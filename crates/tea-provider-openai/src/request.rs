//! Maps an immutable `ModelRequest` to an `OpenAI` Chat Completions request body.

use std::collections::BTreeMap;

use serde_json::{Map, Value, json};
use tea_model::{MAX_SYSTEM_PROMPT_BYTES, ModelRequest};
use tea_protocol::{
    CanonicalMessage, ContentBlock, FINAL_JSON_OBJECT_INSTRUCTION, FinalOutputFormat, ToolCallId,
};

use crate::credential::{OpenAiApiMode, OpenAiCompatibilityProfile, OpenAiConfig};
use crate::error::{OpenAiError, OpenAiErrorCode};
use crate::reasoning::{OpenAiReasoningEffortMap, request_wire_effort};

pub(crate) const JSON_OBJECT_INSTRUCTION: &str = FINAL_JSON_OBJECT_INSTRUCTION;

/// Builds the `OpenAI` Chat Completions JSON body for one request.
///
/// # Errors
///
/// Returns an error when a message or tool call cannot be normalized.
pub fn build_chat_completions_body(
    request: &ModelRequest,
    config: &OpenAiConfig,
) -> Result<Value, OpenAiError> {
    let map = OpenAiReasoningEffortMap::default();
    build_chat_completions_body_with_reasoning_map(request, config, Some(&map))
}

/// Builds a Chat Completions body using the selected model's validated effort map.
///
/// # Errors
///
/// Returns an error when request content cannot be normalized or a requested
/// non-off reasoning effort has no model-level wire mapping.
pub fn build_chat_completions_body_with_reasoning_map(
    request: &ModelRequest,
    config: &OpenAiConfig,
    reasoning_map: Option<&OpenAiReasoningEffortMap>,
) -> Result<Value, OpenAiError> {
    let final_output = validated_final_output(request, config, OpenAiApiMode::ChatCompletions)?;
    let stream = !matches!(
        final_output,
        Some((profile, FinalOutputFormat::JsonSchema { .. }))
            if !profile.supports_streaming_final_json_schema(OpenAiApiMode::ChatCompletions)
    );
    let mut body = Map::new();
    body.insert("model".to_owned(), json!(request.model_id().as_str()));
    body.insert("stream".to_owned(), json!(stream));
    if stream {
        // Request usage in the final stream chunk so the adapter can normalize tokens.
        body.insert("stream_options".to_owned(), json!({"include_usage": true}));
    }

    let mut messages = Vec::with_capacity(request.messages().len() + 1);
    if let Some(system) = request.system_prompt() {
        messages.push(json!({"role": "system", "content": system}));
    }
    let provider_call_ids = provider_tool_call_ids(request.messages());
    for message in request.messages() {
        messages.push(map_message(message, &provider_call_ids)?);
    }
    let structured_instruction = match final_output {
        Some((_, FinalOutputFormat::JsonObject)) => {
            Some(json_object_instructions(request.system_prompt())?)
        }
        Some((OpenAiCompatibilityProfile::Together, FinalOutputFormat::JsonSchema { schema })) => {
            Some(together_schema_instructions(
                request.system_prompt(),
                schema,
            )?)
        }
        _ => None,
    };
    if let Some(instruction) = structured_instruction {
        if request.system_prompt().is_some()
            && let Some(message) = messages.first_mut().and_then(Value::as_object_mut)
        {
            message.insert("content".to_owned(), json!(instruction));
        } else {
            messages.insert(0, json!({"role": "system", "content": instruction}));
        }
    }
    body.insert("messages".to_owned(), Value::Array(messages));

    if !request.tools().is_empty() {
        let tools: Vec<Value> = request
            .tools()
            .iter()
            .map(|tool| {
                let function = tool.as_function().ok_or_else(|| {
                    OpenAiError::new(
                        OpenAiErrorCode::InvalidRequest,
                        "hosted tools require the OpenAI Responses API",
                    )
                })?;
                Ok(json!({
                    "type": "function",
                    "function": {
                        "name": function.name(),
                        "description": function.description(),
                        "parameters": function.input_schema(),
                    }
                }))
            })
            .collect::<Result<_, OpenAiError>>()?;
        body.insert("tools".to_owned(), Value::Array(tools));
        body.insert(
            "parallel_tool_calls".to_owned(),
            json!(request.allow_parallel_tool_calls()),
        );
    }

    let reasoning_effort = request_wire_effort(request, reasoning_map)?;
    let using_reasoning = reasoning_effort.is_some();
    if let Some(effort) = reasoning_effort {
        body.insert("reasoning_effort".to_owned(), json!(effort));
    }
    if let Some(max_output) = request.max_output_tokens() {
        let key = if using_reasoning {
            "max_completion_tokens"
        } else {
            "max_tokens"
        };
        body.insert(key.to_owned(), json!(max_output.get()));
    }

    if let Some((profile, format)) = final_output {
        let response_format = match format {
            FinalOutputFormat::JsonObject => json!({"type": "json_object"}),
            FinalOutputFormat::JsonSchema { schema } => json!({
                "type": "json_schema",
                "json_schema": {
                    "name": "tea_output",
                    "strict": profile.uses_strict_json_schema(OpenAiApiMode::ChatCompletions),
                    "schema": schema,
                }
            }),
        };
        body.insert("response_format".to_owned(), response_format);
        if profile.requires_parameter_support() {
            body.insert("provider".to_owned(), json!({"require_parameters": true}));
        }
    }

    Ok(Value::Object(body))
}

pub(crate) fn json_object_instructions(existing: Option<&str>) -> Result<String, OpenAiError> {
    if let Some(existing) = existing
        && existing.lines().any(|line| line == JSON_OBJECT_INSTRUCTION)
    {
        return Ok(existing.to_owned());
    }
    append_system_instruction(
        existing,
        JSON_OBJECT_INSTRUCTION,
        "JSON-object instruction exceeds system prompt bounds",
    )
}

fn together_schema_instructions(
    existing: Option<&str>,
    schema: &Value,
) -> Result<String, OpenAiError> {
    let schema = serde_json::to_string(schema).map_err(|_| {
        invalid_structured_output("Together JSON Schema instruction could not be serialized")
    })?;
    let instruction = format!(
        "Return the final response as JSON only. Do not use Markdown.\nJSON Schema:\n{schema}"
    );
    if existing.is_some_and(|existing| {
        existing
            .strip_suffix(&instruction)
            .is_some_and(|prefix| prefix.is_empty() || prefix.ends_with("\n\n"))
    }) {
        return Ok(existing.unwrap_or_default().to_owned());
    }
    append_system_instruction(
        existing,
        &instruction,
        "Together JSON Schema instruction exceeds system prompt bounds",
    )
}

fn append_system_instruction(
    existing: Option<&str>,
    instruction: &str,
    overflow_message: &'static str,
) -> Result<String, OpenAiError> {
    let existing_bytes = existing.map_or(0, str::len);
    let separator_bytes = usize::from(existing.is_some()) * 2;
    let encoded_bytes = existing_bytes
        .checked_add(separator_bytes)
        .and_then(|bytes| bytes.checked_add(instruction.len()))
        .ok_or_else(|| invalid_structured_output(overflow_message))?;
    if encoded_bytes > MAX_SYSTEM_PROMPT_BYTES {
        return Err(invalid_structured_output(overflow_message));
    }
    Ok(match existing {
        Some(existing) => format!("{existing}\n\n{instruction}"),
        None => instruction.to_owned(),
    })
}

pub(crate) fn validated_final_output<'a>(
    request: &'a ModelRequest,
    config: &OpenAiConfig,
    api_mode: OpenAiApiMode,
) -> Result<Option<(OpenAiCompatibilityProfile, &'a FinalOutputFormat)>, OpenAiError> {
    let Some(format) = request.final_output_format() else {
        return Ok(None);
    };
    format.validate().map_err(|error| {
        OpenAiError::new(
            OpenAiErrorCode::InvalidRequest,
            format!("OpenAI final-output format is invalid: {error}"),
        )
    })?;
    if config.api_mode() != api_mode {
        return Err(invalid_structured_output(
            "structured output request builder does not match the configured API mode",
        ));
    }
    let Some(profile) = config.compatibility_profile() else {
        return Err(invalid_structured_output(
            "structured output requires an explicit compatibility profile",
        ));
    };
    let supported = match format {
        FinalOutputFormat::JsonObject => profile.supports_final_json_object(api_mode),
        FinalOutputFormat::JsonSchema { .. } => profile.supports_final_json_schema(api_mode),
    };
    if !supported {
        return Err(invalid_structured_output(
            "structured output is unsupported by the compatibility profile and API mode",
        ));
    }
    if matches!(format, FinalOutputFormat::JsonSchema { .. })
        && !request.tools().is_empty()
        && !profile.supports_final_json_schema_with_tools(api_mode)
    {
        return Err(invalid_structured_output(
            "JSON Schema output cannot be combined with tools for this profile",
        ));
    }
    if matches!(format, FinalOutputFormat::JsonSchema { .. })
        && request.allow_parallel_tool_calls()
        && request
            .tools()
            .iter()
            .any(|tool| tool.as_function().is_some())
        && !profile.supports_parallel_tools_with_json_schema()
    {
        return Err(invalid_structured_output(
            "JSON Schema output cannot be combined with parallel function calls for this profile",
        ));
    }
    Ok(Some((profile, format)))
}

fn invalid_structured_output(message: &'static str) -> OpenAiError {
    OpenAiError::new(OpenAiErrorCode::InvalidRequest, message)
}

/// Returns the full Chat Completions endpoint URL for the supplied config.
#[must_use]
pub fn chat_completions_url(config: &OpenAiConfig) -> String {
    let base = config.base_url().trim_end_matches('/');
    format!("{base}/chat/completions")
}

/// Returns the per-request headers for the supplied config.
#[must_use]
pub fn request_headers(config: &OpenAiConfig) -> Vec<(String, String)> {
    let mut headers = Vec::new();
    let key_value = if config.api_key_prefix().is_empty() {
        config.api_key().as_str().to_owned()
    } else {
        format!("{}{}", config.api_key_prefix(), config.api_key().as_str())
    };
    headers.push((config.api_key_header().to_owned(), key_value));
    if let Some(org) = config.org_id() {
        headers.push(("OpenAI-Organization".to_owned(), org.to_owned()));
    }
    if let Some(project) = config.project_id() {
        headers.push(("OpenAI-Project".to_owned(), project.to_owned()));
    }
    headers
}

fn map_message(
    message: &CanonicalMessage,
    provider_call_ids: &BTreeMap<ToolCallId, String>,
) -> Result<Value, OpenAiError> {
    match message {
        CanonicalMessage::User { content, .. } => Ok(json!({
            "role": "user",
            "content": map_user_content(content),
        })),
        CanonicalMessage::Assistant { content, .. } => map_assistant(content, provider_call_ids),
        CanonicalMessage::ToolResult {
            tool_call_id,
            content,
            ..
        } => Ok(json!({
            "role": "tool",
            "tool_call_id": request_tool_call_id(tool_call_id, provider_call_ids),
            "content": map_text_content(content),
        })),
    }
}

fn map_user_content(content: &[ContentBlock]) -> Value {
    let has_image = content
        .iter()
        .any(|block| matches!(block, ContentBlock::Image { .. }));
    if !has_image {
        return Value::String(map_text_content(content));
    }
    let parts: Vec<Value> = content
        .iter()
        .map(|block| match block {
            ContentBlock::Text { text }
            | ContentBlock::ContextualText { text }
            | ContentBlock::Thinking { text } => {
                json!({"type": "text", "text": text})
            }
            ContentBlock::Image { mime_type, source } => json!({
                "type": "image_url",
                "image_url": {"url": image_data_url(mime_type, source)},
            }),
            ContentBlock::ToolCall { .. }
            | ContentBlock::HostedTool { .. }
            | ContentBlock::Citation { .. } => Value::Null,
        })
        .filter(|value| !value.is_null())
        .collect();
    Value::Array(parts)
}

fn map_assistant(
    content: &[ContentBlock],
    provider_call_ids: &BTreeMap<ToolCallId, String>,
) -> Result<Value, OpenAiError> {
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();
    let mut seen_index = 0usize;
    for block in content {
        match block {
            ContentBlock::Text { text } | ContentBlock::Thinking { text } => {
                text_parts.push(text.clone());
            }
            ContentBlock::ToolCall {
                tool_call_id,
                tool_name,
                arguments,
                ..
            } => {
                let arguments_string = serde_json::to_string(arguments).map_err(|error| {
                    OpenAiError::new(
                        OpenAiErrorCode::MalformedResponse,
                        format!("tool arguments serialization failed: {error}"),
                    )
                })?;
                tool_calls.push(json!({
                    "id": request_tool_call_id(tool_call_id, provider_call_ids),
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": arguments_string,
                    },
                    "_index": seen_index,
                }));
                seen_index += 1;
            }
            ContentBlock::ContextualText { .. }
            | ContentBlock::Image { .. }
            | ContentBlock::HostedTool { .. }
            | ContentBlock::Citation { .. } => {}
        }
    }
    // OpenAI expects a numeric index per tool call; strip the helper key after.
    let tool_calls: Vec<Value> = tool_calls
        .into_iter()
        .map(|mut value| {
            if let Some(obj) = value.as_object_mut()
                && let Some(index) = obj.remove("_index")
            {
                obj.insert("index".to_owned(), index);
            }
            value
        })
        .collect();
    let mut message = Map::new();
    message.insert("role".to_owned(), json!("assistant"));
    let text = text_parts.join("");
    if !text.is_empty() {
        message.insert("content".to_owned(), Value::String(text));
    } else if tool_calls.is_empty() {
        message.insert("content".to_owned(), Value::Null);
    }
    if !tool_calls.is_empty() {
        message.insert("tool_calls".to_owned(), Value::Array(tool_calls));
    }
    Ok(Value::Object(message))
}

pub(crate) fn provider_tool_call_ids(
    messages: &[CanonicalMessage],
) -> BTreeMap<ToolCallId, String> {
    let mut provider_call_ids = BTreeMap::new();
    for message in messages {
        let CanonicalMessage::Assistant { content, .. } = message else {
            continue;
        };
        for block in content {
            let ContentBlock::ToolCall {
                tool_call_id,
                provider_call_id: Some(provider_call_id),
                ..
            } = block
            else {
                continue;
            };
            provider_call_ids.insert(*tool_call_id, provider_call_id.clone());
        }
    }
    provider_call_ids
}

pub(crate) fn request_tool_call_id(
    tool_call_id: &ToolCallId,
    provider_call_ids: &BTreeMap<ToolCallId, String>,
) -> String {
    provider_call_ids
        .get(tool_call_id)
        .cloned()
        .unwrap_or_else(|| tool_call_id.to_string())
}

fn map_text_content(content: &[ContentBlock]) -> String {
    content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text }
            | ContentBlock::ContextualText { text }
            | ContentBlock::Thinking { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

fn image_data_url(mime_type: &str, source: &tea_protocol::ImageSource) -> String {
    match source {
        tea_protocol::ImageSource::InlineBase64 { data } => {
            format!("data:{mime_type};base64,{data}")
        }
        tea_protocol::ImageSource::Reference { reference } => reference.clone(),
    }
}
