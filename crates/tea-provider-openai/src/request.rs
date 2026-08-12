//! Maps an immutable `ModelRequest` to an `OpenAI` Chat Completions request body.

use std::collections::BTreeMap;

use serde_json::{Map, Value, json};
use tea_model::ModelRequest;
use tea_protocol::{CanonicalMessage, ContentBlock, ToolCallId};

use crate::credential::OpenAiConfig;
use crate::error::{OpenAiError, OpenAiErrorCode};
use crate::reasoning::{OpenAiReasoningEffortMap, request_wire_effort};

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
    _config: &OpenAiConfig,
    reasoning_map: Option<&OpenAiReasoningEffortMap>,
) -> Result<Value, OpenAiError> {
    let mut body = Map::new();
    body.insert("model".to_owned(), json!(request.model_id().as_str()));
    body.insert("stream".to_owned(), json!(true));
    // Request usage in the final stream chunk so the adapter can normalize tokens.
    body.insert("stream_options".to_owned(), json!({"include_usage": true}));

    let mut messages = Vec::with_capacity(request.messages().len() + 1);
    if let Some(system) = request.system_prompt() {
        messages.push(json!({"role": "system", "content": system}));
    }
    let provider_call_ids = provider_tool_call_ids(request.messages());
    for message in request.messages() {
        messages.push(map_message(message, &provider_call_ids)?);
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

    Ok(Value::Object(body))
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
