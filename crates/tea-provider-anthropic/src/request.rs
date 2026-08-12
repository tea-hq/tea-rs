//! Provider-neutral request mapping for Anthropic Messages.

use std::collections::BTreeMap;
use std::fmt;

use serde_json::{Map, Value, json};
use tea_model::{HostedToolDefinition, ModelRequest, ModelToolDefinition};
use tea_protocol::{CanonicalMessage, ContentBlock, ImageSource, ProviderContinuation, ToolCallId};

use crate::WEB_SEARCH_CONTINUATION_FORMAT;
use crate::credential::AnthropicConfig;
use crate::error::{AnthropicError, AnthropicErrorCode};

const DEFAULT_MAX_OUTPUT_TOKENS: u64 = 4096;

struct AnthropicMessage {
    role: &'static str,
    content: Vec<Value>,
}

impl fmt::Debug for AnthropicMessage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnthropicMessage")
            .field("role", &self.role)
            .field("content_count", &self.content.len())
            .finish_non_exhaustive()
    }
}

impl AnthropicMessage {
    fn into_value(self) -> Value {
        json!({
            "role": self.role,
            "content": self.content,
        })
    }
}

/// Builds one streaming Anthropic Messages request body.
///
/// # Errors
///
/// Returns an error when provider-neutral message content cannot be represented
/// by the Messages API.
pub fn build_messages_body(
    request: &ModelRequest,
    config: &AnthropicConfig,
) -> Result<Value, AnthropicError> {
    let mut body = Map::new();
    body.insert("model".to_owned(), json!(request.model_id().as_str()));
    body.insert("stream".to_owned(), Value::Bool(true));
    body.insert(
        "max_tokens".to_owned(),
        json!(
            request
                .max_output_tokens()
                .map_or(DEFAULT_MAX_OUTPUT_TOKENS, tea_protocol::TokenCount::get)
        ),
    );
    if let Some(system) = request.system_prompt() {
        body.insert("system".to_owned(), json!(system));
    }
    let provider_call_ids = provider_tool_call_ids(request.messages());
    let messages = map_messages(request.messages(), &provider_call_ids)?;
    body.insert("messages".to_owned(), Value::Array(messages));
    if !request.tools().is_empty() {
        let tools = request
            .tools()
            .iter()
            .map(|tool| map_tool_definition(tool, config))
            .collect();
        body.insert("tools".to_owned(), Value::Array(tools));
    }
    Ok(Value::Object(body))
}

/// Returns the Anthropic Messages endpoint for a configuration.
#[must_use]
pub fn messages_url(config: &AnthropicConfig) -> String {
    format!("{}/v1/messages", config.base_url().trim_end_matches('/'))
}

/// Returns request headers without exposing them through debug output.
#[must_use]
pub fn request_headers(config: &AnthropicConfig) -> Vec<(String, String)> {
    vec![
        ("x-api-key".to_owned(), config.api_key().as_str().to_owned()),
        (
            "anthropic-version".to_owned(),
            config.api_version().to_owned(),
        ),
    ]
}

fn map_messages(
    messages: &[CanonicalMessage],
    provider_call_ids: &BTreeMap<ToolCallId, String>,
) -> Result<Vec<Value>, AnthropicError> {
    let mut mapped: Vec<AnthropicMessage> = Vec::with_capacity(messages.len());
    let mut previous_was_tool_result = false;
    for message in messages {
        let is_tool_result = matches!(message, CanonicalMessage::ToolResult { .. });
        let mut message = map_message(message, provider_call_ids)?;
        // Parallel tool results must share one user turn for Anthropic-compatible gateways.
        if is_tool_result
            && previous_was_tool_result
            && let Some(previous) = mapped.last_mut()
        {
            previous.content.append(&mut message.content);
        } else {
            mapped.push(message);
        }
        previous_was_tool_result = is_tool_result;
    }
    Ok(mapped
        .into_iter()
        .map(AnthropicMessage::into_value)
        .collect())
}

fn map_message(
    message: &CanonicalMessage,
    provider_call_ids: &BTreeMap<ToolCallId, String>,
) -> Result<AnthropicMessage, AnthropicError> {
    match message {
        CanonicalMessage::User { content, .. } => Ok(AnthropicMessage {
            role: "user",
            content: map_user_content(content)?,
        }),
        CanonicalMessage::Assistant { content, .. } => Ok(AnthropicMessage {
            role: "assistant",
            content: map_assistant_content(content, provider_call_ids)?,
        }),
        CanonicalMessage::ToolResult {
            tool_call_id,
            content,
            is_error,
            ..
        } => Ok(AnthropicMessage {
            role: "user",
            content: vec![json!({
                "type": "tool_result",
                "tool_use_id": request_tool_call_id(tool_call_id, provider_call_ids),
                "content": map_tool_result_content(content)?,
                "is_error": is_error,
            })],
        }),
    }
}

fn map_user_content(content: &[ContentBlock]) -> Result<Vec<Value>, AnthropicError> {
    content.iter().map(map_user_block).collect()
}

fn map_user_block(block: &ContentBlock) -> Result<Value, AnthropicError> {
    match block {
        ContentBlock::Text { text } | ContentBlock::ContextualText { text } => {
            Ok(json!({"type": "text", "text": text}))
        }
        ContentBlock::Image { mime_type, source } => match source {
            ImageSource::InlineBase64 { data } => Ok(json!({
                "type": "image",
                "source": {"type": "base64", "media_type": mime_type, "data": data},
            })),
            ImageSource::Reference { .. } => Err(invalid("referenced images are not supported")),
        },
        ContentBlock::Thinking { .. }
        | ContentBlock::ToolCall { .. }
        | ContentBlock::HostedTool { .. }
        | ContentBlock::Citation { .. } => Err(invalid("invalid user content block")),
    }
}

fn map_assistant_content(
    content: &[ContentBlock],
    provider_call_ids: &BTreeMap<ToolCallId, String>,
) -> Result<Vec<Value>, AnthropicError> {
    let mut mapped = Vec::with_capacity(content.len());
    for block in content {
        match block {
            ContentBlock::HostedTool { activity } => {
                append_hosted_replay(activity.continuation(), &mut mapped)?;
            }
            ContentBlock::Citation { citation } => {
                append_citation_replay(citation.continuation(), &mut mapped)?;
            }
            _ => mapped.push(map_assistant_block(block, provider_call_ids)?),
        }
    }
    Ok(mapped)
}

fn map_assistant_block(
    block: &ContentBlock,
    provider_call_ids: &BTreeMap<ToolCallId, String>,
) -> Result<Value, AnthropicError> {
    match block {
        ContentBlock::Text { text } => Ok(json!({"type": "text", "text": text})),
        ContentBlock::ToolCall {
            tool_call_id,
            tool_name,
            arguments,
            ..
        } => Ok(json!({
            "type": "tool_use",
            "id": request_tool_call_id(tool_call_id, provider_call_ids),
            "name": tool_name,
            "input": arguments,
        })),
        ContentBlock::ContextualText { .. } | ContentBlock::Thinking { .. } => Err(invalid(
            "contextual or extended thinking content is not assistant output",
        )),
        ContentBlock::Image { .. } => Err(invalid("invalid assistant content block")),
        ContentBlock::HostedTool { .. } | ContentBlock::Citation { .. } => {
            Err(invalid("invalid hosted replay content block"))
        }
    }
}

fn provider_tool_call_ids(messages: &[CanonicalMessage]) -> BTreeMap<ToolCallId, String> {
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

fn request_tool_call_id(
    tool_call_id: &ToolCallId,
    provider_call_ids: &BTreeMap<ToolCallId, String>,
) -> String {
    provider_call_ids
        .get(tool_call_id)
        .cloned()
        .unwrap_or_else(|| tool_call_id.to_string())
}

fn map_tool_result_content(content: &[ContentBlock]) -> Result<Vec<Value>, AnthropicError> {
    content
        .iter()
        .map(|block| match block {
            ContentBlock::Text { text } => Ok(json!({"type": "text", "text": text})),
            ContentBlock::Image { mime_type, source } => match source {
                ImageSource::InlineBase64 { data } => Ok(json!({
                    "type": "image",
                    "source": {"type": "base64", "media_type": mime_type, "data": data},
                })),
                ImageSource::Reference { .. } => {
                    Err(invalid("referenced images are not supported"))
                }
            },
            ContentBlock::ContextualText { .. }
            | ContentBlock::Thinking { .. }
            | ContentBlock::ToolCall { .. }
            | ContentBlock::HostedTool { .. }
            | ContentBlock::Citation { .. } => Err(invalid("invalid tool result content block")),
        })
        .collect()
}

fn map_tool_definition(tool: &ModelToolDefinition, config: &AnthropicConfig) -> Value {
    match tool {
        ModelToolDefinition::Function(tool) => json!({
            "name": tool.name(),
            "description": tool.description(),
            "input_schema": tool.input_schema(),
        }),
        ModelToolDefinition::Hosted(tool) => map_hosted_tool(tool, config),
    }
}

fn map_hosted_tool(tool: &HostedToolDefinition, config: &AnthropicConfig) -> Value {
    let options = tool.options().web_search();
    let mut mapped = Map::new();
    mapped.insert("type".to_owned(), json!(config.web_search().tool_type()));
    mapped.insert("name".to_owned(), json!(tool.name()));
    mapped.insert("max_uses".to_owned(), json!(config.web_search().max_uses()));
    if !options.allowed_domains().is_empty() {
        mapped.insert(
            "allowed_domains".to_owned(),
            json!(options.allowed_domains()),
        );
    }
    if !options.blocked_domains().is_empty() {
        mapped.insert(
            "blocked_domains".to_owned(),
            json!(options.blocked_domains()),
        );
    }
    if let Some(location) = options.location() {
        let mut mapped_location = Map::new();
        mapped_location.insert("type".to_owned(), json!("approximate"));
        for (key, value) in [
            ("country", location.country()),
            ("city", location.city()),
            ("region", location.region()),
            ("timezone", location.timezone()),
        ] {
            if let Some(value) = value {
                mapped_location.insert(key.to_owned(), json!(value));
            }
        }
        mapped.insert("user_location".to_owned(), Value::Object(mapped_location));
    }
    Value::Object(mapped)
}

fn append_hosted_replay(
    continuation: Option<&ProviderContinuation>,
    mapped: &mut Vec<Value>,
) -> Result<(), AnthropicError> {
    let Some(payload) = matching_continuation(continuation) else {
        return Ok(());
    };
    let blocks = payload
        .get("content_blocks")
        .and_then(Value::as_array)
        .filter(|blocks| !blocks.is_empty())
        .ok_or_else(|| invalid("Anthropic hosted continuation is malformed"))?;
    if blocks.iter().any(|block| !block.is_object()) {
        return Err(invalid("Anthropic hosted continuation is malformed"));
    }
    mapped.extend(blocks.iter().cloned());
    Ok(())
}

fn append_citation_replay(
    continuation: Option<&ProviderContinuation>,
    mapped: &mut [Value],
) -> Result<(), AnthropicError> {
    let Some(payload) = matching_continuation(continuation) else {
        return Ok(());
    };
    let citation = payload
        .get("citation")
        .filter(|citation| citation.is_object())
        .cloned()
        .ok_or_else(|| invalid("Anthropic citation continuation is malformed"))?;
    let text = mapped
        .last_mut()
        .and_then(Value::as_object_mut)
        .filter(|block| block.get("type").and_then(Value::as_str) == Some("text"))
        .ok_or_else(|| invalid("Anthropic citation has no preceding text block"))?;
    let citations = text
        .entry("citations".to_owned())
        .or_insert_with(|| Value::Array(Vec::new()))
        .as_array_mut()
        .ok_or_else(|| invalid("Anthropic text citations are malformed"))?;
    citations.push(citation);
    Ok(())
}

fn matching_continuation(continuation: Option<&ProviderContinuation>) -> Option<&Value> {
    continuation
        .filter(|continuation| {
            continuation.provider() == crate::credential::PROVIDER_ID
                && continuation.format() == WEB_SEARCH_CONTINUATION_FORMAT
        })
        .map(ProviderContinuation::payload)
}

fn invalid(message: &'static str) -> AnthropicError {
    AnthropicError::new(AnthropicErrorCode::InvalidRequest, message)
}

#[cfg(test)]
mod tests {
    use super::*;

    const OPAQUE_SENTINEL: &str = "opaque-continuation-must-not-appear-in-debug";

    #[test]
    fn anthropic_message_debug_redacts_content() {
        let message = AnthropicMessage {
            role: "assistant",
            content: vec![json!({"opaque": OPAQUE_SENTINEL})],
        };

        let debug = format!("{message:?}");
        assert!(!debug.contains(OPAQUE_SENTINEL), "{debug}");
    }
}
