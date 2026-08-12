//! Request mapping for the `OpenAI` Responses API.

use std::collections::BTreeMap;

use serde_json::Value;
use tea_model::{HostedToolKind, ModelRequest, ModelToolDefinition};
use tea_protocol::{
    CanonicalMessage, ContentBlock, ImageSource, ProviderContinuation, SourceCitation, ToolCallId,
};

use crate::credential::OpenAiConfig;
use crate::error::{OpenAiError, OpenAiErrorCode};
use crate::reasoning::{OpenAiReasoningEffortMap, request_wire_effort};
use crate::request::{provider_tool_call_ids, request_tool_call_id};
use crate::responses_model::{
    FunctionCallOutputPayload, ResponseItem, ResponsesApiRequest, ResponsesApiTool,
    ResponsesContentItem, ResponsesInputItem, ResponsesReasoning, ResponsesWebSearchFilters,
    ResponsesWebSearchLocation,
};

pub(crate) const WEB_SEARCH_CONTINUATION_FORMAT: &str = "openai.responses.web_search.v1";
pub(crate) const URL_CITATION_CONTINUATION_FORMAT: &str = "openai.responses.url_citation.v1";

/// Builds the `OpenAI` Responses API JSON body for one request.
///
/// # Errors
///
/// Returns an error when a message or tool call cannot be serialized.
pub fn build_responses_body(
    request: &ModelRequest,
    config: &OpenAiConfig,
) -> Result<Value, OpenAiError> {
    let map = OpenAiReasoningEffortMap::default();
    build_responses_body_with_reasoning_map(request, config, Some(&map))
}

/// Builds a Responses body using the selected model's validated effort map.
///
/// # Errors
///
/// Returns an error when request content cannot be serialized or a requested
/// non-off reasoning effort has no model-level wire mapping.
pub fn build_responses_body_with_reasoning_map(
    request: &ModelRequest,
    config: &OpenAiConfig,
    reasoning_map: Option<&OpenAiReasoningEffortMap>,
) -> Result<Value, OpenAiError> {
    let mut input = Vec::new();
    let provider_call_ids = provider_tool_call_ids(request.messages());
    for message in request.messages() {
        input.extend(map_message(
            message,
            &provider_call_ids,
            config.provider_id().as_str(),
        )?);
    }
    let tools: Vec<ResponsesApiTool> = request
        .tools()
        .iter()
        .map(|tool| match tool {
            ModelToolDefinition::Function(function) => ResponsesApiTool::Function {
                name: function.name().to_owned(),
                description: function.description().to_owned(),
                parameters: function.input_schema().clone(),
                strict: false,
            },
            ModelToolDefinition::Hosted(hosted) => match hosted.kind() {
                HostedToolKind::WebSearch => {
                    let options = hosted.options().web_search();
                    let filters = if options.allowed_domains().is_empty()
                        && options.blocked_domains().is_empty()
                    {
                        None
                    } else {
                        Some(ResponsesWebSearchFilters {
                            allowed_domains: options.allowed_domains().to_vec(),
                            blocked_domains: options.blocked_domains().to_vec(),
                        })
                    };
                    let user_location =
                        options
                            .location()
                            .map(|location| ResponsesWebSearchLocation {
                                r#type: "approximate".to_owned(),
                                country: location.country().map(str::to_owned),
                                city: location.city().map(str::to_owned),
                                region: location.region().map(str::to_owned),
                                timezone: location.timezone().map(str::to_owned),
                            });
                    ResponsesApiTool::WebSearch {
                        filters,
                        user_location,
                    }
                }
            },
        })
        .collect();
    let reasoning = request_wire_effort(request, reasoning_map)?.map(|effort| ResponsesReasoning {
        effort: effort.to_owned(),
        summary: "auto".to_owned(),
    });
    let mut include = Vec::new();
    if reasoning.is_some() {
        include.push("reasoning.encrypted_content".to_owned());
    }
    if request
        .tools()
        .iter()
        .any(|tool| tool.hosted_kind() == Some(HostedToolKind::WebSearch))
    {
        include.push("web_search_call.action.sources".to_owned());
    }
    let has_tools = !tools.is_empty();
    let body = ResponsesApiRequest {
        model: request.model_id().as_str().to_owned(),
        instructions: request.system_prompt().map(str::to_owned),
        input,
        tools,
        tool_choice: has_tools.then(|| "auto".to_owned()),
        parallel_tool_calls: has_tools.then(|| request.allow_parallel_tool_calls()),
        reasoning,
        store: false,
        stream: true,
        include,
        max_output_tokens: request
            .max_output_tokens()
            .map(tea_protocol::TokenCount::get),
        service_tier: None,
        prompt_cache_key: None,
    };
    serde_json::to_value(body).map_err(|error| {
        OpenAiError::new(
            OpenAiErrorCode::Internal,
            format!("Responses request serialization failed: {error}"),
        )
    })
}

/// Returns the full Responses endpoint URL for the supplied config.
#[must_use]
pub fn responses_url(config: &OpenAiConfig) -> String {
    let base = config.base_url().trim_end_matches('/');
    format!("{base}/responses")
}

fn map_message(
    message: &CanonicalMessage,
    provider_call_ids: &BTreeMap<ToolCallId, String>,
    provider_id: &str,
) -> Result<Vec<ResponsesInputItem>, OpenAiError> {
    match message {
        CanonicalMessage::User { content, .. } => {
            Ok(vec![ResponsesInputItem::Canonical(ResponseItem::Message {
                id: None,
                role: "user".to_owned(),
                content: map_input_content(content),
                status: None,
            })])
        }
        CanonicalMessage::Assistant { id, content, .. } => {
            let message_id = id.to_string();
            map_assistant(&message_id, content, provider_call_ids, provider_id)
        }
        CanonicalMessage::ToolResult {
            tool_call_id,
            content,
            ..
        } => Ok(vec![ResponsesInputItem::Canonical(
            ResponseItem::FunctionCallOutput {
                call_id: request_tool_call_id(tool_call_id, provider_call_ids),
                output: map_tool_output(content),
            },
        )]),
    }
}

fn map_assistant(
    message_id: &str,
    content: &[ContentBlock],
    provider_call_ids: &BTreeMap<ToolCallId, String>,
    provider_id: &str,
) -> Result<Vec<ResponsesInputItem>, OpenAiError> {
    let mut items = Vec::new();
    let mut text_blocks = Vec::new();
    let mut citations = Vec::new();
    let mut message_index = 0_usize;
    for block in content {
        match block {
            ContentBlock::Text { text } => text_blocks.push(text.as_str()),
            ContentBlock::Citation { citation } => citations.push(citation),
            ContentBlock::ToolCall {
                tool_call_id,
                tool_name,
                arguments,
                ..
            } => {
                flush_assistant_message(
                    &mut items,
                    message_id,
                    &mut message_index,
                    &mut text_blocks,
                    &mut citations,
                    provider_id,
                )?;
                let arguments = serde_json::to_string(arguments).map_err(|error| {
                    OpenAiError::new(
                        OpenAiErrorCode::MalformedResponse,
                        format!("tool arguments serialization failed: {error}"),
                    )
                })?;
                items.push(ResponsesInputItem::Canonical(ResponseItem::FunctionCall {
                    id: None,
                    call_id: request_tool_call_id(tool_call_id, provider_call_ids),
                    name: tool_name.clone(),
                    arguments,
                }));
            }
            ContentBlock::HostedTool { activity } => {
                flush_assistant_message(
                    &mut items,
                    message_id,
                    &mut message_index,
                    &mut text_blocks,
                    &mut citations,
                    provider_id,
                )?;
                if let Some(item) = replay_hosted_tool(
                    activity.continuation(),
                    activity.provider_call_id(),
                    provider_id,
                )? {
                    items.push(ResponsesInputItem::ProviderContinuation(item));
                }
            }
            ContentBlock::ContextualText { .. }
            | ContentBlock::Thinking { .. }
            | ContentBlock::Image { .. } => {}
        }
    }
    flush_assistant_message(
        &mut items,
        message_id,
        &mut message_index,
        &mut text_blocks,
        &mut citations,
        provider_id,
    )?;
    Ok(items)
}

fn flush_assistant_message(
    items: &mut Vec<ResponsesInputItem>,
    message_id: &str,
    message_index: &mut usize,
    text_blocks: &mut Vec<&str>,
    citations: &mut Vec<&SourceCitation>,
    provider_id: &str,
) -> Result<(), OpenAiError> {
    if text_blocks.is_empty() {
        citations.clear();
        return Ok(());
    }
    let joined_text = text_blocks.concat();
    let annotations = citations
        .iter()
        .filter_map(|citation| map_citation(citation, provider_id, &joined_text).transpose())
        .collect::<Result<Vec<_>, _>>()?;
    let output_text = if annotations.is_empty() {
        text_blocks
            .iter()
            .map(|text| ResponsesContentItem::OutputText {
                text: (*text).to_owned(),
                annotations: Vec::new(),
            })
            .collect()
    } else {
        vec![ResponsesContentItem::OutputText {
            text: joined_text,
            annotations,
        }]
    };
    let id = if *message_index == 0 {
        format!("msg_{message_id}")
    } else {
        format!("msg_{message_id}_{message_index}")
    };
    items.push(ResponsesInputItem::Canonical(ResponseItem::Message {
        id: Some(id),
        role: "assistant".to_owned(),
        content: output_text,
        status: Some("completed".to_owned()),
    }));
    *message_index += 1;
    text_blocks.clear();
    citations.clear();
    Ok(())
}

fn replay_hosted_tool(
    continuation: Option<&ProviderContinuation>,
    provider_call_id: &str,
    provider_id: &str,
) -> Result<Option<Value>, OpenAiError> {
    let Some(continuation) = continuation.filter(|continuation| {
        continuation.provider() == provider_id
            && continuation.format() == WEB_SEARCH_CONTINUATION_FORMAT
    }) else {
        return Ok(None);
    };
    let item = continuation.payload();
    let action_is_valid = match item.get("status").and_then(Value::as_str) {
        Some("completed") => item.get("action").is_some_and(Value::is_object),
        Some("failed") => item
            .get("action")
            .is_none_or(|action| action.is_null() || action.is_object()),
        _ => false,
    };
    if !item.is_object()
        || item.get("type").and_then(Value::as_str) != Some("web_search_call")
        || item.get("id").and_then(Value::as_str) != Some(provider_call_id)
        || !action_is_valid
    {
        return Err(OpenAiError::new(
            OpenAiErrorCode::InvalidRequest,
            "OpenAI web-search continuation payload is invalid",
        ));
    }
    Ok(Some(item.clone()))
}

fn map_citation(
    citation: &SourceCitation,
    provider_id: &str,
    text: &str,
) -> Result<Option<Value>, OpenAiError> {
    if let Some(continuation) = citation.continuation().filter(|continuation| {
        continuation.provider() == provider_id
            && continuation.format() == URL_CITATION_CONTINUATION_FORMAT
    }) {
        let annotation = continuation.payload();
        let Some(annotation) = normalize_citation_continuation(annotation, citation, text) else {
            return Err(OpenAiError::new(
                OpenAiErrorCode::InvalidRequest,
                "OpenAI URL-citation continuation payload is invalid",
            ));
        };
        return Ok(Some(annotation));
    }
    let Some((start_index, end_index)) = citation.range() else {
        return Ok(None);
    };
    let (start_index, end_index) = utf8_range_to_character_range(
        text,
        usize::try_from(start_index).map_err(|_| {
            OpenAiError::new(
                OpenAiErrorCode::InvalidRequest,
                "OpenAI URL citation range is invalid",
            )
        })?,
        usize::try_from(end_index).map_err(|_| {
            OpenAiError::new(
                OpenAiErrorCode::InvalidRequest,
                "OpenAI URL citation range is invalid",
            )
        })?,
    )
    .ok_or_else(|| {
        OpenAiError::new(
            OpenAiErrorCode::InvalidRequest,
            "OpenAI URL citation range is invalid",
        )
    })?;
    Ok(Some(serde_json::json!({
        "type": "url_citation",
        "start_index": start_index,
        "end_index": end_index,
        "url": citation.source().url(),
        "title": citation.source().title().unwrap_or(citation.source().url()),
    })))
}

fn normalize_citation_continuation(
    annotation: &Value,
    citation: &SourceCitation,
    text: &str,
) -> Option<Value> {
    if !annotation.is_object()
        || annotation.get("type").and_then(Value::as_str) != Some("url_citation")
        || annotation.get("url").and_then(Value::as_str) != Some(citation.source().url())
    {
        return None;
    }
    let annotation_title = match annotation.get("title") {
        Some(Value::String(title)) => Some(title.as_str()),
        Some(Value::Null) | None => None,
        _ => return None,
    };
    if annotation_title != citation.source().title() {
        return None;
    }
    let start_index = annotation
        .get("start_index")
        .and_then(Value::as_u64)
        .and_then(|index| usize::try_from(index).ok())?;
    let end_index = annotation
        .get("end_index")
        .and_then(Value::as_u64)
        .and_then(|index| usize::try_from(index).ok())?;
    let (byte_start, byte_end) = citation.range().and_then(|(start, end)| {
        Some((usize::try_from(start).ok()?, usize::try_from(end).ok()?))
    })?;
    let (canonical_start, canonical_end) =
        utf8_range_to_character_range(text, byte_start, byte_end)?;
    if start_index >= end_index
        || end_index.checked_sub(start_index) != canonical_end.checked_sub(canonical_start)
        || citation.cited_text() != Some(&text[byte_start..byte_end])
    {
        return None;
    }
    let mut normalized = annotation.clone();
    let object = normalized.as_object_mut()?;
    object.insert("start_index".to_owned(), canonical_start.into());
    object.insert("end_index".to_owned(), canonical_end.into());
    Some(normalized)
}

fn utf8_range_to_character_range(text: &str, start: usize, end: usize) -> Option<(usize, usize)> {
    if start >= end
        || end > text.len()
        || !text.is_char_boundary(start)
        || !text.is_char_boundary(end)
    {
        return None;
    }
    Some((text[..start].chars().count(), text[..end].chars().count()))
}

fn map_input_content(content: &[ContentBlock]) -> Vec<ResponsesContentItem> {
    content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text }
            | ContentBlock::ContextualText { text }
            | ContentBlock::Thinking { text } => {
                Some(ResponsesContentItem::InputText { text: text.clone() })
            }
            ContentBlock::Image { mime_type, source } => Some(ResponsesContentItem::InputImage {
                image_url: image_data_url(mime_type, source),
                detail: "auto".to_owned(),
            }),
            ContentBlock::ToolCall { .. }
            | ContentBlock::HostedTool { .. }
            | ContentBlock::Citation { .. } => None,
        })
        .collect()
}

fn map_tool_output(content: &[ContentBlock]) -> FunctionCallOutputPayload {
    let has_image = content
        .iter()
        .any(|block| matches!(block, ContentBlock::Image { .. }));
    if has_image {
        return FunctionCallOutputPayload::Content(map_input_content(content));
    }
    let text = content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text }
            | ContentBlock::ContextualText { text }
            | ContentBlock::Thinking { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("");
    FunctionCallOutputPayload::Text(if text.is_empty() {
        "(no tool output)".to_owned()
    } else {
        text
    })
}

fn image_data_url(mime_type: &str, source: &ImageSource) -> String {
    match source {
        ImageSource::InlineBase64 { data } => format!("data:{mime_type};base64,{data}"),
        ImageSource::Reference { reference } => reference.clone(),
    }
}
