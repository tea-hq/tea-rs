//! Maps parsed `OpenAI` streaming chunks to normalized `ModelEvent`s.
//!
//! The [`ChunkReducer`] is transport-agnostic: the live `reqwest` stream and
//! the fixture-backed conformance tests both feed parsed JSON payloads through
//! it so the mapping logic is identical. Tool-call argument fragments are
//! accumulated per stream index and completed on `finish_reason`.

use std::collections::{BTreeMap, BTreeSet};
use std::str::FromStr;

use serde_json::Value;
use tea_model::{
    MAX_MODEL_DELTA_BYTES, ModelCompletion, ModelEvent, ModelFailure, ModelFailureCode,
    ModelResponseInfo, ModelStreamIndex, ProviderResponseId, ProviderToolCallId,
    ToolArgumentsDelta, ToolCallCompleted, ToolCallStarted, Utf8Delta,
};
use tea_protocol::{MAX_TEXT_BLOCK_BYTES, ModelId, RetryClass, StopReason, TokenCount, Usage};
use tea_provider_http::normalize_provider_error;

use crate::error::{OpenAiError, OpenAiErrorCode};

/// Accumulator for one streaming tool call.
#[derive(Debug, Default, Clone)]
struct ToolCallAccumulator {
    index: u16,
    provider_call_id: Option<String>,
    name: Option<String>,
    arguments: String,
}

/// Stateful reducer mapping `OpenAI` chunk JSON to normalized `ModelEvent`s.
#[derive(Debug, Default)]
pub struct ChunkReducer {
    started: bool,
    tool_calls: BTreeMap<u16, ToolCallAccumulator>,
    stop_reason: Option<StopReason>,
    saw_refusal: bool,
    terminal_emitted: bool,
}

impl ChunkReducer {
    /// Creates an empty reducer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns whether a terminal event has already been emitted.
    #[must_use]
    pub const fn terminal_emitted(&self) -> bool {
        self.terminal_emitted
    }

    /// Maps one parsed JSON chunk to zero or more normalized events.
    ///
    /// # Errors
    ///
    /// Returns an error when the chunk is malformed or a normalized event
    /// cannot be constructed.
    pub fn map_chunk(&mut self, value: &Value) -> Result<Vec<ModelEvent>, OpenAiError> {
        if self.terminal_emitted {
            return Err(malformed(
                "Chat Completions chunk arrived after terminal event",
            ));
        }
        let mut events = Vec::new();
        if !self.started {
            events.push(ModelEvent::Started(ModelResponseInfo::new()));
            self.started = true;
        }
        if let Some(error) = value.get("error") {
            self.terminal_emitted = true;
            events.push(ModelEvent::Failed(map_stream_error(error)));
            return Ok(events);
        }
        let choices = value.get("choices").and_then(Value::as_array);
        if let Some(usage) = value.get("usage").filter(|usage| !usage.is_null()) {
            if !matches!(choices, Some(choices) if choices.is_empty()) {
                return Err(malformed(
                    "Chat Completions usage chunk must contain empty choices",
                ));
            }
            let stop_reason = self.stop_reason.clone().ok_or_else(|| {
                malformed("Chat Completions usage chunk arrived before finish_reason")
            })?;
            events.push(self.complete(Some(stop_reason), usage)?);
            return Ok(events);
        }
        if self.stop_reason.is_some() {
            return Err(malformed(
                "Chat Completions chunk arrived after finish_reason",
            ));
        }
        let Some(choices) = choices else {
            return Ok(events);
        };
        if choices.len() > 1 {
            return Err(malformed(
                "Chat Completions chunk contains multiple choices",
            ));
        }
        if let Some(choice) = choices.first() {
            if choice.get("index").and_then(Value::as_u64) != Some(0) {
                return Err(malformed(
                    "Chat Completions choice index is missing or unsupported",
                ));
            }
            self.map_choice(choice, &mut events)?;
        }
        Ok(events)
    }

    /// Flushes the terminal `Completed` event when the stream ends after a
    /// provider finish reason but without a usage chunk.
    ///
    /// # Errors
    ///
    /// Returns an error when the provider did not send a finish reason or when
    /// accumulated tool-call arguments cannot parse.
    pub fn finish(&mut self) -> Result<Option<ModelEvent>, OpenAiError> {
        if self.terminal_emitted {
            return Ok(None);
        }
        if self.stop_reason.is_some() {
            let stop = self.stop_reason.clone();
            Ok(Some(self.complete(stop, &Value::Null)?))
        } else {
            Err(malformed(
                "Chat Completions stream ended without finish_reason",
            ))
        }
    }

    fn map_choice(
        &mut self,
        choice: &Value,
        events: &mut Vec<ModelEvent>,
    ) -> Result<(), OpenAiError> {
        let delta = choice.get("delta").unwrap_or(&Value::Null);
        if let Some(reasoning) = delta.get("reasoning").and_then(Value::as_str)
            && !reasoning.is_empty()
        {
            events.push(ModelEvent::ThinkingDelta(
                Utf8Delta::new(reasoning.to_owned()).map_err(|_| malformed("reasoning delta"))?,
            ));
        }
        if let Some(content) = delta.get("content").and_then(Value::as_str)
            && !content.is_empty()
        {
            events.push(ModelEvent::TextDelta(
                Utf8Delta::new(content.to_owned()).map_err(|_| malformed("text delta"))?,
            ));
        }
        if let Some(refusal) = delta.get("refusal").and_then(Value::as_str) {
            self.saw_refusal = true;
            if !refusal.is_empty() {
                events.push(ModelEvent::TextDelta(
                    Utf8Delta::new(refusal.to_owned()).map_err(|_| malformed("refusal delta"))?,
                ));
            }
        }
        if let Some(tool_calls) = delta.get("tool_calls").and_then(Value::as_array) {
            for tool_call in tool_calls {
                self.map_tool_call_delta(tool_call, events)?;
            }
        }
        if let Some(finish) = choice
            .get("finish_reason")
            .and_then(Value::as_str)
            .map(str::to_owned)
            && !finish.is_empty()
            && self.stop_reason.is_none()
        {
            let stop_reason = if self.saw_refusal {
                StopReason::Refusal
            } else {
                match finish.as_str() {
                    "stop" => StopReason::Completed,
                    "length" => StopReason::Length,
                    "tool_calls" | "function_call" => StopReason::ToolUse,
                    "refusal" => StopReason::Refusal,
                    _ => StopReason::Unknown(finish),
                }
            };
            self.stop_reason = Some(stop_reason.clone());
            for event in self.complete_tool_calls()? {
                events.push(event);
            }
        }
        Ok(())
    }

    fn map_tool_call_delta(
        &mut self,
        tool_call: &Value,
        events: &mut Vec<ModelEvent>,
    ) -> Result<(), OpenAiError> {
        let index = tool_call
            .get("index")
            .and_then(Value::as_u64)
            .ok_or_else(|| malformed("tool call index missing"))
            .and_then(|value| {
                u16::try_from(value.min(u64::from(u16::MAX)))
                    .map_err(|_| malformed("tool call index out of range"))
            })?;
        let accumulator = self
            .tool_calls
            .entry(index)
            .or_insert_with(|| ToolCallAccumulator {
                index,
                ..Default::default()
            });
        if let Some(id) = tool_call.get("id").and_then(Value::as_str)
            && accumulator.provider_call_id.is_none()
        {
            let provider_id =
                ProviderToolCallId::from_str(id).map_err(|_| malformed("tool call id"))?;
            accumulator.provider_call_id = Some(id.to_owned());
            let name = tool_call
                .get("function")
                .and_then(|function| function.get("name"))
                .and_then(Value::as_str)
                .ok_or_else(|| malformed("tool call name missing"))?;
            accumulator.name = Some(name.to_owned());
            let stream_index =
                ModelStreamIndex::new(index).map_err(|_| malformed("stream index"))?;
            events.push(ModelEvent::ToolCallStarted(
                ToolCallStarted::new(stream_index, provider_id, name)
                    .map_err(|_| malformed("tool call started"))?,
            ));
        }
        if let Some(arguments) = tool_call
            .get("function")
            .and_then(|function| function.get("arguments"))
            .and_then(Value::as_str)
            && !arguments.is_empty()
        {
            accumulator.arguments.push_str(arguments);
            let stream_index =
                ModelStreamIndex::new(index).map_err(|_| malformed("stream index"))?;
            let provider_id = accumulator
                .provider_call_id
                .as_ref()
                .and_then(|id| ProviderToolCallId::from_str(id).ok())
                .ok_or_else(|| malformed("tool arguments without a started call"))?;
            events.push(ModelEvent::ToolArgumentsDelta(
                ToolArgumentsDelta::new(stream_index, provider_id, arguments.to_owned())
                    .map_err(|_| malformed("tool arguments delta"))?,
            ));
        }
        Ok(())
    }

    fn complete_tool_calls(&mut self) -> Result<Vec<ModelEvent>, OpenAiError> {
        let mut events = Vec::new();
        // Complete every accumulated tool call in index order.
        let mut completed: Vec<(u16, ToolCallAccumulator)> = self
            .tool_calls
            .iter()
            .map(|(index, accumulator)| (*index, accumulator.clone()))
            .collect();
        completed.sort_by_key(|(index, _)| *index);
        for (_, accumulator) in completed {
            let arguments: Value = if accumulator.arguments.is_empty() {
                Value::Object(serde_json::Map::new())
            } else {
                serde_json::from_str(&accumulator.arguments)
                    .map_err(|_| malformed("tool arguments did not parse"))?
            };
            let stream_index =
                ModelStreamIndex::new(accumulator.index).map_err(|_| malformed("stream index"))?;
            let provider_id = accumulator
                .provider_call_id
                .as_ref()
                .and_then(|id| ProviderToolCallId::from_str(id).ok())
                .ok_or_else(|| malformed("completed tool call without id"))?;
            let name = accumulator.name.clone().unwrap_or_default();
            events.push(ModelEvent::ToolCallCompleted(
                ToolCallCompleted::new(stream_index, provider_id, name, arguments)
                    .map_err(|_| malformed("tool call completed"))?,
            ));
        }
        Ok(events)
    }

    fn complete(
        &mut self,
        stop_reason: Option<StopReason>,
        usage_value: &Value,
    ) -> Result<ModelEvent, OpenAiError> {
        let stop = stop_reason.unwrap_or(StopReason::Completed);
        let mut completion =
            ModelCompletion::new(stop).map_err(|_| malformed("invalid completion reason"))?;
        if let (Some(prompt), Some(completion_tokens)) = (
            usage_value.get("prompt_tokens").and_then(Value::as_u64),
            usage_value.get("completion_tokens").and_then(Value::as_u64),
        ) {
            let prompt_details = usage_value.get("prompt_tokens_details");
            let cached = prompt_details
                .and_then(|details| details.get("cached_tokens"))
                .and_then(Value::as_u64);
            let cache_write = prompt_details
                .and_then(|details| details.get("cache_write_tokens"))
                .and_then(Value::as_u64);
            let billable_input = prompt
                .checked_sub(cached.unwrap_or(0))
                .and_then(|value| value.checked_sub(cache_write.unwrap_or(0)))
                .ok_or_else(|| malformed("cache usage exceeds prompt tokens"))?;
            if let (Ok(input), Ok(output)) = (
                TokenCount::new(billable_input),
                TokenCount::new(completion_tokens),
            ) {
                let mut usage = Usage::new(input, output);
                if let Some(cached) = cached.and_then(|cached| TokenCount::new(cached).ok()) {
                    usage = usage.with_cache_read(cached);
                }
                if let Some(cache_write) =
                    cache_write.and_then(|cache_write| TokenCount::new(cache_write).ok())
                {
                    usage = usage.with_cache_write(cache_write);
                }
                if let Some(reasoning) = usage_value
                    .get("completion_tokens_details")
                    .and_then(|details| details.get("reasoning_tokens"))
                    .and_then(Value::as_u64)
                    .and_then(|reasoning| TokenCount::new(reasoning).ok())
                    && let Ok(with_reasoning) = usage.clone().with_reasoning(reasoning)
                {
                    usage = with_reasoning;
                }
                completion = completion.with_usage(usage);
            }
        }
        self.terminal_emitted = true;
        Ok(ModelEvent::Completed(completion))
    }
}

/// Maps one non-streaming Chat Completions response to normalized model events.
///
/// This is used by compatibility profiles whose strict structured-output mode
/// forbids streaming. The response must contain exactly one choice at index
/// zero and a complete assistant message.
///
/// # Errors
///
/// Returns an error when the response envelope, assistant message, tool calls,
/// finish reason, or usage payload is malformed.
#[allow(clippy::too_many_lines)]
pub fn map_chat_completion_response(value: &Value) -> Result<Vec<ModelEvent>, OpenAiError> {
    if let Some(error) = value.get("error") {
        return Ok(vec![
            ModelEvent::Started(ModelResponseInfo::new()),
            ModelEvent::Failed(map_stream_error(error)),
        ]);
    }

    let response_id = value
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| malformed("Chat Completions response id is missing"))?
        .parse::<ProviderResponseId>()
        .map_err(|_| malformed("Chat Completions response id is invalid"))?;
    let response_model = value
        .get("model")
        .and_then(Value::as_str)
        .ok_or_else(|| malformed("Chat Completions response model is missing"))?
        .parse::<ModelId>()
        .map_err(|_| malformed("Chat Completions response model is invalid"))?;
    let choices = value
        .get("choices")
        .and_then(Value::as_array)
        .ok_or_else(|| malformed("Chat Completions response choices are missing"))?;
    if choices.len() != 1 {
        return Err(malformed(
            "Chat Completions response must contain exactly one choice",
        ));
    }
    let choice = &choices[0];
    if choice.get("index").and_then(Value::as_u64) != Some(0) {
        return Err(malformed(
            "Chat Completions response choice index is missing or unsupported",
        ));
    }
    let message = choice
        .get("message")
        .and_then(Value::as_object)
        .ok_or_else(|| malformed("Chat Completions assistant message is missing"))?;
    if message.get("role").and_then(Value::as_str) != Some("assistant") {
        return Err(malformed(
            "Chat Completions response message role is not assistant",
        ));
    }

    let reasoning = optional_message_text(message, "reasoning")?;
    let content = optional_message_text(message, "content")?;
    let refusal = optional_message_text(message, "refusal")?;
    if content.map_or(0, str::len) + refusal.map_or(0, str::len) > MAX_TEXT_BLOCK_BYTES {
        return Err(malformed(
            "Chat Completions assistant text exceeds the supported bound",
        ));
    }
    if message
        .get("function_call")
        .is_some_and(|value| !value.is_null())
    {
        return Err(malformed(
            "Chat Completions legacy function_call response is unsupported",
        ));
    }
    let tool_calls = parse_complete_tool_calls(message.get("tool_calls"))?;
    if refusal.is_some() && !tool_calls.is_empty() {
        return Err(malformed(
            "Chat Completions response combines refusal and tool calls",
        ));
    }

    let raw_finish = choice
        .get("finish_reason")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| malformed("Chat Completions finish_reason is missing"))?;
    let finish_is_tool_use = matches!(raw_finish, "tool_calls" | "function_call");
    if finish_is_tool_use == tool_calls.is_empty() {
        return Err(malformed(
            "Chat Completions tool calls do not match finish_reason",
        ));
    }
    let stop_reason = if refusal.is_some() {
        StopReason::Refusal
    } else {
        match raw_finish {
            "stop" => StopReason::Completed,
            "length" => StopReason::Length,
            "tool_calls" | "function_call" => StopReason::ToolUse,
            "refusal" => StopReason::Refusal,
            other => StopReason::Unknown(other.to_owned()),
        }
    };
    let usage = parse_complete_usage(value.get("usage"))?;

    let mut events = vec![ModelEvent::Started(
        ModelResponseInfo::new()
            .with_response_id(response_id)
            .with_response_model(response_model),
    )];
    append_text_events(reasoning, true, &mut events)?;
    append_text_events(content, false, &mut events)?;
    append_text_events(refusal, false, &mut events)?;
    for call in tool_calls {
        let stream_index = ModelStreamIndex::new(call.index)
            .map_err(|_| malformed("Chat Completions tool call index is out of range"))?;
        let provider_id = call
            .id
            .parse::<ProviderToolCallId>()
            .map_err(|_| malformed("Chat Completions tool call id is invalid"))?;
        events.push(ModelEvent::ToolCallStarted(
            ToolCallStarted::new(stream_index, provider_id.clone(), &call.name)
                .map_err(|_| malformed("Chat Completions tool call identity is invalid"))?,
        ));
        for chunk in utf8_chunks(&call.arguments) {
            events.push(ModelEvent::ToolArgumentsDelta(
                ToolArgumentsDelta::new(stream_index, provider_id.clone(), chunk.to_owned())
                    .map_err(|_| malformed("Chat Completions tool arguments are invalid"))?,
            ));
        }
        events.push(ModelEvent::ToolCallCompleted(
            ToolCallCompleted::new(stream_index, provider_id, call.name, call.parsed_arguments)
                .map_err(|_| malformed("Chat Completions completed tool call is invalid"))?,
        ));
    }
    let mut completion = ModelCompletion::new(stop_reason)
        .map_err(|_| malformed("Chat Completions finish_reason is invalid"))?;
    if let Some(usage) = usage {
        completion = completion.with_usage(usage);
    }
    events.push(ModelEvent::Completed(completion));
    Ok(events)
}

#[derive(Debug)]
struct CompleteToolCall {
    index: u16,
    id: String,
    name: String,
    arguments: String,
    parsed_arguments: Value,
}

fn optional_message_text<'a>(
    message: &'a serde_json::Map<String, Value>,
    key: &str,
) -> Result<Option<&'a str>, OpenAiError> {
    match message.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) if value.len() <= MAX_TEXT_BLOCK_BYTES => Ok(Some(value)),
        Some(Value::String(_)) => Err(malformed(&format!(
            "Chat Completions assistant {key} exceeds the supported bound"
        ))),
        Some(_) => Err(malformed(&format!(
            "Chat Completions assistant {key} is not text"
        ))),
    }
}

fn parse_complete_tool_calls(value: Option<&Value>) -> Result<Vec<CompleteToolCall>, OpenAiError> {
    let Some(value) = value.filter(|value| !value.is_null()) else {
        return Ok(Vec::new());
    };
    let calls = value
        .as_array()
        .ok_or_else(|| malformed("Chat Completions tool_calls is not an array"))?;
    let mut ids = BTreeSet::new();
    calls
        .iter()
        .enumerate()
        .map(|(index, call)| {
            let index = u16::try_from(index)
                .map_err(|_| malformed("Chat Completions tool call index is out of range"))?;
            if call
                .get("index")
                .is_some_and(|value| value.as_u64() != Some(u64::from(index)))
            {
                return Err(malformed("Chat Completions tool call index changed"));
            }
            if call.get("type").and_then(Value::as_str) != Some("function") {
                return Err(malformed("Chat Completions tool call type is invalid"));
            }
            let id = call
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| malformed("Chat Completions tool call id is missing"))?
                .to_owned();
            if !ids.insert(id.clone()) {
                return Err(malformed("Chat Completions tool call id is duplicated"));
            }
            let function = call
                .get("function")
                .and_then(Value::as_object)
                .ok_or_else(|| malformed("Chat Completions tool call function is missing"))?;
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .ok_or_else(|| malformed("Chat Completions tool call name is missing"))?
                .to_owned();
            let arguments = function
                .get("arguments")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .ok_or_else(|| malformed("Chat Completions tool arguments are missing"))?
                .to_owned();
            let parsed_arguments = serde_json::from_str(&arguments)
                .map_err(|_| malformed("Chat Completions tool arguments did not parse"))?;
            Ok(CompleteToolCall {
                index,
                id,
                name,
                arguments,
                parsed_arguments,
            })
        })
        .collect()
}

fn append_text_events(
    value: Option<&str>,
    thinking: bool,
    events: &mut Vec<ModelEvent>,
) -> Result<(), OpenAiError> {
    let Some(value) = value else { return Ok(()) };
    for chunk in utf8_chunks(value) {
        let delta = Utf8Delta::new(chunk.to_owned())
            .map_err(|_| malformed("Chat Completions assistant text is invalid"))?;
        events.push(if thinking {
            ModelEvent::ThinkingDelta(delta)
        } else {
            ModelEvent::TextDelta(delta)
        });
    }
    Ok(())
}

fn utf8_chunks(value: &str) -> impl Iterator<Item = &str> {
    let mut offset = 0;
    std::iter::from_fn(move || {
        if offset == value.len() {
            return None;
        }
        let mut end = offset
            .saturating_add(MAX_MODEL_DELTA_BYTES)
            .min(value.len());
        while !value.is_char_boundary(end) {
            end -= 1;
        }
        let chunk = &value[offset..end];
        offset = end;
        Some(chunk)
    })
}

fn parse_complete_usage(value: Option<&Value>) -> Result<Option<Usage>, OpenAiError> {
    let value = value
        .filter(|value| !value.is_null())
        .ok_or_else(|| malformed("Chat Completions usage is missing"))?;
    let usage = value
        .as_object()
        .ok_or_else(|| malformed("Chat Completions usage is not an object"))?;
    let prompt = required_usage_count(usage, "prompt_tokens")?;
    let output = required_usage_count(usage, "completion_tokens")?;
    if let Some(total) = optional_usage_count(usage, "total_tokens")?
        && total
            != prompt
                .checked_add(output)
                .ok_or_else(|| malformed("Chat Completions usage total overflowed"))?
    {
        return Err(malformed("Chat Completions usage total is inconsistent"));
    }
    let prompt_details = optional_usage_object(usage, "prompt_tokens_details")?;
    let completion_details = optional_usage_object(usage, "completion_tokens_details")?;
    let cached = prompt_details
        .map(|details| optional_usage_count(details, "cached_tokens"))
        .transpose()?
        .flatten();
    let cache_write = prompt_details
        .map(|details| optional_usage_count(details, "cache_write_tokens"))
        .transpose()?
        .flatten();
    let reasoning = completion_details
        .map(|details| optional_usage_count(details, "reasoning_tokens"))
        .transpose()?
        .flatten();
    let billable_input = prompt
        .checked_sub(cached.unwrap_or(0))
        .and_then(|value| value.checked_sub(cache_write.unwrap_or(0)))
        .ok_or_else(|| malformed("Chat Completions cache usage exceeds prompt tokens"))?;
    let mut normalized = Usage::new(
        TokenCount::new(billable_input)
            .map_err(|_| malformed("Chat Completions prompt token count is invalid"))?,
        TokenCount::new(output)
            .map_err(|_| malformed("Chat Completions completion token count is invalid"))?,
    );
    if let Some(cached) = cached {
        normalized = normalized.with_cache_read(
            TokenCount::new(cached)
                .map_err(|_| malformed("Chat Completions cached token count is invalid"))?,
        );
    }
    if let Some(cache_write) = cache_write {
        normalized = normalized.with_cache_write(
            TokenCount::new(cache_write)
                .map_err(|_| malformed("Chat Completions cache-write token count is invalid"))?,
        );
    }
    if let Some(reasoning) = reasoning {
        normalized = normalized
            .with_reasoning(
                TokenCount::new(reasoning)
                    .map_err(|_| malformed("Chat Completions reasoning token count is invalid"))?,
            )
            .map_err(|_| malformed("Chat Completions reasoning usage exceeds output tokens"))?;
    }
    normalized
        .total_tokens()
        .map_err(|_| malformed("Chat Completions usage total is invalid"))?;
    Ok(Some(normalized))
}

fn required_usage_count(
    usage: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<u64, OpenAiError> {
    usage.get(key).and_then(Value::as_u64).ok_or_else(|| {
        malformed(&format!(
            "Chat Completions usage {key} is missing or invalid"
        ))
    })
}

fn optional_usage_count(
    usage: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<Option<u64>, OpenAiError> {
    match usage.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_u64()
            .map(Some)
            .ok_or_else(|| malformed(&format!("Chat Completions usage {key} is invalid"))),
    }
}

fn optional_usage_object<'a>(
    usage: &'a serde_json::Map<String, Value>,
    key: &str,
) -> Result<Option<&'a serde_json::Map<String, Value>>, OpenAiError> {
    match usage.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Object(value)) => Ok(Some(value)),
        Some(_) => Err(malformed(&format!(
            "Chat Completions usage {key} is invalid"
        ))),
    }
}

/// Maps an HTTP error status / transport failure to a normalized `ModelFailure`.
#[must_use]
pub fn map_http_failure(status: u16, body: &str) -> ModelFailure {
    let (code, retry) = match status {
        401 => (ModelFailureCode::Authentication, RetryClass::Never),
        403 => (ModelFailureCode::PermissionDenied, RetryClass::Never),
        400 => (ModelFailureCode::InvalidRequest, RetryClass::Never),
        429 => (ModelFailureCode::RateLimited, RetryClass::AfterBackoff),
        408 => (ModelFailureCode::Transport, RetryClass::Immediate),
        500..=599 if status != 501 => (ModelFailureCode::Unavailable, RetryClass::AfterBackoff),
        _ => (ModelFailureCode::Internal, RetryClass::Never),
    };
    ModelFailure::safe(code, normalize_provider_error(Some(status), body), retry)
        .unwrap_or_else(|_| ModelFailure::internal_adapter_failure())
}

/// Maps an `OpenAI` streaming `error` payload to a normalized `ModelFailure`.
#[must_use]
pub fn map_stream_error(error: &Value) -> ModelFailure {
    let serialized = serde_json::to_string(error).unwrap_or_default();
    let message = normalize_provider_error(None, &serialized);
    let error_type = error.get("type").and_then(Value::as_str).unwrap_or("");
    let (code, retry) = match error_type {
        "server_error" | "overloaded" => (ModelFailureCode::Unavailable, RetryClass::AfterBackoff),
        "rate_limit_exceeded" => (ModelFailureCode::RateLimited, RetryClass::AfterBackoff),
        _ => (ModelFailureCode::MalformedResponse, RetryClass::Never),
    };
    ModelFailure::safe(code, message, retry)
        .unwrap_or_else(|_| ModelFailure::internal_adapter_failure())
}

fn malformed(message: &str) -> OpenAiError {
    OpenAiError::new(OpenAiErrorCode::MalformedResponse, message)
}
