//! Maps `OpenAI` Responses SSE events to normalized model events.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

use serde_json::Value;
use tea_model::{
    HostedToolCompleted, HostedToolStarted, ModelCompletion, ModelEvent, ModelFailure,
    ModelFailureCode, ModelResponseInfo, ModelSourceCitation, ModelStreamIndex, ProviderResponseId,
    ProviderToolCallId, ToolArgumentsDelta, ToolCallCompleted, ToolCallStarted, Utf8Delta,
};
use tea_protocol::{
    ExternalSource, HostedToolError, HostedToolOutcome, MAX_PROVIDER_CONTINUATION_BYTES, ModelId,
    ProviderContinuation, RetryClass, SourceCitation, StopReason, TokenCount, Usage,
};

use crate::credential::PROVIDER_ID;
use crate::error::{OpenAiError, OpenAiErrorCode};
use crate::responses::{URL_CITATION_CONTINUATION_FORMAT, WEB_SEARCH_CONTINUATION_FORMAT};

const MAX_PENDING_CITATIONS: usize = 256;

#[derive(Debug, Clone)]
struct ToolCallAccumulator {
    index: u16,
    provider_call_id: String,
    name: String,
    arguments: String,
    completed: bool,
}

#[derive(Clone)]
struct HostedSearchAccumulator {
    provider_call_id: String,
    completed: bool,
    source_urls: BTreeSet<String>,
    completed_item: Option<Value>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum OutputItemKind {
    FunctionCall,
    HostedSearch,
    Message,
    Reasoning,
    Other(String),
}

impl OutputItemKind {
    const fn label(&self) -> &'static str {
        match self {
            Self::FunctionCall => "function_call",
            Self::HostedSearch => "web_search_call",
            Self::Message => "message",
            Self::Reasoning => "reasoning",
            Self::Other(_) => "other",
        }
    }

    const fn is_non_executable_text(&self) -> bool {
        matches!(self, Self::Message | Self::Reasoning)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputIndexCompatibility {
    Strict,
    ReusedNonExecutable,
}

/// Stateful reducer for one `OpenAI` Responses stream.
pub struct ResponsesReducer {
    started: bool,
    terminal_emitted: bool,
    saw_tool_call: bool,
    output_index_compatibility: OutputIndexCompatibility,
    output_item_kinds: BTreeMap<u16, OutputItemKind>,
    output_item_indices: BTreeMap<String, u16>,
    tool_calls: BTreeMap<u16, ToolCallAccumulator>,
    hosted_searches: BTreeMap<u16, HostedSearchAccumulator>,
    text_by_index: BTreeMap<u16, String>,
    text_by_content: BTreeMap<(u16, u16), String>,
    emitted_text: String,
    reasoning_by_index: BTreeMap<(u16, u16), String>,
    citation_payloads: BTreeMap<(u16, u16, u16), Value>,
    pending_citations: BTreeMap<(u16, u16, u16), SourceCitation>,
    pending_citation_bytes: usize,
    provider_id: String,
}

impl fmt::Debug for ResponsesReducer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ResponsesReducer")
            .field("started", &self.started)
            .field("terminal_emitted", &self.terminal_emitted)
            .field("saw_tool_call", &self.saw_tool_call)
            .field(
                "output_index_compatibility",
                &self.output_index_compatibility,
            )
            .field("output_item_kind_count", &self.output_item_kinds.len())
            .field(
                "output_item_identity_count",
                &self.output_item_indices.len(),
            )
            .field("tool_call_count", &self.tool_calls.len())
            .field("hosted_search_count", &self.hosted_searches.len())
            .field("text_output_count", &self.text_by_index.len())
            .field("text_content_count", &self.text_by_content.len())
            .field("emitted_text_bytes", &self.emitted_text.len())
            .field("reasoning_output_count", &self.reasoning_by_index.len())
            .field("citation_payload_count", &self.citation_payloads.len())
            .field("pending_citation_count", &self.pending_citations.len())
            .field("pending_citation_bytes", &self.pending_citation_bytes)
            .finish_non_exhaustive()
    }
}

impl Default for ResponsesReducer {
    fn default() -> Self {
        Self {
            started: false,
            terminal_emitted: false,
            saw_tool_call: false,
            output_index_compatibility: OutputIndexCompatibility::Strict,
            output_item_kinds: BTreeMap::new(),
            output_item_indices: BTreeMap::new(),
            tool_calls: BTreeMap::new(),
            hosted_searches: BTreeMap::new(),
            text_by_index: BTreeMap::new(),
            text_by_content: BTreeMap::new(),
            emitted_text: String::new(),
            reasoning_by_index: BTreeMap::new(),
            citation_payloads: BTreeMap::new(),
            pending_citations: BTreeMap::new(),
            pending_citation_bytes: 0,
            provider_id: PROVIDER_ID.to_owned(),
        }
    }
}

impl ResponsesReducer {
    /// Creates an empty Responses reducer.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub(crate) fn for_provider(provider_id: impl Into<String>) -> Self {
        Self {
            provider_id: provider_id.into(),
            ..Self::default()
        }
    }

    /// Returns whether a terminal event has already been emitted.
    #[must_use]
    pub const fn terminal_emitted(&self) -> bool {
        self.terminal_emitted
    }

    /// Maps one parsed Responses event to normalized model events.
    ///
    /// # Errors
    ///
    /// Returns an error when an event violates the Responses stream contract.
    #[allow(clippy::too_many_lines)]
    pub fn map_chunk(&mut self, value: &Value) -> Result<Vec<ModelEvent>, OpenAiError> {
        if self.terminal_emitted {
            return Ok(Vec::new());
        }
        let mut events = Vec::new();
        let kind = value.get("type").and_then(Value::as_str).unwrap_or("");
        let response = value.get("response");
        if let Some(started) = self.ensure_started(response)? {
            events.push(started);
        }
        match kind {
            "response.created" => {}
            "response.output_item.added" => {
                let raw_index = output_index(value)?;
                let item = value
                    .get("item")
                    .ok_or_else(|| malformed("Responses output item missing"))?;
                let index = self.register_output_item_identity(raw_index, item)?;
                let kind = output_item_kind(item)?;
                self.register_output_item_kind(index, kind.clone())?;
                match kind {
                    OutputItemKind::FunctionCall => {
                        self.start_tool_call(index, item, &mut events)?;
                    }
                    OutputItemKind::HostedSearch => {
                        self.start_hosted_search(index, item, &mut events)?;
                    }
                    OutputItemKind::Message => {
                        self.text_by_index.entry(index).or_default();
                    }
                    OutputItemKind::Reasoning => {
                        self.reasoning_by_index.entry((index, 0)).or_default();
                    }
                    OutputItemKind::Other(_) => {}
                }
            }
            "response.output_text.delta" | "response.refusal.delta" => {
                let index = self.resolve_output_index(value)?;
                self.register_output_item_kind(index, OutputItemKind::Message)?;
                let content_index = bounded_event_index(value, "content_index")?;
                if let Some(delta) = nonempty_string(value, "delta") {
                    self.append_text(index, content_index, delta, &mut events)?;
                }
            }
            "response.output_text.done" => {
                let index = self.resolve_output_index(value)?;
                self.register_output_item_kind(index, OutputItemKind::Message)?;
                let content_index = bounded_event_index(value, "content_index")?;
                if let Some(text) = value.get("text").and_then(Value::as_str) {
                    self.reconcile_text(index, content_index, text, &mut events)?;
                }
            }
            "response.output_text.annotation.added" => {
                let index = self.resolve_output_index(value)?;
                self.register_output_item_kind(index, OutputItemKind::Message)?;
                let content_index = bounded_event_index(value, "content_index")?;
                let annotation_index = bounded_event_index(value, "annotation_index")?;
                let annotation = value
                    .get("annotation")
                    .ok_or_else(|| malformed("Responses annotation missing"))?;
                self.emit_annotation(
                    index,
                    content_index,
                    annotation_index,
                    annotation,
                    &mut events,
                )?;
            }
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => {
                let index = self.resolve_output_index(value)?;
                let summary_index = value
                    .get("summary_index")
                    .map(|_| bounded_event_index(value, "summary_index"))
                    .transpose()?
                    .unwrap_or(0);
                self.register_output_item_kind(index, OutputItemKind::Reasoning)?;
                if let Some(delta) = nonempty_string(value, "delta") {
                    self.reasoning_by_index
                        .entry((index, summary_index))
                        .or_default()
                        .push_str(delta);
                    events.push(ModelEvent::ThinkingDelta(
                        Utf8Delta::new(delta.to_owned())
                            .map_err(|_| malformed("Responses reasoning delta"))?,
                    ));
                }
            }
            "response.reasoning_summary_text.done" => {
                let index = self.resolve_output_index(value)?;
                self.register_output_item_kind(index, OutputItemKind::Reasoning)?;
                if let Some(text) = value.get("text").and_then(Value::as_str) {
                    let summary_index = bounded_event_index(value, "summary_index")?;
                    self.reconcile_reasoning(index, summary_index, text, &mut events)?;
                }
            }
            "response.function_call_arguments.delta" => {
                let index = self.resolve_output_index(value)?;
                if let Some(delta) = nonempty_string(value, "delta") {
                    self.append_tool_arguments(index, delta, &mut events)?;
                }
            }
            "response.function_call_arguments.done" => {
                let index = self.resolve_output_index(value)?;
                if let Some(arguments) = value.get("arguments").and_then(Value::as_str) {
                    self.reconcile_tool_arguments(index, arguments, &mut events)?;
                }
            }
            "response.web_search_call.in_progress"
            | "response.web_search_call.searching"
            | "response.web_search_call.completed" => {
                let index = self.resolve_output_index(value)?;
                let item_id = value
                    .get("item_id")
                    .and_then(Value::as_str)
                    .ok_or_else(|| malformed("Responses web-search item id missing"))?;
                self.start_hosted_search(index, &serde_json::json!({"id": item_id}), &mut events)?;
            }
            "response.output_item.done" => {
                let raw_index = output_index(value)?;
                let item = value
                    .get("item")
                    .ok_or_else(|| malformed("Responses completed output item missing"))?;
                let index = self.register_output_item_identity(raw_index, item)?;
                self.complete_output_item(index, item, &mut events)?;
            }
            "response.completed" => {
                let response = response.ok_or_else(|| malformed("completed response missing"))?;
                self.complete_response_output(response, &mut events)?;
                self.ensure_output_items_complete()?;
                self.flush_pending_citations(&mut events, true)?;
                let mut completion = ModelCompletion::new(if self.saw_tool_call {
                    StopReason::ToolUse
                } else {
                    StopReason::Completed
                })
                .map_err(|_| malformed("Responses completion reason"))?;
                if let Some(usage) = parse_usage(response)? {
                    completion = completion.with_usage(usage);
                }
                self.terminal_emitted = true;
                events.push(ModelEvent::Completed(completion));
            }
            "response.failed" => {
                let response = response.unwrap_or(value);
                self.terminal_emitted = true;
                events.push(ModelEvent::Failed(map_response_failure(response, false)));
            }
            "response.incomplete" => {
                let response = response.unwrap_or(value);
                self.terminal_emitted = true;
                events.push(ModelEvent::Failed(map_response_failure(response, true)));
            }
            "error" => {
                self.terminal_emitted = true;
                events.push(ModelEvent::Failed(map_response_failure(value, false)));
            }
            _ if value.get("error").is_some() => {
                self.terminal_emitted = true;
                events.push(ModelEvent::Failed(map_response_failure(value, false)));
            }
            _ => {}
        }
        Ok(events)
    }

    /// Produces a terminal failure when the SSE transport ends before a
    /// Responses terminal event.
    ///
    /// # Errors
    ///
    /// This method currently has no error path; it returns `Result` to match
    /// the Chat Completions reducer contract.
    pub fn finish(&mut self) -> Result<Option<ModelEvent>, OpenAiError> {
        if self.terminal_emitted {
            return Ok(None);
        }
        self.terminal_emitted = true;
        Ok(Some(ModelEvent::Failed(
            ModelFailure::new(
                ModelFailureCode::MalformedResponse,
                "Responses stream ended before a terminal event",
                RetryClass::Never,
            )
            .unwrap_or_else(|_| ModelFailure::internal_adapter_failure()),
        )))
    }

    fn ensure_started(
        &mut self,
        response: Option<&Value>,
    ) -> Result<Option<ModelEvent>, OpenAiError> {
        if self.started {
            return Ok(None);
        }
        let mut info = ModelResponseInfo::new();
        if let Some(response) = response {
            if let Some(id) = response.get("id").and_then(Value::as_str) {
                let id = ProviderResponseId::from_str(id)
                    .map_err(|_| malformed("Responses response id"))?;
                info = info.with_response_id(id);
            }
            if let Some(model) = response.get("model").and_then(Value::as_str) {
                let model =
                    ModelId::from_str(model).map_err(|_| malformed("Responses response model"))?;
                info = info.with_response_model(model);
            }
        }
        self.started = true;
        Ok(Some(ModelEvent::Started(info)))
    }

    fn register_output_item_kind(
        &mut self,
        index: u16,
        kind: OutputItemKind,
    ) -> Result<(), OpenAiError> {
        if let Some(existing) = self.output_item_kinds.get(&index) {
            if *existing != kind {
                if existing.is_non_executable_text() && kind.is_non_executable_text() {
                    self.output_index_compatibility = OutputIndexCompatibility::ReusedNonExecutable;
                    return Ok(());
                }
                return Err(malformed(&format!(
                    "Responses output item {index} changed type from {} to {}",
                    existing.label(),
                    kind.label()
                )));
            }
            return Ok(());
        }
        self.output_item_kinds.insert(index, kind);
        Ok(())
    }

    fn register_output_item_identity(
        &mut self,
        raw_index: u16,
        item: &Value,
    ) -> Result<u16, OpenAiError> {
        let Some(item_id) = item.get("id").and_then(Value::as_str) else {
            return Ok(raw_index);
        };
        if let Some(existing) = self.output_item_indices.get(item_id) {
            return Ok(*existing);
        }
        let index = if self
            .output_item_indices
            .values()
            .any(|index| *index == raw_index)
        {
            self.next_output_item_index(raw_index)?
        } else {
            raw_index
        };
        self.output_item_indices.insert(item_id.to_owned(), index);
        Ok(index)
    }

    fn resolve_output_index(&self, value: &Value) -> Result<u16, OpenAiError> {
        let raw = output_index(value)?;
        Ok(value
            .get("item_id")
            .and_then(Value::as_str)
            .and_then(|item_id| self.output_item_indices.get(item_id).copied())
            .unwrap_or(raw))
    }

    fn next_output_item_index(&self, raw_index: u16) -> Result<u16, OpenAiError> {
        let mut candidate = raw_index;
        loop {
            candidate = candidate
                .checked_add(1)
                .ok_or_else(|| malformed("Responses output index out of range"))?;
            if !self
                .output_item_indices
                .values()
                .any(|index| *index == candidate)
                && !self.output_item_kinds.contains_key(&candidate)
            {
                let _ = stream_index(candidate)?;
                return Ok(candidate);
            }
        }
    }

    fn start_tool_call(
        &mut self,
        index: u16,
        item: &Value,
        events: &mut Vec<ModelEvent>,
    ) -> Result<(), OpenAiError> {
        let call_id = item
            .get("call_id")
            .and_then(Value::as_str)
            .ok_or_else(|| malformed("Responses function call id missing"))?;
        let name = item
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| malformed("Responses function name missing"))?;
        self.register_output_item_kind(index, OutputItemKind::FunctionCall)?;
        if let Some(existing) = self.tool_calls.get(&index) {
            if existing.provider_call_id != call_id || existing.name != name {
                return Err(malformed("Responses function call changed identity"));
            }
            return Ok(());
        }
        let provider_id = ProviderToolCallId::from_str(call_id)
            .map_err(|_| malformed("Responses function call id"))?;
        let stream_index = stream_index(index)?;
        let arguments = item
            .get("arguments")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        self.tool_calls.insert(
            index,
            ToolCallAccumulator {
                index,
                provider_call_id: call_id.to_owned(),
                name: name.to_owned(),
                arguments,
                completed: false,
            },
        );
        self.saw_tool_call = true;
        events.push(ModelEvent::ToolCallStarted(
            ToolCallStarted::new(stream_index, provider_id, name)
                .map_err(|_| malformed("Responses function call start"))?,
        ));
        Ok(())
    }

    fn append_tool_arguments(
        &mut self,
        index: u16,
        delta: &str,
        events: &mut Vec<ModelEvent>,
    ) -> Result<(), OpenAiError> {
        let accumulator = self
            .tool_calls
            .get_mut(&index)
            .ok_or_else(|| malformed("Responses arguments without function call"))?;
        if accumulator.completed {
            return Err(malformed(
                "Responses arguments after completed function call",
            ));
        }
        accumulator.arguments.push_str(delta);
        let provider_id = ProviderToolCallId::from_str(&accumulator.provider_call_id)
            .map_err(|_| malformed("Responses function call id"))?;
        events.push(ModelEvent::ToolArgumentsDelta(
            ToolArgumentsDelta::new(stream_index(index)?, provider_id, delta.to_owned())
                .map_err(|_| malformed("Responses function arguments delta"))?,
        ));
        Ok(())
    }

    fn reconcile_tool_arguments(
        &mut self,
        index: u16,
        arguments: &str,
        events: &mut Vec<ModelEvent>,
    ) -> Result<(), OpenAiError> {
        let current = self
            .tool_calls
            .get(&index)
            .ok_or_else(|| malformed("Responses arguments without function call"))?
            .arguments
            .clone();
        if arguments == current {
            return Ok(());
        }
        let suffix = arguments
            .strip_prefix(&current)
            .ok_or_else(|| malformed("Responses function arguments changed"))?;
        if !suffix.is_empty() {
            self.append_tool_arguments(index, suffix, events)?;
        }
        Ok(())
    }

    fn start_hosted_search(
        &mut self,
        index: u16,
        item: &Value,
        events: &mut Vec<ModelEvent>,
    ) -> Result<(), OpenAiError> {
        let call_id = item
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| malformed("Responses web-search call id missing"))?;
        self.register_output_item_kind(index, OutputItemKind::HostedSearch)?;
        if let Some(existing) = self.hosted_searches.get(&index) {
            if existing.provider_call_id != call_id {
                return Err(malformed("Responses web-search call changed identity"));
            }
            return Ok(());
        }
        let provider_id = ProviderToolCallId::from_str(call_id)
            .map_err(|_| malformed("Responses web-search call id"))?;
        self.hosted_searches.insert(
            index,
            HostedSearchAccumulator {
                provider_call_id: call_id.to_owned(),
                completed: false,
                source_urls: BTreeSet::new(),
                completed_item: None,
            },
        );
        events.push(ModelEvent::HostedToolStarted(
            HostedToolStarted::new(stream_index(index)?, provider_id, "web_search")
                .map_err(|_| malformed("Responses web-search call start"))?,
        ));
        Ok(())
    }

    fn complete_hosted_search(
        &mut self,
        index: u16,
        item: &Value,
        events: &mut Vec<ModelEvent>,
    ) -> Result<(), OpenAiError> {
        self.start_hosted_search(index, item, events)?;
        if let Some(search) = self
            .hosted_searches
            .get(&index)
            .filter(|search| search.completed)
        {
            if search.completed_item.as_ref() == Some(item) {
                return Ok(());
            }
            return Err(malformed(
                "Responses completed web-search call changed payload",
            ));
        }

        let status = item
            .get("status")
            .and_then(Value::as_str)
            .ok_or_else(|| malformed("Responses web-search status missing"))?;
        let outcome = match status {
            "completed" => HostedToolOutcome::Success,
            "failed" => HostedToolOutcome::Error(
                HostedToolError::new("provider_failed", "OpenAI web search call failed")
                    .map_err(|_| malformed("Responses web-search failure"))?,
            ),
            _ => {
                return Err(malformed(
                    "Responses web-search call completed without terminal status",
                ));
            }
        };
        let action = match item.get("action") {
            Some(Value::Object(action)) => Value::Object(action.clone()),
            Some(Value::Null) | None if matches!(outcome, HostedToolOutcome::Error(_)) => {
                Value::Object(serde_json::Map::new())
            }
            _ => return Err(malformed("Responses web-search action missing")),
        };
        let sources = parse_web_search_sources(&action)?;
        let source_urls = sources
            .iter()
            .map(|source| source.url().to_owned())
            .collect();
        let mut arguments = action;
        arguments
            .as_object_mut()
            .expect("web-search action was normalized as an object")
            .remove("sources");
        let continuation = ProviderContinuation::new(
            self.provider_id.clone(),
            WEB_SEARCH_CONTINUATION_FORMAT,
            item.clone(),
        )
        .map_err(|_| malformed("Responses web-search continuation exceeded bounds"))?;
        let provider_call_id = self
            .hosted_searches
            .get(&index)
            .expect("hosted search was started above")
            .provider_call_id
            .clone();
        let provider_id = ProviderToolCallId::from_str(&provider_call_id)
            .map_err(|_| malformed("Responses web-search call id"))?;
        let completed = HostedToolCompleted::new(
            stream_index(index)?,
            provider_id,
            "web_search",
            arguments,
            outcome,
            sources,
            Some(continuation),
        )
        .map_err(|_| malformed("Responses completed web-search call"))?;
        let accumulator = self
            .hosted_searches
            .get_mut(&index)
            .expect("hosted search was started above");
        accumulator.completed = true;
        accumulator.source_urls = source_urls;
        accumulator.completed_item = Some(item.clone());
        events.push(ModelEvent::HostedToolCompleted(completed));
        self.flush_pending_citations(events, false)
    }

    fn complete_output_item(
        &mut self,
        index: u16,
        item: &Value,
        events: &mut Vec<ModelEvent>,
    ) -> Result<(), OpenAiError> {
        let kind = output_item_kind(item)?;
        self.register_output_item_kind(index, kind.clone())?;
        match kind {
            OutputItemKind::FunctionCall => {
                self.start_tool_call(index, item, events)?;
                if let Some(arguments) = item.get("arguments").and_then(Value::as_str) {
                    self.reconcile_tool_arguments(index, arguments, events)?;
                }
                self.complete_tool_call(index, events)?;
            }
            OutputItemKind::HostedSearch => {
                self.complete_hosted_search(index, item, events)?;
            }
            OutputItemKind::Message => {
                self.reconcile_message_content(index, item, events)?;
                self.emit_message_annotations(index, item, events)?;
            }
            OutputItemKind::Reasoning => {
                if let Some(summary) = item.get("summary").and_then(Value::as_array) {
                    for (summary_index, summary) in summary.iter().enumerate() {
                        let summary_index = u16::try_from(summary_index).map_err(|_| {
                            malformed("Responses reasoning summary index out of range")
                        })?;
                        if let Some(text) = summary.get("text").and_then(Value::as_str)
                            && !text.is_empty()
                        {
                            self.reconcile_reasoning(index, summary_index, text, events)?;
                        }
                    }
                }
            }
            OutputItemKind::Other(_) => {}
        }
        Ok(())
    }

    fn reconcile_message_content(
        &mut self,
        index: u16,
        item: &Value,
        events: &mut Vec<ModelEvent>,
    ) -> Result<(), OpenAiError> {
        let mut complete_text = String::new();
        if let Some(content) = item.get("content").and_then(Value::as_array) {
            for (content_index, part) in content.iter().enumerate() {
                let content_index = u16::try_from(content_index)
                    .map_err(|_| malformed("Responses content index out of range"))?;
                let Some(text) = part
                    .get("text")
                    .or_else(|| part.get("refusal"))
                    .and_then(Value::as_str)
                else {
                    continue;
                };
                self.reconcile_text(index, content_index, text, events)?;
                complete_text.push_str(text);
            }
        }
        if self
            .text_by_index
            .get(&index)
            .is_some_and(|text| text != &complete_text)
        {
            return Err(malformed("Responses output text changed"));
        }
        Ok(())
    }

    fn emit_message_annotations(
        &mut self,
        index: u16,
        item: &Value,
        events: &mut Vec<ModelEvent>,
    ) -> Result<(), OpenAiError> {
        let Some(content) = item.get("content").and_then(Value::as_array) else {
            return Ok(());
        };
        for (content_index, part) in content.iter().enumerate() {
            let content_index = u16::try_from(content_index)
                .map_err(|_| malformed("Responses content index out of range"))?;
            let Some(annotations) = part.get("annotations").and_then(Value::as_array) else {
                continue;
            };
            for (annotation_index, annotation) in annotations.iter().enumerate() {
                let annotation_index = u16::try_from(annotation_index)
                    .map_err(|_| malformed("Responses annotation index out of range"))?;
                self.emit_annotation(index, content_index, annotation_index, annotation, events)?;
            }
        }
        Ok(())
    }

    fn emit_annotation(
        &mut self,
        index: u16,
        content_index: u16,
        annotation_index: u16,
        annotation: &Value,
        events: &mut Vec<ModelEvent>,
    ) -> Result<(), OpenAiError> {
        if annotation.get("type").and_then(Value::as_str) != Some("url_citation") {
            return Ok(());
        }
        let key = (index, content_index, annotation_index);
        if let Some(existing) = self.citation_payloads.get(&key) {
            if existing == annotation {
                return Ok(());
            }
            return Err(malformed("Responses URL citation changed payload"));
        }
        let url = annotation
            .get("url")
            .and_then(Value::as_str)
            .ok_or_else(|| malformed("Responses URL citation URL missing"))?;
        let mut source = ExternalSource::new(url)
            .map_err(|_| malformed("Responses URL citation URL invalid"))?;
        match annotation.get("title") {
            Some(Value::String(title)) => {
                source = source
                    .with_title(title)
                    .map_err(|_| malformed("Responses URL citation title invalid"))?;
            }
            Some(Value::Null) | None => {}
            _ => return Err(malformed("Responses URL citation title invalid")),
        }
        let start = annotation
            .get("start_index")
            .and_then(Value::as_u64)
            .ok_or_else(|| malformed("Responses URL citation start index missing"))?;
        let end = annotation
            .get("end_index")
            .and_then(Value::as_u64)
            .ok_or_else(|| malformed("Responses URL citation end index missing"))?;
        let start = usize::try_from(start)
            .map_err(|_| malformed("Responses URL citation range out of bounds"))?;
        let end = usize::try_from(end)
            .map_err(|_| malformed("Responses URL citation range out of bounds"))?;
        let text = self
            .text_by_content
            .get(&(index, content_index))
            .ok_or_else(|| malformed("Responses URL citation without output text"))?;
        let (local_start, local_end) = character_range_to_bytes(text, start, end)
            .ok_or_else(|| malformed("Responses URL citation range invalid"))?;
        let prefix_bytes = self
            .text_by_content
            .iter()
            .filter(|((output, content), _)| *output == index && *content < content_index)
            .try_fold(0_usize, |total, (_, text)| total.checked_add(text.len()))
            .ok_or_else(|| malformed("Responses URL citation range out of bounds"))?;
        let start = prefix_bytes
            .checked_add(local_start)
            .ok_or_else(|| malformed("Responses URL citation range out of bounds"))?;
        let end = prefix_bytes
            .checked_add(local_end)
            .ok_or_else(|| malformed("Responses URL citation range out of bounds"))?;
        let start_index = u32::try_from(start)
            .map_err(|_| malformed("Responses URL citation range out of bounds"))?;
        let end_index = u32::try_from(end)
            .map_err(|_| malformed("Responses URL citation range out of bounds"))?;
        let continuation = ProviderContinuation::new(
            self.provider_id.clone(),
            URL_CITATION_CONTINUATION_FORMAT,
            annotation.clone(),
        )
        .map_err(|_| malformed("Responses URL citation continuation exceeded bounds"))?;
        let citation = SourceCitation::new(source)
            .with_range(start_index, end_index)
            .and_then(|citation| citation.with_cited_text(&text[local_start..local_end]))
            .map(|citation| citation.with_continuation(continuation))
            .map_err(|_| malformed("Responses URL citation invalid"))?;
        self.emit_or_defer_citation(key, citation, events, false)?;
        self.citation_payloads.insert(key, annotation.clone());
        Ok(())
    }

    fn emit_or_defer_citation(
        &mut self,
        key: (u16, u16, u16),
        citation: SourceCitation,
        events: &mut Vec<ModelEvent>,
        finalize_unmatched: bool,
    ) -> Result<(), OpenAiError> {
        let provider_call_id = self.provider_call_for_source(citation.source().url())?;
        let hosted_search_pending = self
            .hosted_searches
            .values()
            .any(|search| !search.completed);
        if !finalize_unmatched && (provider_call_id.is_none() || hosted_search_pending) {
            let continuation_bytes = citation
                .continuation()
                .map(|continuation| serde_json::to_vec(continuation.payload()))
                .transpose()
                .map_err(|_| malformed("Responses URL citation continuation serialization"))?
                .map_or(0, |payload| payload.len());
            let pending_bytes = self
                .pending_citation_bytes
                .checked_add(continuation_bytes)
                .filter(|bytes| *bytes <= MAX_PROVIDER_CONTINUATION_BYTES)
                .ok_or_else(|| malformed("Responses pending URL citations exceeded bounds"))?;
            if self.pending_citations.len() >= MAX_PENDING_CITATIONS {
                return Err(malformed("Responses pending URL citations exceeded bounds"));
            }
            self.pending_citations.insert(key, citation);
            self.pending_citation_bytes = pending_bytes;
            return Ok(());
        }
        let citation = ModelSourceCitation::new(provider_call_id, citation)
            .map_err(|_| malformed("Responses URL citation invalid"))?;
        events.push(ModelEvent::SourceCitation(citation));
        Ok(())
    }

    fn flush_pending_citations(
        &mut self,
        events: &mut Vec<ModelEvent>,
        finalize_unmatched: bool,
    ) -> Result<(), OpenAiError> {
        let pending = std::mem::take(&mut self.pending_citations);
        self.pending_citation_bytes = 0;
        for (key, citation) in pending {
            self.emit_or_defer_citation(key, citation, events, finalize_unmatched)?;
        }
        Ok(())
    }

    fn provider_call_for_source(
        &self,
        url: &str,
    ) -> Result<Option<ProviderToolCallId>, OpenAiError> {
        let mut matching = self
            .hosted_searches
            .values()
            .filter(|search| search.completed && search.source_urls.contains(url));
        let Some(search) = matching.next() else {
            return Ok(None);
        };
        if matching.next().is_some() {
            return Ok(None);
        }
        ProviderToolCallId::from_str(&search.provider_call_id)
            .map(Some)
            .map_err(|_| malformed("Responses web-search call id"))
    }

    fn complete_tool_call(
        &mut self,
        index: u16,
        events: &mut Vec<ModelEvent>,
    ) -> Result<(), OpenAiError> {
        let accumulator = self
            .tool_calls
            .get_mut(&index)
            .ok_or_else(|| malformed("Responses completed function call missing"))?;
        if accumulator.completed {
            return Ok(());
        }
        let arguments = if accumulator.arguments.is_empty() {
            Value::Object(serde_json::Map::new())
        } else {
            serde_json::from_str(&accumulator.arguments)
                .map_err(|_| malformed("Responses function arguments did not parse"))?
        };
        let provider_id = ProviderToolCallId::from_str(&accumulator.provider_call_id)
            .map_err(|_| malformed("Responses function call id"))?;
        let stream_index = stream_index(accumulator.index)?;
        let name = accumulator.name.clone();
        accumulator.completed = true;
        events.push(ModelEvent::ToolCallCompleted(
            ToolCallCompleted::new(stream_index, provider_id, name, arguments)
                .map_err(|_| malformed("Responses completed function call"))?,
        ));
        Ok(())
    }

    fn reconcile_text(
        &mut self,
        index: u16,
        content_index: u16,
        text: &str,
        events: &mut Vec<ModelEvent>,
    ) -> Result<(), OpenAiError> {
        let reused_non_executable_index =
            self.output_index_compatibility == OutputIndexCompatibility::ReusedNonExecutable;
        let streamed_under_reused_index = reused_non_executable_index
            && !self.text_by_content.contains_key(&(index, content_index))
            && self
                .text_by_content
                .iter()
                .any(|((other_index, other_content_index), existing)| {
                    *other_index != index
                        && *other_content_index == content_index
                        && existing == text
                });
        let replayed_complete_stream = reused_non_executable_index
            && !self.emitted_text.is_empty()
            && self.emitted_text == text;
        if streamed_under_reused_index || replayed_complete_stream {
            self.text_by_content
                .insert((index, content_index), text.to_owned());
            self.text_by_index.insert(index, text.to_owned());
            return Ok(());
        }
        let current = self
            .text_by_content
            .entry((index, content_index))
            .or_default();
        let suffix = text
            .strip_prefix(current.as_str())
            .ok_or_else(|| malformed("Responses output text changed"))?;
        if !suffix.is_empty() {
            self.append_text(index, content_index, suffix, events)?;
        }
        Ok(())
    }

    fn append_text(
        &mut self,
        index: u16,
        content_index: u16,
        delta: &str,
        events: &mut Vec<ModelEvent>,
    ) -> Result<(), OpenAiError> {
        self.text_by_content
            .entry((index, content_index))
            .or_default()
            .push_str(delta);
        self.text_by_index.entry(index).or_default().push_str(delta);
        self.emitted_text.push_str(delta);
        events.push(ModelEvent::TextDelta(
            Utf8Delta::new(delta.to_owned()).map_err(|_| malformed("Responses text delta"))?,
        ));
        Ok(())
    }

    fn reconcile_reasoning(
        &mut self,
        index: u16,
        summary_index: u16,
        text: &str,
        events: &mut Vec<ModelEvent>,
    ) -> Result<(), OpenAiError> {
        let current = self
            .reasoning_by_index
            .entry((index, summary_index))
            .or_default();
        let suffix = text
            .strip_prefix(current.as_str())
            .ok_or_else(|| malformed("Responses reasoning text changed"))?;
        if !suffix.is_empty() {
            current.push_str(suffix);
            events.push(ModelEvent::ThinkingDelta(
                Utf8Delta::new(suffix.to_owned())
                    .map_err(|_| malformed("Responses reasoning delta"))?,
            ));
        }
        Ok(())
    }

    fn complete_response_output(
        &mut self,
        response: &Value,
        events: &mut Vec<ModelEvent>,
    ) -> Result<(), OpenAiError> {
        let Some(output) = response.get("output").and_then(Value::as_array) else {
            return Ok(());
        };
        for (index, item) in output.iter().enumerate() {
            let raw_index = u16::try_from(index)
                .map_err(|_| malformed("Responses output index out of range"))?;
            let index = self.terminal_output_item_index(raw_index, item)?;
            self.complete_output_item(index, item, events)?;
        }
        Ok(())
    }

    fn terminal_output_item_index(
        &mut self,
        raw_index: u16,
        item: &Value,
    ) -> Result<u16, OpenAiError> {
        let Some(item_id) = item.get("id").and_then(Value::as_str) else {
            return Ok(raw_index);
        };
        if let Some(existing) = self.output_item_indices.get(item_id) {
            return Ok(*existing);
        }
        let kind = output_item_kind(item)?;
        if self.output_item_kinds.get(&raw_index) == Some(&kind) {
            self.output_item_indices
                .insert(item_id.to_owned(), raw_index);
            return Ok(raw_index);
        }
        self.register_output_item_identity(raw_index, item)
    }

    fn ensure_output_items_complete(&self) -> Result<(), OpenAiError> {
        if self.tool_calls.values().any(|call| !call.completed) {
            return Err(malformed(
                "Responses stream ended with function call without completion",
            ));
        }
        if self
            .hosted_searches
            .values()
            .any(|search| !search.completed)
        {
            return Err(malformed(
                "Responses stream ended with web-search call without completion",
            ));
        }
        Ok(())
    }
}

fn parse_web_search_sources(action: &Value) -> Result<Vec<ExternalSource>, OpenAiError> {
    let Some(raw_sources) = action.get("sources") else {
        return Ok(Vec::new());
    };
    let raw_sources = raw_sources
        .as_array()
        .ok_or_else(|| malformed("Responses web-search sources are not an array"))?;
    raw_sources
        .iter()
        .filter(|source| source.get("type").and_then(Value::as_str) == Some("url"))
        .map(|source| {
            let url = source
                .get("url")
                .and_then(Value::as_str)
                .ok_or_else(|| malformed("Responses web-search source URL missing"))?;
            ExternalSource::new(url)
                .map_err(|_| malformed("Responses web-search source URL invalid"))
        })
        .collect()
}

fn output_item_kind(item: &Value) -> Result<OutputItemKind, OpenAiError> {
    match item.get("type").and_then(Value::as_str) {
        Some("function_call") => Ok(OutputItemKind::FunctionCall),
        Some("web_search_call") => Ok(OutputItemKind::HostedSearch),
        Some("message") => Ok(OutputItemKind::Message),
        Some("reasoning") => Ok(OutputItemKind::Reasoning),
        Some(other) => Ok(OutputItemKind::Other(other.to_owned())),
        None => Err(malformed("Responses output item type missing")),
    }
}

fn character_range_to_bytes(text: &str, start: usize, end: usize) -> Option<(usize, usize)> {
    if start >= end {
        return None;
    }
    let char_count = text.chars().count();
    if end > char_count {
        return None;
    }
    let start = if start == char_count {
        text.len()
    } else {
        text.char_indices().nth(start)?.0
    };
    let end = if end == char_count {
        text.len()
    } else {
        text.char_indices().nth(end)?.0
    };
    Some((start, end))
}

fn bounded_event_index(value: &Value, key: &str) -> Result<u16, OpenAiError> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|raw| u16::try_from(raw).ok())
        .ok_or_else(|| malformed("Responses event index missing or out of range"))
}

fn output_index(value: &Value) -> Result<u16, OpenAiError> {
    let raw = value
        .get("output_index")
        .and_then(Value::as_u64)
        .ok_or_else(|| malformed("Responses output index missing"))?;
    let index = u16::try_from(raw).map_err(|_| malformed("Responses output index out of range"))?;
    let _ = stream_index(index)?;
    Ok(index)
}

fn stream_index(index: u16) -> Result<ModelStreamIndex, OpenAiError> {
    ModelStreamIndex::new(index).map_err(|_| malformed("Responses output index out of range"))
}

fn nonempty_string<'a>(value: &'a Value, key: &str) -> Option<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
}

fn parse_usage(response: &Value) -> Result<Option<Usage>, OpenAiError> {
    let Some(value) = response.get("usage").filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let raw_input = token_value(value, "input_tokens")?;
    let output = token_value(value, "output_tokens")?;
    let details = value.get("input_tokens_details");
    let cache_read = optional_token_value(details, "cached_tokens")?;
    let cache_write = optional_token_value(details, "cache_write_tokens")?;
    let billable_input = raw_input
        .saturating_sub(cache_read.map_or(0, TokenCount::get))
        .saturating_sub(cache_write.map_or(0, TokenCount::get));
    let mut usage = Usage::new(
        TokenCount::new(billable_input).map_err(|_| malformed("Responses input usage"))?,
        TokenCount::new(output).map_err(|_| malformed("Responses output usage"))?,
    );
    if let Some(cache_read) = cache_read {
        usage = usage.with_cache_read(cache_read);
    }
    if let Some(cache_write) = cache_write {
        usage = usage.with_cache_write(cache_write);
    }
    if let Some(reasoning) =
        optional_token_value(value.get("output_tokens_details"), "reasoning_tokens")?
    {
        usage = usage
            .with_reasoning(reasoning)
            .map_err(|_| malformed("Responses reasoning usage"))?;
    }
    Ok(Some(usage))
}

fn token_value(value: &Value, key: &str) -> Result<u64, OpenAiError> {
    value
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| malformed("Responses usage field missing"))
}

fn optional_token_value(
    value: Option<&Value>,
    key: &str,
) -> Result<Option<TokenCount>, OpenAiError> {
    let Some(raw) = value.and_then(|value| value.get(key)) else {
        return Ok(None);
    };
    let raw = raw
        .as_u64()
        .ok_or_else(|| malformed("Responses usage field invalid"))?;
    TokenCount::new(raw)
        .map(Some)
        .map_err(|_| malformed("Responses usage field out of range"))
}

fn map_response_failure(response: &Value, incomplete: bool) -> ModelFailure {
    let error = response.get("error").unwrap_or(response);
    let details = response.get("incomplete_details");
    let code = error
        .get("code")
        .or_else(|| error.get("type"))
        .and_then(Value::as_str)
        .or_else(|| {
            details
                .and_then(|value| value.get("reason"))
                .and_then(Value::as_str)
        })
        .unwrap_or(if incomplete { "incomplete" } else { "unknown" });
    let message = error
        .get("message")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| {
            details
                .and_then(|value| value.get("reason"))
                .and_then(Value::as_str)
                .map(|reason| format!("response incomplete: {reason}"))
        })
        .unwrap_or_else(|| "OpenAI Responses request failed".to_owned());
    let (failure_code, retry) = match code {
        "invalid_api_key" | "authentication_error" => {
            (ModelFailureCode::Authentication, RetryClass::Never)
        }
        "permission_denied" => (ModelFailureCode::PermissionDenied, RetryClass::Never),
        "rate_limit_exceeded" | "insufficient_quota" => {
            (ModelFailureCode::RateLimited, RetryClass::AfterBackoff)
        }
        "server_error" | "overloaded" => (ModelFailureCode::Unavailable, RetryClass::AfterBackoff),
        "context_length_exceeded" | "max_output_tokens" => {
            (ModelFailureCode::ContextOverflow, RetryClass::Never)
        }
        "cancelled" => (ModelFailureCode::Cancelled, RetryClass::Never),
        "invalid_request_error" => (ModelFailureCode::InvalidRequest, RetryClass::Never),
        _ => (ModelFailureCode::MalformedResponse, RetryClass::Never),
    };
    ModelFailure::new(failure_code, message, retry)
        .unwrap_or_else(|_| ModelFailure::internal_adapter_failure())
}

fn malformed(message: &str) -> OpenAiError {
    OpenAiError::new(OpenAiErrorCode::MalformedResponse, message)
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    const OPAQUE_SENTINEL: &str = "opaque-continuation-must-not-appear-in-debug";

    #[test]
    fn reducer_debug_redacts_opaque_provider_payloads() {
        let mut reducer = ResponsesReducer::new();
        reducer.hosted_searches.insert(
            0,
            HostedSearchAccumulator {
                provider_call_id: "search_0".to_owned(),
                completed: true,
                source_urls: BTreeSet::new(),
                completed_item: Some(json!({"opaque": OPAQUE_SENTINEL})),
            },
        );
        reducer
            .citation_payloads
            .insert((0, 0, 0), json!({"opaque": OPAQUE_SENTINEL}));

        let debug = format!("{reducer:?}");
        assert!(!debug.contains(OPAQUE_SENTINEL), "{debug}");
    }
}
