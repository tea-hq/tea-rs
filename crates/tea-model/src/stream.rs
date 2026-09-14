use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::str::FromStr;

use serde_json::Value;
use tea_protocol::{
    ExactCost, ExternalSource, HostedToolOutcome, MAX_HOSTED_TOOL_SOURCES, ModelId,
    ProtocolMetadata, ProviderContinuation, SourceCitation, StopReason, Usage,
};
use thiserror::Error;

use crate::ModelFailure;

/// Maximum UTF-8 bytes in one normalized text, thinking, or argument delta.
pub const MAX_MODEL_DELTA_BYTES: usize = 64 * 1024;
/// Maximum UTF-8 bytes in one opaque provider response or tool-call ID.
pub const MAX_PROVIDER_OPAQUE_ID_BYTES: usize = 256;
/// Largest concurrent tool-call index supported in one response.
pub const MAX_MODEL_STREAM_INDEX: u16 = 1023;
const MAX_COMPLETED_TOOL_ARGUMENT_BYTES: usize = 256 * 1024;
const MAX_COMPLETED_TOOL_ARGUMENT_DEPTH: usize = 32;

macro_rules! opaque_id {
    ($name:ident, $doc:literal) => {
        #[doc = $doc]
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            /// Returns the bounded opaque provider value.
            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl FromStr for $name {
            type Err = ModelStreamValueError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                if value.is_empty()
                    || value.len() > MAX_PROVIDER_OPAQUE_ID_BYTES
                    || value.chars().any(char::is_control)
                {
                    return Err(ModelStreamValueError::InvalidProviderOpaqueId);
                }
                Ok(Self(value.to_owned()))
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }
    };
}

opaque_id!(
    ProviderResponseId,
    "Bounded opaque response identifier returned by a provider."
);
opaque_id!(
    ProviderToolCallId,
    "Bounded provider-scoped identifier joining streamed tool-call fragments."
);

/// Bounded zero-based content/tool index in one provider response.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModelStreamIndex(u16);

impl ModelStreamIndex {
    /// Creates a bounded stream index.
    ///
    /// # Errors
    ///
    /// Returns an error above [`MAX_MODEL_STREAM_INDEX`].
    pub const fn new(value: u16) -> Result<Self, ModelStreamValueError> {
        if value > MAX_MODEL_STREAM_INDEX {
            Err(ModelStreamValueError::InvalidStreamIndex)
        } else {
            Ok(Self(value))
        }
    }

    /// Returns the numeric index.
    #[must_use]
    pub const fn get(self) -> u16 {
        self.0
    }
}

/// Bounded non-empty UTF-8 text or thinking delta.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Utf8Delta(String);

impl Utf8Delta {
    /// Creates a bounded delta.
    ///
    /// # Errors
    ///
    /// Returns an error when empty, oversized, or containing a null character.
    pub fn new(value: impl Into<String>) -> Result<Self, ModelStreamValueError> {
        let value = value.into();
        validate_delta(&value)?;
        Ok(Self(value))
    }

    /// Returns delta text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Metadata reported when a normalized provider response starts.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ModelResponseInfo {
    response_id: Option<ProviderResponseId>,
    response_model: Option<ModelId>,
    metadata: ProtocolMetadata,
}

impl ModelResponseInfo {
    /// Creates empty response metadata.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an opaque provider response ID.
    #[must_use]
    pub fn with_response_id(mut self, response_id: ProviderResponseId) -> Self {
        self.response_id = Some(response_id);
        self
    }

    /// Adds the concrete response model when it differs or is informative.
    #[must_use]
    pub fn with_response_model(mut self, response_model: ModelId) -> Self {
        self.response_model = Some(response_model);
        self
    }

    /// Adds bounded namespaced response metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: ProtocolMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Returns the provider response ID.
    #[must_use]
    pub const fn response_id(&self) -> Option<&ProviderResponseId> {
        self.response_id.as_ref()
    }

    /// Returns the concrete response model.
    #[must_use]
    pub const fn response_model(&self) -> Option<&ModelId> {
        self.response_model.as_ref()
    }

    /// Returns bounded response metadata.
    #[must_use]
    pub const fn metadata(&self) -> &ProtocolMetadata {
        &self.metadata
    }
}

/// Start of one provider-streamed tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallStarted {
    index: ModelStreamIndex,
    provider_call_id: ProviderToolCallId,
    tool_name: String,
}

impl ToolCallStarted {
    /// Creates a validated tool-call start.
    ///
    /// # Errors
    ///
    /// Returns an error when the tool name is not canonical.
    pub fn new(
        index: ModelStreamIndex,
        provider_call_id: ProviderToolCallId,
        tool_name: impl Into<String>,
    ) -> Result<Self, ModelStreamValueError> {
        let tool_name = tool_name.into();
        validate_tool_name(&tool_name)?;
        Ok(Self {
            index,
            provider_call_id,
            tool_name,
        })
    }

    /// Returns the response-local stream index.
    #[must_use]
    pub const fn index(&self) -> ModelStreamIndex {
        self.index
    }

    /// Returns the provider call ID.
    #[must_use]
    pub const fn provider_call_id(&self) -> &ProviderToolCallId {
        &self.provider_call_id
    }

    /// Returns the canonical tool name.
    #[must_use]
    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }
}

/// Incomplete provider tool-arguments fragment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolArgumentsDelta {
    index: ModelStreamIndex,
    provider_call_id: ProviderToolCallId,
    delta: String,
}

impl ToolArgumentsDelta {
    /// Creates a bounded incomplete argument fragment.
    ///
    /// # Errors
    ///
    /// Returns an error when the fragment is empty, oversized, or contains a
    /// null character.
    pub fn new(
        index: ModelStreamIndex,
        provider_call_id: ProviderToolCallId,
        delta: impl Into<String>,
    ) -> Result<Self, ModelStreamValueError> {
        let delta = delta.into();
        validate_delta(&delta)?;
        Ok(Self {
            index,
            provider_call_id,
            delta,
        })
    }

    /// Returns the response-local stream index.
    #[must_use]
    pub const fn index(&self) -> ModelStreamIndex {
        self.index
    }

    /// Returns the provider call ID.
    #[must_use]
    pub const fn provider_call_id(&self) -> &ProviderToolCallId {
        &self.provider_call_id
    }

    /// Returns the incomplete UTF-8 argument fragment.
    #[must_use]
    pub fn delta(&self) -> &str {
        &self.delta
    }
}

/// Completed, parsed provider tool call safe for later canonical projection.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCallCompleted {
    index: ModelStreamIndex,
    provider_call_id: ProviderToolCallId,
    tool_name: String,
    arguments: Value,
}

impl ToolCallCompleted {
    /// Creates a validated completed call with object arguments.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid name, non-object arguments, or bounded
    /// JSON violations.
    pub fn new(
        index: ModelStreamIndex,
        provider_call_id: ProviderToolCallId,
        tool_name: impl Into<String>,
        arguments: Value,
    ) -> Result<Self, ModelStreamValueError> {
        let tool_name = tool_name.into();
        validate_tool_name(&tool_name)?;
        validate_completed_arguments(&arguments)?;
        Ok(Self {
            index,
            provider_call_id,
            tool_name,
            arguments,
        })
    }

    /// Returns the response-local stream index.
    #[must_use]
    pub const fn index(&self) -> ModelStreamIndex {
        self.index
    }

    /// Returns the provider call ID.
    #[must_use]
    pub const fn provider_call_id(&self) -> &ProviderToolCallId {
        &self.provider_call_id
    }

    /// Returns the canonical tool name.
    #[must_use]
    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }

    /// Returns complete parsed object arguments.
    #[must_use]
    pub const fn arguments(&self) -> &Value {
        &self.arguments
    }
}

/// Start of one provider-hosted tool activity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostedToolStarted {
    index: ModelStreamIndex,
    provider_call_id: ProviderToolCallId,
    tool_name: String,
}

impl HostedToolStarted {
    /// Creates a validated hosted tool start.
    ///
    /// # Errors
    ///
    /// Returns an error when the tool name is not canonical.
    pub fn new(
        index: ModelStreamIndex,
        provider_call_id: ProviderToolCallId,
        tool_name: impl Into<String>,
    ) -> Result<Self, ModelStreamValueError> {
        let tool_name = tool_name.into();
        validate_tool_name(&tool_name)?;
        Ok(Self {
            index,
            provider_call_id,
            tool_name,
        })
    }

    /// Returns the response-local stream index.
    #[must_use]
    pub const fn index(&self) -> ModelStreamIndex {
        self.index
    }

    /// Returns the provider activity identifier.
    #[must_use]
    pub const fn provider_call_id(&self) -> &ProviderToolCallId {
        &self.provider_call_id
    }

    /// Returns the canonical hosted tool name.
    #[must_use]
    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }
}

/// Complete provider-hosted tool activity safe for canonical projection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostedToolCompleted {
    index: ModelStreamIndex,
    provider_call_id: ProviderToolCallId,
    tool_name: String,
    arguments: Value,
    outcome: HostedToolOutcome,
    sources: Vec<ExternalSource>,
    continuation: Option<ProviderContinuation>,
}

impl HostedToolCompleted {
    /// Creates one validated complete hosted activity.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid identity, arguments, or source bounds.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        index: ModelStreamIndex,
        provider_call_id: ProviderToolCallId,
        tool_name: impl Into<String>,
        arguments: Value,
        outcome: HostedToolOutcome,
        sources: Vec<ExternalSource>,
        continuation: Option<ProviderContinuation>,
    ) -> Result<Self, ModelStreamValueError> {
        let tool_name = tool_name.into();
        validate_tool_name(&tool_name)?;
        validate_completed_arguments(&arguments)?;
        if sources.len() > MAX_HOSTED_TOOL_SOURCES {
            return Err(ModelStreamValueError::TooManyHostedToolSources);
        }
        Ok(Self {
            index,
            provider_call_id,
            tool_name,
            arguments,
            outcome,
            sources,
            continuation,
        })
    }

    /// Returns the response-local stream index.
    #[must_use]
    pub const fn index(&self) -> ModelStreamIndex {
        self.index
    }

    /// Returns the provider activity identifier.
    #[must_use]
    pub const fn provider_call_id(&self) -> &ProviderToolCallId {
        &self.provider_call_id
    }

    /// Returns the canonical hosted tool name.
    #[must_use]
    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }

    /// Returns normalized object arguments.
    #[must_use]
    pub const fn arguments(&self) -> &Value {
        &self.arguments
    }

    /// Returns the provider-reported terminal outcome.
    #[must_use]
    pub const fn outcome(&self) -> &HostedToolOutcome {
        &self.outcome
    }

    /// Returns normalized sources in provider order.
    #[must_use]
    pub fn sources(&self) -> &[ExternalSource] {
        &self.sources
    }

    /// Returns opaque same-provider continuation data.
    #[must_use]
    pub const fn continuation(&self) -> Option<&ProviderContinuation> {
        self.continuation.as_ref()
    }
}

/// Provider-streamed citation awaiting canonical hosted call identity mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSourceCitation {
    provider_call_id: Option<ProviderToolCallId>,
    citation: SourceCitation,
}

impl ModelSourceCitation {
    /// Creates a normalized citation without a canonical tool-call identifier.
    ///
    /// # Errors
    ///
    /// Returns an error when an adapter attempts to assign kernel-owned identity.
    pub fn new(
        provider_call_id: Option<ProviderToolCallId>,
        citation: SourceCitation,
    ) -> Result<Self, ModelStreamValueError> {
        if citation.tool_call_id().is_some() {
            return Err(ModelStreamValueError::CitationAlreadyCanonical);
        }
        Ok(Self {
            provider_call_id,
            citation,
        })
    }

    /// Returns the related provider activity identifier, when available.
    #[must_use]
    pub const fn provider_call_id(&self) -> Option<&ProviderToolCallId> {
        self.provider_call_id.as_ref()
    }

    /// Returns the normalized provider-neutral citation.
    #[must_use]
    pub const fn citation(&self) -> &SourceCitation {
        &self.citation
    }
}

/// Provider-terminal model completion data.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelCompletion {
    stop_reason: StopReason,
    usage: Option<Usage>,
    cost: Option<ExactCost>,
    metadata: ProtocolMetadata,
}

impl ModelCompletion {
    /// Creates a normal successful completion.
    #[must_use]
    pub fn completed() -> Self {
        Self {
            stop_reason: StopReason::Completed,
            usage: None,
            cost: None,
            metadata: ProtocolMetadata::default(),
        }
    }

    /// Creates provider-terminal completion data.
    ///
    /// # Errors
    ///
    /// Returns an error for cancelled or error stop reasons. Those outcomes
    /// must use [`ModelEvent::Failed`]. Refusal and unknown provider-terminal
    /// reasons remain completions, but are not normal successful output.
    pub fn new(stop_reason: StopReason) -> Result<Self, ModelStreamValueError> {
        if !matches!(
            stop_reason,
            StopReason::Completed
                | StopReason::Length
                | StopReason::ToolUse
                | StopReason::PauseTurn
                | StopReason::Refusal
                | StopReason::Unknown(_)
        ) {
            return Err(ModelStreamValueError::InvalidCompletionReason);
        }
        Ok(Self {
            stop_reason,
            usage: None,
            cost: None,
            metadata: ProtocolMetadata::default(),
        })
    }

    /// Adds normalized token usage.
    #[must_use]
    pub fn with_usage(mut self, usage: Usage) -> Self {
        self.usage = Some(usage);
        self
    }

    /// Adds exact provider-reported cost.
    #[must_use]
    pub fn with_cost(mut self, cost: ExactCost) -> Self {
        self.cost = Some(cost);
        self
    }

    /// Adds bounded terminal metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: ProtocolMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Returns the normalized stop reason.
    #[must_use]
    pub const fn stop_reason(&self) -> &StopReason {
        &self.stop_reason
    }

    /// Returns normalized token usage when reported.
    #[must_use]
    pub const fn usage(&self) -> Option<&Usage> {
        self.usage.as_ref()
    }

    /// Returns exact provider-reported cost when available.
    #[must_use]
    pub const fn cost(&self) -> Option<&ExactCost> {
        self.cost.as_ref()
    }

    /// Returns bounded terminal metadata.
    #[must_use]
    pub const fn metadata(&self) -> &ProtocolMetadata {
        &self.metadata
    }
}

/// One normalized provider stream event.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelEvent {
    /// Provider accepted the request and began one response.
    Started(ModelResponseInfo),
    /// Visible assistant text fragment.
    TextDelta(Utf8Delta),
    /// Assistant reasoning fragment.
    ThinkingDelta(Utf8Delta),
    /// Start of a provider tool call.
    ToolCallStarted(ToolCallStarted),
    /// Incomplete tool argument fragment, never executable by itself.
    ToolArgumentsDelta(ToolArgumentsDelta),
    /// Completed tool call with parsed object arguments.
    ToolCallCompleted(ToolCallCompleted),
    /// Start of a provider-hosted tool activity.
    HostedToolStarted(HostedToolStarted),
    /// Completed provider-hosted tool activity with normalized sources.
    HostedToolCompleted(HostedToolCompleted),
    /// Citation emitted for assistant text and an external source.
    SourceCitation(ModelSourceCitation),
    /// Provider-terminal event.
    Completed(ModelCompletion),
    /// Failed or cancelled terminal event.
    Failed(ModelFailure),
}

impl ModelEvent {
    /// Returns visible text for a text-delta event.
    #[must_use]
    pub fn as_text_delta(&self) -> Option<&str> {
        match self {
            Self::TextDelta(delta) => Some(delta.as_str()),
            _ => None,
        }
    }

    /// Returns reasoning text for a thinking-delta event.
    #[must_use]
    pub fn as_thinking_delta(&self) -> Option<&str> {
        match self {
            Self::ThinkingDelta(delta) => Some(delta.as_str()),
            _ => None,
        }
    }

    /// Returns the tool-call start payload.
    #[must_use]
    pub const fn as_tool_call_started(&self) -> Option<&ToolCallStarted> {
        match self {
            Self::ToolCallStarted(call) => Some(call),
            _ => None,
        }
    }

    /// Returns the incomplete tool-arguments payload.
    #[must_use]
    pub const fn as_tool_arguments_delta(&self) -> Option<&ToolArgumentsDelta> {
        match self {
            Self::ToolArgumentsDelta(delta) => Some(delta),
            _ => None,
        }
    }

    /// Returns the completed tool-call payload.
    #[must_use]
    pub const fn as_tool_call_completed(&self) -> Option<&ToolCallCompleted> {
        match self {
            Self::ToolCallCompleted(call) => Some(call),
            _ => None,
        }
    }

    /// Returns the hosted tool start payload.
    #[must_use]
    pub const fn as_hosted_tool_started(&self) -> Option<&HostedToolStarted> {
        match self {
            Self::HostedToolStarted(call) => Some(call),
            _ => None,
        }
    }

    /// Returns the hosted tool completion payload.
    #[must_use]
    pub const fn as_hosted_tool_completed(&self) -> Option<&HostedToolCompleted> {
        match self {
            Self::HostedToolCompleted(call) => Some(call),
            _ => None,
        }
    }

    /// Returns the normalized source citation payload.
    #[must_use]
    pub const fn as_source_citation(&self) -> Option<&ModelSourceCitation> {
        match self {
            Self::SourceCitation(citation) => Some(citation),
            _ => None,
        }
    }
}

/// Deterministic validator for normalized provider stream grammar.
#[derive(Debug, Default)]
pub struct ModelStreamValidator {
    started: bool,
    terminal: Option<bool>,
    event_count: usize,
    completed_tool_calls: usize,
    completed_hosted_tools: usize,
    active_tools: BTreeMap<ModelStreamIndex, (ProviderToolCallId, String)>,
    active_hosted_tools: BTreeMap<ModelStreamIndex, (ProviderToolCallId, String)>,
    completed_hosted_ids: BTreeSet<ProviderToolCallId>,
    seen_tool_indexes: BTreeSet<ModelStreamIndex>,
}

impl ModelStreamValidator {
    /// Creates an empty stream grammar validator.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Observes one event in source order.
    ///
    /// # Errors
    ///
    /// Returns a typed violation when the event does not follow normalized
    /// stream grammar. Rejected events do not advance validator state.
    pub fn observe(&mut self, event: &ModelEvent) -> Result<(), ModelStreamViolation> {
        if self.terminal.is_some() {
            return Err(ModelStreamViolation::EventAfterTerminal);
        }
        if !self.started {
            if matches!(event, ModelEvent::Started(_)) {
                self.started = true;
                self.event_count += 1;
                return Ok(());
            }
            return Err(ModelStreamViolation::EventBeforeStart);
        }

        match event {
            ModelEvent::Started(_) => return Err(ModelStreamViolation::DuplicateStart),
            ModelEvent::ToolCallStarted(call) => {
                if self.seen_tool_indexes.contains(&call.index()) {
                    return Err(ModelStreamViolation::DuplicateToolIndex);
                }
                self.seen_tool_indexes.insert(call.index());
                self.active_tools.insert(
                    call.index(),
                    (call.provider_call_id().clone(), call.tool_name().to_owned()),
                );
            }
            ModelEvent::HostedToolStarted(call) => {
                if self.seen_tool_indexes.contains(&call.index()) {
                    return Err(ModelStreamViolation::DuplicateToolIndex);
                }
                self.seen_tool_indexes.insert(call.index());
                self.active_hosted_tools.insert(
                    call.index(),
                    (call.provider_call_id().clone(), call.tool_name().to_owned()),
                );
            }
            ModelEvent::ToolArgumentsDelta(delta) => {
                let Some((call_id, _)) = self.active_tools.get(&delta.index()) else {
                    return Err(ModelStreamViolation::UnknownToolIndex);
                };
                if call_id != delta.provider_call_id() {
                    return Err(ModelStreamViolation::ToolIdentityMismatch);
                }
            }
            ModelEvent::ToolCallCompleted(call) => {
                let Some((call_id, tool_name)) = self.active_tools.get(&call.index()) else {
                    return Err(ModelStreamViolation::UnknownToolIndex);
                };
                if call_id != call.provider_call_id() || tool_name != call.tool_name() {
                    return Err(ModelStreamViolation::ToolIdentityMismatch);
                }
                self.active_tools.remove(&call.index());
                self.completed_tool_calls += 1;
            }
            ModelEvent::HostedToolCompleted(call) => {
                let Some((call_id, tool_name)) = self.active_hosted_tools.get(&call.index()) else {
                    return Err(ModelStreamViolation::UnknownToolIndex);
                };
                if call_id != call.provider_call_id() || tool_name != call.tool_name() {
                    return Err(ModelStreamViolation::ToolIdentityMismatch);
                }
                self.active_hosted_tools.remove(&call.index());
                self.completed_hosted_ids
                    .insert(call.provider_call_id().clone());
                self.completed_hosted_tools += 1;
            }
            ModelEvent::SourceCitation(citation) => {
                if citation
                    .provider_call_id()
                    .is_some_and(|call_id| !self.completed_hosted_ids.contains(call_id))
                {
                    return Err(ModelStreamViolation::UnknownHostedCitation);
                }
            }
            ModelEvent::Completed(_) => {
                if !self.active_tools.is_empty() || !self.active_hosted_tools.is_empty() {
                    return Err(ModelStreamViolation::IncompleteToolCalls);
                }
                self.terminal = Some(true);
            }
            ModelEvent::Failed(_) => {
                self.terminal = Some(false);
            }
            ModelEvent::TextDelta(_) | ModelEvent::ThinkingDelta(_) => {}
        }
        self.event_count += 1;
        Ok(())
    }

    /// Finishes validation after the stream returns `None`.
    ///
    /// # Errors
    ///
    /// Returns an error when start or terminal events are missing.
    pub fn finish(self) -> Result<ModelStreamSummary, ModelStreamViolation> {
        if !self.started {
            return Err(ModelStreamViolation::MissingStart);
        }
        let succeeded = self.terminal.ok_or(ModelStreamViolation::MissingTerminal)?;
        Ok(ModelStreamSummary {
            event_count: self.event_count,
            completed_tool_calls: self.completed_tool_calls,
            completed_hosted_tools: self.completed_hosted_tools,
            succeeded,
        })
    }
}

/// Summary of one fully validated normalized stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelStreamSummary {
    event_count: usize,
    completed_tool_calls: usize,
    completed_hosted_tools: usize,
    succeeded: bool,
}

impl ModelStreamSummary {
    /// Returns accepted event count including start and terminal events.
    #[must_use]
    pub const fn event_count(self) -> usize {
        self.event_count
    }

    /// Returns the number of completed tool calls.
    #[must_use]
    pub const fn completed_tool_calls(self) -> usize {
        self.completed_tool_calls
    }

    /// Returns the number of completed provider-hosted activities.
    #[must_use]
    pub const fn completed_hosted_tools(self) -> usize {
        self.completed_hosted_tools
    }

    /// Returns whether termination was successful rather than failed.
    #[must_use]
    pub const fn succeeded(self) -> bool {
        self.succeeded
    }
}

/// Normalized stream grammar violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ModelStreamViolation {
    /// A non-start event appeared before `Started`.
    #[error("model stream event appeared before start")]
    EventBeforeStart,
    /// No start event was observed.
    #[error("model stream is missing start")]
    MissingStart,
    /// More than one start event appeared.
    #[error("model stream contains duplicate start")]
    DuplicateStart,
    /// Stream ended without success or failure.
    #[error("model stream is missing terminal event")]
    MissingTerminal,
    /// An event appeared after success or failure.
    #[error("model stream event appeared after terminal event")]
    EventAfterTerminal,
    /// A response-local tool index was started more than once concurrently.
    #[error("model stream contains duplicate active tool index")]
    DuplicateToolIndex,
    /// A tool fragment or completion references no active call.
    #[error("model stream references an unknown tool index")]
    UnknownToolIndex,
    /// Tool index, provider ID, or name changed within one call.
    #[error("model stream tool identity changed")]
    ToolIdentityMismatch,
    /// Successful termination left one or more calls incomplete.
    #[error("model stream completed with incomplete tool calls")]
    IncompleteToolCalls,
    /// A citation references no completed provider-hosted activity.
    #[error("model stream citation references an unknown hosted tool")]
    UnknownHostedCitation,
}

/// Error returned by normalized stream value constructors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ModelStreamValueError {
    /// Provider opaque identifier is empty, oversized, or contains controls.
    #[error("provider opaque identifier is invalid")]
    InvalidProviderOpaqueId,
    /// Stream index exceeds the supported response-local range.
    #[error("model stream index is invalid")]
    InvalidStreamIndex,
    /// Delta is empty, oversized, or contains a null character.
    #[error("model stream delta is invalid")]
    InvalidDelta,
    /// Tool name is not canonical.
    #[error("model tool name is invalid")]
    InvalidToolName,
    /// Completed tool arguments must be an object.
    #[error("completed tool arguments must be a JSON object")]
    ToolArgumentsMustBeObject,
    /// Completed tool arguments exceed JSON bounds.
    #[error("completed tool arguments exceed supported bounds")]
    ToolArgumentsOutOfBounds,
    /// Hosted tool returned too many normalized sources.
    #[error("hosted tool returned too many sources")]
    TooManyHostedToolSources,
    /// Provider adapters cannot assign kernel-owned citation identity.
    #[error("model citation already contains a canonical tool-call identifier")]
    CitationAlreadyCanonical,
    /// Successful completion used a failure-only or unknown stop reason.
    #[error("successful completion stop reason is invalid")]
    InvalidCompletionReason,
    /// Failure message is empty, oversized, or contains a null character.
    #[error("model failure message is invalid")]
    InvalidFailureMessage,
}

fn validate_delta(value: &str) -> Result<(), ModelStreamValueError> {
    if value.is_empty() || value.len() > MAX_MODEL_DELTA_BYTES || value.contains('\0') {
        Err(ModelStreamValueError::InvalidDelta)
    } else {
        Ok(())
    }
}

fn validate_tool_name(value: &str) -> Result<(), ModelStreamValueError> {
    let mut bytes = value.bytes();
    if value.len() > 128
        || !bytes.next().is_some_and(|byte| byte.is_ascii_lowercase())
        || !bytes.all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-' | b'.')
        })
    {
        Err(ModelStreamValueError::InvalidToolName)
    } else {
        Ok(())
    }
}

fn validate_completed_arguments(arguments: &Value) -> Result<(), ModelStreamValueError> {
    if !arguments.is_object() {
        return Err(ModelStreamValueError::ToolArgumentsMustBeObject);
    }
    if serde_json::to_vec(arguments)
        .map_err(|_| ModelStreamValueError::ToolArgumentsOutOfBounds)?
        .len()
        > MAX_COMPLETED_TOOL_ARGUMENT_BYTES
        || json_depth(arguments) > MAX_COMPLETED_TOOL_ARGUMENT_DEPTH
    {
        return Err(ModelStreamValueError::ToolArgumentsOutOfBounds);
    }
    Ok(())
}

fn json_depth(value: &Value) -> usize {
    match value {
        Value::Array(values) => 1 + values.iter().map(json_depth).max().unwrap_or(0),
        Value::Object(values) => 1 + values.values().map(json_depth).max().unwrap_or(0),
        _ => 1,
    }
}
