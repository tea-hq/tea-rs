#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Canonical, provider-neutral protocol types for agent runtimes.
//!
//! Protocol compatibility is versioned independently from this crate's
//! semantic version. This crate contains data and validation only; it does not
//! perform model requests, tool execution, policy evaluation, or persistence.
//!
//! # Example
//!
//! ```
//! use std::str::FromStr;
//!
//! use tea_protocol::{
//!     AgentCommand, CommandEnvelope, CommandId, ProfileId, ProtocolMetadata,
//!     ProtocolTimestamp,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let command = CommandEnvelope::new(
//!     CommandId::from_str("0195a0b1-5e3b-7ef0-8ec1-0aa7aa000001")?,
//!     None,
//!     ProtocolTimestamp::from_str("2026-07-23T09:30:12.123Z")?,
//!     AgentCommand::CreateSession {
//!         profile_id: ProfileId::from_str("minimal-assistant")?,
//!         metadata: ProtocolMetadata::default(),
//!     },
//! )?;
//!
//! let json = serde_json::to_string(&command)?;
//! assert!(json.contains(r#""type":"create_session""#));
//! # Ok(())
//! # }
//! ```

mod change;
mod command;
mod content;
mod envelope;
mod error;
mod event;
mod external;
mod final_output;
mod id;
mod message;
mod metadata;
mod model;
mod reasoning;
mod record;
mod sequence;
mod timestamp;
mod usage;
mod version;

pub use change::{
    CodeChange, CodeChangeHunk, CodeChangeKind, CodeChangeLine, CodeChangeLineKind,
    CodeChangeTruncation, CodeChangeValidationError, MAX_CODE_CHANGE_HUNKS,
    MAX_CODE_CHANGE_LINE_BYTES, MAX_CODE_CHANGE_LINES, MAX_CODE_CHANGE_LINES_PER_HUNK,
    MAX_CODE_CHANGE_PATCH_BYTES, MAX_CODE_CHANGE_PATH_BYTES, ToolPresentation,
};
pub use command::{
    AgentCommand, AgentCommandType, ApprovalDecision, CommandDecodeError, CommandEnvelope,
    CommandText, CommandValidationError, MAX_COMMAND_TEXT_BYTES, MAX_SELECTOR_BYTES, ModelId,
    ProfileId, ProviderId, SelectorParseError,
};
pub use content::{
    ContentBlock, ContentValidationError, ImageSource, MAX_INLINE_IMAGE_BASE64_BYTES,
    MAX_TEXT_BLOCK_BYTES, MAX_TOOL_ARGUMENT_BYTES, MAX_TOOL_ARGUMENT_DEPTH,
};
pub use error::{
    AgentErrorCode, MAX_ERROR_MESSAGE_BYTES, ProtocolError, ProtocolErrorEnvelope,
    ProtocolErrorValidationError, RetryClass,
};
pub use event::{
    AgentEvent, AgentEventType, EventCompatibility, EventDecodeError, EventDelta, EventEnvelope,
    EventInspection, EventValidationError, MAX_APPROVAL_ITEM_BYTES, MAX_APPROVAL_ITEMS,
    MAX_EVENT_DELTA_BYTES, MAX_PROGRESS_MESSAGE_BYTES, MAX_UNKNOWN_EVENT_BYTES, RunStatus,
    UnknownSkippableEvent,
};
pub use external::{
    ExternalContentError, ExternalSource, HostedToolActivity, HostedToolError, HostedToolOutcome,
    MAX_EXTERNAL_SOURCE_TEXT_BYTES, MAX_EXTERNAL_SOURCE_TITLE_BYTES, MAX_EXTERNAL_SOURCE_URL_BYTES,
    MAX_HOSTED_TOOL_SOURCES, MAX_PROVIDER_CONTINUATION_BYTES, MAX_PROVIDER_CONTINUATION_DEPTH,
    MAX_WEB_FETCH_BODY_BYTES, MAX_WEB_FETCH_BODY_CHARS, MAX_WEB_FETCH_MIME_BYTES,
    MAX_WEB_FETCH_REDIRECTS, MAX_WEB_FETCH_TITLE_BYTES, MAX_WEB_FETCH_URL_BYTES,
    ProviderContinuation, SourceCitation, WebFetchPresentation, WebFetchRedirect,
    WebFetchTruncation,
};
pub use final_output::{
    FINAL_JSON_OBJECT_INSTRUCTION, FinalOutputFormat, FinalOutputFormatError,
    MAX_FINAL_OUTPUT_SCHEMA_BYTES, MAX_FINAL_OUTPUT_SCHEMA_DEPTH,
};
pub use id::{
    ApprovalId, BranchId, CausationId, CommandId, CorrelationId, EventId, MessageId,
    ProtocolIdParseError, RecordId, RunId, SessionId, ToolCallId, TurnId,
};
pub use message::{CanonicalMessage, MessageRole, MessageValidationError, StopReason, ToolFailure};
pub use metadata::{
    MAX_METADATA_BYTES, MAX_METADATA_DEPTH, MAX_METADATA_NAMESPACES, ProtocolMetadata,
    ProtocolMetadataError,
};
pub use model::ModelRef;
pub use reasoning::{ReasoningEffort, ReasoningEffortParseError};
pub use record::{
    ExecutionTarget, MAX_RECORD_CONTENT_BLOCKS, NextTurnAction, PolicyDecision, RecordDecodeError,
    RecordEnvelope, RecordValidationError, SessionRecord, SessionRecordType, ToolIdempotency,
};
pub use sequence::{SessionSequence, SessionSequenceParseError};
pub use timestamp::{ProtocolTimestamp, ProtocolTimestampParseError};
pub use usage::{
    CostUnit, CurrencyCode, CurrencyCodeParseError, DecimalAmount, DecimalAmountParseError,
    ExactCost, MAX_SAFE_INTEGER, TokenCount, Usage, UsageError,
};
pub use version::{
    CURRENT_PROTOCOL_VERSION, PROTOCOL_V1_0, ProtocolVersion, ProtocolVersionParseError,
};
