use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

use crate::content::{ContentBlock, ContentValidationError, validate_tool_name};
use crate::{MessageId, ProtocolTimestamp, ToolCallId};

/// Role of a canonical message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageRole {
    /// User-authored input.
    User,
    /// Model-authored output.
    Assistant,
    /// Result of a tool call.
    ToolResult,
}

/// A provider-neutral conversation message.
#[derive(Debug, Clone, PartialEq)]
pub enum CanonicalMessage {
    /// User-authored input.
    User {
        /// Stable message identifier.
        id: MessageId,
        /// User-visible text or images.
        content: Vec<ContentBlock>,
        /// Message creation time.
        timestamp: ProtocolTimestamp,
    },
    /// Model-authored output.
    Assistant {
        /// Stable message identifier.
        id: MessageId,
        /// Text, thinking, and tool calls.
        content: Vec<ContentBlock>,
        /// Normalized completion reason.
        stop_reason: StopReason,
        /// Message completion time.
        timestamp: ProtocolTimestamp,
    },
    /// Result returned for a tool call.
    ToolResult {
        /// Stable message identifier.
        id: MessageId,
        /// Tool call receiving this result.
        tool_call_id: ToolCallId,
        /// Registered tool name.
        tool_name: String,
        /// Text or image result content.
        content: Vec<ContentBlock>,
        /// Whether execution failed.
        is_error: bool,
        /// Machine-readable failure details, present only on failure.
        error: Option<ToolFailure>,
        /// Result completion time.
        timestamp: ProtocolTimestamp,
    },
}

impl CanonicalMessage {
    /// Creates a validated user message.
    ///
    /// # Errors
    ///
    /// Returns an error when content is empty, too numerous, or contains a
    /// block that is invalid for the user role.
    pub fn user(
        id: MessageId,
        content: Vec<ContentBlock>,
        timestamp: ProtocolTimestamp,
    ) -> Result<Self, MessageValidationError> {
        validate_content(&content, ContentBlock::valid_for_user)?;
        Ok(Self::User {
            id,
            content,
            timestamp,
        })
    }

    /// Creates a validated assistant message.
    ///
    /// # Errors
    ///
    /// Returns an error when content is empty, too numerous, or contains a
    /// block that is invalid for the assistant role.
    pub fn assistant(
        id: MessageId,
        content: Vec<ContentBlock>,
        stop_reason: StopReason,
        timestamp: ProtocolTimestamp,
    ) -> Result<Self, MessageValidationError> {
        validate_content(&content, ContentBlock::valid_for_assistant)?;
        Ok(Self::Assistant {
            id,
            content,
            stop_reason,
            timestamp,
        })
    }

    /// Creates a successful tool-result message.
    ///
    /// # Errors
    ///
    /// Returns an error when the tool name or result content is invalid.
    pub fn tool_result_success(
        id: MessageId,
        tool_call_id: ToolCallId,
        tool_name: impl Into<String>,
        content: Vec<ContentBlock>,
        timestamp: ProtocolTimestamp,
    ) -> Result<Self, MessageValidationError> {
        Self::tool_result(id, tool_call_id, tool_name.into(), content, None, timestamp)
    }

    /// Creates a failed tool-result message.
    ///
    /// # Errors
    ///
    /// Returns an error when the tool name or result content is invalid.
    pub fn tool_result_failure(
        id: MessageId,
        tool_call_id: ToolCallId,
        tool_name: impl Into<String>,
        content: Vec<ContentBlock>,
        error: ToolFailure,
        timestamp: ProtocolTimestamp,
    ) -> Result<Self, MessageValidationError> {
        Self::tool_result(
            id,
            tool_call_id,
            tool_name.into(),
            content,
            Some(error),
            timestamp,
        )
    }

    /// Returns the message role.
    #[must_use]
    pub const fn role(&self) -> MessageRole {
        match self {
            Self::User { .. } => MessageRole::User,
            Self::Assistant { .. } => MessageRole::Assistant,
            Self::ToolResult { .. } => MessageRole::ToolResult,
        }
    }

    fn validate(&self) -> Result<(), MessageValidationError> {
        match self {
            Self::User { content, .. } => validate_content(content, ContentBlock::valid_for_user),
            Self::Assistant { content, .. } => {
                validate_content(content, ContentBlock::valid_for_assistant)
            }
            Self::ToolResult {
                tool_name,
                content,
                is_error,
                error,
                ..
            } => {
                validate_tool_name(tool_name)?;
                validate_content(content, ContentBlock::valid_for_tool_result)?;
                if *is_error != error.is_some() {
                    return Err(MessageValidationError::InconsistentToolFailure);
                }
                Ok(())
            }
        }
    }

    fn tool_result(
        id: MessageId,
        tool_call_id: ToolCallId,
        tool_name: String,
        content: Vec<ContentBlock>,
        error: Option<ToolFailure>,
        timestamp: ProtocolTimestamp,
    ) -> Result<Self, MessageValidationError> {
        validate_tool_name(&tool_name)?;
        validate_content(&content, ContentBlock::valid_for_tool_result)?;
        Ok(Self::ToolResult {
            id,
            tool_call_id,
            tool_name,
            content,
            is_error: error.is_some(),
            error,
            timestamp,
        })
    }
}

impl Serialize for CanonicalMessage {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.validate().map_err(serde::ser::Error::custom)?;
        SerializableCanonicalMessage::from(self).serialize(serializer)
    }
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SerializableCanonicalMessage<'a> {
    User {
        id: &'a MessageId,
        content: &'a [ContentBlock],
        timestamp: &'a ProtocolTimestamp,
    },
    Assistant {
        id: &'a MessageId,
        content: &'a [ContentBlock],
        #[serde(rename = "stopReason")]
        stop_reason: &'a StopReason,
        timestamp: &'a ProtocolTimestamp,
    },
    ToolResult {
        id: &'a MessageId,
        #[serde(rename = "toolCallId")]
        tool_call_id: &'a ToolCallId,
        #[serde(rename = "toolName")]
        tool_name: &'a str,
        content: &'a [ContentBlock],
        #[serde(rename = "isError")]
        is_error: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: &'a Option<ToolFailure>,
        timestamp: &'a ProtocolTimestamp,
    },
}

impl<'a> From<&'a CanonicalMessage> for SerializableCanonicalMessage<'a> {
    fn from(value: &'a CanonicalMessage) -> Self {
        match value {
            CanonicalMessage::User {
                id,
                content,
                timestamp,
            } => Self::User {
                id,
                content,
                timestamp,
            },
            CanonicalMessage::Assistant {
                id,
                content,
                stop_reason,
                timestamp,
            } => Self::Assistant {
                id,
                content,
                stop_reason,
                timestamp,
            },
            CanonicalMessage::ToolResult {
                id,
                tool_call_id,
                tool_name,
                content,
                is_error,
                error,
                timestamp,
            } => Self::ToolResult {
                id,
                tool_call_id,
                tool_name,
                content,
                is_error: *is_error,
                error,
                timestamp,
            },
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RawCanonicalMessage {
    User {
        id: MessageId,
        content: Vec<ContentBlock>,
        timestamp: ProtocolTimestamp,
    },
    Assistant {
        id: MessageId,
        content: Vec<ContentBlock>,
        #[serde(rename = "stopReason")]
        stop_reason: StopReason,
        timestamp: ProtocolTimestamp,
    },
    ToolResult {
        id: MessageId,
        #[serde(rename = "toolCallId")]
        tool_call_id: ToolCallId,
        #[serde(rename = "toolName")]
        tool_name: String,
        content: Vec<ContentBlock>,
        #[serde(rename = "isError")]
        is_error: bool,
        #[serde(default)]
        error: Option<ToolFailure>,
        timestamp: ProtocolTimestamp,
    },
}

impl<'de> Deserialize<'de> for CanonicalMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        match RawCanonicalMessage::deserialize(deserializer)? {
            RawCanonicalMessage::User {
                id,
                content,
                timestamp,
            } => Self::user(id, content, timestamp),
            RawCanonicalMessage::Assistant {
                id,
                content,
                stop_reason,
                timestamp,
            } => Self::assistant(id, content, stop_reason, timestamp),
            RawCanonicalMessage::ToolResult {
                id,
                tool_call_id,
                tool_name,
                content,
                is_error,
                error,
                timestamp,
            } => {
                if is_error != error.is_some() {
                    return Err(serde::de::Error::custom(
                        "tool-result isError must match error presence",
                    ));
                }
                Self::tool_result(id, tool_call_id, tool_name, content, error, timestamp)
            }
        }
        .map_err(serde::de::Error::custom)
    }
}

/// Normalized assistant stop reason with forward-compatible unknown values.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StopReason {
    /// Model completed normally.
    Completed,
    /// Model reached an output limit.
    Length,
    /// Model requested one or more tools.
    ToolUse,
    /// Provider paused a server-side tool loop and requires transcript replay.
    PauseTurn,
    /// Provider refused to produce the requested output.
    Refusal,
    /// Operation was cancelled.
    Cancelled,
    /// Provider or runtime failed.
    Error,
    /// A future provider-neutral value not known by this crate.
    Unknown(String),
}

impl StopReason {
    /// Returns the serialized reason.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Completed => "completed",
            Self::Length => "length",
            Self::ToolUse => "tool_use",
            Self::PauseTurn => "pause_turn",
            Self::Refusal => "refusal",
            Self::Cancelled => "cancelled",
            Self::Error => "error",
            Self::Unknown(value) => value,
        }
    }

    /// Returns whether this reason represents normal completion.
    #[must_use]
    pub const fn is_success(&self) -> bool {
        matches!(self, Self::Completed)
    }
}

impl Serialize for StopReason {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if !valid_error_code(self.as_str()) {
            return Err(serde::ser::Error::custom("invalid stop reason"));
        }
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for StopReason {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        if !valid_error_code(&value) {
            return Err(serde::de::Error::custom("invalid stop reason"));
        }
        Ok(match value.as_str() {
            "completed" => Self::Completed,
            "length" => Self::Length,
            "tool_use" => Self::ToolUse,
            "pause_turn" => Self::PauseTurn,
            "refusal" => Self::Refusal,
            "cancelled" => Self::Cancelled,
            "error" => Self::Error,
            _ => Self::Unknown(value),
        })
    }
}

/// Machine-readable failure returned by a tool.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ToolFailure {
    code: String,
    message: String,
}

impl ToolFailure {
    /// Creates the canonical model-visible approval-denied failure.
    #[must_use]
    pub fn approval_denied() -> Self {
        Self {
            code: "approval_denied".to_owned(),
            message: "tool invocation was denied by approval".to_owned(),
        }
    }

    /// Creates validated tool failure details.
    ///
    /// # Errors
    ///
    /// Returns an error when the code is not canonical snake case or the
    /// technical message is empty, oversized, or contains a null character.
    pub fn new(
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Result<Self, MessageValidationError> {
        let code = code.into();
        let message = message.into();
        if !valid_error_code(&code)
            || message.is_empty()
            || message.len() > 4096
            || message.contains('\0')
        {
            return Err(MessageValidationError::InvalidToolFailure);
        }
        Ok(Self { code, message })
    }

    /// Returns the machine-readable failure code.
    #[must_use]
    pub fn code(&self) -> &str {
        &self.code
    }

    /// Returns the English technical diagnostic message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

#[derive(Deserialize)]
struct RawToolFailure {
    code: String,
    message: String,
}

impl<'de> Deserialize<'de> for ToolFailure {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawToolFailure::deserialize(deserializer)?;
        Self::new(raw.code, raw.message).map_err(serde::de::Error::custom)
    }
}

/// Error returned when validating canonical messages.
#[derive(Debug, Error)]
pub enum MessageValidationError {
    /// The content array is empty or contains a block invalid for the role.
    #[error("message content is empty or contains a block invalid for its role")]
    InvalidContent,
    /// Content or tool-name validation failed.
    #[error("invalid message content: {0}")]
    InvalidContentValue(#[from] ContentValidationError),
    /// Tool failure code or message is invalid.
    #[error("tool failure code or message is invalid")]
    InvalidToolFailure,
    /// Tool-result error flag does not match failure presence.
    #[error("tool-result isError must match error presence")]
    InconsistentToolFailure,
}

fn validate_content(
    content: &[ContentBlock],
    predicate: impl Fn(&ContentBlock) -> bool,
) -> Result<(), MessageValidationError> {
    if content.is_empty() || content.len() > 256 || !content.iter().all(predicate) {
        return Err(MessageValidationError::InvalidContent);
    }
    for block in content {
        block.validate()?;
    }
    Ok(())
}

fn valid_error_code(value: &str) -> bool {
    let mut bytes = value.bytes();
    bytes.next().is_some_and(|byte| byte.is_ascii_lowercase())
        && value.len() <= 128
        && bytes.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
}

impl fmt::Display for StopReason {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}
