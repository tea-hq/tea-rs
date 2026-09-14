use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::{Map, Value, json};
use thiserror::Error;

use crate::envelope::{deserialize_unique_value, validate_read_version};
use crate::{
    ApprovalId, BranchId, CURRENT_PROTOCOL_VERSION, CanonicalMessage, CommandId, CorrelationId,
    FinalOutputFormat, MessageId, MessageRole, ProtocolError, ProtocolMetadata, ProtocolTimestamp,
    ProtocolVersion, SessionId,
};

/// Maximum UTF-8 bytes in a command text fragment.
pub const MAX_COMMAND_TEXT_BYTES: usize = 256 * 1024;
/// Maximum UTF-8 bytes in a model or profile selector.
pub const MAX_SELECTOR_BYTES: usize = 128;

macro_rules! selector {
    ($name:ident, $doc:literal, $validate:ident) => {
        #[doc = $doc]
        #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $name(String);

        impl $name {
            /// Returns the canonical selector text.
            #[must_use]
            pub fn as_str(&self) -> &str {
                &self.0
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str(&self.0)
            }
        }

        impl FromStr for $name {
            type Err = SelectorParseError;

            fn from_str(value: &str) -> Result<Self, Self::Err> {
                $validate(value)?;
                Ok(Self(value.to_owned()))
            }
        }

        impl Serialize for $name {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                $validate(&self.0).map_err(serde::ser::Error::custom)?;
                serializer.serialize_str(&self.0)
            }
        }

        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                let value = String::deserialize(deserializer)?;
                value.parse().map_err(serde::de::Error::custom)
            }
        }
    };
}

selector!(
    ProfileId,
    "A bounded product profile selector.",
    validate_profile_selector
);
selector!(
    ModelId,
    "A bounded canonical model selector.",
    validate_model_selector
);
selector!(
    ProviderId,
    "A bounded canonical model-provider selector.",
    validate_profile_selector
);

/// Error returned when parsing a model or profile selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SelectorParseError {
    /// Selector text is empty, oversized, or contains non-canonical characters.
    #[error(
        "selector must start with lowercase ASCII and contain only supported canonical characters"
    )]
    Invalid,
}

/// A user decision for a pending approval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ApprovalDecision {
    /// Allow only this tool call.
    AllowOnce,
    /// Allow matching operations for the current session.
    AllowSession,
    /// Deny this tool call.
    Deny,
}

/// Stable initial command discriminators.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentCommandType {
    /// Create a new session.
    CreateSession,
    /// Submit a canonical user prompt.
    Prompt,
    /// Inject steering text into the active run.
    Steer,
    /// Queue a follow-up user message.
    FollowUp,
    /// Abort the active run.
    Abort,
    /// Resolve a pending approval.
    ResolveApproval,
    /// Select a model.
    SetModel,
    /// Select a reasoning effort for subsequent turns.
    SetReasoningEffort,
    /// Select a product profile.
    SetProfile,
    /// Request session compaction.
    CompactSession,
    /// Fork from a durable message or branch point.
    ForkSession,
}

impl AgentCommandType {
    /// All initial protocol 1.0 command types.
    pub const ALL: [Self; 11] = [
        Self::CreateSession,
        Self::Prompt,
        Self::Steer,
        Self::FollowUp,
        Self::Abort,
        Self::ResolveApproval,
        Self::SetModel,
        Self::SetReasoningEffort,
        Self::SetProfile,
        Self::CompactSession,
        Self::ForkSession,
    ];
}

/// A provider- and transport-neutral agent command payload.
#[derive(Debug, Clone, PartialEq)]
pub enum AgentCommand {
    /// Create a session using a profile and optional extension metadata.
    CreateSession {
        /// Product profile selector.
        profile_id: ProfileId,
        /// Bounded session extension metadata.
        metadata: ProtocolMetadata,
    },
    /// Submit a canonical user message.
    Prompt {
        /// Canonical user message.
        message: CanonicalMessage,
        /// Optional contract for the run's final visible assistant output.
        final_output_format: Option<FinalOutputFormat>,
    },
    /// Steer the currently running turn.
    Steer {
        /// Bounded steering text.
        text: CommandText,
    },
    /// Queue a follow-up canonical user message.
    FollowUp {
        /// Canonical user message.
        message: CanonicalMessage,
    },
    /// Abort the active run.
    Abort {},
    /// Resolve a pending approval request.
    ResolveApproval {
        /// Approval request identifier.
        approval_id: ApprovalId,
        /// User or policy decision.
        decision: ApprovalDecision,
    },
    /// Select a model for subsequent runs.
    SetModel {
        /// Complete provider-qualified model selector.
        model: crate::ModelRef,
    },
    /// Select reasoning effort for subsequent turns.
    SetReasoningEffort {
        /// Provider-neutral reasoning effort.
        reasoning_effort: crate::ReasoningEffort,
    },
    /// Select a product profile for subsequent runs.
    SetProfile {
        /// Product profile selector.
        profile_id: ProfileId,
    },
    /// Request compaction with an optional bounded instruction.
    CompactSession {
        /// Optional compaction instruction.
        instruction: Option<CommandText>,
    },
    /// Fork the session from a message into a new branch.
    ForkSession {
        /// Source message for the fork.
        from_message_id: MessageId,
        /// Identifier assigned to the new branch.
        branch_id: BranchId,
    },
}

impl AgentCommand {
    /// Returns the stable command discriminator.
    #[must_use]
    pub const fn command_type(&self) -> AgentCommandType {
        match self {
            Self::CreateSession { .. } => AgentCommandType::CreateSession,
            Self::Prompt { .. } => AgentCommandType::Prompt,
            Self::Steer { .. } => AgentCommandType::Steer,
            Self::FollowUp { .. } => AgentCommandType::FollowUp,
            Self::Abort {} => AgentCommandType::Abort,
            Self::ResolveApproval { .. } => AgentCommandType::ResolveApproval,
            Self::SetModel { .. } => AgentCommandType::SetModel,
            Self::SetReasoningEffort { .. } => AgentCommandType::SetReasoningEffort,
            Self::SetProfile { .. } => AgentCommandType::SetProfile,
            Self::CompactSession { .. } => AgentCommandType::CompactSession,
            Self::ForkSession { .. } => AgentCommandType::ForkSession,
        }
    }

    fn validate(&self) -> Result<(), CommandValidationError> {
        match self {
            Self::Prompt { message, .. } | Self::FollowUp { message }
                if message.role() != MessageRole::User =>
            {
                Err(CommandValidationError::MessageMustBeUser)
            }
            Self::Prompt {
                final_output_format: Some(format),
                ..
            } if format.validate().is_err() => {
                Err(CommandValidationError::InvalidFinalOutputFormat)
            }
            _ => Ok(()),
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(
    remote = "AgentCommand",
    tag = "type",
    content = "payload",
    rename_all = "snake_case",
    deny_unknown_fields
)]
enum AgentCommandDef {
    CreateSession {
        #[serde(rename = "profileId")]
        profile_id: ProfileId,
        #[serde(default, skip_serializing_if = "ProtocolMetadata::is_empty")]
        metadata: ProtocolMetadata,
    },
    Prompt {
        message: CanonicalMessage,
        #[serde(
            rename = "finalOutputFormat",
            default,
            skip_serializing_if = "Option::is_none"
        )]
        final_output_format: Option<FinalOutputFormat>,
    },
    Steer {
        text: CommandText,
    },
    FollowUp {
        message: CanonicalMessage,
    },
    Abort {},
    ResolveApproval {
        #[serde(rename = "approvalId")]
        approval_id: ApprovalId,
        decision: ApprovalDecision,
    },
    SetModel {
        model: crate::ModelRef,
    },
    SetReasoningEffort {
        #[serde(rename = "reasoningEffort")]
        reasoning_effort: crate::ReasoningEffort,
    },
    SetProfile {
        #[serde(rename = "profileId")]
        profile_id: ProfileId,
    },
    CompactSession {
        #[serde(skip_serializing_if = "Option::is_none")]
        instruction: Option<CommandText>,
    },
    ForkSession {
        #[serde(rename = "fromMessageId")]
        from_message_id: MessageId,
        #[serde(rename = "branchId")]
        branch_id: BranchId,
    },
}

impl Serialize for AgentCommand {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.validate().map_err(serde::ser::Error::custom)?;
        AgentCommandDef::serialize(self, serializer)
    }
}

impl<'de> Deserialize<'de> for AgentCommand {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let command = AgentCommandDef::deserialize(deserializer)?;
        command.validate().map_err(serde::de::Error::custom)?;
        Ok(command)
    }
}

/// Bounded command text that rejects controls unsafe for logs and transports.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct CommandText(String);

impl CommandText {
    /// Creates validated command text.
    ///
    /// # Errors
    ///
    /// Returns an error when text is empty, oversized, or contains a null character.
    pub fn new(value: impl Into<String>) -> Result<Self, CommandValidationError> {
        let value = value.into();
        if value.is_empty() || value.len() > MAX_COMMAND_TEXT_BYTES || value.contains('\0') {
            return Err(CommandValidationError::InvalidText);
        }
        Ok(Self(value))
    }

    /// Returns the bounded command text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl TryFrom<String> for CommandText {
    type Error = CommandValidationError;

    fn try_from(value: String) -> Result<Self, Self::Error> {
        Self::new(value)
    }
}

impl From<CommandText> for String {
    fn from(value: CommandText) -> Self {
        value.0
    }
}

/// A versioned command transport envelope.
#[derive(Debug, Clone, PartialEq)]
pub struct CommandEnvelope {
    protocol_version: ProtocolVersion,
    command_id: CommandId,
    session_id: Option<SessionId>,
    timestamp: ProtocolTimestamp,
    command: AgentCommand,
}

impl CommandEnvelope {
    /// Creates a validated current-version command envelope.
    ///
    /// # Errors
    ///
    /// Returns an error when session presence or command payload invariants fail.
    pub fn new(
        command_id: CommandId,
        session_id: Option<SessionId>,
        timestamp: ProtocolTimestamp,
        command: AgentCommand,
    ) -> Result<Self, CommandValidationError> {
        let envelope = Self {
            protocol_version: CURRENT_PROTOCOL_VERSION,
            command_id,
            session_id,
            timestamp,
            command,
        };
        envelope.validate()?;
        Ok(envelope)
    }

    /// Decodes a JSON value while preserving unsupported-command classification.
    ///
    /// # Errors
    ///
    /// Returns [`CommandDecodeError::UnsupportedType`] for an unknown safe
    /// discriminator and [`CommandDecodeError::Invalid`] for malformed input.
    pub fn decode_value(value: Value) -> Result<Self, CommandDecodeError> {
        let version = decode_version(&value).map_err(CommandDecodeError::Invalid)?;
        if validate_read_version(version).is_err() {
            return Err(CommandDecodeError::UnsupportedVersion { version });
        }
        let discriminator = value
            .as_object()
            .and_then(|object| object.get("type"))
            .and_then(Value::as_str)
            .ok_or_else(|| CommandDecodeError::Invalid("missing command type".to_owned()))?;
        if discriminator.parse::<AgentCommandTypeText>().is_err() {
            if valid_discriminator(discriminator) {
                return Err(CommandDecodeError::UnsupportedType {
                    command_type: discriminator.to_owned(),
                });
            }
            return Err(CommandDecodeError::Invalid(
                "invalid command type".to_owned(),
            ));
        }
        serde_json::from_value(value)
            .map_err(|error| CommandDecodeError::Invalid(error.to_string()))
    }

    /// Returns the protocol version read from or written to the envelope.
    #[must_use]
    pub const fn protocol_version(&self) -> ProtocolVersion {
        self.protocol_version
    }

    /// Returns the command identifier.
    #[must_use]
    pub const fn command_id(&self) -> CommandId {
        self.command_id
    }

    /// Returns the target session, absent only for session creation.
    #[must_use]
    pub const fn session_id(&self) -> Option<SessionId> {
        self.session_id
    }

    /// Returns the command timestamp.
    #[must_use]
    pub const fn timestamp(&self) -> ProtocolTimestamp {
        self.timestamp
    }

    /// Returns the command payload.
    #[must_use]
    pub const fn command(&self) -> &AgentCommand {
        &self.command
    }

    /// Returns the stable command discriminator.
    #[must_use]
    pub const fn command_type(&self) -> AgentCommandType {
        self.command.command_type()
    }

    fn validate(&self) -> Result<(), CommandValidationError> {
        let is_create = matches!(self.command, AgentCommand::CreateSession { .. });
        if is_create == self.session_id.is_some() {
            return Err(CommandValidationError::InvalidSessionPresence);
        }
        self.command.validate()
    }
}

impl Serialize for CommandEnvelope {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.validate().map_err(serde::ser::Error::custom)?;
        let mut value = serde_json::to_value(&self.command).map_err(serde::ser::Error::custom)?;
        let object = value
            .as_object_mut()
            .ok_or_else(|| serde::ser::Error::custom("command must encode as object"))?;
        insert_envelope_fields(
            object,
            self.protocol_version,
            self.command_id,
            self.session_id,
            self.timestamp,
        );
        value.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CommandEnvelope {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mut value = deserialize_unique_value(deserializer)?;
        let object = value
            .as_object_mut()
            .ok_or_else(|| serde::de::Error::custom("command envelope must be an object"))?;
        let protocol_version = take(object, "protocolVersion").map_err(serde::de::Error::custom)?;
        validate_read_version(protocol_version).map_err(serde::de::Error::custom)?;
        let command_id = take(object, "commandId").map_err(serde::de::Error::custom)?;
        let session_id = take_optional(object, "sessionId").map_err(serde::de::Error::custom)?;
        let timestamp = take(object, "timestamp").map_err(serde::de::Error::custom)?;
        let command = AgentCommand::deserialize(Value::Object(std::mem::take(object)))
            .map_err(serde::de::Error::custom)?;
        let envelope = Self {
            protocol_version,
            command_id,
            session_id,
            timestamp,
            command,
        };
        envelope.validate().map_err(serde::de::Error::custom)?;
        Ok(envelope)
    }
}

/// Failure while decoding an untrusted command envelope.
#[derive(Debug, Error)]
pub enum CommandDecodeError {
    /// The protocol major is unsupported and takes precedence over command type.
    #[error("unsupported protocol version: {version}")]
    UnsupportedVersion {
        /// Received canonical protocol version.
        version: ProtocolVersion,
    },
    /// The discriminator is canonical but unsupported by this host.
    #[error("unsupported command type: {command_type}")]
    UnsupportedType {
        /// Bounded canonical unsupported discriminator.
        command_type: String,
    },
    /// The command envelope or known payload is malformed.
    #[error("invalid command: {0}")]
    Invalid(String),
}

impl CommandDecodeError {
    /// Converts a decode failure to a safe protocol error.
    #[must_use]
    pub fn into_protocol_error(self, correlation_id: CorrelationId) -> ProtocolError {
        match self {
            Self::UnsupportedVersion { version } => {
                ProtocolError::unsupported_protocol_version(correlation_id, version)
            }
            Self::UnsupportedType { command_type } => {
                let details = ProtocolMetadata::protocol_compatibility_details(Some(&command_type));
                ProtocolError::unsupported_command(correlation_id).with_details(details)
            }
            Self::Invalid(_) => ProtocolError::invalid_command(correlation_id),
        }
    }
}

/// Error returned when validating command data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum CommandValidationError {
    /// Only create-session omits a session identifier.
    #[error("sessionId must be absent for create_session and present for every other command")]
    InvalidSessionPresence,
    /// Prompt and follow-up require canonical user messages.
    #[error("prompt and follow_up messages must have user role")]
    MessageMustBeUser,
    /// Prompt final-output format is malformed or exceeds protocol bounds.
    #[error("prompt final-output format is invalid")]
    InvalidFinalOutputFormat,
    /// Command text is empty, oversized, or contains a null character.
    #[error("command text is invalid")]
    InvalidText,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AgentCommandTypeText;

impl FromStr for AgentCommandTypeText {
    type Err = ();

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if [
            "create_session",
            "prompt",
            "steer",
            "follow_up",
            "abort",
            "resolve_approval",
            "set_model",
            "set_reasoning_effort",
            "set_profile",
            "compact_session",
            "fork_session",
        ]
        .contains(&value)
        {
            Ok(Self)
        } else {
            Err(())
        }
    }
}

fn decode_version(value: &Value) -> Result<ProtocolVersion, String> {
    let version = value
        .as_object()
        .and_then(|object| object.get("protocolVersion"))
        .cloned()
        .ok_or_else(|| "missing protocolVersion".to_owned())?;
    serde_json::from_value(version).map_err(|error| error.to_string())
}

fn validate_profile_selector(value: &str) -> Result<(), SelectorParseError> {
    validate_selector(value, false)
}

fn validate_model_selector(value: &str) -> Result<(), SelectorParseError> {
    validate_selector(value, true)
}

fn validate_selector(value: &str, allow_colon: bool) -> Result<(), SelectorParseError> {
    let mut bytes = value.bytes();
    if value.len() > MAX_SELECTOR_BYTES
        || !bytes.next().is_some_and(|byte| byte.is_ascii_lowercase())
        || !bytes.all(|byte| {
            byte.is_ascii_lowercase()
                || byte.is_ascii_digit()
                || matches!(byte, b'_' | b'-' | b'.' | b'/')
                || (allow_colon && byte == b':')
        })
    {
        Err(SelectorParseError::Invalid)
    } else {
        Ok(())
    }
}

fn valid_discriminator(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_'))
}

fn insert_envelope_fields(
    object: &mut Map<String, Value>,
    version: ProtocolVersion,
    command_id: CommandId,
    session_id: Option<SessionId>,
    timestamp: ProtocolTimestamp,
) {
    object.insert("protocolVersion".to_owned(), json!(version));
    object.insert("commandId".to_owned(), json!(command_id));
    if let Some(session_id) = session_id {
        object.insert("sessionId".to_owned(), json!(session_id));
    }
    object.insert("timestamp".to_owned(), json!(timestamp));
}

fn take<T>(object: &mut Map<String, Value>, key: &str) -> Result<T, serde_json::Error>
where
    T: for<'de> Deserialize<'de>,
{
    serde_json::from_value(object.remove(key).unwrap_or(Value::Null))
}

fn take_optional<T>(
    object: &mut Map<String, Value>,
    key: &str,
) -> Result<Option<T>, serde_json::Error>
where
    T: for<'de> Deserialize<'de>,
{
    object.remove(key).map_or(Ok(None), serde_json::from_value)
}
