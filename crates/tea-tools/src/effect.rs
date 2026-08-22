use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

/// Declared capability or external effect of a tool invocation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ToolEffect {
    /// Read filesystem content.
    FsRead,
    /// Create or modify filesystem content.
    FsWrite,
    /// Delete filesystem content.
    FsDelete,
    /// Spawn or control an operating-system process.
    ProcessSpawn,
    /// Perform a network request.
    NetworkRequest,
    /// Read credentials or secret material.
    CredentialRead,
    /// Read clipboard content.
    ClipboardRead,
    /// Read bounded data from a host-owned external source.
    ExternalRead,
    /// Require direct user interaction.
    UserInteraction,
    /// Mutate state in an external system.
    ExternalMutation,
    /// Future namespaced effect unknown to this runtime version.
    Unknown(String),
}

impl ToolEffect {
    /// Returns the canonical dotted effect name.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::FsRead => "fs.read",
            Self::FsWrite => "fs.write",
            Self::FsDelete => "fs.delete",
            Self::ProcessSpawn => "process.spawn",
            Self::NetworkRequest => "network.request",
            Self::CredentialRead => "credential.read",
            Self::ClipboardRead => "clipboard.read",
            Self::ExternalRead => "external.read",
            Self::UserInteraction => "user.interaction",
            Self::ExternalMutation => "external.mutation",
            Self::Unknown(value) => value,
        }
    }

    /// Returns whether this runtime does not understand the effect semantics.
    #[must_use]
    pub const fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown(_))
    }

    pub(crate) const fn is_read_only(&self) -> bool {
        matches!(
            self,
            Self::FsRead | Self::CredentialRead | Self::ClipboardRead | Self::ExternalRead
        )
    }
}

impl Serialize for ToolEffect {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ToolEffect {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        String::deserialize(deserializer)?
            .parse()
            .map_err(serde::de::Error::custom)
    }
}

impl FromStr for ToolEffect {
    type Err = ToolEffectParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let known = match value {
            "fs.read" => Some(Self::FsRead),
            "fs.write" => Some(Self::FsWrite),
            "fs.delete" => Some(Self::FsDelete),
            "process.spawn" => Some(Self::ProcessSpawn),
            "network.request" => Some(Self::NetworkRequest),
            "credential.read" => Some(Self::CredentialRead),
            "clipboard.read" => Some(Self::ClipboardRead),
            "external.read" => Some(Self::ExternalRead),
            "user.interaction" => Some(Self::UserInteraction),
            "external.mutation" => Some(Self::ExternalMutation),
            _ => None,
        };
        if let Some(known) = known {
            return Ok(known);
        }
        if value.len() > 256 || !valid_namespaced_effect(value) {
            return Err(ToolEffectParseError);
        }
        Ok(Self::Unknown(value.to_owned()))
    }
}

/// Error returned when parsing a tool effect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
#[error("tool effect must be known or use a lowercase namespaced dotted name")]
pub struct ToolEffectParseError;

fn valid_namespaced_effect(value: &str) -> bool {
    let segments = value.split('.').collect::<Vec<_>>();
    segments.len() >= 3
        && segments.iter().all(|segment| {
            let mut bytes = segment.bytes();
            bytes.next().is_some_and(|byte| byte.is_ascii_lowercase())
                && bytes.all(|byte| {
                    byte.is_ascii_lowercase()
                        || byte.is_ascii_digit()
                        || matches!(byte, b'-' | b'_')
                })
        })
}
