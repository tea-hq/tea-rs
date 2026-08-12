use std::fmt;
use std::str::FromStr;

use thiserror::Error;

use crate::SkillId;

/// Maximum UTF-8 bytes in one skill description.
pub const MAX_SKILL_DESCRIPTION_BYTES: usize = 4096;
/// Maximum UTF-8 bytes in explicit skill arguments.
pub const MAX_SKILL_ARGUMENT_BYTES: usize = 16 * 1024;

/// Bounded declarative skill metadata; it does not execute the skill.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillMetadata {
    id: SkillId,
    description: String,
}

impl SkillMetadata {
    /// Creates one skill metadata entry.
    ///
    /// # Errors
    ///
    /// Returns an error for empty, oversized, or null-containing description.
    pub fn new(id: SkillId, description: impl Into<String>) -> Result<Self, SkillError> {
        let description = description.into();
        if description.is_empty()
            || description.len() > MAX_SKILL_DESCRIPTION_BYTES
            || description.contains('\0')
        {
            return Err(SkillError::InvalidDescription);
        }
        Ok(Self { id, description })
    }
    /// Returns skill identity.
    #[must_use]
    pub const fn id(&self) -> &SkillId {
        &self.id
    }
    /// Returns model-visible skill description.
    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }
    /// Returns the sole explicit invocation form.
    #[must_use]
    pub fn invocation(&self) -> SkillInvocation {
        SkillInvocation {
            skill_id: self.id.clone(),
        }
    }

    /// Returns the canonical slash command without explicit arguments.
    #[must_use]
    pub fn command(&self) -> SkillCommand {
        self.invocation().command()
    }
}

/// Parsed explicit `@skill <skill-id>` invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillInvocation {
    skill_id: SkillId,
}

impl SkillInvocation {
    /// Returns invoked skill.
    #[must_use]
    pub const fn skill_id(&self) -> &SkillId {
        &self.skill_id
    }

    /// Converts the legacy metadata invocation to a typed command without
    /// arguments.
    #[must_use]
    pub fn command(&self) -> SkillCommand {
        self.clone().into()
    }
}

impl FromStr for SkillInvocation {
    type Err = SkillError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let skill_id = value
            .strip_prefix("@skill ")
            .filter(|remaining| !remaining.contains(' '))
            .ok_or(SkillError::InvalidInvocation)?
            .parse()
            .map_err(|_| SkillError::InvalidInvocation)?;
        Ok(Self { skill_id })
    }
}

impl fmt::Display for SkillInvocation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "@skill {}", self.skill_id)
    }
}

/// Typed canonical `/skill:<skill-id> [args]` command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillCommand {
    skill_id: SkillId,
    arguments: String,
}

impl SkillCommand {
    /// Creates a validated skill command.
    ///
    /// # Errors
    ///
    /// Returns an error for null-containing or oversized arguments.
    pub fn new(skill_id: SkillId, arguments: impl Into<String>) -> Result<Self, SkillError> {
        let arguments = arguments.into();
        if arguments.len() > MAX_SKILL_ARGUMENT_BYTES || arguments.contains('\0') {
            return Err(SkillError::InvalidArguments);
        }
        Ok(Self {
            skill_id,
            arguments,
        })
    }

    /// Returns the invoked skill.
    #[must_use]
    pub const fn skill_id(&self) -> &SkillId {
        &self.skill_id
    }

    /// Returns uninterpreted invocation arguments.
    #[must_use]
    pub fn arguments(&self) -> &str {
        &self.arguments
    }
}

impl From<SkillInvocation> for SkillCommand {
    fn from(invocation: SkillInvocation) -> Self {
        Self {
            skill_id: invocation.skill_id,
            arguments: String::new(),
        }
    }
}

impl FromStr for SkillCommand {
    type Err = SkillError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let remaining = value
            .strip_prefix("/skill:")
            .ok_or(SkillError::InvalidCommand)?;
        let (skill_id, arguments) = remaining
            .split_once(' ')
            .map_or((remaining, ""), |(skill_id, arguments)| {
                (skill_id, arguments.trim_start())
            });
        let skill_id = skill_id.parse().map_err(|_| SkillError::InvalidCommand)?;
        Self::new(skill_id, arguments)
    }
}

impl fmt::Display for SkillCommand {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "/skill:{}", self.skill_id)?;
        if !self.arguments.is_empty() {
            write!(formatter, " {}", self.arguments)?;
        }
        Ok(())
    }
}

/// Invalid skill metadata or invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SkillError {
    /// Description violates bounds.
    #[error("skill description is invalid")]
    InvalidDescription,
    /// Invocation is not exact explicit skill syntax.
    #[error("skill invocation must use exact '@skill <skill-id>' syntax")]
    InvalidInvocation,
    /// Slash command is not canonical skill command syntax.
    #[error("skill command must use '/skill:<skill-id> [args]' syntax")]
    InvalidCommand,
    /// Explicit arguments violate their byte or null bound.
    #[error("skill command arguments are invalid")]
    InvalidArguments,
}
