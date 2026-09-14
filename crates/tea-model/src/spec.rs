use std::fmt;
use std::str::FromStr;

use tea_protocol::{ModelId, ModelRef, ProviderId, ReasoningEffort, TokenCount};
use thiserror::Error;

use crate::HostedToolKind;

const MAX_DISPLAY_NAME_BYTES: usize = 256;

/// Human-readable bounded model name.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ModelDisplayName(String);

impl ModelDisplayName {
    /// Returns the model display name.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for ModelDisplayName {
    type Err = ModelTextParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        if value.is_empty()
            || value.len() > MAX_DISPLAY_NAME_BYTES
            || value.chars().any(char::is_control)
        {
            return Err(ModelTextParseError::InvalidDisplayName);
        }
        Ok(Self(value.to_owned()))
    }
}

impl fmt::Display for ModelDisplayName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Error returned when parsing bounded model text values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ModelTextParseError {
    /// Display name is empty, oversized, or contains control characters.
    #[error("model display name is invalid")]
    InvalidDisplayName,
}

const CAP_IMAGE_INPUT: u16 = 1 << 0;
const CAP_REASONING: u16 = 1 << 1;
const CAP_TOOLS: u16 = 1 << 2;
const CAP_PARALLEL_TOOLS: u16 = 1 << 3;
const CAP_USAGE: u16 = 1 << 4;
const CAP_HOSTED_WEB_SEARCH: u16 = 1 << 5;
const CAP_FINAL_JSON_OBJECT: u16 = 1 << 6;
const CAP_FINAL_JSON_SCHEMA: u16 = 1 << 7;
const CAP_FINAL_JSON_SCHEMA_WITH_TOOLS: u16 = 1 << 8;

/// Provider-neutral capabilities advertised by one model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelCapabilities(u16);

impl ModelCapabilities {
    /// Creates the minimum supported capability set: text input only.
    #[must_use]
    pub const fn text() -> Self {
        Self(0)
    }

    /// Enables image input.
    #[must_use]
    pub const fn with_image_input(mut self) -> Self {
        self.0 |= CAP_IMAGE_INPUT;
        self
    }

    /// Enables reasoning output and options.
    #[must_use]
    pub const fn with_reasoning(mut self) -> Self {
        self.0 |= CAP_REASONING;
        self
    }

    /// Enables tool calls and optionally parallel tool calls.
    #[must_use]
    pub const fn with_tools(mut self, parallel: bool) -> Self {
        self.0 |= CAP_TOOLS;
        if parallel {
            self.0 |= CAP_PARALLEL_TOOLS;
        } else {
            self.0 &= !CAP_PARALLEL_TOOLS;
        }
        self
    }

    /// Enables normalized usage reporting.
    #[must_use]
    pub const fn with_usage_reporting(mut self) -> Self {
        self.0 |= CAP_USAGE;
        self
    }

    /// Enables one provider-hosted tool kind for this model and endpoint.
    #[must_use]
    pub const fn with_hosted_tool(mut self, kind: HostedToolKind) -> Self {
        match kind {
            HostedToolKind::WebSearch => self.0 |= CAP_HOSTED_WEB_SEARCH,
        }
        self
    }

    /// Enables JSON-object final output for this model and endpoint.
    #[must_use]
    pub const fn with_final_json_object(mut self) -> Self {
        self.0 |= CAP_FINAL_JSON_OBJECT;
        self
    }

    /// Enables JSON-Schema-constrained final output for this model and endpoint.
    #[must_use]
    pub const fn with_final_json_schema(mut self) -> Self {
        self.0 |= CAP_FINAL_JSON_SCHEMA;
        self
    }

    /// Enables JSON-Schema final output combined with model-visible tools.
    #[must_use]
    pub const fn with_final_json_schema_with_tools(mut self) -> Self {
        self.0 |= CAP_FINAL_JSON_SCHEMA | CAP_FINAL_JSON_SCHEMA_WITH_TOOLS;
        self
    }

    /// Returns whether text input is accepted.
    #[must_use]
    pub const fn accepts_text(self) -> bool {
        true
    }

    /// Returns whether image input is accepted.
    #[must_use]
    pub const fn accepts_images(self) -> bool {
        self.0 & CAP_IMAGE_INPUT != 0
    }

    /// Returns whether reasoning is supported.
    #[must_use]
    pub const fn supports_reasoning(self) -> bool {
        self.0 & CAP_REASONING != 0
    }

    /// Returns whether tool calls are supported.
    #[must_use]
    pub const fn supports_tools(self) -> bool {
        self.0 & CAP_TOOLS != 0
    }

    /// Returns whether several tool calls may be requested in one response.
    #[must_use]
    pub const fn supports_parallel_tool_calls(self) -> bool {
        self.0 & CAP_PARALLEL_TOOLS != 0
    }

    /// Returns whether usage is reported.
    #[must_use]
    pub const fn reports_usage(self) -> bool {
        self.0 & CAP_USAGE != 0
    }

    /// Returns whether the model/endpoint supports one provider-hosted tool.
    #[must_use]
    pub const fn supports_hosted_tool(self, kind: HostedToolKind) -> bool {
        match kind {
            HostedToolKind::WebSearch => self.0 & CAP_HOSTED_WEB_SEARCH != 0,
        }
    }

    /// Returns whether JSON-object final output is supported.
    #[must_use]
    pub const fn supports_final_json_object(self) -> bool {
        self.0 & CAP_FINAL_JSON_OBJECT != 0
    }

    /// Returns whether JSON-Schema-constrained final output is supported.
    #[must_use]
    pub const fn supports_final_json_schema(self) -> bool {
        self.0 & CAP_FINAL_JSON_SCHEMA != 0
    }

    /// Returns whether JSON-Schema final output may be combined with tools.
    #[must_use]
    pub const fn supports_final_json_schema_with_tools(self) -> bool {
        self.0 & CAP_FINAL_JSON_SCHEMA_WITH_TOOLS != 0
    }
}

impl Default for ModelCapabilities {
    fn default() -> Self {
        Self::text()
    }
}

/// Validated provider-neutral reasoning levels supported by one model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningProfile {
    default_effort: ReasoningEffort,
    supported_efforts: Vec<ReasoningEffort>,
}

impl ReasoningProfile {
    /// Creates a profile with a supported default and unique effort levels.
    ///
    /// # Errors
    ///
    /// Returns an error when no levels are supplied, a level is duplicated,
    /// or the default level is not supported.
    pub fn new(
        default_effort: ReasoningEffort,
        supported_efforts: impl IntoIterator<Item = ReasoningEffort>,
    ) -> Result<Self, ModelSpecError> {
        let mut supported_efforts = supported_efforts.into_iter().collect::<Vec<_>>();
        if supported_efforts.is_empty() {
            return Err(ModelSpecError::EmptyReasoningEfforts);
        }
        supported_efforts.sort_unstable();
        if supported_efforts
            .windows(2)
            .any(|levels| levels[0] == levels[1])
        {
            return Err(ModelSpecError::DuplicateReasoningEffort);
        }
        if !supported_efforts.contains(&default_effort) {
            return Err(ModelSpecError::ReasoningDefaultUnsupported);
        }
        Ok(Self {
            default_effort,
            supported_efforts,
        })
    }

    pub(crate) fn compatible_default() -> Self {
        Self {
            default_effort: ReasoningEffort::Medium,
            supported_efforts: ReasoningEffort::SHORTCUT_LEVELS.to_vec(),
        }
    }

    /// Returns the model default when a session has no explicit selection.
    #[must_use]
    pub const fn default_effort(&self) -> ReasoningEffort {
        self.default_effort
    }

    /// Returns supported levels in canonical ascending order.
    #[must_use]
    pub fn supported_efforts(&self) -> &[ReasoningEffort] {
        &self.supported_efforts
    }

    /// Resolves one requested level using Pi's upward-first clamp rule.
    #[must_use]
    pub fn resolve(&self, requested: ReasoningEffort) -> ReasoningResolution {
        let effective = if self.supported_efforts.contains(&requested) {
            requested
        } else {
            self.supported_efforts
                .iter()
                .copied()
                .find(|candidate| *candidate > requested)
                .or_else(|| {
                    self.supported_efforts
                        .iter()
                        .rev()
                        .copied()
                        .find(|candidate| *candidate < requested)
                })
                .unwrap_or(self.default_effort)
        };
        ReasoningResolution {
            requested,
            effective,
        }
    }
}

/// Result of resolving a requested reasoning level for one model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReasoningResolution {
    requested: ReasoningEffort,
    effective: ReasoningEffort,
}

impl ReasoningResolution {
    /// Returns the caller's requested level.
    #[must_use]
    pub const fn requested(self) -> ReasoningEffort {
        self.requested
    }

    /// Returns the supported level selected for the model.
    #[must_use]
    pub const fn effective(self) -> ReasoningEffort {
        self.effective
    }

    /// Returns whether resolution changed the caller's request.
    #[must_use]
    pub fn was_clamped(self) -> bool {
        self.requested != self.effective
    }
}

/// Validated provider-neutral model specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpec {
    model_ref: ModelRef,
    display_name: ModelDisplayName,
    context_window_tokens: TokenCount,
    max_output_tokens: TokenCount,
    capabilities: ModelCapabilities,
    reasoning_profile: Option<ReasoningProfile>,
}

impl ModelSpec {
    /// Creates a validated model specification.
    ///
    /// # Errors
    ///
    /// Returns an error when context/output limits are zero or output exceeds
    /// the full context window.
    pub fn new(
        model_id: ModelId,
        provider_id: ProviderId,
        display_name: ModelDisplayName,
        context_window_tokens: TokenCount,
        max_output_tokens: TokenCount,
        capabilities: ModelCapabilities,
    ) -> Result<Self, ModelSpecError> {
        if context_window_tokens.get() == 0 {
            return Err(ModelSpecError::EmptyContextWindow);
        }
        if max_output_tokens.get() == 0 {
            return Err(ModelSpecError::EmptyOutputLimit);
        }
        if max_output_tokens > context_window_tokens {
            return Err(ModelSpecError::OutputExceedsContext);
        }
        let reasoning_profile = capabilities
            .supports_reasoning()
            .then(ReasoningProfile::compatible_default);
        Ok(Self {
            model_ref: ModelRef::new(provider_id, model_id),
            display_name,
            context_window_tokens,
            max_output_tokens,
            capabilities,
            reasoning_profile,
        })
    }

    /// Replaces the model's supported/default reasoning metadata.
    #[must_use]
    pub fn with_reasoning_profile(mut self, profile: ReasoningProfile) -> Self {
        self.capabilities = self.capabilities.with_reasoning();
        self.reasoning_profile = Some(profile);
        self
    }

    /// Returns the canonical model selector.
    #[must_use]
    pub const fn model_id(&self) -> &ModelId {
        self.model_ref.model_id()
    }

    /// Returns the owning provider selector.
    #[must_use]
    pub const fn provider_id(&self) -> &ProviderId {
        self.model_ref.provider_id()
    }

    /// Returns the complete provider-qualified model identity.
    #[must_use]
    pub const fn model_ref(&self) -> &ModelRef {
        &self.model_ref
    }

    /// Returns the human-readable model name.
    #[must_use]
    pub const fn display_name(&self) -> &ModelDisplayName {
        &self.display_name
    }

    /// Returns the full context-window limit.
    #[must_use]
    pub const fn context_window_tokens(&self) -> TokenCount {
        self.context_window_tokens
    }

    /// Returns the maximum generated output tokens.
    #[must_use]
    pub const fn max_output_tokens(&self) -> TokenCount {
        self.max_output_tokens
    }

    /// Returns provider-neutral model capabilities.
    #[must_use]
    pub const fn capabilities(&self) -> ModelCapabilities {
        self.capabilities
    }

    /// Returns supported/default reasoning metadata for a reasoning model.
    #[must_use]
    pub const fn reasoning_profile(&self) -> Option<&ReasoningProfile> {
        self.reasoning_profile.as_ref()
    }

    /// Resolves an optional session selection against this model.
    ///
    /// A missing selection inherits a reasoning model's default. A requested
    /// level on a non-reasoning model resolves to explicit `off`.
    #[must_use]
    pub fn resolve_reasoning(
        &self,
        requested: Option<ReasoningEffort>,
    ) -> Option<ReasoningResolution> {
        match (&self.reasoning_profile, requested) {
            (Some(profile), Some(requested)) => Some(profile.resolve(requested)),
            (Some(profile), None) => Some(profile.resolve(profile.default_effort())),
            (None, Some(requested)) => Some(ReasoningResolution {
                requested,
                effective: ReasoningEffort::Off,
            }),
            (None, None) => None,
        }
    }
}

/// Error returned when model limits are inconsistent.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ModelSpecError {
    /// Context window must contain at least one token.
    #[error("model context window must be non-zero")]
    EmptyContextWindow,
    /// Output limit must contain at least one token.
    #[error("model output limit must be non-zero")]
    EmptyOutputLimit,
    /// Output token limit exceeds the full context window.
    #[error("model output limit exceeds context window")]
    OutputExceedsContext,
    /// A reasoning profile must advertise at least one level.
    #[error("model reasoning profile is empty")]
    EmptyReasoningEfforts,
    /// A reasoning profile cannot repeat a canonical level.
    #[error("model reasoning profile contains a duplicate effort")]
    DuplicateReasoningEffort,
    /// The model default must be one of its supported levels.
    #[error("model reasoning default is unsupported")]
    ReasoningDefaultUnsupported,
}
