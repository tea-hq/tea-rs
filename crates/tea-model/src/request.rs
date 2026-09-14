use std::collections::BTreeSet;

use serde_json::Value;
use tea_protocol::{
    CanonicalMessage, ContentBlock, FinalOutputFormat, ModelId, ProtocolMetadata, ReasoningEffort,
    TokenCount,
};
use thiserror::Error;

use crate::{HostedToolKind, HostedToolOptions, ModelSpec};

/// Maximum UTF-8 bytes in a system prompt.
pub const MAX_SYSTEM_PROMPT_BYTES: usize = 1024 * 1024;
/// Maximum canonical messages in one model request.
pub const MAX_REQUEST_MESSAGES: usize = 4096;
/// Maximum model-visible tools in one request.
pub const MAX_MODEL_TOOLS: usize = 256;
/// Maximum UTF-8 bytes in one model-visible tool description.
pub const MAX_TOOL_DESCRIPTION_BYTES: usize = 16 * 1024;
/// Maximum encoded JSON bytes in one tool input schema.
pub const MAX_TOOL_SCHEMA_BYTES: usize = 256 * 1024;
/// Maximum JSON nesting depth in one tool input schema.
pub const MAX_TOOL_SCHEMA_DEPTH: usize = 32;

/// Provider-neutral reasoning request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReasoningOptions {
    effort: ReasoningEffort,
    budget_tokens: Option<TokenCount>,
}

impl ReasoningOptions {
    /// Creates reasoning options with provider/model default token budgeting.
    #[must_use]
    pub const fn new(effort: ReasoningEffort) -> Self {
        Self {
            effort,
            budget_tokens: None,
        }
    }

    /// Adds a requested reasoning-token budget.
    #[must_use]
    pub const fn with_budget(mut self, budget_tokens: TokenCount) -> Self {
        self.budget_tokens = Some(budget_tokens);
        self
    }

    /// Returns the requested effort.
    #[must_use]
    pub const fn effort(self) -> ReasoningEffort {
        self.effort
    }

    /// Returns the optional reasoning-token budget.
    #[must_use]
    pub const fn budget_tokens(self) -> Option<TokenCount> {
        self.budget_tokens
    }
}

/// Model-visible client function tool.
#[derive(Debug, Clone, PartialEq)]
pub struct FunctionToolDefinition {
    name: String,
    description: String,
    input_schema: Value,
}

impl FunctionToolDefinition {
    /// Creates a validated model-visible tool definition.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid names/descriptions or non-object,
    /// oversized, or excessively nested schemas.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
    ) -> Result<Self, ModelRequestError> {
        let name = name.into();
        let description = description.into();
        validate_tool_contract(&name, &description, &input_schema)?;
        Ok(Self {
            name,
            description,
            input_schema,
        })
    }

    /// Returns the canonical tool name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the model-visible description.
    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Returns the provider-neutral input JSON Schema.
    #[must_use]
    pub const fn input_schema(&self) -> &Value {
        &self.input_schema
    }
}

/// Model-visible provider-hosted tool.
#[derive(Debug, Clone, PartialEq)]
pub struct HostedToolDefinition {
    name: String,
    description: String,
    input_schema: Value,
    options: HostedToolOptions,
}

impl HostedToolDefinition {
    fn new(
        description: impl Into<String>,
        input_schema: Value,
        options: HostedToolOptions,
    ) -> Result<Self, ModelRequestError> {
        let name = options.kind().name().to_owned();
        let description = description.into();
        validate_tool_contract(&name, &description, &input_schema)?;
        Ok(Self {
            name,
            description,
            input_schema,
            options,
        })
    }

    /// Returns the canonical hosted tool name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the model-visible description.
    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Returns the stable input schema used by client fallback and accounting.
    #[must_use]
    pub const fn input_schema(&self) -> &Value {
        &self.input_schema
    }

    /// Returns the required hosted capability kind.
    #[must_use]
    pub const fn kind(&self) -> HostedToolKind {
        self.options.kind()
    }

    /// Returns portable hosted-tool options.
    #[must_use]
    pub const fn options(&self) -> &HostedToolOptions {
        &self.options
    }
}

/// One model-visible client function or provider-hosted tool definition.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelToolDefinition {
    /// A function call that the client must execute.
    Function(FunctionToolDefinition),
    /// A tool executed inside the provider response lifecycle.
    Hosted(HostedToolDefinition),
}

impl ModelToolDefinition {
    /// Creates a validated client function definition.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid names/descriptions or non-object,
    /// oversized, or excessively nested schemas.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
    ) -> Result<Self, ModelRequestError> {
        FunctionToolDefinition::new(name, description, input_schema).map(Self::Function)
    }

    /// Creates a validated provider-hosted definition.
    ///
    /// # Errors
    ///
    /// Returns an error when the common model contract violates request bounds.
    pub fn hosted(
        description: impl Into<String>,
        input_schema: Value,
        options: HostedToolOptions,
    ) -> Result<Self, ModelRequestError> {
        HostedToolDefinition::new(description, input_schema, options).map(Self::Hosted)
    }

    /// Returns the canonical model-visible name.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Function(tool) => &tool.name,
            Self::Hosted(tool) => &tool.name,
        }
    }

    /// Returns the model-visible description.
    #[must_use]
    pub fn description(&self) -> &str {
        match self {
            Self::Function(tool) => &tool.description,
            Self::Hosted(tool) => &tool.description,
        }
    }

    /// Returns the provider-neutral input JSON Schema.
    #[must_use]
    pub const fn input_schema(&self) -> &Value {
        match self {
            Self::Function(tool) => &tool.input_schema,
            Self::Hosted(tool) => &tool.input_schema,
        }
    }

    /// Returns this definition as a client function.
    #[must_use]
    pub const fn as_function(&self) -> Option<&FunctionToolDefinition> {
        match self {
            Self::Function(tool) => Some(tool),
            Self::Hosted(_) => None,
        }
    }

    /// Returns this definition as a hosted tool.
    #[must_use]
    pub const fn as_hosted(&self) -> Option<&HostedToolDefinition> {
        match self {
            Self::Function(_) => None,
            Self::Hosted(tool) => Some(tool),
        }
    }

    /// Returns the hosted capability kind, if any.
    #[must_use]
    pub const fn hosted_kind(&self) -> Option<HostedToolKind> {
        match self {
            Self::Function(_) => None,
            Self::Hosted(tool) => Some(tool.kind()),
        }
    }
}

/// Immutable provider-neutral snapshot for one model request.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelRequest {
    model_id: ModelId,
    system_prompt: Option<String>,
    messages: Vec<CanonicalMessage>,
    tools: Vec<ModelToolDefinition>,
    allow_parallel_tool_calls: bool,
    reasoning: Option<ReasoningOptions>,
    final_output_format: Option<FinalOutputFormat>,
    max_output_tokens: Option<TokenCount>,
    metadata: ProtocolMetadata,
}

impl ModelRequest {
    /// Creates a request for a non-empty canonical transcript.
    ///
    /// # Errors
    ///
    /// Returns an error when messages are empty, too numerous, or invalid at
    /// the canonical protocol wire boundary.
    pub fn new(
        model_id: ModelId,
        messages: Vec<CanonicalMessage>,
    ) -> Result<Self, ModelRequestError> {
        validate_messages(&messages)?;
        Ok(Self {
            model_id,
            system_prompt: None,
            messages,
            tools: Vec::new(),
            allow_parallel_tool_calls: false,
            reasoning: None,
            final_output_format: None,
            max_output_tokens: None,
            metadata: ProtocolMetadata::default(),
        })
    }

    /// Adds a bounded non-empty system prompt.
    ///
    /// # Errors
    ///
    /// Returns an error when the prompt is empty, oversized, or contains a
    /// null character.
    pub fn with_system_prompt(
        mut self,
        system_prompt: impl Into<String>,
    ) -> Result<Self, ModelRequestError> {
        let system_prompt = system_prompt.into();
        if system_prompt.is_empty()
            || system_prompt.len() > MAX_SYSTEM_PROMPT_BYTES
            || system_prompt.contains('\0')
        {
            return Err(ModelRequestError::InvalidSystemPrompt);
        }
        self.system_prompt = Some(system_prompt);
        Ok(self)
    }

    /// Adds model-visible tools in deterministic source order.
    ///
    /// # Errors
    ///
    /// Returns an error for an oversized list or duplicate tool names.
    pub fn with_tools(
        mut self,
        tools: Vec<ModelToolDefinition>,
        allow_parallel: bool,
    ) -> Result<Self, ModelRequestError> {
        if tools.len() > MAX_MODEL_TOOLS {
            return Err(ModelRequestError::TooManyTools);
        }
        let mut names = BTreeSet::new();
        if tools.iter().any(|tool| !names.insert(tool.name())) {
            return Err(ModelRequestError::DuplicateToolName);
        }
        self.tools = tools;
        self.allow_parallel_tool_calls = allow_parallel;
        Ok(self)
    }

    /// Adds provider-neutral reasoning options.
    #[must_use]
    pub const fn with_reasoning(mut self, reasoning: ReasoningOptions) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    /// Adds the required format for the run's final visible assistant output.
    #[must_use]
    pub fn with_final_output_format(mut self, final_output_format: FinalOutputFormat) -> Self {
        self.final_output_format = Some(final_output_format);
        self
    }

    /// Adds a requested output-token limit.
    #[must_use]
    pub const fn with_max_output_tokens(mut self, max_output_tokens: TokenCount) -> Self {
        self.max_output_tokens = Some(max_output_tokens);
        self
    }

    /// Adds bounded namespaced request metadata.
    #[must_use]
    pub fn with_metadata(mut self, metadata: ProtocolMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Validates this request against one advertised model.
    ///
    /// # Errors
    ///
    /// Returns an error when model identity, capability, or output limits do
    /// not satisfy the request.
    pub fn validate_for(&self, model: &ModelSpec) -> Result<(), ModelRequestError> {
        if self.model_id != *model.model_id() {
            return Err(ModelRequestError::ModelMismatch);
        }
        validate_messages(&self.messages)?;
        if self
            .final_output_format
            .as_ref()
            .is_some_and(|format| format.validate().is_err())
        {
            return Err(ModelRequestError::InvalidFinalOutputFormat);
        }
        let capabilities = model.capabilities();
        if request_contains_image(&self.messages) && !capabilities.accepts_images() {
            return Err(ModelRequestError::ImageInputUnsupported);
        }
        if let Some(reasoning) = self.reasoning {
            let Some(profile) = model.reasoning_profile() else {
                return Err(ModelRequestError::ReasoningUnsupported);
            };
            if !profile.supported_efforts().contains(&reasoning.effort()) {
                return Err(ModelRequestError::ReasoningEffortUnsupported);
            }
        }
        if self.tools.iter().any(|tool| tool.as_function().is_some())
            && !capabilities.supports_tools()
        {
            return Err(ModelRequestError::ToolsUnsupported);
        }
        if self.tools.iter().any(|tool| {
            tool.hosted_kind()
                .is_some_and(|kind| !capabilities.supports_hosted_tool(kind))
        }) {
            return Err(ModelRequestError::HostedToolUnsupported);
        }
        if self.allow_parallel_tool_calls
            && self.tools.iter().any(|tool| tool.as_function().is_some())
            && !capabilities.supports_parallel_tool_calls()
        {
            return Err(ModelRequestError::ParallelToolsUnsupported);
        }
        match self.final_output_format.as_ref() {
            Some(FinalOutputFormat::JsonObject) if !capabilities.supports_final_json_object() => {
                return Err(ModelRequestError::FinalJsonObjectUnsupported);
            }
            Some(FinalOutputFormat::JsonSchema { .. })
                if !capabilities.supports_final_json_schema() =>
            {
                return Err(ModelRequestError::FinalJsonSchemaUnsupported);
            }
            Some(FinalOutputFormat::JsonSchema { .. })
                if !self.tools.is_empty()
                    && !capabilities.supports_final_json_schema_with_tools() =>
            {
                return Err(ModelRequestError::FinalJsonSchemaWithToolsUnsupported);
            }
            _ => {}
        }
        let output_limit = self
            .max_output_tokens
            .unwrap_or_else(|| model.max_output_tokens());
        if output_limit.get() == 0 || output_limit > model.max_output_tokens() {
            return Err(ModelRequestError::OutputLimitUnsupported);
        }
        if self
            .reasoning
            .and_then(ReasoningOptions::budget_tokens)
            .is_some_and(|budget| budget.get() == 0 || budget > output_limit)
        {
            return Err(ModelRequestError::ReasoningBudgetUnsupported);
        }
        Ok(())
    }

    /// Returns the selected model.
    #[must_use]
    pub const fn model_id(&self) -> &ModelId {
        &self.model_id
    }

    /// Returns the optional system prompt.
    #[must_use]
    pub fn system_prompt(&self) -> Option<&str> {
        self.system_prompt.as_deref()
    }

    /// Returns canonical messages in source order.
    #[must_use]
    pub fn messages(&self) -> &[CanonicalMessage] {
        &self.messages
    }

    /// Returns model-visible tools in source order.
    #[must_use]
    pub fn tools(&self) -> &[ModelToolDefinition] {
        &self.tools
    }

    /// Returns whether the model may request several tools in one response.
    #[must_use]
    pub const fn allow_parallel_tool_calls(&self) -> bool {
        self.allow_parallel_tool_calls
    }

    /// Returns reasoning options.
    #[must_use]
    pub const fn reasoning(&self) -> Option<ReasoningOptions> {
        self.reasoning
    }

    /// Returns the required final-output format, when present.
    #[must_use]
    pub const fn final_output_format(&self) -> Option<&FinalOutputFormat> {
        self.final_output_format.as_ref()
    }

    /// Returns the request-specific output limit.
    #[must_use]
    pub const fn max_output_tokens(&self) -> Option<TokenCount> {
        self.max_output_tokens
    }

    /// Returns bounded request metadata.
    #[must_use]
    pub const fn metadata(&self) -> &ProtocolMetadata {
        &self.metadata
    }
}

/// Error returned when building or validating a model request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ModelRequestError {
    /// At least one canonical message is required.
    #[error("model request requires at least one message")]
    EmptyMessages,
    /// The canonical message count exceeds the request limit.
    #[error("model request contains too many messages")]
    TooManyMessages,
    /// A message fails its protocol wire invariant.
    #[error("model request contains an invalid canonical message")]
    InvalidMessage,
    /// System prompt is empty, oversized, or contains a null character.
    #[error("system prompt is invalid")]
    InvalidSystemPrompt,
    /// Tool name is not canonical.
    #[error("tool name is invalid")]
    InvalidToolName,
    /// Tool description is empty, oversized, or contains a null character.
    #[error("tool description is invalid")]
    InvalidToolDescription,
    /// A portable web-search domain is invalid.
    #[error("web-search domain must be a canonical lowercase hostname")]
    InvalidWebSearchDomain,
    /// A portable web-search policy exceeds the domain limit.
    #[error("web-search domain policy contains too many domains")]
    TooManyWebSearchDomains,
    /// Portable allow and block domain policies cannot be combined.
    #[error("web-search allowed and blocked domains are mutually exclusive")]
    ConflictingWebSearchDomainFilters,
    /// An approximate web-search location field is invalid.
    #[error("web-search location is invalid")]
    InvalidWebSearchLocation,
    /// Tool schema root must be a JSON object value.
    #[error("tool input schema must be a JSON object")]
    ToolSchemaMustBeObject,
    /// Tool schema must explicitly declare `type: object`.
    #[error("tool input schema must declare object type")]
    ToolSchemaMustDeclareObject,
    /// Tool schema exceeds encoded-byte or nesting limits.
    #[error("tool input schema exceeds supported bounds")]
    ToolSchemaOutOfBounds,
    /// Request contains too many tool definitions.
    #[error("model request contains too many tools")]
    TooManyTools,
    /// Request contains duplicate tool names.
    #[error("model request contains a duplicate tool name")]
    DuplicateToolName,
    /// Request selects a different model than the specification.
    #[error("model request does not match model specification")]
    ModelMismatch,
    /// Request includes an image but model accepts text only.
    #[error("model does not support image input")]
    ImageInputUnsupported,
    /// Request asks for reasoning from a model without reasoning support.
    #[error("model does not support reasoning")]
    ReasoningUnsupported,
    /// Request asks for an effort not supported by the selected model.
    #[error("model does not support the requested reasoning effort")]
    ReasoningEffortUnsupported,
    /// Request includes tools for a model without tool support.
    #[error("model does not support tools")]
    ToolsUnsupported,
    /// Request allows parallel tools for a serial-tool model.
    #[error("model does not support parallel tool calls")]
    ParallelToolsUnsupported,
    /// Request contains a hosted tool unsupported by the selected model.
    #[error("model does not support a requested hosted tool")]
    HostedToolUnsupported,
    /// Request contains a malformed or out-of-bounds final-output contract.
    #[error("model request final-output format is invalid")]
    InvalidFinalOutputFormat,
    /// Request asks for JSON-object output from a model without that capability.
    #[error("model does not support JSON-object final output")]
    FinalJsonObjectUnsupported,
    /// Request asks for JSON-Schema output from a model without that capability.
    #[error("model does not support JSON-Schema final output")]
    FinalJsonSchemaUnsupported,
    /// Request combines JSON-Schema output with tools without that capability.
    #[error("model does not support JSON-Schema final output with tools")]
    FinalJsonSchemaWithToolsUnsupported,
    /// Request output limit is zero or exceeds the model limit.
    #[error("requested output limit is unsupported")]
    OutputLimitUnsupported,
    /// Reasoning budget is zero or exceeds the selected output limit.
    #[error("requested reasoning budget is unsupported")]
    ReasoningBudgetUnsupported,
}

fn validate_messages(messages: &[CanonicalMessage]) -> Result<(), ModelRequestError> {
    if messages.is_empty() {
        return Err(ModelRequestError::EmptyMessages);
    }
    if messages.len() > MAX_REQUEST_MESSAGES {
        return Err(ModelRequestError::TooManyMessages);
    }
    if messages
        .iter()
        .any(|message| serde_json::to_value(message).is_err())
    {
        return Err(ModelRequestError::InvalidMessage);
    }
    Ok(())
}

fn request_contains_image(messages: &[CanonicalMessage]) -> bool {
    messages.iter().any(|message| {
        let content = match message {
            CanonicalMessage::User { content, .. }
            | CanonicalMessage::Assistant { content, .. }
            | CanonicalMessage::ToolResult { content, .. } => content,
        };
        content
            .iter()
            .any(|block| matches!(block, ContentBlock::Image { .. }))
    })
}

fn validate_tool_name(value: &str) -> Result<(), ModelRequestError> {
    let mut bytes = value.bytes();
    if value.len() > 128
        || !bytes.next().is_some_and(|byte| byte.is_ascii_lowercase())
        || !bytes.all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-' | b'.')
        })
    {
        return Err(ModelRequestError::InvalidToolName);
    }
    Ok(())
}

fn validate_tool_contract(
    name: &str,
    description: &str,
    input_schema: &Value,
) -> Result<(), ModelRequestError> {
    validate_tool_name(name)?;
    if description.is_empty()
        || description.len() > MAX_TOOL_DESCRIPTION_BYTES
        || description.contains('\0')
    {
        return Err(ModelRequestError::InvalidToolDescription);
    }
    let object = input_schema
        .as_object()
        .ok_or(ModelRequestError::ToolSchemaMustBeObject)?;
    if object.get("type").and_then(Value::as_str) != Some("object") {
        return Err(ModelRequestError::ToolSchemaMustDeclareObject);
    }
    validate_schema_bounds(input_schema)
}

fn validate_schema_bounds(value: &Value) -> Result<(), ModelRequestError> {
    if serde_json::to_vec(value)
        .map_err(|_| ModelRequestError::ToolSchemaOutOfBounds)?
        .len()
        > MAX_TOOL_SCHEMA_BYTES
        || json_depth(value) > MAX_TOOL_SCHEMA_DEPTH
    {
        return Err(ModelRequestError::ToolSchemaOutOfBounds);
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
