use std::time::Duration;

use tea_policy::{ActorId, PolicyEnvironment, WorkspaceId};
use tea_protocol::{FinalOutputFormat, ProtocolMetadata};
use tea_tools::CompiledToolSchema;

use crate::compaction::{CompactionPolicy, CompactionSummarizer, NeverCompactPolicy};
use crate::retry::ModelRetryPolicy;
use crate::{KernelError, KernelErrorCode};

#[derive(Debug, Clone)]
pub(crate) enum CompiledFinalOutputFormat {
    JsonObject,
    JsonSchema(CompiledToolSchema),
}

impl CompiledFinalOutputFormat {
    pub(crate) fn as_protocol(&self) -> FinalOutputFormat {
        match self {
            Self::JsonObject => FinalOutputFormat::JsonObject,
            Self::JsonSchema(schema) => FinalOutputFormat::JsonSchema {
                schema: schema.source().clone(),
            },
        }
    }
}

/// Hard deterministic limits applied to one kernel run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_field_names)] // `max_*` expresses the stable limit vocabulary.
pub struct RunLimits {
    max_tool_iterations: u32,
    max_elapsed: Duration,
    max_assistant_output_bytes: usize,
    max_events: u64,
    max_queued_messages: usize,
}

impl RunLimits {
    /// Creates validated non-zero run limits.
    ///
    /// # Errors
    ///
    /// Returns an error for zero or unsupported values.
    pub fn new(
        max_tool_iterations: u32,
        max_elapsed: Duration,
        max_assistant_output_bytes: usize,
        max_events: u64,
        max_queued_messages: usize,
    ) -> Result<Self, KernelError> {
        if max_tool_iterations == 0
            || max_elapsed.is_zero()
            || max_elapsed > Duration::from_hours(24)
            || max_assistant_output_bytes == 0
            || max_assistant_output_bytes > 16 * 1024 * 1024
            || max_events == 0
            || max_events > 1_000_000
            || max_queued_messages == 0
            || max_queued_messages > 1024
        {
            return Err(KernelError::new(
                KernelErrorCode::InvalidRequest,
                "run limits are invalid",
            ));
        }
        Ok(Self {
            max_tool_iterations,
            max_elapsed,
            max_assistant_output_bytes,
            max_events,
            max_queued_messages,
        })
    }

    /// Returns the maximum model responses containing tools.
    #[must_use]
    pub const fn max_tool_iterations(self) -> u32 {
        self.max_tool_iterations
    }
    /// Returns maximum elapsed run time.
    #[must_use]
    pub const fn max_elapsed(self) -> Duration {
        self.max_elapsed
    }
    /// Returns maximum accumulated assistant output bytes.
    #[must_use]
    pub const fn max_assistant_output_bytes(self) -> usize {
        self.max_assistant_output_bytes
    }
    /// Returns maximum emitted observations.
    #[must_use]
    pub const fn max_events(self) -> u64 {
        self.max_events
    }
    /// Returns maximum accepted queued messages.
    #[must_use]
    pub const fn max_queued_messages(self) -> usize {
        self.max_queued_messages
    }
}

impl Default for RunLimits {
    fn default() -> Self {
        Self {
            max_tool_iterations: 16,
            max_elapsed: Duration::from_mins(5),
            max_assistant_output_bytes: 4 * 1024 * 1024,
            max_events: 100_000,
            max_queued_messages: 64,
        }
    }
}

/// Immutable product-independent context frozen for one run.
#[derive(Debug, Clone)]
pub struct KernelRunConfig {
    actor_id: ActorId,
    workspace_id: Option<WorkspaceId>,
    environment: PolicyEnvironment,
    system_prompt: Option<String>,
    compiled_prompt: Option<tea_context::CompiledPrompt>,
    final_output_format: Option<CompiledFinalOutputFormat>,
    request_metadata: ProtocolMetadata,
    approval_ttl: Duration,
    retry_policy: ModelRetryPolicy,
    compaction_policy: std::sync::Arc<dyn CompactionPolicy>,
    compaction_summarizer: Option<std::sync::Arc<dyn CompactionSummarizer>>,
    limits: RunLimits,
}

impl KernelRunConfig {
    /// Creates a run snapshot with default hard limits.
    #[must_use]
    pub fn new(actor_id: ActorId, environment: PolicyEnvironment) -> Self {
        Self {
            actor_id,
            workspace_id: None,
            environment,
            system_prompt: None,
            compiled_prompt: None,
            final_output_format: None,
            request_metadata: ProtocolMetadata::default(),
            approval_ttl: Duration::from_mins(10),
            retry_policy: ModelRetryPolicy::default(),
            compaction_policy: std::sync::Arc::new(NeverCompactPolicy),
            compaction_summarizer: None,
            limits: RunLimits {
                max_tool_iterations: 16,
                max_elapsed: Duration::from_mins(5),
                max_assistant_output_bytes: 4 * 1024 * 1024,
                max_events: 100_000,
                max_queued_messages: 64,
            },
        }
    }

    /// Adds a workspace identity.
    #[must_use]
    pub fn with_workspace(mut self, workspace_id: WorkspaceId) -> Self {
        self.workspace_id = Some(workspace_id);
        self
    }

    /// Adds a bounded non-empty system prompt.
    ///
    /// # Errors
    ///
    /// Returns an error when the prompt is empty, oversized, or contains null.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Result<Self, KernelError> {
        let prompt = prompt.into();
        if self.compiled_prompt.is_some()
            || prompt.is_empty()
            || prompt.len() > 1024 * 1024
            || prompt.contains('\0')
        {
            return Err(KernelError::new(
                KernelErrorCode::InvalidRequest,
                "system prompt is invalid",
            ));
        }
        self.system_prompt = Some(prompt);
        Ok(self)
    }

    /// Attaches one immutable provenance-preserving compiled prompt snapshot.
    ///
    /// # Errors
    ///
    /// Returns an error when a legacy system prompt is already configured.
    pub fn with_compiled_prompt(
        mut self,
        prompt: tea_context::CompiledPrompt,
    ) -> Result<Self, KernelError> {
        if self.system_prompt.is_some() {
            return Err(KernelError::new(
                KernelErrorCode::InvalidRequest,
                "compiled and legacy system prompts are mutually exclusive",
            ));
        }
        self.compiled_prompt = Some(prompt);
        Ok(self)
    }

    /// Freezes an optional structured final-output contract for this run.
    ///
    /// # Errors
    ///
    /// Returns an error when a JSON Schema is not an object or cannot be
    /// compiled as a bounded, self-contained Draft 2020-12 schema.
    pub fn with_final_output_format(
        mut self,
        format: Option<FinalOutputFormat>,
    ) -> Result<Self, KernelError> {
        self.final_output_format = match format {
            None => None,
            Some(FinalOutputFormat::JsonObject) => Some(CompiledFinalOutputFormat::JsonObject),
            Some(FinalOutputFormat::JsonSchema { schema }) => {
                if !schema.is_object() {
                    return Err(KernelError::new(
                        KernelErrorCode::InvalidRequest,
                        "final output schema is invalid",
                    ));
                }
                let schema = CompiledToolSchema::compile(schema).map_err(|_| {
                    KernelError::new(
                        KernelErrorCode::InvalidRequest,
                        "final output schema is invalid",
                    )
                })?;
                Some(CompiledFinalOutputFormat::JsonSchema(schema))
            }
        };
        Ok(self)
    }

    /// Adds bounded model request metadata.
    #[must_use]
    pub fn with_request_metadata(mut self, metadata: ProtocolMetadata) -> Self {
        self.request_metadata = metadata;
        self
    }

    /// Replaces the approval lifetime used for newly persisted requests.
    ///
    /// # Errors
    ///
    /// Returns an error for zero or values above 24 hours.
    pub fn with_approval_ttl(mut self, ttl: Duration) -> Result<Self, KernelError> {
        if ttl.is_zero() || ttl > Duration::from_hours(24) {
            return Err(KernelError::new(
                KernelErrorCode::InvalidRequest,
                "approval lifetime is invalid",
            ));
        }
        self.approval_ttl = ttl;
        Ok(self)
    }

    /// Replaces default run limits.
    #[must_use]
    pub const fn with_limits(mut self, limits: RunLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Replaces the default model retry policy.
    #[must_use]
    pub const fn with_retry_policy(mut self, policy: ModelRetryPolicy) -> Self {
        self.retry_policy = policy;
        self
    }

    /// Replaces the default automatic compaction policy.
    #[must_use]
    pub fn with_compaction_policy(mut self, policy: std::sync::Arc<dyn CompactionPolicy>) -> Self {
        self.compaction_policy = policy;
        self
    }

    /// Attaches a product-supplied compaction summarizer.
    #[must_use]
    pub fn with_compaction_summarizer(
        mut self,
        summarizer: std::sync::Arc<dyn CompactionSummarizer>,
    ) -> Self {
        self.compaction_summarizer = Some(summarizer);
        self
    }

    /// Returns the actor snapshot.
    #[must_use]
    pub const fn actor_id(&self) -> &ActorId {
        &self.actor_id
    }
    /// Returns optional workspace identity.
    #[must_use]
    pub const fn workspace_id(&self) -> Option<&WorkspaceId> {
        self.workspace_id.as_ref()
    }
    /// Returns the policy environment snapshot.
    #[must_use]
    pub const fn environment(&self) -> &PolicyEnvironment {
        &self.environment
    }
    /// Returns optional product-supplied system prompt.
    #[must_use]
    pub fn system_prompt(&self) -> Option<&str> {
        self.compiled_prompt.as_ref().map_or_else(
            || self.system_prompt.as_deref(),
            |prompt| (!prompt.text().is_empty()).then_some(prompt.text()),
        )
    }
    /// Returns the immutable compiled prompt and inspection snapshot.
    #[must_use]
    pub const fn compiled_prompt(&self) -> Option<&tea_context::CompiledPrompt> {
        self.compiled_prompt.as_ref()
    }
    /// Returns the provider-neutral final-output contract frozen for this run.
    #[must_use]
    pub fn final_output_format(&self) -> Option<FinalOutputFormat> {
        self.final_output_format
            .as_ref()
            .map(CompiledFinalOutputFormat::as_protocol)
    }
    pub(crate) const fn compiled_final_output_format(&self) -> Option<&CompiledFinalOutputFormat> {
        self.final_output_format.as_ref()
    }
    /// Returns model request metadata.
    #[must_use]
    pub const fn request_metadata(&self) -> &ProtocolMetadata {
        &self.request_metadata
    }
    /// Returns the lifetime assigned to new approval requests.
    #[must_use]
    pub const fn approval_ttl(&self) -> Duration {
        self.approval_ttl
    }
    /// Returns hard run limits.
    #[must_use]
    pub const fn limits(&self) -> RunLimits {
        self.limits
    }
    /// Returns the model retry policy.
    #[must_use]
    pub const fn retry_policy(&self) -> ModelRetryPolicy {
        self.retry_policy
    }
    /// Returns the automatic compaction policy.
    #[must_use]
    pub fn compaction_policy(&self) -> &std::sync::Arc<dyn CompactionPolicy> {
        &self.compaction_policy
    }
    /// Returns the product-supplied compaction summarizer when configured.
    #[must_use]
    pub fn compaction_summarizer(&self) -> Option<&std::sync::Arc<dyn CompactionSummarizer>> {
        self.compaction_summarizer.as_ref()
    }
}
