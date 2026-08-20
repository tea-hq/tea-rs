use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use tea_context::{
    ContextProvider, PromptBudget, ToolHintProvider, TrustLevel, WorkspaceInstruction,
    WorkspaceInstructionProvider,
};
use tea_kernel::RunLimits;
use tea_policy::{ActorId, PolicyEngine, PolicyEnvironment, WorkspaceId};
use tea_profile::ProfileWorkspaceInstruction;
use tea_protocol::{ModelId, ModelRef, ProfileId, ProviderId};
use tea_tools::{ToolBinding, ToolName, ToolRegistry, ToolSpec};

use crate::{RuntimeError, RuntimeErrorCode};

/// One registered master tool contract.
#[derive(Debug, Clone)]
pub(crate) struct ToolRegistration {
    pub(crate) spec: ToolSpec,
    pub(crate) binding: ToolBinding,
    pub(crate) kind: ToolRegistrationKind,
}

/// Registration precedence assigned by the runtime builder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ToolRegistrationKind {
    /// A product-owned native tool.
    Product,
    /// An extension, MCP, or other externally supplied tool.
    Extension,
}

/// Immutable per-profile precomputed artifacts reused across runs.
///
/// Built once at `AgentRuntimeBuilder::build`, a binding holds the filtered
/// tool registry, composed policy engine, ordered context providers, converted
/// run limits and prompt budget, and the actor/workspace/environment snapshot.
#[derive(Debug)]
pub struct ProfileBinding {
    profile_id: ProfileId,
    model: ModelRef,
    all_tools: Arc<ToolRegistry>,
    tools: Arc<ToolRegistry>,
    policy: Arc<PolicyEngine>,
    context_providers: Vec<Arc<dyn ContextProvider>>,
    active_tool_names: Vec<ToolName>,
    active_tool_specs: Vec<ToolSpec>,
    prompt_budget: PromptBudget,
    run_limits: RunLimits,
    environment: PolicyEnvironment,
    approval_ttl: Duration,
    actor_id: ActorId,
    workspace_id: Option<WorkspaceId>,
}

impl ProfileBinding {
    /// Creates a binding from precomputed owned artifacts.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        profile_id: ProfileId,
        model: ModelRef,
        all_tools: Arc<ToolRegistry>,
        tools: Arc<ToolRegistry>,
        policy: Arc<PolicyEngine>,
        context_providers: Vec<Arc<dyn ContextProvider>>,
        active_tool_names: Vec<ToolName>,
        active_tool_specs: Vec<ToolSpec>,
        prompt_budget: PromptBudget,
        run_limits: RunLimits,
        environment: PolicyEnvironment,
        approval_ttl: Duration,
        actor_id: ActorId,
        workspace_id: Option<WorkspaceId>,
    ) -> Self {
        Self {
            profile_id,
            model,
            all_tools,
            tools,
            policy,
            context_providers,
            active_tool_names,
            active_tool_specs,
            prompt_budget,
            run_limits,
            environment,
            approval_ttl,
            actor_id,
            workspace_id,
        }
    }

    /// Returns the bound profile selector.
    #[must_use]
    pub fn profile_id(&self) -> &ProfileId {
        &self.profile_id
    }
    /// Returns the bound model selector.
    #[must_use]
    pub fn model_id(&self) -> &ModelId {
        self.model.model_id()
    }
    /// Returns the bound provider selector.
    #[must_use]
    pub const fn provider_id(&self) -> &ProviderId {
        self.model.provider_id()
    }
    /// Returns the complete bound model selector.
    #[must_use]
    pub const fn model_ref(&self) -> &ModelRef {
        &self.model
    }
    /// Returns every tool registered with the runtime.
    #[must_use]
    pub fn all_tools(&self) -> &ToolRegistry {
        &self.all_tools
    }
    /// Returns the filtered active tool registry.
    #[must_use]
    pub fn tools(&self) -> &ToolRegistry {
        &self.tools
    }
    /// Returns the composed policy engine.
    #[must_use]
    pub fn policy(&self) -> &PolicyEngine {
        &self.policy
    }
    /// Returns the ordered context providers.
    #[must_use]
    pub fn context_providers(&self) -> &[Arc<dyn ContextProvider>] {
        &self.context_providers
    }
    /// Returns the profile's default active tool names in canonical order.
    #[must_use]
    pub fn active_tool_names(&self) -> &[ToolName] {
        &self.active_tool_names
    }
    /// Returns the active tool specifications in canonical order.
    #[must_use]
    pub fn active_tool_specs(&self) -> &[ToolSpec] {
        &self.active_tool_specs
    }
    /// Returns the converted prompt budget.
    #[must_use]
    pub const fn prompt_budget(&self) -> PromptBudget {
        self.prompt_budget
    }
    /// Returns the converted run limits.
    #[must_use]
    pub const fn run_limits(&self) -> RunLimits {
        self.run_limits
    }
    /// Returns the policy environment snapshot.
    #[must_use]
    pub fn environment(&self) -> &PolicyEnvironment {
        &self.environment
    }
    /// Returns the approval lifetime.
    #[must_use]
    pub fn approval_ttl(&self) -> Duration {
        self.approval_ttl
    }
    /// Returns the actor identity.
    #[must_use]
    pub fn actor_id(&self) -> &ActorId {
        &self.actor_id
    }
    /// Returns the optional workspace identity.
    #[must_use]
    pub fn workspace_id(&self) -> Option<&WorkspaceId> {
        self.workspace_id.as_ref()
    }
}

/// Builds a filtered `ToolRegistry` containing only the profile's active tools.
pub(crate) fn build_filtered_registry(
    active_tool_names: &[ToolName],
    registrations: &[ToolRegistration],
) -> Result<(Arc<ToolRegistry>, Vec<ToolSpec>), RuntimeError> {
    let mut registry = ToolRegistry::new();
    let mut specs = Vec::with_capacity(active_tool_names.len());
    for name in active_tool_names {
        let registration = registrations
            .iter()
            .find(|entry| entry.spec.name() == name)
            .ok_or_else(|| {
                RuntimeError::new(
                    RuntimeErrorCode::UnknownTool,
                    format!("profile references unregistered tool {name}"),
                )
            })?;
        registry
            .register_binding(registration.spec.clone(), registration.binding.clone())
            .map_err(|error| {
                RuntimeError::new(RuntimeErrorCode::InvalidRequest, error.to_string())
            })?;
        specs.push(registration.spec.clone());
    }
    Ok((Arc::new(registry), specs))
}

/// Builds the ordered context provider list for one profile binding.
///
/// Order is: workspace instructions (when present), optional tool hints, then
/// builder-supplied providers.
pub(crate) fn build_context_providers(
    workspace_instructions: &[ProfileWorkspaceInstruction],
    builtin_tool_hints: bool,
    extra: &[Arc<dyn ContextProvider>],
) -> Result<Vec<Arc<dyn ContextProvider>>, RuntimeError> {
    let mut providers: Vec<Arc<dyn ContextProvider>> = Vec::new();
    if !workspace_instructions.is_empty() {
        let instructions = workspace_instructions
            .iter()
            .map(convert_workspace_instruction)
            .collect::<Result<Vec<_>, _>>()?;
        let provider = WorkspaceInstructionProvider::new(instructions).map_err(|error| {
            RuntimeError::new(RuntimeErrorCode::InvalidRequest, error.message().to_owned())
        })?;
        providers.push(Arc::new(provider));
    }
    if builtin_tool_hints {
        let tool_hints = ToolHintProvider::new().map_err(|error| {
            RuntimeError::new(
                RuntimeErrorCode::InvalidRequest,
                format!("tool hint provider failed: {error}"),
            )
        })?;
        providers.push(Arc::new(tool_hints));
    }
    for provider in extra {
        providers.push(Arc::clone(provider));
    }
    Ok(providers)
}

fn convert_workspace_instruction(
    instruction: &ProfileWorkspaceInstruction,
) -> Result<WorkspaceInstruction, RuntimeError> {
    let segment_id = tea_context::PromptSegmentId::from_str(instruction.segment_id().as_str())
        .map_err(|_| {
            RuntimeError::new(
                RuntimeErrorCode::InvalidRequest,
                "workspace instruction segment id is not canonical",
            )
        })?;
    let trust = match instruction.trust() {
        tea_profile::ProfileTrustLevel::Trusted => TrustLevel::Trusted,
        tea_profile::ProfileTrustLevel::Delegated => TrustLevel::Delegated,
        tea_profile::ProfileTrustLevel::Untrusted => TrustLevel::Untrusted,
    };
    WorkspaceInstruction::new(
        segment_id,
        instruction.content(),
        instruction.locator(),
        trust,
    )
    .map_err(|error| {
        RuntimeError::new(RuntimeErrorCode::InvalidRequest, error.message().to_owned())
    })
}
