use std::collections::HashMap;
use std::sync::Arc;

use tea_context::{ContextProvider, PromptBudget, PromptCompiler};
use tea_kernel::{KernelClock, KernelIdSource, RunLimits, TokioKernelClock, UuidV7KernelIdSource};
use tea_model::{ModelProvider, ModelRegistry};
use tea_policy::{ActorId, PolicyRule, UnknownEffectPolicy, WorkspaceId};
use tea_profile::{AgentProfile, ProfileRuleId};
use tea_protocol::{ProfileId, ReasoningEffort};
use tea_session::{InMemorySessionStore, SessionStore};
use tea_tools::{ToolBinding, ToolExecutor, ToolName, ToolResourceResolver, ToolSpec};

use crate::binding::{
    ProfileBinding, ToolRegistration, ToolRegistrationKind, build_context_providers,
    build_filtered_registry,
};
use crate::policy_wiring::{RegisteredPolicyRule, SharedPolicyRule, build_engine};
use crate::runtime::AgentRuntime;
use crate::{RuntimeError, RuntimeErrorCode};

/// Ergonomic builder wiring runtime ports and profiles into an [`AgentRuntime`].
#[derive(Debug)]
#[must_use]
pub struct AgentRuntimeBuilder {
    providers: Vec<Arc<dyn ModelProvider>>,
    clock: Option<Arc<dyn KernelClock>>,
    ids: Option<Arc<dyn KernelIdSource>>,
    session_id_source: Option<Arc<dyn crate::id::SessionIdSource>>,
    session_store: Option<Arc<dyn SessionStore>>,
    session_catalog: Option<Arc<dyn tea_session::SessionCatalog>>,
    event_capacity: usize,
    actor_id: Option<ActorId>,
    workspace_id: Option<WorkspaceId>,
    retry_policy: tea_kernel::ModelRetryPolicy,
    compaction_policy: Option<Arc<dyn tea_kernel::CompactionPolicy>>,
    compaction_summarizer: Option<Arc<dyn tea_kernel::CompactionSummarizer>>,
    tool_registrations: Vec<ToolRegistration>,
    policy_rules: Vec<RegisteredPolicyRule>,
    context_providers: Vec<Arc<dyn ContextProvider>>,
    builtin_tool_hints: bool,
    profiles: Vec<AgentProfile>,
    default_reasoning_effort: Option<ReasoningEffort>,
}

impl AgentRuntimeBuilder {
    /// Creates an empty builder with conservative defaults.
    pub fn new() -> Self {
        Self {
            providers: Vec::new(),
            clock: None,
            ids: None,
            session_id_source: None,
            session_store: None,
            session_catalog: None,
            event_capacity: crate::DEFAULT_EVENT_CHANNEL_CAPACITY,
            actor_id: None,
            workspace_id: None,
            retry_policy: tea_kernel::ModelRetryPolicy::default(),
            compaction_policy: None,
            compaction_summarizer: None,
            tool_registrations: Vec::new(),
            policy_rules: Vec::new(),
            context_providers: Vec::new(),
            builtin_tool_hints: true,
            profiles: Vec::new(),
            default_reasoning_effort: None,
        }
    }

    /// Registers one model provider. At least one is required before building.
    pub fn provider(mut self, provider: Arc<dyn ModelProvider>) -> Self {
        self.providers.push(provider);
        self
    }

    /// Sets the runtime clock. Defaults to `TokioKernelClock`.
    pub fn clock(mut self, clock: Arc<dyn KernelClock>) -> Self {
        self.clock = Some(clock);
        self
    }

    /// Sets the runtime ID source. Defaults to `UuidV7KernelIdSource`.
    pub fn ids(mut self, ids: Arc<dyn KernelIdSource>) -> Self {
        self.ids = Some(ids);
        self
    }

    /// Sets the session identity source. Defaults to `UuidSessionIdSource`.
    pub fn session_id_source(mut self, source: Arc<dyn crate::id::SessionIdSource>) -> Self {
        self.session_id_source = Some(source);
        self
    }

    /// Sets the session store. Defaults to an in-memory reference store.
    pub fn session_store(mut self, store: Arc<dyn SessionStore>) -> Self {
        self.session_store = Some(store);
        self
    }

    /// Sets the optional host-facing session listing/name query port.
    pub fn session_catalog(mut self, catalog: Arc<dyn tea_session::SessionCatalog>) -> Self {
        self.session_catalog = Some(catalog);
        self
    }

    /// Sets the bounded event channel capacity per subscriber.
    pub fn event_capacity(mut self, capacity: usize) -> Self {
        self.event_capacity = capacity;
        self
    }

    /// Sets the default actor identity applied to every run.
    pub fn actor(mut self, actor_id: ActorId) -> Self {
        self.actor_id = Some(actor_id);
        self
    }

    /// Sets an optional default workspace identity applied to every run.
    pub fn workspace(mut self, workspace_id: WorkspaceId) -> Self {
        self.workspace_id = Some(workspace_id);
        self
    }

    /// Sets the model retry policy applied to every run.
    pub fn retry_policy(mut self, policy: tea_kernel::ModelRetryPolicy) -> Self {
        self.retry_policy = policy;
        self
    }

    /// Sets the provider-neutral reasoning default captured by new sessions.
    pub fn default_reasoning_effort(mut self, effort: ReasoningEffort) -> Self {
        self.default_reasoning_effort = Some(effort);
        self
    }

    /// Sets the automatic compaction policy applied to every run.
    pub fn compaction_policy(mut self, policy: Arc<dyn tea_kernel::CompactionPolicy>) -> Self {
        self.compaction_policy = Some(policy);
        self
    }

    /// Attaches a product-supplied compaction summarizer.
    pub fn compaction_summarizer(
        mut self,
        summarizer: Arc<dyn tea_kernel::CompactionSummarizer>,
    ) -> Self {
        self.compaction_summarizer = Some(summarizer);
        self
    }

    /// Registers one product-owned native tool contract.
    ///
    /// Product tools take precedence over extension tools with the same name.
    /// Use [`Self::register_tool`] for extension and user-supplied tools.
    ///
    /// # Errors
    ///
    /// Returns an error when another product tool has the same name.
    pub fn tool(
        self,
        spec: ToolSpec,
        resolver: Arc<dyn ToolResourceResolver>,
        executor: Arc<dyn ToolExecutor>,
    ) -> Result<Self, RuntimeError> {
        self.tool_binding(spec, ToolBinding::client(resolver, executor))
    }

    /// Registers one complete product-owned tool binding.
    ///
    /// A binding is atomic: precedence replaces or retains the complete client,
    /// hosted, or hybrid route rather than merging routes across registrations.
    /// Product tools take precedence over extension tools with the same name.
    ///
    /// # Errors
    ///
    /// Returns an error for non-product provenance or another product tool with
    /// the same name.
    pub fn tool_binding(self, spec: ToolSpec, binding: ToolBinding) -> Result<Self, RuntimeError> {
        if !spec.source().is_native_product() {
            return Err(RuntimeError::new(
                RuntimeErrorCode::InvalidRequest,
                "product tools must declare product provenance",
            ));
        }
        self.register_tool_binding_with_kind(spec, binding, ToolRegistrationKind::Product)
    }

    /// Registers one extension or user-supplied tool contract.
    ///
    /// The specification must carry explicit non-product provenance. Product
    /// tools always take precedence when their name collides with this tool.
    ///
    /// # Errors
    ///
    /// Returns an error for product provenance or another extension with the
    /// same name.
    pub fn register_tool(
        self,
        spec: ToolSpec,
        resolver: Arc<dyn ToolResourceResolver>,
        executor: Arc<dyn ToolExecutor>,
    ) -> Result<Self, RuntimeError> {
        self.register_tool_binding(spec, ToolBinding::client(resolver, executor))
    }

    /// Registers one complete extension or user-supplied tool binding.
    ///
    /// A binding is atomic: precedence never combines routes from separate
    /// registrations. Product bindings retain precedence on name collisions.
    ///
    /// # Errors
    ///
    /// Returns an error for product provenance or another extension binding
    /// with the same name.
    pub fn register_tool_binding(
        self,
        spec: ToolSpec,
        binding: ToolBinding,
    ) -> Result<Self, RuntimeError> {
        if spec.source().is_native_product() {
            return Err(RuntimeError::new(
                RuntimeErrorCode::InvalidRequest,
                "extension tools must declare non-product provenance",
            ));
        }
        self.register_tool_binding_with_kind(spec, binding, ToolRegistrationKind::Extension)
    }

    fn register_tool_binding_with_kind(
        mut self,
        spec: ToolSpec,
        binding: ToolBinding,
        kind: ToolRegistrationKind,
    ) -> Result<Self, RuntimeError> {
        let registration = ToolRegistration {
            spec,
            binding,
            kind,
        };
        if let Some(index) = self
            .tool_registrations
            .iter()
            .position(|entry| entry.spec.name() == registration.spec.name())
        {
            match (self.tool_registrations[index].kind, registration.kind) {
                (ToolRegistrationKind::Product, ToolRegistrationKind::Extension) => {}
                (ToolRegistrationKind::Extension, ToolRegistrationKind::Product) => {
                    self.tool_registrations[index] = registration;
                }
                _ => {
                    return Err(RuntimeError::new(
                        RuntimeErrorCode::DuplicateEntry,
                        "tool name is already registered at the same precedence",
                    ));
                }
            }
        } else {
            self.tool_registrations.push(registration);
        }
        Ok(self)
    }

    /// Registers one policy rule keyed by its canonical reference id.
    ///
    /// # Errors
    ///
    /// Returns an error when a rule id is already registered.
    pub fn policy_rule(
        mut self,
        id: ProfileRuleId,
        rule: Arc<dyn PolicyRule>,
    ) -> Result<Self, RuntimeError> {
        if self.policy_rules.iter().any(|entry| entry.id == id) {
            return Err(RuntimeError::new(
                RuntimeErrorCode::DuplicateEntry,
                "policy rule id is already registered",
            ));
        }
        self.policy_rules.push(RegisteredPolicyRule { id, rule });
        Ok(self)
    }

    /// Adds an extra context provider appended after the built-in providers.
    pub fn context_provider(mut self, provider: Arc<dyn ContextProvider>) -> Self {
        self.context_providers.push(provider);
        self
    }

    /// Selects whether the runtime adds its generic active-tool hint provider.
    ///
    /// The default is `true`. Product prompt builders that already compose
    /// active-tool modules may disable it to keep one authoritative module set.
    pub fn builtin_tool_hints(mut self, enabled: bool) -> Self {
        self.builtin_tool_hints = enabled;
        self
    }

    /// Registers one product profile. Duplicate profile ids fail at build.
    pub fn profile(mut self, profile: AgentProfile) -> Self {
        self.profiles.push(profile);
        self
    }

    /// Builds the runtime and precomputes one [`ProfileBinding`] per profile.
    ///
    /// # Panics
    ///
    /// Panics only if the built-in platform rule selector is no longer canonical.
    ///
    /// # Errors
    ///
    /// Returns an error when the provider is missing, no profile is registered,
    /// a profile references an unregistered tool or policy rule, a profile
    /// advertises a model the provider does not, or duplicate profile ids exist.
    #[allow(clippy::too_many_lines)]
    pub fn build(self) -> Result<AgentRuntime, RuntimeError> {
        let models = Arc::new(ModelRegistry::new(self.providers).map_err(|error| {
            RuntimeError::new(RuntimeErrorCode::InvalidRequest, error.to_string())
        })?);
        if self.profiles.is_empty() {
            return Err(RuntimeError::new(
                RuntimeErrorCode::UnknownProfile,
                "at least one profile must be registered",
            ));
        }
        let clock = self.clock.unwrap_or_else(|| Arc::new(TokioKernelClock));
        let ids = self.ids.unwrap_or_else(|| Arc::new(UuidV7KernelIdSource));
        let session_id_source = self
            .session_id_source
            .unwrap_or_else(|| Arc::new(crate::id::UuidSessionIdSource));
        let compaction_policy = self
            .compaction_policy
            .unwrap_or_else(|| Arc::new(tea_kernel::NeverCompactPolicy));
        let session_store = self
            .session_store
            .unwrap_or_else(|| Arc::new(InMemorySessionStore::new()));
        let actor_id = self.actor_id.ok_or_else(|| {
            RuntimeError::new(
                RuntimeErrorCode::InvalidRequest,
                "a default actor identity is required",
            )
        })?;

        // Always register the platform fail-closed rule so it is resolvable by
        // any profile that references it; duplicate registration is tolerated.
        let mut policy_rules = self.policy_rules;
        let platform_id = "platform.unknown_effect"
            .parse::<ProfileRuleId>()
            .expect("platform rule id is canonical");
        if !policy_rules.iter().any(|entry| entry.id == platform_id) {
            policy_rules.push(RegisteredPolicyRule {
                id: platform_id,
                rule: Arc::new(UnknownEffectPolicy),
            });
        }

        let all_tool_names = self
            .tool_registrations
            .iter()
            .map(|entry| entry.spec.name().clone())
            .collect::<Vec<ToolName>>();
        let (all_tools, _) = build_filtered_registry(&all_tool_names, &self.tool_registrations)?;

        let mut bindings: HashMap<ProfileId, Arc<ProfileBinding>> = HashMap::new();
        for profile in self.profiles {
            let profile_id = profile.profile_id().clone();
            if bindings.contains_key(&profile_id) {
                return Err(RuntimeError::new(
                    RuntimeErrorCode::DuplicateEntry,
                    "profile id is already registered",
                ));
            }
            let model_ref = profile.model_ref().clone();
            let model = crate::runtime::resolve_model(models.as_ref(), &model_ref)?;
            if profile.active_tool_names().is_empty()
                && !profile.workspace_instructions().is_empty()
            {
                // Permitted; workspace instructions may exist without tools.
            }
            let (tools, active_tool_specs) =
                build_filtered_registry(profile.active_tool_names(), &self.tool_registrations)?;
            tools.model_definitions(model).map_err(|error| {
                RuntimeError::new(RuntimeErrorCode::InvalidRequest, error.to_string())
            })?;
            let resolved_rules: Vec<SharedPolicyRule> = profile
                .policy_rule_ids()
                .iter()
                .map(|id| {
                    policy_rules
                        .iter()
                        .find(|entry| &entry.id == id)
                        .map(|entry| SharedPolicyRule::new(Arc::clone(&entry.rule)))
                        .ok_or_else(|| {
                            RuntimeError::new(
                                RuntimeErrorCode::UnknownPolicyRule,
                                format!(
                                    "profile {profile_id} references unregistered policy rule {id}"
                                ),
                            )
                        })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let policy = Arc::new(build_engine(&resolved_rules).map_err(|error| {
                RuntimeError::new(RuntimeErrorCode::PolicyFailure, error.to_string())
            })?);
            let context_providers = build_context_providers(
                profile.workspace_instructions(),
                self.builtin_tool_hints,
                &self.context_providers,
            )?;
            let prompt_budget = PromptBudget::new(
                profile.prompt_budget().max_bytes(),
                profile.prompt_budget().max_estimated_tokens(),
            )?;
            let run_limits = RunLimits::new(
                profile.run_limits().max_tool_iterations(),
                profile.run_limits().max_elapsed(),
                profile.run_limits().max_assistant_output_bytes(),
                profile.run_limits().max_events(),
                profile.run_limits().max_queued_messages(),
            )
            .map_err(|error| {
                RuntimeError::new(RuntimeErrorCode::InvalidRequest, error.message().to_owned())
            })?;
            let environment = profile.environment().clone();
            let approval_ttl = profile.approval_ttl();
            let binding = ProfileBinding::new(
                profile_id.clone(),
                model_ref,
                Arc::clone(&all_tools),
                tools,
                policy,
                context_providers,
                profile.active_tool_names().to_vec(),
                active_tool_specs,
                prompt_budget,
                run_limits,
                environment,
                approval_ttl,
                actor_id.clone(),
                self.workspace_id.clone(),
            );
            bindings.insert(profile_id, Arc::new(binding));
        }

        let event_sink = Arc::new(crate::RuntimeEventSink::with_capacity(self.event_capacity));
        Ok(AgentRuntime::new(
            models,
            clock,
            ids,
            session_id_source,
            session_store,
            self.session_catalog,
            event_sink,
            Arc::new(PromptCompiler),
            bindings,
            self.tool_registrations,
            policy_rules,
            actor_id,
            self.workspace_id,
            self.retry_policy,
            compaction_policy,
            self.compaction_summarizer,
            self.default_reasoning_effort,
        ))
    }
}

impl Default for AgentRuntimeBuilder {
    fn default() -> Self {
        Self::new()
    }
}
