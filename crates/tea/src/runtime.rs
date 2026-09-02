use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};

use tea_context::PromptCompiler;
use tea_kernel::{KernelClock, KernelIdSource};
use tea_model::{ModelProvider, ModelRegistry, ModelRegistryError, ModelRouter, ModelSpec};
use tea_policy::{ActorId, WorkspaceId};
use tea_protocol::{ModelRef, ProfileId, ReasoningEffort, SessionId};
use tea_session::SessionStore;
use tea_tools::{ToolName, ToolRegistry, ToolSpec};

use crate::RuntimePromptInspection;
use crate::binding::{ToolRegistration, build_filtered_registry};
use crate::id::SessionIdSource;
use crate::policy_wiring::RegisteredPolicyRule;
use crate::{ProfileBinding, RuntimeError, RuntimeErrorCode, RuntimeEventSink, RuntimeHealth};

/// Runtime-owned cancellation scopes for active runs, keyed by session.
type ActiveRuns = Mutex<HashMap<tea_protocol::SessionId, tea_control::CancellationScope>>;

/// Sessions created through this runtime.
type TrackedSessions = Mutex<HashSet<tea_protocol::SessionId>>;

/// Per-session bounded steering and follow-up queues.
type SessionQueues = Mutex<HashMap<tea_protocol::SessionId, Arc<tea_kernel::KernelInputQueue>>>;

/// Per-session runtime-only active-tool replacements.
type ActiveToolOverrides = Mutex<HashMap<SessionId, ActiveToolOverride>>;

/// Content-free last-successful prompt metadata keyed by live session.
type PromptInspections = Mutex<HashMap<SessionId, RuntimePromptInspection>>;

#[derive(Debug)]
struct ActiveToolOverride {
    profile_id: ProfileId,
    names: Vec<ToolName>,
}

pub(crate) fn resolve_model<'a>(
    models: &'a dyn ModelRouter,
    model_ref: &ModelRef,
) -> Result<&'a ModelSpec, RuntimeError> {
    if models.provider(model_ref.provider_id()).is_none() {
        return Err(RuntimeError::new(
            RuntimeErrorCode::UnknownProvider,
            format!(
                "model provider {} is not registered",
                model_ref.provider_id()
            ),
        ));
    }
    models.model(model_ref).ok_or_else(|| {
        RuntimeError::new(
            RuntimeErrorCode::UnknownModel,
            format!("model {model_ref} is not advertised by its provider"),
        )
    })
}

fn registry_error(error: &ModelRegistryError) -> RuntimeError {
    let code = match error {
        ModelRegistryError::DuplicateProvider(_) => RuntimeErrorCode::DuplicateEntry,
        ModelRegistryError::UnknownProvider(_) => RuntimeErrorCode::UnknownProvider,
        ModelRegistryError::ProviderCatalogMismatch(_) => RuntimeErrorCode::InvalidRequest,
    };
    RuntimeError::new(code, error.to_string())
}

/// Ergonomic embedding facade owning replaceable ports and profile bindings.
///
/// Construct through [`crate::AgentRuntimeBuilder`]. The runtime constructs a
/// fresh borrowed [`tea_kernel::AgentKernel`] for each run.
#[allow(dead_code)]
#[derive(Debug)]
pub struct AgentRuntime {
    models: RwLock<Arc<ModelRegistry>>,
    pub(crate) clock: Arc<dyn KernelClock>,
    pub(crate) ids: Arc<dyn KernelIdSource>,
    pub(crate) session_id_source: Arc<dyn SessionIdSource>,
    pub(crate) sessions: Arc<dyn SessionStore>,
    pub(crate) session_catalog: Option<Arc<dyn tea_session::SessionCatalog>>,
    pub(crate) event_sink: Arc<RuntimeEventSink>,
    pub(crate) compiler: Arc<PromptCompiler>,
    pub(crate) bindings: HashMap<ProfileId, Arc<ProfileBinding>>,
    pub(crate) tool_registrations: Vec<ToolRegistration>,
    pub(crate) policy_rules: Vec<RegisteredPolicyRule>,
    pub(crate) actor_id: ActorId,
    pub(crate) workspace_id: Option<WorkspaceId>,
    pub(crate) retry_policy: tea_kernel::ModelRetryPolicy,
    pub(crate) compaction_policy: Arc<dyn tea_kernel::CompactionPolicy>,
    pub(crate) compaction_summarizer: Option<Arc<dyn tea_kernel::CompactionSummarizer>>,
    pub(crate) default_reasoning_effort: Option<ReasoningEffort>,
    pub(crate) active_runs: ActiveRuns,
    active_tool_overrides: ActiveToolOverrides,
    pub(crate) sessions_created: TrackedSessions,
    pub(crate) queues: SessionQueues,
    prompt_inspections: PromptInspections,
}

impl AgentRuntime {
    /// Creates a runtime from prebuilt owned wiring.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        models: Arc<ModelRegistry>,
        clock: Arc<dyn KernelClock>,
        ids: Arc<dyn KernelIdSource>,
        session_id_source: Arc<dyn SessionIdSource>,
        sessions: Arc<dyn SessionStore>,
        session_catalog: Option<Arc<dyn tea_session::SessionCatalog>>,
        event_sink: Arc<RuntimeEventSink>,
        compiler: Arc<PromptCompiler>,
        bindings: HashMap<ProfileId, Arc<ProfileBinding>>,
        tool_registrations: Vec<ToolRegistration>,
        policy_rules: Vec<RegisteredPolicyRule>,
        actor_id: ActorId,
        workspace_id: Option<WorkspaceId>,
        retry_policy: tea_kernel::ModelRetryPolicy,
        compaction_policy: Arc<dyn tea_kernel::CompactionPolicy>,
        compaction_summarizer: Option<Arc<dyn tea_kernel::CompactionSummarizer>>,
        default_reasoning_effort: Option<ReasoningEffort>,
    ) -> Self {
        Self {
            models: RwLock::new(models),
            clock,
            ids,
            session_id_source,
            sessions,
            session_catalog,
            event_sink,
            compiler,
            bindings,
            tool_registrations,
            policy_rules,
            actor_id,
            workspace_id,
            retry_policy,
            compaction_policy,
            compaction_summarizer,
            default_reasoning_effort,
            active_runs: Mutex::new(HashMap::new()),
            active_tool_overrides: Mutex::new(HashMap::new()),
            sessions_created: Mutex::new(HashSet::new()),
            queues: Mutex::new(HashMap::new()),
            prompt_inspections: Mutex::new(HashMap::new()),
        }
    }

    /// Returns provider-advertised models from the current immutable generation.
    ///
    #[must_use]
    pub fn models(&self) -> Vec<tea_model::ModelSpec> {
        self.model_registry().models().to_vec()
    }

    pub(crate) fn resolve_model(&self, model_ref: &ModelRef) -> Result<ModelSpec, RuntimeError> {
        let models = self.model_registry();
        resolve_model(models.as_ref(), model_ref).cloned()
    }

    /// Returns the current immutable model-provider generation.
    ///
    /// Active runs retain their own returned generation while future runs read
    /// whatever generation is current when they start.
    ///
    #[must_use]
    pub fn model_registry(&self) -> Arc<ModelRegistry> {
        let models = self
            .models
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        Arc::clone(&models)
    }

    /// Atomically publishes a generation containing additional providers.
    ///
    /// Registration does not change any session's selected model.
    ///
    /// # Errors
    ///
    /// Returns a duplicate-entry error when a provider identity is already
    /// registered.
    pub fn register_model_providers(
        &self,
        providers: impl IntoIterator<Item = Arc<dyn ModelProvider>>,
    ) -> Result<Arc<ModelRegistry>, RuntimeError> {
        self.update_model_providers([], providers)
    }

    /// Atomically publishes a generation without the requested providers.
    ///
    /// Existing runs keep their starting generation. Sessions keep their
    /// selected [`ModelRef`] and future prompts fail until it is available again.
    ///
    /// # Errors
    ///
    /// Returns an unknown-provider error when any requested identity is absent,
    /// or the requested generation is otherwise invalid.
    pub fn remove_model_providers(
        &self,
        provider_ids: impl IntoIterator<Item = tea_model::ProviderId>,
    ) -> Result<Arc<ModelRegistry>, RuntimeError> {
        self.update_model_providers(provider_ids, [])
    }

    /// Atomically removes and registers providers as one generation update.
    ///
    /// This supports replacing one host-owned provider set without exposing a
    /// transient partial catalog. Provider identities removed by the same call
    /// may be registered again with a new immutable adapter.
    ///
    /// # Errors
    ///
    /// Returns an error without publishing a partial generation when removal or
    /// registration validation fails.
    pub fn update_model_providers(
        &self,
        provider_ids: impl IntoIterator<Item = tea_model::ProviderId>,
        providers: impl IntoIterator<Item = Arc<dyn ModelProvider>>,
    ) -> Result<Arc<ModelRegistry>, RuntimeError> {
        let provider_ids = provider_ids.into_iter().collect::<Vec<_>>();
        let providers = providers.into_iter().collect::<Vec<_>>();
        let mut current = self
            .models
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let next = current
            .without_providers(provider_ids)
            .and_then(|registry| registry.with_registered(providers))
            .map_err(|error| registry_error(&error))?;
        let next = Arc::new(next);
        *current = Arc::clone(&next);
        Ok(next)
    }

    /// Returns the bound profile configuration, if registered.
    #[must_use]
    pub fn binding(&self, profile_id: &ProfileId) -> Option<&Arc<ProfileBinding>> {
        self.bindings.get(profile_id)
    }

    pub(crate) fn active_tool_snapshot(
        &self,
        session_id: SessionId,
        binding: &ProfileBinding,
    ) -> Result<(Arc<ToolRegistry>, Vec<ToolSpec>), RuntimeError> {
        let names = {
            let mut overrides = self.active_tool_overrides.lock().map_err(|_| {
                RuntimeError::new(
                    RuntimeErrorCode::InvalidState,
                    "runtime active-tool selector is poisoned",
                )
            })?;
            match overrides.get(&session_id) {
                Some(active) if active.profile_id == *binding.profile_id() => active.names.clone(),
                Some(_) => {
                    overrides.remove(&session_id);
                    binding.active_tool_names().to_vec()
                }
                None => binding.active_tool_names().to_vec(),
            }
        };
        build_filtered_registry(&names, &self.tool_registrations)
    }

    pub(crate) fn replace_active_tool_override(
        &self,
        session_id: SessionId,
        profile_id: ProfileId,
        names: Vec<ToolName>,
    ) -> Result<(), RuntimeError> {
        let runs = self.active_runs.lock().map_err(|_| {
            RuntimeError::new(
                RuntimeErrorCode::InvalidState,
                "runtime active-run tracker is poisoned",
            )
        })?;
        if runs.contains_key(&session_id) {
            return Err(RuntimeError::new(
                RuntimeErrorCode::RunAlreadyActive,
                "active tools cannot change while a run is active",
            ));
        }
        self.active_tool_overrides
            .lock()
            .map_err(|_| {
                RuntimeError::new(
                    RuntimeErrorCode::InvalidState,
                    "runtime active-tool selector is poisoned",
                )
            })?
            .insert(session_id, ActiveToolOverride { profile_id, names });
        Ok(())
    }

    pub(crate) fn clear_active_tool_override(
        &self,
        session_id: SessionId,
    ) -> Result<(), RuntimeError> {
        self.active_tool_overrides
            .lock()
            .map_err(|_| {
                RuntimeError::new(
                    RuntimeErrorCode::InvalidState,
                    "runtime active-tool selector is poisoned",
                )
            })?
            .remove(&session_id);
        Ok(())
    }

    /// Returns the event sink used to subscribe to runtime events.
    #[must_use]
    pub fn event_sink(&self) -> &RuntimeEventSink {
        &self.event_sink
    }

    /// Subscribes to canonical events for one session.
    ///
    /// # Errors
    ///
    /// Returns an error when too many subscribers are already registered.
    pub fn subscribe(
        &self,
        session_id: tea_protocol::SessionId,
    ) -> Result<tokio::sync::mpsc::Receiver<tea_protocol::EventEnvelope>, RuntimeError> {
        self.event_sink.subscribe(session_id).map_err(|error| {
            RuntimeError::new(RuntimeErrorCode::InvalidRequest, error.message().to_owned())
        })
    }

    /// Returns the count of sessions created through this runtime.
    #[must_use]
    pub fn session_count(&self) -> usize {
        self.sessions_created
            .lock()
            .map_or(0, |sessions| sessions.len())
    }

    #[allow(dead_code)]
    pub(crate) fn track_session_created(
        &self,
        session_id: tea_protocol::SessionId,
    ) -> Result<(), RuntimeError> {
        let mut sessions = self.lock_sessions_created()?;
        if !sessions.insert(session_id) {
            return Err(RuntimeError::new(
                RuntimeErrorCode::InvalidRequest,
                "session was already created through this runtime",
            ));
        }
        Ok(())
    }

    pub(crate) fn track_session_attached(
        &self,
        session_id: tea_protocol::SessionId,
    ) -> Result<(), RuntimeError> {
        self.lock_sessions_created()?.insert(session_id);
        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) fn lock_sessions_created(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, HashSet<tea_protocol::SessionId>>, RuntimeError> {
        self.sessions_created.lock().map_err(|_| {
            RuntimeError::new(
                RuntimeErrorCode::InvalidState,
                "runtime session tracker is poisoned",
            )
        })
    }

    /// Returns a health summary of the runtime configuration.
    #[must_use]
    pub fn health(&self) -> RuntimeHealth {
        let models = self.model_registry();
        let provider_ids = models.provider_ids();
        let model_refs = models
            .models()
            .iter()
            .map(|model| model.model_ref().clone())
            .collect::<Vec<_>>();
        let mut profile_ids = self.bindings.keys().cloned().collect::<Vec<_>>();
        profile_ids.sort();
        let mut policy_rule_ids = self
            .policy_rules
            .iter()
            .map(|rule| rule.id.clone())
            .collect::<Vec<_>>();
        policy_rule_ids.sort();
        let tool_count = self.tool_registrations.len();
        let session_count = self.session_count();
        RuntimeHealth::new(
            provider_ids,
            model_refs,
            profile_ids,
            policy_rule_ids,
            tool_count,
            session_count,
        )
    }

    /// Returns the runtime actor identity.
    #[must_use]
    pub fn actor_id(&self) -> &ActorId {
        &self.actor_id
    }

    /// Returns the runtime ID source.
    #[must_use]
    pub fn ids(&self) -> &Arc<dyn KernelIdSource> {
        &self.ids
    }

    /// Returns (or lazily creates) the bounded input queue for one session.
    pub(crate) fn session_queue(
        &self,
        session_id: SessionId,
    ) -> Result<Arc<tea_kernel::KernelInputQueue>, RuntimeError> {
        let mut queues = self.queues.lock().map_err(|_| {
            RuntimeError::new(
                RuntimeErrorCode::InvalidState,
                "runtime queue map is poisoned",
            )
        })?;
        if let Some(queue) = queues.get(&session_id) {
            return Ok(Arc::clone(queue));
        }
        let queue = Arc::new(tea_kernel::KernelInputQueue::new(64, 64 * 1024).map_err(
            |error| RuntimeError::new(RuntimeErrorCode::InvalidRequest, error.message().to_owned()),
        )?);
        queues.insert(session_id, Arc::clone(&queue));
        Ok(queue)
    }

    /// Returns the runtime clock.
    #[must_use]
    pub fn clock(&self) -> &Arc<dyn KernelClock> {
        &self.clock
    }

    /// Returns the runtime session store.
    #[must_use]
    pub fn sessions(&self) -> &Arc<dyn SessionStore> {
        &self.sessions
    }

    /// Returns one runtime model provider by canonical identity.
    #[must_use]
    pub fn provider(&self, provider_id: &tea_model::ProviderId) -> Option<Arc<dyn ModelProvider>> {
        self.model_registry().provider_arc(provider_id)
    }

    /// Returns the prompt compiler.
    #[must_use]
    pub fn compiler(&self) -> &Arc<PromptCompiler> {
        &self.compiler
    }

    /// Returns the last successfully compiled prompt metadata for one session.
    ///
    /// The result contains no prompt text and exists only in this runtime
    /// process. `None` is returned before the first successful compilation.
    ///
    /// # Errors
    ///
    /// Returns an invalid-state error if the runtime inspection lock is poisoned.
    pub fn prompt_inspection(
        &self,
        session_id: SessionId,
    ) -> Result<Option<RuntimePromptInspection>, RuntimeError> {
        self.prompt_inspections
            .lock()
            .map_err(|_| {
                RuntimeError::new(
                    RuntimeErrorCode::InvalidState,
                    "runtime prompt inspection state is poisoned",
                )
            })
            .map(|inspections| inspections.get(&session_id).cloned())
    }

    pub(crate) fn record_prompt_inspection(
        &self,
        session_id: SessionId,
        run_id: Option<tea_protocol::RunId>,
        prompt: &tea_context::CompiledPrompt,
    ) -> Result<(), RuntimeError> {
        self.prompt_inspections
            .lock()
            .map_err(|_| {
                RuntimeError::new(
                    RuntimeErrorCode::InvalidState,
                    "runtime prompt inspection state is poisoned",
                )
            })?
            .insert(
                session_id,
                RuntimePromptInspection::new(session_id, run_id, prompt.inspection_snapshot()),
            );
        Ok(())
    }
}
