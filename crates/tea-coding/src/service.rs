use std::collections::HashMap;
use std::future::{Future as _, poll_fn};
use std::pin::Pin;
use std::str::FromStr;
use std::sync::{Arc, Mutex};
use std::task::Poll;

use tea::{AgentRuntime, RuntimeCommandOutcome, RuntimeSessionState, SessionStats};
use tea_coding_tools::WorkspaceRoot;
use tea_mcp::{McpError, McpManager, McpServerHealth, McpServerId};
use tea_model::{ModelCapabilities, ModelProvider, ModelRegistry, ModelSpec, ProviderId};
use tea_policy::WorkspaceId;
use tea_protocol::{
    AgentCommand, ApprovalDecision, ApprovalId, BranchId, CanonicalMessage, CommandEnvelope,
    CommandId, CommandText, ContentBlock, MessageId, ModelRef, ProfileId, ProtocolMetadata,
    ProtocolTimestamp, ReasoningEffort, SessionId,
};
use tea_session::{SessionCatalogEntry, SessionName};
use tea_tools::ToolName;

use crate::config::CodingSettings;
use crate::mcp::{self, McpServiceSnapshot};
use crate::resources::ResourceCatalog;
use crate::{CodingError, CodingErrorCode};

const MAX_OWNED_RUNS: usize = 64;

/// Immediate acknowledgement that a long-running command has an owned task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandAcceptance {
    command_id: CommandId,
    session_id: SessionId,
}

impl CommandAcceptance {
    /// Returns the accepted command identity.
    #[must_use]
    pub const fn command_id(self) -> CommandId {
        self.command_id
    }
    /// Returns the affected session.
    #[must_use]
    pub const fn session_id(self) -> SessionId {
        self.session_id
    }
}

/// One mode-neutral application service over the canonical runtime/session model.
#[derive(Debug)]
pub struct CodingAgentService {
    runtime: Arc<AgentRuntime>,
    workspace: WorkspaceRoot,
    resources: Arc<ResourceCatalog>,
    settings: CodingSettings,
    workspace_id: WorkspaceId,
    runs: Mutex<
        HashMap<SessionId, tokio::task::JoinHandle<Result<RuntimeCommandOutcome, CodingError>>>,
    >,
    mcp_manager: Mutex<Option<Arc<McpManager>>>,
}

#[allow(clippy::missing_errors_doc)] // All facade errors use the documented `CodingError` boundary.
impl CodingAgentService {
    pub(crate) fn new(
        runtime: AgentRuntime,
        workspace: WorkspaceRoot,
        resources: Arc<ResourceCatalog>,
        settings: CodingSettings,
        workspace_id: WorkspaceId,
        mcp_manager: Option<Arc<McpManager>>,
    ) -> Self {
        Self {
            runtime: Arc::new(runtime),
            workspace,
            resources,
            settings,
            workspace_id,
            runs: Mutex::new(HashMap::new()),
            mcp_manager: Mutex::new(mcp_manager),
        }
    }

    /// Returns the validated workspace capability.
    #[must_use]
    pub const fn workspace(&self) -> &WorkspaceRoot {
        &self.workspace
    }
    /// Returns the safe canonical workspace identity.
    #[must_use]
    pub const fn workspace_id(&self) -> &WorkspaceId {
        &self.workspace_id
    }
    /// Returns the immutable resource catalog.
    #[must_use]
    pub fn resources(&self) -> &ResourceCatalog {
        self.resources.as_ref()
    }
    /// Returns the resolved secret-free product settings snapshot.
    #[must_use]
    pub const fn settings(&self) -> &CodingSettings {
        &self.settings
    }
    /// Returns advertised provider models.
    #[must_use]
    pub fn models(&self) -> Vec<ModelRef> {
        self.runtime
            .models()
            .iter()
            .map(|model| model.model_ref().clone())
            .collect()
    }
    /// Returns capabilities for one model in the current provider generation.
    #[must_use]
    pub fn model_capabilities(&self, model_ref: &ModelRef) -> Option<ModelCapabilities> {
        self.runtime
            .models()
            .iter()
            .find(|model| model.model_ref() == model_ref)
            .map(tea_model::ModelSpec::capabilities)
    }
    /// Returns one advertised model contract from the current provider generation.
    #[must_use]
    pub fn model_spec(&self, model_ref: &ModelRef) -> Option<ModelSpec> {
        self.runtime
            .models()
            .into_iter()
            .find(|model| model.model_ref() == model_ref)
    }

    /// Validates a model against the current provider and profile tool contracts.
    pub fn validate_model(&self, model_ref: &ModelRef) -> Result<(), CodingError> {
        let model = self.model_spec(model_ref).ok_or_else(|| {
            CodingError::new(
                CodingErrorCode::InvalidInput,
                "requested model is unavailable",
            )
        })?;
        let profile_id = ProfileId::from_str("coding-agent").map_err(|_| {
            CodingError::new(CodingErrorCode::Runtime, "coding profile is unavailable")
        })?;
        let binding = self.runtime.binding(&profile_id).ok_or_else(|| {
            CodingError::new(CodingErrorCode::Runtime, "coding profile is unavailable")
        })?;
        binding
            .tools()
            .model_definitions(&model)
            .map(|_| ())
            .map_err(|error| CodingError::new(CodingErrorCode::InvalidInput, error.to_string()))
    }

    /// Returns the current immutable provider generation.
    #[must_use]
    pub fn model_registry(&self) -> Arc<ModelRegistry> {
        self.runtime.model_registry()
    }

    /// Atomically registers providers without changing any session selection.
    pub fn register_model_providers(
        &self,
        providers: impl IntoIterator<Item = Arc<dyn ModelProvider>>,
    ) -> Result<Arc<ModelRegistry>, CodingError> {
        self.runtime
            .register_model_providers(providers)
            .map_err(CodingError::from)
    }

    /// Atomically removes providers while preserving session model references.
    pub fn remove_model_providers(
        &self,
        provider_ids: impl IntoIterator<Item = ProviderId>,
    ) -> Result<Arc<ModelRegistry>, CodingError> {
        self.runtime
            .remove_model_providers(provider_ids)
            .map_err(CodingError::from)
    }

    /// Atomically replaces part of the provider generation.
    pub fn update_model_providers(
        &self,
        provider_ids: impl IntoIterator<Item = ProviderId>,
        providers: impl IntoIterator<Item = Arc<dyn ModelProvider>>,
    ) -> Result<Arc<ModelRegistry>, CodingError> {
        self.runtime
            .update_model_providers(provider_ids, providers)
            .map_err(CodingError::from)
    }

    /// Returns a safe immutable MCP lifecycle and catalog projection.
    ///
    /// The projection excludes server-provided descriptions, annotations,
    /// process details, environment values, stderr, and result bodies.
    pub fn mcp_snapshot(&self) -> Result<McpServiceSnapshot, McpError> {
        let manager = self
            .mcp_manager
            .lock()
            .map_err(|_| tea_mcp::McpError::new(tea_mcp::McpErrorCode::Unavailable))?
            .clone();
        mcp::snapshot(manager.as_deref(), now_mcp()?)
    }

    /// Returns the frozen MCP tool names currently owned by this service.
    #[must_use]
    pub fn mcp_tool_names(&self) -> Vec<ToolName> {
        self.mcp_manager
            .lock()
            .ok()
            .and_then(|manager| manager.as_ref().cloned())
            .map(|manager| {
                manager
                    .catalog()
                    .bindings()
                    .map(|binding| binding.spec().name().clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Returns frozen MCP tool names owned by the supplied server identities.
    #[must_use]
    pub fn mcp_tool_names_for_servers(&self, server_ids: &[McpServerId]) -> Vec<ToolName> {
        self.mcp_manager
            .lock()
            .ok()
            .and_then(|manager| manager.as_ref().cloned())
            .map(|manager| {
                manager
                    .catalog()
                    .bindings()
                    .filter(|binding| {
                        server_ids
                            .iter()
                            .any(|server_id| server_id == binding.server_id())
                    })
                    .map(|binding| binding.spec().name().clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Returns the profile's default active tool set for one session.
    pub async fn default_active_tool_names(
        &self,
        session_id: SessionId,
    ) -> Result<Vec<ToolName>, CodingError> {
        let state = self.snapshot(session_id).await?;
        let Some(binding) = self.runtime.binding(state.profile_id()) else {
            return Err(CodingError::new(
                CodingErrorCode::Runtime,
                "session profile is not registered",
            ));
        };
        Ok(binding.active_tool_names().to_vec())
    }

    /// Reconnects one configured MCP server only when its new discovery
    /// snapshot exactly matches the frozen runtime catalog.
    ///
    /// A descriptor or stale-catalog error requires rebuilding this service;
    /// reconnect never mutates its live tool registry or profile binding.
    pub async fn reconnect_mcp(
        &self,
        server_id: &McpServerId,
    ) -> Result<McpServerHealth, McpError> {
        let manager = self
            .mcp_manager
            .lock()
            .map_err(|_| tea_mcp::McpError::new(tea_mcp::McpErrorCode::Unavailable))?
            .clone()
            .ok_or_else(|| tea_mcp::McpError::new(tea_mcp::McpErrorCode::Unavailable))?;
        manager.reconnect(server_id, now_mcp()?).await
    }
    /// Subscribes before prompt submission to avoid missing early events.
    ///
    /// # Errors
    ///
    /// Returns an error when the bounded subscriber registry is full.
    pub fn subscribe(
        &self,
        session_id: SessionId,
    ) -> Result<tokio::sync::mpsc::Receiver<tea_protocol::EventEnvelope>, CodingError> {
        self.runtime
            .subscribe(session_id)
            .map_err(CodingError::from)
    }
    /// Creates a durable coding session.
    ///
    /// # Errors
    ///
    /// Returns a runtime or canonical command error.
    pub async fn create_session(&self) -> Result<SessionId, CodingError> {
        let outcome = self
            .runtime
            .send(envelope(
                None,
                AgentCommand::CreateSession {
                    profile_id: ProfileId::from_str("coding-agent").map_err(|_| invalid())?,
                    metadata: ProtocolMetadata::default(),
                },
            )?)
            .await
            .map_err(CodingError::from)?;
        match outcome {
            RuntimeCommandOutcome::Created { session_id } => Ok(session_id),
            _ => Err(runtime_error()),
        }
    }
    /// Attaches an existing durable session.
    pub async fn open_session(
        &self,
        session_id: SessionId,
    ) -> Result<RuntimeSessionState, CodingError> {
        self.runtime
            .attach_session(session_id)
            .await
            .map_err(CodingError::from)
    }
    /// Lists durable sessions.
    pub async fn list_sessions(&self) -> Result<Vec<SessionCatalogEntry>, CodingError> {
        self.runtime
            .list_sessions()
            .await
            .map_err(CodingError::from)
    }
    /// Sets a durable display name.
    pub async fn name_session(
        &self,
        session_id: SessionId,
        name: Option<SessionName>,
    ) -> Result<(), CodingError> {
        self.runtime
            .set_session_name(session_id, name)
            .await
            .map_err(CodingError::from)
    }
    /// Returns a compact mode-neutral state projection.
    pub async fn snapshot(
        &self,
        session_id: SessionId,
    ) -> Result<RuntimeSessionState, CodingError> {
        self.runtime
            .session_state(session_id)
            .await
            .map_err(CodingError::from)
    }
    /// Returns the complete immutable canonical session snapshot.
    ///
    /// Hosts use this after startup, reconnect, or an observed sequence gap to
    /// rebuild projections instead of treating transient deltas as durable.
    pub async fn session_snapshot(
        &self,
        session_id: SessionId,
    ) -> Result<tea_session::SessionSnapshot, CodingError> {
        self.runtime
            .snapshot(session_id)
            .await
            .map_err(CodingError::from)
    }
    /// Returns rebuildable session statistics.
    pub async fn stats(&self, session_id: SessionId) -> Result<SessionStats, CodingError> {
        self.runtime
            .session_stats(session_id)
            .await
            .map_err(CodingError::from)
    }

    /// Returns content-free metadata for the session's last compiled prompt.
    ///
    /// The snapshot is runtime-local and unavailable before the first prompt
    /// compilation or after rebuilding the service process.
    pub fn prompt_inspection(
        &self,
        session_id: SessionId,
    ) -> Result<Option<tea::RuntimePromptInspection>, CodingError> {
        self.runtime
            .prompt_inspection(session_id)
            .map_err(CodingError::from)
    }

    /// Accepts a prompt into one owned task; completion is observed separately.
    ///
    /// # Errors
    ///
    /// Rejects duplicate/overflowing owned runs and invalid prompt text.
    pub fn prompt(
        &self,
        session_id: SessionId,
        text: impl Into<String>,
    ) -> Result<CommandAcceptance, CodingError> {
        let content = vec![ContentBlock::text(text.into()).map_err(|_| invalid())?];
        self.prompt_content(session_id, content)
    }

    /// Accepts typed user content into one owned task.
    ///
    /// # Errors
    ///
    /// Rejects duplicate/overflowing owned runs and content that cannot form a
    /// canonical user message.
    pub fn prompt_content(
        &self,
        session_id: SessionId,
        content: Vec<ContentBlock>,
    ) -> Result<CommandAcceptance, CodingError> {
        let timestamp = now()?;
        let message = CanonicalMessage::user(new_id::<MessageId>()?, content, timestamp)
            .map_err(|_| invalid())?;
        let envelope = envelope_at(
            Some(session_id),
            AgentCommand::Prompt { message },
            timestamp,
        )?;
        self.start_owned(session_id, envelope)
    }

    /// Awaits and removes one owned run.
    pub async fn wait(&self, session_id: SessionId) -> Result<RuntimeCommandOutcome, CodingError> {
        poll_fn(|context| {
            let Ok(mut runs) = self.runs.lock() else {
                return Poll::Ready(Err(runtime_error()));
            };
            let Some(handle) = runs.get_mut(&session_id) else {
                return Poll::Ready(Err(CodingError::new(
                    CodingErrorCode::Runtime,
                    "session has no owned run",
                )));
            };
            match Pin::new(handle).poll(context) {
                Poll::Pending => Poll::Pending,
                Poll::Ready(joined) => {
                    runs.remove(&session_id);
                    Poll::Ready(
                        joined
                            .map_err(|_| runtime_error())
                            .and_then(|result| result),
                    )
                }
            }
        })
        .await
    }

    /// Waits for the currently owned command and returns its terminal outcome.
    ///
    /// This is equivalent to [`Self::wait`], but is named for protocol hosts
    /// that need to distinguish a completed turn from a turn paused for
    /// approval before deciding whether to continue it.
    pub async fn wait_owned(
        &self,
        session_id: SessionId,
    ) -> Result<RuntimeCommandOutcome, CodingError> {
        self.wait(session_id).await
    }

    /// Sends steering text to an active run.
    pub async fn steer(
        &self,
        session_id: SessionId,
        text: impl Into<String>,
    ) -> Result<RuntimeCommandOutcome, CodingError> {
        self.send(
            session_id,
            AgentCommand::Steer {
                text: CommandText::new(text).map_err(|_| invalid())?,
            },
        )
        .await
    }
    /// Queues a follow-up user message.
    pub async fn follow_up(
        &self,
        session_id: SessionId,
        text: impl Into<String>,
    ) -> Result<RuntimeCommandOutcome, CodingError> {
        self.follow_up_content(
            session_id,
            vec![ContentBlock::text(text.into()).map_err(|_| invalid())?],
        )
        .await
    }

    /// Queues typed user content as a follow-up message.
    ///
    /// # Errors
    ///
    /// Returns an error when the content cannot form a canonical user message
    /// or the active run's follow-up queue rejects it.
    pub async fn follow_up_content(
        &self,
        session_id: SessionId,
        content: Vec<ContentBlock>,
    ) -> Result<RuntimeCommandOutcome, CodingError> {
        let timestamp = now()?;
        let message = CanonicalMessage::user(new_id::<MessageId>()?, content, timestamp)
            .map_err(|_| invalid())?;
        self.send_at(session_id, AgentCommand::FollowUp { message }, timestamp)
            .await
    }
    /// Cancels an active run.
    pub async fn abort(&self, session_id: SessionId) -> Result<RuntimeCommandOutcome, CodingError> {
        self.send(session_id, AgentCommand::Abort {}).await
    }
    /// Accepts approval resolution into an owned continuation task.
    pub fn approve(
        &self,
        session_id: SessionId,
        approval_id: ApprovalId,
        decision: ApprovalDecision,
    ) -> Result<CommandAcceptance, CodingError> {
        let command = envelope(
            Some(session_id),
            AgentCommand::ResolveApproval {
                approval_id,
                decision,
            },
        )?;
        self.start_owned(session_id, command)
    }
    /// Selects a model for future turns.
    pub async fn set_model(
        &self,
        session_id: SessionId,
        model: ModelRef,
    ) -> Result<RuntimeCommandOutcome, CodingError> {
        self.send(session_id, AgentCommand::SetModel { model })
            .await
    }
    /// Selects provider-neutral reasoning effort for future turns.
    pub async fn set_reasoning_effort(
        &self,
        session_id: SessionId,
        reasoning_effort: ReasoningEffort,
    ) -> Result<RuntimeCommandOutcome, CodingError> {
        self.send(
            session_id,
            AgentCommand::SetReasoningEffort { reasoning_effort },
        )
        .await
    }
    /// Selects a registered product profile for future turns.
    pub async fn set_profile(
        &self,
        session_id: SessionId,
        profile_id: ProfileId,
    ) -> Result<RuntimeCommandOutcome, CodingError> {
        self.send(session_id, AgentCommand::SetProfile { profile_id })
            .await
    }
    /// Replaces the active tool set for future runs in this session.
    pub async fn set_active_tools(
        &self,
        session_id: SessionId,
        names: Vec<ToolName>,
    ) -> Result<(), CodingError> {
        self.runtime
            .set_active_tools(session_id, names)
            .await
            .map_err(CodingError::from)
    }
    /// Requests manual compaction.
    pub async fn compact(
        &self,
        session_id: SessionId,
    ) -> Result<RuntimeCommandOutcome, CodingError> {
        self.send(
            session_id,
            AgentCommand::CompactSession { instruction: None },
        )
        .await
    }
    /// Forks and activates a branch from one message.
    pub async fn fork(
        &self,
        session_id: SessionId,
        from_message_id: MessageId,
        branch_id: BranchId,
    ) -> Result<RuntimeCommandOutcome, CodingError> {
        self.send(
            session_id,
            AgentCommand::ForkSession {
                from_message_id,
                branch_id,
            },
        )
        .await
    }

    /// Cancels active runtime runs, aborts task owners, and awaits shutdown.
    pub async fn shutdown(&self) {
        let session_ids = self
            .runs
            .lock()
            .map(|runs| runs.keys().copied().collect::<Vec<_>>())
            .unwrap_or_default();
        for session_id in session_ids {
            if let Ok(command) = envelope(Some(session_id), AgentCommand::Abort {}) {
                let _ = self.runtime.send(command).await;
            }
        }
        let handles = self
            .runs
            .lock()
            .map(|mut runs| runs.drain().map(|(_, handle)| handle).collect::<Vec<_>>())
            .unwrap_or_default();
        for handle in &handles {
            handle.abort();
        }
        for handle in handles {
            let _ = handle.await;
        }
        let manager = self
            .mcp_manager
            .lock()
            .ok()
            .and_then(|mut manager| manager.take());
        if let Some(manager) = manager {
            let _ = manager.shutdown().await;
        }
    }

    fn start_owned(
        &self,
        session_id: SessionId,
        command: CommandEnvelope,
    ) -> Result<CommandAcceptance, CodingError> {
        let command_id = command.command_id();
        let runtime = Arc::clone(&self.runtime);
        let mut runs = self.runs.lock().map_err(|_| runtime_error())?;
        if runs.contains_key(&session_id) || runs.len() >= MAX_OWNED_RUNS {
            return Err(CodingError::new(
                CodingErrorCode::Runtime,
                "session already has an owned run",
            ));
        }
        let handle =
            tokio::spawn(async move { runtime.send(command).await.map_err(CodingError::from) });
        runs.insert(session_id, handle);
        Ok(CommandAcceptance {
            command_id,
            session_id,
        })
    }

    async fn send(
        &self,
        session_id: SessionId,
        command: AgentCommand,
    ) -> Result<RuntimeCommandOutcome, CodingError> {
        self.runtime
            .send(envelope(Some(session_id), command)?)
            .await
            .map_err(CodingError::from)
    }
    async fn send_at(
        &self,
        session_id: SessionId,
        command: AgentCommand,
        timestamp: ProtocolTimestamp,
    ) -> Result<RuntimeCommandOutcome, CodingError> {
        self.runtime
            .send(envelope_at(Some(session_id), command, timestamp)?)
            .await
            .map_err(CodingError::from)
    }
}

impl Drop for CodingAgentService {
    fn drop(&mut self) {
        if let Ok(runs) = self.runs.get_mut() {
            for handle in runs.values() {
                handle.abort();
            }
        }
        let Some(manager) = self.mcp_manager.get_mut().ok().and_then(Option::take) else {
            return;
        };
        if let Ok(runtime) = tokio::runtime::Handle::try_current() {
            runtime.spawn(async move {
                let _ = manager.shutdown().await;
            });
            return;
        }
        if let Ok(runtime) = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            let _ = runtime.block_on(manager.shutdown());
        }
    }
}

fn envelope(
    session_id: Option<SessionId>,
    command: AgentCommand,
) -> Result<CommandEnvelope, CodingError> {
    envelope_at(session_id, command, now()?)
}
fn envelope_at(
    session_id: Option<SessionId>,
    command: AgentCommand,
    timestamp: ProtocolTimestamp,
) -> Result<CommandEnvelope, CodingError> {
    CommandEnvelope::new(new_id::<CommandId>()?, session_id, timestamp, command)
        .map_err(|_| invalid())
}
fn new_id<T: FromStr>() -> Result<T, CodingError> {
    UuidString::new().0.parse().map_err(|_| invalid())
}
struct UuidString(String);
impl UuidString {
    fn new() -> Self {
        Self(uuid::Uuid::now_v7().hyphenated().to_string())
    }
}
fn now() -> Result<ProtocolTimestamp, CodingError> {
    chrono::Utc::now()
        .to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
        .parse()
        .map_err(|_| invalid())
}
fn now_mcp() -> Result<ProtocolTimestamp, McpError> {
    chrono::Utc::now()
        .to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
        .parse()
        .map_err(|_| tea_mcp::McpError::new(tea_mcp::McpErrorCode::Configuration))
}
fn invalid() -> CodingError {
    CodingError::new(
        CodingErrorCode::InvalidInput,
        "canonical coding command is invalid",
    )
}
fn runtime_error() -> CodingError {
    CodingError::new(CodingErrorCode::Runtime, "coding runtime operation failed")
}
