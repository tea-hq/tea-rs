use tea_protocol::{ApprovalId, BranchId, ModelId, ModelRef, ProfileId, ProviderId, SessionId};
use tea_session::{SessionCatalogEntry, SessionName};

use crate::{AgentRuntime, RuntimeError, RuntimeErrorCode};

/// Immutable mode-neutral state suitable for CLI/RPC host projections.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeSessionState {
    session_id: SessionId,
    name: Option<SessionName>,
    profile_id: ProfileId,
    model: Option<ModelRef>,
    active_branch_id: Option<BranchId>,
    message_count: usize,
    pending_approval_id: Option<ApprovalId>,
    is_running: bool,
}

impl RuntimeSessionState {
    /// Returns session identity.
    #[must_use]
    pub const fn session_id(&self) -> SessionId {
        self.session_id
    }

    /// Returns the optional display name.
    #[must_use]
    pub const fn name(&self) -> Option<&SessionName> {
        self.name.as_ref()
    }

    /// Returns the active product profile.
    #[must_use]
    pub const fn profile_id(&self) -> &ProfileId {
        &self.profile_id
    }

    /// Returns the active model when configured.
    #[must_use]
    pub const fn model_id(&self) -> Option<&ModelId> {
        match &self.model {
            Some(model) => Some(model.model_id()),
            None => None,
        }
    }

    /// Returns the selected provider when configured.
    #[must_use]
    pub const fn provider_id(&self) -> Option<&ProviderId> {
        match &self.model {
            Some(model) => Some(model.provider_id()),
            None => None,
        }
    }

    /// Returns the provider-qualified selected model when configured.
    #[must_use]
    pub const fn model_ref(&self) -> Option<&ModelRef> {
        self.model.as_ref()
    }

    /// Returns the active branch for branch-aware sessions.
    #[must_use]
    pub const fn active_branch_id(&self) -> Option<BranchId> {
        self.active_branch_id
    }

    /// Returns active transcript message count.
    #[must_use]
    pub const fn message_count(&self) -> usize {
        self.message_count
    }

    /// Returns one pending approval when present.
    #[must_use]
    pub const fn pending_approval_id(&self) -> Option<ApprovalId> {
        self.pending_approval_id
    }

    /// Returns whether this runtime currently owns an active run.
    #[must_use]
    pub const fn is_running(&self) -> bool {
        self.is_running
    }
}

/// Rebuildable aggregate statistics for one active transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionStats {
    message_count: usize,
    user_messages: usize,
    assistant_messages: usize,
    tool_result_messages: usize,
    tool_calls: usize,
}

impl SessionStats {
    /// Returns total active transcript messages.
    #[must_use]
    pub const fn message_count(self) -> usize {
        self.message_count
    }

    /// Returns user-message count.
    #[must_use]
    pub const fn user_messages(self) -> usize {
        self.user_messages
    }

    /// Returns assistant-message count.
    #[must_use]
    pub const fn assistant_messages(self) -> usize {
        self.assistant_messages
    }

    /// Returns tool-result-message count.
    #[must_use]
    pub const fn tool_result_messages(self) -> usize {
        self.tool_result_messages
    }

    /// Returns requested tool-call count.
    #[must_use]
    pub const fn tool_calls(self) -> usize {
        self.tool_calls
    }
}

impl AgentRuntime {
    /// Validates and attaches one existing durable session to this runtime.
    ///
    /// # Errors
    ///
    /// Returns an error when the session is missing or references a profile not
    /// registered by this runtime. Model availability is checked when a run starts.
    pub async fn attach_session(
        &self,
        session_id: SessionId,
    ) -> Result<RuntimeSessionState, RuntimeError> {
        let snapshot = self.load_snapshot(session_id).await?;
        let configuration = snapshot.state().configuration();
        if self.binding(configuration.profile_id()).is_none() {
            return Err(RuntimeError::new(
                RuntimeErrorCode::UnknownProfile,
                format!(
                    "session profile {} is not registered",
                    configuration.profile_id()
                ),
            ));
        }
        self.track_session_attached(session_id)?;
        self.runtime_session_state_from_snapshot(&snapshot).await
    }

    /// Lists host-facing session catalog entries.
    ///
    /// # Errors
    ///
    /// Returns an error when no catalog was configured or the catalog fails.
    pub async fn list_sessions(&self) -> Result<Vec<SessionCatalogEntry>, RuntimeError> {
        self.catalog()?
            .list_sessions()
            .await
            .map_err(RuntimeError::from)
    }

    /// Sets or clears a host-facing session display name.
    ///
    /// # Errors
    ///
    /// Returns an error when no catalog was configured or the session is missing.
    pub async fn set_session_name(
        &self,
        session_id: SessionId,
        name: Option<SessionName>,
    ) -> Result<(), RuntimeError> {
        self.catalog()?
            .set_session_name(session_id, name)
            .await
            .map_err(RuntimeError::from)
    }

    /// Returns one immutable host state projection.
    ///
    /// # Errors
    ///
    /// Returns an error when the session or catalog cannot be read.
    pub async fn session_state(
        &self,
        session_id: SessionId,
    ) -> Result<RuntimeSessionState, RuntimeError> {
        let snapshot = self.load_snapshot(session_id).await?;
        self.runtime_session_state_from_snapshot(&snapshot).await
    }

    /// Returns rebuildable active-transcript counts.
    ///
    /// # Errors
    ///
    /// Returns an error when the session cannot be loaded.
    pub async fn session_stats(&self, session_id: SessionId) -> Result<SessionStats, RuntimeError> {
        let snapshot = self.load_snapshot(session_id).await?;
        let mut user_messages = 0;
        let mut assistant_messages = 0;
        let mut tool_result_messages = 0;
        for message in snapshot.state().messages() {
            match message {
                tea_protocol::CanonicalMessage::User { .. } => user_messages += 1,
                tea_protocol::CanonicalMessage::Assistant { .. } => {
                    assistant_messages += 1;
                }
                tea_protocol::CanonicalMessage::ToolResult { .. } => {
                    tool_result_messages += 1;
                }
            }
        }
        Ok(SessionStats {
            message_count: snapshot.state().messages().len(),
            user_messages,
            assistant_messages,
            tool_result_messages,
            tool_calls: snapshot.state().tool_calls().len(),
        })
    }

    fn catalog(&self) -> Result<&dyn tea_session::SessionCatalog, RuntimeError> {
        self.session_catalog.as_deref().ok_or_else(|| {
            RuntimeError::new(
                RuntimeErrorCode::InvalidState,
                "runtime has no session catalog configured",
            )
        })
    }

    async fn runtime_session_state_from_snapshot(
        &self,
        snapshot: &tea_session::SessionSnapshot,
    ) -> Result<RuntimeSessionState, RuntimeError> {
        let state = snapshot.state();
        let name = match &self.session_catalog {
            Some(catalog) => catalog
                .session_name(state.session_id())
                .await
                .map_err(RuntimeError::from)?,
            None => None,
        };
        Ok(RuntimeSessionState {
            session_id: state.session_id(),
            name,
            profile_id: state.configuration().profile_id().clone(),
            model: state.configuration().model_ref().cloned(),
            active_branch_id: state.active_branch_id(),
            message_count: state.messages().len(),
            pending_approval_id: state.pending_approvals().keys().next().copied(),
            is_running: self.has_active_run(state.session_id()),
        })
    }
}
