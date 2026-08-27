use std::sync::Arc;
use tea_control::CancellationScope;
use tea_kernel::{AgentKernel, KernelRunConfig};
use tea_model::ModelRouter;
use tea_policy::{ApprovalResolution, GrantScope, PolicyGrant, ResourcePattern};
use tea_protocol::{
    AgentCommand, AgentCommandType, ApprovalDecision, BranchId, CanonicalMessage, CommandEnvelope,
    MessageId, ModelRef, ProfileId, ProtocolMetadata, ReasoningEffort, RecordEnvelope, RecordId,
    SessionId, SessionRecord, SessionSequence,
};
use tea_session::ApprovalArtifactEntry;

use crate::prompt::compile_prompt;
use crate::runtime::resolve_model;
use crate::{AgentRuntime, RuntimeError, RuntimeErrorCode};

struct ActiveRunGuard<'a> {
    runtime: &'a AgentRuntime,
    session_id: SessionId,
    cancellation: CancellationScope,
}

impl ActiveRunGuard<'_> {
    fn cancellation(&self) -> CancellationScope {
        self.cancellation.clone()
    }
}

impl Drop for ActiveRunGuard<'_> {
    fn drop(&mut self) {
        self.cancellation.cancel();
        self.runtime.clear_active_run(self.session_id);
    }
}

/// Terminal result of one dispatched command.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)] // Boxing would break the public facade value API.
pub enum RuntimeCommandOutcome {
    /// A session was created.
    Created {
        /// New session identity.
        session_id: SessionId,
    },
    /// A run reached a terminal state (Task 8).
    RunCompleted {
        /// Final run state.
        state: tea_kernel::RunState,
        /// Committed materialized session state.
        session: tea_session::MaterializedSessionState,
        /// Pending approval id when the run paused for approval.
        pending_approval_id: Option<tea_protocol::ApprovalId>,
    },
    /// A queued operation was accepted (Task 9).
    Enqueued {
        /// Accepted follow-up count.
        follow_ups: usize,
        /// Accepted steering count.
        steering: usize,
    },
    /// An active run was cancelled (Task 8).
    Aborted {
        /// Session whose run was cancelled.
        session_id: SessionId,
    },
    /// A model or profile configuration was changed.
    ConfigurationChanged {
        /// Affected session.
        session_id: SessionId,
        /// New provider-qualified model selector when changed.
        model: Option<ModelRef>,
        /// New profile selector when changed.
        profile_id: Option<ProfileId>,
        /// Effective reasoning effort when changed or adjusted.
        reasoning_effort: Option<ReasoningEffort>,
        /// Original requested effort when it was adjusted for the model.
        requested_reasoning_effort: Option<ReasoningEffort>,
    },
    /// A session compaction was committed.
    SessionCompacted {
        /// Affected session.
        session_id: SessionId,
        /// Durable compaction record.
        record_id: RecordId,
    },
    /// A branch was created and activated.
    SessionForked {
        /// Affected session.
        session_id: SessionId,
        /// Source branch.
        source_branch_id: BranchId,
        /// Newly active branch.
        branch_id: BranchId,
    },
    /// The command is not supported by this runtime version.
    Unsupported {
        /// Command discriminator.
        command_type: AgentCommandType,
    },
}

impl AgentRuntime {
    /// Dispatches one canonical command and returns its terminal outcome.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown profiles/models, missing sessions, active
    /// run conflicts, store failures, or unsupported commands.
    pub async fn send(
        &self,
        envelope: CommandEnvelope,
    ) -> Result<RuntimeCommandOutcome, RuntimeError> {
        match envelope.command() {
            AgentCommand::CreateSession {
                profile_id,
                metadata,
            } => {
                self.create_session(profile_id.clone(), metadata.clone())
                    .await
            }
            AgentCommand::SetModel { model } => {
                self.set_model(envelope.session_id(), model.clone()).await
            }
            AgentCommand::SetReasoningEffort { reasoning_effort } => {
                self.set_reasoning_effort(envelope.session_id(), *reasoning_effort)
                    .await
            }
            AgentCommand::SetProfile { profile_id } => {
                self.set_profile(envelope.session_id(), profile_id.clone())
                    .await
            }
            AgentCommand::Prompt { message } => {
                self.handle_prompt(envelope.session_id(), message.clone())
                    .await
            }
            AgentCommand::Steer { text } => {
                self.handle_steer(envelope.session_id(), text.clone()).await
            }
            AgentCommand::FollowUp { message } => {
                self.handle_follow_up(envelope.session_id(), message.clone())
                    .await
            }
            AgentCommand::Abort {} => self.handle_abort(envelope.session_id()).await,
            AgentCommand::ResolveApproval {
                approval_id,
                decision,
            } => {
                self.handle_resolve_approval(envelope.session_id(), *approval_id, *decision)
                    .await
            }
            AgentCommand::CompactSession { instruction } => {
                self.handle_compact(envelope.session_id(), instruction.as_ref())
                    .await
            }
            AgentCommand::ForkSession {
                from_message_id,
                branch_id,
            } => {
                self.handle_fork(envelope.session_id(), *from_message_id, *branch_id)
                    .await
            }
        }
    }

    /// Creates a session using a registered profile, appending the durable
    /// `SessionCreated` and `ConfigurationChanged` records.
    ///
    /// # Errors
    ///
    /// Returns an error for an unknown profile or a store failure.
    pub async fn create_session(
        &self,
        profile_id: ProfileId,
        metadata: ProtocolMetadata,
    ) -> Result<RuntimeCommandOutcome, RuntimeError> {
        let binding = self.binding(&profile_id).ok_or_else(|| {
            RuntimeError::new(
                RuntimeErrorCode::UnknownProfile,
                format!("profile {profile_id} is not registered"),
            )
        })?;
        let session_id = self.session_id_source.next_session_id()?;
        let timestamp = self.clock.now().map_err(|error| {
            RuntimeError::new(RuntimeErrorCode::InvalidState, error.message().to_owned())
        })?;
        let root_branch_id = session_id.to_string().parse::<BranchId>().map_err(|_| {
            RuntimeError::new(
                RuntimeErrorCode::InvalidState,
                "session id cannot be converted to a root branch id",
            )
        })?;
        let models = self.model_registry();
        let reasoning_effort = models.model(binding.model_ref()).and_then(|model| {
            self.default_reasoning_effort
                .and_then(|requested| model.resolve_reasoning(Some(requested)))
                .map(tea_model::ReasoningResolution::effective)
        });
        let records = self.build_records(
            session_id,
            timestamp,
            None,
            Some(root_branch_id),
            [
                SessionRecord::SessionCreated {
                    profile_id: profile_id.clone(),
                    metadata,
                },
                SessionRecord::ConfigurationChanged {
                    model: Some(binding.model_ref().clone()),
                    profile_id: None,
                    reasoning_effort,
                },
            ],
        )?;
        self.append_records(session_id, None, records).await?;
        self.track_session_created(session_id)?;
        Ok(RuntimeCommandOutcome::Created { session_id })
    }

    async fn set_model(
        &self,
        session_id: Option<SessionId>,
        model_ref: ModelRef,
    ) -> Result<RuntimeCommandOutcome, RuntimeError> {
        let session_id = Self::require_session(session_id)?;
        self.ensure_configuration_mutable(session_id)?;
        let model = self.resolve_model(&model_ref)?;
        let snapshot = self.load_snapshot(session_id).await?;
        let profile_id = snapshot.state().configuration().profile_id().clone();
        let binding = self.binding(&profile_id).ok_or_else(|| {
            RuntimeError::new(
                RuntimeErrorCode::UnknownProfile,
                format!("session profile {profile_id} is not registered"),
            )
        })?;
        let (active_tools, _) = self.active_tool_snapshot(session_id, binding)?;
        active_tools.model_definitions(&model).map_err(|error| {
            RuntimeError::new(RuntimeErrorCode::InvalidRequest, error.to_string())
        })?;
        let requested_reasoning_effort = snapshot.state().configuration().reasoning_effort();
        let adjusted_reasoning = requested_reasoning_effort
            .and_then(|requested| model.resolve_reasoning(Some(requested)))
            .filter(|resolution| resolution.was_clamped());
        let reasoning_effort = adjusted_reasoning.map(tea_model::ReasoningResolution::effective);
        self.append_configuration_change(
            session_id,
            Some(model_ref.clone()),
            None,
            reasoning_effort,
        )
        .await?;
        Ok(RuntimeCommandOutcome::ConfigurationChanged {
            session_id,
            model: Some(model_ref),
            profile_id: None,
            reasoning_effort,
            requested_reasoning_effort: adjusted_reasoning
                .map(tea_model::ReasoningResolution::requested),
        })
    }

    async fn set_reasoning_effort(
        &self,
        session_id: Option<SessionId>,
        requested: ReasoningEffort,
    ) -> Result<RuntimeCommandOutcome, RuntimeError> {
        let session_id = Self::require_session(session_id)?;
        self.ensure_configuration_mutable(session_id)?;
        let snapshot = self.load_snapshot(session_id).await?;
        let model_ref = snapshot
            .state()
            .configuration()
            .model_ref()
            .ok_or_else(|| {
                RuntimeError::new(RuntimeErrorCode::UnknownModel, "session has no model")
            })?;
        let model = self.resolve_model(model_ref)?;
        let resolution = model
            .resolve_reasoning(Some(requested))
            .expect("an explicit reasoning request always resolves");
        let effective = resolution.effective();
        self.append_configuration_change(session_id, None, None, Some(effective))
            .await?;
        Ok(RuntimeCommandOutcome::ConfigurationChanged {
            session_id,
            model: None,
            profile_id: None,
            reasoning_effort: Some(effective),
            requested_reasoning_effort: resolution.was_clamped().then_some(requested),
        })
    }

    async fn set_profile(
        &self,
        session_id: Option<SessionId>,
        profile_id: ProfileId,
    ) -> Result<RuntimeCommandOutcome, RuntimeError> {
        let session_id = Self::require_session(session_id)?;
        self.ensure_configuration_mutable(session_id)?;
        let binding = self.binding(&profile_id).ok_or_else(|| {
            RuntimeError::new(
                RuntimeErrorCode::UnknownProfile,
                format!("profile {profile_id} is not registered"),
            )
        })?;
        let snapshot = self.load_snapshot(session_id).await?;
        let model = self.resolve_model(binding.model_ref())?;
        let requested_reasoning_effort = snapshot.state().configuration().reasoning_effort();
        let adjusted_reasoning = requested_reasoning_effort
            .and_then(|requested| model.resolve_reasoning(Some(requested)))
            .filter(|resolution| resolution.was_clamped());
        let reasoning_effort = adjusted_reasoning.map(tea_model::ReasoningResolution::effective);
        self.append_configuration_change(
            session_id,
            Some(binding.model_ref().clone()),
            Some(profile_id.clone()),
            reasoning_effort,
        )
        .await?;
        self.clear_active_tool_override(session_id)?;
        Ok(RuntimeCommandOutcome::ConfigurationChanged {
            session_id,
            model: Some(binding.model_ref().clone()),
            profile_id: Some(profile_id),
            reasoning_effort,
            requested_reasoning_effort: adjusted_reasoning
                .map(tea_model::ReasoningResolution::requested),
        })
    }

    fn ensure_configuration_mutable(&self, session_id: SessionId) -> Result<(), RuntimeError> {
        let active_runs = self.active_runs.lock().map_err(|_| {
            RuntimeError::new(
                RuntimeErrorCode::InvalidState,
                "runtime active-run tracker is poisoned",
            )
        })?;
        if active_runs.contains_key(&session_id) {
            Err(RuntimeError::new(
                RuntimeErrorCode::RunAlreadyActive,
                "session configuration cannot change while a run is active",
            ))
        } else {
            Ok(())
        }
    }

    fn require_session(session_id: Option<SessionId>) -> Result<SessionId, RuntimeError> {
        session_id.ok_or_else(|| {
            RuntimeError::new(
                RuntimeErrorCode::InvalidRequest,
                "command requires an existing session id",
            )
        })
    }

    async fn append_configuration_change(
        &self,
        session_id: SessionId,
        model: Option<ModelRef>,
        profile_id: Option<ProfileId>,
        reasoning_effort: Option<ReasoningEffort>,
    ) -> Result<(), RuntimeError> {
        let snapshot = self.load_snapshot(session_id).await?;
        let timestamp = self.clock_now()?;
        let records = self.build_records(
            session_id,
            timestamp,
            Some(snapshot.state().tail_sequence()),
            snapshot.state().active_branch_id(),
            [SessionRecord::ConfigurationChanged {
                model,
                profile_id,
                reasoning_effort,
            }],
        )?;
        self.append_records(session_id, Some(snapshot.state().tail_sequence()), records)
            .await?;
        Ok(())
    }

    pub(crate) fn build_records(
        &self,
        session_id: SessionId,
        timestamp: tea_protocol::ProtocolTimestamp,
        previous: Option<SessionSequence>,
        branch_id: Option<BranchId>,
        records: impl IntoIterator<Item = SessionRecord>,
    ) -> Result<Vec<RecordEnvelope>, RuntimeError> {
        let mut sequence = previous;
        let mut envelopes = Vec::new();
        for record in records {
            let next = match sequence {
                None => SessionSequence::new(0),
                Some(prev) => prev.checked_next().ok_or_else(|| {
                    RuntimeError::new(
                        RuntimeErrorCode::SessionFailure,
                        "durable session sequence cannot advance",
                    )
                })?,
            };
            sequence = Some(next);
            let record_id = self.ids.next_record_id().map_err(|error| {
                RuntimeError::new(RuntimeErrorCode::InvalidState, error.message().to_owned())
            })?;
            envelopes.push(
                RecordEnvelope::new(
                    record_id,
                    session_id,
                    next,
                    timestamp,
                    None,
                    None,
                    branch_id,
                    ProtocolMetadata::default(),
                    record,
                )
                .map_err(|error| {
                    RuntimeError::new(RuntimeErrorCode::SessionFailure, error.to_string())
                })?,
            );
        }
        Ok(envelopes)
    }

    pub(crate) async fn append_records(
        &self,
        session_id: SessionId,
        expected: Option<SessionSequence>,
        records: Vec<RecordEnvelope>,
    ) -> Result<tea_session::SessionSnapshot, RuntimeError> {
        let transaction = tea_session::AppendTransaction::new(session_id, expected, records);
        self.sessions
            .append(transaction)
            .await
            .map_err(RuntimeError::from)?;
        self.load_snapshot(session_id).await
    }

    pub(crate) async fn load_snapshot(
        &self,
        session_id: SessionId,
    ) -> Result<tea_session::SessionSnapshot, RuntimeError> {
        self.sessions
            .load(session_id)
            .await
            .map_err(RuntimeError::from)
    }

    pub(crate) fn clock_now(&self) -> Result<tea_protocol::ProtocolTimestamp, RuntimeError> {
        self.clock.now().map_err(|error| {
            RuntimeError::new(RuntimeErrorCode::InvalidState, error.message().to_owned())
        })
    }

    async fn handle_compact(
        &self,
        session_id: Option<SessionId>,
        instruction: Option<&tea_protocol::CommandText>,
    ) -> Result<RuntimeCommandOutcome, RuntimeError> {
        let session_id = Self::require_session(session_id)?;
        if self.has_active_run(session_id) {
            return Err(RuntimeError::new(
                RuntimeErrorCode::RunAlreadyActive,
                "cannot compact while a run is active",
            ));
        }
        if instruction.is_some() {
            return Err(RuntimeError::new(
                RuntimeErrorCode::InvalidRequest,
                "custom compaction instructions are not supported by the configured summarizer contract",
            ));
        }
        let summarizer = self.compaction_summarizer.as_ref().ok_or_else(|| {
            RuntimeError::new(
                RuntimeErrorCode::InvalidState,
                "manual compaction requires a configured summarizer",
            )
        })?;
        let snapshot = self.load_snapshot(session_id).await?;
        if snapshot.state().messages().is_empty() {
            return Err(RuntimeError::new(
                RuntimeErrorCode::InvalidRequest,
                "cannot compact an empty transcript",
            ));
        }
        let compacted_through_record_id = snapshot.state().tail_record_id();
        let summary = summarizer
            .summarize(snapshot.state().messages().to_vec())
            .await
            .map_err(RuntimeError::from)?;
        let snapshot = self
            .compact(session_id, summary, compacted_through_record_id)
            .await?;
        Ok(RuntimeCommandOutcome::SessionCompacted {
            session_id,
            record_id: snapshot.state().tail_record_id(),
        })
    }

    async fn handle_fork(
        &self,
        session_id: Option<SessionId>,
        from_message_id: MessageId,
        branch_id: BranchId,
    ) -> Result<RuntimeCommandOutcome, RuntimeError> {
        let session_id = Self::require_session(session_id)?;
        if self.has_active_run(session_id) {
            return Err(RuntimeError::new(
                RuntimeErrorCode::RunAlreadyActive,
                "cannot fork while a run is active",
            ));
        }
        let snapshot = self.load_snapshot(session_id).await?;
        let source_branch_id = snapshot.state().active_branch_id().ok_or_else(|| {
            RuntimeError::new(
                RuntimeErrorCode::InvalidState,
                "legacy unbranched sessions cannot be forked",
            )
        })?;
        let from_record_id = snapshot
            .records()
            .iter()
            .find_map(|record| match record.record() {
                SessionRecord::MessageCommitted { message }
                    if canonical_message_id(message) == from_message_id =>
                {
                    Some(record.record_id())
                }
                _ => None,
            })
            .ok_or_else(|| {
                RuntimeError::new(
                    RuntimeErrorCode::InvalidRequest,
                    "fork source message does not exist in the session",
                )
            })?;
        let timestamp = self.clock_now()?;
        let records = self.build_records(
            session_id,
            timestamp,
            Some(snapshot.state().tail_sequence()),
            Some(branch_id),
            [
                SessionRecord::BranchCreated {
                    source_branch_id,
                    branch_id,
                    from_record_id,
                },
                SessionRecord::ActiveBranchChanged { branch_id },
            ],
        )?;
        self.append_records(session_id, Some(snapshot.state().tail_sequence()), records)
            .await?;
        Ok(RuntimeCommandOutcome::SessionForked {
            session_id,
            source_branch_id,
            branch_id,
        })
    }

    /// Returns an immutable snapshot of one stored session.
    ///
    /// # Errors
    ///
    /// Returns an error when the session store fails to load.
    pub async fn snapshot(
        &self,
        session_id: SessionId,
    ) -> Result<tea_session::SessionSnapshot, RuntimeError> {
        self.load_snapshot(session_id).await
    }

    /// Appends a manual compaction summary, replacing the compacted transcript
    /// prefix with the summary message.
    ///
    /// # Errors
    ///
    /// Returns an error when the session cannot be loaded, the summary is not
    /// an assistant message, the referenced record is not on the active
    /// branch, or the append transaction fails.
    pub async fn compact(
        &self,
        session_id: SessionId,
        summary: tea_protocol::CanonicalMessage,
        compacted_through_record_id: tea_protocol::RecordId,
    ) -> Result<tea_session::SessionSnapshot, RuntimeError> {
        let snapshot = self.load_snapshot(session_id).await?;
        let binding = self
            .binding(snapshot.state().configuration().profile_id())
            .ok_or_else(|| {
                RuntimeError::new(
                    RuntimeErrorCode::UnknownProfile,
                    "session profile is not registered",
                )
            })?;
        let (tools, _) = self.active_tool_snapshot(session_id, binding)?;
        let models = self.model_registry();
        let kernel = tea_kernel::AgentKernel::new(
            models.as_ref(),
            tools.as_ref(),
            binding.policy(),
            self.sessions.as_ref(),
            self.clock.as_ref(),
            self.ids.as_ref(),
            self.event_sink.as_ref(),
        );
        kernel
            .compact(session_id, summary, compacted_through_record_id)
            .await
            .map_err(RuntimeError::from)
    }

    /// Replaces the active tool set for future runs in one session.
    ///
    /// The replacement is runtime-local and is cleared when the session changes
    /// profiles. A run observes one immutable registry for its entire lifetime.
    ///
    /// # Errors
    ///
    /// Returns an error when the session or a tool is unknown, the requested set
    /// is invalid, a run is active, or runtime state cannot be read safely.
    pub async fn set_active_tools(
        &self,
        session_id: SessionId,
        mut names: Vec<tea_tools::ToolName>,
    ) -> Result<(), RuntimeError> {
        let snapshot = self.load_snapshot(session_id).await?;
        let profile_id = snapshot.state().configuration().profile_id().clone();
        self.binding(&profile_id).ok_or_else(|| {
            RuntimeError::new(
                RuntimeErrorCode::UnknownProfile,
                format!("session profile {profile_id} is not registered"),
            )
        })?;
        names.sort();
        let (tools, _) = crate::binding::build_filtered_registry(&names, &self.tool_registrations)?;
        let model_ref = snapshot
            .state()
            .configuration()
            .model_ref()
            .ok_or_else(|| {
                RuntimeError::new(
                    RuntimeErrorCode::UnknownModel,
                    "session has no active model",
                )
            })?;
        let models = self.model_registry();
        if let Some(model) = models.model(model_ref) {
            tools.model_definitions(model).map_err(|error| {
                RuntimeError::new(RuntimeErrorCode::InvalidRequest, error.to_string())
            })?;
        }
        self.replace_active_tool_override(session_id, profile_id, names)
    }

    async fn handle_prompt(
        &self,
        session_id: Option<SessionId>,
        message: CanonicalMessage,
    ) -> Result<RuntimeCommandOutcome, RuntimeError> {
        let session_id = Self::require_session(session_id)?;
        let active_run = self.begin_active_run(session_id)?;
        let snapshot = self.load_snapshot(session_id).await?;
        let models = self.model_registry();
        let model_ref = snapshot
            .state()
            .configuration()
            .model_ref()
            .ok_or_else(|| {
                RuntimeError::new(
                    RuntimeErrorCode::UnknownModel,
                    "session has no active model",
                )
            })?;
        let model = resolve_model(models.as_ref(), model_ref)?;
        let profile_id = snapshot.state().configuration().profile_id().clone();
        let binding = self.binding(&profile_id).ok_or_else(|| {
            RuntimeError::new(
                RuntimeErrorCode::UnknownProfile,
                format!("session profile {profile_id} is not registered"),
            )
        })?;
        let (active_tools, active_tool_specs) = self.active_tool_snapshot(session_id, binding)?;
        active_tools.model_definitions(model).map_err(|error| {
            RuntimeError::new(RuntimeErrorCode::InvalidRequest, error.to_string())
        })?;
        let run_id = self.ids.next_run_id().map_err(|error| {
            RuntimeError::new(RuntimeErrorCode::InvalidState, error.message().to_owned())
        })?;
        let prompt = compile_prompt(
            &self.compiler,
            binding.context_providers(),
            profile_id.clone(),
            session_id,
            Some(run_id),
            &active_tool_specs,
            ProtocolMetadata::default(),
            binding.prompt_budget(),
        )
        .await?;
        self.record_prompt_inspection(session_id, Some(run_id), &prompt)?;
        let config = self.build_run_config(binding, Some(prompt))?;
        self.append_user_message(session_id, snapshot.state().tail_sequence(), message)
            .await?;
        let outcome = self
            .run_kernel(
                session_id,
                models,
                binding,
                active_tools.as_ref(),
                &config,
                active_run.cancellation(),
            )
            .await?;
        Ok(RuntimeCommandOutcome::RunCompleted {
            state: outcome.state(),
            session: outcome.session().clone(),
            pending_approval_id: outcome.pending_approval_id(),
        })
    }

    #[allow(clippy::unused_async)]
    async fn handle_abort(
        &self,
        session_id: Option<SessionId>,
    ) -> Result<RuntimeCommandOutcome, RuntimeError> {
        let session_id = Self::require_session(session_id)?;
        let cancellation = self
            .active_runs
            .lock()
            .map_err(|_| {
                RuntimeError::new(
                    RuntimeErrorCode::InvalidState,
                    "runtime active-run tracker is poisoned",
                )
            })?
            .get(&session_id)
            .cloned()
            .ok_or_else(|| {
                RuntimeError::new(
                    RuntimeErrorCode::NoActiveRun,
                    "no active run to abort on this session",
                )
            })?;
        cancellation.cancel();
        Ok(RuntimeCommandOutcome::Aborted { session_id })
    }

    async fn handle_resolve_approval(
        &self,
        session_id: Option<SessionId>,
        approval_id: tea_protocol::ApprovalId,
        decision: ApprovalDecision,
    ) -> Result<RuntimeCommandOutcome, RuntimeError> {
        let session_id = Self::require_session(session_id)?;
        let active_run = self.begin_active_run(session_id)?;
        let snapshot = self.load_snapshot(session_id).await?;
        let models = self.model_registry();
        let model_ref = snapshot
            .state()
            .configuration()
            .model_ref()
            .ok_or_else(|| {
                RuntimeError::new(
                    RuntimeErrorCode::UnknownModel,
                    "session has no active model",
                )
            })?;
        resolve_model(models.as_ref(), model_ref)?;
        let profile_id = snapshot.state().configuration().profile_id().clone();
        let binding = self.binding(&profile_id).ok_or_else(|| {
            RuntimeError::new(
                RuntimeErrorCode::UnknownProfile,
                format!("session profile {profile_id} is not registered"),
            )
        })?;
        let (active_tools, active_tool_specs) = self.active_tool_snapshot(session_id, binding)?;
        let approval_artifacts = snapshot.approval_artifacts();
        let already_resolved = approval_artifacts.iter().any(|entry| match entry {
            ApprovalArtifactEntry::Resolved { resolution, .. } => {
                resolution.request().approval_id() == &approval_id
            }
            ApprovalArtifactEntry::Requested { .. } => false,
        });
        let request = (!already_resolved)
            .then(|| {
                approval_artifacts.iter().find_map(|entry| match entry {
                    ApprovalArtifactEntry::Requested { request, .. }
                        if request.approval_id() == &approval_id =>
                    {
                        Some(request.clone())
                    }
                    ApprovalArtifactEntry::Requested { .. }
                    | ApprovalArtifactEntry::Resolved { .. } => None,
                })
            })
            .flatten()
            .ok_or_else(|| {
                RuntimeError::new(
                    RuntimeErrorCode::PolicyFailure,
                    "persisted approval request is missing or already resolved",
                )
            })?;
        let decided_at = self.clock_now()?;
        let issued_grant = if decision == ApprovalDecision::AllowSession {
            let grant_id = self.ids.next_grant_id().map_err(|error| {
                RuntimeError::new(RuntimeErrorCode::InvalidState, error.message().to_owned())
            })?;
            Some(session_grant(grant_id, &request, session_id, decided_at)?)
        } else {
            None
        };
        let resolution = ApprovalResolution::new(&request, decision, decided_at, issued_grant)
            .map_err(|error| {
                RuntimeError::new(RuntimeErrorCode::PolicyFailure, error.to_string())
            })?;
        let run_id = resolution.request().run_id().copied();
        let prompt = compile_prompt(
            &self.compiler,
            binding.context_providers(),
            profile_id.clone(),
            session_id,
            run_id,
            &active_tool_specs,
            ProtocolMetadata::default(),
            binding.prompt_budget(),
        )
        .await?;
        self.record_prompt_inspection(session_id, run_id, &prompt)?;
        let config = self.build_run_config(binding, Some(prompt))?;
        let outcome = self
            .resume_kernel(
                session_id,
                models,
                binding,
                active_tools.as_ref(),
                &resolution,
                &config,
                active_run.cancellation(),
            )
            .await?;
        Ok(RuntimeCommandOutcome::RunCompleted {
            state: outcome.state(),
            session: outcome.session().clone(),
            pending_approval_id: outcome.pending_approval_id(),
        })
    }

    fn build_run_config(
        &self,
        binding: &crate::ProfileBinding,
        prompt: Option<tea_context::CompiledPrompt>,
    ) -> Result<KernelRunConfig, RuntimeError> {
        let mut config =
            KernelRunConfig::new(binding.actor_id().clone(), binding.environment().clone());
        if let Some(workspace) = binding.workspace_id() {
            config = config.with_workspace(workspace.clone());
        }
        if let Some(prompt) = prompt {
            config = config.with_compiled_prompt(prompt).map_err(|error| {
                RuntimeError::new(RuntimeErrorCode::InvalidRequest, error.message().to_owned())
            })?;
        }
        config = config
            .with_approval_ttl(binding.approval_ttl())
            .map_err(|error| {
                RuntimeError::new(RuntimeErrorCode::InvalidRequest, error.message().to_owned())
            })?
            .with_limits(binding.run_limits())
            .with_retry_policy(self.retry_policy)
            .with_compaction_policy(Arc::clone(&self.compaction_policy))
            .with_request_metadata(ProtocolMetadata::default());
        if let Some(summarizer) = &self.compaction_summarizer {
            config = config.with_compaction_summarizer(Arc::clone(summarizer));
        }
        Ok(config)
    }

    async fn append_user_message(
        &self,
        session_id: SessionId,
        tail: SessionSequence,
        message: CanonicalMessage,
    ) -> Result<(), RuntimeError> {
        let timestamp = self.clock_now()?;
        let snapshot = self.load_snapshot(session_id).await?;
        let records = self.build_records(
            session_id,
            timestamp,
            Some(tail),
            snapshot.state().active_branch_id(),
            [SessionRecord::MessageCommitted { message }],
        )?;
        self.append_records(session_id, Some(tail), records).await?;
        Ok(())
    }

    async fn run_kernel(
        &self,
        session_id: SessionId,
        models: Arc<tea_model::ModelRegistry>,
        binding: &crate::ProfileBinding,
        tools: &tea_tools::ToolRegistry,
        config: &KernelRunConfig,
        cancellation: CancellationScope,
    ) -> Result<tea_kernel::KernelRunOutcome, RuntimeError> {
        let queue = self.session_queue(session_id)?;
        let kernel = AgentKernel::new(
            models.as_ref(),
            tools,
            binding.policy(),
            self.sessions.as_ref(),
            self.clock.as_ref(),
            self.ids.as_ref(),
            self.event_sink.as_ref(),
        )
        .with_input_queue(queue.as_ref());
        kernel
            .run(session_id, config, cancellation)
            .await
            .map_err(RuntimeError::from)
    }

    async fn resume_kernel(
        &self,
        session_id: SessionId,
        models: Arc<tea_model::ModelRegistry>,
        binding: &crate::ProfileBinding,
        tools: &tea_tools::ToolRegistry,
        resolution: &ApprovalResolution,
        config: &KernelRunConfig,
        cancellation: CancellationScope,
    ) -> Result<tea_kernel::KernelRunOutcome, RuntimeError> {
        let queue = self.session_queue(session_id)?;
        let kernel = AgentKernel::new(
            models.as_ref(),
            tools,
            binding.policy(),
            self.sessions.as_ref(),
            self.clock.as_ref(),
            self.ids.as_ref(),
            self.event_sink.as_ref(),
        )
        .with_input_queue(queue.as_ref());
        kernel
            .resume_approval(session_id, resolution, config, cancellation)
            .await
            .map_err(RuntimeError::from)
    }

    #[allow(clippy::unused_async)]
    async fn handle_steer(
        &self,
        session_id: Option<SessionId>,
        text: tea_protocol::CommandText,
    ) -> Result<RuntimeCommandOutcome, RuntimeError> {
        let session_id = Self::require_session(session_id)?;
        let queue = self.session_queue(session_id)?;
        queue.enqueue_steering(text).map_err(|error| {
            RuntimeError::new(RuntimeErrorCode::InvalidRequest, error.message().to_owned())
        })?;
        let (follow_ups, steering) = queue.lengths().map_err(|error| {
            RuntimeError::new(RuntimeErrorCode::InvalidRequest, error.message().to_owned())
        })?;
        Ok(RuntimeCommandOutcome::Enqueued {
            follow_ups,
            steering,
        })
    }

    #[allow(clippy::unused_async)]
    async fn handle_follow_up(
        &self,
        session_id: Option<SessionId>,
        message: CanonicalMessage,
    ) -> Result<RuntimeCommandOutcome, RuntimeError> {
        let session_id = Self::require_session(session_id)?;
        let queue = self.session_queue(session_id)?;
        queue.enqueue_follow_up(message).map_err(|error| {
            RuntimeError::new(RuntimeErrorCode::InvalidRequest, error.message().to_owned())
        })?;
        let (follow_ups, steering) = queue.lengths().map_err(|error| {
            RuntimeError::new(RuntimeErrorCode::InvalidRequest, error.message().to_owned())
        })?;
        Ok(RuntimeCommandOutcome::Enqueued {
            follow_ups,
            steering,
        })
    }

    pub(crate) fn has_active_run(&self, session_id: SessionId) -> bool {
        self.active_runs
            .lock()
            .is_ok_and(|runs| runs.contains_key(&session_id))
    }

    fn begin_active_run(&self, session_id: SessionId) -> Result<ActiveRunGuard<'_>, RuntimeError> {
        let cancellation = CancellationScope::new();
        let mut runs = self.active_runs.lock().map_err(|_| {
            RuntimeError::new(
                RuntimeErrorCode::InvalidState,
                "runtime active-run tracker is poisoned",
            )
        })?;
        if runs.contains_key(&session_id) {
            return Err(RuntimeError::new(
                RuntimeErrorCode::RunAlreadyActive,
                "a run is already active on this session",
            ));
        }
        runs.insert(session_id, cancellation.clone());
        drop(runs);
        Ok(ActiveRunGuard {
            runtime: self,
            session_id,
            cancellation,
        })
    }

    fn clear_active_run(&self, session_id: SessionId) {
        if let Ok(mut runs) = self.active_runs.lock() {
            runs.remove(&session_id);
        }
    }
}

fn session_grant(
    grant_id: tea_policy::GrantId,
    request: &tea_policy::ApprovalRequest,
    session_id: SessionId,
    issued_at: tea_protocol::ProtocolTimestamp,
) -> Result<PolicyGrant, RuntimeError> {
    let resources = request
        .resources()
        .iter()
        .map(|resource| {
            ResourcePattern::new(
                resource.scheme(),
                resource.locator(),
                Some(resource.access()),
            )
        })
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| RuntimeError::new(RuntimeErrorCode::PolicyFailure, error.to_string()))?;
    PolicyGrant::new(
        grant_id,
        request.actor_id().clone(),
        request.profile_id().clone(),
        request.tool_name().clone(),
        request.tool_version().clone(),
        request.effects().iter().cloned(),
        resources,
        GrantScope::SessionResource { session_id },
        issued_at,
    )
    .map(|grant| grant.with_source(request.tool_source().clone()))
    .map_err(|error| RuntimeError::new(RuntimeErrorCode::PolicyFailure, error.to_string()))
}

const fn canonical_message_id(message: &CanonicalMessage) -> MessageId {
    match message {
        CanonicalMessage::User { id, .. }
        | CanonicalMessage::Assistant { id, .. }
        | CanonicalMessage::ToolResult { id, .. } => *id,
    }
}
