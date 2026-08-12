use std::collections::BTreeSet;

use tea_control::CancellationScope;
use tea_model::ModelRouter;
use tea_policy::{ApprovalRequest, ApprovalResolution, PolicyDecision, PolicyEngine, PolicyInput};
use tea_protocol::{
    AgentEvent, ApprovalId, CanonicalMessage, ContentBlock, ExecutionTarget, NextTurnAction,
    ProtocolMetadata, RunId, RunStatus, SessionId, SessionRecord, StopReason, ToolCallId, TurnId,
};
use tea_session::{
    AppendTransaction, ApprovalArtifactEntry, GrantJournalEntry, SessionSnapshot, SessionStore,
};
use tea_tools::{ToolRegistry, ValidatedToolInvocation};

use crate::approval::{add_duration, request as approval_request};
use crate::model_turn::{CompletedToolCall, ModelTurnOutput, stream_turn};
use crate::observe::EventEmitter;
use crate::policy::{denial_reason, durable_decision, selected_target};
use crate::record::{envelopes, envelopes_at, envelopes_with_metadata, tool_audit_metadata};
use crate::scheduler::{SchedulePlan, Scheduler};
use crate::tool_call::{PreparedToolCall, prepare};
use crate::tool_execution::{
    CollectedExecution, ToolExecutionContext, ToolTerminal, collect_execution, execute,
    failed_terminal,
};
use crate::{
    KernelClock, KernelError, KernelErrorCode, KernelEventSink, KernelIdSource, KernelInputQueue,
    KernelRunConfig, RunState, TurnRequestSnapshot,
};

/// Terminal result of one kernel invocation.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelRunOutcome {
    run_id: RunId,
    state: RunState,
    session: tea_session::MaterializedSessionState,
    pending_approval_id: Option<ApprovalId>,
}

impl KernelRunOutcome {
    /// Returns the stable run identity.
    #[must_use]
    pub const fn run_id(&self) -> RunId {
        self.run_id
    }

    /// Returns terminal run state.
    #[must_use]
    pub const fn state(&self) -> RunState {
        self.state
    }

    /// Returns the committed materialized session state.
    #[must_use]
    pub const fn session(&self) -> &tea_session::MaterializedSessionState {
        &self.session
    }

    /// Returns the durable approval that paused this invocation.
    #[must_use]
    pub const fn pending_approval_id(&self) -> Option<ApprovalId> {
        self.pending_approval_id
    }
}

enum ToolProcessOutcome {
    Continue(SessionSnapshot),
    Waiting(KernelRunOutcome),
}

struct RunProgress {
    tool_iterations: u32,
    assistant_output_bytes: usize,
    deadline: tea_protocol::ProtocolTimestamp,
}

impl RunProgress {
    fn output_budget(&self, config: &KernelRunConfig) -> Result<usize, KernelError> {
        config
            .limits()
            .max_assistant_output_bytes()
            .checked_sub(self.assistant_output_bytes)
            .ok_or_else(|| {
                KernelError::new(
                    KernelErrorCode::LimitExceeded,
                    "assistant output byte limit was reached",
                )
            })
    }

    fn observe_output(&mut self, bytes: usize) -> Result<(), KernelError> {
        self.assistant_output_bytes =
            self.assistant_output_bytes
                .checked_add(bytes)
                .ok_or_else(|| {
                    KernelError::new(
                        KernelErrorCode::LimitExceeded,
                        "assistant output byte count cannot advance",
                    )
                })?;
        Ok(())
    }

    fn advance_tools(&mut self, config: &KernelRunConfig) -> Result<(), KernelError> {
        self.tool_iterations = self.tool_iterations.checked_add(1).ok_or_else(|| {
            KernelError::new(
                KernelErrorCode::LimitExceeded,
                "tool iteration count cannot advance",
            )
        })?;
        if self.tool_iterations > config.limits().max_tool_iterations() {
            return Err(KernelError::new(
                KernelErrorCode::LimitExceeded,
                "tool iteration limit was reached",
            ));
        }
        Ok(())
    }
}

/// Headless coordinator for model, tool, policy, session, and observation ports.
#[derive(Debug)]
pub struct AgentKernel<'a> {
    models: &'a dyn ModelRouter,
    tools: &'a ToolRegistry,
    policy: &'a PolicyEngine,
    sessions: &'a dyn SessionStore,
    clock: &'a dyn KernelClock,
    ids: &'a dyn KernelIdSource,
    events: &'a dyn KernelEventSink,
    input_queue: Option<&'a KernelInputQueue>,
}

impl<'a> AgentKernel<'a> {
    /// Creates a kernel from replaceable inward-facing ports.
    #[must_use]
    pub const fn new(
        models: &'a dyn ModelRouter,
        tools: &'a ToolRegistry,
        policy: &'a PolicyEngine,
        sessions: &'a dyn SessionStore,
        clock: &'a dyn KernelClock,
        ids: &'a dyn KernelIdSource,
        events: &'a dyn KernelEventSink,
    ) -> Self {
        Self {
            models,
            tools,
            policy,
            sessions,
            clock,
            ids,
            events,
            input_queue: None,
        }
    }

    /// Attaches a bounded queue consumed only between durable turns.
    #[must_use]
    pub const fn with_input_queue(mut self, queue: &'a KernelInputQueue) -> Self {
        self.input_queue = Some(queue);
        self
    }

    /// Appends a manual compaction summary, replacing the compacted prefix in
    /// the model-visible transcript with the supplied summary message.
    ///
    /// The summary must be an assistant message. `compacted_through_record_id`
    /// must be a record on the active branch. Original records remain in the
    /// durable log for audit and replay.
    ///
    /// # Errors
    ///
    /// Returns an error when the session cannot be loaded, the summary is not
    /// an assistant message, the referenced record is not on the active
    /// branch, or the append transaction fails.
    pub async fn compact(
        &self,
        session_id: SessionId,
        summary: CanonicalMessage,
        compacted_through_record_id: tea_protocol::RecordId,
    ) -> Result<tea_session::SessionSnapshot, KernelError> {
        let snapshot = self
            .sessions
            .load(session_id)
            .await
            .map_err(KernelError::from)?;
        let timestamp = self.clock.now()?;
        let records = envelopes_at(
            self.ids,
            session_id,
            snapshot.state().tail_sequence(),
            snapshot.state().active_branch_id(),
            timestamp,
            [SessionRecord::SessionCompacted {
                summary,
                compacted_through_record_id,
            }],
        )?;
        self.sessions
            .append(
                AppendTransaction::new(session_id, Some(snapshot.state().tail_sequence()), records)
                    .with_expected_journal_revision(snapshot.journal_revision()),
            )
            .await
            .map_err(KernelError::from)?;
        self.sessions
            .load(session_id)
            .await
            .map_err(KernelError::from)
    }

    /// Runs serial model/tool turns until completion or durable approval pause.
    ///
    /// # Errors
    ///
    /// Returns a typed failure for invalid durable state, adapter contract
    /// violations, rejected events, policy/tool failures, cancellation, or
    /// deterministic run limits.
    pub async fn run(
        &self,
        session_id: SessionId,
        config: &KernelRunConfig,
        cancellation: CancellationScope,
    ) -> Result<KernelRunOutcome, KernelError> {
        let snapshot = self
            .sessions
            .load(session_id)
            .await
            .map_err(KernelError::from)?;
        validate_new_run_state(&snapshot)?;
        let run_id = self.ids.next_run_id()?;
        let mut emitter = EventEmitter::new(
            self.ids,
            self.clock,
            self.events,
            session_id,
            snapshot.state().tail_sequence(),
            config.limits(),
        );
        emitter
            .emit(Some(run_id), None, AgentEvent::RunStarted {})
            .await?;

        let deadline = add_duration(self.clock.now()?, config.limits().max_elapsed())?;
        self.run_loop(
            snapshot,
            config,
            cancellation,
            run_id,
            &mut emitter,
            RunProgress {
                tool_iterations: 0,
                assistant_output_bytes: 0,
                deadline,
            },
        )
        .await
    }

    /// Resolves one persisted approval and continues its original run.
    ///
    /// The supplied resolution must contain the exact rich request snapshot
    /// stored with the canonical pending approval. Tool arguments and resources
    /// are always loaded from durable session state.
    ///
    /// # Errors
    ///
    /// Returns a typed failure for missing, stale, mismatched, expired, or
    /// concurrently resolved approvals and for subsequent runtime failures.
    #[allow(clippy::too_many_lines)]
    pub async fn resume_approval(
        &self,
        session_id: SessionId,
        resolution: &ApprovalResolution,
        config: &KernelRunConfig,
        cancellation: CancellationScope,
    ) -> Result<KernelRunOutcome, KernelError> {
        let snapshot = self
            .sessions
            .load(session_id)
            .await
            .map_err(KernelError::from)?;
        let request = persisted_request(&snapshot, *resolution.request().approval_id())?.clone();
        if &request != resolution.request()
            || request.actor_id() != config.actor_id()
            || request.workspace_id() != config.workspace_id()
            || request.environment() != config.environment()
        {
            return Err(KernelError::new(
                KernelErrorCode::PolicyFailure,
                "approval resolution does not match persisted request context",
            ));
        }
        let run_id = request.run_id().copied().ok_or_else(|| {
            KernelError::new(
                KernelErrorCode::InvalidState,
                "persisted approval has no active run",
            )
        })?;
        let turn_id = snapshot
            .state()
            .latest_checkpoint()
            .filter(|checkpoint| {
                checkpoint.run_id() == run_id
                    && checkpoint.next_action() == NextTurnAction::WaitForApproval
            })
            .map(tea_session::TurnCheckpoint::turn_id)
            .ok_or_else(|| {
                KernelError::new(
                    KernelErrorCode::InvalidState,
                    "persisted approval has no matching wait checkpoint",
                )
            })?;
        let tool = snapshot
            .state()
            .tool_calls()
            .get(request.tool_call_id())
            .ok_or_else(|| {
                KernelError::new(
                    KernelErrorCode::InvalidState,
                    "persisted approval tool call is missing",
                )
            })?;
        let call = CompletedToolCall {
            tool_call_id: tool.tool_call_id(),
            tool_name: tool.tool_name().to_owned(),
            arguments: tool.arguments().clone(),
        };
        let invocation = match prepare(self.tools, &call) {
            PreparedToolCall::Valid(invocation) => invocation,
            PreparedToolCall::Rejected { .. } => {
                return Err(KernelError::new(
                    KernelErrorCode::ToolFailure,
                    "persisted approval tool no longer validates",
                ));
            }
        };
        if invocation.name() != request.tool_name()
            || invocation.spec().version() != request.tool_version()
            || invocation.source() != request.tool_source()
            || invocation.spec().effects() != request.effects()
            || invocation.resources() != request.resources()
        {
            return Err(KernelError::new(
                KernelErrorCode::PolicyFailure,
                "persisted approval no longer matches registered tool context",
            ));
        }
        let deadline = add_duration(self.clock.now()?, config.limits().max_elapsed())?;
        let mut emitter = EventEmitter::new(
            self.ids,
            self.clock,
            self.events,
            session_id,
            snapshot.state().tail_sequence(),
            config.limits(),
        );
        let denied = matches!(resolution.decision(), tea_protocol::ApprovalDecision::Deny);
        let snapshot = if denied {
            self.commit_denied_resolution(snapshot, resolution, &call)
                .await?
        } else {
            let snapshot = self
                .commit_resolution(
                    snapshot,
                    resolution,
                    &invocation,
                    config.environment().target(),
                )
                .await?;
            let terminal = match execute(
                self.tools,
                invocation,
                cancellation.child(),
                &mut ToolExecutionContext {
                    emitter: &mut emitter,
                    run_id,
                    turn_id,
                    clock: self.clock,
                    deadline,
                },
            )
            .await
            {
                Ok(terminal) => terminal,
                Err(error) => {
                    self.record_tool_interruption(
                        &snapshot,
                        &call,
                        run_id,
                        turn_id,
                        &error,
                        &mut emitter,
                    )
                    .await?;
                    return Err(error);
                }
            };
            self.commit_terminal(snapshot, &call, terminal).await?
        };
        let mut snapshot = snapshot;
        for pending_call in remaining_tool_calls(&snapshot, call.tool_call_id) {
            match self
                .process_tool(
                    snapshot,
                    &pending_call,
                    config,
                    run_id,
                    turn_id,
                    cancellation.child(),
                    &mut emitter,
                    deadline,
                )
                .await?
            {
                ToolProcessOutcome::Continue(next) => snapshot = next,
                ToolProcessOutcome::Waiting(outcome) => return Ok(outcome),
            }
        }
        let snapshot = self
            .checkpoint_model_request(snapshot, run_id, turn_id, &mut emitter)
            .await?;
        let tool_iterations = completed_tool_iterations(&snapshot, run_id);
        let output_bytes = committed_assistant_output_bytes(&snapshot, run_id)?;
        self.run_loop(
            snapshot,
            config,
            cancellation,
            run_id,
            &mut emitter,
            RunProgress {
                tool_iterations,
                assistant_output_bytes: output_bytes,
                deadline,
            },
        )
        .await
    }

    async fn checkpoint_model_request(
        &self,
        snapshot: SessionSnapshot,
        run_id: RunId,
        turn_id: TurnId,
        emitter: &mut EventEmitter<'_>,
    ) -> Result<SessionSnapshot, KernelError> {
        let snapshot = self
            .append_records(
                &snapshot,
                [SessionRecord::TurnCheckpointed {
                    run_id,
                    turn_id,
                    next_action: NextTurnAction::ModelRequest,
                }],
            )
            .await?;
        emitter
            .emit(Some(run_id), Some(turn_id), AgentEvent::TurnCheckpointed {})
            .await?;
        Ok(snapshot)
    }

    #[allow(clippy::too_many_lines)]
    async fn run_loop(
        &self,
        mut snapshot: SessionSnapshot,
        config: &KernelRunConfig,
        cancellation: CancellationScope,
        run_id: RunId,
        emitter: &mut EventEmitter<'_>,
        mut progress: RunProgress,
    ) -> Result<KernelRunOutcome, KernelError> {
        let mut auto_compacted = false;
        loop {
            if cancellation.is_cancelled() {
                let error = KernelError::new(KernelErrorCode::Cancelled, "agent run was cancelled");
                self.record_run_terminal(&snapshot, run_id, None, &error, emitter)
                    .await?;
                return Err(error);
            }
            snapshot = self.apply_queued_input(snapshot).await?;
            let turn_id = self.ids.next_turn_id()?;
            let model_ref = snapshot
                .state()
                .configuration()
                .model_ref()
                .ok_or_else(|| {
                    KernelError::new(KernelErrorCode::InvalidModel, "session has no active model")
                })?;
            let provider = self
                .models
                .provider(model_ref.provider_id())
                .ok_or_else(|| {
                    KernelError::new(
                        KernelErrorCode::InvalidModel,
                        "active model provider is not registered",
                    )
                })?;
            let model = self.models.model(model_ref).ok_or_else(|| {
                KernelError::new(
                    KernelErrorCode::InvalidModel,
                    "active model is not advertised by its provider",
                )
            })?;
            let request = TurnRequestSnapshot::build(snapshot.state(), config, self.tools, model)?;
            let accountant = crate::ContextWindowAccountant::new(model);
            let estimated = crate::ContextWindowAccountant::estimate_input_tokens(
                config.compiled_prompt(),
                config.system_prompt(),
                request.request().tools(),
                snapshot.state().messages(),
            );
            if let Err(error) = accountant.check_overflow(estimated) {
                if !auto_compacted
                    && config
                        .compaction_policy()
                        .should_compact(estimated, accountant.context_window().get())
                    && let Some(summarizer) = config.compaction_summarizer()
                {
                    let summary = summarizer
                        .summarize(snapshot.state().messages().to_vec())
                        .await?;
                    let tail = snapshot.state().tail_record_id();
                    snapshot = self
                        .compact(snapshot.state().session_id(), summary, tail)
                        .await?;
                    auto_compacted = true;
                    continue;
                }
                self.record_run_terminal(&snapshot, run_id, Some(turn_id), &error, emitter)
                    .await?;
                return Err(error);
            }
            let output = match stream_turn(
                provider,
                request,
                self.ids,
                emitter,
                run_id,
                turn_id,
                cancellation.child(),
                self.clock,
                progress.deadline,
                progress.output_budget(config)?,
                config.retry_policy(),
            )
            .await
            {
                Ok(output) => output,
                Err(error) => {
                    self.record_run_terminal(&snapshot, run_id, Some(turn_id), &error, emitter)
                        .await?;
                    return Err(error);
                }
            };
            progress.observe_output(output.output_bytes)?;
            if output.tool_calls.is_empty() {
                if output.completion.stop_reason() == &StopReason::PauseTurn {
                    if let Err(error) = progress.advance_tools(config) {
                        self.record_run_terminal(&snapshot, run_id, Some(turn_id), &error, emitter)
                            .await?;
                        return Err(error);
                    }
                    snapshot = self
                        .commit_declarations(snapshot, &output, run_id, turn_id, emitter)
                        .await?;
                    snapshot = self
                        .append_records(
                            &snapshot,
                            [SessionRecord::TurnCheckpointed {
                                run_id,
                                turn_id,
                                next_action: NextTurnAction::ModelRequest,
                            }],
                        )
                        .await?;
                    emitter
                        .emit(Some(run_id), Some(turn_id), AgentEvent::TurnCheckpointed {})
                        .await?;
                    continue;
                }
                return self
                    .finish(snapshot, output, run_id, turn_id, emitter)
                    .await;
            }
            if let Err(error) = progress.advance_tools(config) {
                self.record_run_terminal(&snapshot, run_id, Some(turn_id), &error, emitter)
                    .await?;
                return Err(error);
            }
            snapshot = self
                .commit_declarations(snapshot, &output, run_id, turn_id, emitter)
                .await?;
            match self
                .process_tool_turn(
                    snapshot,
                    &output,
                    config,
                    run_id,
                    turn_id,
                    cancellation.child(),
                    emitter,
                    progress.deadline,
                )
                .await?
            {
                ToolProcessOutcome::Continue(next) => snapshot = next,
                ToolProcessOutcome::Waiting(outcome) => return Ok(outcome),
            }
            snapshot = self
                .append_records(
                    &snapshot,
                    [SessionRecord::TurnCheckpointed {
                        run_id,
                        turn_id,
                        next_action: NextTurnAction::ModelRequest,
                    }],
                )
                .await?;
            emitter
                .emit(Some(run_id), Some(turn_id), AgentEvent::TurnCheckpointed {})
                .await?;
        }
    }

    async fn apply_queued_input(
        &self,
        snapshot: SessionSnapshot,
    ) -> Result<SessionSnapshot, KernelError> {
        let Some(queue) = self.input_queue else {
            return Ok(snapshot);
        };
        let queued = queue.snapshot()?;
        if queued.follow_ups.is_empty() && queued.steering.is_empty() {
            return Ok(snapshot);
        }
        let mut records = Vec::with_capacity(queued.follow_ups.len() + 1);
        if !queued.steering.is_empty() {
            let text = queued
                .steering
                .iter()
                .map(tea_protocol::CommandText::as_str)
                .collect::<Vec<_>>()
                .join("\n");
            let message = CanonicalMessage::user(
                self.ids.next_message_id()?,
                vec![ContentBlock::text(text)?],
                self.clock.now()?,
            )?;
            records.push(SessionRecord::MessageCommitted { message });
        }
        records.extend(
            queued
                .follow_ups
                .iter()
                .cloned()
                .map(|message| SessionRecord::MessageCommitted { message }),
        );
        let snapshot = self.append_records(&snapshot, records).await?;
        queue.acknowledge(&queued)?;
        Ok(snapshot)
    }

    async fn commit_declarations(
        &self,
        snapshot: SessionSnapshot,
        output: &ModelTurnOutput,
        run_id: RunId,
        turn_id: TurnId,
        emitter: &mut EventEmitter<'_>,
    ) -> Result<SessionSnapshot, KernelError> {
        let mut records = Vec::with_capacity(output.tool_calls.len() + 1);
        records.push((
            ProtocolMetadata::default(),
            SessionRecord::MessageCommitted {
                message: output.message.clone(),
            },
        ));
        for call in &output.tool_calls {
            let metadata = match prepare(self.tools, call) {
                PreparedToolCall::Valid(invocation) => tool_audit_metadata(&invocation)?,
                PreparedToolCall::Rejected { .. } => ProtocolMetadata::default(),
            };
            records.push((
                metadata,
                SessionRecord::ToolCallRequested {
                    tool_call_id: call.tool_call_id,
                    tool_name: call.tool_name.clone(),
                    arguments: call.arguments.clone(),
                },
            ));
        }
        let snapshot = self
            .append_records_with_metadata(&snapshot, records)
            .await?;
        for call in &output.tool_calls {
            emitter
                .emit(
                    Some(run_id),
                    Some(turn_id),
                    AgentEvent::ToolCallRequested {
                        tool_call_id: call.tool_call_id,
                        tool_name: call.tool_name.clone(),
                        arguments: call.arguments.clone(),
                    },
                )
                .await?;
        }
        Ok(snapshot)
    }

    /// Processes all tool calls declared by a model turn with effect-aware
    /// parallel scheduling and source-order result commits.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    async fn process_tool_turn(
        &self,
        mut snapshot: SessionSnapshot,
        output: &ModelTurnOutput,
        config: &KernelRunConfig,
        run_id: RunId,
        turn_id: TurnId,
        cancellation: CancellationScope,
        emitter: &mut EventEmitter<'_>,
        deadline: tea_protocol::ProtocolTimestamp,
    ) -> Result<ToolProcessOutcome, KernelError> {
        #[allow(clippy::large_enum_variant)] // Short-lived turn values avoid extra allocation.
        enum Classified {
            Rejected {
                index: usize,
                code: &'static str,
                message: &'static str,
            },
            Denied {
                index: usize,
                reason: String,
            },
            Allow {
                invocation: ValidatedToolInvocation,
                target: ExecutionTarget,
            },
            Ask {
                // An approval is rare and ends the turn. Keep its validated
                // invocation out of the turn-wide classification enum.
                invocation: Box<ValidatedToolInvocation>,
                input: PolicyInput,
                reason: String,
            },
        }

        let mut classified: Vec<Classified> = Vec::with_capacity(output.tool_calls.len());
        for (index, call) in output.tool_calls.iter().enumerate() {
            match prepare(self.tools, call) {
                PreparedToolCall::Rejected { code, message } => {
                    classified.push(Classified::Rejected {
                        index,
                        code,
                        message,
                    });
                }
                PreparedToolCall::Valid(invocation) => {
                    let grant_candidates = self
                        .sessions
                        .active_grants_for_actor(config.actor_id().clone())
                        .await
                        .map_err(KernelError::from)?;
                    let input = PolicyInput::from_validated(
                        config.actor_id().clone(),
                        snapshot.state().configuration().profile_id().clone(),
                        snapshot.state().session_id(),
                        Some(run_id),
                        config.workspace_id().cloned(),
                        &invocation,
                        config.environment().clone(),
                        self.clock.now()?,
                        grant_candidates,
                    )
                    .map_err(|error| {
                        KernelError::new(KernelErrorCode::PolicyFailure, error.to_string())
                    })?;
                    let evaluation = self.policy.evaluate(&input);
                    match evaluation.decision() {
                        PolicyDecision::Allow | PolicyDecision::Redirect { .. } => {
                            let target = selected_target(
                                evaluation.decision(),
                                config.environment().target(),
                                invocation.source().kind(),
                            );
                            classified.push(Classified::Allow { invocation, target });
                        }
                        PolicyDecision::Deny { .. } | PolicyDecision::HardDeny { .. } => {
                            let reason = denial_reason(evaluation.decision())
                                .unwrap_or("policy denied tool");
                            classified.push(Classified::Denied {
                                index,
                                reason: reason.to_owned(),
                            });
                        }
                        PolicyDecision::Ask(requirement) => {
                            classified.push(Classified::Ask {
                                invocation: Box::new(invocation),
                                input,
                                reason: requirement.reason().to_owned(),
                            });
                            break;
                        }
                    }
                }
            }
        }

        let mut batch: Vec<(ValidatedToolInvocation, ExecutionTarget)> = Vec::new();
        for entry in classified {
            match entry {
                Classified::Rejected {
                    index,
                    code,
                    message,
                } => {
                    snapshot = self
                        .flush_tool_batch(
                            snapshot,
                            &batch,
                            run_id,
                            turn_id,
                            cancellation.child(),
                            emitter,
                            deadline,
                        )
                        .await?;
                    batch.clear();
                    let terminal = failed_terminal(code, message)?;
                    snapshot = self
                        .commit_denied(snapshot, &output.tool_calls[index], terminal)
                        .await?;
                }
                Classified::Denied { index, reason } => {
                    snapshot = self
                        .flush_tool_batch(
                            snapshot,
                            &batch,
                            run_id,
                            turn_id,
                            cancellation.child(),
                            emitter,
                            deadline,
                        )
                        .await?;
                    let terminal = failed_terminal("policy_denied", &reason)?;
                    snapshot = self
                        .commit_denied(snapshot, &output.tool_calls[index], terminal)
                        .await?;
                }
                Classified::Allow { invocation, target } => {
                    batch.push((invocation, target));
                }
                Classified::Ask {
                    invocation,
                    input,
                    reason,
                } => {
                    snapshot = self
                        .flush_tool_batch(
                            snapshot,
                            &batch,
                            run_id,
                            turn_id,
                            cancellation.child(),
                            emitter,
                            deadline,
                        )
                        .await?;
                    return self
                        .wait_for_approval(
                            snapshot,
                            &invocation,
                            &input,
                            &reason,
                            config,
                            run_id,
                            turn_id,
                            emitter,
                        )
                        .await
                        .map(ToolProcessOutcome::Waiting);
                }
            }
        }
        snapshot = self
            .flush_tool_batch(
                snapshot,
                &batch,
                run_id,
                turn_id,
                cancellation.child(),
                emitter,
                deadline,
            )
            .await?;
        Ok(ToolProcessOutcome::Continue(snapshot))
    }

    /// Executes a batch of policy-allowed invocations with effect-aware
    /// scheduling, committing durable results in canonical source order.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_lines)]
    async fn flush_tool_batch(
        &self,
        snapshot: SessionSnapshot,
        batch: &[(ValidatedToolInvocation, ExecutionTarget)],
        run_id: RunId,
        turn_id: TurnId,
        cancellation: CancellationScope,
        emitter: &mut EventEmitter<'_>,
        deadline: tea_protocol::ProtocolTimestamp,
    ) -> Result<SessionSnapshot, KernelError> {
        if batch.is_empty() {
            return Ok(snapshot);
        }
        let invocations: Vec<&ValidatedToolInvocation> =
            batch.iter().map(|(invocation, _)| invocation).collect();
        let plan = Scheduler.plan(&invocations)?;
        // Commit policy decisions and durable starts in source order.
        let start_records: Vec<SessionRecord> = batch
            .iter()
            .flat_map(|(invocation, target)| {
                [
                    SessionRecord::PolicyDecisionRecorded {
                        tool_call_id: *invocation.tool_call_id(),
                        decision: tea_protocol::PolicyDecision::Allow,
                    },
                    SessionRecord::ToolExecutionStarted {
                        tool_call_id: *invocation.tool_call_id(),
                        execution_target: *target,
                        idempotency: invocation.spec().execution().idempotency(),
                    },
                ]
            })
            .collect();
        let mut snapshot = self.append_records(&snapshot, start_records).await?;
        // Execute lanes; collect per-invocation results keyed by tool-call id.
        let mut results: std::collections::HashMap<ToolCallId, CollectedExecution> =
            std::collections::HashMap::new();
        let exec_result = self
            .execute_schedule_lanes(&plan, &mut results, cancellation, deadline)
            .await;
        if let Err(error) = exec_result {
            let interrupted: Vec<SessionRecord> = batch
                .iter()
                .filter(|(invocation, _)| !results.contains_key(invocation.tool_call_id()))
                .map(|(invocation, _)| SessionRecord::ToolExecutionInterrupted {
                    tool_call_id: *invocation.tool_call_id(),
                    reason: error.message().to_owned(),
                })
                .collect();
            if !interrupted.is_empty() {
                snapshot = self.append_records(&snapshot, interrupted).await?;
            }
            self.record_run_terminal(&snapshot, run_id, Some(turn_id), &error, emitter)
                .await?;
            return Err(error);
        }
        // Emit collected progress in source order.
        for (invocation, _) in batch {
            if let Some(collected) = results.get(invocation.tool_call_id()) {
                for progress in &collected.progress {
                    emitter
                        .emit(
                            Some(run_id),
                            Some(turn_id),
                            AgentEvent::ToolExecutionProgress {
                                tool_call_id: progress.tool_call_id,
                                message: progress.message.clone(),
                                completed_units: progress.completed_units,
                                total_units: progress.total_units,
                            },
                        )
                        .await?;
                }
            }
        }
        // Commit terminals and tool-result messages in source order.
        let mut records: Vec<SessionRecord> = Vec::with_capacity(batch.len() * 2);
        for (invocation, _) in batch {
            let collected = results.get(invocation.tool_call_id()).ok_or_else(|| {
                KernelError::new(
                    KernelErrorCode::InvalidState,
                    "parallel tool batch is missing a collected terminal",
                )
            })?;
            let terminal = &collected.terminal;
            let is_error = terminal.failure.is_some();
            records.push(SessionRecord::ToolExecutionFinished {
                tool_call_id: *invocation.tool_call_id(),
                is_error,
                content: terminal.content.clone(),
                error: terminal.failure.clone(),
                presentation: terminal.presentation.clone(),
            });
            let message_id = self.ids.next_message_id()?;
            let message = if let Some(failure) = &terminal.failure {
                CanonicalMessage::tool_result_failure(
                    message_id,
                    *invocation.tool_call_id(),
                    invocation.name().to_string(),
                    terminal.content.clone(),
                    failure.clone(),
                    self.clock.now()?,
                )?
            } else {
                CanonicalMessage::tool_result_success(
                    message_id,
                    *invocation.tool_call_id(),
                    invocation.name().to_string(),
                    terminal.content.clone(),
                    self.clock.now()?,
                )?
            };
            records.push(SessionRecord::MessageCommitted { message });
        }
        snapshot = self.append_records(&snapshot, records).await?;
        Ok(snapshot)
    }

    /// Runs every lane of a schedule, filling `results` keyed by tool-call id.
    ///
    /// The parallel lane runs concurrently via `join_all`; serial and exclusive
    /// lanes run one invocation at a time. The first execution error aborts the
    /// remaining work; the caller records interruptions for started-but-
    /// unterminated invocations.
    async fn execute_schedule_lanes(
        &self,
        plan: &SchedulePlan<'_>,
        results: &mut std::collections::HashMap<ToolCallId, CollectedExecution>,
        cancellation: CancellationScope,
        deadline: tea_protocol::ProtocolTimestamp,
    ) -> Result<(), KernelError> {
        for lane in plan.lanes() {
            if lane.allows_parallel() {
                let invocations = lane.invocations().to_vec();
                let futures = invocations.iter().map(|invocation| {
                    collect_execution(
                        self.tools,
                        (*invocation).clone(),
                        cancellation.child(),
                        self.clock,
                        deadline,
                    )
                });
                let lane_results = futures_util::future::join_all(futures).await;
                for (invocation, collected) in invocations.into_iter().zip(lane_results) {
                    let collected = collected?;
                    results.insert(*invocation.tool_call_id(), collected);
                }
            } else {
                for invocation in lane.invocations() {
                    let collected = collect_execution(
                        self.tools,
                        (*invocation).clone(),
                        cancellation.child(),
                        self.clock,
                        deadline,
                    )
                    .await?;
                    results.insert(*invocation.tool_call_id(), collected);
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn process_tool(
        &self,
        snapshot: SessionSnapshot,
        call: &CompletedToolCall,
        config: &KernelRunConfig,
        run_id: RunId,
        turn_id: TurnId,
        cancellation: CancellationScope,
        emitter: &mut EventEmitter<'_>,
        deadline: tea_protocol::ProtocolTimestamp,
    ) -> Result<ToolProcessOutcome, KernelError> {
        match prepare(self.tools, call) {
            PreparedToolCall::Rejected { code, message } => {
                let terminal = failed_terminal(code, message)?;
                self.commit_denied(snapshot, call, terminal)
                    .await
                    .map(ToolProcessOutcome::Continue)
            }
            PreparedToolCall::Valid(invocation) => {
                self.process_validated(
                    snapshot,
                    call,
                    invocation,
                    config,
                    run_id,
                    turn_id,
                    cancellation,
                    emitter,
                    deadline,
                )
                .await
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn process_validated(
        &self,
        snapshot: SessionSnapshot,
        call: &CompletedToolCall,
        invocation: ValidatedToolInvocation,
        config: &KernelRunConfig,
        run_id: RunId,
        turn_id: TurnId,
        cancellation: CancellationScope,
        emitter: &mut EventEmitter<'_>,
        deadline: tea_protocol::ProtocolTimestamp,
    ) -> Result<ToolProcessOutcome, KernelError> {
        let grant_candidates = self
            .sessions
            .active_grants_for_actor(config.actor_id().clone())
            .await
            .map_err(KernelError::from)?;
        let input = PolicyInput::from_validated(
            config.actor_id().clone(),
            snapshot.state().configuration().profile_id().clone(),
            snapshot.state().session_id(),
            Some(run_id),
            config.workspace_id().cloned(),
            &invocation,
            config.environment().clone(),
            self.clock.now()?,
            grant_candidates,
        )
        .map_err(|error| KernelError::new(KernelErrorCode::PolicyFailure, error.to_string()))?;
        let evaluation = self.policy.evaluate(&input);
        match evaluation.decision() {
            PolicyDecision::Allow | PolicyDecision::Redirect { .. } => {
                let target = selected_target(
                    evaluation.decision(),
                    config.environment().target(),
                    invocation.source().kind(),
                );
                let snapshot = self
                    .append_records(
                        &snapshot,
                        [
                            SessionRecord::PolicyDecisionRecorded {
                                tool_call_id: call.tool_call_id,
                                decision: durable_decision(evaluation.decision()),
                            },
                            SessionRecord::ToolExecutionStarted {
                                tool_call_id: call.tool_call_id,
                                execution_target: target,
                                idempotency: invocation.spec().execution().idempotency(),
                            },
                        ],
                    )
                    .await?;
                let terminal = match execute(
                    self.tools,
                    invocation,
                    cancellation,
                    &mut ToolExecutionContext {
                        emitter,
                        run_id,
                        turn_id,
                        clock: self.clock,
                        deadline,
                    },
                )
                .await
                {
                    Ok(terminal) => terminal,
                    Err(error) => {
                        self.record_tool_interruption(
                            &snapshot, call, run_id, turn_id, &error, emitter,
                        )
                        .await?;
                        return Err(error);
                    }
                };
                self.commit_terminal(snapshot, call, terminal)
                    .await
                    .map(ToolProcessOutcome::Continue)
            }
            PolicyDecision::Deny { .. } | PolicyDecision::HardDeny { .. } => {
                let reason = denial_reason(evaluation.decision()).unwrap_or("policy denied tool");
                let terminal = failed_terminal("policy_denied", reason)?;
                self.commit_denied(snapshot, call, terminal)
                    .await
                    .map(ToolProcessOutcome::Continue)
            }
            PolicyDecision::Ask(requirement) => self
                .wait_for_approval(
                    snapshot,
                    &invocation,
                    &input,
                    requirement.reason(),
                    config,
                    run_id,
                    turn_id,
                    emitter,
                )
                .await
                .map(ToolProcessOutcome::Waiting),
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn wait_for_approval(
        &self,
        snapshot: SessionSnapshot,
        invocation: &ValidatedToolInvocation,
        input: &PolicyInput,
        reason: &str,
        config: &KernelRunConfig,
        run_id: RunId,
        turn_id: TurnId,
        emitter: &mut EventEmitter<'_>,
    ) -> Result<KernelRunOutcome, KernelError> {
        let approval_id = self.ids.next_approval_id()?;
        let created_at = self.clock.now()?;
        let request = approval_request(
            approval_id,
            input,
            reason,
            created_at,
            config.approval_ttl(),
        )?;
        let records = envelopes_at(
            self.ids,
            snapshot.state().session_id(),
            snapshot.state().tail_sequence(),
            snapshot.state().active_branch_id(),
            created_at,
            [
                SessionRecord::PolicyDecisionRecorded {
                    tool_call_id: *input.tool_call_id(),
                    decision: tea_protocol::PolicyDecision::RequireApproval,
                },
                SessionRecord::ApprovalRequested {
                    approval_id,
                    tool_call_id: *input.tool_call_id(),
                    expires_at: request.expires_at(),
                },
                SessionRecord::TurnCheckpointed {
                    run_id,
                    turn_id,
                    next_action: NextTurnAction::WaitForApproval,
                },
            ],
        )?;
        let approval_record_id = records
            .get(1)
            .map(tea_protocol::RecordEnvelope::record_id)
            .ok_or_else(|| {
                KernelError::new(
                    KernelErrorCode::InvalidState,
                    "approval transaction is missing its request record",
                )
            })?;
        self.sessions
            .append(
                AppendTransaction::new(
                    snapshot.state().session_id(),
                    Some(snapshot.state().tail_sequence()),
                    records,
                )
                .with_expected_journal_revision(snapshot.journal_revision())
                .with_approval_artifacts([ApprovalArtifactEntry::Requested {
                    record_id: approval_record_id,
                    request: request.clone(),
                }]),
            )
            .await
            .map_err(KernelError::from)?;
        let snapshot = self
            .sessions
            .load(snapshot.state().session_id())
            .await
            .map_err(KernelError::from)?;
        let capabilities = request
            .effects()
            .iter()
            .map(|effect| effect.as_str().to_owned())
            .collect();
        let mut resources = request.presentation().resources().to_vec();
        if resources.is_empty() {
            resources.push(format!("tool:{}", request.tool_name().as_str()));
        }
        self.emit_tool_preview(invocation, run_id, turn_id, emitter)
            .await?;
        emitter
            .emit(
                Some(run_id),
                Some(turn_id),
                AgentEvent::ApprovalRequested {
                    approval_id,
                    tool_call_id: *input.tool_call_id(),
                    capabilities,
                    resources,
                    expires_at: request.expires_at(),
                },
            )
            .await?;
        emitter
            .emit(Some(run_id), Some(turn_id), AgentEvent::TurnCheckpointed {})
            .await?;
        Ok(KernelRunOutcome {
            run_id,
            state: RunState::WaitingApproval,
            session: snapshot.state().clone(),
            pending_approval_id: Some(approval_id),
        })
    }

    async fn emit_tool_preview(
        &self,
        invocation: &ValidatedToolInvocation,
        run_id: RunId,
        turn_id: TurnId,
        emitter: &mut EventEmitter<'_>,
    ) -> Result<(), KernelError> {
        let Some(presentation) = self.tools.preview_validated(invocation) else {
            return Ok(());
        };
        emitter
            .emit(
                Some(run_id),
                Some(turn_id),
                AgentEvent::ToolExecutionPreview {
                    tool_call_id: *invocation.tool_call_id(),
                    presentation,
                },
            )
            .await
    }

    async fn commit_denied_resolution(
        &self,
        snapshot: SessionSnapshot,
        resolution: &ApprovalResolution,
        call: &CompletedToolCall,
    ) -> Result<SessionSnapshot, KernelError> {
        let terminal =
            failed_terminal("approval_denied", "tool invocation was denied by approval")?;
        let message = CanonicalMessage::tool_result_failure(
            self.ids.next_message_id()?,
            call.tool_call_id,
            call.tool_name.clone(),
            terminal.content.clone(),
            terminal.failure.clone().ok_or_else(|| {
                KernelError::new(
                    KernelErrorCode::InvalidState,
                    "approval denial has no failure",
                )
            })?,
            resolution.decided_at(),
        )?;
        let records = envelopes_at(
            self.ids,
            snapshot.state().session_id(),
            snapshot.state().tail_sequence(),
            snapshot.state().active_branch_id(),
            resolution.decided_at(),
            [
                SessionRecord::ApprovalResolved {
                    approval_id: *resolution.request().approval_id(),
                    decision: resolution.decision(),
                },
                SessionRecord::ToolExecutionFinished {
                    tool_call_id: call.tool_call_id,
                    is_error: true,
                    content: terminal.content,
                    error: terminal.failure,
                    presentation: None,
                },
                SessionRecord::MessageCommitted { message },
            ],
        )?;
        let record_id = records[0].record_id();
        self.sessions
            .append(
                AppendTransaction::new(
                    snapshot.state().session_id(),
                    Some(snapshot.state().tail_sequence()),
                    records,
                )
                .with_expected_journal_revision(snapshot.journal_revision())
                .with_approval_artifacts([ApprovalArtifactEntry::Resolved {
                    record_id,
                    resolution: resolution.clone(),
                }]),
            )
            .await
            .map_err(KernelError::from)?;
        self.sessions
            .load(snapshot.state().session_id())
            .await
            .map_err(KernelError::from)
    }

    async fn commit_resolution(
        &self,
        snapshot: SessionSnapshot,
        resolution: &ApprovalResolution,
        invocation: &ValidatedToolInvocation,
        target: tea_policy::PolicyExecutionTarget,
    ) -> Result<SessionSnapshot, KernelError> {
        let allowed = matches!(
            resolution.decision(),
            tea_protocol::ApprovalDecision::AllowOnce
                | tea_protocol::ApprovalDecision::AllowSession
        );
        let mut facts = vec![SessionRecord::ApprovalResolved {
            approval_id: *resolution.request().approval_id(),
            decision: resolution.decision(),
        }];
        if allowed {
            facts.push(SessionRecord::ToolExecutionStarted {
                tool_call_id: *resolution.request().tool_call_id(),
                execution_target: selected_target(
                    &PolicyDecision::Allow,
                    target,
                    invocation.source().kind(),
                ),
                idempotency: invocation.spec().execution().idempotency(),
            });
        }
        let records = envelopes_at(
            self.ids,
            snapshot.state().session_id(),
            snapshot.state().tail_sequence(),
            snapshot.state().active_branch_id(),
            resolution.decided_at(),
            facts,
        )?;
        let resolution_record_id = records
            .first()
            .map(tea_protocol::RecordEnvelope::record_id)
            .ok_or_else(|| {
                KernelError::new(
                    KernelErrorCode::InvalidState,
                    "approval resolution transaction is empty",
                )
            })?;
        let mut transaction = AppendTransaction::new(
            snapshot.state().session_id(),
            Some(snapshot.state().tail_sequence()),
            records,
        )
        .with_expected_journal_revision(snapshot.journal_revision())
        .with_approval_artifacts([ApprovalArtifactEntry::Resolved {
            record_id: resolution_record_id,
            resolution: resolution.clone(),
        }]);
        if let Some(grant) = resolution.issued_grant() {
            transaction = transaction.with_grant_entries([GrantJournalEntry::Issued {
                approval_record_id: resolution_record_id,
                grant: grant.clone(),
            }]);
        }
        self.sessions
            .append(transaction)
            .await
            .map_err(KernelError::from)?;
        self.sessions
            .load(snapshot.state().session_id())
            .await
            .map_err(KernelError::from)
    }

    async fn commit_denied(
        &self,
        snapshot: SessionSnapshot,
        call: &CompletedToolCall,
        terminal: ToolTerminal,
    ) -> Result<SessionSnapshot, KernelError> {
        self.commit_terminal_with_prefix(
            snapshot,
            call,
            terminal,
            Some(SessionRecord::PolicyDecisionRecorded {
                tool_call_id: call.tool_call_id,
                decision: tea_protocol::PolicyDecision::Deny,
            }),
        )
        .await
    }

    async fn commit_terminal(
        &self,
        snapshot: SessionSnapshot,
        call: &CompletedToolCall,
        terminal: ToolTerminal,
    ) -> Result<SessionSnapshot, KernelError> {
        self.commit_terminal_with_prefix(snapshot, call, terminal, None)
            .await
    }

    async fn commit_terminal_with_prefix(
        &self,
        snapshot: SessionSnapshot,
        call: &CompletedToolCall,
        terminal: ToolTerminal,
        prefix: Option<SessionRecord>,
    ) -> Result<SessionSnapshot, KernelError> {
        let mut records = Vec::with_capacity(3);
        records.extend(prefix);
        let is_error = terminal.failure.is_some();
        records.push(SessionRecord::ToolExecutionFinished {
            tool_call_id: call.tool_call_id,
            is_error,
            content: terminal.content.clone(),
            error: terminal.failure.clone(),
            presentation: terminal.presentation.clone(),
        });
        let message_id = self.ids.next_message_id()?;
        let message = if let Some(failure) = terminal.failure {
            CanonicalMessage::tool_result_failure(
                message_id,
                call.tool_call_id,
                call.tool_name.clone(),
                terminal.content,
                failure,
                self.clock.now()?,
            )?
        } else {
            CanonicalMessage::tool_result_success(
                message_id,
                call.tool_call_id,
                call.tool_name.clone(),
                terminal.content,
                self.clock.now()?,
            )?
        };
        records.push(SessionRecord::MessageCommitted { message });
        self.append_records(&snapshot, records).await
    }

    async fn finish(
        &self,
        snapshot: SessionSnapshot,
        output: ModelTurnOutput,
        run_id: RunId,
        turn_id: TurnId,
        emitter: &mut EventEmitter<'_>,
    ) -> Result<KernelRunOutcome, KernelError> {
        let snapshot = self
            .append_records(
                &snapshot,
                [
                    SessionRecord::MessageCommitted {
                        message: output.message,
                    },
                    SessionRecord::TurnCheckpointed {
                        run_id,
                        turn_id,
                        next_action: NextTurnAction::FinishRun,
                    },
                ],
            )
            .await?;
        emitter
            .emit(Some(run_id), Some(turn_id), AgentEvent::TurnCheckpointed {})
            .await?;
        emitter
            .emit(
                Some(run_id),
                None,
                AgentEvent::RunFinished {
                    status: RunStatus::Completed,
                    usage: output.completion.usage().cloned(),
                    cost: output.completion.cost().cloned(),
                },
            )
            .await?;
        Ok(KernelRunOutcome {
            run_id,
            state: RunState::Completed,
            session: snapshot.state().clone(),
            pending_approval_id: None,
        })
    }

    async fn record_tool_interruption(
        &self,
        snapshot: &SessionSnapshot,
        call: &CompletedToolCall,
        run_id: RunId,
        turn_id: TurnId,
        error: &KernelError,
        emitter: &mut EventEmitter<'_>,
    ) -> Result<(), KernelError> {
        let snapshot = self
            .append_records(
                snapshot,
                [SessionRecord::ToolExecutionInterrupted {
                    tool_call_id: call.tool_call_id,
                    reason: error.message().to_owned(),
                }],
            )
            .await?;
        self.record_run_terminal(&snapshot, run_id, Some(turn_id), error, emitter)
            .await
    }

    async fn record_run_terminal(
        &self,
        snapshot: &SessionSnapshot,
        run_id: RunId,
        turn_id: Option<TurnId>,
        error: &KernelError,
        emitter: &mut EventEmitter<'_>,
    ) -> Result<(), KernelError> {
        let (record, status) = if error.code() == KernelErrorCode::Cancelled {
            (SessionRecord::RunCancelled { run_id }, RunStatus::Cancelled)
        } else {
            let turn_id = turn_id.ok_or_else(|| {
                KernelError::new(
                    KernelErrorCode::InvalidState,
                    "interrupted run has no active turn",
                )
            })?;
            (
                SessionRecord::RunInterrupted {
                    run_id,
                    turn_id,
                    reason: error.message().to_owned(),
                },
                RunStatus::Interrupted,
            )
        };
        self.append_records(snapshot, [record]).await?;
        emitter
            .emit(
                Some(run_id),
                None,
                AgentEvent::RunFinished {
                    status,
                    usage: None,
                    cost: None,
                },
            )
            .await
    }

    async fn append_records(
        &self,
        snapshot: &SessionSnapshot,
        records: impl IntoIterator<Item = SessionRecord>,
    ) -> Result<SessionSnapshot, KernelError> {
        let records = envelopes(
            self.ids,
            self.clock,
            snapshot.state().session_id(),
            snapshot.state().tail_sequence(),
            snapshot.state().active_branch_id(),
            records,
        )?;
        self.sessions
            .append(AppendTransaction::new(
                snapshot.state().session_id(),
                Some(snapshot.state().tail_sequence()),
                records,
            ))
            .await
            .map_err(KernelError::from)?;
        self.sessions
            .load(snapshot.state().session_id())
            .await
            .map_err(KernelError::from)
    }

    async fn append_records_with_metadata(
        &self,
        snapshot: &SessionSnapshot,
        records: impl IntoIterator<Item = (ProtocolMetadata, SessionRecord)>,
    ) -> Result<SessionSnapshot, KernelError> {
        let records = envelopes_with_metadata(
            self.ids,
            self.clock,
            snapshot.state().session_id(),
            snapshot.state().tail_sequence(),
            snapshot.state().active_branch_id(),
            records,
        )?;
        self.sessions
            .append(AppendTransaction::new(
                snapshot.state().session_id(),
                Some(snapshot.state().tail_sequence()),
                records,
            ))
            .await
            .map_err(KernelError::from)?;
        self.sessions
            .load(snapshot.state().session_id())
            .await
            .map_err(KernelError::from)
    }
}

fn validate_new_run_state(snapshot: &SessionSnapshot) -> Result<(), KernelError> {
    if !snapshot.state().pending_approvals().is_empty() {
        return Err(KernelError::new(
            KernelErrorCode::InvalidState,
            "session has a pending approval; resume it explicitly",
        ));
    }
    if snapshot
        .state()
        .tool_calls()
        .values()
        .any(|tool| tool.result_message_id().is_none())
    {
        return Err(KernelError::new(
            KernelErrorCode::InvalidState,
            "session has an incomplete or uncertain tool outcome",
        ));
    }
    Ok(())
}

fn remaining_tool_calls(
    snapshot: &SessionSnapshot,
    completed_call_id: tea_protocol::ToolCallId,
) -> Vec<CompletedToolCall> {
    let mut after_completed = false;
    snapshot
        .records()
        .iter()
        .filter_map(|record| match record.record() {
            SessionRecord::ToolCallRequested { tool_call_id, .. }
                if *tool_call_id == completed_call_id =>
            {
                after_completed = true;
                None
            }
            SessionRecord::ToolCallRequested {
                tool_call_id,
                tool_name,
                arguments,
            } if after_completed
                && snapshot
                    .state()
                    .tool_calls()
                    .get(tool_call_id)
                    .is_some_and(|tool| {
                        tool.policy_decision().is_none()
                            && matches!(
                                tool.execution(),
                                tea_session::ToolExecutionState::NotStarted
                            )
                    }) =>
            {
                Some(CompletedToolCall {
                    tool_call_id: *tool_call_id,
                    tool_name: tool_name.clone(),
                    arguments: arguments.clone(),
                })
            }
            SessionRecord::SessionCreated { .. }
            | SessionRecord::MessageCommitted { .. }
            | SessionRecord::ConfigurationChanged { .. }
            | SessionRecord::ToolCallRequested { .. }
            | SessionRecord::PolicyDecisionRecorded { .. }
            | SessionRecord::ApprovalRequested { .. }
            | SessionRecord::ApprovalResolved { .. }
            | SessionRecord::ToolExecutionStarted { .. }
            | SessionRecord::ToolExecutionFinished { .. }
            | SessionRecord::ToolExecutionInterrupted { .. }
            | SessionRecord::RunInterrupted { .. }
            | SessionRecord::RunCancelled { .. }
            | SessionRecord::BranchCreated { .. }
            | SessionRecord::ActiveBranchChanged { .. }
            | SessionRecord::SessionCompacted { .. }
            | SessionRecord::TurnCheckpointed { .. } => None,
        })
        .collect()
}

fn completed_tool_iterations(snapshot: &SessionSnapshot, run_id: RunId) -> u32 {
    snapshot
        .records()
        .iter()
        .filter_map(|record| match record.record() {
            SessionRecord::TurnCheckpointed {
                run_id: checkpoint_run,
                turn_id,
                ..
            } if *checkpoint_run == run_id => Some(*turn_id),
            _ => None,
        })
        .collect::<BTreeSet<_>>()
        .len()
        .try_into()
        .unwrap_or(u32::MAX)
}

fn committed_assistant_output_bytes(
    snapshot: &SessionSnapshot,
    run_id: RunId,
) -> Result<usize, KernelError> {
    let mut segment_bytes = 0usize;
    let mut run_bytes = 0usize;
    for record in snapshot.records() {
        match record.record() {
            SessionRecord::MessageCommitted {
                message: CanonicalMessage::Assistant { content, .. },
            } => segment_bytes = checked_content_bytes(segment_bytes, content)?,
            SessionRecord::TurnCheckpointed {
                run_id: checkpoint_run,
                ..
            } => {
                if *checkpoint_run == run_id {
                    run_bytes = run_bytes
                        .checked_add(segment_bytes)
                        .ok_or_else(byte_count_error)?;
                }
                segment_bytes = 0;
            }
            _ => {}
        }
    }
    Ok(run_bytes)
}

fn checked_content_bytes(initial: usize, content: &[ContentBlock]) -> Result<usize, KernelError> {
    content.iter().try_fold(initial, |total, block| {
        let bytes = match block {
            ContentBlock::Text { text }
            | ContentBlock::ContextualText { text }
            | ContentBlock::Thinking { text } => text.len(),
            ContentBlock::Image { .. } | ContentBlock::ToolCall { .. } => 0,
            ContentBlock::HostedTool { .. } | ContentBlock::Citation { .. } => {
                serde_json::to_vec(block)
                    .map_err(|_| byte_count_error())?
                    .len()
            }
        };
        total.checked_add(bytes).ok_or_else(byte_count_error)
    })
}

fn byte_count_error() -> KernelError {
    KernelError::new(
        KernelErrorCode::LimitExceeded,
        "assistant output byte count cannot be reconstructed",
    )
}

fn persisted_request(
    snapshot: &SessionSnapshot,
    approval_id: ApprovalId,
) -> Result<&ApprovalRequest, KernelError> {
    snapshot
        .approval_artifacts()
        .iter()
        .find_map(|entry| match entry {
            ApprovalArtifactEntry::Requested { request, .. }
                if request.approval_id() == &approval_id =>
            {
                Some(request)
            }
            ApprovalArtifactEntry::Requested { .. } | ApprovalArtifactEntry::Resolved { .. } => {
                None
            }
        })
        .ok_or_else(|| {
            KernelError::new(
                KernelErrorCode::PolicyFailure,
                "persisted approval request is missing",
            )
        })
}

impl From<tea_session::SessionStoreError> for KernelError {
    fn from(error: tea_session::SessionStoreError) -> Self {
        Self::new(KernelErrorCode::SessionFailure, error.to_string())
    }
}
