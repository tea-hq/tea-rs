use std::collections::BTreeMap;

use serde_json::Value;
use tea_protocol::{
    ApprovalDecision, ApprovalId, BranchId, CanonicalMessage, ContentBlock, ExecutionTarget,
    MessageId, ModelRef, PolicyDecision, ProfileId, ProtocolMetadata, ProtocolTimestamp,
    ReasoningEffort, RecordId, RunId, SessionId, SessionSequence, ToolCallId, ToolFailure,
    ToolIdempotency, ToolPresentation, TurnId,
};

/// Active model and product-profile configuration derived from durable records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionConfiguration {
    model: Option<ModelRef>,
    profile_id: ProfileId,
    reasoning_effort: Option<ReasoningEffort>,
}

impl SessionConfiguration {
    /// Returns the selected model, when one has been configured.
    #[must_use]
    pub const fn model_ref(&self) -> Option<&ModelRef> {
        self.model.as_ref()
    }

    /// Returns the provider-local model selector, when one is configured.
    #[must_use]
    pub const fn model_id(&self) -> Option<&tea_protocol::ModelId> {
        match &self.model {
            Some(model) => Some(model.model_id()),
            None => None,
        }
    }

    /// Returns the selected provider, when one is configured.
    #[must_use]
    pub const fn provider_id(&self) -> Option<&tea_protocol::ProviderId> {
        match &self.model {
            Some(model) => Some(model.provider_id()),
            None => None,
        }
    }

    /// Returns the selected product profile.
    #[must_use]
    pub const fn profile_id(&self) -> &ProfileId {
        &self.profile_id
    }

    /// Returns the explicit session reasoning effort, when configured.
    #[must_use]
    pub const fn reasoning_effort(&self) -> Option<ReasoningEffort> {
        self.reasoning_effort
    }
}

/// Durable pending approval reconstructed from canonical records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PendingApproval {
    approval_id: ApprovalId,
    tool_call_id: ToolCallId,
    expires_at: ProtocolTimestamp,
    requested_at: ProtocolTimestamp,
}

impl PendingApproval {
    /// Returns the approval identity.
    #[must_use]
    pub const fn approval_id(&self) -> ApprovalId {
        self.approval_id
    }

    /// Returns the tool call awaiting a decision.
    #[must_use]
    pub const fn tool_call_id(&self) -> ToolCallId {
        self.tool_call_id
    }

    /// Returns the caller-clock expiry boundary.
    #[must_use]
    pub const fn expires_at(&self) -> ProtocolTimestamp {
        self.expires_at
    }

    /// Returns when the canonical request became durable.
    #[must_use]
    pub const fn requested_at(&self) -> ProtocolTimestamp {
        self.requested_at
    }
}

/// Durable execution/recovery state for a tool call.
#[derive(Debug, Clone, PartialEq)]
pub enum ToolExecutionState {
    /// The tool call exists but execution has not started.
    NotStarted,
    /// Execution crossed its durable start boundary.
    Started {
        /// Selected execution boundary.
        execution_target: ExecutionTarget,
        /// Recovery semantics declared at start.
        idempotency: ToolIdempotency,
    },
    /// Execution reached a durable terminal result.
    Finished {
        /// Whether the terminal result is an error.
        is_error: bool,
        /// Canonical terminal content committed before a tool-result message.
        content: Vec<ContentBlock>,
        /// Machine-readable failure when terminal result is an error.
        error: Option<ToolFailure>,
        /// Optional UI-only presentation retained out of model context.
        presentation: Option<ToolPresentation>,
    },
    /// Execution was interrupted after start and has uncertain outcome.
    Interrupted {
        /// English technical recovery diagnostic.
        reason: String,
        /// Selected execution boundary.
        execution_target: ExecutionTarget,
        /// Recovery semantics declared at start.
        idempotency: ToolIdempotency,
    },
}

/// Materialized lifecycle of one requested tool call.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCallState {
    tool_call_id: ToolCallId,
    tool_name: String,
    arguments: Value,
    policy_decision: Option<PolicyDecision>,
    approval_id: Option<ApprovalId>,
    approval_decision: Option<ApprovalDecision>,
    execution: ToolExecutionState,
    result_message_id: Option<MessageId>,
}

impl ToolCallState {
    /// Returns the stable tool-call identity.
    #[must_use]
    pub const fn tool_call_id(&self) -> ToolCallId {
        self.tool_call_id
    }

    /// Returns the registered tool name.
    #[must_use]
    pub fn tool_name(&self) -> &str {
        &self.tool_name
    }

    /// Returns validated provider-neutral arguments.
    #[must_use]
    pub const fn arguments(&self) -> &Value {
        &self.arguments
    }

    /// Returns the durable policy decision, when evaluated.
    #[must_use]
    pub const fn policy_decision(&self) -> Option<PolicyDecision> {
        self.policy_decision
    }

    /// Returns the associated approval identity, when requested.
    #[must_use]
    pub const fn approval_id(&self) -> Option<ApprovalId> {
        self.approval_id
    }

    /// Returns the terminal approval decision, when resolved.
    #[must_use]
    pub const fn approval_decision(&self) -> Option<ApprovalDecision> {
        self.approval_decision
    }

    /// Returns durable execution/recovery state.
    #[must_use]
    pub const fn execution(&self) -> &ToolExecutionState {
        &self.execution
    }

    /// Returns the sole committed result message, when present.
    #[must_use]
    pub const fn result_message_id(&self) -> Option<MessageId> {
        self.result_message_id
    }
}

/// Provider-run state required for restart diagnostics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunRecoveryState {
    /// Provider streaming was interrupted before durable terminal output.
    Interrupted {
        /// Active turn at interruption.
        turn_id: TurnId,
        /// English technical diagnostic.
        reason: String,
    },
    /// The run was explicitly cancelled.
    Cancelled,
}

/// Durable turn boundary reconstructed during replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TurnCheckpoint {
    run_id: RunId,
    turn_id: TurnId,
    record_id: RecordId,
    sequence: SessionSequence,
    next_action: tea_protocol::NextTurnAction,
}

impl TurnCheckpoint {
    pub(crate) const fn new(
        run_id: RunId,
        turn_id: TurnId,
        record_id: RecordId,
        sequence: SessionSequence,
        next_action: tea_protocol::NextTurnAction,
    ) -> Self {
        Self {
            run_id,
            turn_id,
            record_id,
            sequence,
            next_action,
        }
    }

    /// Returns the checkpointed run.
    #[must_use]
    pub const fn run_id(&self) -> RunId {
        self.run_id
    }

    /// Returns the checkpointed turn.
    #[must_use]
    pub const fn turn_id(&self) -> TurnId {
        self.turn_id
    }

    /// Returns the record establishing this checkpoint.
    #[must_use]
    pub const fn record_id(&self) -> RecordId {
        self.record_id
    }

    /// Returns the authoritative sequence.
    #[must_use]
    pub const fn sequence(&self) -> SessionSequence {
        self.sequence
    }

    /// Returns the next action allowed by the checkpoint.
    #[must_use]
    pub const fn next_action(&self) -> tea_protocol::NextTurnAction {
        self.next_action
    }
}

/// Latest durable compaction summary and its source provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionCompaction {
    summary: CanonicalMessage,
    compacted_through_record_id: RecordId,
}

impl SessionCompaction {
    /// Returns the canonical summary message used for future model context.
    #[must_use]
    pub const fn summary(&self) -> &CanonicalMessage {
        &self.summary
    }

    /// Returns the last source record replaced in model context.
    #[must_use]
    pub const fn compacted_through_record_id(&self) -> RecordId {
        self.compacted_through_record_id
    }
}

/// Summary of one durable branch.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::struct_field_names)] // Explicit `_id` fields retain strong identifier semantics.
pub struct BranchSummary {
    branch_id: BranchId,
    source_branch_id: Option<BranchId>,
    from_record_id: RecordId,
    leaf_record_id: RecordId,
}

impl BranchSummary {
    pub(crate) const fn new(
        branch_id: BranchId,
        source_branch_id: Option<BranchId>,
        from_record_id: RecordId,
        leaf_record_id: RecordId,
    ) -> Self {
        Self {
            branch_id,
            source_branch_id,
            from_record_id,
            leaf_record_id,
        }
    }

    pub(crate) fn set_leaf(&mut self, record_id: RecordId) {
        self.leaf_record_id = record_id;
    }

    /// Returns branch identity.
    #[must_use]
    pub const fn branch_id(&self) -> BranchId {
        self.branch_id
    }

    /// Returns source branch for a fork, or `None` for the root branch.
    #[must_use]
    pub const fn source_branch_id(&self) -> Option<BranchId> {
        self.source_branch_id
    }

    /// Returns the durable source position used to create this branch.
    #[must_use]
    pub const fn from_record_id(&self) -> RecordId {
        self.from_record_id
    }

    /// Returns the current durable leaf record.
    #[must_use]
    pub const fn leaf_record_id(&self) -> RecordId {
        self.leaf_record_id
    }
}

/// Deterministic session projection derived only from the append-only record log.
#[derive(Debug, Clone, PartialEq)]
pub struct MaterializedSessionState {
    pub(crate) session_id: SessionId,
    pub(crate) tail_sequence: SessionSequence,
    pub(crate) tail_record_id: RecordId,
    pub(crate) metadata: ProtocolMetadata,
    pub(crate) configuration: SessionConfiguration,
    pub(crate) messages: Vec<CanonicalMessage>,
    pub(crate) pending_approvals: BTreeMap<ApprovalId, PendingApproval>,
    pub(crate) tool_calls: BTreeMap<ToolCallId, ToolCallState>,
    pub(crate) active_branch_id: Option<BranchId>,
    pub(crate) branches: BTreeMap<BranchId, BranchSummary>,
    pub(crate) run_recovery: BTreeMap<RunId, RunRecoveryState>,
    pub(crate) latest_checkpoint: Option<TurnCheckpoint>,
    pub(crate) latest_compaction: Option<SessionCompaction>,
}

impl MaterializedSessionState {
    /// Returns session identity.
    #[must_use]
    pub const fn session_id(&self) -> SessionId {
        self.session_id
    }

    /// Returns the current authoritative tail sequence.
    #[must_use]
    pub const fn tail_sequence(&self) -> SessionSequence {
        self.tail_sequence
    }

    /// Returns the current tail record identity.
    #[must_use]
    pub const fn tail_record_id(&self) -> RecordId {
        self.tail_record_id
    }

    /// Returns bounded creation metadata.
    #[must_use]
    pub const fn metadata(&self) -> &ProtocolMetadata {
        &self.metadata
    }

    /// Returns active configuration.
    #[must_use]
    pub const fn configuration(&self) -> &SessionConfiguration {
        &self.configuration
    }

    /// Returns the active durable transcript.
    #[must_use]
    pub fn messages(&self) -> &[CanonicalMessage] {
        &self.messages
    }

    /// Returns pending approvals in stable ID order.
    #[must_use]
    pub const fn pending_approvals(&self) -> &BTreeMap<ApprovalId, PendingApproval> {
        &self.pending_approvals
    }

    /// Returns tool-call lifecycle projections in stable ID order.
    #[must_use]
    pub const fn tool_calls(&self) -> &BTreeMap<ToolCallId, ToolCallState> {
        &self.tool_calls
    }

    /// Returns active branch, or `None` for a legacy unbranched Protocol 1.0 log.
    #[must_use]
    pub const fn active_branch_id(&self) -> Option<BranchId> {
        self.active_branch_id
    }

    /// Returns durable branch summaries in stable ID order.
    #[must_use]
    pub const fn branches(&self) -> &BTreeMap<BranchId, BranchSummary> {
        &self.branches
    }

    /// Returns provider-run restart diagnostics.
    #[must_use]
    pub const fn run_recovery(&self) -> &BTreeMap<RunId, RunRecoveryState> {
        &self.run_recovery
    }

    /// Returns the latest durable turn checkpoint.
    #[must_use]
    pub const fn latest_checkpoint(&self) -> Option<&TurnCheckpoint> {
        self.latest_checkpoint.as_ref()
    }

    /// Returns the latest compaction summary without removing original messages.
    #[must_use]
    pub const fn latest_compaction(&self) -> Option<&SessionCompaction> {
        self.latest_compaction.as_ref()
    }
}

pub(crate) fn message_id(message: &CanonicalMessage) -> MessageId {
    match message {
        CanonicalMessage::User { id, .. }
        | CanonicalMessage::Assistant { id, .. }
        | CanonicalMessage::ToolResult { id, .. } => *id,
    }
}

pub(crate) fn declared_tool_calls(
    message: &CanonicalMessage,
) -> impl Iterator<Item = (ToolCallId, &str, &Value)> {
    let content = match message {
        CanonicalMessage::Assistant { content, .. } => Some(content.as_slice()),
        CanonicalMessage::User { .. } | CanonicalMessage::ToolResult { .. } => None,
    };
    content
        .into_iter()
        .flatten()
        .filter_map(|block| match block {
            ContentBlock::ToolCall {
                tool_call_id,
                tool_name,
                arguments,
                ..
            } => Some((*tool_call_id, tool_name.as_str(), arguments)),
            ContentBlock::Text { .. }
            | ContentBlock::ContextualText { .. }
            | ContentBlock::Thinking { .. }
            | ContentBlock::Image { .. }
            | ContentBlock::HostedTool { .. }
            | ContentBlock::Citation { .. } => None,
        })
}

pub(crate) fn new_state(
    record_id: RecordId,
    session_id: SessionId,
    sequence: SessionSequence,
    profile_id: ProfileId,
    metadata: ProtocolMetadata,
    root_branch_id: Option<BranchId>,
) -> MaterializedSessionState {
    let branches = root_branch_id.map_or_else(BTreeMap::new, |branch_id| {
        BTreeMap::from([(
            branch_id,
            BranchSummary {
                branch_id,
                source_branch_id: None,
                from_record_id: record_id,
                leaf_record_id: record_id,
            },
        )])
    });
    MaterializedSessionState {
        session_id,
        tail_sequence: sequence,
        tail_record_id: record_id,
        metadata,
        configuration: SessionConfiguration {
            model: None,
            profile_id,
            reasoning_effort: None,
        },
        messages: Vec::new(),
        pending_approvals: BTreeMap::new(),
        tool_calls: BTreeMap::new(),
        active_branch_id: root_branch_id,
        branches,
        run_recovery: BTreeMap::new(),
        latest_checkpoint: None,
        latest_compaction: None,
    }
}

pub(crate) fn new_pending_approval(
    approval_id: ApprovalId,
    tool_call_id: ToolCallId,
    expires_at: ProtocolTimestamp,
    requested_at: ProtocolTimestamp,
) -> PendingApproval {
    PendingApproval {
        approval_id,
        tool_call_id,
        expires_at,
        requested_at,
    }
}

pub(crate) fn new_tool_call(
    tool_call_id: ToolCallId,
    tool_name: String,
    arguments: Value,
) -> ToolCallState {
    ToolCallState {
        tool_call_id,
        tool_name,
        arguments,
        policy_decision: None,
        approval_id: None,
        approval_decision: None,
        execution: ToolExecutionState::NotStarted,
        result_message_id: None,
    }
}

pub(crate) fn set_model(configuration: &mut SessionConfiguration, model: ModelRef) {
    configuration.model = Some(model);
}

pub(crate) fn set_profile(configuration: &mut SessionConfiguration, profile_id: ProfileId) {
    configuration.profile_id = profile_id;
}

pub(crate) fn set_reasoning_effort(
    configuration: &mut SessionConfiguration,
    reasoning_effort: ReasoningEffort,
) {
    configuration.reasoning_effort = Some(reasoning_effort);
}

pub(crate) fn set_policy(tool: &mut ToolCallState, decision: PolicyDecision) {
    tool.policy_decision = Some(decision);
}

pub(crate) fn set_approval(tool: &mut ToolCallState, approval_id: ApprovalId) {
    tool.approval_id = Some(approval_id);
}

pub(crate) fn resolve_approval(tool: &mut ToolCallState, decision: ApprovalDecision) {
    tool.approval_decision = Some(decision);
}

pub(crate) fn start_tool(
    tool: &mut ToolCallState,
    execution_target: ExecutionTarget,
    idempotency: ToolIdempotency,
) {
    tool.execution = ToolExecutionState::Started {
        execution_target,
        idempotency,
    };
}

pub(crate) fn finish_tool(
    tool: &mut ToolCallState,
    is_error: bool,
    content: Vec<ContentBlock>,
    error: Option<ToolFailure>,
    presentation: Option<ToolPresentation>,
) {
    tool.execution = ToolExecutionState::Finished {
        is_error,
        content,
        error,
        presentation,
    };
}

pub(crate) fn set_compaction(
    state: &mut MaterializedSessionState,
    summary: CanonicalMessage,
    compacted_through_record_id: RecordId,
) {
    // Replace the compacted prefix with the summary message so the model-visible
    // transcript begins with the summary followed by later uncompacted records.
    // Original records remain in the durable log for audit and replay.
    state.messages.clear();
    state.messages.push(summary.clone());
    state.latest_compaction = Some(SessionCompaction {
        summary,
        compacted_through_record_id,
    });
}

pub(crate) fn commit_tool_result(tool: &mut ToolCallState, message_id: MessageId) {
    tool.result_message_id = Some(message_id);
}

pub(crate) fn interrupt_tool(tool: &mut ToolCallState, reason: String) {
    if let ToolExecutionState::Started {
        execution_target,
        idempotency,
    } = tool.execution
    {
        tool.execution = ToolExecutionState::Interrupted {
            reason,
            execution_target,
            idempotency,
        };
    }
}
