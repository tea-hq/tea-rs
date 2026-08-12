use std::collections::{BTreeMap, BTreeSet, VecDeque};

use tea_protocol::{
    ApprovalDecision, BranchId, CanonicalMessage, ContentBlock, ExactCost, HostedToolOutcome,
    MessageId, ModelId, ModelRef, ProfileId, ReasoningEffort, RunStatus, SessionId,
    SessionSequence, ToolCallId, ToolPresentation, Usage,
};
use tea_session::{ApprovalArtifactEntry, SessionSnapshot, ToolExecutionState};

use super::Overlay;
use super::attachment::{AttachmentError, ComposerAttachment, validate_addition};
use super::selectors::Selector;

/// Maximum local notifications retained by the terminal projection.
pub const MAX_NOTIFICATIONS: usize = 32;
/// Maximum configured MCP server rows retained in the terminal projection.
pub const MAX_MCP_HEALTH_ROWS: usize = 64;
/// Maximum discovered skill rows retained in the terminal projection.
pub const MAX_SKILL_CATALOG_ROWS: usize = 128;
/// Maximum steering or follow-up entries retained for rendering.
pub const MAX_VISIBLE_QUEUE_ITEMS: usize = 64;
/// Maximum event identities retained for duplicate observational-event detection.
pub const MAX_OBSERVED_EVENT_IDS: usize = 1024;
/// Maximum incomplete streamed messages retained in arrival order.
pub const MAX_ACTIVE_STREAM_MESSAGES: usize = 64;
/// Maximum incomplete or uncommitted tool activities retained in arrival order.
pub const MAX_ACTIVE_TOOL_ITEMS: usize = 128;

/// Immutable startup context counts shown above the transcript.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StartupContext {
    workspace: String,
    context_documents: usize,
    skills: usize,
    prompt_templates: usize,
    diagnostics: usize,
}

impl StartupContext {
    /// Creates a bounded, path-safe startup summary.
    #[must_use]
    pub fn new(
        workspace: impl Into<String>,
        context_documents: usize,
        skills: usize,
        prompt_templates: usize,
        diagnostics: usize,
    ) -> Self {
        Self {
            workspace: sanitize_summary(&workspace.into()),
            context_documents,
            skills,
            prompt_templates,
            diagnostics,
        }
    }

    /// Returns the safe workspace label.
    #[must_use]
    pub fn workspace(&self) -> &str {
        &self.workspace
    }

    /// Returns loaded context-document count.
    #[must_use]
    pub const fn context_documents(&self) -> usize {
        self.context_documents
    }

    /// Returns discovered skill count.
    #[must_use]
    pub const fn skills(&self) -> usize {
        self.skills
    }

    /// Returns discovered prompt-template count.
    #[must_use]
    pub const fn prompt_templates(&self) -> usize {
        self.prompt_templates
    }

    /// Returns safe resource diagnostic count.
    #[must_use]
    pub const fn diagnostics(&self) -> usize {
        self.diagnostics
    }
}

/// Local-only view preferences; toggling them never changes durable state.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ViewPreferences {
    /// Whether all reasoning blocks are collapsed.
    pub thinking_collapsed: bool,
    /// Tool calls whose arguments/output were explicitly expanded.
    pub expanded_tools: BTreeSet<ToolCallId>,
}

/// Local transcript navigation state. It is never persisted in session records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptViewport {
    offset_from_tail_rows: usize,
    follow_tail: bool,
    unread_items: usize,
}

impl Default for TranscriptViewport {
    fn default() -> Self {
        Self {
            offset_from_tail_rows: 0,
            follow_tail: true,
            unread_items: 0,
        }
    }
}

impl TranscriptViewport {
    /// Returns whether incoming transcript content is followed immediately.
    #[must_use]
    pub const fn follows_tail(&self) -> bool {
        self.follow_tail
    }

    /// Returns the current line offset from the latest transcript content.
    #[must_use]
    pub const fn offset_from_tail_rows(&self) -> usize {
        self.offset_from_tail_rows
    }

    /// Returns the number of incoming items received while browsing history.
    #[must_use]
    pub const fn unread_items(&self) -> usize {
        self.unread_items
    }

    pub(crate) fn scroll_up(&mut self, rows: usize) {
        self.follow_tail = false;
        self.offset_from_tail_rows = self.offset_from_tail_rows.saturating_add(rows.max(1));
    }

    pub(crate) fn scroll_down(&mut self, rows: usize) {
        self.offset_from_tail_rows = self.offset_from_tail_rows.saturating_sub(rows.max(1));
        if self.offset_from_tail_rows == 0 {
            self.follow_tail = true;
            self.unread_items = 0;
        }
    }

    pub(crate) fn follow_tail(&mut self) {
        self.offset_from_tail_rows = 0;
        self.follow_tail = true;
        self.unread_items = 0;
    }

    pub(crate) fn note_new_item(&mut self) {
        if !self.follow_tail {
            self.unread_items = self.unread_items.saturating_add(1);
            self.offset_from_tail_rows = self.offset_from_tail_rows.saturating_add(1);
        }
    }
}

/// One streamed content block not yet guaranteed durable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StreamingBlock {
    /// Whether this is hidden reasoning rather than visible assistant text.
    pub thinking: bool,
    /// Accumulated bounded deltas.
    pub text: String,
    /// Content generation used by render caches.
    pub generation: u64,
}

/// Ephemeral streamed assistant message assembled by content index.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StreamingMessage {
    /// Ordered partial blocks.
    pub blocks: BTreeMap<u32, StreamingBlock>,
}

/// Latest transient progress for one tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolProgressView {
    /// Bounded technical progress text.
    pub message: String,
    /// Completed work units.
    pub completed_units: u64,
    /// Total units when known.
    pub total_units: Option<u64>,
}

/// Durable tool state plus optional transient progress.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolView {
    /// Stable tool-call identity.
    pub tool_call_id: ToolCallId,
    /// Registered tool name.
    pub tool_name: String,
    /// Validated provider-neutral arguments.
    pub arguments: serde_json::Value,
    /// Human-readable durable execution state.
    pub status: String,
    /// Durable approval decision when the canonical session retained one.
    pub approval_decision: Option<ApprovalDecision>,
    /// Latest non-durable progress.
    pub progress: Option<ToolProgressView>,
    /// Optional durable UI presentation kept out of model context.
    pub presentation: Option<ToolPresentation>,
    /// Optional non-durable preview shown while this call awaits approval.
    pub preview: Option<ToolPresentation>,
}

/// Ephemeral projection of provider-hosted work before its assistant message is durable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostedToolView {
    /// Kernel-owned hosted activity identifier.
    pub tool_call_id: ToolCallId,
    /// Stable hosted tool name.
    pub tool_name: String,
    /// Arguments available after the provider completes the activity.
    pub arguments: Option<serde_json::Value>,
    /// Terminal outcome available after completion.
    pub outcome: Option<HostedToolOutcome>,
    /// Number of normalized durable sources reported at completion.
    pub source_count: Option<u32>,
}

/// Ephemeral projection of one scheduled model retry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelRetryView {
    /// Ephemeral assistant message reused by the retry.
    pub message_id: MessageId,
    /// One-based retry number.
    pub attempt: u32,
    /// Maximum retry count, excluding the initial request.
    pub max_retries: u32,
    delay_ms: u64,
    elapsed_seconds: u64,
}

impl ModelRetryView {
    pub(crate) const fn new(
        message_id: MessageId,
        attempt: u32,
        max_retries: u32,
        delay_ms: u64,
    ) -> Self {
        Self {
            message_id,
            attempt,
            max_retries,
            delay_ms,
            elapsed_seconds: 0,
        }
    }

    /// Returns the rounded-up whole seconds remaining before the retry.
    #[must_use]
    pub const fn remaining_seconds(&self) -> u64 {
        let elapsed_ms = self.elapsed_seconds.saturating_mul(1_000);
        self.delay_ms.saturating_sub(elapsed_ms).saturating_add(999) / 1_000
    }

    pub(crate) fn advance(&mut self, seconds: u64) {
        self.elapsed_seconds = self.elapsed_seconds.saturating_add(seconds);
    }
}

/// Locally visible prompt accepted by the service but not yet durable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PendingUserPrompt {
    text: String,
    durable_message_count: usize,
}

impl PendingUserPrompt {
    pub(crate) fn text(&self) -> &str {
        &self.text
    }
}

/// Persisted redacted approval projection suitable for terminal display.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ApprovalView {
    /// Stable approval identity.
    pub approval_id: tea_protocol::ApprovalId,
    /// Stable tool-call identity.
    pub tool_call_id: ToolCallId,
    /// Registered tool name.
    pub tool_name: String,
    /// Declared effect names.
    pub effects: Vec<String>,
    /// Persisted policy execution target.
    pub target: String,
    /// Redacted resource locators.
    pub resources: Vec<String>,
    /// Technical reason from the persisted presentation.
    pub reason: String,
    /// Persisted redacted argument JSON.
    pub arguments: String,
    /// Caller-clock expiry.
    pub expires_at: tea_protocol::ProtocolTimestamp,
}

/// One canonical choice supported by the durable approval contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ApprovalChoice {
    /// Authorize only the pending tool call.
    #[default]
    AllowOnce,
    /// Authorize matching resources for this session.
    AllowSession,
    /// Deny the pending tool call.
    Deny,
}

impl ApprovalChoice {
    /// Returns the protocol decision represented by this choice.
    #[must_use]
    pub const fn decision(self) -> tea_protocol::ApprovalDecision {
        match self {
            Self::AllowOnce => tea_protocol::ApprovalDecision::AllowOnce,
            Self::AllowSession => tea_protocol::ApprovalDecision::AllowSession,
            Self::Deny => tea_protocol::ApprovalDecision::Deny,
        }
    }

    /// Returns the next choice in display order.
    #[must_use]
    pub const fn next(self) -> Self {
        match self {
            Self::AllowOnce => Self::AllowSession,
            Self::AllowSession => Self::Deny,
            Self::Deny => Self::AllowOnce,
        }
    }

    /// Returns the previous choice in display order.
    #[must_use]
    pub const fn previous(self) -> Self {
        match self {
            Self::AllowOnce => Self::Deny,
            Self::AllowSession => Self::AllowOnce,
            Self::Deny => Self::AllowSession,
        }
    }

    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::AllowOnce => "allow once",
            Self::AllowSession => "allow for session",
            Self::Deny => "deny",
        }
    }
}

/// Pure terminal projection of canonical durable state and bounded ephemeral observations.
#[derive(Debug, Clone, PartialEq)]
pub struct TuiState {
    pub(crate) session_id: SessionId,
    pub(crate) profile_id: ProfileId,
    pub(crate) model_ref: Option<ModelRef>,
    pub(crate) reasoning_effort: Option<ReasoningEffort>,
    pub(crate) model_default_reasoning_effort: Option<ReasoningEffort>,
    pub(crate) pending_reasoning_effort: Option<ReasoningEffort>,
    pub(crate) active_branch_id: Option<BranchId>,
    pub(crate) durable_tail: SessionSequence,
    pub(crate) event_cursor: SessionSequence,
    pub(crate) messages: Vec<CanonicalMessage>,
    pub(crate) pending_user_prompt: Option<PendingUserPrompt>,
    pub(crate) tools: BTreeMap<ToolCallId, ToolView>,
    pub(crate) hosted_tools: BTreeMap<ToolCallId, HostedToolView>,
    pub(crate) approval: Option<ApprovalView>,
    pub(crate) approval_choice: ApprovalChoice,
    pub(crate) approval_submitting: bool,
    pub(crate) streaming: BTreeMap<MessageId, StreamingMessage>,
    /// Arrival order for incomplete streamed messages. This is local UI state;
    /// canonical snapshots remain the source of durable transcript ordering.
    pub(crate) streaming_order: VecDeque<MessageId>,
    /// Event/snapshot order for tool activities not yet represented by a
    /// committed transcript cell.
    pub(crate) tool_order: VecDeque<ToolCallId>,
    pub(crate) hosted_tool_order: VecDeque<ToolCallId>,
    pub(crate) model_retry: Option<ModelRetryView>,
    pub(crate) observed_event_ids: VecDeque<tea_protocol::EventId>,
    pub(crate) running: bool,
    /// Locally measured whole seconds since the active request was accepted.
    pub(crate) run_elapsed_seconds: u64,
    pub(crate) observation_run_id: Option<tea_protocol::RunId>,
    pub(crate) run_status: Option<RunStatus>,
    pub(crate) usage: Option<Usage>,
    pub(crate) cost: Option<ExactCost>,
    pub(crate) steering_queue: VecDeque<String>,
    pub(crate) follow_up_queue: VecDeque<String>,
    pub(crate) attachments: Vec<ComposerAttachment>,
    pub(crate) notifications: VecDeque<String>,
    pub(crate) mcp_health: Vec<String>,
    pub(crate) skill_catalog: Vec<String>,
    pub(crate) editor: String,
    pub(crate) editor_skill_mention: Option<String>,
    pub(crate) overlay: Option<Overlay>,
    pub(crate) viewport_width: u16,
    pub(crate) viewport_height: u16,
    pub(crate) transcript_viewport: TranscriptViewport,
    pub(crate) preferences: ViewPreferences,
    pub(crate) startup: StartupContext,
    pub(crate) generation: u64,
    pub(crate) resyncing: bool,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct SessionEphemera {
    steering_queue: VecDeque<String>,
    follow_up_queue: VecDeque<String>,
    attachments: Vec<ComposerAttachment>,
    pending_reasoning_effort: Option<ReasoningEffort>,
}

impl TuiState {
    /// Rebuilds a projection from one immutable canonical snapshot.
    #[must_use]
    pub fn from_snapshot(snapshot: &SessionSnapshot, startup: StartupContext) -> Self {
        let state = snapshot.state();
        let tools = durable_tools(snapshot);
        let tool_order = tool_order(state.messages(), &tools);
        let approval = pending_approval(snapshot);
        Self {
            session_id: state.session_id(),
            profile_id: state.configuration().profile_id().clone(),
            model_ref: state.configuration().model_ref().cloned(),
            reasoning_effort: state.configuration().reasoning_effort(),
            model_default_reasoning_effort: None,
            pending_reasoning_effort: None,
            active_branch_id: state.active_branch_id(),
            durable_tail: state.tail_sequence(),
            event_cursor: state.tail_sequence(),
            messages: state.messages().to_vec(),
            pending_user_prompt: None,
            tools,
            hosted_tools: BTreeMap::new(),
            approval,
            approval_choice: ApprovalChoice::default(),
            approval_submitting: false,
            streaming: BTreeMap::new(),
            streaming_order: VecDeque::new(),
            tool_order,
            hosted_tool_order: VecDeque::new(),
            model_retry: None,
            observed_event_ids: VecDeque::new(),
            running: false,
            run_elapsed_seconds: 0,
            observation_run_id: None,
            run_status: None,
            usage: None,
            cost: None,
            steering_queue: VecDeque::new(),
            follow_up_queue: VecDeque::new(),
            attachments: Vec::new(),
            notifications: VecDeque::new(),
            mcp_health: Vec::new(),
            skill_catalog: Vec::new(),
            editor: String::new(),
            editor_skill_mention: None,
            overlay: None,
            viewport_width: 80,
            viewport_height: 24,
            transcript_viewport: TranscriptViewport::default(),
            preferences: ViewPreferences::default(),
            startup,
            generation: 0,
            resyncing: false,
        }
    }

    /// Returns session identity.
    #[must_use]
    pub const fn session_id(&self) -> SessionId {
        self.session_id
    }

    /// Returns the active durable model selection.
    #[must_use]
    pub const fn model_id(&self) -> Option<&ModelId> {
        match &self.model_ref {
            Some(model) => Some(model.model_id()),
            None => None,
        }
    }

    /// Returns the active durable provider-qualified model selection.
    #[must_use]
    pub const fn model_ref(&self) -> Option<&ModelRef> {
        self.model_ref.as_ref()
    }

    /// Returns the durable session reasoning selection, when explicit.
    #[must_use]
    pub const fn reasoning_effort(&self) -> Option<ReasoningEffort> {
        self.reasoning_effort
    }

    /// Returns a queued next-turn reasoning selection, when present.
    #[must_use]
    pub const fn pending_reasoning_effort(&self) -> Option<ReasoningEffort> {
        self.pending_reasoning_effort
    }

    /// Returns the effective durable or pending effort shown by the TUI.
    #[must_use]
    pub fn displayed_reasoning_effort(&self) -> &'static str {
        self.pending_reasoning_effort
            .or(self.reasoning_effort)
            .or(self.model_default_reasoning_effort)
            .map_or("default", ReasoningEffort::as_str)
    }

    /// Returns the current durable transcript.
    #[must_use]
    pub fn messages(&self) -> &[CanonicalMessage] {
        &self.messages
    }

    /// Returns provider-neutral usage from the latest observed terminal run event.
    #[must_use]
    pub const fn usage(&self) -> Option<&Usage> {
        self.usage.as_ref()
    }

    /// Returns the current event cursor.
    #[must_use]
    pub const fn event_cursor(&self) -> SessionSequence {
        self.event_cursor
    }

    /// Returns whether a snapshot reload is pending after reconnect or a gap.
    #[must_use]
    pub const fn is_resyncing(&self) -> bool {
        self.resyncing
    }

    /// Returns current render generation.
    #[must_use]
    pub const fn generation(&self) -> u64 {
        self.generation
    }

    /// Returns local view preferences.
    #[must_use]
    pub const fn preferences(&self) -> &ViewPreferences {
        &self.preferences
    }

    /// Returns local transcript navigation state.
    #[must_use]
    pub const fn transcript_viewport(&self) -> &TranscriptViewport {
        &self.transcript_viewport
    }

    /// Returns the persisted approval projection, when pending.
    #[must_use]
    pub const fn approval(&self) -> Option<&ApprovalView> {
        self.approval.as_ref()
    }

    /// Returns the selected approval choice.
    #[must_use]
    pub const fn approval_choice(&self) -> ApprovalChoice {
        self.approval_choice
    }

    /// Returns whether the approval command is already owned by the service.
    #[must_use]
    pub const fn approval_submitting(&self) -> bool {
        self.approval_submitting
    }

    /// Returns streamed incomplete messages.
    #[must_use]
    pub const fn streaming(&self) -> &BTreeMap<MessageId, StreamingMessage> {
        &self.streaming
    }

    /// Returns durable/transient tool projections.
    #[must_use]
    pub const fn tools(&self) -> &BTreeMap<ToolCallId, ToolView> {
        &self.tools
    }

    /// Returns ephemeral provider-hosted activity projections.
    #[must_use]
    pub const fn hosted_tools(&self) -> &BTreeMap<ToolCallId, HostedToolView> {
        &self.hosted_tools
    }

    /// Returns the currently scheduled model retry, when waiting in backoff.
    #[must_use]
    pub const fn model_retry(&self) -> Option<&ModelRetryView> {
        self.model_retry.as_ref()
    }

    /// Returns current editor contents.
    #[must_use]
    pub fn editor(&self) -> &str {
        &self.editor
    }

    /// Returns the selected skill mention styled in the local editor.
    #[must_use]
    pub fn editor_skill_mention(&self) -> Option<&str> {
        self.editor_skill_mention.as_deref()
    }

    /// Returns whether a model/tool run is currently active.
    #[must_use]
    pub const fn is_running(&self) -> bool {
        self.running
    }

    /// Returns elapsed whole seconds for the active request.
    #[must_use]
    pub const fn run_elapsed_seconds(&self) -> u64 {
        self.run_elapsed_seconds
    }

    /// Returns the number of session-scoped ephemeral queued inputs shown.
    #[must_use]
    pub fn queued_message_count(&self) -> usize {
        self.steering_queue.len() + self.follow_up_queue.len()
    }

    /// Returns validated images retained for the next idle prompt submission.
    #[must_use]
    pub fn attachments(&self) -> &[ComposerAttachment] {
        &self.attachments
    }

    /// Returns the currently displayed frozen skill catalog rows.
    #[must_use]
    pub fn skill_catalog(&self) -> &[String] {
        &self.skill_catalog
    }

    /// Clones the canonical image blocks in submission order.
    #[must_use]
    pub fn attachment_blocks(&self) -> Vec<ContentBlock> {
        self.attachments
            .iter()
            .map(|attachment| attachment.content().clone())
            .collect()
    }

    /// Returns aggregate encoded image bytes retained by the composer.
    #[must_use]
    pub fn attachment_encoded_bytes(&self) -> usize {
        self.attachments
            .iter()
            .map(ComposerAttachment::encoded_bytes)
            .sum()
    }

    /// Returns the active local selector, when open.
    #[must_use]
    pub fn selector(&self) -> Option<&Selector> {
        self.overlay.as_ref().and_then(Overlay::selector)
    }

    pub(crate) fn take_queued_text(&mut self) -> String {
        let mut queued = self
            .steering_queue
            .drain(..)
            .chain(self.follow_up_queue.drain(..))
            .collect::<Vec<_>>();
        queued.retain(|text| !text.is_empty());
        queued.join("\n")
    }

    pub(crate) fn take_session_ephemera(&mut self) -> SessionEphemera {
        SessionEphemera {
            steering_queue: std::mem::take(&mut self.steering_queue),
            follow_up_queue: std::mem::take(&mut self.follow_up_queue),
            attachments: std::mem::take(&mut self.attachments),
            pending_reasoning_effort: self.pending_reasoning_effort.take(),
        }
    }

    pub(crate) fn restore_session_ephemera(&mut self, ephemera: SessionEphemera) {
        self.steering_queue = ephemera.steering_queue;
        self.follow_up_queue = ephemera.follow_up_queue;
        self.attachments = ephemera.attachments;
        self.pending_reasoning_effort = ephemera.pending_reasoning_effort;
    }

    pub(crate) fn add_attachment(
        &mut self,
        attachment: ComposerAttachment,
    ) -> Result<(), AttachmentError> {
        validate_addition(
            self.attachments.len(),
            self.attachment_encoded_bytes(),
            attachment.encoded_bytes(),
        )?;
        self.attachments.push(attachment);
        Ok(())
    }

    pub(crate) fn remove_attachment(&mut self, index: usize) -> bool {
        let Some(index) = index
            .checked_sub(1)
            .filter(|index| *index < self.attachments.len())
        else {
            return false;
        };
        self.attachments.remove(index);
        true
    }

    pub(crate) fn clear_attachments(&mut self) {
        self.attachments.clear();
    }

    pub(crate) fn rebuild(&mut self, snapshot: &SessionSnapshot) {
        let state = snapshot.state();
        let previous_message_count = self.messages.len();
        self.session_id = state.session_id();
        self.profile_id = state.configuration().profile_id().clone();
        self.model_ref = state.configuration().model_ref().cloned();
        self.reasoning_effort = state.configuration().reasoning_effort();
        self.active_branch_id = state.active_branch_id();
        self.durable_tail = state.tail_sequence();
        // Event sequence is monotonic within the observation stream but may
        // diverge from record sequence after a run starts. Preserve the live
        // event cursor during an in-process rebuild; startup begins at the
        // durable tail in `from_snapshot`.
        self.messages = state.messages().to_vec();
        if self.messages.len() > previous_message_count {
            for _ in previous_message_count..self.messages.len() {
                self.transcript_viewport.note_new_item();
            }
        }
        if self
            .pending_user_prompt
            .as_ref()
            .is_some_and(|pending| prompt_is_durable(&self.messages, pending))
        {
            self.pending_user_prompt = None;
        }
        let previews = self
            .tools
            .iter()
            .filter_map(|(tool_call_id, tool)| {
                tool.preview.clone().map(|preview| (*tool_call_id, preview))
            })
            .collect::<BTreeMap<_, _>>();
        self.tools = durable_tools(snapshot);
        self.tool_order = tool_order(&self.messages, &self.tools);
        self.hosted_tools.clear();
        self.hosted_tool_order.clear();
        self.model_retry = None;
        let approval = pending_approval(snapshot);
        if let Some(pending) = &approval
            && let Some(preview) = previews.get(&pending.tool_call_id)
            && let Some(tool) = self.tools.get_mut(&pending.tool_call_id)
        {
            tool.preview = Some(preview.clone());
        }
        let approval_changed = self.approval.as_ref().map(|view| view.approval_id)
            != approval.as_ref().map(|view| view.approval_id);
        self.approval = approval;
        if approval_changed || self.approval.is_none() {
            self.approval_choice = ApprovalChoice::default();
            self.approval_submitting = false;
        }
        // Deltas and progress cannot be reconstructed reliably after a gap.
        self.streaming.clear();
        self.streaming_order.clear();
        self.resyncing = false;
        self.bump_generation();
    }

    pub(crate) fn set_model_default_reasoning_effort(
        &mut self,
        effort: Option<ReasoningEffort>,
    ) -> bool {
        if self.model_default_reasoning_effort == effort {
            false
        } else {
            self.model_default_reasoning_effort = effort;
            self.bump_generation();
            true
        }
    }

    pub(crate) fn bump_generation(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    pub(crate) fn show_pending_user_prompt(&mut self, text: String) {
        self.pending_user_prompt = (!text.is_empty()).then_some(PendingUserPrompt {
            text,
            durable_message_count: self.messages.len(),
        });
        self.transcript_viewport.note_new_item();
    }

    pub(crate) fn record_stream(&mut self, message_id: MessageId) {
        push_unique(
            &mut self.streaming_order,
            message_id,
            MAX_ACTIVE_STREAM_MESSAGES,
        );
    }

    pub(crate) fn record_tool_activity(&mut self, tool_call_id: ToolCallId) {
        push_unique(&mut self.tool_order, tool_call_id, MAX_ACTIVE_TOOL_ITEMS);
    }

    pub(crate) fn record_hosted_tool_activity(&mut self, tool_call_id: ToolCallId) {
        push_unique(
            &mut self.hosted_tool_order,
            tool_call_id,
            MAX_ACTIVE_TOOL_ITEMS,
        );
    }

    pub(crate) fn remember_event(&mut self, event_id: tea_protocol::EventId) {
        if self.observed_event_ids.len() == MAX_OBSERVED_EVENT_IDS {
            self.observed_event_ids.pop_front();
        }
        self.observed_event_ids.push_back(event_id);
    }

    pub(crate) fn notify(&mut self, message: impl Into<String>) {
        if self.notifications.len() == MAX_NOTIFICATIONS {
            self.notifications.pop_front();
        }
        self.notifications
            .push_back(sanitize_summary(&message.into()));
    }

    pub(crate) fn set_mcp_health(&mut self, rows: Vec<String>) {
        self.mcp_health = rows
            .into_iter()
            .take(MAX_MCP_HEALTH_ROWS)
            .map(|row| sanitize_summary(&row))
            .collect();
    }

    pub(crate) fn set_skill_catalog(&mut self, rows: Vec<String>) {
        self.skill_catalog = rows
            .into_iter()
            .take(MAX_SKILL_CATALOG_ROWS)
            .map(|row| sanitize_summary(&row))
            .collect();
    }

    pub(crate) fn note_transcript_item(&mut self) {
        self.transcript_viewport.note_new_item();
    }
}

fn durable_tools(snapshot: &SessionSnapshot) -> BTreeMap<ToolCallId, ToolView> {
    snapshot
        .state()
        .tool_calls()
        .iter()
        .map(|(id, tool)| {
            let status = match tool.execution() {
                ToolExecutionState::NotStarted => "proposed".to_owned(),
                ToolExecutionState::Started { .. } => "running".to_owned(),
                ToolExecutionState::Finished { is_error, .. } if *is_error => "failed".to_owned(),
                ToolExecutionState::Finished { .. } => "succeeded".to_owned(),
                ToolExecutionState::Interrupted { .. } => "interrupted/uncertain".to_owned(),
            };
            let presentation = match tool.execution() {
                ToolExecutionState::Finished { presentation, .. } => presentation.clone(),
                ToolExecutionState::NotStarted
                | ToolExecutionState::Started { .. }
                | ToolExecutionState::Interrupted { .. } => None,
            };
            (
                *id,
                ToolView {
                    tool_call_id: *id,
                    tool_name: tool.tool_name().to_owned(),
                    arguments: tool.arguments().clone(),
                    status,
                    approval_decision: tool.approval_decision(),
                    progress: None,
                    presentation,
                    preview: None,
                },
            )
        })
        .collect()
}

fn tool_order(
    messages: &[CanonicalMessage],
    tools: &BTreeMap<ToolCallId, ToolView>,
) -> VecDeque<ToolCallId> {
    let mut order = VecDeque::new();
    for message in messages {
        let CanonicalMessage::Assistant { content, .. } = message else {
            continue;
        };
        for block in content {
            if let tea_protocol::ContentBlock::ToolCall { tool_call_id, .. } = block {
                push_unique(&mut order, *tool_call_id, MAX_ACTIVE_TOOL_ITEMS);
            }
        }
    }
    // A valid snapshot normally contains the corresponding assistant tool
    // call. Keep malformed/legacy projections visible with a deterministic
    // fallback instead of silently hiding a known tool state.
    for tool_call_id in tools.keys() {
        push_unique(&mut order, *tool_call_id, MAX_ACTIVE_TOOL_ITEMS);
    }
    order
}

fn prompt_is_durable(messages: &[CanonicalMessage], pending: &PendingUserPrompt) -> bool {
    messages
        .iter()
        .skip(pending.durable_message_count)
        .any(|message| match message {
            CanonicalMessage::User { content, .. } => content
                .iter()
                .any(|block| matches!(block, ContentBlock::Text { text } if text == &pending.text)),
            CanonicalMessage::Assistant { .. } | CanonicalMessage::ToolResult { .. } => false,
        })
}

fn push_unique<T: Copy + PartialEq>(queue: &mut VecDeque<T>, value: T, capacity: usize) {
    if queue.iter().any(|item| *item == value) {
        return;
    }
    if queue.len() == capacity {
        queue.pop_front();
    }
    queue.push_back(value);
}

fn pending_approval(snapshot: &SessionSnapshot) -> Option<ApprovalView> {
    let pending = snapshot.state().pending_approvals();
    snapshot
        .approval_artifacts()
        .iter()
        .rev()
        .find_map(|artifact| match artifact {
            ApprovalArtifactEntry::Requested { request, .. }
                if pending.contains_key(request.approval_id()) =>
            {
                Some(ApprovalView {
                    approval_id: *request.approval_id(),
                    tool_call_id: *request.tool_call_id(),
                    tool_name: request.tool_name().to_string(),
                    effects: request
                        .effects()
                        .iter()
                        .map(|effect| effect.as_str().to_owned())
                        .collect(),
                    target: policy_target(request.environment().target()).to_owned(),
                    resources: request.presentation().resources().to_vec(),
                    reason: request.presentation().reason().to_owned(),
                    arguments: serde_json::to_string(request.presentation().arguments())
                        .unwrap_or_else(|_| "{\"redacted\":true}".to_owned()),
                    expires_at: request.expires_at(),
                })
            }
            ApprovalArtifactEntry::Requested { .. } | ApprovalArtifactEntry::Resolved { .. } => {
                None
            }
        })
}

const fn policy_target(target: tea_policy::PolicyExecutionTarget) -> &'static str {
    match target {
        tea_policy::PolicyExecutionTarget::Native => "native",
        tea_policy::PolicyExecutionTarget::Subprocess => "subprocess",
        tea_policy::PolicyExecutionTarget::Sandbox => "sandbox",
        tea_policy::PolicyExecutionTarget::Mcp => "mcp",
        tea_policy::PolicyExecutionTarget::Remote => "remote",
        tea_policy::PolicyExecutionTarget::Wasm => "wasm",
    }
}

fn sanitize_summary(value: &str) -> String {
    value
        .chars()
        .take(4096)
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .collect::<String>()
        .trim()
        .to_owned()
}
