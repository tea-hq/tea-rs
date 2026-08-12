use std::borrow::Cow;
use std::collections::BTreeSet;
use std::fmt::Write as _;

use tea_protocol::{
    ApprovalDecision, CanonicalMessage, CodeChange, CodeChangeLineKind, CodeChangeTruncation,
    ContentBlock, ExternalSource, HostedToolActivity, HostedToolOutcome, ImageSource, MessageId,
    ToolCallId, ToolPresentation, WebFetchPresentation, WebFetchTruncation,
};
use unicode_segmentation::UnicodeSegmentation as _;
use unicode_width::UnicodeWidthStr as _;

use super::attachment::{decoded_inline_image_bytes, format_byte_count};
use super::state::{ApprovalView, HostedToolView, StreamingMessage, ToolView, TuiState};

mod cell;

pub use cell::{
    CellContent, CellId, CellLane, CellNode, DecisionCell, DiffCell, LifecycleCell, LifecycleKind,
    MessageAuthor, MessageCell, MessageCellFacet, NoticeCell, NoticeKind, PlanCell,
    QueuedInputCell, ReasoningCell, ResultCell, SourcesCell, StreamCellFacet, ToolCellFacet,
};

const MAX_CELL_TEXT_BYTES: usize = 64 * 1024;
const MAX_QUEUE_PREVIEW_CELLS: usize = 160;
const QUEUE_PREVIEW_ELLIPSIS: &str = "...";

/// Source format owned by a typed timeline body.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OutputFormat {
    /// Preserve the source as terminal-safe plain text.
    Plain,
    /// Project the source through the message-level Markdown writer.
    Markdown,
    /// Preserve process output whitespace without Markdown interpretation.
    Terminal,
}

/// Observable lifecycle state used to choose markers and action emphasis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LifecycleStatus {
    /// Work has been proposed but not requested.
    Proposed,
    /// Work has been requested but has not started.
    Requested,
    /// Work is waiting for an explicit approval decision.
    ApprovalPending,
    /// Work is currently executing.
    Running,
    /// Work completed successfully.
    Succeeded,
    /// Work completed with an explicit failure.
    Failed,
    /// Work was interrupted before completion.
    Interrupted,
    /// The final effect of interrupted work is unknown.
    Uncertain,
    /// Work or input is queued for later execution.
    Queued,
}

/// Semantic treatment for one structured detail row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TimelineDetailKind {
    /// Arguments, targets, identifiers, and other metadata.
    Metadata,
    /// Observable progress reported by the runtime.
    Progress,
    /// Successful tool or process output.
    Output,
    /// Explicit failure output.
    Error,
}

/// One renderer-neutral, tree-prefixed detail row.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TimelineDetail {
    kind: TimelineDetailKind,
    label: Option<String>,
    text: String,
}

/// One terminal-safe source label with an optional validated OSC8 destination.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct TimelineSource {
    label: String,
    url: String,
    destination: Option<String>,
}

impl TimelineSource {
    pub(crate) fn from_external(source: &ExternalSource) -> Self {
        Self {
            label: terminal_safe_text(source.title().unwrap_or(source.url())),
            url: terminal_safe_text(source.url()),
            destination: super::hyperlink::validate_destination(source.url()),
        }
    }

    /// Returns the visible source label.
    #[must_use]
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Returns the original validated source URL for copy-friendly output.
    #[must_use]
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Returns the canonical destination when it is safe for OSC8 output.
    #[must_use]
    pub fn destination(&self) -> Option<&str> {
        self.destination.as_deref()
    }
}

impl TimelineDetail {
    pub(crate) fn new(kind: TimelineDetailKind, label: Option<&str>, text: &str) -> Self {
        Self {
            kind,
            label: label.map(bounded_terminal_text),
            text: bounded_terminal_text(text),
        }
    }

    /// Returns the semantic detail category.
    #[must_use]
    pub const fn kind(&self) -> TimelineDetailKind {
        self.kind
    }

    /// Returns the optional compact detail label.
    #[must_use]
    pub fn label(&self) -> Option<&str> {
        self.label.as_deref()
    }

    /// Returns the terminal-safe source text.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }
}

/// Severity for a typed notice surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum NoticeSeverity {
    /// Low-salience informational output.
    Information,
    /// A warning that may require attention.
    Warning,
    /// An explicit failure or error diagnostic.
    Error,
}

/// Durable approval or decision state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DecisionStatus {
    /// A user decision is still required.
    Pending,
    /// The action was approved.
    Approved,
    /// The action was denied.
    Denied,
    /// The request expired before a decision.
    TimedOut,
    /// The decision flow was aborted.
    Aborted,
    /// A selected decision is being submitted.
    Submitting,
}

/// Semantic type for one queued input preview.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QueuedInputKind {
    /// Input intended to steer the active run.
    Steering,
    /// Input intended for the next turn.
    FollowUp,
}

/// Protocol-ready status for one plan step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum PlanStepStatus {
    /// Work has not started.
    Pending,
    /// Work is currently active.
    InProgress,
    /// Work has completed.
    Completed,
}

/// One bounded renderer-neutral plan step.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PlanStep {
    status: PlanStepStatus,
    text: String,
}

impl PlanStep {
    /// Creates one terminal-safe plan step.
    #[must_use]
    pub fn new(status: PlanStepStatus, text: &str) -> Self {
        Self {
            status,
            text: bounded_terminal_text(text),
        }
    }

    /// Returns the observable step status.
    #[must_use]
    pub const fn status(&self) -> PlanStepStatus {
        self.status
    }

    /// Returns the terminal-safe step text.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }
}

/// Pure presentation projection split by durable and transient lifetime.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Presentation {
    history: Vec<CellNode>,
    active: Vec<CellNode>,
    notifications: Vec<CellNode>,
}

impl Presentation {
    /// Projects one terminal state without changing canonical or local state.
    #[must_use]
    pub fn from_state(state: &TuiState) -> Self {
        let history = HistoryProjection::from_state(state);
        let active = project_active_cells(
            state,
            &history.represented_tools,
            &history.represented_hosted_tools,
        );

        Self {
            history: history.cells,
            active,
            notifications: project_notifications(state),
        }
    }

    /// Returns cells backed by the durable session transcript.
    #[must_use]
    pub fn history(&self) -> &[CellNode] {
        &self.history
    }

    /// Returns active or session-local activity below the durable transcript.
    #[must_use]
    pub fn active(&self) -> &[CellNode] {
        &self.active
    }

    /// Returns bounded transient notifications separate from the transcript.
    #[must_use]
    pub fn notifications(&self) -> &[CellNode] {
        &self.notifications
    }
}

fn project_active_cells(
    state: &TuiState,
    represented_tools: &BTreeSet<ToolCallId>,
    represented_hosted_tools: &BTreeSet<ToolCallId>,
) -> Vec<CellNode> {
    let mut active = Vec::new();
    if let Some(pending) = &state.pending_user_prompt {
        active.push(message_cell(
            CellId::Synthetic {
                lane: CellLane::Active,
                index: 0,
            },
            MessageAuthor::User,
            pending.text(),
            OutputFormat::Plain,
        ));
    }
    for message_id in &state.streaming_order {
        if let Some(message) = state.streaming.get(message_id) {
            project_stream(
                *message_id,
                message,
                state.preferences.thinking_collapsed,
                &mut active,
            );
        }
    }
    for tool_call_id in &state.tool_order {
        if !represented_tools.contains(tool_call_id)
            && let Some(tool) = state.tools.get(tool_call_id)
        {
            active.push(project_tool(
                state,
                tool_call_id,
                &tool.tool_name,
                &tool.arguments,
                Some(tool),
            ));
        }
    }
    for tool_call_id in &state.hosted_tool_order {
        if !represented_hosted_tools.contains(tool_call_id)
            && let Some(tool) = state.hosted_tools.get(tool_call_id)
        {
            active.push(project_hosted_view(tool, state.run_elapsed_seconds));
        }
    }
    project_queues(state, &mut active);
    for (index, row) in state.mcp_health.iter().enumerate() {
        active.push(notice_cell(
            CellId::McpHealth {
                index: bounded_index(index),
            },
            NoticeKind::McpHealth,
            NoticeSeverity::Information,
            row,
            None,
        ));
    }
    for (index, row) in state.skill_catalog.iter().enumerate() {
        active.push(notice_cell(
            CellId::Synthetic {
                lane: CellLane::Notifications,
                index: bounded_index(index),
            },
            NoticeKind::General,
            NoticeSeverity::Information,
            row,
            None,
        ));
    }
    if let Some(approval) = &state.approval {
        active.push(project_approval(approval, state.approval_submitting));
        if let Some(preview) = state
            .tools
            .get(&approval.tool_call_id)
            .and_then(|tool| tool.preview.as_ref())
            .and_then(ToolPresentation::code_change)
        {
            active.push(preview_diff_cell(preview, approval.tool_call_id));
        }
    }
    active
}

fn project_queues(state: &TuiState, active: &mut Vec<CellNode>) {
    for (input_kind, queue) in [
        (QueuedInputKind::Steering, &state.steering_queue),
        (QueuedInputKind::FollowUp, &state.follow_up_queue),
    ] {
        for (index, text) in queue.iter().enumerate() {
            active.push(CellNode::new(
                CellId::Queue {
                    kind: input_kind,
                    index: bounded_index(index),
                },
                CellContent::QueuedInput(QueuedInputCell::new(input_kind, text)),
                None,
            ));
        }
    }
}

fn project_approval(approval: &ApprovalView, submitting: bool) -> CellNode {
    let status = if submitting {
        DecisionStatus::Submitting
    } else {
        DecisionStatus::Pending
    };
    let detail =
        |label, text: &str| TimelineDetail::new(TimelineDetailKind::Metadata, Some(label), text);
    decision_cell(
        CellId::Tool {
            tool_call_id: approval.tool_call_id,
            facet: ToolCellFacet::Decision,
            part_index: 0,
        },
        "Approval required",
        &approval.tool_name,
        status,
        vec![
            detail("target", &approval.target),
            detail("effects", &approval.effects.join(", ")),
            detail("resources", &approval.resources.join(", ")),
            detail("reason", &approval.reason),
            detail("arguments", &approval.arguments),
            detail("expires", &approval.expires_at.to_string()),
        ],
        Some(approval.tool_call_id),
    )
}

fn project_notifications(state: &TuiState) -> Vec<CellNode> {
    state
        .notifications
        .iter()
        .enumerate()
        .map(|(index, message)| {
            notice_cell(
                CellId::Notification {
                    index: bounded_index(index),
                },
                NoticeKind::General,
                NoticeSeverity::Information,
                message,
                None,
            )
        })
        .collect()
}

fn bounded_index(index: usize) -> u16 {
    u16::try_from(index).unwrap_or(u16::MAX)
}

fn bounded_content_index(index: u32) -> u16 {
    u16::try_from(index).unwrap_or(u16::MAX)
}

struct HistoryProjection<'a> {
    state: &'a TuiState,
    cells: Vec<CellNode>,
    represented_tools: BTreeSet<ToolCallId>,
    represented_hosted_tools: BTreeSet<ToolCallId>,
}

impl<'a> HistoryProjection<'a> {
    fn from_state(state: &'a TuiState) -> Self {
        let mut projection = Self {
            state,
            cells: Vec::new(),
            represented_tools: BTreeSet::new(),
            represented_hosted_tools: BTreeSet::new(),
        };
        for message in &state.messages {
            projection.project_message(message);
        }
        projection
    }

    fn project_message(&mut self, message: &CanonicalMessage) {
        match message {
            CanonicalMessage::User { id, content, .. } => {
                self.project_content(*id, content, MessageAuthor::User, OutputFormat::Plain);
            }
            CanonicalMessage::Assistant { id, content, .. } => {
                self.project_content(
                    *id,
                    content,
                    MessageAuthor::Assistant,
                    OutputFormat::Markdown,
                );
                self.project_assistant_sources(*id, content);
            }
            CanonicalMessage::ToolResult {
                tool_call_id,
                tool_name,
                content,
                is_error,
                ..
            } => self.project_tool_result(*tool_call_id, tool_name, content, *is_error),
        }
    }

    fn project_content(
        &mut self,
        message_id: MessageId,
        content: &[ContentBlock],
        author: MessageAuthor,
        format: OutputFormat,
    ) {
        for (block_index, block) in content.iter().enumerate() {
            self.project_content_block(
                message_id,
                bounded_index(block_index),
                block,
                author,
                format,
            );
        }
    }

    fn project_assistant_sources(&mut self, message_id: MessageId, content: &[ContentBlock]) {
        let sources = assistant_sources(content);
        if sources.is_empty() {
            return;
        }
        self.cells.push(CellNode::new(
            CellId::Message {
                message_id,
                block_index: bounded_index(content.len()),
                facet: MessageCellFacet::Sources,
            },
            CellContent::Sources(SourcesCell::new(
                sources.iter().map(TimelineSource::from_external).collect(),
            )),
            None,
        ));
    }

    fn project_tool_result(
        &mut self,
        tool_call_id: ToolCallId,
        tool_name: &str,
        content: &[ContentBlock],
        is_error: bool,
    ) {
        if !is_error
            && let Some(presentation) = self
                .state
                .tools
                .get(&tool_call_id)
                .and_then(|tool| tool.presentation.as_ref())
        {
            let cell = match presentation {
                ToolPresentation::CodeChange(change) => diff_cell(change, tool_call_id),
                ToolPresentation::WebFetch(fetch) => web_fetch_cell(fetch, tool_call_id),
            };
            self.cells.push(cell);
            return;
        }

        let action = if is_error { "Failed" } else { "Returned" };
        for (part_index, block) in content.iter().enumerate() {
            let (text, format) = match block {
                ContentBlock::Text { text } | ContentBlock::Thinking { text } => {
                    (Cow::Borrowed(text.as_str()), OutputFormat::Terminal)
                }
                ContentBlock::Image { mime_type, source } => (
                    Cow::Owned(image_metadata(mime_type, source)),
                    OutputFormat::Plain,
                ),
                ContentBlock::ContextualText { .. }
                | ContentBlock::ToolCall { .. }
                | ContentBlock::HostedTool { .. }
                | ContentBlock::Citation { .. } => continue,
            };
            self.cells.push(result_cell(
                CellId::Tool {
                    tool_call_id,
                    facet: ToolCellFacet::Result,
                    part_index: bounded_index(part_index),
                },
                action,
                Some(tool_name),
                text.as_ref(),
                format,
                is_error,
                Some(tool_call_id),
            ));
        }
    }

    fn project_content_block(
        &mut self,
        message_id: MessageId,
        block_index: u16,
        block: &ContentBlock,
        author: MessageAuthor,
        format: OutputFormat,
    ) {
        let message_cell_id = CellId::Message {
            message_id,
            block_index,
            facet: MessageCellFacet::Content,
        };
        match block {
            ContentBlock::Text { text } => {
                self.cells
                    .push(message_cell(message_cell_id, author, text, format));
            }
            ContentBlock::Thinking { text } => {
                self.cells.push(reasoning_cell(
                    message_cell_id,
                    text,
                    self.state.preferences.thinking_collapsed,
                ));
            }
            ContentBlock::Image { mime_type, source } => self.cells.push(message_cell(
                message_cell_id,
                author,
                &image_metadata(mime_type, source),
                OutputFormat::Plain,
            )),
            ContentBlock::ToolCall {
                tool_call_id,
                tool_name,
                arguments,
                ..
            } => {
                self.represented_tools.insert(*tool_call_id);
                let tool = self.state.tools.get(tool_call_id);
                self.cells.push(project_tool(
                    self.state,
                    tool_call_id,
                    tool_name,
                    arguments,
                    tool,
                ));
                if let Some(decision) = tool.and_then(project_tool_decision) {
                    self.cells.push(decision);
                }
            }
            ContentBlock::HostedTool { activity } => {
                self.represented_hosted_tools
                    .insert(activity.tool_call_id());
                self.cells.push(project_hosted_activity(activity));
            }
            ContentBlock::ContextualText { .. } | ContentBlock::Citation { .. } => {}
        }
    }
}

fn assistant_sources(content: &[ContentBlock]) -> Vec<ExternalSource> {
    let mut seen = BTreeSet::new();
    let mut sources = Vec::new();
    for block in content {
        let candidates: &[ExternalSource] = match block {
            ContentBlock::HostedTool { activity } => activity.sources(),
            ContentBlock::Citation { citation } => std::slice::from_ref(citation.source()),
            ContentBlock::Text { .. }
            | ContentBlock::ContextualText { .. }
            | ContentBlock::Thinking { .. }
            | ContentBlock::Image { .. }
            | ContentBlock::ToolCall { .. } => &[],
        };
        for source in candidates {
            let deduplication_key = super::hyperlink::validate_destination(source.url())
                .unwrap_or_else(|| source.url().to_owned());
            if seen.insert(deduplication_key) {
                sources.push(source.clone());
            }
        }
    }
    sources
}

fn project_hosted_activity(activity: &HostedToolActivity) -> CellNode {
    hosted_tool_cell(
        activity.tool_call_id(),
        activity.tool_name(),
        Some(activity.arguments()),
        Some(activity.outcome()),
        Some(u32::try_from(activity.sources().len()).unwrap_or(u32::MAX)),
        0,
    )
}

fn project_hosted_view(tool: &HostedToolView, tick: u64) -> CellNode {
    hosted_tool_cell(
        tool.tool_call_id,
        &tool.tool_name,
        tool.arguments.as_ref(),
        tool.outcome.as_ref(),
        tool.source_count,
        tick,
    )
}

fn hosted_tool_cell(
    tool_call_id: ToolCallId,
    tool_name: &str,
    arguments: Option<&serde_json::Value>,
    outcome: Option<&HostedToolOutcome>,
    source_count: Option<u32>,
    tick: u64,
) -> CellNode {
    let (action, status) = match outcome {
        None if tool_name == "web_search" => ("Searching web", LifecycleStatus::Running),
        None => ("Running hosted tool", LifecycleStatus::Running),
        Some(HostedToolOutcome::Success) if tool_name == "web_search" => {
            ("Searched web", LifecycleStatus::Succeeded)
        }
        Some(HostedToolOutcome::Success) => ("Ran hosted tool", LifecycleStatus::Succeeded),
        Some(HostedToolOutcome::Error(_)) if tool_name == "web_search" => {
            ("Web search failed", LifecycleStatus::Failed)
        }
        Some(HostedToolOutcome::Error(_)) => ("Hosted tool failed", LifecycleStatus::Failed),
    };
    let target = arguments
        .and_then(|arguments| arguments.get("query"))
        .and_then(serde_json::Value::as_str)
        .map(bounded_terminal_text);
    let mut details = Vec::new();
    if let Some(source_count) = source_count {
        details.push(TimelineDetail::new(
            TimelineDetailKind::Output,
            Some("sources"),
            &source_count.to_string(),
        ));
    }
    if let Some(HostedToolOutcome::Error(error)) = outcome {
        details.push(TimelineDetail::new(
            TimelineDetailKind::Error,
            Some(error.code()),
            error.message(),
        ));
    }
    CellNode::new(
        CellId::Tool {
            tool_call_id,
            facet: ToolCellFacet::Call,
            part_index: 0,
        },
        CellContent::Lifecycle(LifecycleCell::new(
            LifecycleKind::HostedTool,
            action,
            target.as_deref(),
            status,
            details,
            false,
            if status == LifecycleStatus::Running {
                tick
            } else {
                0
            },
        )),
        Some(tool_call_id),
    )
}

fn image_metadata(mime_type: &str, source: &ImageSource) -> String {
    match source {
        ImageSource::InlineBase64 { data } => decoded_inline_image_bytes(data).map_or_else(
            || format!("[image {mime_type}]"),
            |bytes| format!("[image {mime_type} · {}]", format_byte_count(bytes)),
        ),
        ImageSource::Reference { .. } => format!("[image {mime_type}]"),
    }
}

fn project_tool_decision(tool: &ToolView) -> Option<CellNode> {
    let (action, status, scope) = match tool.approval_decision? {
        ApprovalDecision::AllowOnce => ("Approved", DecisionStatus::Approved, Some("this call")),
        ApprovalDecision::AllowSession => ("Approved", DecisionStatus::Approved, Some("session")),
        ApprovalDecision::Deny => ("Denied", DecisionStatus::Denied, None),
    };
    let details = scope.map_or_else(Vec::new, |scope| {
        vec![TimelineDetail::new(
            TimelineDetailKind::Metadata,
            Some("scope"),
            scope,
        )]
    });
    Some(decision_cell(
        CellId::Tool {
            tool_call_id: tool.tool_call_id,
            facet: ToolCellFacet::Decision,
            part_index: 0,
        },
        action,
        &tool.tool_name,
        status,
        details,
        Some(tool.tool_call_id),
    ))
}

fn project_tool(
    state: &TuiState,
    tool_call_id: &ToolCallId,
    tool_name: &str,
    arguments: &serde_json::Value,
    tool: Option<&ToolView>,
) -> CellNode {
    let pending_approval = state
        .approval
        .as_ref()
        .is_some_and(|approval| approval.tool_call_id == *tool_call_id);
    let status = if pending_approval {
        LifecycleStatus::ApprovalPending
    } else {
        tool.map_or(LifecycleStatus::Requested, |tool| {
            lifecycle_status(&tool.status)
        })
    };
    let mut details = Vec::new();
    let expanded = !pending_approval && state.preferences.expanded_tools.contains(tool_call_id);
    if !pending_approval {
        let arguments = serde_json::to_string(arguments).unwrap_or_else(|_| "{}".to_owned());
        details.push(TimelineDetail::new(
            TimelineDetailKind::Metadata,
            Some("arguments"),
            &arguments,
        ));
    }
    if let Some(progress) = tool.and_then(|tool| tool.progress.as_ref()) {
        let units = progress.total_units.map_or_else(
            || progress.completed_units.to_string(),
            |total| format!("{}/{}", progress.completed_units, total),
        );
        details.push(TimelineDetail::new(
            TimelineDetailKind::Progress,
            Some("progress"),
            &format!("{units} {}", progress.message),
        ));
    }
    let target = tool_target(tool_name, arguments);
    CellNode::new(
        CellId::Tool {
            tool_call_id: *tool_call_id,
            facet: ToolCellFacet::Call,
            part_index: 0,
        },
        CellContent::Lifecycle(LifecycleCell::new(
            LifecycleKind::ToolCall,
            lifecycle_action(status),
            Some(&target),
            status,
            details,
            expanded,
            if status == LifecycleStatus::Running {
                state.run_elapsed_seconds
            } else {
                0
            },
        )),
        Some(*tool_call_id),
    )
}

fn tool_target(tool_name: &str, arguments: &serde_json::Value) -> String {
    let target = arguments.as_object().and_then(|arguments| {
        ["path", "command", "query", "url"]
            .into_iter()
            .find_map(|key| arguments.get(key).and_then(serde_json::Value::as_str))
    });
    target.map_or_else(
        || tool_name.to_owned(),
        |target| format!("{tool_name} {target}"),
    )
}

fn lifecycle_status(status: &str) -> LifecycleStatus {
    match status {
        "proposed" => LifecycleStatus::Proposed,
        "requested" => LifecycleStatus::Requested,
        "running" => LifecycleStatus::Running,
        "succeeded" => LifecycleStatus::Succeeded,
        "failed" => LifecycleStatus::Failed,
        "interrupted" => LifecycleStatus::Interrupted,
        _ => LifecycleStatus::Uncertain,
    }
}

const fn lifecycle_action(status: LifecycleStatus) -> &'static str {
    match status {
        LifecycleStatus::Proposed => "Proposed",
        LifecycleStatus::Requested => "Requested",
        LifecycleStatus::ApprovalPending => "Waiting for approval",
        LifecycleStatus::Running => "Running",
        LifecycleStatus::Succeeded => "Ran",
        LifecycleStatus::Failed => "Failed",
        LifecycleStatus::Interrupted | LifecycleStatus::Uncertain => "Interrupted",
        LifecycleStatus::Queued => "Queued",
    }
}

fn project_stream(
    message_id: MessageId,
    message: &StreamingMessage,
    thinking_collapsed: bool,
    output: &mut Vec<CellNode>,
) {
    for (block_index, block) in &message.blocks {
        if block.thinking {
            output.push(reasoning_cell(
                CellId::Stream {
                    message_id,
                    block_index: bounded_content_index(*block_index),
                    facet: StreamCellFacet::Reasoning,
                },
                &block.text,
                thinking_collapsed,
            ));
        } else {
            output.push(message_cell(
                CellId::Stream {
                    message_id,
                    block_index: bounded_content_index(*block_index),
                    facet: StreamCellFacet::Message,
                },
                MessageAuthor::Assistant,
                &block.text,
                OutputFormat::Markdown,
            ));
        }
    }
}

fn message_cell(id: CellId, author: MessageAuthor, source: &str, format: OutputFormat) -> CellNode {
    CellNode::new(
        id,
        CellContent::Message(MessageCell::new(author, source, format)),
        None,
    )
}

fn reasoning_cell(id: CellId, source: &str, collapsed: bool) -> CellNode {
    CellNode::new(
        id,
        CellContent::Reasoning(ReasoningCell::new(source, collapsed)),
        None,
    )
}

fn result_cell(
    id: CellId,
    action: &str,
    source_name: Option<&str>,
    content: &str,
    format: OutputFormat,
    is_error: bool,
    tool_call_id: Option<ToolCallId>,
) -> CellNode {
    CellNode::new(
        id,
        CellContent::Result(ResultCell::new(
            action,
            source_name,
            content,
            format,
            is_error,
        )),
        tool_call_id,
    )
}

fn web_fetch_cell(fetch: &WebFetchPresentation, tool_call_id: ToolCallId) -> CellNode {
    result_cell(
        CellId::Tool {
            tool_call_id,
            facet: ToolCellFacet::Result,
            part_index: 0,
        },
        "Fetched",
        Some("web_fetch"),
        &web_fetch_content(fetch),
        OutputFormat::Plain,
        false,
        Some(tool_call_id),
    )
}

fn web_fetch_content(fetch: &WebFetchPresentation) -> String {
    let mut content = format!(
        "URL: {}\nContent-Type: {}",
        fetch.final_url(),
        fetch.mime_type()
    );
    if let Some(title) = fetch.title() {
        content.push_str("\nTitle: ");
        content.push_str(title);
    }
    if fetch.requested_url() != fetch.final_url() {
        content.push_str("\nRequested URL: ");
        content.push_str(fetch.requested_url());
    }
    if let Some(truncation) = fetch.truncation() {
        content.push_str("\nTruncated: ");
        content.push_str(match truncation {
            WebFetchTruncation::CompressedBytes => "compressed bytes",
            WebFetchTruncation::DecodedBytes => "decoded bytes",
            WebFetchTruncation::BodyCharacters => "body characters",
            WebFetchTruncation::ParserComplexity => "parser complexity",
        });
    }
    if !fetch.redirects().is_empty() {
        content.push_str("\nRedirects: ");
        content.push_str(&fetch.redirects().len().to_string());
    }
    content.push_str("\n\n");
    content.push_str(fetch.body());
    content
}

fn diff_cell(change: &CodeChange, tool_call_id: ToolCallId) -> CellNode {
    let action = match change.kind() {
        tea_protocol::CodeChangeKind::Create => "Created",
        tea_protocol::CodeChangeKind::Update => "Updated",
        tea_protocol::CodeChangeKind::Delete => "Deleted",
    };
    CellNode::new(
        CellId::Tool {
            tool_call_id,
            facet: ToolCellFacet::Diff,
            part_index: 0,
        },
        CellContent::Diff(DiffCell::new(action, change.clone())),
        Some(tool_call_id),
    )
}

fn preview_diff_cell(change: &CodeChange, tool_call_id: ToolCallId) -> CellNode {
    CellNode::new(
        CellId::Tool {
            tool_call_id,
            facet: ToolCellFacet::ApprovalPreview,
            part_index: 0,
        },
        CellContent::Diff(DiffCell::new("Preview", change.clone())),
        Some(tool_call_id),
    )
}

fn notice_cell(
    id: CellId,
    kind: NoticeKind,
    severity: NoticeSeverity,
    message: &str,
    hint: Option<&str>,
) -> CellNode {
    CellNode::new(
        id,
        CellContent::Notice(NoticeCell::new(kind, severity, message, hint)),
        None,
    )
}

fn decision_cell(
    id: CellId,
    action: &str,
    subject: &str,
    status: DecisionStatus,
    details: Vec<TimelineDetail>,
    tool_call_id: Option<ToolCallId>,
) -> CellNode {
    CellNode::new(
        id,
        CellContent::Decision(DecisionCell::new(action, subject, status, details)),
        tool_call_id,
    )
}

fn raw_plan_text(
    title: &str,
    source: Option<&str>,
    steps: &[PlanStep],
    note: Option<&str>,
) -> String {
    if let Some(source) = source {
        return source.to_owned();
    }
    let mut raw = title.to_owned();
    for step in steps {
        let marker = match step.status() {
            PlanStepStatus::Pending => "[ ]",
            PlanStepStatus::InProgress => "[>]",
            PlanStepStatus::Completed => "[x]",
        };
        if !raw.is_empty() {
            raw.push('\n');
        }
        raw.push_str(marker);
        raw.push(' ');
        raw.push_str(step.text());
    }
    if let Some(note) = note {
        if !raw.is_empty() {
            raw.push('\n');
        }
        raw.push_str(note);
    }
    raw
}

fn append_raw_details(raw: &mut String, details: &[TimelineDetail]) {
    for detail in details {
        raw.push('\n');
        if let Some(label) = detail.label() {
            raw.push_str(label);
            raw.push(' ');
        }
        raw.push_str(detail.text());
    }
}

pub(super) fn terminal_safe_text(value: &str) -> String {
    value.chars().map(terminal_safe_character).collect()
}

fn code_change_raw_text(change: &CodeChange) -> String {
    if let Some(patch) = change.patch() {
        return patch.to_owned();
    }
    let mut raw = format!("{}\n", change.path());
    for hunk in change.hunks() {
        writeln!(
            raw,
            "@@ -{},{} +{},{} @@",
            hunk.old_start(),
            hunk.old_lines(),
            hunk.new_start(),
            hunk.new_lines()
        )
        .expect("writing to a String cannot fail");
        for line in hunk.lines() {
            let marker = match line.kind() {
                CodeChangeLineKind::Context => ' ',
                CodeChangeLineKind::Addition => '+',
                CodeChangeLineKind::Deletion => '-',
            };
            raw.push(marker);
            raw.push_str(line.text());
            raw.push('\n');
        }
    }
    if change.truncated() {
        raw.push_str("[diff truncated: ");
        raw.push_str(truncation_text(change.truncation()));
        raw.push_str("]\n");
    }
    raw
}

fn truncation_text(truncation: Option<CodeChangeTruncation>) -> &'static str {
    match truncation {
        Some(CodeChangeTruncation::Hunks) => "hunks",
        Some(CodeChangeTruncation::Lines) => "lines",
        Some(CodeChangeTruncation::LineBytes) => "line bytes",
        Some(CodeChangeTruncation::PatchBytes) => "patch bytes",
        None => "unknown",
    }
}

fn terminal_safe_character(character: char) -> char {
    match character {
        '\n' => '\n',
        '\t' => ' ',
        '\u{1b}' => '␛',
        value if value.is_control() => '�',
        value => value,
    }
}

fn bounded_terminal_text(value: &str) -> String {
    let mut text = String::new();
    for character in value.chars() {
        let mapped = terminal_safe_character(character);
        if text.len().saturating_add(mapped.len_utf8()) > MAX_CELL_TEXT_BYTES {
            break;
        }
        text.push(mapped);
    }
    text
}

fn queue_preview(value: &str) -> String {
    let safe = bounded_terminal_text(value);
    let compact = safe.split_whitespace().collect::<Vec<_>>().join(" ");
    if compact.width() <= MAX_QUEUE_PREVIEW_CELLS {
        return compact;
    }

    let available = MAX_QUEUE_PREVIEW_CELLS.saturating_sub(QUEUE_PREVIEW_ELLIPSIS.width());
    let mut preview = String::new();
    for grapheme in compact.graphemes(true) {
        if preview.width().saturating_add(grapheme.width()) > available {
            break;
        }
        preview.push_str(grapheme);
    }
    preview.push_str(QUEUE_PREVIEW_ELLIPSIS);
    preview
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use tea_protocol::{CodeChange, CodeChangeKind, MessageId, ToolCallId};

    use super::{
        CellContent, CellId, CellLane, CellNode, DecisionCell, DecisionStatus, DiffCell,
        LifecycleCell, LifecycleKind, LifecycleStatus, MessageAuthor, MessageCell,
        MessageCellFacet, NoticeCell, NoticeKind, NoticeSeverity, OutputFormat, PlanCell,
        QueuedInputCell, QueuedInputKind, ReasoningCell, ResultCell, SourcesCell, StreamCellFacet,
        ToolCellFacet, bounded_terminal_text,
    };

    fn message_id(value: &str) -> MessageId {
        value.parse().expect("test message ID must be valid")
    }

    fn tool_call_id(value: &str) -> ToolCallId {
        value.parse().expect("test tool-call ID must be valid")
    }

    fn code_change() -> CodeChange {
        CodeChange::new(
            "src/lib.rs",
            CodeChangeKind::Update,
            Vec::new(),
            false,
            None,
            Some("@@ -1 +1 @@\n-old\n+new\n".to_owned()),
            Some(1),
        )
        .expect("test code change must be valid")
    }

    #[test]
    fn terminal_text_is_control_safe_and_bounded() {
        assert_eq!(bounded_terminal_text("a\u{1b}[31m\t\0"), "a␛[31m �");
    }

    #[test]
    fn raw_text_is_control_safe_and_preserves_empty_canonical_results() {
        let notice = CellContent::Notice(NoticeCell::new(
            NoticeKind::General,
            NoticeSeverity::Warning,
            "warning\u{1b}[31m",
            Some("hint\t\0"),
        ));
        assert_eq!(notice.raw_text(), "warning␛[31m\nhint �");

        let empty_result = CellContent::Result(ResultCell::new(
            "Returned",
            Some("custom_tool"),
            "",
            OutputFormat::Plain,
            false,
        ));
        assert_eq!(empty_result.raw_text(), "");
    }

    #[test]
    fn semantic_subtypes_live_inside_typed_payloads_without_parallel_kind() {
        let message = MessageCell::new(MessageAuthor::User, "hello", OutputFormat::Plain);
        let lifecycle = LifecycleCell::new(
            LifecycleKind::HostedTool,
            "Searched",
            Some("docs"),
            LifecycleStatus::Succeeded,
            Vec::new(),
            false,
            0,
        );
        let notice = NoticeCell::new(
            NoticeKind::McpHealth,
            NoticeSeverity::Warning,
            "offline",
            None,
        );

        assert_eq!(message.author(), MessageAuthor::User);
        assert_eq!(lifecycle.kind(), LifecycleKind::HostedTool);
        assert_eq!(notice.kind(), NoticeKind::McpHealth);
        assert!(matches!(
            CellContent::Message(message),
            CellContent::Message(_)
        ));
        assert!(matches!(
            CellContent::Lifecycle(lifecycle),
            CellContent::Lifecycle(_)
        ));
        assert!(matches!(
            CellContent::Notice(notice),
            CellContent::Notice(_)
        ));
    }

    #[test]
    fn raw_text_is_renderer_independent_for_every_content_variant() {
        let values = vec![
            (
                CellContent::Message(MessageCell::new(
                    MessageAuthor::Assistant,
                    "message",
                    OutputFormat::Markdown,
                )),
                "message",
            ),
            (
                CellContent::Reasoning(ReasoningCell::new("reasoning", true)),
                "reasoning",
            ),
            (
                CellContent::Plan(PlanCell::new("Plan", Some("plan source"), Vec::new(), None)),
                "plan source",
            ),
            (
                CellContent::Lifecycle(LifecycleCell::new(
                    LifecycleKind::RunActivity,
                    "Running",
                    Some("tests"),
                    LifecycleStatus::Running,
                    Vec::new(),
                    false,
                    1,
                )),
                "Running tests",
            ),
            (
                CellContent::Result(ResultCell::new(
                    "Returned",
                    Some("tool"),
                    "result",
                    OutputFormat::Plain,
                    false,
                )),
                "result",
            ),
            (CellContent::Sources(SourcesCell::new(Vec::new())), ""),
            (
                CellContent::Diff(DiffCell::new("Updated", code_change())),
                "@@ -1 +1 @@\n-old\n+new\n",
            ),
            (
                CellContent::QueuedInput(QueuedInputCell::new(QueuedInputKind::Steering, "queue")),
                "queue",
            ),
            (
                CellContent::Notice(NoticeCell::new(
                    NoticeKind::General,
                    NoticeSeverity::Information,
                    "notice",
                    Some("hint"),
                )),
                "notice\nhint",
            ),
            (
                CellContent::Decision(DecisionCell::new(
                    "Approve",
                    "workspace",
                    DecisionStatus::Pending,
                    Vec::new(),
                )),
                "Approve workspace",
            ),
        ];

        for (content, expected) in values {
            assert_eq!(content.raw_text(), expected);
        }
    }

    #[test]
    fn tool_ownership_is_metadata_not_a_kind_body_convention() {
        let owner = tool_call_id("0195a0b1-7e00-7000-8000-000000000099");
        let content = CellContent::Result(ResultCell::new(
            "Returned",
            Some("custom_tool"),
            "ok",
            OutputFormat::Plain,
            false,
        ));
        let node = CellNode::new(
            CellId::Tool {
                tool_call_id: owner,
                facet: ToolCellFacet::Result,
                part_index: 0,
            },
            content.clone(),
            Some(owner),
        );

        assert_eq!(node.content(), &content);
        assert_eq!(node.tool_call_id(), Some(owner));
    }

    #[test]
    fn stable_cell_ids_distinguish_message_blocks_and_tool_facets() {
        let message = message_id("0195a0b1-7e00-7000-8000-000000000001");
        let tool = tool_call_id("0195a0b1-7e00-7000-8000-000000000099");
        let ids = BTreeSet::from([
            CellId::Message {
                message_id: message,
                block_index: 0,
                facet: MessageCellFacet::Content,
            },
            CellId::Message {
                message_id: message,
                block_index: 1,
                facet: MessageCellFacet::Content,
            },
            CellId::Tool {
                tool_call_id: tool,
                facet: ToolCellFacet::Call,
                part_index: 0,
            },
            CellId::Tool {
                tool_call_id: tool,
                facet: ToolCellFacet::Decision,
                part_index: 0,
            },
            CellId::Tool {
                tool_call_id: tool,
                facet: ToolCellFacet::Result,
                part_index: 0,
            },
            CellId::Stream {
                message_id: message,
                block_index: 0,
                facet: StreamCellFacet::Message,
            },
            CellId::Synthetic {
                lane: CellLane::Active,
                index: 0,
            },
        ]);

        assert_eq!(ids.len(), 7);
    }
}
