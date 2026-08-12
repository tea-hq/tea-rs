use std::future::Future;
use std::pin::Pin;

use futures_util::stream::{FuturesOrdered, StreamExt as _};
use tea_protocol::{
    AgentEvent, AgentEventType, EventDelta, EventEnvelope, ReasoningEffort, SessionId,
};
use tea_session::SessionSnapshot;
use tokio::sync::mpsc;

use super::ComposerAttachment;
use super::Overlay;
use super::state::{
    ApprovalChoice, HostedToolView, MAX_VISIBLE_QUEUE_ITEMS, StreamingBlock, ToolProgressView,
    ToolView, TuiState,
};

/// Default bound for terminal actions awaiting reduction.
pub const DEFAULT_ACTION_CAPACITY: usize = 128;
/// Default bound for effects awaiting an executor.
pub const DEFAULT_EFFECT_CAPACITY: usize = 32;
const MAX_EPHEMERAL_TEXT_BYTES: usize = 1024 * 1024;

/// Typed inputs reduced into [`TuiState`].
#[derive(Debug)]
pub enum Action {
    /// One ordered canonical runtime event.
    Event(Box<EventEnvelope>),
    /// A fresh canonical snapshot returned by an effect.
    SnapshotLoaded(Box<SessionSnapshot>),
    /// A snapshot query failed with a safe diagnostic.
    SnapshotLoadFailed(String),
    /// The subscriber reconnected and requires authoritative rebuild.
    Reconnected,
    /// Terminal dimensions changed.
    Resize {
        /// New display-cell width.
        width: u16,
        /// New row height.
        height: u16,
    },
    /// Replace local editor contents without a selected skill mention.
    SetEditor(String),
    /// Replace local editor contents and its selected skill mention atomically.
    SetComposer {
        /// Bounded visible draft text.
        text: String,
        /// Selected skill token when the draft starts with one.
        skill_mention: Option<String>,
    },
    /// Add one validated image to the active session composer.
    AddAttachment(ComposerAttachment),
    /// Remove one image by its one-based composer index.
    RemoveAttachment {
        /// One-based attachment index shown to the user.
        index: usize,
    },
    /// Remove every image from the active session composer.
    ClearAttachments,
    /// Show an accepted prompt until its canonical user message is durable.
    ShowPendingUserPrompt(String),
    /// Begin the local activity projection once a prompt is accepted by the service.
    StartRunActivity,
    /// Advance the active-run elapsed counter by one or more whole seconds.
    AdvanceRunElapsed(u64),
    /// Stop local activity without discarding its measured elapsed time.
    FinishRunActivity,
    /// Replace the queued next-turn reasoning selection.
    SetPendingReasoningEffort(Option<ReasoningEffort>),
    /// Browse older transcript content without changing canonical state.
    ScrollTranscriptUp {
        /// Number of wrapped terminal rows to move toward older content.
        rows: usize,
    },
    /// Browse newer transcript content without changing canonical state.
    ScrollTranscriptDown {
        /// Number of wrapped terminal rows to move toward the live tail.
        rows: usize,
    },
    /// Return the transcript to the live tail and clear unread state.
    FollowTranscriptTail,
    /// Select one supported resolution for the pending approval.
    SelectApproval(ApprovalChoice),
    /// Mark whether an approval command has been accepted by the service.
    SetApprovalSubmitting(bool),
    /// Toggle all reasoning blocks.
    ToggleThinking,
    /// Toggle one tool's detail view.
    ToggleTool(tea_protocol::ToolCallId),
    /// Toggle every currently visible tool detail.
    ToggleAllTools,
    /// Project one queued steering message.
    QueueSteering(String),
    /// Project one queued follow-up message.
    QueueFollowUp(String),
    /// Replace the active local interaction overlay.
    SetOverlay(Option<Overlay>),
    /// Add one bounded local notification.
    Notify(String),
    /// Replace safe MCP lifecycle rows from a host projection.
    SetMcpHealth(Vec<String>),
    /// Replace the bounded frozen skill catalog projection.
    SetSkillCatalog(Vec<String>),
}

/// Side effects requested by the pure reducer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Effect {
    /// Reload one immutable canonical session snapshot.
    ReloadSnapshot {
        /// Session requiring an authoritative rebuild.
        session_id: SessionId,
    },
    /// Request a new terminal frame.
    Render,
}

/// Failure to enqueue work into a bounded application loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DispatchError {
    /// The bounded action queue is full.
    #[error("terminal action queue is full")]
    Full,
    /// The terminal application loop has stopped.
    #[error("terminal action loop is closed")]
    Closed,
}

/// Cloneable bounded sender for terminal actions.
#[derive(Debug, Clone)]
pub struct ActionSender {
    sender: mpsc::Sender<Action>,
}

impl ActionSender {
    /// Enqueues one action without waiting.
    ///
    /// # Errors
    ///
    /// Returns [`DispatchError::Full`] under backpressure or
    /// [`DispatchError::Closed`] after shutdown.
    pub fn try_send(&self, action: Action) -> Result<(), DispatchError> {
        self.sender.try_send(action).map_err(|error| match error {
            mpsc::error::TrySendError::Full(_) => DispatchError::Full,
            mpsc::error::TrySendError::Closed(_) => DispatchError::Closed,
        })
    }

    /// Waits for bounded action capacity.
    ///
    /// # Errors
    ///
    /// Returns [`DispatchError::Closed`] after shutdown.
    pub async fn send(&self, action: Action) -> Result<(), DispatchError> {
        self.sender
            .send(action)
            .await
            .map_err(|_| DispatchError::Closed)
    }
}

/// Async boundary used by the bounded application loop.
pub trait EffectExecutor: Send + Sync {
    /// Starts one typed effect and optionally returns a typed completion action.
    ///
    /// The executor must copy any needed projection data from `state` before
    /// returning. The returned owned future lets the application loop poll
    /// several effects concurrently without spawning detached tasks.
    fn execute(
        &self,
        effect: Effect,
        state: &TuiState,
    ) -> Pin<Box<dyn Future<Output = Option<Action>> + Send + 'static>>;
}

/// Owned bounded Action → Effect application loop.
///
/// The reducer runs synchronously. Effects are polled concurrently through a
/// bounded owned set, so no task is detached and shutdown is deterministic.
#[derive(Debug)]
pub struct ActionLoop {
    actions: mpsc::Receiver<Action>,
    effect_capacity: usize,
}

impl ActionLoop {
    /// Creates an application loop with explicit non-zero queue bounds.
    ///
    /// # Panics
    ///
    /// Panics when either capacity is zero.
    #[must_use]
    pub fn new(action_capacity: usize, effect_capacity: usize) -> (ActionSender, Self) {
        assert!(action_capacity > 0, "action capacity must be non-zero");
        assert!(effect_capacity > 0, "effect capacity must be non-zero");
        let (action_sender, actions) = mpsc::channel(action_capacity);
        (
            ActionSender {
                sender: action_sender,
            },
            Self {
                actions,
                effect_capacity,
            },
        )
    }

    /// Runs until every action sender is dropped and all accepted effects finish.
    ///
    /// Effects are polled concurrently up to the configured bound and their
    /// completion actions are reduced in dispatch order. Action reception is
    /// backpressured while that bound is full, so terminal input remains
    /// bounded without dropping authoritative snapshot reloads.
    pub async fn run(mut self, state: &mut TuiState, executor: &dyn EffectExecutor) {
        type EffectFuture = Pin<Box<dyn Future<Output = Option<Action>> + Send + 'static>>;
        let mut running = FuturesOrdered::<EffectFuture>::new();
        let mut actions_open = true;
        loop {
            if !actions_open && running.is_empty() {
                break;
            }
            tokio::select! {
                action = self.actions.recv(), if actions_open && running.len() < self.effect_capacity => {
                    match action {
                        Some(action) => {
                            let effects = reduce(state, action);
                            enqueue_effects(
                                &mut running,
                                effects,
                                self.effect_capacity,
                                executor,
                                state,
                            );
                        }
                        None => actions_open = false,
                    }
                }
                result = running.next(), if !running.is_empty() => {
                    if let Some(Some(action)) = result {
                        let effects = reduce(state, action);
                        enqueue_effects(
                            &mut running,
                            effects,
                            self.effect_capacity,
                            executor,
                            state,
                        );
                    }
                }
            }
        }
    }
}

fn enqueue_effects(
    running: &mut FuturesOrdered<Pin<Box<dyn Future<Output = Option<Action>> + Send + 'static>>>,
    effects: Vec<Effect>,
    capacity: usize,
    executor: &dyn EffectExecutor,
    state: &TuiState,
) {
    for effect in effects {
        if running.len() == capacity {
            // A reducer emits authoritative reload before its coalescible frame,
            // so only an observational render can reach this branch.
            debug_assert!(matches!(effect, Effect::Render));
            continue;
        }
        running.push_back(executor.execute(effect, state));
    }
}

/// Pure deterministic state transition.
#[must_use]
#[allow(clippy::too_many_lines)] // Keep the complete state-action grammar auditable together.
pub fn reduce(state: &mut TuiState, action: Action) -> Vec<Effect> {
    match action {
        Action::Event(event) => reduce_event(state, &event),
        Action::SnapshotLoaded(snapshot) => {
            if snapshot.state().session_id() == state.session_id {
                state.rebuild(&snapshot);
            } else {
                state.notify("ignored snapshot for a different session");
            }
            vec![Effect::Render]
        }
        Action::SnapshotLoadFailed(message) => {
            state.resyncing = false;
            state.notify(format!("session resync failed: {message}"));
            state.bump_generation();
            vec![Effect::Render]
        }
        Action::Reconnected => request_resync(state, "event stream reconnected"),
        Action::Resize { width, height } => {
            state.viewport_width = width.max(1);
            state.viewport_height = height.max(1);
            state.bump_generation();
            vec![Effect::Render]
        }
        Action::SetEditor(text) => {
            state.editor = bounded_text(text, MAX_EPHEMERAL_TEXT_BYTES);
            state.editor_skill_mention = None;
            state.bump_generation();
            vec![Effect::Render]
        }
        Action::SetComposer {
            text,
            skill_mention,
        } => {
            state.editor = bounded_text(text, MAX_EPHEMERAL_TEXT_BYTES);
            state.editor_skill_mention = skill_mention
                .map(|mention| bounded_text(mention, 512))
                .filter(|mention| valid_editor_skill_mention(&state.editor, mention));
            state.bump_generation();
            vec![Effect::Render]
        }
        Action::AddAttachment(attachment) => {
            if let Err(error) = state.add_attachment(attachment) {
                state.notify(error.to_string());
            }
            state.bump_generation();
            vec![Effect::Render]
        }
        Action::RemoveAttachment { index } => {
            if !state.remove_attachment(index) {
                state.notify("attachment index is out of range");
            }
            state.bump_generation();
            vec![Effect::Render]
        }
        Action::ClearAttachments => {
            state.clear_attachments();
            state.bump_generation();
            vec![Effect::Render]
        }
        Action::ShowPendingUserPrompt(text) => {
            state.show_pending_user_prompt(bounded_text(text, MAX_EPHEMERAL_TEXT_BYTES));
            state.bump_generation();
            vec![Effect::Render]
        }
        Action::StartRunActivity => {
            state.running = true;
            state.run_elapsed_seconds = 0;
            state.model_retry = None;
            state.run_status = None;
            state.usage = None;
            state.cost = None;
            state.notifications.clear();
            state.bump_generation();
            vec![Effect::Render]
        }
        Action::AdvanceRunElapsed(seconds) => {
            if state.running && seconds > 0 {
                state.run_elapsed_seconds = state.run_elapsed_seconds.saturating_add(seconds);
                if let Some(retry) = state.model_retry.as_mut() {
                    retry.advance(seconds);
                }
                state.bump_generation();
                vec![Effect::Render]
            } else {
                Vec::new()
            }
        }
        Action::FinishRunActivity => {
            if state.running {
                state.running = false;
                state.model_retry = None;
                state.bump_generation();
                vec![Effect::Render]
            } else {
                Vec::new()
            }
        }
        Action::SetPendingReasoningEffort(effort) => {
            if state.pending_reasoning_effort == effort {
                Vec::new()
            } else {
                state.pending_reasoning_effort = effort;
                state.bump_generation();
                vec![Effect::Render]
            }
        }
        Action::ScrollTranscriptUp { rows } => {
            state.transcript_viewport.scroll_up(rows);
            state.bump_generation();
            vec![Effect::Render]
        }
        Action::ScrollTranscriptDown { rows } => {
            state.transcript_viewport.scroll_down(rows);
            state.bump_generation();
            vec![Effect::Render]
        }
        Action::FollowTranscriptTail => {
            state.transcript_viewport.follow_tail();
            state.bump_generation();
            vec![Effect::Render]
        }
        Action::SelectApproval(choice) => {
            if state.approval.is_some() && !state.approval_submitting {
                state.approval_choice = choice;
                state.bump_generation();
                vec![Effect::Render]
            } else {
                Vec::new()
            }
        }
        Action::SetApprovalSubmitting(submitting) => {
            if state.approval.is_some() && state.approval_submitting != submitting {
                state.approval_submitting = submitting;
                state.bump_generation();
                vec![Effect::Render]
            } else {
                Vec::new()
            }
        }
        Action::ToggleThinking => {
            state.preferences.thinking_collapsed = !state.preferences.thinking_collapsed;
            state.bump_generation();
            vec![Effect::Render]
        }
        Action::ToggleTool(tool_call_id) => {
            if !state.preferences.expanded_tools.remove(&tool_call_id) {
                state.preferences.expanded_tools.insert(tool_call_id);
            }
            state.bump_generation();
            vec![Effect::Render]
        }
        Action::ToggleAllTools => {
            if state
                .tools
                .keys()
                .all(|id| state.preferences.expanded_tools.contains(id))
            {
                state.preferences.expanded_tools.clear();
            } else {
                state
                    .preferences
                    .expanded_tools
                    .extend(state.tools.keys().copied());
            }
            state.bump_generation();
            vec![Effect::Render]
        }
        Action::QueueSteering(text) => {
            push_queue(&mut state.steering_queue, text);
            state.bump_generation();
            vec![Effect::Render]
        }
        Action::QueueFollowUp(text) => {
            push_queue(&mut state.follow_up_queue, text);
            state.bump_generation();
            vec![Effect::Render]
        }
        Action::SetOverlay(overlay) => {
            state.overlay = overlay;
            state.bump_generation();
            vec![Effect::Render]
        }
        Action::Notify(message) => {
            state.notify(message);
            state.bump_generation();
            vec![Effect::Render]
        }
        Action::SetMcpHealth(rows) => {
            state.set_mcp_health(rows);
            state.bump_generation();
            vec![Effect::Render]
        }
        Action::SetSkillCatalog(rows) => {
            state.set_skill_catalog(rows);
            state.bump_generation();
            vec![Effect::Render]
        }
    }
}

fn valid_editor_skill_mention(editor: &str, mention: &str) -> bool {
    mention.starts_with('$')
        && mention.len() > 1
        && !mention.chars().any(char::is_whitespace)
        && editor.strip_prefix(mention).is_some_and(|remaining| {
            remaining.is_empty() || remaining.chars().next().is_some_and(char::is_whitespace)
        })
}

#[allow(clippy::too_many_lines)] // Keep the complete protocol-event grammar auditable together.
fn reduce_event(state: &mut TuiState, envelope: &EventEnvelope) -> Vec<Effect> {
    if envelope.session_id() != state.session_id {
        state.notify("ignored event for a different session");
        return vec![Effect::Render];
    }
    if state
        .observed_event_ids
        .iter()
        .any(|event_id| *event_id == envelope.event_id())
    {
        return Vec::new();
    }
    if envelope.sequence() <= state.event_cursor {
        if envelope.event_type().compatibility()
            == tea_protocol::EventCompatibility::SkippableObservation
        {
            state.remember_event(envelope.event_id());
            return Vec::new();
        }
        return request_resync(state, "late state-bearing event detected");
    }
    // Durable records and observations do not advance one-for-one, so the
    // protocol guarantees monotonic event order rather than contiguity.
    if state.resyncing {
        state.event_cursor = envelope.sequence();
        state.remember_event(envelope.event_id());
        return Vec::new();
    }

    state.event_cursor = envelope.sequence();
    state.remember_event(envelope.event_id());
    let mut effects = vec![Effect::Render];
    match envelope.event() {
        AgentEvent::RunStarted {} => {
            state.running = true;
            state.run_elapsed_seconds = 0;
            state.model_retry = None;
            state.observation_run_id = envelope.run_id();
            state.run_status = None;
            state.usage = None;
            state.cost = None;
            state.notifications.clear();
            state.hosted_tools.clear();
            state.hosted_tool_order.clear();
        }
        AgentEvent::MessageDelta {
            message_id,
            content_index,
            delta,
        } => {
            let (thinking, text) = match delta {
                EventDelta::TextDelta { text } => (false, text.as_str()),
                EventDelta::ThinkingDelta { text } => (true, text.as_str()),
            };
            if !state.streaming.contains_key(message_id) {
                state.record_stream(*message_id);
            }
            let message = state.streaming.entry(*message_id).or_default();
            let block = message
                .blocks
                .entry(*content_index)
                .or_insert_with(|| StreamingBlock {
                    thinking,
                    text: String::new(),
                    generation: 0,
                });
            if block.thinking != thinking {
                return request_resync(state, "stream block type changed");
            }
            if block.text.len().saturating_add(text.len()) > MAX_EPHEMERAL_TEXT_BYTES {
                return request_resync(state, "streamed message exceeded UI bound");
            }
            block.text.push_str(text);
            block.generation = block.generation.wrapping_add(1);
            state.note_transcript_item();
        }
        AgentEvent::ToolCallRequested {
            tool_call_id,
            tool_name,
            arguments,
        } => {
            state.record_tool_activity(*tool_call_id);
            state
                .tools
                .entry(*tool_call_id)
                .or_insert_with(|| ToolView {
                    tool_call_id: *tool_call_id,
                    tool_name: tool_name.clone(),
                    arguments: arguments.clone(),
                    status: "requested".to_owned(),
                    approval_decision: None,
                    progress: None,
                    presentation: None,
                    preview: None,
                });
            state.note_transcript_item();
        }
        AgentEvent::ToolExecutionProgress {
            tool_call_id,
            message,
            completed_units,
            total_units,
        } => {
            if let Some(tool) = state.tools.get_mut(tool_call_id) {
                "running".clone_into(&mut tool.status);
                tool.progress = Some(ToolProgressView {
                    message: message.clone(),
                    completed_units: *completed_units,
                    total_units: *total_units,
                });
                state.note_transcript_item();
            } else {
                return request_resync(state, "tool progress referenced an unknown call");
            }
        }
        AgentEvent::ToolExecutionPreview {
            tool_call_id,
            presentation,
        } => {
            if let Some(tool) = state.tools.get_mut(tool_call_id) {
                tool.preview = Some(presentation.clone());
            } else {
                return request_resync(state, "tool preview referenced an unknown call");
            }
        }
        AgentEvent::HostedToolStarted {
            tool_call_id,
            tool_name,
        } => {
            if state.hosted_tools.contains_key(tool_call_id) {
                return request_resync(state, "hosted tool reused an active identity");
            }
            state.record_hosted_tool_activity(*tool_call_id);
            state.hosted_tools.insert(
                *tool_call_id,
                HostedToolView {
                    tool_call_id: *tool_call_id,
                    tool_name: tool_name.clone(),
                    arguments: None,
                    outcome: None,
                    source_count: None,
                },
            );
            state.note_transcript_item();
        }
        AgentEvent::HostedToolCompleted {
            tool_call_id,
            tool_name,
            arguments,
            outcome,
            source_count,
        } => {
            let Some(tool) = state.hosted_tools.get_mut(tool_call_id) else {
                return request_resync(state, "hosted completion referenced an unknown call");
            };
            if tool.tool_name != *tool_name || tool.outcome.is_some() {
                return request_resync(state, "hosted completion changed an active identity");
            }
            tool.arguments = Some(arguments.clone());
            tool.outcome = Some(outcome.clone());
            tool.source_count = Some(*source_count);
            state.note_transcript_item();
        }
        AgentEvent::ModelRetryScheduled {
            message_id,
            attempt,
            max_retries,
            delay_ms,
        } => {
            state.streaming.remove(message_id);
            state.streaming_order.retain(|id| id != message_id);
            state.hosted_tools.clear();
            state.hosted_tool_order.clear();
            state.model_retry = Some(super::state::ModelRetryView::new(
                *message_id,
                *attempt,
                *max_retries,
                *delay_ms,
            ));
        }
        AgentEvent::ModelRetryStarted { .. } => {
            state.model_retry = None;
        }
        AgentEvent::ApprovalRequested { .. }
        | AgentEvent::TurnCheckpointed {}
        | AgentEvent::SessionCompacted { .. }
        | AgentEvent::SessionForked { .. } => {
            state.resyncing = true;
            effects.insert(
                0,
                Effect::ReloadSnapshot {
                    session_id: state.session_id,
                },
            );
        }
        AgentEvent::RunFinished {
            status,
            usage,
            cost,
        } => {
            state.running = false;
            state.model_retry = None;
            state.run_status = Some(*status);
            state.usage.clone_from(usage);
            state.cost.clone_from(cost);
            state.resyncing = true;
            effects.insert(
                0,
                Effect::ReloadSnapshot {
                    session_id: state.session_id,
                },
            );
        }
    }
    state.bump_generation();
    effects
}

fn request_resync(state: &mut TuiState, reason: &str) -> Vec<Effect> {
    if state.resyncing {
        Vec::new()
    } else {
        state.resyncing = true;
        state.notify(reason);
        state.bump_generation();
        vec![
            Effect::ReloadSnapshot {
                session_id: state.session_id,
            },
            Effect::Render,
        ]
    }
}

fn push_queue(queue: &mut std::collections::VecDeque<String>, text: String) {
    if queue.len() == MAX_VISIBLE_QUEUE_ITEMS {
        queue.pop_front();
    }
    queue.push_back(bounded_text(text, 64 * 1024));
}

fn bounded_text(mut text: String, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text;
    }
    let mut boundary = max_bytes;
    while !text.is_char_boundary(boundary) {
        boundary -= 1;
    }
    text.truncate(boundary);
    text
}

/// Returns whether an event is observational and safe to drop after a newer cursor.
#[must_use]
pub const fn is_observational(event: AgentEventType) -> bool {
    matches!(
        event,
        AgentEventType::MessageDelta
            | AgentEventType::ToolExecutionProgress
            | AgentEventType::ToolExecutionPreview
            | AgentEventType::HostedToolStarted
            | AgentEventType::HostedToolCompleted
            | AgentEventType::ModelRetryScheduled
            | AgentEventType::ModelRetryStarted
    )
}
