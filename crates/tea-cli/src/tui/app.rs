use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::path::{Path, PathBuf};
use std::str::FromStr as _;
use std::time::Duration;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::backend::{Backend as _, CrosstermBackend};
use ratatui::layout::{Position, Rect, Size};
use tea::RuntimeCommandOutcome;
use tea_coding::{CodingAgentService, config::persist_global_model_settings};
use tea_mcp::McpServerId;
use tea_protocol::{
    AgentEvent, CanonicalMessage, ContentBlock, ModelRef, ReasoningEffort, SessionId,
};
use tea_session::SessionName;
use tokio::sync::mpsc;
use tokio::time;
use unicode_segmentation::UnicodeSegmentation as _;

use super::attachment::{load_local_image, validate_addition};
use super::clipboard::{Clipboard, SystemClipboard};
use super::commands::{CommandCatalog, SlashCommand};
use super::custom_terminal::Terminal;
use super::editor::Editor;
use super::input::{InputEvent, spawn_input_pump};
use super::keymap::{BindingAction, EditorAction, KeyMap, resolve_editor_action};
use super::layout::draw_lines;
use super::overlay::{CommandCompletion, Overlay};
use super::presentation::{CellNode, Presentation};
use super::reducer::{Action, Effect, reduce};
use super::render::{Renderer, workspace_trust_panel};
use super::selectors::{Selector, SelectorItem, SelectorValue};
use super::state::{ApprovalChoice, SessionEphemera, StartupContext, TuiState};
use super::status::StatusIndicator;
use super::terminal::{
    CrosstermDriver, TerminalCapabilities, TerminalGuard, TerminalOptions, TerminalTitle,
    ViewportMode,
};
use super::theme::Theme;
use crate::args::{CliArgs, SessionSelection};
use crate::bootstrap::WorkspaceTrustPrompt;
use crate::session_views::{mcp_servers, session_list, session_tree};
use crate::{CliBootstrap, CliFailure, ExitCategory};

const INPUT_CAPACITY: usize = 128;
const FRAME_DEADLINE: Duration = Duration::from_millis(100);
const STREAM_FRAME_CADENCE: Duration = Duration::from_millis(50);
const RUN_ACTIVITY_TICK: Duration = Duration::from_secs(1);

fn sync_reasoning_projection(state: &mut TuiState, service: &CodingAgentService) -> bool {
    let model_default = state
        .model_ref()
        .and_then(|model| service.model_spec(model))
        .map(|model| {
            model.reasoning_profile().map_or(
                ReasoningEffort::Off,
                tea_model::ReasoningProfile::default_effort,
            )
        });
    state.set_model_default_reasoning_effort(model_default)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum WorkspaceTrustChoice {
    Trust,
    #[default]
    Exit,
}

impl WorkspaceTrustChoice {
    const fn toggled(self) -> Self {
        match self {
            Self::Trust => Self::Exit,
            Self::Exit => Self::Trust,
        }
    }

    const fn should_trust(self) -> bool {
        matches!(self, Self::Trust)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum WorkspaceTrustKeyAction {
    Select(WorkspaceTrustChoice),
    Submit(bool),
}

/// Coalesces visible stream deltas while preserving immediate interactive frames.
#[derive(Debug, Default)]
struct FrameScheduler {
    last_frame: Option<time::Instant>,
    pending_stream_frame: bool,
}

impl FrameScheduler {
    fn request(&mut self, stream_delta: bool, now: time::Instant) -> bool {
        if !stream_delta {
            self.pending_stream_frame = false;
            return true;
        }
        if self
            .last_frame
            .is_none_or(|last_frame| now >= last_frame + STREAM_FRAME_CADENCE)
        {
            self.pending_stream_frame = false;
            true
        } else {
            self.pending_stream_frame = true;
            false
        }
    }

    fn next_deadline(&self) -> Option<time::Instant> {
        self.pending_stream_frame
            .then(|| self.last_frame.map(|frame| frame + STREAM_FRAME_CADENCE))
            .flatten()
    }

    fn take_pending(&mut self) -> bool {
        std::mem::take(&mut self.pending_stream_frame)
    }

    fn record_frame(&mut self, now: time::Instant) {
        self.last_frame = Some(now);
        self.pending_stream_frame = false;
    }
}

/// Tracks whole elapsed seconds from run acceptance without depending on wall-clock sleeps in tests.
#[derive(Debug, Default)]
struct RunActivityTimer {
    started_at: Option<time::Instant>,
    reported_seconds: u64,
}

impl RunActivityTimer {
    fn from_state(running: bool, elapsed_seconds: u64, now: time::Instant) -> Self {
        let started_at = running.then(|| {
            now.checked_sub(Duration::from_secs(elapsed_seconds))
                .unwrap_or(now)
        });
        Self {
            started_at,
            reported_seconds: elapsed_seconds,
        }
    }

    fn sync(&mut self, was_running: bool, running: bool, elapsed_seconds: u64, now: time::Instant) {
        match (was_running, running) {
            (false, true) => {
                self.started_at = Some(
                    now.checked_sub(Duration::from_secs(elapsed_seconds))
                        .unwrap_or(now),
                );
                self.reported_seconds = elapsed_seconds;
            }
            (true, false) => {
                self.started_at = None;
                self.reported_seconds = elapsed_seconds;
            }
            (true, true) => {
                self.reported_seconds = self.reported_seconds.max(elapsed_seconds);
            }
            (false, false) => {}
        }
    }

    fn next_deadline(&self) -> Option<time::Instant> {
        self.started_at.map(|started_at| {
            started_at + Duration::from_secs(self.reported_seconds) + RUN_ACTIVITY_TICK
        })
    }

    fn advance(&mut self, now: time::Instant) -> u64 {
        let Some(started_at) = self.started_at else {
            return 0;
        };
        let elapsed_seconds = now.saturating_duration_since(started_at).as_secs();
        let advance = elapsed_seconds.saturating_sub(self.reported_seconds);
        self.reported_seconds = self.reported_seconds.max(elapsed_seconds);
        advance
    }
}

/// Synchronous frame boundary injected into the asynchronous application loop.
pub trait FrameSink {
    /// Queues finalized cells for one-time insertion into native terminal scrollback.
    ///
    /// # Errors
    ///
    /// Returns a terminal output failure.
    fn insert_history_cells(&mut self, _cells: &[CellNode]) -> io::Result<()> {
        Ok(())
    }

    /// Replaces source-backed history before a resize or canonical rebuild replay.
    ///
    /// # Errors
    ///
    /// Returns a terminal output failure.
    fn replace_history_cells(&mut self, _cells: &[CellNode]) -> io::Result<()> {
        Ok(())
    }

    /// Renders the current pure state projection and editor cursor.
    ///
    /// # Errors
    ///
    /// Returns a terminal output failure.
    fn render(&mut self, state: &TuiState, cursor_byte: usize) -> io::Result<()>;
}

/// Runs the production crossterm interactive mode.
///
/// # Errors
///
/// Returns stable CLI failures for bootstrap, terminal, runtime, or input errors.
pub async fn run(args: &CliArgs, bootstrap: &CliBootstrap) -> Result<(), CliFailure> {
    if args.print || args.json || args.rpc {
        return Err(CliFailure::usage(
            "interactive mode cannot be combined with a headless mode",
        ));
    }
    if let Some(prompt) = bootstrap.workspace_trust_prompt(args)? {
        if !prompt_for_workspace_trust(&prompt).await? {
            return Err(CliFailure::new(
                ExitCategory::TrustOrConfig,
                "project-local configuration is not trusted",
            ));
        }
        CliBootstrap::accept_workspace_trust(&prompt)?;
    }
    let initial_prompt = interactive_initial_prompt(args, bootstrap)?;
    let global_settings_path = bootstrap.global_settings_path(args)?;
    let (service, selection, startup_notices) = bootstrap.build_tui_async(args).await?;
    let mut interrupt = tokio::spawn(tokio::signal::ctrl_c());
    tokio::task::yield_now().await;
    let mut frames = match TerminalFrameSink::new(
        service.settings().tui.viewport.as_str(),
        service.settings().tui.reduced_motion,
    ) {
        Ok(frames) => frames,
        Err(error) => {
            interrupt.abort();
            service.shutdown().await;
            return Err(error);
        }
    };
    let (pump, input) = spawn_input_pump(INPUT_CAPACITY);
    let mut clipboard = SystemClipboard;
    let interactive = Box::pin(run_with_channels_context(
        &service,
        selection,
        input,
        &mut frames,
        &mut clipboard,
        InteractiveStartup {
            initial_prompt,
            notices: startup_notices,
            global_settings_path: Some(global_settings_path),
        },
    ));
    let result = tokio::select! {
        result = interactive => result.map(drop),
        signal = &mut interrupt => match signal {
            Ok(Ok(())) => Err(CliFailure::new(ExitCategory::Cancelled, "interrupted")),
            Ok(Err(_)) | Err(_) => Err(CliFailure::new(
                ExitCategory::Internal,
                "interrupt handler failed",
            )),
        },
    };
    interrupt.abort();
    pump.shutdown().await;
    let _ = frames.clear_inline_viewport();
    drop(frames);
    service.shutdown().await;
    result
}

async fn prompt_for_workspace_trust(prompt: &WorkspaceTrustPrompt) -> Result<bool, CliFailure> {
    let mut frames = TerminalFrameSink::new("inline", true)?;
    let (pump, input) = spawn_input_pump(INPUT_CAPACITY);
    let mut interrupt = tokio::spawn(tokio::signal::ctrl_c());
    tokio::task::yield_now().await;
    let prompt_loop = Box::pin(run_workspace_trust_prompt(prompt, input, &mut frames));
    let result = tokio::select! {
        result = prompt_loop => result,
        signal = &mut interrupt => match signal {
            Ok(Ok(())) => Err(CliFailure::new(ExitCategory::Cancelled, "interrupted")),
            Ok(Err(_)) | Err(_) => Err(CliFailure::new(
                ExitCategory::Internal,
                "interrupt handler failed",
            )),
        },
    };
    interrupt.abort();
    pump.shutdown().await;
    let cleanup = frames
        .clear_inline_viewport()
        .map_err(|_| CliFailure::new(ExitCategory::Internal, "terminal rendering failed"));
    drop(frames);
    cleanup?;
    result
}

async fn run_workspace_trust_prompt(
    prompt: &WorkspaceTrustPrompt,
    mut input: mpsc::Receiver<InputEvent>,
    frames: &mut TerminalFrameSink,
) -> Result<bool, CliFailure> {
    let mut choice = WorkspaceTrustChoice::default();
    frames
        .render_workspace_trust(prompt, choice)
        .map_err(|_| CliFailure::new(ExitCategory::Internal, "terminal rendering failed"))?;
    while let Some(input) = input.recv().await {
        match input {
            InputEvent::Resize { .. } => {
                frames.render_workspace_trust(prompt, choice).map_err(|_| {
                    CliFailure::new(ExitCategory::Internal, "terminal rendering failed")
                })?;
            }
            InputEvent::Key(key) => match workspace_trust_key_action(choice, key) {
                WorkspaceTrustKeyAction::Select(selected) => {
                    choice = selected;
                    frames.render_workspace_trust(prompt, choice).map_err(|_| {
                        CliFailure::new(ExitCategory::Internal, "terminal rendering failed")
                    })?;
                }
                WorkspaceTrustKeyAction::Submit(trusted) => return Ok(trusted),
            },
            InputEvent::Focus(_) | InputEvent::Paste(_) => {}
        }
    }
    Ok(false)
}

fn workspace_trust_key_action(
    choice: WorkspaceTrustChoice,
    key: KeyEvent,
) -> WorkspaceTrustKeyAction {
    match key.code {
        KeyCode::Left
        | KeyCode::Right
        | KeyCode::Up
        | KeyCode::Down
        | KeyCode::Tab
        | KeyCode::BackTab => WorkspaceTrustKeyAction::Select(choice.toggled()),
        KeyCode::Char('1') => WorkspaceTrustKeyAction::Select(WorkspaceTrustChoice::Trust),
        KeyCode::Char('2') => WorkspaceTrustKeyAction::Select(WorkspaceTrustChoice::Exit),
        KeyCode::Enter => WorkspaceTrustKeyAction::Submit(choice.should_trust()),
        KeyCode::Esc => WorkspaceTrustKeyAction::Submit(false),
        KeyCode::Char('c' | 'd') if key.modifiers.contains(KeyModifiers::CONTROL) => {
            WorkspaceTrustKeyAction::Submit(false)
        }
        _ => WorkspaceTrustKeyAction::Select(choice),
    }
}

/// Runs the bounded interactive loop with injected input, frame, and clipboard ports.
///
/// # Errors
///
/// Returns stable failures from session selection, settings, runtime, or rendering.
pub async fn run_with_channels(
    service: &CodingAgentService,
    selection: SessionSelection,
    input: mpsc::Receiver<InputEvent>,
    frames: &mut dyn FrameSink,
    clipboard: &mut dyn Clipboard,
    initial_prompt: Option<String>,
) -> Result<TuiState, CliFailure> {
    Box::pin(run_with_channels_context(
        service,
        selection,
        input,
        frames,
        clipboard,
        InteractiveStartup {
            initial_prompt,
            notices: Vec::new(),
            global_settings_path: None,
        },
    ))
    .await
}

/// Runs the injected interactive loop with an explicit persistent settings target.
///
/// This is the embeddable counterpart of the production TUI startup path. Quick
/// reasoning changes remain session-local; accepted model-and-effort choices may
/// update only the supplied sparse global settings file.
///
/// # Errors
///
/// Returns stable failures from session selection, settings, runtime, or rendering.
pub async fn run_with_channels_with_settings_path(
    service: &CodingAgentService,
    selection: SessionSelection,
    input: mpsc::Receiver<InputEvent>,
    frames: &mut dyn FrameSink,
    clipboard: &mut dyn Clipboard,
    initial_prompt: Option<String>,
    global_settings_path: PathBuf,
) -> Result<TuiState, CliFailure> {
    Box::pin(run_with_channels_context(
        service,
        selection,
        input,
        frames,
        clipboard,
        InteractiveStartup {
            initial_prompt,
            notices: Vec::new(),
            global_settings_path: Some(global_settings_path),
        },
    ))
    .await
}

struct InteractiveStartup {
    initial_prompt: Option<String>,
    notices: Vec<String>,
    global_settings_path: Option<PathBuf>,
}

async fn run_with_channels_context(
    service: &CodingAgentService,
    selection: SessionSelection,
    input: mpsc::Receiver<InputEvent>,
    frames: &mut dyn FrameSink,
    clipboard: &mut dyn Clipboard,
    startup_options: InteractiveStartup,
) -> Result<TuiState, CliFailure> {
    let InteractiveStartup {
        initial_prompt,
        notices,
        global_settings_path,
    } = startup_options;
    let session_id = crate::modes::print::select_session(service, selection).await?;
    let events = service.subscribe(session_id).map_err(CliFailure::from)?;
    let snapshot = service
        .session_snapshot(session_id)
        .await
        .map_err(CliFailure::from)?;
    let startup = StartupContext::new(
        service.workspace().host_path().display().to_string(),
        service.resources().context().len(),
        service.resources().skills().len(),
        service.resources().prompts().len(),
        service.resources().diagnostics().len(),
    );
    let mut state = TuiState::from_snapshot(&snapshot, startup.clone());
    sync_reasoning_projection(&mut state, service);
    for notice in notices {
        let _ = reduce(&mut state, Action::Notify(notice));
    }
    if service.settings().tui.collapse_thinking {
        let _ = reduce(&mut state, Action::ToggleThinking);
    }
    let editor = Editor::new();
    let keymap = KeyMap::from_settings(&service.settings().tui)
        .map_err(|_| CliFailure::new(ExitCategory::TrustOrConfig, "keybindings are invalid"))?;
    let commands = command_catalog(service)?;
    let activity_timer = RunActivityTimer::from_state(
        StatusIndicator::needs_elapsed_tick(&state),
        state.run_elapsed_seconds(),
        time::Instant::now(),
    );
    let mut app = InteractiveApp {
        service,
        session_id,
        events,
        input,
        frames,
        clipboard,
        state,
        editor,
        keymap,
        commands,
        startup,
        global_settings_path,
        pending_model_selection: None,
        owned_runs: BTreeSet::new(),
        session_ephemera: BTreeMap::new(),
        frame_scheduler: FrameScheduler::default(),
        activity_timer,
        transcript_cells: Vec::new(),
        quit: false,
    };
    app.sync_committed_history()?;
    app.render()?;
    if let Some(prompt) = initial_prompt {
        app.start_prompt(prompt.clone()).await?;
        app.apply(Action::ShowPendingUserPrompt(prompt)).await?;
    }
    Box::pin(app.run_loop()).await
}

struct InteractiveApp<'a> {
    service: &'a CodingAgentService,
    session_id: SessionId,
    events: mpsc::Receiver<tea_protocol::EventEnvelope>,
    input: mpsc::Receiver<InputEvent>,
    frames: &'a mut dyn FrameSink,
    clipboard: &'a mut dyn Clipboard,
    state: TuiState,
    editor: Editor,
    keymap: KeyMap,
    commands: CommandCatalog,
    startup: StartupContext,
    global_settings_path: Option<PathBuf>,
    pending_model_selection: Option<ModelRef>,
    owned_runs: BTreeSet<SessionId>,
    session_ephemera: BTreeMap<SessionId, SessionEphemera>,
    frame_scheduler: FrameScheduler,
    activity_timer: RunActivityTimer,
    transcript_cells: Vec<CellNode>,
    quit: bool,
}

impl InteractiveApp<'_> {
    async fn run_loop(mut self) -> Result<TuiState, CliFailure> {
        while !self.quit {
            let session_id = self.session_id;
            let frame_deadline = self.frame_scheduler.next_deadline();
            let activity_deadline = self.activity_timer.next_deadline();
            tokio::select! {
                () = time::sleep_until(frame_deadline.unwrap_or_else(time::Instant::now)), if frame_deadline.is_some() => {
                    self.flush_pending_frame()?;
                }
                input = self.input.recv() => {
                    match input {
                        Some(input) => Box::pin(self.handle_input(input)).await?,
                        None => self.quit = true,
                    }
                }
                event = self.events.recv() => {
                    match event {
                        Some(event) => {
                            let owned_command_finished = matches!(
                                event.event(),
                                AgentEvent::ApprovalRequested { .. }
                                    | AgentEvent::RunFinished { .. }
                            );
                            self.apply(Action::Event(Box::new(event))).await?;
                            if owned_command_finished {
                                if self.owned_runs.contains(&session_id) {
                                    let outcome = self.service.wait(session_id).await;
                                    self.finish_owned_run(session_id, outcome).await?;
                                } else {
                                    tokio::task::yield_now().await;
                                }
                                self.flush_pending_reasoning_effort().await?;
                            }
                        }
                        None => self.reconnect().await?,
                    }
                }
                outcome = self.service.wait(session_id), if self.owned_runs.contains(&session_id) => {
                    self.finish_owned_run(session_id, outcome).await?;
                    self.flush_pending_reasoning_effort().await?;
                }
                () = time::sleep_until(activity_deadline.unwrap_or_else(time::Instant::now)), if activity_deadline.is_some() => {
                    let elapsed_seconds = self.activity_timer.advance(time::Instant::now());
                    if elapsed_seconds > 0 {
                        self.apply(Action::AdvanceRunElapsed(elapsed_seconds)).await?;
                    }
                }
            }
        }
        self.flush_pending_frame()?;
        for session_id in self.owned_runs.iter().copied().collect::<Vec<_>>() {
            let _ = self.service.abort(session_id).await;
        }
        Ok(self.state)
    }

    async fn handle_input(&mut self, input: InputEvent) -> Result<(), CliFailure> {
        match input {
            InputEvent::Resize { width, height } => {
                self.apply(Action::Resize { width, height }).await?;
            }
            InputEvent::Focus(_) => {}
            InputEvent::Paste(_) if self.state.approval.is_some() => {}
            InputEvent::Paste(text)
                if self
                    .state
                    .overlay
                    .as_ref()
                    .and_then(Overlay::selector)
                    .is_some() =>
            {
                let selector = self
                    .state
                    .overlay
                    .as_mut()
                    .and_then(Overlay::selector_mut)
                    .expect("selector checked above");
                let mut query = selector.query().to_owned();
                query.push_str(&text);
                selector.set_query(&query);
                self.state.bump_generation();
                self.render()?;
            }
            InputEvent::Paste(text) => {
                if self.editor.insert_paste(&text).is_err() {
                    self.notify("paste exceeds the editor size limit").await?;
                } else {
                    self.sync_editor().await?;
                }
            }
            InputEvent::Key(key) if self.state.approval.is_some() => {
                self.handle_approval_key(key).await?;
            }
            InputEvent::Key(key) if self.state.overlay.is_some() => {
                self.handle_overlay_key(key).await?;
            }
            InputEvent::Key(key) => Box::pin(self.handle_editor_key(key)).await?,
        }
        Ok(())
    }

    async fn handle_approval_key(&mut self, key: KeyEvent) -> Result<(), CliFailure> {
        if self.state.approval_submitting() {
            if self.keymap.resolve(key, false) == Some(BindingAction::Exit) {
                self.quit = true;
            }
            return Ok(());
        }
        match key.code {
            KeyCode::Left | KeyCode::Up | KeyCode::BackTab => {
                let choice = self.state.approval_choice().previous();
                self.apply(Action::SelectApproval(choice)).await?;
            }
            KeyCode::Right | KeyCode::Down | KeyCode::Tab => {
                let choice = self.state.approval_choice().next();
                self.apply(Action::SelectApproval(choice)).await?;
            }
            KeyCode::Char('1') => {
                self.apply(Action::SelectApproval(ApprovalChoice::AllowOnce))
                    .await?;
            }
            KeyCode::Char('2') => {
                self.apply(Action::SelectApproval(ApprovalChoice::AllowSession))
                    .await?;
            }
            KeyCode::Char('3') => {
                self.apply(Action::SelectApproval(ApprovalChoice::Deny))
                    .await?;
            }
            KeyCode::Enter => self.submit_approval().await?,
            _ if self.keymap.resolve(key, false) == Some(BindingAction::Exit) => {
                self.quit = true;
            }
            _ => {}
        }
        Ok(())
    }

    async fn submit_approval(&mut self) -> Result<(), CliFailure> {
        if self.owns_current_run() || self.state.approval_submitting() {
            return Ok(());
        }
        let Some(approval_id) = self.state.approval().map(|approval| approval.approval_id) else {
            return Ok(());
        };
        let decision = self.state.approval_choice().decision();
        match self.service.approve(self.session_id, approval_id, decision) {
            Ok(_) => {
                self.owned_runs.insert(self.session_id);
                self.apply(Action::SetApprovalSubmitting(true)).await?;
                self.apply(Action::StartRunActivity).await?;
            }
            Err(error) => self.notify(error.message()).await?,
        }
        Ok(())
    }

    async fn handle_editor_key(&mut self, key: KeyEvent) -> Result<(), CliFailure> {
        if self.handle_transcript_navigation(key).await? {
            return Ok(());
        }
        if key.code == KeyCode::BackTab {
            return self.cycle_reasoning_effort().await;
        }
        let editor_action = resolve_editor_action(key);
        let editor_preempts_binding = matches!(editor_action, Some(EditorAction::DeleteForward))
            && !self.editor.text().is_empty()
            || matches!(editor_action, Some(EditorAction::Yank)) && self.editor.has_kill_buffer();
        if editor_preempts_binding {
            return self
                .handle_editor_action(editor_action.expect("editor action checked above"))
                .await;
        }
        if let Some(action) = self
            .keymap
            .resolve(key, self.owns_current_run() || self.state.running)
        {
            return Box::pin(self.handle_binding(action)).await;
        }
        if let Some(action) = editor_action {
            return self.handle_editor_action(action).await;
        }
        let changed = match key.code {
            KeyCode::Tab => {
                if let Some(completion) = self
                    .commands
                    .complete(self.editor.text(), 1)
                    .into_iter()
                    .next()
                {
                    let draft = self.commands.completion_draft(&completion);
                    self.editor = Editor::with_limit(draft, super::MAX_EDITOR_BYTES)
                        .expect("bounded command completion");
                    true
                } else {
                    false
                }
            }
            KeyCode::Char(character)
                if key
                    .modifiers
                    .contains(KeyModifiers::CONTROL | KeyModifiers::ALT)
                    && !key.modifiers.contains(KeyModifiers::SUPER) =>
            {
                self.editor.insert_char(character).is_ok()
            }
            KeyCode::Char(character)
                if !key.modifiers.intersects(
                    KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SUPER,
                ) =>
            {
                self.editor.insert_char(character).is_ok()
            }
            _ => false,
        };
        if changed {
            self.sync_editor().await?;
        }
        Ok(())
    }

    async fn handle_editor_action(&mut self, action: EditorAction) -> Result<(), CliFailure> {
        let changed = match action {
            EditorAction::InsertNewline => {
                if let Ok(()) = self.editor.insert_newline() {
                    true
                } else {
                    self.notify("editor size limit reached").await?;
                    false
                }
            }
            EditorAction::MoveLeft => {
                self.editor.move_left();
                true
            }
            EditorAction::MoveRight => {
                self.editor.move_right();
                true
            }
            EditorAction::MoveUp if self.editor.text().contains('\n') => {
                self.editor.move_up();
                true
            }
            EditorAction::MoveDown if self.editor.text().contains('\n') => {
                self.editor.move_down();
                true
            }
            EditorAction::MoveUp => self.editor.previous_history(),
            EditorAction::MoveDown => self.editor.next_history(),
            EditorAction::MoveWordLeft => {
                self.editor.move_word_left();
                true
            }
            EditorAction::MoveWordRight => {
                self.editor.move_word_right();
                true
            }
            EditorAction::MoveLineStart { cross_line: true } => {
                self.editor.move_home_or_previous_line();
                true
            }
            EditorAction::MoveLineStart { cross_line: false } => {
                self.editor.move_home();
                true
            }
            EditorAction::MoveLineEnd { cross_line: true } => {
                self.editor.move_end_or_next_line();
                true
            }
            EditorAction::MoveLineEnd { cross_line: false } => {
                self.editor.move_end();
                true
            }
            EditorAction::DeleteBackward => self.editor.delete_backward(),
            EditorAction::DeleteForward => self.editor.delete_forward(),
            EditorAction::DeleteWordBackward => self.editor.delete_word_backward(),
            EditorAction::DeleteWordForward => self.editor.delete_word_forward(),
            EditorAction::KillLineStart => self.editor.kill_to_line_start(),
            EditorAction::KillLineEnd => self.editor.kill_to_line_end(),
            EditorAction::Yank => {
                if let Ok(changed) = self.editor.yank() {
                    changed
                } else {
                    self.notify("editor size limit reached").await?;
                    false
                }
            }
            EditorAction::Undo => self.editor.undo(),
        };
        if changed {
            self.sync_editor().await?;
        }
        Ok(())
    }

    async fn handle_transcript_navigation(&mut self, key: KeyEvent) -> Result<bool, CliFailure> {
        let page_rows = usize::from((self.state.viewport_height / 2).max(1));
        let action = match (key.code, key.modifiers) {
            (KeyCode::PageUp, KeyModifiers::NONE) => {
                Some(Action::ScrollTranscriptUp { rows: page_rows })
            }
            (KeyCode::PageDown, KeyModifiers::NONE) => {
                Some(Action::ScrollTranscriptDown { rows: page_rows })
            }
            (KeyCode::Up, KeyModifiers::SHIFT) => Some(Action::ScrollTranscriptUp { rows: 1 }),
            (KeyCode::Down, KeyModifiers::SHIFT) => Some(Action::ScrollTranscriptDown { rows: 1 }),
            (KeyCode::End, KeyModifiers::CONTROL) => Some(Action::FollowTranscriptTail),
            _ => None,
        };
        if let Some(action) = action {
            self.apply(action).await?;
            return Ok(true);
        }
        Ok(false)
    }

    async fn handle_binding(&mut self, action: BindingAction) -> Result<(), CliFailure> {
        match action {
            BindingAction::Submit | BindingAction::Steer => {
                let Ok(skill) = self.commands.parse_skill_mention(self.editor.text()) else {
                    self.notify("skill invocation is invalid").await?;
                    return Ok(());
                };
                let Some(text) = self.editor.submit() else {
                    return Ok(());
                };
                self.sync_editor().await?;
                if let Some(command) = skill {
                    self.handle_skill_invocation(&command, &text).await?;
                } else if text.starts_with('/') {
                    self.handle_command(&text).await?;
                } else if !self.state.attachments().is_empty()
                    && (self.owns_current_run() || self.state.running)
                {
                    self.restore_failed_input(text, "image attachments require an idle session")
                        .await?;
                } else if action == BindingAction::Steer {
                    match self.service.steer(self.session_id, text.clone()).await {
                        Ok(_) => self.apply(Action::QueueSteering(text)).await?,
                        Err(error) => self.restore_failed_input(text, error.message()).await?,
                    }
                } else {
                    match self.start_prompt(text.clone()).await {
                        Ok(()) => self.apply(Action::ShowPendingUserPrompt(text)).await?,
                        Err(error) => self.restore_failed_input(text, error.message()).await?,
                    }
                }
            }
            BindingAction::FollowUp => {
                let Ok(skill) = self.commands.parse_skill_mention(self.editor.text()) else {
                    self.notify("skill invocation is invalid").await?;
                    return Ok(());
                };
                let Some(text) = self.editor.submit() else {
                    return Ok(());
                };
                self.sync_editor().await?;
                if let Some(command) = skill {
                    self.handle_skill_invocation(&command, &text).await?;
                } else if !self.state.attachments().is_empty()
                    && (self.owns_current_run() || self.state.running)
                {
                    self.restore_failed_input(text, "image attachments require an idle session")
                        .await?;
                } else {
                    match self.service.follow_up(self.session_id, text.clone()).await {
                        Ok(_) => self.apply(Action::QueueFollowUp(text)).await?,
                        Err(error) => self.restore_failed_input(text, error.message()).await?,
                    }
                }
            }
            BindingAction::Newline => {
                if self.editor.insert_newline().is_err() {
                    self.notify("editor size limit reached").await?;
                } else {
                    self.sync_editor().await?;
                }
            }
            BindingAction::Abort => {
                if self.owns_current_run() || self.state.running {
                    let queued = self.state.take_queued_text();
                    if !queued.is_empty() {
                        self.editor = Editor::with_limit(queued, super::MAX_EDITOR_BYTES)
                            .expect("queued input is bounded");
                        self.sync_editor().await?;
                    }
                    if let Err(error) = self.service.abort(self.session_id).await {
                        self.notify(error.message()).await?;
                    }
                }
            }
            BindingAction::Clear => {
                if self.editor.clear() {
                    self.sync_editor().await?;
                }
            }
            BindingAction::Exit => self.quit = true,
            BindingAction::SelectModel => self.open_model_selector().await?,
            BindingAction::ToggleThinking => self.apply(Action::ToggleThinking).await?,
            BindingAction::ToggleTools => self.apply(Action::ToggleAllTools).await?,
            BindingAction::Copy => self.copy_last_response().await?,
            BindingAction::RetrieveQueued => {
                let queued = self.state.take_queued_text();
                if !queued.is_empty() {
                    self.editor = Editor::with_limit(queued, super::MAX_EDITOR_BYTES)
                        .expect("queued input is bounded");
                    self.sync_editor().await?;
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    async fn handle_command(&mut self, input: &str) -> Result<(), CliFailure> {
        let Ok(command) = self.commands.parse(input) else {
            self.notify("unknown or invalid slash command").await?;
            return Ok(());
        };
        match command {
            SlashCommand::New => {
                let session_id = self
                    .service
                    .create_session()
                    .await
                    .map_err(CliFailure::from)?;
                self.switch_session(session_id, false).await?;
            }
            SlashCommand::Resume(Some(session_id)) => {
                self.switch_session(session_id, true).await?;
            }
            SlashCommand::Resume(None) | SlashCommand::Session => {
                self.open_session_selector().await?;
            }
            SlashCommand::Name(name) => {
                if self.ensure_idle().await? {
                    let name = name
                        .map(|name| SessionName::from_str(&name))
                        .transpose()
                        .map_err(|_| CliFailure::usage("session name is invalid"))?;
                    self.service
                        .name_session(self.session_id, name)
                        .await
                        .map_err(CliFailure::from)?;
                    self.notify("session name updated").await?;
                }
            }
            SlashCommand::Model(Some(model_id)) => {
                let model = self.resolve_model_id(&model_id)?;
                self.select_model(model).await?;
            }
            SlashCommand::Model(None) => self.open_model_selector().await?,
            SlashCommand::Reasoning(Some(reasoning_effort)) => {
                self.request_reasoning_effort(reasoning_effort).await?;
            }
            SlashCommand::Reasoning(None) => self.open_reasoning_selector().await?,
            SlashCommand::Compact => {
                if self.ensure_idle().await? {
                    self.service
                        .compact(self.session_id)
                        .await
                        .map_err(CliFailure::from)?;
                    self.reload_snapshot().await?;
                }
            }
            SlashCommand::Tree => self.open_tree_selector().await?,
            SlashCommand::Fork(message_id) => {
                if self.ensure_idle().await? {
                    let branch_id = uuid::Uuid::now_v7()
                        .hyphenated()
                        .to_string()
                        .parse()
                        .map_err(|_| {
                            CliFailure::new(ExitCategory::Internal, "branch identity failed")
                        })?;
                    self.service
                        .fork(self.session_id, message_id, branch_id)
                        .await
                        .map_err(CliFailure::from)?;
                    self.reload_snapshot().await?;
                }
            }
            SlashCommand::Copy => self.copy_last_response().await?,
            SlashCommand::Image(path) => self.attach_image(input, &path).await?,
            SlashCommand::ImageRemove(index) => {
                self.apply(Action::RemoveAttachment { index }).await?;
            }
            SlashCommand::ImageClear => self.apply(Action::ClearAttachments).await?,
            SlashCommand::Mcp => self.show_mcp_health().await?,
            SlashCommand::Skills => self.show_skills().await?,
            SlashCommand::McpReconnect(server_id) => self.reconnect_mcp(server_id).await?,
            SlashCommand::Help => {
                self.notify(
                    "/new /resume /session /name /model /reasoning /compact /tree /fork /copy /image /mcp /skills /help /quit",
                )
                .await?;
            }
            SlashCommand::Quit => self.quit = true,
            SlashCommand::Template { name, arguments } => {
                let prompt = self
                    .service
                    .resources()
                    .prompts()
                    .iter()
                    .find(|template| template.name() == name)
                    .map(|template| template.expand(&arguments, &BTreeMap::new()))
                    .ok_or_else(|| CliFailure::usage("prompt template is unavailable"))?;
                self.submit_generated(prompt).await?;
            }
            SlashCommand::Skill(command) => {
                self.handle_skill_invocation(&command, input).await?;
            }
        }
        Ok(())
    }

    async fn handle_selector_key(&mut self, key: KeyEvent) -> Result<(), CliFailure> {
        match key.code {
            KeyCode::Esc => {
                self.pending_model_selection = None;
                self.apply(Action::SetOverlay(None)).await?;
            }
            KeyCode::Up => {
                self.state
                    .overlay
                    .as_mut()
                    .and_then(Overlay::selector_mut)
                    .expect("selector exists")
                    .move_previous();
                self.state.bump_generation();
                self.render()?;
            }
            KeyCode::Down => {
                self.state
                    .overlay
                    .as_mut()
                    .and_then(Overlay::selector_mut)
                    .expect("selector exists")
                    .move_next();
                self.state.bump_generation();
                self.render()?;
            }
            KeyCode::Backspace => {
                let selector = self
                    .state
                    .overlay
                    .as_mut()
                    .and_then(Overlay::selector_mut)
                    .expect("selector exists");
                let mut query = selector.query().to_owned();
                if let Some((index, _)) = query.grapheme_indices(true).next_back() {
                    query.truncate(index);
                    selector.set_query(&query);
                    self.state.bump_generation();
                    self.render()?;
                }
            }
            KeyCode::Enter => {
                let value = self.state.selector().and_then(Selector::accept);
                self.apply(Action::SetOverlay(None)).await?;
                if let Some(value) = value {
                    Box::pin(self.accept_selector(value)).await?;
                }
            }
            KeyCode::Char(character)
                if !key.modifiers.intersects(
                    KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SUPER,
                ) =>
            {
                let selector = self
                    .state
                    .overlay
                    .as_mut()
                    .and_then(Overlay::selector_mut)
                    .expect("selector exists");
                let mut query = selector.query().to_owned();
                query.push(character);
                selector.set_query(&query);
                self.state.bump_generation();
                self.render()?;
            }
            _ => {}
        }
        Ok(())
    }

    async fn handle_overlay_key(&mut self, key: KeyEvent) -> Result<(), CliFailure> {
        if self
            .state
            .overlay
            .as_ref()
            .and_then(Overlay::selector)
            .is_some()
        {
            return Box::pin(self.handle_selector_key(key)).await;
        }
        Box::pin(self.handle_command_completion_key(key)).await
    }

    async fn handle_command_completion_key(&mut self, key: KeyEvent) -> Result<(), CliFailure> {
        let tab_accepts = self
            .state
            .overlay
            .as_ref()
            .and_then(Overlay::command_completion)
            .and_then(CommandCompletion::selected)
            .is_some_and(|selected| {
                selected.kind() == super::commands::CommandCompletionKind::Skill
            });
        match key.code {
            KeyCode::Esc => self.apply(Action::SetOverlay(None)).await?,
            KeyCode::Up | KeyCode::BackTab => {
                self.state
                    .overlay
                    .as_mut()
                    .and_then(Overlay::command_completion_mut)
                    .expect("command completion exists")
                    .move_previous();
                self.state.bump_generation();
                self.render()?;
            }
            code if code == KeyCode::Down || (code == KeyCode::Tab && !tab_accepts) => {
                self.state
                    .overlay
                    .as_mut()
                    .and_then(Overlay::command_completion_mut)
                    .expect("command completion exists")
                    .move_next();
                self.state.bump_generation();
                self.render()?;
            }
            KeyCode::Enter | KeyCode::Tab => {
                let selected = self
                    .state
                    .overlay
                    .as_ref()
                    .and_then(Overlay::command_completion)
                    .and_then(CommandCompletion::selected)
                    .cloned();
                self.apply(Action::SetOverlay(None)).await?;
                if let Some(selected) = selected {
                    let draft = self.commands.completion_draft(&selected);
                    self.editor = Editor::with_limit(draft, super::MAX_EDITOR_BYTES)
                        .expect("catalog completion remains bounded");
                    self.sync_editor().await?;
                }
            }
            _ => Box::pin(self.handle_editor_key(key)).await?,
        }
        Ok(())
    }

    async fn accept_selector(&mut self, value: SelectorValue) -> Result<(), CliFailure> {
        match value {
            SelectorValue::Session(session_id) => {
                self.switch_session(session_id, true).await?;
            }
            SelectorValue::Model(model_id) => self.select_model(model_id).await?,
            SelectorValue::Reasoning(reasoning_effort) => {
                if let Some(model_id) = self.pending_model_selection.take() {
                    self.select_model_and_reasoning(model_id, reasoning_effort)
                        .await?;
                } else {
                    self.request_reasoning_effort(reasoning_effort).await?;
                }
            }
            SelectorValue::Branch(branch_id) => {
                let snapshot = self
                    .service
                    .session_snapshot(self.session_id)
                    .await
                    .map_err(CliFailure::from)?;
                let tree = session_tree(&snapshot);
                if let Some(branch) = tree
                    .branches()
                    .iter()
                    .find(|branch| branch.branch_id() == branch_id)
                {
                    let source = branch
                        .source_branch_id()
                        .map_or_else(|| "root".to_owned(), |source| source.to_string());
                    let marker = if branch.is_active() {
                        "active"
                    } else {
                        "inactive"
                    };
                    self.notify(format!(
                        "{marker} branch {branch_id} · source {source} · from {} · leaf {}",
                        branch.from_record_id(),
                        branch.leaf_record_id()
                    ))
                    .await?;
                }
            }
        }
        Ok(())
    }

    async fn open_session_selector(&mut self) -> Result<(), CliFailure> {
        let entries = self
            .service
            .list_sessions()
            .await
            .map_err(CliFailure::from)?;
        let items = session_list(&entries).into_iter().map(|entry| {
            let name = entry
                .name()
                .map_or_else(|| entry.session_id().to_string(), str::to_owned);
            SelectorItem::new(
                format!("{name} · {} messages", entry.message_count()),
                SelectorValue::Session(entry.session_id()),
            )
        });
        let selector = Selector::new("sessions", items)
            .map_err(|_| CliFailure::new(ExitCategory::Internal, "session selector failed"))?;
        self.apply(Action::SetOverlay(Some(Overlay::Selector(selector))))
            .await
    }

    async fn open_model_selector(&mut self) -> Result<(), CliFailure> {
        let items = self
            .service
            .models()
            .into_iter()
            .map(|model| SelectorItem::new(model.to_string(), SelectorValue::Model(model)));
        let mut selector = Selector::new("models", items)
            .map_err(|_| CliFailure::new(ExitCategory::Internal, "model selector failed"))?;
        if let Some(model) = self.state.model_ref() {
            selector.select_value(&SelectorValue::Model(model.clone()));
        }
        self.apply(Action::SetOverlay(Some(Overlay::Selector(selector))))
            .await
    }

    async fn open_reasoning_selector(&mut self) -> Result<(), CliFailure> {
        let model = self
            .state
            .model_ref()
            .cloned()
            .ok_or_else(|| CliFailure::usage("session has no selected model"))?;
        self.open_reasoning_selector_for(model, false).await
    }

    async fn open_reasoning_selector_for(
        &mut self,
        model_ref: ModelRef,
        persist_model: bool,
    ) -> Result<(), CliFailure> {
        let model = self
            .service
            .model_spec(&model_ref)
            .ok_or_else(|| CliFailure::usage("selected model is unavailable"))?;
        let efforts = model.reasoning_profile().map_or_else(
            || vec![ReasoningEffort::Off],
            |profile| profile.supported_efforts().to_vec(),
        );
        let current = model
            .resolve_reasoning(
                self.state
                    .pending_reasoning_effort()
                    .or(self.state.reasoning_effort()),
            )
            .map_or(
                ReasoningEffort::Off,
                tea_model::ReasoningResolution::effective,
            );
        let items = efforts
            .into_iter()
            .map(|effort| SelectorItem::new(effort.as_str(), SelectorValue::Reasoning(effort)));
        let mut selector = Selector::new("reasoning effort", items)
            .map_err(|_| CliFailure::new(ExitCategory::Internal, "reasoning selector failed"))?;
        selector.select_value(&SelectorValue::Reasoning(current));
        self.pending_model_selection = persist_model.then_some(model_ref);
        self.apply(Action::SetOverlay(Some(Overlay::Selector(selector))))
            .await
    }

    async fn open_tree_selector(&mut self) -> Result<(), CliFailure> {
        let snapshot = self
            .service
            .session_snapshot(self.session_id)
            .await
            .map_err(CliFailure::from)?;
        let tree = session_tree(&snapshot);
        let items = tree.branches().iter().map(|branch| {
            let marker = if branch.is_active() { "*" } else { " " };
            let source = branch
                .source_branch_id()
                .map_or_else(|| "root".to_owned(), |source| source.to_string());
            SelectorItem::new(
                format!(
                    "{marker} {} · {source} · {}",
                    branch.branch_id(),
                    branch.leaf_record_id()
                ),
                SelectorValue::Branch(branch.branch_id()),
            )
        });
        let selector = Selector::new("branches", items)
            .map_err(|_| CliFailure::new(ExitCategory::Internal, "branch selector failed"))?;
        self.apply(Action::SetOverlay(Some(Overlay::Selector(selector))))
            .await
    }

    async fn select_model(&mut self, model: ModelRef) -> Result<(), CliFailure> {
        if self.ensure_idle().await? {
            self.open_reasoning_selector_for(model, true).await?;
        }
        Ok(())
    }

    fn resolve_model_id(&self, model_id: &tea_protocol::ModelId) -> Result<ModelRef, CliFailure> {
        let mut matches = self
            .service
            .models()
            .into_iter()
            .filter(|model| model.model_id() == model_id);
        let model = matches
            .next()
            .ok_or_else(|| CliFailure::usage("selected model is unavailable"))?;
        if matches.next().is_some() {
            return Err(CliFailure::usage(
                "model id is ambiguous; choose a provider-qualified model from the selector",
            ));
        }
        Ok(model)
    }

    async fn request_reasoning_effort(
        &mut self,
        requested: ReasoningEffort,
    ) -> Result<(), CliFailure> {
        if self.owns_current_run() || self.state.running {
            let effective = self.resolve_current_reasoning_effort(requested)?;
            if effective != requested {
                self.notify(format!(
                    "reasoning {} adjusted to {} for the selected model",
                    requested.as_str(),
                    effective.as_str()
                ))
                .await?;
            }
            self.apply(Action::SetPendingReasoningEffort(Some(effective)))
                .await?;
            self.notify(format!(
                "reasoning {} queued for the next turn",
                effective.as_str()
            ))
            .await?;
            return Ok(());
        }
        self.apply_reasoning_effort_now(requested).await.map(drop)
    }

    fn resolve_current_reasoning_effort(
        &self,
        requested: ReasoningEffort,
    ) -> Result<ReasoningEffort, CliFailure> {
        let model_ref = self
            .state
            .model_ref()
            .ok_or_else(|| CliFailure::usage("session has no selected model"))?;
        let model = self
            .service
            .model_spec(model_ref)
            .ok_or_else(|| CliFailure::usage("selected model is unavailable"))?;
        Ok(model
            .resolve_reasoning(Some(requested))
            .expect("an explicit effort always resolves")
            .effective())
    }

    async fn apply_reasoning_effort_now(
        &mut self,
        requested: ReasoningEffort,
    ) -> Result<ReasoningEffort, CliFailure> {
        let outcome = self
            .service
            .set_reasoning_effort(self.session_id, requested)
            .await
            .map_err(CliFailure::from)?;
        let RuntimeCommandOutcome::ConfigurationChanged {
            reasoning_effort: Some(effective),
            requested_reasoning_effort,
            ..
        } = outcome
        else {
            return Err(CliFailure::new(
                ExitCategory::Internal,
                "reasoning change returned an unexpected result",
            ));
        };
        self.apply(Action::SetPendingReasoningEffort(None)).await?;
        self.reload_snapshot().await?;
        if let Some(requested) = requested_reasoning_effort {
            self.notify(format!(
                "reasoning {} adjusted to {} for the selected model",
                requested.as_str(),
                effective.as_str()
            ))
            .await?;
        }
        Ok(effective)
    }

    async fn flush_pending_reasoning_effort(&mut self) -> Result<(), CliFailure> {
        if self.owns_current_run() || self.state.running {
            return Ok(());
        }
        let Some(effort) = self.state.pending_reasoning_effort() else {
            return Ok(());
        };
        if let Err(error) = self.apply_reasoning_effort_now(effort).await {
            self.notify(format!(
                "queued reasoning change failed: {}",
                error.message()
            ))
            .await?;
        }
        Ok(())
    }

    async fn finish_owned_run(
        &mut self,
        session_id: SessionId,
        outcome: Result<RuntimeCommandOutcome, tea_coding::CodingError>,
    ) -> Result<(), CliFailure> {
        self.owned_runs.remove(&session_id);
        self.apply(Action::FinishRunActivity).await?;
        self.apply(Action::SetApprovalSubmitting(false)).await?;
        match outcome {
            Ok(RuntimeCommandOutcome::RunCompleted { .. }) => self.reload_snapshot().await,
            Ok(_) => {
                self.notify("runtime command returned an unexpected result")
                    .await
            }
            Err(error) => self.notify(error.message()).await,
        }
    }

    async fn cycle_reasoning_effort(&mut self) -> Result<(), CliFailure> {
        let model_ref = self
            .state
            .model_ref()
            .ok_or_else(|| CliFailure::usage("session has no selected model"))?;
        let model = self
            .service
            .model_spec(model_ref)
            .ok_or_else(|| CliFailure::usage("selected model is unavailable"))?;
        let supported = model.reasoning_profile().map_or_else(
            || vec![ReasoningEffort::Off],
            |profile| {
                profile
                    .supported_efforts()
                    .iter()
                    .copied()
                    .filter(|effort| *effort <= ReasoningEffort::High)
                    .collect::<Vec<_>>()
            },
        );
        let Some(first) = supported.first().copied() else {
            return self
                .notify("selected model has no shortcut reasoning levels")
                .await;
        };
        let current = self
            .state
            .pending_reasoning_effort()
            .or(self.state.reasoning_effort())
            .or_else(|| {
                model
                    .reasoning_profile()
                    .map(tea_model::ReasoningProfile::default_effort)
            })
            .unwrap_or(ReasoningEffort::Off);
        let next = supported
            .iter()
            .position(|effort| *effort == current)
            .map_or(first, |index| supported[(index + 1) % supported.len()]);
        self.request_reasoning_effort(next).await
    }

    async fn select_model_and_reasoning(
        &mut self,
        model_ref: ModelRef,
        requested: ReasoningEffort,
    ) -> Result<(), CliFailure> {
        if !self.ensure_idle().await? {
            return Ok(());
        }
        self.service
            .set_model(self.session_id, model_ref.clone())
            .await
            .map_err(CliFailure::from)?;
        let outcome = self
            .service
            .set_reasoning_effort(self.session_id, requested)
            .await
            .map_err(CliFailure::from)?;
        let RuntimeCommandOutcome::ConfigurationChanged {
            reasoning_effort: Some(effective),
            requested_reasoning_effort,
            ..
        } = outcome
        else {
            return Err(CliFailure::new(
                ExitCategory::Internal,
                "model reasoning change returned an unexpected result",
            ));
        };
        self.reload_snapshot().await?;
        if let Some(requested) = requested_reasoning_effort {
            self.notify(format!(
                "reasoning {} adjusted to {} for {model_ref}",
                requested.as_str(),
                effective.as_str()
            ))
            .await?;
        }
        self.persist_model_defaults(model_ref, effective).await
    }

    async fn persist_model_defaults(
        &mut self,
        model: ModelRef,
        effort: ReasoningEffort,
    ) -> Result<(), CliFailure> {
        let Some(path) = self.global_settings_path.clone() else {
            return Ok(());
        };
        let result = tokio::task::spawn_blocking(move || {
            persist_global_model_settings(&path, &model, effort)
        })
        .await;
        if !matches!(result, Ok(Ok(()))) {
            self.notify("session updated, but global model defaults could not be saved")
                .await?;
        }
        Ok(())
    }

    async fn switch_session(
        &mut self,
        session_id: SessionId,
        open: bool,
    ) -> Result<(), CliFailure> {
        let runtime_state = if open {
            self.service
                .open_session(session_id)
                .await
                .map_err(CliFailure::from)?
        } else {
            self.service
                .snapshot(session_id)
                .await
                .map_err(CliFailure::from)?
        };
        let events = self
            .service
            .subscribe(session_id)
            .map_err(CliFailure::from)?;
        let snapshot = self
            .service
            .session_snapshot(session_id)
            .await
            .map_err(CliFailure::from)?;
        let viewport = (self.state.viewport_width, self.state.viewport_height);
        let preferences = self.state.preferences.clone();
        let previous_ephemera = self.state.take_session_ephemera();
        self.session_ephemera
            .insert(self.session_id, previous_ephemera);
        self.session_id = session_id;
        self.events = events;
        let mut state = TuiState::from_snapshot(&snapshot, self.startup.clone());
        state.viewport_width = viewport.0;
        state.viewport_height = viewport.1;
        state.preferences = preferences;
        state.running = runtime_state.is_running();
        if let Some(ephemera) = self.session_ephemera.remove(&session_id) {
            state.restore_session_ephemera(ephemera);
        }
        sync_reasoning_projection(&mut state, self.service);
        self.state = state;
        self.sync_editor().await
    }

    async fn start_prompt(&mut self, text: String) -> Result<(), CliFailure> {
        let has_attachments = !self.state.attachments().is_empty();
        if has_attachments {
            self.ensure_image_capability()?;
            let mut content = Vec::with_capacity(self.state.attachments().len() + 1);
            content.push(
                ContentBlock::text(text)
                    .map_err(|_| CliFailure::usage("prompt text is invalid"))?,
            );
            content.extend(self.state.attachment_blocks());
            self.service
                .prompt_content(self.session_id, content)
                .map_err(CliFailure::from)?;
            self.apply(Action::ClearAttachments).await?;
        } else {
            self.service
                .prompt(self.session_id, text)
                .map_err(CliFailure::from)?;
        }
        self.owned_runs.insert(self.session_id);
        self.apply(Action::StartRunActivity).await
    }

    async fn attach_image(&mut self, input: &str, path: &str) -> Result<(), CliFailure> {
        if let Err(error) = self.ensure_image_capability() {
            return self
                .restore_failed_input(input.to_owned(), error.message())
                .await;
        }
        let path = Path::new(path);
        let workspace_path = path
            .is_relative()
            .then(|| self.service.workspace().host_path().join(path));
        let path = workspace_path.as_deref().unwrap_or(path);
        let attachment = match load_local_image(path) {
            Ok(attachment) => attachment,
            Err(error) => {
                return self
                    .restore_failed_input(input.to_owned(), &error.to_string())
                    .await;
            }
        };
        if let Err(error) = validate_addition(
            self.state.attachments().len(),
            self.state.attachment_encoded_bytes(),
            attachment.encoded_bytes(),
        ) {
            return self
                .restore_failed_input(input.to_owned(), &error.to_string())
                .await;
        }
        self.apply(Action::AddAttachment(attachment)).await
    }

    fn ensure_image_capability(&self) -> Result<(), CliFailure> {
        let model_ref = self
            .state
            .model_ref()
            .ok_or_else(|| CliFailure::usage("no model is selected for image input"))?;
        let capabilities = self
            .service
            .model_capabilities(model_ref)
            .ok_or_else(|| CliFailure::usage("selected model is unavailable for image input"))?;
        if capabilities.accepts_images() {
            Ok(())
        } else {
            Err(CliFailure::usage(
                "selected model does not support image input",
            ))
        }
    }

    async fn submit_generated(&mut self, prompt: String) -> Result<(), CliFailure> {
        if self.owns_current_run() || self.state.running {
            self.service
                .steer(self.session_id, prompt.clone())
                .await
                .map_err(CliFailure::from)?;
            self.apply(Action::QueueSteering(prompt)).await
        } else {
            self.start_prompt(prompt).await
        }
    }

    async fn submit_skill(
        &mut self,
        invocation: &str,
        instructions: String,
    ) -> Result<(), CliFailure> {
        let invocation = invocation.to_owned();
        let mut content = vec![
            ContentBlock::text(invocation.clone())
                .map_err(|_| CliFailure::usage("skill invocation is invalid"))?,
            ContentBlock::contextual_text(instructions)
                .map_err(|_| CliFailure::usage("skill instructions are invalid"))?,
        ];
        if self.owns_current_run() || self.state.running {
            if !self.state.attachments().is_empty() {
                return self
                    .restore_failed_input(invocation, "image attachments require an idle session")
                    .await;
            }
            self.service
                .follow_up_content(self.session_id, content)
                .await
                .map_err(CliFailure::from)?;
            self.apply(Action::QueueFollowUp(invocation)).await
        } else {
            if !self.state.attachments().is_empty() {
                self.ensure_image_capability()?;
                content.extend(self.state.attachment_blocks());
            }
            self.service
                .prompt_content(self.session_id, content)
                .map_err(CliFailure::from)?;
            self.owned_runs.insert(self.session_id);
            self.apply(Action::ClearAttachments).await?;
            self.apply(Action::StartRunActivity).await?;
            self.apply(Action::ShowPendingUserPrompt(invocation)).await
        }
    }

    async fn handle_skill_invocation(
        &mut self,
        command: &tea::context::SkillCommand,
        invocation: &str,
    ) -> Result<(), CliFailure> {
        let skill = self
            .service
            .resources()
            .load_skill(command)
            .map_err(CliFailure::from)?;
        self.submit_skill(invocation, skill.prompt_fragment()).await
    }

    async fn ensure_idle(&mut self) -> Result<bool, CliFailure> {
        if self.owns_current_run() || self.state.running {
            self.notify("command is unavailable while a run is active")
                .await?;
            Ok(false)
        } else {
            Ok(true)
        }
    }

    fn owns_current_run(&self) -> bool {
        self.owned_runs.contains(&self.session_id)
    }

    async fn copy_last_response(&mut self) -> Result<(), CliFailure> {
        let text = self.state.messages().iter().rev().find_map(assistant_text);
        match text {
            Some(text) => match self.clipboard.copy(&text) {
                Ok(()) => self.notify("copied last assistant response").await?,
                Err(_) => self.notify("system clipboard is unavailable").await?,
            },
            None => self.notify("no assistant response is available").await?,
        }
        Ok(())
    }

    async fn show_mcp_health(&mut self) -> Result<(), CliFailure> {
        let snapshot = self.service.mcp_snapshot().map_err(mcp_failure)?;
        let servers = mcp_servers(&snapshot);
        if servers.is_empty() {
            return self
                .apply(Action::SetMcpHealth(vec![
                    "no MCP servers configured".to_owned(),
                ]))
                .await;
        }
        self.apply(Action::SetMcpHealth(
            servers.iter().map(mcp_health_row).collect::<Vec<_>>(),
        ))
        .await
    }

    async fn show_skills(&mut self) -> Result<(), CliFailure> {
        let rows = if self.service.resources().skills().is_empty() {
            vec!["no skills loaded".to_owned()]
        } else {
            self.service
                .resources()
                .skills()
                .iter()
                .map(skill_catalog_row)
                .collect::<Vec<_>>()
        };
        self.apply(Action::SetSkillCatalog(rows)).await
    }

    async fn reconnect_mcp(&mut self, server_id: McpServerId) -> Result<(), CliFailure> {
        match self.service.reconnect_mcp(&server_id).await {
            Ok(health) => {
                let _ = health;
                self.show_mcp_health().await
            }
            Err(error) => self.notify(mcp_error_message(error)).await,
        }
    }

    async fn restore_failed_input(
        &mut self,
        text: String,
        message: &str,
    ) -> Result<(), CliFailure> {
        self.editor = Editor::with_limit(text, super::MAX_EDITOR_BYTES)
            .expect("previous editor input remains bounded");
        self.sync_editor().await?;
        self.notify(message).await
    }

    async fn sync_editor(&mut self) -> Result<(), CliFailure> {
        let mention = self.commands.skill_mention_token(self.editor.text());
        self.apply(Action::SetComposer {
            text: self.editor.text().to_owned(),
            skill_mention: mention,
        })
        .await?;
        self.sync_command_completion().await
    }

    async fn sync_command_completion(&mut self) -> Result<(), CliFailure> {
        if self
            .state
            .overlay
            .as_ref()
            .and_then(Overlay::selector)
            .is_some()
        {
            return Ok(());
        }
        let editor = self.editor.text();
        let completion = if self.state.approval().is_none()
            && matches!(editor.as_bytes().first(), Some(b'/' | b'$'))
            && !editor.chars().any(char::is_whitespace)
        {
            let completion = CommandCompletion::new(self.commands.complete(editor, 16));
            completion.should_show(editor).then_some(completion)
        } else {
            None
        };
        self.apply(Action::SetOverlay(
            completion.map(Overlay::CommandCompletion),
        ))
        .await
    }

    async fn notify(&mut self, message: impl Into<String>) -> Result<(), CliFailure> {
        self.apply(Action::Notify(message.into())).await
    }

    async fn reload_snapshot(&mut self) -> Result<(), CliFailure> {
        let snapshot = self
            .service
            .session_snapshot(self.session_id)
            .await
            .map_err(CliFailure::from)?;
        self.apply(Action::SnapshotLoaded(Box::new(snapshot))).await
    }

    async fn reconnect(&mut self) -> Result<(), CliFailure> {
        self.events = self
            .service
            .subscribe(self.session_id)
            .map_err(CliFailure::from)?;
        self.apply(Action::Reconnected).await
    }

    async fn apply(&mut self, action: Action) -> Result<(), CliFailure> {
        let stream_delta = is_stream_delta(&action);
        let was_running = StatusIndicator::needs_elapsed_tick(&self.state);
        let mut effects = reduce(&mut self.state, action);
        let requires_snapshot = effects
            .iter()
            .any(|effect| matches!(effect, Effect::ReloadSnapshot { .. }));
        if requires_snapshot {
            let snapshot = self
                .service
                .session_snapshot(self.session_id)
                .await
                .map_err(CliFailure::from)?;
            effects.extend(reduce(
                &mut self.state,
                Action::SnapshotLoaded(Box::new(snapshot)),
            ));
        }
        if sync_reasoning_projection(&mut self.state, self.service)
            && !effects.contains(&Effect::Render)
        {
            effects.push(Effect::Render);
        }
        let now = time::Instant::now();
        self.activity_timer.sync(
            was_running,
            StatusIndicator::needs_elapsed_tick(&self.state),
            self.state.run_elapsed_seconds(),
            now,
        );
        self.sync_committed_history()?;
        if effects.contains(&Effect::Render)
            && self
                .frame_scheduler
                .request(stream_delta && !requires_snapshot, now)
        {
            self.render()?;
        }
        Ok(())
    }

    fn sync_committed_history(&mut self) -> Result<(), CliFailure> {
        let next = Presentation::from_state(&self.state).history().to_vec();
        if next == self.transcript_cells {
            return Ok(());
        }

        let output = if next.starts_with(&self.transcript_cells) {
            self.frames
                .insert_history_cells(&next[self.transcript_cells.len()..])
        } else {
            self.frames.replace_history_cells(&next)
        };
        output.map_err(|_| {
            CliFailure::new(ExitCategory::Internal, "terminal history insertion failed")
        })?;
        self.transcript_cells = next;
        Ok(())
    }

    fn flush_pending_frame(&mut self) -> Result<(), CliFailure> {
        if self.frame_scheduler.take_pending() {
            self.render()?;
        }
        Ok(())
    }

    fn render(&mut self) -> Result<(), CliFailure> {
        self.frames
            .render(&self.state, self.editor.cursor_byte())
            .map_err(|_| CliFailure::new(ExitCategory::Internal, "terminal render failed"))?;
        self.frame_scheduler.record_frame(time::Instant::now());
        Ok(())
    }
}

fn is_stream_delta(action: &Action) -> bool {
    matches!(
        action,
        Action::Event(envelope)
            if matches!(envelope.event(), AgentEvent::MessageDelta { .. })
    )
}

fn command_catalog(service: &CodingAgentService) -> Result<CommandCatalog, CliFailure> {
    let catalog = CommandCatalog::new(
        service
            .resources()
            .prompts()
            .iter()
            .map(tea_coding::resources::PromptTemplate::name),
        service
            .resources()
            .skills()
            .iter()
            .map(|skill| skill.metadata().id().as_str()),
    )
    .map_err(|_| CliFailure::new(ExitCategory::TrustOrConfig, "command catalog is invalid"))?;
    Ok(
        catalog.with_skill_descriptions(service.resources().skills().iter().map(|skill| {
            (
                skill.metadata().id().as_str(),
                skill.metadata().description(),
            )
        })),
    )
}

fn mcp_failure(error: tea_mcp::McpError) -> CliFailure {
    CliFailure::new(ExitCategory::Internal, mcp_error_message(error))
}

fn mcp_error_message(error: tea_mcp::McpError) -> &'static str {
    match error.code() {
        tea_mcp::McpErrorCode::Descriptor
        | tea_mcp::McpErrorCode::Identity
        | tea_mcp::McpErrorCode::StaleCatalog => {
            "MCP catalog changed; close and rebuild the CLI service"
        }
        tea_mcp::McpErrorCode::Unavailable => "MCP server is unavailable for reconnect",
        tea_mcp::McpErrorCode::Cancellation => "MCP reconnect was cancelled",
        _ => "MCP service operation failed",
    }
}

fn mcp_health_row(server: &crate::session_views::McpServerView) -> String {
    const MAX_TUI_MCP_TOOLS: usize = 8;

    let mut tools = server
        .tools()
        .iter()
        .take(MAX_TUI_MCP_TOOLS)
        .cloned()
        .collect::<Vec<_>>();
    if server.tools().len() > tools.len() {
        tools.push(format!("+{} more", server.tools().len() - tools.len()));
    }
    let tools = if tools.is_empty() {
        "no enabled tools".to_owned()
    } else {
        tools.join(", ")
    };
    let code = server
        .code()
        .map(|code| format!("; {code:?}"))
        .unwrap_or_default();
    format!(
        "{}: {:?}{code}; restarts {}; {tools}",
        server.server_id(),
        server.state(),
        server.restart_count(),
    )
}

fn skill_catalog_row(skill: &tea_coding::resources::DiscoveredSkill) -> String {
    let visibility = if skill.model_invocable() {
        skill.source().label().to_owned()
    } else {
        format!("{}; explicit-only", skill.source().label())
    };
    format!(
        "{} [{}] - {}",
        skill.metadata().id(),
        visibility,
        skill.metadata().description()
    )
}

fn assistant_text(message: &CanonicalMessage) -> Option<String> {
    let CanonicalMessage::Assistant { content, .. } = message else {
        return None;
    };
    let text = content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            ContentBlock::ContextualText { .. }
            | ContentBlock::Thinking { .. }
            | ContentBlock::Image { .. }
            | ContentBlock::ToolCall { .. }
            | ContentBlock::HostedTool { .. }
            | ContentBlock::Citation { .. } => None,
        })
        .collect::<String>();
    (!text.is_empty()).then_some(text)
}

fn interactive_initial_prompt(
    args: &CliArgs,
    bootstrap: &CliBootstrap,
) -> Result<Option<String>, CliFailure> {
    if args.prompt.is_empty() {
        return Ok(None);
    }
    if args.prompt.len() > 128 {
        return Err(CliFailure::usage("too many prompt arguments"));
    }
    let mut prompt = String::new();
    for value in &args.prompt {
        let part = if let Some(path) = value.strip_prefix('@') {
            bootstrap.read_prompt_file(args, path)?
        } else {
            value.clone()
        };
        if prompt.len().saturating_add(part.len()).saturating_add(1) > super::MAX_EDITOR_BYTES {
            return Err(CliFailure::usage("prompt exceeds input size limit"));
        }
        if !prompt.is_empty() {
            prompt.push('\n');
        }
        prompt.push_str(&part);
    }
    Ok(Some(prompt))
}

struct TerminalFrameSink {
    guard: TerminalGuard,
    terminal: Terminal<CrosstermBackend<io::Stdout>>,
    renderer: Renderer,
    theme: Theme,
    viewport: ViewportMode,
    transcript_cells: Vec<CellNode>,
    pending_history_cells: Vec<CellNode>,
    rebuild_history: bool,
    initial_clear: bool,
}

impl TerminalFrameSink {
    fn new(viewport: &str, reduced_motion: bool) -> Result<Self, CliFailure> {
        let capabilities = TerminalCapabilities::detect(reduced_motion);
        let options = terminal_options(viewport, capabilities)?;
        let guard =
            TerminalGuard::enter(CrosstermDriver::new(io::stdout()), options).map_err(|_| {
                CliFailure::new(ExitCategory::Internal, "terminal initialization failed")
            })?;
        let mut backend = CrosstermBackend::new(io::stdout());
        let screen_size = backend.size().map_err(|_| {
            CliFailure::new(ExitCategory::Internal, "terminal initialization failed")
        })?;
        let cursor_position = if options.viewport == ViewportMode::Inline {
            backend.get_cursor_position().unwrap_or(Position::ORIGIN)
        } else {
            Position::ORIGIN
        };
        let mut terminal = Terminal::with_cursor_position(
            backend,
            screen_size,
            cursor_position,
            options.hyperlinks,
        );
        if options.viewport == ViewportMode::Fullscreen {
            terminal.set_viewport_area(Rect::new(0, 0, screen_size.width, screen_size.height));
        }
        Ok(Self {
            guard,
            terminal,
            renderer: Renderer::new(),
            theme: Theme::for_capabilities(capabilities),
            viewport: options.viewport,
            transcript_cells: Vec::new(),
            pending_history_cells: Vec::new(),
            rebuild_history: false,
            initial_clear: options.viewport == ViewportMode::Inline,
        })
    }

    fn render_workspace_trust(
        &mut self,
        prompt: &WorkspaceTrustPrompt,
        choice: WorkspaceTrustChoice,
    ) -> io::Result<()> {
        self.guard.begin_frame()?;
        let draw = self.draw_workspace_trust(prompt, choice);
        let end = self.guard.end_frame(FRAME_DEADLINE);
        draw.and(end)
    }

    fn draw_workspace_trust(
        &mut self,
        prompt: &WorkspaceTrustPrompt,
        choice: WorkspaceTrustChoice,
    ) -> io::Result<()> {
        let screen_size = self.terminal.size()?;
        if screen_size.width == 0 || screen_size.height == 0 {
            return Ok(());
        }
        let lines = workspace_trust_panel(
            prompt.workspace(),
            usize::from(screen_size.width),
            choice.should_trust(),
            &self.theme,
        );
        let height = u16::try_from(lines.len())
            .unwrap_or(u16::MAX)
            .min(screen_size.height);
        match self.viewport {
            ViewportMode::Inline => {
                if std::mem::take(&mut self.initial_clear) {
                    self.terminal.clear_visible_screen()?;
                    self.terminal.set_viewport_area(Rect::new(
                        0,
                        screen_size.height.saturating_sub(height),
                        screen_size.width,
                        height,
                    ));
                } else {
                    Self::update_inline_viewport(&mut self.terminal, height, screen_size)?;
                }
            }
            ViewportMode::Fullscreen => {
                let area = Rect::new(0, 0, screen_size.width, screen_size.height);
                if area != self.terminal.viewport_area {
                    self.terminal.clear_after_position(Position::ORIGIN)?;
                    self.terminal.set_viewport_area(area);
                }
            }
        }
        self.terminal.draw(|frame| {
            let area = frame.area();
            let visible_start = lines.len().saturating_sub(usize::from(area.height));
            let (buffer, _) = frame.buffers_mut();
            draw_lines(&lines[visible_start..], area, buffer);
        })
    }

    fn clear_inline_viewport(&mut self) -> io::Result<()> {
        if self.viewport == ViewportMode::Inline {
            self.terminal.clear()?;
        }
        Ok(())
    }

    fn update_inline_viewport(
        terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
        height: u16,
        screen_size: Size,
    ) -> io::Result<()> {
        let terminal_height_shrank = screen_size.height < terminal.last_known_screen_size.height;
        let terminal_height_grew = screen_size.height > terminal.last_known_screen_size.height;
        let viewport_was_bottom_aligned =
            terminal.viewport_area.bottom() == terminal.last_known_screen_size.height;
        let previous_area = terminal.viewport_area;

        let mut area = terminal.viewport_area;
        area.height = height.min(screen_size.height);
        area.width = screen_size.width;
        if area.bottom() > screen_size.height {
            let scroll_by = area.bottom() - screen_size.height;
            if !terminal_height_shrank && area.top() > 0 {
                terminal.scroll_region_up(0..area.top(), scroll_by)?;
            }
            area.y = screen_size.height - area.height;
        } else if terminal_height_grew && viewport_was_bottom_aligned {
            area.y = screen_size.height - area.height;
        }

        if area != previous_area {
            terminal.clear_after_position(Position::new(0, previous_area.y.min(area.y)))?;
            terminal.set_viewport_area(area);
        }
        Ok(())
    }

    fn draw_inline(&mut self, state: &TuiState, cursor_byte: usize) -> io::Result<()> {
        let screen_size = self.terminal.size()?;
        let screen_changed = screen_size != self.terminal.last_known_screen_size;
        let height = self.renderer.inline_height(
            state,
            screen_size.width,
            screen_size.height,
            &self.theme,
            cursor_byte,
        );
        let initial_frame = std::mem::take(&mut self.initial_clear);
        if initial_frame {
            self.terminal.clear_visible_screen()?;
            self.terminal.set_viewport_area(Rect::new(
                0,
                screen_size.height.saturating_sub(height),
                screen_size.width,
                height.min(screen_size.height),
            ));
        } else {
            Self::update_inline_viewport(&mut self.terminal, height, screen_size)?;
        }

        if (!initial_frame && screen_changed) || self.rebuild_history {
            self.terminal.clear_scrollback_and_visible_screen()?;
            let mut area = self.terminal.viewport_area;
            area.y = 0;
            self.terminal.set_viewport_area(area);
            let lines =
                self.renderer
                    .history_lines(&self.transcript_cells, screen_size.width, &self.theme);
            super::insert_history::insert_history_lines(&mut self.terminal, &lines)?;
            self.pending_history_cells.clear();
            self.rebuild_history = false;
        } else if !self.pending_history_cells.is_empty() {
            let lines = self.renderer.history_lines(
                &self.pending_history_cells,
                screen_size.width,
                &self.theme,
            );
            super::insert_history::insert_history_lines(&mut self.terminal, &lines)?;
            self.pending_history_cells.clear();
        }

        let terminal = &mut self.terminal;
        let renderer = &mut self.renderer;
        let theme = &self.theme;
        terminal.draw(|frame| {
            let area = frame.area();
            let (buffer, hyperlinks) = frame.buffers_mut();
            renderer.render_inline_with_cursor(state, area, buffer, hyperlinks, theme, cursor_byte);
            if let Some((column, row)) =
                renderer.cursor_position(state, area.width, area.height, cursor_byte)
            {
                frame.set_cursor_position((area.x + column, area.y + row));
            }
        })
    }

    fn draw_fullscreen(&mut self, state: &TuiState, cursor_byte: usize) -> io::Result<()> {
        let size = self.terminal.size()?;
        let area = Rect::new(0, 0, size.width, size.height);
        if area != self.terminal.viewport_area {
            self.terminal.clear_after_position(Position::ORIGIN)?;
            self.terminal.set_viewport_area(area);
        }
        let terminal = &mut self.terminal;
        let renderer = &mut self.renderer;
        let theme = &self.theme;
        terminal.draw(|frame| {
            let area = frame.area();
            let (buffer, hyperlinks) = frame.buffers_mut();
            renderer.render_with_cursor_and_hyperlinks(
                state,
                area,
                buffer,
                hyperlinks,
                theme,
                cursor_byte,
            );
            if let Some((column, row)) =
                renderer.cursor_position(state, area.width, area.height, cursor_byte)
            {
                frame.set_cursor_position((area.x + column, area.y + row));
            }
        })
    }
}

fn terminal_options(
    viewport: &str,
    capabilities: TerminalCapabilities,
) -> Result<TerminalOptions, CliFailure> {
    let viewport = match viewport {
        "fullscreen" => ViewportMode::Fullscreen,
        "inline" => ViewportMode::Inline,
        _ => {
            return Err(CliFailure::new(
                ExitCategory::TrustOrConfig,
                "viewport is invalid",
            ));
        }
    };
    Ok(TerminalOptions {
        title: capabilities
            .supports_title()
            .then_some(TerminalTitle::default()),
        hyperlinks: capabilities.supports_hyperlinks(),
        viewport,
        synchronized_output: capabilities.supports_synchronized_output(),
        focus_events: capabilities.supports_focus_events(),
        mouse_capture: false,
        keyboard_enhancement: true,
        cursor_visible: true,
    })
}

impl FrameSink for TerminalFrameSink {
    fn insert_history_cells(&mut self, cells: &[CellNode]) -> io::Result<()> {
        self.transcript_cells.extend_from_slice(cells);
        if self.viewport == ViewportMode::Inline {
            self.pending_history_cells.extend_from_slice(cells);
        }
        Ok(())
    }

    fn replace_history_cells(&mut self, cells: &[CellNode]) -> io::Result<()> {
        self.transcript_cells.clear();
        self.transcript_cells.extend_from_slice(cells);
        self.pending_history_cells.clear();
        self.rebuild_history = self.viewport == ViewportMode::Inline;
        Ok(())
    }

    fn render(&mut self, state: &TuiState, cursor_byte: usize) -> io::Result<()> {
        self.guard.begin_frame()?;
        let draw = match self.viewport {
            ViewportMode::Fullscreen => self.draw_fullscreen(state, cursor_byte),
            ViewportMode::Inline => self.draw_inline(state, cursor_byte),
        };
        let end = self.guard.end_frame(FRAME_DEADLINE);
        draw.and(end)
    }
}

#[cfg(test)]
mod frame_scheduler_tests {
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use tokio::time::Instant;

    use super::{
        FrameScheduler, RUN_ACTIVITY_TICK, RunActivityTimer, STREAM_FRAME_CADENCE,
        WorkspaceTrustChoice, WorkspaceTrustKeyAction, terminal_options,
        workspace_trust_key_action,
    };
    use crate::tui::TerminalCapabilities;

    #[test]
    fn stream_bursts_coalesce_without_dropping_the_latest_content() {
        let mut scheduler = FrameScheduler::default();
        let start = Instant::now();
        let mut visible = String::new();
        let mut latest = "first".to_owned();

        assert!(scheduler.request(true, start));
        visible.clone_from(&latest);
        scheduler.record_frame(start);

        latest.push_str(" second");
        assert!(!scheduler.request(true, start + STREAM_FRAME_CADENCE / 4));
        latest.push_str(" final");
        assert!(!scheduler.request(true, start + STREAM_FRAME_CADENCE / 2));
        assert_eq!(
            scheduler.next_deadline(),
            Some(start + STREAM_FRAME_CADENCE)
        );

        assert!(scheduler.take_pending());
        visible.clone_from(&latest);
        scheduler.record_frame(start + STREAM_FRAME_CADENCE);
        assert_eq!(visible, "first second final");
        assert!(scheduler.next_deadline().is_none());
    }

    #[test]
    fn non_stream_frames_cancel_a_pending_stream_frame() {
        let mut scheduler = FrameScheduler::default();
        let start = Instant::now();
        scheduler.record_frame(start);

        assert!(!scheduler.request(true, start + STREAM_FRAME_CADENCE / 4));
        assert!(scheduler.request(false, start + STREAM_FRAME_CADENCE / 2));
        assert!(!scheduler.take_pending());
    }

    #[test]
    fn run_activity_timer_starts_at_acceptance_and_catches_up_missed_ticks() {
        let start = Instant::now();
        let mut timer = RunActivityTimer::from_state(false, 0, start);
        assert!(timer.next_deadline().is_none());

        timer.sync(false, true, 0, start);
        assert_eq!(timer.next_deadline(), Some(start + RUN_ACTIVITY_TICK));
        assert_eq!(timer.advance(start + RUN_ACTIVITY_TICK / 2), 0);
        assert_eq!(timer.advance(start + RUN_ACTIVITY_TICK * 3), 3);
        assert_eq!(timer.next_deadline(), Some(start + RUN_ACTIVITY_TICK * 4));

        timer.sync(true, false, 3, start + RUN_ACTIVITY_TICK * 3);
        assert!(timer.next_deadline().is_none());
    }

    #[test]
    fn run_activity_timer_resumes_from_existing_elapsed_state() {
        let now = Instant::now();
        let timer = RunActivityTimer::from_state(true, 5, now);
        assert_eq!(timer.next_deadline(), Some(now + RUN_ACTIVITY_TICK));
    }

    #[test]
    fn production_options_leave_mouse_scrollback_to_the_terminal() {
        let interactive =
            TerminalCapabilities::from_environment(Some("xterm-256color"), None, false, false);
        let fullscreen = terminal_options("fullscreen", interactive).unwrap();
        let inline = terminal_options("inline", interactive).unwrap();
        assert!(!fullscreen.mouse_capture);
        assert!(!inline.mouse_capture);
        assert!(fullscreen.keyboard_enhancement);
        assert!(inline.keyboard_enhancement);
    }

    #[test]
    fn workspace_trust_requires_an_explicit_acceptance() {
        let choice = WorkspaceTrustChoice::default();
        assert_eq!(choice, WorkspaceTrustChoice::Exit);
        assert_eq!(
            workspace_trust_key_action(choice, KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            WorkspaceTrustKeyAction::Submit(false)
        );

        let selected =
            workspace_trust_key_action(choice, KeyEvent::new(KeyCode::Up, KeyModifiers::NONE));
        assert_eq!(
            selected,
            WorkspaceTrustKeyAction::Select(WorkspaceTrustChoice::Trust)
        );
        assert_eq!(
            workspace_trust_key_action(
                WorkspaceTrustChoice::Trust,
                KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)
            ),
            WorkspaceTrustKeyAction::Submit(true)
        );
        assert_eq!(
            workspace_trust_key_action(
                WorkspaceTrustChoice::Trust,
                KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL)
            ),
            WorkspaceTrustKeyAction::Submit(false)
        );
    }
}
