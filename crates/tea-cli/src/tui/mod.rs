//! Pure terminal state reduction, deterministic rendering, bounded effect routing,
//! and panic-safe terminal lifecycle ownership.

mod app;
mod attachment;
mod bottom_pane;
mod cells;
mod clipboard;
mod commands;
mod components;
mod custom_terminal;
mod editor;
mod hyperlink;
mod input;
mod insert_history;
mod keymap;
mod layout;
mod markdown;
mod overlay;
mod presentation;
mod reducer;
mod render;
mod render_output;
mod selectors;
mod state;
mod status;
mod terminal;
mod theme;
mod transcript;

pub use app::{FrameSink, run, run_with_channels, run_with_channels_with_settings_path};
pub use attachment::{
    AttachmentError, ComposerAttachment, MAX_COMPOSER_ATTACHMENTS, MAX_COMPOSER_IMAGE_BASE64_BYTES,
};
pub use clipboard::{Clipboard, MemoryClipboard, SystemClipboard};
pub use commands::{
    CommandCatalog, CommandCompletionItem, CommandCompletionKind, CommandError, SlashCommand,
};
pub use editor::{Editor, EditorError, MAX_EDITOR_BYTES, MAX_EDITOR_HISTORY};
pub use input::{InputEvent, InputPump, spawn_input_pump};
pub use keymap::{BindingAction, KeyMap, KeyMapError};
pub use overlay::{CommandCompletion, Overlay};
pub use presentation::{
    CellContent, CellId, CellLane, CellNode, DecisionCell, DecisionStatus, DiffCell, LifecycleCell,
    LifecycleKind, LifecycleStatus, MessageAuthor, MessageCell, MessageCellFacet, NoticeCell,
    NoticeKind, NoticeSeverity, OutputFormat, PlanCell, PlanStep, PlanStepStatus, Presentation,
    QueuedInputCell, QueuedInputKind, ReasoningCell, ResultCell, SourcesCell, StreamCellFacet,
    TimelineDetail, TimelineDetailKind, TimelineSource, ToolCellFacet,
};
pub use reducer::{
    Action, ActionLoop, ActionSender, DEFAULT_ACTION_CAPACITY, DEFAULT_EFFECT_CAPACITY,
    DispatchError, Effect, EffectExecutor, is_observational, reduce,
};
pub use render::Renderer;
pub use render_output::RenderedLine;
pub use selectors::{Selector, SelectorError, SelectorItem, SelectorValue};
pub use state::{
    ApprovalChoice, ApprovalView, MAX_MCP_HEALTH_ROWS, MAX_NOTIFICATIONS, MAX_OBSERVED_EVENT_IDS,
    MAX_SKILL_CATALOG_ROWS, MAX_VISIBLE_QUEUE_ITEMS, ModelRetryView, StartupContext,
    StreamingBlock, StreamingMessage, ToolProgressView, ToolView, TranscriptViewport, TuiState,
    ViewPreferences,
};
pub use terminal::{
    ColorCapability, CrosstermDriver, TerminalCapabilities, TerminalDriver, TerminalGuard,
    TerminalMode, TerminalOptions, TerminalTitle, ViewportMode,
};
pub use theme::Theme;
