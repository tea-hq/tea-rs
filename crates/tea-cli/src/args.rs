use std::path::PathBuf;
use std::str::FromStr;

use clap::{ArgAction, Parser, ValueEnum};
use tea_protocol::{ProfileId, ReasoningEffort, SessionId};
use tea_provider_openai::ApiKey;

/// Stable non-interactive project trust selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum TrustArg {
    /// Use persisted/default policy.
    Default,
    /// Trust project-local resources for this invocation.
    Once,
    /// Trust and persist the canonical workspace decision.
    Persist,
    /// Reject project-local resources and fail.
    Reject,
    /// Ignore project-local resources.
    Ignore,
}

/// Durable session selection shared by headless and future interactive modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionSelection {
    /// Create a new durable session.
    New,
    /// Continue the most recently updated durable session.
    Continue,
    /// Open one explicit durable session.
    Existing(SessionId),
    /// Use an invocation-local in-memory `SQLite` session.
    NoSession,
}

/// Shared command-line contract for the Coding CLI.
#[derive(Debug, Clone, Parser)]
#[command(name = "tea", version = crate::version())]
#[allow(clippy::struct_excessive_bools)] // Clap presence flags map to one validated selection.
pub struct CliArgs {
    /// Run the protocol-neutral Tea app-server over stdin/stdout.
    #[arg(long = "app-server", conflicts_with_all = ["print", "json", "rpc", "prompt"])]
    pub app_server: bool,
    /// Run one script-safe prompt and print only the final assistant text.
    #[arg(long, conflicts_with_all = ["json", "rpc"])]
    pub print: bool,
    /// Stream one mode header and canonical event envelope per JSON/LF line.
    #[arg(long, conflicts_with_all = ["print", "rpc"])]
    pub json: bool,
    /// Run the strict bounded LF-delimited JSON/RPC process interface.
    #[arg(long, conflicts_with_all = ["print", "json"])]
    pub rpc: bool,
    /// Workspace directory (defaults to the process current directory).
    #[arg(long, value_name = "DIR")]
    pub cwd: Option<PathBuf>,
    /// Provider selector.
    #[arg(long)]
    pub provider: Option<String>,
    /// Model selector.
    #[arg(long)]
    pub model: Option<String>,
    /// Invocation-local provider-neutral reasoning effort.
    #[arg(long)]
    pub reasoning_effort: Option<ReasoningEffort>,
    /// Invocation-local provider API key override.
    #[arg(long, value_name = "SECRET")]
    pub api_key: Option<ApiKey>,
    /// Product profile selector.
    #[arg(long, default_value = "coding-agent")]
    pub profile: String,
    /// Active tools, as comma-separated names. Repeatable.
    #[arg(long, value_delimiter = ',', action = ArgAction::Append)]
    pub tools: Vec<String>,
    /// Additional workspace-relative context file. Repeatable.
    #[arg(long = "context-file", value_name = "PATH")]
    pub context_files: Vec<String>,
    /// Open one durable session ID.
    #[arg(long, conflicts_with_all = ["new_session", "continue_session", "no_session"])]
    pub session: Option<String>,
    /// Force a new durable session.
    #[arg(long = "new", conflicts_with_all = ["session", "continue_session", "no_session"])]
    pub new_session: bool,
    /// Continue the most recently updated durable session.
    #[arg(long = "continue", conflicts_with_all = ["session", "new_session", "no_session"])]
    pub continue_session: bool,
    /// Do not persist this invocation's session.
    #[arg(long, conflicts_with_all = ["session", "new_session", "continue_session"])]
    pub no_session: bool,
    /// Absolute application configuration directory.
    #[arg(long, value_name = "DIR")]
    pub config_dir: Option<PathBuf>,
    /// Absolute durable state directory.
    #[arg(long, value_name = "DIR")]
    pub state_dir: Option<PathBuf>,
    /// Absolute declarative resource directory.
    #[arg(long, value_name = "DIR")]
    pub data_dir: Option<PathBuf>,
    /// Explicit `SQLite` path; relative values are resolved beneath state-dir.
    #[arg(long, value_name = "PATH")]
    pub session_db: Option<PathBuf>,
    /// Project-local configuration/resource trust behavior.
    #[arg(long, value_enum, default_value = "default")]
    pub trust: TrustArg,
    /// Increase diagnostic verbosity (stdout remains machine-safe in headless modes).
    #[arg(short, long, action = ArgAction::Count)]
    pub verbose: u8,
    /// Initial prompt text; values beginning with @ load a workspace-relative file.
    #[arg(value_name = "PROMPT")]
    pub prompt: Vec<String>,
}

impl CliArgs {
    /// Resolves and validates the requested session behavior.
    ///
    /// # Errors
    ///
    /// Rejects malformed explicit session IDs.
    pub fn session_selection(&self) -> Result<SessionSelection, crate::CliFailure> {
        if let Some(session) = &self.session {
            return SessionId::from_str(session)
                .map(SessionSelection::Existing)
                .map_err(|_| crate::CliFailure::usage("session identifier is invalid"));
        }
        if self.continue_session {
            Ok(SessionSelection::Continue)
        } else if self.no_session {
            Ok(SessionSelection::NoSession)
        } else {
            Ok(SessionSelection::New)
        }
    }

    /// Resolves the registered profile selector.
    ///
    /// # Errors
    ///
    /// Rejects malformed profile IDs.
    pub fn profile_id(&self) -> Result<ProfileId, crate::CliFailure> {
        ProfileId::from_str(&self.profile)
            .map_err(|_| crate::CliFailure::usage("profile selector is invalid"))
    }
}
