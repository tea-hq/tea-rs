//! Tea's protocol-neutral app-server contract and process adapter.

mod server;
mod types;

pub use server::{run, run_service};
pub use types::{
    APP_SERVER_VERSION, AppRequestId, AppServerCapabilities, AppServerError, AppServerNotification,
    AppServerRequest, AppServerRequestMessage, AppServerResponse, ApprovalResponseParams,
    CreateSessionParams, CreateSessionResult, EventParams, InitializeParams, InitializeResult,
    McpEnvironmentVariable, McpServerDescriptor, PromptParams, SessionParams, SetConfigParams,
};
