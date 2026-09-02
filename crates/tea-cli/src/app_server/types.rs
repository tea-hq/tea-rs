use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tea_protocol::{ApprovalDecision, ModelRef, SessionId};

/// Independent version of the Tea app-server wire contract.
pub const APP_SERVER_VERSION: &str = "1.0";

const MAX_REQUEST_ID_BYTES: usize = 128;
const MAX_METHOD_BYTES: usize = 128;
const MAX_STRING_BYTES: usize = 1024 * 1024;

/// A bounded JSON-RPC request identifier.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AppRequestId(String);

impl AppRequestId {
    /// Creates a bounded identifier without control characters.
    pub fn new(value: impl Into<String>) -> Result<Self, AppServerError> {
        let value = value.into();
        if value.is_empty()
            || value.len() > MAX_REQUEST_ID_BYTES
            || value.chars().any(char::is_control)
        {
            return Err(AppServerError::invalid_request("request id is invalid"));
        }
        Ok(Self(value))
    }

    /// Returns the identifier text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for AppRequestId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Serialize for AppRequestId {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for AppRequestId {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Self::new(String::deserialize(deserializer)?)
            .map_err(|error| serde::de::Error::custom(error.message))
    }
}

/// One Tea app-server JSON-RPC request.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AppServerRequest {
    /// JSON-RPC version marker.
    #[serde(rename = "jsonrpc")]
    pub jsonrpc: String,
    /// Optional request correlation identifier.
    pub id: Option<AppRequestId>,
    /// Method name.
    pub method: String,
    /// Method parameters, validated by the dispatcher.
    #[serde(default)]
    pub params: Value,
}

impl AppServerRequest {
    /// Validates the JSON-RPC envelope and returns its parameters.
    pub fn validate(self) -> Result<(Option<AppRequestId>, String, Value), AppServerError> {
        if self.jsonrpc != "2.0" {
            return Err(AppServerError::invalid_request(
                "JSON-RPC version must be 2.0",
            ));
        }
        if self.method.is_empty()
            || self.method.len() > MAX_METHOD_BYTES
            || self.method.chars().any(char::is_control)
        {
            return Err(AppServerError::invalid_request("method is invalid"));
        }
        Ok((self.id, self.method, self.params))
    }
}

/// One JSON-RPC response emitted by the app-server.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AppServerResponse {
    /// JSON-RPC version marker.
    #[serde(rename = "jsonrpc")]
    pub jsonrpc: &'static str,
    /// Request correlation identifier.
    pub id: Option<AppRequestId>,
    /// Successful result, when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    /// Failed result, when present.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<AppServerError>,
}

impl AppServerResponse {
    /// Creates a successful response.
    #[must_use]
    pub fn success(id: Option<AppRequestId>, result: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: Some(result),
            error: None,
        }
    }

    /// Creates a failed response.
    #[must_use]
    pub fn failure(id: Option<AppRequestId>, error: AppServerError) -> Self {
        Self {
            jsonrpc: "2.0",
            id,
            result: None,
            error: Some(error),
        }
    }
}

/// One server-to-client JSON-RPC notification.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AppServerNotification {
    /// JSON-RPC version marker.
    #[serde(rename = "jsonrpc")]
    pub jsonrpc: &'static str,
    /// Notification method.
    pub method: &'static str,
    /// Notification parameters.
    pub params: Value,
}

impl AppServerNotification {
    /// Creates an event notification.
    #[must_use]
    pub fn event(params: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            method: "event/session",
            params,
        }
    }

    /// Creates a permission request notification.
    #[must_use]
    pub fn permission(params: Value) -> Self {
        Self {
            jsonrpc: "2.0",
            method: "permission/request",
            params,
        }
    }
}

/// A request sent from the app-server to its adapter.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AppServerRequestMessage {
    /// JSON-RPC version marker.
    #[serde(rename = "jsonrpc")]
    pub jsonrpc: &'static str,
    /// Request correlation identifier.
    pub id: AppRequestId,
    /// Request method.
    pub method: &'static str,
    /// Request parameters.
    pub params: Value,
}

/// The app-server's bounded error object.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AppServerError {
    /// Stable machine-readable code.
    pub code: i32,
    /// Safe diagnostic message.
    pub message: String,
}

impl AppServerError {
    /// Creates an invalid-request error.
    #[must_use]
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Self {
            code: -32600,
            message: bounded_message(message.into()),
        }
    }

    /// Creates a method-not-found error.
    #[must_use]
    pub fn method_not_found() -> Self {
        Self {
            code: -32601,
            message: "method is not supported".to_owned(),
        }
    }

    /// Creates an invalid-params error.
    #[must_use]
    pub fn invalid_params(message: impl Into<String>) -> Self {
        Self {
            code: -32602,
            message: bounded_message(message.into()),
        }
    }

    /// Creates an internal error without exposing implementation details.
    #[must_use]
    pub fn internal() -> Self {
        Self {
            code: -32603,
            message: "app-server internal error".to_owned(),
        }
    }
}

/// App-server handshake parameters.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct InitializeParams {
    /// Client display name.
    pub client_name: String,
    /// Client version.
    pub client_version: String,
    /// Requested app-server contract version.
    pub app_server_version: String,
}

/// App-server handshake result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InitializeResult {
    /// App-server contract version.
    pub app_server_version: &'static str,
    /// Human-readable server name.
    pub server_name: &'static str,
    /// Server version.
    pub server_version: &'static str,
    /// Supported feature flags.
    pub capabilities: AppServerCapabilities,
}

/// Capabilities exposed by the Tea app-server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AppServerCapabilities {
    /// Whether session resume is supported.
    pub session_resume: bool,
    /// Whether session-scoped MCP is supported.
    pub session_mcp: bool,
    /// Whether permission requests are supported.
    pub permission_requests: bool,
}

/// Session creation parameters.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct CreateSessionParams {
    /// Canonical workspace path.
    pub cwd: String,
    /// Optional provider-qualified model.
    pub model: Option<ModelRef>,
    /// Optional Tea permission mode.
    pub mode: Option<String>,
    /// Client-provided MCP servers.
    #[serde(default)]
    pub mcp_servers: Vec<McpServerDescriptor>,
}

/// Stdio MCP descriptor accepted by Tea app-server.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct McpServerDescriptor {
    /// Stable server name.
    pub name: String,
    /// Executable path or command.
    pub command: String,
    /// Process arguments.
    #[serde(default)]
    pub args: Vec<String>,
    /// Explicit environment variables.
    #[serde(default)]
    pub env: Vec<McpEnvironmentVariable>,
}

/// One explicitly supplied MCP environment variable.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct McpEnvironmentVariable {
    /// Variable name.
    pub name: String,
    /// Variable value.
    pub value: String,
}

/// Session creation result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CreateSessionResult {
    /// Tea session identity.
    pub session_id: SessionId,
    /// Current model, when configured.
    pub model: Option<ModelRef>,
    /// Current mode.
    pub mode: String,
}

/// Prompt/turn parameters.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PromptParams {
    /// Target session.
    pub session_id: SessionId,
    /// User prompt text.
    pub text: String,
}

/// Session cancellation parameters.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SessionParams {
    /// Target session.
    pub session_id: SessionId,
}

/// Session configuration parameters.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct SetConfigParams {
    /// Target session.
    pub session_id: SessionId,
    /// Stable configuration identifier.
    pub config_id: String,
    /// New configuration value.
    pub value: String,
}

/// Approval response parameters.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ApprovalResponseParams {
    /// Target session.
    pub session_id: SessionId,
    /// Pending Tea approval identity.
    pub approval_id: tea_protocol::ApprovalId,
    /// Client decision.
    pub decision: ApprovalDecision,
}

/// Canonical event notification parameters.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct EventParams {
    /// Event session identity.
    pub session_id: SessionId,
    /// Canonical event sequence, when available.
    pub sequence: Option<tea_protocol::SessionSequence>,
    /// Serialized Tea event envelope.
    pub event: Value,
}

fn bounded_message(mut value: String) -> String {
    if value.len() > MAX_STRING_BYTES {
        value.truncate(MAX_STRING_BYTES);
    }
    value
}
