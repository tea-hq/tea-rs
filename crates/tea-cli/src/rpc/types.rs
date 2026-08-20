use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tea_coding::{CodingError, CodingErrorCode};
use tea_mcp::{McpError, McpErrorCode, McpServerId};
use tea_policy::WorkspaceId;
use tea_protocol::{
    ApprovalDecision, ApprovalId, BranchId, CommandId, EventEnvelope, MessageId, ModelRef,
    SessionId, SessionSequence,
};

use crate::session_views::{
    McpServerView, SessionListItem, SessionSnapshotPage, SessionStateView, SessionStatsView,
    SessionTreeView,
};

/// Current independent RPC schema version.
pub const RPC_VERSION: &str = "1.0";
const MAX_REQUEST_ID_BYTES: usize = 128;

/// Bounded client-supplied response correlation identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RpcRequestId(String);

impl RpcRequestId {
    /// Creates a bounded, control-free request identity.
    ///
    /// # Errors
    ///
    /// Rejects empty, oversized, or control-containing values.
    pub fn new(value: impl Into<String>) -> Result<Self, RpcError> {
        let value = value.into();
        if value.is_empty()
            || value.len() > MAX_REQUEST_ID_BYTES
            || value.chars().any(char::is_control)
        {
            return Err(RpcError::new(
                RpcErrorCode::InvalidRequest,
                "request id is invalid",
            ));
        }
        Ok(Self(value))
    }

    /// Returns the correlation text.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for RpcRequestId {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Serialize for RpcRequestId {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for RpcRequestId {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Self::new(String::deserialize(deserializer)?)
            .map_err(|_| serde::de::Error::custom("request id is invalid"))
    }
}

/// One strict versioned client request.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct RpcRequest {
    rpc_version: String,
    #[serde(default)]
    id: Option<RpcRequestId>,
    #[serde(flatten)]
    request: RpcRequestKind,
}

impl RpcRequest {
    /// Returns the optional client correlation identity.
    #[must_use]
    pub const fn id(&self) -> Option<&RpcRequestId> {
        self.id.as_ref()
    }

    /// Splits the request after validating the independent RPC version.
    ///
    /// # Errors
    ///
    /// Returns `unsupported_version` for any schema other than 1.0.
    pub fn into_parts(self) -> Result<(Option<RpcRequestId>, RpcRequestKind), RpcError> {
        if self.rpc_version != RPC_VERSION {
            return Err(RpcError::new(
                RpcErrorCode::UnsupportedVersion,
                "RPC version is unsupported",
            ));
        }
        Ok((self.id, self.request))
    }
}

/// Supported command and host-query request families.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type", content = "payload", rename_all = "snake_case")]
pub enum RpcRequestKind {
    /// Create and select a new durable session.
    NewSession {},
    /// Attach and select an existing durable session.
    OpenSession {
        /// Durable session identity.
        #[serde(rename = "sessionId")]
        session_id: SessionId,
    },
    /// Set or clear the selected session display name.
    NameSession {
        /// Optional bounded display name.
        name: Option<String>,
    },
    /// Submit one user prompt as an owned long-running command.
    Prompt {
        /// User text.
        text: String,
    },
    /// Steer the active run.
    Steer {
        /// Bounded steering text.
        text: String,
    },
    /// Queue a user follow-up.
    FollowUp {
        /// Bounded user text.
        text: String,
    },
    /// Cancel the active run.
    Abort {},
    /// Resolve a durable approval.
    ResolveApproval {
        /// Pending approval identity.
        #[serde(rename = "approvalId")]
        approval_id: ApprovalId,
        /// Supported bounded decision.
        decision: ApprovalDecision,
    },
    /// Select a model for future turns.
    SetModel {
        /// Registered provider-qualified model selector.
        model: ModelRef,
    },
    /// Compact the selected session.
    Compact {},
    /// Fork and activate a new branch.
    Fork {
        /// Durable source message.
        #[serde(rename = "fromMessageId")]
        from_message_id: MessageId,
        /// Client-assigned branch identity.
        #[serde(rename = "branchId")]
        branch_id: BranchId,
    },
    /// List durable sessions.
    ListSessions {},
    /// Query compact selected-session state.
    QueryState {},
    /// Query bounded canonical records after a durable cursor.
    QuerySnapshot {
        /// Last durable sequence already held by the client.
        #[serde(default, rename = "afterSequence")]
        after_sequence: Option<SessionSequence>,
        /// Requested record count, clamped to the server maximum.
        #[serde(default = "default_snapshot_limit")]
        limit: usize,
    },
    /// Query transcript statistics.
    QueryStats {},
    /// Query the append-only branch tree.
    QueryTree {},
    /// List registered provider models.
    ListModels {},
    /// List safe MCP health and frozen catalog metadata.
    ListMcpServers {},
    /// Reconnect one MCP server only when discovery matches the frozen catalog.
    ReconnectMcp {
        /// Configured MCP server identity.
        #[serde(rename = "serverId")]
        server_id: McpServerId,
    },
}

const fn default_snapshot_limit() -> usize {
    32
}

/// Stable safe RPC failure classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RpcErrorCode {
    /// A complete LF frame was not valid request JSON.
    ParseError,
    /// Request fields or state were invalid.
    InvalidRequest,
    /// The independent RPC version is unsupported.
    UnsupportedVersion,
    /// A requested session or artifact does not exist.
    NotFound,
    /// The requested session already owns conflicting work.
    Busy,
    /// Policy rejected the operation.
    PolicyDenied,
    /// Durable state could not be read or committed.
    Persistence,
    /// Provider execution or selection failed.
    Provider,
    /// Work was cancelled.
    Cancelled,
    /// An internal boundary failed.
    Internal,
}

/// Bounded path- and secret-independent RPC error.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RpcError {
    code: RpcErrorCode,
    message: String,
}

impl RpcError {
    /// Creates one safe error with bounded text.
    #[must_use]
    pub fn new(code: RpcErrorCode, message: impl Into<String>) -> Self {
        let mut message = message.into();
        if message.is_empty() {
            "RPC request failed".clone_into(&mut message);
        }
        message.retain(|character| !character.is_control() || character == ' ');
        if message.len() > 512 {
            let mut boundary = 512;
            while !message.is_char_boundary(boundary) {
                boundary -= 1;
            }
            message.truncate(boundary);
        }
        Self { code, message }
    }

    /// Returns the stable code.
    #[must_use]
    pub const fn code(&self) -> RpcErrorCode {
        self.code
    }
}

impl From<CodingError> for RpcError {
    fn from(error: CodingError) -> Self {
        let code = match error.code() {
            CodingErrorCode::InvalidInput | CodingErrorCode::InvalidRequest => {
                RpcErrorCode::InvalidRequest
            }
            CodingErrorCode::NotFound => RpcErrorCode::NotFound,
            CodingErrorCode::ProjectNotTrusted | CodingErrorCode::PolicyDenied => {
                RpcErrorCode::PolicyDenied
            }
            CodingErrorCode::Persistence => RpcErrorCode::Persistence,
            CodingErrorCode::Credential
            | CodingErrorCode::Authentication
            | CodingErrorCode::PermissionDenied
            | CodingErrorCode::RateLimited
            | CodingErrorCode::ContextOverflow
            | CodingErrorCode::Unavailable
            | CodingErrorCode::Transport => RpcErrorCode::Provider,
            CodingErrorCode::Cancelled => RpcErrorCode::Cancelled,
            CodingErrorCode::Runtime | CodingErrorCode::Internal => RpcErrorCode::Internal,
        };
        Self::new(code, error.message())
    }
}

impl From<McpError> for RpcError {
    fn from(error: McpError) -> Self {
        let (code, message) = match error.code() {
            McpErrorCode::Configuration | McpErrorCode::PolicyDeclaration => {
                (RpcErrorCode::InvalidRequest, "MCP configuration is invalid")
            }
            McpErrorCode::Descriptor | McpErrorCode::Identity | McpErrorCode::StaleCatalog => (
                RpcErrorCode::InvalidRequest,
                "MCP catalog changed; close and rebuild the CLI service",
            ),
            McpErrorCode::Unavailable => (RpcErrorCode::Busy, "MCP server is unavailable"),
            McpErrorCode::Cancellation => (RpcErrorCode::Cancelled, "MCP operation was cancelled"),
            _ => (RpcErrorCode::Internal, "MCP service operation failed"),
        };
        Self::new(code, message)
    }
}

/// Successful or failed response payload correlated to one request.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum RpcResponse {
    /// An owned prompt/approval command was accepted.
    CommandAccepted {
        /// Server-generated command identity.
        #[serde(rename = "commandId")]
        command_id: CommandId,
        /// Affected session.
        #[serde(rename = "sessionId")]
        session_id: SessionId,
    },
    /// A synchronous canonical mutation completed.
    CommandCompleted {
        /// Affected session.
        #[serde(rename = "sessionId")]
        session_id: SessionId,
    },
    /// The active session changed.
    SessionSelected {
        /// Compact authoritative state.
        state: SessionStateView,
        /// Clients must query a durable snapshot after rebinding.
        #[serde(rename = "resnapshotRequired")]
        resnapshot_required: bool,
    },
    /// Durable session catalog result.
    Sessions {
        /// Stable latest-updated ordering from `SessionCatalog`.
        sessions: Vec<SessionListItem>,
    },
    /// Compact selected-session state.
    State {
        /// Authoritative host state.
        state: SessionStateView,
    },
    /// Bounded canonical replay page.
    Snapshot {
        /// Durable records and tail cursor.
        snapshot: SessionSnapshotPage,
    },
    /// Rebuildable transcript statistics.
    Stats {
        /// Aggregate counts.
        stats: SessionStatsView,
    },
    /// Append-only branch tree.
    Tree {
        /// Stable branch projection.
        tree: SessionTreeView,
    },
    /// Registered models.
    Models {
        /// Provider-advertised model selectors.
        models: Vec<ModelRef>,
    },
    /// Safe MCP lifecycle and frozen-catalog projection.
    McpServers {
        /// Servers in canonical server-ID order.
        servers: Vec<McpServerView>,
    },
    /// Stable safe request failure.
    Error {
        /// Machine-readable error.
        error: RpcError,
    },
}

/// One server output frame: correlated response or asynchronous canonical event.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RpcOutput {
    /// Initial connection metadata.
    Ready {
        /// Independent RPC schema version.
        #[serde(rename = "rpcVersion")]
        rpc_version: &'static str,
        /// Initially selected session.
        #[serde(rename = "sessionId")]
        session_id: SessionId,
        /// Safe canonical workspace identity.
        #[serde(rename = "workspaceId")]
        workspace_id: WorkspaceId,
    },
    /// Correlated request response.
    Response {
        /// Independent RPC schema version.
        #[serde(rename = "rpcVersion")]
        rpc_version: &'static str,
        /// Optional client correlation identity.
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<RpcRequestId>,
        /// Typed response payload.
        payload: RpcResponse,
    },
    /// Asynchronous canonical event.
    Event {
        /// Independent RPC schema version.
        #[serde(rename = "rpcVersion")]
        rpc_version: &'static str,
        /// Unchanged canonical runtime event envelope.
        payload: EventEnvelope,
    },
    /// Owned command task reached its terminal result.
    CommandFinished {
        /// Independent RPC schema version.
        #[serde(rename = "rpcVersion")]
        rpc_version: &'static str,
        /// Accepted command identity.
        #[serde(rename = "commandId")]
        command_id: CommandId,
        /// Affected session.
        #[serde(rename = "sessionId")]
        session_id: SessionId,
        /// Safe failure when the command did not complete successfully.
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<RpcError>,
    },
    /// The event subscription was replaced after a gap or rebind.
    ResnapshotRequired {
        /// Independent RPC schema version.
        #[serde(rename = "rpcVersion")]
        rpc_version: &'static str,
        /// Session requiring canonical replay.
        #[serde(rename = "sessionId")]
        session_id: SessionId,
        /// Current durable tail at notification time.
        #[serde(rename = "tailSequence")]
        tail_sequence: SessionSequence,
    },
}

impl RpcOutput {
    /// Creates initial connection metadata.
    #[must_use]
    pub const fn ready(session_id: SessionId, workspace_id: WorkspaceId) -> Self {
        Self::Ready {
            rpc_version: RPC_VERSION,
            session_id,
            workspace_id,
        }
    }

    /// Creates one correlated response frame.
    #[must_use]
    pub const fn response(id: Option<RpcRequestId>, payload: RpcResponse) -> Self {
        Self::Response {
            rpc_version: RPC_VERSION,
            id,
            payload,
        }
    }

    /// Creates one asynchronous canonical event frame.
    #[must_use]
    pub const fn event(payload: EventEnvelope) -> Self {
        Self::Event {
            rpc_version: RPC_VERSION,
            payload,
        }
    }

    /// Creates one asynchronous owned-command terminal notification.
    #[must_use]
    pub const fn command_finished(
        command_id: CommandId,
        session_id: SessionId,
        error: Option<RpcError>,
    ) -> Self {
        Self::CommandFinished {
            rpc_version: RPC_VERSION,
            command_id,
            session_id,
            error,
        }
    }

    /// Requires the client to rebuild from a durable snapshot page.
    #[must_use]
    pub const fn resnapshot_required(
        session_id: SessionId,
        tail_sequence: SessionSequence,
    ) -> Self {
        Self::ResnapshotRequired {
            rpc_version: RPC_VERSION,
            session_id,
            tail_sequence,
        }
    }

    /// Creates one safe error response.
    #[must_use]
    pub const fn error(id: Option<RpcRequestId>, error: RpcError) -> Self {
        Self::response(id, RpcResponse::Error { error })
    }
}
