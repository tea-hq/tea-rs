use std::{
    collections::BTreeSet,
    ffi::{OsStr, OsString},
    fmt,
    path::{Path, PathBuf},
    str::FromStr,
    time::Duration,
};

use tea_tools::{
    ArgumentResourceResolver, ToolEffect, ToolExecutionSemantics, ToolName, ToolResourceAccess,
};

use crate::{McpError, McpErrorCode, McpRemoteToolName, McpServerId};

/// Maximum bytes in an exact absolute executable path.
pub const MAX_MCP_EXECUTABLE_BYTES: usize = 4 * 1024;
/// Maximum number of exact process arguments.
pub const MAX_MCP_ARGUMENTS: usize = 128;
/// Maximum bytes in one exact process argument.
pub const MAX_MCP_ARGUMENT_BYTES: usize = 16 * 1024;
/// Maximum aggregate bytes across exact process arguments.
pub const MAX_MCP_ARGUMENT_TOTAL_BYTES: usize = 64 * 1024;
/// Maximum number of explicitly inherited environment variable names.
pub const MAX_MCP_ENVIRONMENT_VARIABLES: usize = 64;
/// Maximum bytes in one inherited environment variable name.
pub const MAX_MCP_ENVIRONMENT_NAME_BYTES: usize = 128;
/// Maximum aggregate bytes across inherited environment variable names.
pub const MAX_MCP_ENVIRONMENT_TOTAL_BYTES: usize = 4 * 1024;
/// Hard maximum raw JSONL frame bytes selected by the SDK transport spike.
pub const MAX_MCP_FRAME_BYTES: usize = 1024 * 1024;
/// Hard maximum bytes in one retained remote descriptor.
pub const MAX_MCP_DESCRIPTOR_BYTES: usize = 256 * 1024;
/// Hard maximum bytes in one mapped terminal tool result.
pub const MAX_MCP_RESULT_BYTES: usize = 256 * 1024;
/// Hard maximum retained server stderr bytes.
pub const MAX_MCP_STDERR_BYTES: usize = 64 * 1024;
/// Hard maximum tools admitted from one server descriptor snapshot.
pub const MAX_MCP_TOOLS_PER_SERVER: usize = 256;
/// Hard maximum admitted notifications between lifecycle checkpoints.
pub const MAX_MCP_NOTIFICATIONS: usize = 1_024;
/// Hard maximum progress events accepted for one invocation.
pub const MAX_MCP_PROGRESS_EVENTS: usize = 1_024;
/// Hard maximum concurrent in-flight requests to one server.
pub const MAX_MCP_IN_FLIGHT_REQUESTS: usize = 64;
/// Hard maximum effects in one host tool declaration.
pub const MAX_MCP_TOOL_EFFECTS: usize = 64;
/// Hard maximum argument-derived resources in one host tool declaration.
pub const MAX_MCP_TOOL_RESOURCES: usize = 64;
/// Hard maximum for one startup, handshake, cancellation, or shutdown stage.
pub const MAX_MCP_LIFECYCLE_TIMEOUT: Duration = Duration::from_mins(5);
/// Hard maximum reconnect attempts owned by one explicit operation.
pub const MAX_MCP_RECONNECT_ATTEMPTS: u32 = 8;
/// Hard maximum reconnect backoff.
pub const MAX_MCP_RECONNECT_BACKOFF: Duration = Duration::from_mins(1);

/// Validated stdio process configuration.
#[derive(Clone, PartialEq, Eq)]
pub struct McpStdioConfig {
    executable: PathBuf,
    arguments: Vec<OsString>,
}

impl McpStdioConfig {
    /// Creates an exact shell-free stdio command.
    ///
    /// # Errors
    ///
    /// Rejects non-absolute, oversized, NUL-containing paths or argument vectors.
    pub fn new(
        executable: impl Into<PathBuf>,
        arguments: impl IntoIterator<Item = OsString>,
    ) -> Result<Self, McpError> {
        let executable = executable.into();
        let arguments = arguments.into_iter().collect::<Vec<_>>();
        let (executable_bytes, executable_has_nul) = os_bytes(executable.as_os_str());
        if !executable.is_absolute()
            || executable_bytes == 0
            || executable_bytes > MAX_MCP_EXECUTABLE_BYTES
            || executable_has_nul
            || arguments.len() > MAX_MCP_ARGUMENTS
        {
            return Err(McpError::new(McpErrorCode::Configuration));
        }

        let mut total = 0usize;
        for argument in &arguments {
            let (bytes, has_nul) = os_bytes(argument);
            total = total
                .checked_add(bytes)
                .ok_or_else(|| McpError::new(McpErrorCode::Configuration))?;
            if bytes > MAX_MCP_ARGUMENT_BYTES || total > MAX_MCP_ARGUMENT_TOTAL_BYTES || has_nul {
                return Err(McpError::new(McpErrorCode::Configuration));
            }
        }
        Ok(Self {
            executable,
            arguments,
        })
    }

    /// Returns the exact absolute executable without resolving it through a shell.
    #[must_use]
    pub fn executable(&self) -> &Path {
        &self.executable
    }

    /// Returns the exact ordered argument vector.
    #[must_use]
    pub fn arguments(&self) -> &[OsString] {
        &self.arguments
    }
}

impl fmt::Debug for McpStdioConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("McpStdioConfig")
            .field("executable", &"<redacted>")
            .field("argument_count", &self.arguments.len())
            .finish()
    }
}

/// Supported MCP server transport configuration.
#[derive(Clone, PartialEq, Eq)]
pub enum McpTransportConfig {
    /// A caller-owned child process connected through exact stdio pipes.
    Stdio(McpStdioConfig),
}

impl McpTransportConfig {
    /// Creates a validated stdio transport configuration.
    ///
    /// # Errors
    ///
    /// Rejects invalid executable or argument values before a process can start.
    pub fn stdio(
        executable: impl Into<PathBuf>,
        arguments: impl IntoIterator<Item = OsString>,
    ) -> Result<Self, McpError> {
        Ok(Self::Stdio(McpStdioConfig::new(executable, arguments)?))
    }

    /// Returns the validated stdio values.
    #[must_use]
    pub const fn as_stdio(&self) -> &McpStdioConfig {
        match self {
            Self::Stdio(config) => config,
        }
    }
}

impl fmt::Debug for McpTransportConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stdio(config) => formatter.debug_tuple("Stdio").field(config).finish(),
        }
    }
}

/// Validated per-server protocol and output limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_field_names)] // `max_*` is the stable limit vocabulary.
pub struct McpLimits {
    max_frame_bytes: usize,
    max_descriptor_bytes: usize,
    max_result_bytes: usize,
    max_stderr_bytes: usize,
    max_tools: usize,
    max_notifications: usize,
    max_progress_events: usize,
    max_in_flight_requests: usize,
}

impl McpLimits {
    /// Replaces the raw JSONL frame bound.
    ///
    /// # Errors
    ///
    /// Rejects zero or values over one MiB.
    pub fn with_max_frame_bytes(mut self, value: usize) -> Result<Self, McpError> {
        self.max_frame_bytes = bounded(value, MAX_MCP_FRAME_BYTES)?;
        Ok(self)
    }

    /// Replaces the descriptor byte bound.
    ///
    /// # Errors
    ///
    /// Rejects zero or values over the hard descriptor maximum.
    pub fn with_max_descriptor_bytes(mut self, value: usize) -> Result<Self, McpError> {
        self.max_descriptor_bytes = bounded(value, MAX_MCP_DESCRIPTOR_BYTES)?;
        Ok(self)
    }

    /// Replaces the terminal result byte bound.
    ///
    /// # Errors
    ///
    /// Rejects zero or values over the hard result maximum.
    pub fn with_max_result_bytes(mut self, value: usize) -> Result<Self, McpError> {
        self.max_result_bytes = bounded(value, MAX_MCP_RESULT_BYTES)?;
        Ok(self)
    }

    /// Replaces the private stderr retention bound.
    ///
    /// # Errors
    ///
    /// Rejects zero or values over the hard stderr maximum.
    pub fn with_max_stderr_bytes(mut self, value: usize) -> Result<Self, McpError> {
        self.max_stderr_bytes = bounded(value, MAX_MCP_STDERR_BYTES)?;
        Ok(self)
    }

    /// Replaces the descriptor tool-count bound.
    ///
    /// # Errors
    ///
    /// Rejects zero or values over the per-server hard maximum.
    pub fn with_max_tools(mut self, value: usize) -> Result<Self, McpError> {
        self.max_tools = bounded(value, MAX_MCP_TOOLS_PER_SERVER)?;
        Ok(self)
    }

    /// Replaces the notification admission bound.
    ///
    /// # Errors
    ///
    /// Rejects zero or values over the hard notification maximum.
    pub fn with_max_notifications(mut self, value: usize) -> Result<Self, McpError> {
        self.max_notifications = bounded(value, MAX_MCP_NOTIFICATIONS)?;
        Ok(self)
    }

    /// Replaces the per-invocation progress bound.
    ///
    /// # Errors
    ///
    /// Rejects zero or values over the hard progress maximum.
    pub fn with_max_progress_events(mut self, value: usize) -> Result<Self, McpError> {
        self.max_progress_events = bounded(value, MAX_MCP_PROGRESS_EVENTS)?;
        Ok(self)
    }

    /// Replaces the concurrent in-flight request bound.
    ///
    /// # Errors
    ///
    /// Rejects zero or values over the per-server hard maximum.
    pub fn with_max_in_flight_requests(mut self, value: usize) -> Result<Self, McpError> {
        self.max_in_flight_requests = bounded(value, MAX_MCP_IN_FLIGHT_REQUESTS)?;
        Ok(self)
    }

    /// Returns the raw frame byte bound.
    #[must_use]
    pub const fn max_frame_bytes(self) -> usize {
        self.max_frame_bytes
    }

    /// Returns the remote descriptor byte bound.
    #[must_use]
    pub const fn max_descriptor_bytes(self) -> usize {
        self.max_descriptor_bytes
    }

    /// Returns the terminal result byte bound.
    #[must_use]
    pub const fn max_result_bytes(self) -> usize {
        self.max_result_bytes
    }

    /// Returns the private stderr retention bound.
    #[must_use]
    pub const fn max_stderr_bytes(self) -> usize {
        self.max_stderr_bytes
    }

    /// Returns the descriptor tool-count bound.
    #[must_use]
    pub const fn max_tools(self) -> usize {
        self.max_tools
    }

    /// Returns the notification admission bound.
    #[must_use]
    pub const fn max_notifications(self) -> usize {
        self.max_notifications
    }

    /// Returns the per-invocation progress bound.
    #[must_use]
    pub const fn max_progress_events(self) -> usize {
        self.max_progress_events
    }

    /// Returns the concurrent in-flight request bound.
    #[must_use]
    pub const fn max_in_flight_requests(self) -> usize {
        self.max_in_flight_requests
    }
}

impl Default for McpLimits {
    fn default() -> Self {
        Self {
            max_frame_bytes: MAX_MCP_FRAME_BYTES,
            max_descriptor_bytes: 128 * 1024,
            max_result_bytes: 256 * 1024,
            max_stderr_bytes: 16 * 1024,
            max_tools: 128,
            max_notifications: 128,
            max_progress_events: 256,
            max_in_flight_requests: 16,
        }
    }
}

/// Caller-owned startup, handshake, cancellation, and shutdown deadlines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::struct_field_names)] // Stage names remain explicit beside `Duration` values.
pub struct McpLifecyclePolicy {
    startup_timeout: Duration,
    handshake_timeout: Duration,
    cancellation_timeout: Duration,
    graceful_shutdown_timeout: Duration,
    termination_timeout: Duration,
    kill_timeout: Duration,
}

impl McpLifecyclePolicy {
    /// Creates validated non-zero lifecycle stage deadlines.
    ///
    /// # Errors
    ///
    /// Rejects a zero stage or a stage over five minutes.
    pub fn new(
        startup_timeout: Duration,
        handshake_timeout: Duration,
        cancellation_timeout: Duration,
        graceful_shutdown_timeout: Duration,
        termination_timeout: Duration,
        kill_timeout: Duration,
    ) -> Result<Self, McpError> {
        for timeout in [
            startup_timeout,
            handshake_timeout,
            cancellation_timeout,
            graceful_shutdown_timeout,
            termination_timeout,
            kill_timeout,
        ] {
            if timeout.is_zero() || timeout > MAX_MCP_LIFECYCLE_TIMEOUT {
                return Err(McpError::new(McpErrorCode::Configuration));
            }
        }
        Ok(Self {
            startup_timeout,
            handshake_timeout,
            cancellation_timeout,
            graceful_shutdown_timeout,
            termination_timeout,
            kill_timeout,
        })
    }

    /// Returns the process startup deadline.
    #[must_use]
    pub const fn startup_timeout(self) -> Duration {
        self.startup_timeout
    }

    /// Returns the initialize/initialized handshake deadline.
    #[must_use]
    pub const fn handshake_timeout(self) -> Duration {
        self.handshake_timeout
    }

    /// Returns the cooperative request cancellation deadline.
    #[must_use]
    pub const fn cancellation_timeout(self) -> Duration {
        self.cancellation_timeout
    }

    /// Returns the graceful service and stdin-close deadline.
    #[must_use]
    pub const fn graceful_shutdown_timeout(self) -> Duration {
        self.graceful_shutdown_timeout
    }

    /// Returns the process termination deadline.
    #[must_use]
    pub const fn termination_timeout(self) -> Duration {
        self.termination_timeout
    }

    /// Returns the final forced-kill deadline.
    #[must_use]
    pub const fn kill_timeout(self) -> Duration {
        self.kill_timeout
    }
}

impl Default for McpLifecyclePolicy {
    fn default() -> Self {
        Self {
            startup_timeout: Duration::from_secs(10),
            handshake_timeout: Duration::from_secs(10),
            cancellation_timeout: Duration::from_secs(2),
            graceful_shutdown_timeout: Duration::from_secs(5),
            termination_timeout: Duration::from_secs(2),
            kill_timeout: Duration::from_secs(5),
        }
    }
}

/// Explicit bounded reconnect policy; disabled by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct McpReconnectPolicy {
    max_attempts: u32,
    initial_backoff: Duration,
    max_backoff: Duration,
}

impl McpReconnectPolicy {
    /// Creates an enabled capped reconnect policy.
    ///
    /// # Errors
    ///
    /// Rejects zero attempts/backoffs, too many attempts, excessive backoff, or
    /// an initial backoff greater than the cap.
    pub fn bounded(
        max_attempts: u32,
        initial_backoff: Duration,
        max_backoff: Duration,
    ) -> Result<Self, McpError> {
        if max_attempts == 0
            || max_attempts > MAX_MCP_RECONNECT_ATTEMPTS
            || initial_backoff.is_zero()
            || initial_backoff > max_backoff
            || max_backoff > MAX_MCP_RECONNECT_BACKOFF
        {
            return Err(McpError::new(McpErrorCode::Configuration));
        }
        Ok(Self {
            max_attempts,
            initial_backoff,
            max_backoff,
        })
    }

    /// Returns whether reconnect is explicitly enabled.
    #[must_use]
    pub const fn is_enabled(self) -> bool {
        self.max_attempts > 0
    }

    /// Returns the maximum attempts owned by one operation.
    #[must_use]
    pub const fn max_attempts(self) -> u32 {
        self.max_attempts
    }

    /// Returns the first deterministic backoff.
    #[must_use]
    pub const fn initial_backoff(self) -> Duration {
        self.initial_backoff
    }

    /// Returns the deterministic backoff cap.
    #[must_use]
    pub const fn max_backoff(self) -> Duration {
        self.max_backoff
    }
}

/// Pure argument-to-resource mapping supplied by the host.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct McpArgumentResource {
    argument: String,
    scheme: String,
    access: ToolResourceAccess,
}

impl McpArgumentResource {
    /// Creates a bounded top-level string argument resource mapping.
    ///
    /// # Errors
    ///
    /// Rejects names, schemes, or access values unsupported by the existing
    /// pure tool resource resolver.
    pub fn new(
        argument: impl Into<String>,
        scheme: impl Into<String>,
        access: ToolResourceAccess,
    ) -> Result<Self, McpError> {
        let argument = argument.into();
        let scheme = scheme.into();
        ArgumentResourceResolver::new(&argument, &scheme, access)
            .map_err(|_| McpError::new(McpErrorCode::PolicyDeclaration))?;
        Ok(Self {
            argument,
            scheme,
            access,
        })
    }

    /// Returns the top-level string argument name.
    #[must_use]
    pub fn argument(&self) -> &str {
        &self.argument
    }

    /// Returns the canonical resource scheme.
    #[must_use]
    pub fn scheme(&self) -> &str {
        &self.scheme
    }

    /// Returns the declared access mode.
    #[must_use]
    pub const fn access(&self) -> ToolResourceAccess {
        self.access
    }
}

/// Complete host-owned safety declaration for one enabled remote tool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpToolDeclaration {
    effects: Vec<ToolEffect>,
    resources: Vec<McpArgumentResource>,
    execution: ToolExecutionSemantics,
}

impl McpToolDeclaration {
    /// Creates a canonical bounded host declaration.
    ///
    /// # Errors
    ///
    /// Rejects empty, duplicate, or oversized effects and duplicate or
    /// oversized resource mappings.
    pub fn new(
        effects: impl IntoIterator<Item = ToolEffect>,
        resources: impl IntoIterator<Item = McpArgumentResource>,
        execution: ToolExecutionSemantics,
    ) -> Result<Self, McpError> {
        let mut effects = effects.into_iter().collect::<Vec<_>>();
        let effect_count = effects
            .iter()
            .map(ToolEffect::as_str)
            .collect::<BTreeSet<_>>()
            .len();
        if effects.is_empty()
            || effects.len() > MAX_MCP_TOOL_EFFECTS
            || effect_count != effects.len()
        {
            return Err(McpError::new(McpErrorCode::PolicyDeclaration));
        }
        effects.sort();

        let mut resources = resources.into_iter().collect::<Vec<_>>();
        let resource_count = resources.iter().collect::<BTreeSet<_>>().len();
        if resources.len() > MAX_MCP_TOOL_RESOURCES || resource_count != resources.len() {
            return Err(McpError::new(McpErrorCode::PolicyDeclaration));
        }
        resources.sort();
        Ok(Self {
            effects,
            resources,
            execution,
        })
    }

    /// Returns sorted authoritative host effects.
    #[must_use]
    pub fn effects(&self) -> &[ToolEffect] {
        &self.effects
    }

    /// Returns sorted pure argument resource mappings.
    #[must_use]
    pub fn resources(&self) -> &[McpArgumentResource] {
        &self.resources
    }

    /// Returns idempotency, retry, concurrency, and timeout semantics.
    #[must_use]
    pub const fn execution(&self) -> ToolExecutionSemantics {
        self.execution
    }
}

/// Host policy for one exact remote tool; disabled until a declaration exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct McpToolPolicy {
    remote_name: McpRemoteToolName,
    alias: Option<ToolName>,
    declaration: Option<McpToolDeclaration>,
}

impl McpToolPolicy {
    /// Creates a disabled host policy for one exact remote name.
    #[must_use]
    pub const fn new(remote_name: McpRemoteToolName) -> Self {
        Self {
            remote_name,
            alias: None,
            declaration: None,
        }
    }

    /// Creates an enabled host policy with a complete safety declaration.
    #[must_use]
    pub const fn enabled(remote_name: McpRemoteToolName, declaration: McpToolDeclaration) -> Self {
        Self {
            remote_name,
            alias: None,
            declaration: Some(declaration),
        }
    }

    /// Sets an explicit canonical local alias.
    #[must_use]
    pub fn with_alias(mut self, alias: ToolName) -> Self {
        self.alias = Some(alias);
        self
    }

    /// Returns the exact remote tool name.
    #[must_use]
    pub const fn remote_name(&self) -> &McpRemoteToolName {
        &self.remote_name
    }

    /// Returns the explicit local alias, when configured.
    #[must_use]
    pub const fn alias(&self) -> Option<&ToolName> {
        self.alias.as_ref()
    }

    /// Returns the complete host declaration when this tool is enabled.
    #[must_use]
    pub const fn declaration(&self) -> Option<&McpToolDeclaration> {
        self.declaration.as_ref()
    }

    /// Returns whether this tool may enter a frozen catalog.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.declaration.is_some()
    }

    /// Resolves the explicit alias or a lossless default MCP alias.
    #[must_use]
    pub fn resolved_alias(&self, server_id: &McpServerId) -> Option<ToolName> {
        self.alias.clone().or_else(|| {
            ToolName::from_str(&format!(
                "mcp.{}.{}",
                server_id.as_str(),
                self.remote_name.as_str()
            ))
            .ok()
        })
    }
}

/// Fully validated pure server configuration.
#[derive(Clone, PartialEq, Eq)]
pub struct McpServerConfig {
    id: McpServerId,
    transport: McpTransportConfig,
    inherited_environment: Vec<String>,
    tools: Vec<McpToolPolicy>,
    default_tool_declaration: Option<McpToolDeclaration>,
    limits: McpLimits,
    lifecycle: McpLifecyclePolicy,
    reconnect: McpReconnectPolicy,
}

impl McpServerConfig {
    /// Creates a deterministic server configuration without performing I/O.
    ///
    /// # Errors
    ///
    /// Rejects duplicate or oversized environment/tool collections, duplicate
    /// remote tools or effective aliases, and enabled tools without a lossless
    /// local alias.
    pub fn new(
        id: McpServerId,
        transport: McpTransportConfig,
        mut inherited_environment: Vec<String>,
        mut tools: Vec<McpToolPolicy>,
        limits: McpLimits,
        lifecycle: McpLifecyclePolicy,
        reconnect: McpReconnectPolicy,
    ) -> Result<Self, McpError> {
        validate_environment(&inherited_environment)?;
        if tools.len() > limits.max_tools() || tools.len() > MAX_MCP_TOOLS_PER_SERVER {
            return Err(McpError::new(McpErrorCode::Configuration));
        }

        let mut remote_names = BTreeSet::new();
        let mut aliases = BTreeSet::new();
        for tool in &tools {
            if !remote_names.insert(tool.remote_name.as_str()) {
                return Err(McpError::new(McpErrorCode::Configuration));
            }
            let alias = tool.resolved_alias(&id);
            if tool.is_enabled() && alias.is_none() {
                return Err(McpError::new(McpErrorCode::PolicyDeclaration));
            }
            if let Some(alias) = alias
                && !aliases.insert(alias)
            {
                return Err(McpError::new(McpErrorCode::Configuration));
            }
        }

        inherited_environment.sort_unstable();
        tools.sort_by(|left, right| left.remote_name.cmp(&right.remote_name));
        Ok(Self {
            id,
            transport,
            inherited_environment,
            tools,
            default_tool_declaration: None,
            limits,
            lifecycle,
            reconnect,
        })
    }

    /// Returns the canonical server identity.
    #[must_use]
    pub const fn id(&self) -> &McpServerId {
        &self.id
    }

    /// Returns the validated exact transport configuration.
    #[must_use]
    pub const fn transport(&self) -> &McpTransportConfig {
        &self.transport
    }

    /// Returns sorted explicitly inherited environment variable names.
    #[must_use]
    pub fn inherited_environment(&self) -> &[String] {
        &self.inherited_environment
    }

    /// Returns host tool policies sorted by exact remote name.
    #[must_use]
    pub fn tools(&self) -> &[McpToolPolicy] {
        &self.tools
    }

    /// Sets a fallback declaration for tools discovered from this server.
    ///
    /// This is intended for protocol hosts such as ACP where a client can
    /// provide a server before its tool names are known. Explicit per-tool
    /// policies still take precedence over this fallback.
    #[must_use]
    pub fn with_default_tool_declaration(mut self, declaration: McpToolDeclaration) -> Self {
        self.default_tool_declaration = Some(declaration);
        self
    }

    /// Returns the fallback declaration for otherwise unconfigured tools.
    #[must_use]
    pub const fn default_tool_declaration(&self) -> Option<&McpToolDeclaration> {
        self.default_tool_declaration.as_ref()
    }

    /// Returns the validated protocol and output limits.
    #[must_use]
    pub const fn limits(&self) -> McpLimits {
        self.limits
    }

    /// Returns caller-owned lifecycle deadlines.
    #[must_use]
    pub const fn lifecycle(&self) -> McpLifecyclePolicy {
        self.lifecycle
    }

    /// Returns the explicit reconnect policy.
    #[must_use]
    pub const fn reconnect(&self) -> McpReconnectPolicy {
        self.reconnect
    }
}

impl fmt::Debug for McpServerConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("McpServerConfig")
            .field("id", &self.id)
            .field("transport", &self.transport)
            .field(
                "inherited_environment_count",
                &self.inherited_environment.len(),
            )
            .field("tool_count", &self.tools.len())
            .field(
                "has_default_tool_declaration",
                &self.default_tool_declaration.is_some(),
            )
            .field("limits", &self.limits)
            .field("lifecycle", &self.lifecycle)
            .field("reconnect", &self.reconnect)
            .finish()
    }
}

fn bounded(value: usize, maximum: usize) -> Result<usize, McpError> {
    if value == 0 || value > maximum {
        Err(McpError::new(McpErrorCode::Configuration))
    } else {
        Ok(value)
    }
}

fn validate_environment(names: &[String]) -> Result<(), McpError> {
    if names.len() > MAX_MCP_ENVIRONMENT_VARIABLES {
        return Err(McpError::new(McpErrorCode::Configuration));
    }
    let mut canonical = BTreeSet::new();
    let mut total = 0usize;
    for name in names {
        total = total
            .checked_add(name.len())
            .ok_or_else(|| McpError::new(McpErrorCode::Configuration))?;
        let mut bytes = name.bytes();
        if name.len() > MAX_MCP_ENVIRONMENT_NAME_BYTES
            || total > MAX_MCP_ENVIRONMENT_TOTAL_BYTES
            || !bytes
                .next()
                .is_some_and(|byte| byte.is_ascii_alphabetic() || byte == b'_')
            || !bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
            || !canonical.insert(name.to_ascii_uppercase())
        {
            return Err(McpError::new(McpErrorCode::Configuration));
        }
    }
    Ok(())
}

#[cfg(unix)]
fn os_bytes(value: &OsStr) -> (usize, bool) {
    use std::os::unix::ffi::OsStrExt;

    let bytes = value.as_bytes();
    (bytes.len(), bytes.contains(&0))
}

#[cfg(windows)]
fn os_bytes(value: &OsStr) -> (usize, bool) {
    use std::os::windows::ffi::OsStrExt;

    value
        .encode_wide()
        .fold((0usize, false), |(bytes, nul), unit| {
            (bytes.saturating_add(2), nul || unit == 0)
        })
}

#[cfg(not(any(unix, windows)))]
fn os_bytes(value: &OsStr) -> (usize, bool) {
    let value = value.to_string_lossy();
    (value.len(), value.contains('\0'))
}
