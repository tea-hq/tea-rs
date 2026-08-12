#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Workspace-scoped native coding tool adapters for the reference Coding CLI.
//!
//! Native executors are rooted in an explicit [`WorkspaceRoot`] capability;
//! they never accept an unconstrained ambient current directory. Path values
//! from models are bounded, relative, normalized, symlink-contained, and
//! separated into existing-read and prospective-mutation resolutions.

use std::sync::Arc;

use tea_tools::{ToolExecutor, ToolResourceAccess, ToolResourceResolver, ToolSpec, ToolSpecError};

mod bash;
mod edit;
mod edit_diff;
mod error;
mod file;
mod file_error;
#[cfg(test)]
mod file_tests;
mod output;
mod path;
mod read;
mod resource;
mod search;
mod web_fetch;
mod web_search;
mod workspace;
mod write;

pub use bash::{BashConfig, BashOutputDirectory, BashShell, BashTool};
pub use edit::EditTool;
pub use error::{MAX_WORKSPACE_ERROR_MESSAGE_BYTES, WorkspacePathError, WorkspacePathErrorCode};
pub use file::{
    DEFAULT_READ_LINE_LIMIT, MAX_READ_BYTES, MAX_READ_LINE_LIMIT, MAX_WRITE_BYTES,
    read_bounded_utf8,
};
pub use file_error::{FileToolError, FileToolErrorCode};
pub use path::{MAX_WORKSPACE_PATH_BYTES, MAX_WORKSPACE_PATH_COMPONENTS};
pub use read::ReadTool;
pub use resource::WorkspaceFileResourceResolver;
pub use search::{
    DEFAULT_SEARCH_RESULT_LIMIT, FindTool, GrepTool, LsTool, MAX_SEARCH_DEPTH,
    MAX_SEARCH_RESULT_LIMIT,
};
pub use web_fetch::{
    DEFAULT_FETCH_CACHE_ENTRIES, DEFAULT_FETCH_CACHE_ENTRY_BYTES, DEFAULT_FETCH_CACHE_TOTAL_BYTES,
    DEFAULT_FETCH_CACHE_TTL, DEFAULT_FETCH_DECODED_BYTES, DEFAULT_FETCH_HTML_BYTES,
    DEFAULT_FETCH_HTML_ELEMENTS, DEFAULT_FETCH_MAX_CHARS, DEFAULT_FETCH_RESPONSE_BYTES,
    DecodedFetchBody, FETCH_SECURITY_POLICY_VERSION, FetchAddressPolicy, FetchAddressPolicyError,
    FetchBodyDecoder, FetchBodyLimits, FetchCacheConfig, FetchCacheScope, FetchCacheStats,
    FetchContentKind, FetchDnsResolver, FetchFuture, FetchHttpConfig, FetchHttpHeaders,
    FetchHttpLimits, FetchHttpResponse, FetchHttpTimeouts, FetchHttpTransport, FetchProvider,
    FetchProviderError, FetchProviderErrorCode, FetchRedirect, FetchRequest, FetchRequestError,
    FetchResolveFuture, FetchResult, FetchResultCache, FetchRetryDisposition,
    FetchTruncationReason, FetchUrlPolicy, FetchUrlPolicyError, FetchUrlScheme, HttpFetchProvider,
    MAX_FETCH_CACHE_ENTRIES, MAX_FETCH_CACHE_ENTRY_BYTES, MAX_FETCH_CACHE_TOTAL_BYTES,
    MAX_FETCH_CACHE_TTL, MAX_FETCH_DECODED_BYTES, MAX_FETCH_MAX_CHARS, MAX_FETCH_MIME_BYTES,
    MAX_FETCH_REDIRECTS, MAX_FETCH_RESPONSE_BYTES, MAX_FETCH_TITLE_BYTES, MAX_FETCH_URL_BYTES,
    SystemFetchDnsResolver, ValidatedFetchAddresses, ValidatedFetchUrl, WebFetchTool,
};
pub use web_search::{
    DEFAULT_TAVILY_SEARCH_ENDPOINT, DEFAULT_WEB_SEARCH_RESULT_LIMIT, MAX_WEB_SEARCH_QUERY_BYTES,
    MAX_WEB_SEARCH_RESPONSE_BYTES, MAX_WEB_SEARCH_RESULT_LIMIT, SearchFuture, SearchProvider,
    SearchProviderError, SearchProviderErrorCode, SearchRequest, SearchRequestError,
    SearchResponse, SearchResult, TavilyApiKey, TavilySearchConfig, TavilySearchProvider,
    WebSearchTool,
};
pub use workspace::{ResolvedExistingPath, ResolvedMutationPath, WorkspaceRoot};
pub use write::WriteTool;

/// One native client-tool registration ready for a runtime or session builder.
pub type NativeToolRegistration = (
    ToolSpec,
    Arc<dyn ToolResourceResolver>,
    Arc<dyn ToolExecutor>,
);

/// Builds the workspace-confined `read`, `grep`, `find`, and `ls` tool preset.
///
/// # Errors
///
/// Returns an error when a built-in tool specification cannot be constructed.
pub fn read_only_workspace_tools(
    workspace: &WorkspaceRoot,
) -> Result<Vec<NativeToolRegistration>, ToolSpecError> {
    let resolver = || {
        Arc::new(WorkspaceFileResourceResolver::new(ToolResourceAccess::Read))
            as Arc<dyn ToolResourceResolver>
    };
    Ok(vec![
        (
            ReadTool::spec()?,
            resolver(),
            Arc::new(ReadTool::new(workspace.clone())),
        ),
        (
            GrepTool::spec()?,
            resolver(),
            Arc::new(GrepTool::new(workspace.clone())),
        ),
        (
            FindTool::spec()?,
            resolver(),
            Arc::new(FindTool::new(workspace.clone())),
        ),
        (
            LsTool::spec()?,
            resolver(),
            Arc::new(LsTool::new(workspace.clone())),
        ),
    ])
}

/// Returns the package version embedded at compile time.
#[must_use]
pub const fn package_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
