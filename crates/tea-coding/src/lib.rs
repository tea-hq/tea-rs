#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Mode-neutral product assembly for the reference Coding CLI.
//!
//! This crate owns coding-specific configuration, project trust, declarative
//! resources, profile assembly, and session-service behavior. It never depends
//! on a terminal renderer or command-line parser.

mod builder;
pub mod config;
mod credentials;
mod error;
mod mcp;
pub mod mcp_config;
mod mcp_policy;
mod paths;
mod profile;
pub mod resources;
mod service;
mod skill_tool;
mod system_prompt;
mod trust;

pub use builder::CodingAgentBuilder;
pub use credentials::CodingCredentialResolver;
pub use error::{CodingError, CodingErrorCode};
pub use mcp::{McpCatalogEntry, McpServiceSnapshot};
pub use mcp_config::McpEnvironmentResolver;
pub use paths::AppPaths;
pub use service::{CodingAgentService, CommandAcceptance};
pub use skill_tool::{READ_SKILL_RESOURCE_TOOL_NAME, ReadSkillResourceTool, SkillResourceResolver};
pub use system_prompt::CodingSystemPromptBuilder;
pub use trust::{
    InteractionMode, PersistedTrustDecision, ProjectAccess, ProjectTrustStore, TrustRequest,
};

/// Returns the package version embedded at compile time.
#[must_use]
pub const fn package_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
