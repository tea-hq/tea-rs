#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Ergonomic embedding facade and product profile wiring for `tea-rs`.
//!
//! The runtime owns replaceable inward-facing ports (model provider, tool
//! registry, policy engine, session store, clock, ID source, event sink,
//! context providers, and prompt compiler) and exposes an in-process command
//! sender, bounded event subscription, session snapshots, and health
//! inspection. It contains no product prompt, live provider, UI, filesystem,
//! process, network, or database adapter.
//!
//! Core contracts are available through the [`context`], [`control`],
//! [`kernel`], [`model`], [`policy`], [`profile`], [`protocol`], [`session`],
//! and [`tools`] namespaces. Concrete provider, persistence, MCP, native-tool,
//! and product adapters remain separate dependencies selected by the host.
//!
//! # Example
//!
//! ```no_run
//! use std::str::FromStr;
//!
//! use tea::{AgentRuntimeBuilder, RuntimeError, profile::AgentProfile};
//!
//! # async fn _example() -> Result<(), RuntimeError> {
//! let runtime = AgentRuntimeBuilder::new()
//!     .profile(AgentProfile::minimal_assistant()?)
//!     // Wire a model provider, tools, and policy rules before building.
//!     .build()?;
//! # Ok(())
//! # }
//! ```

mod agent_session;
mod binding;
mod builder;
mod command;
mod error;
mod event;
mod health;
mod id;
mod policy_wiring;
mod prompt;
mod prompt_inspection;
mod runtime;
mod session_host;

/// Deterministic prompt compilation and context-provider contracts.
pub use tea_context as context;
/// Shared cancellation contracts used by model and tool operations.
pub use tea_control as control;
/// Agent loop, reliability, clock, and identity-source contracts.
pub use tea_kernel as kernel;
/// Provider-neutral model specifications, requests, events, and ports.
pub use tea_model as model;
/// Policy, approval, grant, identity, and redaction contracts.
pub use tea_policy as policy;
/// Versioned product profile schema and composition contracts.
pub use tea_profile as profile;
/// Canonical commands, events, messages, identifiers, and metadata.
pub use tea_protocol as protocol;
/// Append-only session, replay, snapshot, and storage contracts.
pub use tea_session as session;
/// Portable tool specifications, resources, bindings, and executor ports.
pub use tea_tools as tools;

pub use agent_session::{AgentResponse, AgentSession, AgentSessionBuilder, AgentToolRegistration};
pub use binding::ProfileBinding;
pub use builder::AgentRuntimeBuilder;
pub use command::RuntimeCommandOutcome;
pub use error::{RuntimeError, RuntimeErrorCode};
pub use event::{DEFAULT_EVENT_CHANNEL_CAPACITY, MAX_EVENT_SUBSCRIBERS, RuntimeEventSink};
pub use health::RuntimeHealth;
pub use id::{SessionIdSource, UuidSessionIdSource};
pub use prompt_inspection::RuntimePromptInspection;
pub use runtime::AgentRuntime;
pub use session_host::{RuntimeSessionState, SessionStats};
