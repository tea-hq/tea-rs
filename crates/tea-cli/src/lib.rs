#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Interactive and headless interfaces for the reference Coding CLI.
//!
//! Terminal, argument parser, JSONL/RPC, and presentation dependencies stop at
//! this crate. Runtime and product state remain authoritative in inward crates.

/// Versioned protocol-neutral app-server contract.
pub mod app_server;
/// Shared command-line argument contract.
pub mod args;
/// Injected production/test service bootstrap.
pub mod bootstrap;
mod exit;
mod jsonl;
/// Presentation modes over the application service.
pub mod modes;
/// Strict bounded JSONL/RPC process interface.
pub mod rpc;
/// Stable host projections shared by terminal and RPC session views.
pub mod session_views;
/// Pure terminal projection, rendering, routing, and lifecycle boundaries.
pub mod tui;

pub use bootstrap::{BootstrapEnvironment, CliBootstrap};
pub use exit::{CliFailure, ExitCategory};

/// Returns the stable private-pre-release executable version.
#[must_use]
pub const fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}
