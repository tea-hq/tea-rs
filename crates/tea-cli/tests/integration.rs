#![forbid(unsafe_code)]

#[path = "app_server_live_smoke.rs"]
mod app_server_live_smoke;
#[path = "app_server_mcp.rs"]
mod app_server_mcp;
#[path = "app_server_process.rs"]
mod app_server_process;
#[path = "app_server_protocol.rs"]
mod app_server_protocol;
#[path = "app_server_session.rs"]
mod app_server_session;
#[path = "args.rs"]
mod args;
#[path = "tui/common.rs"]
mod common;
#[path = "cross_mode.rs"]
mod cross_mode;
#[path = "dependency_boundary.rs"]
mod dependency_boundary;
#[path = "faults.rs"]
mod faults;
#[path = "json_mode.rs"]
mod json_mode;
#[path = "live_smoke.rs"]
mod live_smoke;
#[path = "mcp_live_smoke.rs"]
mod mcp_live_smoke;
#[path = "print_mode.rs"]
mod print_mode;
#[path = "pty.rs"]
mod pty;
#[path = "rpc.rs"]
mod rpc;
#[path = "rpc_backpressure.rs"]
mod rpc_backpressure;
#[path = "secrets.rs"]
mod secrets;
#[path = "session_ux.rs"]
mod session_ux;
#[path = "skills.rs"]
mod skills;
#[path = "tui.rs"]
mod tui;
#[path = "version.rs"]
mod version;
