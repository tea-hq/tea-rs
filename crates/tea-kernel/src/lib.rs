#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Tokio-native headless agent state machine for `tea-rs`.
//!
//! The kernel coordinates provider-neutral model streams, validated tools,
//! pure policy, durable sessions, cooperative cancellation, and awaited
//! canonical observations. It contains no product prompt, live provider, UI,
//! filesystem, process, or network implementation.

mod approval;
mod clock;
mod compaction;
mod config;
mod context_window;
mod error;
mod event;
mod id;
mod kernel;
mod model_turn;
mod observe;
mod policy;
mod queue;
mod record;
mod request;
mod retry;
mod scheduler;
mod state;
mod tool_call;
mod tool_execution;

pub use clock::{KernelClock, KernelDeadlineFuture, TokioKernelClock};
pub use compaction::{
    CompactionPolicy, CompactionSummarizer, CompactionSummaryFuture, NeverCompactPolicy,
};
pub use config::{KernelRunConfig, RunLimits};
pub(crate) use context_window::ContextWindowAccountant;
pub use error::{KernelError, KernelErrorCode};
pub use event::{DiscardEventSink, KernelEventFuture, KernelEventSink};
pub use id::{KernelIdSource, UuidV7KernelIdSource};
pub use kernel::{AgentKernel, KernelRunOutcome, validate_new_run_state};
pub use queue::KernelInputQueue;
pub use request::TurnRequestSnapshot;
pub use retry::ModelRetryPolicy;
pub use scheduler::{Lane, SchedulePlan, Scheduler};
pub use state::{RunState, StateTransitionError, TurnState};
