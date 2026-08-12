#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! Deterministic prompt modules, context providers, compilation, and
//! inspection for `tea-rs`.
//!
//! This pure context layer performs no filesystem, process, network, provider,
//! executor, clock, or async-runtime work. Trust labels preserve provenance for
//! inspection; they do not claim to prevent prompt injection.

mod budget;
mod compiler;
mod diagnostic;
mod error;
mod identity;
mod inspection;
mod module;
mod provenance;
mod provider;
mod providers;
mod segment;
mod skill;

pub use budget::{PROMPT_SEPARATOR, PromptBudget, TRUNCATION_MARKER, estimate_tokens};
pub use compiler::{CompiledPrompt, MAX_COMPILE_MODULES, PromptCompiler};
pub use diagnostic::{PromptDiagnostic, PromptDiagnosticCode};
pub use error::{ContextError, ContextErrorCode};
pub use identity::{
    ConflictKey, ContextIdentityError, ContextProviderId, PromptModuleId, PromptSegmentId, SkillId,
};
pub use inspection::{PromptInspectionEntry, SegmentDisposition};
pub use module::{MAX_MODULE_SEGMENTS, ModuleError, PromptModule, PromptPriority};
pub use provenance::{CacheScope, PromptAuthority, PromptProvenance, ProvenanceError, TrustLevel};
pub use provider::{
    ContextProvider, ContextProviderFuture, ContextRequest, MAX_CONTEXT_TOOLS,
    StaticContextProvider,
};
pub use providers::{
    SessionSummaryProvider, SkillMetadataProvider, ToolHintProvider, WorkspaceInstruction,
    WorkspaceInstructionProvider,
};
pub use segment::{
    BudgetBehavior, ConflictClaim, ConflictMode, MAX_SEGMENT_BYTES, PromptSegment, SegmentError,
};
pub use skill::{
    MAX_SKILL_ARGUMENT_BYTES, MAX_SKILL_DESCRIPTION_BYTES, SkillCommand, SkillError,
    SkillInvocation, SkillMetadata,
};
