use serde::Serialize;
use tea_context::PromptInspection;
use tea_protocol::{RunId, SessionId};

/// Last successfully compiled prompt metadata retained for one live session.
///
/// This snapshot contains no prompt text and is never written to the session
/// store. It is unavailable before the first successful compilation and after
/// rebuilding the runtime process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RuntimePromptInspection {
    session_id: SessionId,
    #[serde(skip_serializing_if = "Option::is_none")]
    run_id: Option<RunId>,
    #[serde(flatten)]
    prompt: PromptInspection,
}

impl RuntimePromptInspection {
    pub(crate) const fn new(
        session_id: SessionId,
        run_id: Option<RunId>,
        prompt: PromptInspection,
    ) -> Self {
        Self {
            session_id,
            run_id,
            prompt,
        }
    }

    /// Returns the owning session.
    #[must_use]
    pub const fn session_id(&self) -> SessionId {
        self.session_id
    }
    /// Returns the run whose prompt was compiled, when available.
    #[must_use]
    pub const fn run_id(&self) -> Option<RunId> {
        self.run_id
    }
    /// Returns content-free compiler metadata.
    #[must_use]
    pub const fn prompt(&self) -> &PromptInspection {
        &self.prompt
    }
}
