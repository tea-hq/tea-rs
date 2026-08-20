use serde::Serialize;

use crate::{ConflictKey, PromptSegmentId};

/// Stable prompt-compiler diagnostic classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PromptDiagnosticCode {
    /// An exact duplicate segment was emitted only once.
    ExactDuplicate,
    /// A lower-precedence conflict was shadowed by a replaceable winner.
    ConflictShadowed,
    /// A lower-precedence claim could not override a protected winner.
    ProtectedConflict,
    /// A segment was omitted due to explicit budget behavior.
    OmittedForBudget,
    /// A segment was deterministically truncated.
    TruncatedForBudget,
}

/// Bounded deterministic compiler diagnostic without prompt content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptDiagnostic {
    code: PromptDiagnosticCode,
    segment_id: PromptSegmentId,
    winner_id: Option<PromptSegmentId>,
    conflict_key: Option<ConflictKey>,
}

impl PromptDiagnostic {
    pub(crate) const fn new(
        code: PromptDiagnosticCode,
        segment_id: PromptSegmentId,
        winner_id: Option<PromptSegmentId>,
        conflict_key: Option<ConflictKey>,
    ) -> Self {
        Self {
            code,
            segment_id,
            winner_id,
            conflict_key,
        }
    }
    /// Returns stable diagnostic code.
    #[must_use]
    pub const fn code(&self) -> PromptDiagnosticCode {
        self.code
    }
    /// Returns affected segment.
    #[must_use]
    pub const fn segment_id(&self) -> &PromptSegmentId {
        &self.segment_id
    }
    /// Returns winning segment for conflict/duplicate diagnostics.
    #[must_use]
    pub const fn winner_id(&self) -> Option<&PromptSegmentId> {
        self.winner_id.as_ref()
    }
    /// Returns affected conflict key when applicable.
    #[must_use]
    pub const fn conflict_key(&self) -> Option<&ConflictKey> {
        self.conflict_key.as_ref()
    }
}
