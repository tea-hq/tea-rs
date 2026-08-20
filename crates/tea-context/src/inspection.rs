use std::ops::Range;

use serde::Serialize;

use crate::{
    CacheScope, PromptAuthority, PromptDiagnostic, PromptModuleId, PromptProvenance,
    PromptSegmentId, TrustLevel,
};

/// Final compiler disposition of one input segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SegmentDisposition {
    /// Full content appears in output.
    Included,
    /// Truncated content with marker appears in output.
    Truncated,
    /// Exact duplicate was deduplicated.
    Duplicate,
    /// Lower-precedence conflict was shadowed.
    ConflictShadowed,
    /// Explicit omit/truncate behavior could not fit the budget.
    OmittedForBudget,
}

/// Explainable inspection row for one input prompt segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptInspectionEntry {
    module_id: PromptModuleId,
    segment_id: PromptSegmentId,
    authority: PromptAuthority,
    provenance: PromptProvenance,
    trust: TrustLevel,
    cache_scope: CacheScope,
    disposition: SegmentDisposition,
    byte_range: Option<Range<usize>>,
    rendered_bytes: usize,
    estimated_tokens: usize,
}

impl PromptInspectionEntry {
    #[allow(clippy::too_many_arguments)]
    pub(crate) const fn new(
        module_id: PromptModuleId,
        segment_id: PromptSegmentId,
        authority: PromptAuthority,
        provenance: PromptProvenance,
        trust: TrustLevel,
        cache_scope: CacheScope,
        disposition: SegmentDisposition,
        byte_range: Option<Range<usize>>,
        rendered_bytes: usize,
        estimated_tokens: usize,
    ) -> Self {
        Self {
            module_id,
            segment_id,
            authority,
            provenance,
            trust,
            cache_scope,
            disposition,
            byte_range,
            rendered_bytes,
            estimated_tokens,
        }
    }
    /// Returns source module.
    #[must_use]
    pub const fn module_id(&self) -> &PromptModuleId {
        &self.module_id
    }
    /// Returns source segment.
    #[must_use]
    pub const fn segment_id(&self) -> &PromptSegmentId {
        &self.segment_id
    }
    /// Returns the source module's authority class.
    #[must_use]
    pub const fn authority(&self) -> PromptAuthority {
        self.authority
    }
    /// Returns source provenance.
    #[must_use]
    pub const fn provenance(&self) -> &PromptProvenance {
        &self.provenance
    }
    /// Returns trust label.
    #[must_use]
    pub const fn trust(&self) -> TrustLevel {
        self.trust
    }
    /// Returns cache scope.
    #[must_use]
    pub const fn cache_scope(&self) -> CacheScope {
        self.cache_scope
    }
    /// Returns final disposition.
    #[must_use]
    pub const fn disposition(&self) -> SegmentDisposition {
        self.disposition
    }
    /// Returns exact output content range, excluding separators.
    #[must_use]
    pub const fn byte_range(&self) -> Option<&Range<usize>> {
        self.byte_range.as_ref()
    }
    /// Returns rendered content bytes excluding separators.
    #[must_use]
    pub const fn rendered_bytes(&self) -> usize {
        self.rendered_bytes
    }
    /// Returns conservative rendered-content token estimate.
    #[must_use]
    pub const fn estimated_tokens(&self) -> usize {
        self.estimated_tokens
    }
}

/// Content-free explainability row suitable for host inspection APIs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptInspectionSegment {
    module_id: PromptModuleId,
    segment_id: PromptSegmentId,
    authority: PromptAuthority,
    provenance: PromptProvenance,
    trust: TrustLevel,
    cache_scope: CacheScope,
    disposition: SegmentDisposition,
    rendered_bytes: usize,
    estimated_tokens: usize,
}

impl PromptInspectionSegment {
    fn from_entry(entry: &PromptInspectionEntry) -> Self {
        Self {
            module_id: entry.module_id.clone(),
            segment_id: entry.segment_id.clone(),
            authority: entry.authority,
            provenance: entry.provenance.clone(),
            trust: entry.trust,
            cache_scope: entry.cache_scope,
            disposition: entry.disposition,
            rendered_bytes: entry.rendered_bytes,
            estimated_tokens: entry.estimated_tokens,
        }
    }

    /// Returns the source module identity.
    #[must_use]
    pub const fn module_id(&self) -> &PromptModuleId {
        &self.module_id
    }
    /// Returns the source segment identity.
    #[must_use]
    pub const fn segment_id(&self) -> &PromptSegmentId {
        &self.segment_id
    }
    /// Returns the module authority used for precedence.
    #[must_use]
    pub const fn authority(&self) -> PromptAuthority {
        self.authority
    }
    /// Returns prompt-safe source provenance.
    #[must_use]
    pub const fn provenance(&self) -> &PromptProvenance {
        &self.provenance
    }
    /// Returns the declared source trust.
    #[must_use]
    pub const fn trust(&self) -> TrustLevel {
        self.trust
    }
    /// Returns the intended cache scope.
    #[must_use]
    pub const fn cache_scope(&self) -> CacheScope {
        self.cache_scope
    }
    /// Returns the compiler's final disposition.
    #[must_use]
    pub const fn disposition(&self) -> SegmentDisposition {
        self.disposition
    }
    /// Returns rendered bytes without exposing content or byte positions.
    #[must_use]
    pub const fn rendered_bytes(&self) -> usize {
        self.rendered_bytes
    }
    /// Returns the conservative token estimate for rendered content.
    #[must_use]
    pub const fn estimated_tokens(&self) -> usize {
        self.estimated_tokens
    }
}

/// Content-free metadata for one successfully compiled prompt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PromptInspection {
    rendered_bytes: usize,
    estimated_tokens: usize,
    segments: Vec<PromptInspectionSegment>,
    diagnostics: Vec<PromptDiagnostic>,
}

impl PromptInspection {
    pub(crate) fn new(
        rendered_bytes: usize,
        estimated_tokens: usize,
        entries: &[PromptInspectionEntry],
        diagnostics: &[PromptDiagnostic],
    ) -> Self {
        Self {
            rendered_bytes,
            estimated_tokens,
            segments: entries
                .iter()
                .map(PromptInspectionSegment::from_entry)
                .collect(),
            diagnostics: diagnostics.to_vec(),
        }
    }

    /// Returns complete rendered prompt bytes without exposing text.
    #[must_use]
    pub const fn rendered_bytes(&self) -> usize {
        self.rendered_bytes
    }
    /// Returns the conservative token estimate for the complete prompt.
    #[must_use]
    pub const fn estimated_tokens(&self) -> usize {
        self.estimated_tokens
    }
    /// Returns stable segment metadata in compiler inspection order.
    #[must_use]
    pub fn segments(&self) -> &[PromptInspectionSegment] {
        &self.segments
    }
    /// Returns stable compiler diagnostics without prompt content.
    #[must_use]
    pub fn diagnostics(&self) -> &[PromptDiagnostic] {
        &self.diagnostics
    }
}
