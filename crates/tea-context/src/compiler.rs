use std::collections::BTreeMap;

use crate::budget::{PROMPT_SEPARATOR, effective_remaining_bytes, estimate_tokens, truncate};
use crate::{
    BudgetBehavior, ConflictKey, ConflictMode, ContextError, ContextErrorCode, PromptBudget,
    PromptDiagnostic, PromptDiagnosticCode, PromptInspection, PromptInspectionEntry, PromptModule,
    PromptModuleId, PromptSegment, PromptSegmentId, SegmentDisposition,
};

/// Maximum modules accepted by one compilation.
pub const MAX_COMPILE_MODULES: usize = 1024;

/// Byte-identical compiled system prompt and explainability data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledPrompt {
    text: String,
    estimated_tokens: usize,
    diagnostics: Vec<PromptDiagnostic>,
    inspection: Vec<PromptInspectionEntry>,
}

impl CompiledPrompt {
    /// Returns exact rendered prompt text without trailing newline.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }
    /// Returns exact UTF-8 output bytes.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.text.len()
    }
    /// Returns conservative token estimate for the complete output.
    #[must_use]
    pub const fn estimated_tokens(&self) -> usize {
        self.estimated_tokens
    }
    /// Returns stable ordered diagnostics.
    #[must_use]
    pub fn diagnostics(&self) -> &[PromptDiagnostic] {
        &self.diagnostics
    }
    /// Returns one explainability row per unique input segment.
    #[must_use]
    pub fn inspection(&self) -> &[PromptInspectionEntry] {
        &self.inspection
    }
    /// Returns content-free aggregate metadata for host inspection APIs.
    #[must_use]
    pub fn inspection_snapshot(&self) -> PromptInspection {
        PromptInspection::new(
            self.bytes(),
            self.estimated_tokens,
            &self.inspection,
            &self.diagnostics,
        )
    }
}

#[derive(Clone)]
struct Candidate {
    module_id: PromptModuleId,
    authority: crate::PromptAuthority,
    priority: crate::PromptPriority,
    segment_order: usize,
    segment: PromptSegment,
}

impl Candidate {
    fn same_precedence(&self, other: &Self) -> bool {
        self.authority == other.authority && self.priority == other.priority
    }
}

type Selection = (
    Vec<Candidate>,
    Vec<PromptDiagnostic>,
    Vec<PromptInspectionEntry>,
);

/// Pure deterministic prompt compiler.
#[derive(Debug, Clone, Copy, Default)]
pub struct PromptCompiler;

impl PromptCompiler {
    /// Selects, budgets, renders, and inspects prompt modules.
    ///
    /// # Errors
    ///
    /// Returns an error for oversized input, divergent duplicate identities,
    /// ambiguous equal-precedence conflicts, or required budget overflow.
    pub fn compile(
        &self,
        modules: impl IntoIterator<Item = PromptModule>,
        budget: PromptBudget,
    ) -> Result<CompiledPrompt, ContextError> {
        let modules = modules.into_iter().collect::<Vec<_>>();
        if modules.len() > MAX_COMPILE_MODULES {
            return Err(ContextError::new(
                ContextErrorCode::BoundsExceeded,
                "prompt compilation contains too many modules",
            ));
        }
        let mut candidates = flatten(modules);
        candidates.sort_by(|left, right| {
            left.authority
                .cmp(&right.authority)
                .then_with(|| right.priority.cmp(&left.priority))
                .then_with(|| left.module_id.cmp(&right.module_id))
                .then_with(|| left.segment_order.cmp(&right.segment_order))
        });
        let (selected, mut diagnostics, mut inspection) = select(candidates)?;
        render(selected, budget, &mut diagnostics, &mut inspection)
    }
}

fn flatten(modules: Vec<PromptModule>) -> Vec<Candidate> {
    modules
        .into_iter()
        .flat_map(|module| {
            let module_id = module.id().clone();
            let authority = module.authority();
            let priority = module.priority();
            let segments = module.segments().to_vec();
            segments
                .into_iter()
                .enumerate()
                .map(move |(segment_order, segment)| Candidate {
                    module_id: module_id.clone(),
                    authority,
                    priority,
                    segment_order,
                    segment,
                })
        })
        .collect()
}

fn select(candidates: Vec<Candidate>) -> Result<Selection, ContextError> {
    let mut identities: BTreeMap<PromptSegmentId, Candidate> = BTreeMap::new();
    let mut conflicts: BTreeMap<ConflictKey, Candidate> = BTreeMap::new();
    let mut selected = Vec::new();
    let mut diagnostics = Vec::new();
    let mut inspection = Vec::new();
    for candidate in candidates {
        if let Some(existing) = identities.get(candidate.segment.id()) {
            if existing.segment != candidate.segment {
                return Err(ContextError::new(
                    ContextErrorCode::DuplicateIdentity,
                    "prompt segment identity has divergent definitions",
                ));
            }
            diagnostics.push(PromptDiagnostic::new(
                PromptDiagnosticCode::ExactDuplicate,
                candidate.segment.id().clone(),
                Some(existing.segment.id().clone()),
                None,
            ));
            inspection.push(nonrendered(&candidate, SegmentDisposition::Duplicate));
            continue;
        }
        identities.insert(candidate.segment.id().clone(), candidate.clone());
        if let Some(claim) = candidate.segment.conflict() {
            if let Some(winner) = conflicts.get(claim.key()) {
                if candidate.same_precedence(winner)
                    && candidate.segment.content() != winner.segment.content()
                {
                    return Err(ContextError::new(
                        ContextErrorCode::AmbiguousConflict,
                        "equal-precedence prompt conflict is ambiguous",
                    ));
                }
                let code = if winner
                    .segment
                    .conflict()
                    .is_some_and(|value| value.mode() == ConflictMode::Protected)
                {
                    PromptDiagnosticCode::ProtectedConflict
                } else {
                    PromptDiagnosticCode::ConflictShadowed
                };
                diagnostics.push(PromptDiagnostic::new(
                    code,
                    candidate.segment.id().clone(),
                    Some(winner.segment.id().clone()),
                    Some(claim.key().clone()),
                ));
                inspection.push(nonrendered(
                    &candidate,
                    SegmentDisposition::ConflictShadowed,
                ));
                continue;
            }
            conflicts.insert(claim.key().clone(), candidate.clone());
        }
        selected.push(candidate);
    }
    Ok((selected, diagnostics, inspection))
}

fn render(
    selected: Vec<Candidate>,
    budget: PromptBudget,
    diagnostics: &mut Vec<PromptDiagnostic>,
    inspection: &mut Vec<PromptInspectionEntry>,
) -> Result<CompiledPrompt, ContextError> {
    let mut text = String::new();
    for candidate in selected {
        let separator_bytes = if text.is_empty() {
            0
        } else {
            PROMPT_SEPARATOR.len()
        };
        let used_with_separator = text.len().saturating_add(separator_bytes);
        let remaining = effective_remaining_bytes(budget, used_with_separator);
        let content = candidate.segment.content();
        let (rendered, disposition) = if content.len() <= remaining {
            (Some(content.to_owned()), SegmentDisposition::Included)
        } else {
            match candidate.segment.budget_behavior() {
                BudgetBehavior::Required => {
                    return Err(ContextError::new(
                        ContextErrorCode::BudgetExceeded,
                        "required prompt segment exceeds compilation budget",
                    ));
                }
                BudgetBehavior::Omit => (None, SegmentDisposition::OmittedForBudget),
                BudgetBehavior::Truncate => truncate(content, remaining)
                    .map_or((None, SegmentDisposition::OmittedForBudget), |value| {
                        (Some(value), SegmentDisposition::Truncated)
                    }),
            }
        };
        let Some(rendered) = rendered else {
            diagnostics.push(PromptDiagnostic::new(
                PromptDiagnosticCode::OmittedForBudget,
                candidate.segment.id().clone(),
                None,
                None,
            ));
            inspection.push(nonrendered(
                &candidate,
                SegmentDisposition::OmittedForBudget,
            ));
            continue;
        };
        if !text.is_empty() {
            text.push_str(PROMPT_SEPARATOR);
        }
        let start = text.len();
        text.push_str(&rendered);
        let end = text.len();
        if disposition == SegmentDisposition::Truncated {
            diagnostics.push(PromptDiagnostic::new(
                PromptDiagnosticCode::TruncatedForBudget,
                candidate.segment.id().clone(),
                None,
                None,
            ));
        }
        inspection.push(PromptInspectionEntry::new(
            candidate.module_id,
            candidate.segment.id().clone(),
            candidate.authority,
            candidate.segment.provenance().clone(),
            candidate.segment.trust(),
            candidate.segment.cache_scope(),
            disposition,
            Some(start..end),
            rendered.len(),
            estimate_tokens(rendered.len()),
        ));
    }
    inspection.sort_by(|left, right| {
        left.byte_range()
            .map_or(usize::MAX, |range| range.start)
            .cmp(&right.byte_range().map_or(usize::MAX, |range| range.start))
            .then_with(|| left.module_id().cmp(right.module_id()))
            .then_with(|| left.segment_id().cmp(right.segment_id()))
    });
    diagnostics.sort_by(|left, right| {
        left.segment_id()
            .cmp(right.segment_id())
            .then_with(|| left.code().cmp(&right.code()))
    });
    let estimated_tokens = estimate_tokens(text.len());
    if text.len() > budget.max_bytes() || estimated_tokens > budget.max_estimated_tokens() {
        return Err(ContextError::new(
            ContextErrorCode::BudgetExceeded,
            "compiled prompt is empty or exceeds final budget",
        ));
    }
    Ok(CompiledPrompt {
        text,
        estimated_tokens,
        diagnostics: diagnostics.clone(),
        inspection: inspection.clone(),
    })
}

fn nonrendered(candidate: &Candidate, disposition: SegmentDisposition) -> PromptInspectionEntry {
    PromptInspectionEntry::new(
        candidate.module_id.clone(),
        candidate.segment.id().clone(),
        candidate.authority,
        candidate.segment.provenance().clone(),
        candidate.segment.trust(),
        candidate.segment.cache_scope(),
        disposition,
        None,
        0,
        0,
    )
}
