use crate::common;

use common::{module, segment};
use tea_context::{
    BudgetBehavior, CacheScope, PromptAuthority, PromptBudget, PromptCompiler, SegmentDisposition,
    TrustLevel,
};

#[test]
fn included_byte_ranges_slice_exact_content_and_preserve_provenance() {
    let prompt = PromptCompiler
        .compile(
            [module(
                "inspect",
                PromptAuthority::Workspace,
                0,
                vec![
                    segment("one", "alpha", BudgetBehavior::Required),
                    segment("two", "βeta", BudgetBehavior::Required),
                ],
            )],
            PromptBudget::new(100, 100).unwrap(),
        )
        .unwrap();
    assert_eq!(prompt.text(), "alpha\n\nβeta");
    for entry in prompt.inspection() {
        let range = entry.byte_range().unwrap().clone();
        let slice = &prompt.text()[range];
        assert_eq!(slice.len(), entry.rendered_bytes());
        assert_eq!(entry.disposition(), SegmentDisposition::Included);
        assert_eq!(entry.authority(), PromptAuthority::Workspace);
        assert_eq!(entry.trust(), TrustLevel::Delegated);
        assert_eq!(entry.cache_scope(), CacheScope::Run);
        assert_eq!(entry.provenance().source_kind(), "test");
        assert!(!slice.is_empty());
    }
    assert_eq!(prompt.inspection()[0].byte_range().unwrap(), &(0..5));
    assert_eq!(prompt.inspection()[1].byte_range().unwrap(), &(7..12));
}

#[test]
fn public_inspection_snapshot_exposes_metadata_without_content_or_byte_ranges() {
    let secret = "private prompt body";
    let prompt = PromptCompiler
        .compile(
            [module(
                "inspect",
                PromptAuthority::Workspace,
                0,
                vec![segment("one", secret, BudgetBehavior::Required)],
            )],
            PromptBudget::new(100, 100).unwrap(),
        )
        .unwrap();

    let snapshot = prompt.inspection_snapshot();
    assert_eq!(snapshot.rendered_bytes(), secret.len());
    assert_eq!(snapshot.estimated_tokens(), prompt.estimated_tokens());
    assert_eq!(snapshot.segments().len(), 1);
    assert_eq!(
        snapshot.segments()[0].authority(),
        PromptAuthority::Workspace
    );
    let encoded = serde_json::to_string(&snapshot).unwrap();
    assert!(!encoded.contains(secret));
    assert!(!encoded.contains("byteRange"));
    assert!(!encoded.contains("content"));
}

#[test]
fn inspection_is_identical_for_permuted_module_inputs() {
    let modules = vec![
        module(
            "later",
            PromptAuthority::Session,
            0,
            vec![segment("later", "later", BudgetBehavior::Required)],
        ),
        module(
            "earlier",
            PromptAuthority::Product,
            0,
            vec![segment("earlier", "earlier", BudgetBehavior::Required)],
        ),
    ];
    let forward = PromptCompiler
        .compile(modules.clone(), PromptBudget::new(100, 100).unwrap())
        .unwrap();
    let reverse = PromptCompiler
        .compile(
            modules.into_iter().rev(),
            PromptBudget::new(100, 100).unwrap(),
        )
        .unwrap();
    assert_eq!(forward.inspection(), reverse.inspection());
    assert_eq!(forward.diagnostics(), reverse.diagnostics());
}
