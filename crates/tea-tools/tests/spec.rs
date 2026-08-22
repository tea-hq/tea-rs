use std::str::FromStr;

use serde_json::json;
use tea_protocol::ToolIdempotency;
use tea_tools::{
    SchedulerClass, ToolConcurrency, ToolEffect, ToolExecutionSemantics, ToolName, ToolRetrySafety,
    ToolSource, ToolSourceKind, ToolSpec, ToolSpecError, ToolTimeout, ToolTrust, ToolVersion,
};

const DIGEST: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

fn object_schema() -> serde_json::Value {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "path": { "type": "string" }
        },
        "required": ["path"],
        "additionalProperties": false
    })
}

fn semantics(
    idempotency: ToolIdempotency,
    retry: ToolRetrySafety,
    concurrency: ToolConcurrency,
) -> ToolExecutionSemantics {
    ToolExecutionSemantics::new(
        idempotency,
        retry,
        concurrency,
        ToolTimeout::from_millis(30_000).unwrap(),
    )
    .unwrap()
}

fn spec(
    name: &str,
    effects: impl IntoIterator<Item = ToolEffect>,
    execution: ToolExecutionSemantics,
) -> ToolSpec {
    ToolSpec::new(
        ToolName::from_str(name).unwrap(),
        ToolVersion::from_str("1.2.3").unwrap(),
        "Operates on a declared resource.",
        object_schema(),
        object_schema(),
        effects,
        execution,
    )
    .unwrap()
}

#[test]
fn tool_identity_is_canonical_and_versioned() {
    assert_eq!(
        ToolName::from_str("read_file").unwrap().as_str(),
        "read_file"
    );
    assert!(ToolName::from_str("").is_err());
    assert!(ToolName::from_str("ReadFile").is_err());
    assert!(ToolName::from_str("read file").is_err());
    assert!(ToolName::from_str(&"x".repeat(129)).is_err());

    let version = ToolVersion::from_str("1.2.3-beta.1").unwrap();
    assert_eq!(version.to_string(), "1.2.3-beta.1");
    assert!(ToolVersion::from_str("v1.2.3").is_err());
    assert!(ToolVersion::from_str("1.2").is_err());
}

#[test]
fn known_and_unknown_effects_have_stable_values() {
    let expected = [
        ("fs.read", ToolEffect::FsRead),
        ("fs.write", ToolEffect::FsWrite),
        ("fs.delete", ToolEffect::FsDelete),
        ("process.spawn", ToolEffect::ProcessSpawn),
        ("network.request", ToolEffect::NetworkRequest),
        ("credential.read", ToolEffect::CredentialRead),
        ("clipboard.read", ToolEffect::ClipboardRead),
        ("external.read", ToolEffect::ExternalRead),
        ("user.interaction", ToolEffect::UserInteraction),
        ("external.mutation", ToolEffect::ExternalMutation),
    ];
    for (wire, effect) in expected {
        assert_eq!(ToolEffect::from_str(wire).unwrap(), effect);
        assert_eq!(effect.as_str(), wire);
    }

    let unknown = ToolEffect::from_str("com.example.device.control").unwrap();
    assert_eq!(unknown.as_str(), "com.example.device.control");
    assert!(unknown.is_unknown());
    assert!(ToolEffect::from_str("unknown").is_err());
    assert!(ToolEffect::from_str("Bad.effect").is_err());
}

#[test]
fn execution_semantics_reject_unsafe_automatic_retry() {
    assert_eq!(
        ToolExecutionSemantics::new(
            ToolIdempotency::NonIdempotent,
            ToolRetrySafety::Automatic,
            ToolConcurrency::Parallel,
            ToolTimeout::from_millis(1_000).unwrap(),
        )
        .unwrap_err(),
        ToolSpecError::UnsafeAutomaticRetry
    );

    let safe = semantics(
        ToolIdempotency::Idempotent,
        ToolRetrySafety::Automatic,
        ToolConcurrency::Parallel,
    );
    assert_eq!(safe.idempotency(), ToolIdempotency::Idempotent);
    assert_eq!(safe.retry_safety(), ToolRetrySafety::Automatic);
    assert_eq!(safe.concurrency(), ToolConcurrency::Parallel);
    assert_eq!(safe.timeout().as_millis(), 30_000);

    assert!(ToolTimeout::from_millis(0).is_err());
    assert!(ToolTimeout::from_millis(86_400_001).is_err());
}

#[test]
fn specification_is_bounded_and_projects_to_model_definition() {
    let tool = spec(
        "read_file",
        [ToolEffect::FsRead],
        semantics(
            ToolIdempotency::Idempotent,
            ToolRetrySafety::Automatic,
            ToolConcurrency::Parallel,
        ),
    )
    .with_label("Read file")
    .unwrap()
    .with_prompt_snippet("Use for bounded workspace reads.")
    .unwrap()
    .with_prompt_guidelines([
        "Keep every path within the workspace.",
        "Request only the data needed for the task.",
    ])
    .unwrap()
    .with_ui_renderer("file-preview")
    .unwrap();

    assert_eq!(tool.name().as_str(), "read_file");
    assert_eq!(tool.version().to_string(), "1.2.3");
    assert_eq!(tool.effects(), &[ToolEffect::FsRead]);
    assert_eq!(tool.source(), &ToolSource::native_product());
    assert_eq!(tool.label(), Some("Read file"));
    assert_eq!(tool.prompt_hint(), Some("Use for bounded workspace reads."));
    assert_eq!(
        tool.prompt_snippet(),
        Some("Use for bounded workspace reads.")
    );
    assert_eq!(
        tool.prompt_guidelines(),
        [
            "Keep every path within the workspace.",
            "Request only the data needed for the task.",
        ]
    );
    assert_eq!(tool.ui_renderer(), Some("file-preview"));

    let model = tool.to_model_definition().unwrap();
    assert_eq!(model.name(), "read_file");
    assert_eq!(model.description(), "Operates on a declared resource.");
    assert_eq!(model.input_schema(), tool.input_schema());

    assert_eq!(
        ToolSpec::new(
            ToolName::from_str("empty_effects").unwrap(),
            ToolVersion::from_str("1.0.0").unwrap(),
            "description",
            object_schema(),
            object_schema(),
            [],
            semantics(
                ToolIdempotency::Idempotent,
                ToolRetrySafety::ExplicitOnly,
                ToolConcurrency::Serial,
            ),
        )
        .unwrap_err(),
        ToolSpecError::MissingEffects
    );
}

#[test]
fn tool_presentation_metadata_is_bounded() {
    let execution = semantics(
        ToolIdempotency::Idempotent,
        ToolRetrySafety::Automatic,
        ToolConcurrency::Parallel,
    );
    assert_eq!(
        spec("invalid_label", [ToolEffect::FsRead], execution)
            .with_label("")
            .unwrap_err(),
        ToolSpecError::InvalidLabel
    );
    assert_eq!(
        spec("too_many_guidelines", [ToolEffect::FsRead], execution)
            .with_prompt_guidelines((0..17).map(|_| "Guideline"))
            .unwrap_err(),
        ToolSpecError::TooManyPromptGuidelines
    );
    assert_eq!(
        spec("oversized_guidelines", [ToolEffect::FsRead], execution)
            .with_prompt_guidelines(["x".repeat(1_025)])
            .unwrap_err(),
        ToolSpecError::InvalidPromptGuideline
    );
}

#[test]
fn specification_accepts_an_explicit_frozen_source() {
    let source = ToolSource::new(
        ToolSourceKind::Mcp,
        "workspace.files",
        ToolTrust::Workspace,
        DIGEST,
    )
    .unwrap();
    let tool = spec(
        "read_file",
        [ToolEffect::FsRead],
        semantics(
            ToolIdempotency::Idempotent,
            ToolRetrySafety::Automatic,
            ToolConcurrency::Parallel,
        ),
    )
    .with_source(source.clone());
    assert_eq!(tool.source(), &source);
}

#[test]
fn scheduler_classification_uses_metadata_not_tool_names() {
    let read_a = spec(
        "read_file",
        [ToolEffect::FsRead],
        semantics(
            ToolIdempotency::Idempotent,
            ToolRetrySafety::Automatic,
            ToolConcurrency::Parallel,
        ),
    );
    let read_b = spec(
        "totally_different_name",
        [ToolEffect::FsRead],
        semantics(
            ToolIdempotency::Idempotent,
            ToolRetrySafety::Automatic,
            ToolConcurrency::Parallel,
        ),
    );
    assert_eq!(read_a.scheduler_class(), SchedulerClass::ParallelReadOnly);
    assert_eq!(read_a.scheduler_class(), read_b.scheduler_class());

    let retry_safe_mutation = spec(
        "write_file",
        [ToolEffect::FsWrite],
        semantics(
            ToolIdempotency::Idempotent,
            ToolRetrySafety::Automatic,
            ToolConcurrency::Parallel,
        ),
    );
    assert_eq!(
        retry_safe_mutation.scheduler_class(),
        SchedulerClass::ParallelRetrySafe
    );

    let non_idempotent = spec(
        "send_message",
        [ToolEffect::ExternalMutation],
        semantics(
            ToolIdempotency::NonIdempotent,
            ToolRetrySafety::Never,
            ToolConcurrency::Parallel,
        ),
    );
    assert_eq!(non_idempotent.scheduler_class(), SchedulerClass::Serial);

    let exclusive = spec(
        "exclusive_read",
        [ToolEffect::FsRead],
        semantics(
            ToolIdempotency::Idempotent,
            ToolRetrySafety::ExplicitOnly,
            ToolConcurrency::Exclusive,
        ),
    );
    assert_eq!(exclusive.scheduler_class(), SchedulerClass::Exclusive);

    let unknown = spec(
        "future_tool",
        [ToolEffect::from_str("com.example.future.effect").unwrap()],
        semantics(
            ToolIdempotency::Idempotent,
            ToolRetrySafety::Automatic,
            ToolConcurrency::Parallel,
        ),
    );
    assert_eq!(unknown.scheduler_class(), SchedulerClass::PolicyRequired);
    assert!(unknown.scheduler_class().requires_policy());
    assert!(!unknown.scheduler_class().allows_parallel_execution());
}
