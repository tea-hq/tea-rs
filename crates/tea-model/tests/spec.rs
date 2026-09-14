use std::str::FromStr;

use tea_model::{
    HostedToolKind, ModelCapabilities, ModelDisplayName, ModelSpec, ModelSpecError, ProviderId,
    ReasoningEffort, ReasoningProfile,
};
use tea_protocol::{ModelId, TokenCount};

fn tokens(value: u64) -> TokenCount {
    TokenCount::new(value).unwrap()
}

#[test]
fn provider_and_display_names_are_bounded() {
    assert_eq!(
        ProviderId::from_str("openai-compatible").unwrap().as_str(),
        "openai-compatible"
    );
    assert!(ProviderId::from_str("").is_err());
    assert!(ProviderId::from_str("OpenAI").is_err());
    assert!(ProviderId::from_str("openai compatible").is_err());
    assert!(ProviderId::from_str(&"p".repeat(129)).is_err());

    assert_eq!(
        ModelDisplayName::from_str("Claude Sonnet 4")
            .unwrap()
            .as_str(),
        "Claude Sonnet 4"
    );
    assert!(ModelDisplayName::from_str("").is_err());
    assert!(ModelDisplayName::from_str("bad\nname").is_err());
    assert!(ModelDisplayName::from_str(&"m".repeat(257)).is_err());
}

#[test]
fn capabilities_are_explicit_and_internally_consistent() {
    let text_only = ModelCapabilities::text();
    assert!(text_only.accepts_text());
    assert!(!text_only.accepts_images());
    assert!(!text_only.supports_reasoning());
    assert!(!text_only.supports_tools());
    assert!(!text_only.supports_parallel_tool_calls());
    assert!(!text_only.supports_hosted_tool(HostedToolKind::WebSearch));
    assert!(!text_only.reports_usage());

    let capable = text_only
        .with_image_input()
        .with_reasoning()
        .with_tools(true)
        .with_hosted_tool(HostedToolKind::WebSearch)
        .with_usage_reporting();
    assert!(capable.accepts_text());
    assert!(capable.accepts_images());
    assert!(capable.supports_reasoning());
    assert!(capable.supports_tools());
    assert!(capable.supports_parallel_tool_calls());
    assert!(capable.supports_hosted_tool(HostedToolKind::WebSearch));
    assert!(capable.reports_usage());

    let serial_tools = ModelCapabilities::text().with_tools(false);
    assert!(serial_tools.supports_tools());
    assert!(!serial_tools.supports_parallel_tool_calls());
}

#[test]
fn final_json_capabilities_are_independent() {
    let neither = ModelCapabilities::text();
    let object_only = ModelCapabilities::text().with_final_json_object();
    let schema_only = ModelCapabilities::text().with_final_json_schema();
    let schema_with_tools = ModelCapabilities::text().with_final_json_schema_with_tools();
    let both = ModelCapabilities::text()
        .with_final_json_object()
        .with_final_json_schema();
    assert_eq!(
        [neither, object_only, schema_only, schema_with_tools, both].map(|capabilities| (
            capabilities.supports_final_json_object(),
            capabilities.supports_final_json_schema(),
            capabilities.supports_final_json_schema_with_tools(),
        )),
        [
            (false, false, false),
            (true, false, false),
            (false, true, false),
            (false, true, true),
            (true, true, false),
        ]
    );
}

#[test]
fn model_spec_enforces_context_and_output_limits() {
    let model_id = ModelId::from_str("anthropic/claude-sonnet-4").unwrap();
    let provider_id = ProviderId::from_str("anthropic").unwrap();
    let display_name = ModelDisplayName::from_str("Claude Sonnet 4").unwrap();
    let capabilities = ModelCapabilities::text().with_reasoning().with_tools(true);

    let spec = ModelSpec::new(
        model_id.clone(),
        provider_id.clone(),
        display_name.clone(),
        tokens(200_000),
        tokens(64_000),
        capabilities,
    )
    .unwrap();

    assert_eq!(spec.model_id(), &model_id);
    assert_eq!(spec.provider_id(), &provider_id);
    assert_eq!(spec.display_name(), &display_name);
    assert_eq!(spec.context_window_tokens(), tokens(200_000));
    assert_eq!(spec.max_output_tokens(), tokens(64_000));
    assert_eq!(spec.capabilities(), capabilities);

    assert_eq!(
        ModelSpec::new(
            model_id.clone(),
            provider_id.clone(),
            display_name.clone(),
            tokens(0),
            tokens(1),
            capabilities,
        )
        .unwrap_err(),
        ModelSpecError::EmptyContextWindow
    );
    assert_eq!(
        ModelSpec::new(
            model_id.clone(),
            provider_id.clone(),
            display_name.clone(),
            tokens(100),
            tokens(0),
            capabilities,
        )
        .unwrap_err(),
        ModelSpecError::EmptyOutputLimit
    );
    assert_eq!(
        ModelSpec::new(
            model_id,
            provider_id,
            display_name,
            tokens(100),
            tokens(101),
            capabilities,
        )
        .unwrap_err(),
        ModelSpecError::OutputExceedsContext
    );
}

#[test]
fn model_types_are_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<ModelSpec>();
    assert_send_sync::<ModelCapabilities>();
    assert_send_sync::<ReasoningProfile>();
}

#[test]
fn reasoning_profiles_validate_defaults_and_clamp_upward_first() {
    assert_eq!(
        ReasoningProfile::new(ReasoningEffort::Medium, []).unwrap_err(),
        ModelSpecError::EmptyReasoningEfforts
    );
    assert_eq!(
        ReasoningProfile::new(
            ReasoningEffort::Medium,
            [ReasoningEffort::Low, ReasoningEffort::High],
        )
        .unwrap_err(),
        ModelSpecError::ReasoningDefaultUnsupported
    );
    assert_eq!(
        ReasoningProfile::new(
            ReasoningEffort::Low,
            [ReasoningEffort::Low, ReasoningEffort::Low],
        )
        .unwrap_err(),
        ModelSpecError::DuplicateReasoningEffort
    );

    let profile = ReasoningProfile::new(
        ReasoningEffort::Medium,
        [
            ReasoningEffort::Off,
            ReasoningEffort::Medium,
            ReasoningEffort::Maximum,
        ],
    )
    .unwrap();
    assert_eq!(
        profile.supported_efforts(),
        &[
            ReasoningEffort::Off,
            ReasoningEffort::Medium,
            ReasoningEffort::Maximum,
        ]
    );
    assert_eq!(
        profile.resolve(ReasoningEffort::Low).effective(),
        ReasoningEffort::Medium
    );
    assert!(profile.resolve(ReasoningEffort::Low).was_clamped());
    assert_eq!(
        profile.resolve(ReasoningEffort::High).effective(),
        ReasoningEffort::Maximum
    );
    assert_eq!(
        profile.resolve(ReasoningEffort::Maximum).effective(),
        ReasoningEffort::Maximum
    );
    assert!(!profile.resolve(ReasoningEffort::Maximum).was_clamped());
}

#[test]
fn model_spec_carries_explicit_or_compatible_reasoning_profiles() {
    let explicit = ReasoningProfile::new(
        ReasoningEffort::High,
        [ReasoningEffort::Low, ReasoningEffort::High],
    )
    .unwrap();
    let spec = ModelSpec::new(
        ModelId::from_str("test/reasoning").unwrap(),
        ProviderId::from_str("test").unwrap(),
        ModelDisplayName::from_str("Reasoning").unwrap(),
        tokens(10_000),
        tokens(2_000),
        ModelCapabilities::text(),
    )
    .unwrap()
    .with_reasoning_profile(explicit.clone());
    assert_eq!(spec.reasoning_profile(), Some(&explicit));
    assert!(spec.capabilities().supports_reasoning());

    let compatible = ModelSpec::new(
        ModelId::from_str("test/legacy").unwrap(),
        ProviderId::from_str("test").unwrap(),
        ModelDisplayName::from_str("Legacy").unwrap(),
        tokens(10_000),
        tokens(2_000),
        ModelCapabilities::text().with_reasoning(),
    )
    .unwrap();
    let compatible = compatible.reasoning_profile().unwrap();
    assert_eq!(compatible.default_effort(), ReasoningEffort::Medium);
    assert!(
        !compatible
            .supported_efforts()
            .contains(&ReasoningEffort::ExtraHigh)
    );
    assert!(
        !compatible
            .supported_efforts()
            .contains(&ReasoningEffort::Maximum)
    );

    let plain = ModelSpec::new(
        ModelId::from_str("test/plain").unwrap(),
        ProviderId::from_str("test").unwrap(),
        ModelDisplayName::from_str("Plain").unwrap(),
        tokens(10_000),
        tokens(2_000),
        ModelCapabilities::text(),
    )
    .unwrap();
    assert_eq!(plain.reasoning_profile(), None);
    let disabled = plain
        .resolve_reasoning(Some(ReasoningEffort::High))
        .unwrap();
    assert_eq!(disabled.requested(), ReasoningEffort::High);
    assert_eq!(disabled.effective(), ReasoningEffort::Off);
    assert!(disabled.was_clamped());
    assert_eq!(plain.resolve_reasoning(None), None);
}
