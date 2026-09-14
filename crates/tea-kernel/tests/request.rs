use crate::common;

use std::str::FromStr;
use std::sync::Arc;

use serde_json::json;
use tea_context::{
    BudgetBehavior, CacheScope, ContextProviderId, PromptAuthority, PromptBudget, PromptCompiler,
    PromptModule, PromptModuleId, PromptPriority, PromptProvenance, PromptSegment, PromptSegmentId,
    TrustLevel,
};
use tea_control::CancellationScope;
use tea_kernel::{AgentKernel, KernelErrorCode, KernelRunConfig, TurnRequestSnapshot};
use tea_model::{
    HostedToolKind, HostedToolOptions, ModelCapabilities, ModelDisplayName, ModelProvider,
    ModelSpec, ProviderId, WebSearchOptions,
};
use tea_policy::{
    ActorId, ExecutionSurface, PolicyEngine, PolicyEnvironment, PolicyExecutionTarget,
};
use tea_protocol::{
    CanonicalMessage, ContentBlock, FINAL_JSON_OBJECT_INSTRUCTION, FinalOutputFormat, MessageId,
    ModelId, ProfileId, ProtocolMetadata, ProtocolTimestamp, ReasoningEffort, RecordEnvelope,
    RecordId, SessionId, SessionRecord, SessionSequence, StopReason, TokenCount, ToolIdempotency,
};
use tea_session::{SessionReducer, SessionStore};
use tea_testkit::{FakeReadTool, ScriptedModelProvider, ScriptedModelResponse};
use tea_tools::{
    StaticResourceResolver, ToolConcurrency, ToolEffect, ToolExecutionSemantics, ToolName,
    ToolRegistry, ToolRetrySafety, ToolRoutePreference, ToolSpec, ToolTimeout, ToolVersion,
};

const SESSION: &str = "0195a0b1-5e3a-7d72-a902-c4e85d828bf1";
const CREATED: &str = "0195a0b1-5e50-7af4-8972-0aa7aa000022";
const CONFIGURED: &str = "0195a0b1-5e51-79e1-8f4a-0aa7aa000023";
const MESSAGE_RECORD: &str = "0195a0b1-5e52-7b3e-93f1-0aa7aa000024";
const MESSAGE: &str = "0195a0b1-5e53-74b2-8c25-0aa7aa000025";

fn timestamp() -> ProtocolTimestamp {
    ProtocolTimestamp::from_str("2026-07-23T09:30:12.125Z").unwrap()
}

fn envelope(sequence: u64, record_id: &str, record: SessionRecord) -> RecordEnvelope {
    RecordEnvelope::new(
        RecordId::from_str(record_id).unwrap(),
        SessionId::from_str(SESSION).unwrap(),
        SessionSequence::new(sequence),
        timestamp(),
        None,
        None,
        None,
        ProtocolMetadata::default(),
        record,
    )
    .unwrap()
}

fn state(include_model: bool) -> tea_session::MaterializedSessionState {
    state_with_reasoning(include_model, None)
}

fn state_with_reasoning(
    include_model: bool,
    reasoning_effort: Option<ReasoningEffort>,
) -> tea_session::MaterializedSessionState {
    let mut records = vec![envelope(
        0,
        CREATED,
        SessionRecord::SessionCreated {
            profile_id: ProfileId::from_str("coding").unwrap(),
            metadata: ProtocolMetadata::default(),
        },
    )];
    if include_model {
        records.push(envelope(
            1,
            CONFIGURED,
            SessionRecord::ConfigurationChanged {
                model: Some(tea_protocol::ModelRef::new(
                    "fake".parse().unwrap(),
                    ModelId::from_str("fake/model").unwrap(),
                )),
                profile_id: None,
                reasoning_effort,
            },
        ));
    }
    let sequence = records.len() as u64;
    records.push(envelope(
        sequence,
        MESSAGE_RECORD,
        SessionRecord::MessageCommitted {
            message: CanonicalMessage::user(
                MessageId::from_str(MESSAGE).unwrap(),
                vec![ContentBlock::text("inspect the workspace").unwrap()],
                timestamp(),
            )
            .unwrap(),
        },
    ));
    SessionReducer::replay(records).unwrap()
}

fn reasoning_provider() -> ScriptedModelProvider {
    provider_with_capabilities(ModelCapabilities::text().with_tools(false).with_reasoning())
}

fn provider_with_capabilities(capabilities: ModelCapabilities) -> ScriptedModelProvider {
    let provider_id = ProviderId::from_str("fake").unwrap();
    let model = ModelSpec::new(
        ModelId::from_str("fake/model").unwrap(),
        provider_id.clone(),
        ModelDisplayName::from_str("Fake Model").unwrap(),
        TokenCount::new(32_000).unwrap(),
        TokenCount::new(4_000).unwrap(),
        capabilities,
    )
    .unwrap();
    ScriptedModelProvider::new(provider_id, vec![model], [])
}

fn provider() -> ScriptedModelProvider {
    provider_with_capabilities(ModelCapabilities::text().with_tools(false))
}

fn advertised_model(provider: &ScriptedModelProvider) -> &ModelSpec {
    provider
        .models()
        .first()
        .expect("test provider has a model")
}

#[test]
fn prospective_request_validates_committed_and_pending_messages_together() {
    let state = state(true);
    let provider = provider();
    let pending = (0..tea_model::MAX_REQUEST_MESSAGES)
        .map(|index| {
            CanonicalMessage::user(
                MessageId::from_str(&format!("0196a0b1-{index:04x}-7000-8000-000000000001"))
                    .unwrap(),
                vec![ContentBlock::text("pending").unwrap()],
                timestamp(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();

    let error = TurnRequestSnapshot::build_with_pending_messages(
        &state,
        &pending,
        &config(),
        &registry(),
        advertised_model(&provider),
    )
    .unwrap_err();

    assert_eq!(error.code(), KernelErrorCode::InvalidRequest);
    assert_eq!(error.message(), "model request contains too many messages");
    assert_eq!(state.messages().len(), 1);
}

fn registry() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    for name in ["z_read", "a_read"] {
        let spec = ToolSpec::new(
            ToolName::from_str(name).unwrap(),
            ToolVersion::from_str("1.0.0").unwrap(),
            "Reads deterministic fake data.",
            json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}),
            json!({"type":"object","properties":{"content":{"type":"string"}},"required":["content"]}),
            [ToolEffect::FsRead],
            ToolExecutionSemantics::new(
                ToolIdempotency::Idempotent,
                ToolRetrySafety::Automatic,
                ToolConcurrency::Parallel,
                ToolTimeout::from_millis(1_000).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        registry
            .register(
                spec,
                Arc::new(StaticResourceResolver::new([]).unwrap()),
                Arc::new(FakeReadTool::new([])),
            )
            .unwrap();
    }
    registry
}

fn web_search_spec() -> ToolSpec {
    ToolSpec::new(
        ToolName::from_str("web_search").unwrap(),
        ToolVersion::from_str("1.0.0").unwrap(),
        "Searches the public web and returns cited sources.",
        json!({
            "type":"object",
            "properties":{"query":{"type":"string"}},
            "required":["query"]
        }),
        json!({"type":"object","properties":{}}),
        [ToolEffect::NetworkRequest],
        ToolExecutionSemantics::new(
            ToolIdempotency::Idempotent,
            ToolRetrySafety::Automatic,
            ToolConcurrency::Serial,
            ToolTimeout::from_millis(1_000).unwrap(),
        )
        .unwrap(),
    )
    .unwrap()
}

fn hybrid_web_search_registry() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry
        .register_hybrid(
            web_search_spec(),
            HostedToolOptions::WebSearch(WebSearchOptions::new()),
            ToolRoutePreference::PreferHosted,
            Arc::new(StaticResourceResolver::new([]).unwrap()),
            Arc::new(FakeReadTool::new([])),
        )
        .unwrap();
    registry
}

fn config() -> KernelRunConfig {
    KernelRunConfig::new(
        ActorId::from_str("user:alice").unwrap(),
        PolicyEnvironment::new(
            ExecutionSurface::Test,
            PolicyExecutionTarget::Native,
            ProtocolMetadata::default(),
        ),
    )
    .with_system_prompt("Follow the test contract.")
    .unwrap()
}

#[test]
fn request_uses_committed_state_and_canonical_serial_tools() {
    let state = state(true);
    let provider = provider();
    let snapshot =
        TurnRequestSnapshot::build(&state, &config(), &registry(), advertised_model(&provider))
            .unwrap();
    assert_eq!(snapshot.durable_tail(), SessionSequence::new(2));
    assert_eq!(snapshot.request().messages(), state.messages());
    assert_eq!(
        snapshot.request().system_prompt(),
        Some("Follow the test contract.")
    );
    assert_eq!(
        snapshot
            .request()
            .tools()
            .iter()
            .map(tea_model::ModelToolDefinition::name)
            .collect::<Vec<_>>(),
        ["a_read", "z_read"]
    );
    assert!(!snapshot.request().allow_parallel_tool_calls());
    assert!(snapshot.request().metadata().is_empty());
    assert_eq!(
        snapshot
            .client_tool_names()
            .iter()
            .map(ToolName::as_str)
            .collect::<Vec<_>>(),
        ["a_read", "z_read"]
    );
    assert!(snapshot.allows_client_tool_call("a_read"));
    assert!(!snapshot.allows_client_tool_call("web_search"));
}

#[test]
fn explicit_session_reasoning_is_frozen_into_the_request() {
    let state = state_with_reasoning(true, Some(ReasoningEffort::High));
    let provider = reasoning_provider();
    let snapshot =
        TurnRequestSnapshot::build(&state, &config(), &registry(), advertised_model(&provider))
            .unwrap();

    assert_eq!(
        snapshot
            .request()
            .reasoning()
            .map(tea_model::ReasoningOptions::effort),
        Some(ReasoningEffort::High)
    );
}

#[test]
fn request_snapshot_copies_the_frozen_final_output_format() {
    let format = FinalOutputFormat::JsonSchema {
        schema: json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": false
        }),
    };
    let provider = provider_with_capabilities(
        ModelCapabilities::text()
            .with_tools(false)
            .with_final_json_schema_with_tools(),
    );
    let config = config()
        .with_final_output_format(Some(format.clone()))
        .unwrap();

    let snapshot = TurnRequestSnapshot::build(
        &state(true),
        &config,
        &registry(),
        advertised_model(&provider),
    )
    .unwrap();

    assert_eq!(snapshot.request().final_output_format(), Some(&format));
}

#[test]
fn json_object_request_includes_the_required_instruction_once() {
    let provider = provider_with_capabilities(
        ModelCapabilities::text()
            .with_tools(false)
            .with_final_json_object(),
    );
    let config = config()
        .with_final_output_format(Some(FinalOutputFormat::JsonObject))
        .unwrap();
    let snapshot = TurnRequestSnapshot::build(
        &state(true),
        &config,
        &registry(),
        advertised_model(&provider),
    )
    .unwrap();
    let expected = format!("Follow the test contract.\n\n{FINAL_JSON_OBJECT_INSTRUCTION}");
    assert_eq!(snapshot.request().system_prompt(), Some(expected.as_str()));

    let config_with_instruction = KernelRunConfig::new(
        ActorId::from_str("user:alice").unwrap(),
        PolicyEnvironment::new(
            ExecutionSurface::Test,
            PolicyExecutionTarget::Native,
            ProtocolMetadata::default(),
        ),
    )
    .with_system_prompt(expected.clone())
    .unwrap()
    .with_final_output_format(Some(FinalOutputFormat::JsonObject))
    .unwrap();
    let idempotent = TurnRequestSnapshot::build(
        &state(true),
        &config_with_instruction,
        &registry(),
        advertised_model(&provider),
    )
    .unwrap();

    assert_eq!(
        idempotent.request().system_prompt(),
        Some(expected.as_str())
    );
    assert_eq!(
        idempotent
            .request()
            .system_prompt()
            .unwrap()
            .lines()
            .filter(|line| *line == FINAL_JSON_OBJECT_INSTRUCTION)
            .count(),
        1
    );
}

#[test]
fn invalid_final_output_schema_is_rejected_by_run_config() {
    let error = config()
        .with_final_output_format(Some(FinalOutputFormat::JsonSchema {
            schema: json!({"type": 7}),
        }))
        .unwrap_err();

    assert_eq!(error.code(), KernelErrorCode::InvalidRequest);
}

#[tokio::test]
async fn unsupported_final_output_fails_before_run_started_or_durable_change() {
    let provider = provider_with_capabilities(ModelCapabilities::text().with_tools(false));
    let store = common::store().await;
    let before = store.load(common::session_id()).await.unwrap();
    let events = common::EventCollector::default();
    let tools = ToolRegistry::new();
    let config = config()
        .with_final_output_format(Some(FinalOutputFormat::JsonObject))
        .unwrap();

    let error = AgentKernel::new(
        &provider,
        &tools,
        &PolicyEngine::new(),
        &store,
        &common::FixedClock,
        &common::TestIds::default(),
        &events,
    )
    .run(common::session_id(), &config, CancellationScope::new())
    .await
    .unwrap_err();

    assert_eq!(error.code(), KernelErrorCode::InvalidRequest);
    assert!(events.events().is_empty());
    assert!(provider.captured_requests().unwrap().is_empty());
    let after = store.load(common::session_id()).await.unwrap();
    assert_eq!(after.records(), before.records());
    assert_eq!(after.state().messages(), before.state().messages());
}

#[tokio::test]
async fn provider_preflight_fails_before_run_started_stream_or_durable_change() {
    let provider = common::RejectingPreflightProvider::new(
        [ScriptedModelResponse::text(["must not stream"])],
        1,
    );
    let store = common::store().await;
    let before = store.load(common::session_id()).await.unwrap();
    let events = common::EventCollector::default();
    let tools = ToolRegistry::new();

    let error = AgentKernel::new(
        &provider,
        &tools,
        &PolicyEngine::new(),
        &store,
        &common::FixedClock,
        &common::TestIds::default(),
        &events,
    )
    .run(common::session_id(), &config(), CancellationScope::new())
    .await
    .unwrap_err();

    assert_eq!(error.code(), KernelErrorCode::InvalidRequest);
    assert_eq!(provider.validate_calls(), 1);
    assert!(events.events().is_empty());
    assert!(provider.captured_requests().is_empty());
    assert_eq!(provider.remaining_scripts(), 1);
    let after = store.load(common::session_id()).await.unwrap();
    assert_eq!(after.journal_revision(), before.journal_revision());
    assert_eq!(after.records(), before.records());
    assert_eq!(after.approval_artifacts(), before.approval_artifacts());
}

#[test]
fn model_default_remains_an_implicit_provider_choice() {
    let provider = reasoning_provider();
    let snapshot = TurnRequestSnapshot::build(
        &state(true),
        &config(),
        &registry(),
        advertised_model(&provider),
    )
    .unwrap();

    assert_eq!(
        reasoning_provider().models()[0]
            .reasoning_profile()
            .unwrap()
            .default_effort(),
        ReasoningEffort::Medium
    );
    assert!(snapshot.request().reasoning().is_none());
}

#[test]
fn explicit_off_is_distinct_from_inheriting_the_model_default() {
    let state = state_with_reasoning(true, Some(ReasoningEffort::Off));
    let provider = reasoning_provider();
    let snapshot =
        TurnRequestSnapshot::build(&state, &config(), &registry(), advertised_model(&provider))
            .unwrap();

    assert_eq!(
        snapshot
            .request()
            .reasoning()
            .map(tea_model::ReasoningOptions::effort),
        Some(ReasoningEffort::Off)
    );
}

#[test]
fn non_reasoning_models_do_not_receive_reasoning_options() {
    let state = state_with_reasoning(true, Some(ReasoningEffort::Off));
    let provider = provider();
    let snapshot =
        TurnRequestSnapshot::build(&state, &config(), &registry(), advertised_model(&provider))
            .unwrap();

    assert!(snapshot.request().reasoning().is_none());
}

#[test]
fn hybrid_projection_freezes_only_client_executable_names() {
    let registry = hybrid_web_search_registry();
    let hosted_provider = provider_with_capabilities(
        ModelCapabilities::text().with_hosted_tool(HostedToolKind::WebSearch),
    );
    let hosted = TurnRequestSnapshot::build(
        &state(true),
        &config(),
        &registry,
        advertised_model(&hosted_provider),
    )
    .unwrap();
    assert!(hosted.request().tools()[0].as_hosted().is_some());
    assert!(hosted.client_tool_names().is_empty());
    assert!(!hosted.allows_client_tool_call("web_search"));
    assert!(hosted.is_hosted_tool_projection("web_search"));

    let client_provider = provider_with_capabilities(ModelCapabilities::text().with_tools(false));
    let client = TurnRequestSnapshot::build(
        &state(true),
        &config(),
        &registry,
        advertised_model(&client_provider),
    )
    .unwrap();
    assert!(client.request().tools()[0].as_function().is_some());
    assert_eq!(
        client.client_tool_names(),
        [ToolName::from_str("web_search").unwrap()]
    );
    assert!(client.allows_client_tool_call("web_search"));
    assert!(!client.is_hosted_tool_projection("web_search"));
}

#[test]
fn request_snapshot_is_immutable_when_later_state_changes() {
    let original = state_with_reasoning(true, Some(ReasoningEffort::High));
    let provider = reasoning_provider();
    let snapshot = TurnRequestSnapshot::build(
        &original,
        &config(),
        &registry(),
        advertised_model(&provider),
    )
    .unwrap();
    let captured = snapshot.request().clone();
    let mut changed = original.messages().to_vec();
    changed.push(
        CanonicalMessage::user(
            MessageId::from_str("0195a0b1-5e54-7a4d-8000-0aa7aa000026").unwrap(),
            vec![ContentBlock::text("later input").unwrap()],
            timestamp(),
        )
        .unwrap(),
    );
    assert_eq!(captured.messages().len(), 1);
    assert_eq!(changed.len(), 2);
    assert_eq!(
        captured
            .reasoning()
            .map(tea_model::ReasoningOptions::effort),
        Some(ReasoningEffort::High)
    );
}

#[test]
fn consecutive_turn_requests_preserve_the_cacheable_prefix() {
    let first_state = state(true);
    let config = config();
    let tools = registry();
    let provider = provider();
    let first =
        TurnRequestSnapshot::build(&first_state, &config, &tools, advertised_model(&provider))
            .unwrap();

    let mut records = vec![
        envelope(
            0,
            CREATED,
            SessionRecord::SessionCreated {
                profile_id: ProfileId::from_str("coding").unwrap(),
                metadata: ProtocolMetadata::default(),
            },
        ),
        envelope(
            1,
            CONFIGURED,
            SessionRecord::ConfigurationChanged {
                model: Some(tea_protocol::ModelRef::new(
                    "fake".parse().unwrap(),
                    ModelId::from_str("fake/model").unwrap(),
                )),
                profile_id: None,
                reasoning_effort: None,
            },
        ),
        envelope(
            2,
            MESSAGE_RECORD,
            SessionRecord::MessageCommitted {
                message: first_state.messages()[0].clone(),
            },
        ),
    ];
    records.push(envelope(
        3,
        "0195a0b1-5e54-7a4d-8000-0aa7aa000026",
        SessionRecord::MessageCommitted {
            message: CanonicalMessage::assistant(
                MessageId::from_str("0195a0b1-5e55-76b8-8000-0aa7aa000027").unwrap(),
                vec![ContentBlock::text("workspace inspected").unwrap()],
                StopReason::Completed,
                timestamp(),
            )
            .unwrap(),
        },
    ));
    let second_state = SessionReducer::replay(records).unwrap();
    let second =
        TurnRequestSnapshot::build(&second_state, &config, &tools, advertised_model(&provider))
            .unwrap();

    assert_eq!(
        second.request().system_prompt(),
        first.request().system_prompt()
    );
    assert_eq!(second.request().tools(), first.request().tools());
    assert_eq!(second.request().metadata(), first.request().metadata());
    assert!(
        second
            .request()
            .messages()
            .starts_with(first.request().messages())
    );
    assert_eq!(
        second.request().messages().len(),
        first.request().messages().len() + 1
    );
}

#[test]
fn missing_or_mismatched_model_fails_closed() {
    let provider = provider();
    let error = TurnRequestSnapshot::build(
        &state(false),
        &config(),
        &registry(),
        advertised_model(&provider),
    )
    .unwrap_err();
    assert_eq!(error.code(), KernelErrorCode::InvalidModel);

    let mismatched = ModelSpec::new(
        ModelId::from_str("fake/other").unwrap(),
        ProviderId::from_str("fake").unwrap(),
        ModelDisplayName::from_str("Other Model").unwrap(),
        TokenCount::new(32_000).unwrap(),
        TokenCount::new(4_000).unwrap(),
        ModelCapabilities::text(),
    )
    .unwrap();
    let error =
        TurnRequestSnapshot::build(&state(true), &config(), &registry(), &mismatched).unwrap_err();
    assert_eq!(error.code(), KernelErrorCode::InvalidModel);
}

#[test]
fn compiled_prompt_is_immutable_and_exclusive_with_legacy_prompt() {
    let segment = PromptSegment::new(
        PromptSegmentId::from_str("product.identity").unwrap(),
        "Compiled identity.",
        PromptProvenance::new(
            ContextProviderId::from_str("test.context").unwrap(),
            "test",
            None,
        )
        .unwrap(),
        TrustLevel::Trusted,
        CacheScope::Profile,
        BudgetBehavior::Required,
    )
    .unwrap();
    let module = PromptModule::new(
        PromptModuleId::from_str("product.identity").unwrap(),
        PromptAuthority::Product,
        PromptPriority::new(0),
        vec![segment],
    )
    .unwrap();
    let compiled = PromptCompiler
        .compile([module], PromptBudget::new(1024, 1024).unwrap())
        .unwrap();
    let config = KernelRunConfig::new(
        ActorId::from_str("user:alice").unwrap(),
        PolicyEnvironment::new(
            ExecutionSurface::Test,
            PolicyExecutionTarget::Native,
            ProtocolMetadata::default(),
        ),
    )
    .with_compiled_prompt(compiled.clone())
    .unwrap();
    let provider = provider();
    let snapshot = TurnRequestSnapshot::build(
        &state(true),
        &config,
        &registry(),
        advertised_model(&provider),
    )
    .unwrap();
    assert_eq!(
        snapshot.request().system_prompt(),
        Some("Compiled identity.")
    );
    assert_eq!(config.compiled_prompt(), Some(&compiled));
    assert!(config.clone().with_system_prompt("legacy").is_err());

    let legacy = KernelRunConfig::new(
        ActorId::from_str("user:alice").unwrap(),
        PolicyEnvironment::new(
            ExecutionSurface::Test,
            PolicyExecutionTarget::Native,
            ProtocolMetadata::default(),
        ),
    )
    .with_system_prompt("legacy")
    .unwrap();
    assert!(legacy.with_compiled_prompt(compiled).is_err());
}

#[test]
fn run_config_rejects_invalid_system_prompts() {
    let base = KernelRunConfig::new(
        ActorId::from_str("user:alice").unwrap(),
        PolicyEnvironment::new(
            ExecutionSurface::Test,
            PolicyExecutionTarget::Native,
            ProtocolMetadata::default(),
        ),
    );
    assert!(base.clone().with_system_prompt("").is_err());
    assert!(base.with_system_prompt("invalid\0prompt").is_err());
}
