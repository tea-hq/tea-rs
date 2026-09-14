//! Context-window accounting fails closed before any provider call.

use crate::common;

use std::str::FromStr;

use tea_control::CancellationScope;
use tea_kernel::{AgentKernel, KernelErrorCode, KernelRunConfig};
use tea_model::{ModelCapabilities, ModelDisplayName, ModelSpec, ProviderId};
use tea_policy::{
    ActorId, CodingWorkspacePolicy, ExecutionSurface, PolicyEngine, PolicyEnvironment,
    PolicyExecutionTarget,
};
use tea_protocol::{FinalOutputFormat, ModelId, ProtocolMetadata, TokenCount};
use tea_testkit::ScriptedModelProvider;
use tea_tools::ToolRegistry;

use common::{EventCollector, FixedClock, TestIds, session_id, timestamp};

fn tiny_window_provider(context: u64, output: u64) -> ScriptedModelProvider {
    tiny_window_provider_with_capabilities(context, output, ModelCapabilities::text())
}

fn tiny_window_provider_with_capabilities(
    context: u64,
    output: u64,
    capabilities: ModelCapabilities,
) -> ScriptedModelProvider {
    let provider_id = ProviderId::from_str("tiny").unwrap();
    let model = ModelSpec::new(
        ModelId::from_str("tiny/model").unwrap(),
        provider_id.clone(),
        ModelDisplayName::from_str("Tiny Model").unwrap(),
        TokenCount::new(context).unwrap(),
        TokenCount::new(output).unwrap(),
        capabilities,
    )
    .unwrap();
    // The provider should never be called when the window is exceeded.
    ScriptedModelProvider::new(provider_id, vec![model], [])
}

fn config_for(actor: &str) -> KernelRunConfig {
    KernelRunConfig::new(
        ActorId::from_str(actor).unwrap(),
        PolicyEnvironment::new(
            ExecutionSurface::Test,
            PolicyExecutionTarget::Native,
            ProtocolMetadata::default(),
        ),
    )
}

#[tokio::test]
async fn overflow_fails_before_any_provider_call() {
    // Context window barely fits the reserved output; any prompt overflows.
    let provider = tiny_window_provider(8, 8);
    let store = store_with_tiny_model().await;
    let tools = ToolRegistry::new();
    let mut policy = PolicyEngine::new();
    policy.add_rule(CodingWorkspacePolicy).unwrap();
    let events = EventCollector::default();

    let error = AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &store,
        &FixedClock,
        &TestIds::default(),
        &events,
    )
    .run(
        session_id(),
        &config_for("user:alice"),
        CancellationScope::new(),
    )
    .await
    .unwrap_err();
    assert_eq!(error.code(), KernelErrorCode::ContextOverflow);
    assert!(
        provider.captured_requests().unwrap().is_empty(),
        "no model request should be made on overflow"
    );
}

#[tokio::test]
async fn json_object_instruction_is_included_in_context_accounting() {
    let provider = tiny_window_provider_with_capabilities(
        32,
        8,
        ModelCapabilities::text().with_final_json_object(),
    );
    let store = store_with_tiny_model().await;
    let tools = ToolRegistry::new();
    let events = EventCollector::default();
    let config = config_for("user:alice")
        .with_final_output_format(Some(FinalOutputFormat::JsonObject))
        .unwrap();

    let error = AgentKernel::new(
        &provider,
        &tools,
        &PolicyEngine::new(),
        &store,
        &FixedClock,
        &TestIds::default(),
        &events,
    )
    .run(session_id(), &config, CancellationScope::new())
    .await
    .unwrap_err();

    assert_eq!(error.code(), KernelErrorCode::ContextOverflow);
    assert!(provider.captured_requests().unwrap().is_empty());
}

#[tokio::test]
async fn small_session_under_window_proceeds() {
    let provider = tiny_window_provider(1_000_000, 1_000);
    let store = store_with_tiny_model().await;
    let tools = ToolRegistry::new();
    let mut policy = PolicyEngine::new();
    policy.add_rule(CodingWorkspacePolicy).unwrap();
    let events = EventCollector::default();
    // A run that overflows the tiny window still fails, but the accountant must
    // be the reason only when the session is large. Here the window is huge so
    // the run fails for a different reason (no script for the text response),
    // proving the accountant did not reject it.
    let error = AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &store,
        &FixedClock,
        &TestIds::default(),
        &events,
    )
    .run(
        session_id(),
        &config_for("user:alice"),
        CancellationScope::new(),
    )
    .await
    .unwrap_err();
    assert_ne!(error.code(), KernelErrorCode::ContextOverflow);
}

async fn store_with_tiny_model() -> tea_session::InMemorySessionStore {
    use tea_protocol::{
        CanonicalMessage, ContentBlock, MessageId, ProtocolMetadata, SessionRecord,
    };
    use tea_session::{AppendTransaction, SessionStore};
    let store = tea_session::InMemorySessionStore::new();
    store
        .append(AppendTransaction::new(
            session_id(),
            None,
            vec![
                tea_protocol::RecordEnvelope::new(
                    "0195a0b1-5e50-7af4-8972-0aa7aa000022".parse().unwrap(),
                    session_id(),
                    tea_protocol::SessionSequence::new(0),
                    timestamp(),
                    None,
                    None,
                    None,
                    ProtocolMetadata::default(),
                    SessionRecord::SessionCreated {
                        profile_id: "coding".parse().unwrap(),
                        metadata: ProtocolMetadata::default(),
                    },
                )
                .unwrap(),
                tea_protocol::RecordEnvelope::new(
                    "0195a0b1-5e51-79e1-8f4a-0aa7aa000023".parse().unwrap(),
                    session_id(),
                    tea_protocol::SessionSequence::new(1),
                    timestamp(),
                    None,
                    None,
                    None,
                    ProtocolMetadata::default(),
                    SessionRecord::ConfigurationChanged {
                        model: Some(tea_protocol::ModelRef::new(
                            "tiny".parse().unwrap(),
                            ModelId::from_str("tiny/model").unwrap(),
                        )),
                        profile_id: None,
                        reasoning_effort: None,
                    },
                )
                .unwrap(),
                tea_protocol::RecordEnvelope::new(
                    "0195a0b1-5e52-7b3e-93f1-0aa7aa000024".parse().unwrap(),
                    session_id(),
                    tea_protocol::SessionSequence::new(2),
                    timestamp(),
                    None,
                    None,
                    None,
                    ProtocolMetadata::default(),
                    SessionRecord::MessageCommitted {
                        message: CanonicalMessage::user(
                            MessageId::from_str("0195a0b1-5e53-74b2-8c25-0aa7aa000025").unwrap(),
                            vec![ContentBlock::text("answer briefly").unwrap()],
                            timestamp(),
                        )
                        .unwrap(),
                    },
                )
                .unwrap(),
            ],
        ))
        .await
        .unwrap();
    store
}
