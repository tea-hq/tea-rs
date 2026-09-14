//! Automatic compaction policy on context overflow.

use crate::common;

use std::collections::BTreeMap;
use std::future;
use std::pin::Pin;
use std::str::FromStr;
use std::sync::Arc;

use tea_control::CancellationScope;
use tea_kernel::{
    AgentKernel, CompactionPolicy, CompactionSummarizer, KernelError, KernelErrorCode,
    KernelRunConfig,
};
use tea_model::{ModelCapabilities, ModelDisplayName, ModelSpec, ProviderId};
use tea_policy::{
    ActorId, CodingWorkspacePolicy, ExecutionSurface, PolicyEngine, PolicyEnvironment,
    PolicyExecutionTarget,
};
use tea_protocol::{
    AgentEventType, CanonicalMessage, ContentBlock, MessageId, ModelId, ProtocolMetadata,
    ProtocolTimestamp, SessionRecord, StopReason, TokenCount,
};
use tea_provider_anthropic::{
    credential::{CredentialResolver, MapCredentialResolver},
    request::build_messages_body,
};
use tea_session::SessionStore;
use tea_testkit::ScriptedModelProvider;
use tea_testkit::ScriptedModelResponse;
use tea_tools::ToolRegistry;

use common::{EventCollector, FixedClock, TestIds, session_id, timestamp};

#[derive(Debug, Clone, Copy, Default)]
struct AlwaysCompact;
impl CompactionPolicy for AlwaysCompact {
    fn should_compact(&self, _estimated_input_tokens: usize, _context_window: u64) -> bool {
        true
    }
}

#[derive(Debug, Default)]
struct FixedSummarizer;
impl CompactionSummarizer for FixedSummarizer {
    fn summarize(
        &self,
        _messages: Vec<CanonicalMessage>,
    ) -> Pin<
        Box<
            dyn future::Future<Output = Result<CanonicalMessage, tea_kernel::KernelError>>
                + Send
                + '_,
        >,
    > {
        Box::pin(future::ready(Ok(summary("auto-compacted summary"))))
    }
}

#[derive(Debug, Default)]
struct FailingSummarizer;
impl CompactionSummarizer for FailingSummarizer {
    fn summarize(
        &self,
        _messages: Vec<CanonicalMessage>,
    ) -> Pin<Box<dyn future::Future<Output = Result<CanonicalMessage, KernelError>> + Send + '_>>
    {
        Box::pin(future::ready(Err(KernelError::new(
            KernelErrorCode::InvalidRequest,
            "compaction summary failed",
        ))))
    }
}

fn summary(text: &str) -> CanonicalMessage {
    CanonicalMessage::assistant(
        MessageId::from_str("0195a0b1-5e90-7000-8000-0000000000c1").unwrap(),
        vec![ContentBlock::text(text).unwrap()],
        StopReason::Completed,
        timestamp(),
    )
    .unwrap()
}

fn tiny_provider() -> ScriptedModelProvider {
    tiny_provider_with_capabilities(
        ScriptedModelResponse::text(["after compaction"]),
        ModelCapabilities::text(),
    )
}

fn tiny_provider_with_capabilities(
    script: ScriptedModelResponse,
    capabilities: ModelCapabilities,
) -> ScriptedModelProvider {
    let provider_id = ProviderId::from_str("tiny").unwrap();
    let model = ModelSpec::new(
        ModelId::from_str("tiny/model").unwrap(),
        provider_id.clone(),
        ModelDisplayName::from_str("Tiny Model").unwrap(),
        // Window fits the summary (small) but not the pre-compaction transcript.
        TokenCount::new(64).unwrap(),
        TokenCount::new(8).unwrap(),
        capabilities,
    )
    .unwrap();
    ScriptedModelProvider::new(provider_id, vec![model], [script])
}

async fn tiny_store() -> tea_session::InMemorySessionStore {
    use tea_protocol::{SessionRecord, SessionSequence};
    let store = tea_session::InMemorySessionStore::new();
    store
        .append(tea_session::AppendTransaction::new(
            session_id(),
            None,
            vec![
                tea_protocol::RecordEnvelope::new(
                    "0195a0b1-5e50-7af4-8972-0aa7aa000022".parse().unwrap(),
                    session_id(),
                    SessionSequence::new(0),
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
                    SessionSequence::new(1),
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
                // A large user message that overflows the tiny window.
                tea_protocol::RecordEnvelope::new(
                    "0195a0b1-5e52-7b3e-93f1-0aa7aa000024".parse().unwrap(),
                    session_id(),
                    SessionSequence::new(2),
                    timestamp(),
                    None,
                    None,
                    None,
                    ProtocolMetadata::default(),
                    SessionRecord::MessageCommitted {
                        message: CanonicalMessage::user(
                            MessageId::from_str("0195a0b1-5e53-74b2-8c25-0aa7aa000025").unwrap(),
                            vec![ContentBlock::text("x".repeat(2_000)).unwrap()],
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

fn auto_config() -> KernelRunConfig {
    KernelRunConfig::new(
        ActorId::from_str("user:alice").unwrap(),
        PolicyEnvironment::new(
            ExecutionSurface::Test,
            PolicyExecutionTarget::Native,
            ProtocolMetadata::default(),
        ),
    )
    .with_compaction_policy(Arc::new(AlwaysCompact))
    .with_compaction_summarizer(Arc::new(FixedSummarizer))
}

fn structured_anthropic_config() -> tea_provider_anthropic::AnthropicConfig {
    MapCredentialResolver::new(
        [
            ("TEA_ANTHROPIC_API_KEY", "sk-ant-test"),
            ("TEA_ANTHROPIC_MODEL", "tiny/model"),
            ("TEA_ANTHROPIC_BASE_URL", "https://api.example.test"),
            ("TEA_ANTHROPIC_FINAL_JSON_SCHEMA", "true"),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_owned(), value.to_owned()))
        .collect::<BTreeMap<_, _>>(),
    )
    .resolve()
    .unwrap()
}

#[tokio::test]
async fn overflow_triggers_auto_compaction_then_completes() {
    let provider = tiny_provider();
    let store = tiny_store().await;
    let tools = ToolRegistry::new();
    let mut policy = PolicyEngine::new();
    policy.add_rule(CodingWorkspacePolicy).unwrap();
    let events = EventCollector::default();

    let outcome = AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &store,
        &FixedClock,
        &TestIds::default(),
        &events,
    )
    .run(session_id(), &auto_config(), CancellationScope::new())
    .await
    .unwrap();
    assert_eq!(outcome.state(), tea_kernel::RunState::Completed);
    let snapshot = store.load(session_id()).await.unwrap();
    assert!(snapshot.state().latest_compaction().is_some());
    // The materialized transcript now starts with the summary.
    assert_eq!(
        snapshot.state().messages()[0],
        summary("auto-compacted summary"),
    );
}

#[tokio::test]
async fn auto_compaction_preserves_the_frozen_final_output_format() {
    let format = tea_protocol::FinalOutputFormat::JsonSchema {
        schema: serde_json::json!({
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": false
        }),
    };
    let provider = tiny_provider_with_capabilities(
        ScriptedModelResponse::text([r#"{"summary":"after compaction"}"#]),
        ModelCapabilities::text().with_final_json_schema(),
    );
    let store = tiny_store().await;
    let tools = ToolRegistry::new();
    let mut policy = PolicyEngine::new();
    policy.add_rule(CodingWorkspacePolicy).unwrap();
    let events = EventCollector::default();
    let config = auto_config()
        .with_final_output_format(Some(format.clone()))
        .unwrap();

    let outcome = AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &store,
        &FixedClock,
        &TestIds::default(),
        &events,
    )
    .run(session_id(), &config, CancellationScope::new())
    .await
    .unwrap();

    assert_eq!(outcome.state(), tea_kernel::RunState::Completed);
    assert!(
        store
            .load(session_id())
            .await
            .unwrap()
            .state()
            .latest_compaction()
            .is_some()
    );
    let requests = provider.captured_requests().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].final_output_format(), Some(&format));
    let body = build_messages_body(&requests[0], &structured_anthropic_config()).unwrap();
    assert_eq!(body["messages"][0]["role"], "user");
    assert_eq!(
        body["messages"][0]["content"][0]["text"],
        "auto-compacted summary"
    );
    assert_eq!(body["output_config"]["format"]["type"], "json_schema");
}

#[tokio::test]
async fn never_policy_terminates_with_context_overflow() {
    let provider = tiny_provider();
    let store = tiny_store().await;
    let tools = ToolRegistry::new();
    let mut policy = PolicyEngine::new();
    policy.add_rule(CodingWorkspacePolicy).unwrap();
    let events = EventCollector::default();
    let config = KernelRunConfig::new(
        ActorId::from_str("user:alice").unwrap(),
        PolicyEnvironment::new(
            ExecutionSurface::Test,
            PolicyExecutionTarget::Native,
            ProtocolMetadata::default(),
        ),
    );
    let error = AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &store,
        &FixedClock,
        &TestIds::default(),
        &events,
    )
    .run(session_id(), &config, CancellationScope::new())
    .await
    .unwrap_err();
    assert_eq!(error.code(), KernelErrorCode::ContextOverflow);
    let snapshot = store.load(session_id()).await.unwrap();
    assert!(snapshot.state().latest_compaction().is_none());
    // Silence unused import.
    let _ = ProtocolTimestamp::from_str(common::NOW).unwrap();
}

#[tokio::test]
async fn summarizer_failure_records_one_durable_and_observed_run_terminal() {
    let provider = tiny_provider();
    let store = tiny_store().await;
    let tools = ToolRegistry::new();
    let policy = PolicyEngine::new();
    let events = EventCollector::default();
    let config = KernelRunConfig::new(
        ActorId::from_str("user:alice").unwrap(),
        PolicyEnvironment::new(
            ExecutionSurface::Test,
            PolicyExecutionTarget::Native,
            ProtocolMetadata::default(),
        ),
    )
    .with_compaction_policy(Arc::new(AlwaysCompact))
    .with_compaction_summarizer(Arc::new(FailingSummarizer));

    let error = AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &store,
        &FixedClock,
        &TestIds::default(),
        &events,
    )
    .run(session_id(), &config, CancellationScope::new())
    .await
    .unwrap_err();

    assert_eq!(error.code(), KernelErrorCode::InvalidRequest);
    let snapshot = store.load(session_id()).await.unwrap();
    assert_eq!(
        snapshot
            .records()
            .iter()
            .filter(|record| matches!(record.record(), SessionRecord::RunInterrupted { .. }))
            .count(),
        1
    );
    assert_eq!(
        events
            .events()
            .iter()
            .filter(|event| event.event_type() == AgentEventType::RunFinished)
            .count(),
        1
    );
}
