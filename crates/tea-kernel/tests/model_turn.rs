use std::future::pending;
use std::str::FromStr;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::json;
use tea_control::CancellationScope;
use tea_kernel::{
    AgentKernel, KernelClock, KernelDeadlineFuture, KernelError, KernelErrorCode,
    KernelEventFuture, KernelEventSink, KernelIdSource, KernelRunConfig, RunState,
};
use tea_model::{
    HostedToolCompleted, HostedToolKind, HostedToolOptions, HostedToolStarted, ModelCapabilities,
    ModelCompletion, ModelDisplayName, ModelEvent, ModelFailureCode, ModelResponseInfo,
    ModelSourceCitation, ModelSpec, ModelStreamIndex, ProviderId, ProviderToolCallId,
    ToolCallStarted, Utf8Delta, WebSearchOptions,
};
use tea_policy::{
    ActorId, ExecutionSurface, GrantId, PolicyEngine, PolicyEnvironment, PolicyExecutionTarget,
};
use tea_protocol::{
    AgentEventType, ApprovalId, CanonicalMessage, ContentBlock, EventEnvelope, EventId,
    ExternalSource, FinalOutputFormat, HostedToolOutcome, MessageId, ModelId, ProfileId,
    ProtocolMetadata, ProtocolTimestamp, ProviderContinuation, RecordEnvelope, RecordId, RunId,
    SessionId, SessionRecord, SessionSequence, SourceCitation, StopReason, TokenCount, ToolCallId,
    ToolIdempotency, TurnId,
};
use tea_session::{AppendTransaction, InMemorySessionStore, SessionStore};
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
const NOW: &str = "2026-07-23T09:30:12.125Z";

#[derive(Debug)]
struct FixedClock;

impl KernelClock for FixedClock {
    fn now(&self) -> Result<ProtocolTimestamp, KernelError> {
        Ok(ProtocolTimestamp::from_str(NOW).unwrap())
    }

    fn sleep_until(&self, _deadline: ProtocolTimestamp) -> KernelDeadlineFuture<'_> {
        Box::pin(pending())
    }
}

#[derive(Debug, Default)]
struct DeterministicIds(AtomicU16);

impl DeterministicIds {
    fn next<T>(&self) -> Result<T, KernelError>
    where
        T: FromStr,
        T::Err: std::fmt::Display,
    {
        let value = self.0.fetch_add(1, Ordering::SeqCst);
        let text = format!("0195a0b1-{value:04x}-7000-8000-000000000001");
        text.parse().map_err(|error: T::Err| {
            KernelError::new(tea_kernel::KernelErrorCode::IdExhausted, error.to_string())
        })
    }
}

impl KernelIdSource for DeterministicIds {
    fn next_run_id(&self) -> Result<RunId, KernelError> {
        self.next()
    }
    fn next_turn_id(&self) -> Result<TurnId, KernelError> {
        self.next()
    }
    fn next_message_id(&self) -> Result<MessageId, KernelError> {
        self.next()
    }
    fn next_tool_call_id(&self) -> Result<ToolCallId, KernelError> {
        self.next()
    }
    fn next_approval_id(&self) -> Result<ApprovalId, KernelError> {
        self.next()
    }
    fn next_grant_id(&self) -> Result<GrantId, KernelError> {
        self.next()
    }
    fn next_event_id(&self) -> Result<EventId, KernelError> {
        self.next()
    }
    fn next_record_id(&self) -> Result<RecordId, KernelError> {
        self.next()
    }
}

#[derive(Debug, Default)]
struct EventCollector(Mutex<Vec<EventEnvelope>>);

impl EventCollector {
    fn events(&self) -> Vec<EventEnvelope> {
        self.0.lock().unwrap().clone()
    }
}

impl KernelEventSink for EventCollector {
    fn emit(&self, event: EventEnvelope) -> KernelEventFuture<'_> {
        Box::pin(async move {
            self.0.lock().unwrap().push(event);
            Ok(())
        })
    }
}

fn timestamp() -> ProtocolTimestamp {
    ProtocolTimestamp::from_str(NOW).unwrap()
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

async fn store() -> InMemorySessionStore {
    let store = InMemorySessionStore::new();
    let records = vec![
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
                message: CanonicalMessage::user(
                    MessageId::from_str(MESSAGE).unwrap(),
                    vec![ContentBlock::text("answer briefly").unwrap()],
                    timestamp(),
                )
                .unwrap(),
            },
        ),
    ];
    store
        .append(AppendTransaction::new(
            SessionId::from_str(SESSION).unwrap(),
            None,
            records,
        ))
        .await
        .unwrap();
    store
}

fn provider_with_capabilities(
    script: ScriptedModelResponse,
    capabilities: ModelCapabilities,
) -> ScriptedModelProvider {
    provider_with_scripts([script], capabilities)
}

fn provider_with_scripts<const N: usize>(
    scripts: [ScriptedModelResponse; N],
    capabilities: ModelCapabilities,
) -> ScriptedModelProvider {
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
    ScriptedModelProvider::new(provider_id, vec![model], scripts)
}

fn provider(script: ScriptedModelResponse) -> ScriptedModelProvider {
    provider_with_capabilities(script, ModelCapabilities::text())
}

fn web_search_spec() -> ToolSpec {
    ToolSpec::new(
        ToolName::from_str("web_search").unwrap(),
        ToolVersion::from_str("1.0.0").unwrap(),
        "Searches the public web.",
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

fn hosted_tools() -> ToolRegistry {
    let mut tools = ToolRegistry::new();
    tools
        .register_hosted(
            web_search_spec(),
            HostedToolOptions::WebSearch(WebSearchOptions::new()),
        )
        .unwrap();
    tools
}

fn hybrid_tools() -> ToolRegistry {
    let mut tools = ToolRegistry::new();
    tools
        .register_hybrid(
            web_search_spec(),
            HostedToolOptions::WebSearch(WebSearchOptions::new()),
            ToolRoutePreference::PreferHosted,
            Arc::new(StaticResourceResolver::new([]).unwrap()),
            Arc::new(FakeReadTool::new([])),
        )
        .unwrap();
    tools
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
}

async fn run_with_final_output(
    format: FinalOutputFormat,
    text: &str,
) -> (
    Result<tea_kernel::KernelRunOutcome, KernelError>,
    InMemorySessionStore,
) {
    let capabilities = match &format {
        FinalOutputFormat::JsonObject => ModelCapabilities::text().with_final_json_object(),
        FinalOutputFormat::JsonSchema { .. } => ModelCapabilities::text().with_final_json_schema(),
    };
    let tools = ToolRegistry::new();
    run_final_output_script(
        format,
        ScriptedModelResponse::text([text]),
        capabilities,
        &tools,
    )
    .await
}

async fn run_final_output_script(
    format: FinalOutputFormat,
    script: ScriptedModelResponse,
    capabilities: ModelCapabilities,
    tools: &ToolRegistry,
) -> (
    Result<tea_kernel::KernelRunOutcome, KernelError>,
    InMemorySessionStore,
) {
    let provider = provider_with_capabilities(script, capabilities);
    let store = store().await;
    let events = EventCollector::default();
    let ids = DeterministicIds::default();
    let policy = PolicyEngine::new();
    let run_config = config().with_final_output_format(Some(format)).unwrap();
    let outcome = AgentKernel::new(
        &provider,
        tools,
        &policy,
        &store,
        &FixedClock,
        &ids,
        &events,
    )
    .run(
        SessionId::from_str(SESSION).unwrap(),
        &run_config,
        CancellationScope::new(),
    )
    .await;
    (outcome, store)
}

async fn assert_structured_output_rejected(
    outcome: Result<tea_kernel::KernelRunOutcome, KernelError>,
    store: &InMemorySessionStore,
    expected_message: &str,
) {
    let error = outcome.unwrap_err();
    assert_eq!(error.code(), KernelErrorCode::ModelFailure);
    assert_eq!(
        error.model_failure_code(),
        Some(ModelFailureCode::MalformedResponse)
    );
    assert_eq!(error.message(), expected_message);
    let snapshot = store
        .load(SessionId::from_str(SESSION).unwrap())
        .await
        .unwrap();
    assert_eq!(snapshot.state().messages().len(), 1);
    assert!(!snapshot.records().iter().any(|record| matches!(
        record.record(),
        SessionRecord::MessageCommitted {
            message: CanonicalMessage::Assistant { .. }
        }
    )));
    assert!(matches!(
        snapshot.records().last().unwrap().record(),
        SessionRecord::RunInterrupted { .. }
    ));
}

#[tokio::test]
async fn json_object_final_output_is_committed_after_validation() {
    let (outcome, store) =
        run_with_final_output(FinalOutputFormat::JsonObject, r#"{"answer":"ok"}"#).await;

    assert_eq!(outcome.unwrap().state(), RunState::Completed);
    assert_eq!(
        store
            .load(SessionId::from_str(SESSION).unwrap())
            .await
            .unwrap()
            .state()
            .messages()
            .len(),
        2
    );
}

#[tokio::test]
async fn json_schema_output_uses_the_run_output_budget_instead_of_tool_value_bounds() {
    let text = format!(r#"{{"payload":"{}"}}"#, "x".repeat(300 * 1024));
    let script = ScriptedModelResponse::text(
        text.as_bytes()
            .chunks(60 * 1024)
            .map(|chunk| String::from_utf8(chunk.to_vec()).unwrap()),
    );
    let format = FinalOutputFormat::JsonSchema {
        schema: json!({
            "type": "object",
            "properties": {"payload": {"type": "string"}},
            "required": ["payload"],
            "additionalProperties": false
        }),
    };
    let tools = ToolRegistry::new();

    let (outcome, store) = run_final_output_script(
        format,
        script,
        ModelCapabilities::text().with_final_json_schema(),
        &tools,
    )
    .await;

    assert_eq!(outcome.unwrap().state(), RunState::Completed);
    let snapshot = store
        .load(SessionId::from_str(SESSION).unwrap())
        .await
        .unwrap();
    assert_eq!(snapshot.state().messages().len(), 2);
}

#[tokio::test]
async fn json_schema_output_does_not_inherit_the_tool_value_depth_limit() {
    let text = format!("{}null{}", "{\"next\":".repeat(40), "}".repeat(40));
    let tools = ToolRegistry::new();

    let (outcome, _) = run_final_output_script(
        FinalOutputFormat::JsonSchema { schema: json!({}) },
        ScriptedModelResponse::text([text]),
        ModelCapabilities::text().with_final_json_schema(),
        &tools,
    )
    .await;

    assert_eq!(outcome.unwrap().state(), RunState::Completed);
}

#[tokio::test]
async fn json_object_array_is_rejected_before_assistant_commit() {
    let (outcome, store) = run_with_final_output(FinalOutputFormat::JsonObject, "[]").await;

    assert_structured_output_rejected(outcome, &store, "structured output is not a JSON object")
        .await;
}

#[tokio::test]
async fn malformed_json_is_rejected_before_assistant_commit() {
    let (outcome, store) = run_with_final_output(FinalOutputFormat::JsonObject, "{").await;

    assert_structured_output_rejected(outcome, &store, "structured output is not valid JSON").await;
}

#[tokio::test]
async fn json_schema_mismatch_is_rejected_before_assistant_commit() {
    let format = FinalOutputFormat::JsonSchema {
        schema: json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": false
        }),
    };
    let (outcome, store) = run_with_final_output(format, r#"{"answer":7}"#).await;

    assert_structured_output_rejected(
        outcome,
        &store,
        "structured output violates its JSON Schema",
    )
    .await;
}

#[tokio::test]
async fn refusal_and_length_are_rejected_before_assistant_commit() {
    for stop_reason in [StopReason::Refusal, StopReason::Length] {
        let script = ScriptedModelResponse::events([
            ModelEvent::Started(ModelResponseInfo::new()),
            ModelEvent::TextDelta(Utf8Delta::new(r#"{"answer":"partial"}"#).unwrap()),
            ModelEvent::Completed(ModelCompletion::new(stop_reason).unwrap()),
        ]);
        let tools = ToolRegistry::new();
        let (outcome, store) = run_final_output_script(
            FinalOutputFormat::JsonObject,
            script,
            ModelCapabilities::text().with_final_json_object(),
            &tools,
        )
        .await;

        assert_structured_output_rejected(
            outcome,
            &store,
            "structured output did not complete normally",
        )
        .await;
    }
}

#[tokio::test]
async fn thinking_and_multiple_text_blocks_form_one_json_value() {
    let script = ScriptedModelResponse::events([
        ModelEvent::Started(ModelResponseInfo::new()),
        ModelEvent::TextDelta(Utf8Delta::new("{\"answer\":\"").unwrap()),
        ModelEvent::ThinkingDelta(Utf8Delta::new("checking schema").unwrap()),
        ModelEvent::TextDelta(Utf8Delta::new("ok\"}").unwrap()),
        ModelEvent::Completed(ModelCompletion::completed()),
    ]);
    let tools = ToolRegistry::new();
    let (outcome, store) = run_final_output_script(
        FinalOutputFormat::JsonObject,
        script,
        ModelCapabilities::text().with_final_json_object(),
        &tools,
    )
    .await;

    assert_eq!(outcome.unwrap().state(), RunState::Completed);
    let snapshot = store
        .load(SessionId::from_str(SESSION).unwrap())
        .await
        .unwrap();
    let CanonicalMessage::Assistant { content, .. } = &snapshot.state().messages()[1] else {
        panic!("expected assistant message")
    };
    assert!(matches!(
        content.as_slice(),
        [
            ContentBlock::Text { text: first },
            ContentBlock::Thinking { .. },
            ContentBlock::Text { text: second }
        ] if first == "{\"answer\":\"" && second == "ok\"}"
    ));
}

#[tokio::test]
async fn hosted_tool_activity_and_citations_do_not_invalidate_structured_json() {
    let tools = hosted_tools();
    let (outcome, store) = run_final_output_script(
        FinalOutputFormat::JsonObject,
        hosted_search_response_with_text(r#"{"answer":"Cited answer"}"#),
        ModelCapabilities::text()
            .with_final_json_object()
            .with_hosted_tool(HostedToolKind::WebSearch),
        &tools,
    )
    .await;

    assert_eq!(outcome.unwrap().state(), RunState::Completed);
    let snapshot = store
        .load(SessionId::from_str(SESSION).unwrap())
        .await
        .unwrap();
    let CanonicalMessage::Assistant { content, .. } = &snapshot.state().messages()[1] else {
        panic!("expected assistant message");
    };
    assert!(matches!(content[0], ContentBlock::HostedTool { .. }));
    assert!(matches!(
        &content[1],
        ContentBlock::Text { text } if text == r#"{"answer":"Cited answer"}"#
    ));
    assert!(matches!(content[2], ContentBlock::Citation { .. }));
}

#[tokio::test]
async fn model_only_turn_streams_then_commits_message_and_checkpoint() {
    let provider = provider(ScriptedModelResponse::events([
        ModelEvent::Started(ModelResponseInfo::new()),
        ModelEvent::ThinkingDelta(Utf8Delta::new("brief thought").unwrap()),
        ModelEvent::TextDelta(Utf8Delta::new("final answer").unwrap()),
        ModelEvent::Completed(ModelCompletion::completed()),
    ]));
    let store = store().await;
    let events = EventCollector::default();
    let ids = DeterministicIds::default();
    let tools = ToolRegistry::new();
    let policy = PolicyEngine::new();
    let kernel = AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &store,
        &FixedClock,
        &ids,
        &events,
    );

    let outcome = kernel
        .run(
            SessionId::from_str(SESSION).unwrap(),
            &config(),
            CancellationScope::new(),
        )
        .await
        .unwrap();
    assert_eq!(outcome.state(), RunState::Completed);
    assert_eq!(outcome.session().tail_sequence(), SessionSequence::new(4));
    assert_eq!(outcome.session().messages().len(), 2);
    assert_eq!(
        events
            .events()
            .iter()
            .map(EventEnvelope::event_type)
            .collect::<Vec<_>>(),
        [
            AgentEventType::RunStarted,
            AgentEventType::MessageDelta,
            AgentEventType::MessageDelta,
            AgentEventType::TurnCheckpointed,
            AgentEventType::RunFinished,
        ]
    );
    let snapshot = store
        .load(SessionId::from_str(SESSION).unwrap())
        .await
        .unwrap();
    assert!(matches!(
        snapshot.records()[3].record(),
        SessionRecord::MessageCommitted {
            message: CanonicalMessage::Assistant { .. }
        }
    ));
    assert!(matches!(
        snapshot.records()[4].record(),
        SessionRecord::TurnCheckpointed { .. }
    ));
    let requests = provider.captured_requests().unwrap();
    assert_eq!(requests.len(), 1);
    assert_eq!(requests[0].messages().len(), 1);
}

#[tokio::test]
async fn hosted_tool_activity_is_committed_without_local_execution() {
    let provider = provider_with_capabilities(
        hosted_search_response(),
        ModelCapabilities::text().with_hosted_tool(HostedToolKind::WebSearch),
    );
    let store = store().await;
    let events = EventCollector::default();
    let ids = DeterministicIds::default();
    let tools = hosted_tools();
    let policy = PolicyEngine::new();
    let kernel = AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &store,
        &FixedClock,
        &ids,
        &events,
    );

    let outcome = kernel
        .run(
            SessionId::from_str(SESSION).unwrap(),
            &config(),
            CancellationScope::new(),
        )
        .await
        .unwrap();
    let CanonicalMessage::Assistant { content, .. } = &outcome.session().messages()[1] else {
        panic!("expected assistant message");
    };
    assert!(matches!(
        &content[0],
        ContentBlock::HostedTool { activity }
            if activity.tool_name() == "web_search"
                && activity.sources()[0].title() == Some("Example result")
    ));
    assert!(matches!(&content[1], ContentBlock::Text { text } if text == "Cited answer"));
    assert!(matches!(
        &content[2],
        ContentBlock::Citation { citation } if citation.tool_call_id().is_some()
    ));
    assert_hosted_observations(&events);
}

#[tokio::test]
async fn unprojected_hosted_activity_is_rejected_before_observation_or_commit() {
    let index = ModelStreamIndex::new(0).unwrap();
    let provider_call_id = ProviderToolCallId::from_str("unexpected_hosted_call").unwrap();
    let provider = provider_with_capabilities(
        ScriptedModelResponse::events([
            ModelEvent::Started(ModelResponseInfo::new()),
            ModelEvent::HostedToolStarted(
                HostedToolStarted::new(index, provider_call_id, "web_search").unwrap(),
            ),
        ]),
        ModelCapabilities::text().with_hosted_tool(HostedToolKind::WebSearch),
    );
    let store = store().await;
    let events = EventCollector::default();

    let error = AgentKernel::new(
        &provider,
        &ToolRegistry::new(),
        &PolicyEngine::new(),
        &store,
        &FixedClock,
        &DeterministicIds::default(),
        &events,
    )
    .run(
        SessionId::from_str(SESSION).unwrap(),
        &config(),
        CancellationScope::new(),
    )
    .await
    .unwrap_err();

    assert_eq!(error.code(), KernelErrorCode::ModelFailure);
    assert!(!events.events().iter().any(|event| {
        matches!(
            event.event_type(),
            AgentEventType::HostedToolStarted
                | AgentEventType::HostedToolCompleted
                | AgentEventType::ToolCallRequested
        )
    }));
    let snapshot = store
        .load(SessionId::from_str(SESSION).unwrap())
        .await
        .unwrap();
    assert_eq!(snapshot.state().messages().len(), 1);
}

fn hosted_search_response() -> ScriptedModelResponse {
    hosted_search_response_with_text("Cited answer")
}

fn hosted_search_response_with_text(text: &str) -> ScriptedModelResponse {
    let index = ModelStreamIndex::new(0).unwrap();
    let provider_call_id = ProviderToolCallId::from_str("ws_123").unwrap();
    let source = ExternalSource::new("https://example.com/result")
        .unwrap()
        .with_title("Example result")
        .unwrap();
    let continuation = ProviderContinuation::new(
        "openai",
        "openai.responses.web_search.v1",
        json!({"type":"web_search_call","id":"ws_123"}),
    )
    .unwrap();
    ScriptedModelResponse::events([
        ModelEvent::Started(ModelResponseInfo::new()),
        ModelEvent::HostedToolStarted(
            HostedToolStarted::new(index, provider_call_id.clone(), "web_search").unwrap(),
        ),
        ModelEvent::HostedToolCompleted(
            HostedToolCompleted::new(
                index,
                provider_call_id.clone(),
                "web_search",
                json!({"query":"example"}),
                HostedToolOutcome::Success,
                vec![source.clone()],
                Some(continuation.clone()),
            )
            .unwrap(),
        ),
        ModelEvent::TextDelta(Utf8Delta::new(text).unwrap()),
        ModelEvent::SourceCitation(
            ModelSourceCitation::new(
                Some(provider_call_id),
                SourceCitation::new(source)
                    .with_range(0, 5)
                    .unwrap()
                    .with_continuation(continuation),
            )
            .unwrap(),
        ),
        ModelEvent::Completed(ModelCompletion::completed()),
    ])
}

fn assert_hosted_observations(events: &EventCollector) {
    assert!(
        !events
            .events()
            .iter()
            .any(|event| event.event_type() == AgentEventType::ToolCallRequested)
    );
    let hosted_events = events
        .events()
        .into_iter()
        .filter(|event| {
            matches!(
                event.event_type(),
                AgentEventType::HostedToolStarted | AgentEventType::HostedToolCompleted
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(hosted_events.len(), 2);
    assert!(matches!(
        hosted_events[0].event(),
        tea_protocol::AgentEvent::HostedToolStarted { tool_name, .. }
            if tool_name == "web_search"
    ));
    assert!(matches!(
        hosted_events[1].event(),
        tea_protocol::AgentEvent::HostedToolCompleted {
            arguments,
            outcome: HostedToolOutcome::Success,
            source_count: 1,
            ..
        } if arguments == &json!({"query":"example"})
    ));
    assert!(
        !serde_json::to_string(&hosted_events)
            .unwrap()
            .contains("web_search_call")
    );
}

#[tokio::test]
async fn hosted_projection_rejects_same_named_client_call_before_local_routing() {
    let index = ModelStreamIndex::new(0).unwrap();
    let provider_call_id = ProviderToolCallId::from_str("unexpected_function_call").unwrap();
    let provider = provider_with_capabilities(
        ScriptedModelResponse::events([
            ModelEvent::Started(ModelResponseInfo::new()),
            ModelEvent::ToolCallStarted(
                ToolCallStarted::new(index, provider_call_id, "web_search").unwrap(),
            ),
        ]),
        ModelCapabilities::text().with_hosted_tool(HostedToolKind::WebSearch),
    );
    let store = store().await;
    let events = EventCollector::default();
    let tools = hybrid_tools();

    let error = AgentKernel::new(
        &provider,
        &tools,
        &PolicyEngine::new(),
        &store,
        &FixedClock,
        &DeterministicIds::default(),
        &events,
    )
    .run(
        SessionId::from_str(SESSION).unwrap(),
        &config(),
        CancellationScope::new(),
    )
    .await
    .unwrap_err();

    assert_eq!(error.code(), KernelErrorCode::ModelFailure);
    assert!(
        !events
            .events()
            .iter()
            .any(|event| event.event_type() == AgentEventType::ToolCallRequested)
    );
    let snapshot = store
        .load(SessionId::from_str(SESSION).unwrap())
        .await
        .unwrap();
    assert!(
        !snapshot
            .records()
            .iter()
            .any(|record| matches!(record.record(), SessionRecord::ToolCallRequested { .. }))
    );
}

#[tokio::test]
async fn provider_pause_turn_commits_and_replays_without_local_tool_execution() {
    let provider = provider_with_scripts(
        [
            ScriptedModelResponse::events([
                ModelEvent::Started(ModelResponseInfo::new()),
                ModelEvent::TextDelta(Utf8Delta::new("Search still running").unwrap()),
                ModelEvent::Completed(ModelCompletion::new(StopReason::PauseTurn).unwrap()),
            ]),
            ScriptedModelResponse::text(["Final answer"]),
        ],
        ModelCapabilities::text(),
    );
    let store = store().await;
    let events = EventCollector::default();

    let outcome = AgentKernel::new(
        &provider,
        &ToolRegistry::new(),
        &PolicyEngine::new(),
        &store,
        &FixedClock,
        &DeterministicIds::default(),
        &events,
    )
    .run(
        SessionId::from_str(SESSION).unwrap(),
        &config(),
        CancellationScope::new(),
    )
    .await
    .unwrap();

    assert_eq!(outcome.state(), RunState::Completed);
    assert_eq!(provider.captured_requests().unwrap().len(), 2);
    assert!(matches!(
        &outcome.session().messages()[1],
        CanonicalMessage::Assistant {
            stop_reason: StopReason::PauseTurn,
            ..
        }
    ));
    assert!(
        !events
            .events()
            .iter()
            .any(|event| event.event_type() == AgentEventType::ToolCallRequested)
    );
}
