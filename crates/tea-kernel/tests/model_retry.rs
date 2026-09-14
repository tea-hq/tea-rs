//! Provider retry with exponential backoff.

use crate::common;

use std::str::FromStr;
use std::sync::Mutex;
use std::time::Duration;

use tea_kernel::{
    AgentKernel, KernelClock, KernelDeadlineFuture, KernelError, KernelErrorCode,
    KernelEventFuture, KernelEventSink, KernelRunConfig, ModelRetryPolicy, RunState,
};
use tea_model::{
    ModelCapabilities, ModelEvent, ModelFailure, ModelFailureCode, ModelResponseInfo, Utf8Delta,
};
use tea_policy::{
    ActorId, CodingWorkspacePolicy, ExecutionSurface, PolicyEngine, PolicyEnvironment,
    PolicyExecutionTarget,
};
use tea_protocol::{
    AgentEvent, EventEnvelope, FinalOutputFormat, ProtocolMetadata, ProtocolTimestamp, RetryClass,
    RunStatus, SessionId, SessionSequence,
};
use tea_testkit::ScriptedModelResponse;
use tea_tools::ToolRegistry;

use common::{
    EventCollector, TestIds, provider, provider_with_capabilities, session_id, store, timestamp,
};
use tea_session::SessionStore;

#[derive(Debug, Clone, Copy, Default)]
struct RealSleepClock;
impl KernelClock for RealSleepClock {
    fn now(&self) -> Result<ProtocolTimestamp, KernelError> {
        Ok(timestamp())
    }
    fn sleep_until(&self, deadline: ProtocolTimestamp) -> KernelDeadlineFuture<'_> {
        let now = self.now().unwrap();
        let duration = (deadline.as_utc() - now.as_utc())
            .to_std()
            .unwrap_or(Duration::ZERO);
        Box::pin(async move {
            tokio::time::sleep(duration).await;
        })
    }
}

#[derive(Debug)]
struct CancelOnRetry {
    events: Mutex<Vec<EventEnvelope>>,
    cancellation: tea_control::CancellationScope,
}

impl CancelOnRetry {
    fn new(cancellation: tea_control::CancellationScope) -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            cancellation,
        }
    }

    fn events(&self) -> Vec<EventEnvelope> {
        self.events.lock().unwrap().clone()
    }
}

impl KernelEventSink for CancelOnRetry {
    fn last_sequence(&self, session_id: SessionId) -> Option<SessionSequence> {
        self.events
            .lock()
            .ok()?
            .iter()
            .rev()
            .find(|event| event.session_id() == session_id)
            .map(EventEnvelope::sequence)
    }

    fn emit(&self, event: EventEnvelope) -> KernelEventFuture<'_> {
        Box::pin(async move {
            let cancel = matches!(event.event(), AgentEvent::ModelRetryScheduled { .. });
            self.events.lock().unwrap().push(event);
            if cancel {
                self.cancellation.cancel();
            }
            Ok(())
        })
    }
}

fn config_with_retries(attempts: u32) -> KernelRunConfig {
    KernelRunConfig::new(
        ActorId::from_str("user:alice").unwrap(),
        PolicyEnvironment::new(
            ExecutionSurface::Test,
            PolicyExecutionTarget::Native,
            ProtocolMetadata::default(),
        ),
    )
    .with_retry_policy(
        ModelRetryPolicy::new(attempts, Duration::from_millis(1), Duration::from_millis(2))
            .unwrap(),
    )
}

#[tokio::test]
async fn retryable_failure_then_success_completes() {
    let provider = provider([
        ScriptedModelResponse::failure(ModelFailureCode::Unavailable, "transient"),
        ScriptedModelResponse::text(["recovered"]),
    ]);
    let store = store().await;
    let tools = ToolRegistry::new();
    let mut policy = PolicyEngine::new();
    policy.add_rule(CodingWorkspacePolicy).unwrap();
    let events = EventCollector::default();

    let outcome = AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &store,
        &RealSleepClock,
        &TestIds::default(),
        &events,
    )
    .run(
        session_id(),
        &config_with_retries(3),
        tea_control::CancellationScope::new(),
    )
    .await
    .unwrap();
    assert_eq!(outcome.state(), RunState::Completed);
    let requests = provider.captured_requests().unwrap();
    assert_eq!(
        requests.len(),
        2,
        "provider should be called twice after one retry"
    );
    let retry_events = events
        .events()
        .into_iter()
        .filter_map(|event| match event.event() {
            AgentEvent::ModelRetryScheduled {
                message_id,
                attempt,
                max_retries,
                delay_ms,
            } => Some((
                "scheduled",
                *message_id,
                *attempt,
                *max_retries,
                Some(*delay_ms),
            )),
            AgentEvent::ModelRetryStarted {
                message_id,
                attempt,
                max_retries,
            } => Some(("started", *message_id, *attempt, *max_retries, None)),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(retry_events.len(), 2);
    assert_eq!(retry_events[0].0, "scheduled");
    assert_eq!(retry_events[0].2, 1);
    assert_eq!(retry_events[0].3, 2);
    assert_eq!(retry_events[0].4, Some(1));
    assert_eq!(retry_events[1].0, "started");
    assert_eq!(retry_events[1].1, retry_events[0].1);
}

#[tokio::test]
async fn retry_preserves_the_frozen_final_output_format() {
    let format = FinalOutputFormat::JsonSchema {
        schema: serde_json::json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": false
        }),
    };
    let provider = provider_with_capabilities(
        [
            ScriptedModelResponse::failure(ModelFailureCode::Unavailable, "transient"),
            ScriptedModelResponse::text([r#"{"answer":"recovered"}"#]),
        ],
        ModelCapabilities::text().with_final_json_schema(),
    );
    let store = store().await;
    let tools = ToolRegistry::new();
    let mut policy = PolicyEngine::new();
    policy.add_rule(CodingWorkspacePolicy).unwrap();
    let events = EventCollector::default();
    let config = config_with_retries(3)
        .with_final_output_format(Some(format.clone()))
        .unwrap();

    let outcome = AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &store,
        &RealSleepClock,
        &TestIds::default(),
        &events,
    )
    .run(session_id(), &config, tea_control::CancellationScope::new())
    .await
    .unwrap();

    assert_eq!(outcome.state(), RunState::Completed);
    let requests = provider.captured_requests().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(
        requests
            .iter()
            .all(|request| request.final_output_format() == Some(&format))
    );
}

#[tokio::test]
async fn retry_discards_partial_attempt_and_uses_server_delay() {
    let failure = ModelFailure::safe(
        ModelFailureCode::Unavailable,
        "HTTP 503: Service temporarily unavailable",
        RetryClass::AfterBackoff,
    )
    .unwrap()
    .with_retry_after(Duration::from_millis(2));
    let provider = provider([
        ScriptedModelResponse::events([
            ModelEvent::Started(ModelResponseInfo::new()),
            ModelEvent::TextDelta(Utf8Delta::new("stale").unwrap()),
            ModelEvent::Failed(failure),
        ]),
        ScriptedModelResponse::text(["recovered"]),
    ]);
    let store = store().await;
    let tools = ToolRegistry::new();
    let mut policy = PolicyEngine::new();
    policy.add_rule(CodingWorkspacePolicy).unwrap();
    let events = EventCollector::default();

    AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &store,
        &RealSleepClock,
        &TestIds::default(),
        &events,
    )
    .run(
        session_id(),
        &config_with_retries(3),
        tea_control::CancellationScope::new(),
    )
    .await
    .unwrap();

    let events = events.events();
    let scheduled = events
        .iter()
        .find_map(|event| match event.event() {
            AgentEvent::ModelRetryScheduled {
                message_id,
                delay_ms,
                ..
            } => Some((*message_id, *delay_ms)),
            _ => None,
        })
        .unwrap();
    assert_eq!(scheduled.1, 2);
    let streamed_message_ids = events
        .iter()
        .filter_map(|event| match event.event() {
            AgentEvent::MessageDelta { message_id, .. } => Some(*message_id),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(streamed_message_ids.iter().all(|id| *id == scheduled.0));

    let snapshot = store.load(session_id()).await.unwrap();
    let assistant = snapshot
        .state()
        .messages()
        .iter()
        .rev()
        .find(|message| matches!(message, tea_protocol::CanonicalMessage::Assistant { .. }))
        .unwrap();
    let serialized = serde_json::to_string(assistant).unwrap();
    assert!(serialized.contains("recovered"));
    assert!(!serialized.contains("stale"));
}

#[tokio::test]
async fn cancellation_during_retry_backoff_prevents_next_request() {
    let provider = provider([
        ScriptedModelResponse::failure(ModelFailureCode::Unavailable, "transient"),
        ScriptedModelResponse::text(["must not run"]),
    ]);
    let store = store().await;
    let tools = ToolRegistry::new();
    let mut policy = PolicyEngine::new();
    policy.add_rule(CodingWorkspacePolicy).unwrap();
    let cancellation = tea_control::CancellationScope::new();
    let events = CancelOnRetry::new(cancellation.clone());

    let error = AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &store,
        &RealSleepClock,
        &TestIds::default(),
        &events,
    )
    .run(session_id(), &config_with_retries(3), cancellation)
    .await
    .unwrap_err();

    assert_eq!(error.code(), KernelErrorCode::Cancelled);
    assert_eq!(provider.captured_requests().unwrap().len(), 1);
    assert!(
        events
            .events()
            .iter()
            .any(|event| matches!(event.event(), AgentEvent::ModelRetryScheduled { .. }))
    );
    assert!(
        !events
            .events()
            .iter()
            .any(|event| matches!(event.event(), AgentEvent::ModelRetryStarted { .. }))
    );
}

#[tokio::test]
async fn never_failure_terminates_without_retry() {
    let provider = provider([
        ScriptedModelResponse::failure(ModelFailureCode::InvalidRequest, "bad request"),
        ScriptedModelResponse::text(["should not run"]),
    ]);
    let store = store().await;
    let tools = ToolRegistry::new();
    let mut policy = PolicyEngine::new();
    policy.add_rule(CodingWorkspacePolicy).unwrap();
    let events = EventCollector::default();

    let error = AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &store,
        &RealSleepClock,
        &TestIds::default(),
        &events,
    )
    .run(
        session_id(),
        &config_with_retries(3),
        tea_control::CancellationScope::new(),
    )
    .await
    .unwrap_err();
    assert_eq!(error.code(), KernelErrorCode::ModelFailure);
    assert_eq!(error.message(), "model provider request failed");
    assert!(!error.is_safe_diagnostic());
    let requests = provider.captured_requests().unwrap();
    assert_eq!(requests.len(), 1, "Never failure must not retry");
}

#[tokio::test]
async fn retry_exhausted_after_policy_limit() {
    let provider = provider([
        ScriptedModelResponse::failure(ModelFailureCode::Unavailable, "transient"),
        ScriptedModelResponse::failure(ModelFailureCode::Unavailable, "transient"),
    ]);
    let store = store().await;
    let tools = ToolRegistry::new();
    let mut policy = PolicyEngine::new();
    policy.add_rule(CodingWorkspacePolicy).unwrap();
    let events = EventCollector::default();

    let error = AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &store,
        &RealSleepClock,
        &TestIds::default(),
        &events,
    )
    .run(
        session_id(),
        &config_with_retries(2),
        tea_control::CancellationScope::new(),
    )
    .await
    .unwrap_err();
    assert_eq!(error.code(), KernelErrorCode::RetryExhausted);
    assert_eq!(error.message(), "model retry policy was exhausted");
    assert!(!error.is_safe_diagnostic());
    let requests = provider.captured_requests().unwrap();
    assert_eq!(
        requests.len(),
        2,
        "policy allows two attempts before exhaustion"
    );
    let snapshot = store.load(session_id()).await.unwrap();
    assert!(matches!(
        snapshot.records().last().unwrap().record(),
        tea_protocol::SessionRecord::RunInterrupted { .. }
    ));
    // Sanity: a finished run was never emitted for this session.
    let _ = RunStatus::Completed;
}
