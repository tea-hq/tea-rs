use crate::common;

use std::str::FromStr;
use std::sync::Arc;

use serde_json::json;
use tea_control::CancellationScope;
use tea_kernel::{
    AgentKernel, KernelErrorCode, KernelEventFuture, KernelEventSink, KernelInputQueue,
    KernelRunConfig,
};
use tea_model::{
    ModelCapabilities, ModelCompletion, ModelEvent, ModelResponseInfo, ModelStreamIndex,
    ProviderToolCallId, ToolCallCompleted, ToolCallStarted, Utf8Delta,
};
use tea_policy::{
    ActorId, CodingWorkspacePolicy, ExecutionSurface, PolicyEngine, PolicyEnvironment,
    PolicyExecutionTarget,
};
use tea_protocol::{
    AgentEvent, AgentEventType, CanonicalMessage, CommandText, ContentBlock, EventEnvelope,
    FinalOutputFormat, MessageId, ProtocolMetadata, RecordEnvelope, RecordId, RunStatus,
    SessionRecord, SessionSequence, StopReason, ToolIdempotency,
};
use tea_session::{AppendTransaction, SessionStore};
use tea_testkit::{FakeReadTool, ScriptedModelResponse};
use tea_tools::{
    ArgumentResourceResolver, ToolConcurrency, ToolEffect, ToolExecutionSemantics, ToolName,
    ToolRegistry, ToolResourceAccess, ToolRetrySafety, ToolSpec, ToolTimeout, ToolVersion,
};

use common::{FixedClock, TestIds, provider_with_capabilities, session_id, store, timestamp};

fn user(id: &str, text: &str) -> CanonicalMessage {
    CanonicalMessage::user(
        MessageId::from_str(id).unwrap(),
        vec![ContentBlock::text(text).unwrap()],
        timestamp(),
    )
    .unwrap()
}

#[test]
fn queue_bounds_preserve_already_accepted_entries() {
    let queue = KernelInputQueue::new(1, 8).unwrap();
    let first = user("0195a0b1-6100-7000-8000-000000000001", "first");
    queue.enqueue_follow_up(first).unwrap();
    assert!(
        queue
            .enqueue_follow_up(user("0195a0b1-6101-7000-8000-000000000001", "second"))
            .is_err()
    );
    assert_eq!(queue.lengths().unwrap(), (1, 0));

    queue
        .enqueue_steering(CommandText::new("steer").unwrap())
        .unwrap();
    assert!(
        queue
            .enqueue_steering(CommandText::new("more").unwrap())
            .is_err()
    );
    assert_eq!(queue.lengths().unwrap(), (1, 1));
}

#[test]
fn queue_rejects_non_user_follow_up_and_invalid_limits() {
    assert!(KernelInputQueue::new(0, 1).is_err());
    let queue = KernelInputQueue::new(2, 32).unwrap();
    let assistant = CanonicalMessage::assistant(
        MessageId::from_str("0195a0b1-6102-7000-8000-000000000001").unwrap(),
        vec![ContentBlock::text("assistant").unwrap()],
        tea_protocol::StopReason::Completed,
        timestamp(),
    )
    .unwrap();
    assert!(queue.enqueue_follow_up(assistant).is_err());
    assert_eq!(queue.lengths().unwrap(), (0, 0));
}

#[test]
fn queue_and_run_configuration_are_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<KernelInputQueue>();
    assert_send_sync::<KernelRunConfig>();
    let _config = KernelRunConfig::new(
        ActorId::from_str("user:alice").unwrap(),
        PolicyEnvironment::new(
            ExecutionSurface::Test,
            PolicyExecutionTarget::Native,
            ProtocolMetadata::default(),
        ),
    );
}

#[derive(Debug)]
struct QueueingSink<'a> {
    queue: &'a KernelInputQueue,
    trigger: AgentEventType,
    events: common::EventCollector,
}

impl<'a> QueueingSink<'a> {
    fn new(queue: &'a KernelInputQueue, trigger: AgentEventType) -> Self {
        Self {
            queue,
            trigger,
            events: common::EventCollector::default(),
        }
    }

    fn events(&self) -> Vec<EventEnvelope> {
        self.events.events()
    }
}

impl KernelEventSink for QueueingSink<'_> {
    fn emit(&self, event: EventEnvelope) -> KernelEventFuture<'_> {
        Box::pin(async move {
            if event.event_type() == self.trigger && self.queue.lengths()? == (0, 0) {
                self.queue
                    .enqueue_steering(CommandText::new("steer next").map_err(|error| {
                        tea_kernel::KernelError::new(
                            tea_kernel::KernelErrorCode::InvalidRequest,
                            error.to_string(),
                        )
                    })?)?;
                self.queue.enqueue_follow_up(user(
                    "0195a0b1-6103-7000-8000-000000000001",
                    "follow next",
                ))?;
            }
            self.events.emit(event).await
        })
    }
}

#[tokio::test]
async fn active_request_is_immutable_and_queue_applies_to_next_turn() {
    let format = FinalOutputFormat::JsonSchema {
        schema: json!({
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
            "additionalProperties": false
        }),
    };
    let index = ModelStreamIndex::new(0).unwrap();
    let call_id = ProviderToolCallId::from_str("queue-read").unwrap();
    let tool_response = ScriptedModelResponse::events([
        ModelEvent::Started(ModelResponseInfo::new()),
        ModelEvent::ToolCallStarted(
            ToolCallStarted::new(index, call_id.clone(), "read_file").unwrap(),
        ),
        ModelEvent::ToolCallCompleted(
            ToolCallCompleted::new(index, call_id, "read_file", json!({"path":"/notes.txt"}))
                .unwrap(),
        ),
        ModelEvent::Completed(ModelCompletion::new(StopReason::ToolUse).unwrap()),
    ]);
    let provider = provider_with_capabilities(
        [
            tool_response,
            ScriptedModelResponse::text([r#"{"status":"queued input observed"}"#]),
        ],
        ModelCapabilities::text()
            .with_tools(false)
            .with_final_json_schema_with_tools(),
    );
    let store = store().await;
    let mut tools = ToolRegistry::new();
    tools
        .register(
            ToolSpec::new(
                ToolName::from_str("read_file").unwrap(),
                ToolVersion::from_str("1.0.0").unwrap(),
                "Read one fake file.",
                json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}),
                json!({"type":"object","properties":{"content":{"type":"string"}},"required":["content"]}),
                [ToolEffect::FsRead],
                ToolExecutionSemantics::new(
                    ToolIdempotency::Idempotent,
                    ToolRetrySafety::Automatic,
                    ToolConcurrency::Serial,
                    ToolTimeout::from_millis(1_000).unwrap(),
                )
                .unwrap(),
            )
            .unwrap(),
            Arc::new(
                ArgumentResourceResolver::new("path", "file", ToolResourceAccess::Read).unwrap(),
            ),
            Arc::new(FakeReadTool::new([(
                "/notes.txt".to_owned(),
                "hello".to_owned(),
            )])),
        )
        .unwrap();
    let mut policy = PolicyEngine::new();
    policy.add_rule(CodingWorkspacePolicy).unwrap();
    let queue = KernelInputQueue::new(4, 1024).unwrap();
    let sink = QueueingSink::new(&queue, AgentEventType::ToolCallRequested);
    let ids = TestIds::default();
    let config = KernelRunConfig::new(
        ActorId::from_str("user:alice").unwrap(),
        PolicyEnvironment::new(
            ExecutionSurface::Test,
            PolicyExecutionTarget::Native,
            ProtocolMetadata::default(),
        ),
    )
    .with_final_output_format(Some(format.clone()))
    .unwrap();
    AgentKernel::new(&provider, &tools, &policy, &store, &FixedClock, &ids, &sink)
        .with_input_queue(&queue)
        .run(session_id(), &config, CancellationScope::new())
        .await
        .unwrap();

    let requests = provider.captured_requests().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(
        requests
            .iter()
            .all(|request| request.final_output_format() == Some(&format))
    );
    assert_eq!(requests[0].messages().len(), 1);
    assert_eq!(requests[1].messages().len(), 5);
    assert_eq!(queue.lengths().unwrap(), (0, 0));
    let snapshot = store.load(session_id()).await.unwrap();
    assert_eq!(snapshot.state().messages().len(), 6);
}

#[tokio::test]
async fn queued_input_provider_preflight_rejection_commits_nothing_and_terminates_once() {
    let provider = common::RejectingPreflightProvider::new(
        [
            ScriptedModelResponse::events([
                ModelEvent::Started(ModelResponseInfo::new()),
                ModelEvent::TextDelta(Utf8Delta::new("first turn").unwrap()),
                ModelEvent::Completed(ModelCompletion::new(StopReason::PauseTurn).unwrap()),
            ]),
            ScriptedModelResponse::text(["must not stream"]),
        ],
        3,
    );
    let store = store().await;
    let queue = KernelInputQueue::new(4, 1024).unwrap();
    let sink = QueueingSink::new(&queue, AgentEventType::MessageDelta);

    let error = AgentKernel::new(
        &provider,
        &ToolRegistry::new(),
        &PolicyEngine::new(),
        &store,
        &FixedClock,
        &TestIds::default(),
        &sink,
    )
    .with_input_queue(&queue)
    .run(
        session_id(),
        &KernelRunConfig::new(
            ActorId::from_str("user:alice").unwrap(),
            PolicyEnvironment::new(
                ExecutionSurface::Test,
                PolicyExecutionTarget::Native,
                ProtocolMetadata::default(),
            ),
        ),
        CancellationScope::new(),
    )
    .await
    .unwrap_err();

    assert_eq!(error.code(), KernelErrorCode::InvalidRequest);
    assert_eq!(provider.validate_calls(), 3);
    assert_eq!(provider.captured_requests().len(), 1);
    assert_eq!(provider.remaining_scripts(), 1);
    assert_eq!(queue.lengths().unwrap(), (1, 1));
    let after = store.load(session_id()).await.unwrap();
    assert_eq!(after.state().messages().len(), 2);
    assert_eq!(after.state().run_recovery().len(), 1);
    let terminal_statuses = sink
        .events()
        .iter()
        .filter_map(|event| match event.event() {
            AgentEvent::RunFinished { status, .. } => Some(*status),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(terminal_statuses, [RunStatus::Interrupted]);
}

#[tokio::test]
async fn oversized_queued_transcript_is_not_persisted_or_acknowledged_and_terminates_once() {
    let provider = provider_with_capabilities([], ModelCapabilities::text());
    let store = store().await;
    let initial = store.load(session_id()).await.unwrap();
    let tail = initial.state().tail_sequence();
    let branch_id = initial.state().active_branch_id();
    let records = (0..(tea_model::MAX_REQUEST_MESSAGES - initial.state().messages().len()))
        .map(|index| {
            RecordEnvelope::new(
                RecordId::from_str(&format!("0197a0b1-{index:04x}-7000-8000-000000000001"))
                    .unwrap(),
                session_id(),
                SessionSequence::new(tail.get() + 1 + index as u64),
                timestamp(),
                None,
                None,
                branch_id,
                ProtocolMetadata::default(),
                SessionRecord::MessageCommitted {
                    message: user(
                        &format!("0197a0b1-{index:04x}-7000-8000-000000000002"),
                        "seed",
                    ),
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    store
        .append(AppendTransaction::new(session_id(), Some(tail), records))
        .await
        .unwrap();
    let queue = KernelInputQueue::new(4, 1024).unwrap();
    queue
        .enqueue_follow_up(user(
            "0197a0b1-ffff-7000-8000-000000000003",
            "must remain queued",
        ))
        .unwrap();
    let events = common::EventCollector::default();
    let tools = ToolRegistry::new();
    let policy = PolicyEngine::new();

    let error = AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &store,
        &FixedClock,
        &TestIds::default(),
        &events,
    )
    .with_input_queue(&queue)
    .run(
        session_id(),
        &KernelRunConfig::new(
            ActorId::from_str("user:alice").unwrap(),
            PolicyEnvironment::new(
                ExecutionSurface::Test,
                PolicyExecutionTarget::Native,
                ProtocolMetadata::default(),
            ),
        ),
        CancellationScope::new(),
    )
    .await
    .unwrap_err();

    assert_eq!(error.code(), KernelErrorCode::InvalidRequest);
    assert_eq!(queue.lengths().unwrap(), (1, 0));
    assert!(provider.captured_requests().unwrap().is_empty());
    let after = store.load(session_id()).await.unwrap();
    assert_eq!(
        after.state().messages().len(),
        tea_model::MAX_REQUEST_MESSAGES
    );
    assert_eq!(after.state().run_recovery().len(), 1);
    assert_eq!(
        events
            .events()
            .iter()
            .filter(|event| event.event_type() == AgentEventType::RunFinished)
            .count(),
        1
    );
}
