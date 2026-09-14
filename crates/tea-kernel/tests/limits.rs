use crate::common;

use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use serde_json::json;
use tea_control::CancellationScope;
use tea_kernel::{
    AgentKernel, KernelClock, KernelDeadlineFuture, KernelError, KernelErrorCode,
    KernelEventFuture, KernelEventSink, KernelRunConfig, RunLimits,
};
use tea_model::{
    ModelCompletion, ModelEvent, ModelResponseInfo, ModelStreamIndex, ProviderToolCallId,
    ToolCallCompleted, ToolCallStarted,
};
use tea_policy::{
    ActorId, CodingWorkspacePolicy, ExecutionSurface, PolicyEngine, PolicyEnvironment,
    PolicyExecutionTarget,
};
use tea_protocol::{
    AgentEvent, EventEnvelope, ProtocolMetadata, ProtocolTimestamp, SessionRecord, StopReason,
    ToolIdempotency,
};
use tea_session::SessionStore;
use tea_testkit::{FakeProcessScript, FakeProcessTool, ScriptedModelResponse};
use tea_tools::{
    StaticResourceResolver, ToolConcurrency, ToolEffect, ToolExecutionSemantics, ToolName,
    ToolRegistry, ToolRetrySafety, ToolSpec, ToolTimeout, ToolVersion,
};

use common::{EventCollector, FixedClock, TestIds, provider, session_id, store, timestamp};

fn config(limits: RunLimits) -> KernelRunConfig {
    KernelRunConfig::new(
        ActorId::from_str("user:alice").unwrap(),
        PolicyEnvironment::new(
            ExecutionSurface::Test,
            PolicyExecutionTarget::Native,
            ProtocolMetadata::default(),
        ),
    )
    .with_limits(limits)
}

#[derive(Debug)]
struct ImmediateDeadlineClock;
impl KernelClock for ImmediateDeadlineClock {
    fn now(&self) -> Result<ProtocolTimestamp, KernelError> {
        Ok(timestamp())
    }
    fn sleep_until(&self, _deadline: ProtocolTimestamp) -> KernelDeadlineFuture<'_> {
        Box::pin(async {})
    }
}

#[derive(Debug, Default)]
struct FailSecondNowClock(AtomicUsize);
impl KernelClock for FailSecondNowClock {
    fn now(&self) -> Result<ProtocolTimestamp, KernelError> {
        if self.0.fetch_add(1, Ordering::SeqCst) == 0 {
            Ok(timestamp())
        } else {
            Err(KernelError::new(
                KernelErrorCode::ClockFailure,
                "clock failed",
            ))
        }
    }

    fn sleep_until(&self, _deadline: ProtocolTimestamp) -> KernelDeadlineFuture<'_> {
        Box::pin(std::future::pending())
    }
}

#[derive(Debug)]
struct CancellingSink<'a> {
    cancellation: &'a CancellationScope,
}
impl KernelEventSink for CancellingSink<'_> {
    fn emit(&self, event: EventEnvelope) -> KernelEventFuture<'_> {
        Box::pin(async move {
            if matches!(event.event(), AgentEvent::ToolCallRequested { .. }) {
                self.cancellation.cancel();
            }
            Ok(())
        })
    }
}

fn tool_registry(script: FakeProcessScript) -> ToolRegistry {
    let mut tools = ToolRegistry::new();
    tools
        .register(
            ToolSpec::new(
                ToolName::from_str("read_pending").unwrap(),
                ToolVersion::from_str("1.0.0").unwrap(),
                "Run one deterministic fake read.",
                json!({"type":"object"}),
                json!({"type":"object","properties":{"stdout":{"type":"string"},"exitCode":{"type":"integer"}},"required":["stdout","exitCode"]}),
                [ToolEffect::FsRead],
                ToolExecutionSemantics::new(
                    ToolIdempotency::NonIdempotent,
                    ToolRetrySafety::Never,
                    ToolConcurrency::Serial,
                    ToolTimeout::from_millis(1_000).unwrap(),
                )
                .unwrap(),
            )
            .unwrap(),
            Arc::new(StaticResourceResolver::new([]).unwrap()),
            Arc::new(FakeProcessTool::new(script)),
        )
        .unwrap();
    tools
}

fn pending_tool_registry() -> ToolRegistry {
    tool_registry(FakeProcessScript::AwaitCancellation)
}

fn tool_script() -> ScriptedModelResponse {
    let index = ModelStreamIndex::new(0).unwrap();
    let provider_id = ProviderToolCallId::from_str("pending-call").unwrap();
    ScriptedModelResponse::events([
        ModelEvent::Started(ModelResponseInfo::new()),
        ModelEvent::ToolCallStarted(
            ToolCallStarted::new(index, provider_id.clone(), "read_pending").unwrap(),
        ),
        ModelEvent::ToolCallCompleted(
            ToolCallCompleted::new(index, provider_id, "read_pending", json!({})).unwrap(),
        ),
        ModelEvent::Completed(ModelCompletion::new(StopReason::ToolUse).unwrap()),
    ])
}

#[tokio::test]
async fn assistant_output_limit_interrupts_without_committing_message() {
    let provider = provider([ScriptedModelResponse::text(["too much output"])]);
    let store = store().await;
    let tools = ToolRegistry::new();
    let policy = PolicyEngine::new();
    let ids = TestIds::default();
    let events = EventCollector::default();
    let kernel = AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &store,
        &FixedClock,
        &ids,
        &events,
    );
    let limits = RunLimits::new(2, Duration::from_secs(30), 4, 100, 4).unwrap();
    let error = kernel
        .run(session_id(), &config(limits), CancellationScope::new())
        .await
        .unwrap_err();
    assert_eq!(error.code(), KernelErrorCode::LimitExceeded);
    let snapshot = store.load(session_id()).await.unwrap();
    assert_eq!(snapshot.state().messages().len(), 1);
    assert!(matches!(
        snapshot.records().last().unwrap().record(),
        SessionRecord::RunInterrupted { .. }
    ));
}

#[tokio::test]
async fn deadline_is_constructed_before_run_started_is_emitted() {
    let provider = provider([ScriptedModelResponse::text(["unused"])]);
    let store = store().await;
    let tools = ToolRegistry::new();
    let policy = PolicyEngine::new();
    let events = EventCollector::default();

    let error = AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &store,
        &FailSecondNowClock::default(),
        &TestIds::default(),
        &events,
    )
    .run(
        session_id(),
        &config(RunLimits::default()),
        CancellationScope::new(),
    )
    .await
    .unwrap_err();

    assert_eq!(error.code(), KernelErrorCode::ClockFailure);
    assert!(events.events().is_empty());
    assert!(
        store
            .load(session_id())
            .await
            .unwrap()
            .state()
            .run_recovery()
            .is_empty()
    );
}

#[tokio::test]
async fn tool_iteration_limit_stops_before_second_declaration() {
    let provider = provider([tool_script(), tool_script()]);
    let store = store().await;
    let tools = tool_registry(FakeProcessScript::Complete {
        stdout: "ok".to_owned(),
    });
    let mut policy = PolicyEngine::new();
    policy.add_rule(CodingWorkspacePolicy).unwrap();
    let ids = TestIds::default();
    let events = EventCollector::default();
    let kernel = AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &store,
        &FixedClock,
        &ids,
        &events,
    );
    let limits = RunLimits::new(1, Duration::from_secs(30), 4096, 100, 4).unwrap();
    let error = kernel
        .run(session_id(), &config(limits), CancellationScope::new())
        .await
        .unwrap_err();
    assert_eq!(error.code(), KernelErrorCode::LimitExceeded);
    let snapshot = store.load(session_id()).await.unwrap();
    assert_eq!(
        snapshot
            .records()
            .iter()
            .filter(|record| matches!(record.record(), SessionRecord::ToolCallRequested { .. }))
            .count(),
        1
    );
}

#[tokio::test]
async fn pre_cancel_is_durable_and_never_polls_model() {
    let provider = provider([ScriptedModelResponse::text(["unused"])]);
    let store = store().await;
    let tools = ToolRegistry::new();
    let policy = PolicyEngine::new();
    let ids = TestIds::default();
    let events = EventCollector::default();
    let cancellation = CancellationScope::new();
    cancellation.cancel();
    let kernel = AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &store,
        &FixedClock,
        &ids,
        &events,
    );
    let error = kernel
        .run(session_id(), &config(RunLimits::default()), cancellation)
        .await
        .unwrap_err();
    assert_eq!(error.code(), KernelErrorCode::Cancelled);
    assert!(provider.captured_requests().unwrap().is_empty());
    let snapshot = store.load(session_id()).await.unwrap();
    assert!(matches!(
        snapshot.records().last().unwrap().record(),
        SessionRecord::RunCancelled { .. }
    ));
}

#[tokio::test]
async fn deadline_interrupts_model_without_partial_message() {
    let provider = provider([ScriptedModelResponse::await_cancellation()]);
    let store = store().await;
    let tools = ToolRegistry::new();
    let policy = PolicyEngine::new();
    let ids = TestIds::default();
    let events = EventCollector::default();
    let kernel = AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &store,
        &ImmediateDeadlineClock,
        &ids,
        &events,
    );
    let error = kernel
        .run(
            session_id(),
            &config(RunLimits::default()),
            CancellationScope::new(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.code(), KernelErrorCode::LimitExceeded);
    let snapshot = store.load(session_id()).await.unwrap();
    assert_eq!(snapshot.state().messages().len(), 1);
    assert!(matches!(
        snapshot.records().last().unwrap().record(),
        SessionRecord::RunInterrupted { .. }
    ));
}

#[tokio::test]
async fn cancel_after_tool_start_records_uncertain_outcome_without_result() {
    let provider = provider([tool_script()]);
    let store = store().await;
    let tools = pending_tool_registry();
    let mut policy = PolicyEngine::new();
    policy.add_rule(CodingWorkspacePolicy).unwrap();
    let ids = TestIds::default();
    let cancellation = CancellationScope::new();
    let sink = CancellingSink {
        cancellation: &cancellation,
    };
    let kernel = AgentKernel::new(&provider, &tools, &policy, &store, &FixedClock, &ids, &sink);
    let error = kernel
        .run(
            session_id(),
            &config(RunLimits::new(2, Duration::from_secs(30), 1024, 100, 4).unwrap()),
            cancellation.clone(),
        )
        .await
        .unwrap_err();
    assert_eq!(error.code(), KernelErrorCode::Cancelled);
    let snapshot = store.load(session_id()).await.unwrap();
    assert!(snapshot.records().iter().any(|record| matches!(
        record.record(),
        SessionRecord::ToolExecutionInterrupted { .. }
    )));
    assert!(
        !snapshot
            .records()
            .iter()
            .any(|record| matches!(record.record(), SessionRecord::ToolExecutionFinished { .. }))
    );

    let fresh_ids = TestIds::with_start(500);
    let fresh_events = EventCollector::default();
    let fresh = AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &store,
        &FixedClock,
        &fresh_ids,
        &fresh_events,
    );
    let retry_error = fresh
        .run(
            session_id(),
            &config(RunLimits::default()),
            CancellationScope::new(),
        )
        .await
        .unwrap_err();
    assert_eq!(retry_error.code(), KernelErrorCode::InvalidState);
}
