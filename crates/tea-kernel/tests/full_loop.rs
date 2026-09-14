use crate::common;

use std::str::FromStr;
use std::sync::Arc;

use serde_json::{Value, json};
use tea_control::CancellationScope;
use tea_kernel::{AgentKernel, KernelErrorCode, KernelRunConfig, RunState};
use tea_model::{
    ModelCapabilities, ModelCompletion, ModelEvent, ModelResponseInfo, ModelStreamIndex,
    ProviderToolCallId, ToolCallCompleted, ToolCallStarted,
};
use tea_policy::{
    ActorId, ApprovalResolution, CodingWorkspacePolicy, ExecutionSurface, PolicyEngine,
    PolicyEnvironment, PolicyExecutionTarget,
};
use tea_protocol::{
    AgentEventType, ApprovalDecision, FinalOutputFormat, ProtocolMetadata, SessionRecord,
    StopReason, ToolIdempotency,
};
use tea_session::{ApprovalArtifactEntry, SessionStore};
use tea_testkit::{FakeReadTool, FakeWriteTool, ScriptedModelResponse};
use tea_tools::{
    ArgumentResourceResolver, ToolConcurrency, ToolEffect, ToolExecutionSemantics, ToolName,
    ToolRegistry, ToolResourceAccess, ToolRetrySafety, ToolSpec, ToolTimeout, ToolVersion,
};

use common::{
    EventCollector, FixedClock, TestIds, provider_with_capabilities, session_id, store, timestamp,
};

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

fn tools(read: FakeReadTool, write: FakeWriteTool) -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    registry
        .register(
            spec(
                "read_file",
                ToolEffect::FsRead,
                ToolIdempotency::Idempotent,
                json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}),
                json!({"type":"object","properties":{"content":{"type":"string"}},"required":["content"]}),
            ),
            Arc::new(
                ArgumentResourceResolver::new("path", "file", ToolResourceAccess::Read).unwrap(),
            ),
            Arc::new(read),
        )
        .unwrap();
    registry
        .register(
            spec(
                "write_file",
                ToolEffect::FsWrite,
                ToolIdempotency::NonIdempotent,
                json!({"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}),
                json!({"type":"object","properties":{"path":{"type":"string"},"writtenBytes":{"type":"integer"}},"required":["path","writtenBytes"]}),
            ),
            Arc::new(
                ArgumentResourceResolver::new("path", "file", ToolResourceAccess::Write).unwrap(),
            ),
            Arc::new(write),
        )
        .unwrap();
    registry
}

fn spec(
    name: &str,
    effect: ToolEffect,
    idempotency: ToolIdempotency,
    input: Value,
    output: Value,
) -> ToolSpec {
    ToolSpec::new(
        ToolName::from_str(name).unwrap(),
        ToolVersion::from_str("1.0.0").unwrap(),
        format!("Deterministic {name}."),
        input,
        output,
        [effect],
        ToolExecutionSemantics::new(
            idempotency,
            if idempotency == ToolIdempotency::Idempotent {
                ToolRetrySafety::Automatic
            } else {
                ToolRetrySafety::ExplicitOnly
            },
            ToolConcurrency::Serial,
            ToolTimeout::from_millis(1_000).unwrap(),
        )
        .unwrap(),
    )
    .unwrap()
}

fn tool_script(name: &str, arguments: Value, opaque_id: &str) -> ScriptedModelResponse {
    let index = ModelStreamIndex::new(0).unwrap();
    let provider_id = ProviderToolCallId::from_str(opaque_id).unwrap();
    ScriptedModelResponse::events([
        ModelEvent::Started(ModelResponseInfo::new()),
        ModelEvent::ToolCallStarted(
            ToolCallStarted::new(index, provider_id.clone(), name).unwrap(),
        ),
        ModelEvent::ToolCallCompleted(
            ToolCallCompleted::new(index, provider_id, name, arguments).unwrap(),
        ),
        ModelEvent::Completed(ModelCompletion::new(StopReason::ToolUse).unwrap()),
    ])
}

#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn final_output_format_survives_tools_and_approval_resume() {
    let format = FinalOutputFormat::JsonSchema {
        schema: json!({
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": false
        }),
    };
    let provider = provider_with_capabilities(
        [
            tool_script("read_file", json!({"path":"/notes.txt"}), "read-1"),
            tool_script(
                "write_file",
                json!({"path":"/summary.txt","content":"hello"}),
                "write-1",
            ),
            ScriptedModelResponse::text([r#"{"summary":"written"}"#]),
        ],
        ModelCapabilities::text()
            .with_tools(false)
            .with_final_json_schema_with_tools(),
    );
    let store = store().await;
    let write = FakeWriteTool::new();
    let tools = tools(
        FakeReadTool::new([("/notes.txt".to_owned(), "hello".to_owned())]),
        write.clone(),
    );
    let mut policy = PolicyEngine::new();
    policy.add_rule(CodingWorkspacePolicy).unwrap();
    let events = EventCollector::default();
    let first_ids = TestIds::default();
    let run_config = config()
        .with_final_output_format(Some(format.clone()))
        .unwrap();

    let waiting = AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &store,
        &FixedClock,
        &first_ids,
        &events,
    )
    .run(session_id(), &run_config, CancellationScope::new())
    .await
    .unwrap();
    assert_eq!(waiting.state(), RunState::WaitingApproval);
    assert!(write.writes().unwrap().is_empty());

    let paused = store.load(session_id()).await.unwrap();
    let (request, persisted_format) = paused
        .approval_artifacts()
        .iter()
        .find_map(|entry| match entry {
            ApprovalArtifactEntry::Requested {
                request,
                final_output_format,
                ..
            } => Some((request.clone(), final_output_format.clone())),
            ApprovalArtifactEntry::Resolved { .. } => None,
        })
        .unwrap();
    assert_eq!(persisted_format, Some(format.clone()));
    let resolution =
        ApprovalResolution::new(&request, ApprovalDecision::AllowOnce, timestamp(), None).unwrap();

    let resume_ids = TestIds::with_start(200);
    let completed = AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &store,
        &FixedClock,
        &resume_ids,
        &events,
    )
    .resume_approval(
        session_id(),
        &resolution,
        &config(),
        CancellationScope::new(),
    )
    .await
    .unwrap();

    assert_eq!(completed.state(), RunState::Completed);
    assert_eq!(
        write.writes().unwrap(),
        [("/summary.txt".to_owned(), "hello".to_owned())]
    );
    let requests = provider.captured_requests().unwrap();
    assert_eq!(requests.len(), 3);
    assert!(
        requests
            .iter()
            .all(|request| request.final_output_format() == Some(&format))
    );
    assert_eq!(
        requests
            .iter()
            .map(|request| request.messages().len())
            .collect::<Vec<_>>(),
        [1, 3, 5]
    );

    let snapshot = store.load(session_id()).await.unwrap();
    assert_eq!(snapshot.approval_artifacts().len(), 2);
    assert!(snapshot.state().pending_approvals().is_empty());
    assert!(matches!(
        snapshot.records().last().unwrap().record(),
        SessionRecord::TurnCheckpointed {
            next_action: tea_protocol::NextTurnAction::FinishRun,
            ..
        }
    ));
    let emitted = events.events();
    assert!(
        emitted
            .windows(2)
            .all(|events| events[0].sequence() < events[1].sequence())
    );
    let event_types = emitted
        .iter()
        .map(tea_protocol::EventEnvelope::event_type)
        .collect::<Vec<_>>();
    assert_eq!(event_types.first(), Some(&AgentEventType::RunStarted));
    assert_eq!(event_types.last(), Some(&AgentEventType::RunFinished));
    assert!(event_types.contains(&AgentEventType::ApprovalRequested));
    assert_eq!(
        event_types
            .iter()
            .filter(|event| **event == AgentEventType::RunStarted)
            .count(),
        1
    );
}

#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn approval_capability_drift_blocks_allow_but_deny_clears_pending_without_side_effects() {
    let format = FinalOutputFormat::JsonSchema {
        schema: json!({
            "type": "object",
            "properties": {"summary": {"type": "string"}},
            "required": ["summary"],
            "additionalProperties": false
        }),
    };
    let initial_provider = provider_with_capabilities(
        [tool_script(
            "write_file",
            json!({"path":"/summary.txt","content":"must not be written"}),
            "write-drift",
        )],
        ModelCapabilities::text()
            .with_tools(false)
            .with_final_json_schema_with_tools(),
    );
    let store = store().await;
    let write = FakeWriteTool::new();
    let tools = tools(FakeReadTool::new([]), write.clone());
    let mut policy = PolicyEngine::new();
    policy.add_rule(CodingWorkspacePolicy).unwrap();
    let events = EventCollector::default();
    let run_config = config().with_final_output_format(Some(format)).unwrap();

    let waiting = AgentKernel::new(
        &initial_provider,
        &tools,
        &policy,
        &store,
        &FixedClock,
        &TestIds::default(),
        &events,
    )
    .run(session_id(), &run_config, CancellationScope::new())
    .await
    .unwrap();
    assert_eq!(waiting.state(), RunState::WaitingApproval);

    let paused = store.load(session_id()).await.unwrap();
    let request = paused
        .approval_artifacts()
        .iter()
        .find_map(|entry| match entry {
            ApprovalArtifactEntry::Requested { request, .. } => Some(request.clone()),
            ApprovalArtifactEntry::Resolved { .. } => None,
        })
        .unwrap();
    let drifted_provider =
        provider_with_capabilities([], ModelCapabilities::text().with_tools(false));
    let resume_ids = TestIds::with_start(200);
    let resumed = AgentKernel::new(
        &drifted_provider,
        &tools,
        &policy,
        &store,
        &FixedClock,
        &resume_ids,
        &events,
    );

    let allow =
        ApprovalResolution::new(&request, ApprovalDecision::AllowOnce, timestamp(), None).unwrap();
    let error = resumed
        .resume_approval(session_id(), &allow, &config(), CancellationScope::new())
        .await
        .unwrap_err();
    assert_eq!(error.code(), KernelErrorCode::InvalidRequest);
    let after_allow = store.load(session_id()).await.unwrap();
    assert_eq!(after_allow.state().pending_approvals().len(), 1);
    assert!(write.writes().unwrap().is_empty());

    let deny =
        ApprovalResolution::new(&request, ApprovalDecision::Deny, timestamp(), None).unwrap();
    let error = resumed
        .resume_approval(session_id(), &deny, &config(), CancellationScope::new())
        .await
        .unwrap_err();
    assert_eq!(error.code(), KernelErrorCode::InvalidRequest);

    let after_deny = store.load(session_id()).await.unwrap();
    assert!(after_deny.state().pending_approvals().is_empty());
    assert!(write.writes().unwrap().is_empty());
    assert!(drifted_provider.captured_requests().unwrap().is_empty());
    assert!(matches!(
        after_deny.records().last().unwrap().record(),
        SessionRecord::RunInterrupted { .. }
    ));
    assert_eq!(
        after_deny
            .records()
            .iter()
            .filter(|record| matches!(
                record.record(),
                SessionRecord::ApprovalResolved { approval_id, .. }
                    if approval_id == request.approval_id()
            ))
            .count(),
        1
    );
    assert!(
        !after_deny
            .records()
            .iter()
            .any(|record| matches!(record.record(), SessionRecord::ToolExecutionStarted { .. }))
    );
}
