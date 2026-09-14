use crate::common;

use std::str::FromStr;
use std::sync::Arc;

use serde_json::json;
use tea_control::CancellationScope;
use tea_kernel::{AgentKernel, KernelRunConfig, RunState};
use tea_model::{
    ModelCapabilities, ModelCompletion, ModelEvent, ModelResponseInfo, ModelStreamIndex,
    ProviderToolCallId, ToolCallCompleted, ToolCallStarted,
};
use tea_policy::{
    ActorId, ApprovalResolution, CodingWorkspacePolicy, ExecutionSurface, PolicyEngine,
    PolicyEnvironment, PolicyExecutionTarget,
};
use tea_protocol::{
    AgentEventType, CanonicalMessage, CodeChange, CodeChangeKind, ContentBlock, FinalOutputFormat,
    MessageId, NextTurnAction, ProtocolMetadata, SessionRecord, StopReason, ToolIdempotency,
    ToolPresentation,
};
use tea_session::{AppendTransaction, ApprovalArtifactEntry, InMemorySessionStore, SessionStore};
use tea_testkit::{FakeWriteTool, ScriptedModelResponse};
use tea_tools::{
    ArgumentResourceResolver, BoxToolExecutionStream, ToolConcurrency, ToolEffect,
    ToolExecutionSemantics, ToolExecutor, ToolName, ToolRegistry, ToolResourceAccess,
    ToolRetrySafety, ToolSource, ToolSourceKind, ToolSpec, ToolTimeout, ToolTrust, ToolVersion,
};

use common::{
    EventCollector, FixedClock, RejectingPreflightProvider, TestIds, provider, session_id, store,
    timestamp,
};

fn write_registry(fake: FakeWriteTool) -> ToolRegistry {
    write_registry_with_source(fake, ToolSource::native_product())
}

fn write_registry_with_source(fake: FakeWriteTool, source: ToolSource) -> ToolRegistry {
    let mut tools = ToolRegistry::new();
    tools
        .register(
            ToolSpec::new(
                ToolName::from_str("write_file").unwrap(),
                ToolVersion::from_str("1.0.0").unwrap(),
                "Write one workspace file.",
                json!({"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}),
                json!({"type":"object","properties":{"path":{"type":"string"},"writtenBytes":{"type":"integer"}},"required":["path","writtenBytes"]}),
                [ToolEffect::FsWrite],
                ToolExecutionSemantics::new(
                    ToolIdempotency::NonIdempotent,
                    ToolRetrySafety::ExplicitOnly,
                    ToolConcurrency::Serial,
                    ToolTimeout::from_millis(1_000).unwrap(),
                )
                .unwrap(),
            )
            .unwrap()
            .with_source(source),
            Arc::new(
                ArgumentResourceResolver::new("path", "file", ToolResourceAccess::Write).unwrap(),
            ),
            Arc::new(fake),
        )
        .unwrap();
    tools
}

#[derive(Debug)]
struct PreviewWriteTool;

impl ToolExecutor for PreviewWriteTool {
    fn preview(
        &self,
        _invocation: &tea_tools::ValidatedToolInvocation,
    ) -> Option<ToolPresentation> {
        Some(ToolPresentation::CodeChange(
            CodeChange::new(
                "notes.txt",
                CodeChangeKind::Update,
                Vec::new(),
                false,
                None,
                None,
                None,
            )
            .unwrap(),
        ))
    }

    fn execute(
        &self,
        _invocation: tea_tools::ValidatedToolInvocation,
        _cancellation: CancellationScope,
    ) -> BoxToolExecutionStream {
        panic!("an approval preview must not execute the tool")
    }
}

fn preview_registry() -> ToolRegistry {
    let mut tools = ToolRegistry::new();
    tools
        .register(
            ToolSpec::new(
                ToolName::from_str("write_file").unwrap(),
                ToolVersion::from_str("1.0.0").unwrap(),
                "Write one workspace file.",
                json!({"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}),
                json!({"type":"object","properties":{"path":{"type":"string"},"writtenBytes":{"type":"integer"}},"required":["path","writtenBytes"]}),
                [ToolEffect::FsWrite],
                ToolExecutionSemantics::new(
                    ToolIdempotency::NonIdempotent,
                    ToolRetrySafety::ExplicitOnly,
                    ToolConcurrency::Serial,
                    ToolTimeout::from_millis(1_000).unwrap(),
                )
                .unwrap(),
            )
            .unwrap(),
            Arc::new(
                ArgumentResourceResolver::new("path", "file", ToolResourceAccess::Write).unwrap(),
            ),
            Arc::new(PreviewWriteTool),
        )
        .unwrap();
    tools
}

fn mcp_source(digest: &str) -> ToolSource {
    mcp_source_for("workspace.files", digest)
}

fn mcp_source_for(source_id: &str, digest: &str) -> ToolSource {
    ToolSource::new(ToolSourceKind::Mcp, source_id, ToolTrust::Workspace, digest).unwrap()
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

fn write_script() -> ScriptedModelResponse {
    let index = ModelStreamIndex::new(0).unwrap();
    let provider_call_id = ProviderToolCallId::from_str("provider-write").unwrap();
    ScriptedModelResponse::events([
        ModelEvent::Started(ModelResponseInfo::new()),
        ModelEvent::ToolCallStarted(
            ToolCallStarted::new(index, provider_call_id.clone(), "write_file").unwrap(),
        ),
        ModelEvent::ToolCallCompleted(
            ToolCallCompleted::new(
                index,
                provider_call_id,
                "write_file",
                json!({"path":"/notes.txt","content":"hello"}),
            )
            .unwrap(),
        ),
        ModelEvent::Completed(ModelCompletion::new(StopReason::ToolUse).unwrap()),
    ])
}

#[tokio::test]
async fn denial_resolution_and_failure_result_commit_atomically() {
    let provider = provider([
        write_script(),
        ScriptedModelResponse::text(["denial handled"]),
    ]);
    let store = store().await;
    let fake = FakeWriteTool::new();
    let tools = write_registry(fake.clone());
    let mut policy = PolicyEngine::new();
    policy.add_rule(CodingWorkspacePolicy).unwrap();
    let events = EventCollector::default();
    let first_ids = TestIds::default();
    AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &store,
        &FixedClock,
        &first_ids,
        &events,
    )
    .run(session_id(), &config(), CancellationScope::new())
    .await
    .unwrap();
    let snapshot = store.load(session_id()).await.unwrap();
    let request = match &snapshot.approval_artifacts()[0] {
        tea_session::ApprovalArtifactEntry::Requested { request, .. } => request.clone(),
        tea_session::ApprovalArtifactEntry::Resolved { .. } => panic!("expected request"),
    };
    let resolution = ApprovalResolution::new(
        &request,
        tea_protocol::ApprovalDecision::Deny,
        timestamp(),
        None,
    )
    .unwrap();
    let resume_ids = TestIds::with_start(300);
    let outcome = AgentKernel::new(
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
    assert_eq!(outcome.state(), RunState::Completed);
    assert!(fake.writes().unwrap().is_empty());
    let snapshot = store.load(session_id()).await.unwrap();
    let resolution_index = snapshot
        .records()
        .iter()
        .position(|record| matches!(record.record(), SessionRecord::ApprovalResolved { .. }))
        .unwrap();
    assert!(matches!(
        snapshot.records()[resolution_index + 1].record(),
        SessionRecord::ToolExecutionFinished { is_error: true, .. }
    ));
    assert!(matches!(
        snapshot.records()[resolution_index + 2].record(),
        SessionRecord::MessageCommitted { .. }
    ));
}

#[tokio::test]
async fn denial_clears_pending_without_a_current_tool_registry() {
    let provider = provider([
        write_script(),
        ScriptedModelResponse::text(["denial handled"]),
    ]);
    let store = store().await;
    let fake = FakeWriteTool::new();
    let original_tools = write_registry(fake.clone());
    let mut policy = PolicyEngine::new();
    policy.add_rule(CodingWorkspacePolicy).unwrap();
    let events = EventCollector::default();
    AgentKernel::new(
        &provider,
        &original_tools,
        &policy,
        &store,
        &FixedClock,
        &TestIds::default(),
        &events,
    )
    .run(session_id(), &config(), CancellationScope::new())
    .await
    .unwrap();
    let paused = store.load(session_id()).await.unwrap();
    let request = paused
        .approval_artifacts()
        .iter()
        .find_map(|entry| match entry {
            ApprovalArtifactEntry::Requested { request, .. } => Some(request.clone()),
            ApprovalArtifactEntry::Resolved { .. } => None,
        })
        .unwrap();
    let resolution = ApprovalResolution::new(
        &request,
        tea_protocol::ApprovalDecision::Deny,
        timestamp(),
        None,
    )
    .unwrap();
    let empty_tools = ToolRegistry::new();

    let outcome = AgentKernel::new(
        &provider,
        &empty_tools,
        &policy,
        &store,
        &FixedClock,
        &TestIds::with_start(300),
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

    assert_eq!(outcome.state(), RunState::Completed);
    assert!(fake.writes().unwrap().is_empty());
    assert!(
        store
            .load(session_id())
            .await
            .unwrap()
            .state()
            .pending_approvals()
            .is_empty()
    );
}

#[tokio::test]
async fn denial_ignores_current_tool_and_policy_context_drift() {
    let provider = provider([
        write_script(),
        ScriptedModelResponse::text(["denial handled"]),
    ]);
    let store = store().await;
    let fake = FakeWriteTool::new();
    let original_tools = write_registry_with_source(
        fake.clone(),
        mcp_source("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
    );
    let mut policy = PolicyEngine::new();
    policy.add_rule(CodingWorkspacePolicy).unwrap();
    let events = EventCollector::default();
    AgentKernel::new(
        &provider,
        &original_tools,
        &policy,
        &store,
        &FixedClock,
        &TestIds::default(),
        &events,
    )
    .run(session_id(), &config(), CancellationScope::new())
    .await
    .unwrap();
    let paused = store.load(session_id()).await.unwrap();
    let request = paused
        .approval_artifacts()
        .iter()
        .find_map(|entry| match entry {
            ApprovalArtifactEntry::Requested { request, .. } => Some(request.clone()),
            ApprovalArtifactEntry::Resolved { .. } => None,
        })
        .unwrap();
    let resolution = ApprovalResolution::new(
        &request,
        tea_protocol::ApprovalDecision::Deny,
        timestamp(),
        None,
    )
    .unwrap();
    let drifted_tools = write_registry_with_source(
        fake.clone(),
        mcp_source("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
    );
    let drifted_config = KernelRunConfig::new(
        ActorId::from_str("user:bob").unwrap(),
        PolicyEnvironment::new(
            ExecutionSurface::Test,
            PolicyExecutionTarget::Remote,
            ProtocolMetadata::default(),
        ),
    );

    let outcome = AgentKernel::new(
        &provider,
        &drifted_tools,
        &policy,
        &store,
        &FixedClock,
        &TestIds::with_start(300),
        &events,
    )
    .resume_approval(
        session_id(),
        &resolution,
        &drifted_config,
        CancellationScope::new(),
    )
    .await
    .unwrap();

    assert_eq!(outcome.state(), RunState::Completed);
    assert!(fake.writes().unwrap().is_empty());
    assert!(
        store
            .load(session_id())
            .await
            .unwrap()
            .state()
            .pending_approvals()
            .is_empty()
    );
}

#[tokio::test]
async fn denial_clears_pending_before_rejecting_a_persisted_invalid_schema() {
    let provider = common::provider_with_capabilities(
        [write_script()],
        ModelCapabilities::text()
            .with_tools(false)
            .with_final_json_schema_with_tools(),
    );
    let original_store = store().await;
    let fake = FakeWriteTool::new();
    let tools = write_registry(fake.clone());
    let mut policy = PolicyEngine::new();
    policy.add_rule(CodingWorkspacePolicy).unwrap();
    let events = EventCollector::default();
    let valid_config = config()
        .with_final_output_format(Some(FinalOutputFormat::JsonSchema {
            schema: json!({"type":"object"}),
        }))
        .unwrap();
    AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &original_store,
        &FixedClock,
        &TestIds::default(),
        &events,
    )
    .run(session_id(), &valid_config, CancellationScope::new())
    .await
    .unwrap();
    let paused = original_store.load(session_id()).await.unwrap();
    let (record_id, request) = paused
        .approval_artifacts()
        .iter()
        .find_map(|entry| match entry {
            ApprovalArtifactEntry::Requested {
                record_id, request, ..
            } => Some((*record_id, request.clone())),
            ApprovalArtifactEntry::Resolved { .. } => None,
        })
        .unwrap();
    let corrupted_store = InMemorySessionStore::new();
    corrupted_store
        .append(
            AppendTransaction::new(session_id(), None, paused.records().to_vec())
                .with_expected_journal_revision(0)
                .with_approval_artifacts([ApprovalArtifactEntry::Requested {
                    record_id,
                    request: request.clone(),
                    final_output_format: Some(FinalOutputFormat::JsonSchema {
                        schema: json!({"type": 7}),
                    }),
                }]),
        )
        .await
        .unwrap();
    let resolution = ApprovalResolution::new(
        &request,
        tea_protocol::ApprovalDecision::Deny,
        timestamp(),
        None,
    )
    .unwrap();

    let error = AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &corrupted_store,
        &FixedClock,
        &TestIds::with_start(300),
        &events,
    )
    .resume_approval(
        session_id(),
        &resolution,
        &config(),
        CancellationScope::new(),
    )
    .await
    .unwrap_err();

    assert_eq!(error.code(), tea_kernel::KernelErrorCode::InvalidRequest);
    let after = corrupted_store.load(session_id()).await.unwrap();
    assert!(after.state().pending_approvals().is_empty());
    assert!(fake.writes().unwrap().is_empty());
    assert_eq!(
        after
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
    assert_eq!(provider.captured_requests().unwrap().len(), 1);
}

#[tokio::test]
async fn manual_compact_rejects_pending_structured_approval_without_mutation() {
    let format = FinalOutputFormat::JsonSchema {
        schema: json!({
            "type": "object",
            "properties": {"status": {"const": "done"}},
            "required": ["status"],
            "additionalProperties": false
        }),
    };
    let provider = common::provider_with_capabilities(
        [
            write_script(),
            ScriptedModelResponse::text([r#"{"status":"done"}"#]),
        ],
        ModelCapabilities::text()
            .with_tools(false)
            .with_final_json_schema_with_tools(),
    );
    let store = store().await;
    let fake = FakeWriteTool::new();
    let tools = write_registry(fake.clone());
    let mut policy = PolicyEngine::new();
    policy.add_rule(CodingWorkspacePolicy).unwrap();
    let events = EventCollector::default();
    let ids = TestIds::default();
    let kernel = AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &store,
        &FixedClock,
        &ids,
        &events,
    );
    let run_config = config()
        .with_final_output_format(Some(format.clone()))
        .unwrap();

    let waiting = kernel
        .run(session_id(), &run_config, CancellationScope::new())
        .await
        .unwrap();
    assert_eq!(waiting.state(), RunState::WaitingApproval);
    let before = store.load(session_id()).await.unwrap();
    let request = match &before.approval_artifacts()[0] {
        ApprovalArtifactEntry::Requested { request, .. } => request.clone(),
        ApprovalArtifactEntry::Resolved { .. } => panic!("expected pending request"),
    };
    let summary = CanonicalMessage::assistant(
        MessageId::from_str("0195a0b1-5e90-7000-8000-000000000092").unwrap(),
        vec![ContentBlock::text("compacted summary").unwrap()],
        StopReason::Completed,
        timestamp(),
    )
    .unwrap();

    let error = kernel
        .compact(session_id(), summary, before.state().tail_record_id())
        .await
        .unwrap_err();
    assert_eq!(error.code(), tea_kernel::KernelErrorCode::InvalidState);
    let after = store.load(session_id()).await.unwrap();
    assert_eq!(after.journal_revision(), before.journal_revision());
    assert_eq!(after.records(), before.records());
    assert_eq!(
        after.state().latest_compaction(),
        before.state().latest_compaction()
    );

    let resolution = ApprovalResolution::new(
        &request,
        tea_protocol::ApprovalDecision::AllowOnce,
        timestamp(),
        None,
    )
    .unwrap();
    let resumed = kernel
        .resume_approval(
            session_id(),
            &resolution,
            &config(),
            CancellationScope::new(),
        )
        .await
        .unwrap();

    assert_eq!(resumed.state(), RunState::Completed);
    assert_eq!(
        fake.writes().unwrap(),
        [("/notes.txt".to_owned(), "hello".to_owned())]
    );
    let requests = provider.captured_requests().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(
        requests
            .iter()
            .all(|request| request.final_output_format() == Some(&format))
    );
}

#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn fresh_kernel_resumes_allowed_approval_from_persisted_context() {
    let provider = provider([
        write_script(),
        ScriptedModelResponse::text(["write complete"]),
    ]);
    let store = store().await;
    let fake = FakeWriteTool::new();
    let tools = write_registry(fake.clone());
    let mut policy = PolicyEngine::new();
    policy.add_rule(CodingWorkspacePolicy).unwrap();
    let events = EventCollector::default();
    let first_ids = TestIds::default();
    let waiting = AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &store,
        &FixedClock,
        &first_ids,
        &events,
    )
    .run(session_id(), &config(), CancellationScope::new())
    .await
    .unwrap();

    let snapshot = store.load(session_id()).await.unwrap();
    let request = match &snapshot.approval_artifacts()[0] {
        tea_session::ApprovalArtifactEntry::Requested { request, .. } => request.clone(),
        tea_session::ApprovalArtifactEntry::Resolved { .. } => {
            panic!("expected pending request")
        }
    };
    let resolution = ApprovalResolution::new(
        &request,
        tea_protocol::ApprovalDecision::AllowOnce,
        timestamp(),
        None,
    )
    .unwrap();
    let mismatch_ids = TestIds::with_start(50);
    let mismatched_config = KernelRunConfig::new(
        ActorId::from_str("user:alice").unwrap(),
        PolicyEnvironment::new(
            ExecutionSurface::Test,
            PolicyExecutionTarget::Remote,
            ProtocolMetadata::default(),
        ),
    );
    let mismatch = AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &store,
        &FixedClock,
        &mismatch_ids,
        &events,
    )
    .resume_approval(
        session_id(),
        &resolution,
        &mismatched_config,
        CancellationScope::new(),
    )
    .await
    .unwrap_err();
    assert_eq!(mismatch.code(), tea_kernel::KernelErrorCode::PolicyFailure);
    assert!(fake.writes().unwrap().is_empty());

    let resume_ids = TestIds::with_start(100);
    let resumed = AgentKernel::new(
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

    assert_eq!(waiting.state(), RunState::WaitingApproval);
    assert_eq!(resumed.state(), RunState::Completed);
    assert_eq!(
        fake.writes().unwrap(),
        [("/notes.txt".to_owned(), "hello".to_owned())]
    );
    let snapshot = store.load(session_id()).await.unwrap();
    assert_eq!(snapshot.approval_artifacts().len(), 2);
    assert!(snapshot.state().pending_approvals().is_empty());
    assert!(
        snapshot
            .records()
            .iter()
            .any(|record| matches!(record.record(), SessionRecord::ToolExecutionStarted { .. }))
    );
}

#[tokio::test]
async fn provider_preflight_rejects_allow_before_resolution_or_tool_side_effect() {
    let provider = RejectingPreflightProvider::new(
        [
            write_script(),
            ScriptedModelResponse::text(["must not stream"]),
        ],
        3,
    );
    let store = store().await;
    let fake = FakeWriteTool::new();
    let tools = write_registry(fake.clone());
    let mut policy = PolicyEngine::new();
    policy.add_rule(CodingWorkspacePolicy).unwrap();
    let events = EventCollector::default();

    let waiting = AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &store,
        &FixedClock,
        &TestIds::default(),
        &events,
    )
    .run(session_id(), &config(), CancellationScope::new())
    .await
    .unwrap();
    assert_eq!(waiting.state(), RunState::WaitingApproval);
    assert_eq!(provider.validate_calls(), 2);

    let paused = store.load(session_id()).await.unwrap();
    let request = match &paused.approval_artifacts()[0] {
        ApprovalArtifactEntry::Requested {
            request,
            final_output_format,
            ..
        } => {
            assert!(final_output_format.is_none());
            request.clone()
        }
        ApprovalArtifactEntry::Resolved { .. } => panic!("expected pending request"),
    };
    let resolution = ApprovalResolution::new(
        &request,
        tea_protocol::ApprovalDecision::AllowOnce,
        timestamp(),
        None,
    )
    .unwrap();
    let event_count = events.events().len();

    let error = AgentKernel::new(
        &provider,
        &tools,
        &policy,
        &store,
        &FixedClock,
        &TestIds::with_start(300),
        &events,
    )
    .resume_approval(
        session_id(),
        &resolution,
        &config(),
        CancellationScope::new(),
    )
    .await
    .unwrap_err();

    assert_eq!(error.code(), tea_kernel::KernelErrorCode::InvalidRequest);
    assert_eq!(provider.validate_calls(), 3);
    assert!(fake.writes().unwrap().is_empty());
    assert_eq!(provider.captured_requests().len(), 1);
    assert_eq!(provider.remaining_scripts(), 1);
    assert_eq!(events.events().len(), event_count);
    let after = store.load(session_id()).await.unwrap();
    assert_eq!(after.journal_revision(), paused.journal_revision());
    assert_eq!(after.records(), paused.records());
    assert_eq!(after.approval_artifacts(), paused.approval_artifacts());
    assert_eq!(after.state().pending_approvals().len(), 1);
}

#[tokio::test]
async fn ask_persists_request_artifact_and_wait_checkpoint_atomically() {
    let provider = provider([write_script()]);
    let store = store().await;
    let fake = FakeWriteTool::new();
    let tools = write_registry(fake.clone());
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

    let outcome = kernel
        .run(session_id(), &config(), CancellationScope::new())
        .await
        .unwrap();
    assert_eq!(outcome.state(), RunState::WaitingApproval);
    let approval_id = outcome.pending_approval_id().unwrap();
    assert!(
        outcome
            .session()
            .pending_approvals()
            .contains_key(&approval_id)
    );
    assert!(fake.writes().unwrap().is_empty());

    let snapshot = store.load(session_id()).await.unwrap();
    assert_eq!(snapshot.approval_artifacts().len(), 1);
    assert_eq!(snapshot.journal_revision(), 1);
    assert_eq!(
        snapshot.state().latest_checkpoint().unwrap().next_action(),
        NextTurnAction::WaitForApproval
    );
    assert!(matches!(
        snapshot.records()[snapshot.records().len() - 3].record(),
        SessionRecord::PolicyDecisionRecorded { .. }
    ));
    assert!(matches!(
        snapshot.records()[snapshot.records().len() - 2].record(),
        SessionRecord::ApprovalRequested { .. }
    ));
    assert!(matches!(
        snapshot.records().last().unwrap().record(),
        SessionRecord::TurnCheckpointed { .. }
    ));
}

#[tokio::test]
async fn approval_preview_is_ephemeral_and_precedes_the_approval_event() {
    let provider = provider([write_script()]);
    let store = store().await;
    let tools = preview_registry();
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
    .run(session_id(), &config(), CancellationScope::new())
    .await
    .unwrap();

    assert_eq!(outcome.state(), RunState::WaitingApproval);
    let events = events.events();
    let preview_index = events
        .iter()
        .position(|event| {
            matches!(
                event.event(),
                tea_protocol::AgentEvent::ToolExecutionPreview {
                    presentation: ToolPresentation::CodeChange(change),
                    ..
                } if change.path() == "notes.txt"
            )
        })
        .expect("preview event must be emitted");
    let approval_index = events
        .iter()
        .position(|event| {
            matches!(
                event.event(),
                tea_protocol::AgentEvent::ApprovalRequested { .. }
            )
        })
        .expect("approval event must be emitted");
    assert!(preview_index < approval_index);

    let snapshot = store.load(session_id()).await.unwrap();
    assert!(snapshot.records().iter().all(|record| !matches!(
        record.record(),
        SessionRecord::ToolExecutionFinished {
            presentation: Some(_),
            ..
        }
    )));
}

#[tokio::test]
async fn fresh_kernel_rejects_approval_after_source_digest_drift() {
    let provider = provider([write_script()]);
    let store = store().await;
    let fake = FakeWriteTool::new();
    let original_source =
        mcp_source("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    let original_tools = write_registry_with_source(fake.clone(), original_source.clone());
    let mut policy = PolicyEngine::new();
    policy.add_rule(CodingWorkspacePolicy).unwrap();
    let events = EventCollector::default();
    AgentKernel::new(
        &provider,
        &original_tools,
        &policy,
        &store,
        &FixedClock,
        &TestIds::default(),
        &events,
    )
    .run(session_id(), &config(), CancellationScope::new())
    .await
    .unwrap();

    let snapshot = store.load(session_id()).await.unwrap();
    let request = match &snapshot.approval_artifacts()[0] {
        tea_session::ApprovalArtifactEntry::Requested { request, .. } => request.clone(),
        tea_session::ApprovalArtifactEntry::Resolved { .. } => panic!("expected request"),
    };
    assert_eq!(request.tool_source(), &original_source);
    let resolution = ApprovalResolution::new(
        &request,
        tea_protocol::ApprovalDecision::AllowOnce,
        timestamp(),
        None,
    )
    .unwrap();
    let drifted_tools = write_registry_with_source(
        fake.clone(),
        mcp_source("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"),
    );
    let error = AgentKernel::new(
        &provider,
        &drifted_tools,
        &policy,
        &store,
        &FixedClock,
        &TestIds::with_start(900),
        &events,
    )
    .resume_approval(
        session_id(),
        &resolution,
        &config(),
        CancellationScope::new(),
    )
    .await
    .unwrap_err();
    assert_eq!(error.code(), tea_kernel::KernelErrorCode::PolicyFailure);
    assert!(fake.writes().unwrap().is_empty());
}

#[tokio::test]
async fn fresh_kernel_rejects_approval_after_mcp_server_substitution() {
    let provider = provider([write_script()]);
    let store = store().await;
    let fake = FakeWriteTool::new();
    let source = mcp_source("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
    let original_tools = write_registry_with_source(fake.clone(), source.clone());
    let mut policy = PolicyEngine::new();
    policy.add_rule(CodingWorkspacePolicy).unwrap();
    let events = EventCollector::default();
    AgentKernel::new(
        &provider,
        &original_tools,
        &policy,
        &store,
        &FixedClock,
        &TestIds::default(),
        &events,
    )
    .run(session_id(), &config(), CancellationScope::new())
    .await
    .unwrap();

    let snapshot = store.load(session_id()).await.unwrap();
    let request = match &snapshot.approval_artifacts()[0] {
        tea_session::ApprovalArtifactEntry::Requested { request, .. } => request.clone(),
        tea_session::ApprovalArtifactEntry::Resolved { .. } => panic!("expected request"),
    };
    let resolution = ApprovalResolution::new(
        &request,
        tea_protocol::ApprovalDecision::AllowOnce,
        timestamp(),
        None,
    )
    .unwrap();
    let substituted_tools = write_registry_with_source(
        fake.clone(),
        mcp_source_for(
            "workspace.replacement",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ),
    );
    let error = AgentKernel::new(
        &provider,
        &substituted_tools,
        &policy,
        &store,
        &FixedClock,
        &TestIds::with_start(900),
        &events,
    )
    .resume_approval(
        session_id(),
        &resolution,
        &config(),
        CancellationScope::new(),
    )
    .await
    .unwrap_err();
    assert_eq!(error.code(), tea_kernel::KernelErrorCode::PolicyFailure);
    assert!(fake.writes().unwrap().is_empty());
}
