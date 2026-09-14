//! Reliability ports wired through the runtime facade.

use crate::common;

use std::future;
use std::pin::Pin;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use tea::{AgentRuntimeBuilder, RuntimeCommandOutcome};
use tea_kernel::{CompactionPolicy, CompactionSummarizer, ModelRetryPolicy, RunState};
use tea_model::{
    ModelCapabilities, ModelCompletion, ModelEvent, ModelResponseInfo, ModelStreamIndex,
    ProviderToolCallId, ToolCallCompleted, ToolCallStarted,
};
use tea_profile::ProfileRuleId;
use tea_protocol::{
    AgentCommand, BranchId, CanonicalMessage, ContentBlock, FinalOutputFormat, MessageId,
    ProtocolTimestamp, SessionRecord, StopReason,
};
use tea_testkit::ScriptedModelResponse;

use common::{TestIds, TestSessionIds, build_runtime, user_message};

#[derive(Debug, Default)]
struct FixedSummarizer(AtomicUsize);

impl FixedSummarizer {
    fn calls(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }
}

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
        self.0.fetch_add(1, Ordering::SeqCst);
        Box::pin(future::ready(Ok(CanonicalMessage::assistant(
            MessageId::from_str("0195a0b1-5e90-7000-8000-0000000000d1").unwrap(),
            vec![ContentBlock::text("summary").unwrap()],
            StopReason::Completed,
            ProtocolTimestamp::from_str(common::NOW).unwrap(),
        )
        .unwrap())))
    }
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
                serde_json::json!({"path":"/notes.txt","content":"hello"}),
            )
            .unwrap(),
        ),
        ModelEvent::Completed(ModelCompletion::new(StopReason::ToolUse).unwrap()),
    ])
}

#[derive(Debug, Clone, Copy, Default)]
struct AlwaysCompact;
impl CompactionPolicy for AlwaysCompact {
    fn should_compact(&self, _: usize, _: u64) -> bool {
        true
    }
}

#[tokio::test]
async fn builder_accepts_reliability_ports() {
    let runtime = AgentRuntimeBuilder::new()
        .provider(common::provider_with([ScriptedModelResponse::text([
            "done",
        ])]))
        .ids(Arc::new(TestIds::default()))
        .session_id_source(Arc::new(TestSessionIds::default()))
        .actor("user:alice".parse().unwrap())
        .retry_policy(
            ModelRetryPolicy::new(
                2,
                std::time::Duration::from_millis(1),
                std::time::Duration::from_millis(2),
            )
            .unwrap(),
        )
        .compaction_policy(Arc::new(AlwaysCompact))
        .compaction_summarizer(Arc::new(FixedSummarizer::default()))
        .tool(
            common::spec("read_file", tea_tools::ToolEffect::FsRead),
            Arc::new(
                tea_tools::ArgumentResourceResolver::new(
                    "path",
                    "file",
                    tea_tools::ToolResourceAccess::Read,
                )
                .unwrap(),
            ),
            Arc::new(tea_testkit::FakeReadTool::new([(
                "/a".to_owned(),
                "x".to_owned(),
            )])),
        )
        .unwrap()
        .tool(
            common::spec("write_file", tea_tools::ToolEffect::FsWrite),
            Arc::new(
                tea_tools::ArgumentResourceResolver::new(
                    "path",
                    "file",
                    tea_tools::ToolResourceAccess::Write,
                )
                .unwrap(),
            ),
            Arc::new(tea_testkit::FakeWriteTool::new()),
        )
        .unwrap()
        .policy_rule(
            ProfileRuleId::from_str("product.coding_workspace").unwrap(),
            Arc::new(tea_policy::CodingWorkspacePolicy),
        )
        .unwrap()
        .profile(tea_profile::AgentProfile::coding_agent().unwrap())
        .build()
        .unwrap();
    assert_eq!(runtime.health().profile_ids().len(), 1);
}

#[tokio::test]
async fn runtime_compact_appends_summary() {
    let runtime = build_runtime(
        common::provider_with([ScriptedModelResponse::text(["done"])]),
        Arc::new(TestIds::default()),
        Arc::new(TestSessionIds::default()),
    )
    .unwrap();
    let session_id = match runtime
        .send(common::envelope_create("coding-agent"))
        .await
        .unwrap()
    {
        RuntimeCommandOutcome::Created { session_id } => session_id,
        other => panic!("expected Created, got {other:?}"),
    };
    let _ = runtime
        .send(common::envelope_prompt(user_message("hello"), session_id))
        .await
        .unwrap();
    let snapshot = runtime.snapshot(session_id).await.unwrap();
    let tail = snapshot.state().tail_record_id();
    let summary = CanonicalMessage::assistant(
        MessageId::from_str("0195a0b1-5e90-7000-8000-0000000000d2").unwrap(),
        vec![ContentBlock::text("manual summary").unwrap()],
        StopReason::Completed,
        ProtocolTimestamp::from_str(common::NOW).unwrap(),
    )
    .unwrap();
    let after = runtime
        .compact(session_id, summary.clone(), tail)
        .await
        .unwrap();
    assert_eq!(after.state().messages().len(), 1);
    assert_eq!(after.state().messages()[0], summary);
    assert!(after.state().latest_compaction().is_some());
}

#[tokio::test]
async fn compact_command_uses_the_configured_summarizer() {
    let runtime = AgentRuntimeBuilder::new()
        .provider(common::provider_with([ScriptedModelResponse::text([
            "done",
        ])]))
        .clock(Arc::new(common::FixedClock))
        .ids(Arc::new(TestIds::default()))
        .session_id_source(Arc::new(TestSessionIds::default()))
        .actor("user:alice".parse().unwrap())
        .compaction_summarizer(Arc::new(FixedSummarizer::default()))
        .tool(
            common::spec("read_file", tea_tools::ToolEffect::FsRead),
            Arc::new(
                tea_tools::ArgumentResourceResolver::new(
                    "path",
                    "file",
                    tea_tools::ToolResourceAccess::Read,
                )
                .unwrap(),
            ),
            Arc::new(tea_testkit::FakeReadTool::new([(
                "/a".to_owned(),
                "x".to_owned(),
            )])),
        )
        .unwrap()
        .tool(
            common::spec("write_file", tea_tools::ToolEffect::FsWrite),
            Arc::new(
                tea_tools::ArgumentResourceResolver::new(
                    "path",
                    "file",
                    tea_tools::ToolResourceAccess::Write,
                )
                .unwrap(),
            ),
            Arc::new(tea_testkit::FakeWriteTool::new()),
        )
        .unwrap()
        .policy_rule(
            ProfileRuleId::from_str("product.coding_workspace").unwrap(),
            Arc::new(tea_policy::CodingWorkspacePolicy),
        )
        .unwrap()
        .profile(tea_profile::AgentProfile::coding_agent().unwrap())
        .build()
        .unwrap();
    let session_id = match runtime
        .send(common::envelope_create("coding-agent"))
        .await
        .unwrap()
    {
        RuntimeCommandOutcome::Created { session_id } => session_id,
        other => panic!("expected Created, got {other:?}"),
    };
    let _ = runtime
        .send(common::envelope_prompt(user_message("hello"), session_id))
        .await
        .unwrap();
    let outcome = runtime
        .send(common::envelope(
            AgentCommand::CompactSession { instruction: None },
            Some(session_id),
        ))
        .await
        .unwrap();
    assert!(matches!(
        outcome,
        RuntimeCommandOutcome::SessionCompacted { .. }
    ));
    let snapshot = runtime.snapshot(session_id).await.unwrap();
    assert_eq!(snapshot.state().messages().len(), 1);
    let CanonicalMessage::Assistant { content, .. } = &snapshot.state().messages()[0] else {
        panic!("expected assistant summary");
    };
    assert!(matches!(
        &content[0],
        ContentBlock::Text { text } if text == "summary"
    ));
}

#[tokio::test]
async fn compact_command_rejects_pending_approval_before_summarizing() {
    let provider = common::provider_with_capabilities(
        [write_script()],
        ModelCapabilities::text()
            .with_tools(true)
            .with_final_json_schema_with_tools(),
    );
    let summarizer = Arc::new(FixedSummarizer::default());
    let runtime = common::runtime_builder(
        provider,
        Arc::new(TestIds::default()),
        Arc::new(TestSessionIds::default()),
    )
    .unwrap()
    .compaction_summarizer(summarizer.clone())
    .build()
    .unwrap();
    let session_id = match runtime
        .send(common::envelope_create("coding-agent"))
        .await
        .unwrap()
    {
        RuntimeCommandOutcome::Created { session_id } => session_id,
        other => panic!("expected Created, got {other:?}"),
    };
    let waiting = runtime
        .send(common::envelope(
            AgentCommand::Prompt {
                message: user_message("write the notes"),
                final_output_format: Some(FinalOutputFormat::JsonSchema {
                    schema: serde_json::json!({"type":"object"}),
                }),
            },
            Some(session_id),
        ))
        .await
        .unwrap();
    assert!(matches!(
        waiting,
        RuntimeCommandOutcome::RunCompleted {
            state: RunState::WaitingApproval,
            ..
        }
    ));
    let before = runtime.snapshot(session_id).await.unwrap();

    let error = runtime
        .send(common::envelope(
            AgentCommand::CompactSession { instruction: None },
            Some(session_id),
        ))
        .await
        .unwrap_err();

    assert_eq!(error.code(), tea::RuntimeErrorCode::KernelFailure);
    assert_eq!(
        error.message(),
        "session has a pending approval; resume it explicitly"
    );
    assert_eq!(summarizer.calls(), 0);
    let after = runtime.snapshot(session_id).await.unwrap();
    assert_eq!(after.journal_revision(), before.journal_revision());
    assert_eq!(after, before);
}

#[tokio::test]
async fn fork_command_creates_and_activates_a_branch_from_message() {
    let runtime = build_runtime(
        common::provider_with([ScriptedModelResponse::text(["done"])]),
        Arc::new(TestIds::default()),
        Arc::new(TestSessionIds::default()),
    )
    .unwrap();
    let session_id = match runtime
        .send(common::envelope_create("coding-agent"))
        .await
        .unwrap()
    {
        RuntimeCommandOutcome::Created { session_id } => session_id,
        other => panic!("expected Created, got {other:?}"),
    };
    let prompt = user_message("fork here");
    let CanonicalMessage::User { id: message_id, .. } = &prompt else {
        panic!("expected user prompt");
    };
    let message_id = *message_id;
    let _ = runtime
        .send(common::envelope_prompt(prompt, session_id))
        .await
        .unwrap();
    let branch_id = BranchId::from_str("0195a0b1-5e90-7000-8000-0000000000e1").unwrap();
    let outcome = runtime
        .send(common::envelope(
            AgentCommand::ForkSession {
                from_message_id: message_id,
                branch_id,
            },
            Some(session_id),
        ))
        .await
        .unwrap();
    assert!(matches!(
        outcome,
        RuntimeCommandOutcome::SessionForked {
            branch_id: actual,
            ..
        } if actual == branch_id
    ));
    let snapshot = runtime.snapshot(session_id).await.unwrap();
    assert_eq!(snapshot.state().active_branch_id(), Some(branch_id));
    assert!(matches!(
        snapshot.records()[snapshot.records().len() - 2].record(),
        SessionRecord::BranchCreated { branch_id: actual, .. } if *actual == branch_id
    ));
    assert!(matches!(
        snapshot.records().last().unwrap().record(),
        SessionRecord::ActiveBranchChanged { branch_id: actual } if *actual == branch_id
    ));
}
