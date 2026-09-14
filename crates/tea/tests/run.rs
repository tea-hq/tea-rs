use crate::common;

use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use common::{
    TestIds, TestSessionIds, build_runtime, runtime_builder, runtime_builder_with_write_tool,
    user_message,
};
use tea::{RuntimeCommandOutcome, RuntimeErrorCode};
use tea_model::{
    BoxModelStream, ModelCancellation, ModelCapabilities, ModelCompletion, ModelDisplayName,
    ModelEvent, ModelFailure, ModelFailureCode, ModelProvider, ModelRequest, ModelResponseInfo,
    ModelSpec, ModelStreamIndex, ProviderId, ProviderToolCallId, ToolCallCompleted,
    ToolCallStarted,
};
use tea_policy::GrantScope;
use tea_protocol::{
    AgentCommand, ApprovalDecision, CanonicalMessage, CommandEnvelope, CommandId, ContentBlock,
    MessageId, ModelId, ProtocolMetadata, ProtocolTimestamp, RecordEnvelope, RecordId, RetryClass,
    SessionId, SessionRecord, SessionSequence, StopReason, TokenCount,
};
use tea_session::{AppendTransaction, InMemorySessionStore, SessionStore};
use tea_testkit::{FakeWriteTool, ScriptStep, ScriptedModelProvider, ScriptedModelResponse};

const NOW: &str = "2026-07-23T09:30:12.125Z";

fn envelope(command: AgentCommand, session_id: Option<SessionId>) -> CommandEnvelope {
    CommandEnvelope::new(
        CommandId::from_str("0195a0b1-0000-7000-8000-000000000010").unwrap(),
        session_id,
        ProtocolTimestamp::from_str(NOW).unwrap(),
        command,
    )
    .unwrap()
}

fn tool_script(name: &str, arguments: serde_json::Value, opaque_id: &str) -> ScriptedModelResponse {
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

fn provider(
    scripts: impl IntoIterator<Item = ScriptedModelResponse>,
) -> Arc<ScriptedModelProvider> {
    let scripts = scripts.into_iter().collect::<Vec<_>>();
    let provider_id = ProviderId::from_str("fake").unwrap();
    let model = ModelSpec::new(
        ModelId::from_str("fake/model").unwrap(),
        provider_id.clone(),
        ModelDisplayName::from_str("Fake Model").unwrap(),
        TokenCount::new(32_000).unwrap(),
        TokenCount::new(4_000).unwrap(),
        ModelCapabilities::text().with_tools(true),
    )
    .unwrap();
    Arc::new(ScriptedModelProvider::new(
        provider_id,
        vec![model],
        scripts,
    ))
}

#[derive(Debug)]
struct RejectingPreflightProvider {
    inner: ScriptedModelProvider,
    fail_on_call: usize,
    preflight_calls: AtomicUsize,
}

impl RejectingPreflightProvider {
    fn new(
        scripts: impl IntoIterator<Item = ScriptedModelResponse>,
        capabilities: ModelCapabilities,
        fail_on_call: usize,
    ) -> Self {
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
        Self {
            inner: ScriptedModelProvider::new(provider_id, vec![model], scripts),
            fail_on_call,
            preflight_calls: AtomicUsize::new(0),
        }
    }

    fn captured_requests(&self) -> Vec<ModelRequest> {
        self.inner.captured_requests().unwrap()
    }

    fn preflight_calls(&self) -> usize {
        self.preflight_calls.load(Ordering::SeqCst)
    }
}

impl ModelProvider for RejectingPreflightProvider {
    fn provider_id(&self) -> &ProviderId {
        self.inner.provider_id()
    }

    fn models(&self) -> &[ModelSpec] {
        self.inner.models()
    }

    fn validate_request(&self, request: &ModelRequest) -> Result<(), ModelFailure> {
        <ScriptedModelProvider as ModelProvider>::validate_request(&self.inner, request)?;
        let call = self.preflight_calls.fetch_add(1, Ordering::SeqCst) + 1;
        if call == self.fail_on_call {
            return Err(ModelFailure::new(
                ModelFailureCode::InvalidRequest,
                "provider-specific preflight rejected request",
                RetryClass::Never,
            )
            .unwrap());
        }
        Ok(())
    }

    fn stream(&self, request: ModelRequest, cancellation: ModelCancellation) -> BoxModelStream {
        self.inner.stream(request, cancellation)
    }
}

async fn create_session(runtime: &tea::AgentRuntime, profile_id: &str) -> SessionId {
    match runtime
        .send(envelope(
            AgentCommand::CreateSession {
                profile_id: profile_id.parse().unwrap(),
                metadata: ProtocolMetadata::default(),
            },
            None,
        ))
        .await
        .unwrap()
    {
        RuntimeCommandOutcome::Created { session_id } => session_id,
        other => panic!("expected Created, got {other:?}"),
    }
}

#[tokio::test]
async fn provider_preflight_rejection_leaves_no_orphan_user_message() {
    let provider = Arc::new(RejectingPreflightProvider::new(
        [ScriptedModelResponse::text([r#"{"answer":"unused"}"#])],
        ModelCapabilities::text()
            .with_tools(true)
            .with_final_json_schema_with_tools(),
        1,
    ));
    let write_tool = Arc::new(FakeWriteTool::new());
    let runtime_provider: Arc<dyn ModelProvider> = provider.clone();
    let runtime = runtime_builder_with_write_tool(
        runtime_provider,
        Arc::new(TestIds::default()),
        Arc::new(TestSessionIds::default()),
        write_tool,
    )
    .unwrap()
    .build()
    .unwrap();
    let session_id = create_session(&runtime, "coding-agent").await;

    let error = runtime
        .send(envelope(
            AgentCommand::Prompt {
                message: user_message("return schema-constrained JSON"),
                final_output_format: Some(tea_protocol::FinalOutputFormat::JsonSchema {
                    schema: serde_json::json!({"type": "object"}),
                }),
            },
            Some(session_id),
        ))
        .await
        .unwrap_err();

    assert_eq!(error.code(), RuntimeErrorCode::InvalidRequest);
    assert_eq!(
        error.message(),
        "provider-specific preflight rejected request"
    );
    assert_eq!(provider.preflight_calls(), 1);
    assert!(provider.captured_requests().is_empty());
    assert!(
        runtime
            .snapshot(session_id)
            .await
            .unwrap()
            .state()
            .messages()
            .is_empty()
    );
}

#[tokio::test]
async fn approval_resume_preflight_rejection_commits_nothing_and_executes_no_side_effect() {
    let provider = Arc::new(RejectingPreflightProvider::new(
        [
            tool_script(
                "write_file",
                serde_json::json!({"path":"/summary.txt","content":"hello"}),
                "write-preflight-1",
            ),
            ScriptedModelResponse::text([r#"{"status":"unused"}"#]),
        ],
        ModelCapabilities::text()
            .with_tools(true)
            .with_final_json_schema_with_tools(),
        4,
    ));
    let write_tool = Arc::new(FakeWriteTool::new());
    let runtime_provider: Arc<dyn ModelProvider> = provider.clone();
    let runtime = runtime_builder_with_write_tool(
        runtime_provider,
        Arc::new(TestIds::default()),
        Arc::new(TestSessionIds::default()),
        Arc::clone(&write_tool),
    )
    .unwrap()
    .build()
    .unwrap();
    let session_id = create_session(&runtime, "coding-agent").await;
    let waiting = runtime
        .send(envelope(
            AgentCommand::Prompt {
                message: user_message("write one file and return JSON"),
                final_output_format: Some(tea_protocol::FinalOutputFormat::JsonSchema {
                    schema: serde_json::json!({"type": "object"}),
                }),
            },
            Some(session_id),
        ))
        .await
        .unwrap();
    let approval_id = match waiting {
        RuntimeCommandOutcome::RunCompleted {
            state: tea_kernel::RunState::WaitingApproval,
            pending_approval_id: Some(approval_id),
            ..
        } => approval_id,
        other => panic!("expected WaitingApproval, got {other:?}"),
    };
    let before = runtime.snapshot(session_id).await.unwrap();

    let error = runtime
        .send(envelope(
            AgentCommand::ResolveApproval {
                approval_id,
                decision: ApprovalDecision::AllowOnce,
            },
            Some(session_id),
        ))
        .await
        .unwrap_err();

    assert_eq!(error.code(), RuntimeErrorCode::InvalidRequest);
    assert_eq!(
        error.message(),
        "provider-specific preflight rejected request"
    );
    assert_eq!(provider.preflight_calls(), 4);
    assert_eq!(provider.captured_requests().len(), 1);
    assert!(write_tool.writes().unwrap().is_empty());
    let after = runtime.snapshot(session_id).await.unwrap();
    assert_eq!(after.records(), before.records());
    assert_eq!(after.approval_artifacts(), before.approval_artifacts());
    assert_eq!(after.state().pending_approvals().len(), 1);

    let denied = runtime
        .send(envelope(
            AgentCommand::ResolveApproval {
                approval_id,
                decision: ApprovalDecision::Deny,
            },
            Some(session_id),
        ))
        .await
        .unwrap();

    assert!(matches!(
        denied,
        RuntimeCommandOutcome::RunCompleted {
            state: tea_kernel::RunState::Completed,
            pending_approval_id: None,
            ..
        }
    ));
    assert_eq!(provider.preflight_calls(), 6);
    assert!(write_tool.writes().unwrap().is_empty());
}

#[tokio::test]
async fn prompt_during_pending_approval_is_not_persisted_or_mixed_into_the_suspended_run() {
    let provider = Arc::new(RejectingPreflightProvider::new(
        [
            tool_script(
                "write_file",
                serde_json::json!({"path":"/summary.txt","content":"hello"}),
                "write-pending-1",
            ),
            ScriptedModelResponse::text([r#"{"status":"done"}"#]),
        ],
        ModelCapabilities::text()
            .with_tools(true)
            .with_final_json_object()
            .with_final_json_schema_with_tools(),
        usize::MAX,
    ));
    let runtime_provider: Arc<dyn ModelProvider> = provider.clone();
    let runtime = runtime_builder_with_write_tool(
        runtime_provider,
        Arc::new(TestIds::default()),
        Arc::new(TestSessionIds::default()),
        Arc::new(FakeWriteTool::new()),
    )
    .unwrap()
    .build()
    .unwrap();
    let session_id = create_session(&runtime, "coding-agent").await;
    let waiting = runtime
        .send(envelope(
            AgentCommand::Prompt {
                message: user_message("write one file, then return JSON"),
                final_output_format: Some(tea_protocol::FinalOutputFormat::JsonObject),
            },
            Some(session_id),
        ))
        .await
        .unwrap();
    let approval_id = match waiting {
        RuntimeCommandOutcome::RunCompleted {
            state: tea_kernel::RunState::WaitingApproval,
            pending_approval_id: Some(approval_id),
            ..
        } => approval_id,
        other => panic!("expected WaitingApproval, got {other:?}"),
    };
    let before = runtime.snapshot(session_id).await.unwrap();

    let error = runtime
        .send(envelope(
            AgentCommand::Prompt {
                message: user_message("this prompt belongs to a different run"),
                final_output_format: Some(tea_protocol::FinalOutputFormat::JsonSchema {
                    schema: serde_json::json!({"type": "object"}),
                }),
            },
            Some(session_id),
        ))
        .await
        .unwrap_err();

    assert_eq!(
        error.message(),
        "session has a pending approval; resume it explicitly"
    );
    let after = runtime.snapshot(session_id).await.unwrap();
    assert_eq!(after.records(), before.records());
    assert_eq!(after.approval_artifacts(), before.approval_artifacts());

    let completed = runtime
        .send(envelope(
            AgentCommand::ResolveApproval {
                approval_id,
                decision: ApprovalDecision::AllowOnce,
            },
            Some(session_id),
        ))
        .await
        .unwrap();
    assert!(matches!(
        completed,
        RuntimeCommandOutcome::RunCompleted {
            state: tea_kernel::RunState::Completed,
            pending_approval_id: None,
            ..
        }
    ));
    let requests = provider.captured_requests();
    assert_eq!(requests.len(), 2);
    assert!(requests.iter().all(|request| {
        request.final_output_format() == Some(&tea_protocol::FinalOutputFormat::JsonObject)
    }));
}

#[tokio::test]
async fn prompt_pauses_for_approval_then_resumes() {
    let provider = provider([
        tool_script(
            "read_file",
            serde_json::json!({"path":"/notes.txt"}),
            "read-1",
        ),
        tool_script(
            "write_file",
            serde_json::json!({"path":"/summary.txt","content":"hello"}),
            "write-1",
        ),
        ScriptedModelResponse::text(["summary written"]),
    ]);
    let runtime = build_runtime(
        provider.clone(),
        Arc::new(TestIds::default()),
        Arc::new(TestSessionIds::default()),
    )
    .unwrap();
    let session_id = match runtime
        .send(envelope(
            AgentCommand::CreateSession {
                profile_id: "coding-agent".parse().unwrap(),
                metadata: ProtocolMetadata::default(),
            },
            None,
        ))
        .await
        .unwrap()
    {
        RuntimeCommandOutcome::Created { session_id } => session_id,
        other => panic!("expected Created, got {other:?}"),
    };

    let waiting = runtime
        .send(envelope(
            AgentCommand::Prompt {
                message: user_message("summarize the notes"),
                final_output_format: None,
            },
            Some(session_id),
        ))
        .await
        .unwrap();
    let pending_approval_id = match waiting {
        RuntimeCommandOutcome::RunCompleted {
            state,
            pending_approval_id,
            ..
        } => {
            assert_eq!(state, tea_kernel::RunState::WaitingApproval);
            pending_approval_id
        }
        other => panic!("expected WaitingApproval, got {other:?}"),
    };
    let approval_id = pending_approval_id.unwrap();

    let completed = runtime
        .send(envelope(
            AgentCommand::ResolveApproval {
                approval_id,
                decision: ApprovalDecision::AllowOnce,
            },
            Some(session_id),
        ))
        .await
        .unwrap();
    match completed {
        RuntimeCommandOutcome::RunCompleted { state, session, .. } => {
            assert_eq!(state, tea_kernel::RunState::Completed);
            assert!(session.pending_approvals().is_empty());
        }
        other => panic!("expected Completed, got {other:?}"),
    }

    let requests = provider.captured_requests().unwrap();
    assert_eq!(requests.len(), 3);
    // The compiled prompt must reach the model as the system prompt.
    assert!(requests.iter().all(|request| {
        request
            .system_prompt()
            .is_some_and(|prompt| !prompt.is_empty())
    }));
    assert_eq!(
        requests
            .iter()
            .map(|request| request.messages().len())
            .collect::<Vec<_>>(),
        [1, 3, 5]
    );
}

#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn allow_session_issues_a_bounded_grant_and_context_drift_asks_again() {
    let provider = provider([
        tool_script(
            "write_file",
            serde_json::json!({"path":"/summary.txt","content":"one"}),
            "write-1",
        ),
        tool_script(
            "write_file",
            serde_json::json!({"path":"/summary.txt","content":"two"}),
            "write-2",
        ),
        tool_script(
            "write_file",
            serde_json::json!({"path":"/other.txt","content":"other"}),
            "write-3",
        ),
        ScriptedModelResponse::text(["done"]),
    ]);
    let runtime = build_runtime(
        provider,
        Arc::new(TestIds::default()),
        Arc::new(TestSessionIds::default()),
    )
    .unwrap();
    let session_id = match runtime
        .send(envelope(
            AgentCommand::CreateSession {
                profile_id: "coding-agent".parse().unwrap(),
                metadata: ProtocolMetadata::default(),
            },
            None,
        ))
        .await
        .unwrap()
    {
        RuntimeCommandOutcome::Created { session_id } => session_id,
        other => panic!("expected Created, got {other:?}"),
    };
    let first_approval = match runtime
        .send(envelope(
            AgentCommand::Prompt {
                message: user_message("write two files"),
                final_output_format: None,
            },
            Some(session_id),
        ))
        .await
        .unwrap()
    {
        RuntimeCommandOutcome::RunCompleted {
            pending_approval_id: Some(approval_id),
            ..
        } => approval_id,
        other => panic!("expected approval, got {other:?}"),
    };

    let drift_approval = match runtime
        .send(envelope(
            AgentCommand::ResolveApproval {
                approval_id: first_approval,
                decision: ApprovalDecision::AllowSession,
            },
            Some(session_id),
        ))
        .await
        .unwrap()
    {
        RuntimeCommandOutcome::RunCompleted {
            state: tea_kernel::RunState::WaitingApproval,
            pending_approval_id: Some(approval_id),
            ..
        } => approval_id,
        other => panic!("expected context-drift approval, got {other:?}"),
    };

    let snapshot = runtime.sessions().load(session_id).await.unwrap();
    assert_eq!(snapshot.grant_journal().len(), 1);
    let tea_session::GrantJournalEntry::Issued { grant, .. } = &snapshot.grant_journal()[0] else {
        panic!("expected issued grant")
    };
    assert_eq!(grant.tool_name().as_str(), "write_file");
    assert_eq!(grant.effects(), [tea_tools::ToolEffect::FsWrite]);
    assert_eq!(grant.resources().len(), 1);
    assert_eq!(grant.resources()[0].locator_prefix(), "/summary.txt");
    assert_eq!(grant.scope(), &GrantScope::SessionResource { session_id });

    let completed = runtime
        .send(envelope(
            AgentCommand::ResolveApproval {
                approval_id: drift_approval,
                decision: ApprovalDecision::Deny,
            },
            Some(session_id),
        ))
        .await
        .unwrap();
    assert!(matches!(
        completed,
        RuntimeCommandOutcome::RunCompleted {
            state: tea_kernel::RunState::Completed,
            pending_approval_id: None,
            ..
        }
    ));
}

#[tokio::test]
async fn dropped_prompt_future_releases_active_run() {
    let provider = provider([
        ScriptedModelResponse::new([ScriptStep::AwaitCancellation]),
        ScriptedModelResponse::text(["recovered"]),
    ]);
    let runtime = Arc::new(
        build_runtime(
            provider.clone(),
            Arc::new(TestIds::default()),
            Arc::new(TestSessionIds::default()),
        )
        .unwrap(),
    );
    let session_id = match runtime
        .send(envelope(
            AgentCommand::CreateSession {
                profile_id: "coding-agent".parse().unwrap(),
                metadata: ProtocolMetadata::default(),
            },
            None,
        ))
        .await
        .unwrap()
    {
        RuntimeCommandOutcome::Created { session_id } => session_id,
        other => panic!("expected Created, got {other:?}"),
    };

    let task_runtime = Arc::clone(&runtime);
    let task = tokio::spawn(async move {
        task_runtime
            .send(envelope(
                AgentCommand::Prompt {
                    message: user_message("wait"),
                    final_output_format: None,
                },
                Some(session_id),
            ))
            .await
    });
    while provider.captured_requests().unwrap().is_empty() {
        tokio::task::yield_now().await;
    }
    task.abort();
    let _ = task.await;

    let recovered = runtime
        .send(envelope(
            AgentCommand::Prompt {
                message: user_message("continue"),
                final_output_format: None,
            },
            Some(session_id),
        ))
        .await
        .unwrap();
    assert!(matches!(
        recovered,
        RuntimeCommandOutcome::RunCompleted { .. }
    ));
}

#[tokio::test]
async fn prompt_rejects_an_oversized_prospective_transcript_without_an_orphan_message() {
    let provider = provider([]);
    let store = Arc::new(InMemorySessionStore::new());
    let runtime = runtime_builder(
        Arc::clone(&provider),
        Arc::new(TestIds::default()),
        Arc::new(TestSessionIds::default()),
    )
    .unwrap()
    .session_store(store.clone())
    .build()
    .unwrap();
    let session_id = create_session(&runtime, "coding-agent").await;
    let initial = store.load(session_id).await.unwrap();
    let tail = initial.state().tail_sequence();
    let branch_id = initial.state().active_branch_id();
    let records = (0..tea_model::MAX_REQUEST_MESSAGES)
        .map(|index| {
            RecordEnvelope::new(
                RecordId::from_str(&format!("0198a0b1-{index:04x}-7000-8000-000000000001"))
                    .unwrap(),
                session_id,
                SessionSequence::new(tail.get() + 1 + index as u64),
                ProtocolTimestamp::from_str(NOW).unwrap(),
                None,
                None,
                branch_id,
                ProtocolMetadata::default(),
                SessionRecord::MessageCommitted {
                    message: CanonicalMessage::user(
                        MessageId::from_str(&format!(
                            "0198a0b1-{index:04x}-7000-8000-000000000002"
                        ))
                        .unwrap(),
                        vec![ContentBlock::text("seed").unwrap()],
                        ProtocolTimestamp::from_str(NOW).unwrap(),
                    )
                    .unwrap(),
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    store
        .append(AppendTransaction::new(session_id, Some(tail), records))
        .await
        .unwrap();
    let before = store.load(session_id).await.unwrap();

    let error = runtime
        .send(envelope(
            AgentCommand::Prompt {
                message: user_message("must not be persisted"),
                final_output_format: None,
            },
            Some(session_id),
        ))
        .await
        .unwrap_err();

    assert_eq!(error.code(), RuntimeErrorCode::InvalidRequest);
    let after = store.load(session_id).await.unwrap();
    assert_eq!(
        after.state().messages().len(),
        tea_model::MAX_REQUEST_MESSAGES
    );
    assert_eq!(
        after.state().tail_sequence(),
        before.state().tail_sequence()
    );
    assert!(provider.captured_requests().unwrap().is_empty());
}

#[tokio::test]
async fn active_tool_overrides_are_isolated_switchable_and_cleared_by_profiles() {
    let provider = provider([
        ScriptedModelResponse::text(["one"]),
        ScriptedModelResponse::text(["two"]),
        ScriptedModelResponse::text(["three"]),
        ScriptedModelResponse::text(["four"]),
    ]);
    let runtime = build_runtime(
        Arc::clone(&provider),
        Arc::new(TestIds::default()),
        Arc::new(TestSessionIds::default()),
    )
    .unwrap();
    let first_session = create_session(&runtime, "coding-agent").await;
    let second_session = create_session(&runtime, "coding-agent").await;

    let error = runtime
        .set_active_tools(first_session, vec!["ghost".parse().unwrap()])
        .await
        .unwrap_err();
    assert_eq!(error.code(), RuntimeErrorCode::UnknownTool);

    runtime
        .set_active_tools(first_session, vec!["clipboard_read".parse().unwrap()])
        .await
        .unwrap();
    runtime
        .send(envelope(
            AgentCommand::Prompt {
                message: user_message("first"),
                final_output_format: None,
            },
            Some(first_session),
        ))
        .await
        .unwrap();
    runtime
        .send(envelope(
            AgentCommand::Prompt {
                message: user_message("second"),
                final_output_format: None,
            },
            Some(second_session),
        ))
        .await
        .unwrap();

    runtime
        .set_active_tools(first_session, vec!["read_file".parse().unwrap()])
        .await
        .unwrap();
    runtime
        .send(envelope(
            AgentCommand::Prompt {
                message: user_message("third"),
                final_output_format: None,
            },
            Some(first_session),
        ))
        .await
        .unwrap();
    runtime
        .send(envelope(
            AgentCommand::SetProfile {
                profile_id: "desktop-assistant".parse().unwrap(),
            },
            Some(first_session),
        ))
        .await
        .unwrap();
    runtime
        .send(envelope(
            AgentCommand::Prompt {
                message: user_message("fourth"),
                final_output_format: None,
            },
            Some(first_session),
        ))
        .await
        .unwrap();

    let requests = provider.captured_requests().unwrap();
    let names = requests
        .iter()
        .map(|request| {
            request
                .tools()
                .iter()
                .map(|tool| tool.name().to_owned())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        [
            vec!["clipboard_read"],
            vec!["read_file", "write_file"],
            vec!["read_file"],
            vec!["clipboard_read", "write_file"],
        ]
    );
    let first_prompt = requests[0].system_prompt().unwrap();
    assert!(first_prompt.contains("clipboard_read"));
    assert!(!first_prompt.contains("Invoke read_file"));
}

#[tokio::test]
async fn inactive_model_tool_call_is_rejected_without_execution() {
    let provider = provider([
        tool_script(
            "clipboard_read",
            serde_json::json!({"path":"clip"}),
            "inactive-1",
        ),
        ScriptedModelResponse::text(["recovered"]),
    ]);
    let runtime = build_runtime(
        Arc::clone(&provider),
        Arc::new(TestIds::default()),
        Arc::new(TestSessionIds::default()),
    )
    .unwrap();
    let session_id = create_session(&runtime, "coding-agent").await;
    runtime
        .set_active_tools(session_id, vec!["read_file".parse().unwrap()])
        .await
        .unwrap();
    runtime
        .send(envelope(
            AgentCommand::Prompt {
                message: user_message("try an inactive tool"),
                final_output_format: None,
            },
            Some(session_id),
        ))
        .await
        .unwrap();

    let requests = provider.captured_requests().unwrap();
    assert_eq!(
        requests[0]
            .tools()
            .iter()
            .map(tea_model::ModelToolDefinition::name)
            .collect::<Vec<_>>(),
        ["read_file"]
    );
    match requests[1].messages().last().unwrap() {
        CanonicalMessage::ToolResult {
            is_error,
            error: Some(error),
            content,
            ..
        } => {
            assert!(*is_error);
            assert_eq!(error.code(), "unknown_tool");
            assert!(matches!(content[0], ContentBlock::Text { .. }));
        }
        other => panic!("unexpected tool result: {other:?}"),
    }
}

#[tokio::test]
async fn active_tools_cannot_change_during_a_run() {
    let provider = provider([ScriptedModelResponse::new([ScriptStep::AwaitCancellation])]);
    let runtime = Arc::new(
        build_runtime(
            Arc::clone(&provider),
            Arc::new(TestIds::default()),
            Arc::new(TestSessionIds::default()),
        )
        .unwrap(),
    );
    let session_id = create_session(&runtime, "coding-agent").await;
    let task_runtime = Arc::clone(&runtime);
    let task = tokio::spawn(async move {
        task_runtime
            .send(envelope(
                AgentCommand::Prompt {
                    message: user_message("wait"),
                    final_output_format: None,
                },
                Some(session_id),
            ))
            .await
    });
    while provider.captured_requests().unwrap().is_empty() {
        tokio::task::yield_now().await;
    }

    let error = runtime
        .set_active_tools(session_id, vec!["read_file".parse().unwrap()])
        .await
        .unwrap_err();
    assert_eq!(error.code(), RuntimeErrorCode::RunAlreadyActive);

    task.abort();
    let _ = task.await;
    runtime
        .set_active_tools(session_id, vec!["read_file".parse().unwrap()])
        .await
        .unwrap();
}

#[tokio::test]
async fn prompt_rejects_active_run_and_abort_requires_active_run() {
    let provider = provider([
        ScriptedModelResponse::text(["done"]),
        ScriptedModelResponse::text(["done"]),
    ]);
    let runtime = build_runtime(
        provider,
        Arc::new(TestIds::default()),
        Arc::new(TestSessionIds::default()),
    )
    .unwrap();
    let session_id = match runtime
        .send(envelope(
            AgentCommand::CreateSession {
                profile_id: "coding-agent".parse().unwrap(),
                metadata: ProtocolMetadata::default(),
            },
            None,
        ))
        .await
        .unwrap()
    {
        RuntimeCommandOutcome::Created { session_id } => session_id,
        other => panic!("expected Created, got {other:?}"),
    };

    // No active run → abort fails.
    let err = runtime
        .send(envelope(AgentCommand::Abort {}, Some(session_id)))
        .await
        .unwrap_err();
    assert_eq!(err.code(), RuntimeErrorCode::NoActiveRun);

    // Run to completion, then a second prompt must succeed (no lingering active run).
    let _completed = runtime
        .send(envelope(
            AgentCommand::Prompt {
                message: user_message("hello"),
                final_output_format: None,
            },
            Some(session_id),
        ))
        .await
        .unwrap();
    let _second = runtime
        .send(envelope(
            AgentCommand::Prompt {
                message: user_message("again"),
                final_output_format: None,
            },
            Some(session_id),
        ))
        .await
        .unwrap();
}
