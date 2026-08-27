use crate::common;

use std::str::FromStr;
use std::sync::{Arc, Mutex};

use common::{FixedClock, TestIds, TestSessionIds, build_runtime, runtime_builder, user_message};
use tea::{RuntimeCommandOutcome, RuntimeErrorCode};
use tea_context::{
    BudgetBehavior, CacheScope, ContextProvider, ContextProviderFuture, ContextProviderId,
    ContextRequest, PromptAuthority, PromptModule, PromptModuleId, PromptPriority,
    PromptProvenance, PromptSegment, PromptSegmentId, TrustLevel,
};
use tea_model::{ModelCapabilities, ModelDisplayName, ModelSpec, ProviderId};
use tea_protocol::{
    AgentCommand, BranchId, CommandEnvelope, CommandId, ModelRef, ProtocolMetadata,
    ProtocolTimestamp, ReasoningEffort, TokenCount,
};
use tea_session::InMemorySessionStore;
use tea_testkit::{ScriptedModelProvider, ScriptedModelResponse};
use tokio::sync::Notify;

fn envelope(command: AgentCommand, session_id: Option<tea_protocol::SessionId>) -> CommandEnvelope {
    CommandEnvelope::new(
        CommandId::from_str("0195a0b1-0000-7000-8000-000000000001").unwrap(),
        session_id,
        ProtocolTimestamp::from_str(common::NOW).unwrap(),
        command,
    )
    .unwrap()
}

fn runtime() -> tea::AgentRuntime {
    build_runtime(
        common::provider(),
        Arc::new(TestIds::default()),
        Arc::new(TestSessionIds::default()),
    )
    .unwrap()
}

fn model_ref(model_id: &str) -> ModelRef {
    ModelRef::new("fake".parse().unwrap(), model_id.parse().unwrap())
}

fn reasoning_provider(
    scripts: impl IntoIterator<Item = ScriptedModelResponse>,
) -> Arc<ScriptedModelProvider> {
    let provider_id = ProviderId::from_str("fake").unwrap();
    let model = ModelSpec::new(
        "fake/model".parse().unwrap(),
        provider_id.clone(),
        ModelDisplayName::from_str("Fake Reasoning Model").unwrap(),
        TokenCount::new(32_000).unwrap(),
        TokenCount::new(4_000).unwrap(),
        ModelCapabilities::text().with_tools(true).with_reasoning(),
    )
    .unwrap();
    Arc::new(ScriptedModelProvider::new(
        provider_id,
        vec![model],
        scripts,
    ))
}

#[derive(Debug)]
struct TurnPromptProvider {
    id: ContextProviderId,
    requests: Mutex<Vec<ContextRequest>>,
}

impl TurnPromptProvider {
    fn new() -> Self {
        Self {
            id: "test.turn_prompt".parse().unwrap(),
            requests: Mutex::new(Vec::new()),
        }
    }

    fn requests(&self) -> Vec<ContextRequest> {
        self.requests.lock().unwrap().clone()
    }
}

impl ContextProvider for TurnPromptProvider {
    fn id(&self) -> &ContextProviderId {
        &self.id
    }

    fn provide(&self, request: ContextRequest) -> ContextProviderFuture<'_> {
        self.requests.lock().unwrap().push(request.clone());
        let run_id = request.run_id().expect("prompt turns have a run id");
        let content = format!("PRIVATE-TURN-CONTENT-{run_id}");
        let segment = PromptSegment::new(
            PromptSegmentId::from_str("extension.turn").unwrap(),
            content,
            PromptProvenance::new(
                self.id.clone(),
                "turn_extension",
                Some("extension:turn".to_owned()),
            )
            .unwrap(),
            TrustLevel::Delegated,
            CacheScope::Run,
            BudgetBehavior::Omit,
        )
        .unwrap();
        let module = PromptModule::new(
            PromptModuleId::from_str("extension.turn").unwrap(),
            PromptAuthority::UserAddition,
            PromptPriority::new(0),
            vec![segment],
        )
        .unwrap();
        Box::pin(async move { Ok(vec![module]) })
    }
}

#[derive(Debug)]
struct GatedContextProvider {
    id: ContextProviderId,
    entered: Arc<Notify>,
    release: Arc<Notify>,
}

impl GatedContextProvider {
    fn new() -> Self {
        Self {
            id: "test.gated_prompt".parse().unwrap(),
            entered: Arc::new(Notify::new()),
            release: Arc::new(Notify::new()),
        }
    }
}

impl ContextProvider for GatedContextProvider {
    fn id(&self) -> &ContextProviderId {
        &self.id
    }

    fn provide(&self, _request: ContextRequest) -> ContextProviderFuture<'_> {
        let entered = Arc::clone(&self.entered);
        let release = Arc::clone(&self.release);
        Box::pin(async move {
            entered.notify_one();
            release.notified().await;
            Ok(Vec::new())
        })
    }
}

#[tokio::test]
async fn context_provider_is_recomputed_per_turn_and_inspection_excludes_content() {
    let provider = common::provider_with([
        ScriptedModelResponse::text(["first response"]),
        ScriptedModelResponse::text(["second response"]),
    ]);
    let extension = Arc::new(TurnPromptProvider::new());
    let runtime = runtime_builder(
        Arc::clone(&provider),
        Arc::new(TestIds::default()),
        Arc::new(TestSessionIds::default()),
    )
    .unwrap()
    .context_provider(extension.clone())
    .build()
    .unwrap();
    let session_id = create_session(&runtime, "coding-agent").await;
    assert!(runtime.prompt_inspection(session_id).unwrap().is_none());

    runtime
        .send(envelope(
            AgentCommand::Prompt {
                message: user_message("first"),
            },
            Some(session_id),
        ))
        .await
        .unwrap();
    let first_inspection = runtime.prompt_inspection(session_id).unwrap().unwrap();

    runtime
        .send(envelope(
            AgentCommand::Prompt {
                message: user_message("second"),
            },
            Some(session_id),
        ))
        .await
        .unwrap();
    let second_inspection = runtime.prompt_inspection(session_id).unwrap().unwrap();

    let requests = extension.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].session_id(), session_id);
    assert_eq!(requests[1].session_id(), session_id);
    assert_ne!(requests[0].run_id(), requests[1].run_id());
    assert_eq!(first_inspection.run_id(), requests[0].run_id());
    assert_eq!(second_inspection.run_id(), requests[1].run_id());

    let model_requests = provider.captured_requests().unwrap();
    assert_eq!(model_requests.len(), 2);
    for (request, context_request) in model_requests.iter().zip(&requests) {
        let marker = format!("PRIVATE-TURN-CONTENT-{}", context_request.run_id().unwrap());
        assert!(request.system_prompt().unwrap().contains(&marker));
    }
    let extension_segment = second_inspection
        .prompt()
        .segments()
        .iter()
        .find(|entry| entry.module_id().as_str() == "extension.turn")
        .unwrap();
    assert_eq!(extension_segment.authority(), PromptAuthority::UserAddition);
    assert_eq!(extension_segment.trust(), TrustLevel::Delegated);
    assert_eq!(extension_segment.cache_scope(), CacheScope::Run);
    assert_eq!(
        extension_segment.provenance().locator(),
        Some("extension:turn")
    );
    let encoded = serde_json::to_string(&second_inspection).unwrap();
    assert!(!encoded.contains("PRIVATE-TURN-CONTENT"));
    assert!(!encoded.contains("byteRange"));

    let durable = runtime.snapshot(session_id).await.unwrap();
    assert!(!format!("{durable:?}").contains("PRIVATE-TURN-CONTENT"));
}

#[tokio::test]
async fn create_session_appends_records_and_tracks_session() {
    let runtime = runtime();
    let outcome = runtime
        .send(envelope(
            AgentCommand::CreateSession {
                profile_id: "coding-agent".parse().unwrap(),
                metadata: ProtocolMetadata::default(),
            },
            None,
        ))
        .await
        .unwrap();
    let session_id = match outcome {
        RuntimeCommandOutcome::Created { session_id } => session_id,
        other => panic!("expected Created, got {other:?}"),
    };
    let snapshot = runtime.snapshot(session_id).await.unwrap();
    assert_eq!(
        snapshot.state().configuration().profile_id().as_str(),
        "coding-agent"
    );
    assert_eq!(
        snapshot.state().configuration().model_id(),
        Some(&"fake/model".parse().unwrap())
    );
    assert_eq!(snapshot.records().len(), 2);
    let expected_root = BranchId::from_str(&session_id.to_string()).unwrap();
    assert_eq!(snapshot.state().active_branch_id(), Some(expected_root));
    assert!(
        snapshot
            .records()
            .iter()
            .all(|record| record.branch_id() == Some(expected_root))
    );
    assert_eq!(runtime.health().session_count(), 1);
}

#[tokio::test]
async fn create_session_rejects_unknown_profile() {
    let runtime = runtime();
    let err = runtime
        .send(envelope(
            AgentCommand::CreateSession {
                profile_id: "ghost".parse().unwrap(),
                metadata: ProtocolMetadata::default(),
            },
            None,
        ))
        .await
        .unwrap_err();
    assert_eq!(err.code(), RuntimeErrorCode::UnknownProfile);
}

#[tokio::test]
async fn set_model_appends_configuration_change() {
    let runtime = runtime();
    let session_id = create_session(&runtime, "coding-agent").await;
    let outcome = runtime
        .send(envelope(
            AgentCommand::SetModel {
                model: model_ref("fake/model"),
            },
            Some(session_id),
        ))
        .await
        .unwrap();
    match outcome {
        RuntimeCommandOutcome::ConfigurationChanged {
            model, profile_id, ..
        } => {
            assert_eq!(model.as_ref().unwrap(), &model_ref("fake/model"));
            assert!(profile_id.is_none());
        }
        other => panic!("expected ConfigurationChanged, got {other:?}"),
    }
    let snapshot = runtime.snapshot(session_id).await.unwrap();
    assert_eq!(snapshot.records().len(), 3);
}

#[tokio::test]
async fn providerless_session_keeps_its_model_until_the_provider_is_registered() {
    let provider = common::provider_with([ScriptedModelResponse::text(["registered later"])]);
    let runtime = build_runtime(
        Arc::clone(&provider),
        Arc::new(TestIds::default()),
        Arc::new(TestSessionIds::default()),
    )
    .unwrap();
    runtime
        .remove_model_providers([ProviderId::from_str("fake").unwrap()])
        .unwrap();

    let session_id = create_session(&runtime, "coding-agent").await;
    let selected = runtime
        .snapshot(session_id)
        .await
        .unwrap()
        .state()
        .configuration()
        .model_ref()
        .cloned();
    assert_eq!(selected, Some(model_ref("fake/model")));

    let error = runtime
        .send(envelope(
            AgentCommand::Prompt {
                message: user_message("before registration"),
            },
            Some(session_id),
        ))
        .await
        .unwrap_err();
    assert_eq!(error.code(), RuntimeErrorCode::UnknownProvider);

    runtime
        .register_model_providers([Arc::clone(&provider) as Arc<dyn tea_model::ModelProvider>])
        .unwrap();
    assert_eq!(
        runtime
            .snapshot(session_id)
            .await
            .unwrap()
            .state()
            .configuration()
            .model_ref(),
        Some(&model_ref("fake/model"))
    );
    runtime
        .send(envelope(
            AgentCommand::Prompt {
                message: user_message("after registration"),
            },
            Some(session_id),
        ))
        .await
        .unwrap();
    assert_eq!(provider.captured_requests().unwrap().len(), 1);
}

#[tokio::test]
async fn registration_is_atomic_and_does_not_change_session_selection() {
    let runtime = runtime();
    let session_id = create_session(&runtime, "coding-agent").await;
    let alternate_id = ProviderId::from_str("alternate").unwrap();
    let alternate_model = ModelSpec::new(
        "alternate/model".parse().unwrap(),
        alternate_id.clone(),
        ModelDisplayName::from_str("Alternate Model").unwrap(),
        TokenCount::new(32_000).unwrap(),
        TokenCount::new(4_000).unwrap(),
        ModelCapabilities::text().with_tools(true),
    )
    .unwrap();
    let alternate = Arc::new(ScriptedModelProvider::new(
        alternate_id.clone(),
        vec![alternate_model],
        [],
    ));

    runtime
        .register_model_providers([Arc::clone(&alternate) as Arc<dyn tea_model::ModelProvider>])
        .unwrap();
    assert_eq!(
        runtime
            .snapshot(session_id)
            .await
            .unwrap()
            .state()
            .configuration()
            .model_ref(),
        Some(&model_ref("fake/model"))
    );

    let error = runtime
        .register_model_providers([alternate as Arc<dyn tea_model::ModelProvider>])
        .unwrap_err();
    assert_eq!(error.code(), RuntimeErrorCode::DuplicateEntry);
    assert_eq!(runtime.model_registry().provider_count(), 2);
}

#[tokio::test]
async fn active_run_finishes_against_its_starting_generation_after_removal() {
    let provider = common::provider_with([ScriptedModelResponse::text(["old generation"])]);
    let gate = Arc::new(GatedContextProvider::new());
    let runtime = Arc::new(
        runtime_builder(
            Arc::clone(&provider),
            Arc::new(TestIds::default()),
            Arc::new(TestSessionIds::default()),
        )
        .unwrap()
        .context_provider(gate.clone())
        .build()
        .unwrap(),
    );
    let session_id = create_session(&runtime, "coding-agent").await;
    let running = {
        let runtime = Arc::clone(&runtime);
        tokio::spawn(async move {
            runtime
                .send(envelope(
                    AgentCommand::Prompt {
                        message: user_message("use the captured generation"),
                    },
                    Some(session_id),
                ))
                .await
        })
    };

    gate.entered.notified().await;
    runtime
        .remove_model_providers([ProviderId::from_str("fake").unwrap()])
        .unwrap();
    assert!(runtime.models().is_empty());
    gate.release.notify_one();

    running.await.unwrap().unwrap();
    assert_eq!(provider.captured_requests().unwrap().len(), 1);

    let error = runtime
        .send(envelope(
            AgentCommand::Prompt {
                message: user_message("future run"),
            },
            Some(session_id),
        ))
        .await
        .unwrap_err();
    assert_eq!(error.code(), RuntimeErrorCode::UnknownProvider);
}

#[tokio::test]
async fn set_model_rejects_unknown_model() {
    let runtime = runtime();
    let session_id = create_session(&runtime, "coding-agent").await;
    let err = runtime
        .send(envelope(
            AgentCommand::SetModel {
                model: model_ref("fake/missing"),
            },
            Some(session_id),
        ))
        .await
        .unwrap_err();
    assert_eq!(err.code(), RuntimeErrorCode::UnknownModel);
}

#[tokio::test]
async fn set_model_rejects_unknown_provider_distinctly() {
    let runtime = runtime();
    let session_id = create_session(&runtime, "coding-agent").await;
    let err = runtime
        .send(envelope(
            AgentCommand::SetModel {
                model: ModelRef::new(
                    "missing-provider".parse().unwrap(),
                    "fake/model".parse().unwrap(),
                ),
            },
            Some(session_id),
        ))
        .await
        .unwrap_err();
    assert_eq!(err.code(), RuntimeErrorCode::UnknownProvider);
}

#[tokio::test]
async fn set_model_routes_same_model_id_to_the_selected_provider() {
    let primary = common::provider();
    let alternate_id = ProviderId::from_str("alternate").unwrap();
    let alternate_model = ModelSpec::new(
        "fake/model".parse().unwrap(),
        alternate_id.clone(),
        ModelDisplayName::from_str("Alternate Model").unwrap(),
        TokenCount::new(32_000).unwrap(),
        TokenCount::new(4_000).unwrap(),
        ModelCapabilities::text().with_tools(true),
    )
    .unwrap();
    let alternate = Arc::new(ScriptedModelProvider::new(
        alternate_id.clone(),
        vec![alternate_model],
        [ScriptedModelResponse::text(["alternate response"])],
    ));
    let runtime = runtime_builder(
        Arc::clone(&primary),
        Arc::new(TestIds::default()),
        Arc::new(TestSessionIds::default()),
    )
    .unwrap()
    .provider(Arc::clone(&alternate) as Arc<dyn tea_model::ModelProvider>)
    .build()
    .unwrap();
    let session_id = create_session(&runtime, "coding-agent").await;
    let alternate_ref = ModelRef::new(alternate_id.clone(), "fake/model".parse().unwrap());

    runtime
        .send(envelope(
            AgentCommand::SetModel {
                model: alternate_ref.clone(),
            },
            Some(session_id),
        ))
        .await
        .unwrap();
    runtime
        .send(envelope(
            AgentCommand::Prompt {
                message: user_message("route this request"),
            },
            Some(session_id),
        ))
        .await
        .unwrap();

    assert!(primary.captured_requests().unwrap().is_empty());
    assert_eq!(alternate.captured_requests().unwrap().len(), 1);
    assert_eq!(
        runtime
            .snapshot(session_id)
            .await
            .unwrap()
            .state()
            .configuration()
            .model_ref(),
        Some(&alternate_ref)
    );
}

#[tokio::test]
async fn reopened_session_routes_by_persisted_provider_identity() {
    let store: Arc<dyn tea_session::SessionStore> = Arc::new(InMemorySessionStore::new());
    let alternate_id = ProviderId::from_str("alternate").unwrap();
    let alternate_provider = |scripts: Vec<ScriptedModelResponse>| {
        let model = ModelSpec::new(
            "fake/model".parse().unwrap(),
            alternate_id.clone(),
            ModelDisplayName::from_str("Alternate Model").unwrap(),
            TokenCount::new(32_000).unwrap(),
            TokenCount::new(4_000).unwrap(),
            ModelCapabilities::text().with_tools(true),
        )
        .unwrap();
        Arc::new(ScriptedModelProvider::new(
            alternate_id.clone(),
            vec![model],
            scripts,
        ))
    };
    let alternate = alternate_provider(Vec::new());
    let ids = Arc::new(TestIds::default());
    let first = runtime_builder(
        common::provider(),
        Arc::clone(&ids),
        Arc::new(TestSessionIds::default()),
    )
    .unwrap()
    .session_store(Arc::clone(&store))
    .provider(Arc::clone(&alternate) as Arc<dyn tea_model::ModelProvider>)
    .build()
    .unwrap();
    let session_id = create_session(&first, "coding-agent").await;
    let alternate_ref = ModelRef::new(alternate_id.clone(), "fake/model".parse().unwrap());
    first
        .send(envelope(
            AgentCommand::SetModel {
                model: alternate_ref.clone(),
            },
            Some(session_id),
        ))
        .await
        .unwrap();
    drop(first);

    let missing_provider = runtime_builder(
        common::provider(),
        Arc::clone(&ids),
        Arc::new(TestSessionIds::default()),
    )
    .unwrap()
    .session_store(Arc::clone(&store))
    .build()
    .unwrap();
    let state = missing_provider.attach_session(session_id).await.unwrap();
    assert_eq!(state.model_ref(), Some(&alternate_ref));
    let error = missing_provider
        .send(envelope(
            AgentCommand::Prompt {
                message: user_message("provider is temporarily unavailable"),
            },
            Some(session_id),
        ))
        .await
        .unwrap_err();
    assert_eq!(error.code(), RuntimeErrorCode::UnknownProvider);
    drop(missing_provider);

    let reopened_alternate = alternate_provider(vec![ScriptedModelResponse::text(["reopened"])]);
    let reopened = runtime_builder(common::provider(), ids, Arc::new(TestSessionIds::default()))
        .unwrap()
        .session_store(store)
        .provider(Arc::clone(&reopened_alternate) as Arc<dyn tea_model::ModelProvider>)
        .build()
        .unwrap();
    let state = reopened.attach_session(session_id).await.unwrap();
    assert_eq!(state.model_ref(), Some(&alternate_ref));
    reopened
        .send(envelope(
            AgentCommand::Prompt {
                message: user_message("continue after reopen"),
            },
            Some(session_id),
        ))
        .await
        .unwrap();

    assert_eq!(reopened_alternate.captured_requests().unwrap().len(), 1);
}

#[tokio::test]
async fn set_reasoning_effort_clamps_and_persists_the_effective_level() {
    let provider = reasoning_provider([]);
    let runtime = build_runtime(
        provider,
        Arc::new(TestIds::default()),
        Arc::new(TestSessionIds::default()),
    )
    .unwrap();
    let session_id = create_session(&runtime, "coding-agent").await;
    let outcome = runtime
        .send(envelope(
            AgentCommand::SetReasoningEffort {
                reasoning_effort: ReasoningEffort::ExtraHigh,
            },
            Some(session_id),
        ))
        .await
        .unwrap();

    match outcome {
        RuntimeCommandOutcome::ConfigurationChanged {
            reasoning_effort,
            requested_reasoning_effort,
            ..
        } => {
            assert_eq!(reasoning_effort, Some(ReasoningEffort::High));
            assert_eq!(requested_reasoning_effort, Some(ReasoningEffort::ExtraHigh));
        }
        other => panic!("expected ConfigurationChanged, got {other:?}"),
    }
    let snapshot = runtime.snapshot(session_id).await.unwrap();
    assert_eq!(
        snapshot.state().configuration().reasoning_effort(),
        Some(ReasoningEffort::High)
    );
}

#[tokio::test]
async fn reasoning_request_on_non_reasoning_model_resolves_to_off() {
    let runtime = runtime();
    let session_id = create_session(&runtime, "coding-agent").await;
    let outcome = runtime
        .send(envelope(
            AgentCommand::SetReasoningEffort {
                reasoning_effort: ReasoningEffort::High,
            },
            Some(session_id),
        ))
        .await
        .unwrap();

    assert!(matches!(
        outcome,
        RuntimeCommandOutcome::ConfigurationChanged {
            reasoning_effort: Some(ReasoningEffort::Off),
            requested_reasoning_effort: Some(ReasoningEffort::High),
            ..
        }
    ));
    assert_eq!(
        runtime
            .snapshot(session_id)
            .await
            .unwrap()
            .state()
            .configuration()
            .reasoning_effort(),
        Some(ReasoningEffort::Off)
    );
}

#[tokio::test]
async fn new_session_captures_the_resolved_runtime_default() {
    let runtime = runtime_builder(
        reasoning_provider([]),
        Arc::new(TestIds::default()),
        Arc::new(TestSessionIds::default()),
    )
    .unwrap()
    .default_reasoning_effort(ReasoningEffort::ExtraHigh)
    .build()
    .unwrap();
    let session_id = create_session(&runtime, "coding-agent").await;

    assert_eq!(
        runtime
            .snapshot(session_id)
            .await
            .unwrap()
            .state()
            .configuration()
            .reasoning_effort(),
        Some(ReasoningEffort::High)
    );
}

#[tokio::test]
async fn reasoning_change_is_rejected_while_a_run_is_active() {
    let provider = reasoning_provider([ScriptedModelResponse::await_cancellation()]);
    let runtime = Arc::new(
        build_runtime(
            Arc::clone(&provider),
            Arc::new(TestIds::default()),
            Arc::new(TestSessionIds::default()),
        )
        .unwrap(),
    );
    let session_id = create_session(&runtime, "coding-agent").await;
    let running = {
        let runtime = Arc::clone(&runtime);
        tokio::spawn(async move {
            runtime
                .send(envelope(
                    AgentCommand::Prompt {
                        message: user_message("keep running"),
                    },
                    Some(session_id),
                ))
                .await
        })
    };

    for _ in 0..100 {
        if !provider.captured_requests().unwrap().is_empty() {
            break;
        }
        tokio::task::yield_now().await;
    }
    assert_eq!(provider.captured_requests().unwrap().len(), 1);
    let error = runtime
        .send(envelope(
            AgentCommand::SetReasoningEffort {
                reasoning_effort: ReasoningEffort::High,
            },
            Some(session_id),
        ))
        .await
        .unwrap_err();
    assert_eq!(error.code(), RuntimeErrorCode::RunAlreadyActive);

    runtime
        .send(envelope(AgentCommand::Abort {}, Some(session_id)))
        .await
        .unwrap();
    let _ = running.await.unwrap();
}

#[tokio::test]
async fn set_model_rejects_a_model_that_cannot_route_active_tools_without_committing() {
    let provider_id = ProviderId::from_str("fake").unwrap();
    let model = |id: &str, capabilities| {
        ModelSpec::new(
            id.parse().unwrap(),
            provider_id.clone(),
            ModelDisplayName::from_str(id).unwrap(),
            TokenCount::new(32_000).unwrap(),
            TokenCount::new(4_000).unwrap(),
            capabilities,
        )
        .unwrap()
    };
    let provider = Arc::new(ScriptedModelProvider::new(
        provider_id.clone(),
        vec![
            model("fake/model", ModelCapabilities::text().with_tools(true)),
            model("fake/text-only", ModelCapabilities::text()),
        ],
        [],
    ));
    let runtime = build_runtime(
        provider,
        Arc::new(TestIds::default()),
        Arc::new(TestSessionIds::default()),
    )
    .unwrap();
    let session_id = create_session(&runtime, "coding-agent").await;
    let error = runtime
        .send(envelope(
            AgentCommand::SetModel {
                model: model_ref("fake/text-only"),
            },
            Some(session_id),
        ))
        .await
        .unwrap_err();
    assert_eq!(error.code(), RuntimeErrorCode::InvalidRequest);
    assert_eq!(
        error.message(),
        "active tool read_file has no execution route supported by selected model fake/text-only; declare the model capability or configure a supported client route"
    );
    let snapshot = runtime.snapshot(session_id).await.unwrap();
    assert_eq!(snapshot.records().len(), 2);
    assert_eq!(
        snapshot.state().configuration().model_id(),
        Some(&"fake/model".parse().unwrap())
    );
}

#[tokio::test]
async fn set_profile_appends_configuration_change() {
    let runtime = runtime();
    let session_id = create_session(&runtime, "coding-agent").await;
    let outcome = runtime
        .send(envelope(
            AgentCommand::SetProfile {
                profile_id: "desktop-assistant".parse().unwrap(),
            },
            Some(session_id),
        ))
        .await
        .unwrap();
    match outcome {
        RuntimeCommandOutcome::ConfigurationChanged { profile_id, .. } => {
            assert_eq!(profile_id.unwrap().as_str(), "desktop-assistant");
        }
        other => panic!("expected ConfigurationChanged, got {other:?}"),
    }
    let snapshot = runtime.snapshot(session_id).await.unwrap();
    assert_eq!(
        snapshot.state().configuration().profile_id().as_str(),
        "desktop-assistant"
    );
}

#[tokio::test]
async fn fork_rejects_an_unknown_source_message() {
    let runtime = runtime();
    let session_id = create_session(&runtime, "coding-agent").await;
    let err = runtime
        .send(envelope(
            AgentCommand::ForkSession {
                from_message_id: "0195a0b1-5e3a-7000-8000-000000000099".parse().unwrap(),
                branch_id: "0195a0b1-5e3a-7000-8000-000000000098".parse().unwrap(),
            },
            Some(session_id),
        ))
        .await
        .unwrap_err();
    assert_eq!(err.code(), RuntimeErrorCode::InvalidRequest);
}

#[tokio::test]
async fn create_session_emits_no_events_without_subscribers() {
    let runtime = runtime();
    // Subscribe before creating to confirm no events block and the session is
    // created cleanly even though the kernel has not run yet.
    let _receiver = runtime.subscribe("0195a0b1-5e3a-7000-8000-000000000002".parse().unwrap());
    let outcome = runtime
        .send(envelope(
            AgentCommand::CreateSession {
                profile_id: "coding-agent".parse().unwrap(),
                metadata: ProtocolMetadata::default(),
            },
            None,
        ))
        .await
        .unwrap();
    assert!(matches!(outcome, RuntimeCommandOutcome::Created { .. }));
}

async fn create_session(runtime: &tea::AgentRuntime, profile: &str) -> tea_protocol::SessionId {
    match runtime
        .send(envelope(
            AgentCommand::CreateSession {
                profile_id: profile.parse().unwrap(),
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

// Keep FixedClock reachable for future task wiring without unused warnings.
#[allow(dead_code)]
fn _clock() -> FixedClock {
    FixedClock
}
