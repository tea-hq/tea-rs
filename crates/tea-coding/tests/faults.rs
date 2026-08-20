use std::fs;
use std::str::FromStr as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use serde_json::Value;
use tea_coding::config::CodingSettings;
use tea_coding::resources::ResourceCatalog;
use tea_coding::{CodingAgentBuilder, CodingAgentService, CodingErrorCode, ProjectAccess};
use tea_coding_tools::{BashConfig, BashOutputDirectory, BashShell, WorkspaceRoot};
use tea_model::{
    ModelCapabilities, ModelCompletion, ModelDisplayName, ModelEvent, ModelFailure,
    ModelFailureCode, ModelResponseInfo, ModelSpec, ModelStreamIndex, ProviderId,
    ProviderToolCallId, ToolCallCompleted, ToolCallStarted, Utf8Delta,
};
use tea_policy::{ActorId, PolicyGrant, WorkspaceId};
use tea_protocol::{
    ApprovalDecision, MessageRole, ModelId, RetryClass, SessionId, SessionRecord, StopReason,
    TokenCount,
};
use tea_session::{
    AppendOutcome, AppendTransaction, InMemorySessionStore, SessionArchive, SessionCatalog,
    SessionCatalogEntry, SessionName, SessionSnapshot, SessionStore, SessionStoreError,
    SessionStoreErrorCode, SessionStoreFuture,
};
use tea_testkit::{ScriptStep, ScriptedModelProvider, ScriptedModelResponse};

#[derive(Debug, Default)]
struct FailingStore {
    inner: InMemorySessionStore,
    fail_append: AtomicBool,
}

impl FailingStore {
    fn set_append_failure(&self, fail: bool) {
        self.fail_append.store(fail, Ordering::SeqCst);
    }
}

impl SessionStore for FailingStore {
    fn load(&self, session_id: SessionId) -> SessionStoreFuture<'_, SessionSnapshot> {
        self.inner.load(session_id)
    }

    fn append(&self, transaction: AppendTransaction) -> SessionStoreFuture<'_, AppendOutcome> {
        Box::pin(async move {
            if self.fail_append.load(Ordering::SeqCst) {
                return Err(SessionStoreError::new(
                    SessionStoreErrorCode::StorageUnavailable,
                    "injected append failure",
                ));
            }
            self.inner.append(transaction).await
        })
    }

    fn active_grants_for_actor(
        &self,
        actor_id: ActorId,
    ) -> SessionStoreFuture<'_, Vec<PolicyGrant>> {
        self.inner.active_grants_for_actor(actor_id)
    }
}

impl SessionCatalog for FailingStore {
    fn list_sessions(&self) -> SessionStoreFuture<'_, Vec<SessionCatalogEntry>> {
        self.inner.list_sessions()
    }

    fn set_session_name(
        &self,
        session_id: SessionId,
        name: Option<SessionName>,
    ) -> SessionStoreFuture<'_, ()> {
        self.inner.set_session_name(session_id, name)
    }

    fn session_name(&self, session_id: SessionId) -> SessionStoreFuture<'_, Option<SessionName>> {
        self.inner.session_name(session_id)
    }
}

struct Fixture {
    root: std::path::PathBuf,
}

impl Fixture {
    fn new(label: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "tea-coding-fault-{label}-{}",
            uuid::Uuid::now_v7().hyphenated()
        ));
        fs::create_dir_all(&root).unwrap();
        Self { root }
    }

    fn service<S>(&self, provider: Arc<ScriptedModelProvider>, store: Arc<S>) -> CodingAgentService
    where
        S: SessionStore + SessionCatalog + 'static,
    {
        let workspace = WorkspaceRoot::new(&self.root).unwrap();
        let resources = ResourceCatalog::discover(
            &self.root,
            &self.root,
            ProjectAccess::Ignored,
            &[],
            &[],
            None,
            None,
        )
        .unwrap();
        let settings = CodingSettings {
            provider: "fake".to_owned(),
            model: "fake/model".to_owned(),
            max_retries: 0,
            ..CodingSettings::default()
        };
        let bash = BashConfig::new(
            test_shell(),
            BashOutputDirectory::new(&self.root).unwrap(),
            Duration::from_secs(5),
        )
        .unwrap();
        CodingAgentBuilder::new(
            provider,
            workspace,
            resources,
            store,
            bash,
            settings,
            ActorId::from_str("local:user").unwrap(),
            WorkspaceId::from_str("workspace/fault-test").unwrap(),
        )
        .build()
        .unwrap()
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[cfg(unix)]
fn test_shell() -> BashShell {
    BashShell::new("/bin/sh", "-c").unwrap()
}

#[cfg(windows)]
fn test_shell() -> BashShell {
    BashShell::new(r"C:\Windows\System32\cmd.exe", "/C").unwrap()
}

fn model() -> ModelSpec {
    ModelSpec::new(
        ModelId::from_str("fake/model").unwrap(),
        ProviderId::from_str("fake").unwrap(),
        ModelDisplayName::from_str("Fake Model").unwrap(),
        TokenCount::new(32_000).unwrap(),
        TokenCount::new(4_000).unwrap(),
        ModelCapabilities::text().with_tools(true),
    )
    .unwrap()
}

fn provider(
    scripts: impl IntoIterator<Item = ScriptedModelResponse>,
) -> Arc<ScriptedModelProvider> {
    Arc::new(ScriptedModelProvider::new(
        ProviderId::from_str("fake").unwrap(),
        vec![model()],
        scripts,
    ))
}

fn tool_response(id: &str, name: &str, arguments: Value) -> ScriptedModelResponse {
    let index = ModelStreamIndex::new(0).unwrap();
    let provider_id = ProviderToolCallId::from_str(id).unwrap();
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

fn approval_id(outcome: tea::RuntimeCommandOutcome) -> tea_protocol::ApprovalId {
    match outcome {
        tea::RuntimeCommandOutcome::RunCompleted {
            pending_approval_id: Some(approval_id),
            ..
        } => approval_id,
        other => panic!("expected approval checkpoint, got {other:?}"),
    }
}

#[tokio::test(flavor = "current_thread")]
async fn append_failure_rolls_back_and_the_service_can_retry() {
    let fixture = Fixture::new("append");
    let store = Arc::new(FailingStore::default());
    let provider = provider([ScriptedModelResponse::text(["recovered"])]);
    let service = fixture.service(Arc::clone(&provider), Arc::clone(&store));
    let session_id = service.create_session().await.unwrap();
    let baseline = service
        .session_snapshot(session_id)
        .await
        .unwrap()
        .records()
        .len();

    store.set_append_failure(true);
    service.prompt(session_id, "must roll back").unwrap();
    let error = service.wait(session_id).await.unwrap_err();
    assert!(!error.message().contains("injected append failure"));
    assert!(error.message().len() <= 4096);
    assert_eq!(
        service
            .session_snapshot(session_id)
            .await
            .unwrap()
            .records()
            .len(),
        baseline
    );
    assert_eq!(provider.remaining_scripts().unwrap(), 1);

    store.set_append_failure(false);
    service.prompt(session_id, "retry safely").unwrap();
    service.wait(session_id).await.unwrap();
    assert_eq!(provider.remaining_scripts().unwrap(), 0);
    service.shutdown().await;
}

#[tokio::test(flavor = "current_thread")]
async fn provider_failures_before_and_during_stream_are_durable_and_bounded() {
    let fixture = Fixture::new("provider");
    let before =
        ScriptedModelResponse::new([ScriptStep::event(ScriptedModelResponse::failure_event(
            ModelFailureCode::Authentication,
            "seeded-provider-secret-before",
        ))]);
    let during = ScriptedModelResponse::events([
        ModelEvent::Started(ModelResponseInfo::new()),
        ModelEvent::TextDelta(Utf8Delta::new("partial text").unwrap()),
        ScriptedModelResponse::failure_event(
            ModelFailureCode::Authentication,
            "seeded-provider-secret-during",
        ),
    ]);
    let provider = provider([before, during]);
    let service = fixture.service(provider, Arc::new(InMemorySessionStore::new()));

    for prompt in ["fail before stream", "fail during stream"] {
        let session_id = service.create_session().await.unwrap();
        service.prompt(session_id, prompt).unwrap();
        let error = service.wait(session_id).await.unwrap_err();
        assert!(!format!("{error:?}").contains("seeded-provider-secret"));
        let snapshot = service.session_snapshot(session_id).await.unwrap();
        let archive = SessionArchive::from_snapshot(&snapshot).unwrap();
        assert!(
            !serde_json::to_string(&archive)
                .unwrap()
                .contains("seeded-provider-secret")
        );
        let reason = match snapshot.records().last().unwrap().record() {
            SessionRecord::RunInterrupted { reason, .. } => reason,
            record => panic!("expected interrupted run, got {record:?}"),
        };
        assert!(!reason.contains("seeded-provider-secret"));
        assert!(reason.len() <= 4_096);
        assert!(!snapshot.records().iter().any(|record| matches!(
            record.record(),
            SessionRecord::MessageCommitted { message }
                if message.role() == MessageRole::Assistant
        )));
    }
    service.shutdown().await;
}

#[tokio::test(flavor = "current_thread")]
async fn wait_preserves_provider_neutral_terminal_failure_codes() {
    let fixture = Fixture::new("provider-failure-codes");
    let cases = [
        (ModelFailureCode::RateLimited, CodingErrorCode::RateLimited),
        (
            ModelFailureCode::Authentication,
            CodingErrorCode::Authentication,
        ),
        (
            ModelFailureCode::PermissionDenied,
            CodingErrorCode::PermissionDenied,
        ),
        (
            ModelFailureCode::ContextOverflow,
            CodingErrorCode::ContextOverflow,
        ),
        (ModelFailureCode::Unavailable, CodingErrorCode::Unavailable),
        (ModelFailureCode::Transport, CodingErrorCode::Transport),
        (
            ModelFailureCode::InvalidRequest,
            CodingErrorCode::InvalidRequest,
        ),
        (ModelFailureCode::Cancelled, CodingErrorCode::Cancelled),
        (ModelFailureCode::Internal, CodingErrorCode::Internal),
        (
            ModelFailureCode::MalformedResponse,
            CodingErrorCode::Internal,
        ),
    ];
    let provider = provider(cases.map(|(source, _)| {
        let retry = match source {
            ModelFailureCode::RateLimited | ModelFailureCode::Unavailable => {
                RetryClass::AfterBackoff
            }
            ModelFailureCode::Transport => RetryClass::Immediate,
            _ => RetryClass::Never,
        };
        let failure =
            ModelFailure::safe(source, format!("safe {source:?} diagnostic"), retry).unwrap();
        ScriptedModelResponse::events([
            ModelEvent::Started(ModelResponseInfo::new()),
            ModelEvent::Failed(failure),
        ])
    }));
    let service = fixture.service(provider, Arc::new(InMemorySessionStore::new()));

    for (source, expected) in cases {
        let session_id = service.create_session().await.unwrap();
        service
            .prompt(session_id, format!("exercise {source:?}"))
            .unwrap();
        let error = service.wait(session_id).await.unwrap_err();
        assert_eq!(error.code(), expected, "source code: {source:?}");
        if source == ModelFailureCode::RateLimited {
            assert_eq!(
                error.message(),
                "model retry policy was exhausted: safe RateLimited diagnostic"
            );
        }
    }

    service.shutdown().await;
}

#[cfg(unix)]
#[tokio::test(flavor = "current_thread")]
async fn tool_failures_before_and_after_a_side_effect_are_not_retried() {
    let fixture = Fixture::new("tools");
    fs::write(fixture.root.join("file.txt"), "original\n").unwrap();
    let provider = provider([
        tool_response(
            "edit-fail",
            "edit",
            serde_json::json!({
                "path":"file.txt","oldText":"missing","newText":"changed"
            }),
        ),
        ScriptedModelResponse::text(["edit failure observed"]),
        tool_response(
            "bash-fail",
            "bash",
            serde_json::json!({"command":"printf x >> marker; exit 9"}),
        ),
        ScriptedModelResponse::text(["bash failure observed"]),
    ]);
    let service = fixture.service(Arc::clone(&provider), Arc::new(InMemorySessionStore::new()));

    let edit_session = service.create_session().await.unwrap();
    service.prompt(edit_session, "edit").unwrap();
    let edit_approval = approval_id(service.wait(edit_session).await.unwrap());
    service
        .approve(edit_session, edit_approval, ApprovalDecision::AllowOnce)
        .unwrap();
    service.wait(edit_session).await.unwrap();
    assert_eq!(
        fs::read_to_string(fixture.root.join("file.txt")).unwrap(),
        "original\n"
    );

    let bash_session = service.create_session().await.unwrap();
    service.prompt(bash_session, "bash").unwrap();
    let bash_approval = approval_id(service.wait(bash_session).await.unwrap());
    service
        .approve(bash_session, bash_approval, ApprovalDecision::AllowOnce)
        .unwrap();
    service.wait(bash_session).await.unwrap();
    assert_eq!(fs::read(fixture.root.join("marker")).unwrap(), b"x");
    assert_eq!(provider.remaining_scripts().unwrap(), 0);
    service.shutdown().await;
}

#[tokio::test(flavor = "current_thread")]
async fn event_sink_backpressure_cannot_block_service_shutdown() {
    let fixture = Fixture::new("backpressure");
    let mut events = vec![ModelEvent::Started(ModelResponseInfo::new())];
    events.extend((0..400).map(|_| ModelEvent::TextDelta(Utf8Delta::new("x").unwrap())));
    events.push(ModelEvent::Completed(
        ModelCompletion::new(StopReason::Completed).unwrap(),
    ));
    let provider = provider([ScriptedModelResponse::events(events)]);
    let service = fixture.service(Arc::clone(&provider), Arc::new(InMemorySessionStore::new()));
    let session_id = service.create_session().await.unwrap();
    let receiver = service.subscribe(session_id).unwrap();
    service.prompt(session_id, "fill subscriber").unwrap();
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if provider.captured_requests().unwrap().len() == 1 && receiver.len() == 256 {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("event channel should reach its documented bound");

    tokio::time::timeout(Duration::from_secs(1), service.shutdown())
        .await
        .expect("shutdown must cancel a run blocked on event backpressure");
}

#[tokio::test(flavor = "current_thread")]
async fn abrupt_owner_drop_allows_reopen_and_a_new_run() {
    let fixture = Fixture::new("restart");
    let store = Arc::new(InMemorySessionStore::new());
    let first_provider = provider([ScriptedModelResponse::await_cancellation()]);
    let service = fixture.service(Arc::clone(&first_provider), Arc::clone(&store));
    let session_id = service.create_session().await.unwrap();
    service.prompt(session_id, "interrupted prompt").unwrap();
    tokio::time::timeout(Duration::from_secs(1), async {
        loop {
            if first_provider.captured_requests().unwrap().len() == 1 {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    drop(service);

    let rebuilt = fixture.service(
        provider([ScriptedModelResponse::text(["after restart"])]),
        store,
    );
    rebuilt.open_session(session_id).await.unwrap();
    rebuilt.prompt(session_id, "continue").unwrap();
    rebuilt.wait(session_id).await.unwrap();
    let snapshot = rebuilt.session_snapshot(session_id).await.unwrap();
    assert!(snapshot.records().iter().any(|record| matches!(
        record.record(),
        SessionRecord::MessageCommitted { message }
            if message.role() == MessageRole::Assistant
    )));
    rebuilt.shutdown().await;
}
