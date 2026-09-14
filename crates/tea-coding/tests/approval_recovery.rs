use std::fs;
use std::str::FromStr as _;
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;
use tea_coding::config::{CodingSettings, SettingsLayer, merge_settings};
use tea_coding::resources::ResourceCatalog;
use tea_coding::{CodingAgentBuilder, CodingAgentService, ProjectAccess};
use tea_coding_tools::{BashConfig, BashOutputDirectory, BashShell, WorkspaceRoot};
use tea_model::{
    ModelCapabilities, ModelCompletion, ModelDisplayName, ModelEvent, ModelResponseInfo, ModelSpec,
    ModelStreamIndex, ProviderId, ProviderToolCallId, ToolCallCompleted, ToolCallStarted,
};
use tea_policy::{ActorId, GrantScope, WorkspaceId};
use tea_protocol::{ApprovalDecision, ModelId, SessionRecord, StopReason, TokenCount};
use tea_session::{ApprovalArtifactEntry, GrantJournalEntry, SessionArchive, ToolExecutionState};
use tea_session_sqlite::SqliteSessionStore;
use tea_testkit::{ScriptedModelProvider, ScriptedModelResponse};

fn provider(
    scripts: impl IntoIterator<Item = ScriptedModelResponse>,
) -> Arc<ScriptedModelProvider> {
    provider_with_capabilities(scripts, ModelCapabilities::text().with_tools(true))
}

fn provider_with_capabilities(
    scripts: impl IntoIterator<Item = ScriptedModelResponse>,
    capabilities: ModelCapabilities,
) -> Arc<ScriptedModelProvider> {
    let model = ModelSpec::new(
        ModelId::from_str("fake/model").unwrap(),
        ProviderId::from_str("fake").unwrap(),
        ModelDisplayName::from_str("Fake Model").unwrap(),
        TokenCount::new(32_000).unwrap(),
        TokenCount::new(4_000).unwrap(),
        capabilities,
    )
    .unwrap();
    Arc::new(ScriptedModelProvider::new(
        ProviderId::from_str("fake").unwrap(),
        vec![model],
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

struct Fixture {
    root: std::path::PathBuf,
    workspace: WorkspaceRoot,
    resources: ResourceCatalog,
    database: std::path::PathBuf,
    bash: BashConfig,
    settings: CodingSettings,
}

impl Fixture {
    fn new(name: &str) -> Self {
        let root = std::env::temp_dir().join(format!(
            "tea-approval-{name}-{}",
            uuid::Uuid::now_v7().hyphenated()
        ));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("file.txt"), "old\n").unwrap();
        let workspace = WorkspaceRoot::new(&root).unwrap();
        let resources =
            ResourceCatalog::discover(&root, &root, ProjectAccess::Trusted, &[], &[], None, None)
                .unwrap();
        let database = root.join("sessions.sqlite3");
        let settings = merge_settings(
            CodingSettings {
                provider: "fake".to_owned(),
                ..CodingSettings::default()
            },
            None,
            None,
            None,
            Some(&SettingsLayer {
                model: Some("fake/model".to_owned()),
                ..Default::default()
            }),
        )
        .unwrap();
        let bash = BashConfig::new(
            BashShell::new("/bin/sh", "-c").unwrap(),
            BashOutputDirectory::new(&root).unwrap(),
            Duration::from_mins(1),
        )
        .unwrap();
        Self {
            root,
            workspace,
            resources,
            database,
            bash,
            settings,
        }
    }

    fn service(&self, provider: Arc<ScriptedModelProvider>) -> CodingAgentService {
        self.service_at(provider, &self.database)
    }

    fn service_at(
        &self,
        provider: Arc<ScriptedModelProvider>,
        database: &std::path::Path,
    ) -> CodingAgentService {
        let store = Arc::new(SqliteSessionStore::open(database.to_str().unwrap()).unwrap());
        CodingAgentBuilder::new(
            provider,
            self.workspace.clone(),
            self.resources.clone(),
            store,
            self.bash.clone(),
            self.settings.clone(),
            ActorId::from_str("local:user").unwrap(),
            WorkspaceId::from_str("workspace/local").unwrap(),
        )
        .build()
        .unwrap()
    }
}

async fn import_archive(database: &std::path::Path, value: &Value) {
    let archive = SessionArchive::decode_json(&serde_json::to_string(value).unwrap()).unwrap();
    let store = SqliteSessionStore::open(database.to_str().unwrap()).unwrap();
    archive.import_into(&store).await.unwrap();
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn pending_request(snapshot: &tea_session::SessionSnapshot) -> &tea_policy::ApprovalRequest {
    snapshot
        .approval_artifacts()
        .iter()
        .find_map(|entry| match entry {
            ApprovalArtifactEntry::Requested { request, .. } => Some(request),
            ApprovalArtifactEntry::Resolved { .. } => None,
        })
        .expect("pending request artifact")
}

#[tokio::test(flavor = "current_thread")]
#[allow(clippy::too_many_lines)]
async fn pending_request_survives_restart_and_resolves_exactly_once() {
    let fixture = Fixture::new("restart");
    let format = tea_protocol::FinalOutputFormat::JsonSchema {
        schema: serde_json::json!({
            "type": "object",
            "properties": {"status": {"type": "string"}},
            "required": ["status"],
            "additionalProperties": false
        }),
    };
    let provider = provider_with_capabilities(
        [
            tool_response(
                "edit-1",
                "edit",
                serde_json::json!({
                    "path":"file.txt",
                    "oldText":"old",
                    "newText":"new"
                }),
            ),
            ScriptedModelResponse::text([r#"{"status":"done"}"#]),
        ],
        ModelCapabilities::text()
            .with_tools(true)
            .with_final_json_schema_with_tools(),
    );
    let service = fixture.service(Arc::clone(&provider));
    let session_id = service.create_session().await.unwrap();
    service
        .prompt(session_id, "edit the file", Some(format.clone()))
        .await
        .unwrap();
    let approval_id = match service.wait(session_id).await.unwrap() {
        tea::RuntimeCommandOutcome::RunCompleted {
            pending_approval_id: Some(approval_id),
            ..
        } => approval_id,
        other => panic!("expected approval, got {other:?}"),
    };
    let before = service.session_snapshot(session_id).await.unwrap();
    let request = pending_request(&before).clone();

    // Closing the owning UI/service without a decision leaves durable pending state intact.
    service.shutdown().await;
    drop(service);

    let rebuilt = fixture.service(Arc::clone(&provider));
    rebuilt.open_session(session_id).await.unwrap();
    let reopened = rebuilt.session_snapshot(session_id).await.unwrap();
    assert_eq!(pending_request(&reopened), &request);
    let reopened_format = reopened
        .approval_artifacts()
        .iter()
        .find_map(|entry| match entry {
            ApprovalArtifactEntry::Requested {
                final_output_format,
                ..
            } => final_output_format.as_ref(),
            ApprovalArtifactEntry::Resolved { .. } => None,
        });
    assert_eq!(reopened_format, Some(&format));

    rebuilt
        .approve(session_id, approval_id, ApprovalDecision::AllowSession)
        .unwrap();
    let duplicate = rebuilt
        .approve(session_id, approval_id, ApprovalDecision::AllowOnce)
        .unwrap_err();
    assert!(duplicate.message().contains("owned run"));
    let outcome = rebuilt.wait(session_id).await.unwrap();
    assert!(matches!(
        outcome,
        tea::RuntimeCommandOutcome::RunCompleted {
            pending_approval_id: None,
            ..
        }
    ));

    // A stale UI may retry after completion; the command is accepted for ownership,
    // then fails against the persisted terminal artifact without adding facts.
    rebuilt
        .approve(session_id, approval_id, ApprovalDecision::AllowOnce)
        .unwrap();
    let stale = rebuilt.wait(session_id).await.unwrap_err();
    assert!(
        stale.message().contains("missing or already resolved"),
        "unexpected stale-resolution error: {}",
        stale.message()
    );

    let snapshot = rebuilt.session_snapshot(session_id).await.unwrap();
    assert!(snapshot.state().pending_approvals().is_empty());
    assert_eq!(
        snapshot
            .records()
            .iter()
            .filter(|record| matches!(
                record.record(),
                SessionRecord::ApprovalResolved { approval_id: id, .. } if *id == approval_id
            ))
            .count(),
        1
    );
    assert_eq!(
        snapshot
            .records()
            .iter()
            .filter(|record| matches!(
                record.record(),
                SessionRecord::ToolExecutionStarted { tool_call_id, .. }
                    if tool_call_id == request.tool_call_id()
            ))
            .count(),
        1
    );
    let [GrantJournalEntry::Issued { grant, .. }] = snapshot.grant_journal() else {
        panic!("expected exactly one issued grant")
    };
    assert_eq!(grant.scope(), &GrantScope::SessionResource { session_id });
    assert_eq!(
        fs::read_to_string(fixture.root.join("file.txt")).unwrap(),
        "new\n"
    );
    assert_eq!(provider.remaining_scripts().unwrap(), 0);
    let requests = provider.captured_requests().unwrap();
    assert_eq!(requests.len(), 2);
    assert!(
        requests
            .iter()
            .all(|request| request.final_output_format() == Some(&format))
    );
    rebuilt.shutdown().await;
}

#[tokio::test(flavor = "current_thread")]
async fn denial_is_terminal_and_never_starts_the_tool() {
    let fixture = Fixture::new("deny");
    let provider = provider([
        tool_response(
            "edit-denied",
            "edit",
            serde_json::json!({
                "path":"file.txt",
                "oldText":"old",
                "newText":"denied"
            }),
        ),
        ScriptedModelResponse::text(["understood"]),
    ]);
    let service = fixture.service(provider);
    let session_id = service.create_session().await.unwrap();
    service
        .prompt(session_id, "edit the file", None)
        .await
        .unwrap();
    let approval_id = match service.wait(session_id).await.unwrap() {
        tea::RuntimeCommandOutcome::RunCompleted {
            pending_approval_id: Some(approval_id),
            ..
        } => approval_id,
        other => panic!("expected approval, got {other:?}"),
    };
    service
        .approve(session_id, approval_id, ApprovalDecision::Deny)
        .unwrap();
    service.wait(session_id).await.unwrap();

    let snapshot = service.session_snapshot(session_id).await.unwrap();
    assert_eq!(
        fs::read_to_string(fixture.root.join("file.txt")).unwrap(),
        "old\n"
    );
    assert!(
        !snapshot
            .records()
            .iter()
            .any(|record| matches!(record.record(), SessionRecord::ToolExecutionStarted { .. }))
    );
    assert_eq!(snapshot.grant_journal().len(), 0);
    service.shutdown().await;
}

#[tokio::test(flavor = "current_thread")]
async fn missing_and_expired_artifacts_fail_before_execution() {
    let fixture = Fixture::new("invalid-artifacts");
    let provider = provider([
        tool_response(
            "edit-invalid",
            "edit",
            serde_json::json!({
                "path":"file.txt",
                "oldText":"old",
                "newText":"invalid"
            }),
        ),
        ScriptedModelResponse::text(["must remain unused"]),
    ]);
    let service = fixture.service(Arc::clone(&provider));
    let session_id = service.create_session().await.unwrap();
    service
        .prompt(session_id, "edit the file", None)
        .await
        .unwrap();
    let approval_id = match service.wait(session_id).await.unwrap() {
        tea::RuntimeCommandOutcome::RunCompleted {
            pending_approval_id: Some(approval_id),
            ..
        } => approval_id,
        other => panic!("expected approval, got {other:?}"),
    };
    let snapshot = service.session_snapshot(session_id).await.unwrap();
    let archive = SessionArchive::from_snapshot(&snapshot).unwrap();
    let archive_value = serde_json::to_value(archive).unwrap();
    service.shutdown().await;
    drop(service);

    let mut missing_value = archive_value.clone();
    missing_value["approvalArtifacts"] = serde_json::json!([]);
    let missing_database = fixture.root.join("missing.sqlite3");
    import_archive(&missing_database, &missing_value).await;
    let missing = fixture.service_at(Arc::clone(&provider), &missing_database);
    missing.open_session(session_id).await.unwrap();
    missing
        .approve(session_id, approval_id, ApprovalDecision::AllowOnce)
        .unwrap();
    let error = missing.wait(session_id).await.unwrap_err();
    assert!(error.message().contains("missing or already resolved"));
    let unchanged = missing.session_snapshot(session_id).await.unwrap();
    assert!(
        !unchanged
            .records()
            .iter()
            .any(|record| matches!(record.record(), SessionRecord::ToolExecutionStarted { .. }))
    );
    missing.shutdown().await;

    let mut expired_value = archive_value;
    let request = &mut expired_value["approvalArtifacts"][0]["request"];
    let created_at = chrono::DateTime::parse_from_rfc3339(
        request["createdAt"]
            .as_str()
            .expect("request creation time"),
    )
    .unwrap();
    let expires_at = (created_at + chrono::Duration::milliseconds(1))
        .to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
    request["expiresAt"] = Value::String(expires_at.clone());
    let approval_record = expired_value["records"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|record| record["type"] == "approval_requested")
        .expect("approval request record");
    approval_record["payload"]["expiresAt"] = Value::String(expires_at);
    let expired_database = fixture.root.join("expired.sqlite3");
    import_archive(&expired_database, &expired_value).await;
    let expired = fixture.service_at(Arc::clone(&provider), &expired_database);
    expired.open_session(session_id).await.unwrap();
    expired
        .approve(session_id, approval_id, ApprovalDecision::AllowOnce)
        .unwrap();
    let error = expired.wait(session_id).await.unwrap_err();
    assert!(error.message().contains("outside request lifetime"));
    let unchanged = expired.session_snapshot(session_id).await.unwrap();
    assert!(
        !unchanged
            .records()
            .iter()
            .any(|record| matches!(record.record(), SessionRecord::ToolExecutionStarted { .. }))
    );
    assert_eq!(provider.remaining_scripts().unwrap(), 1);
    expired.shutdown().await;
}

#[tokio::test(flavor = "current_thread")]
async fn cancellation_after_execution_start_is_durable_and_never_replayed() {
    let fixture = Fixture::new("started");
    let provider = provider([
        tool_response(
            "bash-1",
            "bash",
            serde_json::json!({"command":"sleep 30; printf replayed > should-not-exist"}),
        ),
        ScriptedModelResponse::text(["should not be consumed"]),
    ]);
    let service = fixture.service(Arc::clone(&provider));
    let session_id = service.create_session().await.unwrap();
    service
        .prompt(session_id, "run the command", None)
        .await
        .unwrap();
    let approval_id = match service.wait(session_id).await.unwrap() {
        tea::RuntimeCommandOutcome::RunCompleted {
            pending_approval_id: Some(approval_id),
            ..
        } => approval_id,
        other => panic!("expected approval, got {other:?}"),
    };
    let tool_call_id =
        *pending_request(&service.session_snapshot(session_id).await.unwrap()).tool_call_id();
    service
        .approve(session_id, approval_id, ApprovalDecision::AllowOnce)
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let snapshot = service.session_snapshot(session_id).await.unwrap();
            if matches!(
                snapshot.state().tool_calls()[&tool_call_id].execution(),
                ToolExecutionState::Started { .. }
            ) {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("tool execution did not start");

    service.abort(session_id).await.unwrap();
    let _ = service.wait(session_id).await;
    let interrupted = service.session_snapshot(session_id).await.unwrap();
    assert!(matches!(
        interrupted.state().tool_calls()[&tool_call_id].execution(),
        ToolExecutionState::Interrupted { .. }
    ));
    assert!(!fixture.root.join("should-not-exist").exists());
    service.shutdown().await;
    drop(service);

    let rebuilt = fixture.service(Arc::clone(&provider));
    rebuilt.open_session(session_id).await.unwrap();
    let reopened = rebuilt.session_snapshot(session_id).await.unwrap();
    assert!(matches!(
        reopened.state().tool_calls()[&tool_call_id].execution(),
        ToolExecutionState::Interrupted { .. }
    ));
    assert!(!fixture.root.join("should-not-exist").exists());
    assert_eq!(provider.remaining_scripts().unwrap(), 1);
    rebuilt.shutdown().await;
}
