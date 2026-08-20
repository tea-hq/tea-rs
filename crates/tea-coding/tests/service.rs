use std::fs;
use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use serde_json::Value;
use tea_coding::config::{
    ClientWebSearchSettingsLayer, CodingSettings, SettingsLayer, WebFetchSettingsLayer,
    WebSearchRoutePreference, WebSearchSettingsLayer, merge_settings,
};
use tea_coding::resources::{ResourceCatalog, SkillRoot, SkillSource};
use tea_coding::{CodingAgentBuilder, CodingError, CodingErrorCode, ProjectAccess};
use tea_coding_tools::{
    BashConfig, BashOutputDirectory, BashShell, FetchFuture, FetchProvider, FetchRequest,
    FetchResult, SearchFuture, SearchProvider, SearchRequest, SearchResponse, WorkspaceRoot,
};
use tea_context::{
    BudgetBehavior, CacheScope, ContextProvider, ContextProviderFuture, ContextProviderId,
    ContextRequest, PromptAuthority, PromptModule, PromptModuleId, PromptPriority,
    PromptProvenance, PromptSegment, PromptSegmentId, TrustLevel,
};
use tea_control::CancellationScope;
use tea_model::{
    HostedToolKind, ModelCapabilities, ModelCompletion, ModelDisplayName, ModelEvent,
    ModelResponseInfo, ModelSpec, ModelStreamIndex, ProviderId, ProviderToolCallId,
    ToolCallCompleted, ToolCallStarted,
};
use tea_policy::{ActorId, WorkspaceId};
use tea_protocol::{
    ApprovalDecision, CanonicalMessage, ContentBlock, ModelId, ModelRef, StopReason, TokenCount,
};
use tea_session::{ApprovalArtifactEntry, SessionArchive, SessionSnapshot};
use tea_session_sqlite::SqliteSessionStore;
use tea_testkit::{ScriptedModelProvider, ScriptedModelResponse};

static ID: AtomicU64 = AtomicU64::new(0);

#[derive(Debug)]
struct ServiceTurnProvider {
    id: ContextProviderId,
}

impl ServiceTurnProvider {
    fn new() -> Self {
        Self {
            id: "test.service_turn".parse().unwrap(),
        }
    }
}

impl ContextProvider for ServiceTurnProvider {
    fn id(&self) -> &ContextProviderId {
        &self.id
    }

    fn provide(&self, request: ContextRequest) -> ContextProviderFuture<'_> {
        let text = format!("service turn {}", request.run_id().unwrap());
        let segment = PromptSegment::new(
            PromptSegmentId::from_str("service.turn").unwrap(),
            text,
            PromptProvenance::new(
                self.id.clone(),
                "turn_extension",
                Some("extension:service-turn".to_owned()),
            )
            .unwrap(),
            TrustLevel::Delegated,
            CacheScope::Run,
            BudgetBehavior::Omit,
        )
        .unwrap();
        let module = PromptModule::new(
            PromptModuleId::from_str("service.turn").unwrap(),
            PromptAuthority::UserAddition,
            PromptPriority::new(0),
            vec![segment],
        )
        .unwrap();
        Box::pin(async move { Ok(vec![module]) })
    }
}

#[derive(Debug)]
struct FakeSearchProvider;

impl SearchProvider for FakeSearchProvider {
    fn destination(&self) -> &'static str {
        "https://search.example.com/query"
    }

    fn search(
        &self,
        _request: SearchRequest,
        _cancellation: CancellationScope,
    ) -> SearchFuture<'_> {
        Box::pin(async { SearchResponse::new(Vec::new(), false) })
    }
}

#[derive(Debug, Clone, Default)]
struct FakeFetchProvider {
    calls: Arc<AtomicUsize>,
}

impl FakeFetchProvider {
    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl FetchProvider for FakeFetchProvider {
    fn fetch(&self, request: FetchRequest, _cancellation: CancellationScope) -> FetchFuture<'_> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let url = request.url().to_owned();
        Box::pin(async move { FetchResult::new(&url, url.clone(), "text/plain", "fetched") })
    }
}

fn provider(scripts: Vec<ScriptedModelResponse>) -> Arc<ScriptedModelProvider> {
    provider_with_capabilities(scripts, ModelCapabilities::text().with_tools(true))
}

fn model_ref(model_id: &str) -> ModelRef {
    ModelRef::new("fake".parse().unwrap(), model_id.parse().unwrap())
}

fn fake_settings() -> CodingSettings {
    CodingSettings {
        provider: "fake".to_owned(),
        model: "fake/model".to_owned(),
        ..CodingSettings::default()
    }
}

fn provider_with_capabilities(
    scripts: Vec<ScriptedModelResponse>,
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

fn assert_persisted_image_privacy(
    snapshot: &SessionSnapshot,
    content: &[ContentBlock],
    local_path: &Path,
) {
    assert!(snapshot.state().messages().iter().any(|message| matches!(
        message,
        CanonicalMessage::User {
            content: persisted,
            ..
        } if persisted == content
    )));
    let archive_json =
        serde_json::to_string(&SessionArchive::from_snapshot(snapshot).unwrap()).unwrap();
    let local_path = local_path.to_str().unwrap();
    assert!(archive_json.contains("iVBORw0KGgo="));
    assert!(!archive_json.contains(local_path));
    assert!(!format!("{snapshot:?}").contains(local_path));
}

#[tokio::test(flavor = "current_thread")]
async fn typed_prompt_content_reaches_model_and_session() {
    let root = std::env::temp_dir().join(format!(
        "coding-service-image-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&root).unwrap();
    let local_path = root.join("private-source-image.png");
    fs::write(&local_path, b"\x89PNG\r\n\x1a\nprivate-image").unwrap();
    let workspace = WorkspaceRoot::new(&root).unwrap();
    let resources =
        ResourceCatalog::discover(&root, &root, ProjectAccess::Trusted, &[], &[], None, None)
            .unwrap();
    let database = root.join("sessions.sqlite3");
    let store = Arc::new(SqliteSessionStore::open(database.to_str().unwrap()).unwrap());
    let settings = merge_settings(
        fake_settings(),
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
        Duration::from_secs(5),
    )
    .unwrap();
    let provider = provider_with_capabilities(
        vec![ScriptedModelResponse::text(["described image"])],
        ModelCapabilities::text()
            .with_image_input()
            .with_tools(true),
    );
    let service = build_service(
        Arc::clone(&provider),
        workspace.clone(),
        resources.clone(),
        store.clone(),
        bash.clone(),
        settings.clone(),
    );
    let model = model_ref("fake/model");
    assert!(
        service
            .model_capabilities(&model)
            .is_some_and(ModelCapabilities::accepts_images)
    );
    assert!(
        service
            .model_capabilities(&model_ref("fake/missing"))
            .is_none()
    );

    let session_id = service.create_session().await.unwrap();
    let content = vec![
        ContentBlock::text("describe this image").unwrap(),
        ContentBlock::inline_image("image/png", "iVBORw0KGgo=").unwrap(),
    ];
    service.prompt_content(session_id, content.clone()).unwrap();
    service.wait(session_id).await.unwrap();

    let requests = provider.captured_requests().unwrap();
    assert!(matches!(
        &requests[0].messages()[0],
        CanonicalMessage::User {
            content: captured,
            ..
        } if captured == &content
    ));
    let snapshot = service.session_snapshot(session_id).await.unwrap();
    assert_persisted_image_privacy(&snapshot, &content, &local_path);

    service.shutdown().await;
    drop(service);
    drop(store);

    let reopened_store = Arc::new(SqliteSessionStore::open(database.to_str().unwrap()).unwrap());
    let reopened = build_service(
        provider_with_capabilities(
            Vec::new(),
            ModelCapabilities::text()
                .with_image_input()
                .with_tools(true),
        ),
        workspace,
        resources,
        reopened_store,
        bash,
        settings,
    );
    reopened.open_session(session_id).await.unwrap();
    let reopened_snapshot = reopened.session_snapshot(session_id).await.unwrap();
    assert_persisted_image_privacy(&reopened_snapshot, &content, &local_path);
    reopened.shutdown().await;
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn service_exposes_content_free_last_prompt_inspection() {
    let root = std::env::temp_dir().join(format!(
        "coding-service-prompt-inspection-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("AGENTS.md"), "PRIVATE PROJECT INSTRUCTION").unwrap();
    let workspace = WorkspaceRoot::new(&root).unwrap();
    let resources =
        ResourceCatalog::discover(&root, &root, ProjectAccess::Trusted, &[], &[], None, None)
            .unwrap();
    let store = Arc::new(
        SqliteSessionStore::open(root.join("sessions.sqlite3").to_str().unwrap()).unwrap(),
    );
    let bash = BashConfig::new(
        BashShell::new("/bin/sh", "-c").unwrap(),
        BashOutputDirectory::new(&root).unwrap(),
        Duration::from_secs(5),
    )
    .unwrap();
    let service = build_service(
        provider(vec![ScriptedModelResponse::text(["done"])]),
        workspace,
        resources,
        store,
        bash,
        fake_settings(),
    );
    let session = service.create_session().await.unwrap();
    assert!(service.prompt_inspection(session).unwrap().is_none());

    service.prompt(session, "inspect").unwrap();
    service.wait(session).await.unwrap();
    let inspection = service.prompt_inspection(session).unwrap().unwrap();
    assert_eq!(inspection.session_id(), session);
    assert!(inspection.run_id().is_some());
    assert!(
        inspection
            .prompt()
            .segments()
            .iter()
            .any(|entry| entry.module_id().as_str() == "workspace.instructions")
    );
    let encoded = serde_json::to_string(&inspection).unwrap();
    assert!(!encoded.contains("PRIVATE PROJECT INSTRUCTION"));
    assert!(!encoded.contains("byteRange"));
    service.shutdown().await;
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn coding_builder_accepts_typed_per_turn_context_providers() {
    let root = std::env::temp_dir().join(format!(
        "coding-service-turn-provider-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&root).unwrap();
    let workspace = WorkspaceRoot::new(&root).unwrap();
    let resources =
        ResourceCatalog::discover(&root, &root, ProjectAccess::Trusted, &[], &[], None, None)
            .unwrap();
    let store = Arc::new(
        SqliteSessionStore::open(root.join("sessions.sqlite3").to_str().unwrap()).unwrap(),
    );
    let bash = BashConfig::new(
        BashShell::new("/bin/sh", "-c").unwrap(),
        BashOutputDirectory::new(&root).unwrap(),
        Duration::from_secs(5),
    )
    .unwrap();
    let provider = provider(vec![ScriptedModelResponse::text(["done"])]);
    let service = service_builder(
        Arc::clone(&provider),
        workspace,
        resources,
        store,
        bash,
        fake_settings(),
    )
    .context_provider(Arc::new(ServiceTurnProvider::new()))
    .build()
    .unwrap();
    let session = service.create_session().await.unwrap();
    service.prompt(session, "turn").unwrap();
    service.wait(session).await.unwrap();

    let request = &provider.captured_requests().unwrap()[0];
    assert!(request.system_prompt().unwrap().contains("service turn "));
    assert!(
        service
            .prompt_inspection(session)
            .unwrap()
            .unwrap()
            .prompt()
            .segments()
            .iter()
            .any(|segment| segment.module_id().as_str() == "service.turn")
    );
    service.shutdown().await;
    fs::remove_dir_all(root).unwrap();
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

fn build_service(
    provider: Arc<ScriptedModelProvider>,
    workspace: WorkspaceRoot,
    resources: ResourceCatalog,
    store: Arc<SqliteSessionStore>,
    bash: BashConfig,
    settings: CodingSettings,
) -> tea_coding::CodingAgentService {
    service_builder(provider, workspace, resources, store, bash, settings)
        .build()
        .unwrap()
}

fn service_builder(
    provider: Arc<ScriptedModelProvider>,
    workspace: WorkspaceRoot,
    resources: ResourceCatalog,
    store: Arc<SqliteSessionStore>,
    bash: BashConfig,
    settings: CodingSettings,
) -> CodingAgentBuilder {
    CodingAgentBuilder::new(
        provider,
        workspace,
        resources,
        store,
        bash,
        settings,
        ActorId::from_str("local:user").unwrap(),
        WorkspaceId::from_str("workspace/local").unwrap(),
    )
}

fn search_settings(
    active: bool,
    client_enabled: bool,
    route_preference: WebSearchRoutePreference,
) -> CodingSettings {
    merge_settings(
        fake_settings(),
        None,
        None,
        None,
        Some(&SettingsLayer {
            model: Some("fake/model".to_owned()),
            active_tools: active.then(|| vec!["web_search".to_owned()]),
            web_search: Some(WebSearchSettingsLayer {
                route_preference: Some(route_preference),
                client: Some(ClientWebSearchSettingsLayer {
                    enabled: Some(client_enabled),
                    api_key_environment: Some("TAVILY_API_KEY".to_owned()),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        }),
    )
    .unwrap()
}

type SearchServiceBuild = (
    Result<tea_coding::CodingAgentService, CodingError>,
    Arc<ScriptedModelProvider>,
    Arc<SqliteSessionStore>,
    std::path::PathBuf,
);

fn assemble_search_service(
    label: &str,
    capabilities: ModelCapabilities,
    scripts: Vec<ScriptedModelResponse>,
    settings: CodingSettings,
    search_provider: Option<Arc<dyn SearchProvider>>,
) -> SearchServiceBuild {
    let root = std::env::temp_dir().join(format!(
        "coding-service-{label}-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&root).unwrap();
    let workspace = WorkspaceRoot::new(&root).unwrap();
    let resources =
        ResourceCatalog::discover(&root, &root, ProjectAccess::Trusted, &[], &[], None, None)
            .unwrap();
    let store = Arc::new(
        SqliteSessionStore::open(root.join("sessions.sqlite3").to_str().unwrap()).unwrap(),
    );
    let bash = BashConfig::new(
        BashShell::new("/bin/sh", "-c").unwrap(),
        BashOutputDirectory::new(&root).unwrap(),
        Duration::from_secs(5),
    )
    .unwrap();
    let provider = provider_with_capabilities(scripts, capabilities);
    let mut builder = service_builder(
        Arc::clone(&provider),
        workspace,
        resources,
        Arc::clone(&store),
        bash,
        settings,
    );
    if let Some(search_provider) = search_provider {
        builder = builder.search_provider(search_provider);
    }
    (builder.build(), provider, store, root)
}

fn fetch_settings(active: bool, enabled: bool) -> CodingSettings {
    merge_settings(
        fake_settings(),
        None,
        None,
        None,
        Some(&SettingsLayer {
            model: Some("fake/model".to_owned()),
            active_tools: active.then(|| vec!["web_fetch".to_owned()]),
            web_fetch: Some(WebFetchSettingsLayer {
                enabled: Some(enabled),
                ..Default::default()
            }),
            ..Default::default()
        }),
    )
    .unwrap()
}

type FetchServiceBuild = (
    Result<tea_coding::CodingAgentService, CodingError>,
    Arc<ScriptedModelProvider>,
    Arc<SqliteSessionStore>,
    std::path::PathBuf,
);

fn assemble_fetch_service(
    label: &str,
    capabilities: ModelCapabilities,
    scripts: Vec<ScriptedModelResponse>,
    settings: CodingSettings,
    fetch_provider: Option<Arc<dyn FetchProvider>>,
) -> FetchServiceBuild {
    let root = std::env::temp_dir().join(format!(
        "coding-service-{label}-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&root).unwrap();
    let workspace = WorkspaceRoot::new(&root).unwrap();
    let resources =
        ResourceCatalog::discover(&root, &root, ProjectAccess::Trusted, &[], &[], None, None)
            .unwrap();
    let store = Arc::new(
        SqliteSessionStore::open(root.join("sessions.sqlite3").to_str().unwrap()).unwrap(),
    );
    let bash = BashConfig::new(
        BashShell::new("/bin/sh", "-c").unwrap(),
        BashOutputDirectory::new(&root).unwrap(),
        Duration::from_secs(5),
    )
    .unwrap();
    let provider = provider_with_capabilities(scripts, capabilities);
    let mut builder = service_builder(
        Arc::clone(&provider),
        workspace,
        resources,
        Arc::clone(&store),
        bash,
        settings,
    );
    if let Some(fetch_provider) = fetch_provider {
        builder = builder.fetch_provider(fetch_provider);
    }
    (builder.build(), provider, store, root)
}

#[tokio::test(flavor = "current_thread")]
async fn coding_service_defaults_to_four_tools_and_can_activate_all_seven() {
    let root = std::env::temp_dir().join(format!(
        "coding-service-tools-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&root).unwrap();
    let workspace = WorkspaceRoot::new(&root).unwrap();
    let resources =
        ResourceCatalog::discover(&root, &root, ProjectAccess::Trusted, &[], &[], None, None)
            .unwrap();
    let store = Arc::new(
        SqliteSessionStore::open(root.join("sessions.sqlite3").to_str().unwrap()).unwrap(),
    );
    let settings = merge_settings(
        fake_settings(),
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
        Duration::from_secs(5),
    )
    .unwrap();
    let provider = provider(vec![
        ScriptedModelResponse::text(["default"]),
        ScriptedModelResponse::text(["all"]),
    ]);
    let service = build_service(
        Arc::clone(&provider),
        workspace,
        resources,
        Arc::clone(&store),
        bash,
        settings,
    );
    let session_id = service.create_session().await.unwrap();

    service.prompt(session_id, "default tools").unwrap();
    service.wait(session_id).await.unwrap();
    service
        .set_active_tools(
            session_id,
            ["read", "write", "edit", "bash", "grep", "find", "ls"]
                .into_iter()
                .map(|name| name.parse().unwrap())
                .collect(),
        )
        .await
        .unwrap();
    service.prompt(session_id, "all tools").unwrap();
    service.wait(session_id).await.unwrap();

    let requests = provider.captured_requests().unwrap();
    let tool_names = requests
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
        tool_names,
        [
            vec!["bash", "edit", "read", "write"],
            vec!["bash", "edit", "find", "grep", "ls", "read", "write"],
        ]
    );

    service.shutdown().await;
    drop(service);
    drop(store);
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn coding_system_prompt_uses_logical_sources_and_tracks_active_tools() {
    let root = std::env::temp_dir().join(format!(
        "private-user-coding-prompt-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("AGENTS.md"), "project rules\n").unwrap();
    let workspace = WorkspaceRoot::new(&root).unwrap();
    let resources =
        ResourceCatalog::discover(&root, &root, ProjectAccess::Trusted, &[], &[], None, None)
            .unwrap();
    let store = Arc::new(
        SqliteSessionStore::open(root.join("sessions.sqlite3").to_str().unwrap()).unwrap(),
    );
    let settings = CodingSettings {
        active_tools: vec!["read".to_owned()],
        ..fake_settings()
    };
    let bash = BashConfig::new(
        BashShell::new("/bin/sh", "-c").unwrap(),
        BashOutputDirectory::new(&root).unwrap(),
        Duration::from_secs(5),
    )
    .unwrap();
    let provider = provider(vec![
        ScriptedModelResponse::text(["read"]),
        ScriptedModelResponse::text(["write"]),
    ]);
    let service = build_service(
        Arc::clone(&provider),
        workspace,
        resources,
        Arc::clone(&store),
        bash,
        settings,
    );
    let session_id = service.create_session().await.unwrap();

    service.prompt(session_id, "first").unwrap();
    service.wait(session_id).await.unwrap();
    service
        .set_active_tools(session_id, vec!["write".parse().unwrap()])
        .await
        .unwrap();
    service.prompt(session_id, "second").unwrap();
    service.wait(session_id).await.unwrap();

    let requests = provider.captured_requests().unwrap();
    let read_prompt = requests[0].system_prompt().unwrap();
    let write_prompt = requests[1].system_prompt().unwrap();
    for prompt in [read_prompt, write_prompt] {
        assert!(prompt.contains("<workspace>"));
        assert!(prompt.contains("<workspace>/AGENTS.md"));
        assert!(prompt.contains("project rules\n"));
        assert!(!prompt.contains(root.to_str().unwrap()));
        assert!(!prompt.contains("private-user"));
    }
    assert!(read_prompt.contains("Tool `read`"));
    assert!(!read_prompt.contains("Tool `write`"));
    assert!(write_prompt.contains("Tool `write`"));
    assert!(!write_prompt.contains("Tool `read`"));

    let snapshot = service.session_snapshot(session_id).await.unwrap();
    let archive =
        serde_json::to_string(&SessionArchive::from_snapshot(&snapshot).unwrap()).unwrap();
    assert!(!archive.contains("<workspace>"));
    assert!(!archive.contains("minimal verified changes"));

    service.shutdown().await;
    drop(service);
    drop(store);
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
#[allow(clippy::too_many_lines)]
async fn skill_resource_tool_is_conditional_and_executes_against_the_catalog() {
    let empty_root = std::env::temp_dir().join(format!(
        "coding-service-empty-skills-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&empty_root).unwrap();
    let empty_workspace = WorkspaceRoot::new(&empty_root).unwrap();
    let empty_resources = ResourceCatalog::discover(
        &empty_root,
        &empty_root,
        ProjectAccess::Trusted,
        &[],
        &[],
        None,
        None,
    )
    .unwrap();
    let empty_store = Arc::new(
        SqliteSessionStore::open(empty_root.join("sessions.sqlite3").to_str().unwrap()).unwrap(),
    );
    let empty_bash = BashConfig::new(
        BashShell::new("/bin/sh", "-c").unwrap(),
        BashOutputDirectory::new(&empty_root).unwrap(),
        Duration::from_secs(5),
    )
    .unwrap();
    let empty_provider = provider(vec![ScriptedModelResponse::text(["empty"])]);
    let empty_service = build_service(
        Arc::clone(&empty_provider),
        empty_workspace,
        empty_resources,
        empty_store,
        empty_bash,
        fake_settings(),
    );
    let empty_session = empty_service.create_session().await.unwrap();
    empty_service.prompt(empty_session, "empty").unwrap();
    empty_service.wait(empty_session).await.unwrap();
    assert!(
        !empty_provider.captured_requests().unwrap()[0]
            .tools()
            .iter()
            .any(|tool| tool.name() == "read_skill_resource")
    );
    empty_service.shutdown().await;
    fs::remove_dir_all(&empty_root).unwrap();

    let root = std::env::temp_dir().join(format!(
        "coding-service-skill-resource-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let skill = root.join("skills/review");
    fs::create_dir_all(skill.join("references")).unwrap();
    fs::write(
        skill.join("SKILL.md"),
        "---\nname: review\ndescription: Review code\n---\nUse the checklist.\n",
    )
    .unwrap();
    fs::write(skill.join("references/checklist.md"), "one\ntwo\nthree\n").unwrap();
    let workspace = WorkspaceRoot::new(&root).unwrap();
    let resources = ResourceCatalog::discover_with_skill_roots(
        &root,
        &root,
        ProjectAccess::Trusted,
        &[SkillRoot::new(root.join("skills"), SkillSource::UserTea).unwrap()],
        None,
        None,
    )
    .unwrap();
    let store = Arc::new(
        SqliteSessionStore::open(root.join("sessions.sqlite3").to_str().unwrap()).unwrap(),
    );
    let bash = BashConfig::new(
        BashShell::new("/bin/sh", "-c").unwrap(),
        BashOutputDirectory::new(&root).unwrap(),
        Duration::from_secs(5),
    )
    .unwrap();
    let provider = provider(vec![
        tool_response(
            "skill-resource",
            "read_skill_resource",
            serde_json::json!({
                "skill":"review",
                "path":"references/checklist.md",
                "offset":1,
                "limit":2
            }),
        ),
        ScriptedModelResponse::text(["read"]),
    ]);
    let service = build_service(
        Arc::clone(&provider),
        workspace,
        resources,
        store,
        bash,
        fake_settings(),
    );
    let session = service.create_session().await.unwrap();
    service.prompt(session, "read the checklist").unwrap();
    let outcome = service.wait(session).await.unwrap();
    assert!(matches!(
        outcome,
        tea::RuntimeCommandOutcome::RunCompleted {
            pending_approval_id: None,
            ..
        }
    ));
    let requests = provider.captured_requests().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(
        requests[0]
            .tools()
            .iter()
            .filter(|tool| tool.name() == "read_skill_resource")
            .count(),
        1
    );
    assert!(
        requests[1]
            .messages()
            .iter()
            .any(|message| format!("{message:?}").contains("one\\ntwo\\n"))
    );
    service.shutdown().await;
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn hosted_search_is_registered_but_requires_explicit_activation() {
    let capabilities = ModelCapabilities::text()
        .with_tools(true)
        .with_hosted_tool(HostedToolKind::WebSearch);
    let (service, provider, store, root) = assemble_search_service(
        "hosted-search",
        capabilities,
        vec![
            ScriptedModelResponse::text(["default"]),
            ScriptedModelResponse::text(["search"]),
        ],
        search_settings(false, false, WebSearchRoutePreference::PreferHosted),
        None,
    );
    let service = service.unwrap();
    assert_eq!(
        service.settings().active_tools,
        ["read", "write", "edit", "bash"]
    );
    let session_id = service.create_session().await.unwrap();
    service.prompt(session_id, "default tools").unwrap();
    service.wait(session_id).await.unwrap();
    service
        .set_active_tools(session_id, ["web_search".parse().unwrap()].into())
        .await
        .unwrap();
    service.prompt(session_id, "search tools").unwrap();
    service.wait(session_id).await.unwrap();

    let requests = provider.captured_requests().unwrap();
    assert_eq!(
        requests[0]
            .tools()
            .iter()
            .map(tea_model::ModelToolDefinition::name)
            .collect::<Vec<_>>(),
        ["bash", "edit", "read", "write"]
    );
    assert_eq!(requests[1].tools().len(), 1);
    assert!(requests[1].tools()[0].as_hosted().is_some());
    assert_eq!(requests[1].tools()[0].name(), "web_search");

    service.shutdown().await;
    drop(service);
    drop(store);
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn configured_client_backend_does_not_activate_search_and_prefers_hosted() {
    let capabilities = ModelCapabilities::text()
        .with_tools(true)
        .with_hosted_tool(HostedToolKind::WebSearch);
    let search_provider: Arc<dyn SearchProvider> = Arc::new(FakeSearchProvider);
    let (service, provider, store, root) = assemble_search_service(
        "hybrid-search",
        capabilities,
        vec![
            ScriptedModelResponse::text(["default"]),
            ScriptedModelResponse::text(["search"]),
        ],
        search_settings(false, true, WebSearchRoutePreference::PreferHosted),
        Some(search_provider),
    );
    let service = service.unwrap();
    let session_id = service.create_session().await.unwrap();
    service.prompt(session_id, "default tools").unwrap();
    service.wait(session_id).await.unwrap();
    service
        .set_active_tools(session_id, ["web_search".parse().unwrap()].into())
        .await
        .unwrap();
    service.prompt(session_id, "search tools").unwrap();
    service.wait(session_id).await.unwrap();

    let requests = provider.captured_requests().unwrap();
    assert!(
        requests[0]
            .tools()
            .iter()
            .all(|tool| tool.name() != "web_search")
    );
    assert!(requests[1].tools()[0].as_hosted().is_some());

    service.shutdown().await;
    drop(service);
    drop(store);
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn hybrid_search_falls_back_to_client_for_models_without_hosted_search() {
    let search_provider: Arc<dyn SearchProvider> = Arc::new(FakeSearchProvider);
    let (service, provider, store, root) = assemble_search_service(
        "client-fallback-search",
        ModelCapabilities::text().with_tools(true),
        vec![ScriptedModelResponse::text(["client search"])],
        search_settings(true, true, WebSearchRoutePreference::PreferHosted),
        Some(search_provider),
    );
    let service = service.unwrap();
    let session_id = service.create_session().await.unwrap();
    service.prompt(session_id, "search tools").unwrap();
    service.wait(session_id).await.unwrap();

    let requests = provider.captured_requests().unwrap();
    assert_eq!(requests[0].tools().len(), 1);
    assert!(requests[0].tools()[0].as_function().is_some());
    assert_eq!(requests[0].tools()[0].name(), "web_search");

    service.shutdown().await;
    drop(service);
    drop(store);
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn force_client_projects_function_and_fails_without_a_real_backend() {
    let capabilities = ModelCapabilities::text()
        .with_tools(true)
        .with_hosted_tool(HostedToolKind::WebSearch);
    let settings = search_settings(true, true, WebSearchRoutePreference::ForceClient);
    let search_provider: Arc<dyn SearchProvider> = Arc::new(FakeSearchProvider);
    let (service, provider, store, root) = assemble_search_service(
        "forced-client-search",
        capabilities,
        vec![ScriptedModelResponse::text(["client search"])],
        settings.clone(),
        Some(search_provider),
    );
    let service = service.unwrap();
    let session_id = service.create_session().await.unwrap();
    service.prompt(session_id, "search tools").unwrap();
    service.wait(session_id).await.unwrap();
    let requests = provider.captured_requests().unwrap();
    assert!(requests[0].tools()[0].as_function().is_some());
    service.shutdown().await;
    drop(service);
    drop(store);
    fs::remove_dir_all(root).unwrap();

    let (missing, _, store, root) = assemble_search_service(
        "missing-client-search",
        capabilities,
        Vec::new(),
        settings,
        None,
    );
    let Err(error) = missing else {
        panic!("force_client must require a real search provider");
    };
    assert_eq!(error.code(), CodingErrorCode::InvalidInput);
    assert_eq!(
        error.message(),
        "client web search requires an injected search provider"
    );
    drop(store);
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn hosted_search_route_is_rejected_before_provider_io_when_model_lacks_capability() {
    let settings = search_settings(true, false, WebSearchRoutePreference::PreferHosted);
    let (service, provider, store, root) = assemble_search_service(
        "unsupported-hosted-search",
        ModelCapabilities::text().with_tools(true),
        Vec::new(),
        settings,
        None,
    );
    let error = service.unwrap_err();
    assert_eq!(error.code(), CodingErrorCode::InvalidInput);
    assert_eq!(
        error.message(),
        "active tool web_search has no execution route supported by selected model fake/model; declare the model capability or configure a supported client route"
    );
    assert!(provider.captured_requests().unwrap().is_empty());
    drop(store);
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn active_tool_override_rejects_an_unsupported_hosted_route_immediately() {
    let settings = search_settings(false, false, WebSearchRoutePreference::PreferHosted);
    let (service, provider, store, root) = assemble_search_service(
        "unsupported-hosted-override",
        ModelCapabilities::text().with_tools(true),
        Vec::new(),
        settings,
        None,
    );
    let service = service.unwrap();
    let session_id = service.create_session().await.unwrap();
    let error = service
        .set_active_tools(session_id, vec!["web_search".parse().unwrap()])
        .await
        .unwrap_err();
    assert_eq!(error.code(), CodingErrorCode::InvalidInput);
    assert_eq!(
        error.message(),
        "active tool web_search has no execution route supported by selected model fake/model; declare the model capability or configure a supported client route"
    );
    assert!(provider.captured_requests().unwrap().is_empty());
    service.shutdown().await;
    drop(service);
    drop(store);
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn web_fetch_registration_requires_activation_enablement_and_provider() {
    let fetch = FakeFetchProvider::default();
    let fetch_provider: Arc<dyn FetchProvider> = Arc::new(fetch.clone());
    let (inactive, model, store, root) = assemble_fetch_service(
        "inactive-fetch",
        ModelCapabilities::text().with_tools(true),
        Vec::new(),
        fetch_settings(false, true),
        Some(fetch_provider),
    );
    let inactive = inactive.unwrap();
    let session_id = inactive.create_session().await.unwrap();
    let error = inactive
        .set_active_tools(session_id, vec!["web_fetch".parse().unwrap()])
        .await
        .unwrap_err();
    assert_eq!(error.code(), CodingErrorCode::InvalidInput);
    assert!(model.captured_requests().unwrap().is_empty());
    assert_eq!(fetch.calls(), 0);
    inactive.shutdown().await;
    drop(inactive);
    drop(store);
    fs::remove_dir_all(root).unwrap();

    let provider: Arc<dyn FetchProvider> = Arc::new(FakeFetchProvider::default());
    let (disabled, _, store, root) = assemble_fetch_service(
        "disabled-fetch",
        ModelCapabilities::text().with_tools(true),
        Vec::new(),
        fetch_settings(true, false),
        Some(provider),
    );
    let error = disabled.unwrap_err();
    assert_eq!(error.code(), CodingErrorCode::InvalidInput);
    assert_eq!(
        error.message(),
        "active web fetch requires an enabled client backend"
    );
    drop(store);
    fs::remove_dir_all(root).unwrap();

    let (missing, _, store, root) = assemble_fetch_service(
        "missing-fetch",
        ModelCapabilities::text().with_tools(true),
        Vec::new(),
        fetch_settings(true, true),
        None,
    );
    let error = missing.unwrap_err();
    assert_eq!(error.code(), CodingErrorCode::InvalidInput);
    assert_eq!(
        error.message(),
        "active web fetch requires an injected fetch provider"
    );
    drop(store);
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn web_fetch_is_a_client_function_independent_of_hosted_model_capabilities() {
    let fetch_provider: Arc<dyn FetchProvider> = Arc::new(FakeFetchProvider::default());
    let capabilities = ModelCapabilities::text()
        .with_tools(true)
        .with_hosted_tool(HostedToolKind::WebSearch);
    let (service, provider, store, root) = assemble_fetch_service(
        "client-fetch",
        capabilities,
        vec![ScriptedModelResponse::text(["fetch available"])],
        fetch_settings(true, true),
        Some(fetch_provider),
    );
    let service = service.unwrap();
    let session_id = service.create_session().await.unwrap();
    service.prompt(session_id, "fetch tools").unwrap();
    service.wait(session_id).await.unwrap();

    let requests = provider.captured_requests().unwrap();
    assert_eq!(requests[0].tools().len(), 1);
    assert_eq!(requests[0].tools()[0].name(), "web_fetch");
    assert!(requests[0].tools()[0].as_function().is_some());

    service.shutdown().await;
    drop(service);
    drop(store);
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn web_fetch_approval_precedes_provider_and_denial_never_calls_it() {
    const URL: &str = "https://example.com/weather?token=private";
    let fetch = FakeFetchProvider::default();
    let fetch_provider: Arc<dyn FetchProvider> = Arc::new(fetch.clone());
    let (service, _, store, root) = assemble_fetch_service(
        "approved-fetch",
        ModelCapabilities::text().with_tools(true),
        vec![
            tool_response(
                "fetch-approved",
                "web_fetch",
                serde_json::json!({"url":URL}),
            ),
            ScriptedModelResponse::text(["done"]),
        ],
        fetch_settings(true, true),
        Some(fetch_provider),
    );
    let service = service.unwrap();
    let session_id = service.create_session().await.unwrap();
    service.prompt(session_id, "fetch the URL").unwrap();
    let outcome = service.wait(session_id).await.unwrap();
    let approval_id = match outcome {
        tea::RuntimeCommandOutcome::RunCompleted {
            pending_approval_id: Some(approval_id),
            ..
        } => approval_id,
        other => panic!("expected web fetch approval, got {other:?}"),
    };
    assert_eq!(fetch.calls(), 0);
    let snapshot = service.session_snapshot(session_id).await.unwrap();
    let request = snapshot
        .approval_artifacts()
        .iter()
        .find_map(|artifact| match artifact {
            ApprovalArtifactEntry::Requested { request, .. } => Some(request),
            ApprovalArtifactEntry::Resolved { .. } => None,
        })
        .unwrap();
    assert_eq!(request.resources()[0].scheme(), "url");
    assert_eq!(request.resources()[0].locator(), URL);
    assert_eq!(
        request.presentation().resources(),
        ["url:https://example.com/weather?[REDACTED]"]
    );
    service
        .approve(session_id, approval_id, ApprovalDecision::AllowOnce)
        .unwrap();
    service.wait(session_id).await.unwrap();
    assert_eq!(fetch.calls(), 1);
    service.shutdown().await;
    drop(service);
    drop(store);
    fs::remove_dir_all(root).unwrap();

    let denied_fetch = FakeFetchProvider::default();
    let denied_provider: Arc<dyn FetchProvider> = Arc::new(denied_fetch.clone());
    let (service, _, store, root) = assemble_fetch_service(
        "denied-fetch",
        ModelCapabilities::text().with_tools(true),
        vec![
            tool_response("fetch-denied", "web_fetch", serde_json::json!({"url":URL})),
            ScriptedModelResponse::text(["denied"]),
        ],
        fetch_settings(true, true),
        Some(denied_provider),
    );
    let service = service.unwrap();
    let session_id = service.create_session().await.unwrap();
    service.prompt(session_id, "fetch the URL").unwrap();
    let outcome = service.wait(session_id).await.unwrap();
    let approval_id = match outcome {
        tea::RuntimeCommandOutcome::RunCompleted {
            pending_approval_id: Some(approval_id),
            ..
        } => approval_id,
        other => panic!("expected web fetch approval, got {other:?}"),
    };
    assert_eq!(denied_fetch.calls(), 0);
    service
        .approve(session_id, approval_id, ApprovalDecision::Deny)
        .unwrap();
    service.wait(session_id).await.unwrap();
    assert_eq!(denied_fetch.calls(), 0);
    service.shutdown().await;
    drop(service);
    drop(store);
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
#[allow(clippy::too_many_lines)] // Keep the end-to-end approval/restart sequence visible.
async fn coding_loop_rebuilds_between_approval_and_resolution() {
    let root = std::env::temp_dir().join(format!(
        "coding-service-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("file.txt"), "old\n").unwrap();
    let workspace = WorkspaceRoot::new(&root).unwrap();
    let resources =
        ResourceCatalog::discover(&root, &root, ProjectAccess::Trusted, &[], &[], None, None)
            .unwrap();
    let database = root.join("sessions.sqlite3");
    let store = Arc::new(SqliteSessionStore::open(database.to_str().unwrap()).unwrap());
    let settings = merge_settings(
        fake_settings(),
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
        Duration::from_secs(5),
    )
    .unwrap();
    let provider = provider(vec![
        tool_response("call_read", "read", serde_json::json!({"path":"file.txt"})),
        tool_response(
            "call_edit",
            "edit",
            serde_json::json!({"path":"file.txt","oldText":"old","newText":"new"}),
        ),
        tool_response(
            "call_bash",
            "bash",
            serde_json::json!({"command":"grep -q new file.txt && touch passed"}),
        ),
        ScriptedModelResponse::text(["done"]),
    ]);

    let service = build_service(
        provider.clone(),
        workspace.clone(),
        resources.clone(),
        store.clone(),
        bash.clone(),
        settings.clone(),
    );
    assert_eq!(service.models(), [model_ref("fake/model")]);
    assert!(service.resources().skill_metadata().is_empty());
    let session_id = service.create_session().await.unwrap();
    let _events = service.subscribe(session_id).unwrap();
    service.prompt(session_id, "fix and test the file").unwrap();
    let first = service.wait(session_id).await.unwrap();
    let edit_approval = match first {
        tea::RuntimeCommandOutcome::RunCompleted {
            pending_approval_id: Some(approval_id),
            ..
        } => approval_id,
        outcome => panic!("expected edit approval, got {outcome:?}"),
    };
    let canonical_snapshot = service.session_snapshot(session_id).await.unwrap();
    assert_eq!(
        canonical_snapshot.state().pending_approvals().len(),
        1,
        "host snapshot queries must expose complete canonical recovery state"
    );
    assert_eq!(canonical_snapshot.approval_artifacts().len(), 1);
    service.shutdown().await;
    drop(service);
    drop(store);

    let reopened_store = Arc::new(SqliteSessionStore::open(database.to_str().unwrap()).unwrap());
    let rebuilt = build_service(
        provider.clone(),
        workspace,
        resources,
        reopened_store,
        bash,
        settings,
    );
    rebuilt.open_session(session_id).await.unwrap();
    rebuilt
        .approve(session_id, edit_approval, ApprovalDecision::AllowOnce)
        .unwrap();
    let second = rebuilt.wait(session_id).await.unwrap();
    let bash_approval = match second {
        tea::RuntimeCommandOutcome::RunCompleted {
            pending_approval_id: Some(approval_id),
            ..
        } => approval_id,
        outcome => panic!("expected bash approval, got {outcome:?}"),
    };
    rebuilt
        .approve(session_id, bash_approval, ApprovalDecision::AllowOnce)
        .unwrap();
    let final_outcome = rebuilt.wait(session_id).await.unwrap();
    assert!(matches!(
        final_outcome,
        tea::RuntimeCommandOutcome::RunCompleted {
            pending_approval_id: None,
            ..
        }
    ));
    assert_eq!(fs::read_to_string(root.join("file.txt")).unwrap(), "new\n");
    assert!(root.join("passed").exists());
    assert_eq!(
        rebuilt
            .stats(session_id)
            .await
            .unwrap()
            .assistant_messages(),
        4
    );
    assert_eq!(rebuilt.list_sessions().await.unwrap().len(), 1);
    assert_eq!(provider.remaining_scripts().unwrap(), 0);
    assert!(
        provider.captured_requests().unwrap()[0]
            .system_prompt()
            .is_some_and(|prompt| prompt.contains("smallest coherent change"))
    );
    rebuilt.shutdown().await;
    fs::remove_dir_all(root).unwrap();
}
