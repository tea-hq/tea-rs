use std::collections::BTreeMap;
use std::fs;
use std::str::FromStr as _;
use std::sync::Arc;

use clap::Parser as _;
use tea::RuntimeCommandOutcome;
use tea_cli::args::CliArgs;
use tea_cli::{BootstrapEnvironment, CliBootstrap};
use tea_model::{
    ModelCapabilities, ModelCompletion, ModelDisplayName, ModelEvent, ModelProvider,
    ModelResponseInfo, ModelSpec, ModelStreamIndex, ProviderId, ProviderToolCallId,
    ToolCallCompleted, ToolCallStarted,
};
use tea_protocol::{ApprovalDecision, ModelId, StopReason, TokenCount};
use tea_testkit::{ScriptedModelProvider, ScriptedModelResponse};

const FIXTURE: &str = env!("CARGO_BIN_EXE_tea-cli-mcp-fixture-server");

fn temp_root() -> std::path::PathBuf {
    let root = std::env::temp_dir().join(format!(
        "tea-cli-mcp-integration-{}",
        uuid::Uuid::now_v7().hyphenated()
    ));
    fs::create_dir_all(&root).unwrap();
    root
}

fn provider() -> Arc<ScriptedModelProvider> {
    let model = ModelSpec::new(
        ModelId::from_str("fake/model").unwrap(),
        ProviderId::from_str("fake").unwrap(),
        ModelDisplayName::from_str("Fake Model").unwrap(),
        TokenCount::new(32_000).unwrap(),
        TokenCount::new(4_000).unwrap(),
        ModelCapabilities::text().with_tools(true),
    )
    .unwrap();
    let index = ModelStreamIndex::new(0).unwrap();
    let call_id = ProviderToolCallId::from_str("mcp-call").unwrap();
    Arc::new(ScriptedModelProvider::new(
        ProviderId::from_str("fake").unwrap(),
        vec![model],
        [
            ScriptedModelResponse::events([
                ModelEvent::Started(ModelResponseInfo::new()),
                ModelEvent::ToolCallStarted(
                    ToolCallStarted::new(index, call_id.clone(), "mcp.fixture.echo").unwrap(),
                ),
                ModelEvent::ToolCallCompleted(
                    ToolCallCompleted::new(
                        index,
                        call_id,
                        "mcp.fixture.echo",
                        serde_json::json!({"value":"hello"}),
                    )
                    .unwrap(),
                ),
                ModelEvent::Completed(ModelCompletion::new(StopReason::ToolUse).unwrap()),
            ]),
            ScriptedModelResponse::text(["done"]),
        ],
    ))
}

#[tokio::test(flavor = "current_thread")]
async fn prepared_manager_registers_frozen_mcp_tools_and_awaits_shutdown() {
    let root = temp_root();
    let marker = root.join("fixture.marker");
    fs::create_dir_all(root.join(".tea")).unwrap();
    fs::write(
        root.join(".tea/settings.json"),
        format!(
            r#"{{
                "schemaVersion":1,
                "activeTools":["mcp.fixture.echo"],
                "mcpServers":[{{
                    "id":"fixture",
                    "transport":{{
                        "type":"stdio",
                        "executable":"{FIXTURE}",
                        "arguments":["execute","success","{}"]
                    }},
                    "tools":[{{
                        "remoteName":"echo",
                        "alias":"mcp.fixture.echo",
                        "declaration":{{
                            "effects":["fs.read"],
                            "idempotency":"idempotent",
                            "retrySafety":"never",
                            "concurrency":"serial",
                            "timeoutMillis":1000
                        }}
                    }}]
                }}]
            }}"#,
            marker.display(),
        ),
    )
    .unwrap();
    let args = CliArgs::try_parse_from([
        "tea",
        "--no-session",
        "--provider",
        "fake",
        "--model",
        "fake/model",
        "--trust",
        "ignore",
    ])
    .unwrap();
    let provider = provider();
    let model_provider: Arc<dyn ModelProvider> = provider.clone();
    let bootstrap = CliBootstrap::new(BootstrapEnvironment::new(
        &root,
        Some(root.clone()),
        BTreeMap::new(),
    ))
    .with_provider(model_provider);
    let (service, _) = bootstrap.build_async(&args).await.unwrap();

    let catalog = service.mcp_snapshot().unwrap();
    assert_eq!(catalog.servers().len(), 1);
    assert_eq!(catalog.catalog().len(), 1);
    assert_eq!(
        catalog.catalog()[0].tool_name().as_str(),
        "mcp.fixture.echo"
    );

    let session_id = service.create_session().await.unwrap();
    service
        .prompt(session_id, "call the MCP tool", None)
        .await
        .unwrap();
    let approval = match service.wait(session_id).await.unwrap() {
        RuntimeCommandOutcome::RunCompleted {
            pending_approval_id: Some(approval_id),
            ..
        } => approval_id,
        outcome => panic!("expected MCP approval, got {outcome:?}"),
    };
    service
        .approve(session_id, approval, ApprovalDecision::AllowOnce)
        .unwrap();
    assert!(matches!(
        service.wait(session_id).await.unwrap(),
        RuntimeCommandOutcome::RunCompleted {
            pending_approval_id: None,
            ..
        }
    ));
    assert_eq!(provider.remaining_scripts().unwrap(), 0);
    let requests = provider.captured_requests().unwrap();
    assert_eq!(requests[0].tools()[0].name(), "mcp.fixture.echo");
    assert_eq!(fs::read_to_string(&marker).unwrap(), "called\n");

    service.shutdown().await;
    assert!(service.mcp_snapshot().unwrap().servers().is_empty());
    fs::remove_dir_all(root).unwrap();
}
