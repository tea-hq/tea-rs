use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr as _;
use std::sync::Arc;

use clap::Parser as _;
use serde_json::{Value, json};
use tea::RuntimeCommandOutcome;
use tea_cli::args::CliArgs;
use tea_cli::{BootstrapEnvironment, CliBootstrap};
use tea_model::{
    ModelCapabilities, ModelCompletion, ModelDisplayName, ModelEvent, ModelProvider,
    ModelResponseInfo, ModelSpec, ModelStreamIndex, ProviderId, ProviderToolCallId,
    ToolCallCompleted, ToolCallStarted,
};
use tea_protocol::{
    ApprovalDecision, ExecutionTarget, ModelId, RecordEnvelope, SessionRecord, SessionRecordType,
    StopReason, TokenCount,
};
use tea_provider_openai::env_file::load_env_file;
use tea_testkit::{ScriptedModelProvider, ScriptedModelResponse};
use tea_tools::TOOL_AUDIT_METADATA_NAMESPACE;

const LIVE_GATE: &str = "TEA_MCP_LIVE_SMOKE";
const LIVE_COMMAND: &str = "TEA_MCP_LIVE_COMMAND";
const LIVE_ARGUMENTS: &str = "TEA_MCP_LIVE_ARGUMENTS_JSON";
const LIVE_SERVER_ID: &str = "TEA_MCP_LIVE_SERVER_ID";
const LIVE_REMOTE_TOOL: &str = "TEA_MCP_LIVE_REMOTE_TOOL";
const LIVE_TOOL_ALIAS: &str = "TEA_MCP_LIVE_TOOL_ALIAS";
const LIVE_TOOL_ARGUMENTS: &str = "TEA_MCP_LIVE_TOOL_ARGUMENTS_JSON";
const LIVE_INHERITED_ENVIRONMENT: &str = "TEA_MCP_LIVE_INHERITED_ENVIRONMENT_JSON";
const FINAL_MARKER: &str = "MCP_LIVE_SMOKE_OK";

struct LiveMcpConfig {
    command: PathBuf,
    arguments: Vec<String>,
    server_id: String,
    remote_tool: String,
    tool_alias: String,
    tool_arguments: Value,
    inherited_environment: Vec<String>,
}

struct LiveWorkspace {
    root: PathBuf,
}

impl LiveWorkspace {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "tea-cli-mcp-live-smoke-{}",
            uuid::Uuid::now_v7().hyphenated()
        ));
        fs::create_dir_all(root.join("workspace")).unwrap();
        Self { root }
    }

    fn workspace(&self) -> PathBuf {
        self.root.join("workspace")
    }

    fn write_settings(&self, config: &LiveMcpConfig) {
        let config_dir = self.root.join("config");
        fs::create_dir_all(&config_dir).unwrap();
        let settings = json!({
            "schemaVersion": 1,
            "activeTools": [config.tool_alias],
            "mcpServers": [{
                "id": config.server_id,
                "transport": {
                    "type": "stdio",
                    "executable": config.command,
                    "arguments": config.arguments
                },
                "inheritedEnvironment": config.inherited_environment,
                "tools": [{
                    "remoteName": config.remote_tool,
                    "alias": config.tool_alias,
                    "declaration": {
                        "effects": ["fs.read"],
                        "idempotency": "idempotent",
                        "retrySafety": "never",
                        "concurrency": "serial",
                        "timeoutMillis": 10000
                    }
                }]
            }]
        });
        fs::write(
            config_dir.join("settings.json"),
            serde_json::to_vec_pretty(&settings).unwrap(),
        )
        .unwrap();
    }
}

impl Drop for LiveWorkspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn environment_value(name: &str) -> Option<String> {
    let dotenv = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.env");
    let values = if dotenv.exists() {
        load_env_file(&dotenv).unwrap_or_default()
    } else {
        BTreeMap::new()
    };
    std::env::var(name)
        .ok()
        .or_else(|| values.get(name).cloned())
}

fn required_value(name: &str) -> Option<String> {
    match environment_value(name) {
        Some(value) if !value.is_empty() => Some(value),
        _ => {
            eprintln!("skipping MCP live smoke: {name} is required");
            None
        }
    }
}

fn live_config() -> Result<Option<LiveMcpConfig>, String> {
    if environment_value(LIVE_GATE).as_deref() != Some("1") {
        eprintln!("skipping MCP live smoke: set {LIVE_GATE}=1 to opt in");
        return Ok(None);
    }

    let Some(command) = required_value(LIVE_COMMAND) else {
        return Ok(None);
    };
    let Some(server_id) = required_value(LIVE_SERVER_ID) else {
        return Ok(None);
    };
    let Some(remote_tool) = required_value(LIVE_REMOTE_TOOL) else {
        return Ok(None);
    };
    let Some(tool_alias) = required_value(LIVE_TOOL_ALIAS) else {
        return Ok(None);
    };
    let Some(tool_arguments) = required_value(LIVE_TOOL_ARGUMENTS) else {
        return Ok(None);
    };

    let command = PathBuf::from(command);
    if !command.is_absolute() {
        return Err(format!(
            "{LIVE_COMMAND} must be an absolute executable path"
        ));
    }

    let arguments = environment_value(LIVE_ARGUMENTS).unwrap_or_else(|| "[]".to_owned());
    let arguments = serde_json::from_str::<Vec<String>>(&arguments)
        .map_err(|_| format!("{LIVE_ARGUMENTS} must be a JSON string array"))?;
    let tool_arguments = serde_json::from_str::<Value>(&tool_arguments)
        .map_err(|_| format!("{LIVE_TOOL_ARGUMENTS} must be a JSON object"))?;
    if !tool_arguments.is_object() {
        return Err(format!("{LIVE_TOOL_ARGUMENTS} must be a JSON object"));
    }

    let inherited_environment =
        environment_value(LIVE_INHERITED_ENVIRONMENT).unwrap_or_else(|| "[]".to_owned());
    let inherited_environment = serde_json::from_str::<Vec<String>>(&inherited_environment)
        .map_err(|_| format!("{LIVE_INHERITED_ENVIRONMENT} must be a JSON string array"))?;

    Ok(Some(LiveMcpConfig {
        command,
        arguments,
        server_id,
        remote_tool,
        tool_alias,
        tool_arguments,
        inherited_environment,
    }))
}

fn child_environment(config: &LiveMcpConfig) -> Result<BTreeMap<String, String>, String> {
    config
        .inherited_environment
        .iter()
        .map(|name| {
            environment_value(name)
                .map(|value| (name.clone(), value))
                .ok_or_else(|| format!("configured MCP environment variable {name} is missing"))
        })
        .collect()
}

fn provider(alias: &str, arguments: Value) -> Arc<ScriptedModelProvider> {
    let model = ModelSpec::new(
        ModelId::from_str("fake/model").unwrap(),
        ProviderId::from_str("fake").unwrap(),
        ModelDisplayName::from_str("MCP Live Smoke Model").unwrap(),
        TokenCount::new(32_000).unwrap(),
        TokenCount::new(4_000).unwrap(),
        ModelCapabilities::text().with_tools(true),
    )
    .unwrap();
    let index = ModelStreamIndex::new(0).unwrap();
    let call_id = ProviderToolCallId::from_str("mcp-live-call").unwrap();
    Arc::new(ScriptedModelProvider::new(
        ProviderId::from_str("fake").unwrap(),
        vec![model],
        [
            ScriptedModelResponse::events([
                ModelEvent::Started(ModelResponseInfo::new()),
                ModelEvent::ToolCallStarted(
                    ToolCallStarted::new(index, call_id.clone(), alias).unwrap(),
                ),
                ModelEvent::ToolCallCompleted(
                    ToolCallCompleted::new(index, call_id, alias, arguments).unwrap(),
                ),
                ModelEvent::Completed(ModelCompletion::new(StopReason::ToolUse).unwrap()),
            ]),
            ScriptedModelResponse::text([FINAL_MARKER]),
        ],
    ))
}

fn cli_args(workspace: &LiveWorkspace, selection: &str) -> CliArgs {
    CliArgs::try_parse_from([
        "tea",
        selection,
        "--provider",
        "fake",
        "--model",
        "fake/model",
        "--trust",
        "ignore",
        "--cwd",
        workspace.workspace().to_str().unwrap(),
        "--config-dir",
        workspace.root.join("config").to_str().unwrap(),
        "--state-dir",
        workspace.root.join("state").to_str().unwrap(),
        "--data-dir",
        workspace.root.join("data").to_str().unwrap(),
    ])
    .unwrap()
}

fn bootstrap(
    workspace: &LiveWorkspace,
    values: BTreeMap<String, String>,
    provider: Arc<dyn ModelProvider>,
) -> CliBootstrap {
    CliBootstrap::new(BootstrapEnvironment::new(
        workspace.workspace(),
        Some(workspace.root.join("home")),
        values,
    ))
    .with_provider(provider)
}

fn assert_mcp_durable_records(records: &[RecordEnvelope], config: &LiveMcpConfig) {
    let requested = records
        .iter()
        .find(|record| {
            matches!(
                record.record(),
                SessionRecord::ToolCallRequested { tool_name, .. } if tool_name == &config.tool_alias
            )
        })
        .expect("the configured MCP tool request must be durable");
    let audit = requested
        .metadata()
        .get(TOOL_AUDIT_METADATA_NAMESPACE)
        .expect("the MCP tool request must retain audit metadata");
    assert_eq!(audit["source"]["kind"], json!("mcp"));
    assert_eq!(
        audit["source"]["sourceId"],
        json!(format!("mcp.{}", config.server_id))
    );
    assert_eq!(audit["source"]["trust"], json!("user"));
    assert!(
        audit["source"]["descriptorDigest"]
            .as_str()
            .is_some_and(|digest| digest.len() == 64),
        "the frozen descriptor digest must be durable"
    );
    assert!(
        audit["resources"]
            .as_array()
            .is_some_and(|resources| resources.iter().any(|resource| {
                resource["scheme"] == "mcp-server" && resource["access"] == "execute"
            })),
        "the host-created MCP execute resource must be audited"
    );
    assert!(
        records.iter().any(|record| {
            matches!(
                record.record(),
                SessionRecord::ToolExecutionStarted {
                    execution_target: ExecutionTarget::Mcp,
                    ..
                }
            )
        }),
        "MCP dispatch must use the durable MCP execution target"
    );
    assert!(
        records
            .iter()
            .any(|record| record.record_type() == SessionRecordType::ApprovalResolved),
        "the live MCP call must have a durable approval decision"
    );
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires TEA_MCP_LIVE_SMOKE=1 and an explicit local MCP stdio command"]
async fn configured_mcp_tool_is_approved_audited_and_never_replayed_after_reopen() {
    let Some(config) = live_config().unwrap() else {
        return;
    };
    let values = child_environment(&config).unwrap();
    let workspace = LiveWorkspace::new();
    workspace.write_settings(&config);

    let scripted = provider(&config.tool_alias, config.tool_arguments.clone());
    let model_provider: Arc<dyn ModelProvider> = scripted.clone();
    let initial = bootstrap(&workspace, values.clone(), model_provider);
    let (service, _) = initial
        .build_async(&cli_args(&workspace, "--new"))
        .await
        .unwrap();

    let catalog = service.mcp_snapshot().unwrap();
    assert_eq!(catalog.servers().len(), 1);
    assert_eq!(catalog.servers()[0].server_id().as_str(), config.server_id);
    assert_eq!(catalog.catalog().len(), 1);
    assert_eq!(catalog.catalog()[0].tool_name().as_str(), config.tool_alias);

    let session_id = service.create_session().await.unwrap();
    service
        .prompt(
            session_id,
            "Call the configured MCP tool with the scripted arguments.",
            None,
        )
        .await
        .unwrap();
    let approval = match service.wait(session_id).await.unwrap() {
        RuntimeCommandOutcome::RunCompleted {
            pending_approval_id: Some(approval_id),
            ..
        } => approval_id,
        outcome => panic!("expected an MCP approval, got {outcome:?}"),
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
    assert_eq!(scripted.remaining_scripts().unwrap(), 0);

    let before = service.session_snapshot(session_id).await.unwrap();
    let before_records = serde_json::to_value(before.records()).unwrap();
    assert_mcp_durable_records(before.records(), &config);
    assert!(
        serde_json::to_string(before.records())
            .unwrap()
            .contains(FINAL_MARKER),
        "the second scripted turn must produce the final response"
    );
    let branch = before.state().active_branch_id();
    service.shutdown().await;
    drop(service);

    let reopened_provider: Arc<dyn ModelProvider> = scripted;
    let reopened_bootstrap = bootstrap(&workspace, values, reopened_provider);
    let (reopened, _) = reopened_bootstrap
        .build_async(&cli_args(&workspace, "--continue"))
        .await
        .unwrap();
    reopened.open_session(session_id).await.unwrap();
    let after = reopened.session_snapshot(session_id).await.unwrap();
    assert_eq!(after.state().active_branch_id(), branch);
    assert_eq!(
        serde_json::to_value(after.records()).unwrap(),
        before_records
    );
    assert_mcp_durable_records(after.records(), &config);
    reopened.shutdown().await;
}
