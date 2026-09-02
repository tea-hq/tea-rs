#![cfg(feature = "fixture-server")]
#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::fs;
use std::str::FromStr;
use std::sync::Arc;

use clap::Parser as _;
use serde_json::{Value, json};
use tea_cli::app_server;
use tea_cli::args::CliArgs;
use tea_cli::{BootstrapEnvironment, CliBootstrap};
use tea_model::{
    ModelCapabilities, ModelCompletion, ModelDisplayName, ModelEvent, ModelResponseInfo, ModelSpec,
    ModelStreamIndex, ProviderId, ProviderToolCallId, ToolCallCompleted, ToolCallStarted,
};
use tea_protocol::{ModelId, StopReason, TokenCount};
use tea_testkit::{ScriptedModelProvider, ScriptedModelResponse};
use tokio::io::{AsyncBufReadExt as _, AsyncWriteExt as _, BufReader, DuplexStream};

const FIXTURE: &str = env!("CARGO_BIN_EXE_tea-cli-mcp-fixture-server");

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

fn provider() -> Arc<ScriptedModelProvider> {
    let index = ModelStreamIndex::new(0).unwrap();
    let call_id = ProviderToolCallId::from_str("mcp-call").unwrap();
    Arc::new(ScriptedModelProvider::new(
        ProviderId::from_str("fake").unwrap(),
        vec![model()],
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
                        json!({"value": "hello"}),
                    )
                    .unwrap(),
                ),
                ModelEvent::Completed(ModelCompletion::new(StopReason::ToolUse).unwrap()),
            ]),
            ScriptedModelResponse::text(["MCP app-server done"]),
        ],
    ))
}

fn args(root: &std::path::Path) -> CliArgs {
    CliArgs::try_parse_from([
        "tea",
        "--app-server",
        "--provider",
        "fake",
        "--model",
        "fake/model",
        "--trust",
        "ignore",
        "--cwd",
        root.to_str().unwrap(),
        "--config-dir",
        root.join("config").to_str().unwrap(),
        "--state-dir",
        root.join("state").to_str().unwrap(),
        "--data-dir",
        root.join("data").to_str().unwrap(),
    ])
    .unwrap()
}

async fn send(input: &mut DuplexStream, value: Value) {
    let mut bytes = serde_json::to_vec(&value).unwrap();
    bytes.push(b'\n');
    input.write_all(&bytes).await.unwrap();
    input.flush().await.unwrap();
}

async fn receive(output: &mut BufReader<DuplexStream>) -> Value {
    let mut line = String::new();
    tokio::time::timeout(
        std::time::Duration::from_secs(5),
        output.read_line(&mut line),
    )
    .await
    .unwrap()
    .unwrap();
    serde_json::from_str(line.trim_end()).unwrap()
}

#[tokio::test(flavor = "current_thread")]
#[allow(clippy::too_many_lines)]
async fn app_server_accepts_session_mcp_and_bridges_approval() {
    let root = std::env::temp_dir().join(format!("tea-app-server-mcp-{}", uuid::Uuid::now_v7()));
    fs::create_dir_all(&root).unwrap();
    let marker = root.join("mcp.marker");
    let provider = provider();
    let bootstrap = CliBootstrap::new(BootstrapEnvironment::new(
        &root,
        Some(root.clone()),
        BTreeMap::new(),
    ))
    .with_provider(Arc::clone(&provider) as Arc<dyn tea_model::ModelProvider>);
    let cli_args = args(&root);
    let (mut input, server_input) = tokio::io::duplex(64 * 1024);
    let (server_output, client_output) = tokio::io::duplex(64 * 1024);
    let server_args = cli_args.clone();
    let server_bootstrap = bootstrap.clone();
    let server = tokio::spawn(async move {
        Box::pin(app_server::run(
            &server_args,
            &server_bootstrap,
            server_input,
            server_output,
        ))
        .await
    });
    let mut output = BufReader::new(client_output);

    send(
        &mut input,
        json!({
            "jsonrpc": "2.0",
            "id": "init",
            "method": "server/initialize",
            "params": {
                "clientName": "app-server-mcp-test",
                "clientVersion": "0.1.0",
                "appServerVersion": "1.0"
            }
        }),
    )
    .await;
    let initialized = receive(&mut output).await;
    assert_eq!(initialized["result"]["capabilities"]["sessionMcp"], true);

    // Listing first lazily creates the base service. The subsequent session
    // must still rebuild it with the session-scoped MCP descriptor.
    send(
        &mut input,
        json!({
            "jsonrpc": "2.0",
            "id": "list",
            "method": "session/list",
            "params": {"cwd": root}
        }),
    )
    .await;
    let listed = receive(&mut output).await;
    assert!(
        listed.get("error").is_none(),
        "session list failed: {listed}"
    );

    send(
        &mut input,
        json!({
            "jsonrpc": "2.0",
            "id": "create",
            "method": "session/create",
            "params": {
                "cwd": root,
                "mcpServers": [{
                    "name": "fixture",
                    "command": FIXTURE,
                    "args": ["execute", "success", marker],
                    "env": []
                }]
            }
        }),
    )
    .await;
    let created = receive(&mut output).await;
    assert!(
        created.get("error").is_none(),
        "session creation failed: {created}"
    );
    let session_id = created["result"]["sessionId"].as_str().unwrap().to_owned();

    send(
        &mut input,
        json!({
            "jsonrpc": "2.0",
            "id": "prompt",
            "method": "session/prompt",
            "params": {"sessionId": session_id, "text": "call the MCP tool"}
        }),
    )
    .await;

    let mut saw_prompt_response = false;
    let mut saw_tool_call = false;
    let mut saw_approval = false;
    let mut saw_approval_response = false;
    let mut saw_text = false;
    let mut approval_id = None;
    let mut completed = false;
    for _ in 0..32 {
        let message = receive(&mut output).await;
        if message["id"] == "prompt" {
            assert_eq!(message["result"]["accepted"], true);
            saw_prompt_response = true;
        } else if message["id"] == "approval" {
            assert!(message.get("error").is_none(), "approval failed: {message}");
            saw_approval_response = true;
        } else if message["method"] == "event/session" {
            let event = &message["params"]["event"];
            let event_type = event["type"]
                .as_str()
                .or_else(|| event["payload"]["type"].as_str())
                .unwrap_or_default();
            match event_type {
                "tool_call_requested" => saw_tool_call = true,
                "approval_requested" => {
                    saw_approval = true;
                    let id = event["approvalId"]
                        .as_str()
                        .or_else(|| event["payload"]["approvalId"].as_str())
                        .unwrap()
                        .to_owned();
                    approval_id = Some(id.clone());
                    send(
                        &mut input,
                        json!({
                            "jsonrpc": "2.0",
                            "id": "approval",
                            "method": "approval/respond",
                            "params": {
                                "sessionId": session_id,
                                "approvalId": id,
                                "decision": {"type": "allow_once"}
                            }
                        }),
                    )
                    .await;
                }
                "message_delta" => {
                    let delta = event["payload"]["delta"].get("text");
                    if delta.and_then(Value::as_str) == Some("MCP app-server done") {
                        saw_text = true;
                    }
                }
                _ => {}
            }
        } else if message["method"] == "event/turn_completed"
            && message["params"]["status"] == "completed"
        {
            completed = true;
            break;
        }
    }

    assert!(saw_prompt_response);
    assert!(saw_tool_call);
    assert!(saw_approval);
    assert!(saw_approval_response);
    assert!(approval_id.is_some());
    assert!(saw_text);
    assert!(completed);
    assert_eq!(fs::read_to_string(&marker).unwrap(), "called\n");
    assert_eq!(provider.remaining_scripts().unwrap(), 0);

    send(
        &mut input,
        json!({
            "jsonrpc": "2.0",
            "id": "close",
            "method": "session/close",
            "params": {"sessionId": session_id}
        }),
    )
    .await;
    loop {
        let closed = receive(&mut output).await;
        if closed["id"] == "close" {
            assert_eq!(closed["result"]["closed"], true);
            break;
        }
    }
    send(
        &mut input,
        json!({
            "jsonrpc": "2.0",
            "id": "stop",
            "method": "server/shutdown",
            "params": {}
        }),
    )
    .await;
    loop {
        let stopped = receive(&mut output).await;
        if stopped["id"] == "stop" {
            assert_eq!(stopped["result"]["stopping"], true);
            break;
        }
    }
    drop(input);
    assert!(
        tokio::time::timeout(std::time::Duration::from_secs(5), server)
            .await
            .unwrap()
            .unwrap()
            .is_ok()
    );
    fs::remove_dir_all(root).unwrap();
}
