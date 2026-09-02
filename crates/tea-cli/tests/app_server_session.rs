use std::collections::BTreeMap;
use std::fs;
use std::str::FromStr;
use std::sync::Arc;

use clap::Parser as _;
use serde_json::{Value, json};
use tea_cli::app_server;
use tea_cli::args::CliArgs;
use tea_cli::{BootstrapEnvironment, CliBootstrap};
use tea_model::{ModelCapabilities, ModelDisplayName, ModelSpec, ProviderId};
use tea_protocol::{ModelId, TokenCount};
use tea_testkit::{ScriptedModelProvider, ScriptedModelResponse};
use tokio::io::{AsyncBufReadExt as _, AsyncWriteExt as _, BufReader, DuplexStream};

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
    .unwrap_or_else(|_| panic!("timed out waiting for app-server output"))
    .unwrap();
    serde_json::from_str(line.trim_end()).unwrap()
}

#[tokio::test(flavor = "current_thread")]
async fn app_server_routes_prompt_events_through_coding_service() {
    let root = std::env::temp_dir().join(format!("tea-app-session-{}", uuid::Uuid::now_v7()));
    fs::create_dir_all(&root).unwrap();
    let provider = Arc::new(ScriptedModelProvider::new(
        ProviderId::from_str("fake").unwrap(),
        vec![model()],
        [ScriptedModelResponse::text(["hello from tea"])],
    ));
    let bootstrap = CliBootstrap::new(BootstrapEnvironment::new(
        &root,
        Some(root.clone()),
        BTreeMap::new(),
    ))
    .with_provider(provider);
    let cli_args = args(&root);
    let (mut input, server_input) = tokio::io::duplex(64 * 1024);
    let (server_output, client_output) = tokio::io::duplex(64 * 1024);
    let server_args = cli_args.clone();
    let server_bootstrap = bootstrap.clone();
    let server = tokio::spawn(async move {
        app_server::run(&server_args, &server_bootstrap, server_input, server_output).await
    });
    let mut output = BufReader::new(client_output);

    send(
        &mut input,
        json!({"jsonrpc":"2.0","id":"init","method":"server/initialize","params":{"clientName":"test","clientVersion":"1","appServerVersion":"1.0"}}),
    )
    .await;
    assert_eq!(receive(&mut output).await["id"], "init");
    send(
        &mut input,
        json!({"jsonrpc":"2.0","id":"create","method":"session/create","params":{"cwd":root,"mcpServers":[]}}),
    )
    .await;
    let created = receive(&mut output).await;
    let session_id = created["result"]["sessionId"].as_str().unwrap().to_owned();
    assert_eq!(
        created["result"]["availableModels"][0]["providerId"],
        "fake"
    );
    assert_eq!(
        created["result"]["availableModels"][0]["modelId"],
        "fake/model"
    );
    send(
        &mut input,
        json!({"jsonrpc":"2.0","id":"prompt","method":"session/prompt","params":{"sessionId":session_id,"text":"say hello"}}),
    )
    .await;

    let mut saw_text = false;
    let mut saw_finished = false;
    for _ in 0..16 {
        let message = receive(&mut output).await;
        if message["id"] == "prompt" {
            assert_eq!(message["result"]["accepted"], true);
        }
        if message["method"] == "event/session"
            && message["params"]["event"]["type"] == "message_delta"
            && message["params"]["event"]["payload"]["delta"]["text"] == "hello from tea"
        {
            saw_text = true;
        }
        if message["method"] == "event/turn_completed" && message["params"]["status"] == "completed"
        {
            saw_finished = true;
            break;
        }
    }
    assert!(saw_text);
    assert!(saw_finished);
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
    let closed = loop {
        let message = receive(&mut output).await;
        if message["id"] == "close" {
            break message;
        }
    };
    assert_eq!(closed["id"], "close");
    assert_eq!(closed["result"]["closed"], true);

    // Closing the session releases the service owner. A subsequent session
    // must be created through a fresh runtime assembly rather than reusing the
    // previous session-scoped resources.
    send(
        &mut input,
        json!({
            "jsonrpc": "2.0",
            "id": "recreate",
            "method": "session/create",
            "params": {"cwd": root, "mcpServers": []}
        }),
    )
    .await;
    let recreated = receive(&mut output).await;
    let recreated_id = recreated["result"]["sessionId"].as_str().unwrap();
    assert_ne!(recreated_id, session_id);

    send(
        &mut input,
        json!({"jsonrpc":"2.0","id":"stop","method":"server/shutdown","params":{}}),
    )
    .await;
    let mut saw_shutdown = false;
    for _ in 0..16 {
        let response = receive(&mut output).await;
        if response["id"] == "stop" {
            assert_eq!(response["result"]["stopping"], true);
            saw_shutdown = true;
            break;
        }
    }
    assert!(saw_shutdown);
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

#[tokio::test(flavor = "current_thread")]
async fn app_server_cancels_an_owned_prompt_and_reports_terminal_status() {
    let root = std::env::temp_dir().join(format!("tea-app-cancel-{}", uuid::Uuid::now_v7()));
    fs::create_dir_all(&root).unwrap();
    let provider = Arc::new(ScriptedModelProvider::new(
        ProviderId::from_str("fake").unwrap(),
        vec![model()],
        [ScriptedModelResponse::await_cancellation()],
    ));
    let bootstrap = CliBootstrap::new(BootstrapEnvironment::new(
        &root,
        Some(root.clone()),
        BTreeMap::new(),
    ))
    .with_provider(provider);
    let cli_args = args(&root);
    let (mut input, server_input) = tokio::io::duplex(64 * 1024);
    let (server_output, client_output) = tokio::io::duplex(64 * 1024);
    let server_args = cli_args.clone();
    let server_bootstrap = bootstrap.clone();
    let server = tokio::spawn(async move {
        app_server::run(&server_args, &server_bootstrap, server_input, server_output).await
    });
    let mut output = BufReader::new(client_output);

    send(
        &mut input,
        json!({
            "jsonrpc": "2.0",
            "id": "init",
            "method": "server/initialize",
            "params": {
                "clientName": "app-server-cancel-test",
                "clientVersion": "0.1.0",
                "appServerVersion": "1.0"
            }
        }),
    )
    .await;
    assert_eq!(receive(&mut output).await["id"], "init");

    send(
        &mut input,
        json!({
            "jsonrpc": "2.0",
            "id": "create",
            "method": "session/create",
            "params": {"cwd": root, "mcpServers": []}
        }),
    )
    .await;
    let created = receive(&mut output).await;
    let session_id = created["result"]["sessionId"].as_str().unwrap().to_owned();

    send(
        &mut input,
        json!({
            "jsonrpc": "2.0",
            "id": "prompt",
            "method": "session/prompt",
            "params": {"sessionId": session_id, "text": "wait"}
        }),
    )
    .await;
    send(
        &mut input,
        json!({
            "jsonrpc": "2.0",
            "id": "cancel",
            "method": "session/cancel",
            "params": {"sessionId": session_id}
        }),
    )
    .await;

    let mut saw_prompt = false;
    let mut saw_cancel = false;
    let mut saw_cancelled = false;
    for _ in 0..32 {
        let message = receive(&mut output).await;
        if message["id"] == "prompt" {
            assert_eq!(message["result"]["accepted"], true);
            saw_prompt = true;
        } else if message["id"] == "cancel" {
            assert_eq!(message["result"]["accepted"], true);
            saw_cancel = true;
        } else if message["method"] == "event/turn_completed"
            && message["params"]["status"] == "cancelled"
        {
            saw_cancelled = true;
            break;
        }
    }
    assert!(saw_prompt);
    assert!(saw_cancel);
    assert!(saw_cancelled);

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
