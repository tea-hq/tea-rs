use std::process::Stdio;

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt as _, AsyncWriteExt as _, BufReader};
use tokio::process::Command;

#[tokio::test(flavor = "current_thread")]
async fn binary_accepts_handshake_and_shutdown_without_bootstrapping_provider() {
    let root =
        std::env::temp_dir().join(format!("tea-app-server-process-{}", uuid::Uuid::now_v7()));
    std::fs::create_dir_all(&root).expect("create app-server workspace");
    let mut child = Command::new(env!("CARGO_BIN_EXE_tea"))
        .arg("--app-server")
        .arg("--cwd")
        .arg(&root)
        .arg("--config-dir")
        .arg(root.join("config"))
        .arg("--state-dir")
        .arg(root.join("state"))
        .arg("--data-dir")
        .arg(root.join("data"))
        .env("TEA_PROVIDER", "unsupported-test-provider")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start tea app-server");
    let mut input = child.stdin.take().expect("app-server stdin");
    let stdout = child.stdout.take().expect("app-server stdout");
    let mut output = BufReader::new(stdout);

    send(
        &mut input,
        json!({
            "jsonrpc": "2.0",
            "id": "init",
            "method": "server/initialize",
            "params": {
                "clientName": "app-server-process-test",
                "clientVersion": "0.1.0",
                "appServerVersion": "1.0"
            }
        }),
    )
    .await;
    let initialized = receive(&mut output).await;
    assert_eq!(initialized["id"], "init");
    assert_eq!(initialized["result"]["appServerVersion"], "1.0");
    assert_eq!(initialized["result"]["capabilities"]["sessionResume"], true);

    send(
        &mut input,
        json!({
            "jsonrpc": "2.0",
            "id": "session",
            "method": "session/create",
            "params": { "cwd": root, "mcpServers": [] }
        }),
    )
    .await;
    let session = receive(&mut output).await;
    assert!(session["result"]["sessionId"].as_str().is_some());

    send(
        &mut input,
        json!({
            "jsonrpc": "2.0",
            "id": "unknown",
            "method": "server/unknown",
            "params": {}
        }),
    )
    .await;
    let unknown = receive(&mut output).await;
    assert_eq!(unknown["id"], "unknown");
    assert_eq!(unknown["error"]["code"], -32601);

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
    let stopped = receive(&mut output).await;
    assert_eq!(stopped["id"], "stop");
    assert_eq!(stopped["result"]["stopping"], true);
    drop(input);

    let status = tokio::time::timeout(std::time::Duration::from_secs(3), child.wait())
        .await
        .expect("app-server exits after shutdown")
        .expect("app-server wait");
    assert!(status.success(), "unexpected app-server status: {status}");
    std::fs::remove_dir_all(root).expect("remove app-server workspace");
}

async fn send(input: &mut tokio::process::ChildStdin, value: Value) {
    let mut line = serde_json::to_vec(&value).expect("request serializes");
    line.push(b'\n');
    input.write_all(&line).await.expect("request writes");
    input.flush().await.expect("request flushes");
}

async fn receive(output: &mut BufReader<tokio::process::ChildStdout>) -> Value {
    let mut line = String::new();
    tokio::time::timeout(
        std::time::Duration::from_secs(3),
        output.read_line(&mut line),
    )
    .await
    .expect("response deadline")
    .expect("response reads");
    serde_json::from_str(line.trim_end()).expect("response JSON")
}
