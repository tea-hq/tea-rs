use std::collections::BTreeMap;
use std::fs;
use std::str::FromStr as _;
use std::sync::Arc;
use std::time::Duration;

use clap::Parser as _;
use tea_cli::args::{CliArgs, SessionSelection};
use tea_cli::rpc::{
    MAX_RPC_FRAME_BYTES, RPC_VERSION, RpcError, RpcErrorCode, RpcFrameReader, RpcLineWriter,
    RpcOutput, RpcReadError, RpcRequest, RpcResponse,
};
use tea_cli::{BootstrapEnvironment, CliBootstrap};
use tea_model::{
    ModelCapabilities, ModelCompletion, ModelDisplayName, ModelEvent, ModelResponseInfo, ModelSpec,
    ModelStreamIndex, ProviderId, ProviderToolCallId, ToolCallCompleted, ToolCallStarted,
};
use tea_protocol::{ModelId, SessionId, StopReason, TokenCount};
use tea_testkit::{ScriptStep, ScriptedModelProvider, ScriptedModelResponse};
use tokio::io::{AsyncBufReadExt as _, BufReader, DuplexStream};
use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

#[tokio::test(flavor = "current_thread")]
async fn reader_uses_only_lf_and_accepts_crlf_without_unicode_splitting() {
    let (mut client, server) = tokio::io::duplex(4096);
    let input = concat!(
        "{\"rpcVersion\":\"1.0\",\"id\":\"a\",\"type\":\"prompt\",",
        "\"payload\":{\"text\":\"one\\u2028two\\u2029three\"}}\r\n",
        "{\"rpcVersion\":\"1.0\",\"id\":\"b\",\"type\":\"query_state\",\"payload\":{}}\n"
    );
    client.write_all(input.as_bytes()).await.unwrap();
    client.shutdown().await.unwrap();

    let mut reader = RpcFrameReader::new(server);
    let first = reader.read_frame().await.unwrap().unwrap();
    let second = reader.read_frame().await.unwrap().unwrap();
    assert!(!first.ends_with(b"\r"));
    assert!(String::from_utf8(first).unwrap().contains("\\u2028"));
    assert_eq!(
        serde_json::from_slice::<RpcRequest>(&second)
            .unwrap()
            .id()
            .unwrap()
            .as_str(),
        "b"
    );
    assert_eq!(reader.read_frame().await.unwrap(), None);
}

#[tokio::test(flavor = "current_thread")]
async fn malformed_complete_frame_does_not_poison_the_next_request() {
    let (mut client, server) = tokio::io::duplex(1024);
    client
        .write_all(
            b"{malformed}\n{\"rpcVersion\":\"1.0\",\"type\":\"query_state\",\"payload\":{}}\n",
        )
        .await
        .unwrap();
    client.shutdown().await.unwrap();
    let mut reader = RpcFrameReader::new(server);
    let malformed = reader.read_frame().await.unwrap().unwrap();
    assert!(serde_json::from_slice::<RpcRequest>(&malformed).is_err());
    let valid = reader.read_frame().await.unwrap().unwrap();
    assert!(serde_json::from_slice::<RpcRequest>(&valid).is_ok());
}

#[tokio::test(flavor = "current_thread")]
async fn oversized_and_unterminated_frames_are_terminal() {
    let (mut client, server) = tokio::io::duplex(MAX_RPC_FRAME_BYTES + 32);
    let writer = tokio::spawn(async move {
        client
            .write_all(&vec![b'x'; MAX_RPC_FRAME_BYTES + 1])
            .await
            .unwrap();
        client.shutdown().await.unwrap();
    });
    let mut reader = RpcFrameReader::new(server);
    assert_eq!(reader.read_frame().await, Err(RpcReadError::Oversize));
    writer.await.unwrap();

    let (mut client, server) = tokio::io::duplex(32);
    client.write_all(b"{}").await.unwrap();
    client.shutdown().await.unwrap();
    let mut reader = RpcFrameReader::new(server);
    assert_eq!(reader.read_frame().await, Err(RpcReadError::Unterminated));
}

#[test]
fn versioned_fixture_is_strict_and_request_ids_round_trip() {
    let requests = include_str!("fixtures/rpc/requests.jsonl")
        .lines()
        .map(|line| serde_json::from_str::<RpcRequest>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[0].id().unwrap().as_str(), "state-1");
    assert!(
        requests
            .into_iter()
            .all(|request| request.into_parts().is_ok())
    );

    let unsupported: RpcRequest =
        serde_json::from_str(r#"{"rpcVersion":"2.0","type":"query_state","payload":{}}"#).unwrap();
    assert_eq!(
        unsupported.into_parts().unwrap_err().code(),
        RpcErrorCode::UnsupportedVersion
    );
    assert!(
        serde_json::from_str::<RpcRequest>(
            r#"{"rpcVersion":"1.0","type":"query_state","payload":{},"extra":true}"#
        )
        .is_err()
    );
}

#[test]
fn mcp_identity_errors_require_a_rebuild_without_adapter_diagnostics() {
    let error: RpcError = tea_mcp::McpError::new(tea_mcp::McpErrorCode::Identity).into();
    assert_eq!(error.code(), RpcErrorCode::InvalidRequest);
    assert_eq!(
        serde_json::to_value(error).unwrap(),
        serde_json::json!({
            "code": "invalid_request",
            "message": "MCP catalog changed; close and rebuild the CLI service"
        })
    );
}

#[tokio::test(flavor = "current_thread")]
async fn writer_emits_one_compact_flushed_response_per_lf() {
    let (server, mut client) = tokio::io::duplex(4096);
    let writer = RpcLineWriter::spawn_with_deadline(server, Duration::from_secs(1));
    let output = RpcOutput::error(
        None,
        RpcError::new(RpcErrorCode::ParseError, "request JSON is malformed"),
    );
    writer.write(&output).await.unwrap();
    writer.shutdown().await.unwrap();

    let mut bytes = Vec::new();
    client.read_to_end(&mut bytes).await.unwrap();
    assert!(bytes.ends_with(b"\n"));
    assert!(!bytes[..bytes.len() - 1].contains(&b'\n'));
    let value: serde_json::Value = serde_json::from_slice(&bytes[..bytes.len() - 1]).unwrap();
    assert_eq!(value["rpcVersion"], RPC_VERSION);
    assert_eq!(value["type"], "response");
    assert_eq!(value["payload"]["type"], "error");

    let _ = RpcResponse::Models { models: Vec::new() };
}

fn temp_root(label: &str) -> std::path::PathBuf {
    let root = std::env::temp_dir().join(format!(
        "tea-rpc-{label}-{}",
        uuid::Uuid::now_v7().hyphenated()
    ));
    fs::create_dir_all(&root).unwrap();
    root
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

fn bootstrap(root: &std::path::Path, provider: Arc<ScriptedModelProvider>) -> CliBootstrap {
    CliBootstrap::new(BootstrapEnvironment::new(
        root,
        Some(root.to_path_buf()),
        BTreeMap::new(),
    ))
    .with_provider(provider)
}

fn rpc_args(root: &std::path::Path, session_id: Option<SessionId>) -> CliArgs {
    let mut values = vec![
        "tea".to_owned(),
        "--rpc".to_owned(),
        "--provider".to_owned(),
        "fake".to_owned(),
        "--model".to_owned(),
        "fake/model".to_owned(),
        "--trust".to_owned(),
        "ignore".to_owned(),
        "--cwd".to_owned(),
        root.display().to_string(),
        "--config-dir".to_owned(),
        root.join("config").display().to_string(),
        "--state-dir".to_owned(),
        root.join("state").display().to_string(),
        "--data-dir".to_owned(),
        root.join("data").display().to_string(),
    ];
    if let Some(session_id) = session_id {
        values.push("--session".to_owned());
        values.push(session_id.to_string());
    } else {
        values.push("--new".to_owned());
    }
    CliArgs::try_parse_from(values).unwrap()
}

async fn send_json(input: &mut DuplexStream, value: serde_json::Value) {
    let mut bytes = serde_json::to_vec(&value).unwrap();
    bytes.push(b'\n');
    input.write_all(&bytes).await.unwrap();
    input.flush().await.unwrap();
}

async fn receive_json(output: &mut BufReader<DuplexStream>) -> serde_json::Value {
    let mut line = String::new();
    tokio::time::timeout(Duration::from_secs(3), output.read_line(&mut line))
        .await
        .expect("RPC response deadline")
        .unwrap();
    assert!(line.ends_with('\n'));
    serde_json::from_str(&line).unwrap()
}

async fn receive_response(
    output: &mut BufReader<DuplexStream>,
    request_id: &str,
) -> serde_json::Value {
    loop {
        let line = receive_json(output).await;
        if line["id"] == request_id {
            return line;
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn mcp_host_queries_are_correlated_and_do_not_require_a_session_record() {
    let root = temp_root("mcp-host-query");
    let provider = Arc::new(ScriptedModelProvider::new(
        ProviderId::from_str("fake").unwrap(),
        vec![model()],
        [],
    ));
    let args = rpc_args(&root, None);
    let bootstrap = bootstrap(&root, provider);
    let (service, selection) = bootstrap.build(&args).unwrap();
    let (mut client_input, server_input) = tokio::io::duplex(64 * 1024);
    let (server_output, client_output) = tokio::io::duplex(64 * 1024);
    let server = tea_cli::rpc::run_service(&service, selection, server_input, server_output);
    let client = async {
        let mut output = BufReader::new(client_output);
        let ready = receive_json(&mut output).await;
        assert_eq!(ready["type"], "ready");
        send_json(
            &mut client_input,
            serde_json::json!({
                "rpcVersion":"1.0","id":"mcp-list","type":"list_mcp_servers","payload":{}
            }),
        )
        .await;
        let listed = receive_response(&mut output, "mcp-list").await;
        assert_eq!(listed["payload"]["type"], "mcp_servers");
        assert_eq!(listed["payload"]["data"]["servers"], serde_json::json!([]));

        send_json(
            &mut client_input,
            serde_json::json!({
                "rpcVersion":"1.0","id":"mcp-reconnect","type":"reconnect_mcp",
                "payload":{"serverId":"fixture"}
            }),
        )
        .await;
        let reconnect = receive_response(&mut output, "mcp-reconnect").await;
        assert_eq!(reconnect["payload"]["type"], "error");
        assert_eq!(reconnect["payload"]["data"]["error"]["code"], "busy");
        client_input.shutdown().await.unwrap();
    };
    let (server, ()) = tokio::join!(server, client);
    assert!(server.is_ok());
    service.shutdown().await;
    fs::remove_dir_all(root).unwrap();
}

async fn drain_output(output: &mut BufReader<DuplexStream>) {
    let mut trailing = String::new();
    while output.read_line(&mut trailing).await.unwrap() != 0 {
        trailing.clear();
    }
}

#[tokio::test(flavor = "current_thread")]
#[allow(clippy::too_many_lines)] // One connection/restart sequence is clearer as one scenario.
async fn client_steers_queries_disconnects_and_resumes_from_durable_cursor() {
    let root = temp_root("resume");
    let provider = Arc::new(ScriptedModelProvider::new(
        ProviderId::from_str("fake").unwrap(),
        vec![model()],
        [ScriptedModelResponse::new([
            ScriptStep::event(ModelEvent::Started(ModelResponseInfo::new())),
            ScriptStep::AwaitCancellation,
        ])],
    ));
    let args = rpc_args(&root, None);
    let first_bootstrap = bootstrap(&root, Arc::clone(&provider));
    let (service, selection) = first_bootstrap.build(&args).unwrap();
    assert_eq!(selection, SessionSelection::New);
    let (mut client_input, server_input) = tokio::io::duplex(64 * 1024);
    let (server_output, client_output) = tokio::io::duplex(64 * 1024);

    let server = tea_cli::rpc::run_service(&service, selection, server_input, server_output);
    let client = async {
        let mut output = BufReader::new(client_output);
        let ready = receive_json(&mut output).await;
        assert_eq!(ready["type"], "ready");
        let session_id = SessionId::from_str(ready["sessionId"].as_str().unwrap()).unwrap();
        send_json(
            &mut client_input,
            serde_json::json!({
                "rpcVersion":"1.0","id":"prompt","type":"prompt","payload":{"text":"work"}
            }),
        )
        .await;

        let mut accepted = false;
        let mut started = false;
        while !accepted || !started {
            let line = receive_json(&mut output).await;
            accepted |= line["id"] == "prompt" && line["payload"]["type"] == "command_accepted";
            started |= line["type"] == "event" && line["payload"]["type"] == "run_started";
        }

        for request in [
            serde_json::json!({"rpcVersion":"1.0","id":"steer","type":"steer","payload":{"text":"adjust"}}),
            serde_json::json!({"rpcVersion":"1.0","id":"follow","type":"follow_up","payload":{"text":"later"}}),
            serde_json::json!({"rpcVersion":"1.0","id":"state","type":"query_state","payload":{}}),
            serde_json::json!({"rpcVersion":"1.0","id":"abort","type":"abort","payload":{}}),
        ] {
            send_json(&mut client_input, request).await;
        }

        let mut response_ids = std::collections::BTreeSet::new();
        let mut finished = false;
        while response_ids.len() < 4 || !finished {
            let line = receive_json(&mut output).await;
            if let Some(id) = line.get("id").and_then(serde_json::Value::as_str) {
                response_ids.insert(id.to_owned());
            }
            finished |= line["type"] == "command_finished";
        }
        let expected = ["abort", "follow", "state", "steer"]
            .map(str::to_owned)
            .into_iter()
            .collect();
        assert_eq!(response_ids, expected);

        send_json(
            &mut client_input,
            serde_json::json!({
                "rpcVersion":"1.0","id":"snapshot","type":"query_snapshot","payload":{"limit":64}
            }),
        )
        .await;
        let snapshot = loop {
            let line = receive_json(&mut output).await;
            if line["id"] == "snapshot" {
                break line;
            }
        };
        let page = &snapshot["payload"]["data"]["snapshot"];
        assert!(!page["records"].as_array().unwrap().is_empty());
        let tail = page["tailSequence"].as_str().unwrap().to_owned();
        client_input.shutdown().await.unwrap();
        drain_output(&mut output).await;
        (session_id, tail)
    };
    let (server_result, (session_id, tail)) = tokio::join!(server, client);
    server_result.unwrap();
    service.shutdown().await;

    let reconnect_args = rpc_args(&root, Some(session_id));
    let reconnect_bootstrap = bootstrap(&root, provider);
    let (service, selection) = reconnect_bootstrap.build(&reconnect_args).unwrap();
    let (mut client_input, server_input) = tokio::io::duplex(16 * 1024);
    let (server_output, client_output) = tokio::io::duplex(16 * 1024);
    let server = tea_cli::rpc::run_service(&service, selection, server_input, server_output);
    let client = async {
        let mut output = BufReader::new(client_output);
        let ready = receive_json(&mut output).await;
        assert_eq!(ready["sessionId"], session_id.to_string());
        send_json(
            &mut client_input,
            serde_json::json!({
                "rpcVersion":"1.0","id":"resume","type":"query_snapshot",
                "payload":{"afterSequence":tail,"limit":64}
            }),
        )
        .await;
        let resumed = receive_json(&mut output).await;
        assert_eq!(resumed["id"], "resume");
        assert!(
            resumed["payload"]["data"]["snapshot"]["records"]
                .as_array()
                .unwrap()
                .is_empty()
        );
        client_input.shutdown().await.unwrap();
        drain_output(&mut output).await;
    };
    let (server_result, ()) = tokio::join!(server, client);
    server_result.unwrap();
    service.shutdown().await;
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn approval_request_is_correlated_resolved_and_completed_asynchronously() {
    let root = temp_root("approval");
    fs::write(root.join("file.txt"), "old").unwrap();
    let index = ModelStreamIndex::new(0).unwrap();
    let opaque = ProviderToolCallId::from_str("edit-rpc").unwrap();
    let provider = Arc::new(ScriptedModelProvider::new(
        ProviderId::from_str("fake").unwrap(),
        vec![model()],
        [
            ScriptedModelResponse::events([
                ModelEvent::Started(ModelResponseInfo::new()),
                ModelEvent::ToolCallStarted(
                    ToolCallStarted::new(index, opaque.clone(), "edit").unwrap(),
                ),
                ModelEvent::ToolCallCompleted(
                    ToolCallCompleted::new(
                        index,
                        opaque,
                        "edit",
                        serde_json::json!({
                            "path":"file.txt","oldText":"old","newText":"new"
                        }),
                    )
                    .unwrap(),
                ),
                ModelEvent::Completed(ModelCompletion::new(StopReason::ToolUse).unwrap()),
            ]),
            ScriptedModelResponse::text(["denial recorded"]),
        ],
    ));
    let args = rpc_args(&root, None);
    let rpc_bootstrap = bootstrap(&root, provider);
    let (service, selection) = rpc_bootstrap.build(&args).unwrap();
    let (mut client_input, server_input) = tokio::io::duplex(64 * 1024);
    let (server_output, client_output) = tokio::io::duplex(64 * 1024);
    let server = tea_cli::rpc::run_service(&service, selection, server_input, server_output);
    let client = async {
        let mut output = BufReader::new(client_output);
        assert_eq!(receive_json(&mut output).await["type"], "ready");
        send_json(
            &mut client_input,
            serde_json::json!({
                "rpcVersion":"1.0","id":"prompt","type":"prompt","payload":{"text":"edit it"}
            }),
        )
        .await;
        let mut approval_id = None;
        let mut first_finished = false;
        while approval_id.is_none() || !first_finished {
            let line = receive_json(&mut output).await;
            if line["type"] == "event" && line["payload"]["type"] == "approval_requested" {
                approval_id = line["payload"]["payload"]["approvalId"]
                    .as_str()
                    .map(str::to_owned);
            }
            first_finished |= line["type"] == "command_finished";
        }
        send_json(
            &mut client_input,
            serde_json::json!({
                "rpcVersion":"1.0","id":"approval","type":"resolve_approval",
                "payload":{"approvalId":approval_id.unwrap(),"decision":{"type":"deny"}}
            }),
        )
        .await;
        let mut accepted = false;
        let mut second_finished = false;
        while !accepted || !second_finished {
            let line = receive_json(&mut output).await;
            accepted |= line["id"] == "approval" && line["payload"]["type"] == "command_accepted";
            second_finished |= line["type"] == "command_finished";
        }
        send_json(
            &mut client_input,
            serde_json::json!({
                "rpcVersion":"1.0","id":"state","type":"query_state","payload":{}
            }),
        )
        .await;
        let state = receive_response(&mut output, "state").await;
        assert_eq!(state["id"], "state");
        assert!(
            state["payload"]["data"]["state"]
                .get("pendingApprovalId")
                .is_none()
        );
        client_input.shutdown().await.unwrap();
        drain_output(&mut output).await;
    };
    let (server_result, ()) = tokio::join!(server, client);
    server_result.unwrap();
    service.shutdown().await;
    assert_eq!(fs::read_to_string(root.join("file.txt")).unwrap(), "old");
    fs::remove_dir_all(root).unwrap();
}
