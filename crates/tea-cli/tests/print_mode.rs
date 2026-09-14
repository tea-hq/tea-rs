use std::collections::BTreeMap;
use std::fs;
use std::io::{Read as _, Write as _};
use std::net::TcpListener;
use std::path::Path;
use std::process::{Command, Stdio};
use std::str::FromStr;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use clap::Parser as _;
use tea_cli::args::CliArgs;
use tea_cli::modes::print::MAX_INITIAL_PROMPT_BYTES;
use tea_cli::{BootstrapEnvironment, CliBootstrap, ExitCategory};
use tea_model::ModelProvider as _;
use tea_model::{
    ModelCapabilities, ModelCompletion, ModelDisplayName, ModelEvent, ModelResponseInfo, ModelSpec,
    ModelStreamIndex, ProviderId, ProviderToolCallId, ToolCallCompleted, ToolCallStarted,
};
use tea_protocol::{FinalOutputFormat, ModelId, StopReason, TokenCount};
use tea_testkit::{ScriptedModelProvider, ScriptedModelResponse};

static ID: AtomicU64 = AtomicU64::new(0);

fn temp_root(label: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "tea-cli-{label}-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

fn fake_provider(text: &str) -> Arc<ScriptedModelProvider> {
    let provider = ProviderId::from_str("fake").unwrap();
    let model = ModelSpec::new(
        ModelId::from_str("fake/model").unwrap(),
        provider.clone(),
        ModelDisplayName::from_str("Fake Model").unwrap(),
        TokenCount::new(32_000).unwrap(),
        TokenCount::new(4_000).unwrap(),
        ModelCapabilities::text()
            .with_tools(true)
            .with_final_json_schema_with_tools(),
    )
    .unwrap();
    Arc::new(ScriptedModelProvider::new(
        provider,
        vec![model],
        [ScriptedModelResponse::text([text])],
    ))
}

fn injected_bootstrap(root: &Path, text: &str) -> CliBootstrap {
    injected_bootstrap_with(root, fake_provider(text))
}

fn injected_bootstrap_with(root: &Path, provider: Arc<ScriptedModelProvider>) -> CliBootstrap {
    CliBootstrap::new(BootstrapEnvironment::new(
        root,
        Some(root.to_path_buf()),
        BTreeMap::new(),
    ))
    .with_provider(provider)
}

fn edit_provider() -> Arc<ScriptedModelProvider> {
    let provider = fake_provider("unused");
    let model = provider.models()[0].clone();
    let index = ModelStreamIndex::new(0).unwrap();
    let provider_call = ProviderToolCallId::from_str("edit-1").unwrap();
    Arc::new(ScriptedModelProvider::new(
        ProviderId::from_str("fake").unwrap(),
        vec![model],
        [ScriptedModelResponse::events([
            ModelEvent::Started(ModelResponseInfo::new()),
            ModelEvent::ToolCallStarted(
                ToolCallStarted::new(index, provider_call.clone(), "edit").unwrap(),
            ),
            ModelEvent::ToolCallCompleted(
                ToolCallCompleted::new(
                    index,
                    provider_call,
                    "edit",
                    serde_json::json!({"path":"file.txt","oldText":"old","newText":"new"}),
                )
                .unwrap(),
            ),
            ModelEvent::Completed(ModelCompletion::new(StopReason::ToolUse).unwrap()),
        ])],
    ))
}

#[tokio::test(flavor = "current_thread")]
async fn fake_print_mode_is_final_answer_only_and_supports_stdin_and_at_file() {
    let root = temp_root("fake");
    fs::write(root.join("prompt.md"), "from file").unwrap();
    let args = CliArgs::try_parse_from([
        "tea",
        "--print",
        "--no-session",
        "--provider",
        "fake",
        "--model",
        "fake/model",
        "--trust",
        "ignore",
        "@prompt.md",
    ])
    .unwrap();
    let bootstrap = injected_bootstrap(&root, "final answer");
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    Box::pin(tea_cli::modes::print::run(
        &args,
        &bootstrap,
        &mut "from stdin".as_bytes(),
        false,
        &mut stdout,
        &mut stderr,
    ))
    .await
    .unwrap();
    assert_eq!(stdout, b"final answer\n");
    assert!(stderr.is_empty());
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn print_mode_enforces_json_schema_with_active_tools() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": false
    });
    for (label, response, expected) in [
        (
            "schema-success",
            r#"{"answer":"tea"}"#,
            Some(r#"{"answer":"tea"}"#),
        ),
        ("schema-malformed", "not JSON", None),
    ] {
        let root = temp_root(label);
        fs::write(
            root.join("answer.schema.json"),
            serde_json::to_vec(&schema).unwrap(),
        )
        .unwrap();
        let args = CliArgs::try_parse_from([
            "tea",
            "--print",
            "--no-session",
            "--provider",
            "fake",
            "--model",
            "fake/model",
            "--trust",
            "ignore",
            "--cwd",
            root.to_str().unwrap(),
            "--output-schema",
            "answer.schema.json",
            "return the answer",
        ])
        .unwrap();
        let provider = fake_provider(response);
        let mut stdout = Vec::new();
        let mut stderr = Vec::new();
        let result = Box::pin(tea_cli::modes::print::run(
            &args,
            &injected_bootstrap_with(&root, Arc::clone(&provider)),
            &mut std::io::empty(),
            true,
            &mut stdout,
            &mut stderr,
        ))
        .await;
        if let Some(expected) = expected {
            result.unwrap();
            assert_eq!(stdout, format!("{expected}\n").as_bytes());
        } else {
            assert_eq!(result.unwrap_err().category(), ExitCategory::Provider);
            assert!(stdout.is_empty());
        }
        assert!(stderr.is_empty());
        let requests = provider.captured_requests().unwrap();
        assert_eq!(requests.len(), 1);
        assert!(!requests[0].tools().is_empty());
        assert_eq!(
            requests[0].final_output_format(),
            Some(&FinalOutputFormat::JsonSchema {
                schema: schema.clone()
            })
        );
        fs::remove_dir_all(root).unwrap();
    }
}

#[tokio::test(flavor = "current_thread")]
async fn print_mode_rejects_terminal_control_text_without_stdout() {
    let root = temp_root("control-output");
    let args = CliArgs::try_parse_from([
        "tea",
        "--print",
        "--no-session",
        "--provider",
        "fake",
        "--model",
        "fake/model",
        "--trust",
        "ignore",
        "hello",
    ])
    .unwrap();
    let mut stdout = Vec::new();
    let error = Box::pin(tea_cli::modes::print::run(
        &args,
        &injected_bootstrap(&root, "unsafe\u{1b}[31m"),
        &mut std::io::empty(),
        true,
        &mut stdout,
        &mut Vec::new(),
    ))
    .await
    .unwrap_err();
    assert_eq!(error.category(), ExitCategory::Provider);
    assert!(stdout.is_empty());
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn print_mode_fails_closed_on_approval_without_stdout() {
    let root = temp_root("policy");
    fs::write(root.join("file.txt"), "old").unwrap();
    let args = CliArgs::try_parse_from([
        "tea",
        "--print",
        "--no-session",
        "--provider",
        "fake",
        "--model",
        "fake/model",
        "--trust",
        "ignore",
        "edit the file",
    ])
    .unwrap();
    let mut stdout = Vec::new();
    let mut stderr = Vec::new();
    let error = Box::pin(tea_cli::modes::print::run(
        &args,
        &injected_bootstrap_with(&root, edit_provider()),
        &mut std::io::empty(),
        true,
        &mut stdout,
        &mut stderr,
    ))
    .await
    .unwrap_err();
    assert_eq!(error.category(), ExitCategory::PolicyDenied);
    assert!(stdout.is_empty());
    assert!(stderr.is_empty());
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn project_mcp_servers_require_trust_and_replace_global_by_id() {
    let root = temp_root("mcp-settings-trust");
    let home = root.join("home");
    let workspace = root.join("workspace");
    fs::create_dir_all(home.join(".tea")).unwrap();
    fs::create_dir_all(workspace.join(".tea")).unwrap();
    fs::write(
        home.join(".tea/settings.json"),
        r#"{"schemaVersion":1,"mcpServers":[{"id":"shared","transport":{"type":"stdio","executable":"/global/server"}}]}"#,
    )
    .unwrap();
    fs::write(
        workspace.join(".tea/settings.json"),
        r#"{"schemaVersion":1,"mcpServers":[{"id":"shared","transport":{"type":"stdio","executable":"/project/server"}}]}"#,
    )
    .unwrap();

    let ignored_args = CliArgs::try_parse_from([
        "tea",
        "--print",
        "--no-session",
        "--provider",
        "fake",
        "--model",
        "fake/model",
        "--trust",
        "ignore",
        "hello",
    ])
    .unwrap();
    let bootstrap = CliBootstrap::new(BootstrapEnvironment::new(
        &workspace,
        Some(home),
        BTreeMap::new(),
    ))
    .with_provider(fake_provider("unused"));
    let (ignored, _) = bootstrap.build_async(&ignored_args).await.unwrap();
    assert_eq!(ignored.settings().mcp_servers.len(), 1);
    assert_eq!(
        ignored.settings().mcp_servers[0]
            .transport()
            .as_stdio()
            .executable(),
        Path::new("/global/server")
    );
    ignored.shutdown().await;

    let trusted_args = CliArgs::try_parse_from([
        "tea",
        "--print",
        "--no-session",
        "--provider",
        "fake",
        "--model",
        "fake/model",
        "--trust",
        "once",
        "hello",
    ])
    .unwrap();
    let (trusted, _) = bootstrap.build_async(&trusted_args).await.unwrap();
    assert_eq!(trusted.settings().mcp_servers.len(), 1);
    assert_eq!(
        trusted.settings().mcp_servers[0]
            .transport()
            .as_stdio()
            .executable(),
        Path::new("/project/server")
    );
    trusted.shutdown().await;
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn active_mcp_bootstrap_failure_leaves_no_reusable_service_owner() {
    let root = temp_root("mcp-bootstrap-failure");
    fs::create_dir_all(root.join(".tea")).unwrap();
    fs::write(
        root.join(".tea/settings.json"),
        r#"{
            "schemaVersion":1,
            "activeTools":["mcp.broken.ping"],
            "mcpServers":[{
                "id":"broken",
                "transport":{"type":"stdio","executable":"/missing/mcp-server"},
                "tools":[{
                    "remoteName":"ping",
                    "alias":"mcp.broken.ping",
                    "declaration":{
                        "effects":["fs.read"],
                        "idempotency":"idempotent",
                        "retrySafety":"never",
                        "concurrency":"serial",
                        "timeoutMillis":1000
                    }
                }]
            }]
        }"#,
    )
    .unwrap();
    let args = CliArgs::try_parse_from([
        "tea",
        "--print",
        "--no-session",
        "--provider",
        "fake",
        "--model",
        "fake/model",
        "--trust",
        "ignore",
        "hello",
    ])
    .unwrap();
    let bootstrap = injected_bootstrap(&root, "unused");
    let error = bootstrap.build_async(&args).await.unwrap_err();
    assert_eq!(error.category(), ExitCategory::TrustOrConfig);

    fs::remove_file(root.join(".tea/settings.json")).unwrap();
    let (service, _) = bootstrap.build_async(&args).await.unwrap();
    service.shutdown().await;
    fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[test]
fn trusted_project_settings_symlink_cannot_escape_workspace() {
    use std::os::unix::fs::symlink;

    let root = temp_root("settings-symlink");
    let outside = root.with_extension("outside-settings.json");
    fs::create_dir_all(root.join(".tea")).unwrap();
    fs::write(
        &outside,
        r#"{"schemaVersion":1,"mcpServers":[{"id":"outside","transport":{"type":"stdio","executable":"/outside/server"}}]}"#,
    )
    .unwrap();
    symlink(&outside, root.join(".tea/settings.json")).unwrap();
    let args = CliArgs::try_parse_from([
        "tea",
        "--print",
        "--no-session",
        "--provider",
        "fake",
        "--model",
        "fake/model",
        "--trust",
        "once",
        "hello",
    ])
    .unwrap();
    let error = injected_bootstrap(&root, "unused")
        .build(&args)
        .unwrap_err();
    assert_eq!(error.category(), ExitCategory::TrustOrConfig);
    fs::remove_dir_all(root).unwrap();
    fs::remove_file(outside).unwrap();
}

#[test]
fn real_binary_uses_live_bootstrap_and_preserves_stdout_contract() {
    let root = temp_root("process");
    let (base_url, server) = one_shot_openai("process answer");
    let output = Command::new(env!("CARGO_BIN_EXE_tea"))
        .args([
            "--print",
            "--no-session",
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
            "--model",
            "fake/model",
        ])
        .env_clear()
        .env("TEA_OPENAI_API_KEY", "test-secret-never-print")
        .env("TEA_OPENAI_BASE_URL", base_url)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            child
                .stdin
                .take()
                .expect("piped stdin")
                .write_all(b"piped production prompt")?;
            child.wait_with_output()
        })
        .unwrap();
    server.join().unwrap();
    assert!(output.status.success(), "{output:?}");
    assert_eq!(output.stdout, b"process answer\n");
    assert!(output.stderr.is_empty());
    assert!(!String::from_utf8_lossy(&output.stderr).contains("test-secret-never-print"));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn process_failures_have_stable_exit_and_empty_stdout() {
    let root = temp_root("failure");
    let output = Command::new(env!("CARGO_BIN_EXE_tea"))
        .args([
            "--print",
            "--no-session",
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
            "hello",
        ])
        .env_clear()
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(4));
    assert!(output.stdout.is_empty());
    assert_eq!(
        output.stderr,
        b"tea: provider credentials could not be resolved\n"
    );

    fs::write(root.join("AGENTS.md"), "project instruction").unwrap();
    let untrusted = Command::new(env!("CARGO_BIN_EXE_tea"))
        .args([
            "--print",
            "--no-session",
            "--cwd",
            root.to_str().unwrap(),
            "--config-dir",
            root.join("config").to_str().unwrap(),
            "--state-dir",
            root.join("state").to_str().unwrap(),
            "--data-dir",
            root.join("data").to_str().unwrap(),
            "hello",
        ])
        .env_clear()
        .output()
        .unwrap();
    assert_eq!(untrusted.status.code(), Some(3));
    assert!(untrusted.stdout.is_empty());
    assert_eq!(
        untrusted.stderr,
        b"tea: project-local configuration is not trusted\n"
    );

    let mut child = Command::new(env!("CARGO_BIN_EXE_tea"))
        .args(["--print"])
        .env_clear()
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(&vec![b'x'; MAX_INITIAL_PROMPT_BYTES + 1])
        .unwrap();
    let oversized = child.wait_with_output().unwrap();
    assert_eq!(oversized.status.code(), Some(2));
    assert!(oversized.stdout.is_empty());
    assert_eq!(oversized.stderr, b"tea: prompt exceeds input size limit\n");
    fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[test]
fn ctrl_c_cancels_owned_print_run_with_stable_exit() {
    use nix::sys::signal::{Signal, kill};
    use nix::unistd::Pid;
    use std::sync::mpsc;
    use std::time::Duration;

    let root = temp_root("cancel");
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let (requested, receiver) = mpsc::sync_channel(1);
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = vec![0_u8; 64 * 1024];
        let _ = stream.read(&mut request).unwrap();
        requested.send(()).unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        while stream.read(&mut request).is_ok_and(|read| read != 0) {}
    });
    let child = Command::new(env!("CARGO_BIN_EXE_tea"))
        .args([
            "--print",
            "--no-session",
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
            "--model",
            "fake/model",
            "hello",
        ])
        .env_clear()
        .env("TEA_OPENAI_API_KEY", "cancel-secret-never-print")
        .env("TEA_OPENAI_BASE_URL", format!("http://{address}/v1"))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    receiver.recv_timeout(Duration::from_secs(5)).unwrap();
    kill(
        Pid::from_raw(i32::try_from(child.id()).unwrap()),
        Signal::SIGINT,
    )
    .unwrap();
    let output = child.wait_with_output().unwrap();
    assert_eq!(output.status.code(), Some(6));
    assert!(output.stdout.is_empty());
    assert_eq!(output.stderr, b"tea: operation cancelled\n");
    server.join().unwrap();
    fs::remove_dir_all(root).unwrap();
}

fn one_shot_openai(text: &str) -> (String, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let text = text.to_owned();
    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .unwrap();
        let mut request = vec![0_u8; 64 * 1024];
        let read = stream.read(&mut request).unwrap();
        let request = String::from_utf8_lossy(&request[..read]);
        assert!(request.starts_with("POST /v1/chat/completions HTTP/1.1"));
        let body = format!(
            "data: {{\"id\":\"response-1\",\"model\":\"fake/model\",\"choices\":[{{\"index\":0,\"delta\":{{\"role\":\"assistant\",\"content\":{text:?}}},\"finish_reason\":null}}]}}\n\ndata: {{\"id\":\"response-1\",\"model\":\"fake/model\",\"choices\":[{{\"index\":0,\"delta\":{{}},\"finish_reason\":\"stop\"}}]}}\n\ndata: [DONE]\n\n"
        );
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        )
        .unwrap();
        stream.flush().unwrap();
    });
    (format!("http://{address}/v1"), handle)
}
