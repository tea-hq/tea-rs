use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use clap::Parser as _;
use tea_cli::args::CliArgs;
use tea_cli::{BootstrapEnvironment, CliBootstrap, ExitCategory};
use tea_model::{
    ModelCapabilities, ModelCompletion, ModelDisplayName, ModelEvent, ModelFailureCode,
    ModelResponseInfo, ModelSpec, ProviderId, ProviderToolCallId, ToolCallCompleted,
    ToolCallStarted, Utf8Delta,
};
use tea_protocol::{FinalOutputFormat, ModelId, StopReason, TokenCount};
use tea_testkit::{ScriptStep, ScriptedModelProvider, ScriptedModelResponse};

static ID: AtomicU64 = AtomicU64::new(0);

#[derive(Clone)]
struct SharedOutput(Arc<Mutex<Vec<u8>>>);

impl Write for SharedOutput {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

struct HeaderThenSlow {
    writes: usize,
}

impl Write for HeaderThenSlow {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.writes += 1;
        if self.writes > 1 {
            std::thread::sleep(Duration::from_secs(2));
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn temp_root(label: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "tea-json-{label}-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&path).unwrap();
    path
}

fn model() -> ModelSpec {
    ModelSpec::new(
        ModelId::from_str("fake/model").unwrap(),
        ProviderId::from_str("fake").unwrap(),
        ModelDisplayName::from_str("Fake Model").unwrap(),
        TokenCount::new(32_000).unwrap(),
        TokenCount::new(4_000).unwrap(),
        ModelCapabilities::text()
            .with_tools(true)
            .with_final_json_schema_with_tools(),
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

fn bootstrap(root: &Path, provider: Arc<ScriptedModelProvider>) -> CliBootstrap {
    CliBootstrap::new(BootstrapEnvironment::new(
        root,
        Some(root.to_path_buf()),
        BTreeMap::new(),
    ))
    .with_provider(provider)
}

fn args() -> CliArgs {
    CliArgs::try_parse_from([
        "tea",
        "--json",
        "--no-session",
        "--provider",
        "fake",
        "--model",
        "fake/model",
        "--trust",
        "ignore",
        "run",
    ])
    .unwrap()
}

fn schema_args(root: &Path) -> CliArgs {
    CliArgs::try_parse_from([
        "tea",
        "--json",
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
    .unwrap()
}

fn output_lines(bytes: &[u8]) -> Vec<serde_json::Value> {
    assert!(bytes.ends_with(b"\n"));
    bytes
        .split(|byte| *byte == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice(line).unwrap())
        .collect()
}

fn normalized(lines: &[serde_json::Value]) -> Vec<serde_json::Value> {
    lines
        .iter()
        .map(|line| match line["type"].as_str().unwrap() {
            "tea_event_stream" => serde_json::json!({
                "type":"tea_event_stream",
                "modeVersion":line["modeVersion"],
                "protocolVersion":line["protocolVersion"],
                "workspacePrefix":line["workspaceId"].as_str().unwrap().starts_with("workspace/"),
            }),
            "message_delta" => serde_json::json!({
                "type":"message_delta",
                "deltaType":line["payload"]["delta"]["type"],
                "text":line["payload"]["delta"]["text"],
            }),
            kind => serde_json::json!({"type":kind}),
        })
        .collect()
}

fn fixture(name: &str) -> Vec<serde_json::Value> {
    include_str!("fixtures/json/success.jsonl")
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_str(line).unwrap())
        .collect::<Vec<_>>()
        .into_iter()
        .filter(|_| name == "success")
        .collect()
}

#[tokio::test(flavor = "current_thread")]
async fn success_stream_matches_golden_and_every_event_is_canonical() {
    let root = temp_root("success");
    let provider = provider([ScriptedModelResponse::events([
        ModelEvent::Started(ModelResponseInfo::new()),
        ModelEvent::ThinkingDelta(Utf8Delta::new("plan\u{2028}next").unwrap()),
        ModelEvent::TextDelta(Utf8Delta::new("answer").unwrap()),
        ModelEvent::Completed(ModelCompletion::new(StopReason::Completed).unwrap()),
    ])]);
    let bytes = Arc::new(Mutex::new(Vec::new()));
    Box::pin(tea_cli::modes::json::run(
        &args(),
        &bootstrap(&root, provider),
        &mut io::empty(),
        true,
        Box::new(SharedOutput(Arc::clone(&bytes))),
    ))
    .await
    .unwrap();
    let output = bytes.lock().unwrap().clone();
    let lines = output_lines(&output);
    assert_eq!(normalized(&lines), fixture("success"));
    assert!(lines[0]["sessionId"].as_str().is_some());
    assert!(!String::from_utf8_lossy(&output).contains(root.to_str().unwrap()));
    for event in &lines[1..] {
        let encoded = serde_json::to_vec(event).unwrap();
        assert!(serde_json::from_slice::<tea_protocol::EventEnvelope>(&encoded).is_ok());
    }
    assert!(!output.contains(&0x1b));
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn json_mode_enforces_json_schema_with_active_tools() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"answer": {"type": "string"}},
        "required": ["answer"],
        "additionalProperties": false
    });
    for (label, response, expected_status) in [
        ("schema-success", r#"{"answer":"tea"}"#, "completed"),
        ("schema-malformed", "not JSON", "interrupted"),
    ] {
        let root = temp_root(label);
        fs::write(
            root.join("answer.schema.json"),
            serde_json::to_vec(&schema).unwrap(),
        )
        .unwrap();
        let provider = provider([ScriptedModelResponse::text([response])]);
        let bytes = Arc::new(Mutex::new(Vec::new()));
        let result = Box::pin(tea_cli::modes::json::run(
            &schema_args(&root),
            &bootstrap(&root, Arc::clone(&provider)),
            &mut io::empty(),
            true,
            Box::new(SharedOutput(Arc::clone(&bytes))),
        ))
        .await;
        if expected_status == "completed" {
            result.unwrap();
        } else {
            assert_eq!(result.unwrap_err().category(), ExitCategory::Provider);
        }
        let output = bytes.lock().unwrap().clone();
        let lines = output_lines(&output);
        assert!(lines.iter().any(|line| {
            line["type"] == "run_finished" && line["payload"]["status"] == expected_status
        }));
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
async fn approval_pause_and_provider_failure_emit_canonical_terminal_observations() {
    let root = temp_root("terminal");
    fs::write(root.join("file.txt"), "old").unwrap();
    let index = tea_model::ModelStreamIndex::new(0).unwrap();
    let opaque = ProviderToolCallId::from_str("edit-1").unwrap();
    let approval_provider = provider([ScriptedModelResponse::events([
        ModelEvent::Started(ModelResponseInfo::new()),
        ModelEvent::ToolCallStarted(ToolCallStarted::new(index, opaque.clone(), "edit").unwrap()),
        ModelEvent::ToolCallCompleted(
            ToolCallCompleted::new(
                index,
                opaque,
                "edit",
                serde_json::json!({"path":"file.txt","oldText":"old","newText":"new"}),
            )
            .unwrap(),
        ),
        ModelEvent::Completed(ModelCompletion::new(StopReason::ToolUse).unwrap()),
    ])]);
    let approval_bytes = Arc::new(Mutex::new(Vec::new()));
    let approval = Box::pin(tea_cli::modes::json::run(
        &args(),
        &bootstrap(&root, approval_provider),
        &mut io::empty(),
        true,
        Box::new(SharedOutput(Arc::clone(&approval_bytes))),
    ))
    .await
    .unwrap_err();
    assert_eq!(approval.category(), ExitCategory::PolicyDenied);
    let approval_types = output_lines(&approval_bytes.lock().unwrap())
        .into_iter()
        .map(|line| line["type"].as_str().unwrap().to_owned())
        .collect::<Vec<_>>();
    assert!(approval_types.contains(&"approval_requested".to_owned()));
    assert_eq!(approval_types.last().unwrap(), "turn_checkpointed");

    let failure_bytes = Arc::new(Mutex::new(Vec::new()));
    let failure = Box::pin(tea_cli::modes::json::run(
        &args(),
        &bootstrap(
            &root,
            provider([ScriptedModelResponse::failure(
                ModelFailureCode::Authentication,
                "secret-independent failure",
            )]),
        ),
        &mut io::empty(),
        true,
        Box::new(SharedOutput(Arc::clone(&failure_bytes))),
    ))
    .await
    .unwrap_err();
    assert_eq!(failure.category(), ExitCategory::Provider);
    let failure_lines = output_lines(&failure_bytes.lock().unwrap());
    assert_eq!(failure_lines.last().unwrap()["type"], "run_finished");
    assert!(
        failure_lines
            .iter()
            .all(|line| line.get("message").is_none())
    );
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn web_fetch_model_definition_requires_explicit_valid_configuration() {
    let default_root = temp_root("fetch-default");
    let default_provider = provider([ScriptedModelResponse::text(["default"])]);
    Box::pin(tea_cli::modes::json::run(
        &args(),
        &bootstrap(&default_root, Arc::clone(&default_provider)),
        &mut io::empty(),
        true,
        Box::new(io::sink()),
    ))
    .await
    .unwrap();
    let requests = default_provider.captured_requests().unwrap();
    assert_eq!(
        requests[0]
            .tools()
            .iter()
            .map(tea_model::ModelToolDefinition::name)
            .collect::<Vec<_>>(),
        ["bash", "edit", "read", "write"]
    );
    fs::remove_dir_all(default_root).unwrap();

    let missing_root = temp_root("fetch-missing-backend");
    fs::create_dir_all(missing_root.join(".tea")).unwrap();
    fs::write(
        missing_root.join(".tea/settings.json"),
        r#"{"schemaVersion":1,"activeTools":["web_fetch"]}"#,
    )
    .unwrap();
    let missing = Box::pin(tea_cli::modes::json::run(
        &args(),
        &bootstrap(
            &missing_root,
            provider([ScriptedModelResponse::text(["unused"])]),
        ),
        &mut io::empty(),
        true,
        Box::new(io::sink()),
    ))
    .await
    .unwrap_err();
    assert_eq!(missing.category(), ExitCategory::TrustOrConfig);
    assert_eq!(
        missing.message(),
        "active web fetch requires an enabled client backend"
    );
    fs::remove_dir_all(missing_root).unwrap();

    let enabled_root = temp_root("fetch-enabled");
    fs::create_dir_all(enabled_root.join(".tea")).unwrap();
    fs::write(
        enabled_root.join(".tea/settings.json"),
        r#"{
            "schemaVersion":1,
            "activeTools":["web_fetch"],
            "webFetch":{"enabled":true,"backend":"http"}
        }"#,
    )
    .unwrap();
    let enabled_provider = provider([ScriptedModelResponse::text(["enabled"])]);
    Box::pin(tea_cli::modes::json::run(
        &args(),
        &bootstrap(&enabled_root, Arc::clone(&enabled_provider)),
        &mut io::empty(),
        true,
        Box::new(io::sink()),
    ))
    .await
    .unwrap();
    let requests = enabled_provider.captured_requests().unwrap();
    assert_eq!(requests[0].tools().len(), 1);
    assert_eq!(requests[0].tools()[0].name(), "web_fetch");
    assert!(requests[0].tools()[0].as_function().is_some());
    fs::remove_dir_all(enabled_root).unwrap();
}

#[test]
fn real_binary_exits_cleanly_when_json_pipe_is_broken() {
    let root = temp_root("broken-pipe");
    let mut child = Command::new(env!("CARGO_BIN_EXE_tea"))
        .args([
            "--json",
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
        .env("TEA_OPENAI_API_KEY", "broken-pipe-secret")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    drop(child.stdout.take());
    let output = child.wait_with_output().unwrap();
    assert_eq!(output.status.code(), Some(6));
    assert_eq!(output.stderr, b"tea: JSON output is unavailable\n");
    assert!(!String::from_utf8_lossy(&output.stderr).contains("broken-pipe-secret"));
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn slow_writer_cancels_owned_run_without_deadlock() {
    let root = temp_root("slow");
    let provider = provider([ScriptedModelResponse::new([
        ScriptStep::event(ModelEvent::Started(ModelResponseInfo::new())),
        ScriptStep::AwaitCancellation,
    ])]);
    let result = tokio::time::timeout(
        Duration::from_secs(1),
        Box::pin(tea_cli::modes::json::run(
            &args(),
            &bootstrap(&root, provider),
            &mut io::empty(),
            true,
            Box::new(HeaderThenSlow { writes: 0 }),
        )),
    )
    .await
    .expect("JSON mode must not await a blocked writer")
    .unwrap_err();
    assert_eq!(result.category(), ExitCategory::Cancelled);
    fs::remove_dir_all(root).unwrap();
}
