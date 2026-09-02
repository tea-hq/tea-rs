#![forbid(unsafe_code)]

use std::fs;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt as _, AsyncWriteExt as _, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

const LIVE_GATE: &str = "TEA_APP_SERVER_LIVE_SMOKE";
const APP_SERVER_VERSION: &str = "1.0";
const READ_TIMEOUT: Duration = Duration::from_mins(3);
const MAX_APPROVALS: usize = 8;

struct LiveWorkspace {
    root: PathBuf,
    workspace: PathBuf,
    config: PathBuf,
}

impl LiveWorkspace {
    fn new(config: PathBuf) -> Self {
        let root = std::env::temp_dir().join(format!(
            "tea-app-server-live-smoke-{}",
            uuid::Uuid::now_v7().hyphenated()
        ));
        let workspace = root.join("workspace");
        fs::create_dir_all(&workspace).expect("create live workspace");
        fs::write(
            workspace.join("tea-live-marker.txt"),
            "APP_SERVER_LIVE_FILE_OK\n",
        )
        .expect("write live marker");
        Self {
            root,
            workspace,
            config,
        }
    }

    fn state(&self) -> PathBuf {
        self.root.join("state")
    }

    fn data(&self) -> PathBuf {
        self.root.join("data")
    }
}

impl Drop for LiveWorkspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

struct AppServerProcess {
    child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
}

impl AppServerProcess {
    fn spawn(live: &LiveWorkspace) -> Self {
        let mut command = Command::new(env!("CARGO_BIN_EXE_tea"));
        command
            .args([
                "--app-server",
                "--cwd",
                live.workspace.to_str().expect("workspace path is UTF-8"),
                "--config-dir",
                live.config.to_str().expect("config path is UTF-8"),
                "--state-dir",
                live.state().to_str().expect("state path is UTF-8"),
                "--data-dir",
                live.data().to_str().expect("data path is UTF-8"),
                "--trust",
                "ignore",
            ])
            // The test must exercise the checked-in ~/.tea provider selection,
            // not an unrelated shell override.
            .env_remove("TEA_PROVIDER")
            .env_remove("TEA_MODEL")
            .env_remove("TEA_OPENAI_MODEL")
            .env_remove("TEA_ANTHROPIC_MODEL")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        let mut child = command.spawn().expect("start tea app-server");
        let input = child.stdin.take().expect("app-server stdin");
        let output = BufReader::new(child.stdout.take().expect("app-server stdout"));
        Self {
            child,
            input,
            output,
        }
    }

    async fn send(&mut self, value: Value) {
        let mut bytes = serde_json::to_vec(&value).expect("request serializes");
        bytes.push(b'\n');
        self.input.write_all(&bytes).await.expect("request writes");
        self.input.flush().await.expect("request flushes");
    }

    async fn receive(&mut self) -> Value {
        let mut line = String::new();
        let read = tokio::time::timeout(READ_TIMEOUT, self.output.read_line(&mut line))
            .await
            .expect("app-server response deadline")
            .expect("app-server response read");
        assert!(read > 0, "app-server closed stdout before a response");
        serde_json::from_str(line.trim_end()).expect("app-server emitted JSON")
    }

    async fn request(&mut self, id: &str, method: &str, params: Value) -> Value {
        self.send(json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        }))
        .await;
        loop {
            let message = self.receive().await;
            if message.get("id").and_then(Value::as_str) == Some(id) {
                return message;
            }
        }
    }

    async fn shutdown(mut self) {
        let response = self.request("shutdown", "server/shutdown", json!({})).await;
        assert_eq!(response["result"]["stopping"], true);
        self.input.shutdown().await.expect("close app-server stdin");
        tokio::time::timeout(READ_TIMEOUT, self.child.wait())
            .await
            .expect("app-server shutdown deadline")
            .expect("app-server wait");
    }
}

impl Drop for AppServerProcess {
    fn drop(&mut self) {
        if self.child.try_wait().ok().flatten().is_none() {
            let _ = self.child.start_kill();
        }
    }
}

struct TurnObservation {
    text: String,
    saw_tool_call: bool,
    event_types: Vec<String>,
    approvals: usize,
    status: String,
}

#[allow(clippy::too_many_lines)]
async fn prompt(process: &mut AppServerProcess, session_id: &str, text: &str) -> TurnObservation {
    process
        .send(json!({
            "jsonrpc": "2.0",
            "id": "prompt",
            "method": "session/prompt",
            "params": {"sessionId": session_id, "text": text},
        }))
        .await;

    let mut accepted = false;
    let mut saw_tool_call = false;
    let mut event_types = Vec::new();
    let mut approvals = 0;
    let mut text_output = String::new();
    let mut approval_ids = Vec::new();
    let (status, terminal_error) = loop {
        let message = process.receive().await;
        if message.get("id").and_then(Value::as_str) == Some("prompt") {
            assert!(
                message.get("error").is_none(),
                "prompt was rejected: {message}"
            );
            assert_eq!(message["result"]["accepted"], true);
            accepted = true;
            continue;
        }
        if message["method"] == "event/session" {
            let event = &message["params"]["event"];
            let event_type = event
                .get("type")
                .or_else(|| event.get("payload").and_then(|value| value.get("type")))
                .and_then(Value::as_str)
                .unwrap_or_default();
            if !event_type.is_empty() {
                event_types.push(event_type.to_owned());
            }
            match event_type {
                "approval_requested" => {
                    let approval_id = event
                        .get("approvalId")
                        .or_else(|| {
                            event
                                .get("payload")
                                .and_then(|value| value.get("approvalId"))
                        })
                        .and_then(Value::as_str)
                        .expect("approval event has an id");
                    if approval_ids.iter().any(|id| id == approval_id) {
                        continue;
                    }
                    assert!(
                        approvals < MAX_APPROVALS,
                        "live prompt requested too many approvals"
                    );
                    approval_ids.push(approval_id.to_owned());
                    approvals += 1;
                    let response = process
                        .request(
                            &format!("approval-{approvals}"),
                            "approval/respond",
                            json!({
                                "sessionId": session_id,
                                "approvalId": approval_id,
                                "decision": {"type": "allow_once"},
                            }),
                        )
                        .await;
                    assert!(
                        response.get("error").is_none(),
                        "approval was rejected: {response}"
                    );
                }
                "message_delta" => {
                    let delta = event
                        .get("payload")
                        .and_then(|value| value.get("delta"))
                        .or_else(|| event.get("delta"));
                    if delta
                        .and_then(|value| value.get("type"))
                        .and_then(Value::as_str)
                        == Some("text_delta")
                        && let Some(chunk) = delta
                            .and_then(|value| value.get("text"))
                            .and_then(Value::as_str)
                    {
                        text_output.push_str(chunk);
                    }
                }
                "tool_call_requested" | "hosted_tool_started" => saw_tool_call = true,
                _ => {}
            }
        } else if message["method"] == "event/turn_completed" {
            let candidate = message["params"]["status"].as_str().unwrap_or_default();
            if candidate != "awaiting_approval" {
                assert!(accepted, "turn completed before prompt acceptance");
                let terminal_error = message["params"]["error"].as_str().map(str::to_owned);
                break (candidate.to_owned(), terminal_error);
            }
        }
    };
    assert_eq!(
        status, "completed",
        "live turn did not complete: {status}; error={terminal_error:?}; events={event_types:?}"
    );
    TurnObservation {
        text: text_output,
        saw_tool_call,
        event_types,
        approvals,
        status,
    }
}

fn session_message_count(listed: &Value, session_id: &str) -> usize {
    listed["result"]["sessions"]
        .as_array()
        .and_then(|sessions| {
            sessions
                .iter()
                .find(|entry| entry["sessionId"].as_str() == Some(session_id))
        })
        .and_then(|entry| entry["messageCount"].as_u64())
        .and_then(|count| usize::try_from(count).ok())
        .expect("session list contains message count")
}

fn configured_provider() -> Option<(PathBuf, String, String)> {
    if std::env::var(LIVE_GATE).ok().as_deref() != Some("1") {
        eprintln!("skipping app-server live smoke: set {LIVE_GATE}=1 to opt in");
        return None;
    }
    let home = std::env::var_os("HOME").map(PathBuf::from)?;
    let config = home.join(".tea");
    let settings_path = config.join("settings.json");
    let providers_path = config.join("providers.json");
    assert!(settings_path.is_file(), "~/.tea/settings.json is required");
    assert!(
        providers_path.is_file(),
        "~/.tea/providers.json is required"
    );
    let settings: Value =
        serde_json::from_slice(&fs::read(settings_path).expect("read Tea settings"))
            .expect("Tea settings are valid JSON");
    let provider = settings["provider"]
        .as_str()
        .filter(|value| !value.is_empty())
        .expect("Tea settings provider is configured")
        .to_owned();
    let model = settings["model"]
        .as_str()
        .filter(|value| !value.is_empty())
        .expect("Tea settings model is configured")
        .to_owned();
    Some((config, provider, model))
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires TEA_APP_SERVER_LIVE_SMOKE=1, ~/.tea provider credentials, and network"]
#[allow(clippy::too_many_lines)]
async fn app_server_uses_local_provider_and_survives_real_session_lifecycle() {
    let Some((config, provider, model)) = configured_provider() else {
        return;
    };
    let live = LiveWorkspace::new(config);
    let mut first = AppServerProcess::spawn(&live);

    let initialized = first
        .request(
            "initialize",
            "server/initialize",
            json!({
                "clientName": "tea-app-server-live-smoke",
                "clientVersion": "0.1.0",
                "appServerVersion": APP_SERVER_VERSION,
            }),
        )
        .await;
    assert_eq!(
        initialized["result"]["appServerVersion"],
        APP_SERVER_VERSION
    );
    assert_eq!(initialized["result"]["capabilities"]["sessionResume"], true);
    assert_eq!(initialized["result"]["capabilities"]["sessionMcp"], true);
    assert_eq!(
        initialized["result"]["capabilities"]["permissionRequests"],
        true
    );

    let created = first
        .request(
            "create",
            "session/create",
            json!({"cwd": live.workspace, "mcpServers": []}),
        )
        .await;
    assert!(
        created.get("error").is_none(),
        "session creation failed: {created}"
    );
    let session_id = created["result"]["sessionId"]
        .as_str()
        .expect("session id")
        .to_owned();
    let models = created["result"]["availableModels"]
        .as_array()
        .expect("available model list");
    assert!(!models.is_empty(), "local provider must advertise a model");
    assert!(
        models.iter().any(|candidate| {
            candidate["providerId"] == provider && candidate["modelId"] == model
        }),
        "configured provider/model was not advertised: {models:?}"
    );

    let selected = first
        .request(
            "set-model",
            "session/set_config",
            json!({
                "sessionId": session_id,
                "configId": "model",
                "value": format!("{provider}:{model}"),
            }),
        )
        .await;
    assert!(
        selected.get("error").is_none(),
        "model selection failed: {selected}"
    );
    let mode = first
        .request(
            "set-mode",
            "session/set_mode",
            json!({"sessionId": session_id, "mode": "default"}),
        )
        .await;
    assert_eq!(mode["result"]["mode"], "default");

    let simple = prompt(
        &mut first,
        &session_id,
        "Reply with exactly APP_SERVER_LIVE_OK. Do not call any tools.",
    )
    .await;
    assert_eq!(simple.status, "completed");
    assert!(
        simple.text.contains("APP_SERVER_LIVE_OK"),
        "unexpected live response: {}",
        simple.text
    );

    let before_tool = first
        .request(
            "before-tool-list",
            "session/list",
            json!({"cwd": live.workspace}),
        )
        .await;
    let message_count_before_tool = session_message_count(&before_tool, &session_id);

    let marker = live.workspace.join("tea-live-marker.txt");
    let tool = prompt(
        &mut first,
        &session_id,
        &format!(
            "Use only the read tool to inspect this absolute file: {}. Then reply with the exact marker APP_SERVER_LIVE_TOOL_OK and the file contents.",
            marker.display()
        ),
    )
    .await;
    assert!(
        tool.saw_tool_call,
        "the live tool turn did not emit a tool call; events: {:?}",
        tool.event_types
    );
    assert!(tool.approvals <= MAX_APPROVALS);
    assert!(
        tool.text.contains("APP_SERVER_LIVE_TOOL_OK"),
        "unexpected tool response: {}",
        tool.text
    );

    let listed = first
        .request("list", "session/list", json!({"cwd": live.workspace}))
        .await;
    assert!(
        listed["result"]["sessions"]
            .as_array()
            .is_some_and(|sessions| {
                sessions
                    .iter()
                    .any(|entry| entry["sessionId"] == session_id)
            })
    );
    let message_count_after_tool = session_message_count(&listed, &session_id);
    assert!(
        message_count_after_tool >= message_count_before_tool + 3,
        "the completed tool turn did not persist its tool exchange: before={message_count_before_tool}, after={message_count_after_tool}, events={:?}",
        tool.event_types
    );

    let before_approval_tool = first
        .request(
            "before-approval-tool-list",
            "session/list",
            json!({"cwd": live.workspace}),
        )
        .await;
    let message_count_before_approval_tool =
        session_message_count(&before_approval_tool, &session_id);
    let approval_tool = prompt(
        &mut first,
        &session_id,
        "Use only the bash tool to run this harmless command: printf APP_SERVER_LIVE_BASH_OK. Then reply with the exact marker APP_SERVER_LIVE_BASH_OK.",
    )
    .await;
    assert!(
        approval_tool.saw_tool_call,
        "the approval tool turn did not emit a tool call; events: {:?}",
        approval_tool.event_types
    );
    assert!(
        approval_tool.approvals > 0,
        "the bash tool turn did not exercise an approval request; events: {:?}",
        approval_tool.event_types
    );
    assert!(
        approval_tool.text.contains("APP_SERVER_LIVE_BASH_OK"),
        "unexpected approval tool response: {}",
        approval_tool.text
    );
    let listed_after_approval_tool = first
        .request(
            "after-approval-tool-list",
            "session/list",
            json!({"cwd": live.workspace}),
        )
        .await;
    let message_count_after_approval_tool =
        session_message_count(&listed_after_approval_tool, &session_id);
    assert!(
        message_count_after_approval_tool >= message_count_before_approval_tool + 3,
        "the approved tool turn did not persist its tool exchange: before={message_count_before_approval_tool}, after={message_count_after_approval_tool}, events={:?}",
        approval_tool.event_types
    );
    first.shutdown().await;

    // Rebuild the Rust process against the same durable state and attach the
    // exact session that produced the real network/tool transcript.
    let mut second = AppServerProcess::spawn(&live);
    let initialized = second
        .request(
            "initialize",
            "server/initialize",
            json!({
                "clientName": "tea-app-server-live-smoke-reconnect",
                "clientVersion": "0.1.0",
                "appServerVersion": APP_SERVER_VERSION,
            }),
        )
        .await;
    assert_eq!(
        initialized["result"]["appServerVersion"],
        APP_SERVER_VERSION
    );
    let loaded = second
        .request(
            "load",
            "session/load",
            json!({"cwd": live.workspace, "sessionId": session_id, "mcpServers": []}),
        )
        .await;
    assert!(
        loaded.get("error").is_none(),
        "session load failed: {loaded}"
    );
    assert_eq!(loaded["result"]["sessionId"], session_id);
    assert_eq!(loaded["result"]["model"]["providerId"], provider);
    assert_eq!(loaded["result"]["model"]["modelId"], model);

    let closed = second
        .request("close", "session/close", json!({"sessionId": session_id}))
        .await;
    assert_eq!(closed["result"]["closed"], true);
    let recreated = second
        .request(
            "recreate",
            "session/create",
            json!({"cwd": live.workspace, "mcpServers": []}),
        )
        .await;
    let recreated_id = recreated["result"]["sessionId"]
        .as_str()
        .expect("recreated session id");
    assert_ne!(recreated_id, session_id);
    second.shutdown().await;
}
