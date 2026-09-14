use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use clap::Parser as _;
use tea::RuntimeCommandOutcome;
use tea_cli::args::CliArgs;
use tea_cli::{BootstrapEnvironment, CliBootstrap};
use tea_protocol::ApprovalDecision;
use tea_provider_openai::env_file::load_env_file;

const LIVE_GATE: &str = "TEA_CLI_LIVE_SMOKE";
const FINAL_MARKER: &str = "LIVE_SMOKE_OK";
const MAX_APPROVAL_ROUNDS: usize = 8;

#[derive(Debug, Clone, PartialEq, Eq)]
struct LiveProvider {
    id: &'static str,
    model: String,
}

struct LiveWorkspace {
    root: PathBuf,
}

impl LiveWorkspace {
    fn new() -> Self {
        let root = std::env::temp_dir().join(format!(
            "tea-cli-live-smoke-{}",
            uuid::Uuid::now_v7().hyphenated()
        ));
        let workspace = root.join("workspace");
        fs::create_dir_all(workspace.join("src")).unwrap();
        fs::write(
            workspace.join("Cargo.toml"),
            "[package]\nname = \"tea-cli-live-smoke-fixture\"\nversion = \"0.1.0\"\nedition = \"2024\"\n",
        )
        .unwrap();
        fs::write(
            workspace.join("src/lib.rs"),
            "pub fn answer() -> i32 { 1 }\n\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn answer_is_fixed() {\n        assert_eq!(super::answer(), 2);\n    }\n}\n",
        )
        .unwrap();
        let status = Command::new("git")
            .args(["init", "--quiet"])
            .current_dir(&workspace)
            .status()
            .expect("git is required for the opt-in live smoke");
        assert!(status.success(), "temporary Git repository creation failed");
        Self { root }
    }

    fn workspace(&self) -> PathBuf {
        self.root.join("workspace")
    }
}

impl Drop for LiveWorkspace {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

fn select_live_provider(values: &BTreeMap<String, String>) -> Result<LiveProvider, String> {
    let provider = values
        .get("TEA_PROVIDER")
        .map(String::as_str)
        .filter(|value| !value.is_empty())
        .unwrap_or("openai");
    let (id, api_key_name, model_name) = match provider {
        "openai" => ("openai", "TEA_OPENAI_API_KEY", "TEA_OPENAI_MODEL"),
        "anthropic" => ("anthropic", "TEA_ANTHROPIC_API_KEY", "TEA_ANTHROPIC_MODEL"),
        _ => return Err("TEA_PROVIDER must be openai or anthropic".to_owned()),
    };
    let missing = [api_key_name, model_name]
        .into_iter()
        .filter(|key| values.get(*key).is_none_or(String::is_empty))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(format!(
            "{} are required for the {id} CLI live smoke",
            missing.join(" and ")
        ));
    }
    Ok(LiveProvider {
        id,
        model: values[model_name].clone(),
    })
}

fn live_environment() -> Option<(BTreeMap<String, String>, LiveProvider)> {
    let mut values = BTreeMap::new();
    let dotenv = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.env");
    if dotenv.exists() {
        values.extend(load_env_file(&dotenv).unwrap_or_default());
    }
    for key in [
        LIVE_GATE,
        "TEA_PROVIDER",
        "TEA_OPENAI_API_KEY",
        "TEA_OPENAI_MODEL",
        "TEA_OPENAI_BASE_URL",
        "TEA_OPENAI_API_KEY_HEADER",
        "TEA_OPENAI_API_KEY_PREFIX",
        "TEA_OPENAI_ORG_ID",
        "TEA_OPENAI_PROJECT_ID",
        "TEA_OPENAI_REASONING_EFFORT",
        "TEA_OPENAI_VISION",
        "TEA_OPENAI_REQUEST_TIMEOUT_MS",
        "TEA_ANTHROPIC_API_KEY",
        "TEA_ANTHROPIC_MODEL",
        "TEA_ANTHROPIC_BASE_URL",
        "TEA_ANTHROPIC_API_VERSION",
        "TEA_ANTHROPIC_REQUEST_TIMEOUT_MS",
        "TEA_SHELL",
        "TEA_SHELL_FLAG",
        "COMSPEC",
    ] {
        if let Ok(value) = std::env::var(key) {
            values.insert(key.to_owned(), value);
        }
    }
    if values.get(LIVE_GATE).map(String::as_str) != Some("1") {
        eprintln!("skipping CLI live smoke: set {LIVE_GATE}=1 to opt in");
        return None;
    }
    let provider = select_live_provider(&values)
        .unwrap_or_else(|error| panic!("invalid CLI live smoke configuration: {error}"));
    Some((values, provider))
}

fn cli_args(root: &LiveWorkspace, provider: &LiveProvider, selection: &str) -> CliArgs {
    CliArgs::try_parse_from([
        "tea",
        selection,
        "--provider",
        provider.id,
        "--model",
        &provider.model,
        "--tools",
        "read,edit,bash",
        "--trust",
        "ignore",
        "--cwd",
        root.workspace().to_str().unwrap(),
        "--config-dir",
        root.root.join("config").to_str().unwrap(),
        "--state-dir",
        root.root.join("state").to_str().unwrap(),
        "--data-dir",
        root.root.join("data").to_str().unwrap(),
    ])
    .unwrap()
}

fn bootstrap(root: &LiveWorkspace, values: BTreeMap<String, String>) -> CliBootstrap {
    CliBootstrap::new(BootstrapEnvironment::new(
        root.workspace(),
        Some(root.root.join("home")),
        values,
    ))
}

#[test]
fn live_provider_defaults_to_openai() {
    let values = BTreeMap::from([
        ("TEA_OPENAI_API_KEY".to_owned(), "test-key".to_owned()),
        ("TEA_OPENAI_MODEL".to_owned(), "test-model".to_owned()),
    ]);

    let provider = select_live_provider(&values).unwrap();

    assert_eq!(provider.id, "openai");
    assert_eq!(provider.model, "test-model");
}

#[test]
fn live_provider_selects_anthropic_and_reaches_cli_args() {
    let values = BTreeMap::from([
        ("TEA_PROVIDER".to_owned(), "anthropic".to_owned()),
        ("TEA_ANTHROPIC_API_KEY".to_owned(), "test-key".to_owned()),
        ("TEA_ANTHROPIC_MODEL".to_owned(), "claude-test".to_owned()),
    ]);
    let provider = select_live_provider(&values).unwrap();
    let root = LiveWorkspace::new();

    let args = cli_args(&root, &provider, "--new");

    assert_eq!(provider.id, "anthropic");
    assert_eq!(args.provider.as_deref(), Some("anthropic"));
    assert_eq!(args.model.as_deref(), Some("claude-test"));
}

#[test]
fn live_provider_requires_selected_credentials() {
    let values = BTreeMap::from([("TEA_PROVIDER".to_owned(), "anthropic".to_owned())]);

    let error = select_live_provider(&values).unwrap_err();

    assert!(error.contains("TEA_ANTHROPIC_API_KEY"));
    assert!(error.contains("TEA_ANTHROPIC_MODEL"));
}

#[test]
fn live_provider_rejects_unsupported_provider() {
    let values = BTreeMap::from([("TEA_PROVIDER".to_owned(), "unsupported".to_owned())]);

    let error = select_live_provider(&values).unwrap_err();

    assert_eq!(error, "TEA_PROVIDER must be openai or anthropic");
}

#[tokio::test(flavor = "current_thread")]
#[ignore = "requires TEA_CLI_LIVE_SMOKE=1, live credentials, network, Git, and Cargo"]
async fn live_agent_edits_tests_and_reopens_a_temporary_git_repository() {
    let Some((values, provider)) = live_environment() else {
        return;
    };
    let root = LiveWorkspace::new();
    let args = cli_args(&root, &provider, "--new");
    let initial_bootstrap = bootstrap(&root, values.clone());
    let (service, _) = initial_bootstrap.build(&args).unwrap();
    let session_id = service.create_session().await.unwrap();
    service
        .prompt(
            session_id,
            "Use read to inspect Cargo.toml and src/lib.rs. Use edit (not write) to change answer() so the existing test passes. Use bash to run `cargo test --quiet`. Do not finish before the test passes. End the final response with LIVE_SMOKE_OK.",
            None,
        )
        .await
        .unwrap();

    let mut approvals = 0;
    let mut completed = false;
    for _ in 0..MAX_APPROVAL_ROUNDS {
        match service.wait(session_id).await.unwrap() {
            RuntimeCommandOutcome::RunCompleted {
                pending_approval_id: Some(approval_id),
                ..
            } => {
                approvals += 1;
                service
                    .approve(session_id, approval_id, ApprovalDecision::AllowOnce)
                    .unwrap();
            }
            RuntimeCommandOutcome::RunCompleted {
                pending_approval_id: None,
                ..
            } => {
                completed = true;
                break;
            }
            outcome => panic!("unexpected live smoke outcome: {outcome:?}"),
        }
    }
    assert!(completed, "live smoke did not reach final completion");
    assert!(
        approvals >= 2,
        "edit and bash must each be explicitly approved"
    );

    let source = fs::read_to_string(root.workspace().join("src/lib.rs")).unwrap();
    assert!(source.contains("answer() -> i32 { 2 }"));
    let local_test = Command::new("cargo")
        .args(["test", "--quiet"])
        .current_dir(root.workspace())
        .status()
        .unwrap();
    assert!(
        local_test.success(),
        "the edited fixture must pass its test"
    );

    let before = service.session_snapshot(session_id).await.unwrap();
    let before_json = serde_json::to_string(before.records()).unwrap();
    assert!(before_json.contains("cargo test --quiet"));
    assert!(before_json.contains(FINAL_MARKER));
    let branch = before.state().active_branch_id();
    let records = before.records().len();
    service.shutdown().await;
    drop(service);

    let reopened_args = cli_args(&root, &provider, "--continue");
    let reopened_bootstrap = bootstrap(&root, values);
    let (reopened, _) = reopened_bootstrap.build(&reopened_args).unwrap();
    reopened.open_session(session_id).await.unwrap();
    let after = reopened.session_snapshot(session_id).await.unwrap();
    assert_eq!(after.state().active_branch_id(), branch);
    assert_eq!(after.records().len(), records);
    assert!(
        serde_json::to_string(after.records())
            .unwrap()
            .contains(FINAL_MARKER)
    );
    reopened.shutdown().await;
}
