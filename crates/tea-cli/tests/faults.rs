use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::pin::Pin;
use std::str::FromStr as _;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use clap::Parser as _;
use tea_cli::args::CliArgs;
use tea_cli::rpc::MAX_RPC_FRAME_BYTES;
use tea_cli::{BootstrapEnvironment, CliBootstrap, ExitCategory};
use tea_model::{ModelCapabilities, ModelDisplayName, ModelFailureCode, ModelSpec, ProviderId};
use tea_protocol::{ModelId, TokenCount};
use tea_testkit::{ScriptedModelProvider, ScriptedModelResponse};
use tokio::io::{AsyncReadExt as _, AsyncWrite, AsyncWriteExt as _};

const SECRET: &str = "sk-seeded-provider-fault";

struct BrokenOutput;

impl Write for BrokenOutput {
    fn write(&mut self, _bytes: &[u8]) -> io::Result<usize> {
        Err(io::Error::new(io::ErrorKind::BrokenPipe, "injected"))
    }

    fn flush(&mut self) -> io::Result<()> {
        Err(io::Error::new(io::ErrorKind::BrokenPipe, "injected"))
    }
}

struct BrokenAsyncOutput;

impl AsyncWrite for BrokenAsyncOutput {
    fn poll_write(
        self: Pin<&mut Self>,
        _context: &mut Context<'_>,
        _bytes: &[u8],
    ) -> Poll<io::Result<usize>> {
        Poll::Ready(Err(io::Error::new(io::ErrorKind::BrokenPipe, "injected")))
    }

    fn poll_flush(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Err(io::Error::new(io::ErrorKind::BrokenPipe, "injected")))
    }

    fn poll_shutdown(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Ok(()))
    }
}

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

fn temp_root(label: &str) -> std::path::PathBuf {
    let root = std::env::temp_dir().join(format!(
        "tea-cli-fault-{label}-{}",
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

fn bootstrap(root: &Path, script: ScriptedModelResponse) -> CliBootstrap {
    let provider = Arc::new(ScriptedModelProvider::new(
        ProviderId::from_str("fake").unwrap(),
        vec![model()],
        [script],
    ));
    CliBootstrap::new(BootstrapEnvironment::new(
        root,
        Some(root.to_path_buf()),
        BTreeMap::new(),
    ))
    .with_provider(provider)
}

fn args(root: &Path, mode: &str) -> CliArgs {
    let mut values = vec![
        "tea".to_owned(),
        mode.to_owned(),
        "--no-session".to_owned(),
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
    if mode == "--print" || mode == "--json" {
        values.push("hello".to_owned());
    }
    CliArgs::try_parse_from(values).unwrap()
}

#[tokio::test(flavor = "current_thread")]
async fn print_broken_stdout_fails_and_shuts_down_within_deadline() {
    let root = temp_root("print");
    let result = tokio::time::timeout(
        Duration::from_secs(1),
        Box::pin(tea_cli::modes::print::run(
            &args(&root, "--print"),
            &bootstrap(&root, ScriptedModelResponse::text(["answer"])),
            &mut io::empty(),
            true,
            &mut BrokenOutput,
            &mut Vec::new(),
        )),
    )
    .await
    .expect("print mode shutdown deadline")
    .unwrap_err();
    assert_eq!(result.category(), ExitCategory::Internal);
    assert_eq!(result.message(), "stdout write failed");
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn json_provider_failures_are_classified_and_redacted() {
    for code in [
        ModelFailureCode::Authentication,
        ModelFailureCode::MalformedResponse,
    ] {
        let root = temp_root(&format!("json-{code:?}"));
        let bytes = Arc::new(Mutex::new(Vec::new()));
        let result = tokio::time::timeout(
            Duration::from_secs(1),
            Box::pin(tea_cli::modes::json::run(
                &args(&root, "--json"),
                &bootstrap(&root, ScriptedModelResponse::failure(code, SECRET)),
                &mut io::empty(),
                true,
                Box::new(SharedOutput(Arc::clone(&bytes))),
            )),
        )
        .await
        .expect("JSON mode shutdown deadline")
        .unwrap_err();
        assert_eq!(result.category(), ExitCategory::Provider);
        assert!(!format!("{result:?}").contains(SECRET));
        assert!(!String::from_utf8_lossy(&bytes.lock().unwrap()).contains(SECRET));
        fs::remove_dir_all(root).unwrap();
    }
}

async fn rpc_input_failure(root: &Path, input: Vec<u8>) -> tea_cli::CliFailure {
    let (mut client_input, server_input) = tokio::io::duplex(64 * 1024);
    let (server_output, mut client_output) = tokio::io::duplex(64 * 1024);
    let send = tokio::spawn(async move {
        let _ = client_input.write_all(&input).await;
        let _ = client_input.shutdown().await;
    });
    let drain = tokio::spawn(async move {
        let mut output = Vec::new();
        client_output.read_to_end(&mut output).await.unwrap();
        output
    });
    let result = tokio::time::timeout(
        Duration::from_secs(1),
        Box::pin(tea_cli::rpc::run(
            &args(root, "--rpc"),
            &bootstrap(root, ScriptedModelResponse::text(["unused"])),
            server_input,
            server_output,
        )),
    )
    .await
    .expect("RPC input failure shutdown deadline")
    .unwrap_err();
    send.await.unwrap();
    let output = drain.await.unwrap();
    assert!(output.ends_with(b"\n"));
    assert!(!output.contains(&0x1b));
    result
}

#[tokio::test(flavor = "current_thread")]
async fn rpc_oversize_and_unterminated_frames_are_terminal_and_bounded() {
    let root = temp_root("rpc-input");
    let oversized = rpc_input_failure(&root, vec![b'x'; MAX_RPC_FRAME_BYTES + 1]).await;
    assert_eq!(oversized.category(), ExitCategory::Usage);
    assert_eq!(
        oversized.message(),
        "RPC input frame exceeds the size limit"
    );

    let unterminated = rpc_input_failure(&root, b"{}".to_vec()).await;
    assert_eq!(unterminated.category(), ExitCategory::Usage);
    assert_eq!(unterminated.message(), "RPC input ended inside a frame");
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn rpc_broken_output_fails_and_shuts_down_within_deadline() {
    let root = temp_root("rpc-output");
    let result = tokio::time::timeout(
        Duration::from_secs(1),
        Box::pin(tea_cli::rpc::run(
            &args(&root, "--rpc"),
            &bootstrap(&root, ScriptedModelResponse::text(["unused"])),
            tokio::io::empty(),
            BrokenAsyncOutput,
        )),
    )
    .await
    .expect("RPC output failure shutdown deadline")
    .unwrap_err();
    assert_eq!(result.category(), ExitCategory::Cancelled);
    assert_eq!(result.message(), "RPC output is closed");
    fs::remove_dir_all(root).unwrap();
}
