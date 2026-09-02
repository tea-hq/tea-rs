#![forbid(unsafe_code)]

use std::io::IsTerminal as _;
use std::process::ExitCode;

use clap::Parser as _;
use tea_cli::args::CliArgs;
use tea_cli::{BootstrapEnvironment, CliBootstrap, CliFailure};

#[tokio::main(flavor = "current_thread")]
async fn main() -> ExitCode {
    let args = CliArgs::parse();
    match Box::pin(execute(&args)).await {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("tea: {}", error.message());
            error.category().exit_code()
        }
    }
}

async fn execute(args: &CliArgs) -> Result<(), CliFailure> {
    let environment = BootstrapEnvironment::from_process()?;
    let bootstrap = CliBootstrap::new(environment);
    let stdin = std::io::stdin();
    let stdin_is_terminal = stdin.is_terminal();
    let stdout_is_terminal = std::io::stdout().is_terminal();
    if args.app_server {
        Box::pin(tea_cli::app_server::run(
            &args,
            &bootstrap,
            tokio::io::stdin(),
            tokio::io::stdout(),
        ))
        .await
    } else if args.rpc {
        Box::pin(tea_cli::rpc::run(
            args,
            &bootstrap,
            tokio::io::stdin(),
            tokio::io::stdout(),
        ))
        .await
    } else if args.json {
        Box::pin(tea_cli::modes::json::run(
            args,
            &bootstrap,
            &mut stdin.lock(),
            stdin_is_terminal,
            Box::new(std::io::stdout()),
        ))
        .await
    } else if args.print || !stdin_is_terminal {
        Box::pin(tea_cli::modes::print::run(
            args,
            &bootstrap,
            &mut stdin.lock(),
            stdin_is_terminal,
            &mut std::io::stdout().lock(),
            &mut std::io::stderr().lock(),
        ))
        .await
    } else if stdout_is_terminal {
        Box::pin(tea_cli::tui::run(args, &bootstrap)).await
    } else {
        Err(CliFailure::usage(
            "interactive mode requires a terminal on stdin and stdout",
        ))
    }
}
