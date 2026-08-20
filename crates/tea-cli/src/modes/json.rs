use std::io::{Read, Write};

use serde::Serialize;
use tea::RuntimeCommandOutcome;
use tea_coding::CodingAgentService;
use tea_policy::WorkspaceId;
use tea_protocol::{CURRENT_PROTOCOL_VERSION, ProtocolVersion, SessionId};

use crate::args::{CliArgs, SessionSelection};
use crate::jsonl::{JSON_WRITER_DEADLINE, JsonLineFailure, JsonLineWriter};
use crate::{CliBootstrap, CliFailure, ExitCategory};

const JSON_MODE_VERSION: &str = "1.0";

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct JsonModeHeader<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    mode_version: &'static str,
    protocol_version: ProtocolVersion,
    session_id: SessionId,
    workspace_id: &'a WorkspaceId,
}

/// Runs canonical LF-delimited JSON event mode.
///
/// # Errors
///
/// Returns stable categorized failures. This mode writes only its versioned
/// header and canonical event envelopes to stdout.
pub async fn run(
    args: &CliArgs,
    bootstrap: &CliBootstrap,
    input: &mut dyn Read,
    stdin_is_terminal: bool,
    output: Box<dyn Write + Send>,
) -> Result<(), CliFailure> {
    if !args.json {
        return Err(CliFailure::usage("JSON event mode requires --json"));
    }
    let prompt = super::print::initial_prompt(args, input, stdin_is_terminal, bootstrap)?;
    let (service, selection) = bootstrap.build_async(args).await?;
    let result = Box::pin(run_service(&service, selection, &prompt, output)).await;
    service.shutdown().await;
    result
}

async fn run_service(
    service: &CodingAgentService,
    selection: SessionSelection,
    prompt: &str,
    output: Box<dyn Write + Send>,
) -> Result<(), CliFailure> {
    let session_id = super::print::select_session(service, selection).await?;
    let mut events = service.subscribe(session_id).map_err(CliFailure::from)?;
    let writer = JsonLineWriter::spawn(output, JSON_WRITER_DEADLINE).map_err(writer_failure)?;
    writer
        .write(&JsonModeHeader {
            kind: "tea_event_stream",
            mode_version: JSON_MODE_VERSION,
            protocol_version: CURRENT_PROTOCOL_VERSION,
            session_id,
            workspace_id: service.workspace_id(),
        })
        .await
        .map_err(writer_failure)?;
    service
        .prompt(session_id, prompt)
        .map_err(CliFailure::from)?;

    let mut wait = Box::pin(service.wait(session_id));
    let outcome = loop {
        tokio::select! {
            biased;
            event = events.recv() => match event {
                Some(event) => writer.write(&event).await.map_err(writer_failure)?,
                None => {
                    break wait.await.map_err(CliFailure::from);
                }
            },
            outcome = &mut wait => break outcome.map_err(CliFailure::from),
            signal = tokio::signal::ctrl_c() => {
                break match signal {
                    Ok(()) => Err(CliFailure::new(ExitCategory::Cancelled, "operation cancelled")),
                    Err(_) => Err(CliFailure::new(ExitCategory::Internal, "cancellation handler failed")),
                };
            }
        }
    };

    // Runtime emission is awaited before command completion, so every event
    // already accepted for this run is now present in the bounded receiver.
    while let Ok(event) = events.try_recv() {
        writer.write(&event).await.map_err(writer_failure)?;
    }
    let outcome = outcome?;
    let RuntimeCommandOutcome::RunCompleted {
        pending_approval_id,
        ..
    } = outcome
    else {
        return Err(CliFailure::new(
            ExitCategory::Internal,
            "prompt returned an unexpected outcome",
        ));
    };
    if pending_approval_id.is_some() {
        return Err(CliFailure::new(
            ExitCategory::PolicyDenied,
            "approval is required in non-interactive JSON mode",
        ));
    }
    Ok(())
}

fn writer_failure(error: JsonLineFailure) -> CliFailure {
    match error {
        JsonLineFailure::InvalidValue => {
            CliFailure::new(ExitCategory::Internal, "JSON event serialization failed")
        }
        JsonLineFailure::Closed | JsonLineFailure::Deadline => {
            CliFailure::new(ExitCategory::Cancelled, "JSON output is unavailable")
        }
    }
}
