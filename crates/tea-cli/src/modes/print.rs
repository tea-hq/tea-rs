use std::io::{Read, Write};

use tea::RuntimeCommandOutcome;
use tea_coding::CodingAgentService;
use tea_protocol::{CanonicalMessage, ContentBlock, SessionId};
use tokio::sync::mpsc;

use crate::args::{CliArgs, SessionSelection};
use crate::{CliBootstrap, CliFailure, ExitCategory};

/// Maximum aggregate UTF-8 bytes accepted as one initial prompt.
pub const MAX_INITIAL_PROMPT_BYTES: usize = 256 * 1024;

/// Runs script-safe print mode with injected streams.
///
/// # Errors
///
/// Returns stable categorized failures; stdout remains empty on failure.
pub async fn run(
    args: &CliArgs,
    bootstrap: &CliBootstrap,
    input: &mut dyn Read,
    stdin_is_terminal: bool,
    output: &mut dyn Write,
    diagnostics: &mut dyn Write,
) -> Result<(), CliFailure> {
    if !args.print && stdin_is_terminal {
        return Err(CliFailure::usage(
            "print mode requires --print or piped stdin",
        ));
    }
    let prompt = initial_prompt(args, input, stdin_is_terminal, bootstrap)?;
    let (service, selection) = bootstrap.build_async(args).await?;
    let result = run_service(&service, selection, &prompt, output, diagnostics).await;
    service.shutdown().await;
    result
}

async fn run_service(
    service: &CodingAgentService,
    selection: SessionSelection,
    prompt: &str,
    output: &mut dyn Write,
    diagnostics: &mut dyn Write,
) -> Result<(), CliFailure> {
    let session_id = select_session(service, selection).await?;
    let events = service.subscribe(session_id).map_err(CliFailure::from)?;
    service
        .prompt(session_id, prompt)
        .map_err(CliFailure::from)?;
    let event_task = tokio::spawn(drain_events(events));
    let outcome = tokio::select! {
        outcome = service.wait(session_id) => outcome.map_err(CliFailure::from),
        signal = tokio::signal::ctrl_c() => {
            match signal {
                Ok(()) => Err(CliFailure::new(ExitCategory::Cancelled, "operation cancelled")),
                Err(_) => Err(CliFailure::new(ExitCategory::Internal, "cancellation handler failed")),
            }
        }
    };
    event_task.abort();
    let _ = event_task.await;
    let outcome = outcome?;
    let RuntimeCommandOutcome::RunCompleted {
        session,
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
            "approval is required in non-interactive print mode",
        ));
    }
    let text = final_assistant_text(session.messages()).ok_or_else(|| {
        CliFailure::new(
            ExitCategory::Provider,
            "provider returned no final assistant text",
        )
    })?;
    if text
        .chars()
        .any(|character| character.is_control() && !matches!(character, '\n' | '\t'))
    {
        return Err(CliFailure::new(
            ExitCategory::Provider,
            "provider returned unsafe terminal control text",
        ));
    }
    output
        .write_all(text.as_bytes())
        .and_then(|()| {
            if text.ends_with('\n') {
                Ok(())
            } else {
                output.write_all(b"\n")
            }
        })
        .and_then(|()| output.flush())
        .map_err(|_| CliFailure::new(ExitCategory::Internal, "stdout write failed"))?;
    if !service.resources().diagnostics().is_empty() {
        diagnostics
            .write_all(b"tea: optional workspace context was skipped\n")
            .map_err(|_| CliFailure::new(ExitCategory::Internal, "stderr write failed"))?;
    }
    Ok(())
}

pub(crate) async fn select_session(
    service: &CodingAgentService,
    selection: SessionSelection,
) -> Result<SessionId, CliFailure> {
    match selection {
        SessionSelection::New | SessionSelection::NoSession => {
            service.create_session().await.map_err(CliFailure::from)
        }
        SessionSelection::Existing(session_id) => {
            service
                .open_session(session_id)
                .await
                .map_err(CliFailure::from)?;
            Ok(session_id)
        }
        SessionSelection::Continue => {
            let session_id = service
                .list_sessions()
                .await
                .map_err(CliFailure::from)?
                .first()
                .map(tea_session::SessionCatalogEntry::session_id)
                .ok_or_else(|| {
                    CliFailure::new(ExitCategory::TrustOrConfig, "no session is available")
                })?;
            service
                .open_session(session_id)
                .await
                .map_err(CliFailure::from)?;
            Ok(session_id)
        }
    }
}

pub(crate) fn initial_prompt(
    args: &CliArgs,
    input: &mut dyn Read,
    stdin_is_terminal: bool,
    bootstrap: &CliBootstrap,
) -> Result<String, CliFailure> {
    if args.prompt.len() > 128 {
        return Err(CliFailure::usage("too many prompt arguments"));
    }
    let mut prompt = String::new();
    for value in &args.prompt {
        let part = if let Some(path) = value.strip_prefix('@') {
            bootstrap.read_prompt_file(args, path)?
        } else {
            value.clone()
        };
        append_prompt_part(&mut prompt, &part)?;
    }
    if !stdin_is_terminal {
        let remaining = MAX_INITIAL_PROMPT_BYTES.saturating_sub(prompt.len());
        let mut bytes = Vec::new();
        input
            .take(
                u64::try_from(remaining.saturating_add(1))
                    .map_err(|_| CliFailure::usage("prompt size bound is unsupported"))?,
            )
            .read_to_end(&mut bytes)
            .map_err(|_| CliFailure::usage("stdin could not be read"))?;
        if bytes.len() > remaining {
            return Err(CliFailure::usage("prompt exceeds input size limit"));
        }
        let stdin =
            String::from_utf8(bytes).map_err(|_| CliFailure::usage("stdin is not valid UTF-8"))?;
        append_prompt_part(&mut prompt, &stdin)?;
    }
    if prompt.is_empty() || prompt.contains('\0') {
        return Err(CliFailure::usage("one bounded prompt is required"));
    }
    Ok(prompt)
}

fn append_prompt_part(prompt: &mut String, part: &str) -> Result<(), CliFailure> {
    if part.is_empty() {
        return Ok(());
    }
    let separator = usize::from(!prompt.is_empty());
    if prompt
        .len()
        .checked_add(separator)
        .and_then(|length| length.checked_add(part.len()))
        .is_none_or(|length| length > MAX_INITIAL_PROMPT_BYTES)
    {
        return Err(CliFailure::usage("prompt exceeds input size limit"));
    }
    if separator == 1 {
        prompt.push('\n');
    }
    prompt.push_str(part);
    Ok(())
}

fn final_assistant_text(messages: &[CanonicalMessage]) -> Option<String> {
    let CanonicalMessage::Assistant { content, .. } = messages.last()? else {
        return None;
    };
    let text = content
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } => Some(text.as_str()),
            ContentBlock::ContextualText { .. }
            | ContentBlock::Thinking { .. }
            | ContentBlock::Image { .. }
            | ContentBlock::ToolCall { .. }
            | ContentBlock::HostedTool { .. }
            | ContentBlock::Citation { .. } => None,
        })
        .collect::<String>();
    (!text.is_empty()).then_some(text)
}

async fn drain_events(mut events: mpsc::Receiver<tea_protocol::EventEnvelope>) {
    while events.recv().await.is_some() {}
}
