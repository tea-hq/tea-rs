use std::future::Future;
use std::pin::Pin;
use std::str::FromStr as _;

use futures_util::stream::{FuturesUnordered, StreamExt as _};
use tea::RuntimeCommandOutcome;
use tea_coding::{CodingAgentService, CommandAcceptance};
use tea_protocol::{CommandId, SessionId};
use tea_session::SessionName;
use tokio::io::{AsyncRead, AsyncWrite};

use super::reader::{RpcFrameReader, RpcReadError};
use super::types::{
    RpcError, RpcErrorCode, RpcOutput, RpcRequest, RpcRequestId, RpcRequestKind, RpcResponse,
};
use super::writer::{RpcLineWriter, RpcWriteError};
use crate::args::{CliArgs, SessionSelection};
use crate::session_views::{
    SessionStateView, SessionStatsView, mcp_servers, session_list, session_tree, snapshot_page,
};
use crate::{CliBootstrap, CliFailure, ExitCategory};

/// One connection parses and dispatches at most one request at a time.
pub const MAX_RPC_IN_FLIGHT_REQUESTS: usize = 1;

type OwnedRun<'a> = Pin<
    Box<
        dyn Future<
                Output = (
                    CommandId,
                    SessionId,
                    Result<RuntimeCommandOutcome, tea_coding::CodingError>,
                ),
            > + 'a,
    >,
>;

struct HandledRequest {
    response: RpcResponse,
    selected_session: Option<SessionId>,
    accepted: Option<CommandAcceptance>,
}

#[derive(serde::Deserialize)]
struct RequestCorrelation {
    #[serde(default)]
    id: Option<RpcRequestId>,
}

/// Builds and owns a service for strict JSONL/RPC mode.
///
/// # Errors
///
/// Returns stable CLI failures for bootstrap, framing, output, or cancellation.
pub async fn run<R, W>(
    args: &CliArgs,
    bootstrap: &CliBootstrap,
    input: R,
    output: W,
) -> Result<(), CliFailure>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin + Send + 'static,
{
    if !args.rpc {
        return Err(CliFailure::usage("RPC mode requires --rpc"));
    }
    if !args.prompt.is_empty() {
        return Err(CliFailure::usage(
            "RPC mode accepts prompts only through request frames",
        ));
    }
    if args.output_format.is_some() || args.output_schema.is_some() {
        return Err(CliFailure::usage(
            "RPC structured output must be supplied in prompt request frames",
        ));
    }
    let (service, selection) = bootstrap.build_async(args).await?;
    let result = run_service(&service, selection, input, output).await;
    service.shutdown().await;
    result
}

/// Runs one RPC connection over an existing mode-neutral service.
///
/// EOF and signals return only after the writer is drained; the caller remains
/// responsible for shutting down the borrowed service and its owned runs.
///
/// # Errors
///
/// Returns a terminal framing, I/O, output-backpressure, or signal failure.
#[allow(clippy::too_many_lines)] // Keep connection input/event/run/shutdown ordering auditable.
pub async fn run_service<R, W>(
    service: &CodingAgentService,
    selection: SessionSelection,
    input: R,
    output: W,
) -> Result<(), CliFailure>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin + Send + 'static,
{
    let mut session_id = crate::modes::print::select_session(service, selection).await?;
    let mut events = service.subscribe(session_id).map_err(CliFailure::from)?;
    let mut reader = RpcFrameReader::new(input);
    let writer = RpcLineWriter::spawn(output);
    writer
        .write(&RpcOutput::ready(
            session_id,
            service.workspace_id().clone(),
        ))
        .await
        .map_err(write_failure)?;

    let mut owned_runs = FuturesUnordered::<OwnedRun<'_>>::new();
    let interrupt = tokio::signal::ctrl_c();
    tokio::pin!(interrupt);
    let mut terminal_error = None;

    loop {
        tokio::select! {
            frame = reader.read_frame() => {
                let frame = match frame {
                    Ok(Some(frame)) => frame,
                    Ok(None) => break,
                    Err(error) => {
                        terminal_error = Some(read_failure(error));
                        break;
                    }
                };
                let request = match decode_request(&frame) {
                    Ok(request) => request,
                    Err((request_id, error)) => {
                        writer
                            .write(&RpcOutput::error(request_id, error))
                            .await
                            .map_err(write_failure)?;
                        continue;
                    }
                };
                let request_id = request.id().cloned();
                let request = match request.into_parts() {
                    Ok((_, request)) => request,
                    Err(error) => {
                        writer
                            .write(&RpcOutput::error(request_id, error))
                            .await
                            .map_err(write_failure)?;
                        continue;
                    }
                };
                let handled = match handle_request(service, session_id, request).await {
                    Ok(handled) => handled,
                    Err(error) => {
                        writer
                            .write(&RpcOutput::error(request_id, error))
                            .await
                            .map_err(write_failure)?;
                        continue;
                    }
                };
                if let Some(selected) = handled.selected_session {
                    match service.subscribe(selected) {
                        Ok(subscription) => {
                            session_id = selected;
                            events = subscription;
                        }
                        Err(error) => {
                            writer
                                .write(&RpcOutput::error(request_id, RpcError::from(error)))
                                .await
                                .map_err(write_failure)?;
                            continue;
                        }
                    }
                }
                writer
                    .write(&RpcOutput::response(request_id, handled.response))
                    .await
                    .map_err(write_failure)?;
                if let Some(acceptance) = handled.accepted {
                    owned_runs.push(Box::pin(wait_owned(service, acceptance)));
                }
            }
            event = events.recv() => {
                if let Some(event) = event {
                    writer
                        .write(&RpcOutput::event(event))
                        .await
                        .map_err(write_failure)?;
                } else {
                    let snapshot = service
                        .session_snapshot(session_id)
                        .await
                        .map_err(CliFailure::from)?;
                    events = service.subscribe(session_id).map_err(CliFailure::from)?;
                    writer
                        .write(&RpcOutput::resnapshot_required(
                            session_id,
                            snapshot.state().tail_sequence(),
                        ))
                        .await
                        .map_err(write_failure)?;
                }
            }
            completed = owned_runs.next(), if !owned_runs.is_empty() => {
                if let Some((command_id, completed_session, outcome)) = completed {
                    let error = outcome.err().map(RpcError::from);
                    writer
                        .write(&RpcOutput::command_finished(
                            command_id,
                            completed_session,
                            error,
                        ))
                        .await
                        .map_err(write_failure)?;
                }
            }
            signal = &mut interrupt => {
                terminal_error = Some(match signal {
                    Ok(()) => CliFailure::new(ExitCategory::Cancelled, "RPC operation cancelled"),
                    Err(_) => CliFailure::new(
                        ExitCategory::Internal,
                        "RPC cancellation handler failed",
                    ),
                });
                break;
            }
        }
    }

    drop(owned_runs);
    writer.shutdown().await.map_err(write_failure)?;
    match terminal_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

fn decode_request(frame: &[u8]) -> Result<RpcRequest, (Option<RpcRequestId>, RpcError)> {
    serde_json::from_slice(frame).map_err(|_| {
        if serde_json::from_slice::<serde::de::IgnoredAny>(frame).is_err() {
            (
                None,
                RpcError::new(RpcErrorCode::ParseError, "request JSON is malformed"),
            )
        } else {
            let request_id = serde_json::from_slice::<RequestCorrelation>(frame)
                .ok()
                .and_then(|correlation| correlation.id);
            (
                request_id,
                RpcError::new(RpcErrorCode::InvalidRequest, "request fields are invalid"),
            )
        }
    })
}

async fn wait_owned(
    service: &CodingAgentService,
    acceptance: CommandAcceptance,
) -> (
    CommandId,
    SessionId,
    Result<RuntimeCommandOutcome, tea_coding::CodingError>,
) {
    (
        acceptance.command_id(),
        acceptance.session_id(),
        service.wait(acceptance.session_id()).await,
    )
}

#[allow(clippy::too_many_lines)]
async fn handle_request(
    service: &CodingAgentService,
    session_id: SessionId,
    request: RpcRequestKind,
) -> Result<HandledRequest, RpcError> {
    let mut selected_session = None;
    let mut accepted = None;
    let response = match request {
        RpcRequestKind::NewSession {} => {
            let new_session = service.create_session().await.map_err(RpcError::from)?;
            let state = service
                .snapshot(new_session)
                .await
                .map_err(RpcError::from)?;
            selected_session = Some(new_session);
            RpcResponse::SessionSelected {
                state: SessionStateView::from(&state),
                resnapshot_required: true,
            }
        }
        RpcRequestKind::OpenSession {
            session_id: selected,
        } => {
            let state = service
                .open_session(selected)
                .await
                .map_err(RpcError::from)?;
            selected_session = Some(selected);
            RpcResponse::SessionSelected {
                state: SessionStateView::from(&state),
                resnapshot_required: true,
            }
        }
        RpcRequestKind::NameSession { name } => {
            let name = name
                .map(|name| SessionName::from_str(&name))
                .transpose()
                .map_err(|_| {
                    RpcError::new(RpcErrorCode::InvalidRequest, "session name is invalid")
                })?;
            service
                .name_session(session_id, name)
                .await
                .map_err(RpcError::from)?;
            RpcResponse::CommandCompleted { session_id }
        }
        RpcRequestKind::Prompt {
            text,
            final_output_format,
        } => {
            let acceptance = service
                .prompt(session_id, text, final_output_format)
                .await
                .map_err(RpcError::from)?;
            accepted = Some(acceptance);
            RpcResponse::CommandAccepted {
                command_id: acceptance.command_id(),
                session_id,
            }
        }
        RpcRequestKind::Steer { text } => {
            service
                .steer(session_id, text)
                .await
                .map_err(RpcError::from)?;
            RpcResponse::CommandCompleted { session_id }
        }
        RpcRequestKind::FollowUp { text } => {
            service
                .follow_up(session_id, text)
                .await
                .map_err(RpcError::from)?;
            RpcResponse::CommandCompleted { session_id }
        }
        RpcRequestKind::Abort {} => {
            service.abort(session_id).await.map_err(RpcError::from)?;
            RpcResponse::CommandCompleted { session_id }
        }
        RpcRequestKind::ResolveApproval {
            approval_id,
            decision,
        } => {
            let acceptance = service
                .approve(session_id, approval_id, decision)
                .map_err(RpcError::from)?;
            accepted = Some(acceptance);
            RpcResponse::CommandAccepted {
                command_id: acceptance.command_id(),
                session_id,
            }
        }
        RpcRequestKind::SetModel { model } => {
            service
                .set_model(session_id, model)
                .await
                .map_err(RpcError::from)?;
            RpcResponse::CommandCompleted { session_id }
        }
        RpcRequestKind::Compact {} => {
            service.compact(session_id).await.map_err(RpcError::from)?;
            RpcResponse::CommandCompleted { session_id }
        }
        RpcRequestKind::Fork {
            from_message_id,
            branch_id,
        } => {
            service
                .fork(session_id, from_message_id, branch_id)
                .await
                .map_err(RpcError::from)?;
            RpcResponse::CommandCompleted { session_id }
        }
        RpcRequestKind::ListSessions {} => {
            let sessions = service.list_sessions().await.map_err(RpcError::from)?;
            RpcResponse::Sessions {
                sessions: session_list(&sessions),
            }
        }
        RpcRequestKind::QueryState {} => {
            let state = service.snapshot(session_id).await.map_err(RpcError::from)?;
            RpcResponse::State {
                state: SessionStateView::from(&state),
            }
        }
        RpcRequestKind::QuerySnapshot {
            after_sequence,
            limit,
        } => {
            if limit == 0 {
                return Err(RpcError::new(
                    RpcErrorCode::InvalidRequest,
                    "snapshot limit must be non-zero",
                ));
            }
            let snapshot = service
                .session_snapshot(session_id)
                .await
                .map_err(RpcError::from)?;
            RpcResponse::Snapshot {
                snapshot: snapshot_page(&snapshot, after_sequence, limit),
            }
        }
        RpcRequestKind::QueryStats {} => {
            let stats = service.stats(session_id).await.map_err(RpcError::from)?;
            RpcResponse::Stats {
                stats: SessionStatsView::from(stats),
            }
        }
        RpcRequestKind::QueryTree {} => {
            let snapshot = service
                .session_snapshot(session_id)
                .await
                .map_err(RpcError::from)?;
            RpcResponse::Tree {
                tree: session_tree(&snapshot),
            }
        }
        RpcRequestKind::ListModels {} => RpcResponse::Models {
            models: service.models(),
        },
        RpcRequestKind::ListMcpServers {} => {
            let snapshot = service.mcp_snapshot().map_err(RpcError::from)?;
            RpcResponse::McpServers {
                servers: mcp_servers(&snapshot),
            }
        }
        RpcRequestKind::ReconnectMcp { server_id } => {
            service
                .reconnect_mcp(&server_id)
                .await
                .map_err(RpcError::from)?;
            let snapshot = service.mcp_snapshot().map_err(RpcError::from)?;
            RpcResponse::McpServers {
                servers: mcp_servers(&snapshot),
            }
        }
    };
    Ok(HandledRequest {
        response,
        selected_session,
        accepted,
    })
}

fn read_failure(error: RpcReadError) -> CliFailure {
    let message = match error {
        RpcReadError::Oversize => "RPC input frame exceeds the size limit",
        RpcReadError::Unterminated => "RPC input ended inside a frame",
        RpcReadError::Io => "RPC input is unavailable",
    };
    CliFailure::new(ExitCategory::Usage, message)
}

fn write_failure(error: RpcWriteError) -> CliFailure {
    let category = match error {
        RpcWriteError::InvalidValue => ExitCategory::Internal,
        RpcWriteError::Closed | RpcWriteError::Deadline => ExitCategory::Cancelled,
    };
    CliFailure::new(category, error.to_string())
}
