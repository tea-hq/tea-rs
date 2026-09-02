use std::ffi::OsString;
use std::future::{Future, pending};
use std::pin::Pin;
use std::str::FromStr;
use std::sync::Arc;

use tea_coding::{CodingAgentService, CodingError, CommandAcceptance};
use tea_mcp::{
    McpServerConfig, McpServerId, McpServerLaunch, McpToolDeclaration, McpTransportConfig,
};
use tea_protocol::{EventEnvelope, SessionId, ToolIdempotency};
use tea_tools::{
    ToolConcurrency, ToolEffect, ToolExecutionSemantics, ToolRetrySafety, ToolTimeout, ToolTrust,
};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::mpsc;

use super::types::{
    APP_SERVER_VERSION, AppServerCapabilities, AppServerError, AppServerNotification,
    AppServerRequest, AppServerResponse, CreateSessionParams, CreateSessionResult, EventParams,
    InitializeParams, InitializeResult, ListSessionsParams, LoadSessionParams, PromptParams,
    SessionParams, SetConfigParams, SetModeParams,
};
use crate::args::CliArgs;
use crate::rpc::{RpcFrameReader, RpcLineWriter, RpcReadError, RpcWriteError};
use crate::{CliBootstrap, CliFailure, ExitCategory};

const SERVER_NAME: &str = "tea-app-server";
const SERVER_VERSION: &str = env!("CARGO_PKG_VERSION");

type OwnedRun =
    Pin<Box<dyn Future<Output = Result<tea::RuntimeCommandOutcome, CodingError>> + Send + 'static>>;

/// Builds one Tea app-server service and runs it over stdin/stdout.
///
/// # Errors
///
/// Returns a CLI failure when app-server mode is not selected, protocol input
/// cannot be read, or the output stream cannot be written.
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
    if !args.app_server {
        return Err(CliFailure::usage("app-server mode requires --app-server"));
    }
    run_service(bootstrap, args, input, output).await
}

/// Runs the app-server protocol over an already-built service.
///
/// # Errors
///
/// Returns a CLI failure when protocol input cannot be read, output cannot be
/// written, or service shutdown fails.
#[allow(clippy::too_many_lines)]
pub async fn run_service<R, W>(
    bootstrap: &CliBootstrap,
    args: &CliArgs,
    input: R,
    output: W,
) -> Result<(), CliFailure>
where
    R: AsyncRead + Unpin,
    W: AsyncWrite + Unpin + Send + 'static,
{
    let mut reader = RpcFrameReader::new(input);
    let writer = RpcLineWriter::spawn(output);
    let mut service: Option<Arc<CodingAgentService>> = None;
    let mut session_id = None;
    let mut events = None;
    let mut owned_run = None;
    let mut initialized = false;
    let mut mode = "default".to_owned();
    let mut terminal_error = None;
    let interrupt = tokio::signal::ctrl_c();
    tokio::pin!(interrupt);

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
                let Ok(request) = serde_json::from_slice::<AppServerRequest>(&frame) else {
                    writer.write(&AppServerResponse::failure(
                        None,
                        AppServerError::invalid_request("request JSON is malformed"),
                    )).await.map_err(write_failure)?;
                    continue;
                };
                let fallback_id = request.id.clone();
                let (request_id, method, params) = match request.validate() {
                    Ok(parts) => parts,
                    Err(error) => {
                        writer
                            .write(&AppServerResponse::failure(fallback_id, error))
                            .await
                            .map_err(write_failure)?;
                        continue;
                    }
                };
                let (response, should_stop) = handle_request(
                    bootstrap,
                    args,
                    &mut service,
                    &mut session_id,
                    &mut events,
                    &mut owned_run,
                    &mut initialized,
                    &mut mode,
                    &method,
                    params,
                ).await;
                if request_id.is_some() && let Some(response) = response {
                    writer.write(&with_id(response, request_id)).await.map_err(write_failure)?;
                }
                if should_stop {
                    break;
                }
            }
            event = receive_event(&mut events) => {
                if let Some(event) = event {
                    writer.write(&event_notification(&event)).await.map_err(write_failure)?;
                } else if let (Some(session_id), Some(service)) = (session_id, service.as_ref()) {
                    writer.write(&AppServerNotification::event(serde_json::json!({
                        "sessionId": session_id,
                        "type": "resync_required"
                    }))).await.map_err(write_failure)?;
                    events = Some(service.subscribe(session_id).map_err(CliFailure::from)?);
                }
            }
            completed = wait_owned_run(&mut owned_run), if owned_run.is_some() => {
                let result = completed;
                owned_run = None;
                let params = match result {
                    Ok(tea::RuntimeCommandOutcome::RunCompleted { pending_approval_id: Some(approval_id), .. }) => serde_json::json!({
                        "status": "awaiting_approval",
                        "approvalId": approval_id,
                    }),
                    Ok(_) => serde_json::json!({"status": "completed"}),
                    Err(error) => serde_json::json!({
                        "status": if error.code() == tea_coding::CodingErrorCode::Cancelled { "cancelled" } else { "failed" },
                        "error": error.message()
                    }),
                };
                writer.write(&AppServerNotification {
                    jsonrpc: "2.0",
                    method: "event/turn_completed",
                    params: match session_id {
                        Some(session_id) => {
                            let mut params = params;
                            if let Some(object) = params.as_object_mut() {
                                object.insert("sessionId".to_owned(), serde_json::json!(session_id));
                            }
                            params
                        }
                        None => params,
                    },
                }).await.map_err(write_failure)?;
            }
            signal = &mut interrupt => {
                terminal_error = Some(match signal {
                    Ok(()) => CliFailure::new(ExitCategory::Cancelled, "app-server operation cancelled"),
                    Err(_) => CliFailure::new(ExitCategory::Internal, "app-server cancellation handler failed"),
                });
                break;
            }
        }
    }

    drop(owned_run);
    writer.shutdown().await.map_err(write_failure)?;
    if let Some(service) = service {
        service.shutdown().await;
    }
    match terminal_error {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
async fn handle_request(
    bootstrap: &CliBootstrap,
    args: &CliArgs,
    service: &mut Option<Arc<CodingAgentService>>,
    session_id: &mut Option<SessionId>,
    events: &mut Option<mpsc::Receiver<EventEnvelope>>,
    owned_run: &mut Option<OwnedRun>,
    initialized: &mut bool,
    mode: &mut String,
    method: &str,
    params: serde_json::Value,
) -> (Option<AppServerResponse>, bool) {
    let response = match method {
        "server/initialize" => match parse_params::<InitializeParams>(params) {
            Ok(params) => {
                if params.app_server_version == APP_SERVER_VERSION {
                    *initialized = true;
                    Ok(serde_json::to_value(InitializeResult {
                        app_server_version: APP_SERVER_VERSION,
                        server_name: SERVER_NAME,
                        server_version: SERVER_VERSION,
                        capabilities: AppServerCapabilities {
                            session_resume: true,
                            session_mcp: true,
                            permission_requests: true,
                            session_list: true,
                            session_config: true,
                        },
                    })
                    .expect("app-server initialize result serializes"))
                } else {
                    Err(AppServerError::invalid_request(
                        "app-server version is unsupported",
                    ))
                }
            }
            Err(error) => Err(error),
        },
        "session/create" => {
            if !*initialized {
                Err(AppServerError::invalid_request(
                    "server must be initialized first",
                ))
            } else if session_id.is_some() {
                Err(AppServerError::invalid_request(
                    "app-server session already exists",
                ))
            } else if owned_run.is_some() {
                Err(AppServerError::invalid_request(
                    "app-server session is busy",
                ))
            } else {
                create_session(bootstrap, args, service, session_id, events, mode, params).await
            }
        }
        "session/load" | "session/resume" => {
            if !*initialized {
                Err(AppServerError::invalid_request(
                    "server must be initialized first",
                ))
            } else if session_id.is_some() {
                Err(AppServerError::invalid_request(
                    "app-server session already exists",
                ))
            } else {
                load_session(bootstrap, args, service, session_id, events, mode, params).await
            }
        }
        "session/list" => {
            if *initialized {
                list_sessions(bootstrap, args, service, params).await
            } else {
                Err(AppServerError::invalid_request(
                    "server must be initialized first",
                ))
            }
        }
        "session/prompt" => {
            let parsed = parse_params::<PromptParams>(params);
            match parsed {
                Ok(prompt) => service
                    .as_ref()
                    .ok_or_else(|| AppServerError::invalid_request("session has not been created"))
                    .and_then(|service| {
                        start_prompt(Arc::clone(service), *session_id, owned_run, prompt)
                    }),
                Err(error) => Err(error),
            }
        }
        "session/cancel" => match parse_params::<SessionParams>(params) {
            Ok(params) => {
                if Some(params.session_id) == *session_id {
                    let Some(service) = service.as_ref() else {
                        return (
                            Some(AppServerResponse::failure(
                                None,
                                AppServerError::invalid_request("session has not been created"),
                            )),
                            false,
                        );
                    };
                    service
                        .abort(params.session_id)
                        .await
                        .map(|_| serde_json::json!({"accepted": true}))
                        .map_err(coding_error)
                } else {
                    Err(AppServerError::invalid_request("session is not attached"))
                }
            }
            Err(error) => Err(error),
        },
        "session/close" => match parse_params::<SessionParams>(params) {
            Ok(params) if Some(params.session_id) == *session_id => {
                if let Some(service) = service.as_ref() {
                    let _ = service.abort(params.session_id).await;
                }
                if let Some(run) = owned_run.take() {
                    let _ = run.await;
                }
                // The service owns session-scoped MCP processes and an
                // immutable tool catalog. Tear it down with the session so a
                // later session/create can supply a different descriptor set.
                if let Some(service) = service.take() {
                    service.shutdown().await;
                }
                *session_id = None;
                *events = None;
                "default".clone_into(mode);
                Ok(serde_json::json!({"closed": true}))
            }
            Ok(_) => Err(AppServerError::invalid_request("session is not attached")),
            Err(error) => Err(error),
        },
        "session/set_mode" => match parse_params::<SetModeParams>(params) {
            Ok(params) if Some(params.session_id) == *session_id => {
                match validate_mode(&params.mode) {
                    Ok(()) => {
                        mode.clone_from(&params.mode);
                        Ok(serde_json::json!({"mode": mode}))
                    }
                    Err(error) => Err(error),
                }
            }
            Ok(_) => Err(AppServerError::invalid_request("session is not attached")),
            Err(error) => Err(error),
        },
        "session/set_config" => match parse_params::<SetConfigParams>(params) {
            Ok(params) if Some(params.session_id) == *session_id => {
                match params.config_id.as_str() {
                    "mode" | "permission_mode" => match validate_mode(&params.value) {
                        Ok(()) => {
                            mode.clone_from(&params.value);
                            Ok(serde_json::json!({"mode": mode}))
                        }
                        Err(error) => Err(error),
                    },
                    "model" | "model_ref" => {
                        match (service.as_ref(), parse_model_ref(&params.value)) {
                            (Some(service), Ok(model)) => service
                                .set_model(params.session_id, model)
                                .await
                                .map(|_| serde_json::json!({"updated": true}))
                                .map_err(coding_error),
                            (None, _) => Err(AppServerError::invalid_request(
                                "session has not been created",
                            )),
                            (_, Err(error)) => Err(error),
                        }
                    }
                    _ => Err(AppServerError::invalid_params(
                        "configuration option is unsupported",
                    )),
                }
            }
            Ok(_) => Err(AppServerError::invalid_request("session is not attached")),
            Err(error) => Err(error),
        },
        "approval/respond" => match parse_params::<super::types::ApprovalResponseParams>(params) {
            Ok(params) => {
                if Some(params.session_id) == *session_id {
                    // A prompt can still be in the process of returning its
                    // pending-approval checkpoint when the adapter responds.
                    // Let that owned command finish before starting the
                    // explicit approval continuation for the same session.
                    if let Some(run) = owned_run.take()
                        && let Err(error) = run.await
                    {
                        return (
                            Some(AppServerResponse::failure(None, coding_error(error))),
                            false,
                        );
                    }
                    let decision = approval_decision_for_mode(mode, params.decision);
                    match service.as_ref() {
                        Some(service) => match service
                            .approve(params.session_id, params.approval_id, decision)
                            .map_err(coding_error)
                        {
                            Ok(acceptance) => {
                                *owned_run =
                                    Some(Box::pin(wait_owned(Arc::clone(service), acceptance)));
                                Ok(
                                    serde_json::json!({"accepted": true, "commandId": acceptance.command_id()}),
                                )
                            }
                            Err(error) => Err(error),
                        },
                        None => Err(AppServerError::invalid_request(
                            "session has not been created",
                        )),
                    }
                } else {
                    Err(AppServerError::invalid_request("session is not attached"))
                }
            }
            Err(error) => Err(error),
        },
        "server/shutdown" => Ok(serde_json::json!({"stopping": true})),
        _ => Err(AppServerError::method_not_found()),
    };

    match response {
        Ok(result) => (
            Some(AppServerResponse::success(None, result)),
            method == "server/shutdown",
        ),
        Err(error) => (Some(AppServerResponse::failure(None, error)), false),
    }
}

async fn create_session(
    bootstrap: &CliBootstrap,
    args: &CliArgs,
    service: &mut Option<Arc<CodingAgentService>>,
    session_id: &mut Option<SessionId>,
    events: &mut Option<mpsc::Receiver<EventEnvelope>>,
    mode: &mut String,
    params: serde_json::Value,
) -> Result<serde_json::Value, AppServerError> {
    let params = parse_params::<CreateSessionParams>(params)?;
    let requested_mode = params.mode.as_deref().unwrap_or("default");
    validate_mode(requested_mode)?;
    let extra = mcp_descriptors(&params.mcp_servers)?;
    let extra_server_ids = extra
        .iter()
        .map(|launch| launch.config().id().clone())
        .collect::<Vec<McpServerId>>();
    if let Some(previous) = service.take() {
        previous.shutdown().await;
    }
    let candidate = Arc::new(
        bootstrap
            .build_async_with_mcp_launches(args, extra)
            .await
            .map_err(cli_failure)?
            .0,
    );
    let result = async {
        if !workspace_matches(&params.cwd, candidate.workspace().host_path()) {
            return Err(AppServerError::invalid_params(
                "session workspace does not match app-server workspace",
            ));
        }
        if let Some(model) = params.model.clone() {
            candidate.validate_model(&model).map_err(coding_error)?;
        }
        let id = candidate.create_session().await.map_err(coding_error)?;
        if let Some(model) = params.model.clone() {
            candidate.set_model(id, model).await.map_err(coding_error)?;
        }
        activate_mcp_tools(&candidate, id, &extra_server_ids).await?;
        let events_receiver = candidate.subscribe(id).map_err(coding_error)?;
        let snapshot = candidate.snapshot(id).await.map_err(coding_error)?;
        let result = serde_json::to_value(CreateSessionResult {
            session_id: id,
            model: snapshot.model_ref().cloned(),
            available_models: candidate.models(),
            mode: requested_mode.to_owned(),
        })
        .map_err(|_| AppServerError::internal())?;
        *events = Some(events_receiver);
        *session_id = Some(id);
        requested_mode.clone_into(mode);
        Ok(result)
    }
    .await;
    match result {
        Ok(result) => {
            *service = Some(candidate);
            Ok(result)
        }
        Err(error) => {
            candidate.shutdown().await;
            Err(error)
        }
    }
}

async fn load_session(
    bootstrap: &CliBootstrap,
    args: &CliArgs,
    service: &mut Option<Arc<CodingAgentService>>,
    session_id: &mut Option<SessionId>,
    events: &mut Option<mpsc::Receiver<EventEnvelope>>,
    mode: &mut String,
    params: serde_json::Value,
) -> Result<serde_json::Value, AppServerError> {
    let params = parse_params::<LoadSessionParams>(params)?;
    let extra = mcp_descriptors(&params.mcp_servers)?;
    let extra_server_ids = extra
        .iter()
        .map(|launch| launch.config().id().clone())
        .collect::<Vec<McpServerId>>();
    if let Some(previous) = service.take() {
        previous.shutdown().await;
    }
    let candidate = Arc::new(
        bootstrap
            .build_async_with_mcp_launches(args, extra)
            .await
            .map_err(cli_failure)?
            .0,
    );
    let result = async {
        if !workspace_matches(&params.cwd, candidate.workspace().host_path()) {
            return Err(AppServerError::invalid_params(
                "session workspace does not match app-server workspace",
            ));
        }
        candidate
            .open_session(params.session_id)
            .await
            .map_err(coding_error)?;
        activate_mcp_tools(&candidate, params.session_id, &extra_server_ids).await?;
        let events_receiver = candidate
            .subscribe(params.session_id)
            .map_err(coding_error)?;
        let snapshot = candidate
            .snapshot(params.session_id)
            .await
            .map_err(coding_error)?;
        let result = serde_json::json!({
            "sessionId": params.session_id,
            "mode": "default",
            "model": snapshot.model_ref(),
            "availableModels": candidate.models(),
        });
        *events = Some(events_receiver);
        *session_id = Some(params.session_id);
        "default".clone_into(mode);
        Ok(result)
    }
    .await;
    match result {
        Ok(result) => {
            *service = Some(candidate);
            Ok(result)
        }
        Err(error) => {
            candidate.shutdown().await;
            Err(error)
        }
    }
}

async fn list_sessions(
    bootstrap: &CliBootstrap,
    args: &CliArgs,
    service: &mut Option<Arc<CodingAgentService>>,
    params: serde_json::Value,
) -> Result<serde_json::Value, AppServerError> {
    let params = parse_params::<ListSessionsParams>(params)?;
    if service.is_none() {
        *service = Some(Arc::new(
            bootstrap
                .build_app_server_async(args)
                .await
                .map_err(cli_failure)?
                .0,
        ));
    }
    let service = service.as_ref().ok_or_else(AppServerError::internal)?;
    let mut sessions = Vec::new();
    for entry in service.list_sessions().await.map_err(coding_error)? {
        if params
            .cwd
            .as_deref()
            .is_some_and(|cwd| !workspace_matches(cwd, service.workspace().host_path()))
        {
            continue;
        }
        sessions.push(serde_json::json!({
            "sessionId": entry.session_id(),
            "name": entry.name().map(ToString::to_string),
            "model": entry.model_ref(),
            "updatedAt": entry.updated_at(),
            "messageCount": entry.message_count(),
            "pendingApprovalCount": entry.pending_approval_count(),
        }));
    }
    Ok(serde_json::json!({"sessions": sessions, "nextCursor": null}))
}

fn start_prompt(
    service: Arc<CodingAgentService>,
    session_id: Option<SessionId>,
    owned_run: &mut Option<OwnedRun>,
    params: PromptParams,
) -> Result<serde_json::Value, AppServerError> {
    if Some(params.session_id) != session_id {
        return Err(AppServerError::invalid_request("session is not attached"));
    }
    if owned_run.is_some() {
        return Err(AppServerError::invalid_request("session is busy"));
    }
    let acceptance = service
        .prompt(params.session_id, params.text)
        .map_err(coding_error)?;
    *owned_run = Some(Box::pin(wait_owned(service, acceptance)));
    Ok(serde_json::json!({"accepted": true, "commandId": acceptance.command_id()}))
}

async fn wait_owned(
    service: Arc<CodingAgentService>,
    acceptance: CommandAcceptance,
) -> Result<tea::RuntimeCommandOutcome, CodingError> {
    service.wait_owned(acceptance.session_id()).await
}

async fn wait_owned_run(
    owned_run: &mut Option<OwnedRun>,
) -> Result<tea::RuntimeCommandOutcome, CodingError> {
    owned_run
        .as_mut()
        .expect("owned run is present when selected")
        .await
}

async fn receive_event(
    events: &mut Option<mpsc::Receiver<EventEnvelope>>,
) -> Option<EventEnvelope> {
    match events {
        Some(events) => events.recv().await,
        None => pending().await,
    }
}

fn event_notification(event: &EventEnvelope) -> AppServerNotification {
    let value = serde_json::to_value(event).unwrap_or_else(|_| {
        serde_json::json!({
            "type": "event_serialization_failed"
        })
    });
    AppServerNotification::event(
        serde_json::to_value(EventParams {
            session_id: event.session_id(),
            sequence: Some(event.sequence()),
            event: value,
        })
        .expect("event params serialize"),
    )
}

fn with_id(
    mut response: AppServerResponse,
    id: Option<super::types::AppRequestId>,
) -> AppServerResponse {
    response.id = id;
    response
}

fn parse_params<T: serde::de::DeserializeOwned>(
    params: serde_json::Value,
) -> Result<T, AppServerError> {
    serde_json::from_value(params)
        .map_err(|_| AppServerError::invalid_params("request parameters are invalid"))
}

#[allow(clippy::needless_pass_by_value)]
fn coding_error(error: CodingError) -> AppServerError {
    let code = match error.code() {
        tea_coding::CodingErrorCode::InvalidInput => -32602,
        tea_coding::CodingErrorCode::Cancelled => -32800,
        tea_coding::CodingErrorCode::PolicyDenied => -32003,
        tea_coding::CodingErrorCode::Persistence => -32004,
        _ => -32000,
    };
    AppServerError {
        code,
        message: error.message().to_owned(),
    }
}

fn validate_mode(mode: &str) -> Result<(), AppServerError> {
    if matches!(mode, "read-only" | "default" | "full-access") {
        Ok(())
    } else {
        Err(AppServerError::invalid_params(
            "session mode is unsupported",
        ))
    }
}

fn approval_decision_for_mode(
    mode: &str,
    requested: tea_protocol::ApprovalDecision,
) -> tea_protocol::ApprovalDecision {
    match mode {
        "read-only" => tea_protocol::ApprovalDecision::Deny,
        "full-access" => tea_protocol::ApprovalDecision::AllowOnce,
        _ => requested,
    }
}

fn parse_model_ref(value: &str) -> Result<tea_protocol::ModelRef, AppServerError> {
    if let Ok(model) = serde_json::from_str(value) {
        return Ok(model);
    }
    let (provider, model) = value
        .split_once(':')
        .or_else(|| value.split_once('/'))
        .ok_or_else(|| AppServerError::invalid_params("model must be provider:model"))?;
    let provider = tea_protocol::ProviderId::from_str(provider)
        .map_err(|_| AppServerError::invalid_params("model provider is invalid"))?;
    let model = tea_protocol::ModelId::from_str(model)
        .map_err(|_| AppServerError::invalid_params("model identifier is invalid"))?;
    Ok(tea_protocol::ModelRef::new(provider, model))
}

fn mcp_descriptors(
    descriptors: &[super::types::McpServerDescriptor],
) -> Result<Vec<McpServerLaunch>, AppServerError> {
    descriptors
        .iter()
        .map(|descriptor| {
            let id = tea_mcp::McpServerId::from_str(&descriptor.name)
                .map_err(|_| AppServerError::invalid_params("MCP server name is invalid"))?;
            let transport = McpTransportConfig::stdio(
                &descriptor.command,
                descriptor.args.iter().cloned().map(OsString::from),
            )
            .map_err(|_| AppServerError::invalid_params("MCP server command is invalid"))?;
            let environment = descriptor
                .env
                .iter()
                .map(|variable| (variable.name.clone(), variable.value.clone()))
                .collect::<Vec<_>>();
            let declaration = McpToolDeclaration::new(
                [ToolEffect::ExternalMutation],
                std::iter::empty(),
                ToolExecutionSemantics::new(
                    ToolIdempotency::NonIdempotent,
                    ToolRetrySafety::Never,
                    ToolConcurrency::Serial,
                    ToolTimeout::from_millis(120_000).map_err(|_| AppServerError::internal())?,
                )
                .map_err(|_| AppServerError::internal())?,
            )
            .map_err(|_| AppServerError::internal())?;
            let config = McpServerConfig::new(
                id,
                transport,
                environment.iter().map(|(name, _)| name.clone()).collect(),
                Vec::new(),
                tea_mcp::McpLimits::default(),
                tea_mcp::McpLifecyclePolicy::default(),
                tea_mcp::McpReconnectPolicy::default(),
            )
            .map_err(|_| AppServerError::invalid_params("MCP server configuration is invalid"))?
            .with_default_tool_declaration(declaration);
            McpServerLaunch::new(config, ToolTrust::User, environment)
                .map_err(|_| AppServerError::invalid_params("MCP server environment is invalid"))
        })
        .collect()
}

async fn activate_mcp_tools(
    service: &CodingAgentService,
    session_id: SessionId,
    session_mcp_server_ids: &[McpServerId],
) -> Result<(), AppServerError> {
    let mut names = service
        .default_active_tool_names(session_id)
        .await
        .map_err(coding_error)?;
    names.extend(service.mcp_tool_names_for_servers(session_mcp_server_ids));
    names.sort();
    names.dedup();
    service
        .set_active_tools(session_id, names)
        .await
        .map_err(coding_error)
}

#[allow(clippy::needless_pass_by_value)]
fn cli_failure(error: CliFailure) -> AppServerError {
    AppServerError {
        code: -32001,
        message: error.message().to_owned(),
    }
}

fn read_failure(error: RpcReadError) -> CliFailure {
    let message = match error {
        RpcReadError::Oversize => "app-server input frame exceeds the size limit",
        RpcReadError::Unterminated => "app-server input ended inside a frame",
        RpcReadError::Io => "app-server input is unavailable",
    };
    CliFailure::new(ExitCategory::Usage, message)
}

fn workspace_matches(requested: &str, actual: &std::path::Path) -> bool {
    std::path::Path::new(requested).canonicalize().map_or_else(
        |_| requested == actual.to_string_lossy(),
        |path| path == actual,
    )
}

fn write_failure(error: RpcWriteError) -> CliFailure {
    let category = match error {
        RpcWriteError::InvalidValue => ExitCategory::Internal,
        RpcWriteError::Closed | RpcWriteError::Deadline => ExitCategory::Cancelled,
    };
    CliFailure::new(category, error.to_string())
}

#[cfg(test)]
mod tests {
    use super::approval_decision_for_mode;
    use tea_protocol::ApprovalDecision;

    #[test]
    fn approval_decision_is_bounded_by_session_mode() {
        assert_eq!(
            approval_decision_for_mode("read-only", ApprovalDecision::AllowOnce),
            ApprovalDecision::Deny
        );
        assert_eq!(
            approval_decision_for_mode("full-access", ApprovalDecision::Deny),
            ApprovalDecision::AllowOnce
        );
        assert_eq!(
            approval_decision_for_mode("default", ApprovalDecision::AllowSession),
            ApprovalDecision::AllowSession
        );
    }
}
