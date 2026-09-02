use serde_json::json;
use tea_cli::app_server::{
    APP_SERVER_VERSION, AppRequestId, AppServerError, AppServerNotification, AppServerRequest,
    AppServerResponse, CreateSessionParams, InitializeParams,
};

#[test]
fn app_server_request_round_trips_strict_json_rpc_envelope() {
    let request: AppServerRequest = serde_json::from_value(json!({
        "jsonrpc": "2.0",
        "id": "request-1",
        "method": "server/initialize",
        "params": {
            "clientName": "tea-acp",
            "clientVersion": "0.1.0",
            "appServerVersion": APP_SERVER_VERSION
        }
    }))
    .expect("valid app-server request");
    let (id, method, params) = request.validate().expect("valid envelope");
    assert_eq!(id.expect("request id").as_str(), "request-1");
    assert_eq!(method, "server/initialize");
    let parsed: InitializeParams = serde_json::from_value(params).unwrap();
    assert_eq!(parsed.client_name, "tea-acp");

    let encoded = serde_json::to_value(AppServerResponse::success(
        Some(AppRequestId::new("request-1").unwrap()),
        json!({"ok": true}),
    ))
    .unwrap();
    assert_eq!(encoded["jsonrpc"], "2.0");
    assert_eq!(encoded["id"], "request-1");
    assert_eq!(encoded["result"]["ok"], true);
}

#[test]
fn app_server_params_reject_unknown_fields_and_invalid_ids() {
    let unknown = serde_json::from_value::<CreateSessionParams>(json!({
        "cwd": "/tmp/workspace",
        "unexpected": true
    }));
    assert!(unknown.is_err());

    assert!(AppRequestId::new("").is_err());
    assert!(AppRequestId::new("bad\nrequest").is_err());
}

#[test]
fn app_server_notifications_and_errors_are_machine_safe() {
    let notification = serde_json::to_value(AppServerNotification::event(json!({
        "sessionId": "00000000-0000-7000-8000-000000000000"
    })))
    .unwrap();
    assert_eq!(notification["jsonrpc"], "2.0");
    assert_eq!(notification["method"], "event/session");

    let error = serde_json::to_value(AppServerError::internal()).unwrap();
    assert_eq!(
        error,
        json!({"code": -32603, "message": "app-server internal error"})
    );
}
