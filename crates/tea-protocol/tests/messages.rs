use std::str::FromStr;

use serde_json::{Value, json};
use tea_protocol::{
    CanonicalMessage, ContentBlock, ExternalSource, HostedToolActivity, HostedToolError,
    HostedToolOutcome, MessageId, MessageRole, ProtocolTimestamp, ProviderContinuation,
    SourceCitation, StopReason, ToolCallId, ToolFailure,
};

const MESSAGE_ID: &str = "0195a0b1-5e3d-7bb4-863a-0aa7aa000003";
const TOOL_CALL_ID: &str = "0195a0b1-5e45-75be-8284-0aa7aa000011";
const TIMESTAMP: &str = "2026-07-23T09:30:12.124Z";

#[test]
fn content_blocks_use_internal_type_tags() {
    let text = ContentBlock::text("hello").unwrap();
    let context = ContentBlock::contextual_text("hidden instructions").unwrap();
    let thinking = ContentBlock::thinking("checking").unwrap();
    let image = ContentBlock::inline_image("image/png", "aGVsbG8=").unwrap();
    let reference = ContentBlock::image_reference("image/jpeg", "artifact:image-1").unwrap();
    let tool_call = ContentBlock::tool_call(
        ToolCallId::from_str(TOOL_CALL_ID).unwrap(),
        "read_file",
        json!({"path": "/workspace/README.md"}),
    )
    .unwrap();
    let provider_tool_call = ContentBlock::tool_call_with_provider_id(
        ToolCallId::from_str(TOOL_CALL_ID).unwrap(),
        "call_provider_123",
        "read_file",
        json!({"path": "/workspace/README.md"}),
    )
    .unwrap();

    assert_eq!(
        serde_json::to_value(text).unwrap(),
        json!({"type":"text","text":"hello"})
    );
    assert_eq!(
        serde_json::to_value(&context).unwrap(),
        json!({"type":"contextual_text","text":"hidden instructions"})
    );
    assert_eq!(
        serde_json::from_value::<ContentBlock>(serde_json::to_value(&context).unwrap()).unwrap(),
        context
    );
    assert_eq!(
        serde_json::to_value(thinking).unwrap(),
        json!({"type":"thinking","text":"checking"})
    );
    assert_eq!(
        serde_json::to_value(image).unwrap(),
        json!({
            "type":"image",
            "mimeType":"image/png",
            "source":{"type":"inline_base64","data":"aGVsbG8="}
        })
    );
    assert_eq!(
        serde_json::to_value(reference).unwrap(),
        json!({
            "type":"image",
            "mimeType":"image/jpeg",
            "source":{"type":"reference","reference":"artifact:image-1"}
        })
    );
    assert_eq!(
        serde_json::to_value(tool_call).unwrap(),
        json!({
            "type":"tool_call",
            "toolCallId":TOOL_CALL_ID,
            "toolName":"read_file",
            "arguments":{"path":"/workspace/README.md"}
        })
    );
    let provider_tool_call = serde_json::to_value(provider_tool_call).unwrap();
    assert_eq!(provider_tool_call["providerCallId"], "call_provider_123");
    assert_eq!(
        serde_json::from_value::<ContentBlock>(provider_tool_call)
            .unwrap()
            .provider_call_id(),
        Some("call_provider_123")
    );
}

#[test]
fn provider_tool_call_ids_are_bounded() {
    assert!(
        ContentBlock::tool_call_with_provider_id(
            ToolCallId::from_str(TOOL_CALL_ID).unwrap(),
            "",
            "read_file",
            json!({}),
        )
        .is_err()
    );
}

#[test]
fn hosted_activity_sources_citations_and_continuation_round_trip() {
    let tool_call_id = ToolCallId::from_str(TOOL_CALL_ID).unwrap();
    let source = ExternalSource::new("https://docs.rs/serde/latest/serde/")
        .unwrap()
        .with_title("Serde documentation")
        .unwrap()
        .with_snippet("Serialization framework for Rust")
        .unwrap();
    let continuation = ProviderContinuation::new(
        "anthropic",
        "anthropic.messages.web_search.v1",
        json!({"encrypted_content":"secret-provider-state"}),
    )
    .unwrap();
    assert!(!format!("{continuation:?}").contains("secret-provider-state"));

    let activity = HostedToolActivity::new(
        tool_call_id,
        "srvtoolu_123",
        "web_search",
        json!({"query":"serde documentation"}),
        HostedToolOutcome::Success,
        vec![source.clone()],
        Some(continuation.clone()),
    )
    .unwrap();
    let citation = SourceCitation::new(source)
        .with_tool_call_id(tool_call_id)
        .with_range(0, 19)
        .unwrap()
        .with_cited_text("Serde documentation")
        .unwrap()
        .with_continuation(continuation);

    let hosted = ContentBlock::hosted_tool(activity);
    let cited = ContentBlock::citation(citation);
    let hosted_json = serde_json::to_value(&hosted).unwrap();
    assert_eq!(hosted_json["type"], "hosted_tool");
    assert_eq!(hosted_json["activity"]["toolName"], "web_search");
    assert_eq!(
        hosted_json["activity"]["sources"][0]["url"],
        "https://docs.rs/serde/latest/serde/"
    );
    assert_eq!(
        serde_json::from_value::<ContentBlock>(hosted_json).unwrap(),
        hosted
    );
    assert_eq!(
        serde_json::from_value::<ContentBlock>(serde_json::to_value(&cited).unwrap()).unwrap(),
        cited
    );
}

#[test]
fn hosted_content_validation_fails_closed() {
    assert!(ExternalSource::new("ftp://example.com/result").is_err());
    assert!(ExternalSource::new("https://example.com/\nresult").is_err());
    assert!(ExternalSource::new("https://user:secret@example.com/result").is_err());
    assert!(ExternalSource::new("https://").is_err());
    assert_eq!(
        ExternalSource::new("HTTPS://Example.COM:443/a/../result")
            .unwrap()
            .url(),
        "https://example.com/result"
    );
    assert!(
        SourceCitation::new(ExternalSource::new("https://example.com").unwrap())
            .with_range(4, 4)
            .is_err()
    );
    assert!(
        HostedToolError::new("Bad Code", "provider failed").is_err(),
        "error codes must be canonical"
    );
    assert!(
        ProviderContinuation::new(
            "anthropic",
            "anthropic.messages.web_search.v1",
            json!({"payload":"x".repeat(4 * 1024 * 1024)}),
        )
        .is_err()
    );
}

#[test]
fn hosted_content_is_assistant_only() {
    let timestamp = ProtocolTimestamp::from_str(TIMESTAMP).unwrap();
    let message_id = MessageId::from_str(MESSAGE_ID).unwrap();
    let activity = HostedToolActivity::new(
        ToolCallId::from_str(TOOL_CALL_ID).unwrap(),
        "ws_123",
        "web_search",
        json!({"query":"tea-rs"}),
        HostedToolOutcome::Error(HostedToolError::new("unavailable", "try later").unwrap()),
        Vec::new(),
        None,
    )
    .unwrap();
    let block = ContentBlock::hosted_tool(activity);

    assert!(CanonicalMessage::user(message_id, vec![block.clone()], timestamp).is_err());
    CanonicalMessage::assistant(message_id, vec![block], StopReason::Completed, timestamp).unwrap();
}

#[test]
fn contextual_content_is_user_only() {
    let timestamp = ProtocolTimestamp::from_str(TIMESTAMP).unwrap();
    let message_id = MessageId::from_str(MESSAGE_ID).unwrap();
    let block = ContentBlock::contextual_text("model-visible hidden context").unwrap();

    CanonicalMessage::user(message_id, vec![block.clone()], timestamp).unwrap();
    assert!(
        CanonicalMessage::assistant(
            message_id,
            vec![block.clone()],
            StopReason::Completed,
            timestamp
        )
        .is_err()
    );
    assert!(
        CanonicalMessage::tool_result_success(
            message_id,
            ToolCallId::from_str(TOOL_CALL_ID).unwrap(),
            "read_file",
            vec![block],
            timestamp,
        )
        .is_err()
    );
}

#[test]
fn canonical_messages_have_stable_shapes() {
    let message_id = MessageId::from_str(MESSAGE_ID).unwrap();
    let timestamp = ProtocolTimestamp::from_str(TIMESTAMP).unwrap();
    let user = CanonicalMessage::user(
        message_id,
        vec![ContentBlock::text("Inspect the workspace.").unwrap()],
        timestamp,
    )
    .unwrap();

    assert_eq!(user.role(), MessageRole::User);
    assert_eq!(
        serde_json::to_value(&user).unwrap(),
        json!({
            "id":MESSAGE_ID,
            "type":"user",
            "content":[{"type":"text","text":"Inspect the workspace."}],
            "timestamp":TIMESTAMP
        })
    );

    let assistant = CanonicalMessage::assistant(
        message_id,
        vec![ContentBlock::text("Done.").unwrap()],
        StopReason::Completed,
        timestamp,
    )
    .unwrap();
    assert_eq!(assistant.role(), MessageRole::Assistant);
    assert_eq!(
        serde_json::to_value(&assistant).unwrap()["stopReason"],
        "completed"
    );

    let success = CanonicalMessage::tool_result_success(
        message_id,
        ToolCallId::from_str(TOOL_CALL_ID).unwrap(),
        "read_file",
        vec![ContentBlock::text("contents").unwrap()],
        timestamp,
    )
    .unwrap();
    let success_json = serde_json::to_value(success).unwrap();
    assert_eq!(success_json["type"], "tool_result");
    assert_eq!(success_json["isError"], false);
    assert!(success_json.get("error").is_none());

    let failure = CanonicalMessage::tool_result_failure(
        message_id,
        ToolCallId::from_str(TOOL_CALL_ID).unwrap(),
        "read_file",
        vec![ContentBlock::text("Unable to read file.").unwrap()],
        ToolFailure::new("not_found", "file does not exist").unwrap(),
        timestamp,
    )
    .unwrap();
    let failure_json = serde_json::to_value(failure).unwrap();
    assert_eq!(failure_json["isError"], true);
    assert_eq!(failure_json["error"]["code"], "not_found");
}

#[test]
fn message_role_content_invariants_are_enforced() {
    let message_id = MessageId::from_str(MESSAGE_ID).unwrap();
    let timestamp = ProtocolTimestamp::from_str(TIMESTAMP).unwrap();

    assert!(CanonicalMessage::user(message_id, Vec::new(), timestamp).is_err());
    assert!(
        CanonicalMessage::user(
            message_id,
            vec![ContentBlock::thinking("private").unwrap()],
            timestamp,
        )
        .is_err()
    );
    assert!(
        CanonicalMessage::tool_result_success(
            message_id,
            ToolCallId::from_str(TOOL_CALL_ID).unwrap(),
            "read_file",
            vec![
                ContentBlock::tool_call(
                    ToolCallId::from_str(TOOL_CALL_ID).unwrap(),
                    "nested",
                    json!({}),
                )
                .unwrap()
            ],
            timestamp,
        )
        .is_err()
    );
}

#[test]
fn invalid_content_is_rejected_during_construction_and_deserialization() {
    assert!(ContentBlock::text("").is_err());
    assert!(ContentBlock::inline_image("image/png", "not base64!").is_err());
    assert!(ContentBlock::image_reference("invalid", "artifact:1").is_err());
    assert!(
        ContentBlock::tool_call(
            ToolCallId::from_str(TOOL_CALL_ID).unwrap(),
            "invalid tool name",
            json!({}),
        )
        .is_err()
    );

    let invalid_message: Value = json!({
        "id":MESSAGE_ID,
        "type":"user",
        "content":[{"type":"thinking","text":"not allowed"}],
        "timestamp":TIMESTAMP
    });
    assert!(serde_json::from_value::<CanonicalMessage>(invalid_message).is_err());
}

#[test]
fn unknown_stop_reasons_are_preserved_but_not_successful() {
    let reason: StopReason = serde_json::from_str(r#""provider_specific_stop""#).unwrap();
    assert_eq!(reason.as_str(), "provider_specific_stop");
    assert!(!reason.is_success());
    assert_eq!(
        serde_json::to_string(&reason).unwrap(),
        r#""provider_specific_stop""#
    );
}

#[test]
fn provider_pause_turn_is_a_stable_nonterminal_reason() {
    let reason: StopReason = serde_json::from_str(r#""pause_turn""#).unwrap();
    assert_eq!(reason, StopReason::PauseTurn);
    assert_eq!(reason.as_str(), "pause_turn");
    assert!(!reason.is_success());
    assert_eq!(serde_json::to_string(&reason).unwrap(), r#""pause_turn""#);
}

#[test]
fn direct_invalid_enum_construction_cannot_cross_the_wire() {
    let timestamp = ProtocolTimestamp::from_str(TIMESTAMP).unwrap();
    let message_id = MessageId::from_str(MESSAGE_ID).unwrap();

    let empty_text = ContentBlock::Text {
        text: String::new(),
    };
    assert!(serde_json::to_value(empty_text).is_err());

    let empty_message = CanonicalMessage::User {
        id: message_id,
        content: Vec::new(),
        timestamp,
    };
    assert!(serde_json::to_value(empty_message).is_err());

    let invalid_reason = StopReason::Unknown("Invalid Reason".to_owned());
    assert!(serde_json::to_value(invalid_reason).is_err());
}
