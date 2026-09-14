use std::fs;
use std::path::PathBuf;
use std::str::FromStr;

use serde_json::{Value, json};
use tea_protocol::{
    AgentCommand, AgentCommandType, AgentErrorCode, CommandDecodeError, CommandEnvelope,
    CorrelationId, FinalOutputFormat, MessageRole, ModelId, ProfileId, ReasoningEffort,
};

const COMMAND_ID: &str = "0195a0b1-5e3c-70a1-927f-0aa7aa000002";
const SESSION_ID: &str = "0195a0b1-5e3a-7d72-a902-c4e85d828bf1";
const TIMESTAMP: &str = "2026-07-23T09:30:12.124Z";
const CORRELATION_ID: &str = "0195a0b1-5e3b-7ef0-8ec1-0aa7aa000001";

fn fixture(name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/v1.0")
        .join(name);
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn command_fixtures_round_trip_with_explicit_shapes() {
    for name in [
        "command-create-session.json",
        "command-prompt.json",
        "command-prompt-json-schema.json",
        "command-steer.json",
        "command-follow-up.json",
        "command-abort.json",
        "command-resolve-approval.json",
        "command-set-model.json",
        "command-set-profile.json",
        "command-compact-session.json",
        "command-fork-session.json",
    ] {
        let value = fixture(name);
        let envelope: CommandEnvelope = serde_json::from_value(value.clone()).unwrap();
        assert_eq!(serde_json::to_value(envelope).unwrap(), value, "{name}");
    }

    let prompt: CommandEnvelope = serde_json::from_value(fixture("command-prompt.json")).unwrap();
    assert_eq!(prompt.command_type(), AgentCommandType::Prompt);
    let AgentCommand::Prompt {
        message,
        final_output_format,
    } = prompt.command()
    else {
        panic!("expected prompt command");
    };
    assert_eq!(message.role(), MessageRole::User);
    assert!(final_output_format.is_none());
}

#[test]
fn prompt_final_output_formats_have_stable_wire_shapes() {
    let schema_value = fixture("command-prompt-json-schema.json");
    let schema_prompt: CommandEnvelope = serde_json::from_value(schema_value.clone()).unwrap();
    let AgentCommand::Prompt {
        final_output_format: Some(FinalOutputFormat::JsonSchema { schema }),
        ..
    } = schema_prompt.command()
    else {
        panic!("expected JSON Schema prompt command");
    };
    assert_eq!(
        schema,
        &json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": false
        })
    );
    let mut object_value = fixture("command-prompt.json");
    object_value["payload"]["finalOutputFormat"] = json!({"type":"json_object"});
    let object_prompt: CommandEnvelope = serde_json::from_value(object_value.clone()).unwrap();
    assert!(matches!(
        object_prompt.command(),
        AgentCommand::Prompt {
            final_output_format: Some(FinalOutputFormat::JsonObject),
            ..
        }
    ));
}

#[test]
fn every_initial_command_type_has_a_stable_wire_value() {
    let expected = [
        (AgentCommandType::CreateSession, "create_session"),
        (AgentCommandType::Prompt, "prompt"),
        (AgentCommandType::Steer, "steer"),
        (AgentCommandType::FollowUp, "follow_up"),
        (AgentCommandType::Abort, "abort"),
        (AgentCommandType::ResolveApproval, "resolve_approval"),
        (AgentCommandType::SetModel, "set_model"),
        (AgentCommandType::SetReasoningEffort, "set_reasoning_effort"),
        (AgentCommandType::SetProfile, "set_profile"),
        (AgentCommandType::CompactSession, "compact_session"),
        (AgentCommandType::ForkSession, "fork_session"),
    ];
    assert_eq!(AgentCommandType::ALL, expected.map(|(kind, _)| kind));
    for (kind, wire) in expected {
        assert_eq!(serde_json::to_value(kind).unwrap(), wire);
    }
}

#[test]
fn set_reasoning_effort_has_a_typed_wire_payload() {
    let command = CommandEnvelope::new(
        tea_protocol::CommandId::from_str(COMMAND_ID).unwrap(),
        Some(tea_protocol::SessionId::from_str(SESSION_ID).unwrap()),
        tea_protocol::ProtocolTimestamp::from_str(TIMESTAMP).unwrap(),
        AgentCommand::SetReasoningEffort {
            reasoning_effort: ReasoningEffort::ExtraHigh,
        },
    )
    .unwrap();
    let value = serde_json::to_value(&command).unwrap();
    assert_eq!(value["type"], "set_reasoning_effort");
    assert_eq!(value["payload"]["reasoningEffort"], "xhigh");
    assert_eq!(
        serde_json::from_value::<CommandEnvelope>(value)
            .unwrap()
            .command(),
        &AgentCommand::SetReasoningEffort {
            reasoning_effort: ReasoningEffort::ExtraHigh,
        }
    );
}

#[test]
fn unknown_commands_are_rejected_and_classified() {
    let value = json!({
        "protocolVersion":"1.0",
        "type":"launch_missiles",
        "commandId":COMMAND_ID,
        "sessionId":SESSION_ID,
        "timestamp":TIMESTAMP,
        "payload":{}
    });
    let error = CommandEnvelope::decode_value(value).unwrap_err();
    assert!(matches!(error, CommandDecodeError::UnsupportedType { .. }));

    let protocol_error =
        error.into_protocol_error(CorrelationId::from_str(CORRELATION_ID).unwrap());
    assert_eq!(protocol_error.code(), AgentErrorCode::UnsupportedCommand);
    assert_eq!(
        protocol_error.details()["dev.tea-rs.protocol"]["unsupportedType"],
        "launch_missiles"
    );
}

#[test]
fn incompatible_major_takes_precedence_over_unknown_command_type() {
    let value = json!({
        "protocolVersion":"2.0",
        "type":"future_command",
        "commandId":COMMAND_ID,
        "sessionId":SESSION_ID,
        "timestamp":TIMESTAMP,
        "payload":{}
    });
    let error = CommandEnvelope::decode_value(value).unwrap_err();
    assert!(matches!(
        error,
        CommandDecodeError::UnsupportedVersion { .. }
    ));
    assert_eq!(
        error
            .into_protocol_error(CorrelationId::from_str(CORRELATION_ID).unwrap())
            .code(),
        AgentErrorCode::UnsupportedProtocolVersion
    );
}

#[test]
fn command_session_and_payload_invariants_are_enforced() {
    let create_with_session = json!({
        "protocolVersion":"1.0",
        "type":"create_session",
        "commandId":COMMAND_ID,
        "sessionId":SESSION_ID,
        "timestamp":TIMESTAMP,
        "payload":{"profileId":"minimal-assistant"}
    });
    assert!(serde_json::from_value::<CommandEnvelope>(create_with_session).is_err());

    let prompt_without_session = json!({
        "protocolVersion":"1.0",
        "type":"prompt",
        "commandId":COMMAND_ID,
        "timestamp":TIMESTAMP,
        "payload":fixture("command-prompt.json")["payload"].clone()
    });
    assert!(serde_json::from_value::<CommandEnvelope>(prompt_without_session).is_err());

    let assistant_prompt = json!({
        "protocolVersion":"1.0",
        "type":"prompt",
        "commandId":COMMAND_ID,
        "sessionId":SESSION_ID,
        "timestamp":TIMESTAMP,
        "payload":{
            "message":{
                "id":"0195a0b1-5e3d-7bb4-863a-0aa7aa000003",
                "type":"assistant",
                "content":[{"type":"text","text":"not user input"}],
                "stopReason":"completed",
                "timestamp":TIMESTAMP
            }
        }
    });
    assert!(serde_json::from_value::<CommandEnvelope>(assistant_prompt).is_err());

    let mut misspelled_format = fixture("command-prompt.json");
    misspelled_format["payload"]["finalOutputForma"] = json!({"type":"json_object"});
    assert!(serde_json::from_value::<CommandEnvelope>(misspelled_format).is_err());
}

#[test]
fn direct_invalid_command_construction_cannot_cross_the_wire() {
    let assistant: tea_protocol::CanonicalMessage = serde_json::from_value(json!({
        "id":"0195a0b1-5e3d-7bb4-863a-0aa7aa000003",
        "type":"assistant",
        "content":[{"type":"text","text":"not user input"}],
        "stopReason":"completed",
        "timestamp":TIMESTAMP
    }))
    .unwrap();
    let invalid = AgentCommand::Prompt {
        message: assistant,
        final_output_format: None,
    };
    assert!(serde_json::to_value(invalid).is_err());

    let invalid_format = AgentCommand::Prompt {
        message: serde_json::from_value(json!({
            "id":"0195a0b1-5e3d-7bb4-863a-0aa7aa000004",
            "type":"user",
            "content":[{"type":"text","text":"Return JSON."}],
            "timestamp":TIMESTAMP
        }))
        .unwrap(),
        final_output_format: Some(FinalOutputFormat::JsonSchema {
            schema: json!(["not", "a", "schema", "object"]),
        }),
    };
    assert_eq!(
        CommandEnvelope::new(
            COMMAND_ID.parse().unwrap(),
            Some(SESSION_ID.parse().unwrap()),
            TIMESTAMP.parse().unwrap(),
            invalid_format,
        )
        .unwrap_err(),
        tea_protocol::CommandValidationError::InvalidFinalOutputFormat
    );
}

#[test]
fn selectors_and_command_payloads_are_bounded() {
    assert!(ProfileId::from_str("minimal-assistant").is_ok());
    assert!(ProfileId::from_str("").is_err());
    assert!(ProfileId::from_str(&"x".repeat(129)).is_err());
    assert!(ProfileId::from_str("Invalid Profile").is_err());
    assert!(ModelId::from_str("llama3.1:8b").is_ok());
    assert!(ProfileId::from_str("profile:variant").is_err());

    let oversized_profile = json!({
        "protocolVersion":"1.0",
        "type":"set_profile",
        "commandId":COMMAND_ID,
        "sessionId":SESSION_ID,
        "timestamp":TIMESTAMP,
        "payload":{"profileId":"x".repeat(129)}
    });
    assert!(serde_json::from_value::<CommandEnvelope>(oversized_profile).is_err());
}
