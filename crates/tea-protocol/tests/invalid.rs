use std::str::FromStr;

use serde_json::{Value, json};
use tea_protocol::{
    CommandEnvelope, EventEnvelope, FinalOutputFormat, FinalOutputFormatError,
    ProtocolErrorEnvelope, ProtocolMetadata, ProtocolTimestamp, ProtocolVersion, RecordEnvelope,
    SessionId, SessionSequence,
};

const COMMAND_ID: &str = "0195a0b1-5e3c-70a1-927f-0aa7aa000002";
const SESSION_ID: &str = "0195a0b1-5e3a-7d72-a902-c4e85d828bf1";
const MESSAGE_ID: &str = "0195a0b1-5e3d-7bb4-863a-0aa7aa000003";
const TIMESTAMP: &str = "2026-07-23T09:30:12.124Z";

fn prompt_with_final_output_format(final_output_format: &Value) -> Value {
    json!({
        "protocolVersion":"1.0",
        "type":"prompt",
        "commandId":COMMAND_ID,
        "sessionId":SESSION_ID,
        "timestamp":TIMESTAMP,
        "payload":{
            "message":{
                "id":MESSAGE_ID,
                "type":"user",
                "content":[{"type":"text","text":"Return JSON."}],
                "timestamp":TIMESTAMP
            },
            "finalOutputFormat":final_output_format
        }
    })
}

#[test]
fn invalid_protocol_versions_are_rejected() {
    for value in ["", "1", "1.0.0", "01.0", "1.00", "+1.0", " 1.0"] {
        assert!(
            ProtocolVersion::from_str(value).is_err(),
            "accepted {value:?}"
        );
    }
}

#[test]
fn non_canonical_or_non_v7_ids_are_rejected() {
    for value in [
        "",
        "550e8400-e29b-41d4-a716-446655440000",
        "0195A0B1-5E3A-7D72-A902-C4E85D828BF1",
        "0195a0b15e3a7d72a902c4e85d828bf1",
    ] {
        assert!(SessionId::from_str(value).is_err(), "accepted {value:?}");
    }
}

#[test]
fn invalid_sequences_are_rejected() {
    for value in ["", "00", "01", "+1", "-1", " 1", "1 ", "1.0", "1e3"] {
        assert!(
            SessionSequence::from_str(value).is_err(),
            "accepted {value:?}"
        );
    }
    assert!(SessionSequence::new(u64::MAX).checked_next().is_none());
    assert!(serde_json::from_str::<SessionSequence>("42").is_err());
}

#[test]
fn duplicate_fields_are_rejected_recursively_at_envelope_boundaries() {
    let duplicate_command_id = r#"{
        "protocolVersion":"1.0",
        "type":"abort",
        "commandId":"0195a0b1-5e5e-741f-b474-0aa7aa000036",
        "commandId":"0195a0b1-5e5b-739d-bf5c-0aa7aa000033",
        "sessionId":"0195a0b1-5e3a-7d72-a902-c4e85d828bf1",
        "timestamp":"2026-07-23T09:30:15.200Z",
        "payload":{}
    }"#;
    assert!(serde_json::from_str::<CommandEnvelope>(duplicate_command_id).is_err());

    let duplicate_payload = r#"{
        "protocolVersion":"1.0",
        "type":"run_finished",
        "eventId":"0195a0b1-5e49-7ec5-8d81-0aa7aa000015",
        "sessionId":"0195a0b1-5e3a-7d72-a902-c4e85d828bf1",
        "runId":"0195a0b1-5e40-7136-8ae0-0aa7aa000006",
        "sequence":"6",
        "timestamp":"2026-07-23T09:30:15.000Z",
        "payload":{"status":"completed","status":"failed"}
    }"#;
    assert!(serde_json::from_str::<EventEnvelope>(duplicate_payload).is_err());

    let duplicate_record_type = r#"{
        "protocolVersion":"1.0",
        "type":"run_cancelled",
        "type":"run_interrupted",
        "recordId":"0195a0b1-5e57-78ff-80e1-0aa7aa000029",
        "sessionId":"0195a0b1-5e3a-7d72-a902-c4e85d828bf1",
        "sequence":"7",
        "timestamp":"2026-07-23T09:30:14.130Z",
        "payload":{"runId":"0195a0b1-5e40-7136-8ae0-0aa7aa000006"}
    }"#;
    assert!(serde_json::from_str::<RecordEnvelope>(duplicate_record_type).is_err());

    let duplicate_error_code = r#"{
        "protocolVersion":"1.0",
        "type":"protocol_error",
        "error":{"code":"internal","code":"invalid_input","message":"bad","retry":"never"}
    }"#;
    assert!(serde_json::from_str::<ProtocolErrorEnvelope>(duplicate_error_code).is_err());
}

#[test]
fn duplicate_metadata_namespaces_are_rejected_directly() {
    let duplicate = r#"{"com.example":{"value":1},"com.example":{"value":2}}"#;
    assert!(serde_json::from_str::<ProtocolMetadata>(duplicate).is_err());
}

#[test]
fn incompatible_protocol_majors_are_rejected_by_all_envelopes() {
    let cases = [
        r#"{"protocolVersion":"2.0","type":"abort","commandId":"0195a0b1-5e5e-741f-b474-0aa7aa000036","sessionId":"0195a0b1-5e3a-7d72-a902-c4e85d828bf1","timestamp":"2026-07-23T09:30:15.200Z","payload":{}}"#,
        r#"{"protocolVersion":"2.0","type":"run_started","eventId":"0195a0b1-5e3f-742a-9891-0aa7aa000005","sessionId":"0195a0b1-5e3a-7d72-a902-c4e85d828bf1","runId":"0195a0b1-5e40-7136-8ae0-0aa7aa000006","sequence":"1","timestamp":"2026-07-23T09:30:12.125Z","payload":{}}"#,
        r#"{"protocolVersion":"2.0","type":"run_cancelled","recordId":"0195a0b1-5e57-78ff-80e1-0aa7aa000029","sessionId":"0195a0b1-5e3a-7d72-a902-c4e85d828bf1","sequence":"7","timestamp":"2026-07-23T09:30:14.130Z","payload":{"runId":"0195a0b1-5e40-7136-8ae0-0aa7aa000006"}}"#,
    ];
    assert!(serde_json::from_str::<CommandEnvelope>(cases[0]).is_err());
    assert!(serde_json::from_str::<EventEnvelope>(cases[1]).is_err());
    assert!(serde_json::from_str::<RecordEnvelope>(cases[2]).is_err());
}

#[test]
fn invalid_or_lossy_timestamps_are_rejected() {
    for value in [
        "",
        "2026-07-23 09:30:12.123Z",
        "2026-07-23T09:30:12.123",
        "2026-07-23T09:30:12.123456Z",
    ] {
        assert!(
            ProtocolTimestamp::from_str(value).is_err(),
            "accepted {value:?}"
        );
    }
}

#[test]
fn final_output_format_rejects_unknown_or_incomplete_wire_shapes() {
    for format in [
        json!({"type":"json_object","unexpected":true}),
        json!({"type":"json_schema"}),
        json!({"type":"future_format"}),
    ] {
        assert!(
            serde_json::from_value::<CommandEnvelope>(prompt_with_final_output_format(&format))
                .is_err()
        );
    }
}

#[test]
fn final_output_schema_requires_a_bounded_object() {
    let non_object = json!({"type":"json_schema","schema":["not", "an", "object"]});
    let invalid_schema = json!({"type":"json_schema","schema":{"type":7}});
    let oversized = json!({
        "type":"json_schema",
        "schema":{"type":"object","description":"x".repeat(256 * 1024)}
    });
    let mut deeply_nested_schema = json!({"type":"object"});
    for _ in 0..40 {
        deeply_nested_schema = json!({
            "type":"object",
            "properties":{"next":deeply_nested_schema}
        });
    }
    let deeply_nested = json!({"type":"json_schema","schema":deeply_nested_schema});
    for format in [non_object, invalid_schema, oversized, deeply_nested] {
        assert!(
            serde_json::from_value::<CommandEnvelope>(prompt_with_final_output_format(&format))
                .is_err()
        );
    }
}

#[test]
fn deeply_nested_direct_schema_fails_before_recursive_serialization() {
    let mut nested = Value::Null;
    for _ in 0..10_000 {
        nested = Value::Array(vec![nested]);
    }
    let mut schema = serde_json::Map::new();
    schema.insert("allOf".to_owned(), Value::Array(vec![nested]));
    let format = FinalOutputFormat::JsonSchema {
        schema: Value::Object(schema),
    };

    assert_eq!(
        format.validate(),
        Err(FinalOutputFormatError::SchemaOutOfBounds)
    );
    // Dropping an adversarially deep serde_json::Value is itself recursive.
    std::mem::forget(format);
}

#[test]
fn final_output_schema_rejects_external_references() {
    for reference in [
        "https://schemas.example.test/final-output.json",
        "http://schemas.example.test/final-output.json",
        "file:///tmp/final-output.json",
    ] {
        let format = json!({
            "type":"json_schema",
            "schema":{"$ref":reference}
        });

        assert!(
            serde_json::from_value::<CommandEnvelope>(prompt_with_final_output_format(&format))
                .is_err(),
            "accepted external schema reference {reference:?}"
        );
    }
}
