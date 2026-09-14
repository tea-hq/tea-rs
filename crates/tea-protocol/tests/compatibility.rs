use std::any::TypeId;
use std::fs;
use std::path::PathBuf;
use std::str::FromStr;

use serde_json::{Value, json};
use tea_protocol::{
    ApprovalId, CURRENT_PROTOCOL_VERSION, CommandEnvelope, CommandId, CorrelationId, EventEnvelope,
    EventId, MessageId, PROTOCOL_V1_0, ProtocolErrorEnvelope, ProtocolTimestamp, ProtocolVersion,
    RecordEnvelope, RecordId, RunId, SessionId, SessionSequence, ToolCallId, TurnId,
};

const SESSION_ID: &str = "0195a0b1-5e3a-7d72-a902-c4e85d828bf1";
const SECOND_ID: &str = "0195a0b1-5e3b-7ef0-8ec1-0aa7aa000001";

fn fixture(name: &str) -> Value {
    fixture_for("v1.0", name)
}

fn fixture_for(version: &str, name: &str) -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(version)
        .join(name);
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn protocol_version_is_a_canonical_string() {
    assert_eq!(PROTOCOL_V1_0.to_string(), "1.0");
    assert_eq!(CURRENT_PROTOCOL_VERSION, PROTOCOL_V1_0);
    assert_eq!(ProtocolVersion::from_str("1.0").unwrap(), PROTOCOL_V1_0);
    assert_eq!(serde_json::to_string(&PROTOCOL_V1_0).unwrap(), r#""1.0""#);
    assert_eq!(
        serde_json::from_str::<ProtocolVersion>(r#""1.0""#).unwrap(),
        PROTOCOL_V1_0
    );
}

#[test]
fn strong_ids_are_distinct_and_use_canonical_uuid_v7_strings() {
    assert_ne!(TypeId::of::<SessionId>(), TypeId::of::<RunId>());
    assert_ne!(TypeId::of::<MessageId>(), TypeId::of::<ToolCallId>());

    macro_rules! check_id {
        ($id:ty, $value:expr) => {{
            let parsed = <$id>::from_str($value).unwrap();
            assert_eq!(parsed.to_string(), $value);
            assert_eq!(
                serde_json::to_string(&parsed).unwrap(),
                format!(r#""{}""#, $value)
            );
            assert_eq!(
                serde_json::from_str::<$id>(&format!(r#""{}""#, $value)).unwrap(),
                parsed
            );
        }};
    }

    check_id!(SessionId, SESSION_ID);
    check_id!(RunId, SECOND_ID);
    check_id!(TurnId, SESSION_ID);
    check_id!(MessageId, SECOND_ID);
    check_id!(ToolCallId, SESSION_ID);
    check_id!(ApprovalId, SECOND_ID);
    check_id!(EventId, SESSION_ID);
    check_id!(CommandId, SECOND_ID);
    check_id!(RecordId, SESSION_ID);
}

#[test]
fn session_sequence_serializes_as_a_decimal_string() {
    let sequence = SessionSequence::new(42);
    assert_eq!(sequence.get(), 42);
    assert_eq!(sequence.checked_next().unwrap().get(), 43);
    assert_eq!(sequence.to_string(), "42");
    assert_eq!(serde_json::to_string(&sequence).unwrap(), r#""42""#);
    assert_eq!(
        serde_json::from_str::<SessionSequence>(r#""42""#).unwrap(),
        sequence
    );
}

#[test]
fn command_fields_are_strict_while_observation_fields_remain_additive() {
    let mut command = fixture("command-prompt.json");
    command["futureEnvelopeField"] = json!(true);
    command["payload"]["futurePayloadField"] = json!({"value": 1});
    assert!(serde_json::from_value::<CommandEnvelope>(command).is_err());

    let mut event = fixture("event-run-finished.json");
    event["futureEnvelopeField"] = json!(true);
    event["payload"]["futurePayloadField"] = json!([1, 2]);
    assert!(serde_json::from_value::<EventEnvelope>(event).is_ok());

    let mut record = fixture("record-message-committed.json");
    record["futureEnvelopeField"] = json!(true);
    record["payload"]["futurePayloadField"] = json!("future");
    assert!(serde_json::from_value::<RecordEnvelope>(record).is_ok());
}

#[test]
fn same_major_minor_fixtures_decode_known_envelopes() {
    assert!(
        serde_json::from_value::<CommandEnvelope>(fixture_for("v1.1", "command-prompt.json"))
            .is_err()
    );
    let event: EventEnvelope =
        serde_json::from_value(fixture_for("v1.1", "event-run-finished.json")).unwrap();
    let record: RecordEnvelope =
        serde_json::from_value(fixture_for("v1.1", "record-message-committed.json")).unwrap();
    let error: ProtocolErrorEnvelope =
        serde_json::from_value(fixture_for("v1.1", "error-unsupported-command.json")).unwrap();

    for version in [
        event.protocol_version(),
        record.protocol_version(),
        error.protocol_version(),
    ] {
        assert_eq!(version, ProtocolVersion::new(1, 1));
    }
}

#[test]
fn namespaced_metadata_survives_known_envelope_round_trip() {
    let mut event = fixture("event-run-finished.json");
    event["metadata"] = json!({"com.example": {"trace": "safe"}});
    let typed: EventEnvelope = serde_json::from_value(event.clone()).unwrap();
    assert_eq!(serde_json::to_value(typed).unwrap(), event);
}

#[test]
fn correlation_id_is_a_distinct_canonical_id() {
    assert_ne!(TypeId::of::<CorrelationId>(), TypeId::of::<CommandId>());
    let parsed = CorrelationId::from_str(SECOND_ID).unwrap();
    assert_eq!(parsed.to_string(), SECOND_ID);
}

#[test]
fn timestamp_normalizes_to_utc_millisecond_rfc3339() {
    let canonical = ProtocolTimestamp::from_str("2026-07-23T09:30:12.123Z").unwrap();
    assert_eq!(canonical.to_string(), "2026-07-23T09:30:12.123Z");
    assert_eq!(
        serde_json::to_string(&canonical).unwrap(),
        r#""2026-07-23T09:30:12.123Z""#
    );

    let offset = ProtocolTimestamp::from_str("2026-07-23T10:30:12.123+01:00").unwrap();
    assert_eq!(offset, canonical);
    assert_eq!(offset.to_string(), "2026-07-23T09:30:12.123Z");
}
