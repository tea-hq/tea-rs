use crate::common;

use std::fs;
use std::path::PathBuf;
use std::str::FromStr;

use common::{APPROVAL_ID, SESSION_ID, TOOL_CALL_ID, branched_archive_flow};
use serde_json::{Value, json};
use tea_policy::{ApprovalRequest, ApprovalResolution, PolicyGrant};
use tea_protocol::{ApprovalDecision, SessionId};
use tea_session::{
    ApprovalArtifactEntry, GrantJournalEntry, InMemorySessionStore, SessionArchive,
    SessionArchiveError, SessionStore, SessionStoreErrorCode,
};
use tea_tools::{TOOL_AUDIT_METADATA_NAMESPACE, ToolSource};

const GRANT_ID: &str = "0195a0b1-5e69-70ac-807e-0aa7aa000047";
const CREATED_AT: &str = "2026-07-23T09:30:12.005Z";
const DECIDED_AT: &str = "2026-07-23T09:30:12.006Z";
const EXPIRES_AT: &str = "2026-07-23T09:35:13.010Z";

fn request() -> ApprovalRequest {
    serde_json::from_value(json!({
        "approvalId":APPROVAL_ID,
        "toolCallId":TOOL_CALL_ID,
        "actorId":"user:alice",
        "profileId":"minimal-assistant",
        "sessionId":SESSION_ID,
        "workspaceId":"workspace/main",
        "environment":{"surface":"test","target":"native","metadata":{}},
        "toolName":"write_text_file",
        "toolVersion":"1.0.0",
        "toolSource":serde_json::to_value(ToolSource::native_product()).unwrap(),
        "effects":["fs.write"],
        "resources":[{"scheme":"file","locator":"/workspace/notes.txt","access":"write"}],
        "createdAt":CREATED_AT,
        "expiresAt":EXPIRES_AT,
        "presentation":{
            "reason":"workspace mutation requires approval",
            "arguments":{"path":"/workspace/notes.txt"},
            "resources":["file:/workspace/notes.txt"]
        }
    }))
    .unwrap()
}

fn grant() -> PolicyGrant {
    serde_json::from_value(json!({
        "id":GRANT_ID,
        "actorId":"user:alice",
        "profileId":"minimal-assistant",
        "toolName":"write_text_file",
        "toolVersion":"1.0.0",
        "toolSource":serde_json::to_value(ToolSource::native_product()).unwrap(),
        "effects":["fs.write"],
        "resources":[{"scheme":"file","locatorPrefix":"/workspace/","access":"write"}],
        "scope":{"type":"session_resource","session_id":SESSION_ID},
        "issuedAt":DECIDED_AT
    }))
    .unwrap()
}

fn archive() -> SessionArchive {
    let mut records = branched_archive_flow();
    for record in &mut records {
        let mut value = serde_json::to_value(&*record).unwrap();
        match value["type"].as_str().unwrap() {
            "tool_call_requested" => {
                value["metadata"] = serde_json::to_value(
                    tea_protocol::ProtocolMetadata::from_entries([(
                        TOOL_AUDIT_METADATA_NAMESPACE,
                        json!({
                            "toolVersion":"1.0.0",
                            "source":serde_json::to_value(ToolSource::native_product()).unwrap(),
                            "effects":["fs.write"],
                            "resources":[{
                                "scheme":"file",
                                "redactedPresentation":"file:/workspace/notes.txt",
                                "access":"write"
                            }]
                        }),
                    )])
                    .unwrap(),
                )
                .unwrap();
            }
            "approval_requested" => value["timestamp"] = json!(CREATED_AT),
            "approval_resolved" => {
                value["timestamp"] = json!(DECIDED_AT);
                value["payload"]["decision"] = json!({"type":"allow_session"});
            }
            _ => {}
        }
        *record = serde_json::from_value(value).unwrap();
    }
    let request_record_id = records
        .iter()
        .find(|record| record.record_type() == tea_protocol::SessionRecordType::ApprovalRequested)
        .unwrap()
        .record_id();
    let resolution_record_id = records
        .iter()
        .find(|record| record.record_type() == tea_protocol::SessionRecordType::ApprovalResolved)
        .unwrap()
        .record_id();
    let request = request();
    let resolution = ApprovalResolution::new(
        &request,
        ApprovalDecision::AllowSession,
        DECIDED_AT.parse().unwrap(),
        Some(grant()),
    )
    .unwrap();
    SessionArchive::new(
        SessionId::from_str(SESSION_ID).unwrap(),
        records,
        vec![
            ApprovalArtifactEntry::Requested {
                record_id: request_record_id,
                request,
                final_output_format: None,
            },
            ApprovalArtifactEntry::Resolved {
                record_id: resolution_record_id,
                resolution,
            },
        ],
        vec![GrantJournalEntry::Issued {
            approval_record_id: resolution_record_id,
            grant: grant(),
        }],
    )
    .unwrap()
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/v1/session-archive.json")
}

#[test]
fn archive_golden_fixture_is_deterministic_and_round_trips() {
    let archive = archive();
    let encoded = serde_json::to_string_pretty(&archive).unwrap() + "\n";
    if std::env::var_os("UPDATE_SESSION_FIXTURES").is_some() {
        fs::write(fixture_path(), &encoded).unwrap();
    }
    let fixture = fs::read_to_string(fixture_path()).unwrap();
    assert_eq!(encoded, fixture);
    assert_eq!(SessionArchive::decode_json(&fixture).unwrap(), archive);
    assert_eq!(
        serde_json::to_string_pretty(&archive).unwrap() + "\n",
        fixture
    );
}

#[tokio::test(flavor = "current_thread")]
async fn import_validates_then_commits_once_and_export_matches() {
    let store = InMemorySessionStore::new();
    let archive = archive();
    let session_id = archive.session_id();
    archive.clone().import_into(&store).await.unwrap();
    let snapshot = store.load(session_id).await.unwrap();
    assert_eq!(SessionArchive::from_snapshot(&snapshot).unwrap(), archive);
    assert_eq!(snapshot.active_grants().len(), 1);
    assert_eq!(
        snapshot.active_grants()[0].tool_source(),
        Some(&ToolSource::native_product())
    );
    let requested = snapshot
        .records()
        .iter()
        .find(|record| record.record_type() == tea_protocol::SessionRecordType::ToolCallRequested)
        .unwrap();
    assert_eq!(
        requested.metadata().get(TOOL_AUDIT_METADATA_NAMESPACE),
        archive
            .records()
            .iter()
            .find(|record| {
                record.record_type() == tea_protocol::SessionRecordType::ToolCallRequested
            })
            .unwrap()
            .metadata()
            .get(TOOL_AUDIT_METADATA_NAMESPACE)
    );

    let duplicate = archive.import_into(&store).await.unwrap_err();
    assert_eq!(
        duplicate.code(),
        SessionStoreErrorCode::SessionAlreadyExists
    );
}

#[test]
fn malformed_unknown_and_wrong_version_archives_fail_closed() {
    let encoded = serde_json::to_string(&archive()).unwrap();
    let duplicate = encoded.replacen(
        "{\"formatVersion\":1,",
        "{\"formatVersion\":1,\"formatVersion\":1,",
        1,
    );
    assert!(matches!(
        SessionArchive::decode_json(&duplicate),
        Err(SessionArchiveError::Malformed(_))
    ));

    let mut wrong_version: Value = serde_json::from_str(&encoded).unwrap();
    wrong_version["formatVersion"] = json!(2);
    assert!(matches!(
        SessionArchive::decode_json(&wrong_version.to_string()),
        Err(SessionArchiveError::UnsupportedFormatVersion(2))
    ));

    let mut unknown: Value = serde_json::from_str(&encoded).unwrap();
    unknown["records"][1]["type"] = json!("future_required_state");
    assert!(matches!(
        SessionArchive::decode_json(&unknown.to_string()),
        Err(SessionArchiveError::Record(
            tea_protocol::RecordDecodeError::UnsupportedType { .. }
        ))
    ));

    let mut missing_format: Value = serde_json::from_str(&encoded).unwrap();
    missing_format["approvalArtifacts"][0]
        .as_object_mut()
        .unwrap()
        .remove("final_output_format");
    assert!(matches!(
        SessionArchive::decode_json(&missing_format.to_string()),
        Err(SessionArchiveError::Malformed(_))
    ));

    let mut misspelled_format: Value = serde_json::from_str(&encoded).unwrap();
    let requested = misspelled_format["approvalArtifacts"][0]
        .as_object_mut()
        .unwrap();
    let format = requested.remove("final_output_format").unwrap();
    requested.insert("final_output_formatt".to_owned(), format);
    assert!(matches!(
        SessionArchive::decode_json(&misspelled_format.to_string()),
        Err(SessionArchiveError::Malformed(_))
    ));
}

#[tokio::test(flavor = "current_thread")]
async fn invalid_import_leaves_destination_empty() {
    let mut value = serde_json::to_value(archive()).unwrap();
    value["records"][1]["sequence"] = json!("9");
    let error = SessionArchive::decode_json(&value.to_string()).unwrap_err();
    assert!(matches!(error, SessionArchiveError::Replay(_)));

    let store = InMemorySessionStore::new();
    let session_id = SessionId::from_str(SESSION_ID).unwrap();
    assert_eq!(
        store.load(session_id).await.unwrap_err().code(),
        SessionStoreErrorCode::SessionNotFound
    );
}
