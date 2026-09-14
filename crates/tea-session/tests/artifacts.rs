use crate::common;

use common::{APPROVAL_ID, SESSION_ID, TOOL_CALL_ID, approval_flow, envelope};
use serde_json::{Value, json};
use std::str::FromStr;
use tea_policy::{ActorId, ApprovalRequest, ApprovalResolution, PolicyGrant};
use tea_protocol::{FinalOutputFormat, RecordEnvelope, SessionId, SessionSequence};
use tea_session::{
    AppendTransaction, ApprovalArtifactEntry, GrantJournalEntry, InMemorySessionStore,
    SessionStore, SessionStoreErrorCode,
};

const GRANT_ID: &str = "0195a0b1-5e69-70ac-807e-0aa7aa000047";
const CREATED_AT: &str = "2026-07-23T09:30:12.005Z";
const DECIDED_AT: &str = "2026-07-23T09:30:12.006Z";
const EXPIRES_AT: &str = "2026-07-23T09:35:13.010Z";

fn request() -> ApprovalRequest {
    serde_json::from_value(json!({
        "approvalId": APPROVAL_ID,
        "toolCallId": TOOL_CALL_ID,
        "actorId": "user:alice",
        "profileId": "minimal-assistant",
        "sessionId": SESSION_ID,
        "workspaceId": "workspace/main",
        "environment": {"surface":"test","target":"native","metadata":{}},
        "toolName": "write_text_file",
        "toolVersion": "1.0.0",
        "effects": ["fs.write"],
        "resources": [{
            "scheme":"file",
            "locator":"/workspace/notes.txt",
            "access":"write"
        }],
        "createdAt": CREATED_AT,
        "expiresAt": EXPIRES_AT,
        "presentation": {
            "reason":"workspace mutation requires approval",
            "arguments":{"path":"/workspace/notes.txt"},
            "resources":["file:/workspace/notes.txt"]
        }
    }))
    .unwrap()
}

fn final_output_format() -> FinalOutputFormat {
    FinalOutputFormat::JsonSchema {
        schema: json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": false
        }),
    }
}

fn grant() -> PolicyGrant {
    serde_json::from_value(json!({
        "id":GRANT_ID,
        "actorId":"user:alice",
        "profileId":"minimal-assistant",
        "toolName":"write_text_file",
        "toolVersion":"1.0.0",
        "effects":["fs.write"],
        "resources":[{"scheme":"file","locatorPrefix":"/workspace/","access":"write"}],
        "scope":{"type":"session_resource","session_id":SESSION_ID},
        "issuedAt":DECIDED_AT
    }))
    .unwrap()
}

fn resolution() -> ApprovalResolution {
    ApprovalResolution::new(
        &request(),
        tea_protocol::ApprovalDecision::AllowSession,
        DECIDED_AT.parse().unwrap(),
        Some(grant()),
    )
    .unwrap()
}

fn approval_request_record() -> RecordEnvelope {
    let mut value = serde_json::to_value(envelope(
        5,
        "approval_requested",
        json!({"approvalId":APPROVAL_ID,"toolCallId":TOOL_CALL_ID,"expiresAt":EXPIRES_AT}),
    ))
    .unwrap();
    value["timestamp"] = json!(CREATED_AT);
    serde_json::from_value(value).unwrap()
}

fn initial_records() -> Vec<RecordEnvelope> {
    let mut records = approval_flow();
    records[5] = approval_request_record();
    records
}

fn resolution_record() -> RecordEnvelope {
    let mut value = serde_json::to_value(envelope(
        6,
        "approval_resolved",
        json!({"approvalId":APPROVAL_ID,"decision":{"type":"allow_session"}}),
    ))
    .unwrap();
    value["timestamp"] = json!(DECIDED_AT);
    serde_json::from_value(value).unwrap()
}

#[tokio::test(flavor = "current_thread")]
async fn request_resolution_grant_and_revocation_are_atomic() {
    let store = InMemorySessionStore::new();
    let session_id = SessionId::from_str(SESSION_ID).unwrap();
    let request_record = approval_request_record();
    let created = store
        .append(
            AppendTransaction::new(session_id, None, initial_records())
                .with_expected_journal_revision(0)
                .with_approval_artifacts([ApprovalArtifactEntry::Requested {
                    record_id: request_record.record_id(),
                    request: request(),
                    final_output_format: Some(final_output_format()),
                }]),
        )
        .await
        .unwrap();
    assert_eq!(created.journal_revision(), 1);

    let resolved_record = resolution_record();
    let resolution = resolution();
    let issued = store
        .append(
            AppendTransaction::new(
                session_id,
                Some(SessionSequence::new(5)),
                vec![resolved_record.clone()],
            )
            .with_expected_journal_revision(1)
            .with_approval_artifacts([ApprovalArtifactEntry::Resolved {
                record_id: resolved_record.record_id(),
                resolution: resolution.clone(),
            }])
            .with_grant_entries([GrantJournalEntry::Issued {
                approval_record_id: resolved_record.record_id(),
                grant: grant(),
            }]),
        )
        .await
        .unwrap();
    assert_eq!(issued.journal_revision(), 3);
    assert_eq!(
        store.load(session_id).await.unwrap().active_grants().len(),
        1
    );
    assert_eq!(
        store
            .active_grants_for_actor(ActorId::from_str("user:alice").unwrap())
            .await
            .unwrap(),
        vec![grant()]
    );

    let revoked = grant()
        .revoke("2026-07-23T10:00:00.000Z".parse().unwrap())
        .unwrap();
    let revoked_outcome = store
        .append(
            AppendTransaction::new(session_id, Some(SessionSequence::new(6)), vec![])
                .with_expected_journal_revision(3)
                .with_grant_entries([GrantJournalEntry::Revoked { grant: revoked }]),
        )
        .await
        .unwrap();
    assert_eq!(revoked_outcome.current_sequence(), SessionSequence::new(6));
    assert_eq!(revoked_outcome.journal_revision(), 4);
    assert!(
        store
            .load(session_id)
            .await
            .unwrap()
            .active_grants()
            .is_empty()
    );
}

#[tokio::test(flavor = "current_thread")]
async fn mismatched_artifact_and_stale_journal_roll_back() {
    let store = InMemorySessionStore::new();
    let session_id = SessionId::from_str(SESSION_ID).unwrap();
    let request_record = approval_request_record();
    let mut value = serde_json::to_value(request()).unwrap();
    value["toolCallId"] = json!("0195a0b1-5e70-7c8d-9e2f-0aa7aa000048");
    let wrong_request: ApprovalRequest = serde_json::from_value(value).unwrap();
    let error = store
        .append(
            AppendTransaction::new(session_id, None, initial_records())
                .with_expected_journal_revision(0)
                .with_approval_artifacts([ApprovalArtifactEntry::Requested {
                    record_id: request_record.record_id(),
                    request: wrong_request,
                    final_output_format: None,
                }]),
        )
        .await
        .unwrap_err();
    assert_eq!(error.code(), SessionStoreErrorCode::InvalidRecord);
    assert_eq!(
        store.load(session_id).await.unwrap_err().code(),
        SessionStoreErrorCode::SessionNotFound
    );

    store
        .append(
            AppendTransaction::new(session_id, None, initial_records())
                .with_expected_journal_revision(0)
                .with_approval_artifacts([ApprovalArtifactEntry::Requested {
                    record_id: request_record.record_id(),
                    request: request(),
                    final_output_format: None,
                }]),
        )
        .await
        .unwrap();
    let error = store
        .append(
            AppendTransaction::new(session_id, Some(SessionSequence::new(5)), vec![])
                .with_expected_journal_revision(0)
                .with_grant_entries([GrantJournalEntry::Revoked { grant: grant() }]),
        )
        .await
        .unwrap_err();
    assert_eq!(error.code(), SessionStoreErrorCode::SequenceConflict);
    assert_eq!(
        store.load(session_id).await.unwrap().grant_journal().len(),
        0
    );
}

#[test]
fn artifact_values_round_trip() {
    let entries = vec![
        ApprovalArtifactEntry::Requested {
            record_id: approval_request_record().record_id(),
            request: request(),
            final_output_format: Some(final_output_format()),
        },
        ApprovalArtifactEntry::Resolved {
            record_id: resolution_record().record_id(),
            resolution: resolution(),
        },
    ];
    let value: Value = serde_json::to_value(&entries).unwrap();
    let decoded = serde_json::from_value::<Vec<ApprovalArtifactEntry>>(value).unwrap();
    assert!(matches!(
        &decoded[0],
        ApprovalArtifactEntry::Requested {
            final_output_format: Some(format),
            ..
        } if format == &final_output_format()
    ));
    assert_eq!(decoded, entries);
}
