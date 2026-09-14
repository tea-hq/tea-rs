use std::path::{Path, PathBuf};
use std::str::FromStr as _;
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{Value, json};
use tea_policy::{ActorId, ApprovalRequest, ApprovalResolution, PolicyGrant};
use tea_protocol::{
    ApprovalDecision, FinalOutputFormat, RecordEnvelope, SessionId, SessionSequence,
};
use tea_session::{AppendTransaction, ApprovalArtifactEntry, GrantJournalEntry, SessionStore};
use tea_session_sqlite::{SqliteSessionError, SqliteSessionStore};

const PRE_RELEASE_BASE_SCHEMA: &str = "CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
     CREATE TABLE records (
         session_id TEXT NOT NULL,
         sequence INTEGER NOT NULL,
         record_id TEXT NOT NULL,
         envelope TEXT NOT NULL,
         PRIMARY KEY (session_id, sequence)
     );
     CREATE UNIQUE INDEX idx_records_record_id ON records (session_id, record_id);
     CREATE TABLE approval_artifacts (
         session_id TEXT NOT NULL,
         record_id TEXT NOT NULL,
         envelope TEXT NOT NULL,
         PRIMARY KEY (session_id, record_id)
     );
     CREATE TABLE grant_journal (
         session_id TEXT NOT NULL,
         seq INTEGER NOT NULL,
         grant_id TEXT NOT NULL,
         envelope TEXT NOT NULL,
         PRIMARY KEY (session_id, seq)
     );
     CREATE INDEX idx_grant_journal_grant_id ON grant_journal (grant_id);";

static NEXT_DATABASE_ID: AtomicU64 = AtomicU64::new(0);

const SESSION_ID: &str = "0195a0b1-5e3a-7d72-a902-c4e85d828bf1";
const TOOL_CALL_ID: &str = "0195a0b1-5e45-75be-8284-0aa7aa000011";
const APPROVAL_ID: &str = "0195a0b1-5e46-7e2a-b230-0aa7aa000012";
const GRANT_ID: &str = "0195a0b1-5e69-70ac-807e-0aa7aa000047";
const CREATED_AT: &str = "2026-07-23T09:30:12.005Z";
const DECIDED_AT: &str = "2026-07-23T09:30:12.006Z";
const EXPIRES_AT: &str = "2026-07-23T09:35:13.010Z";
const RECORD_IDS: [&str; 7] = [
    "0195a0b1-5e50-7af4-8972-0aa7aa000022",
    "0195a0b1-5e4a-742a-b57f-0aa7aa000016",
    "0195a0b1-5e63-7b8c-a5ad-0aa7aa000041",
    "0195a0b1-5e52-713b-9bfa-0aa7aa000024",
    "0195a0b1-5e53-7771-82ab-0aa7aa000025",
    "0195a0b1-5e4b-712a-9682-0aa7aa000017",
    "0195a0b1-5e54-7c92-b8ca-0aa7aa000026",
];
const RECORD_TIMESTAMPS: [&str; 7] = [
    "2026-07-23T09:30:12.000Z",
    "2026-07-23T09:30:12.001Z",
    "2026-07-23T09:30:12.002Z",
    "2026-07-23T09:30:12.003Z",
    "2026-07-23T09:30:12.004Z",
    CREATED_AT,
    DECIDED_AT,
];

#[test]
fn fresh_database_installs_and_reopens_complete_schema_v1() {
    let database = TestDatabase::new("fresh");
    let store = SqliteSessionStore::open(database.path()).unwrap();
    assert_eq!(store.schema_version(), 1);
    drop(store);

    let connection = rusqlite::Connection::open(database.path()).unwrap();
    assert_eq!(
        object_names(&connection, "table"),
        vec![
            "active_grants",
            "approval_artifacts",
            "grant_journal",
            "records",
            "schema_version",
            "session_catalog",
        ]
    );
    assert_eq!(
        object_names(&connection, "index"),
        vec![
            "idx_active_grants_actor_revoked",
            "idx_grant_journal_grant_id",
            "idx_records_record_id",
        ]
    );
    assert_eq!(
        version_rows(&connection),
        vec![("1".to_owned(), "integer".to_owned())]
    );
    drop(connection);

    let reopened = SqliteSessionStore::open(database.path()).unwrap();
    assert_eq!(reopened.schema_version(), 1);
}

#[test]
fn complete_schema_v1_reopens_without_rebuilding_active_grants() {
    let database = TestDatabase::new("no-startup-backfill");
    drop(SqliteSessionStore::open(database.path()).unwrap());
    let connection = rusqlite::Connection::open(database.path()).unwrap();
    connection
        .execute(
            "INSERT INTO active_grants
             (grant_id, session_id, actor_id, grant_json, revoked)
             VALUES ('sentinel-grant', 'sentinel-session', 'sentinel-actor', '{}', 0)",
            [],
        )
        .unwrap();
    drop(connection);

    drop(SqliteSessionStore::open(database.path()).unwrap());
    let connection = rusqlite::Connection::open(database.path()).unwrap();
    let rows: u32 = connection
        .query_row(
            "SELECT COUNT(*) FROM active_grants WHERE grant_id = 'sentinel-grant'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(rows, 1);
}

#[tokio::test]
async fn active_grant_write_through_survives_reopen_and_revoke() {
    let database = TestDatabase::new("active-grant-write-through");
    let session_id: SessionId = SESSION_ID.parse().unwrap();
    let issued_grant = grant();
    let final_output_format = FinalOutputFormat::JsonSchema {
        schema: json!({
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
            "additionalProperties": false
        }),
    };
    {
        let store = SqliteSessionStore::open(database.path()).unwrap();
        let records = approval_records();
        let requested = records[5].clone();
        let created = store
            .append(
                AppendTransaction::new(session_id, None, records)
                    .with_expected_journal_revision(0)
                    .with_approval_artifacts([ApprovalArtifactEntry::Requested {
                        record_id: requested.record_id(),
                        request: request(),
                        final_output_format: Some(final_output_format.clone()),
                    }]),
            )
            .await
            .unwrap();
        assert_eq!(created.journal_revision(), 1);

        let resolved = approval_resolution_record();
        let issued = store
            .append(
                AppendTransaction::new(
                    session_id,
                    Some(SessionSequence::new(5)),
                    vec![resolved.clone()],
                )
                .with_expected_journal_revision(1)
                .with_approval_artifacts([ApprovalArtifactEntry::Resolved {
                    record_id: resolved.record_id(),
                    resolution: resolution(),
                }])
                .with_grant_entries([GrantJournalEntry::Issued {
                    approval_record_id: resolved.record_id(),
                    grant: issued_grant.clone(),
                }]),
            )
            .await
            .unwrap();
        assert_eq!(issued.journal_revision(), 3);
    }

    {
        let reopened = SqliteSessionStore::open(database.path()).unwrap();
        let snapshot = reopened.load(session_id).await.unwrap();
        assert!(matches!(
            &snapshot.approval_artifacts()[0],
            ApprovalArtifactEntry::Requested {
                final_output_format: Some(format),
                ..
            } if format == &final_output_format
        ));
        assert_eq!(
            reopened
                .active_grants_for_actor(ActorId::from_str("user:alice").unwrap())
                .await
                .unwrap(),
            vec![issued_grant.clone()]
        );
        let revoked = issued_grant
            .revoke("2026-07-23T10:00:00.000Z".parse().unwrap())
            .unwrap();
        let outcome = reopened
            .append(
                AppendTransaction::new(session_id, Some(SessionSequence::new(6)), vec![])
                    .with_expected_journal_revision(3)
                    .with_grant_entries([GrantJournalEntry::Revoked { grant: revoked }]),
            )
            .await
            .unwrap();
        assert_eq!(outcome.journal_revision(), 4);
    }

    let reopened = SqliteSessionStore::open(database.path()).unwrap();
    assert!(
        reopened
            .active_grants_for_actor(ActorId::from_str("user:alice").unwrap())
            .await
            .unwrap()
            .is_empty()
    );
}

#[test]
fn pre_release_v1_v2_and_v3_are_rejected_without_mutation() {
    for version in [1, 2, 3] {
        let database = TestDatabase::new(&format!("pre-release-v{version}"));
        create_pre_release_database(&database, version);
        let before = schema_snapshot(&database);
        let before_record = marker_record(&database);
        assert_eq!(before.journal_mode, "delete");

        let expected = if version == 1 {
            "required table `active_grants` is missing"
        } else {
            "unsupported SQLite schema version"
        };
        assert_schema_error(&database, expected);

        assert_eq!(schema_snapshot(&database), before);
        assert_eq!(marker_record(&database), before_record);
        let connection = rusqlite::Connection::open(database.path()).unwrap();
        assert_eq!(
            version_rows(&connection),
            vec![(version.to_string(), "integer".to_owned())]
        );
        drop(connection);
        assert!(!database.sidecar_exists("-wal"));
        assert!(!database.sidecar_exists("-shm"));
        assert!(!database.sidecar_exists("-journal"));
    }
}

#[test]
fn nonempty_database_without_version_table_is_rejected_without_mutation() {
    let database = TestDatabase::new("missing-version-table");
    let connection = rusqlite::Connection::open(database.path()).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE sqliteXowner (key TEXT PRIMARY KEY, value TEXT NOT NULL);
             INSERT INTO sqliteXowner (key, value) VALUES ('sentinel', 'unchanged');",
        )
        .unwrap();
    drop(connection);
    let before = schema_snapshot(&database);

    assert_schema_error(
        &database,
        "schema_version is missing from a non-empty database",
    );

    assert_eq!(schema_snapshot(&database), before);
    let connection = rusqlite::Connection::open(database.path()).unwrap();
    let value: String = connection
        .query_row(
            "SELECT value FROM sqliteXowner WHERE key = 'sentinel'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(value, "unchanged");
}

#[test]
fn unsupported_future_version_is_rejected_without_mutation() {
    let database = TestDatabase::new("future-version");
    drop(SqliteSessionStore::open(database.path()).unwrap());
    let connection = rusqlite::Connection::open(database.path()).unwrap();
    connection
        .execute("UPDATE schema_version SET version = 99", [])
        .unwrap();
    drop(connection);
    let before = schema_snapshot(&database);

    assert_schema_error(
        &database,
        "unsupported SQLite schema version 99; expected 1",
    );

    assert_eq!(schema_snapshot(&database), before);
    let connection = rusqlite::Connection::open(database.path()).unwrap();
    assert_eq!(
        version_rows(&connection),
        vec![("99".to_owned(), "integer".to_owned())]
    );
}

#[test]
fn schema_version_requires_exactly_one_integer_row() {
    for (case, mutation) in [
        ("empty-version", "DELETE FROM schema_version"),
        (
            "multiple-versions",
            "INSERT INTO schema_version (version) VALUES (2)",
        ),
    ] {
        let database = TestDatabase::new(case);
        drop(SqliteSessionStore::open(database.path()).unwrap());
        let connection = rusqlite::Connection::open(database.path()).unwrap();
        connection.execute(mutation, []).unwrap();
        let before_rows = version_rows(&connection);
        drop(connection);
        let before = schema_snapshot(&database);

        assert_schema_error(
            &database,
            "schema_version must contain exactly one integer row",
        );

        assert_eq!(schema_snapshot(&database), before);
        let connection = rusqlite::Connection::open(database.path()).unwrap();
        assert_eq!(version_rows(&connection), before_rows);
    }
}

#[test]
fn malformed_same_version_layouts_are_rejected_without_repair() {
    for (case, mutation, expected) in [
        (
            "missing-index",
            "DROP INDEX idx_active_grants_actor_revoked",
            "required index `idx_active_grants_actor_revoked` is missing",
        ),
        (
            "wrong-index",
            "DROP INDEX idx_active_grants_actor_revoked;
             CREATE INDEX idx_active_grants_actor_revoked
                 ON active_grants (revoked, actor_id, grant_id);",
            "index `idx_active_grants_actor_revoked` has an unexpected layout",
        ),
        (
            "extra-column",
            "ALTER TABLE session_catalog ADD COLUMN owner_data TEXT",
            "table `session_catalog` has an unexpected layout",
        ),
    ] {
        let database = TestDatabase::new(case);
        drop(SqliteSessionStore::open(database.path()).unwrap());
        let connection = rusqlite::Connection::open(database.path()).unwrap();
        connection.execute_batch(mutation).unwrap();
        drop(connection);
        let before = schema_snapshot(&database);

        assert_schema_error(&database, expected);

        assert_eq!(schema_snapshot(&database), before);
    }
}

fn approval_records() -> Vec<RecordEnvelope> {
    vec![
        record(
            0,
            "session_created",
            &json!({"profileId":"minimal-assistant"}),
        ),
        record(
            1,
            "message_committed",
            &json!({
                "message": {
                    "id":"0195a0b1-5e3d-7bb4-863a-0aa7aa000003",
                    "type":"user",
                    "content":[{"type":"text","text":"Inspect the workspace."}],
                    "timestamp":RECORD_TIMESTAMPS[1]
                }
            }),
        ),
        record(
            2,
            "message_committed",
            &json!({
                "message": {
                    "id":"0195a0b1-5e64-76d6-9a5a-0aa7aa000042",
                    "type":"assistant",
                    "content":[{
                        "type":"tool_call",
                        "toolCallId":TOOL_CALL_ID,
                        "toolName":"write_text_file",
                        "arguments":{"path":"/workspace/notes.txt","content":"done"}
                    }],
                    "stopReason":"tool_use",
                    "timestamp":RECORD_TIMESTAMPS[2]
                }
            }),
        ),
        record(
            3,
            "tool_call_requested",
            &json!({
                "toolCallId":TOOL_CALL_ID,
                "toolName":"write_text_file",
                "arguments":{"path":"/workspace/notes.txt","content":"done"}
            }),
        ),
        record(
            4,
            "policy_decision_recorded",
            &json!({"toolCallId":TOOL_CALL_ID,"decision":"require_approval"}),
        ),
        record(
            5,
            "approval_requested",
            &json!({
                "approvalId":APPROVAL_ID,
                "toolCallId":TOOL_CALL_ID,
                "expiresAt":EXPIRES_AT
            }),
        ),
    ]
}

fn approval_resolution_record() -> RecordEnvelope {
    record(
        6,
        "approval_resolved",
        &json!({"approvalId":APPROVAL_ID,"decision":{"type":"allow_session"}}),
    )
}

fn record(sequence: usize, record_type: &str, payload: &Value) -> RecordEnvelope {
    serde_json::from_value(json!({
        "protocolVersion":"1.0",
        "type":record_type,
        "recordId":RECORD_IDS[sequence],
        "sessionId":SESSION_ID,
        "sequence":sequence.to_string(),
        "timestamp":RECORD_TIMESTAMPS[sequence],
        "payload":payload
    }))
    .unwrap()
}

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
        "effects":["fs.write"],
        "resources":[{
            "scheme":"file",
            "locator":"/workspace/notes.txt",
            "access":"write"
        }],
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
        "effects":["fs.write"],
        "resources":[{
            "scheme":"file",
            "locatorPrefix":"/workspace/",
            "access":"write"
        }],
        "scope":{"type":"session_resource","session_id":SESSION_ID},
        "issuedAt":DECIDED_AT
    }))
    .unwrap()
}

fn resolution() -> ApprovalResolution {
    ApprovalResolution::new(
        &request(),
        ApprovalDecision::AllowSession,
        DECIDED_AT.parse().unwrap(),
        Some(grant()),
    )
    .unwrap()
}

fn create_pre_release_database(database: &TestDatabase, version: u32) {
    let connection = rusqlite::Connection::open(database.path()).unwrap();
    connection.execute_batch(PRE_RELEASE_BASE_SCHEMA).unwrap();
    if version >= 2 {
        connection
            .execute_batch(
                "CREATE TABLE session_catalog (
                     session_id TEXT PRIMARY KEY,
                     display_name TEXT
                 );",
            )
            .unwrap();
    }
    if version >= 3 {
        connection
            .execute_batch(
                "CREATE TABLE active_grants (
                     grant_id TEXT PRIMARY KEY,
                     session_id TEXT NOT NULL,
                     actor_id TEXT NOT NULL,
                     grant_json TEXT NOT NULL,
                     revoked INTEGER NOT NULL CHECK (revoked IN (0, 1))
                 );
                 CREATE INDEX idx_active_grants_actor_revoked
                     ON active_grants (actor_id, revoked, grant_id);",
            )
            .unwrap();
    }
    connection
        .execute("INSERT INTO schema_version (version) VALUES (?)", [version])
        .unwrap();
    connection
        .execute(
            "INSERT INTO records (session_id, sequence, record_id, envelope)
             VALUES ('legacy-session', 0, 'legacy-record', 'unchanged')",
            [],
        )
        .unwrap();
}

fn assert_schema_error(database: &TestDatabase, expected: &str) {
    let error = match SqliteSessionStore::open(database.path()) {
        Ok(store) => {
            drop(store);
            panic!("incompatible database was accepted")
        }
        Err(error) => error,
    };
    let SqliteSessionError::Schema(message) = error else {
        panic!("unexpected error: {error}");
    };
    assert!(
        message.contains(expected),
        "expected `{expected}` in schema error, got `{message}`"
    );
}

fn marker_record(database: &TestDatabase) -> (i64, String) {
    let connection = rusqlite::Connection::open(database.path()).unwrap();
    connection
        .query_row("SELECT rowid, envelope FROM records", [], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })
        .unwrap()
}

fn object_names(connection: &rusqlite::Connection, object_type: &str) -> Vec<String> {
    let mut statement = connection
        .prepare(
            "SELECT name FROM main.sqlite_schema
             WHERE type = ?1 AND name NOT GLOB 'sqlite_*' ORDER BY name",
        )
        .unwrap();
    statement
        .query_map([object_type], |row| row.get(0))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}

fn version_rows(connection: &rusqlite::Connection) -> Vec<(String, String)> {
    let mut statement = connection
        .prepare(
            "SELECT CAST(version AS TEXT), typeof(version)
             FROM schema_version ORDER BY rowid",
        )
        .unwrap();
    statement
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap()
}

#[derive(Debug, PartialEq, Eq)]
struct SchemaSnapshot {
    objects: Vec<(String, String, String, Option<String>)>,
    journal_mode: String,
}

fn schema_snapshot(database: &TestDatabase) -> SchemaSnapshot {
    let connection = rusqlite::Connection::open(database.path()).unwrap();
    let journal_mode = connection
        .query_row("PRAGMA journal_mode", [], |row| row.get(0))
        .unwrap();
    let mut statement = connection
        .prepare(
            "SELECT type, name, tbl_name, sql FROM main.sqlite_schema
             WHERE name NOT GLOB 'sqlite_*' ORDER BY type, name",
        )
        .unwrap();
    let objects = statement
        .query_map([], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        })
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    SchemaSnapshot {
        objects,
        journal_mode,
    }
}

struct TestDatabase {
    path: PathBuf,
}

impl TestDatabase {
    fn new(case: &str) -> Self {
        let id = NEXT_DATABASE_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "tea-schema-v1-{case}-{}-{id}.sqlite",
            std::process::id()
        ));
        Self { path }
    }

    fn path(&self) -> &str {
        self.path.to_str().unwrap()
    }

    fn sidecar_exists(&self, suffix: &str) -> bool {
        path_with_suffix(&self.path, suffix).exists()
    }
}

impl Drop for TestDatabase {
    fn drop(&mut self) {
        for suffix in ["", "-wal", "-shm", "-journal"] {
            let _ = std::fs::remove_file(path_with_suffix(&self.path, suffix));
        }
    }
}

fn path_with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_owned();
    value.push(suffix);
    PathBuf::from(value)
}
