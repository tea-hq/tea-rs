use crate::common;

use std::str::FromStr;
use std::sync::Arc;

use common::{TestIds, TestSessionIds, build_runtime, user_message};
use tea::RuntimeCommandOutcome;
use tea_model::{
    ModelCapabilities, ModelCompletion, ModelDisplayName, ModelEvent, ModelResponseInfo, ModelSpec,
    ModelStreamIndex, ProviderId, ProviderToolCallId, ToolCallCompleted, ToolCallStarted,
};
use tea_protocol::{
    AgentCommand, CommandEnvelope, CommandId, CommandText, ModelId, ProtocolMetadata,
    ProtocolTimestamp, SessionId, StopReason, TokenCount,
};
use tea_testkit::{ScriptedModelProvider, ScriptedModelResponse};

const NOW: &str = "2026-07-23T09:30:12.125Z";

fn envelope(command: AgentCommand, session_id: Option<SessionId>) -> CommandEnvelope {
    CommandEnvelope::new(
        CommandId::from_str("0195a0b1-0000-7000-8000-000000000020").unwrap(),
        session_id,
        ProtocolTimestamp::from_str(NOW).unwrap(),
        command,
    )
    .unwrap()
}

fn steer(text: &str) -> AgentCommand {
    AgentCommand::Steer {
        text: CommandText::new(text).unwrap(),
    }
}

fn tool_script(name: &str, arguments: serde_json::Value, opaque_id: &str) -> ScriptedModelResponse {
    let index = ModelStreamIndex::new(0).unwrap();
    let provider_id = ProviderToolCallId::from_str(opaque_id).unwrap();
    ScriptedModelResponse::events([
        ModelEvent::Started(ModelResponseInfo::new()),
        ModelEvent::ToolCallStarted(
            ToolCallStarted::new(index, provider_id.clone(), name).unwrap(),
        ),
        ModelEvent::ToolCallCompleted(
            ToolCallCompleted::new(index, provider_id, name, arguments).unwrap(),
        ),
        ModelEvent::Completed(ModelCompletion::new(StopReason::ToolUse).unwrap()),
    ])
}

fn provider(
    scripts: impl IntoIterator<Item = ScriptedModelResponse>,
) -> Arc<ScriptedModelProvider> {
    let provider_id = ProviderId::from_str("fake").unwrap();
    let model = ModelSpec::new(
        ModelId::from_str("fake/model").unwrap(),
        provider_id.clone(),
        ModelDisplayName::from_str("Fake Model").unwrap(),
        TokenCount::new(32_000).unwrap(),
        TokenCount::new(4_000).unwrap(),
        ModelCapabilities::text().with_tools(true),
    )
    .unwrap();
    Arc::new(ScriptedModelProvider::new(
        provider_id,
        vec![model],
        scripts,
    ))
}

async fn create_session(runtime: &tea::AgentRuntime) -> SessionId {
    match runtime
        .send(envelope(
            AgentCommand::CreateSession {
                profile_id: "coding-agent".parse().unwrap(),
                metadata: ProtocolMetadata::default(),
            },
            None,
        ))
        .await
        .unwrap()
    {
        RuntimeCommandOutcome::Created { session_id } => session_id,
        other => panic!("expected Created, got {other:?}"),
    }
}

#[tokio::test]
async fn steer_coalesces_into_next_turn() {
    let provider = provider([
        tool_script(
            "read_file",
            serde_json::json!({"path":"/notes.txt"}),
            "read-1",
        ),
        ScriptedModelResponse::text(["summary"]),
    ]);
    let runtime = build_runtime(
        provider.clone(),
        Arc::new(TestIds::default()),
        Arc::new(TestSessionIds::default()),
    )
    .unwrap();
    let session_id = create_session(&runtime).await;

    let outcome = runtime
        .send(envelope(steer("prefer concise answers"), Some(session_id)))
        .await
        .unwrap();
    match outcome {
        RuntimeCommandOutcome::Enqueued {
            follow_ups,
            steering,
        } => {
            assert_eq!(follow_ups, 0);
            assert_eq!(steering, 1);
        }
        other => panic!("expected Enqueued, got {other:?}"),
    }

    let _ = runtime
        .send(envelope(
            AgentCommand::Prompt {
                message: user_message("summarize"),
                final_output_format: None,
            },
            Some(session_id),
        ))
        .await
        .unwrap();
    let requests = provider.captured_requests().unwrap();
    assert!(requests.len() >= 2);
    assert_eq!(
        requests[0].messages().len(),
        2,
        "steering must coalesce into the first model request"
    );
}
