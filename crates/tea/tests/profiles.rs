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
    AgentCommand, CommandEnvelope, CommandId, ModelId, ProtocolMetadata, ProtocolTimestamp,
    SessionId, StopReason, TokenCount,
};
use tea_testkit::{ScriptedModelProvider, ScriptedModelResponse};

const NOW: &str = "2026-07-23T09:30:12.125Z";

fn envelope(command: AgentCommand, session_id: Option<SessionId>) -> CommandEnvelope {
    CommandEnvelope::new(
        CommandId::from_str("0195a0b1-0000-7000-8000-000000000030").unwrap(),
        session_id,
        ProtocolTimestamp::from_str(NOW).unwrap(),
        command,
    )
    .unwrap()
}

fn write_call(opaque_id: &str) -> ScriptedModelResponse {
    let index = ModelStreamIndex::new(0).unwrap();
    let provider_id = ProviderToolCallId::from_str(opaque_id).unwrap();
    ScriptedModelResponse::events([
        ModelEvent::Started(ModelResponseInfo::new()),
        ModelEvent::ToolCallStarted(
            ToolCallStarted::new(index, provider_id.clone(), "write_file").unwrap(),
        ),
        ModelEvent::ToolCallCompleted(
            ToolCallCompleted::new(
                index,
                provider_id,
                "write_file",
                serde_json::json!({"path":"/out.txt","content":"x"}),
            )
            .unwrap(),
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

async fn create_session(runtime: &tea::AgentRuntime, profile: &str) -> SessionId {
    match runtime
        .send(envelope(
            AgentCommand::CreateSession {
                profile_id: profile.parse().unwrap(),
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
async fn two_profiles_differ_in_prompt_tools_limits_and_policy() {
    // Shared provider/scripts consumed in FIFO order across both sessions.
    let provider = provider([
        write_call("write-1"), // coding session: write_file -> Ask (pauses)
        write_call("write-2"), // desktop session: write_file -> HardDeny -> denied
        ScriptedModelResponse::text(["done"]), // desktop session: final response
    ]);
    let runtime = build_runtime(
        provider.clone(),
        Arc::new(TestIds::default()),
        Arc::new(TestSessionIds::default()),
    )
    .unwrap();

    let coding_id = create_session(&runtime, "coding-agent").await;
    let desktop_id = create_session(&runtime, "desktop-assistant").await;

    // Coding profile: write_file pauses for approval (Ask).
    let coding_outcome = runtime
        .send(envelope(
            AgentCommand::Prompt {
                message: user_message("write a note"),
                final_output_format: None,
            },
            Some(coding_id),
        ))
        .await
        .unwrap();
    match coding_outcome {
        RuntimeCommandOutcome::RunCompleted {
            state,
            pending_approval_id,
            ..
        } => {
            assert_eq!(state, tea_kernel::RunState::WaitingApproval);
            assert!(pending_approval_id.is_some());
        }
        other => panic!("coding profile should pause for approval, got {other:?}"),
    }

    // Desktop profile: same write_file is hard-denied (no authorizing rule), so
    // the run completes instead of pausing.
    let desktop_outcome = runtime
        .send(envelope(
            AgentCommand::Prompt {
                message: user_message("write a note"),
                final_output_format: None,
            },
            Some(desktop_id),
        ))
        .await
        .unwrap();
    match desktop_outcome {
        RuntimeCommandOutcome::RunCompleted {
            state,
            pending_approval_id,
            ..
        } => {
            assert_eq!(state, tea_kernel::RunState::Completed);
            assert!(pending_approval_id.is_none());
        }
        other => panic!("desktop profile should complete via denial, got {other:?}"),
    }

    let requests = provider.captured_requests().unwrap();
    let coding_request = &requests[0];
    let desktop_request = &requests[1];
    // Different compiled system prompts (different active tool hints).
    assert_ne!(
        coding_request.system_prompt().unwrap_or(""),
        desktop_request.system_prompt().unwrap_or("")
    );
    // Different active tool sets projected to the model request.
    let coding_tools = coding_request
        .tools()
        .iter()
        .map(|tool| tool.name().to_owned())
        .collect::<Vec<_>>();
    let desktop_tools = desktop_request
        .tools()
        .iter()
        .map(|tool| tool.name().to_owned())
        .collect::<Vec<_>>();
    assert_ne!(coding_tools, desktop_tools);
    assert!(coding_tools.contains(&"write_file".to_owned()));
    assert!(desktop_tools.contains(&"write_file".to_owned()));
    assert!(coding_tools.contains(&"read_file".to_owned()));
    assert!(desktop_tools.contains(&"clipboard_read".to_owned()));

    // Different run limits per profile.
    let coding_binding = runtime.binding(&"coding-agent".parse().unwrap()).unwrap();
    let desktop_binding = runtime
        .binding(&"desktop-assistant".parse().unwrap())
        .unwrap();
    assert_ne!(
        coding_binding.run_limits().max_tool_iterations(),
        desktop_binding.run_limits().max_tool_iterations()
    );
    assert_ne!(
        coding_binding.environment().surface(),
        desktop_binding.environment().surface()
    );
}
