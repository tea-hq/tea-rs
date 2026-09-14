#![allow(dead_code)]

use std::str::FromStr;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::{Arc, Mutex};

use std::future::pending;
use tea::{AgentRuntime, AgentRuntimeBuilder, RuntimeError, SessionIdSource};
use tea_kernel::{KernelClock, KernelDeadlineFuture, KernelError, KernelIdSource};
use tea_model::{ModelCapabilities, ModelDisplayName, ModelProvider, ModelSpec, ProviderId};
use tea_policy::{ActorId, CodingWorkspacePolicy, DesktopPolicy, GrantId};
use tea_profile::ProfileRuleId;
use tea_protocol::{
    ApprovalId, CanonicalMessage, ContentBlock, EventId, MessageId, ProtocolTimestamp, RecordId,
    RunId, SessionId, TokenCount, ToolCallId, TurnId,
};
use tea_testkit::{FakeReadTool, FakeWriteTool, ScriptedModelProvider, ScriptedModelResponse};
use tea_tools::{
    ArgumentResourceResolver, ToolConcurrency, ToolEffect, ToolExecutionSemantics, ToolName,
    ToolResourceAccess, ToolSpec, ToolTimeout, ToolVersion,
};

pub const NOW: &str = "2026-07-23T09:30:12.125Z";

static MESSAGE_COUNTER: std::sync::atomic::AtomicU16 = std::sync::atomic::AtomicU16::new(0);

/// Builds a canonical user message with a unique message id for tests.
pub fn user_message(text: &str) -> CanonicalMessage {
    use std::str::FromStr;
    use std::sync::atomic::Ordering;
    let value = MESSAGE_COUNTER.fetch_add(1, Ordering::SeqCst);
    let message_id = MessageId::from_str(&format!("0195a0b1-5e52-7000-8000-{value:012}")).unwrap();
    CanonicalMessage::user(
        message_id,
        vec![ContentBlock::text(text).unwrap()],
        ProtocolTimestamp::from_str(NOW).unwrap(),
    )
    .unwrap()
}

pub fn envelope(
    command: tea_protocol::AgentCommand,
    session_id: Option<tea_protocol::SessionId>,
) -> tea_protocol::CommandEnvelope {
    tea_protocol::CommandEnvelope::new(
        tea_protocol::CommandId::from_str("0195a0b1-0000-7000-8000-0000000000f0").unwrap(),
        session_id,
        ProtocolTimestamp::from_str(NOW).unwrap(),
        command,
    )
    .unwrap()
}

pub fn envelope_create(profile: &str) -> tea_protocol::CommandEnvelope {
    envelope(
        tea_protocol::AgentCommand::CreateSession {
            profile_id: profile.parse().unwrap(),
            metadata: tea_protocol::ProtocolMetadata::default(),
        },
        None,
    )
}

pub fn envelope_prompt(
    message: CanonicalMessage,
    session_id: tea_protocol::SessionId,
) -> tea_protocol::CommandEnvelope {
    envelope(
        tea_protocol::AgentCommand::Prompt {
            message,
            final_output_format: None,
        },
        Some(session_id),
    )
}

#[derive(Debug, Default)]
pub struct FixedClock;
impl KernelClock for FixedClock {
    fn now(&self) -> Result<ProtocolTimestamp, KernelError> {
        Ok(ProtocolTimestamp::from_str(NOW).unwrap())
    }
    fn sleep_until(&self, _deadline: ProtocolTimestamp) -> KernelDeadlineFuture<'_> {
        Box::pin(pending())
    }
}

#[derive(Debug, Default)]
pub struct TestIds(AtomicU16);
impl TestIds {
    fn next<T: FromStr>(&self) -> Result<T, RuntimeError>
    where
        T::Err: std::fmt::Display,
    {
        let value = self.0.fetch_add(1, Ordering::SeqCst);
        format!("0195a0b1-5e3a-7000-8000-{value:012}")
            .parse()
            .map_err(|error: T::Err| {
                RuntimeError::new(tea::RuntimeErrorCode::InvalidState, error.to_string())
            })
    }
}
impl KernelIdSource for TestIds {
    fn next_run_id(&self) -> Result<RunId, KernelError> {
        self.next::<RunId>().map_err(kernel_err)
    }
    fn next_turn_id(&self) -> Result<TurnId, KernelError> {
        self.next::<TurnId>().map_err(kernel_err)
    }
    fn next_message_id(&self) -> Result<MessageId, KernelError> {
        self.next::<MessageId>().map_err(kernel_err)
    }
    fn next_tool_call_id(&self) -> Result<ToolCallId, KernelError> {
        self.next::<ToolCallId>().map_err(kernel_err)
    }
    fn next_approval_id(&self) -> Result<ApprovalId, KernelError> {
        self.next::<ApprovalId>().map_err(kernel_err)
    }
    fn next_grant_id(&self) -> Result<GrantId, KernelError> {
        self.next::<GrantId>().map_err(kernel_err)
    }
    fn next_event_id(&self) -> Result<EventId, KernelError> {
        self.next::<EventId>().map_err(kernel_err)
    }
    fn next_record_id(&self) -> Result<RecordId, KernelError> {
        self.next::<RecordId>().map_err(kernel_err)
    }
}

#[derive(Debug, Default)]
pub struct TestSessionIds(Mutex<u16>);
impl SessionIdSource for TestSessionIds {
    fn next_session_id(&self) -> Result<SessionId, RuntimeError> {
        let mut value = self.0.lock().unwrap();
        *value += 1;
        let text = format!("0195a0b1-5e3a-7000-8000-{value:012}");
        SessionId::from_str(&text).map_err(|_| {
            RuntimeError::new(
                tea::RuntimeErrorCode::InvalidState,
                "session id source produced an invalid id",
            )
        })
    }
}

#[allow(clippy::needless_pass_by_value)]
fn kernel_err(error: RuntimeError) -> KernelError {
    KernelError::new(
        tea_kernel::KernelErrorCode::IdExhausted,
        error.message().to_owned(),
    )
}

pub fn provider() -> Arc<ScriptedModelProvider> {
    provider_with([])
}

pub fn provider_with(
    scripts: impl IntoIterator<Item = ScriptedModelResponse>,
) -> Arc<ScriptedModelProvider> {
    provider_with_capabilities(scripts, ModelCapabilities::text().with_tools(true))
}

pub fn provider_with_capabilities(
    scripts: impl IntoIterator<Item = ScriptedModelResponse>,
    capabilities: ModelCapabilities,
) -> Arc<ScriptedModelProvider> {
    let provider_id = ProviderId::from_str("fake").unwrap();
    let model = ModelSpec::new(
        tea_protocol::ModelId::from_str("fake/model").unwrap(),
        provider_id.clone(),
        ModelDisplayName::from_str("Fake Model").unwrap(),
        TokenCount::new(32_000).unwrap(),
        TokenCount::new(4_000).unwrap(),
        capabilities,
    )
    .unwrap();
    Arc::new(ScriptedModelProvider::new(
        provider_id,
        vec![model],
        scripts,
    ))
}

pub fn spec(name: &str, effect: ToolEffect) -> ToolSpec {
    use tea_protocol::ToolIdempotency;
    let idempotency = if effect == ToolEffect::FsWrite {
        ToolIdempotency::NonIdempotent
    } else {
        ToolIdempotency::Idempotent
    };
    ToolSpec::new(
        ToolName::from_str(name).unwrap(),
        ToolVersion::from_str("1.0.0").unwrap(),
        format!("Deterministic {name}."),
        serde_json::json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}),
        serde_json::json!({"type":"object","properties":{"content":{"type":"string"}},"required":["content"]}),
        [effect],
        ToolExecutionSemantics::new(
            idempotency,
            if idempotency == ToolIdempotency::Idempotent {
                tea_tools::ToolRetrySafety::Automatic
            } else {
                tea_tools::ToolRetrySafety::ExplicitOnly
            },
            ToolConcurrency::Serial,
            ToolTimeout::from_millis(1_000).unwrap(),
        )
        .unwrap(),
    )
    .unwrap()
    .with_prompt_hint(format!("Invoke {name} when appropriate."))
    .unwrap()
}

pub fn build_runtime(
    provider: Arc<ScriptedModelProvider>,
    ids: Arc<TestIds>,
    session_ids: Arc<TestSessionIds>,
) -> Result<AgentRuntime, RuntimeError> {
    runtime_builder(provider, ids, session_ids)?.build()
}

pub fn runtime_builder(
    provider: Arc<ScriptedModelProvider>,
    ids: Arc<TestIds>,
    session_ids: Arc<TestSessionIds>,
) -> Result<AgentRuntimeBuilder, RuntimeError> {
    runtime_builder_with_write_tool(provider, ids, session_ids, Arc::new(FakeWriteTool::new()))
}

pub fn runtime_builder_with_write_tool(
    provider: Arc<dyn ModelProvider>,
    ids: Arc<TestIds>,
    session_ids: Arc<TestSessionIds>,
    write_tool: Arc<FakeWriteTool>,
) -> Result<AgentRuntimeBuilder, RuntimeError> {
    let builder = AgentRuntimeBuilder::new()
        .provider(provider)
        .clock(Arc::new(FixedClock))
        .ids(ids)
        .session_id_source(session_ids as Arc<dyn SessionIdSource>)
        .actor(ActorId::from_str("user:alice").unwrap())
        .tool(
            spec("read_file", ToolEffect::FsRead),
            Arc::new(
                ArgumentResourceResolver::new("path", "file", ToolResourceAccess::Read).unwrap(),
            ),
            Arc::new(FakeReadTool::new([(
                "/notes.txt".to_owned(),
                "hello".to_owned(),
            )])),
        )?
        .tool(
            spec("write_file", ToolEffect::FsWrite),
            Arc::new(
                ArgumentResourceResolver::new("path", "file", ToolResourceAccess::Write).unwrap(),
            ),
            write_tool,
        )?
        .tool(
            spec("clipboard_read", ToolEffect::ClipboardRead),
            Arc::new(
                ArgumentResourceResolver::new("path", "clipboard", ToolResourceAccess::Read)
                    .unwrap(),
            ),
            Arc::new(FakeReadTool::new([("clip".to_owned(), "data".to_owned())])),
        )?
        .policy_rule(
            ProfileRuleId::from_str("product.coding_workspace").unwrap(),
            Arc::new(CodingWorkspacePolicy),
        )?
        .policy_rule(
            ProfileRuleId::from_str("product.desktop").unwrap(),
            Arc::new(DesktopPolicy),
        )?
        .profile(tea_profile::AgentProfile::coding_agent().unwrap())
        .profile(tea_profile::AgentProfile::desktop_assistant().unwrap());
    Ok(builder)
}
