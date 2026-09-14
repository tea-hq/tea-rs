use std::str::FromStr;
use std::sync::Arc;

use tea_model::ModelProvider;
use tea_policy::{
    ActorId, ExecutionSurface, FilesystemReadPolicy, PolicyEnvironment, PolicyExecutionTarget,
    PolicyRule, WorkspaceId,
};
use tea_profile::{AgentProfile, ProfileRuleId, ProfileTrustLevel, ProfileWorkspaceInstruction};
use tea_protocol::{
    AgentCommand, CanonicalMessage, CommandEnvelope, CommandId, ContentBlock, FinalOutputFormat,
    ModelRef, ProfileId, ProtocolMetadata, SessionId,
};
use tea_tools::{ToolExecutor, ToolResourceResolver, ToolSpec};
use uuid::Uuid;

use crate::{
    AgentRuntime, AgentRuntimeBuilder, RuntimeCommandOutcome, RuntimeError, RuntimeErrorCode,
};

const DEFAULT_ACTOR_ID: &str = "user:host";
const DEFAULT_PROFILE_ID: &str = "agent";
const DEFAULT_PROFILE_NAME: &str = "Agent";
const DEFAULT_WORKSPACE_ID: &str = "workspace";
const SYSTEM_PROMPT_SEGMENT_ID: &str = "host.system-prompt";

/// One client-executed tool registration accepted by [`AgentSessionBuilder`].
pub type AgentToolRegistration = (
    ToolSpec,
    Arc<dyn ToolResourceResolver>,
    Arc<dyn ToolExecutor>,
);

/// High-level builder for one in-memory agent session.
///
/// The builder keeps provider adapters external while supplying conservative
/// defaults for profile identity, policy environment, runtime identities, and
/// session storage. Pure filesystem reads are authorized by default; every
/// other tool effect still requires an explicit policy rule.
#[derive(Debug)]
#[must_use]
pub struct AgentSessionBuilder {
    provider: Arc<dyn ModelProvider>,
    model: ModelRef,
    actor: Option<ActorId>,
    workspace: Option<WorkspaceId>,
    environment: PolicyEnvironment,
    system_prompt: Option<String>,
    tools: Vec<AgentToolRegistration>,
    policy_rules: Vec<(ProfileRuleId, Arc<dyn PolicyRule>)>,
}

impl AgentSessionBuilder {
    /// Creates a session builder from one concrete provider adapter and model.
    pub fn new(provider: Arc<dyn ModelProvider>, model: ModelRef) -> Self {
        Self {
            provider,
            model,
            actor: None,
            workspace: None,
            environment: PolicyEnvironment::new(
                ExecutionSurface::Service,
                PolicyExecutionTarget::Native,
                ProtocolMetadata::default(),
            ),
            system_prompt: None,
            tools: Vec::new(),
            policy_rules: Vec::new(),
        }
    }

    /// Replaces the default host actor identity.
    pub fn actor(mut self, actor: ActorId) -> Self {
        self.actor = Some(actor);
        self
    }

    /// Replaces the default workspace identity.
    pub fn workspace(mut self, workspace: WorkspaceId) -> Self {
        self.workspace = Some(workspace);
        self
    }

    /// Replaces the default service/native policy environment.
    pub fn environment(mut self, environment: PolicyEnvironment) -> Self {
        self.environment = environment;
        self
    }

    /// Adds one trusted host-owned system prompt.
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Adds one client-executed tool and activates it for this session.
    pub fn tool(
        mut self,
        spec: ToolSpec,
        resolver: Arc<dyn ToolResourceResolver>,
        executor: Arc<dyn ToolExecutor>,
    ) -> Self {
        self.tools.push((spec, resolver, executor));
        self
    }

    /// Adds client-executed tools and activates them for this session.
    pub fn tools(mut self, tools: impl IntoIterator<Item = AgentToolRegistration>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Registers and activates one policy rule for this session.
    pub fn policy_rule(mut self, id: ProfileRuleId, rule: Arc<dyn PolicyRule>) -> Self {
        self.policy_rules.push((id, rule));
        self
    }

    /// Builds the runtime and creates one in-memory session.
    ///
    /// # Errors
    ///
    /// Returns an error when an identity, prompt, profile, tool, policy, model,
    /// or session invariant cannot be satisfied.
    pub async fn build(self) -> Result<AgentSession, RuntimeError> {
        let profile_id: ProfileId = parse_identity(DEFAULT_PROFILE_ID, "default profile ID")?;
        let actor: ActorId = self
            .actor
            .map_or_else(|| parse_identity(DEFAULT_ACTOR_ID, "default actor ID"), Ok)?;
        let workspace: WorkspaceId = self.workspace.map_or_else(
            || parse_identity(DEFAULT_WORKSPACE_ID, "default workspace ID"),
            Ok,
        )?;
        let mut profile = AgentProfile::builder(
            profile_id.clone(),
            parse_identity(DEFAULT_PROFILE_NAME, "default profile name")?,
            self.model,
        )
        .environment(self.environment);

        if let Some(system_prompt) = self.system_prompt {
            let instruction = ProfileWorkspaceInstruction::new(
                parse_identity(SYSTEM_PROMPT_SEGMENT_ID, "system prompt segment ID")?,
                system_prompt,
                "host://system-prompt",
                ProfileTrustLevel::Trusted,
            )
            .map_err(|error| invalid_request(error.to_string()))?;
            profile = profile.workspace_instruction(instruction);
        }

        let mut runtime = AgentRuntimeBuilder::new()
            .provider(self.provider)
            .actor(actor)
            .workspace(workspace);
        let default_policy: Arc<dyn PolicyRule> = Arc::new(FilesystemReadPolicy);
        let default_policy_id: ProfileRuleId =
            parse_identity(default_policy.id(), "default policy rule ID")?;
        profile = profile.policy_rule(default_policy_id.clone());
        runtime = runtime.policy_rule(default_policy_id, default_policy)?;
        for (spec, resolver, executor) in self.tools {
            profile = profile.active_tool(spec.name().clone());
            runtime = runtime.tool(spec, resolver, executor)?;
        }
        for (id, rule) in self.policy_rules {
            profile = profile.policy_rule(id.clone());
            runtime = runtime.policy_rule(id, rule)?;
        }
        let profile = profile
            .build()
            .map_err(|error| invalid_request(error.to_string()))?;
        let runtime = runtime.profile(profile).build()?;
        let RuntimeCommandOutcome::Created { session_id } = runtime
            .create_session(profile_id, ProtocolMetadata::default())
            .await?
        else {
            return Err(RuntimeError::new(
                RuntimeErrorCode::InvalidState,
                "session creation returned an unexpected outcome",
            ));
        };
        Ok(AgentSession {
            runtime,
            session_id,
        })
    }
}

/// One ready-to-prompt in-memory agent session.
#[derive(Debug)]
pub struct AgentSession {
    runtime: AgentRuntime,
    session_id: SessionId,
}

impl AgentSession {
    /// Creates a high-level session builder.
    pub fn builder(provider: Arc<dyn ModelProvider>, model: ModelRef) -> AgentSessionBuilder {
        AgentSessionBuilder::new(provider, model)
    }

    /// Returns the generated session identity.
    #[must_use]
    pub const fn session_id(&self) -> SessionId {
        self.session_id
    }

    /// Sends one plain-text user prompt and waits for the completed response.
    ///
    /// The method owns an internal event subscription so bounded event
    /// backpressure cannot stall the run. Advanced hosts can continue to use
    /// [`AgentRuntime`](crate::AgentRuntime) directly for full event control.
    ///
    /// # Errors
    ///
    /// Returns an error when input validation, runtime execution, event
    /// delivery, approval, or terminal run-state validation fails.
    pub async fn prompt(
        &self,
        text: impl Into<String>,
        final_output_format: Option<FinalOutputFormat>,
    ) -> Result<AgentResponse, RuntimeError> {
        let mut events = self.runtime.subscribe(self.session_id)?;
        let timestamp = self.runtime.clock().now().map_err(RuntimeError::from)?;
        let message = CanonicalMessage::user(
            self.runtime
                .ids()
                .next_message_id()
                .map_err(RuntimeError::from)?,
            vec![
                ContentBlock::text(text.into())
                    .map_err(|error| invalid_request(error.to_string()))?,
            ],
            timestamp,
        )
        .map_err(|error| invalid_request(error.to_string()))?;
        let command = CommandEnvelope::new(
            new_command_id()?,
            Some(self.session_id),
            timestamp,
            AgentCommand::Prompt {
                message,
                final_output_format,
            },
        )
        .map_err(|error| invalid_request(error.to_string()))?;

        let mut send = Box::pin(self.runtime.send(command));
        let outcome = loop {
            tokio::select! {
                event = events.recv() => match event {
                    Some(_) => {}
                    None => return Err(RuntimeError::new(
                        RuntimeErrorCode::InvalidState,
                        "runtime event subscription closed before prompt completion",
                    )),
                },
                result = &mut send => break result?,
            }
        };
        while events.try_recv().is_ok() {}
        match outcome {
            RuntimeCommandOutcome::RunCompleted {
                state: tea_kernel::RunState::Completed,
                session,
                pending_approval_id: None,
                ..
            } => AgentResponse::from_messages(session.messages()),
            RuntimeCommandOutcome::RunCompleted {
                pending_approval_id: Some(_),
                ..
            } => Err(RuntimeError::new(
                RuntimeErrorCode::PolicyFailure,
                "prompt paused for approval",
            )),
            _ => Err(RuntimeError::new(
                RuntimeErrorCode::InvalidState,
                "prompt returned an unexpected outcome",
            )),
        }
    }
}

/// Visible output from the final durable assistant message of one completed prompt.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AgentResponse {
    text: String,
}

impl AgentResponse {
    /// Returns the visible text blocks in durable final-message order.
    #[must_use]
    pub fn text(&self) -> &str {
        &self.text
    }

    fn from_messages(messages: &[CanonicalMessage]) -> Result<Self, RuntimeError> {
        let Some(CanonicalMessage::Assistant { content, .. }) = messages.last() else {
            return Err(RuntimeError::new(
                RuntimeErrorCode::InvalidState,
                "completed prompt has no final assistant message",
            ));
        };
        let text = content
            .iter()
            .filter_map(|block| match block {
                ContentBlock::Text { text } => Some(text.as_str()),
                ContentBlock::ContextualText { .. }
                | ContentBlock::Thinking { .. }
                | ContentBlock::Image { .. }
                | ContentBlock::ToolCall { .. }
                | ContentBlock::HostedTool { .. }
                | ContentBlock::Citation { .. } => None,
            })
            .collect();
        Ok(Self { text })
    }
}

fn new_command_id() -> Result<CommandId, RuntimeError> {
    CommandId::from_str(&Uuid::now_v7().hyphenated().to_string())
        .map_err(|error| invalid_request(error.to_string()))
}

fn parse_identity<T>(value: &str, name: &str) -> Result<T, RuntimeError>
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    value
        .parse()
        .map_err(|error: T::Err| invalid_request(format!("{name} is invalid: {error}")))
}

fn invalid_request(message: impl Into<String>) -> RuntimeError {
    RuntimeError::new(RuntimeErrorCode::InvalidRequest, message)
}
