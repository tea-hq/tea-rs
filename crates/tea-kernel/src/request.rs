use tea_model::{ModelRequest, ModelSpec, ReasoningOptions};
use tea_protocol::{
    CanonicalMessage, ContentBlock, FINAL_JSON_OBJECT_INSTRUCTION, FinalOutputFormat,
};
use tea_session::MaterializedSessionState;
use tea_tools::{ToolName, ToolRegistry};

use crate::{KernelError, KernelErrorCode, KernelRunConfig};

/// Immutable model request plus the durable tail used to construct it.
#[derive(Debug, Clone, PartialEq)]
pub struct TurnRequestSnapshot {
    request: ModelRequest,
    client_tool_names: Vec<ToolName>,
    hosted_tool_names: Vec<ToolName>,
    durable_tail: tea_protocol::SessionSequence,
}

impl TurnRequestSnapshot {
    /// Builds and validates one immutable request from committed session state.
    ///
    /// Registered tool specifications are projected in canonical name order and
    /// parallel calls remain disabled for the sequential milestone scheduler.
    ///
    /// # Errors
    ///
    /// Returns a typed error when no model is selected, the provider does not
    /// advertise it, or model/tool/request bounds are incompatible.
    pub fn build(
        state: &MaterializedSessionState,
        config: &KernelRunConfig,
        tools: &ToolRegistry,
        model: &ModelSpec,
    ) -> Result<Self, KernelError> {
        Self::build_with_pending_messages(state, &[], config, tools, model)
    }

    /// Builds and validates a request prospectively, before pending messages
    /// become durable session state.
    ///
    /// # Errors
    ///
    /// Returns the same typed validation failures as [`Self::build`], including
    /// bounds that apply to the complete committed-plus-pending transcript.
    pub fn build_with_pending_messages(
        state: &MaterializedSessionState,
        pending_messages: &[CanonicalMessage],
        config: &KernelRunConfig,
        tools: &ToolRegistry,
        model: &ModelSpec,
    ) -> Result<Self, KernelError> {
        let model_ref = state.configuration().model_ref().ok_or_else(|| {
            KernelError::new(KernelErrorCode::InvalidModel, "session has no active model")
        })?;
        if model.model_ref() != model_ref {
            return Err(KernelError::new(
                KernelErrorCode::InvalidModel,
                "resolved model does not match the active session model",
            ));
        }
        let definitions = tools.model_definitions(model).map_err(|error| {
            KernelError::new(KernelErrorCode::InvalidRequest, error.to_string())
        })?;
        let mut client_tool_names = Vec::new();
        let mut hosted_tool_names = Vec::new();
        for (name, definition) in tools.names().zip(&definitions) {
            if definition.as_function().is_some() {
                client_tool_names.push(name.clone());
            } else if definition.as_hosted().is_some() {
                hosted_tool_names.push(name.clone());
            }
        }
        let mut messages = project_model_messages(state)?;
        messages.reserve(pending_messages.len());
        messages.extend_from_slice(pending_messages);
        let final_output_format = config.final_output_format();
        let mut request = ModelRequest::new(model_ref.model_id().clone(), messages)?
            .with_tools(definitions, false)?
            .with_metadata(config.request_metadata().clone());
        if let (Some(requested), Some(profile)) = (
            state.configuration().reasoning_effort(),
            model.reasoning_profile(),
        ) {
            let effective = profile.resolve(requested).effective();
            request = request.with_reasoning(ReasoningOptions::new(effective));
        }
        if let Some(prompt) =
            model_system_prompt(config.system_prompt(), final_output_format.as_ref())
        {
            request = request.with_system_prompt(prompt)?;
        }
        if let Some(format) = final_output_format {
            request = request.with_final_output_format(format);
        }
        request.validate_for(model)?;
        Ok(Self {
            request,
            client_tool_names,
            hosted_tool_names,
            durable_tail: state.tail_sequence(),
        })
    }

    /// Returns the immutable provider request.
    #[must_use]
    pub const fn request(&self) -> &ModelRequest {
        &self.request
    }

    /// Returns the exact client-executable function names frozen for this turn.
    ///
    /// Active hosted projections are deliberately absent even when their
    /// portable tool specifications also have a client route.
    #[must_use]
    pub fn client_tool_names(&self) -> &[ToolName] {
        &self.client_tool_names
    }

    /// Returns the exact provider-hosted tool names frozen for this turn.
    #[must_use]
    pub fn hosted_tool_names(&self) -> &[ToolName] {
        &self.hosted_tool_names
    }

    /// Returns whether a provider function call may enter local execution.
    #[must_use]
    pub fn allows_client_tool_call(&self, tool_name: &str) -> bool {
        self.client_tool_names
            .iter()
            .any(|name| name.as_str() == tool_name)
    }

    /// Returns whether this name was projected as a provider-hosted tool.
    #[must_use]
    pub fn is_hosted_tool_projection(&self, tool_name: &str) -> bool {
        self.hosted_tool_names
            .iter()
            .any(|name| name.as_str() == tool_name)
    }

    /// Consumes the snapshot into its provider request.
    #[must_use]
    pub fn into_request(self) -> ModelRequest {
        self.request
    }

    /// Returns the durable session tail captured with the request.
    #[must_use]
    pub const fn durable_tail(&self) -> tea_protocol::SessionSequence {
        self.durable_tail
    }
}

fn model_system_prompt(
    existing: Option<&str>,
    final_output_format: Option<&FinalOutputFormat>,
) -> Option<String> {
    if !matches!(final_output_format, Some(FinalOutputFormat::JsonObject))
        || existing.is_some_and(|prompt| {
            prompt
                .lines()
                .any(|line| line == FINAL_JSON_OBJECT_INSTRUCTION)
        })
    {
        return existing.map(str::to_owned);
    }
    Some(existing.map_or_else(
        || FINAL_JSON_OBJECT_INSTRUCTION.to_owned(),
        |prompt| format!("{prompt}\n\n{FINAL_JSON_OBJECT_INSTRUCTION}"),
    ))
}

fn project_model_messages(
    state: &MaterializedSessionState,
) -> Result<Vec<CanonicalMessage>, KernelError> {
    let mut messages = Vec::with_capacity(state.messages().len());
    let Some(compaction) = state.latest_compaction() else {
        messages.extend_from_slice(state.messages());
        return Ok(messages);
    };
    let Some((summary, remainder)) = state.messages().split_first() else {
        return Err(KernelError::new(
            KernelErrorCode::InvalidState,
            "compaction summary is missing from the active transcript",
        ));
    };
    if summary != compaction.summary() {
        return Err(KernelError::new(
            KernelErrorCode::InvalidState,
            "compaction summary is not the active transcript head",
        ));
    }
    messages.push(project_compaction_summary(summary)?);
    messages.extend_from_slice(remainder);
    Ok(messages)
}

fn project_compaction_summary(summary: &CanonicalMessage) -> Result<CanonicalMessage, KernelError> {
    let CanonicalMessage::Assistant {
        id,
        content,
        timestamp,
        ..
    } = summary
    else {
        return Err(KernelError::new(
            KernelErrorCode::InvalidState,
            "compaction summary is not an assistant message",
        ));
    };
    let content = content
        .iter()
        .map(|block| match block {
            ContentBlock::Text { text } | ContentBlock::ContextualText { text } => {
                ContentBlock::contextual_text(text.clone()).map_err(KernelError::from)
            }
            ContentBlock::Thinking { .. }
            | ContentBlock::Image { .. }
            | ContentBlock::ToolCall { .. }
            | ContentBlock::HostedTool { .. }
            | ContentBlock::Citation { .. } => Err(KernelError::new(
                KernelErrorCode::InvalidState,
                "compaction summary contains non-text content",
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    CanonicalMessage::user(*id, content, *timestamp).map_err(KernelError::from)
}

impl From<tea_model::ModelRequestError> for KernelError {
    fn from(error: tea_model::ModelRequestError) -> Self {
        Self::new(KernelErrorCode::InvalidRequest, error.to_string())
    }
}
