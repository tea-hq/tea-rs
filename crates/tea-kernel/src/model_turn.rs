use std::collections::BTreeMap;

use futures_util::StreamExt;
use tea_model::{
    HostedToolCompleted, ModelCompletion, ModelEvent, ModelFailure, ModelFailureCode,
    ModelProvider, ModelSourceCitation, ModelStreamIndex, ModelStreamValidator, ProviderToolCallId,
    ToolCallCompleted,
};
use tea_protocol::{
    AgentEvent, CanonicalMessage, ContentBlock, EventDelta, HostedToolActivity, MessageId,
    RetryClass, RunId, SourceCitation, StopReason, ToolCallId, TurnId,
};

use crate::approval::add_duration;
use crate::observe::EventEmitter;
use crate::retry::ModelRetryPolicy;
use crate::{KernelError, KernelErrorCode, KernelIdSource, TurnRequestSnapshot};

#[derive(Debug)]
pub(crate) struct ModelTurnOutput {
    pub(crate) message: CanonicalMessage,
    pub(crate) tool_calls: Vec<CompletedToolCall>,
    pub(crate) completion: ModelCompletion,
    pub(crate) output_bytes: usize,
}

#[derive(Debug)]
pub(crate) struct CompletedToolCall {
    pub(crate) tool_call_id: ToolCallId,
    pub(crate) tool_name: String,
    pub(crate) arguments: serde_json::Value,
}

#[derive(Debug)]
enum PendingContent {
    Text(String),
    Thinking(String),
    Tool {
        tool_call_id: ToolCallId,
        completed: Option<ToolCallCompleted>,
    },
    HostedTool {
        tool_call_id: ToolCallId,
        completed: Option<HostedToolCompleted>,
    },
    Citation(ModelSourceCitation),
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_lines)]
pub(crate) async fn stream_turn(
    provider: &dyn ModelProvider,
    snapshot: TurnRequestSnapshot,
    ids: &dyn KernelIdSource,
    emitter: &mut EventEmitter<'_>,
    run_id: RunId,
    turn_id: TurnId,
    cancellation: tea_control::CancellationScope,
    clock: &dyn crate::KernelClock,
    deadline: tea_protocol::ProtocolTimestamp,
    output_budget: usize,
    retry_policy: ModelRetryPolicy,
) -> Result<ModelTurnOutput, KernelError> {
    let message_id = ids.next_message_id()?;
    let hosted_tool_names = snapshot.hosted_tool_names().to_vec();
    let request = snapshot.into_request();
    let mut attempt: u32 = 0;
    loop {
        let mut stream = provider.stream(request.clone(), cancellation.clone());
        let mut validator = ModelStreamValidator::new();
        let mut content = Vec::new();
        let mut tool_positions = BTreeMap::new();
        let mut hosted_positions = BTreeMap::new();
        let mut hosted_provider_ids = BTreeMap::new();
        let mut completion = None;
        let mut output_bytes = 0usize;
        let attempt_failure: Option<ModelFailure> = loop {
            let event = tokio::select! {
                biased;
                () = cancellation.cancelled() => {
                    return Err(KernelError::new(
                        KernelErrorCode::Cancelled,
                        "model request was cancelled",
                    ));
                }
                () = clock.sleep_until(deadline) => {
                    return Err(KernelError::new(
                        KernelErrorCode::LimitExceeded,
                        "run deadline was reached during model streaming",
                    ));
                }
                event = stream.next() => event,
            };
            let Some(event) = event else { break None };
            validator.observe(&event).map_err(|error| {
                KernelError::new(KernelErrorCode::ModelFailure, error.to_string())
            })?;
            match event {
                ModelEvent::TextDelta(delta) => {
                    output_bytes = add_output(output_bytes, delta.as_str().len(), output_budget)?;
                    let index = append_text(&mut content, delta.as_str(), false);
                    emitter
                        .emit(
                            Some(run_id),
                            Some(turn_id),
                            AgentEvent::MessageDelta {
                                message_id,
                                content_index: to_content_index(index)?,
                                delta: EventDelta::TextDelta {
                                    text: delta.as_str().to_owned(),
                                },
                            },
                        )
                        .await?;
                }
                ModelEvent::ThinkingDelta(delta) => {
                    output_bytes = add_output(output_bytes, delta.as_str().len(), output_budget)?;
                    let index = append_text(&mut content, delta.as_str(), true);
                    emitter
                        .emit(
                            Some(run_id),
                            Some(turn_id),
                            AgentEvent::MessageDelta {
                                message_id,
                                content_index: to_content_index(index)?,
                                delta: EventDelta::ThinkingDelta {
                                    text: delta.as_str().to_owned(),
                                },
                            },
                        )
                        .await?;
                }
                ModelEvent::ToolCallStarted(call) => {
                    if hosted_tool_names
                        .iter()
                        .any(|name| name.as_str() == call.tool_name())
                    {
                        return Err(model_contract(
                            "provider returned a client tool call that was not projected",
                        ));
                    }
                    let position = content.len();
                    let tool_call_id = ids.next_tool_call_id()?;
                    if tool_positions.insert(call.index(), position).is_some() {
                        return Err(model_contract("model reused a tool-call index"));
                    }
                    content.push(PendingContent::Tool {
                        tool_call_id,
                        completed: None,
                    });
                }
                ModelEvent::Started(_) | ModelEvent::ToolArgumentsDelta(_) => {}
                ModelEvent::ToolCallCompleted(call) => {
                    complete_tool(&mut content, &tool_positions, call)?;
                }
                ModelEvent::HostedToolStarted(call) => {
                    if !hosted_tool_names
                        .iter()
                        .any(|name| name.as_str() == call.tool_name())
                    {
                        return Err(model_contract(
                            "provider returned a hosted tool activity that was not projected",
                        ));
                    }
                    let position = content.len();
                    let tool_call_id = ids.next_tool_call_id()?;
                    if hosted_positions.insert(call.index(), position).is_some()
                        || hosted_provider_ids
                            .insert(call.provider_call_id().clone(), tool_call_id)
                            .is_some()
                    {
                        return Err(model_contract("model reused a hosted tool identity"));
                    }
                    content.push(PendingContent::HostedTool {
                        tool_call_id,
                        completed: None,
                    });
                    emitter
                        .emit(
                            Some(run_id),
                            Some(turn_id),
                            AgentEvent::HostedToolStarted {
                                tool_call_id,
                                tool_name: call.tool_name().to_owned(),
                            },
                        )
                        .await?;
                }
                ModelEvent::HostedToolCompleted(call) => {
                    let block_bytes = hosted_block_bytes(&content, &hosted_positions, &call)?;
                    output_bytes = add_output(output_bytes, block_bytes, output_budget)?;
                    let source_count = u32::try_from(call.sources().len())
                        .map_err(|_| model_contract("hosted source count exceeded event bounds"))?;
                    let observation = AgentEvent::HostedToolCompleted {
                        tool_call_id: *hosted_provider_ids
                            .get(call.provider_call_id())
                            .ok_or_else(|| {
                                model_contract("completed hosted tool has no identity")
                            })?,
                        tool_name: call.tool_name().to_owned(),
                        arguments: call.arguments().clone(),
                        outcome: call.outcome().clone(),
                        source_count,
                    };
                    complete_hosted_tool(&mut content, &hosted_positions, call)?;
                    emitter
                        .emit(Some(run_id), Some(turn_id), observation)
                        .await?;
                }
                ModelEvent::SourceCitation(citation) => {
                    let canonical = canonical_citation(&citation, &hosted_provider_ids)?;
                    let block = ContentBlock::citation(canonical);
                    let block_bytes = serde_json::to_vec(&block)
                        .map_err(|_| model_contract("model citation could not be measured"))?
                        .len();
                    output_bytes = add_output(output_bytes, block_bytes, output_budget)?;
                    content.push(PendingContent::Citation(citation));
                }
                ModelEvent::Completed(value) => completion = Some(value),
                ModelEvent::Failed(failure) => break Some(failure),
            }
        };
        let Some(failure) = attempt_failure else {
            validator.finish().map_err(|error| {
                KernelError::new(KernelErrorCode::ModelFailure, error.to_string())
            })?;
            let completion =
                completion.ok_or_else(|| model_contract("model stream has no completion"))?;
            return finish_turn(message_id, clock.now()?, content, completion, output_bytes);
        };
        // Non-retryable failures terminate immediately.
        if failure.code() == ModelFailureCode::Cancelled || failure.retry() == RetryClass::Never {
            return Err(KernelError::model_failure(&failure, false));
        }
        // Retryable failure within the policy.
        if attempt + 1 >= retry_policy.max_attempts() {
            return Err(KernelError::model_failure(&failure, true));
        }
        let retry_attempt = attempt + 1;
        let max_retries = retry_policy.max_attempts().saturating_sub(1);
        let delay = retry_policy.delay_for_failure(&failure, retry_attempt);
        let delay_ms = u64::try_from(delay.as_millis())
            .map_err(|_| model_contract("model retry delay exceeded event bounds"))?;
        emitter
            .emit(
                Some(run_id),
                Some(turn_id),
                AgentEvent::ModelRetryScheduled {
                    message_id,
                    attempt: retry_attempt,
                    max_retries,
                    delay_ms,
                },
            )
            .await?;
        let wake = add_duration(clock.now()?, delay)?;
        tokio::select! {
            biased;
            () = cancellation.cancelled() => {
                return Err(KernelError::new(
                    KernelErrorCode::Cancelled,
                    "model request was cancelled during retry backoff",
                ));
            }
            () = clock.sleep_until(wake) => {}
        }
        attempt = retry_attempt;
        emitter
            .emit(
                Some(run_id),
                Some(turn_id),
                AgentEvent::ModelRetryStarted {
                    message_id,
                    attempt,
                    max_retries,
                },
            )
            .await?;
    }
}

fn finish_turn(
    message_id: MessageId,
    timestamp: tea_protocol::ProtocolTimestamp,
    pending: Vec<PendingContent>,
    completion: ModelCompletion,
    output_bytes: usize,
) -> Result<ModelTurnOutput, KernelError> {
    let hosted_ids = pending
        .iter()
        .filter_map(|block| match block {
            PendingContent::HostedTool {
                tool_call_id,
                completed: Some(call),
            } => Some((call.provider_call_id().clone(), *tool_call_id)),
            PendingContent::Text(_)
            | PendingContent::Thinking(_)
            | PendingContent::Tool { .. }
            | PendingContent::HostedTool {
                completed: None, ..
            }
            | PendingContent::Citation(_) => None,
        })
        .collect::<BTreeMap<_, _>>();
    let mut content = Vec::with_capacity(pending.len());
    let mut tool_calls = Vec::new();
    for block in pending {
        match block {
            PendingContent::Text(text) => content.push(ContentBlock::text(text)?),
            PendingContent::Thinking(text) => content.push(ContentBlock::thinking(text)?),
            PendingContent::Tool {
                tool_call_id,
                completed: Some(call),
            } => {
                let tool_name = call.tool_name().to_owned();
                let arguments = call.arguments().clone();
                content.push(ContentBlock::tool_call_with_provider_id(
                    tool_call_id,
                    call.provider_call_id().as_str(),
                    tool_name.clone(),
                    arguments.clone(),
                )?);
                tool_calls.push(CompletedToolCall {
                    tool_call_id,
                    tool_name,
                    arguments,
                });
            }
            PendingContent::Tool {
                completed: None, ..
            } => return Err(model_contract("model tool call is incomplete")),
            PendingContent::HostedTool {
                tool_call_id,
                completed: Some(call),
            } => {
                let activity = HostedToolActivity::new(
                    tool_call_id,
                    call.provider_call_id().as_str(),
                    call.tool_name(),
                    call.arguments().clone(),
                    call.outcome().clone(),
                    call.sources().to_vec(),
                    call.continuation().cloned(),
                )?;
                content.push(ContentBlock::hosted_tool(activity));
            }
            PendingContent::HostedTool {
                completed: None, ..
            } => return Err(model_contract("model hosted tool activity is incomplete")),
            PendingContent::Citation(citation) => {
                content.push(ContentBlock::citation(canonical_citation(
                    &citation,
                    &hosted_ids,
                )?));
            }
        }
    }
    if content.is_empty() {
        return Err(model_contract("model completion contains no content"));
    }
    let has_tools = !tool_calls.is_empty();
    if has_tools != matches!(completion.stop_reason(), StopReason::ToolUse) {
        return Err(model_contract(
            "model stop reason does not match completed tool calls",
        ));
    }
    let message = CanonicalMessage::assistant(
        message_id,
        content,
        completion.stop_reason().clone(),
        timestamp,
    )?;
    Ok(ModelTurnOutput {
        message,
        tool_calls,
        completion,
        output_bytes,
    })
}

fn append_text(content: &mut Vec<PendingContent>, delta: &str, thinking: bool) -> usize {
    let last_matches = matches!(
        content.last(),
        Some(PendingContent::Thinking(_)) if thinking
    ) || matches!(content.last(), Some(PendingContent::Text(_)) if !thinking);
    if last_matches {
        match content.last_mut() {
            Some(PendingContent::Text(text) | PendingContent::Thinking(text)) => {
                text.push_str(delta);
            }
            Some(
                PendingContent::Tool { .. }
                | PendingContent::HostedTool { .. }
                | PendingContent::Citation(_),
            )
            | None => {}
        }
        content.len() - 1
    } else {
        content.push(if thinking {
            PendingContent::Thinking(delta.to_owned())
        } else {
            PendingContent::Text(delta.to_owned())
        });
        content.len() - 1
    }
}

fn complete_tool(
    content: &mut [PendingContent],
    positions: &BTreeMap<ModelStreamIndex, usize>,
    call: ToolCallCompleted,
) -> Result<(), KernelError> {
    let position = positions
        .get(&call.index())
        .copied()
        .ok_or_else(|| model_contract("completed tool call has no start"))?;
    match content.get_mut(position) {
        Some(PendingContent::Tool { completed, .. }) if completed.is_none() => {
            *completed = Some(call);
            Ok(())
        }
        Some(
            PendingContent::Text(_)
            | PendingContent::Thinking(_)
            | PendingContent::Tool { .. }
            | PendingContent::HostedTool { .. }
            | PendingContent::Citation(_),
        )
        | None => Err(model_contract("completed tool call has invalid position")),
    }
}

fn complete_hosted_tool(
    content: &mut [PendingContent],
    positions: &BTreeMap<ModelStreamIndex, usize>,
    call: HostedToolCompleted,
) -> Result<(), KernelError> {
    let position = positions
        .get(&call.index())
        .copied()
        .ok_or_else(|| model_contract("completed hosted tool has no start"))?;
    match content.get_mut(position) {
        Some(PendingContent::HostedTool { completed, .. }) if completed.is_none() => {
            *completed = Some(call);
            Ok(())
        }
        Some(
            PendingContent::Text(_)
            | PendingContent::Thinking(_)
            | PendingContent::Tool { .. }
            | PendingContent::HostedTool { .. }
            | PendingContent::Citation(_),
        )
        | None => Err(model_contract("completed hosted tool has invalid position")),
    }
}

fn hosted_block_bytes(
    content: &[PendingContent],
    positions: &BTreeMap<ModelStreamIndex, usize>,
    call: &HostedToolCompleted,
) -> Result<usize, KernelError> {
    let position = positions
        .get(&call.index())
        .copied()
        .ok_or_else(|| model_contract("completed hosted tool has no start"))?;
    let Some(PendingContent::HostedTool { tool_call_id, .. }) = content.get(position) else {
        return Err(model_contract("hosted tool has invalid position"));
    };
    let activity = HostedToolActivity::new(
        *tool_call_id,
        call.provider_call_id().as_str(),
        call.tool_name(),
        call.arguments().clone(),
        call.outcome().clone(),
        call.sources().to_vec(),
        call.continuation().cloned(),
    )?;
    serde_json::to_vec(&ContentBlock::hosted_tool(activity))
        .map(|bytes| bytes.len())
        .map_err(|_| model_contract("hosted tool activity could not be measured"))
}

fn canonical_citation(
    citation: &ModelSourceCitation,
    hosted_ids: &BTreeMap<ProviderToolCallId, ToolCallId>,
) -> Result<SourceCitation, KernelError> {
    let mut canonical = citation.citation().clone();
    if let Some(provider_call_id) = citation.provider_call_id() {
        let tool_call_id = hosted_ids
            .get(provider_call_id)
            .copied()
            .ok_or_else(|| model_contract("citation references an unknown hosted tool"))?;
        canonical = canonical.with_tool_call_id(tool_call_id);
    }
    Ok(canonical)
}

fn add_output(current: usize, delta: usize, output_budget: usize) -> Result<usize, KernelError> {
    let total = current.checked_add(delta).ok_or_else(|| {
        KernelError::new(
            KernelErrorCode::LimitExceeded,
            "assistant output byte count overflowed",
        )
    })?;
    if total > output_budget {
        return Err(KernelError::new(
            KernelErrorCode::LimitExceeded,
            "assistant output byte limit was reached",
        ));
    }
    Ok(total)
}

fn to_content_index(value: usize) -> Result<u32, KernelError> {
    u32::try_from(value).map_err(|_| model_contract("assistant content index is out of range"))
}

fn model_contract(message: &'static str) -> KernelError {
    KernelError::new(KernelErrorCode::ModelFailure, message)
}

impl From<tea_protocol::ContentValidationError> for KernelError {
    fn from(error: tea_protocol::ContentValidationError) -> Self {
        Self::new(KernelErrorCode::ModelFailure, error.to_string())
    }
}

impl From<tea_protocol::MessageValidationError> for KernelError {
    fn from(error: tea_protocol::MessageValidationError) -> Self {
        Self::new(KernelErrorCode::ModelFailure, error.to_string())
    }
}

impl From<tea_protocol::ExternalContentError> for KernelError {
    fn from(error: tea_protocol::ExternalContentError) -> Self {
        Self::new(KernelErrorCode::ModelFailure, error.to_string())
    }
}
