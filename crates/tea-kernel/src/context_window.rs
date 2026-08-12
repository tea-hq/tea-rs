use tea_context::CompiledPrompt;
use tea_context::estimate_tokens;
use tea_model::{ModelSpec, ModelToolDefinition};
use tea_protocol::{CanonicalMessage, ContentBlock, TokenCount};

use crate::{KernelError, KernelErrorCode};

/// Conservative pre-turn context-window accountant.
///
/// Estimates input tokens from the compiled prompt (or legacy system prompt),
/// the active tool definitions, and committed message bytes using the context
/// crate's deterministic `ceil(utf8_bytes / 3)` estimator. Provider tokenization
/// remains outside the kernel; this is a fail-closed pre-turn guard that never
/// silently truncates.
pub(crate) struct ContextWindowAccountant {
    context_window: TokenCount,
    max_output: TokenCount,
}

impl ContextWindowAccountant {
    /// Creates an accountant from the advertised model limits.
    #[must_use]
    pub const fn new(spec: &ModelSpec) -> Self {
        Self {
            context_window: spec.context_window_tokens(),
            max_output: spec.max_output_tokens(),
        }
    }

    /// Returns the estimated input tokens for one turn request.
    ///
    /// `prompt` is the compiled prompt snapshot when present; `legacy_prompt`
    /// is the raw system prompt string used when no compiled prompt is attached.
    #[must_use]
    pub fn estimate_input_tokens(
        prompt: Option<&CompiledPrompt>,
        legacy_prompt: Option<&str>,
        tools: &[ModelToolDefinition],
        messages: &[CanonicalMessage],
    ) -> usize {
        let prompt_tokens = prompt.map_or_else(
            || legacy_prompt.map_or(0, |text| estimate_tokens(text.len())),
            CompiledPrompt::estimated_tokens,
        );
        let mut other_bytes = 0usize;
        for tool in tools {
            other_bytes = other_bytes.saturating_add(tool.name().len());
            other_bytes = other_bytes.saturating_add(tool.description().len());
            other_bytes = other_bytes
                .saturating_add(serde_json::to_vec(tool.input_schema()).map_or(0, |vec| vec.len()));
        }
        for message in messages {
            other_bytes = other_bytes.saturating_add(message_text_bytes(message));
        }
        prompt_tokens.saturating_add(estimate_tokens(other_bytes))
    }

    /// Returns the advertised context-window limit.
    #[must_use]
    pub const fn context_window(self) -> tea_protocol::TokenCount {
        self.context_window
    }

    /// Returns a `ContextOverflow` error when the estimated input plus the
    /// reserved output budget exceeds the model context window.
    pub fn check_overflow(&self, estimated_input_tokens: usize) -> Result<(), KernelError> {
        let reserved = estimated_input_tokens
            .saturating_add(usize::try_from(self.max_output.get()).unwrap_or(usize::MAX));
        if reserved > usize::try_from(self.context_window.get()).unwrap_or(usize::MAX) {
            return Err(KernelError::new(
                KernelErrorCode::ContextOverflow,
                "estimated input tokens exceed the model context window",
            ));
        }
        Ok(())
    }
}

fn message_text_bytes(message: &CanonicalMessage) -> usize {
    match message {
        CanonicalMessage::User { content, .. }
        | CanonicalMessage::Assistant { content, .. }
        | CanonicalMessage::ToolResult { content, .. } => {
            content.iter().map(content_block_bytes).sum()
        }
    }
}

fn content_block_bytes(block: &ContentBlock) -> usize {
    match block {
        ContentBlock::Text { text }
        | ContentBlock::ContextualText { text }
        | ContentBlock::Thinking { text } => text.len(),
        ContentBlock::ToolCall { arguments, .. } => {
            serde_json::to_vec(arguments).map_or(0, |vec| vec.len())
        }
        ContentBlock::HostedTool { .. } | ContentBlock::Citation { .. } => {
            serde_json::to_vec(block).map_or(0, |vec| vec.len())
        }
        ContentBlock::Image { .. } => 0,
    }
}
