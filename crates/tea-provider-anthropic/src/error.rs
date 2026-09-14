//! Stable failure classification for the Anthropic adapter.

use tea_model::ModelFailureCode;
use thiserror::Error;

/// Stable machine-readable failure classification for the Anthropic adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnthropicErrorCode {
    /// The request payload was invalid.
    InvalidRequest,
    /// Credentials are missing or rejected.
    Authentication,
    /// Credentials are valid but the operation is not permitted.
    PermissionDenied,
    /// The provider rate-limited the request.
    RateLimited,
    /// The provider or model is temporarily unavailable.
    Unavailable,
    /// A network or transport error occurred.
    Transport,
    /// The provider response could not be normalized safely.
    MalformedResponse,
    /// The prompt and output exceed the model context window.
    ContextOverflow,
    /// The operation was cooperatively cancelled.
    Cancelled,
    /// An unexpected adapter failure.
    Internal,
}

impl AnthropicErrorCode {
    pub(crate) const fn into_model_failure_code(self) -> ModelFailureCode {
        match self {
            Self::Authentication => ModelFailureCode::Authentication,
            Self::PermissionDenied => ModelFailureCode::PermissionDenied,
            Self::RateLimited => ModelFailureCode::RateLimited,
            Self::Unavailable => ModelFailureCode::Unavailable,
            Self::Transport => ModelFailureCode::Transport,
            Self::InvalidRequest => ModelFailureCode::InvalidRequest,
            Self::MalformedResponse => ModelFailureCode::MalformedResponse,
            Self::ContextOverflow => ModelFailureCode::ContextOverflow,
            Self::Cancelled => ModelFailureCode::Cancelled,
            Self::Internal => ModelFailureCode::Internal,
        }
    }
}

/// Bounded safe failure returned by the Anthropic adapter.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{code:?}: {message}")]
pub struct AnthropicError {
    code: AnthropicErrorCode,
    message: String,
}

impl AnthropicError {
    /// Creates a bounded English technical failure.
    #[must_use]
    pub fn new(code: AnthropicErrorCode, message: impl Into<String>) -> Self {
        let mut message = message.into();
        if message.is_empty() {
            "anthropic adapter operation failed".clone_into(&mut message);
        }
        message.retain(|character| character != '\0');
        if message.len() > 4096 {
            let boundary = message
                .char_indices()
                .map(|(index, _)| index)
                .take_while(|index| *index <= 4096)
                .last()
                .unwrap_or(0);
            message.truncate(boundary);
        }
        Self { code, message }
    }

    /// Returns the stable classification.
    #[must_use]
    pub const fn code(&self) -> AnthropicErrorCode {
        self.code
    }

    /// Returns the bounded safe diagnostic (never the API key).
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}
