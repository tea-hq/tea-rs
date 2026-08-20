use serde::{Deserialize, Serialize};
use tea_model::{ModelFailure, ModelFailureCode};
use thiserror::Error;

/// Stable machine-readable kernel failure classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KernelErrorCode {
    /// The requested model is absent or incompatible with the run snapshot.
    InvalidModel,
    /// Run configuration or immutable request construction failed.
    InvalidRequest,
    /// A runtime state transition is not legal.
    InvalidState,
    /// A model adapter stream violated its normalized contract.
    ModelFailure,
    /// Tool lookup, validation, or execution failed.
    ToolFailure,
    /// Policy or approval context could not be constructed safely.
    PolicyFailure,
    /// Durable session state could not be loaded or appended.
    SessionFailure,
    /// The awaited observation sink rejected an event.
    EventSinkFailure,
    /// The run was cooperatively cancelled.
    Cancelled,
    /// A deterministic run limit was reached.
    LimitExceeded,
    /// A required deterministic ID could not be produced.
    IdExhausted,
    /// The configured clock could not provide a canonical timestamp.
    ClockFailure,
    /// The compiled prompt, tools, and messages exceed the model context window.
    ContextOverflow,
    /// A retryable model request exhausted the configured retry policy.
    RetryExhausted,
    /// The tool scheduler could not place an invocation safely.
    SchedulerConflict,
}

/// Bounded safe failure returned by the agent kernel.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{code:?}: {message}")]
pub struct KernelError {
    code: KernelErrorCode,
    model_failure_code: Option<ModelFailureCode>,
    message: String,
    safe_diagnostic: bool,
}

impl KernelError {
    /// Creates a bounded English technical failure.
    #[must_use]
    pub fn new(code: KernelErrorCode, message: impl Into<String>) -> Self {
        let mut message = message.into();
        if message.is_empty() {
            "kernel operation failed".clone_into(&mut message);
        }
        if message.len() > 4096 {
            let boundary = message
                .char_indices()
                .map(|(index, _)| index)
                .take_while(|index| *index <= 4096)
                .last()
                .unwrap_or(0);
            message.truncate(boundary);
        }
        message.retain(|character| character != '\0');
        Self {
            code,
            model_failure_code: None,
            message,
            safe_diagnostic: false,
        }
    }

    pub(crate) fn model_failure(failure: &ModelFailure, retry_exhausted: bool) -> Self {
        let code = if failure.code() == ModelFailureCode::Cancelled {
            KernelErrorCode::Cancelled
        } else if retry_exhausted {
            KernelErrorCode::RetryExhausted
        } else {
            KernelErrorCode::ModelFailure
        };
        let message = if failure.code() == ModelFailureCode::Cancelled {
            "model request was cancelled".to_owned()
        } else if failure.is_safe_diagnostic() && retry_exhausted {
            format!("model retry policy was exhausted: {}", failure.message())
        } else if failure.is_safe_diagnostic() {
            failure.message().to_owned()
        } else if retry_exhausted {
            "model retry policy was exhausted".to_owned()
        } else {
            "model provider request failed".to_owned()
        };
        let mut error = Self::new(code, message);
        error.model_failure_code = Some(failure.code());
        error.safe_diagnostic =
            failure.is_safe_diagnostic() && failure.code() != ModelFailureCode::Cancelled;
        error
    }

    /// Returns the stable machine-readable code.
    #[must_use]
    pub const fn code(&self) -> KernelErrorCode {
        self.code
    }

    /// Returns the provider-neutral model failure classification, when the
    /// kernel error originated from a terminal model failure.
    #[must_use]
    pub const fn model_failure_code(&self) -> Option<ModelFailureCode> {
        self.model_failure_code
    }

    /// Returns the bounded safe diagnostic.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }

    /// Returns whether the message is safe to expose as provider diagnostics.
    #[must_use]
    pub const fn is_safe_diagnostic(&self) -> bool {
        self.safe_diagnostic
    }
}

#[cfg(test)]
mod tests {
    use tea_protocol::RetryClass;

    use super::*;

    #[test]
    fn model_failure_preserves_classification_and_safe_diagnostic() {
        let failure = ModelFailure::safe(
            ModelFailureCode::RateLimited,
            "HTTP 429: request quota exceeded",
            RetryClass::AfterBackoff,
        )
        .unwrap();

        let error = KernelError::model_failure(&failure, true);

        assert_eq!(error.code(), KernelErrorCode::RetryExhausted);
        assert_eq!(
            error.model_failure_code(),
            Some(ModelFailureCode::RateLimited)
        );
        assert!(error.is_safe_diagnostic());
        assert_eq!(
            error.message(),
            "model retry policy was exhausted: HTTP 429: request quota exceeded"
        );
    }

    #[test]
    fn unsafe_model_failure_keeps_code_but_redacts_diagnostic() {
        let failure = ModelFailure::new(
            ModelFailureCode::Authentication,
            "sk-secret-must-not-cross-the-kernel-boundary",
            RetryClass::Never,
        )
        .unwrap();

        let error = KernelError::model_failure(&failure, false);

        assert_eq!(
            error.model_failure_code(),
            Some(ModelFailureCode::Authentication)
        );
        assert!(!error.is_safe_diagnostic());
        assert_eq!(error.message(), "model provider request failed");
    }
}
