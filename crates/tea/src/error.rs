use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Stable machine-readable runtime failure classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeErrorCode {
    /// A request value was invalid or malformed.
    InvalidRequest,
    /// The runtime or session was in an invalid internal state.
    InvalidState,
    /// A referenced profile selector is not registered.
    UnknownProfile,
    /// A referenced model-provider identity is not registered.
    UnknownProvider,
    /// A referenced model selector is not advertised by the provider.
    UnknownModel,
    /// A referenced tool selector is not registered.
    UnknownTool,
    /// A referenced policy rule is not registered.
    UnknownPolicyRule,
    /// A run is already active on the target session.
    RunAlreadyActive,
    /// No run is active on the target session.
    NoActiveRun,
    /// A session store operation failed.
    SessionFailure,
    /// A policy evaluation or approval resolution failed.
    PolicyFailure,
    /// A kernel operation failed outside a more specific public category.
    KernelFailure,
    /// A model provider request or retry sequence failed.
    ProviderFailure,
    /// The compiled prompt, tools, and messages exceed the model context window.
    ContextOverflow,
    /// An owned run was cooperatively cancelled.
    Cancelled,
    /// The command is not supported by this runtime version.
    UnsupportedCommand,
    /// An event subscriber channel closed during emission.
    EventSinkClosed,
    /// A runtime value exceeded a deterministic bound.
    BoundsExceeded,
    /// A profile, tool, or rule identity was registered more than once.
    DuplicateEntry,
}

/// Bounded safe runtime failure.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{code:?}: {message}")]
pub struct RuntimeError {
    code: RuntimeErrorCode,
    model_failure_code: Option<tea_model::ModelFailureCode>,
    message: String,
    safe_diagnostic: bool,
}

impl RuntimeError {
    /// Creates a bounded English technical runtime error.
    #[must_use]
    pub fn new(code: RuntimeErrorCode, message: impl Into<String>) -> Self {
        let mut message = message.into();
        if message.is_empty() {
            "runtime operation failed".clone_into(&mut message);
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
        Self {
            code,
            model_failure_code: None,
            message,
            safe_diagnostic: false,
        }
    }

    /// Returns the stable classification.
    #[must_use]
    pub const fn code(&self) -> RuntimeErrorCode {
        self.code
    }

    /// Returns the provider-neutral model failure classification, when this
    /// runtime error originated from a terminal model request failure.
    #[must_use]
    pub const fn model_failure_code(&self) -> Option<tea_model::ModelFailureCode> {
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

impl From<tea_kernel::KernelError> for RuntimeError {
    fn from(error: tea_kernel::KernelError) -> Self {
        let code = match error.code() {
            tea_kernel::KernelErrorCode::ModelFailure
            | tea_kernel::KernelErrorCode::RetryExhausted => RuntimeErrorCode::ProviderFailure,
            tea_kernel::KernelErrorCode::ContextOverflow => RuntimeErrorCode::ContextOverflow,
            tea_kernel::KernelErrorCode::PolicyFailure => RuntimeErrorCode::PolicyFailure,
            tea_kernel::KernelErrorCode::Cancelled => RuntimeErrorCode::Cancelled,
            _ => RuntimeErrorCode::KernelFailure,
        };
        let mut runtime = Self::new(code, error.message().to_owned());
        runtime.model_failure_code = error.model_failure_code();
        runtime.safe_diagnostic = error.is_safe_diagnostic();
        runtime
    }
}

impl From<tea_session::SessionStoreError> for RuntimeError {
    fn from(error: tea_session::SessionStoreError) -> Self {
        Self::new(RuntimeErrorCode::SessionFailure, error.to_string())
    }
}

impl From<tea_profile::ProfileError> for RuntimeError {
    fn from(error: tea_profile::ProfileError) -> Self {
        Self::new(RuntimeErrorCode::InvalidRequest, error.message().to_owned())
    }
}

impl From<tea_context::ContextError> for RuntimeError {
    fn from(error: tea_context::ContextError) -> Self {
        Self::new(RuntimeErrorCode::KernelFailure, error.message().to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preflight_context_overflow_keeps_its_runtime_classification() {
        let kernel = tea_kernel::KernelError::new(
            tea_kernel::KernelErrorCode::ContextOverflow,
            "model context window was exceeded",
        );

        let runtime = RuntimeError::from(kernel);

        assert_eq!(runtime.code(), RuntimeErrorCode::ContextOverflow);
        assert_eq!(runtime.model_failure_code(), None);
    }
}
