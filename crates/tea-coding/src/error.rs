use serde::{Deserialize, Serialize};
use tea_model::ModelFailureCode;
use thiserror::Error;

/// Stable coding-product failure classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodingErrorCode {
    /// Configuration or resource input was malformed or exceeded bounds.
    InvalidInput,
    /// A required path or resource was absent.
    NotFound,
    /// Project-local input was not trusted for the requested mode.
    ProjectNotTrusted,
    /// Persistent state could not be read or committed.
    Persistence,
    /// Credential resolution failed without exposing the credential.
    Credential,
    /// Provider credentials were rejected during model execution.
    Authentication,
    /// Provider credentials are valid but the operation is not permitted.
    PermissionDenied,
    /// The model provider rate-limited the request.
    RateLimited,
    /// The prompt and requested output exceed the model context window.
    ContextOverflow,
    /// The provider or selected model is temporarily unavailable.
    Unavailable,
    /// A network or transport operation failed.
    Transport,
    /// A provider-neutral model request was rejected as invalid.
    InvalidRequest,
    /// Policy denied or could not authorize an operation.
    PolicyDenied,
    /// An owned run was cancelled.
    Cancelled,
    /// Runtime assembly or execution failed.
    Runtime,
    /// The failure has no safer or more specific public classification.
    Internal,
}

impl CodingErrorCode {
    /// All stable coding-product failure codes.
    pub const ALL: [Self; 16] = [
        Self::InvalidInput,
        Self::NotFound,
        Self::ProjectNotTrusted,
        Self::Persistence,
        Self::Credential,
        Self::Authentication,
        Self::PermissionDenied,
        Self::RateLimited,
        Self::ContextOverflow,
        Self::Unavailable,
        Self::Transport,
        Self::InvalidRequest,
        Self::PolicyDenied,
        Self::Cancelled,
        Self::Runtime,
        Self::Internal,
    ];

    const fn from_model_failure(code: ModelFailureCode) -> Self {
        match code {
            ModelFailureCode::RateLimited => Self::RateLimited,
            ModelFailureCode::Authentication => Self::Authentication,
            ModelFailureCode::PermissionDenied => Self::PermissionDenied,
            ModelFailureCode::ContextOverflow => Self::ContextOverflow,
            ModelFailureCode::Unavailable => Self::Unavailable,
            ModelFailureCode::Transport => Self::Transport,
            ModelFailureCode::InvalidRequest => Self::InvalidRequest,
            ModelFailureCode::Cancelled => Self::Cancelled,
            ModelFailureCode::MalformedResponse | ModelFailureCode::Internal => Self::Internal,
        }
    }
}

/// Bounded path- and secret-independent coding-product error.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{code:?}: {message}")]
pub struct CodingError {
    code: CodingErrorCode,
    message: String,
}

impl CodingError {
    pub(crate) fn new(code: CodingErrorCode, message: impl Into<String>) -> Self {
        let mut message = message.into();
        if message.is_empty() {
            "coding operation failed".clone_into(&mut message);
        }
        message.retain(|character| character != '\0');
        if message.len() > 512 {
            let boundary = message
                .char_indices()
                .map(|(index, _)| index)
                .take_while(|index| *index <= 512)
                .last()
                .unwrap_or(0);
            message.truncate(boundary);
        }
        Self { code, message }
    }

    /// Returns the stable machine-readable classification.
    #[must_use]
    pub const fn code(&self) -> CodingErrorCode {
        self.code
    }

    /// Returns the bounded safe diagnostic.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl From<tea::RuntimeError> for CodingError {
    fn from(error: tea::RuntimeError) -> Self {
        let model_failure = error.model_failure_code();
        let code = model_failure.map_or_else(
            || match error.code() {
                tea::RuntimeErrorCode::ProviderFailure => CodingErrorCode::Internal,
                tea::RuntimeErrorCode::PolicyFailure => CodingErrorCode::PolicyDenied,
                tea::RuntimeErrorCode::Cancelled => CodingErrorCode::Cancelled,
                tea::RuntimeErrorCode::UnknownProvider => CodingErrorCode::Unavailable,
                tea::RuntimeErrorCode::InvalidRequest
                | tea::RuntimeErrorCode::UnknownProfile
                | tea::RuntimeErrorCode::UnknownModel
                | tea::RuntimeErrorCode::UnknownTool
                | tea::RuntimeErrorCode::UnknownPolicyRule => CodingErrorCode::InvalidInput,
                tea::RuntimeErrorCode::SessionFailure => CodingErrorCode::Persistence,
                tea::RuntimeErrorCode::ContextOverflow => CodingErrorCode::ContextOverflow,
                _ => CodingErrorCode::Runtime,
            },
            CodingErrorCode::from_model_failure,
        );
        let provider_origin =
            model_failure.is_some() || error.code() == tea::RuntimeErrorCode::ProviderFailure;
        let message = match code {
            CodingErrorCode::Cancelled => "coding operation was cancelled",
            _ if provider_origin && error.is_safe_diagnostic() => error.message(),
            _ if provider_origin => "model provider operation failed",
            CodingErrorCode::Persistence => "session persistence operation failed",
            CodingErrorCode::Runtime => "coding runtime operation failed",
            CodingErrorCode::Internal => "coding operation failed internally",
            CodingErrorCode::ContextOverflow => "model context window was exceeded",
            CodingErrorCode::PolicyDenied
            | CodingErrorCode::Authentication
            | CodingErrorCode::PermissionDenied
            | CodingErrorCode::RateLimited
            | CodingErrorCode::Unavailable
            | CodingErrorCode::Transport
            | CodingErrorCode::InvalidRequest
            | CodingErrorCode::Credential
            | CodingErrorCode::InvalidInput
            | CodingErrorCode::NotFound
            | CodingErrorCode::ProjectNotTrusted => error.message(),
        };
        Self::new(code, message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn untrusted_provider_runtime_messages_stay_generic() {
        let runtime = tea::RuntimeError::new(
            tea::RuntimeErrorCode::ProviderFailure,
            "sk-seeded-cli-credential-must-never-persist",
        );
        let coding = CodingError::from(runtime);
        assert_eq!(coding.message(), "model provider operation failed");
        assert_eq!(coding.code(), CodingErrorCode::Internal);
    }

    #[test]
    fn terminal_failure_codes_have_stable_snake_case_serialization() {
        let values = CodingErrorCode::ALL.map(|code| serde_json::to_value(code).unwrap());
        assert_eq!(
            values,
            [
                "invalid_input",
                "not_found",
                "project_not_trusted",
                "persistence",
                "credential",
                "authentication",
                "permission_denied",
                "rate_limited",
                "context_overflow",
                "unavailable",
                "transport",
                "invalid_request",
                "policy_denied",
                "cancelled",
                "runtime",
                "internal",
            ]
            .map(serde_json::Value::from)
        );
    }
}
