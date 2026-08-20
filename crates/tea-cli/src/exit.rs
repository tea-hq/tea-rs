use std::fmt;
use std::process::ExitCode;

use tea_coding::CodingErrorCode;

const MAX_DIAGNOSTIC_BYTES: usize = 512;

/// Stable CLI process exit categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ExitCategory {
    /// Successful completion.
    Success = 0,
    /// Invalid command line or prompt input.
    Usage = 2,
    /// Configuration, state path, or project trust failure.
    TrustOrConfig = 3,
    /// Provider credentials, transport, or model failure.
    Provider = 4,
    /// Policy denial or unresolved approval.
    PolicyDenied = 5,
    /// User-requested cancellation.
    Cancelled = 6,
    /// Unexpected internal/persistence failure.
    Internal = 70,
}

impl ExitCategory {
    /// Returns the stable process exit code.
    #[must_use]
    pub fn exit_code(self) -> ExitCode {
        ExitCode::from(self as u8)
    }
}

/// Bounded secret-independent CLI failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliFailure {
    category: ExitCategory,
    message: String,
}

impl CliFailure {
    /// Creates a usage failure.
    #[must_use]
    pub fn usage(message: impl Into<String>) -> Self {
        Self::new(ExitCategory::Usage, message)
    }

    /// Creates a bounded failure in one stable category.
    #[must_use]
    pub fn new(category: ExitCategory, message: impl Into<String>) -> Self {
        let mut message = message.into();
        message.retain(|character| character != '\0' && character != '\r');
        if message.is_empty() {
            "tea operation failed".clone_into(&mut message);
        }
        if message.len() > MAX_DIAGNOSTIC_BYTES {
            let boundary = message
                .char_indices()
                .map(|(index, _)| index)
                .take_while(|index| *index <= MAX_DIAGNOSTIC_BYTES)
                .last()
                .unwrap_or(0);
            message.truncate(boundary);
        }
        Self { category, message }
    }

    /// Returns the stable category.
    #[must_use]
    pub const fn category(&self) -> ExitCategory {
        self.category
    }

    /// Returns the bounded diagnostic text.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for CliFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for CliFailure {}

impl From<tea_coding::CodingError> for CliFailure {
    fn from(error: tea_coding::CodingError) -> Self {
        let category = match error.code() {
            CodingErrorCode::InvalidInput
            | CodingErrorCode::NotFound
            | CodingErrorCode::ProjectNotTrusted => ExitCategory::TrustOrConfig,
            CodingErrorCode::Credential
            | CodingErrorCode::Authentication
            | CodingErrorCode::PermissionDenied
            | CodingErrorCode::RateLimited
            | CodingErrorCode::ContextOverflow
            | CodingErrorCode::Unavailable
            | CodingErrorCode::Transport
            | CodingErrorCode::InvalidRequest => ExitCategory::Provider,
            CodingErrorCode::PolicyDenied => ExitCategory::PolicyDenied,
            CodingErrorCode::Cancelled => ExitCategory::Cancelled,
            CodingErrorCode::Persistence | CodingErrorCode::Runtime | CodingErrorCode::Internal => {
                ExitCategory::Internal
            }
        };
        Self::new(category, error.message())
    }
}
