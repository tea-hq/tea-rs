use std::str::FromStr;

use crate::{
    BudgetBehavior, CacheScope, ContextError, ContextErrorCode, ContextProvider,
    ContextProviderFuture, ContextProviderId, ContextRequest, PromptAuthority, PromptModule,
    PromptModuleId, PromptPriority, PromptProvenance, PromptSegment, PromptSegmentId, TrustLevel,
};

/// Maximum caller-supplied workspace instruction documents.
pub const MAX_WORKSPACE_INSTRUCTIONS: usize = 128;

/// One caller-loaded workspace instruction document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkspaceInstruction {
    id: PromptSegmentId,
    content: String,
    locator: String,
    trust: TrustLevel,
}

impl WorkspaceInstruction {
    /// Creates one explicit workspace document snapshot.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid content or source locator.
    pub fn new(
        id: PromptSegmentId,
        content: impl Into<String>,
        locator: impl Into<String>,
        trust: TrustLevel,
    ) -> Result<Self, ContextError> {
        let content = content.into();
        let locator = locator.into();
        if locator.is_empty() || locator.len() > 2048 || locator.chars().any(char::is_control) {
            return Err(ContextError::new(
                ContextErrorCode::InvalidValue,
                "workspace instruction locator is invalid",
            ));
        }
        // Reuse segment construction as the authoritative content bound.
        PromptSegment::new(
            id.clone(),
            content.clone(),
            PromptProvenance::new(
                ContextProviderId::from_str("builtin.workspace_instructions")
                    .map_err(value_error)?,
                "workspace_file",
                Some(locator.clone()),
            )
            .map_err(value_error)?,
            trust,
            CacheScope::Session,
            BudgetBehavior::Omit,
        )
        .map_err(value_error)?;
        Ok(Self {
            id,
            content,
            locator,
            trust,
        })
    }

    /// Returns the stable prompt segment identity.
    #[must_use]
    pub const fn id(&self) -> &PromptSegmentId {
        &self.id
    }

    /// Returns the caller-authored instruction body.
    #[must_use]
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Returns the caller-supplied logical source locator.
    #[must_use]
    pub fn locator(&self) -> &str {
        &self.locator
    }

    /// Returns the source trust classification.
    #[must_use]
    pub const fn trust(&self) -> TrustLevel {
        self.trust
    }
}

/// Provider over caller-supplied workspace instruction snapshots.
#[derive(Debug, Clone)]
pub struct WorkspaceInstructionProvider {
    id: ContextProviderId,
    instructions: Vec<WorkspaceInstruction>,
}

impl WorkspaceInstructionProvider {
    /// Creates a canonical instruction provider.
    ///
    /// # Errors
    ///
    /// Returns an error for too many or duplicate instruction identities.
    pub fn new(mut instructions: Vec<WorkspaceInstruction>) -> Result<Self, ContextError> {
        if instructions.len() > MAX_WORKSPACE_INSTRUCTIONS {
            return Err(ContextError::new(
                ContextErrorCode::BoundsExceeded,
                "workspace instruction collection is too large",
            ));
        }
        instructions.sort_by(|left, right| left.id.cmp(&right.id));
        if instructions
            .windows(2)
            .any(|items| items[0].id == items[1].id)
        {
            return Err(ContextError::new(
                ContextErrorCode::DuplicateIdentity,
                "workspace instruction ID is duplicated",
            ));
        }
        Ok(Self {
            id: ContextProviderId::from_str("builtin.workspace_instructions")
                .map_err(value_error)?,
            instructions,
        })
    }
}

impl ContextProvider for WorkspaceInstructionProvider {
    fn id(&self) -> &ContextProviderId {
        &self.id
    }

    fn provide(&self, _request: ContextRequest) -> ContextProviderFuture<'_> {
        let id = self.id.clone();
        let instructions = self.instructions.clone();
        Box::pin(async move {
            if instructions.is_empty() {
                return Ok(Vec::new());
            }
            let segments = instructions
                .into_iter()
                .map(|instruction| {
                    PromptSegment::new(
                        instruction.id,
                        instruction.content,
                        PromptProvenance::new(
                            id.clone(),
                            "workspace_file",
                            Some(instruction.locator),
                        )
                        .map_err(value_error)?,
                        instruction.trust,
                        CacheScope::Session,
                        BudgetBehavior::Omit,
                    )
                    .map_err(value_error)
                })
                .collect::<Result<Vec<_>, ContextError>>()?;
            Ok(vec![
                PromptModule::new(
                    PromptModuleId::from_str("workspace.instructions").map_err(value_error)?,
                    PromptAuthority::Workspace,
                    PromptPriority::new(0),
                    segments,
                )
                .map_err(value_error)?,
            ])
        })
    }
}

fn value_error(error: impl std::fmt::Display) -> ContextError {
    ContextError::new(ContextErrorCode::InvalidValue, error.to_string())
}
