use std::str::FromStr;

use crate::{
    BudgetBehavior, CacheScope, ContextError, ContextProvider, ContextProviderFuture,
    ContextProviderId, ContextRequest, PromptAuthority, PromptModule, PromptModuleId,
    PromptPriority, PromptProvenance, PromptSegment, PromptSegmentId, TrustLevel,
};

/// Generates guidance from prompt hints on the active tool snapshot.
#[derive(Debug, Clone)]
pub struct ToolHintProvider {
    id: ContextProviderId,
}

impl ToolHintProvider {
    /// Creates the built-in active-tool provider.
    ///
    /// # Errors
    ///
    /// Returns an error only when the crate's static provider ID is invalid.
    pub fn new() -> Result<Self, crate::ContextIdentityError> {
        Ok(Self {
            id: ContextProviderId::from_str("builtin.tool_hints")?,
        })
    }

    /// Builds tool guidance modules from one active-tool snapshot.
    ///
    /// # Errors
    ///
    /// Returns an error for duplicate tool names or invalid prompt values.
    pub fn modules(
        &self,
        active_tools: &[tea_tools::ToolSpec],
    ) -> Result<Vec<PromptModule>, ContextError> {
        let mut active_tools = active_tools.iter().collect::<Vec<_>>();
        active_tools.sort_by(|left, right| left.name().cmp(right.name()));
        if active_tools
            .windows(2)
            .any(|tools| tools[0].name() == tools[1].name())
        {
            return Err(ContextError::new(
                crate::ContextErrorCode::DuplicateIdentity,
                "active tool guidance contains a duplicate tool name",
            ));
        }
        let snippets = active_tools
            .iter()
            .filter_map(|tool| tool.prompt_snippet().map(|snippet| (*tool, snippet)))
            .map(|(tool, snippet)| {
                let segment_id = format!("tool.{}.snippet", tool.name().as_str());
                let locator = format!("{}@{}", tool.name(), tool.version());
                PromptSegment::new(
                    PromptSegmentId::from_str(&segment_id).map_err(value_error)?,
                    format!("Tool `{}`: {snippet}", tool.name()),
                    PromptProvenance::new(self.id.clone(), "tool_spec", Some(locator))
                        .map_err(value_error)?,
                    TrustLevel::Delegated,
                    CacheScope::Profile,
                    BudgetBehavior::Omit,
                )
                .map_err(value_error)
            })
            .collect::<Result<Vec<_>, ContextError>>()?;
        let guidelines = active_tools
            .iter()
            .filter(|tool| !tool.prompt_guidelines().is_empty())
            .map(|tool| {
                let segment_id = format!("tool.{}.guidelines", tool.name().as_str());
                let locator = format!("{}@{}", tool.name(), tool.version());
                let guidelines = tool
                    .prompt_guidelines()
                    .iter()
                    .map(|guideline| format!("- {guideline}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                PromptSegment::new(
                    PromptSegmentId::from_str(&segment_id).map_err(value_error)?,
                    format!("Tool `{}` guidelines:\n{guidelines}", tool.name()),
                    PromptProvenance::new(self.id.clone(), "tool_spec", Some(locator))
                        .map_err(value_error)?,
                    TrustLevel::Delegated,
                    CacheScope::Profile,
                    BudgetBehavior::Omit,
                )
                .map_err(value_error)
            })
            .collect::<Result<Vec<_>, ContextError>>()?;
        let mut modules = Vec::new();
        if !snippets.is_empty() {
            modules.push(
                PromptModule::new(
                    PromptModuleId::from_str("tool.active_snippets").map_err(value_error)?,
                    PromptAuthority::Tool,
                    PromptPriority::new(1),
                    snippets,
                )
                .map_err(value_error)?,
            );
        }
        if !guidelines.is_empty() {
            modules.push(
                PromptModule::new(
                    PromptModuleId::from_str("tool.active_guidelines").map_err(value_error)?,
                    PromptAuthority::Tool,
                    PromptPriority::new(0),
                    guidelines,
                )
                .map_err(value_error)?,
            );
        }
        Ok(modules)
    }
}

impl ContextProvider for ToolHintProvider {
    fn id(&self) -> &ContextProviderId {
        &self.id
    }

    fn provide(&self, request: ContextRequest) -> ContextProviderFuture<'_> {
        let modules = self.modules(request.active_tools());
        Box::pin(async move { modules })
    }
}

fn value_error(error: impl std::fmt::Display) -> ContextError {
    ContextError::new(crate::ContextErrorCode::InvalidValue, error.to_string())
}
