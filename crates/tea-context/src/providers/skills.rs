use std::str::FromStr;

use crate::{
    BudgetBehavior, CacheScope, ContextError, ContextErrorCode, ContextProvider,
    ContextProviderFuture, ContextProviderId, ContextRequest, PromptAuthority, PromptModule,
    PromptModuleId, PromptPriority, PromptProvenance, PromptSegment, PromptSegmentId,
    SkillMetadata, TrustLevel,
};

/// Maximum active skill metadata entries.
pub const MAX_ACTIVE_SKILLS: usize = 128;

/// Deterministic model-visible active skill metadata provider.
#[derive(Debug, Clone)]
pub struct SkillMetadataProvider {
    id: ContextProviderId,
    skills: Vec<SkillMetadata>,
}

impl SkillMetadataProvider {
    /// Creates a canonical active skill provider.
    ///
    /// # Errors
    ///
    /// Returns an error for too many or duplicate skill IDs.
    pub fn new(mut skills: Vec<SkillMetadata>) -> Result<Self, ContextError> {
        if skills.len() > MAX_ACTIVE_SKILLS {
            return Err(ContextError::new(
                ContextErrorCode::BoundsExceeded,
                "active skill collection is too large",
            ));
        }
        skills.sort_by(|left, right| left.id().cmp(right.id()));
        if skills
            .windows(2)
            .any(|items| items[0].id() == items[1].id())
        {
            return Err(ContextError::new(
                ContextErrorCode::DuplicateIdentity,
                "active skill ID is duplicated",
            ));
        }
        Ok(Self {
            id: ContextProviderId::from_str("builtin.skill_metadata").map_err(value_error)?,
            skills,
        })
    }
}

impl ContextProvider for SkillMetadataProvider {
    fn id(&self) -> &ContextProviderId {
        &self.id
    }

    fn provide(&self, _request: ContextRequest) -> ContextProviderFuture<'_> {
        let id = self.id.clone();
        let skills = self.skills.clone();
        Box::pin(async move {
            if skills.is_empty() {
                return Ok(Vec::new());
            }
            let segments = skills
                .into_iter()
                .map(|skill| {
                    let segment_id = format!("skill.{}.metadata", skill.id().as_str());
                    PromptSegment::new(
                        PromptSegmentId::from_str(&segment_id).map_err(value_error)?,
                        format!(
                            "Skill `{}`: {} Invoke explicitly with `{}`.",
                            skill.id(),
                            skill.description(),
                            skill.command()
                        ),
                        PromptProvenance::new(
                            id.clone(),
                            "skill_metadata",
                            Some(skill.id().to_string()),
                        )
                        .map_err(value_error)?,
                        TrustLevel::Delegated,
                        CacheScope::Profile,
                        BudgetBehavior::Omit,
                    )
                    .map_err(value_error)
                })
                .collect::<Result<Vec<_>, ContextError>>()?;
            Ok(vec![
                PromptModule::new(
                    PromptModuleId::from_str("skill.active_metadata").map_err(value_error)?,
                    PromptAuthority::Skill,
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
