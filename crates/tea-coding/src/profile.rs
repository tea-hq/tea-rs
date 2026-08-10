use std::str::FromStr;
use std::time::Duration;

use tea_context::{
    BudgetBehavior, CacheScope, ContextProviderId, PromptAuthority, PromptModule, PromptModuleId,
    PromptPriority, PromptProvenance, PromptSegment, PromptSegmentId, StaticContextProvider,
    TrustLevel,
};
use tea_policy::{ExecutionSurface, PolicyEnvironment, PolicyExecutionTarget};
use tea_profile::{
    AgentProfile, ProfileDisplayName, ProfilePromptBudget, ProfileRuleId, ProfileRunLimits,
};
use tea_protocol::{ModelId, ModelRef, ProfileId, ProtocolMetadata, ProviderId};
use tea_tools::ToolName;

use crate::config::CodingSettings;
use crate::{CodingError, CodingErrorCode};

pub(crate) fn coding_profile(
    settings: &CodingSettings,
    execution_surface: ExecutionSurface,
) -> Result<AgentProfile, CodingError> {
    let mut builder = AgentProfile::builder(
        ProfileId::from_str("coding-agent").map_err(|_| invalid())?,
        ProfileDisplayName::new("Coding Agent").map_err(|_| invalid())?,
        ModelRef::new(
            ProviderId::from_str(&settings.provider).map_err(|_| invalid())?,
            ModelId::from_str(&settings.model).map_err(|_| invalid())?,
        ),
    )
    .prompt_budget(ProfilePromptBudget::new(128 * 1024, 32 * 1024).map_err(|_| invalid())?)
    .run_limits(
        ProfileRunLimits::new(64, Duration::from_mins(5), 4 * 1024 * 1024, 100_000, 64)
            .map_err(|_| invalid())?,
    )
    .environment(PolicyEnvironment::new(
        execution_surface,
        PolicyExecutionTarget::Native,
        ProtocolMetadata::default(),
    ))
    .approval_ttl(Duration::from_mins(10))
    .policy_rule(ProfileRuleId::from_str("product.coding_workspace").map_err(|_| invalid())?)
    .policy_rule(ProfileRuleId::from_str("product.coding_mcp").map_err(|_| invalid())?)
    .policy_rule(ProfileRuleId::from_str("platform.external_source").map_err(|_| invalid())?)
    .policy_rule(ProfileRuleId::from_str("platform.unknown_effect").map_err(|_| invalid())?);
    for tool in &settings.active_tools {
        builder = builder.active_tool(ToolName::from_str(tool).map_err(|_| invalid())?);
    }
    builder.build().map_err(|_| invalid())
}

pub(crate) fn coding_identity_provider() -> Result<StaticContextProvider, CodingError> {
    let provider_id =
        ContextProviderId::from_str("product.coding_identity").map_err(|_| invalid())?;
    let segment = PromptSegment::new(
        PromptSegmentId::from_str("product.coding.identity").map_err(|_| invalid())?,
        "You are a coding agent. Inspect relevant workspace context, make minimal verified changes, respect approvals, and report results concisely.",
        PromptProvenance::new(provider_id.clone(), "product_prompt", None)
            .map_err(|_| invalid())?,
        TrustLevel::Trusted,
        CacheScope::Profile,
        BudgetBehavior::Required,
    )
    .map_err(|_| invalid())?;
    let module = PromptModule::new(
        PromptModuleId::from_str("product.coding").map_err(|_| invalid())?,
        PromptAuthority::Product,
        PromptPriority::new(0),
        vec![segment],
    )
    .map_err(|_| invalid())?;
    Ok(StaticContextProvider::new(provider_id, vec![module]))
}

fn invalid() -> CodingError {
    CodingError::new(CodingErrorCode::InvalidInput, "coding profile is invalid")
}

#[cfg(test)]
mod tests {
    use super::coding_profile;
    use crate::config::CodingSettings;
    use tea_policy::ExecutionSurface;
    use tea_profile::ProfileRuleId;

    #[test]
    fn coding_profile_registers_external_source_policy_chain() {
        let profile = coding_profile(&CodingSettings::default(), ExecutionSurface::Cli).unwrap();
        let rule_ids = profile
            .policy_rule_ids()
            .iter()
            .map(ProfileRuleId::as_str)
            .collect::<Vec<_>>();
        assert_eq!(
            rule_ids,
            [
                "platform.external_source",
                "platform.unknown_effect",
                "product.coding_mcp",
                "product.coding_workspace",
            ]
        );
    }

    #[test]
    fn coding_profile_preserves_embedding_surface() {
        let profile =
            coding_profile(&CodingSettings::default(), ExecutionSurface::Desktop).unwrap();

        assert_eq!(profile.environment().surface(), ExecutionSurface::Desktop);
    }
}
