use std::str::FromStr;
use std::time::Duration;

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
    has_skills: bool,
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
    for tool in settings
        .active_tools
        .iter()
        .filter(|tool| tool.as_str() != crate::skill_tool::READ_SKILL_RESOURCE_TOOL_NAME)
    {
        builder = builder.active_tool(ToolName::from_str(tool).map_err(|_| invalid())?);
    }
    if has_skills {
        builder = builder.active_tool(
            ToolName::from_str(crate::skill_tool::READ_SKILL_RESOURCE_TOOL_NAME)
                .map_err(|_| invalid())?,
        );
    }
    builder.build().map_err(|_| invalid())
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
        let profile =
            coding_profile(&CodingSettings::default(), ExecutionSurface::Cli, false).unwrap();
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
            coding_profile(&CodingSettings::default(), ExecutionSurface::Desktop, false).unwrap();

        assert_eq!(profile.environment().surface(), ExecutionSurface::Desktop);
    }

    #[test]
    fn skill_resource_activation_is_catalog_owned_and_deduplicated() {
        let settings = CodingSettings {
            active_tools: vec!["read_skill_resource".to_owned()],
            ..CodingSettings::default()
        };
        let empty = coding_profile(&settings, ExecutionSurface::Cli, false).unwrap();
        assert!(
            !empty
                .active_tool_names()
                .iter()
                .any(|name| name.as_str() == "read_skill_resource")
        );
        let active = coding_profile(&settings, ExecutionSurface::Cli, true).unwrap();
        assert_eq!(
            active
                .active_tool_names()
                .iter()
                .filter(|name| name.as_str() == "read_skill_resource")
                .count(),
            1
        );
    }
}
