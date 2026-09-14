//! Default model catalog advertised by the `OpenAI` adapter.

use std::str::FromStr;

use tea_model::{
    HostedToolKind, ModelCapabilities, ModelDisplayName, ModelSpec, ProviderId, ReasoningEffort,
    ReasoningProfile,
};
use tea_protocol::{ModelId, TokenCount};

use crate::credential::OpenAiConfig;
use crate::error::{OpenAiError, OpenAiErrorCode};

/// Builds a catalog that always includes the configured smoke-test model plus a
/// small set of standard `OpenAI` models with documented limits.
///
/// # Errors
///
/// Returns an error when a catalog entry cannot be constructed.
pub fn default_catalog(config: &OpenAiConfig) -> Result<Vec<ModelSpec>, OpenAiError> {
    let provider_id = config.provider_id().clone();
    Ok(vec![spec(
        &provider_id,
        config.model_id(),
        "Configured model",
        128_000,
        16_384,
        true,
        reasoning_profile(config.model_id().as_str(), config.reasoning_effort())?,
        config.final_json_object(),
        config.final_json_schema(),
        config.final_json_schema_with_tools(),
        config.hosted_web_search(),
    )?])
}

#[allow(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
fn spec(
    provider_id: &ProviderId,
    model_id: &ModelId,
    display: &str,
    context: u64,
    output: u64,
    tools: bool,
    reasoning: Option<ReasoningProfile>,
    final_json_object: bool,
    final_json_schema: bool,
    final_json_schema_with_tools: bool,
    hosted_web_search: bool,
) -> Result<ModelSpec, OpenAiError> {
    let mut capabilities = ModelCapabilities::text();
    if tools {
        capabilities = capabilities.with_tools(true);
    }
    if reasoning.is_some() {
        capabilities = capabilities.with_reasoning();
    }
    if final_json_object {
        capabilities = capabilities.with_final_json_object();
    }
    if final_json_schema {
        capabilities = capabilities.with_final_json_schema();
    }
    if final_json_schema_with_tools {
        capabilities = capabilities.with_final_json_schema_with_tools();
    }
    if hosted_web_search {
        capabilities = capabilities.with_hosted_tool(HostedToolKind::WebSearch);
    }
    let spec = ModelSpec::new(
        model_id.clone(),
        provider_id.clone(),
        ModelDisplayName::from_str(display)
            .map_err(|_| OpenAiError::new(OpenAiErrorCode::Internal, "display name invalid"))?,
        TokenCount::new(context)
            .map_err(|_| OpenAiError::new(OpenAiErrorCode::Internal, "context window invalid"))?,
        TokenCount::new(output)
            .map_err(|_| OpenAiError::new(OpenAiErrorCode::Internal, "output limit invalid"))?,
        capabilities,
    )
    .map_err(|error| OpenAiError::new(OpenAiErrorCode::Internal, error.to_string()))?;
    Ok(reasoning.map_or(spec.clone(), |profile| spec.with_reasoning_profile(profile)))
}

fn reasoning_profile(
    model_id: &str,
    legacy_default: Option<&str>,
) -> Result<Option<ReasoningProfile>, OpenAiError> {
    let family = model_id.split('-').next().unwrap_or(model_id);
    let supported = if model_id.starts_with("gpt-5") {
        Some(vec![
            ReasoningEffort::Off,
            ReasoningEffort::Minimal,
            ReasoningEffort::Low,
            ReasoningEffort::Medium,
            ReasoningEffort::High,
        ])
    } else if matches!(family, "o1" | "o3" | "o4") {
        Some(vec![
            ReasoningEffort::Off,
            ReasoningEffort::Low,
            ReasoningEffort::Medium,
            ReasoningEffort::High,
        ])
    } else if legacy_default.is_some() {
        Some(ReasoningEffort::SHORTCUT_LEVELS.to_vec())
    } else {
        None
    };
    let Some(supported) = supported else {
        return Ok(None);
    };
    let requested = legacy_default
        .and_then(|value| value.parse::<ReasoningEffort>().ok())
        .unwrap_or(ReasoningEffort::Medium);
    let provisional = ReasoningProfile::new(ReasoningEffort::Medium, supported.clone())
        .map_err(|error| OpenAiError::new(OpenAiErrorCode::Internal, error.to_string()))?;
    let default_effort = provisional.resolve(requested).effective();
    ReasoningProfile::new(default_effort, supported)
        .map(Some)
        .map_err(|error| OpenAiError::new(OpenAiErrorCode::Internal, error.to_string()))
}
