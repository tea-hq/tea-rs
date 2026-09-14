//! Default model catalog advertised by the Anthropic adapter.

use std::str::FromStr;

use tea_model::{HostedToolKind, ModelCapabilities, ModelDisplayName, ModelSpec};
use tea_protocol::TokenCount;

use crate::credential::AnthropicConfig;
use crate::error::{AnthropicError, AnthropicErrorCode};

/// Builds the catalog for the model selected in [`AnthropicConfig`].
///
/// # Errors
///
/// Returns an error when the model contract cannot be constructed.
pub fn default_catalog(config: &AnthropicConfig) -> Result<Vec<ModelSpec>, AnthropicError> {
    let mut capabilities = ModelCapabilities::text()
        .with_image_input()
        .with_tools(true)
        .with_hosted_tool(HostedToolKind::WebSearch)
        .with_usage_reporting();
    if config.supports_final_json_schema() {
        capabilities = capabilities.with_final_json_schema_with_tools();
    }
    let model = ModelSpec::new(
        config.model_id().clone(),
        config.provider_id().clone(),
        ModelDisplayName::from_str("Configured Anthropic model").map_err(|_| {
            AnthropicError::new(AnthropicErrorCode::Internal, "display name is invalid")
        })?,
        TokenCount::new(200_000).map_err(|_| {
            AnthropicError::new(AnthropicErrorCode::Internal, "context window is invalid")
        })?,
        TokenCount::new(64_000).map_err(|_| {
            AnthropicError::new(AnthropicErrorCode::Internal, "output limit is invalid")
        })?,
        capabilities,
    )
    .map_err(|error| AnthropicError::new(AnthropicErrorCode::Internal, error.to_string()))?;
    Ok(vec![model])
}
