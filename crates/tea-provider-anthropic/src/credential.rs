//! Credential resolution for the Anthropic Messages adapter.

use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

use tea_model::ProviderId;
use tea_protocol::ModelId;

use crate::error::{AnthropicError, AnthropicErrorCode};

/// Default provider identity advertised by this adapter.
pub const PROVIDER_ID: &str = "anthropic";
/// Stable basic web-search server tool used when no adapter override is set.
pub const DEFAULT_WEB_SEARCH_TOOL_TYPE: &str = "web_search_20250305";
/// Default maximum number of provider-hosted searches in one model request.
pub const DEFAULT_WEB_SEARCH_MAX_USES: u16 = 5;

/// Anthropic-specific controls for the versioned hosted web-search tool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnthropicWebSearchConfig {
    tool_type: String,
    max_uses: u16,
}

impl AnthropicWebSearchConfig {
    /// Creates validated adapter-owned hosted web-search controls.
    ///
    /// # Errors
    ///
    /// Returns an error unless the type is `web_search_YYYYMMDD` and the
    /// per-request use limit is between 1 and 100.
    pub fn new(tool_type: impl Into<String>, max_uses: u16) -> Result<Self, AnthropicError> {
        let tool_type = tool_type.into();
        let version = tool_type.strip_prefix("web_search_");
        if !version.is_some_and(|value| {
            value.len() == 8 && value.bytes().all(|byte| byte.is_ascii_digit())
        }) || !(1..=100).contains(&max_uses)
        {
            return Err(AnthropicError::new(
                AnthropicErrorCode::InvalidRequest,
                "Anthropic web search configuration is invalid",
            ));
        }
        Ok(Self {
            tool_type,
            max_uses,
        })
    }

    /// Returns the versioned Anthropic server-tool type.
    #[must_use]
    pub fn tool_type(&self) -> &str {
        &self.tool_type
    }

    /// Returns the maximum provider-hosted searches per request.
    #[must_use]
    pub const fn max_uses(&self) -> u16 {
        self.max_uses
    }
}

/// Bounded API key value that never appears in debug output.
#[derive(Clone, PartialEq, Eq)]
pub struct ApiKey(String);

impl ApiKey {
    /// Creates a bounded non-empty key.
    ///
    /// # Errors
    ///
    /// Returns an error for empty, oversized, or null-containing values.
    pub fn new(value: impl Into<String>) -> Result<Self, AnthropicError> {
        let value = value.into();
        if value.is_empty() || value.len() > 512 || value.contains('\0') {
            return Err(AnthropicError::new(
                AnthropicErrorCode::Authentication,
                "api key is invalid",
            ));
        }
        Ok(Self(value))
    }

    /// Returns the raw key value (caller must not log it).
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Debug for ApiKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("ApiKey(**REDACTED**)")
    }
}

/// Immutable Anthropic Messages connection configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnthropicConfig {
    provider_id: ProviderId,
    base_url: String,
    api_key: ApiKey,
    api_version: String,
    model_id: ModelId,
    timeout_millis: u64,
    web_search: AnthropicWebSearchConfig,
    final_json_schema: bool,
}

impl AnthropicConfig {
    /// Returns the provider identity.
    #[must_use]
    pub fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }

    /// Returns the base API URL (without `/v1/messages`).
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Returns the API key.
    #[must_use]
    pub fn api_key(&self) -> &ApiKey {
        &self.api_key
    }

    /// Returns the selected Anthropic API version.
    #[must_use]
    pub fn api_version(&self) -> &str {
        &self.api_version
    }

    /// Returns the configured default model identifier.
    #[must_use]
    pub fn model_id(&self) -> &ModelId {
        &self.model_id
    }

    /// Returns the per-request timeout in milliseconds.
    #[must_use]
    pub const fn timeout_millis(&self) -> u64 {
        self.timeout_millis
    }

    /// Returns adapter-owned hosted web-search controls.
    #[must_use]
    pub const fn web_search(&self) -> &AnthropicWebSearchConfig {
        &self.web_search
    }

    /// Returns whether the configured model and endpoint support native final JSON Schema output.
    #[must_use]
    pub const fn supports_final_json_schema(&self) -> bool {
        self.final_json_schema
    }
}

/// Object-safe port that resolves connection configuration at request time.
pub trait CredentialResolver: fmt::Debug + Send + Sync {
    /// Resolves connection configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when required configuration is missing or invalid.
    fn resolve(&self) -> Result<AnthropicConfig, AnthropicError>;
}

/// Resolves [`AnthropicConfig`] from the `TEA_ANTHROPIC_*` environment contract.
#[derive(Debug, Clone, Copy, Default)]
pub struct EnvCredentialResolver;

impl EnvCredentialResolver {
    /// Creates an environment-backed resolver.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl CredentialResolver for EnvCredentialResolver {
    fn resolve(&self) -> Result<AnthropicConfig, AnthropicError> {
        resolve_config(|key| std::env::var(key).ok())
    }
}

/// Resolves configuration from injected values without exposing them in debug output.
#[derive(Clone, Default)]
pub struct MapCredentialResolver {
    values: BTreeMap<String, String>,
}

impl MapCredentialResolver {
    /// Creates a resolver backed by supplied values.
    #[must_use]
    pub fn new(values: BTreeMap<String, String>) -> Self {
        Self { values }
    }
}

impl fmt::Debug for MapCredentialResolver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MapCredentialResolver")
            .field("values", &"**REDACTED**")
            .finish()
    }
}

impl CredentialResolver for MapCredentialResolver {
    fn resolve(&self) -> Result<AnthropicConfig, AnthropicError> {
        resolve_config(|key| self.values.get(key).cloned())
    }
}

fn resolve_config(get: impl Fn(&str) -> Option<String>) -> Result<AnthropicConfig, AnthropicError> {
    let api_key = get("TEA_ANTHROPIC_API_KEY")
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            AnthropicError::new(
                AnthropicErrorCode::Authentication,
                "TEA_ANTHROPIC_API_KEY is not set",
            )
        })?;
    let model = get("TEA_ANTHROPIC_MODEL")
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            AnthropicError::new(
                AnthropicErrorCode::Authentication,
                "TEA_ANTHROPIC_MODEL is not set",
            )
        })?;
    let model_id = ModelId::from_str(&model).map_err(|_| {
        AnthropicError::new(
            AnthropicErrorCode::InvalidRequest,
            "Anthropic model is invalid",
        )
    })?;
    let base_url = get("TEA_ANTHROPIC_BASE_URL")
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "https://api.anthropic.com".to_owned());
    let api_version = get("TEA_ANTHROPIC_API_VERSION")
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "2023-06-01".to_owned());
    if api_version.len() > 64 || api_version.chars().any(char::is_control) {
        return Err(AnthropicError::new(
            AnthropicErrorCode::InvalidRequest,
            "Anthropic API version is invalid",
        ));
    }
    let timeout_millis = get("TEA_ANTHROPIC_REQUEST_TIMEOUT_MS")
        .and_then(|value| value.parse().ok())
        .filter(|value: &u64| *value > 0)
        .unwrap_or(60_000);
    let web_search_tool_type = get("TEA_ANTHROPIC_WEB_SEARCH_TOOL_TYPE")
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_WEB_SEARCH_TOOL_TYPE.to_owned());
    let web_search_max_uses =
        match get("TEA_ANTHROPIC_WEB_SEARCH_MAX_USES").filter(|value| !value.is_empty()) {
            Some(value) => value.parse().map_err(|_| {
                AnthropicError::new(
                    AnthropicErrorCode::InvalidRequest,
                    "Anthropic web search configuration is invalid",
                )
            })?,
            None => DEFAULT_WEB_SEARCH_MAX_USES,
        };
    let final_json_schema = get("TEA_ANTHROPIC_FINAL_JSON_SCHEMA")
        .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"));
    Ok(AnthropicConfig {
        provider_id: ProviderId::from_str(PROVIDER_ID).expect("provider id is canonical"),
        base_url,
        api_key: ApiKey::new(api_key)?,
        api_version,
        model_id,
        timeout_millis,
        web_search: AnthropicWebSearchConfig::new(web_search_tool_type, web_search_max_uses)?,
        final_json_schema,
    })
}
