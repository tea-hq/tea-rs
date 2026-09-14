//! Credential resolution for the OpenAI-compatible adapter.
//!
//! The adapter stores no secrets. A [`CredentialResolver`] returns a bounded
//! [`ApiKey`] and connection config at request time. The crate provides an
//! [`EnvCredentialResolver`] reading the committed `TEA_OPENAI_*` env
//! contract; a test-only `.env` loader (no `dotenv` dependency) populates the
//! process env for the live smoke test.

use std::collections::BTreeMap;
use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use tea_model::ProviderId;
use tea_protocol::ModelId;

use crate::error::{OpenAiError, OpenAiErrorCode};

/// Default provider identity advertised by this adapter.
pub const PROVIDER_ID: &str = "openai";

const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1";

/// `OpenAI` HTTP API used for model requests.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum OpenAiApiMode {
    /// The legacy `/chat/completions` endpoint.
    #[default]
    ChatCompletions,
    /// The `/responses` endpoint.
    Responses,
}

/// Exact structured-output dialect supported by an OpenAI-compatible endpoint.
///
/// Non-default endpoints configure this value explicitly. The built-in resolver
/// supplies [`OpenAiCompatibilityProfile::OpenAi`] only for the exact default
/// `OpenAI` base URL; it never infers a profile from provider or model names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum OpenAiCompatibilityProfile {
    /// Native `OpenAI` Chat Completions and Responses semantics.
    OpenAi,
    /// Azure `OpenAI` v1 Chat Completions and Responses semantics.
    AzureOpenAi,
    /// xAI Chat Completions and Responses semantics.
    Xai,
    /// `DeepSeek` Chat Completions and Responses semantics.
    DeepSeek,
    /// Gemini's documented `OpenAI` Chat Completions compatibility endpoint.
    GeminiOpenAi,
    /// A local Ollama `OpenAI` Chat Completions compatibility endpoint.
    OllamaLocal,
    /// `OpenRouter` Chat Completions with parameter-aware route selection.
    OpenRouter,
    /// Groq Chat Completions and beta Responses semantics.
    Groq,
    /// Mistral Chat Completions semantics.
    Mistral,
    /// Together AI Chat Completions semantics.
    Together,
    /// A vLLM `OpenAI` Chat Completions and Responses server.
    Vllm,
}

impl OpenAiCompatibilityProfile {
    /// Returns whether the profile documents this API mode.
    #[must_use]
    pub const fn supports_api_mode(self, api_mode: OpenAiApiMode) -> bool {
        match api_mode {
            OpenAiApiMode::ChatCompletions => true,
            OpenAiApiMode::Responses => {
                matches!(
                    self,
                    Self::OpenAi
                        | Self::AzureOpenAi
                        | Self::Xai
                        | Self::DeepSeek
                        | Self::Groq
                        | Self::Vllm
                )
            }
        }
    }

    /// Returns whether the profile supports JSON-object final output in this mode.
    #[must_use]
    pub const fn supports_final_json_object(self, api_mode: OpenAiApiMode) -> bool {
        match api_mode {
            OpenAiApiMode::ChatCompletions => true,
            OpenAiApiMode::Responses => {
                matches!(
                    self,
                    Self::OpenAi | Self::AzureOpenAi | Self::Xai | Self::DeepSeek | Self::Vllm
                )
            }
        }
    }

    /// Returns whether the profile supports JSON-Schema final output in this mode.
    #[must_use]
    pub const fn supports_final_json_schema(self, api_mode: OpenAiApiMode) -> bool {
        match api_mode {
            OpenAiApiMode::ChatCompletions => !matches!(self, Self::DeepSeek),
            OpenAiApiMode::Responses => {
                matches!(
                    self,
                    Self::OpenAi | Self::AzureOpenAi | Self::Xai | Self::DeepSeek | Self::Vllm
                )
            }
        }
    }

    /// Returns whether JSON Schema may be combined with tools in this mode.
    ///
    /// A model still has to advertise the corresponding combination capability.
    #[must_use]
    pub const fn supports_final_json_schema_with_tools(self, api_mode: OpenAiApiMode) -> bool {
        self.supports_final_json_schema(api_mode)
            && !matches!(self, Self::Groq)
            && !matches!((self, api_mode), (Self::Vllm, OpenAiApiMode::Responses))
    }

    /// Returns whether JSON Schema output uses an SSE response in this mode.
    #[must_use]
    pub const fn supports_streaming_final_json_schema(self, api_mode: OpenAiApiMode) -> bool {
        self.supports_final_json_schema(api_mode) && !matches!(self, Self::Groq)
    }

    /// Returns whether structured output must select only parameter-capable routes.
    #[must_use]
    pub const fn requires_parameter_support(self) -> bool {
        matches!(self, Self::OpenRouter)
    }

    /// Returns whether JSON Schema may be combined with parallel function calls.
    #[must_use]
    pub const fn supports_parallel_tools_with_json_schema(self) -> bool {
        !matches!(self, Self::AzureOpenAi | Self::Groq)
    }

    pub(crate) const fn uses_strict_json_schema(self, api_mode: OpenAiApiMode) -> bool {
        !matches!((self, api_mode), (Self::DeepSeek, OpenAiApiMode::Responses))
    }
}

impl FromStr for OpenAiCompatibilityProfile {
    type Err = OpenAiError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "open-ai" => Ok(Self::OpenAi),
            "azure-open-ai" => Ok(Self::AzureOpenAi),
            "xai" => Ok(Self::Xai),
            "deep-seek" => Ok(Self::DeepSeek),
            "gemini-open-ai" => Ok(Self::GeminiOpenAi),
            "ollama-local" => Ok(Self::OllamaLocal),
            "open-router" => Ok(Self::OpenRouter),
            "groq" => Ok(Self::Groq),
            "mistral" => Ok(Self::Mistral),
            "together" => Ok(Self::Together),
            "vllm" => Ok(Self::Vllm),
            _ => Err(OpenAiError::new(
                OpenAiErrorCode::InvalidRequest,
                "TEA_OPENAI_COMPATIBILITY_PROFILE is invalid",
            )),
        }
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
    /// Returns an error for empty or oversized values.
    pub fn new(value: impl Into<String>) -> Result<Self, OpenAiError> {
        let value = value.into();
        if value.is_empty() || value.len() > 512 || value.contains('\0') {
            return Err(OpenAiError::new(
                OpenAiErrorCode::Authentication,
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

impl FromStr for ApiKey {
    type Err = OpenAiError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        Self::new(value)
    }
}

/// Immutable OpenAI-compatible connection configuration.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::struct_excessive_bools)]
pub struct OpenAiConfig {
    provider_id: ProviderId,
    model_id: ModelId,
    base_url: String,
    api_key: ApiKey,
    api_key_header: String,
    api_key_prefix: String,
    api_mode: OpenAiApiMode,
    compatibility_profile: Option<OpenAiCompatibilityProfile>,
    org_id: Option<String>,
    project_id: Option<String>,
    reasoning_effort: Option<String>,
    vision: bool,
    final_json_object: bool,
    final_json_schema: bool,
    final_json_schema_with_tools: bool,
    hosted_web_search: bool,
    timeout_millis: u64,
}

impl OpenAiConfig {
    /// Creates an explicit OpenAI-compatible connection configuration.
    ///
    /// The configuration uses the default provider identity, bearer-token
    /// authentication, Chat Completions API mode, no compatibility profile,
    /// and a 60-second timeout. Callers enabling structured output must select
    /// an explicit profile with [`Self::with_compatibility_profile`].
    ///
    /// # Errors
    ///
    /// Returns an error when the base URL is empty or contains a null byte.
    pub fn new(
        model_id: ModelId,
        base_url: impl Into<String>,
        api_key: ApiKey,
    ) -> Result<Self, OpenAiError> {
        let base_url = base_url.into();
        if base_url.is_empty() || base_url.contains('\0') {
            return Err(OpenAiError::new(
                OpenAiErrorCode::InvalidRequest,
                "base URL is invalid",
            ));
        }
        Ok(Self {
            provider_id: default_provider_id(),
            model_id,
            base_url,
            api_key,
            api_key_header: "Authorization".to_owned(),
            api_key_prefix: "Bearer ".to_owned(),
            api_mode: OpenAiApiMode::ChatCompletions,
            compatibility_profile: None,
            org_id: None,
            project_id: None,
            reasoning_effort: None,
            vision: false,
            final_json_object: false,
            final_json_schema: false,
            final_json_schema_with_tools: false,
            hosted_web_search: false,
            timeout_millis: 60_000,
        })
    }

    /// Returns the provider identity.
    #[must_use]
    pub fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }
    /// Returns the configured model used for catalog initialization.
    #[must_use]
    pub const fn model_id(&self) -> &ModelId {
        &self.model_id
    }
    /// Returns the base URL (without `/chat/completions`).
    #[must_use]
    pub fn base_url(&self) -> &str {
        &self.base_url
    }
    /// Returns the API key.
    #[must_use]
    pub fn api_key(&self) -> &ApiKey {
        &self.api_key
    }
    /// Returns the header name carrying the key.
    #[must_use]
    pub fn api_key_header(&self) -> &str {
        &self.api_key_header
    }
    /// Returns the prefix prepended to the key value.
    #[must_use]
    pub fn api_key_prefix(&self) -> &str {
        &self.api_key_prefix
    }
    /// Returns the configured `OpenAI` API mode.
    #[must_use]
    pub const fn api_mode(&self) -> OpenAiApiMode {
        self.api_mode
    }
    /// Overrides the `OpenAI` API mode.
    #[must_use]
    pub const fn with_api_mode(mut self, api_mode: OpenAiApiMode) -> Self {
        self.api_mode = api_mode;
        self
    }
    /// Returns the configured structured-output compatibility profile.
    #[must_use]
    pub const fn compatibility_profile(&self) -> Option<OpenAiCompatibilityProfile> {
        self.compatibility_profile
    }
    /// Selects an explicit structured-output compatibility profile.
    #[must_use]
    pub const fn with_compatibility_profile(mut self, profile: OpenAiCompatibilityProfile) -> Self {
        self.compatibility_profile = Some(profile);
        self
    }
    /// Returns the optional organization id.
    #[must_use]
    pub fn org_id(&self) -> Option<&str> {
        self.org_id.as_deref()
    }
    /// Returns the optional project id.
    #[must_use]
    pub fn project_id(&self) -> Option<&str> {
        self.project_id.as_deref()
    }
    /// Returns the optional reasoning effort.
    #[must_use]
    pub fn reasoning_effort(&self) -> Option<&str> {
        self.reasoning_effort.as_deref()
    }
    /// Returns whether image input is enabled for the smoke test.
    #[must_use]
    pub const fn vision(&self) -> bool {
        self.vision
    }
    /// Returns whether the configured model explicitly supports JSON-object output.
    #[must_use]
    pub const fn final_json_object(&self) -> bool {
        self.final_json_object
    }
    /// Explicitly enables or disables JSON-object output for the configured model.
    #[must_use]
    pub const fn with_final_json_object(mut self, enabled: bool) -> Self {
        self.final_json_object = enabled;
        self
    }
    /// Returns whether the configured model explicitly supports JSON Schema output.
    #[must_use]
    pub const fn final_json_schema(&self) -> bool {
        self.final_json_schema
    }
    /// Explicitly enables or disables JSON Schema output for the configured model.
    #[must_use]
    pub const fn with_final_json_schema(mut self, enabled: bool) -> Self {
        self.final_json_schema = enabled;
        if !enabled {
            self.final_json_schema_with_tools = false;
        }
        self
    }
    /// Returns whether the configured model explicitly supports JSON Schema with tools.
    #[must_use]
    pub const fn final_json_schema_with_tools(&self) -> bool {
        self.final_json_schema_with_tools
    }
    /// Explicitly enables or disables JSON Schema output combined with tools.
    #[must_use]
    pub const fn with_final_json_schema_with_tools(mut self, enabled: bool) -> Self {
        self.final_json_schema_with_tools = enabled;
        if enabled {
            self.final_json_schema = true;
        }
        self
    }
    /// Returns whether the configured model explicitly supports Responses hosted web search.
    #[must_use]
    pub const fn hosted_web_search(&self) -> bool {
        self.hosted_web_search
    }
    /// Explicitly enables or disables Responses hosted web search for the configured model.
    #[must_use]
    pub const fn with_hosted_web_search(mut self, enabled: bool) -> Self {
        self.hosted_web_search = enabled;
        self
    }
    /// Returns the per-request timeout in milliseconds.
    #[must_use]
    pub const fn timeout_millis(&self) -> u64 {
        self.timeout_millis
    }
}

/// Object-safe port that resolves connection configuration at request time.
///
/// Implementations must not retain secrets beyond the returned config.
pub trait CredentialResolver: fmt::Debug + Send + Sync {
    /// Resolves the connection configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when required configuration (api key, base url) is
    /// missing or invalid.
    fn resolve(&self) -> Result<OpenAiConfig, OpenAiError>;
}

/// Resolves [`OpenAiConfig`] from the `TEA_OPENAI_*` environment contract.
#[derive(Debug, Clone, Copy, Default)]
pub struct EnvCredentialResolver;

impl EnvCredentialResolver {
    /// Creates the env-backed resolver.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl CredentialResolver for EnvCredentialResolver {
    fn resolve(&self) -> Result<OpenAiConfig, OpenAiError> {
        resolve_config(
            default_provider_id(),
            Some(OpenAiCompatibilityProfile::OpenAi),
            |key| std::env::var(key).ok(),
        )
    }
}

/// Resolves [`OpenAiConfig`] from a supplied key-value map (e.g. a parsed
/// `.env` file). Used by the live smoke test to avoid mutating the process
/// environment (and thus avoid `unsafe`).
#[derive(Clone)]
pub struct MapCredentialResolver {
    provider_id: ProviderId,
    compatibility_profile: Option<OpenAiCompatibilityProfile>,
    values: BTreeMap<String, String>,
}

impl MapCredentialResolver {
    /// Creates a resolver backed by the supplied map.
    #[must_use]
    pub fn new(values: BTreeMap<String, String>) -> Self {
        Self {
            provider_id: default_provider_id(),
            compatibility_profile: Some(OpenAiCompatibilityProfile::OpenAi),
            values,
        }
    }

    /// Creates a resolver for one custom OpenAI-compatible provider identity.
    #[must_use]
    pub fn for_provider(provider_id: ProviderId, values: BTreeMap<String, String>) -> Self {
        Self {
            provider_id,
            compatibility_profile: None,
            values,
        }
    }
}

impl Default for MapCredentialResolver {
    fn default() -> Self {
        Self::new(BTreeMap::new())
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
    fn resolve(&self) -> Result<OpenAiConfig, OpenAiError> {
        resolve_config(
            self.provider_id.clone(),
            self.compatibility_profile,
            |key| self.values.get(key).cloned(),
        )
    }
}

/// Shared configuration builder parameterized by a value lookup.
fn resolve_config(
    provider_id: ProviderId,
    default_compatibility_profile: Option<OpenAiCompatibilityProfile>,
    get: impl Fn(&str) -> Option<String>,
) -> Result<OpenAiConfig, OpenAiError> {
    let api_key = get("TEA_OPENAI_API_KEY")
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            OpenAiError::new(
                OpenAiErrorCode::Authentication,
                "TEA_OPENAI_API_KEY is not set",
            )
        })?;
    let model_id = get("TEA_OPENAI_MODEL")
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            OpenAiError::new(
                OpenAiErrorCode::Authentication,
                "TEA_OPENAI_MODEL is not set",
            )
        })?
        .parse::<ModelId>()
        .map_err(|_| {
            OpenAiError::new(
                OpenAiErrorCode::InvalidRequest,
                "TEA_OPENAI_MODEL is invalid",
            )
        })?;
    let base_url = get("TEA_OPENAI_BASE_URL")
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_OPENAI_BASE_URL.to_owned());
    let api_key_header = get("TEA_OPENAI_API_KEY_HEADER")
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "Authorization".to_owned());
    let api_key_prefix = get("TEA_OPENAI_API_KEY_PREFIX").map_or_else(
        || "Bearer ".to_owned(),
        |value| {
            if value == "__NONE__" {
                String::new()
            } else {
                value
            }
        },
    );
    let api_mode = match get("TEA_OPENAI_API_MODE").as_deref() {
        None | Some("" | "chat-completions") => OpenAiApiMode::ChatCompletions,
        Some("responses") => OpenAiApiMode::Responses,
        Some(_) => {
            return Err(OpenAiError::new(
                OpenAiErrorCode::InvalidRequest,
                "TEA_OPENAI_API_MODE must be chat-completions or responses",
            ));
        }
    };
    let compatibility_profile = get("TEA_OPENAI_COMPATIBILITY_PROFILE")
        .filter(|value| !value.is_empty())
        .map(|value| value.parse::<OpenAiCompatibilityProfile>())
        .transpose()?
        .or_else(|| {
            (base_url == DEFAULT_OPENAI_BASE_URL)
                .then_some(default_compatibility_profile)
                .flatten()
        });
    if compatibility_profile.is_some_and(|profile| !profile.supports_api_mode(api_mode)) {
        return Err(OpenAiError::new(
            OpenAiErrorCode::InvalidRequest,
            "OpenAI compatibility profile does not support the configured API mode",
        ));
    }
    let org_id = get("TEA_OPENAI_ORG_ID").filter(|value| !value.is_empty());
    let project_id = get("TEA_OPENAI_PROJECT_ID").filter(|value| !value.is_empty());
    let reasoning_effort = get("TEA_OPENAI_REASONING_EFFORT")
        .filter(|value| value.parse::<tea_protocol::ReasoningEffort>().is_ok());
    let vision = flag_enabled(get("TEA_OPENAI_VISION"));
    let final_json_object = flag_enabled(get("TEA_OPENAI_FINAL_JSON_OBJECT"));
    let final_json_schema_with_tools = flag_enabled(get("TEA_OPENAI_FINAL_JSON_SCHEMA_WITH_TOOLS"));
    let final_json_schema =
        final_json_schema_with_tools || flag_enabled(get("TEA_OPENAI_FINAL_JSON_SCHEMA"));
    let hosted_web_search = flag_enabled(get("TEA_OPENAI_HOSTED_WEB_SEARCH"));
    let timeout_millis = get("TEA_OPENAI_REQUEST_TIMEOUT_MS")
        .and_then(|value| value.parse().ok())
        .filter(|value: &u64| *value > 0)
        .unwrap_or(60_000);
    Ok(OpenAiConfig {
        provider_id,
        model_id,
        base_url,
        api_key: ApiKey::new(api_key)?,
        api_key_header,
        api_key_prefix,
        api_mode,
        compatibility_profile,
        org_id,
        project_id,
        reasoning_effort,
        vision,
        final_json_object,
        final_json_schema,
        final_json_schema_with_tools,
        hosted_web_search,
        timeout_millis,
    })
}

fn flag_enabled(value: Option<String>) -> bool {
    value.is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
}

fn default_provider_id() -> ProviderId {
    ProviderId::from_str(PROVIDER_ID).expect("provider id is canonical")
}
