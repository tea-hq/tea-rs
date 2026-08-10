//! Versioned secret-free coding settings and deterministic layer merging.

mod file;
mod merge;
mod process;
mod providers;
mod types;

pub use crate::mcp_config::{
    McpArgumentResourceSettings, McpConcurrencySettings, McpLifecycleSettings, McpLimitsSettings,
    McpReconnectSettings, McpRetrySafetySettings, McpServerSettings, McpToolDeclarationSettings,
    McpToolSettings, McpTransportSettings,
};
pub use file::{MAX_SETTINGS_FILE_BYTES, load_settings_file, persist_global_model_settings};
pub use merge::merge_settings;
pub use process::{
    ConfiguredCodingProvider, ProcessCodingConfiguration, ResolvedOpenAiProvider,
    resolve_openai_compatible_provider,
};
pub use providers::{
    HostedToolCapability, MAX_PROVIDERS_FILE_BYTES, ModelCapabilitiesConfig, ModelDefinition,
    ModelReasoningConfig, ProviderConfig, ProviderValueResolver, ProvidersConfig,
    ProvidersConfigLoad, ProvidersConfigLoadError, load_providers_file,
};
pub use types::{
    CODING_SETTINGS_SCHEMA_VERSION, ClientWebSearchSettings, ClientWebSearchSettingsLayer,
    CodingSettings, DEFAULT_WEB_SEARCH_API_KEY_ENVIRONMENT, DEFAULT_WEB_SEARCH_TIMEOUT_MILLIS,
    MAX_WEB_SEARCH_API_KEY_ENVIRONMENT_BYTES, ProjectTrustDefault, ResourceSettings,
    ResourceSettingsLayer, SettingsLayer, TuiSettings, TuiSettingsLayer, WebFetchBackend,
    WebFetchCacheSettings, WebFetchCacheSettingsLayer, WebFetchSettings, WebFetchSettingsLayer,
    WebSearchClientBackend, WebSearchLocationSettings, WebSearchLocationSettingsLayer,
    WebSearchRoutePreference, WebSearchSettings, WebSearchSettingsLayer,
};

pub(crate) use merge::validate;
