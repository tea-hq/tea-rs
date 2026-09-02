use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;

use tea_model::{
    ModelCapabilities, ModelDisplayName, ModelProvider, ModelSpec, ProviderId, ReasoningProfile,
};
use tea_protocol::{ModelId, ReasoningEffort, TokenCount};
use tea_provider_openai::{MapCredentialResolver, OpenAiProviderBuilder, OpenAiReasoningEffortMap};

use super::{
    CodingSettings, ModelDefinition, ProviderConfig, ProviderValueResolver, load_providers_file,
    load_settings_file, merge_settings,
};
use crate::{CodingCredentialResolver, CodingError, CodingErrorCode};

/// Reads CLI-compatible global coding configuration for a process host.
///
/// The configuration directory follows the reference CLI's precedence:
/// `TEA_CONFIG_DIR`, then `$HOME/.tea`. Environment values are retained only
/// to resolve explicit provider-value templates such as `$MY_API_KEY`; they do
/// not override values declared in `settings.json`.
#[derive(Clone)]
pub struct ProcessCodingConfiguration {
    config_dir: PathBuf,
    environment: BTreeMap<String, String>,
}

impl std::fmt::Debug for ProcessCodingConfiguration {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProcessCodingConfiguration")
            .field("config_dir", &self.config_dir)
            .field("environment", &"**REDACTED**")
            .finish()
    }
}

impl ProcessCodingConfiguration {
    /// Captures process configuration roots and environment values once.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error when no absolute configuration directory
    /// can be determined.
    pub fn from_process() -> Result<Self, CodingError> {
        let environment = std::env::vars_os()
            .filter_map(|(key, value)| Some((key.into_string().ok()?, value.into_string().ok()?)))
            .collect::<BTreeMap<_, _>>();
        let config_dir = configuration_dir(&environment, process_home_dir())?;
        Self::new(config_dir, environment)
    }

    /// Creates a hermetic configuration source for embedding hosts and tests.
    ///
    /// # Errors
    ///
    /// Returns an invalid-input error when `config_dir` is not absolute.
    pub fn new(
        config_dir: impl Into<PathBuf>,
        environment: BTreeMap<String, String>,
    ) -> Result<Self, CodingError> {
        let config_dir = config_dir.into();
        if !config_dir.is_absolute() || config_dir.file_name().is_none() {
            return Err(invalid("configuration directory is invalid"));
        }
        Ok(Self {
            config_dir,
            environment,
        })
    }

    /// Returns the global configuration directory.
    #[must_use]
    pub fn config_dir(&self) -> &Path {
        &self.config_dir
    }

    /// Loads global settings and creates the selected OpenAI-compatible provider.
    ///
    /// `settings.json` determines the provider and model. `providers.json`
    /// supplies custom endpoint, credential, and catalog values. A provider
    /// configuration for `openai` may omit its model list to retain the
    /// adapter's built-in catalog; custom provider selectors must declare one.
    ///
    /// # Errors
    ///
    /// Returns typed failures for unreadable settings, invalid provider files,
    /// missing credentials, unsupported catalogs, or provider construction.
    pub fn load(&self) -> Result<ConfiguredCodingProvider, CodingError> {
        let configured = self.load_all()?;
        let provider = Arc::clone(
            configured
                .providers
                .first()
                .ok_or_else(|| invalid("provider configuration is empty"))?,
        );
        Ok(ConfiguredCodingProvider {
            settings: configured.settings,
            provider,
        })
    }

    /// Loads global settings and every explicitly configured provider.
    ///
    /// The provider selected by `settings.json` is always first. Remaining
    /// providers follow canonical provider-id order from `providers.json`.
    ///
    /// # Errors
    ///
    /// Returns typed failures when any configured provider cannot be resolved
    /// into one immutable provider generation.
    pub fn load_all(&self) -> Result<ConfiguredCodingProviders, CodingError> {
        self.load_all_with_provider_id(|provider_id| ProviderId::from_str(provider_id).ok())
    }

    /// Loads every provider after assigning its final runtime identity.
    ///
    /// The mapper runs before provider construction so every advertised
    /// [`ModelSpec`] is born with the same final identity as its provider. This
    /// is intended for hosts that namespace provider sources, such as
    /// `local.openai`, without teaching `tea-model` about those sources.
    ///
    /// # Errors
    ///
    /// Returns typed failures for configuration, identity mapping, duplicate
    /// mapped identities, credentials, catalogs, or provider construction.
    pub fn load_all_with_provider_id(
        &self,
        map_provider_id: impl Fn(&str) -> Option<ProviderId>,
    ) -> Result<ConfiguredCodingProviders, CodingError> {
        let settings = load_settings_file(&self.config_dir.join("settings.json"))?;
        let providers = load_providers_file(&self.config_dir.join("providers.json"));
        if providers.error.is_some() {
            return Err(invalid("provider configuration is invalid"));
        }
        let mut settings = merge_settings(
            CodingSettings::default(),
            settings.as_ref(),
            None,
            None,
            None,
        )?;
        let selected_source_id = settings.provider.clone();
        let selected_provider_id = map_provider_id(&selected_source_id)
            .ok_or_else(|| invalid("mapped provider identity is invalid"))?;
        let mut mapped_provider_ids = BTreeSet::from([selected_provider_id.clone()]);
        let resolved = resolve_openai_compatible_provider(
            selected_provider_id.as_str(),
            &settings.model,
            providers.config.providers.get(&selected_source_id),
            None,
            self.environment.clone(),
        )?;
        settings.provider = selected_provider_id.to_string();
        let mut resolved_providers = vec![resolved.build()?];
        for (provider_id, provider) in &providers.config.providers {
            if provider_id == &selected_source_id {
                continue;
            }
            let mapped_provider_id = map_provider_id(provider_id)
                .ok_or_else(|| invalid("mapped provider identity is invalid"))?;
            if !mapped_provider_ids.insert(mapped_provider_id.clone()) {
                return Err(invalid("mapped provider identity is duplicated"));
            }
            let model_id = provider
                .models
                .first()
                .map(|model| model.id.as_str())
                .ok_or_else(|| invalid("configured provider model catalog is empty"))?;
            let resolved = resolve_openai_compatible_provider(
                mapped_provider_id.as_str(),
                model_id,
                Some(provider),
                None,
                self.environment.clone(),
            )?;
            resolved_providers.push(resolved.build()?);
        }
        Ok(ConfiguredCodingProviders {
            settings,
            providers: resolved_providers,
        })
    }
}

/// Secret-safe settings and all providers loaded for one process generation.
#[derive(Clone)]
pub struct ConfiguredCodingProviders {
    settings: CodingSettings,
    providers: Vec<Arc<dyn ModelProvider>>,
}

impl std::fmt::Debug for ConfiguredCodingProviders {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ConfiguredCodingProviders")
            .field("settings", &self.settings)
            .field(
                "providers",
                &self
                    .providers
                    .iter()
                    .map(|provider| (provider.provider_id(), provider.models()))
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl ConfiguredCodingProviders {
    /// Returns the selected global settings.
    #[must_use]
    pub const fn settings(&self) -> &CodingSettings {
        &self.settings
    }

    /// Returns all initialized providers with the selected provider first.
    #[must_use]
    pub fn providers(&self) -> &[Arc<dyn ModelProvider>] {
        &self.providers
    }
}

/// Secret-free resolved coding settings paired with an initialized provider.
#[derive(Clone)]
pub struct ConfiguredCodingProvider {
    settings: CodingSettings,
    provider: Arc<dyn ModelProvider>,
}

impl std::fmt::Debug for ConfiguredCodingProvider {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ConfiguredCodingProvider")
            .field("settings", &self.settings)
            .field("provider_id", self.provider.provider_id())
            .field("models", &self.provider.models())
            .finish()
    }
}

impl ConfiguredCodingProvider {
    /// Returns the resolved, secret-free coding settings.
    #[must_use]
    pub const fn settings(&self) -> &CodingSettings {
        &self.settings
    }

    /// Returns the initialized model provider.
    #[must_use]
    pub fn provider(&self) -> Arc<dyn ModelProvider> {
        Arc::clone(&self.provider)
    }
}

/// Resolves an OpenAI-compatible provider from a configured or built-in source.
///
/// This is shared host assembly for the reference CLI and embedding products.
/// A supplied API key takes precedence over the configured provider key; it is
/// never stored in the returned value's debug representation.
///
/// # Errors
///
/// Returns typed failures for invalid selectors/catalogs, unresolved configured
/// credentials, or invalid provider connection values.
pub fn resolve_openai_compatible_provider(
    provider_id: &str,
    model_id: &str,
    provider: Option<&ProviderConfig>,
    api_key_override: Option<&str>,
    environment: BTreeMap<String, String>,
) -> Result<ResolvedOpenAiProvider, CodingError> {
    let provider_id =
        ProviderId::from_str(provider_id).map_err(|_| invalid("provider selector is invalid"))?;
    let process_environment = environment.clone();
    let mut values = environment;
    values.insert("TEA_OPENAI_MODEL".to_owned(), model_id.to_owned());

    let (catalog, reasoning_effort_maps) = if let Some(provider) = provider {
        // Custom providers are configuration-authoritative. Only an explicit
        // `$NAME` value may resolve through the captured process environment.
        values.retain(|key, _| !key.starts_with("TEA_OPENAI_"));
        values.insert("TEA_OPENAI_MODEL".to_owned(), model_id.to_owned());
        let value_resolver = ProviderValueResolver::new(process_environment);
        insert_provider_value(
            &mut values,
            "TEA_OPENAI_BASE_URL",
            provider.base_url.as_deref(),
        );
        insert_provider_value(
            &mut values,
            "TEA_OPENAI_API_KEY_HEADER",
            provider.api_key_header.as_deref(),
        );
        insert_provider_value(
            &mut values,
            "TEA_OPENAI_API_KEY_PREFIX",
            provider.api_key_prefix.as_deref(),
        );
        insert_provider_value(
            &mut values,
            "TEA_OPENAI_API_MODE",
            provider.api_mode.as_deref(),
        );
        insert_provider_value(&mut values, "TEA_OPENAI_ORG_ID", provider.org_id.as_deref());
        insert_provider_value(
            &mut values,
            "TEA_OPENAI_PROJECT_ID",
            provider.project_id.as_deref(),
        );
        insert_provider_value(
            &mut values,
            "TEA_OPENAI_REASONING_EFFORT",
            provider.reasoning_effort.as_deref(),
        );
        if let Some(vision) = provider.vision {
            values.insert("TEA_OPENAI_VISION".to_owned(), vision.to_string());
        }
        if let Some(timeout_millis) = provider.timeout_millis {
            values.insert(
                "TEA_OPENAI_REQUEST_TIMEOUT_MS".to_owned(),
                timeout_millis.to_string(),
            );
        }
        if let Some(api_key) = api_key_override {
            values.insert("TEA_OPENAI_API_KEY".to_owned(), api_key.to_owned());
        } else if let Some(api_key) = &provider.api_key {
            values.insert(
                "TEA_OPENAI_API_KEY".to_owned(),
                value_resolver.resolve(api_key).ok_or_else(|| {
                    credential("configured provider API key could not be resolved")
                })?,
            );
        }

        if provider.models.is_empty() {
            if provider_id.as_str() != tea_provider_openai::PROVIDER_ID {
                return Err(invalid("configured provider has no models"));
            }
            (None, BTreeMap::default())
        } else {
            let catalog = custom_model_catalog(&provider_id, provider)?;
            if !catalog
                .models
                .iter()
                .any(|model| model.model_id().as_str() == model_id)
            {
                return Err(invalid("model is not configured for provider"));
            }
            (Some(catalog.models), catalog.reasoning_effort_maps)
        }
    } else {
        if let Some(api_key) = api_key_override {
            values.insert("TEA_OPENAI_API_KEY".to_owned(), api_key.to_owned());
        }
        (None, BTreeMap::default())
    };

    let resolver = if provider_id.as_str() == tea_provider_openai::PROVIDER_ID {
        MapCredentialResolver::new(values)
    } else {
        MapCredentialResolver::for_provider(provider_id, values)
    };
    let config = CodingCredentialResolver::new(Arc::new(resolver)).resolve()?;
    Ok(ResolvedOpenAiProvider {
        config: Arc::new(config),
        catalog,
        reasoning_effort_maps,
    })
}

/// Resolved OpenAI-compatible connection values ready for host-specific assembly.
///
/// The type keeps connection secrets private while allowing the CLI to attach
/// its HTTP client policy before building a provider.
pub struct ResolvedOpenAiProvider {
    config: Arc<tea_provider_openai::OpenAiConfig>,
    catalog: Option<Vec<ModelSpec>>,
    reasoning_effort_maps: BTreeMap<ModelId, OpenAiReasoningEffortMap>,
}

impl std::fmt::Debug for ResolvedOpenAiProvider {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ResolvedOpenAiProvider")
            .field("provider_id", self.config.provider_id())
            .field("model_id", self.config.model_id())
            .field("catalog", &self.catalog)
            .field("reasoning_effort_maps", &self.reasoning_effort_maps)
            .finish()
    }
}

impl ResolvedOpenAiProvider {
    /// Returns a provider builder with the resolved connection and catalog.
    #[must_use]
    pub fn provider_builder(self) -> OpenAiProviderBuilder {
        let builder = OpenAiProviderBuilder::new()
            .with_config(self.config)
            .with_reasoning_effort_maps(self.reasoning_effort_maps);
        match self.catalog {
            Some(catalog) => builder.with_catalog(catalog),
            None => builder,
        }
    }

    /// Builds the OpenAI-compatible model provider with the default HTTP policy.
    ///
    /// # Errors
    ///
    /// Returns a provider failure when HTTP or catalog assembly fails.
    pub fn build(self) -> Result<Arc<dyn ModelProvider>, CodingError> {
        self.provider_builder()
            .build()
            .map(|provider| Arc::new(provider) as Arc<dyn ModelProvider>)
            .map_err(|_| provider_failure())
    }
}

fn custom_model_catalog(
    provider_id: &ProviderId,
    provider: &ProviderConfig,
) -> Result<CustomModelCatalog, CodingError> {
    let mut models = Vec::with_capacity(provider.models.len());
    let mut reasoning_effort_maps = BTreeMap::new();
    for model in &provider.models {
        let (spec, map) = custom_model_spec(provider_id, provider, model)?;
        if let Some(map) = map {
            reasoning_effort_maps.insert(spec.model_id().clone(), map);
        }
        models.push(spec);
    }
    Ok(CustomModelCatalog {
        models,
        reasoning_effort_maps,
    })
}

struct CustomModelCatalog {
    models: Vec<ModelSpec>,
    reasoning_effort_maps: BTreeMap<ModelId, OpenAiReasoningEffortMap>,
}

fn custom_model_spec(
    provider_id: &ProviderId,
    provider: &ProviderConfig,
    model: &ModelDefinition,
) -> Result<(ModelSpec, Option<OpenAiReasoningEffortMap>), CodingError> {
    let reasoning = custom_model_reasoning(provider, model)?;
    let mut capabilities = ModelCapabilities::text().with_tools(true);
    if provider.vision.unwrap_or(false) {
        capabilities = capabilities.with_image_input();
    }
    if reasoning.is_some() {
        capabilities = capabilities.with_reasoning();
    }
    for hosted_tool in &model.capabilities.hosted_tools {
        capabilities = capabilities.with_hosted_tool(hosted_tool.kind());
    }
    let spec = ModelSpec::new(
        ModelId::from_str(&model.id)
            .map_err(|_| invalid("configured model identifier is invalid"))?,
        provider_id.clone(),
        ModelDisplayName::from_str(model.display_name.as_deref().unwrap_or(&model.id))
            .map_err(|_| invalid("configured model name is invalid"))?,
        TokenCount::new(model.context_window_tokens.unwrap_or(128_000))
            .map_err(|_| invalid("configured model context window is invalid"))?,
        TokenCount::new(model.max_output_tokens.unwrap_or(16_384))
            .map_err(|_| invalid("configured model output limit is invalid"))?,
        capabilities,
    )
    .map_err(|_| invalid("configured model limits are invalid"))?;
    let Some((profile, map)) = reasoning else {
        return Ok((spec, None));
    };
    Ok((spec.with_reasoning_profile(profile), Some(map)))
}

fn custom_model_reasoning(
    provider: &ProviderConfig,
    model: &ModelDefinition,
) -> Result<Option<(ReasoningProfile, OpenAiReasoningEffortMap)>, CodingError> {
    if let Some(reasoning) = &model.capabilities.reasoning {
        let (profile, entries) = reasoning
            .resolved()
            .ok_or_else(|| invalid("configured model reasoning profile is invalid"))?;
        let map = OpenAiReasoningEffortMap::new(entries)
            .map_err(|_| invalid("configured model reasoning wire map is invalid"))?;
        return Ok(Some((profile, map)));
    }
    let Some(default_effort) = provider
        .reasoning_effort
        .as_deref()
        .and_then(|value| ReasoningEffort::from_str(value).ok())
    else {
        return Ok(None);
    };
    let provisional =
        ReasoningProfile::new(ReasoningEffort::Medium, ReasoningEffort::SHORTCUT_LEVELS)
            .map_err(|_| invalid("legacy reasoning default is invalid"))?;
    let default_effort = provisional.resolve(default_effort).effective();
    let profile = ReasoningProfile::new(default_effort, ReasoningEffort::SHORTCUT_LEVELS)
        .map_err(|_| invalid("legacy reasoning default is invalid"))?;
    let map = OpenAiReasoningEffortMap::new(
        ReasoningEffort::SHORTCUT_LEVELS
            .into_iter()
            .filter(|effort| *effort != ReasoningEffort::Off)
            .map(|effort| (effort, effort.as_str().to_owned())),
    )
    .map_err(|_| invalid("legacy reasoning wire map is invalid"))?;
    Ok(Some((profile, map)))
}

fn configuration_dir(
    environment: &BTreeMap<String, String>,
    home_dir: Option<PathBuf>,
) -> Result<PathBuf, CodingError> {
    environment
        .get("TEA_CONFIG_DIR")
        .map(PathBuf::from)
        .or_else(|| home_dir.map(|home| home.join(".tea")))
        .filter(|path| path.is_absolute())
        .ok_or_else(|| invalid("absolute configuration directory is required"))
}

fn process_home_dir() -> Option<PathBuf> {
    home_dir_from(|key| std::env::var_os(key))
}

fn home_dir_from(mut value: impl FnMut(&str) -> Option<OsString>) -> Option<PathBuf> {
    if let Some(home) = value("HOME").filter(|path| !path.is_empty()) {
        return Some(PathBuf::from(home));
    }
    if let Some(profile) = value("USERPROFILE").filter(|path| !path.is_empty()) {
        return Some(PathBuf::from(profile));
    }
    let drive = value("HOMEDRIVE").filter(|path| !path.is_empty())?;
    let path = value("HOMEPATH").filter(|path| !path.is_empty())?;
    let mut home = PathBuf::from(drive);
    home.push(path);
    Some(home)
}

fn insert_provider_value(values: &mut BTreeMap<String, String>, key: &str, value: Option<&str>) {
    if let Some(value) = value {
        values.insert(key.to_owned(), value.to_owned());
    }
}

fn invalid(message: &'static str) -> CodingError {
    CodingError::new(CodingErrorCode::InvalidInput, message)
}

fn credential(message: &'static str) -> CodingError {
    CodingError::new(CodingErrorCode::Credential, message)
}

fn provider_failure() -> CodingError {
    CodingError::new(CodingErrorCode::Internal, "provider configuration failed")
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn config_dir(label: &str) -> PathBuf {
        let directory = std::env::temp_dir().join(format!(
            "tea-coding-process-config-{label}-{}-{}",
            std::process::id(),
            TEST_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&directory).expect("configuration directory");
        directory
    }

    fn write_configuration(directory: &Path, api_key: &str) {
        fs::write(
            directory.join("settings.json"),
            r#"{"schemaVersion":1,"provider":"workshop-dev","model":"debug-model"}"#,
        )
        .expect("settings file");
        fs::write(
            directory.join("providers.json"),
            format!(
                r#"{{"providers":{{"workshop-dev":{{"base_url":"http://127.0.0.1:11434/v1","api_key":"{api_key}","models":[{{"id":"debug-model"}}]}}}}}}"#
            ),
        )
        .expect("providers file");
    }

    #[test]
    fn global_configuration_builds_a_custom_provider_without_environment_credentials() {
        let directory = config_dir("literal");
        write_configuration(&directory, "configured-secret");

        let configured = ProcessCodingConfiguration::new(&directory, BTreeMap::new())
            .unwrap()
            .load()
            .unwrap();

        assert_eq!(configured.settings().provider, "workshop-dev");
        assert_eq!(configured.settings().model, "debug-model");
        assert_eq!(configured.provider().provider_id().as_str(), "workshop-dev");
        assert_eq!(
            configured.provider().models()[0].model_id().as_str(),
            "debug-model"
        );
        assert!(!format!("{configured:?}").contains("configured-secret"));
        fs::remove_dir_all(directory).expect("temporary configuration cleanup");
    }

    #[test]
    fn global_configuration_builds_every_configured_provider_with_the_selected_one_first() {
        let directory = config_dir("multiple-providers");
        fs::write(
            directory.join("settings.json"),
            r#"{"schemaVersion":1,"provider":"local-b","model":"model-b"}"#,
        )
        .expect("settings file");
        fs::write(
            directory.join("providers.json"),
            r#"{
                "providers": {
                    "local-a": {
                        "base_url": "http://127.0.0.1:11434/v1",
                        "api_key": "secret-a",
                        "models": [{"id": "model-a"}]
                    },
                    "local-b": {
                        "base_url": "http://127.0.0.1:11435/v1",
                        "api_key": "secret-b",
                        "models": [{"id": "model-b"}]
                    }
                }
            }"#,
        )
        .expect("providers file");

        let configured = ProcessCodingConfiguration::new(&directory, BTreeMap::new())
            .unwrap()
            .load_all()
            .unwrap();

        assert_eq!(configured.settings().provider, "local-b");
        assert_eq!(configured.settings().model, "model-b");
        assert_eq!(
            configured
                .providers()
                .iter()
                .map(|provider| provider.provider_id().as_str())
                .collect::<Vec<_>>(),
            ["local-b", "local-a"]
        );
        assert!(!format!("{configured:?}").contains("secret-a"));
        assert!(!format!("{configured:?}").contains("secret-b"));

        let mapped = ProcessCodingConfiguration::new(&directory, BTreeMap::new())
            .unwrap()
            .load_all_with_provider_id(|provider_id| {
                ProviderId::from_str(&format!("local.{provider_id}")).ok()
            })
            .unwrap();
        assert_eq!(mapped.settings().provider, "local.local-b");
        assert_eq!(mapped.settings().model, "model-b");
        assert_eq!(
            mapped
                .providers()
                .iter()
                .map(|provider| provider.provider_id().as_str())
                .collect::<Vec<_>>(),
            ["local.local-b", "local.local-a"]
        );
        assert_eq!(
            mapped.providers()[0].models()[0].provider_id().as_str(),
            "local.local-b"
        );
        fs::remove_dir_all(directory).expect("temporary configuration cleanup");
    }

    #[test]
    fn provider_api_key_templates_use_the_injected_environment() {
        let directory = config_dir("template");
        write_configuration(&directory, "$WORKSHOP_TEST_PROVIDER_KEY");
        let environment = BTreeMap::from([(
            "WORKSHOP_TEST_PROVIDER_KEY".to_owned(),
            "resolved-secret".to_owned(),
        )]);

        let configured = ProcessCodingConfiguration::new(&directory, environment)
            .unwrap()
            .load()
            .unwrap();

        assert_eq!(configured.provider().provider_id().as_str(), "workshop-dev");
        assert!(!format!("{configured:?}").contains("resolved-secret"));
        fs::remove_dir_all(directory).expect("temporary configuration cleanup");
    }

    #[test]
    fn custom_provider_configuration_does_not_implicitly_use_legacy_environment_credentials() {
        let directory = config_dir("no-legacy-fallback");
        fs::write(
            directory.join("settings.json"),
            r#"{"schemaVersion":1,"provider":"workshop-dev","model":"debug-model"}"#,
        )
        .expect("settings file");
        fs::write(
            directory.join("providers.json"),
            r#"{"providers":{"workshop-dev":{"models":[{"id":"debug-model"}]}}}"#,
        )
        .expect("providers file");
        let environment =
            BTreeMap::from([("TEA_OPENAI_API_KEY".to_owned(), "legacy-secret".to_owned())]);

        let error = ProcessCodingConfiguration::new(&directory, environment)
            .unwrap()
            .load()
            .expect_err("custom provider requires an explicit credential");

        assert_eq!(error.code(), CodingErrorCode::Credential);
        assert!(!error.message().contains("legacy-secret"));
        fs::remove_dir_all(directory).expect("temporary configuration cleanup");
    }

    #[test]
    fn invalid_provider_documents_fail_at_the_configuration_boundary() {
        let directory = config_dir("invalid");
        fs::write(directory.join("providers.json"), b"not json").expect("providers file");

        let error = ProcessCodingConfiguration::new(&directory, BTreeMap::new())
            .unwrap()
            .load()
            .expect_err("invalid provider file fails");

        assert_eq!(error.code(), CodingErrorCode::InvalidInput);
        assert_eq!(error.message(), "provider configuration is invalid");
        fs::remove_dir_all(directory).expect("temporary configuration cleanup");
    }

    #[test]
    fn configuration_directory_matches_cli_override_precedence() {
        let home = PathBuf::from("/tmp/home");
        let environment = BTreeMap::from([("TEA_CONFIG_DIR".to_owned(), "/tmp/config".to_owned())]);

        assert_eq!(
            configuration_dir(&environment, Some(home.clone())).unwrap(),
            PathBuf::from("/tmp/config")
        );
        assert_eq!(
            configuration_dir(&BTreeMap::new(), Some(home)).unwrap(),
            PathBuf::from("/tmp/home/.tea")
        );
    }
}
