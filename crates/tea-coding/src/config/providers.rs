use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::io::Read as _;
use std::path::Path;
use std::process::{Command, Stdio};
use std::str::FromStr as _;
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};

use serde::Deserialize;
use tea_model::{HostedToolKind, ModelDisplayName, ProviderId, ReasoningProfile};
use tea_protocol::{ModelId, ReasoningEffort, TokenCount};
use tea_provider_openai::{OpenAiApiMode, OpenAiCompatibilityProfile};

/// Maximum accepted size of one provider configuration file.
pub const MAX_PROVIDERS_FILE_BYTES: usize = 256 * 1024;

const MAX_CONFIGURED_PROVIDERS: usize = 64;
const MAX_MODELS_PER_PROVIDER: usize = 256;
const MAX_HOSTED_TOOLS_PER_MODEL: usize = 16;
const MAX_COMMAND_OUTPUT_BYTES: u64 = 16 * 1024;
const COMMAND_TIMEOUT: Duration = Duration::from_secs(10);

/// Provider map loaded from a standalone `providers.json` file.
#[derive(Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProvidersConfig {
    /// OpenAI-compatible providers keyed by canonical provider selector.
    #[serde(default)]
    pub providers: BTreeMap<String, ProviderConfig>,
}

impl fmt::Debug for ProvidersConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProvidersConfig")
            .field("providers", &self.providers)
            .finish()
    }
}

impl ProvidersConfig {
    /// Overlays project/provider values on top of this configuration.
    #[must_use]
    pub fn merged(mut self, overlay: Self) -> Self {
        for (provider_id, provider) in overlay.providers {
            match self.providers.get_mut(&provider_id) {
                Some(base) => base.apply(provider),
                None => {
                    self.providers.insert(provider_id, provider);
                }
            }
        }
        self
    }
}

/// Sparse OpenAI-compatible provider connection and model catalog.
#[derive(Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderConfig {
    /// Optional human-readable provider name.
    pub name: Option<String>,
    /// Optional API base URL.
    pub base_url: Option<String>,
    /// Optional API key literal, environment template, or `!command`.
    pub api_key: Option<String>,
    /// Optional HTTP header carrying the API key.
    pub api_key_header: Option<String>,
    /// Optional prefix prepended to the API key.
    pub api_key_prefix: Option<String>,
    /// Optional API mode: `chat-completions` or `responses`.
    pub api_mode: Option<String>,
    /// Exact structured-output dialect; absent means structured output is unsupported.
    pub compatibility_profile: Option<OpenAiCompatibilityProfile>,
    /// Optional `OpenAI` organization identifier.
    pub org_id: Option<String>,
    /// Optional `OpenAI` project identifier.
    pub project_id: Option<String>,
    /// Deprecated provider-wide reasoning default used only as a model-profile fallback.
    pub reasoning_effort: Option<String>,
    /// Whether models in this provider accept image input.
    pub vision: Option<bool>,
    /// Optional positive request timeout in milliseconds.
    pub timeout_millis: Option<u64>,
    /// Models advertised by this provider. A non-empty overlay replaces the base list.
    #[serde(default)]
    pub models: Vec<ModelDefinition>,
}

impl fmt::Debug for ProviderConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderConfig")
            .field("name", &self.name)
            .field("base_url", &self.base_url)
            .field("api_key", &self.api_key.as_ref().map(|_| "**REDACTED**"))
            .field("api_key_header", &self.api_key_header)
            .field("api_key_prefix", &self.api_key_prefix)
            .field("api_mode", &self.api_mode)
            .field("compatibility_profile", &self.compatibility_profile)
            .field("org_id", &self.org_id)
            .field("project_id", &self.project_id)
            .field("reasoning_effort", &self.reasoning_effort)
            .field("vision", &self.vision)
            .field("timeout_millis", &self.timeout_millis)
            .field("models", &self.models)
            .finish()
    }
}

impl ProviderConfig {
    fn apply(&mut self, overlay: Self) {
        replace(&mut self.name, overlay.name);
        replace(&mut self.base_url, overlay.base_url);
        replace(&mut self.api_key, overlay.api_key);
        replace(&mut self.api_key_header, overlay.api_key_header);
        replace(&mut self.api_key_prefix, overlay.api_key_prefix);
        replace(&mut self.api_mode, overlay.api_mode);
        replace(
            &mut self.compatibility_profile,
            overlay.compatibility_profile,
        );
        replace(&mut self.org_id, overlay.org_id);
        replace(&mut self.project_id, overlay.project_id);
        replace(&mut self.reasoning_effort, overlay.reasoning_effort);
        replace(&mut self.vision, overlay.vision);
        replace(&mut self.timeout_millis, overlay.timeout_millis);
        if !overlay.models.is_empty() {
            self.models = overlay.models;
        }
    }

    pub(super) fn structured_output_configuration_is_valid(&self) -> bool {
        let api_mode = match self.api_mode.as_deref() {
            None | Some("chat-completions") => OpenAiApiMode::ChatCompletions,
            Some("responses") => OpenAiApiMode::Responses,
            Some(_) => return false,
        };
        let Some(profile) = self.compatibility_profile else {
            return self.models.iter().all(|model| {
                !model.capabilities.final_json_object
                    && !model.capabilities.final_json_schema
                    && !model.capabilities.final_json_schema_with_tools
            });
        };
        profile.supports_api_mode(api_mode)
            && self.models.iter().all(|model| {
                (!model.capabilities.final_json_schema_with_tools
                    || model.capabilities.final_json_schema)
                    && (!model.capabilities.final_json_object
                        || profile.supports_final_json_object(api_mode))
                    && (!model.capabilities.final_json_schema
                        || profile.supports_final_json_schema(api_mode))
                    && (!model.capabilities.final_json_schema_with_tools
                        || profile.supports_final_json_schema_with_tools(api_mode))
            })
    }
}

/// One model advertised by a configured provider.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelDefinition {
    /// Canonical model identifier.
    pub id: String,
    /// Optional human-readable model name.
    pub display_name: Option<String>,
    /// Optional context window, defaulting to 128,000 tokens.
    pub context_window_tokens: Option<u64>,
    /// Optional output limit, defaulting to 16,384 tokens.
    pub max_output_tokens: Option<u64>,
    /// Explicit capabilities supported by this model at the configured endpoint.
    #[serde(default)]
    pub capabilities: ModelCapabilitiesConfig,
}

/// Explicit, fail-closed capabilities for one configured model and endpoint.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelCapabilitiesConfig {
    /// Whether this exact model/endpoint supports schema-less JSON-object output.
    #[serde(default)]
    pub final_json_object: bool,
    /// Whether this exact model/endpoint supports native JSON Schema output.
    #[serde(default)]
    pub final_json_schema: bool,
    /// Whether native JSON Schema output may be combined with model-visible tools.
    #[serde(default)]
    pub final_json_schema_with_tools: bool,
    /// Model-specific provider-neutral reasoning support and wire mapping.
    pub reasoning: Option<ModelReasoningConfig>,
    /// Provider-hosted tools supported by this exact model and endpoint.
    #[serde(default)]
    pub hosted_tools: Vec<HostedToolCapability>,
}

/// Strict model-level reasoning profile and provider wire overrides.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelReasoningConfig {
    /// Default canonical effort for this model.
    pub default_effort: ReasoningEffort,
    /// Canonical effort to provider wire value; `null` explicitly disables a level.
    #[serde(default)]
    pub effort_map: BTreeMap<ReasoningEffort, Option<String>>,
}

impl ModelReasoningConfig {
    /// Resolves default support and explicit overrides into a validated profile and wire map.
    #[must_use]
    pub fn resolved(&self) -> Option<(ReasoningProfile, BTreeMap<ReasoningEffort, String>)> {
        if self.effort_map.len() > ReasoningEffort::ALL.len()
            || self
                .effort_map
                .get(&ReasoningEffort::Off)
                .is_some_and(Option::is_some)
        {
            return None;
        }
        let mut supported = Vec::new();
        let mut wire_map = BTreeMap::new();
        for effort in ReasoningEffort::ALL {
            let configured = self.effort_map.get(&effort);
            let enabled = if matches!(
                effort,
                ReasoningEffort::ExtraHigh | ReasoningEffort::Maximum
            ) {
                configured.is_some_and(Option::is_some)
            } else {
                !configured.is_some_and(Option::is_none)
            };
            if !enabled {
                continue;
            }
            supported.push(effort);
            if effort != ReasoningEffort::Off {
                let wire = configured
                    .and_then(Option::as_deref)
                    .unwrap_or_else(|| effort.as_str());
                if !valid_wire_effort(wire) {
                    return None;
                }
                wire_map.insert(effort, wire.to_owned());
            }
        }
        let profile = ReasoningProfile::new(self.default_effort, supported).ok()?;
        Some((profile, wire_map))
    }
}

/// Provider-hosted capability accepted in custom model configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HostedToolCapability {
    /// Searches the public web inside the provider response lifecycle.
    WebSearch,
}

impl HostedToolCapability {
    /// Returns the provider-neutral model capability represented by this value.
    #[must_use]
    pub const fn kind(self) -> HostedToolKind {
        match self {
            Self::WebSearch => HostedToolKind::WebSearch,
        }
    }
}

/// Safe classification for a provider configuration load failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProvidersConfigLoadError {
    /// The file could not be read.
    Read,
    /// The file exceeded [`MAX_PROVIDERS_FILE_BYTES`].
    TooLarge,
    /// JSON, schema, or configured collection bounds were invalid.
    Invalid,
}

/// Soft-failing provider configuration load result.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ProvidersConfigLoad {
    /// Parsed configuration, or an empty map on failure/missing input.
    pub config: ProvidersConfig,
    /// Safe failure classification. Missing files are not errors.
    pub error: Option<ProvidersConfigLoadError>,
}

impl ProvidersConfigLoad {
    fn success(config: ProvidersConfig) -> Self {
        Self {
            config,
            error: None,
        }
    }

    fn failure(error: ProvidersConfigLoadError) -> Self {
        Self {
            config: ProvidersConfig::default(),
            error: Some(error),
        }
    }
}

/// Loads a strict, bounded provider configuration without propagating file errors.
#[must_use]
pub fn load_providers_file(path: &Path) -> ProvidersConfigLoad {
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return ProvidersConfigLoad::success(ProvidersConfig::default());
        }
        Err(_) => return ProvidersConfigLoad::failure(ProvidersConfigLoadError::Read),
    };
    if bytes.len() > MAX_PROVIDERS_FILE_BYTES {
        return ProvidersConfigLoad::failure(ProvidersConfigLoadError::TooLarge);
    }
    let Ok(config) = serde_json::from_slice::<ProvidersConfig>(&bytes) else {
        return ProvidersConfigLoad::failure(ProvidersConfigLoadError::Invalid);
    };
    if !valid_config(&config) {
        return ProvidersConfigLoad::failure(ProvidersConfigLoadError::Invalid);
    }
    ProvidersConfigLoad::success(config)
}

/// Resolves provider values from environment templates and cached shell commands.
pub struct ProviderValueResolver {
    environment: BTreeMap<String, String>,
    commands: Mutex<BTreeMap<String, Option<String>>>,
}

impl fmt::Debug for ProviderValueResolver {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProviderValueResolver")
            .field("environment", &"**REDACTED**")
            .field("commands", &"**REDACTED**")
            .finish()
    }
}

impl ProviderValueResolver {
    /// Creates a resolver with an explicit, hermetic command environment.
    #[must_use]
    pub fn new(environment: BTreeMap<String, String>) -> Self {
        Self {
            environment,
            commands: Mutex::new(BTreeMap::new()),
        }
    }

    /// Resolves `$NAME`/`${NAME}` templates or a leading `!command`.
    ///
    /// Missing environment values and failed, empty, oversized, or timed-out
    /// commands resolve to `None`.
    #[must_use]
    pub fn resolve(&self, configured: &str) -> Option<String> {
        if let Some(command) = configured.strip_prefix('!') {
            return self.resolve_command(command.trim());
        }
        resolve_template(configured, &self.environment)
    }

    fn resolve_command(&self, command: &str) -> Option<String> {
        if command.is_empty() {
            return None;
        }
        if let Ok(cache) = self.commands.lock()
            && let Some(value) = cache.get(command)
        {
            return value.clone();
        }
        let value = execute_command(command, &self.environment);
        if let Ok(mut cache) = self.commands.lock() {
            cache.insert(command.to_owned(), value.clone());
        }
        value
    }
}

fn replace<T>(base: &mut Option<T>, overlay: Option<T>) {
    if overlay.is_some() {
        *base = overlay;
    }
}

fn valid_config(config: &ProvidersConfig) -> bool {
    config.providers.len() <= MAX_CONFIGURED_PROVIDERS
        && config.providers.iter().all(|(provider_id, provider)| {
            ProviderId::from_str(provider_id).is_ok()
                && provider.models.len() <= MAX_MODELS_PER_PROVIDER
                && provider
                    .api_mode
                    .as_deref()
                    .is_none_or(|mode| matches!(mode, "chat-completions" | "responses"))
                && provider
                    .reasoning_effort
                    .as_deref()
                    .is_none_or(|effort| ReasoningEffort::from_str(effort).is_ok())
                && provider.timeout_millis.is_none_or(|timeout| timeout > 0)
                && provider.structured_output_configuration_is_valid()
                && valid_models(&provider.models)
        })
}

fn valid_models(models: &[ModelDefinition]) -> bool {
    let mut ids = std::collections::BTreeSet::new();
    models.iter().all(|model| {
        let context = model.context_window_tokens.unwrap_or(128_000);
        let output = model.max_output_tokens.unwrap_or(16_384);
        ids.insert(model.id.as_str())
            && ModelId::from_str(&model.id).is_ok()
            && ModelDisplayName::from_str(model.display_name.as_deref().unwrap_or(&model.id))
                .is_ok()
            && TokenCount::new(context).is_ok()
            && TokenCount::new(output).is_ok()
            && output <= context
            && (!model.capabilities.final_json_schema_with_tools
                || model.capabilities.final_json_schema)
            && model
                .capabilities
                .reasoning
                .as_ref()
                .is_none_or(|reasoning| reasoning.resolved().is_some())
            && valid_hosted_tools(&model.capabilities.hosted_tools)
    })
}

fn valid_wire_effort(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'))
}

fn valid_hosted_tools(tools: &[HostedToolCapability]) -> bool {
    let mut unique = std::collections::BTreeSet::new();
    tools.len() <= MAX_HOSTED_TOOLS_PER_MODEL && tools.iter().all(|tool| unique.insert(*tool))
}

fn resolve_template(value: &str, environment: &BTreeMap<String, String>) -> Option<String> {
    let bytes = value.as_bytes();
    let mut output = String::with_capacity(value.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'$' {
            let character = value[index..].chars().next()?;
            output.push(character);
            index += character.len_utf8();
            continue;
        }
        match bytes.get(index + 1).copied() {
            Some(b'$') => {
                output.push('$');
                index += 2;
            }
            Some(b'!') => {
                output.push('!');
                index += 2;
            }
            Some(b'{') => {
                let Some(relative_end) = value[index + 2..].find('}') else {
                    output.push('$');
                    index += 1;
                    continue;
                };
                let end = index + 2 + relative_end;
                let name = &value[index + 2..end];
                if valid_environment_name(name) {
                    output.push_str(environment.get(name)?);
                } else {
                    output.push_str(&value[index..=end]);
                }
                index = end + 1;
            }
            Some(_) => {
                let name_end = value[index + 1..]
                    .char_indices()
                    .take_while(|(offset, character)| {
                        if *offset == 0 {
                            *character == '_' || character.is_ascii_alphabetic()
                        } else {
                            *character == '_' || character.is_ascii_alphanumeric()
                        }
                    })
                    .map(|(offset, character)| index + 1 + offset + character.len_utf8())
                    .last();
                if let Some(end) = name_end {
                    output.push_str(environment.get(&value[index + 1..end])?);
                    index = end;
                } else {
                    output.push('$');
                    index += 1;
                }
            }
            None => {
                output.push('$');
                index += 1;
            }
        }
    }
    Some(output)
}

fn valid_environment_name(name: &str) -> bool {
    let mut characters = name.chars();
    characters
        .next()
        .is_some_and(|character| character == '_' || character.is_ascii_alphabetic())
        && characters.all(|character| character == '_' || character.is_ascii_alphanumeric())
}

fn execute_command(command: &str, environment: &BTreeMap<String, String>) -> Option<String> {
    let mut process = configured_shell(environment);
    process
        .env_clear()
        .envs(environment)
        .arg(command)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let mut child = process.spawn().ok()?;
    let stdout = child.stdout.take()?;
    let reader = thread::spawn(move || {
        let mut bytes = Vec::new();
        stdout
            .take(MAX_COMMAND_OUTPUT_BYTES + 1)
            .read_to_end(&mut bytes)
            .ok()?;
        Some(bytes)
    });
    let deadline = Instant::now() + COMMAND_TIMEOUT;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(10)),
            Ok(None) | Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
        }
    };
    let bytes = reader.join().ok()??;
    if !status?.success() || u64::try_from(bytes.len()).ok()? > MAX_COMMAND_OUTPUT_BYTES {
        return None;
    }
    let output = String::from_utf8(bytes).ok()?;
    let output = output.trim();
    (!output.is_empty()).then(|| output.to_owned())
}

#[cfg(unix)]
fn configured_shell(environment: &BTreeMap<String, String>) -> Command {
    let mut command = Command::new(
        environment
            .get("TEA_SHELL")
            .map_or("/bin/sh", String::as_str),
    );
    command.arg(
        environment
            .get("TEA_SHELL_FLAG")
            .map_or("-c", String::as_str),
    );
    command
}

#[cfg(windows)]
fn configured_shell(environment: &BTreeMap<String, String>) -> Command {
    let mut command = Command::new(
        environment
            .get("TEA_SHELL")
            .or_else(|| environment.get("COMSPEC"))
            .map_or("cmd.exe", String::as_str),
    );
    command.arg(
        environment
            .get("TEA_SHELL_FLAG")
            .map_or("/C", String::as_str),
    );
    command
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn templates_expand_and_escape_environment_values() {
        let resolver = ProviderValueResolver::new(BTreeMap::from([
            ("TOKEN".to_owned(), "secret".to_owned()),
            ("SUFFIX_2".to_owned(), "tail".to_owned()),
        ]));
        assert_eq!(
            resolver.resolve("pre-$TOKEN-${SUFFIX_2}-$$-$!"),
            Some("pre-secret-tail-$-!".to_owned())
        );
        assert_eq!(resolver.resolve("$MISSING"), None);
        assert_eq!(
            resolver.resolve("${bad-name}"),
            Some("${bad-name}".to_owned())
        );
    }

    #[cfg(unix)]
    #[test]
    fn command_values_are_trimmed_and_cached() {
        let resolver = ProviderValueResolver::new(BTreeMap::new());
        assert_eq!(
            resolver.resolve("!printf ' key\\n'"),
            Some("key".to_owned())
        );
        assert_eq!(
            resolver.resolve("!printf ' key\\n'"),
            Some("key".to_owned())
        );
        assert_eq!(resolver.commands.lock().unwrap().len(), 1);
    }

    #[test]
    fn provider_debug_output_redacts_api_keys() {
        let provider = ProviderConfig {
            api_key: Some("secret-provider-key".to_owned()),
            ..ProviderConfig::default()
        };
        assert!(!format!("{provider:?}").contains("secret-provider-key"));
    }
}
