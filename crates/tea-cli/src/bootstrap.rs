use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use tea_coding::config::{
    CodingSettings, ProviderConfig, ProvidersConfigLoad, SettingsLayer, WebFetchBackend,
    WebFetchSettings, WebSearchClientBackend, WebSearchSettings, load_providers_file,
    load_settings_file, merge_settings, resolve_openai_compatible_provider,
};
use tea_coding::mcp_config::{
    McpEnvironmentValue, ProcessMcpEnvironmentResolver, resolve_mcp_environment,
};
use tea_coding::resources::{
    CodingPromptResourceRoots, ResourceCatalog, SkillRoot, SkillSource, has_project_instructions,
    project_instruction_boundary,
};
use tea_coding::{
    AppPaths, CodingAgentBuilder, CodingAgentService, InteractionMode, McpEnvironmentResolver,
    PersistedTrustDecision, ProjectAccess, ProjectTrustStore, TrustRequest,
};
use tea_coding_tools::{
    BashConfig, BashOutputDirectory, BashShell, FetchCacheConfig, FetchCacheScope, FetchProvider,
    FetchResultCache, HttpFetchProvider, SearchProvider, TavilyApiKey, TavilySearchConfig,
    TavilySearchProvider, WorkspaceRoot,
};
use tea_mcp::{MAX_MCP_STARTUP_CONCURRENCY, McpManager, McpServerConfig, McpServerLaunch};
use tea_model::{
    BoxModelStream, ModelCancellation, ModelCapabilities, ModelDisplayName, ModelEvent,
    ModelFailure, ModelFailureCode, ModelProvider, ModelRequest, ModelResponseInfo, ModelSpec,
    ProviderId,
};
use tea_policy::{ActorId, WorkspaceId};
use tea_protocol::ProtocolTimestamp;
use tea_protocol::{ModelId, RetryClass, TokenCount};
use tea_provider_anthropic::{
    AnthropicProviderBuilder, CredentialResolver as _,
    MapCredentialResolver as AnthropicCredentialResolver,
};
use tea_provider_http::{ProviderHttpConfig, UserAgent};
use tea_session_sqlite::SqliteSessionStore;
use tea_tools::{ToolName, ToolTrust};

use crate::args::{CliArgs, SessionSelection, TrustArg};
use crate::{CliFailure, ExitCategory};

/// Process values read once at the outward CLI boundary.
#[derive(Clone)]
pub struct BootstrapEnvironment {
    current_dir: PathBuf,
    home_dir: Option<PathBuf>,
    values: BTreeMap<String, String>,
    live_mcp_environment: bool,
}

impl std::fmt::Debug for BootstrapEnvironment {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("BootstrapEnvironment")
            .field("current_dir", &self.current_dir)
            .field("home_dir", &self.home_dir)
            .field("values", &"**REDACTED**")
            .field(
                "mcp_environment",
                &if self.live_mcp_environment {
                    "process"
                } else {
                    "injected"
                },
            )
            .finish()
    }
}

impl BootstrapEnvironment {
    /// Creates a fully injected environment for hermetic embedding/tests.
    #[must_use]
    pub fn new(
        current_dir: impl Into<PathBuf>,
        home_dir: Option<PathBuf>,
        values: BTreeMap<String, String>,
    ) -> Self {
        Self {
            current_dir: current_dir.into(),
            home_dir,
            values,
            live_mcp_environment: false,
        }
    }

    /// Captures the process environment once for later hermetic resolution.
    ///
    /// # Errors
    ///
    /// Returns a configuration failure when the current directory is unavailable.
    pub fn from_process() -> Result<Self, CliFailure> {
        let current_dir = std::env::current_dir()
            .map_err(|_| config_failure("current directory is unavailable"))?;
        let home_dir = process_home_dir();
        let values = std::env::vars_os()
            .filter_map(|(key, value)| Some((key.into_string().ok()?, value.into_string().ok()?)))
            .collect();
        let mut environment = Self::new(current_dir, home_dir, values);
        environment.live_mcp_environment = true;
        Ok(environment)
    }

    fn value(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(String::as_str)
    }
}

fn process_home_dir() -> Option<PathBuf> {
    home_dir_from(|key| std::env::var_os(key))
}

fn home_dir_from(mut value: impl FnMut(&str) -> Option<std::ffi::OsString>) -> Option<PathBuf> {
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

impl McpEnvironmentResolver for BootstrapEnvironment {
    fn resolve(
        &self,
        name: &str,
    ) -> Result<Option<McpEnvironmentValue>, tea_coding::mcp_config::McpEnvironmentError> {
        if self.live_mcp_environment {
            return ProcessMcpEnvironmentResolver.resolve(name);
        }
        self.values
            .get(name)
            .map(|value| McpEnvironmentValue::try_new(name, value))
            .transpose()
    }
}

/// Injectable product bootstrap; production uses the live provider by default.
#[derive(Debug, Clone)]
pub struct CliBootstrap {
    environment: BootstrapEnvironment,
    providers: Vec<Arc<dyn ModelProvider>>,
}

/// The product surface responsible for a model request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ClientSurface {
    Cli,
    Tui,
}

impl ClientSurface {
    const fn name(self) -> &'static str {
        match self {
            Self::Cli => "tea-cli",
            Self::Tui => "tea-tui",
        }
    }
}

/// A workspace whose project-local resources require an interactive trust decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct WorkspaceTrustPrompt {
    workspace: PathBuf,
    trust_file: PathBuf,
}

impl WorkspaceTrustPrompt {
    /// Returns the canonical workspace path displayed to the user.
    #[must_use]
    pub(crate) fn workspace(&self) -> &Path {
        &self.workspace
    }
}

impl CliBootstrap {
    /// Creates a bootstrap from an injected environment.
    #[must_use]
    pub fn new(environment: BootstrapEnvironment) -> Self {
        Self {
            environment,
            providers: Vec::new(),
        }
    }

    /// Injects an additional provider (for embedding and hermetic tests).
    #[must_use]
    pub fn with_provider(mut self, provider: Arc<dyn ModelProvider>) -> Self {
        self.providers.push(provider);
        self
    }

    /// Returns the late MCP environment resolver owned by this bootstrap.
    #[must_use]
    pub fn mcp_environment_resolver(&self) -> &dyn McpEnvironmentResolver {
        &self.environment
    }

    /// Reads one bounded UTF-8 prompt file through the workspace capability.
    ///
    /// # Errors
    ///
    /// Rejects absolute, traversing, escaping, changed, non-file, invalid UTF-8,
    /// or oversized input.
    pub fn read_prompt_file(&self, args: &CliArgs, path: &str) -> Result<String, CliFailure> {
        let workspace_path = args
            .cwd
            .as_deref()
            .unwrap_or(self.environment.current_dir.as_path());
        let workspace = WorkspaceRoot::new(workspace_path)
            .map_err(|_| config_failure("workspace directory is invalid"))?;
        let resolved = workspace
            .resolve_existing(path)
            .map_err(|_| CliFailure::usage("prompt file path is invalid"))?;
        let mut file = fs::File::open(resolved.host_path())
            .map_err(|_| CliFailure::usage("prompt file could not be read"))?;
        let metadata = file
            .metadata()
            .map_err(|_| CliFailure::usage("prompt file could not be read"))?;
        workspace
            .verify_opened_existing(&resolved, &metadata)
            .map_err(|_| CliFailure::usage("prompt file changed during read"))?;
        if !metadata.is_file()
            || metadata.len() > crate::modes::print::MAX_INITIAL_PROMPT_BYTES as u64
        {
            return Err(CliFailure::usage("prompt file exceeds input size limit"));
        }
        let mut bytes = Vec::with_capacity(usize::try_from(metadata.len()).unwrap_or(0));
        file.by_ref()
            .take(crate::modes::print::MAX_INITIAL_PROMPT_BYTES as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| CliFailure::usage("prompt file could not be read"))?;
        workspace
            .revalidate_existing(&resolved)
            .map_err(|_| CliFailure::usage("prompt file changed during read"))?;
        if u64::try_from(bytes.len()).ok() != Some(metadata.len()) {
            return Err(CliFailure::usage("prompt file changed during read"));
        }
        if bytes.len() > crate::modes::print::MAX_INITIAL_PROMPT_BYTES {
            return Err(CliFailure::usage("prompt file exceeds input size limit"));
        }
        String::from_utf8(bytes).map_err(|_| CliFailure::usage("prompt file is not valid UTF-8"))
    }

    /// Builds one mode-neutral coding service and resolves session selection.
    ///
    /// # Errors
    ///
    /// Returns stable trust/config/provider/internal failures without secret values.
    pub fn build(
        &self,
        args: &CliArgs,
    ) -> Result<(CodingAgentService, SessionSelection), CliFailure> {
        let prepared = self.prepare(args, ClientSurface::Cli)?;
        if !prepared.mcp_servers.is_empty() {
            return Err(config_failure(
                "MCP servers require asynchronous CLI bootstrap",
            ));
        }
        let service = prepared.builder.build().map_err(CliFailure::from)?;
        Ok((service, prepared.selection))
    }

    /// Asynchronously initializes MCP servers before immutable runtime binding.
    ///
    /// # Errors
    ///
    /// Returns a stable CLI failure when configuration, environment resolution,
    /// server startup, or runtime assembly fails.
    pub async fn build_async(
        &self,
        args: &CliArgs,
    ) -> Result<(CodingAgentService, SessionSelection), CliFailure> {
        let (service, selection, _) = self
            .build_async_for_surface(args, ClientSurface::Cli)
            .await?;
        Ok((service, selection))
    }

    /// Builds the service used by the interactive terminal interface.
    pub(crate) async fn build_tui_async(
        &self,
        args: &CliArgs,
    ) -> Result<(CodingAgentService, SessionSelection, Vec<String>), CliFailure> {
        self.build_async_for_surface(args, ClientSurface::Tui).await
    }

    /// Returns the trust prompt required before interactive bootstrap may continue.
    pub(crate) fn workspace_trust_prompt(
        &self,
        args: &CliArgs,
    ) -> Result<Option<WorkspaceTrustPrompt>, CliFailure> {
        let (workspace, paths) = self.workspace_and_paths(args)?;
        let project_boundary = project_instruction_boundary(workspace.host_path())?;
        let access = Self::resolve_project_access(
            args,
            ClientSurface::Tui,
            &workspace,
            &project_boundary,
            &paths,
        )?;
        Ok(
            (access == ProjectAccess::Ask).then(|| WorkspaceTrustPrompt {
                workspace: project_boundary,
                trust_file: paths.trust_file(),
            }),
        )
    }

    /// Returns the global sparse settings file used by persistent TUI choices.
    pub(crate) fn global_settings_path(&self, args: &CliArgs) -> Result<PathBuf, CliFailure> {
        self.app_paths(args).map(|paths| paths.settings_file())
    }

    /// Persists an accepted interactive workspace trust decision.
    pub(crate) fn accept_workspace_trust(prompt: &WorkspaceTrustPrompt) -> Result<(), CliFailure> {
        ProjectTrustStore::new(prompt.trust_file.clone())
            .set(&prompt.workspace, PersistedTrustDecision::Trusted)
            .map_err(CliFailure::from)
    }

    async fn build_async_for_surface(
        &self,
        args: &CliArgs,
        surface: ClientSurface,
    ) -> Result<(CodingAgentService, SessionSelection, Vec<String>), CliFailure> {
        let prepared = self.prepare(args, surface)?;
        let manager = self
            .start_mcp_manager(&prepared.mcp_servers, &prepared.active_mcp_tools)
            .await?;
        let mut builder = prepared.builder;
        if let Some(manager) = &manager {
            builder = builder.mcp_manager(Arc::clone(manager));
        }
        match builder.build().map_err(CliFailure::from) {
            Ok(service) => Ok((service, prepared.selection, prepared.provider_notices)),
            Err(error) => {
                if let Some(manager) = manager {
                    let _ = manager.shutdown().await;
                }
                Err(error)
            }
        }
    }

    #[allow(clippy::too_many_lines)] // Bootstrap order preserves trust and resource boundaries.
    fn prepare(
        &self,
        args: &CliArgs,
        surface: ClientSurface,
    ) -> Result<PreparedBootstrap, CliFailure> {
        let selection = args.session_selection()?;
        if args.profile_id()?.as_str() != "coding-agent" {
            return Err(config_failure("profile is not registered"));
        }
        let (workspace, paths) = self.workspace_and_paths(args)?;
        let project_boundary = project_instruction_boundary(workspace.host_path())?;
        let access =
            Self::resolve_project_access(args, surface, &workspace, &project_boundary, &paths)?;
        if access == ProjectAccess::Ask {
            return Err(config_failure("workspace trust confirmation is required"));
        }
        let global = load_settings_file(&paths.settings_file()).map_err(CliFailure::from)?;
        let global_providers = load_providers_file(&paths.providers_file());
        let mut project = if access == tea_coding::ProjectAccess::Trusted {
            load_project_settings(&workspace)?
        } else {
            None
        };
        let project_providers = if access == tea_coding::ProjectAccess::Trusted {
            load_project_providers(&workspace)?
        } else {
            ProvidersConfigLoad::default()
        };
        let providers_config_error =
            global_providers.error.is_some() || project_providers.error.is_some();
        let mut provider_notices = Vec::new();
        if providers_config_error {
            if surface == ClientSurface::Tui {
                provider_notices
                    .push("custom provider configuration could not be loaded".to_owned());
            } else {
                eprintln!("warning: custom provider configuration could not be loaded");
            }
        }
        let providers = global_providers.config.merged(project_providers.config);
        if let Some(project) = &mut project {
            // Durable state locations are host-owned, never project-controlled.
            project.session_database = None;
        }
        let environment = self.environment_settings();
        let cli = cli_settings(args)?;
        let settings = merge_settings(
            CodingSettings::default(),
            global.as_ref(),
            project.as_ref(),
            Some(&environment),
            Some(&cli),
        )
        .map_err(CliFailure::from)?;
        let mcp_servers = configured_mcp_servers(&settings, project.as_ref());
        let active_mcp_tools = active_mcp_tools(&settings, &mcp_servers)?;
        if !matches!(settings.provider.as_str(), "openai" | "anthropic")
            && !providers.providers.contains_key(&settings.provider)
            && !self
                .providers
                .iter()
                .any(|provider| provider.provider_id().as_str() == settings.provider)
            && surface != ClientSurface::Tui
        {
            return Err(config_failure(if providers_config_error {
                "custom provider configuration is invalid"
            } else {
                "provider is not supported"
            }));
        }

        let mut skill_roots = Vec::with_capacity(settings.resources.skill_paths.len() + 5);
        for path in &settings.resources.skill_paths {
            let resolved = workspace
                .resolve_existing(path)
                .map_err(|_| config_failure("configured skill path is invalid"))?;
            skill_roots.push(
                SkillRoot::new(resolved.host_path().to_path_buf(), SkillSource::Explicit)
                    .map_err(|_| config_failure("configured skill path is invalid"))?,
            );
        }
        for (path, source) in [
            (
                workspace.host_path().join(".tea/skills"),
                SkillSource::ProjectTea,
            ),
            (
                workspace.host_path().join(".agents/skills"),
                SkillSource::ProjectAgents,
            ),
            (paths.config_dir().join("skills"), SkillSource::UserTea),
        ] {
            skill_roots.push(
                SkillRoot::new(path, source)
                    .map_err(|_| config_failure("skill root is invalid"))?,
            );
        }
        if let Some(home) = &self.environment.home_dir {
            skill_roots.push(
                SkillRoot::new(home.join(".agents/skills"), SkillSource::UserAgents)
                    .map_err(|_| config_failure("skill root is invalid"))?,
            );
        }
        skill_roots.push(
            SkillRoot::new(paths.data_dir().join("skills"), SkillSource::TeaData)
                .map_err(|_| config_failure("skill root is invalid"))?,
        );
        let global_prompts = paths.data_dir().join("prompts");
        let coding_prompt_roots = CodingPromptResourceRoots::new(paths.data_dir())?
            .with_project_root(workspace.host_path().join(".tea"))?;
        let project_prompts = if access == tea_coding::ProjectAccess::Trusted {
            optional_project_directory(&workspace, ".tea/prompts")?
        } else {
            None
        };
        let mut resources = ResourceCatalog::discover_complete(
            &project_boundary,
            workspace.host_path(),
            access,
            &skill_roots,
            Some(&global_prompts),
            project_prompts.as_deref(),
            Some(&coding_prompt_roots),
        )
        .map_err(CliFailure::from)?;
        resources.apply_settings(
            settings.resources.context_files,
            settings.resources.prompt_templates,
        );
        resources
            .add_explicit_context_files(&workspace, &args.context_files)
            .map_err(CliFailure::from)?;

        let store = if selection == SessionSelection::NoSession {
            Arc::new(
                SqliteSessionStore::in_memory()
                    .map_err(|_| internal_failure("session store initialization failed"))?,
            )
        } else {
            let database = session_database(args, &settings, &paths)?;
            Arc::new(
                SqliteSessionStore::open(path_text(&database)?)
                    .map_err(|_| internal_failure("session store initialization failed"))?,
            )
        };
        let overflow = paths.state_dir().join("bash-output");
        create_private_directory(&overflow)?;
        let bash = BashConfig::new(
            shell(&self.environment)?,
            BashOutputDirectory::new(&overflow)
                .map_err(|_| config_failure("bash output directory is invalid"))?,
            Duration::from_mins(2),
        )
        .map_err(|_| config_failure("bash configuration is invalid"))?;
        let http_config = ProviderHttpConfig::new()
            .with_user_agent(client_user_agent(surface, &self.environment));
        let search_provider =
            client_web_search_provider(&settings.web_search, &self.environment, &http_config)?;
        let workspace_id = workspace_id(workspace.host_path())?;
        let fetch_provider = client_web_fetch_provider(
            &settings.web_fetch,
            &settings.active_tools,
            &workspace_id,
            &http_config,
        )?;
        let provider = (|| -> Result<Arc<dyn ModelProvider>, CliFailure> {
            if let Some(provider) = self
                .providers
                .iter()
                .find(|provider| provider.provider_id().as_str() == settings.provider)
            {
                return Ok(Arc::clone(provider));
            }
            if settings.provider != "anthropic"
                && let Some(config) = providers.providers.get(&settings.provider)
            {
                return custom_openai_provider(
                    &settings.provider,
                    &settings.model,
                    config,
                    args.api_key.as_ref(),
                    &self.environment,
                    http_config.clone(),
                );
            }
            match settings.provider.as_str() {
                "openai" => builtin_openai_provider(
                    &settings.model,
                    args.api_key.as_ref(),
                    &self.environment,
                    http_config.clone(),
                ),
                "anthropic" => builtin_anthropic_provider(
                    &settings.model,
                    args.api_key.as_ref(),
                    &self.environment,
                    http_config.clone(),
                ),
                _ => Err(config_failure(if providers_config_error {
                    "custom provider configuration is invalid"
                } else {
                    "provider is not supported"
                })),
            }
        })();
        let provider = match provider {
            Ok(provider) => provider,
            Err(error) if surface == ClientSurface::Tui => {
                provider_notices.push(format!("provider unavailable: {}", error.message()));
                unavailable_provider(&settings, error.message())?
            }
            Err(error) => return Err(error),
        };
        let mut additional_providers = self
            .providers
            .iter()
            .filter(|candidate| !Arc::ptr_eq(candidate, &provider))
            .cloned()
            .collect::<Vec<_>>();
        for (provider_id, config) in &providers.providers {
            if provider_id == provider.provider_id().as_str()
                || additional_providers
                    .iter()
                    .any(|provider| provider.provider_id().as_str() == provider_id)
            {
                continue;
            }
            let Some(model_id) = config.models.first().map(|model| model.id.as_str()) else {
                continue;
            };
            if let Ok(additional) = custom_openai_provider(
                provider_id,
                model_id,
                config,
                None,
                &self.environment,
                http_config.clone(),
            ) {
                additional_providers.push(additional);
            }
        }
        for (provider_id, model_key, key_key, constructor) in [
            (
                "openai",
                "TEA_OPENAI_MODEL",
                "TEA_OPENAI_API_KEY",
                builtin_openai_provider as BuiltinProviderConstructor,
            ),
            (
                "anthropic",
                "TEA_ANTHROPIC_MODEL",
                "TEA_ANTHROPIC_API_KEY",
                builtin_anthropic_provider as BuiltinProviderConstructor,
            ),
        ] {
            if provider_id == provider.provider_id().as_str()
                || additional_providers
                    .iter()
                    .any(|candidate| candidate.provider_id().as_str() == provider_id)
            {
                continue;
            }
            let Some(model_id) = self.environment.value(model_key) else {
                continue;
            };
            if self.environment.value(key_key).is_none() {
                continue;
            }
            if let Ok(additional) =
                constructor(model_id, None, &self.environment, http_config.clone())
            {
                additional_providers.push(additional);
            }
        }
        let mut builder = CodingAgentBuilder::new(
            provider,
            workspace.clone(),
            resources,
            store,
            bash,
            settings,
            ActorId::from_str("local:user")
                .map_err(|_| internal_failure("actor identity failed"))?,
            workspace_id,
        );
        for provider in additional_providers {
            builder = builder.provider(provider);
        }
        if let Some(search_provider) = search_provider {
            builder = builder.search_provider(search_provider);
        }
        if let Some(fetch_provider) = fetch_provider {
            builder = builder.fetch_provider(fetch_provider);
        }
        Ok(PreparedBootstrap {
            builder,
            selection,
            mcp_servers,
            active_mcp_tools,
            provider_notices,
        })
    }

    fn workspace_and_paths(&self, args: &CliArgs) -> Result<(WorkspaceRoot, AppPaths), CliFailure> {
        let workspace_path = args
            .cwd
            .as_deref()
            .unwrap_or(self.environment.current_dir.as_path());
        let workspace = WorkspaceRoot::new(workspace_path)
            .map_err(|_| config_failure("workspace directory is invalid"))?;
        let paths = self.app_paths(args)?;
        create_private_directories(&paths)?;
        Ok((workspace, paths))
    }

    fn resolve_project_access(
        args: &CliArgs,
        surface: ClientSurface,
        workspace: &WorkspaceRoot,
        project_boundary: &Path,
        paths: &AppPaths,
    ) -> Result<ProjectAccess, CliFailure> {
        let has_project_resources =
            has_project_instructions(project_boundary, workspace.host_path())
                || [
                    workspace.host_path().join(".tea"),
                    workspace.host_path().join(".agents/skills"),
                ]
                .iter()
                .any(|path| path.exists());
        if args.trust == TrustArg::Default && !has_project_resources {
            return Ok(ProjectAccess::Ignored);
        }
        let mode = match surface {
            ClientSurface::Cli => InteractionMode::NonInteractive,
            ClientSurface::Tui => InteractionMode::Interactive,
        };
        ProjectTrustStore::new(paths.trust_file())
            .resolve(project_boundary, trust_request(args.trust), mode)
            .map_err(CliFailure::from)
    }

    async fn start_mcp_manager(
        &self,
        servers: &[(McpServerConfig, ToolTrust)],
        active_tools: &[ToolName],
    ) -> Result<Option<Arc<McpManager>>, CliFailure> {
        if servers.is_empty() {
            return Ok(None);
        }
        let mut launches = Vec::with_capacity(servers.len());
        for (server, trust) in servers {
            let environment = resolve_mcp_environment(server, self.mcp_environment_resolver())
                .map_err(|_| config_failure("MCP environment is unavailable"))?;
            launches.push(
                McpServerLaunch::new(server.clone(), *trust, environment.into_variables())
                    .map_err(mcp_failure)?,
            );
        }
        let observed_at = mcp_timestamp()?;
        let manager = McpManager::start(
            launches,
            active_tools.iter().cloned(),
            MAX_MCP_STARTUP_CONCURRENCY,
            observed_at,
        )
        .await
        .map_err(mcp_failure)?;
        Ok(Some(Arc::new(manager)))
    }

    fn app_paths(&self, args: &CliArgs) -> Result<AppPaths, CliFailure> {
        let home = self.environment.home_dir.as_deref();
        let config = path_override(
            args.config_dir.as_ref(),
            self.environment.value("TEA_CONFIG_DIR"),
            home.map(|path| path.join(".tea")),
        )?;
        let state = path_override(
            args.state_dir.as_ref(),
            self.environment.value("TEA_STATE_DIR"),
            home.map(|path| path.join(".local/state/tea")),
        )?;
        let data = path_override(
            args.data_dir.as_ref(),
            self.environment.value("TEA_DATA_DIR"),
            home.map(|path| path.join(".local/share/tea")),
        )?;
        AppPaths::new(config, state, data).map_err(CliFailure::from)
    }

    fn environment_settings(&self) -> SettingsLayer {
        SettingsLayer {
            provider: self.environment.value("TEA_PROVIDER").map(str::to_owned),
            model: self
                .environment
                .value("TEA_MODEL")
                .or_else(|| self.environment.value("TEA_OPENAI_MODEL"))
                .or_else(|| self.environment.value("TEA_ANTHROPIC_MODEL"))
                .map(str::to_owned),
            thinking: self
                .environment
                .value("TEA_REASONING_EFFORT")
                .or_else(|| self.environment.value("TEA_OPENAI_REASONING_EFFORT"))
                .map(str::to_owned),
            ..Default::default()
        }
    }
}

struct PreparedBootstrap {
    builder: CodingAgentBuilder,
    selection: SessionSelection,
    mcp_servers: Vec<(McpServerConfig, ToolTrust)>,
    active_mcp_tools: Vec<ToolName>,
    provider_notices: Vec<String>,
}

#[derive(Debug)]
struct UnavailableProvider {
    provider_id: ProviderId,
    models: Vec<ModelSpec>,
    message: String,
}

impl ModelProvider for UnavailableProvider {
    fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }

    fn models(&self) -> &[ModelSpec] {
        &self.models
    }

    fn stream(&self, _request: ModelRequest, _cancellation: ModelCancellation) -> BoxModelStream {
        let failure = ModelFailure::new(
            ModelFailureCode::Unavailable,
            self.message.clone(),
            RetryClass::Never,
        )
        .expect("bounded provider failure is valid");
        Box::pin(futures_util::stream::iter([
            ModelEvent::Started(ModelResponseInfo::new()),
            ModelEvent::Failed(failure),
        ]))
    }
}

fn unavailable_provider(
    settings: &CodingSettings,
    reason: &str,
) -> Result<Arc<dyn ModelProvider>, CliFailure> {
    let provider_id = ProviderId::from_str(&settings.provider)
        .map_err(|_| config_failure("provider selector is invalid"))?;
    let model = ModelSpec::new(
        ModelId::from_str(&settings.model)
            .map_err(|_| config_failure("model selector is invalid"))?,
        provider_id.clone(),
        ModelDisplayName::from_str("Unavailable model")
            .map_err(|_| internal_failure("model catalog failed"))?,
        TokenCount::new(128_000).map_err(|_| internal_failure("model catalog failed"))?,
        TokenCount::new(16_384).map_err(|_| internal_failure("model catalog failed"))?,
        ModelCapabilities::text().with_tools(true),
    )
    .map_err(|_| internal_failure("model catalog failed"))?;
    Ok(Arc::new(UnavailableProvider {
        provider_id,
        models: vec![model],
        message: format!("provider is unavailable: {reason}"),
    }))
}

fn configured_mcp_servers(
    settings: &CodingSettings,
    project: Option<&SettingsLayer>,
) -> Vec<(McpServerConfig, ToolTrust)> {
    let project_ids = project
        .and_then(|layer| layer.mcp_servers.as_ref())
        .into_iter()
        .flatten()
        .map(|server| server.id.as_str())
        .collect::<BTreeSet<_>>();
    settings
        .mcp_servers
        .iter()
        .cloned()
        .map(|server| {
            let trust = if project_ids.contains(server.id().as_str()) {
                ToolTrust::Workspace
            } else {
                ToolTrust::User
            };
            (server, trust)
        })
        .collect()
}

fn active_mcp_tools(
    settings: &CodingSettings,
    servers: &[(McpServerConfig, ToolTrust)],
) -> Result<Vec<ToolName>, CliFailure> {
    let aliases = servers
        .iter()
        .flat_map(|(server, _)| {
            server
                .tools()
                .iter()
                .filter_map(|tool| tool.resolved_alias(server.id()))
        })
        .collect::<BTreeSet<_>>();
    settings
        .active_tools
        .iter()
        .map(|alias| {
            ToolName::from_str(alias).map_err(|_| config_failure("active tool is invalid"))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|tools| {
            tools
                .into_iter()
                .filter(|tool| aliases.contains(tool))
                .collect()
        })
}

fn mcp_timestamp() -> Result<ProtocolTimestamp, CliFailure> {
    chrono::Utc::now()
        .to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
        .parse()
        .map_err(|_| internal_failure("MCP clock is unavailable"))
}

fn mcp_failure(error: tea_mcp::McpError) -> CliFailure {
    CliFailure::new(ExitCategory::TrustOrConfig, error.to_string())
}

fn load_project_settings(workspace: &WorkspaceRoot) -> Result<Option<SettingsLayer>, CliFailure> {
    match workspace.resolve_existing(".tea/settings.json") {
        Ok(path) => {
            workspace
                .revalidate_existing(&path)
                .map_err(|_| config_failure("project settings path changed"))?;
            let settings = load_settings_file(path.host_path()).map_err(CliFailure::from)?;
            workspace
                .revalidate_existing(&path)
                .map_err(|_| config_failure("project settings path changed"))?;
            Ok(settings)
        }
        Err(error) if error.code() == tea_coding_tools::WorkspacePathErrorCode::TargetNotFound => {
            Ok(None)
        }
        Err(_) => Err(config_failure("project settings path is invalid")),
    }
}

fn load_project_providers(workspace: &WorkspaceRoot) -> Result<ProvidersConfigLoad, CliFailure> {
    match workspace.resolve_existing(".tea/providers.json") {
        Ok(path) => {
            workspace
                .revalidate_existing(&path)
                .map_err(|_| config_failure("project provider path changed"))?;
            let providers = load_providers_file(path.host_path());
            workspace
                .revalidate_existing(&path)
                .map_err(|_| config_failure("project provider path changed"))?;
            Ok(providers)
        }
        Err(error) if error.code() == tea_coding_tools::WorkspacePathErrorCode::TargetNotFound => {
            Ok(ProvidersConfigLoad::default())
        }
        Err(_) => Err(config_failure("project provider path is invalid")),
    }
}

fn optional_project_directory(
    workspace: &WorkspaceRoot,
    path: &str,
) -> Result<Option<PathBuf>, CliFailure> {
    match workspace.resolve_existing(path) {
        Ok(path) if path.host_path().is_dir() => Ok(Some(path.host_path().to_path_buf())),
        Ok(_) => Err(config_failure("project resource path is not a directory")),
        Err(error) if error.code() == tea_coding_tools::WorkspacePathErrorCode::TargetNotFound => {
            Ok(None)
        }
        Err(_) => Err(config_failure("project resource path is invalid")),
    }
}

fn trust_request(value: TrustArg) -> TrustRequest {
    match value {
        TrustArg::Default => TrustRequest::Default,
        TrustArg::Once => TrustRequest::TrustOnce,
        TrustArg::Persist => TrustRequest::TrustPersisted,
        TrustArg::Reject => TrustRequest::Reject,
        TrustArg::Ignore => TrustRequest::Ignore,
    }
}

fn cli_settings(args: &CliArgs) -> Result<SettingsLayer, CliFailure> {
    let session_database = args
        .session_db
        .as_deref()
        .map(path_text)
        .transpose()?
        .map(str::to_owned);
    Ok(SettingsLayer {
        provider: args.provider.clone(),
        model: args.model.clone(),
        thinking: args
            .reasoning_effort
            .map(|effort| effort.as_str().to_owned()),
        active_tools: (!args.tools.is_empty()).then(|| args.tools.clone()),
        session_database,
        ..Default::default()
    })
}

fn session_database(
    args: &CliArgs,
    settings: &CodingSettings,
    paths: &AppPaths,
) -> Result<PathBuf, CliFailure> {
    let configured = args.session_db.as_ref().map_or_else(
        || settings.session_database.as_ref().map(PathBuf::from),
        |path| Some(path.clone()),
    );
    let path = configured.map_or_else(
        || paths.session_database(),
        |path| {
            if path.is_absolute() {
                path
            } else {
                paths.state_dir().join(path)
            }
        },
    );
    if let Some(parent) = path.parent() {
        create_private_directory(parent)?;
    }
    Ok(path)
}

fn path_override(
    argument: Option<&PathBuf>,
    environment: Option<&str>,
    fallback: Option<PathBuf>,
) -> Result<PathBuf, CliFailure> {
    argument
        .cloned()
        .or_else(|| environment.map(PathBuf::from))
        .or(fallback)
        .filter(|path| path.is_absolute())
        .ok_or_else(|| config_failure("absolute application paths are required"))
}

fn create_private_directories(paths: &AppPaths) -> Result<(), CliFailure> {
    for path in [paths.config_dir(), paths.state_dir(), paths.data_dir()] {
        create_private_directory(path)?;
    }
    Ok(())
}

fn create_private_directory(path: &Path) -> Result<(), CliFailure> {
    if path.exists() {
        return path
            .is_dir()
            .then_some(())
            .ok_or_else(|| config_failure("application directory is invalid"));
    }
    let mut builder = fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt as _;
        builder.mode(0o700);
    }
    builder
        .create(path)
        .map_err(|_| config_failure("application directory is unavailable"))
}

#[cfg(unix)]
fn shell(environment: &BootstrapEnvironment) -> Result<BashShell, CliFailure> {
    BashShell::new(
        environment.value("TEA_SHELL").unwrap_or("/bin/sh"),
        environment.value("TEA_SHELL_FLAG").unwrap_or("-c"),
    )
    .map_err(|_| config_failure("configured shell is invalid"))
}

#[cfg(windows)]
fn shell(environment: &BootstrapEnvironment) -> Result<BashShell, CliFailure> {
    let executable = environment
        .value("TEA_SHELL")
        .map(PathBuf::from)
        .or_else(|| environment.value("COMSPEC").map(PathBuf::from))
        .ok_or_else(|| config_failure("configured shell is unavailable"))?;
    BashShell::new(
        executable,
        environment.value("TEA_SHELL_FLAG").unwrap_or("/C"),
    )
    .map_err(|_| config_failure("configured shell is invalid"))
}

fn custom_openai_provider(
    provider_id: &str,
    model_id: &str,
    provider: &ProviderConfig,
    cli_api_key: Option<&tea_provider_openai::ApiKey>,
    environment: &BootstrapEnvironment,
    http_config: ProviderHttpConfig,
) -> Result<Arc<dyn ModelProvider>, CliFailure> {
    let resolved = resolve_openai_compatible_provider(
        provider_id,
        model_id,
        Some(provider),
        cli_api_key.map(tea_provider_openai::ApiKey::as_str),
        environment.values.clone(),
    )
    .map_err(CliFailure::from)?;
    Ok(Arc::new(
        resolved
            .provider_builder()
            .with_http_config(http_config)
            .build()
            .map_err(|_| provider_failure("provider configuration failed"))?,
    ))
}

type BuiltinProviderConstructor = fn(
    &str,
    Option<&tea_provider_openai::ApiKey>,
    &BootstrapEnvironment,
    ProviderHttpConfig,
) -> Result<Arc<dyn ModelProvider>, CliFailure>;

fn builtin_openai_provider(
    model_id: &str,
    cli_api_key: Option<&tea_provider_openai::ApiKey>,
    environment: &BootstrapEnvironment,
    http_config: ProviderHttpConfig,
) -> Result<Arc<dyn ModelProvider>, CliFailure> {
    let resolved = resolve_openai_compatible_provider(
        "openai",
        model_id,
        None,
        cli_api_key.map(tea_provider_openai::ApiKey::as_str),
        environment.values.clone(),
    )
    .map_err(CliFailure::from)?;
    Ok(Arc::new(
        resolved
            .provider_builder()
            .with_http_config(http_config)
            .build()
            .map_err(|_| provider_failure("provider configuration failed"))?,
    ))
}

fn builtin_anthropic_provider(
    model_id: &str,
    cli_api_key: Option<&tea_provider_openai::ApiKey>,
    environment: &BootstrapEnvironment,
    http_config: ProviderHttpConfig,
) -> Result<Arc<dyn ModelProvider>, CliFailure> {
    let mut values = environment.values.clone();
    values.insert("TEA_ANTHROPIC_MODEL".to_owned(), model_id.to_owned());
    if let Some(api_key) = cli_api_key {
        values.insert(
            "TEA_ANTHROPIC_API_KEY".to_owned(),
            api_key.as_str().to_owned(),
        );
    }
    let config = AnthropicCredentialResolver::new(values)
        .resolve()
        .map_err(|_| provider_failure("provider configuration failed"))?;
    Ok(Arc::new(
        AnthropicProviderBuilder::new()
            .with_config(Arc::new(config))
            .with_http_config(http_config)
            .build()
            .map_err(|_| provider_failure("provider configuration failed"))?,
    ))
}

fn client_web_search_provider(
    settings: &WebSearchSettings,
    environment: &BootstrapEnvironment,
    http: &ProviderHttpConfig,
) -> Result<Option<Arc<dyn SearchProvider>>, CliFailure> {
    if !settings.client.enabled {
        return Ok(None);
    }
    let Some(api_key) = environment
        .value(&settings.client.api_key_environment)
        .filter(|value| !value.trim().is_empty())
    else {
        return Ok(None);
    };
    let api_key = TavilyApiKey::new(api_key.to_owned())
        .map_err(|_| provider_failure("client web search credential is invalid"))?;
    let provider: Arc<dyn SearchProvider> = match settings.client.backend {
        WebSearchClientBackend::Tavily => {
            let config = TavilySearchConfig::new(
                &settings.client.endpoint,
                Duration::from_millis(settings.client.timeout_millis),
            )
            .map_err(|_| config_failure("client web search configuration is invalid"))?;
            Arc::new(
                TavilySearchProvider::new(config, api_key, http).map_err(|_| {
                    provider_failure("client web search provider initialization failed")
                })?,
            )
        }
    };
    Ok(Some(provider))
}

fn client_web_fetch_provider(
    settings: &WebFetchSettings,
    active_tools: &[String],
    workspace_id: &WorkspaceId,
    http: &ProviderHttpConfig,
) -> Result<Option<Arc<dyn FetchProvider>>, CliFailure> {
    if !settings.enabled || !active_tools.iter().any(|tool| tool == "web_fetch") {
        return Ok(None);
    }
    let cache_config = FetchCacheConfig::new(
        Duration::from_secs(settings.cache.ttl_seconds),
        settings.cache.max_entries,
        settings.cache.max_total_bytes,
        settings.cache.max_entry_bytes,
    )
    .map_err(|_| config_failure("client web fetch cache configuration is invalid"))?;
    let scope = FetchCacheScope::new(workspace_id.as_str(), "coding-agent")
        .map_err(|_| config_failure("client web fetch cache scope is invalid"))?;
    let cache = Arc::new(FetchResultCache::new(cache_config));
    let provider: Arc<dyn FetchProvider> = match settings.backend {
        WebFetchBackend::Http => Arc::new(HttpFetchProvider::production(scope, cache, http)),
    };
    Ok(Some(provider))
}

fn client_user_agent(surface: ClientSurface, environment: &BootstrapEnvironment) -> UserAgent {
    let version = env!("CARGO_PKG_VERSION");
    let os_info = os_info::get();
    let client = surface.name();
    let value = format!(
        "{client}/{version} ({} {}; {}) {} ({client}; {version})",
        os_info.os_type(),
        os_info.version(),
        os_info.architecture().unwrap_or("unknown"),
        terminal_user_agent_token(environment),
    );
    UserAgent::new(value).expect("built-in user agent is valid")
}

fn terminal_user_agent_token(environment: &BootstrapEnvironment) -> String {
    if let Some(program) = environment_value(environment, "TERM_PROGRAM") {
        return match environment_value(environment, "TERM_PROGRAM_VERSION") {
            Some(version) => format!("{program}/{version}"),
            None => program.to_owned(),
        };
    }
    if let Some(version) = environment_value(environment, "WEZTERM_VERSION") {
        return format!("WezTerm/{version}");
    }
    if has_environment_value(environment, "ITERM_SESSION_ID")
        || has_environment_value(environment, "ITERM_PROFILE")
        || has_environment_value(environment, "ITERM_PROFILE_NAME")
    {
        return "iTerm.app".to_owned();
    }
    if has_environment_value(environment, "TERM_SESSION_ID") {
        return "Apple_Terminal".to_owned();
    }
    if has_environment_value(environment, "KITTY_WINDOW_ID")
        || environment_value(environment, "TERM").is_some_and(|term| term.contains("kitty"))
    {
        return "kitty".to_owned();
    }
    if has_environment_value(environment, "ALACRITTY_SOCKET")
        || environment_value(environment, "TERM").is_some_and(|term| term == "alacritty")
    {
        return "Alacritty".to_owned();
    }
    if let Some(version) = environment_value(environment, "KONSOLE_VERSION") {
        return format!("Konsole/{version}");
    }
    if has_environment_value(environment, "GNOME_TERMINAL_SCREEN") {
        return "gnome-terminal".to_owned();
    }
    if let Some(version) = environment_value(environment, "VTE_VERSION") {
        return format!("VTE/{version}");
    }
    if has_environment_value(environment, "WT_SESSION") {
        return "WindowsTerminal".to_owned();
    }
    environment_value(environment, "TERM").map_or_else(|| "unknown".to_owned(), str::to_owned)
}

fn has_environment_value(environment: &BootstrapEnvironment, key: &str) -> bool {
    environment_value(environment, key).is_some()
}

fn environment_value<'a>(environment: &'a BootstrapEnvironment, key: &str) -> Option<&'a str> {
    environment
        .value(key)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(sanitize_user_agent_token)
        .filter(|value| !value.is_empty())
}

fn sanitize_user_agent_token(value: &str) -> &str {
    if value
        .bytes()
        .all(|byte| byte.is_ascii() && !byte.is_ascii_control())
    {
        value
    } else {
        ""
    }
}

fn workspace_id(path: &Path) -> Result<WorkspaceId, CliFailure> {
    let bytes = path_text(path)?.as_bytes();
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    WorkspaceId::from_str(&format!("workspace/{hash:016x}"))
        .map_err(|_| internal_failure("workspace identity failed"))
}

fn path_text(path: &Path) -> Result<&str, CliFailure> {
    path.to_str()
        .filter(|value| value.len() <= 4096 && !value.chars().any(char::is_control))
        .ok_or_else(|| config_failure("filesystem path is invalid"))
}

fn config_failure(message: &'static str) -> CliFailure {
    CliFailure::new(ExitCategory::TrustOrConfig, message)
}

fn provider_failure(message: &'static str) -> CliFailure {
    CliFailure::new(ExitCategory::Provider, message)
}

fn internal_failure(message: &'static str) -> CliFailure {
    CliFailure::new(ExitCategory::Internal, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser as _;
    use tea_model::{ModelCapabilities, ModelDisplayName, ModelSpec, ReasoningEffort};
    use tea_protocol::TokenCount;
    use tea_testkit::ScriptedModelProvider;

    fn model_ref(provider_id: &str, model_id: &str) -> tea_protocol::ModelRef {
        tea_protocol::ModelRef::new(provider_id.parse().unwrap(), model_id.parse().unwrap())
    }

    fn fake_provider() -> Arc<ScriptedModelProvider> {
        let provider_id = ProviderId::from_str("fake").unwrap();
        let model = ModelSpec::new(
            ModelId::from_str("fake/model").unwrap(),
            provider_id.clone(),
            ModelDisplayName::from_str("Fake Model").unwrap(),
            TokenCount::new(32_000).unwrap(),
            TokenCount::new(4_000).unwrap(),
            ModelCapabilities::text().with_tools(true),
        )
        .unwrap();
        Arc::new(ScriptedModelProvider::new(provider_id, vec![model], []))
    }

    fn test_environment(
        values: impl IntoIterator<Item = (&'static str, &'static str)>,
    ) -> BootstrapEnvironment {
        BootstrapEnvironment::new(
            ".",
            None,
            values
                .into_iter()
                .map(|(key, value)| (key.to_owned(), value.to_owned()))
                .collect(),
        )
    }

    #[test]
    fn user_agent_identifies_the_cli_and_tui_surfaces() {
        let environment = test_environment([
            ("TERM_PROGRAM", "iTerm.app"),
            ("TERM_PROGRAM_VERSION", "3.6.11"),
        ]);
        let os_info = os_info::get();
        let version = env!("CARGO_PKG_VERSION");

        for (surface, name) in [
            (ClientSurface::Cli, "tea-cli"),
            (ClientSurface::Tui, "tea-tui"),
        ] {
            let expected = format!(
                "{name}/{version} ({} {}; {}) iTerm.app/3.6.11 ({name}; {version})",
                os_info.os_type(),
                os_info.version(),
                os_info.architecture().unwrap_or("unknown"),
            );
            assert_eq!(
                client_user_agent(surface, &environment),
                UserAgent::new(expected).unwrap()
            );
        }
    }

    #[test]
    fn terminal_user_agent_token_follows_codex_environment_priority() {
        let environment = test_environment([
            ("TERM_PROGRAM", "iTerm.app"),
            ("TERM_PROGRAM_VERSION", "3.6.11"),
            ("WEZTERM_VERSION", "20250730"),
        ]);
        assert_eq!(terminal_user_agent_token(&environment), "iTerm.app/3.6.11");

        let environment = test_environment([("ITERM_SESSION_ID", "w0t0p0")]);
        assert_eq!(terminal_user_agent_token(&environment), "iTerm.app");

        let environment = test_environment([("TERM_PROGRAM", "invalid\r\nvalue")]);
        assert_eq!(terminal_user_agent_token(&environment), "unknown");
    }

    #[test]
    fn reasoning_effort_layers_are_provider_neutral_and_cli_wins() {
        let environment = test_environment([
            ("TEA_REASONING_EFFORT", "max"),
            ("TEA_OPENAI_REASONING_EFFORT", "low"),
        ]);
        let bootstrap = CliBootstrap::new(environment);
        assert_eq!(
            bootstrap.environment_settings().thinking.as_deref(),
            Some("max")
        );

        let args = CliArgs::try_parse_from(["tea", "--reasoning-effort", "xhigh"]).unwrap();
        let settings = merge_settings(
            CodingSettings::default(),
            Some(&SettingsLayer {
                thinking: Some("minimal".to_owned()),
                ..Default::default()
            }),
            None,
            Some(&bootstrap.environment_settings()),
            Some(&cli_settings(&args).unwrap()),
        )
        .unwrap();
        assert_eq!(settings.thinking, "xhigh");

        let legacy = test_environment([("TEA_OPENAI_REASONING_EFFORT", "high")]);
        let legacy = CliBootstrap::new(legacy);
        assert_eq!(
            legacy.environment_settings().thinking.as_deref(),
            Some("high")
        );
    }

    #[test]
    fn builtin_openai_bootstrap_preserves_the_adapter_reasoning_profile() {
        let root = std::env::temp_dir().join(format!(
            "tea-cli-openai-reasoning-bootstrap-{}",
            uuid::Uuid::now_v7().hyphenated()
        ));
        let config = root.join("config");
        let state = root.join("state");
        let data = root.join("data");
        fs::create_dir_all(&root).unwrap();
        let bootstrap = CliBootstrap::new(BootstrapEnvironment::new(
            &root,
            Some(root.clone()),
            [("TEA_OPENAI_API_KEY".to_owned(), "test-model-key".to_owned())]
                .into_iter()
                .collect(),
        ));
        let args = CliArgs::try_parse_from([
            "tea",
            "--no-session",
            "--trust",
            "ignore",
            "--cwd",
            root.to_str().unwrap(),
            "--config-dir",
            config.to_str().unwrap(),
            "--state-dir",
            state.to_str().unwrap(),
            "--data-dir",
            data.to_str().unwrap(),
        ])
        .unwrap();

        let (service, _) = bootstrap.build(&args).unwrap();
        let model = model_ref(&service.settings().provider, &service.settings().model);
        let profile = service
            .model_spec(&model)
            .unwrap()
            .reasoning_profile()
            .unwrap();
        assert_eq!(profile.default_effort(), ReasoningEffort::Medium);
        assert_eq!(
            profile.supported_efforts(),
            &[
                ReasoningEffort::Off,
                ReasoningEffort::Minimal,
                ReasoningEffort::Low,
                ReasoningEffort::Medium,
                ReasoningEffort::High,
            ]
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn client_web_search_provider_requires_enablement_and_an_injected_key() {
        let mut settings = tea_coding::config::WebSearchSettings::default();
        settings.client.api_key_environment = "SEARCH_TEST_KEY".to_owned();
        let environment = test_environment([("SEARCH_TEST_KEY", "test-search-key")]);
        let http = ProviderHttpConfig::new();

        assert!(
            client_web_search_provider(&settings, &environment, &http)
                .unwrap()
                .is_none()
        );

        settings.client.enabled = true;
        assert!(
            client_web_search_provider(&settings, &environment, &http)
                .unwrap()
                .is_some()
        );
        assert!(
            client_web_search_provider(&settings, &test_environment([]), &http)
                .unwrap()
                .is_none()
        );
        assert!(
            client_web_search_provider(
                &settings,
                &test_environment([("SEARCH_TEST_KEY", "   ")]),
                &http,
            )
            .unwrap()
            .is_none()
        );
    }

    #[test]
    fn bootstrap_injects_force_client_search_without_activating_it() {
        let root = std::env::temp_dir().join(format!(
            "tea-cli-client-search-bootstrap-{}",
            uuid::Uuid::now_v7().hyphenated()
        ));
        let config = root.join("config");
        let state = root.join("state");
        let data = root.join("data");
        fs::create_dir_all(&config).unwrap();
        fs::write(
            config.join("settings.json"),
            r#"{
                "schemaVersion": 1,
                "webSearch": {
                    "routePreference": "force_client",
                    "client": {
                        "enabled": true,
                        "apiKeyEnvironment": "SEARCH_BOOTSTRAP_KEY"
                    }
                }
            }"#,
        )
        .unwrap();
        let args = CliArgs::try_parse_from([
            "tea",
            "--no-session",
            "--trust",
            "ignore",
            "--cwd",
            root.to_str().unwrap(),
            "--config-dir",
            config.to_str().unwrap(),
            "--state-dir",
            state.to_str().unwrap(),
            "--data-dir",
            data.to_str().unwrap(),
        ])
        .unwrap();

        let missing = CliBootstrap::new(BootstrapEnvironment::new(
            &root,
            Some(root.clone()),
            [("TEA_OPENAI_API_KEY".to_owned(), "test-model-key".to_owned())]
                .into_iter()
                .collect(),
        ));
        let error = missing.build(&args).unwrap_err();
        assert_eq!(error.category(), ExitCategory::TrustOrConfig);
        assert_eq!(
            error.message(),
            "client web search requires an injected search provider"
        );

        let configured = CliBootstrap::new(BootstrapEnvironment::new(
            &root,
            Some(root.clone()),
            [
                ("TEA_OPENAI_API_KEY".to_owned(), "test-model-key".to_owned()),
                (
                    "SEARCH_BOOTSTRAP_KEY".to_owned(),
                    "test-search-key".to_owned(),
                ),
            ]
            .into_iter()
            .collect(),
        ));
        let (service, _) = configured.build(&args).unwrap();
        assert_eq!(
            service.settings().active_tools,
            ["read", "write", "edit", "bash"]
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn app_paths_use_dot_tea_and_preserve_config_override_precedence() {
        let home = std::env::temp_dir().join("tea-cli-app-paths-home");
        let env_config = home.join("env-config");
        let cli_config = home.join("cli-config");
        let args = CliArgs::try_parse_from(["tea"]).unwrap();
        let bootstrap = CliBootstrap::new(BootstrapEnvironment::new(
            &home,
            Some(home.clone()),
            BTreeMap::new(),
        ));

        let paths = bootstrap.app_paths(&args).unwrap();
        assert_eq!(paths.config_dir(), home.join(".tea"));
        assert_eq!(paths.state_dir(), home.join(".local/state/tea"));
        assert_eq!(paths.data_dir(), home.join(".local/share/tea"));

        let bootstrap = CliBootstrap::new(BootstrapEnvironment::new(
            &home,
            Some(home.clone()),
            [(
                "TEA_CONFIG_DIR".to_owned(),
                env_config.display().to_string(),
            )]
            .into_iter()
            .collect(),
        ));
        assert_eq!(bootstrap.app_paths(&args).unwrap().config_dir(), env_config);

        let args =
            CliArgs::try_parse_from(["tea", "--config-dir", cli_config.to_str().unwrap()]).unwrap();
        assert_eq!(bootstrap.app_paths(&args).unwrap().config_dir(), cli_config);
    }

    #[test]
    fn home_directory_resolution_supports_windows_environment() {
        let profile = std::ffi::OsString::from(r"C:\Users\tea");
        assert_eq!(
            home_dir_from(|key| (key == "USERPROFILE").then(|| profile.clone())),
            Some(PathBuf::from(&profile))
        );

        let home = std::ffi::OsString::from("/explicit/home");
        assert_eq!(
            home_dir_from(|key| match key {
                "HOME" => Some(home.clone()),
                "USERPROFILE" => Some(profile.clone()),
                _ => None,
            }),
            Some(PathBuf::from(home))
        );

        let drive = std::ffi::OsString::from("profile-root");
        let path = std::ffi::OsString::from("tea-user");
        assert_eq!(
            home_dir_from(|key| match key {
                "HOMEDRIVE" => Some(drive.clone()),
                "HOMEPATH" => Some(path.clone()),
                _ => None,
            }),
            Some(PathBuf::from(drive).join(path))
        );
    }

    #[test]
    fn tui_prompts_before_project_files_are_loaded_and_acceptance_persists() {
        let root = std::env::temp_dir().join(format!(
            "tea-cli-workspace-trust-bootstrap-{}",
            uuid::Uuid::now_v7().hyphenated()
        ));
        let config = root.join("config");
        let state = root.join("state");
        let data = root.join("data");
        fs::create_dir_all(root.join(".tea")).unwrap();
        fs::write(root.join(".tea/settings.json"), "not valid json").unwrap();
        let bootstrap = CliBootstrap::new(BootstrapEnvironment::new(
            &root,
            Some(root.clone()),
            BTreeMap::new(),
        ));
        let args = CliArgs::try_parse_from([
            "tea",
            "--no-session",
            "--cwd",
            root.to_str().unwrap(),
            "--config-dir",
            config.to_str().unwrap(),
            "--state-dir",
            state.to_str().unwrap(),
            "--data-dir",
            data.to_str().unwrap(),
        ])
        .unwrap();

        let prompt = bootstrap.workspace_trust_prompt(&args).unwrap().unwrap();
        assert_eq!(prompt.workspace(), fs::canonicalize(&root).unwrap());
        let error = bootstrap.prepare(&args, ClientSurface::Tui).err().unwrap();
        assert_eq!(error.category(), ExitCategory::TrustOrConfig);
        assert_eq!(error.message(), "workspace trust confirmation is required");

        CliBootstrap::accept_workspace_trust(&prompt).unwrap();
        assert!(bootstrap.workspace_trust_prompt(&args).unwrap().is_none());
        assert_eq!(
            ProjectTrustStore::new(state.join("project-trust.json"))
                .get(&root)
                .unwrap(),
            Some(PersistedTrustDecision::Trusted)
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bootstrap_inherits_repository_instructions_without_replacing_the_selected_workspace() {
        let root = std::env::temp_dir().join(format!(
            "tea-cli-project-boundary-bootstrap-{}",
            uuid::Uuid::now_v7().hyphenated()
        ));
        let workspace = root.join("src/nested");
        let config = root.join("config");
        let state = root.join("state");
        let data = root.join("data");
        fs::create_dir_all(root.join(".git")).unwrap();
        fs::create_dir_all(&workspace).unwrap();
        fs::write(root.join(".git/HEAD"), "ref: refs/heads/main\n").unwrap();
        fs::write(root.join("AGENTS.md"), "repository instructions").unwrap();
        let bootstrap = CliBootstrap::new(BootstrapEnvironment::new(
            &workspace,
            Some(root.clone()),
            BTreeMap::new(),
        ))
        .with_provider(fake_provider());
        let args = CliArgs::try_parse_from([
            "tea",
            "--no-session",
            "--provider",
            "fake",
            "--model",
            "fake/model",
            "--trust",
            "once",
            "--cwd",
            workspace.to_str().unwrap(),
            "--config-dir",
            config.to_str().unwrap(),
            "--state-dir",
            state.to_str().unwrap(),
            "--data-dir",
            data.to_str().unwrap(),
        ])
        .unwrap();

        let (service, _) = bootstrap.build(&args).unwrap();
        assert_eq!(service.resources().context().len(), 1);
        assert_eq!(
            service.resources().context()[0].content(),
            "repository instructions"
        );
        assert_eq!(
            service.resources().logical_workspace(),
            "<workspace>/src/nested"
        );
        assert_eq!(
            service.workspace().host_path(),
            fs::canonicalize(&workspace).unwrap()
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bootstrap_builds_anthropic_provider_from_injected_configuration() {
        let root = std::env::temp_dir().join(format!(
            "tea-cli-anthropic-bootstrap-{}",
            uuid::Uuid::now_v7().hyphenated()
        ));
        fs::create_dir_all(&root).unwrap();
        let config = root.join("config");
        let state = root.join("state");
        let data = root.join("data");
        let model = "claude-sonnet-4-20250514";
        let bootstrap = CliBootstrap::new(BootstrapEnvironment::new(
            &root,
            Some(root.clone()),
            [
                ("TEA_PROVIDER".to_owned(), "anthropic".to_owned()),
                (
                    "TEA_ANTHROPIC_API_KEY".to_owned(),
                    "test-anthropic-key".to_owned(),
                ),
                ("TEA_ANTHROPIC_MODEL".to_owned(), model.to_owned()),
            ]
            .into_iter()
            .collect(),
        ));
        let args = CliArgs::try_parse_from([
            "tea",
            "--no-session",
            "--provider",
            "anthropic",
            "--model",
            model,
            "--trust",
            "ignore",
            "--cwd",
            root.to_str().unwrap(),
            "--config-dir",
            config.to_str().unwrap(),
            "--state-dir",
            state.to_str().unwrap(),
            "--data-dir",
            data.to_str().unwrap(),
        ])
        .unwrap();

        let (service, _) = bootstrap.build(&args).unwrap();
        assert_eq!(service.settings().provider, "anthropic");
        assert_eq!(service.models(), &[model_ref("anthropic", model)]);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bootstrap_builds_custom_provider_with_cli_credential_precedence() {
        let root = std::env::temp_dir().join(format!(
            "tea-cli-custom-provider-bootstrap-{}",
            uuid::Uuid::now_v7().hyphenated()
        ));
        let config = root.join("config");
        let state = root.join("state");
        let data = root.join("data");
        fs::create_dir_all(&config).unwrap();
        fs::write(
            config.join("providers.json"),
            r#"{
                "providers": {
                    "ollama": {
                        "name": "Ollama",
                        "base_url": "http://localhost:11434/v1",
                        "api_key": "$MISSING_KEY",
                        "api_mode": "responses",
                        "reasoning_effort": "max",
                        "vision": true,
                        "models": [{
                            "id": "llama3.1:8b",
                            "display_name": "Llama 3.1 8B",
                            "context_window_tokens": 64000,
                            "max_output_tokens": 8000,
                            "capabilities": {
                                "hosted_tools": ["web_search"]
                            }
                        }]
                    }
                }
            }"#,
        )
        .unwrap();
        let bootstrap = CliBootstrap::new(BootstrapEnvironment::new(
            &root,
            Some(root.clone()),
            BTreeMap::new(),
        ));
        let args = CliArgs::try_parse_from([
            "tea",
            "--no-session",
            "--provider",
            "ollama",
            "--model",
            "llama3.1:8b",
            "--api-key",
            "cli-secret-key",
            "--trust",
            "ignore",
            "--cwd",
            root.to_str().unwrap(),
            "--config-dir",
            config.to_str().unwrap(),
            "--state-dir",
            state.to_str().unwrap(),
            "--data-dir",
            data.to_str().unwrap(),
        ])
        .unwrap();
        assert!(!format!("{args:?}").contains("cli-secret-key"));

        let (service, _) = bootstrap.build(&args).unwrap();
        assert_eq!(service.settings().provider, "ollama");
        assert_eq!(service.models(), &[model_ref("ollama", "llama3.1:8b")]);
        let model = model_ref("ollama", "llama3.1:8b");
        let capabilities = service.model_capabilities(&model).unwrap();
        assert!(capabilities.accepts_images());
        assert!(capabilities.supports_reasoning());
        assert!(capabilities.supports_hosted_tool(tea_model::HostedToolKind::WebSearch));
        assert_eq!(
            service
                .model_spec(&model)
                .unwrap()
                .reasoning_profile()
                .unwrap()
                .default_effort(),
            ReasoningEffort::High
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bootstrap_registers_all_available_custom_providers_with_same_model_id() {
        let root = std::env::temp_dir().join(format!(
            "tea-cli-multi-provider-bootstrap-{}",
            uuid::Uuid::now_v7().hyphenated()
        ));
        let config = root.join("config");
        let state = root.join("state");
        let data = root.join("data");
        fs::create_dir_all(&config).unwrap();
        fs::write(
            config.join("providers.json"),
            r#"{
                "providers": {
                    "one": {
                        "api_key": "key-one",
                        "models": [{"id": "shared"}]
                    },
                    "two": {
                        "api_key": "key-two",
                        "models": [{"id": "shared"}]
                    }
                }
            }"#,
        )
        .unwrap();
        let bootstrap = CliBootstrap::new(BootstrapEnvironment::new(
            &root,
            Some(root.clone()),
            BTreeMap::new(),
        ));
        let args = CliArgs::try_parse_from([
            "tea",
            "--no-session",
            "--provider",
            "one",
            "--model",
            "shared",
            "--trust",
            "ignore",
            "--cwd",
            root.to_str().unwrap(),
            "--config-dir",
            config.to_str().unwrap(),
            "--state-dir",
            state.to_str().unwrap(),
            "--data-dir",
            data.to_str().unwrap(),
        ])
        .unwrap();

        let (service, _) = bootstrap.build(&args).unwrap();
        assert_eq!(
            service.models(),
            &[model_ref("one", "shared"), model_ref("two", "shared")]
        );
        assert!(service.model_spec(&model_ref("one", "shared")).is_some());
        assert!(service.model_spec(&model_ref("two", "shared")).is_some());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn bootstrap_rejects_active_hosted_tool_missing_from_custom_model_capabilities() {
        let root = std::env::temp_dir().join(format!(
            "tea-cli-custom-hosted-capability-bootstrap-{}",
            uuid::Uuid::now_v7().hyphenated()
        ));
        let config = root.join("config");
        let state = root.join("state");
        let data = root.join("data");
        fs::create_dir_all(&config).unwrap();
        fs::write(
            config.join("settings.json"),
            r#"{
                "schemaVersion": 1,
                "provider": "gateway",
                "model": "gpt-test",
                "activeTools": ["web_search"]
            }"#,
        )
        .unwrap();
        fs::write(
            config.join("providers.json"),
            r#"{
                "providers": {
                    "gateway": {
                        "base_url": "https://gateway.example/v1",
                        "api_key": "test-key",
                        "api_mode": "responses",
                        "models": [{"id": "gpt-test"}]
                    }
                }
            }"#,
        )
        .unwrap();
        let bootstrap = CliBootstrap::new(BootstrapEnvironment::new(
            &root,
            Some(root.clone()),
            BTreeMap::new(),
        ));
        let args = CliArgs::try_parse_from([
            "tea",
            "--no-session",
            "--trust",
            "ignore",
            "--cwd",
            root.to_str().unwrap(),
            "--config-dir",
            config.to_str().unwrap(),
            "--state-dir",
            state.to_str().unwrap(),
            "--data-dir",
            data.to_str().unwrap(),
        ])
        .unwrap();

        let error = bootstrap.build(&args).unwrap_err();
        assert_eq!(error.category(), ExitCategory::TrustOrConfig);
        assert_eq!(
            error.message(),
            "active tool web_search has no execution route supported by selected model gpt-test; declare the model capability or configure a supported client route"
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn tui_defers_missing_custom_provider_credentials() {
        let root = std::env::temp_dir().join(format!(
            "tea-cli-unavailable-provider-bootstrap-{}",
            uuid::Uuid::now_v7().hyphenated()
        ));
        let config = root.join("config");
        let state = root.join("state");
        let data = root.join("data");
        fs::create_dir_all(&config).unwrap();
        fs::write(
            config.join("providers.json"),
            r#"{
                "providers": {
                    "local": {
                        "api_key": "$MISSING_TUI_PROVIDER_KEY",
                        "models": [{"id": "local-model"}]
                    }
                }
            }"#,
        )
        .unwrap();
        let bootstrap = CliBootstrap::new(BootstrapEnvironment::new(
            &root,
            Some(root.clone()),
            BTreeMap::new(),
        ));
        let args = CliArgs::try_parse_from([
            "tea",
            "--no-session",
            "--provider",
            "local",
            "--model",
            "local-model",
            "--trust",
            "ignore",
            "--cwd",
            root.to_str().unwrap(),
            "--config-dir",
            config.to_str().unwrap(),
            "--state-dir",
            state.to_str().unwrap(),
            "--data-dir",
            data.to_str().unwrap(),
        ])
        .unwrap();

        let headless_error = bootstrap.build(&args).unwrap_err();
        assert_eq!(headless_error.category(), ExitCategory::Provider);
        let (service, _, notices) = bootstrap.build_tui_async(&args).await.unwrap();
        assert_eq!(service.models(), &[model_ref("local", "local-model")]);
        assert_eq!(
            notices,
            ["provider unavailable: configured provider API key could not be resolved"]
        );
        service.shutdown().await;
        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn tui_defers_custom_provider_model_mismatch() {
        let root = std::env::temp_dir().join(format!(
            "tea-cli-unconfigured-model-bootstrap-{}",
            uuid::Uuid::now_v7().hyphenated()
        ));
        let config = root.join("config");
        let state = root.join("state");
        let data = root.join("data");
        fs::create_dir_all(&config).unwrap();
        fs::write(
            config.join("providers.json"),
            r#"{
                "providers": {
                    "local": {
                        "api_key": "configured-secret",
                        "models": [{"id": "available-model"}]
                    }
                }
            }"#,
        )
        .unwrap();
        let bootstrap = CliBootstrap::new(BootstrapEnvironment::new(
            &root,
            Some(root.clone()),
            BTreeMap::new(),
        ));
        let args = CliArgs::try_parse_from([
            "tea",
            "--no-session",
            "--provider",
            "local",
            "--model",
            "missing-model",
            "--trust",
            "ignore",
            "--cwd",
            root.to_str().unwrap(),
            "--config-dir",
            config.to_str().unwrap(),
            "--state-dir",
            state.to_str().unwrap(),
            "--data-dir",
            data.to_str().unwrap(),
        ])
        .unwrap();

        let headless_error = bootstrap.build(&args).unwrap_err();
        assert_eq!(headless_error.category(), ExitCategory::TrustOrConfig);
        assert_eq!(
            headless_error.message(),
            "model is not configured for provider"
        );
        let (service, _, notices) = bootstrap.build_tui_async(&args).await.unwrap();
        assert_eq!(service.models(), &[model_ref("local", "missing-model")]);
        assert_eq!(
            notices,
            ["provider unavailable: model is not configured for provider"]
        );
        assert!(
            notices
                .iter()
                .all(|notice| !notice.contains("configured-secret"))
        );
        service.shutdown().await;
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn project_provider_configuration_requires_trust() {
        let root = std::env::temp_dir().join(format!(
            "tea-cli-project-provider-bootstrap-{}",
            uuid::Uuid::now_v7().hyphenated()
        ));
        let config = root.join("config");
        let state = root.join("state");
        let data = root.join("data");
        fs::create_dir_all(root.join(".tea")).unwrap();
        fs::write(
            root.join(".tea/providers.json"),
            r#"{"providers":{"local":{"api_key":"local","models":[{"id":"local-model"}]}}}"#,
        )
        .unwrap();
        let bootstrap = CliBootstrap::new(BootstrapEnvironment::new(
            &root,
            Some(root.clone()),
            BTreeMap::new(),
        ));
        let common = [
            "tea",
            "--no-session",
            "--provider",
            "local",
            "--model",
            "local-model",
            "--cwd",
            root.to_str().unwrap(),
            "--config-dir",
            config.to_str().unwrap(),
            "--state-dir",
            state.to_str().unwrap(),
            "--data-dir",
            data.to_str().unwrap(),
        ];
        let ignored =
            CliArgs::try_parse_from(common.into_iter().chain(["--trust", "ignore"])).unwrap();
        assert!(bootstrap.build(&ignored).is_err());
        let trusted =
            CliArgs::try_parse_from(common.into_iter().chain(["--trust", "once"])).unwrap();
        let (service, _) = bootstrap.build(&trusted).unwrap();
        assert_eq!(service.settings().provider, "local");
        fs::remove_dir_all(root).unwrap();
    }
}
