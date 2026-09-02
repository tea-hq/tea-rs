use std::str::FromStr;
use std::sync::Arc;
use std::time::Duration;

use tea::{AgentRuntimeBuilder, AgentToolRegistration};
use tea_coding_tools::{
    BashConfig, BashTool, EditTool, FetchProvider, FindTool, GrepTool, LsTool, ReadTool,
    SearchProvider, WebFetchTool, WebSearchTool, WorkspaceFileResourceResolver, WorkspaceRoot,
    WriteTool,
};
use tea_context::ContextProvider;
use tea_kernel::ModelRetryPolicy;
use tea_mcp::McpManager;
use tea_model::{HostedToolOptions, ModelProvider, ReasoningEffort};
use tea_policy::{
    ActorId, CodingWorkspacePolicy, ExecutionSurface, ExternalSourcePolicy, WorkspaceId,
};
use tea_profile::ProfileRuleId;
use tea_session::{SessionCatalog, SessionStore};
use tea_tools::{
    ArgumentResourceResolver, StaticResourceResolver, ToolBinding, ToolExecutor,
    ToolResourceAccess, ToolResourceResolver, ToolRoutePreference, ToolSpec,
};

use crate::config::{
    CodingSettings, WebFetchSettings, WebSearchRoutePreference, WebSearchSettings,
};
use crate::mcp_policy::CodingMcpPolicy;
use crate::profile::coding_profile;
use crate::resources::ResourceCatalog;
use crate::skill_tool::{ReadSkillResourceTool, SkillResourceResolver};
use crate::system_prompt::CodingSystemPromptBuilder;
use crate::{CodingAgentService, CodingError, CodingErrorCode};

/// Product builder assembling one mode-neutral coding service.
#[derive(Debug)]
pub struct CodingAgentBuilder {
    providers: Vec<Arc<dyn ModelProvider>>,
    workspace: WorkspaceRoot,
    resources: Arc<ResourceCatalog>,
    store: Arc<dyn SessionStore>,
    catalog: Arc<dyn SessionCatalog>,
    bash: BashConfig,
    settings: CodingSettings,
    actor: ActorId,
    workspace_id: WorkspaceId,
    execution_surface: ExecutionSurface,
    compaction_summarizer: Option<Arc<dyn tea_kernel::CompactionSummarizer>>,
    mcp_manager: Option<Arc<McpManager>>,
    search_provider: Option<Arc<dyn SearchProvider>>,
    fetch_provider: Option<Arc<dyn FetchProvider>>,
    context_providers: Vec<Arc<dyn ContextProvider>>,
    host_tools: Vec<AgentToolRegistration>,
}

impl CodingAgentBuilder {
    /// Creates a fully explicit, test-isolated builder.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new<S>(
        provider: Arc<dyn ModelProvider>,
        workspace: WorkspaceRoot,
        resources: ResourceCatalog,
        store: Arc<S>,
        bash: BashConfig,
        settings: CodingSettings,
        actor: ActorId,
        workspace_id: WorkspaceId,
    ) -> Self
    where
        S: SessionStore + SessionCatalog + 'static,
    {
        Self::without_provider(
            workspace,
            resources,
            store,
            bash,
            settings,
            actor,
            workspace_id,
        )
        .provider(provider)
    }

    /// Creates a service whose model providers will be registered dynamically.
    ///
    /// The resulting service can create and restore sessions immediately. A
    /// prompt fails with an unknown-provider error until the selected model's
    /// provider is registered through [`CodingAgentService`](crate::CodingAgentService).
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn without_provider<S>(
        workspace: WorkspaceRoot,
        resources: ResourceCatalog,
        store: Arc<S>,
        bash: BashConfig,
        settings: CodingSettings,
        actor: ActorId,
        workspace_id: WorkspaceId,
    ) -> Self
    where
        S: SessionStore + SessionCatalog + 'static,
    {
        let session_store: Arc<dyn SessionStore> = store.clone();
        let session_catalog: Arc<dyn SessionCatalog> = store;
        Self {
            providers: Vec::new(),
            workspace,
            resources: Arc::new(resources),
            store: session_store,
            catalog: session_catalog,
            bash,
            settings,
            actor,
            workspace_id,
            execution_surface: ExecutionSurface::Cli,
            compaction_summarizer: None,
            mcp_manager: None,
            search_provider: None,
            fetch_provider: None,
            context_providers: Vec::new(),
            host_tools: Vec::new(),
        }
    }

    /// Selects the outward product surface used for policy evaluation.
    ///
    /// The default remains [`ExecutionSurface::Cli`] for compatibility with
    /// existing coding CLI hosts. Desktop and IDE embeddings must set their
    /// actual surface before building the service.
    #[must_use]
    pub const fn execution_surface(mut self, execution_surface: ExecutionSurface) -> Self {
        self.execution_surface = execution_surface;
        self
    }

    /// Registers a model provider in the initial runtime generation.
    #[must_use]
    pub fn provider(mut self, provider: Arc<dyn ModelProvider>) -> Self {
        self.providers.push(provider);
        self
    }

    /// Supplies the required summarizer when automatic compaction is enabled.
    #[must_use]
    pub fn compaction_summarizer(
        mut self,
        summarizer: Arc<dyn tea_kernel::CompactionSummarizer>,
    ) -> Self {
        self.compaction_summarizer = Some(summarizer);
        self
    }

    /// Injects a manager initialized before immutable runtime binding.
    ///
    /// The resulting service owns the manager and must be shut down normally
    /// to await its child processes. The manager's catalog is never refreshed
    /// during this service lifetime.
    #[must_use]
    pub fn mcp_manager(mut self, manager: Arc<McpManager>) -> Self {
        self.mcp_manager = Some(manager);
        self
    }

    /// Injects a real client web-search backend.
    ///
    /// Injection alone does not enable the client route and never adds
    /// `web_search` to the active-tool allowlist. Both remain controlled by
    /// explicit product settings.
    #[must_use]
    pub fn search_provider(mut self, provider: Arc<dyn SearchProvider>) -> Self {
        self.search_provider = Some(provider);
        self
    }

    /// Injects a real client web-fetch backend.
    ///
    /// Injection alone never registers or activates `web_fetch`; both the
    /// active-tool allowlist and explicit backend enablement remain required.
    #[must_use]
    pub fn fetch_provider(mut self, provider: Arc<dyn FetchProvider>) -> Self {
        self.fetch_provider = Some(provider);
        self
    }

    /// Registers a typed context provider evaluated before every prompt compilation.
    ///
    /// Providers receive immutable session/run/tool context and return bounded
    /// `PromptModule` values. Registration is additive and cannot bypass the
    /// prompt compiler or replace required coding modules.
    #[must_use]
    pub fn context_provider(mut self, provider: Arc<dyn ContextProvider>) -> Self {
        self.context_providers.push(provider);
        self
    }

    /// Registers one product-owned host tool in the frozen runtime catalog.
    ///
    /// The tool remains inactive until a host explicitly selects its name for
    /// an individual session through [`CodingAgentService::set_active_tools`].
    #[must_use]
    pub fn host_tool(
        mut self,
        spec: ToolSpec,
        resolver: Arc<dyn ToolResourceResolver>,
        executor: Arc<dyn ToolExecutor>,
    ) -> Self {
        self.host_tools.push((spec, resolver, executor));
        self
    }

    /// Assembles provider, profile, policy, context, registered tools, and session ports.
    ///
    /// # Errors
    ///
    /// Returns a bounded product error for any invalid contract or registration.
    pub fn build(self) -> Result<CodingAgentService, CodingError> {
        crate::config::validate(&self.settings)?;
        let has_skills = !self.resources.skills().is_empty();
        let profile = coding_profile(&self.settings, self.execution_surface, has_skills)?;
        let retry_attempts = self.settings.max_retries.saturating_add(1);
        let default_reasoning_effort =
            ReasoningEffort::from_str(&self.settings.thinking).map_err(|_| invalid_settings())?;
        let mut builder = AgentRuntimeBuilder::new()
            .session_store(self.store)
            .session_catalog(self.catalog)
            .actor(self.actor)
            .workspace(self.workspace_id.clone())
            .default_reasoning_effort(default_reasoning_effort)
            .retry_policy(
                ModelRetryPolicy::new(
                    retry_attempts,
                    Duration::from_millis(self.settings.retry_base_delay_ms),
                    Duration::from_millis(self.settings.retry_max_delay_ms),
                )
                .map_err(|_| runtime_error())?,
            )
            .profile(profile)
            .builtin_tool_hints(false)
            .context_provider(Arc::new(
                CodingSystemPromptBuilder::new(
                    self.resources.logical_workspace(),
                    self.resources.context().to_vec(),
                    self.resources.skill_metadata(),
                )
                .map_err(|_| runtime_error())?
                .with_prompt_resources(
                    self.resources.system_prompt().cloned(),
                    self.resources.append_system_prompt().cloned(),
                ),
            ));
        for context_provider in self.context_providers {
            builder = builder.context_provider(context_provider);
        }
        for provider in self.providers {
            builder = builder.provider(provider);
        }
        if self.settings.compaction_enabled {
            let summarizer = self.compaction_summarizer.ok_or_else(runtime_error)?;
            builder = builder
                .compaction_policy(Arc::new(OverflowCompactionPolicy))
                .compaction_summarizer(summarizer);
        }
        builder = builder
            .policy_rule(
                ProfileRuleId::from_str("platform.external_source").map_err(|_| runtime_error())?,
                Arc::new(ExternalSourcePolicy),
            )
            .map_err(|_| runtime_error())?
            .policy_rule(
                ProfileRuleId::from_str("product.coding_workspace").map_err(|_| runtime_error())?,
                Arc::new(CodingWorkspacePolicy),
            )
            .map_err(|_| runtime_error())?
            .policy_rule(
                ProfileRuleId::from_str("product.coding_mcp").map_err(|_| runtime_error())?,
                Arc::new(CodingMcpPolicy),
            )
            .map_err(|_| runtime_error())?;
        builder = register_native_tools(
            builder,
            &self.workspace,
            self.bash,
            Arc::clone(&self.resources),
        )?;
        builder = register_web_search(builder, &self.settings.web_search, self.search_provider)?;
        builder = register_web_fetch(
            builder,
            &self.settings.active_tools,
            &self.settings.web_fetch,
            self.fetch_provider,
        )?;
        builder = register_mcp_tools(builder, self.mcp_manager.as_deref())?;
        for (spec, resolver, executor) in self.host_tools {
            builder = builder
                .tool(spec, resolver, executor)
                .map_err(|_| runtime_error())?;
        }
        let runtime = builder.build().map_err(CodingError::from)?;
        Ok(CodingAgentService::new(
            runtime,
            self.workspace,
            self.resources,
            self.settings,
            self.workspace_id,
            self.mcp_manager,
        ))
    }
}

fn register_web_fetch(
    builder: AgentRuntimeBuilder,
    active_tools: &[String],
    settings: &WebFetchSettings,
    provider: Option<Arc<dyn FetchProvider>>,
) -> Result<AgentRuntimeBuilder, CodingError> {
    if !active_tools.iter().any(|tool| tool == "web_fetch") {
        return Ok(builder);
    }
    if !settings.enabled {
        return Err(CodingError::new(
            CodingErrorCode::InvalidInput,
            "active web fetch requires an enabled client backend",
        ));
    }
    let provider = provider.ok_or_else(|| {
        CodingError::new(
            CodingErrorCode::InvalidInput,
            "active web fetch requires an injected fetch provider",
        )
    })?;
    let resolver = ArgumentResourceResolver::new("url", "url", ToolResourceAccess::Read)
        .map_err(|_| invalid_settings())?;
    builder
        .tool(
            WebFetchTool::spec().map_err(|_| runtime_error())?,
            Arc::new(resolver),
            Arc::new(WebFetchTool::new(provider)),
        )
        .map_err(|_| runtime_error())
}

fn register_web_search(
    builder: AgentRuntimeBuilder,
    settings: &WebSearchSettings,
    provider: Option<Arc<dyn SearchProvider>>,
) -> Result<AgentRuntimeBuilder, CodingError> {
    let options = settings.runtime_options().map_err(|_| invalid_settings())?;
    let hosted = HostedToolOptions::WebSearch(options.clone());
    let spec = WebSearchTool::spec().map_err(|_| runtime_error())?;
    if settings.client.enabled {
        if let Some(provider) = provider {
            let tool = WebSearchTool::new(provider, options);
            let resolver = Arc::new(
                StaticResourceResolver::new([tool.resource().map_err(|_| invalid_settings())?])
                    .map_err(|_| invalid_settings())?,
            );
            let preference = match settings.route_preference {
                WebSearchRoutePreference::PreferHosted => ToolRoutePreference::PreferHosted,
                WebSearchRoutePreference::ForceClient => ToolRoutePreference::ForceClient,
            };
            return builder
                .tool_binding(
                    spec,
                    ToolBinding::hybrid(hosted, preference, resolver, Arc::new(tool)),
                )
                .map_err(|_| runtime_error());
        }
        if settings.route_preference == WebSearchRoutePreference::ForceClient {
            return Err(CodingError::new(
                CodingErrorCode::InvalidInput,
                "client web search requires an injected search provider",
            ));
        }
    }
    builder
        .tool_binding(spec, ToolBinding::hosted(hosted))
        .map_err(|_| runtime_error())
}

fn register_native_tools(
    mut builder: AgentRuntimeBuilder,
    workspace: &WorkspaceRoot,
    bash: BashConfig,
    resources: Arc<ResourceCatalog>,
) -> Result<AgentRuntimeBuilder, CodingError> {
    builder = builder
        .tool(
            ReadTool::spec().map_err(|_| runtime_error())?,
            Arc::new(WorkspaceFileResourceResolver::new(ToolResourceAccess::Read)),
            Arc::new(ReadTool::new(workspace.clone())),
        )
        .map_err(|_| runtime_error())?;
    builder = builder
        .tool(
            GrepTool::spec().map_err(|_| runtime_error())?,
            Arc::new(WorkspaceFileResourceResolver::new(ToolResourceAccess::Read)),
            Arc::new(GrepTool::new(workspace.clone())),
        )
        .map_err(|_| runtime_error())?;
    builder = builder
        .tool(
            FindTool::spec().map_err(|_| runtime_error())?,
            Arc::new(WorkspaceFileResourceResolver::new(ToolResourceAccess::Read)),
            Arc::new(FindTool::new(workspace.clone())),
        )
        .map_err(|_| runtime_error())?;
    builder = builder
        .tool(
            LsTool::spec().map_err(|_| runtime_error())?,
            Arc::new(WorkspaceFileResourceResolver::new(ToolResourceAccess::Read)),
            Arc::new(LsTool::new(workspace.clone())),
        )
        .map_err(|_| runtime_error())?;
    builder = builder
        .tool(
            WriteTool::spec().map_err(|_| runtime_error())?,
            Arc::new(WorkspaceFileResourceResolver::new(
                ToolResourceAccess::Write,
            )),
            Arc::new(WriteTool::new(workspace.clone())),
        )
        .map_err(|_| runtime_error())?;
    builder = builder
        .tool(
            EditTool::spec().map_err(|_| runtime_error())?,
            Arc::new(WorkspaceFileResourceResolver::read_write()),
            Arc::new(EditTool::new(workspace.clone())),
        )
        .map_err(|_| runtime_error())?;
    builder = builder
        .tool(
            BashTool::spec().map_err(|_| runtime_error())?,
            Arc::new(
                StaticResourceResolver::new([
                    BashTool::workspace_resource().map_err(|_| runtime_error())?
                ])
                .map_err(|_| runtime_error())?,
            ),
            Arc::new(BashTool::new(workspace.clone(), bash)),
        )
        .map_err(|_| runtime_error())?;
    if !resources.skills().is_empty() {
        builder = builder
            .tool(
                ReadSkillResourceTool::spec().map_err(|_| runtime_error())?,
                Arc::new(SkillResourceResolver),
                Arc::new(ReadSkillResourceTool::new(resources)),
            )
            .map_err(|_| runtime_error())?;
    }
    Ok(builder)
}

fn register_mcp_tools(
    mut builder: AgentRuntimeBuilder,
    manager: Option<&McpManager>,
) -> Result<AgentRuntimeBuilder, CodingError> {
    let Some(manager) = manager else {
        return Ok(builder);
    };
    for binding in manager.catalog().bindings() {
        let executor = manager
            .tool_executor(binding.spec().name())
            .map_err(|_| runtime_error())?;
        builder = builder
            .register_tool(
                binding.spec().clone(),
                Arc::new(binding.clone()),
                Arc::new(executor),
            )
            .map_err(|_| runtime_error())?;
    }
    Ok(builder)
}

#[derive(Debug)]
struct OverflowCompactionPolicy;

impl tea_kernel::CompactionPolicy for OverflowCompactionPolicy {
    fn should_compact(&self, estimated_input_tokens: usize, context_window: u64) -> bool {
        u64::try_from(estimated_input_tokens).is_ok_and(|tokens| tokens >= context_window)
    }
}

fn runtime_error() -> CodingError {
    CodingError::new(CodingErrorCode::Runtime, "coding runtime assembly failed")
}

fn invalid_settings() -> CodingError {
    CodingError::new(CodingErrorCode::InvalidInput, "coding settings are invalid")
}
