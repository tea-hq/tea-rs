use std::sync::Arc;

use std::str::FromStr;
use tea::{AgentRuntimeBuilder, RuntimeErrorCode};
use tea_model::{
    HostedToolKind, HostedToolOptions, ModelCapabilities, ModelDisplayName, ModelSpec, ProviderId,
    WebSearchOptions,
};
use tea_policy::{
    ActorId, CodingWorkspacePolicy, DesktopPolicy, ExecutionSurface, PolicyEnvironment,
    PolicyExecutionTarget,
};
use tea_profile::ProfileRuleId;
use tea_protocol::{ModelId, ModelRef, ProtocolMetadata, TokenCount, ToolIdempotency};
use tea_testkit::{FakeReadTool, FakeWriteTool, ScriptedModelProvider};
use tea_tools::{
    ArgumentResourceResolver, ToolBinding, ToolConcurrency, ToolEffect, ToolExecutionSemantics,
    ToolName, ToolResourceAccess, ToolRoutePreference, ToolSource, ToolSourceKind, ToolSpec,
    ToolTimeout, ToolTrust, ToolVersion,
};

fn provider_with_capabilities(capabilities: ModelCapabilities) -> Arc<ScriptedModelProvider> {
    let provider_id = ProviderId::from_str("fake").unwrap();
    let model = ModelSpec::new(
        ModelId::from_str("fake/model").unwrap(),
        provider_id.clone(),
        ModelDisplayName::from_str("Fake Model").unwrap(),
        TokenCount::new(32_000).unwrap(),
        TokenCount::new(4_000).unwrap(),
        capabilities,
    )
    .unwrap();
    Arc::new(ScriptedModelProvider::new(provider_id, vec![model], []))
}

fn provider() -> Arc<ScriptedModelProvider> {
    provider_with_capabilities(ModelCapabilities::text().with_tools(true))
}

fn model_ref(model_id: &str) -> ModelRef {
    ModelRef::new("fake".parse().unwrap(), model_id.parse().unwrap())
}

fn environment(surface: ExecutionSurface) -> PolicyEnvironment {
    PolicyEnvironment::new(
        surface,
        PolicyExecutionTarget::Native,
        ProtocolMetadata::default(),
    )
}

fn spec(name: &str, effect: ToolEffect, idempotency: ToolIdempotency) -> ToolSpec {
    ToolSpec::new(
        ToolName::from_str(name).unwrap(),
        ToolVersion::from_str("1.0.0").unwrap(),
        format!("Deterministic {name}."),
        serde_json::json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}),
        serde_json::json!({"type":"object","properties":{"content":{"type":"string"}},"required":["content"]}),
        [effect],
        ToolExecutionSemantics::new(
            idempotency,
            if idempotency == ToolIdempotency::Idempotent {
                tea_tools::ToolRetrySafety::Automatic
            } else {
                tea_tools::ToolRetrySafety::ExplicitOnly
            },
            ToolConcurrency::Serial,
            ToolTimeout::from_millis(1_000).unwrap(),
        )
        .unwrap(),
    )
    .unwrap()
}

fn user_source(source_id: &str) -> ToolSource {
    ToolSource::new(
        ToolSourceKind::Native,
        source_id,
        ToolTrust::User,
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    )
    .unwrap()
}

fn extension_spec(name: &str, source_id: &str) -> ToolSpec {
    spec(name, ToolEffect::FsRead, ToolIdempotency::Idempotent).with_source(user_source(source_id))
}

fn web_search_options() -> HostedToolOptions {
    HostedToolOptions::WebSearch(WebSearchOptions::new())
}

fn profile_with_tools(id: &str, name: &str, tools: &[&str]) -> tea_profile::AgentProfile {
    let mut builder = tea_profile::AgentProfile::builder(
        id.parse().unwrap(),
        name.parse().unwrap(),
        model_ref("fake/model"),
    );
    for tool in tools {
        builder = builder.active_tool((*tool).parse().unwrap());
    }
    builder
        .policy_rule(ProfileRuleId::from_str("product.coding_workspace").unwrap())
        .environment(environment(ExecutionSurface::Cli))
        .build()
        .unwrap()
}

fn fake_read_executor() -> Arc<FakeReadTool> {
    Arc::new(FakeReadTool::new([(
        "/notes.txt".to_owned(),
        "hello".to_owned(),
    )]))
}

fn read_resolver() -> Arc<ArgumentResourceResolver> {
    Arc::new(ArgumentResourceResolver::new("path", "file", ToolResourceAccess::Read).unwrap())
}

fn register_tools(builder: AgentRuntimeBuilder) -> Result<AgentRuntimeBuilder, tea::RuntimeError> {
    builder
        .tool(
            spec("read_file", ToolEffect::FsRead, ToolIdempotency::Idempotent),
            Arc::new(
                ArgumentResourceResolver::new("path", "file", ToolResourceAccess::Read).unwrap(),
            ),
            Arc::new(FakeReadTool::new([(
                "/notes.txt".to_owned(),
                "hello".to_owned(),
            )])),
        )?
        .tool(
            spec(
                "write_file",
                ToolEffect::FsWrite,
                ToolIdempotency::NonIdempotent,
            ),
            Arc::new(
                ArgumentResourceResolver::new("path", "file", ToolResourceAccess::Write).unwrap(),
            ),
            Arc::new(FakeWriteTool::new()),
        )?
        .tool(
            spec(
                "clipboard_read",
                ToolEffect::ClipboardRead,
                ToolIdempotency::Idempotent,
            ),
            Arc::new(
                ArgumentResourceResolver::new("path", "clipboard", ToolResourceAccess::Read)
                    .unwrap(),
            ),
            Arc::new(FakeReadTool::new([("clip".to_owned(), "data".to_owned())])),
        )
}

fn coding_profile() -> tea_profile::AgentProfile {
    tea_profile::AgentProfile::coding_agent().unwrap()
}

fn desktop_profile() -> tea_profile::AgentProfile {
    tea_profile::AgentProfile::desktop_assistant().unwrap()
}

#[test]
fn build_without_provider_keeps_a_valid_empty_generation() {
    let profile = tea_profile::AgentProfile::builder(
        "providerless".parse().unwrap(),
        "Providerless".parse().unwrap(),
        model_ref("fake/model"),
    )
    .environment(environment(ExecutionSurface::Cli))
    .build()
    .unwrap();
    let runtime = AgentRuntimeBuilder::new()
        .actor(ActorId::from_str("user:alice").unwrap())
        .profile(profile)
        .build()
        .unwrap();

    assert!(runtime.models().is_empty());
    assert!(runtime.health().provider_ids().is_empty());
}

#[test]
fn build_without_profiles_fails() {
    let err = AgentRuntimeBuilder::new()
        .provider(provider())
        .actor(ActorId::from_str("user:alice").unwrap())
        .build()
        .unwrap_err();
    assert_eq!(err.code(), RuntimeErrorCode::UnknownProfile);
}

#[test]
fn build_without_actor_fails() {
    let err = AgentRuntimeBuilder::new()
        .provider(provider())
        .profile(coding_profile())
        .build()
        .unwrap_err();
    assert_eq!(err.code(), RuntimeErrorCode::InvalidRequest);
}

#[test]
fn build_rejects_unregistered_tool() {
    let profile = tea_profile::AgentProfile::builder(
        "coding-agent".parse().unwrap(),
        "Coding Agent".parse().unwrap(),
        model_ref("fake/model"),
    )
    .active_tool("ghost_tool".parse().unwrap())
    .policy_rule(ProfileRuleId::from_str("product.coding_workspace").unwrap())
    .environment(environment(ExecutionSurface::Cli))
    .build()
    .unwrap();
    let err = AgentRuntimeBuilder::new()
        .provider(provider())
        .actor(ActorId::from_str("user:alice").unwrap())
        .profile(profile)
        .build()
        .unwrap_err();
    assert_eq!(err.code(), RuntimeErrorCode::UnknownTool);
}

#[test]
fn build_rejects_unregistered_policy_rule() {
    let profile = tea_profile::AgentProfile::builder(
        "coding-agent".parse().unwrap(),
        "Coding Agent".parse().unwrap(),
        model_ref("fake/model"),
    )
    .active_tool("read_file".parse().unwrap())
    .policy_rule(ProfileRuleId::from_str("product.missing").unwrap())
    .environment(environment(ExecutionSurface::Cli))
    .build()
    .unwrap();
    let err = register_tools(AgentRuntimeBuilder::new())
        .unwrap()
        .provider(provider())
        .actor(ActorId::from_str("user:alice").unwrap())
        .profile(profile)
        .build()
        .unwrap_err();
    assert_eq!(err.code(), RuntimeErrorCode::UnknownPolicyRule);
}

#[test]
fn build_rejects_unregistered_model() {
    let profile = tea_profile::AgentProfile::builder(
        "coding-agent".parse().unwrap(),
        "Coding Agent".parse().unwrap(),
        model_ref("fake/missing-model"),
    )
    .active_tool("read_file".parse().unwrap())
    .policy_rule(ProfileRuleId::from_str("product.coding_workspace").unwrap())
    .environment(environment(ExecutionSurface::Cli))
    .build()
    .unwrap();
    let err = AgentRuntimeBuilder::new()
        .provider(provider())
        .actor(ActorId::from_str("user:alice").unwrap())
        .profile(profile)
        .build()
        .unwrap_err();
    assert_eq!(err.code(), RuntimeErrorCode::UnknownModel);
}

#[test]
fn build_defers_an_unregistered_profile_provider_until_run_time() {
    let profile = tea_profile::AgentProfile::builder(
        "coding-agent".parse().unwrap(),
        "Coding Agent".parse().unwrap(),
        ModelRef::new(
            "missing-provider".parse().unwrap(),
            "fake/model".parse().unwrap(),
        ),
    )
    .environment(environment(ExecutionSurface::Cli))
    .build()
    .unwrap();
    let runtime = AgentRuntimeBuilder::new()
        .provider(provider())
        .actor(ActorId::from_str("user:alice").unwrap())
        .profile(profile)
        .build()
        .unwrap();

    assert_eq!(runtime.models().len(), 1);
}

#[test]
fn build_rejects_duplicate_profile_ids() {
    let err = register_tools(AgentRuntimeBuilder::new())
        .unwrap()
        .provider(provider())
        .actor(ActorId::from_str("user:alice").unwrap())
        .policy_rule(
            ProfileRuleId::from_str("product.coding_workspace").unwrap(),
            Arc::new(CodingWorkspacePolicy),
        )
        .unwrap()
        .profile(tea_profile::AgentProfile::coding_agent().unwrap())
        .profile(tea_profile::AgentProfile::coding_agent().unwrap())
        .build()
        .unwrap_err();
    assert_eq!(err.code(), RuntimeErrorCode::DuplicateEntry);
}

#[test]
fn custom_tools_preserve_provenance_and_only_activate_in_selected_profiles() {
    let runtime = AgentRuntimeBuilder::new()
        .register_tool(
            extension_spec("extension_read", "extension.user.read"),
            read_resolver(),
            fake_read_executor(),
        )
        .unwrap()
        .provider(provider())
        .actor(ActorId::from_str("user:alice").unwrap())
        .policy_rule(
            ProfileRuleId::from_str("product.coding_workspace").unwrap(),
            Arc::new(CodingWorkspacePolicy),
        )
        .unwrap()
        .profile(profile_with_tools(
            "extension-active",
            "Extension Active",
            &["extension_read"],
        ))
        .profile(profile_with_tools(
            "extension-inactive",
            "Extension Inactive",
            &[],
        ))
        .build()
        .unwrap();

    let active = runtime
        .binding(&"extension-active".parse().unwrap())
        .unwrap();
    let inactive = runtime
        .binding(&"extension-inactive".parse().unwrap())
        .unwrap();
    assert_eq!(
        active
            .active_tool_specs()
            .iter()
            .map(|spec| spec.to_model_definition().unwrap().name().to_owned())
            .collect::<Vec<_>>(),
        ["extension_read"]
    );
    assert_eq!(
        active
            .tools()
            .specs()
            .map(|spec| spec.source().source_id())
            .collect::<Vec<_>>(),
        ["extension.user.read"]
    );
    assert_eq!(inactive.tools().specs().count(), 0);
    assert_eq!(
        inactive
            .all_tools()
            .specs()
            .map(|spec| spec.name().as_str())
            .collect::<Vec<_>>(),
        ["extension_read"]
    );
}

#[test]
fn product_tools_win_custom_name_collisions_in_both_registration_orders() {
    for product_first in [true, false] {
        let product = || {
            (
                spec(
                    "shared_read",
                    ToolEffect::FsRead,
                    ToolIdempotency::Idempotent,
                ),
                read_resolver(),
                fake_read_executor(),
            )
        };
        let extension = || {
            (
                extension_spec("shared_read", "extension.user.shared"),
                read_resolver(),
                fake_read_executor(),
            )
        };
        let builder = if product_first {
            let (spec, resolver, executor) = product();
            let builder = AgentRuntimeBuilder::new()
                .tool(spec, resolver, executor)
                .unwrap();
            let (spec, resolver, executor) = extension();
            builder.register_tool(spec, resolver, executor).unwrap()
        } else {
            let (spec, resolver, executor) = extension();
            let builder = AgentRuntimeBuilder::new()
                .register_tool(spec, resolver, executor)
                .unwrap();
            let (spec, resolver, executor) = product();
            builder.tool(spec, resolver, executor).unwrap()
        };
        let runtime = builder
            .provider(provider())
            .actor(ActorId::from_str("user:alice").unwrap())
            .policy_rule(
                ProfileRuleId::from_str("product.coding_workspace").unwrap(),
                Arc::new(CodingWorkspacePolicy),
            )
            .unwrap()
            .profile(profile_with_tools(
                "collision-profile",
                "Collision Profile",
                &["shared_read"],
            ))
            .build()
            .unwrap();
        let tool = runtime
            .binding(&"collision-profile".parse().unwrap())
            .unwrap()
            .tools()
            .specs()
            .next()
            .unwrap();
        assert!(tool.source().is_native_product());
    }
}

#[test]
fn custom_tools_reject_product_provenance_and_equal_precedence_duplicates() {
    let extension_error = AgentRuntimeBuilder::new()
        .tool(
            extension_spec("extension_read", "extension.user.read"),
            read_resolver(),
            fake_read_executor(),
        )
        .unwrap_err();
    assert_eq!(extension_error.code(), RuntimeErrorCode::InvalidRequest);

    let product_error = AgentRuntimeBuilder::new()
        .register_tool(
            spec(
                "extension_read",
                ToolEffect::FsRead,
                ToolIdempotency::Idempotent,
            ),
            read_resolver(),
            fake_read_executor(),
        )
        .unwrap_err();
    assert_eq!(product_error.code(), RuntimeErrorCode::InvalidRequest);

    let duplicate_error = AgentRuntimeBuilder::new()
        .register_tool(
            extension_spec("extension_read", "extension.user.first"),
            read_resolver(),
            fake_read_executor(),
        )
        .unwrap()
        .register_tool(
            extension_spec("extension_read", "extension.user.second"),
            read_resolver(),
            fake_read_executor(),
        )
        .unwrap_err();
    assert_eq!(duplicate_error.code(), RuntimeErrorCode::DuplicateEntry);
}

#[test]
fn hosted_tool_is_available_but_only_active_for_explicit_profiles() {
    let runtime = AgentRuntimeBuilder::new()
        .tool_binding(
            spec(
                "web_search",
                ToolEffect::NetworkRequest,
                ToolIdempotency::Idempotent,
            ),
            ToolBinding::hosted(web_search_options()),
        )
        .unwrap()
        .provider(provider_with_capabilities(
            ModelCapabilities::text().with_hosted_tool(HostedToolKind::WebSearch),
        ))
        .actor(ActorId::from_str("user:alice").unwrap())
        .policy_rule(
            ProfileRuleId::from_str("product.coding_workspace").unwrap(),
            Arc::new(CodingWorkspacePolicy),
        )
        .unwrap()
        .profile(profile_with_tools(
            "hosted-inactive",
            "Hosted Inactive",
            &[],
        ))
        .profile(profile_with_tools(
            "hosted-active",
            "Hosted Active",
            &["web_search"],
        ))
        .build()
        .unwrap();

    let inactive = runtime
        .binding(&"hosted-inactive".parse().unwrap())
        .unwrap();
    let active = runtime.binding(&"hosted-active".parse().unwrap()).unwrap();
    let provider = runtime.provider(&"fake".parse().unwrap()).unwrap();
    let model = provider.model(&"fake/model".parse().unwrap()).unwrap();

    assert_eq!(inactive.all_tools().names().count(), 1);
    assert!(
        inactive
            .tools()
            .model_definitions(model)
            .unwrap()
            .is_empty()
    );
    let definitions = active.tools().model_definitions(model).unwrap();
    assert_eq!(definitions.len(), 1);
    assert_eq!(
        definitions[0].hosted_kind(),
        Some(HostedToolKind::WebSearch)
    );
}

#[test]
fn extension_hybrid_binding_projects_per_model_capability() {
    let runtime = AgentRuntimeBuilder::new()
        .register_tool_binding(
            extension_spec("web_search", "extension.user.web_search"),
            ToolBinding::hybrid(
                web_search_options(),
                ToolRoutePreference::PreferHosted,
                read_resolver(),
                fake_read_executor(),
            ),
        )
        .unwrap()
        .provider(provider())
        .actor(ActorId::from_str("user:alice").unwrap())
        .policy_rule(
            ProfileRuleId::from_str("product.coding_workspace").unwrap(),
            Arc::new(CodingWorkspacePolicy),
        )
        .unwrap()
        .profile(profile_with_tools(
            "hybrid-active",
            "Hybrid Active",
            &["web_search"],
        ))
        .build()
        .unwrap();
    let tools = runtime
        .binding(&"hybrid-active".parse().unwrap())
        .unwrap()
        .tools();

    let client_model = ModelSpec::new(
        "fake/client".parse().unwrap(),
        ProviderId::from_str("fake").unwrap(),
        ModelDisplayName::from_str("Client Model").unwrap(),
        TokenCount::new(32_000).unwrap(),
        TokenCount::new(4_000).unwrap(),
        ModelCapabilities::text().with_tools(false),
    )
    .unwrap();
    let hosted_model = ModelSpec::new(
        "fake/hosted".parse().unwrap(),
        ProviderId::from_str("fake").unwrap(),
        ModelDisplayName::from_str("Hosted Model").unwrap(),
        TokenCount::new(32_000).unwrap(),
        TokenCount::new(4_000).unwrap(),
        ModelCapabilities::text().with_hosted_tool(HostedToolKind::WebSearch),
    )
    .unwrap();

    assert!(
        tools.model_definitions(&client_model).unwrap()[0]
            .as_function()
            .is_some()
    );
    assert!(
        tools.model_definitions(&hosted_model).unwrap()[0]
            .as_hosted()
            .is_some()
    );
}

#[test]
fn precedence_replaces_entire_binding_without_merging_routes() {
    for product_first in [true, false] {
        let product = || {
            (
                spec(
                    "web_search",
                    ToolEffect::NetworkRequest,
                    ToolIdempotency::Idempotent,
                ),
                ToolBinding::hosted(web_search_options()),
            )
        };
        let extension = || {
            (
                extension_spec("web_search", "extension.user.web_search"),
                ToolBinding::client(read_resolver(), fake_read_executor()),
            )
        };
        let builder = if product_first {
            let (spec, binding) = product();
            let builder = AgentRuntimeBuilder::new()
                .tool_binding(spec, binding)
                .unwrap();
            let (spec, binding) = extension();
            builder.register_tool_binding(spec, binding).unwrap()
        } else {
            let (spec, binding) = extension();
            let builder = AgentRuntimeBuilder::new()
                .register_tool_binding(spec, binding)
                .unwrap();
            let (spec, binding) = product();
            builder.tool_binding(spec, binding).unwrap()
        };
        let error = builder
            .provider(provider())
            .actor(ActorId::from_str("user:alice").unwrap())
            .policy_rule(
                ProfileRuleId::from_str("product.coding_workspace").unwrap(),
                Arc::new(CodingWorkspacePolicy),
            )
            .unwrap()
            .profile(profile_with_tools(
                "precedence",
                "Precedence",
                &["web_search"],
            ))
            .build()
            .unwrap_err();
        assert_eq!(error.code(), RuntimeErrorCode::InvalidRequest);
        assert_eq!(
            error.message(),
            "active tool web_search has no execution route supported by selected model fake/model; declare the model capability or configure a supported client route"
        );
    }
}

#[test]
fn binding_exposes_filtered_tools_and_distinct_engines() {
    let runtime = register_tools(AgentRuntimeBuilder::new())
        .unwrap()
        .provider(provider())
        .actor(ActorId::from_str("user:alice").unwrap())
        .policy_rule(
            ProfileRuleId::from_str("product.coding_workspace").unwrap(),
            Arc::new(CodingWorkspacePolicy),
        )
        .unwrap()
        .policy_rule(
            ProfileRuleId::from_str("product.desktop").unwrap(),
            Arc::new(DesktopPolicy),
        )
        .unwrap()
        .profile(coding_profile())
        .profile(desktop_profile())
        .build()
        .unwrap();

    let coding = runtime.binding(&"coding-agent".parse().unwrap()).unwrap();
    let desktop = runtime
        .binding(&"desktop-assistant".parse().unwrap())
        .unwrap();

    // Different filtered tool sets.
    assert_eq!(
        coding
            .tools()
            .specs()
            .map(|spec| spec.name().to_string())
            .collect::<Vec<_>>(),
        ["read_file", "write_file"]
    );
    assert_eq!(
        desktop
            .tools()
            .specs()
            .map(|spec| spec.name().to_string())
            .collect::<Vec<_>>(),
        ["clipboard_read", "write_file"]
    );
    assert_eq!(
        coding
            .all_tools()
            .specs()
            .map(|spec| spec.name().to_string())
            .collect::<Vec<_>>(),
        ["clipboard_read", "read_file", "write_file"]
    );
    assert_eq!(
        coding
            .active_tool_names()
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        ["read_file", "write_file"]
    );

    // Distinct composed engines (different instances backing different rules).
    assert!(!std::ptr::eq(
        std::ptr::from_ref::<tea_policy::PolicyEngine>(coding.policy()),
        std::ptr::from_ref::<tea_policy::PolicyEngine>(desktop.policy()),
    ));

    // Converted limits and environments differ per profile.
    assert_ne!(
        coding.run_limits().max_tool_iterations(),
        desktop.run_limits().max_tool_iterations()
    );
    assert_ne!(
        coding.environment().surface(),
        desktop.environment().surface()
    );

    let health = runtime.health();
    assert_eq!(
        health
            .profile_ids()
            .iter()
            .map(tea_protocol::ProfileId::as_str)
            .collect::<Vec<_>>(),
        ["coding-agent", "desktop-assistant"]
    );
    assert_eq!(health.tool_count(), 3);
    assert_eq!(health.provider_id(), Some("fake"));
}
