#![forbid(unsafe_code)]

use std::{ffi::OsString, str::FromStr};

use serde_json::{Value, json};
use tea_mcp::{
    MAX_MCP_DESCRIPTOR_BYTES, McpArgumentResource, McpErrorCode, McpLifecyclePolicy, McpLimits,
    McpReconnectPolicy, McpRemoteToolDescriptor, McpRemoteToolName, McpServerConfig, McpServerId,
    McpToolCatalog, McpToolDeclaration, McpToolPolicy, McpTransportConfig,
};
use tea_protocol::ToolIdempotency;
use tea_tools::{
    ToolConcurrency, ToolEffect, ToolExecutionSemantics, ToolName, ToolResourceAccess,
    ToolRetrySafety, ToolTimeout, ToolTrust,
};

#[test]
fn descriptor_and_host_policy_hash_deterministically_across_json_key_order() {
    let first = McpRemoteToolDescriptor::from_value(
        serde_json::from_str(
            r#"{"name":"inspect","description":"Inspect a path.","inputSchema":{"type":"object","properties":{"z":{"type":"string"},"a":{"type":"string"}}},"outputSchema":{"type":"object","properties":{"ok":{"type":"boolean"}}}}"#,
        )
        .unwrap(),
    )
    .unwrap();
    let second = McpRemoteToolDescriptor::from_value(
        serde_json::from_str(
            r#"{"outputSchema":{"properties":{"ok":{"type":"boolean"}},"type":"object"},"inputSchema":{"properties":{"a":{"type":"string"},"z":{"type":"string"}},"type":"object"},"description":"Inspect a path.","name":"inspect"}"#,
        )
        .unwrap(),
    )
    .unwrap();
    let config = config(
        "first",
        vec![enabled("inspect", None, conservative_declaration())],
    );

    let first = McpToolCatalog::freeze(&config, ToolTrust::Workspace, [first]).unwrap();
    let second = McpToolCatalog::freeze(&config, ToolTrust::Workspace, [second]).unwrap();
    let first = first.specs().next().unwrap();
    let second = second.specs().next().unwrap();

    assert_eq!(first, second);
    assert_eq!(first.source().descriptor_digest().len(), 64);
    assert!(
        first
            .source()
            .descriptor_digest()
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
    );
    assert_eq!(
        first.version().to_string(),
        format!("0.0.0+mcp.{}", &first.source().descriptor_digest()[..16])
    );
}

#[test]
fn host_policy_order_is_canonical_and_policy_changes_rotate_identity() {
    let remote = || {
        descriptor(json!({
            "name":"inspect",
            "description":"Inspect.",
            "inputSchema":{"type":"object"},
            "outputSchema":{"type":"object"}
        }))
    };
    let first = config(
        "ordered",
        vec![enabled("inspect", None, ordered_declaration(false, 4_000))],
    );
    let reversed = config(
        "ordered",
        vec![enabled("inspect", None, ordered_declaration(true, 4_000))],
    );
    let changed = config(
        "ordered",
        vec![enabled("inspect", None, ordered_declaration(false, 4_001))],
    );

    let first = McpToolCatalog::freeze(&first, ToolTrust::User, [remote()]).unwrap();
    let reversed = McpToolCatalog::freeze(&reversed, ToolTrust::User, [remote()]).unwrap();
    let changed = McpToolCatalog::freeze(&changed, ToolTrust::User, [remote()]).unwrap();
    let first = first.specs().next().unwrap();
    let reversed = reversed.specs().next().unwrap();
    let changed = changed.specs().next().unwrap();

    assert_eq!(first, reversed);
    assert_ne!(first.version(), changed.version());
    assert_ne!(
        first.source().descriptor_digest(),
        changed.source().descriptor_digest()
    );
}

#[test]
fn only_host_declarations_control_policy_and_execute_resources_are_mandatory() {
    let descriptor = descriptor(json!({
        "name": "mutate",
        "description": "Mutates an external object.",
        "inputSchema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"]
        },
        "outputSchema": {"type": "object"},
        "annotations": {
            "readOnlyHint": true,
            "destructiveHint": false,
            "idempotentHint": true,
            "openWorldHint": false
        }
    }));
    let declaration = conservative_declaration();
    let expected_execution = declaration.execution();
    let config = config("host", vec![enabled("mutate", None, declaration)]);

    let catalog = McpToolCatalog::freeze(&config, ToolTrust::User, [descriptor]).unwrap();
    let binding = catalog.binding("mcp.host.mutate").unwrap();
    let spec = binding.spec();

    assert_eq!(spec.effects(), [ToolEffect::ExternalMutation]);
    assert_eq!(spec.execution(), expected_execution);
    assert_eq!(binding.remote_annotations().unwrap()["readOnlyHint"], true);

    let resources = binding
        .resolve_resources(&json!({"path":"ticket/42"}))
        .unwrap();
    assert_eq!(resources.len(), 2);
    assert!(resources.iter().any(|resource| {
        resource.scheme() == "mcp-server"
            && resource.locator() == "host/mutate"
            && resource.access() == ToolResourceAccess::Execute
    }));
    assert!(resources.iter().any(|resource| {
        resource.scheme() == "ticket"
            && resource.locator() == "ticket/42"
            && resource.access() == ToolResourceAccess::Write
    }));
}

#[test]
fn explicit_disabled_policy_overrides_default_declaration() {
    let config = config(
        "server",
        vec![McpToolPolicy::new(
            McpRemoteToolName::new("disabled").unwrap(),
        )],
    )
    .with_default_tool_declaration(conservative_declaration());
    let catalog = McpToolCatalog::freeze(
        &config,
        ToolTrust::User,
        [descriptor(json!({
            "name": "disabled",
            "description": "Disabled by host policy.",
            "inputSchema": {"type": "object"},
            "outputSchema": {"type": "object"}
        }))],
    )
    .unwrap();

    assert!(catalog.is_empty());
}

#[test]
fn lossy_default_aliases_require_an_explicit_canonical_alias() {
    let declaration = conservative_declaration();
    let lossy = McpToolPolicy::enabled(
        McpRemoteToolName::new("Mixed Case").unwrap(),
        declaration.clone(),
    );
    assert_eq!(
        config_result("server", vec![lossy]).unwrap_err().code(),
        McpErrorCode::PolicyDeclaration
    );

    let explicit = enabled(
        "Mixed Case",
        Some(ToolName::from_str("mcp.server.mixed_case").unwrap()),
        declaration,
    );
    let config = config("server", vec![explicit]);
    let catalog = McpToolCatalog::freeze(
        &config,
        ToolTrust::User,
        [descriptor(json!({
            "name": "Mixed Case",
            "description": "Mixed name.",
            "inputSchema": {"type":"object"},
            "outputSchema": {"type":"object"}
        }))],
    )
    .unwrap();
    assert!(catalog.binding("mcp.server.mixed_case").is_some());
}

#[test]
fn cross_server_alias_collisions_fail_closed() {
    let alias = ToolName::from_str("mcp.shared.inspect").unwrap();
    let first_config = config(
        "first",
        vec![enabled(
            "inspect",
            Some(alias.clone()),
            conservative_declaration(),
        )],
    );
    let second_config = config(
        "second",
        vec![enabled("inspect", Some(alias), conservative_declaration())],
    );
    let remote = || {
        descriptor(json!({
            "name":"inspect",
            "description":"Inspect.",
            "inputSchema":{"type":"object"},
            "outputSchema":{"type":"object"}
        }))
    };
    let first = McpToolCatalog::freeze(&first_config, ToolTrust::User, [remote()]).unwrap();
    let second = McpToolCatalog::freeze(&second_config, ToolTrust::User, [remote()]).unwrap();

    assert_eq!(
        McpToolCatalog::combine([first, second]).unwrap_err().code(),
        McpErrorCode::Descriptor
    );
}

#[test]
fn invalid_and_oversized_descriptors_are_rejected_before_freezing() {
    for invalid in [
        json!({"name":"","description":"Valid.","inputSchema":{"type":"object"}}),
        json!({"name":"valid","description":"","inputSchema":{"type":"object"}}),
        json!({"name":"valid","description":"bad\0text","inputSchema":{"type":"object"}}),
        json!({"name":"valid","inputSchema":{"type":"object"}}),
    ] {
        assert_eq!(
            McpRemoteToolDescriptor::from_value(invalid)
                .unwrap_err()
                .code(),
            McpErrorCode::Descriptor
        );
    }

    let oversized = json!({
        "name":"valid",
        "description":"Valid.",
        "inputSchema":{"type":"object","$comment":"x".repeat(MAX_MCP_DESCRIPTOR_BYTES)}
    });
    assert_eq!(
        McpRemoteToolDescriptor::from_value(oversized)
            .unwrap_err()
            .code(),
        McpErrorCode::OutputBound
    );
}

fn descriptor(value: Value) -> McpRemoteToolDescriptor {
    McpRemoteToolDescriptor::from_value(value).unwrap()
}

fn conservative_declaration() -> McpToolDeclaration {
    let execution = ToolExecutionSemantics::new(
        ToolIdempotency::NonIdempotent,
        ToolRetrySafety::Never,
        ToolConcurrency::Serial,
        ToolTimeout::from_millis(7_500).unwrap(),
    )
    .unwrap();
    let resource = McpArgumentResource::new("path", "ticket", ToolResourceAccess::Write).unwrap();
    McpToolDeclaration::new([ToolEffect::ExternalMutation], [resource], execution).unwrap()
}

fn ordered_declaration(reverse: bool, timeout_millis: u64) -> McpToolDeclaration {
    let execution = ToolExecutionSemantics::new(
        ToolIdempotency::Idempotent,
        ToolRetrySafety::ExplicitOnly,
        ToolConcurrency::Serial,
        ToolTimeout::from_millis(timeout_millis).unwrap(),
    )
    .unwrap();
    let first = McpArgumentResource::new("path", "file", ToolResourceAccess::Read).unwrap();
    let second = McpArgumentResource::new("url", "https", ToolResourceAccess::Write).unwrap();
    let (effects, resources) = if reverse {
        (
            vec![ToolEffect::NetworkRequest, ToolEffect::FsRead],
            vec![second, first],
        )
    } else {
        (
            vec![ToolEffect::FsRead, ToolEffect::NetworkRequest],
            vec![first, second],
        )
    };
    McpToolDeclaration::new(effects, resources, execution).unwrap()
}

fn enabled(
    remote_name: &str,
    alias: Option<ToolName>,
    declaration: McpToolDeclaration,
) -> McpToolPolicy {
    let policy = McpToolPolicy::enabled(McpRemoteToolName::new(remote_name).unwrap(), declaration);
    match alias {
        Some(alias) => policy.with_alias(alias),
        None => policy,
    }
}

fn config(id: &str, tools: Vec<McpToolPolicy>) -> McpServerConfig {
    config_result(id, tools).unwrap()
}

fn config_result(
    id: &str,
    tools: Vec<McpToolPolicy>,
) -> Result<McpServerConfig, tea_mcp::McpError> {
    McpServerConfig::new(
        McpServerId::from_str(id).unwrap(),
        McpTransportConfig::stdio(
            if cfg!(windows) { r"C:\mcp.exe" } else { "/mcp" },
            Vec::<OsString>::new(),
        )
        .unwrap(),
        Vec::new(),
        tools,
        McpLimits::default(),
        McpLifecyclePolicy::default(),
        McpReconnectPolicy::default(),
    )
}
