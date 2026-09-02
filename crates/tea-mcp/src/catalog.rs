use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::Write as _,
    str::FromStr,
};

use rmcp::model::ListToolsResult;
use serde_json::json;
use sha2::{Digest, Sha256};
use tea_tools::{
    ToolConcurrency, ToolName, ToolRetrySafety, ToolSource, ToolSourceKind, ToolSpec, ToolTrust,
    ToolVersion,
};
use tokio::time::Instant;

use crate::{
    McpDescriptorDigest, McpError, McpErrorCode, McpRemoteToolDescriptor, McpServerConfig,
    McpToolBinding, descriptor, transport::SdkClient,
};

const VERSION_DIGEST_PREFIX_BYTES: usize = 16;
const MAX_CURSOR_BYTES: usize = 4 * 1024;
const MAX_COMBINED_CATALOGS: usize = 64;
const MAX_COMBINED_TOOLS: usize = 4_096;

/// Immutable MCP tool bindings ordered by canonical local alias.
#[derive(Debug, Clone, Default)]
pub struct McpToolCatalog {
    bindings: BTreeMap<ToolName, McpToolBinding>,
}

impl McpToolCatalog {
    /// Freezes already bounded remote descriptors against one host configuration.
    ///
    /// Only exact remote names with enabled host declarations enter the result.
    /// Remote annotations never determine effects, resources, idempotency, retry,
    /// concurrency, timeout, or trust.
    ///
    /// # Errors
    ///
    /// Rejects duplicate names, descriptor/count bounds, invalid aliases,
    /// schema/spec failures, or resource declarations.
    pub fn freeze(
        config: &McpServerConfig,
        trust: ToolTrust,
        descriptors: impl IntoIterator<Item = McpRemoteToolDescriptor>,
    ) -> Result<Self, McpError> {
        let limits = config.limits();
        let policies = config
            .tools()
            .iter()
            .map(|policy| (policy.remote_name(), policy))
            .collect::<BTreeMap<_, _>>();
        let mut remote_names = BTreeSet::new();
        let mut bindings = BTreeMap::new();
        for descriptor in descriptors {
            if remote_names.len() >= limits.max_tools()
                || descriptor.canonical_json().len() > limits.max_descriptor_bytes()
            {
                return Err(McpError::new(McpErrorCode::OutputBound));
            }
            if !remote_names.insert(descriptor.name().clone()) {
                return Err(McpError::new(McpErrorCode::Descriptor));
            }
            let policy = policies.get(descriptor.name());
            let declaration = policy
                .and_then(|policy| policy.declaration())
                .or_else(|| config.default_tool_declaration());
            let Some(declaration) = declaration else {
                continue;
            };
            let alias = policy
                .and_then(|policy| policy.resolved_alias(config.id()))
                .or_else(|| {
                    tea_tools::ToolName::from_str(&format!(
                        "mcp.{}.{}",
                        config.id().as_str(),
                        descriptor.name().as_str()
                    ))
                    .ok()
                })
                .ok_or_else(|| McpError::new(McpErrorCode::PolicyDeclaration))?;
            let host_policy = host_policy_json(
                config,
                descriptor.name().as_str(),
                &alias,
                trust,
                declaration,
            )?;
            let digest = hash_descriptor(descriptor.canonical_json(), &host_policy)?;
            let version = ToolVersion::from_str(&format!(
                "0.0.0+mcp.{}",
                &digest.as_str()[..VERSION_DIGEST_PREFIX_BYTES]
            ))
            .map_err(|_| McpError::new(McpErrorCode::Descriptor))?;
            let source = ToolSource::new(
                ToolSourceKind::Mcp,
                format!("mcp.{}", config.id().as_str()),
                trust,
                digest.as_str(),
            )
            .map_err(|_| McpError::new(McpErrorCode::Descriptor))?;
            let spec = ToolSpec::new(
                alias,
                version,
                descriptor.description(),
                descriptor.input_schema().clone(),
                descriptor.output_schema().clone(),
                declaration.effects().iter().cloned(),
                declaration.execution(),
            )
            .map_err(|_| McpError::new(McpErrorCode::Descriptor))?
            .with_source(source);
            let binding = McpToolBinding::new(
                config.id(),
                descriptor.name().clone(),
                spec,
                declaration.resources(),
                descriptor.annotations().cloned(),
            )?;
            let (name, binding) = binding.into_parts();
            if bindings.insert(name, binding).is_some() {
                return Err(McpError::new(McpErrorCode::Descriptor));
            }
        }
        Ok(Self { bindings })
    }

    /// Combines bounded per-server catalogs and rejects cross-server aliases.
    ///
    /// # Errors
    ///
    /// Rejects duplicate aliases or more than 64 catalogs/4096 total tools.
    pub fn combine(catalogs: impl IntoIterator<Item = Self>) -> Result<Self, McpError> {
        let mut bindings = BTreeMap::new();
        for (index, catalog) in catalogs.into_iter().enumerate() {
            if index >= MAX_COMBINED_CATALOGS {
                return Err(McpError::new(McpErrorCode::OutputBound));
            }
            for (name, binding) in catalog.bindings {
                if bindings.len() >= MAX_COMBINED_TOOLS {
                    return Err(McpError::new(McpErrorCode::OutputBound));
                }
                if bindings.insert(name, binding).is_some() {
                    return Err(McpError::new(McpErrorCode::Descriptor));
                }
            }
        }
        Ok(Self { bindings })
    }

    /// Returns the number of enabled frozen bindings.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Returns whether no enabled descriptor entered the catalog.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Returns bindings in canonical local-alias order.
    pub fn bindings(&self) -> impl Iterator<Item = &McpToolBinding> {
        self.bindings.values()
    }

    /// Returns ordinary tool specifications in canonical local-alias order.
    pub fn specs(&self) -> impl Iterator<Item = &ToolSpec> {
        self.bindings.values().map(McpToolBinding::spec)
    }

    /// Looks up one canonical local alias.
    #[must_use]
    pub fn binding(&self, alias: &str) -> Option<&McpToolBinding> {
        ToolName::from_str(alias)
            .ok()
            .and_then(|alias| self.bindings.get(&alias))
    }
}

pub(crate) async fn discover(
    client: &SdkClient,
    config: &McpServerConfig,
    trust: ToolTrust,
) -> Result<McpToolCatalog, McpError> {
    let limits = config.limits();
    let mut descriptors = Vec::new();
    let mut cursor = None;
    let mut cursors = BTreeSet::new();
    let deadline = Instant::now() + config.lifecycle().handshake_timeout();
    for _ in 0..=limits.max_tools() {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(McpError::new(McpErrorCode::Timeout));
        }
        let page = client.list_tools_page(cursor, remaining).await?;
        let ListToolsResult {
            tools, next_cursor, ..
        } = page;
        for tool in tools {
            if descriptors.len() >= limits.max_tools() {
                return Err(McpError::new(McpErrorCode::OutputBound));
            }
            descriptors.push(McpRemoteToolDescriptor::from_sdk_tool(
                &tool,
                limits.max_descriptor_bytes(),
            )?);
        }
        if Instant::now() >= deadline {
            return Err(McpError::new(McpErrorCode::Timeout));
        }
        let Some(next_cursor) = next_cursor else {
            return McpToolCatalog::freeze(config, trust, descriptors);
        };
        if next_cursor.is_empty()
            || next_cursor.len() > MAX_CURSOR_BYTES
            || !cursors.insert(next_cursor.clone())
        {
            return Err(McpError::new(McpErrorCode::Descriptor));
        }
        cursor = Some(next_cursor);
    }
    Err(McpError::new(McpErrorCode::OutputBound))
}

fn host_policy_json(
    config: &McpServerConfig,
    remote_name: &str,
    alias: &ToolName,
    trust: ToolTrust,
    declaration: &crate::McpToolDeclaration,
) -> Result<Vec<u8>, McpError> {
    let execution = declaration.execution();
    let retry = match execution.retry_safety() {
        ToolRetrySafety::Never => "never",
        ToolRetrySafety::ExplicitOnly => "explicit_only",
        ToolRetrySafety::Automatic => "automatic",
    };
    let concurrency = match execution.concurrency() {
        ToolConcurrency::Parallel => "parallel",
        ToolConcurrency::Serial => "serial",
        ToolConcurrency::Exclusive => "exclusive",
    };
    let resources = declaration
        .resources()
        .iter()
        .map(|resource| {
            json!({
                "access": resource.access(),
                "argument": resource.argument(),
                "scheme": resource.scheme(),
            })
        })
        .collect::<Vec<_>>();
    let value = json!({
        "alias": alias.as_str(),
        "effects": declaration.effects().iter().map(tea_tools::ToolEffect::as_str).collect::<Vec<_>>(),
        "executeResource": format!("mcp-server://{}/{remote_name}", config.id().as_str()),
        "execution": {
            "concurrency": concurrency,
            "idempotency": execution.idempotency(),
            "retrySafety": retry,
            "timeoutMillis": execution.timeout().as_millis(),
        },
        "resources": resources,
        "serverId": config.id().as_str(),
        "trust": trust,
    });
    descriptor::canonical_json(value, config.limits().max_descriptor_bytes())
}

fn hash_descriptor(
    descriptor_json: &[u8],
    host_policy_json: &[u8],
) -> Result<McpDescriptorDigest, McpError> {
    let descriptor_len = u64::try_from(descriptor_json.len())
        .map_err(|_| McpError::new(McpErrorCode::OutputBound))?;
    let host_len = u64::try_from(host_policy_json.len())
        .map_err(|_| McpError::new(McpErrorCode::OutputBound))?;
    let mut hasher = Sha256::new();
    hasher.update(b"tea-mcp-descriptor-v1\0");
    hasher.update(descriptor_len.to_be_bytes());
    hasher.update(descriptor_json);
    hasher.update(host_len.to_be_bytes());
    hasher.update(host_policy_json);
    let mut digest = String::with_capacity(64);
    for byte in hasher.finalize() {
        write!(&mut digest, "{byte:02x}").map_err(|_| McpError::new(McpErrorCode::Descriptor))?;
    }
    descriptor::descriptor_digest(&digest)
}
