use tea_tools::{ToolEffect, ToolResourceAccess, ToolSourceKind, ToolTrust};

use crate::{ApprovalRequirement, PolicyInput, PolicyLayer, PolicyRule, PolicyRuleDecision};

/// Fail-closed platform rule for effects unknown to this runtime version.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnknownEffectPolicy;

impl PolicyRule for UnknownEffectPolicy {
    fn id(&self) -> &'static str {
        "platform.unknown_effect"
    }
    fn layer(&self) -> PolicyLayer {
        PolicyLayer::Platform
    }
    fn evaluate(&self, input: &PolicyInput) -> PolicyRuleDecision {
        if input.effects().iter().any(ToolEffect::is_unknown) {
            hard_deny("unknown tool effect is denied by default")
        } else {
            PolicyRuleDecision::abstain()
        }
    }
}

/// Fail-closed platform rule for externally supplied tool capabilities.
#[derive(Debug, Clone, Copy, Default)]
pub struct ExternalSourcePolicy;

impl PolicyRule for ExternalSourcePolicy {
    fn id(&self) -> &'static str {
        "platform.external_source"
    }

    fn layer(&self) -> PolicyLayer {
        PolicyLayer::Platform
    }

    fn evaluate(&self, input: &PolicyInput) -> PolicyRuleDecision {
        let source = input.tool_source();
        if source.trust() == ToolTrust::Untrusted {
            return hard_deny("untrusted tool sources are denied by default");
        }
        if !matches!(source.kind(), ToolSourceKind::Mcp | ToolSourceKind::Remote) {
            return PolicyRuleDecision::abstain();
        }
        if input.effects().is_empty() || input.resources().is_empty() {
            return hard_deny("external tool capabilities must be fully declared");
        }
        if source.kind() == ToolSourceKind::Mcp
            && !input.resources().iter().any(|resource| {
                resource.scheme() == "mcp-server"
                    && resource.access() == ToolResourceAccess::Execute
            })
        {
            return hard_deny("MCP tools require a host-declared server resource");
        }
        PolicyRuleDecision::abstain()
    }
}

/// Minimal product rule that authorizes pure filesystem reads.
///
/// The rule deliberately abstains for every other effect so more capable
/// products must opt into their own explicit policy. Resource confinement is
/// still enforced by the selected tool resource resolver.
#[derive(Debug, Clone, Copy, Default)]
pub struct FilesystemReadPolicy;

impl PolicyRule for FilesystemReadPolicy {
    fn id(&self) -> &'static str {
        "product.filesystem_read"
    }

    fn layer(&self) -> PolicyLayer {
        PolicyLayer::Product
    }

    fn evaluate(&self, input: &PolicyInput) -> PolicyRuleDecision {
        if !input.effects().is_empty()
            && input
                .effects()
                .iter()
                .all(|effect| effect == &ToolEffect::FsRead)
        {
            PolicyRuleDecision::allow()
        } else {
            PolicyRuleDecision::abstain()
        }
    }
}

/// Pure coding-workspace policy used by tests and product profiles.
#[derive(Debug, Clone, Copy, Default)]
pub struct CodingWorkspacePolicy;

impl PolicyRule for CodingWorkspacePolicy {
    fn id(&self) -> &'static str {
        "product.coding_workspace"
    }
    fn layer(&self) -> PolicyLayer {
        PolicyLayer::Product
    }
    fn evaluate(&self, input: &PolicyInput) -> PolicyRuleDecision {
        if input.effects().contains(&ToolEffect::CredentialRead) {
            return hard_deny("credential reads are denied in coding workspace policy");
        }
        if input.effects().contains(&ToolEffect::ProcessSpawn) {
            return if is_read_only_git_status(input.arguments()) {
                PolicyRuleDecision::allow()
            } else {
                ask("process execution requires approval")
            };
        }
        if input.effects().iter().all(is_known_read_only) {
            return PolicyRuleDecision::allow();
        }
        if input.effects().iter().any(|effect| {
            matches!(
                effect,
                ToolEffect::FsWrite
                    | ToolEffect::FsDelete
                    | ToolEffect::NetworkRequest
                    | ToolEffect::UserInteraction
                    | ToolEffect::ExternalMutation
            )
        }) {
            return ask("coding workspace mutation requires approval");
        }
        PolicyRuleDecision::abstain()
    }
}

/// Conservative desktop interaction policy.
#[derive(Debug, Clone, Copy, Default)]
pub struct DesktopPolicy;

impl PolicyRule for DesktopPolicy {
    fn id(&self) -> &'static str {
        "product.desktop"
    }
    fn layer(&self) -> PolicyLayer {
        PolicyLayer::Product
    }
    fn evaluate(&self, input: &PolicyInput) -> PolicyRuleDecision {
        if input.effects().contains(&ToolEffect::CredentialRead) {
            return hard_deny("credential reads are denied by desktop policy");
        }
        if input.effects().iter().any(|effect| {
            matches!(
                effect,
                ToolEffect::ClipboardRead
                    | ToolEffect::UserInteraction
                    | ToolEffect::ExternalMutation
            )
        }) {
            return ask("desktop-sensitive operation requires approval");
        }
        PolicyRuleDecision::abstain()
    }
}

fn is_known_read_only(effect: &ToolEffect) -> bool {
    matches!(
        effect,
        ToolEffect::FsRead | ToolEffect::ClipboardRead | ToolEffect::ExternalRead
    )
}

fn is_read_only_git_status(arguments: &serde_json::Value) -> bool {
    arguments
        .get("command")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|command| command.split_ascii_whitespace().eq(["git", "status"]))
}

fn ask(reason: &str) -> PolicyRuleDecision {
    ApprovalRequirement::new(reason)
        .map_or_else(|_| PolicyRuleDecision::abstain(), PolicyRuleDecision::ask)
}

fn hard_deny(reason: &str) -> PolicyRuleDecision {
    PolicyRuleDecision::hard_deny(reason).unwrap_or_else(|_| PolicyRuleDecision::abstain())
}
