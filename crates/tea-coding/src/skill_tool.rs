use std::path::{Component, Path};
use std::str::FromStr;
use std::sync::Arc;

use futures_util::stream;
use serde_json::{Value, json};
use tea_context::SkillId;
use tea_protocol::ProtocolMetadata;
use tea_protocol::ToolIdempotency;
use tea_tools::{
    BoxToolExecutionStream, ToolConcurrency, ToolEffect, ToolExecutionEvent, ToolExecutionFailure,
    ToolExecutionSemantics, ToolExecutor, ToolName, ToolResource, ToolResourceAccess,
    ToolResourceError, ToolResourceResolver, ToolResult, ToolRetrySafety, ToolSpec, ToolSpecError,
    ToolTimeout, ToolVersion, ValidatedToolInvocation,
};

use crate::resources::ResourceCatalog;
use tea::control::CancellationScope;

/// Stable native tool name for bounded reads below an active skill.
pub const READ_SKILL_RESOURCE_TOOL_NAME: &str = "read_skill_resource";

const MAX_SKILL_ID_BYTES: usize = 128;
const MAX_SKILL_RESOURCE_PATH_BYTES: usize = 1_800;
const ERROR_NAMESPACE: &str = "dev.tea-rs.coding";

/// Native executor for resources contained by one immutable skill catalog.
#[derive(Debug, Clone)]
pub struct ReadSkillResourceTool {
    resources: Arc<ResourceCatalog>,
}

impl ReadSkillResourceTool {
    /// Creates a resource reader bound to one catalog generation.
    #[must_use]
    pub const fn new(resources: Arc<ResourceCatalog>) -> Self {
        Self { resources }
    }

    /// Builds the bounded, read-only native tool contract.
    ///
    /// # Errors
    ///
    /// Returns an error only if the static contract violates tool bounds.
    pub fn spec() -> Result<ToolSpec, ToolSpecError> {
        ToolSpec::new(
            ToolName::from_str(READ_SKILL_RESOURCE_TOOL_NAME)
                .map_err(|_| ToolSpecError::InvalidDescription)?,
            ToolVersion::from_str("1.0.0").map_err(|_| ToolSpecError::InvalidDescription)?,
            "Read bounded UTF-8 text from a file contained by one active skill.",
            json!({
                "type":"object",
                "properties":{
                    "skill":{"type":"string","minLength":1,"maxLength":MAX_SKILL_ID_BYTES},
                    "path":{"type":"string","minLength":1,"maxLength":MAX_SKILL_RESOURCE_PATH_BYTES},
                    "offset":{"type":"integer","minimum":1},
                    "limit":{"type":"integer","minimum":1,"maximum":10_000}
                },
                "required":["skill","path"],
                "additionalProperties":false
            }),
            json!({
                "type":"object",
                "properties":{
                    "skill":{"type":"string"},
                    "path":{"type":"string"},
                    "content":{"type":"string"},
                    "startLine":{"type":"integer","minimum":1},
                    "endLine":{"type":"integer","minimum":0},
                    "totalLines":{"type":"integer","minimum":0},
                    "truncated":{"type":"boolean"}
                },
                "required":["skill","path","content","startLine","endLine","totalLines","truncated"],
                "additionalProperties":false
            }),
            [ToolEffect::FsRead],
            ToolExecutionSemantics::new(
                ToolIdempotency::Idempotent,
                ToolRetrySafety::Automatic,
                ToolConcurrency::Parallel,
                ToolTimeout::from_millis(30_000)?,
            )?,
        )?
        .with_prompt_hint(
            "Use a listed skill ID and a relative file path referenced by that skill.",
        )
    }

    fn run(
        &self,
        invocation: &ValidatedToolInvocation,
    ) -> Result<ToolExecutionEvent, ToolExecutionFailure> {
        let skill = string_argument(invocation, "skill")?;
        let path = string_argument(invocation, "path")?;
        let skill_id = skill.parse::<SkillId>().map_err(|_| invalid_failure())?;
        let offset = integer_argument(invocation, "offset")?;
        let limit = integer_argument(invocation, "limit")?;
        let result = self
            .resources
            .read_skill_resource(&skill_id, path, offset, limit)
            .map_err(|error| failure_for(error.code(), error.message()))?;
        let visible = if result.content().is_empty() {
            "(empty text result)".to_owned()
        } else {
            result.content().to_owned()
        };
        let output = json!({
            "skill": result.skill_id().as_str(),
            "path": result.path(),
            "content": result.content(),
            "startLine": result.start_line(),
            "endLine": result.end_line(),
            "totalLines": result.total_lines(),
            "truncated": result.truncated()
        });
        let content = tea_protocol::ContentBlock::text(visible).map_err(|_| internal_failure())?;
        let result = ToolResult::new(vec![content], output).map_err(|_| internal_failure())?;
        Ok(ToolExecutionEvent::Finished(result))
    }
}

impl ToolExecutor for ReadSkillResourceTool {
    fn execute(
        &self,
        invocation: ValidatedToolInvocation,
        cancellation: CancellationScope,
    ) -> BoxToolExecutionStream {
        let executor = self.clone();
        Box::pin(stream::once(async move {
            if cancellation.is_cancelled() {
                ToolExecutionEvent::Failed(ToolExecutionFailure::cancelled())
            } else {
                executor
                    .run(&invocation)
                    .unwrap_or_else(ToolExecutionEvent::Failed)
            }
        }))
    }
}

/// Filesystem-free resolver for the skill and relative path arguments.
#[derive(Debug, Clone, Default)]
pub struct SkillResourceResolver;

impl ToolResourceResolver for SkillResourceResolver {
    fn resolve(
        &self,
        _tool_name: &ToolName,
        arguments: &Value,
    ) -> Result<Vec<ToolResource>, ToolResourceError> {
        let skill = arguments
            .get("skill")
            .and_then(Value::as_str)
            .ok_or(ToolResourceError::Unresolved)?;
        let path = arguments
            .get("path")
            .and_then(Value::as_str)
            .ok_or(ToolResourceError::Unresolved)?;
        let skill_id = skill
            .parse::<SkillId>()
            .map_err(|_| ToolResourceError::Unresolved)?;
        if skill_id.as_str().len() > MAX_SKILL_ID_BYTES {
            return Err(ToolResourceError::Unresolved);
        }
        let path = canonical_relative_path(path).ok_or(ToolResourceError::Unresolved)?;
        let locator = format!("/{}/{path}", skill_id.as_str());
        Ok(vec![ToolResource::new(
            "skill",
            locator,
            ToolResourceAccess::Read,
        )?])
    }
}

fn canonical_relative_path(input: &str) -> Option<String> {
    if input.is_empty()
        || input.len() > MAX_SKILL_RESOURCE_PATH_BYTES
        || input.as_bytes().contains(&0)
        || input.chars().any(char::is_control)
        || input.contains('\\')
        || looks_like_windows_prefix(input)
    {
        return None;
    }
    let mut components = Vec::new();
    for component in Path::new(input).components() {
        match component {
            Component::CurDir => {}
            Component::Normal(value) => components.push(value.to_str()?.to_owned()),
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => return None,
        }
    }
    (!components.is_empty()).then(|| components.join("/"))
}

fn looks_like_windows_prefix(input: &str) -> bool {
    let bytes = input.as_bytes();
    (bytes.len() >= 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':')
        || input.starts_with("\\\\")
}

fn string_argument<'a>(
    invocation: &'a ValidatedToolInvocation,
    name: &str,
) -> Result<&'a str, ToolExecutionFailure> {
    invocation
        .arguments()
        .get(name)
        .and_then(Value::as_str)
        .ok_or_else(invalid_failure)
}

fn integer_argument(
    invocation: &ValidatedToolInvocation,
    name: &str,
) -> Result<Option<usize>, ToolExecutionFailure> {
    invocation
        .arguments()
        .get(name)
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(invalid_failure)
        })
        .transpose()
}

fn failure_for(code: crate::CodingErrorCode, message: &str) -> ToolExecutionFailure {
    let details = ProtocolMetadata::from_entries([(
        ERROR_NAMESPACE,
        json!({"code": coding_error_code(code)}),
    )])
    .unwrap_or_default();
    ToolExecutionFailure::execution(message)
        .unwrap_or_else(|_| ToolExecutionFailure::internal_contract())
        .with_details(details)
}

fn coding_error_code(code: crate::CodingErrorCode) -> &'static str {
    match code {
        crate::CodingErrorCode::InvalidInput => "invalid_input",
        crate::CodingErrorCode::NotFound => "not_found",
        crate::CodingErrorCode::ProjectNotTrusted => "project_not_trusted",
        crate::CodingErrorCode::Persistence => "persistence",
        crate::CodingErrorCode::Credential => "credential",
        crate::CodingErrorCode::Authentication => "authentication",
        crate::CodingErrorCode::PermissionDenied => "permission_denied",
        crate::CodingErrorCode::RateLimited => "rate_limited",
        crate::CodingErrorCode::ContextOverflow => "context_overflow",
        crate::CodingErrorCode::Unavailable => "unavailable",
        crate::CodingErrorCode::Transport => "transport",
        crate::CodingErrorCode::InvalidRequest => "invalid_request",
        crate::CodingErrorCode::PolicyDenied => "policy_denied",
        crate::CodingErrorCode::Cancelled => "cancelled",
        crate::CodingErrorCode::Runtime => "runtime",
        crate::CodingErrorCode::Internal => "internal",
    }
}

fn invalid_failure() -> ToolExecutionFailure {
    failure_for(
        crate::CodingErrorCode::InvalidInput,
        "skill resource arguments are invalid",
    )
}

fn internal_failure() -> ToolExecutionFailure {
    failure_for(
        crate::CodingErrorCode::Runtime,
        "skill resource tool failed internally",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spec_declares_bounded_read_only_contract() {
        let spec = ReadSkillResourceTool::spec().unwrap();
        assert_eq!(spec.name().as_str(), READ_SKILL_RESOURCE_TOOL_NAME);
        assert_eq!(spec.version().to_string(), "1.0.0");
        assert_eq!(spec.effects(), &[ToolEffect::FsRead]);
        let execution = spec.execution();
        assert_eq!(execution.idempotency(), ToolIdempotency::Idempotent);
        assert_eq!(execution.retry_safety(), ToolRetrySafety::Automatic);
        assert_eq!(execution.concurrency(), ToolConcurrency::Parallel);
        assert_eq!(execution.timeout().as_millis(), 30_000);
        assert_eq!(
            spec.input_schema()["properties"]["path"]["maxLength"],
            1_800
        );
    }

    #[test]
    fn resolver_returns_only_a_canonical_opaque_skill_locator() {
        let resolver = SkillResourceResolver;
        let name = ToolName::from_str(READ_SKILL_RESOURCE_TOOL_NAME).unwrap();
        let resources = resolver
            .resolve(
                &name,
                &json!({"skill":"review","path":"./references/checklist.md"}),
            )
            .unwrap();
        assert_eq!(resources.len(), 1);
        assert_eq!(resources[0].scheme(), "skill");
        assert_eq!(resources[0].locator(), "/review/references/checklist.md");
        assert_eq!(resources[0].access(), ToolResourceAccess::Read);
        for path in ["", "../escape", "/etc/passwd", "C:/secret", "a\\b"] {
            assert!(
                resolver
                    .resolve(&name, &json!({"skill":"review","path":path}))
                    .is_err()
            );
        }
        assert!(
            resolver
                .resolve(&name, &json!({"skill":"../review","path":"x"}))
                .is_err()
        );
    }
}
