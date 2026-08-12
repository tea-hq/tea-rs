use std::str::FromStr;

use futures_util::stream;
use serde_json::json;
use tea_control::CancellationScope;
use tea_protocol::ToolIdempotency;
use tea_tools::{
    BoxToolExecutionStream, ToolConcurrency, ToolEffect, ToolExecutionEvent,
    ToolExecutionSemantics, ToolExecutor, ToolName, ToolRetrySafety, ToolSpec, ToolSpecError,
    ToolTimeout, ToolVersion, ValidatedToolInvocation,
};

use crate::file::{DEFAULT_READ_LINE_LIMIT, MAX_READ_LINE_LIMIT, read_bounded_utf8};
use crate::output::{failure, success};
use crate::{FileToolError, FileToolErrorCode, WorkspaceRoot};

/// Workspace-confined bounded UTF-8 text reader.
#[derive(Debug, Clone)]
pub struct ReadTool {
    workspace: WorkspaceRoot,
}

impl ReadTool {
    /// Creates a reader bound to one validated workspace capability.
    #[must_use]
    pub const fn new(workspace: WorkspaceRoot) -> Self {
        Self { workspace }
    }

    /// Builds the portable `read` tool contract.
    ///
    /// # Errors
    ///
    /// Returns an error only if the static contract violates tool bounds.
    pub fn spec() -> Result<ToolSpec, ToolSpecError> {
        ToolSpec::new(
            ToolName::from_str("read").map_err(|_| ToolSpecError::InvalidDescription)?,
            ToolVersion::from_str("1.0.0").map_err(|_| ToolSpecError::InvalidDescription)?,
            "Read bounded UTF-8 text from one workspace file.",
            json!({
                "type":"object",
                "properties":{
                    "path":{"type":"string","minLength":1,"maxLength":4096},
                    "offset":{"type":"integer","minimum":1},
                    "limit":{"type":"integer","minimum":1,"maximum":MAX_READ_LINE_LIMIT}
                },
                "required":["path"],
                "additionalProperties":false
            }),
            json!({
                "type":"object",
                "properties":{
                    "path":{"type":"string"},
                    "content":{"type":"string"},
                    "startLine":{"type":"integer","minimum":1},
                    "endLine":{"type":"integer","minimum":0},
                    "totalLines":{"type":"integer","minimum":0},
                    "truncated":{"type":"boolean"}
                },
                "required":["path","content","startLine","endLine","totalLines","truncated"],
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
            "Use read with workspace-relative paths; offset and limit are 1-based lines.",
        )
    }

    fn run(
        &self,
        invocation: &ValidatedToolInvocation,
    ) -> Result<ToolExecutionEvent, FileToolError> {
        let path = string_argument(invocation, "path")?;
        let offset = integer_argument(invocation, "offset")?.unwrap_or(1);
        let limit = integer_argument(invocation, "limit")?.unwrap_or(DEFAULT_READ_LINE_LIMIT);
        let target = self.workspace.resolve_existing(path)?;
        let source = read_bounded_utf8(&self.workspace, &target, crate::MAX_READ_BYTES)?;
        let lines = source.split_inclusive('\n').collect::<Vec<_>>();
        let start_index = offset.saturating_sub(1).min(lines.len());
        let end_index = start_index.saturating_add(limit).min(lines.len());
        let content = lines[start_index..end_index].concat();
        let end_line = if end_index == start_index {
            0
        } else {
            end_index
        };
        let truncated = start_index > 0 || end_index < lines.len();
        let visible = if content.is_empty() {
            "(empty text result)".to_owned()
        } else {
            content.clone()
        };
        Ok(success(
            visible,
            json!({
                "path":target.display_path(),
                "content":content,
                "startLine":offset,
                "endLine":end_line,
                "totalLines":lines.len(),
                "truncated":truncated
            }),
        ))
    }
}

impl ToolExecutor for ReadTool {
    fn execute(
        &self,
        invocation: ValidatedToolInvocation,
        cancellation: CancellationScope,
    ) -> BoxToolExecutionStream {
        let executor = self.clone();
        Box::pin(stream::once(async move {
            if cancellation.is_cancelled() {
                ToolExecutionEvent::Failed(tea_tools::ToolExecutionFailure::cancelled())
            } else {
                executor.run(&invocation).unwrap_or_else(failure)
            }
        }))
    }
}

pub(crate) fn string_argument<'a>(
    invocation: &'a ValidatedToolInvocation,
    name: &str,
) -> Result<&'a str, FileToolError> {
    invocation
        .arguments()
        .get(name)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| FileToolError::new(FileToolErrorCode::InvalidArguments))
}

fn integer_argument(
    invocation: &ValidatedToolInvocation,
    name: &str,
) -> Result<Option<usize>, FileToolError> {
    invocation
        .arguments()
        .get(name)
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| FileToolError::new(FileToolErrorCode::InvalidArguments))
        })
        .transpose()
}
