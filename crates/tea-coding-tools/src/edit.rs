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

use crate::edit_diff::code_change;
use crate::file::{atomic_write_if_unchanged, read_bounded_utf8};
use crate::output::{failure, success};
use crate::read::string_argument;
use crate::{FileToolError, FileToolErrorCode, WorkspaceRoot};

/// Workspace-confined exact-match atomic UTF-8 editor.
#[derive(Debug, Clone)]
pub struct EditTool {
    workspace: WorkspaceRoot,
}

impl EditTool {
    /// Creates an editor bound to one validated workspace capability.
    #[must_use]
    pub const fn new(workspace: WorkspaceRoot) -> Self {
        Self { workspace }
    }

    /// Builds the portable `edit` tool contract.
    ///
    /// # Errors
    ///
    /// Returns an error only if the static contract violates tool bounds.
    pub fn spec() -> Result<ToolSpec, ToolSpecError> {
        ToolSpec::new(
            ToolName::from_str("edit").map_err(|_| ToolSpecError::InvalidDescription)?,
            ToolVersion::from_str("1.0.0").map_err(|_| ToolSpecError::InvalidDescription)?,
            "Atomically replace exact UTF-8 text in one workspace file.",
            json!({
                "type":"object",
                "properties":{
                    "path":{"type":"string","minLength":1,"maxLength":4096},
                    "oldText":{"type":"string","minLength":1,"maxLength":crate::MAX_WRITE_BYTES},
                    "newText":{"type":"string","maxLength":crate::MAX_WRITE_BYTES},
                    "expectedReplacements":{"type":"integer","minimum":1,"maximum":100_000}
                },
                "required":["path","oldText","newText"],
                "additionalProperties":false
            }),
            json!({
                "type":"object",
                "properties":{
                    "path":{"type":"string"},
                    "replacements":{"type":"integer","minimum":1},
                    "writtenBytes":{"type":"integer","minimum":0}
                },
                "required":["path","replacements","writtenBytes"],
                "additionalProperties":false
            }),
            [ToolEffect::FsRead, ToolEffect::FsWrite],
            ToolExecutionSemantics::new(
                ToolIdempotency::NonIdempotent,
                ToolRetrySafety::ExplicitOnly,
                ToolConcurrency::Serial,
                ToolTimeout::from_millis(30_000)?,
            )?,
        )?
        .with_prompt_hint(
            "Use edit for exact text replacement; ambiguous matches fail unless expectedReplacements is set.",
        )
    }

    fn run(
        &self,
        invocation: &ValidatedToolInvocation,
    ) -> Result<ToolExecutionEvent, FileToolError> {
        let path = string_argument(invocation, "path")?;
        let old_text = string_argument(invocation, "oldText")?;
        let new_text = string_argument(invocation, "newText")?;
        let expected = expected_replacements(invocation)?;
        let existing = self.workspace.resolve_existing(path)?;
        let source = read_bounded_utf8(&self.workspace, &existing, crate::MAX_WRITE_BYTES)?;
        let matches = source.match_indices(old_text).count();
        if matches == 0 {
            return Err(FileToolError::new(FileToolErrorCode::NoMatch));
        }
        let expected = expected.unwrap_or(1);
        if matches != expected {
            return Err(FileToolError::new(FileToolErrorCode::MatchCountMismatch));
        }
        let result = source.replace(old_text, new_text);
        if result.len() > crate::MAX_WRITE_BYTES {
            return Err(FileToolError::new(FileToolErrorCode::TooLarge));
        }
        let change = code_change(
            existing.display_path(),
            &source,
            &result,
            tea_protocol::CodeChangeKind::Update,
        )?;
        let mutation = self.workspace.resolve_mutation(path)?;
        atomic_write_if_unchanged(
            &self.workspace,
            &existing,
            &mutation,
            &source,
            result.as_bytes(),
        )?;
        let event = success(
            format!(
                "replaced {matches} match(es) in {}",
                existing.display_path()
            ),
            json!({
                "path":existing.display_path(),
                "replacements":matches,
                "writtenBytes":result.len()
            }),
        );
        let ToolExecutionEvent::Finished(result) = event else {
            return Err(FileToolError::new(FileToolErrorCode::Internal));
        };
        Ok(ToolExecutionEvent::Finished(result.with_presentation(
            tea_protocol::ToolPresentation::CodeChange(change),
        )))
    }

    fn preview_change(
        &self,
        invocation: &ValidatedToolInvocation,
    ) -> Result<tea_protocol::CodeChange, FileToolError> {
        let path = string_argument(invocation, "path")?;
        let old_text = string_argument(invocation, "oldText")?;
        let new_text = string_argument(invocation, "newText")?;
        let expected = expected_replacements(invocation)?;
        let existing = self.workspace.resolve_existing(path)?;
        let source = read_bounded_utf8(&self.workspace, &existing, crate::MAX_WRITE_BYTES)?;
        let matches = source.match_indices(old_text).count();
        if matches == 0 {
            return Err(FileToolError::new(FileToolErrorCode::NoMatch));
        }
        let expected = expected.unwrap_or(1);
        if matches != expected {
            return Err(FileToolError::new(FileToolErrorCode::MatchCountMismatch));
        }
        let result = source.replace(old_text, new_text);
        if result.len() > crate::MAX_WRITE_BYTES {
            return Err(FileToolError::new(FileToolErrorCode::TooLarge));
        }
        code_change(
            existing.display_path(),
            &source,
            &result,
            tea_protocol::CodeChangeKind::Update,
        )
    }
}

impl ToolExecutor for EditTool {
    fn preview(
        &self,
        invocation: &ValidatedToolInvocation,
    ) -> Option<tea_protocol::ToolPresentation> {
        self.preview_change(invocation)
            .ok()
            .map(tea_protocol::ToolPresentation::CodeChange)
    }

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

fn expected_replacements(
    invocation: &ValidatedToolInvocation,
) -> Result<Option<usize>, FileToolError> {
    invocation
        .arguments()
        .get("expectedReplacements")
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| FileToolError::new(FileToolErrorCode::InvalidArguments))
        })
        .transpose()
}
