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
use crate::file::{atomic_write, atomic_write_if_unchanged, read_bounded_utf8};
use crate::output::{failure, success};
use crate::read::string_argument;
use crate::{FileToolError, FileToolErrorCode, WorkspaceRoot};

/// Workspace-confined atomic UTF-8 file writer.
#[derive(Debug, Clone)]
pub struct WriteTool {
    workspace: WorkspaceRoot,
}

impl WriteTool {
    /// Creates a writer bound to one validated workspace capability.
    #[must_use]
    pub const fn new(workspace: WorkspaceRoot) -> Self {
        Self { workspace }
    }

    /// Builds the portable `write` tool contract.
    ///
    /// # Errors
    ///
    /// Returns an error only if the static contract violates tool bounds.
    pub fn spec() -> Result<ToolSpec, ToolSpecError> {
        ToolSpec::new(
            ToolName::from_str("write").map_err(|_| ToolSpecError::InvalidDescription)?,
            ToolVersion::from_str("1.0.0").map_err(|_| ToolSpecError::InvalidDescription)?,
            "Atomically create or replace one workspace UTF-8 text file.",
            json!({
                "type":"object",
                "properties":{
                    "path":{"type":"string","minLength":1,"maxLength":4096},
                    "content":{"type":"string","maxLength":crate::MAX_WRITE_BYTES}
                },
                "required":["path","content"],
                "additionalProperties":false
            }),
            json!({
                "type":"object",
                "properties":{
                    "path":{"type":"string"},
                    "writtenBytes":{"type":"integer","minimum":0},
                    "created":{"type":"boolean"}
                },
                "required":["path","writtenBytes","created"],
                "additionalProperties":false
            }),
            [ToolEffect::FsWrite],
            ToolExecutionSemantics::new(
                ToolIdempotency::Idempotent,
                ToolRetrySafety::ExplicitOnly,
                ToolConcurrency::Serial,
                ToolTimeout::from_millis(30_000)?,
            )?,
        )?
        .with_prompt_hint("Use write for complete UTF-8 file content and workspace-relative paths.")
    }

    fn run(
        &self,
        invocation: &ValidatedToolInvocation,
    ) -> Result<ToolExecutionEvent, FileToolError> {
        let path = string_argument(invocation, "path")?;
        let content = string_argument(invocation, "content")?;
        if content.len() > crate::MAX_WRITE_BYTES {
            return Err(FileToolError::new(FileToolErrorCode::TooLarge));
        }
        let target = self.workspace.resolve_mutation(path)?;
        let created = !target.target_existed_at_resolution();
        let existing_text = if created {
            None
        } else {
            let existing = self.workspace.resolve_existing(path)?;
            match read_bounded_utf8(&self.workspace, &existing, crate::MAX_WRITE_BYTES) {
                Ok(source) => Some((existing, source)),
                Err(error)
                    if matches!(
                        error.code(),
                        FileToolErrorCode::BinaryFile
                            | FileToolErrorCode::InvalidUtf8
                            | FileToolErrorCode::TooLarge
                    ) =>
                {
                    None
                }
                Err(error) => return Err(error),
            }
        };
        let change = match existing_text.as_ref() {
            Some((existing, source)) => Some(code_change(
                existing.display_path(),
                source,
                content,
                tea_protocol::CodeChangeKind::Update,
            )?),
            None if created => Some(code_change(
                target.display_path(),
                "",
                content,
                tea_protocol::CodeChangeKind::Create,
            )?),
            None => None,
        };
        if let Some((existing, source)) = existing_text.as_ref() {
            atomic_write_if_unchanged(
                &self.workspace,
                existing,
                &target,
                source,
                content.as_bytes(),
            )?;
        } else {
            atomic_write(&self.workspace, &target, content.as_bytes())?;
        }
        let event = success(
            format!("wrote {} bytes to {}", content.len(), target.display_path()),
            json!({
                "path":target.display_path(),
                "writtenBytes":content.len(),
                "created":created
            }),
        );
        let ToolExecutionEvent::Finished(result) = event else {
            return Err(FileToolError::new(FileToolErrorCode::Internal));
        };
        let result = match change {
            Some(change) => {
                result.with_presentation(tea_protocol::ToolPresentation::CodeChange(change))
            }
            None => result,
        };
        Ok(ToolExecutionEvent::Finished(result))
    }

    fn preview_change(
        &self,
        invocation: &ValidatedToolInvocation,
    ) -> Result<tea_protocol::CodeChange, FileToolError> {
        let path = string_argument(invocation, "path")?;
        let content = string_argument(invocation, "content")?;
        if content.len() > crate::MAX_WRITE_BYTES {
            return Err(FileToolError::new(FileToolErrorCode::TooLarge));
        }
        let target = self.workspace.resolve_mutation(path)?;
        if !target.target_existed_at_resolution() {
            return code_change(
                target.display_path(),
                "",
                content,
                tea_protocol::CodeChangeKind::Create,
            );
        }
        let existing = self.workspace.resolve_existing(path)?;
        let source = read_bounded_utf8(&self.workspace, &existing, crate::MAX_WRITE_BYTES)?;
        code_change(
            existing.display_path(),
            &source,
            content,
            tea_protocol::CodeChangeKind::Update,
        )
    }
}

impl ToolExecutor for WriteTool {
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
