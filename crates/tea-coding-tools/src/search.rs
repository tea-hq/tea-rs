use std::fs;
use std::str::FromStr;

use futures_util::stream;
use globset::{Glob, GlobMatcher};
use ignore::{DirEntry, WalkBuilder};
use regex::{Regex, RegexBuilder};
use serde_json::{Value, json};
use tea_control::CancellationScope;
use tea_protocol::ToolIdempotency;
use tea_tools::{
    BoxToolExecutionStream, ToolConcurrency, ToolEffect, ToolExecutionEvent, ToolExecutionFailure,
    ToolExecutionSemantics, ToolExecutor, ToolName, ToolRetrySafety, ToolSpec, ToolSpecError,
    ToolTimeout, ToolVersion, ValidatedToolInvocation,
};

use crate::file::read_bounded_utf8;
use crate::output::{failure, success};
use crate::read::string_argument;
use crate::{FileToolError, FileToolErrorCode, WorkspaceRoot};

/// Default maximum number of entries returned by a search tool.
pub const DEFAULT_SEARCH_RESULT_LIMIT: usize = 100;
/// Maximum result limit accepted by a search tool invocation.
pub const MAX_SEARCH_RESULT_LIMIT: usize = 1_000;
/// Maximum recursive depth accepted by `ls`.
pub const MAX_SEARCH_DEPTH: usize = 8;

const MAX_SEARCH_PATTERN_BYTES: usize = 1_024;
const MAX_SEARCH_FILE_BYTES: usize = 1024 * 1024;
const MAX_SEARCH_SCANNED_BYTES: usize = 16 * 1024 * 1024;
const MAX_SEARCH_ENTRIES: usize = 20_000;
const MAX_SEARCH_OUTPUT_BYTES: usize = 128 * 1024;
const MAX_VISIBLE_OUTPUT_BYTES: usize = 32 * 1024;
const MAX_MATCH_LINE_BYTES: usize = 1_024;

/// Workspace-confined bounded regular-expression search.
#[derive(Debug, Clone)]
pub struct GrepTool {
    workspace: WorkspaceRoot,
}

impl GrepTool {
    /// Creates a grep tool bound to one validated workspace capability.
    #[must_use]
    pub const fn new(workspace: WorkspaceRoot) -> Self {
        Self { workspace }
    }

    /// Builds the portable `grep` tool contract.
    ///
    /// # Errors
    ///
    /// Returns an error only if the static contract violates tool bounds.
    pub fn spec() -> Result<ToolSpec, ToolSpecError> {
        search_spec(
            "grep",
            "Search bounded UTF-8 workspace files with a regular expression.",
            json!({
                "type":"object",
                "properties":{
                    "pattern":{"type":"string","minLength":1,"maxLength":MAX_SEARCH_PATTERN_BYTES},
                    "path":{"type":"string","minLength":1,"maxLength":4096},
                    "glob":{"type":"string","minLength":1,"maxLength":MAX_SEARCH_PATTERN_BYTES},
                    "caseSensitive":{"type":"boolean"},
                    "limit":{"type":"integer","minimum":1,"maximum":MAX_SEARCH_RESULT_LIMIT}
                },
                "required":["pattern"],
                "additionalProperties":false
            }),
            json!({
                "type":"object",
                "properties":{
                    "matches":{"type":"array","items":{"type":"object"}},
                    "scannedFiles":{"type":"integer","minimum":0},
                    "truncated":{"type":"boolean"}
                },
                "required":["matches","scannedFiles","truncated"],
                "additionalProperties":false
            }),
            "Use grep for regex content search; path defaults to the workspace root and glob filters files.",
        )
    }

    fn run(
        &self,
        invocation: &ValidatedToolInvocation,
        cancellation: &CancellationScope,
    ) -> Result<ToolExecutionEvent, SearchRunError> {
        let pattern = string_argument(invocation, "pattern")?;
        let case_sensitive = bool_argument(invocation, "caseSensitive")?.unwrap_or(true);
        let regex = RegexBuilder::new(pattern)
            .case_insensitive(!case_sensitive)
            .size_limit(1024 * 1024)
            .build()
            .map_err(|_| FileToolError::new(FileToolErrorCode::InvalidArguments))?;
        let glob = optional_glob(invocation, "glob")?;
        let limit = result_limit(invocation)?;
        let root = self.search_root(invocation)?;
        let mut matches = Vec::new();
        let mut visible = String::new();
        let mut scanned_files = 0_usize;
        let mut scanned_bytes = 0_usize;
        let mut output_bytes = 0_usize;
        let mut truncated = false;

        for entry in walker(root.host_path(), None) {
            check_cancelled(cancellation)?;
            let entry =
                entry.map_err(|_| FileToolError::new(FileToolErrorCode::FilesystemFailure))?;
            if !is_regular_file(&entry) {
                continue;
            }
            let Some(relative) = relative_path(&self.workspace, entry.path()) else {
                continue;
            };
            if glob
                .as_ref()
                .is_some_and(|glob| !glob_matches(glob, &relative))
            {
                continue;
            }
            let metadata = fs::metadata(entry.path())
                .map_err(|_| FileToolError::new(FileToolErrorCode::FilesystemFailure))?;
            let file_bytes = usize::try_from(metadata.len()).unwrap_or(usize::MAX);
            if file_bytes > MAX_SEARCH_FILE_BYTES {
                continue;
            }
            if scanned_files >= MAX_SEARCH_ENTRIES
                || scanned_bytes.saturating_add(file_bytes) > MAX_SEARCH_SCANNED_BYTES
            {
                truncated = true;
                break;
            }
            scanned_files += 1;
            scanned_bytes += file_bytes;
            let Ok(target) = self.workspace.resolve_existing(&relative) else {
                continue;
            };
            let source = match read_bounded_utf8(&self.workspace, &target, MAX_SEARCH_FILE_BYTES) {
                Ok(source) => source,
                Err(error)
                    if matches!(
                        error.code(),
                        FileToolErrorCode::BinaryFile
                            | FileToolErrorCode::InvalidUtf8
                            | FileToolErrorCode::TooLarge
                    ) =>
                {
                    continue;
                }
                Err(error) => return Err(error.into()),
            };
            if collect_file_matches(
                &regex,
                &relative,
                &source,
                limit,
                &mut matches,
                &mut visible,
                &mut output_bytes,
            ) {
                truncated = true;
                break;
            }
        }
        if visible.is_empty() {
            visible.push_str("No matches found.");
        }
        Ok(success(
            visible,
            json!({
                "matches": matches,
                "scannedFiles": scanned_files,
                "truncated": truncated
            }),
        ))
    }

    fn search_root(
        &self,
        invocation: &ValidatedToolInvocation,
    ) -> Result<crate::ResolvedExistingPath, FileToolError> {
        let path = optional_string_argument(invocation, "path")?.unwrap_or(".");
        self.workspace.resolve_existing(path).map_err(Into::into)
    }
}

impl ToolExecutor for GrepTool {
    fn execute(
        &self,
        invocation: ValidatedToolInvocation,
        cancellation: CancellationScope,
    ) -> BoxToolExecutionStream {
        let executor = self.clone();
        Box::pin(stream::once(async move {
            terminal(executor.run(&invocation, &cancellation))
        }))
    }
}

/// Workspace-confined bounded path search.
#[derive(Debug, Clone)]
pub struct FindTool {
    workspace: WorkspaceRoot,
}

impl FindTool {
    /// Creates a find tool bound to one validated workspace capability.
    #[must_use]
    pub const fn new(workspace: WorkspaceRoot) -> Self {
        Self { workspace }
    }

    /// Builds the portable `find` tool contract.
    ///
    /// # Errors
    ///
    /// Returns an error only if the static contract violates tool bounds.
    pub fn spec() -> Result<ToolSpec, ToolSpecError> {
        search_spec(
            "find",
            "Find workspace paths with a bounded glob pattern.",
            json!({
                "type":"object",
                "properties":{
                    "pattern":{"type":"string","minLength":1,"maxLength":MAX_SEARCH_PATTERN_BYTES},
                    "path":{"type":"string","minLength":1,"maxLength":4096},
                    "limit":{"type":"integer","minimum":1,"maximum":MAX_SEARCH_RESULT_LIMIT}
                },
                "required":["pattern"],
                "additionalProperties":false
            }),
            list_output_schema("paths"),
            "Use find for glob path search; path defaults to the workspace root.",
        )
    }

    fn run(
        &self,
        invocation: &ValidatedToolInvocation,
        cancellation: &CancellationScope,
    ) -> Result<ToolExecutionEvent, SearchRunError> {
        let pattern = string_argument(invocation, "pattern")?;
        let matcher = compile_glob(pattern)?;
        let limit = result_limit(invocation)?;
        let path = optional_string_argument(invocation, "path")?.unwrap_or(".");
        let root = self
            .workspace
            .resolve_existing(path)
            .map_err(FileToolError::from)?;
        let mut paths = Vec::new();
        let mut output_bytes = 0_usize;
        let mut truncated = false;
        for (visited, entry) in walker(root.host_path(), None).enumerate() {
            check_cancelled(cancellation)?;
            if visited >= MAX_SEARCH_ENTRIES {
                truncated = true;
                break;
            }
            let entry =
                entry.map_err(|_| FileToolError::new(FileToolErrorCode::FilesystemFailure))?;
            if (entry.depth() == 0 && is_directory(&entry)) || is_symlink(&entry) {
                continue;
            }
            let Some(relative) = relative_path(&self.workspace, entry.path()) else {
                continue;
            };
            if !glob_matches(&matcher, &relative) {
                continue;
            }
            if paths.len() >= limit
                || output_bytes.saturating_add(relative.len()) > MAX_SEARCH_OUTPUT_BYTES
            {
                truncated = true;
                break;
            }
            output_bytes += relative.len();
            paths.push(relative);
        }
        let visible = visible_paths(&paths, truncated);
        Ok(success(
            visible,
            json!({"paths":paths,"truncated":truncated}),
        ))
    }
}

impl ToolExecutor for FindTool {
    fn execute(
        &self,
        invocation: ValidatedToolInvocation,
        cancellation: CancellationScope,
    ) -> BoxToolExecutionStream {
        let executor = self.clone();
        Box::pin(stream::once(async move {
            terminal(executor.run(&invocation, &cancellation))
        }))
    }
}

/// Workspace-confined bounded directory listing.
#[derive(Debug, Clone)]
pub struct LsTool {
    workspace: WorkspaceRoot,
}

impl LsTool {
    /// Creates an ls tool bound to one validated workspace capability.
    #[must_use]
    pub const fn new(workspace: WorkspaceRoot) -> Self {
        Self { workspace }
    }

    /// Builds the portable `ls` tool contract.
    ///
    /// # Errors
    ///
    /// Returns an error only if the static contract violates tool bounds.
    pub fn spec() -> Result<ToolSpec, ToolSpecError> {
        search_spec(
            "ls",
            "List bounded workspace directory entries.",
            json!({
                "type":"object",
                "properties":{
                    "path":{"type":"string","minLength":1,"maxLength":4096},
                    "depth":{"type":"integer","minimum":1,"maximum":MAX_SEARCH_DEPTH},
                    "limit":{"type":"integer","minimum":1,"maximum":MAX_SEARCH_RESULT_LIMIT}
                },
                "additionalProperties":false
            }),
            json!({
                "type":"object",
                "properties":{
                    "entries":{"type":"array","items":{"type":"object"}},
                    "truncated":{"type":"boolean"}
                },
                "required":["entries","truncated"],
                "additionalProperties":false
            }),
            "Use ls for directory structure; path defaults to the workspace root and depth defaults to one.",
        )
    }

    fn run(
        &self,
        invocation: &ValidatedToolInvocation,
        cancellation: &CancellationScope,
    ) -> Result<ToolExecutionEvent, SearchRunError> {
        let path = optional_string_argument(invocation, "path")?.unwrap_or(".");
        let depth = usize_argument(invocation, "depth")?.unwrap_or(1);
        let limit = result_limit(invocation)?;
        let root = self
            .workspace
            .resolve_existing(path)
            .map_err(FileToolError::from)?;
        let metadata = fs::metadata(root.host_path())
            .map_err(|_| FileToolError::new(FileToolErrorCode::FilesystemFailure))?;
        if !metadata.is_dir() {
            return Err(FileToolError::new(FileToolErrorCode::NotAFile).into());
        }
        let mut entries = Vec::new();
        let mut visible_lines = Vec::new();
        let mut output_bytes = 0_usize;
        let mut truncated = false;
        for (visited, entry) in walker(root.host_path(), Some(depth)).enumerate() {
            check_cancelled(cancellation)?;
            if visited >= MAX_SEARCH_ENTRIES {
                truncated = true;
                break;
            }
            let entry =
                entry.map_err(|_| FileToolError::new(FileToolErrorCode::FilesystemFailure))?;
            if entry.depth() == 0 || is_symlink(&entry) {
                continue;
            }
            let Some(relative) = relative_path(&self.workspace, entry.path()) else {
                continue;
            };
            let kind = if is_directory(&entry) {
                "directory"
            } else {
                "file"
            };
            if entries.len() >= limit
                || output_bytes.saturating_add(relative.len()) > MAX_SEARCH_OUTPUT_BYTES
            {
                truncated = true;
                break;
            }
            output_bytes += relative.len();
            visible_lines.push(format!(
                "{relative}{}",
                if kind == "directory" { "/" } else { "" }
            ));
            entries.push(json!({"path":relative,"kind":kind}));
        }
        let mut visible = bounded_lines(&visible_lines);
        if visible.is_empty() {
            visible.push_str("(empty directory)");
        }
        if truncated {
            visible.push_str("\n[results truncated]");
        }
        Ok(success(
            visible,
            json!({"entries":entries,"truncated":truncated}),
        ))
    }
}

impl ToolExecutor for LsTool {
    fn execute(
        &self,
        invocation: ValidatedToolInvocation,
        cancellation: CancellationScope,
    ) -> BoxToolExecutionStream {
        let executor = self.clone();
        Box::pin(stream::once(async move {
            terminal(executor.run(&invocation, &cancellation))
        }))
    }
}

fn search_spec(
    name: &str,
    description: &str,
    input_schema: Value,
    output_schema: Value,
    prompt_hint: &str,
) -> Result<ToolSpec, ToolSpecError> {
    ToolSpec::new(
        ToolName::from_str(name).map_err(|_| ToolSpecError::InvalidDescription)?,
        ToolVersion::from_str("1.0.0").map_err(|_| ToolSpecError::InvalidDescription)?,
        description,
        input_schema,
        output_schema,
        [ToolEffect::FsRead],
        ToolExecutionSemantics::new(
            ToolIdempotency::Idempotent,
            ToolRetrySafety::Automatic,
            ToolConcurrency::Parallel,
            ToolTimeout::from_millis(30_000)?,
        )?,
    )?
    .with_prompt_hint(prompt_hint)
}

fn list_output_schema(field: &str) -> Value {
    json!({
        "type":"object",
        "properties":{
            field:{"type":"array","items":{"type":"string"}},
            "truncated":{"type":"boolean"}
        },
        "required":[field,"truncated"],
        "additionalProperties":false
    })
}

fn walker(path: &std::path::Path, max_depth: Option<usize>) -> ignore::Walk {
    let mut builder = WalkBuilder::new(path);
    builder
        .hidden(false)
        .parents(true)
        .require_git(false)
        .git_ignore(true)
        .git_exclude(true)
        .git_global(false)
        .follow_links(false)
        .same_file_system(true)
        .max_depth(max_depth)
        .sort_by_file_path(std::path::Path::cmp);
    builder.build()
}

fn relative_path(workspace: &WorkspaceRoot, path: &std::path::Path) -> Option<String> {
    let relative = path.strip_prefix(workspace.host_path()).ok()?;
    let relative = relative.to_str()?.replace(std::path::MAIN_SEPARATOR, "/");
    Some(if relative.is_empty() {
        ".".to_owned()
    } else {
        relative
    })
}

fn is_regular_file(entry: &DirEntry) -> bool {
    entry.file_type().is_some_and(|kind| kind.is_file())
}

fn is_directory(entry: &DirEntry) -> bool {
    entry.file_type().is_some_and(|kind| kind.is_dir())
}

fn is_symlink(entry: &DirEntry) -> bool {
    entry.file_type().is_some_and(|kind| kind.is_symlink())
}

fn compile_glob(pattern: &str) -> Result<GlobMatcher, FileToolError> {
    Glob::new(pattern)
        .map(|glob| glob.compile_matcher())
        .map_err(|_| FileToolError::new(FileToolErrorCode::InvalidArguments))
}

fn optional_glob(
    invocation: &ValidatedToolInvocation,
    name: &str,
) -> Result<Option<GlobMatcher>, FileToolError> {
    optional_string_argument(invocation, name)?
        .map(compile_glob)
        .transpose()
}

fn glob_matches(matcher: &GlobMatcher, relative: &str) -> bool {
    matcher.is_match(relative)
        || relative
            .rsplit_once('/')
            .is_some_and(|(_, name)| matcher.is_match(name))
}

fn collect_file_matches(
    regex: &Regex,
    path: &str,
    source: &str,
    limit: usize,
    matches: &mut Vec<Value>,
    visible: &mut String,
    output_bytes: &mut usize,
) -> bool {
    for (line_index, line) in source.lines().enumerate() {
        let Some(found) = regex.find(line) else {
            continue;
        };
        let text = truncate_utf8(line, MAX_MATCH_LINE_BYTES);
        let estimated = path.len().saturating_add(text.len()).saturating_add(64);
        if matches.len() >= limit
            || output_bytes.saturating_add(estimated) > MAX_SEARCH_OUTPUT_BYTES
        {
            return true;
        }
        *output_bytes += estimated;
        let line_number = line_index + 1;
        let column = line[..found.start()].chars().count() + 1;
        matches.push(json!({
            "path":path,
            "line":line_number,
            "column":column,
            "text":text
        }));
        if visible.len() < MAX_VISIBLE_OUTPUT_BYTES {
            let rendered = format!("{path}:{line_number}:{column}: {text}\n");
            let remaining = MAX_VISIBLE_OUTPUT_BYTES - visible.len();
            visible.push_str(truncate_utf8(&rendered, remaining));
        }
    }
    false
}

fn visible_paths(paths: &[String], truncated: bool) -> String {
    let mut visible = bounded_lines(paths);
    if visible.is_empty() {
        visible.push_str("No paths found.");
    }
    if truncated {
        visible.push_str("\n[results truncated]");
    }
    visible
}

fn bounded_lines(lines: &[String]) -> String {
    let mut visible = String::new();
    for line in lines {
        if visible.len() >= MAX_VISIBLE_OUTPUT_BYTES {
            break;
        }
        let remaining = MAX_VISIBLE_OUTPUT_BYTES - visible.len();
        let rendered = format!("{line}\n");
        visible.push_str(truncate_utf8(&rendered, remaining));
    }
    visible.trim_end_matches('\n').to_owned()
}

fn truncate_utf8(text: &str, max_bytes: usize) -> &str {
    if text.len() <= max_bytes {
        return text;
    }
    let mut end = max_bytes;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

fn optional_string_argument<'a>(
    invocation: &'a ValidatedToolInvocation,
    name: &str,
) -> Result<Option<&'a str>, FileToolError> {
    invocation
        .arguments()
        .get(name)
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| FileToolError::new(FileToolErrorCode::InvalidArguments))
        })
        .transpose()
}

fn usize_argument(
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

fn bool_argument(
    invocation: &ValidatedToolInvocation,
    name: &str,
) -> Result<Option<bool>, FileToolError> {
    invocation
        .arguments()
        .get(name)
        .map(|value| {
            value
                .as_bool()
                .ok_or_else(|| FileToolError::new(FileToolErrorCode::InvalidArguments))
        })
        .transpose()
}

fn result_limit(invocation: &ValidatedToolInvocation) -> Result<usize, FileToolError> {
    Ok(usize_argument(invocation, "limit")?.unwrap_or(DEFAULT_SEARCH_RESULT_LIMIT))
}

fn check_cancelled(cancellation: &CancellationScope) -> Result<(), SearchRunError> {
    if cancellation.is_cancelled() {
        Err(SearchRunError::Cancelled)
    } else {
        Ok(())
    }
}

fn terminal(result: Result<ToolExecutionEvent, SearchRunError>) -> ToolExecutionEvent {
    match result {
        Ok(event) => event,
        Err(SearchRunError::File(error)) => failure(error),
        Err(SearchRunError::Cancelled) => {
            ToolExecutionEvent::Failed(ToolExecutionFailure::cancelled())
        }
    }
}

enum SearchRunError {
    File(FileToolError),
    Cancelled,
}

impl From<FileToolError> for SearchRunError {
    fn from(error: FileToolError) -> Self {
        Self::File(error)
    }
}
