use std::collections::BTreeSet;
use std::fs::{self, File};
use std::io::Read as _;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use tea_context::{PromptSegmentId, TrustLevel, WorkspaceInstruction};

use crate::resources::ResourceDiagnostic;
use crate::{CodingError, CodingErrorCode, ProjectAccess};

const MAX_CONTEXT_FILES: usize = 32;
const MAX_CONTEXT_TOTAL_BYTES: usize = 256 * 1024;
const MAX_GIT_POINTER_BYTES: usize = 4096;
#[derive(Clone, Copy)]
struct ContextCandidate {
    filename: &'static str,
    locator: &'static str,
}

const CONTEXT_CANDIDATES: [ContextCandidate; 5] = [
    ContextCandidate {
        filename: "AGENTS.override.md",
        locator: "AGENTS.override.md",
    },
    ContextCandidate {
        filename: "AGENTS.md",
        locator: "AGENTS.md",
    },
    ContextCandidate {
        filename: "AGENTS.MD",
        locator: "AGENTS.md",
    },
    ContextCandidate {
        filename: "CLAUDE.md",
        locator: "CLAUDE.md",
    },
    ContextCandidate {
        filename: "CLAUDE.MD",
        locator: "CLAUDE.md",
    },
];

pub(crate) fn project_instruction_boundary(workspace: &Path) -> Result<PathBuf, CodingError> {
    let workspace = fs::canonicalize(workspace).map_err(|_| not_found())?;
    if !workspace.is_dir() {
        return Err(not_found());
    }
    let Some((worktree_root, common_git_dir)) = find_git_paths(&workspace, None) else {
        return Ok(workspace);
    };
    let Some(main_repo_root) = common_git_dir.parent() else {
        return Ok(worktree_root);
    };
    if worktree_root != main_repo_root && worktree_root.starts_with(main_repo_root) {
        let main_git_dir = fs::canonicalize(main_repo_root.join(".git")).ok();
        if main_git_dir.as_deref() == Some(common_git_dir.as_path()) {
            return Ok(main_repo_root.to_path_buf());
        }
    }
    Ok(worktree_root)
}

pub(crate) fn has_project_instructions(boundary: &Path, workspace: &Path) -> bool {
    let (Ok(boundary), Ok(workspace)) = (fs::canonicalize(boundary), fs::canonicalize(workspace))
    else {
        return false;
    };
    let Ok(relative) = workspace.strip_prefix(&boundary) else {
        return false;
    };
    let mut directories = vec![boundary.clone()];
    let mut current = boundary;
    for component in relative.components() {
        current = current.join(component);
        directories.push(current.clone());
    }
    directories.into_iter().any(|directory| {
        CONTEXT_CANDIDATES
            .iter()
            .any(|candidate| directory.join(candidate.filename).exists())
    })
}

pub(crate) fn discover(
    boundary: &Path,
    workspace: &Path,
    access: ProjectAccess,
    global_context_root: Option<&Path>,
) -> Result<(Vec<WorkspaceInstruction>, Vec<ResourceDiagnostic>, String), CodingError> {
    let boundary = fs::canonicalize(boundary).map_err(|_| not_found())?;
    let workspace = fs::canonicalize(workspace).map_err(|_| not_found())?;
    if !workspace.starts_with(&boundary) {
        return Err(invalid());
    }
    let relative = workspace.strip_prefix(&boundary).map_err(|_| invalid())?;
    let logical_workspace = logical_workspace_locator(relative)?;
    let mut instructions = Vec::new();
    let mut diagnostics = Vec::new();
    let mut total = 0_usize;
    let mut seen_targets = BTreeSet::new();
    if let Some(global_root) = global_context_root {
        load_directory(
            global_root,
            global_root,
            "<global>",
            &mut seen_targets,
            &mut instructions,
            &mut diagnostics,
            &mut total,
            None,
        )?;
    }
    if access == ProjectAccess::Trusted {
        let mut directories = vec![boundary.clone()];
        let mut current = boundary.clone();
        for component in relative.components() {
            current = current.join(component);
            directories.push(current.clone());
        }
        let shadowed_context = find_shadowed_context_file(&boundary, &workspace);
        for directory in directories {
            load_directory(
                &directory,
                &boundary,
                "",
                &mut seen_targets,
                &mut instructions,
                &mut diagnostics,
                &mut total,
                shadowed_context.as_deref(),
            )?;
        }
    }
    Ok((instructions, diagnostics, logical_workspace))
}

#[allow(clippy::too_many_arguments)]
fn load_directory(
    directory: &Path,
    containment_root: &Path,
    locator_prefix: &str,
    seen_targets: &mut BTreeSet<std::path::PathBuf>,
    instructions: &mut Vec<WorkspaceInstruction>,
    diagnostics: &mut Vec<ResourceDiagnostic>,
    total: &mut usize,
    shadowed_context: Option<&Path>,
) -> Result<(), CodingError> {
    let canonical_root = match fs::canonicalize(containment_root) {
        Ok(root) => root,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(_) => return Err(invalid()),
    };
    for candidate in CONTEXT_CANDIDATES {
        let path = directory.join(candidate.filename);
        let canonical_path = match fs::canonicalize(&path) {
            Ok(path) if path.starts_with(&canonical_root) => path,
            Ok(_) => return Err(invalid()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(_) => {
                diagnostics.push(ResourceDiagnostic::new(
                    "context_read_failed",
                    candidate.locator,
                ));
                continue;
            }
        };
        if !canonical_path.is_file() {
            continue;
        }
        if shadowed_context == Some(canonical_path.as_path()) {
            return Ok(());
        }
        if !seen_targets.insert(canonical_path.clone()) {
            return Ok(());
        }
        let Ok(bytes) = fs::read(&canonical_path) else {
            diagnostics.push(ResourceDiagnostic::new(
                "context_read_failed",
                candidate.locator,
            ));
            return Ok(());
        };
        let content = String::from_utf8(bytes).map_err(|_| invalid())?;
        if content.trim().is_empty() {
            return Ok(());
        }
        *total = total.saturating_add(content.len());
        if instructions.len() == MAX_CONTEXT_FILES || *total > MAX_CONTEXT_TOTAL_BYTES {
            return Err(CodingError::new(
                CodingErrorCode::InvalidInput,
                "workspace context files exceed bounds",
            ));
        }
        let relative = path
            .with_file_name(candidate.locator)
            .strip_prefix(containment_root)
            .map_err(|_| invalid())?
            .to_string_lossy()
            .replace('\\', "/");
        let locator = if locator_prefix.is_empty() {
            relative
        } else {
            format!("{locator_prefix}/{relative}")
        };
        let id = format!("workspace.context.{}", instructions.len());
        instructions.push(
            WorkspaceInstruction::new(
                PromptSegmentId::from_str(&id).map_err(|_| invalid())?,
                content,
                locator,
                TrustLevel::Delegated,
            )
            .map_err(|_| invalid())?,
        );
        return Ok(());
    }
    Ok(())
}

fn find_shadowed_context_file(boundary: &Path, workspace: &Path) -> Option<std::path::PathBuf> {
    let (worktree_root, common_git_dir) = find_git_paths(workspace, Some(boundary))?;
    let main_repo_root = common_git_dir.parent()?.to_path_buf();
    if worktree_root == main_repo_root || !worktree_root.starts_with(&main_repo_root) {
        return None;
    }
    let main_git_dir = fs::canonicalize(main_repo_root.join(".git")).ok()?;
    if main_git_dir != common_git_dir {
        return None;
    }
    let name = selected_context_name(&worktree_root)?;
    fs::canonicalize(main_repo_root.join(name)).ok()
}

fn find_git_paths(
    workspace: &Path,
    boundary: Option<&Path>,
) -> Option<(std::path::PathBuf, std::path::PathBuf)> {
    let mut directory = workspace.to_path_buf();
    loop {
        let dot_git = directory.join(".git");
        if dot_git.is_file() {
            let pointer = read_bounded_text(&dot_git)?;
            let git_dir_text = pointer.trim().strip_prefix("gitdir: ")?.trim();
            let git_dir = resolve_from(&directory, git_dir_text);
            let git_dir = fs::canonicalize(git_dir).ok()?;
            if !git_dir.join("HEAD").is_file() {
                return None;
            }
            let common_git_dir = if git_dir.join("commondir").is_file() {
                let common = read_bounded_text(&git_dir.join("commondir"))?;
                fs::canonicalize(resolve_from(&git_dir, common.trim())).ok()?
            } else {
                git_dir
            };
            return Some((directory, common_git_dir));
        }
        if dot_git.is_dir() {
            if !dot_git.join("HEAD").is_file() {
                return None;
            }
            return Some((directory, fs::canonicalize(dot_git).ok()?));
        }
        if boundary.is_some_and(|boundary| directory == boundary) {
            return None;
        }
        directory = directory.parent()?.to_path_buf();
        if boundary.is_some_and(|boundary| !directory.starts_with(boundary)) {
            return None;
        }
    }
}

fn selected_context_name(directory: &Path) -> Option<&'static str> {
    CONTEXT_CANDIDATES
        .into_iter()
        .find(|candidate| {
            fs::canonicalize(directory.join(candidate.filename)).is_ok_and(|path| path.is_file())
        })
        .map(|candidate| candidate.filename)
}

fn read_bounded_text(path: &Path) -> Option<String> {
    let bytes = fs::read(path).ok()?;
    if bytes.len() > MAX_GIT_POINTER_BYTES {
        return None;
    }
    String::from_utf8(bytes).ok()
}

fn resolve_from(base: &Path, path: &str) -> std::path::PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

pub(crate) fn add_explicit(
    workspace: &tea_coding_tools::WorkspaceRoot,
    logical_workspace: &str,
    paths: &[String],
    instructions: &mut Vec<WorkspaceInstruction>,
) -> Result<(), CodingError> {
    if paths.len() > MAX_CONTEXT_FILES
        || instructions.len().saturating_add(paths.len()) > MAX_CONTEXT_FILES
    {
        return Err(invalid());
    }
    let mut seen = BTreeSet::new();
    let mut total = 0_usize;
    let mut loaded = Vec::with_capacity(paths.len());
    for path in paths {
        let resolved = workspace.resolve_existing(path).map_err(|_| invalid())?;
        if !seen.insert(resolved.display_path().to_owned()) {
            return Err(invalid());
        }
        let mut file = File::open(resolved.host_path()).map_err(|_| invalid())?;
        let metadata = file.metadata().map_err(|_| invalid())?;
        workspace
            .verify_opened_existing(&resolved, &metadata)
            .map_err(|_| invalid())?;
        if !metadata.is_file() {
            return Err(invalid());
        }
        let declared = usize::try_from(metadata.len()).map_err(|_| invalid())?;
        total = total.checked_add(declared).ok_or_else(invalid)?;
        if total > MAX_CONTEXT_TOTAL_BYTES {
            return Err(invalid());
        }
        let mut bytes = Vec::with_capacity(declared);
        file.by_ref()
            .take(u64::try_from(MAX_CONTEXT_TOTAL_BYTES + 1).map_err(|_| invalid())?)
            .read_to_end(&mut bytes)
            .map_err(|_| invalid())?;
        if bytes.len() != declared {
            return Err(invalid());
        }
        workspace
            .revalidate_existing(&resolved)
            .map_err(|_| invalid())?;
        let content = String::from_utf8(bytes).map_err(|_| invalid())?;
        let id = format!(
            "workspace.explicit_context.{}",
            instructions.len() + loaded.len()
        );
        let locator = if logical_workspace == "<workspace>" {
            resolved.display_path().to_owned()
        } else {
            format!(
                "{}/{}",
                logical_workspace
                    .strip_prefix("<workspace>/")
                    .ok_or_else(invalid)?,
                resolved.display_path()
            )
        };
        loaded.push(
            WorkspaceInstruction::new(
                PromptSegmentId::from_str(&id).map_err(|_| invalid())?,
                content,
                locator,
                TrustLevel::Delegated,
            )
            .map_err(|_| invalid())?,
        );
    }
    instructions.extend(loaded);
    Ok(())
}

fn logical_workspace_locator(relative: &Path) -> Result<String, CodingError> {
    let relative = relative
        .to_str()
        .ok_or_else(invalid)?
        .replace(std::path::MAIN_SEPARATOR, "/");
    if relative.is_empty() {
        Ok("<workspace>".to_owned())
    } else {
        Ok(format!("<workspace>/{relative}"))
    }
}

fn invalid() -> CodingError {
    CodingError::new(
        CodingErrorCode::InvalidInput,
        "workspace context is invalid",
    )
}

fn not_found() -> CodingError {
    CodingError::new(
        CodingErrorCode::NotFound,
        "workspace context boundary is missing",
    )
}
