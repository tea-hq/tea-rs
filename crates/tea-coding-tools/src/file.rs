use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{
    FileToolError, FileToolErrorCode, ResolvedExistingPath, ResolvedMutationPath, WorkspaceRoot,
};

/// Maximum source bytes accepted by the read tool.
pub const MAX_READ_BYTES: usize = 32 * 1024;
/// Maximum bytes accepted or produced by a write/edit tool.
pub const MAX_WRITE_BYTES: usize = 192 * 1024;
/// Default maximum lines returned by one read invocation.
pub const DEFAULT_READ_LINE_LIMIT: usize = 2_000;
/// Maximum line limit accepted by one read invocation.
pub const MAX_READ_LINE_LIMIT: usize = 10_000;

static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

/// Reads a resolved file as bounded UTF-8 text through its workspace capability.
///
/// The target is revalidated before opening and the opened file identity is
/// checked before its bytes are read. This is intentionally not a raw host
/// path reader.
///
/// # Errors
///
/// Returns a bounded file-tool error when the target changes, is not a regular
/// file, exceeds the byte limit, or is not valid UTF-8 text.
pub fn read_bounded_utf8(
    workspace: &WorkspaceRoot,
    target: &ResolvedExistingPath,
    max_bytes: usize,
) -> Result<String, FileToolError> {
    workspace.revalidate_existing(target)?;
    let metadata = fs::metadata(target.host_path())
        .map_err(|_| FileToolError::new(FileToolErrorCode::FilesystemFailure))?;
    if !metadata.is_file() {
        return Err(FileToolError::new(FileToolErrorCode::NotAFile));
    }
    if metadata.len() > max_bytes as u64 {
        return Err(FileToolError::new(FileToolErrorCode::TooLarge));
    }

    let mut file = File::open(target.host_path())
        .map_err(|_| FileToolError::new(FileToolErrorCode::FilesystemFailure))?;
    let opened_metadata = file
        .metadata()
        .map_err(|_| FileToolError::new(FileToolErrorCode::FilesystemFailure))?;
    workspace.verify_opened_existing(target, &opened_metadata)?;
    let capacity = usize::try_from(metadata.len())
        .unwrap_or(max_bytes)
        .min(max_bytes);
    let mut bytes = Vec::with_capacity(capacity);
    Read::by_ref(&mut file)
        .take((max_bytes + 1) as u64)
        .read_to_end(&mut bytes)
        .map_err(|_| FileToolError::new(FileToolErrorCode::FilesystemFailure))?;
    if bytes.len() > max_bytes {
        return Err(FileToolError::new(FileToolErrorCode::TooLarge));
    }
    if bytes.contains(&0) {
        return Err(FileToolError::new(FileToolErrorCode::BinaryFile));
    }
    String::from_utf8(bytes).map_err(|_| FileToolError::new(FileToolErrorCode::InvalidUtf8))
}

pub(crate) fn atomic_write(
    workspace: &WorkspaceRoot,
    target: &ResolvedMutationPath,
    content: &[u8],
) -> Result<(), FileToolError> {
    atomic_write_inner(workspace, target, content, None)
}

pub(crate) fn atomic_write_if_unchanged(
    workspace: &WorkspaceRoot,
    existing: &ResolvedExistingPath,
    target: &ResolvedMutationPath,
    expected: &str,
    content: &[u8],
) -> Result<(), FileToolError> {
    atomic_write_inner(workspace, target, content, Some((existing, expected)))
}

fn atomic_write_inner(
    workspace: &WorkspaceRoot,
    target: &ResolvedMutationPath,
    content: &[u8],
    expected: Option<(&ResolvedExistingPath, &str)>,
) -> Result<(), FileToolError> {
    if content.len() > MAX_WRITE_BYTES {
        return Err(FileToolError::new(FileToolErrorCode::TooLarge));
    }
    let parent = target
        .host_path()
        .parent()
        .ok_or_else(|| FileToolError::new(FileToolErrorCode::InvalidPath))?;
    if !parent.is_dir() {
        return Err(FileToolError::new(FileToolErrorCode::NotFound));
    }
    if target.target_existed_at_resolution() {
        let metadata = fs::metadata(target.host_path())
            .map_err(|_| FileToolError::new(FileToolErrorCode::FilesystemFailure))?;
        if !metadata.is_file() {
            return Err(FileToolError::new(FileToolErrorCode::NotAFile));
        }
    }
    let temporary = create_temporary(parent)?;
    let result = write_and_commit(workspace, target, &temporary, content, expected);
    if result.is_err() {
        let _ = fs::remove_file(&temporary.path);
    }
    result
}

fn create_temporary(parent: &Path) -> Result<TemporaryFile, FileToolError> {
    for _ in 0..32 {
        let id = NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed);
        let path = parent.join(format!(".tea-tmp-{}-{id}", std::process::id()));
        match open_new_temporary(&path) {
            Ok(file) => return Ok(TemporaryFile { path, file }),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(_) => return Err(FileToolError::new(FileToolErrorCode::FilesystemFailure)),
        }
    }
    Err(FileToolError::new(FileToolErrorCode::FilesystemFailure))
}

fn open_new_temporary(path: &Path) -> std::io::Result<File> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)
}

fn write_and_commit(
    workspace: &WorkspaceRoot,
    target: &ResolvedMutationPath,
    temporary: &TemporaryFile,
    content: &[u8],
    expected: Option<(&ResolvedExistingPath, &str)>,
) -> Result<(), FileToolError> {
    if target.target_existed_at_resolution() {
        let permissions = fs::metadata(target.host_path())
            .map_err(|_| FileToolError::new(FileToolErrorCode::FilesystemFailure))?
            .permissions();
        temporary
            .file
            .set_permissions(permissions)
            .map_err(|_| FileToolError::new(FileToolErrorCode::FilesystemFailure))?;
    }
    (&temporary.file)
        .write_all(content)
        .map_err(|_| FileToolError::new(FileToolErrorCode::FilesystemFailure))?;
    temporary
        .file
        .sync_all()
        .map_err(|_| FileToolError::new(FileToolErrorCode::FilesystemFailure))?;
    if let Some((existing, expected)) = expected {
        let current = read_bounded_utf8(workspace, existing, MAX_WRITE_BYTES)?;
        if current != expected {
            return Err(FileToolError::new(FileToolErrorCode::PathChanged));
        }
    }
    workspace.revalidate_mutation(target)?;
    fs::rename(&temporary.path, target.host_path())
        .map_err(|_| FileToolError::new(FileToolErrorCode::AtomicCommitFailed))?;
    sync_parent(target.host_path().parent());
    Ok(())
}

fn sync_parent(parent: Option<&Path>) {
    #[cfg(unix)]
    if let Some(parent) = parent
        && let Ok(directory) = File::open(parent)
    {
        let _ = directory.sync_all();
    }
    #[cfg(not(unix))]
    let _ = parent;
}

struct TemporaryFile {
    path: PathBuf,
    file: File,
}
