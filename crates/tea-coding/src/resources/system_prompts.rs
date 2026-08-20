use std::fs;
use std::path::{Path, PathBuf};

use tea_context::{MAX_SEGMENT_BYTES, TrustLevel};

use crate::{CodingError, CodingErrorCode, ProjectAccess};

/// Explicit filesystem roots used for coding prompt customization resources.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodingPromptResourceRoots {
    global_root: PathBuf,
    project_root: Option<PathBuf>,
}

impl CodingPromptResourceRoots {
    /// Creates roots with a required absolute global resource directory.
    ///
    /// The directory may be absent; missing optional resources are normal.
    ///
    /// # Errors
    ///
    /// Rejects a relative or structurally invalid directory path.
    pub fn new(global_root: impl Into<PathBuf>) -> Result<Self, CodingError> {
        Ok(Self {
            global_root: valid_root(global_root.into())?,
            project_root: None,
        })
    }

    /// Adds an absolute trusted-project resource directory.
    ///
    /// The directory may be absent and is ignored for untrusted projects.
    ///
    /// # Errors
    ///
    /// Rejects a relative or structurally invalid directory path.
    pub fn with_project_root(
        mut self,
        project_root: impl Into<PathBuf>,
    ) -> Result<Self, CodingError> {
        self.project_root = Some(valid_root(project_root.into())?);
        Ok(self)
    }

    pub(crate) fn global_root(&self) -> &Path {
        &self.global_root
    }

    fn project_root(&self) -> Option<&Path> {
        self.project_root.as_deref()
    }
}

/// One bounded prompt customization resource with safe provenance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodingPromptResource {
    content: String,
    locator: String,
    trust: TrustLevel,
}

impl CodingPromptResource {
    /// Returns the exact resource body.
    #[must_use]
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Returns a privacy-safe logical source locator.
    #[must_use]
    pub fn locator(&self) -> &str {
        &self.locator
    }

    /// Returns the configured source trust.
    #[must_use]
    pub const fn trust(&self) -> TrustLevel {
        self.trust
    }
}

pub(crate) struct DiscoveredSystemPrompts {
    pub(crate) system: Option<CodingPromptResource>,
    pub(crate) append: Option<CodingPromptResource>,
}

pub(crate) fn discover(
    roots: Option<&CodingPromptResourceRoots>,
    access: ProjectAccess,
) -> Result<DiscoveredSystemPrompts, CodingError> {
    let Some(roots) = roots else {
        return Ok(DiscoveredSystemPrompts {
            system: None,
            append: None,
        });
    };
    Ok(DiscoveredSystemPrompts {
        system: load_winner(roots, access, "SYSTEM.md")?,
        append: load_winner(roots, access, "APPEND_SYSTEM.md")?,
    })
}

fn load_winner(
    roots: &CodingPromptResourceRoots,
    access: ProjectAccess,
    name: &str,
) -> Result<Option<CodingPromptResource>, CodingError> {
    if access == ProjectAccess::Trusted
        && let Some(project_root) = roots.project_root()
        && let Some(resource) = load(project_root, name, &format!("<workspace>/.tea/{name}"))?
    {
        return Ok(Some(resource));
    }
    load(roots.global_root(), name, &format!("<global>/{name}"))
}

fn load(
    root: &Path,
    name: &str,
    locator: &str,
) -> Result<Option<CodingPromptResource>, CodingError> {
    let canonical_root = match fs::canonicalize(root) {
        Ok(root) if root.is_dir() => root,
        Ok(_) => return Err(invalid()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err(not_found()),
    };
    let path = root.join(name);
    let canonical_path = match fs::canonicalize(&path) {
        Ok(path) if path.starts_with(&canonical_root) && path.is_file() => path,
        Ok(_) => return Err(invalid()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(_) => return Err(not_found()),
    };
    let bytes = fs::read(canonical_path).map_err(|_| not_found())?;
    if bytes.len() > MAX_SEGMENT_BYTES {
        return Err(invalid());
    }
    let content = String::from_utf8(bytes).map_err(|_| invalid())?;
    if content.is_empty() {
        return Ok(None);
    }
    Ok(Some(CodingPromptResource {
        content,
        locator: locator.to_owned(),
        trust: TrustLevel::Delegated,
    }))
}

fn valid_root(path: PathBuf) -> Result<PathBuf, CodingError> {
    if !path.is_absolute() || path.file_name().is_none() {
        return Err(invalid());
    }
    Ok(path)
}

fn invalid() -> CodingError {
    CodingError::new(
        CodingErrorCode::InvalidInput,
        "coding prompt resource is invalid",
    )
}

fn not_found() -> CodingError {
    CodingError::new(
        CodingErrorCode::NotFound,
        "coding prompt resource is unavailable",
    )
}
