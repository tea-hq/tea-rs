use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use ignore::{DirEntry, WalkBuilder};
use serde::Deserialize;
use tea_coding_tools::{
    DEFAULT_READ_LINE_LIMIT, FileToolError, FileToolErrorCode, MAX_READ_BYTES, MAX_READ_LINE_LIMIT,
    WorkspacePathError, WorkspacePathErrorCode, WorkspaceRoot, read_bounded_utf8,
};
use tea_context::{SkillCommand, SkillId, SkillMetadata};

use crate::resources::ResourceDiagnostic;
use crate::resources::frontmatter::parse;
use crate::{CodingError, CodingErrorCode};

const MAX_SKILLS: usize = 128;
const MAX_SKILL_CONTENT_BYTES: usize = 128 * 1024;
const MAX_SKILL_SCAN_DEPTH: usize = 16;
const MAX_SKILL_SCAN_ENTRIES: usize = 8_192;
const MAX_SKILL_CANDIDATES_PER_ROOT: usize = 512;

/// Origin of a discovered skill root.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SkillSource {
    /// A path explicitly configured by the caller.
    Explicit,
    /// The workspace `.tea/skills` directory.
    ProjectTea,
    /// The workspace `.agents/skills` directory.
    ProjectAgents,
    /// The user Tea configuration `skills` directory.
    UserTea,
    /// The user `.agents/skills` directory.
    UserAgents,
    /// The Tea data `skills` directory.
    TeaData,
}

impl SkillSource {
    /// Returns the stable diagnostic and display label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Explicit => "explicit",
            Self::ProjectTea => "project-tea",
            Self::ProjectAgents => "project-agents",
            Self::UserTea => "user-tea",
            Self::UserAgents => "user-agents",
            Self::TeaData => "tea-data",
        }
    }

    pub(crate) const fn priority(self) -> u8 {
        match self {
            Self::Explicit => 0,
            Self::ProjectTea => 1,
            Self::ProjectAgents => 2,
            Self::UserTea => 3,
            Self::UserAgents => 4,
            Self::TeaData => 5,
        }
    }

    pub(crate) const fn requires_project_trust(self) -> bool {
        matches!(self, Self::ProjectTea | Self::ProjectAgents)
    }
}

/// Absolute, syntactically validated root used for skill discovery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillRoot {
    path: PathBuf,
    source: SkillSource,
}

/// Bounded text returned from a skill resource read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkillResourceContent {
    skill_id: SkillId,
    path: String,
    content: String,
    start_line: usize,
    end_line: usize,
    total_lines: usize,
    truncated: bool,
}

impl SkillResourceContent {
    /// Returns the winning skill identity used for the read.
    #[must_use]
    pub const fn skill_id(&self) -> &SkillId {
        &self.skill_id
    }

    /// Returns the capability-relative resource path.
    #[must_use]
    pub fn path(&self) -> &str {
        &self.path
    }

    /// Returns the exact bounded source text for the selected lines.
    #[must_use]
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Returns the requested one-based start line.
    #[must_use]
    pub const fn start_line(&self) -> usize {
        self.start_line
    }

    /// Returns the one-based inclusive end line, or zero for an empty result.
    #[must_use]
    pub const fn end_line(&self) -> usize {
        self.end_line
    }

    /// Returns the total number of source lines.
    #[must_use]
    pub const fn total_lines(&self) -> usize {
        self.total_lines
    }

    /// Returns whether the result omits source lines.
    #[must_use]
    pub const fn truncated(&self) -> bool {
        self.truncated
    }
}

impl SkillRoot {
    /// Creates a root without touching the filesystem.
    ///
    /// # Errors
    ///
    /// Returns an error for relative paths, paths without a final component,
    /// or paths containing control characters.
    pub fn new(path: impl Into<PathBuf>, source: SkillSource) -> Result<Self, CodingError> {
        let path = path.into();
        let valid_text = path
            .to_str()
            .is_some_and(|value| !value.chars().any(char::is_control));
        if !path.is_absolute() || path.file_name().is_none() || !valid_text {
            return Err(invalid());
        }
        Ok(Self { path, source })
    }

    /// Returns the configured, not-yet-canonicalized root path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the root's provenance.
    #[must_use]
    pub const fn source(&self) -> SkillSource {
        self.source
    }
}

#[derive(Debug, Deserialize)]
struct SkillFrontmatter {
    name: String,
    description: String,
    #[serde(default, rename = "disable-model-invocation")]
    disable_model_invocation: bool,
    #[serde(flatten)]
    _extra: BTreeMap<String, serde_yaml::Value>,
}

/// One validated declarative skill whose body is loaded only on invocation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DiscoveredSkill {
    metadata: SkillMetadata,
    manifest_path: PathBuf,
    source: SkillSource,
    model_invocable: bool,
    resource_root: WorkspaceRoot,
}

impl DiscoveredSkill {
    /// Returns prompt-safe skill metadata.
    #[must_use]
    pub const fn metadata(&self) -> &SkillMetadata {
        &self.metadata
    }

    /// Returns the source that supplied this winning skill.
    #[must_use]
    pub const fn source(&self) -> SkillSource {
        self.source
    }

    /// Returns whether this skill may be advertised to the model.
    #[must_use]
    pub const fn model_invocable(&self) -> bool {
        self.model_invocable
    }

    /// Returns the canonical manifest path.
    #[must_use]
    pub fn manifest_path(&self) -> &Path {
        &self.manifest_path
    }

    /// Loads the body after a typed explicit command.
    ///
    /// # Errors
    ///
    /// Rejects changed metadata, oversized content, and unsafe references.
    pub fn load(&self, command: &SkillCommand) -> Result<LoadedSkill, CodingError> {
        if command.skill_id() != self.metadata.id() {
            return Err(invalid());
        }
        let source = self.read_manifest()?;
        let document = parse::<SkillFrontmatter>(&source)?;
        if document.metadata.name != self.metadata.id().as_str()
            || document.metadata.description != self.metadata.description()
            || document.metadata.disable_model_invocation == self.model_invocable
        {
            return Err(invalid());
        }
        Ok(LoadedSkill {
            content: document.body,
            arguments: command.arguments().to_owned(),
            resource_root: self.resource_root.clone(),
        })
    }

    /// Loads the body using the legacy string command wrapper.
    ///
    /// New callers should parse a [`SkillCommand`] and use [`Self::load`].
    ///
    /// # Errors
    ///
    /// Returns an error when the command syntax or manifest is invalid.
    pub fn invoke(&self, invocation: &str) -> Result<LoadedSkill, CodingError> {
        let command = invocation.parse::<SkillCommand>().map_err(|_| invalid())?;
        self.load(&command)
    }

    fn read_manifest(&self) -> Result<String, CodingError> {
        let relative = self
            .manifest_path
            .strip_prefix(self.resource_root.host_path())
            .map_err(|_| invalid())?
            .to_str()
            .ok_or_else(invalid)?
            .replace(std::path::MAIN_SEPARATOR, "/");
        let resolved = self
            .resource_root
            .resolve_existing(&relative)
            .map_err(|_| invalid())?;
        let metadata = fs::metadata(resolved.host_path()).map_err(|_| not_found())?;
        if !metadata.is_file() || metadata.len() > MAX_SKILL_CONTENT_BYTES as u64 {
            return Err(invalid());
        }
        let mut file = File::open(resolved.host_path()).map_err(|_| not_found())?;
        let opened = file.metadata().map_err(|_| not_found())?;
        self.resource_root
            .verify_opened_existing(&resolved, &opened)
            .map_err(|_| invalid())?;
        let mut bytes = Vec::new();
        file.by_ref()
            .take((MAX_SKILL_CONTENT_BYTES + 1) as u64)
            .read_to_end(&mut bytes)
            .map_err(|_| not_found())?;
        self.resource_root
            .revalidate_existing(&resolved)
            .map_err(|_| invalid())?;
        if bytes.len() > MAX_SKILL_CONTENT_BYTES {
            return Err(invalid());
        }
        String::from_utf8(bytes).map_err(|_| invalid())
    }
}

/// Explicitly loaded skill body and its safe reference base.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedSkill {
    content: String,
    arguments: String,
    resource_root: WorkspaceRoot,
}

impl LoadedSkill {
    /// Returns skill instructions.
    #[must_use]
    pub fn content(&self) -> &str {
        &self.content
    }

    /// Returns the prompt body with explicit arguments appended when present.
    ///
    /// The body is preserved exactly as loaded. This is the sole owner of the
    /// argument section used when a skill becomes a user prompt.
    #[must_use]
    pub fn prompt_fragment(&self) -> String {
        if self.arguments.is_empty() {
            self.content.clone()
        } else {
            format!("{}\n\nArguments: {}", self.content, self.arguments)
        }
    }

    /// Returns uninterpreted invocation arguments.
    #[must_use]
    pub fn arguments(&self) -> &str {
        &self.arguments
    }

    /// Resolves one relative existing reference beneath the skill directory.
    ///
    /// # Errors
    ///
    /// Rejects absolute/traversing/missing references and symlink escape.
    pub fn resolve_reference(&self, relative: &str) -> Result<PathBuf, CodingError> {
        self.resource_root
            .resolve_existing(relative)
            .map(|target| target.host_path().to_path_buf())
            .map_err(|error| map_workspace_error(&error))
    }
}

impl DiscoveredSkill {
    pub(crate) fn read_resource(
        &self,
        skill_id: &SkillId,
        path: &str,
        offset: Option<usize>,
        limit: Option<usize>,
    ) -> Result<SkillResourceContent, CodingError> {
        let offset = offset.unwrap_or(1);
        let limit = limit.unwrap_or(DEFAULT_READ_LINE_LIMIT);
        if offset == 0 || limit == 0 || limit > MAX_READ_LINE_LIMIT {
            return Err(invalid());
        }
        let target = self
            .resource_root
            .resolve_existing(path)
            .map_err(|error| map_workspace_error(&error))?;
        let source = read_bounded_utf8(&self.resource_root, &target, MAX_READ_BYTES)
            .map_err(map_file_error)?;
        let lines = source.split_inclusive('\n').collect::<Vec<_>>();
        let start_index = offset.saturating_sub(1).min(lines.len());
        let end_index = start_index.saturating_add(limit).min(lines.len());
        let content = lines[start_index..end_index].concat();
        let end_line = if end_index == start_index {
            0
        } else {
            end_index
        };
        Ok(SkillResourceContent {
            skill_id: skill_id.clone(),
            path: target.display_path().to_owned(),
            content,
            start_line: offset,
            end_line,
            total_lines: lines.len(),
            truncated: start_index > 0 || end_index < lines.len(),
        })
    }
}

#[derive(Debug, Clone)]
struct SkillCandidate {
    manifest_path: PathBuf,
    relative_path: String,
    source: SkillSource,
}

#[derive(Debug, Clone, Copy)]
struct ScanLimits {
    depth: usize,
    entries: usize,
    candidates: usize,
}

impl Default for ScanLimits {
    fn default() -> Self {
        Self {
            depth: MAX_SKILL_SCAN_DEPTH,
            entries: MAX_SKILL_SCAN_ENTRIES,
            candidates: MAX_SKILL_CANDIDATES_PER_ROOT,
        }
    }
}

#[derive(Debug, Default)]
struct RootScan {
    candidates: Vec<SkillCandidate>,
    diagnostics: Vec<ResourceDiagnostic>,
}

pub(crate) fn discover(
    roots: &[SkillRoot],
) -> Result<(Vec<DiscoveredSkill>, Vec<ResourceDiagnostic>), CodingError> {
    let mut ordered = roots.iter().enumerate().collect::<Vec<_>>();
    ordered.sort_by_key(|(index, root)| (root.source().priority(), *index));

    let mut winners = BTreeMap::<SkillId, DiscoveredSkill>::new();
    let mut seen_manifests = BTreeSet::new();
    let mut diagnostics = Vec::new();
    let mut catalog_limit_reported = false;

    for (_, root) in ordered {
        let scan = scan_root(root)?;
        diagnostics.extend(scan.diagnostics);
        for candidate in scan.candidates {
            if !seen_manifests.insert(candidate.manifest_path.clone()) {
                continue;
            }
            if winners.len() >= MAX_SKILLS {
                if !catalog_limit_reported {
                    diagnostics.push(ResourceDiagnostic::skill(
                        "skill_catalog_limit",
                        candidate.source,
                        "catalog",
                    ));
                    catalog_limit_reported = true;
                }
                continue;
            }
            let Ok(skill) = parse_candidate(&candidate) else {
                diagnostics.push(ResourceDiagnostic::skill(
                    "skill_invalid",
                    candidate.source,
                    &candidate.relative_path,
                ));
                continue;
            };
            if let Some(winner) = winners.get(skill.metadata.id()) {
                let subject = format!(
                    "{} <- {}",
                    candidate.source.label(),
                    winner.source().label()
                );
                diagnostics.push(ResourceDiagnostic::skill(
                    "skill_shadowed",
                    candidate.source,
                    &subject,
                ));
                continue;
            }
            winners.insert(skill.metadata.id().clone(), skill);
        }
    }
    Ok((winners.into_values().collect(), diagnostics))
}

fn scan_root(root: &SkillRoot) -> Result<RootScan, CodingError> {
    scan_root_with_limits(root, ScanLimits::default())
}

#[allow(clippy::too_many_lines)]
fn scan_root_with_limits(root: &SkillRoot, limits: ScanLimits) -> Result<RootScan, CodingError> {
    let canonical_root = match fs::canonicalize(root.path()) {
        Ok(path) => path,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            if root.source() == SkillSource::Explicit {
                return Err(not_found());
            }
            return Ok(RootScan::default());
        }
        Err(_) if root.source() == SkillSource::Explicit => return Err(not_found()),
        Err(_) => {
            return Ok(RootScan {
                candidates: Vec::new(),
                diagnostics: vec![ResourceDiagnostic::skill(
                    "skill_root_invalid",
                    root.source(),
                    root.source().label(),
                )],
            });
        }
    };

    let metadata = match fs::metadata(&canonical_root) {
        Ok(metadata) => metadata,
        Err(_) if root.source() == SkillSource::Explicit => return Err(not_found()),
        Err(_) => {
            return Ok(RootScan {
                candidates: Vec::new(),
                diagnostics: vec![ResourceDiagnostic::skill(
                    "skill_root_invalid",
                    root.source(),
                    root.source().label(),
                )],
            });
        }
    };
    if metadata.is_file() {
        if root.source() != SkillSource::Explicit
            || canonical_root
                .extension()
                .is_none_or(|extension| extension != "md")
        {
            return if root.source() == SkillSource::Explicit {
                Err(invalid())
            } else {
                Ok(RootScan {
                    candidates: Vec::new(),
                    diagnostics: vec![ResourceDiagnostic::skill(
                        "skill_root_invalid",
                        root.source(),
                        root.source().label(),
                    )],
                })
            };
        }
        return Ok(RootScan {
            candidates: vec![SkillCandidate {
                relative_path: canonical_root
                    .file_name()
                    .and_then(|name| name.to_str())
                    .ok_or_else(invalid)?
                    .to_owned(),
                manifest_path: canonical_root,
                source: root.source(),
            }],
            diagnostics: Vec::new(),
        });
    }
    if !metadata.is_dir() {
        return Err(invalid());
    }

    let mut builder = WalkBuilder::new(&canonical_root);
    builder
        .hidden(true)
        .parents(false)
        .require_git(false)
        .git_ignore(true)
        .git_exclude(true)
        .git_global(false)
        .ignore(true)
        .follow_links(false)
        .same_file_system(true)
        .max_depth(Some(limits.depth + 1))
        .add_custom_ignore_filename(".fdignore")
        .filter_entry(|entry| !is_node_modules(entry));

    let mut raw_candidates = Vec::new();
    let mut entries = 0_usize;
    let mut limit = false;
    let mut traversal_failed = false;
    for result in builder.build() {
        entries = entries.saturating_add(1);
        if entries > limits.entries {
            limit = true;
            break;
        }
        let Ok(entry) = result else {
            traversal_failed = true;
            continue;
        };
        if entry.depth() > limits.depth {
            limit = true;
            break;
        }
        if entry.file_name() == "SKILL.md"
            && entry
                .file_type()
                .is_some_and(|file_type| file_type.is_file() || file_type.is_symlink())
        {
            if raw_candidates.len() >= limits.candidates {
                limit = true;
                break;
            }
            raw_candidates.push(entry.path().to_path_buf());
        }
    }
    if limit || traversal_failed {
        let code = if limit {
            "skill_scan_limit"
        } else {
            "skill_scan_failed"
        };
        return Ok(RootScan {
            candidates: Vec::new(),
            diagnostics: vec![ResourceDiagnostic::skill(
                code,
                root.source(),
                root.source().label(),
            )],
        });
    }

    let mut candidates = Vec::new();
    let mut diagnostics = Vec::new();
    for raw_path in raw_candidates {
        let Some(path) = raw_path
            .strip_prefix(&canonical_root)
            .ok()
            .and_then(|path| path.to_str())
        else {
            diagnostics.push(ResourceDiagnostic::skill(
                "skill_invalid",
                root.source(),
                "manifest",
            ));
            continue;
        };
        let relative_path = path.replace(std::path::MAIN_SEPARATOR, "/");
        match fs::canonicalize(&raw_path) {
            Ok(manifest_path)
                if manifest_path.starts_with(&canonical_root) && manifest_path.is_file() =>
            {
                candidates.push(SkillCandidate {
                    manifest_path,
                    relative_path,
                    source: root.source(),
                });
            }
            Ok(_) | Err(_) => diagnostics.push(ResourceDiagnostic::skill(
                "skill_invalid",
                root.source(),
                &relative_path,
            )),
        }
    }
    candidates.sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    let manifest_directories = candidates
        .iter()
        .filter_map(|candidate| candidate.manifest_path.parent())
        .map(Path::to_path_buf)
        .collect::<BTreeSet<_>>();
    candidates.retain(|candidate| {
        let Some(mut parent) = candidate.manifest_path.parent() else {
            return true;
        };
        while parent.starts_with(&canonical_root) && parent != canonical_root {
            if parent != candidate.manifest_path.parent().unwrap()
                && manifest_directories.contains(parent)
            {
                return false;
            }
            parent = match parent.parent() {
                Some(parent) => parent,
                None => break,
            };
        }
        true
    });
    Ok(RootScan {
        candidates,
        diagnostics,
    })
}

fn is_node_modules(entry: &DirEntry) -> bool {
    entry.path().components().any(|component| {
        matches!(component, std::path::Component::Normal(name) if name == "node_modules")
    })
}

fn parse_candidate(candidate: &SkillCandidate) -> Result<DiscoveredSkill, CodingError> {
    let source = fs::read_to_string(&candidate.manifest_path).map_err(|_| not_found())?;
    if source.len() > MAX_SKILL_CONTENT_BYTES {
        return Err(invalid());
    }
    let document = parse::<SkillFrontmatter>(&source)?;
    let id = SkillId::from_str(&document.metadata.name).map_err(|_| invalid())?;
    let metadata = SkillMetadata::new(id, document.metadata.description).map_err(|_| invalid())?;
    let resource_root = WorkspaceRoot::new(candidate.manifest_path.parent().ok_or_else(invalid)?)
        .map_err(|_| invalid())?;
    Ok(DiscoveredSkill {
        metadata,
        manifest_path: candidate.manifest_path.clone(),
        source: candidate.source,
        model_invocable: !document.metadata.disable_model_invocation,
        resource_root,
    })
}

fn invalid() -> CodingError {
    CodingError::new(CodingErrorCode::InvalidInput, "skill resource is invalid")
}

fn not_found() -> CodingError {
    CodingError::new(CodingErrorCode::NotFound, "skill resource is missing")
}

fn map_workspace_error(error: &WorkspacePathError) -> CodingError {
    let code = if error.code() == WorkspacePathErrorCode::TargetNotFound {
        CodingErrorCode::NotFound
    } else {
        CodingErrorCode::InvalidInput
    };
    CodingError::new(
        code,
        if code == CodingErrorCode::NotFound {
            "skill resource is missing"
        } else {
            "skill resource path is invalid"
        },
    )
}

fn map_file_error(error: FileToolError) -> CodingError {
    let code = if error.code() == FileToolErrorCode::NotFound {
        CodingErrorCode::NotFound
    } else {
        CodingErrorCode::InvalidInput
    };
    CodingError::new(
        code,
        if code == CodingErrorCode::NotFound {
            "skill resource is missing"
        } else {
            "skill resource could not be read"
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static ID: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn scan_limits_discard_provisional_candidates() {
        let root = std::env::temp_dir().join(format!(
            "coding-skills-scan-limit-{}-{}",
            std::process::id(),
            ID.fetch_add(1, Ordering::Relaxed)
        ));
        let skill = root.join("valid");
        fs::create_dir_all(&skill).unwrap();
        fs::write(
            skill.join("SKILL.md"),
            "---\nname: valid\ndescription: valid\n---\nbody\n",
        )
        .unwrap();
        let typed_root = SkillRoot::new(&root, SkillSource::UserTea).unwrap();
        let scan = scan_root_with_limits(
            &typed_root,
            ScanLimits {
                depth: MAX_SKILL_SCAN_DEPTH,
                entries: 1,
                candidates: MAX_SKILL_CANDIDATES_PER_ROOT,
            },
        )
        .unwrap();
        assert!(scan.candidates.is_empty());
        assert_eq!(scan.diagnostics[0].code(), "skill_scan_limit");
        fs::remove_dir_all(root).unwrap();
    }
}
