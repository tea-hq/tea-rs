//! Trusted bounded declarative context, skill, and prompt-template discovery.

mod context_files;
mod frontmatter;
mod prompts;
mod skills;
mod system_prompts;

use std::path::{Path, PathBuf};

use tea_context::{SkillCommand, SkillMetadata, WorkspaceInstruction};

pub use prompts::PromptTemplate;
pub use skills::{DiscoveredSkill, LoadedSkill, SkillResourceContent, SkillRoot, SkillSource};
pub use system_prompts::{CodingPromptResource, CodingPromptResourceRoots};

use crate::{CodingError, ProjectAccess};

/// Resolves the bounded project root used for inherited instruction discovery.
///
/// Ordinary repositories use their worktree root. A linked worktree nested
/// inside its main repository uses the main repository root so inherited
/// instructions retain deterministic shadowing. Other linked worktrees and
/// submodules remain scoped to their own worktree. Non-Git directories use the
/// supplied workspace itself.
///
/// # Errors
///
/// Returns an error when the workspace is absent or is not a directory.
pub fn project_instruction_boundary(workspace: &Path) -> Result<PathBuf, CodingError> {
    context_files::project_instruction_boundary(workspace)
}

/// Returns whether any supported project instruction candidate exists between
/// the resolved boundary and workspace, inclusive.
#[must_use]
pub fn has_project_instructions(boundary: &Path, workspace: &Path) -> bool {
    context_files::has_project_instructions(boundary, workspace)
}

/// Bounded safe diagnostic produced while optional resources are skipped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceDiagnostic {
    code: &'static str,
    subject: String,
    source: Option<SkillSource>,
}

impl ResourceDiagnostic {
    pub(crate) fn new(code: &'static str, subject: &str) -> Self {
        Self {
            code,
            subject: bounded_subject(subject),
            source: None,
        }
    }

    pub(crate) fn skill(code: &'static str, source: SkillSource, subject: &str) -> Self {
        Self {
            code,
            subject: bounded_subject(subject),
            source: Some(source),
        }
    }
    /// Returns the machine-readable diagnostic code.
    #[must_use]
    pub const fn code(&self) -> &'static str {
        self.code
    }
    /// Returns a bounded workspace-relative subject.
    #[must_use]
    pub fn subject(&self) -> &str {
        &self.subject
    }

    /// Returns the skill source associated with this diagnostic, when present.
    #[must_use]
    pub const fn source(&self) -> Option<SkillSource> {
        self.source
    }
}

const MAX_RESOURCE_DIAGNOSTICS: usize = 256;
const MAX_RESOURCE_DIAGNOSTIC_SUBJECT_BYTES: usize = 512;

fn bounded_subject(subject: &str) -> String {
    let mut bounded = String::new();
    for character in subject.chars().filter(|character| !character.is_control()) {
        if bounded.len() + character.len_utf8() > MAX_RESOURCE_DIAGNOSTIC_SUBJECT_BYTES {
            break;
        }
        bounded.push(character);
    }
    bounded
}

fn append_diagnostics(
    target: &mut Vec<ResourceDiagnostic>,
    incoming: impl IntoIterator<Item = ResourceDiagnostic>,
) {
    let mut truncated = false;
    for diagnostic in incoming {
        if target.len() < MAX_RESOURCE_DIAGNOSTICS - 1 {
            target.push(diagnostic);
        } else {
            truncated = true;
        }
    }
    if truncated
        && !target
            .iter()
            .any(|diagnostic| diagnostic.code() == "resource_diagnostics_truncated")
    {
        target.push(ResourceDiagnostic::new(
            "resource_diagnostics_truncated",
            "catalog",
        ));
    }
}

/// Immutable deterministic catalog discovered for one workspace.
#[derive(Debug, Clone)]
pub struct ResourceCatalog {
    context: Vec<WorkspaceInstruction>,
    logical_workspace: String,
    skills: Vec<DiscoveredSkill>,
    prompts: Vec<PromptTemplate>,
    system_prompt: Option<Box<CodingPromptResource>>,
    append_system_prompt: Option<Box<CodingPromptResource>>,
    diagnostics: Vec<ResourceDiagnostic>,
}

impl ResourceCatalog {
    /// Discovers global resources and project resources only when trusted.
    ///
    /// # Errors
    ///
    /// Rejects malformed, duplicate, oversized, or escaping resources.
    #[allow(clippy::too_many_arguments)]
    pub fn discover(
        boundary: &Path,
        workspace: &Path,
        access: ProjectAccess,
        global_skill_roots: &[PathBuf],
        project_skill_roots: &[PathBuf],
        global_prompt_root: Option<&Path>,
        project_prompt_root: Option<&Path>,
    ) -> Result<Self, CodingError> {
        let mut skill_roots =
            Vec::with_capacity(global_skill_roots.len() + project_skill_roots.len());
        for path in global_skill_roots {
            skill_roots.push(SkillRoot::new(path, SkillSource::UserTea)?);
        }
        for path in project_skill_roots {
            skill_roots.push(SkillRoot::new(path, SkillSource::ProjectTea)?);
        }
        Self::discover_with_skill_roots(
            boundary,
            workspace,
            access,
            &skill_roots,
            global_prompt_root,
            project_prompt_root,
        )
    }

    /// Discovers resources from typed, ordered skill roots.
    ///
    /// Project roots are filtered before canonicalization or filesystem
    /// traversal unless the workspace is trusted. Roots are then merged by
    /// [`SkillSource`] precedence.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid explicit roots, trusted boundary failures,
    /// or invalid configured prompt resources.
    #[allow(clippy::too_many_arguments)]
    pub fn discover_with_skill_roots(
        boundary: &Path,
        workspace: &Path,
        access: ProjectAccess,
        skill_roots: &[SkillRoot],
        global_prompt_root: Option<&Path>,
        project_prompt_root: Option<&Path>,
    ) -> Result<Self, CodingError> {
        Self::discover_complete(
            boundary,
            workspace,
            access,
            skill_roots,
            global_prompt_root,
            project_prompt_root,
            None,
        )
    }

    /// Discovers the complete coding resource set from explicitly injected roots.
    ///
    /// Project-owned roots are ignored unless the project is trusted. Existing
    /// discovery methods are convenience wrappers without coding prompt
    /// customization roots.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid explicit roots, trusted boundary failures,
    /// or malformed selected resources.
    #[allow(clippy::too_many_arguments)]
    pub fn discover_complete(
        boundary: &Path,
        workspace: &Path,
        access: ProjectAccess,
        skill_roots: &[SkillRoot],
        global_prompt_root: Option<&Path>,
        project_prompt_root: Option<&Path>,
        coding_prompt_roots: Option<&CodingPromptResourceRoots>,
    ) -> Result<Self, CodingError> {
        let global_context_root = coding_prompt_roots.map(CodingPromptResourceRoots::global_root);
        let (context, mut diagnostics, logical_workspace) =
            context_files::discover(boundary, workspace, access, global_context_root)?;
        let allowed_roots = skill_roots
            .iter()
            .filter(|root| {
                access == ProjectAccess::Trusted || !root.source().requires_project_trust()
            })
            .cloned()
            .collect::<Vec<_>>();
        let (skills, skill_diagnostics) = skills::discover(&allowed_roots)?;
        append_diagnostics(&mut diagnostics, skill_diagnostics);
        let prompts = prompts::discover(
            global_prompt_root,
            (access == ProjectAccess::Trusted)
                .then_some(project_prompt_root)
                .flatten(),
        )?;
        let system_prompts = system_prompts::discover(coding_prompt_roots, access)?;
        Ok(Self {
            context,
            logical_workspace,
            skills,
            prompts,
            system_prompt: system_prompts.system.map(Box::new),
            append_system_prompt: system_prompts.append.map(Box::new),
            diagnostics,
        })
    }

    /// Adds explicit workspace-relative context files through the workspace capability.
    ///
    /// # Errors
    ///
    /// Rejects duplicate, escaping, non-UTF-8, oversized, changed, or unreadable files.
    pub fn add_explicit_context_files(
        &mut self,
        workspace: &tea_coding_tools::WorkspaceRoot,
        paths: &[String],
    ) -> Result<(), CodingError> {
        context_files::add_explicit(workspace, &self.logical_workspace, paths, &mut self.context)
    }

    /// Applies resolved resource feature switches without re-reading any source.
    pub fn apply_settings(&mut self, context_files: bool, prompt_templates: bool) {
        if !context_files {
            self.context.clear();
        }
        if !prompt_templates {
            self.prompts.clear();
        }
    }

    /// Returns deterministic context instructions.
    #[must_use]
    pub fn context(&self) -> &[WorkspaceInstruction] {
        &self.context
    }
    /// Returns the privacy-safe logical working directory.
    #[must_use]
    pub fn logical_workspace(&self) -> &str {
        &self.logical_workspace
    }
    /// Returns discovered skills sorted by ID.
    #[must_use]
    pub fn skills(&self) -> &[DiscoveredSkill] {
        &self.skills
    }
    /// Returns prompt templates sorted by name.
    #[must_use]
    pub fn prompts(&self) -> &[PromptTemplate] {
        &self.prompts
    }
    /// Returns the selected coding behavior replacement, when configured.
    #[must_use]
    pub fn system_prompt(&self) -> Option<&CodingPromptResource> {
        self.system_prompt.as_deref()
    }
    /// Returns the selected appended coding prompt resource, when configured.
    #[must_use]
    pub fn append_system_prompt(&self) -> Option<&CodingPromptResource> {
        self.append_system_prompt.as_deref()
    }
    /// Returns safe discovery diagnostics.
    #[must_use]
    pub fn diagnostics(&self) -> &[ResourceDiagnostic] {
        &self.diagnostics
    }
    /// Projects skill metadata without loading skill bodies.
    #[must_use]
    pub fn skill_metadata(&self) -> Vec<SkillMetadata> {
        self.skills
            .iter()
            .filter(|skill| skill.model_invocable())
            .map(|skill| skill.metadata().clone())
            .collect()
    }

    /// Loads the winning skill for a typed explicit command.
    ///
    /// # Errors
    ///
    /// Rejects unknown skills, changed manifests, and unsafe content.
    pub fn load_skill(&self, command: &SkillCommand) -> Result<LoadedSkill, CodingError> {
        let skill = self
            .skills
            .iter()
            .find(|skill| skill.metadata().id() == command.skill_id())
            .ok_or_else(|| {
                crate::CodingError::new(crate::CodingErrorCode::NotFound, "skill is not registered")
            })?;
        skill.load(command)
    }
    /// Loads an explicitly invoked skill.
    ///
    /// # Errors
    ///
    /// Rejects unknown or invalid invocation syntax.
    pub fn invoke_skill(&self, invocation: &str) -> Result<LoadedSkill, CodingError> {
        let command = invocation.parse::<SkillCommand>().map_err(|_| {
            crate::CodingError::new(
                crate::CodingErrorCode::InvalidInput,
                "skill invocation is invalid",
            )
        })?;
        self.load_skill(&command)
    }

    /// Reads bounded text from the selected winning skill's resource tree.
    ///
    /// # Errors
    ///
    /// Rejects unknown skills, invalid relative paths, directories, binary or
    /// oversized files, changed targets, and resource paths outside the skill.
    pub fn read_skill_resource(
        &self,
        skill_id: &tea_context::SkillId,
        path: &str,
        offset: Option<usize>,
        limit: Option<usize>,
    ) -> Result<SkillResourceContent, CodingError> {
        let skill = self
            .skills
            .iter()
            .find(|skill| skill.metadata().id() == skill_id)
            .ok_or_else(|| {
                crate::CodingError::new(crate::CodingErrorCode::NotFound, "skill is not registered")
            })?;
        skill.read_resource(skill_id, path, offset, limit)
    }
}
