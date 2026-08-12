//! Trusted bounded declarative context, skill, and prompt-template discovery.

mod context_files;
mod frontmatter;
mod prompts;
mod skills;

use std::path::{Path, PathBuf};

use tea_context::{SkillCommand, SkillMetadata, WorkspaceInstruction};

pub use prompts::PromptTemplate;
pub use skills::{DiscoveredSkill, LoadedSkill, SkillResourceContent, SkillRoot, SkillSource};

use crate::{CodingError, ProjectAccess};

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
    skills: Vec<DiscoveredSkill>,
    prompts: Vec<PromptTemplate>,
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
        let (context, mut diagnostics) = context_files::discover(boundary, workspace, access)?;
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
        Ok(Self {
            context,
            skills,
            prompts,
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
        context_files::add_explicit(workspace, paths, &mut self.context)
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
