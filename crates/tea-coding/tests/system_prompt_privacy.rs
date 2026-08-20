use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use tea_coding::resources::{ResourceCatalog, SkillRoot, SkillSource};
use tea_coding::{CodingSystemPromptBuilder, ProjectAccess};
use tea_coding_tools::ReadTool;
use tea_context::{CompiledPrompt, PromptBudget, PromptCompiler};

static ID: AtomicU64 = AtomicU64::new(0);

struct WorkspaceFixture {
    home: PathBuf,
    boundary: PathBuf,
    workspace: PathBuf,
    skill_root: PathBuf,
}

impl WorkspaceFixture {
    fn new(username: &str) -> Self {
        let home = std::env::temp_dir().join(format!(
            "prompt-home-{username}-{}-{}",
            std::process::id(),
            ID.fetch_add(1, Ordering::Relaxed)
        ));
        let boundary = home.join("projects");
        let workspace = boundary.join("tea/repo");
        let skill_root = home.join("private-config/skills");
        fs::create_dir_all(&workspace).unwrap();
        fs::create_dir_all(skill_root.join("review")).unwrap();
        fs::write(boundary.join("AGENTS.md"), "root rules\n").unwrap();
        fs::write(workspace.join("AGENTS.md"), "workspace rules\n").unwrap();
        fs::write(
            skill_root.join("review/SKILL.md"),
            "---\nname: review\ndescription: Review changes carefully\n---\nReview body\n",
        )
        .unwrap();
        Self {
            home,
            boundary,
            workspace,
            skill_root,
        }
    }

    fn compile(&self) -> (CompiledPrompt, String) {
        let catalog = ResourceCatalog::discover_with_skill_roots(
            &self.boundary,
            &self.workspace,
            ProjectAccess::Trusted,
            &[SkillRoot::new(&self.skill_root, SkillSource::UserTea).unwrap()],
            None,
            None,
        )
        .unwrap();
        let diagnostics = format!("{:?}", catalog.diagnostics());
        let builder = CodingSystemPromptBuilder::new(
            catalog.logical_workspace(),
            catalog.context().to_vec(),
            catalog.skill_metadata(),
        )
        .unwrap();
        let prompt = PromptCompiler
            .compile(
                builder.modules(&[ReadTool::spec().unwrap()]).unwrap(),
                PromptBudget::new(32 * 1024, 32 * 1024).unwrap(),
            )
            .unwrap();
        (prompt, diagnostics)
    }
}

impl Drop for WorkspaceFixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.home);
    }
}

fn assert_private_markers_absent(value: &str, fixture: &WorkspaceFixture, username: &str) {
    assert!(!value.contains(username), "leaked username in {value}");
    for path in [
        &fixture.home,
        &fixture.boundary,
        &fixture.workspace,
        &fixture.skill_root,
    ] {
        assert_path_absent(value, path);
    }
}

fn assert_path_absent(value: &str, path: &Path) {
    let marker = path.to_str().unwrap();
    assert!(
        !value.contains(marker),
        "leaked host path {marker} in {value}"
    );
}

#[test]
fn equivalent_private_workspaces_compile_byte_identical_safe_prompts() {
    let alice = WorkspaceFixture::new("seeded-alice");
    let bob = WorkspaceFixture::new("seeded-bob");
    let (alice_prompt, alice_diagnostics) = alice.compile();
    let (bob_prompt, bob_diagnostics) = bob.compile();

    assert_eq!(alice_prompt, bob_prompt);
    assert_eq!(alice_diagnostics, bob_diagnostics);
    assert!(alice_prompt.text().contains("<workspace>/AGENTS.md"));
    assert!(
        alice_prompt
            .text()
            .contains("The logical working directory is `<workspace>/tea/repo`")
    );
    assert!(alice_prompt.text().contains("Skill `review`"));
    assert!(alice_prompt.text().contains("root rules\n"));
    assert!(alice_prompt.text().contains("workspace rules\n"));

    let inspection = format!("{:?}", alice_prompt.inspection());
    let diagnostics = format!("{:?}", alice_prompt.diagnostics());
    for (fixture, username) in [(&alice, "seeded-alice"), (&bob, "seeded-bob")] {
        assert_private_markers_absent(alice_prompt.text(), fixture, username);
        assert_private_markers_absent(&inspection, fixture, username);
        assert_private_markers_absent(&diagnostics, fixture, username);
        assert_private_markers_absent(&alice_diagnostics, fixture, username);
    }
}
