use std::fs;
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};

use tea_coding::ProjectAccess;
use tea_coding::resources::{
    CodingPromptResourceRoots, ResourceCatalog, has_project_instructions,
    project_instruction_boundary,
};
use tea_coding_tools::WorkspaceRoot;
use tea_context::{ContextProvider, ContextRequest, WorkspaceInstructionProvider};
use tea_protocol::{ProfileId, ProtocolMetadata, SessionId};

static ID: AtomicU64 = AtomicU64::new(0);

#[test]
fn project_instruction_boundary_uses_the_repository_root_without_crawling_past_it() {
    let root = std::env::temp_dir().join(format!(
        "coding-context-boundary-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let repository = root.join("repository");
    let workspace = repository.join("src/nested");
    fs::create_dir_all(repository.join(".git")).unwrap();
    fs::create_dir_all(&workspace).unwrap();
    fs::write(repository.join(".git/HEAD"), "ref: refs/heads/main\n").unwrap();
    fs::write(root.join("AGENTS.md"), "outside instructions").unwrap();
    fs::write(repository.join("CLAUDE.md"), "repository instructions").unwrap();

    let boundary = project_instruction_boundary(&workspace).unwrap();
    assert_eq!(boundary, fs::canonicalize(&repository).unwrap());
    assert!(has_project_instructions(&boundary, &workspace));
    fs::remove_file(repository.join("CLAUDE.md")).unwrap();
    assert!(!has_project_instructions(&boundary, &workspace));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn non_git_project_instruction_boundary_is_the_selected_workspace() {
    let root = std::env::temp_dir().join(format!(
        "coding-context-non-git-boundary-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let workspace = root.join("workspace");
    fs::create_dir_all(&workspace).unwrap();
    fs::write(root.join("AGENTS.md"), "outside instructions").unwrap();

    assert_eq!(
        project_instruction_boundary(&workspace).unwrap(),
        fs::canonicalize(&workspace).unwrap()
    );
    assert!(!has_project_instructions(&workspace, &workspace));
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn context_order_is_deterministic_and_untrusted_projects_are_not_read() {
    let root = std::env::temp_dir().join(format!(
        "coding-context-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let workspace = root.join("repo/sub");
    fs::create_dir_all(&workspace).unwrap();
    fs::write(root.join("AGENTS.md"), "root agents").unwrap();
    fs::write(root.join("repo/CLAUDE.md"), "repo claude").unwrap();
    fs::write(workspace.join("AGENTS.md"), "sub agents").unwrap();

    let ignored = ResourceCatalog::discover(
        &root,
        &workspace,
        ProjectAccess::Ignored,
        &[],
        &[],
        None,
        None,
    )
    .unwrap();
    assert!(ignored.context().is_empty());
    let trusted = ResourceCatalog::discover(
        &root,
        &workspace,
        ProjectAccess::Trusted,
        &[],
        &[],
        None,
        None,
    )
    .unwrap();
    assert_eq!(trusted.context().len(), 3);
    assert_eq!(trusted.logical_workspace(), "<workspace>/repo/sub");
    let provider = WorkspaceInstructionProvider::new(trusted.context().to_vec()).unwrap();
    let modules = provider
        .provide(
            ContextRequest::new(
                ProfileId::from_str("coding-agent").unwrap(),
                SessionId::from_str("0195a0b1-5e45-75be-8284-0aa7aa000011").unwrap(),
                None,
                Vec::new(),
                ProtocolMetadata::default(),
            )
            .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(modules[0].segments()[0].content(), "root agents");
    assert_eq!(modules[0].segments()[1].content(), "repo claude");
    assert_eq!(modules[0].segments()[2].content(), "sub agents");
    assert_eq!(
        modules[0].segments()[0].provenance().locator(),
        Some("AGENTS.md")
    );
    assert_eq!(
        modules[0].segments()[2].provenance().locator(),
        Some("repo/sub/AGENTS.md")
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn blank_context_files_are_ignored() {
    let root = std::env::temp_dir().join(format!(
        "coding-context-blank-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&root).unwrap();
    fs::write(root.join("AGENTS.md"), "").unwrap();
    fs::write(root.join("CLAUDE.md"), " \n\t\r\n").unwrap();

    let catalog =
        ResourceCatalog::discover(&root, &root, ProjectAccess::Trusted, &[], &[], None, None)
            .unwrap();

    assert!(catalog.context().is_empty());
    assert!(catalog.diagnostics().is_empty());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn context_candidates_use_first_match_per_directory_and_ancestor_order() {
    let root = std::env::temp_dir().join(format!(
        "coding-context-priority-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let workspace = root.join("repo/sub/deep");
    fs::create_dir_all(&workspace).unwrap();
    fs::write(root.join("AGENTS.md"), "root agents").unwrap();
    fs::write(root.join("CLAUDE.md"), "root compatibility").unwrap();
    fs::write(root.join("repo/AGENTS.MD"), "repo uppercase").unwrap();
    fs::write(root.join("repo/CLAUDE.md"), "repo compatibility").unwrap();
    fs::write(root.join("repo/sub/AGENTS.override.md"), "sub override").unwrap();
    fs::write(root.join("repo/sub/AGENTS.md"), "sub agents").unwrap();
    fs::write(workspace.join("CLAUDE.MD"), "deep compatibility").unwrap();

    let catalog = ResourceCatalog::discover(
        &root,
        &workspace,
        ProjectAccess::Trusted,
        &[],
        &[],
        None,
        None,
    )
    .unwrap();

    assert_eq!(
        catalog
            .context()
            .iter()
            .map(|instruction| (instruction.locator(), instruction.content()))
            .collect::<Vec<_>>(),
        [
            ("AGENTS.md", "root agents"),
            ("repo/AGENTS.md", "repo uppercase"),
            ("repo/sub/AGENTS.override.md", "sub override"),
            ("repo/sub/deep/CLAUDE.md", "deep compatibility"),
        ]
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn global_context_precedes_project_context_and_is_ignored_with_project_resources_disabled() {
    let root = std::env::temp_dir().join(format!(
        "coding-context-global-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let global = root.join("global");
    let workspace = root.join("workspace");
    fs::create_dir_all(&global).unwrap();
    fs::create_dir_all(&workspace).unwrap();
    fs::write(global.join("AGENTS.md"), "global agents").unwrap();
    fs::write(workspace.join("AGENTS.md"), "project agents").unwrap();
    let roots = CodingPromptResourceRoots::new(&global)
        .unwrap()
        .with_project_root(workspace.join(".tea"))
        .unwrap();

    let trusted = ResourceCatalog::discover_complete(
        &workspace,
        &workspace,
        ProjectAccess::Trusted,
        &[],
        None,
        None,
        Some(&roots),
    )
    .unwrap();
    assert_eq!(
        trusted
            .context()
            .iter()
            .map(tea_context::WorkspaceInstruction::locator)
            .collect::<Vec<_>>(),
        ["<global>/AGENTS.md", "AGENTS.md"]
    );

    let ignored = ResourceCatalog::discover_complete(
        &workspace,
        &workspace,
        ProjectAccess::Ignored,
        &[],
        None,
        None,
        Some(&roots),
    )
    .unwrap();
    assert_eq!(ignored.context().len(), 1);
    assert_eq!(ignored.context()[0].locator(), "<global>/AGENTS.md");
    fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[test]
fn canonical_context_targets_are_loaded_once_for_compatibility_symlinks() {
    use std::os::unix::fs::symlink;

    let root = std::env::temp_dir().join(format!(
        "coding-context-dedupe-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let workspace = root.join("repo/sub");
    fs::create_dir_all(&workspace).unwrap();
    fs::write(root.join("AGENTS.md"), "shared agents").unwrap();
    symlink(root.join("AGENTS.md"), root.join("repo/CLAUDE.md")).unwrap();
    symlink(root.join("AGENTS.md"), workspace.join("CLAUDE.MD")).unwrap();

    let catalog = ResourceCatalog::discover(
        &root,
        &workspace,
        ProjectAccess::Trusted,
        &[],
        &[],
        None,
        None,
    )
    .unwrap();

    assert_eq!(catalog.context().len(), 1);
    assert_eq!(catalog.context()[0].locator(), "AGENTS.md");
    assert_eq!(catalog.context()[0].content(), "shared agents");
    fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[test]
fn nested_linked_worktree_shadows_only_the_main_repo_file_with_the_same_name() {
    let root = std::env::temp_dir().join(format!(
        "coding-context-worktree-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let main = root.join("main");
    let worktree = main.join("worktrees/feature");
    let workspace = worktree.join("src");
    link_worktree(&main, &worktree, "feature");
    fs::create_dir_all(&workspace).unwrap();
    fs::write(main.join("AGENTS.md"), "main instructions").unwrap();
    fs::write(worktree.join("AGENTS.md"), "worktree instructions").unwrap();

    let catalog = ResourceCatalog::discover(
        &root,
        &workspace,
        ProjectAccess::Trusted,
        &[],
        &[],
        None,
        None,
    )
    .unwrap();
    assert_eq!(
        catalog
            .context()
            .iter()
            .map(tea_context::WorkspaceInstruction::content)
            .collect::<Vec<_>>(),
        ["worktree instructions"]
    );

    fs::remove_file(main.join("AGENTS.md")).unwrap();
    fs::write(main.join("CLAUDE.md"), "main compatibility").unwrap();
    let catalog = ResourceCatalog::discover(
        &root,
        &workspace,
        ProjectAccess::Trusted,
        &[],
        &[],
        None,
        None,
    )
    .unwrap();
    assert_eq!(
        catalog
            .context()
            .iter()
            .map(tea_context::WorkspaceInstruction::content)
            .collect::<Vec<_>>(),
        ["main compatibility", "worktree instructions"]
    );
    fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[test]
fn nested_linked_worktree_inherits_main_context_when_worktree_has_none() {
    let root = std::env::temp_dir().join(format!(
        "coding-context-worktree-inherit-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let main = root.join("main");
    let worktree = main.join("worktrees/feature");
    let workspace = worktree.join("src");
    link_worktree(&main, &worktree, "feature");
    fs::create_dir_all(&workspace).unwrap();
    fs::write(main.join("AGENTS.md"), "main instructions").unwrap();

    let catalog = ResourceCatalog::discover(
        &root,
        &workspace,
        ProjectAccess::Trusted,
        &[],
        &[],
        None,
        None,
    )
    .unwrap();
    assert_eq!(catalog.context().len(), 1);
    assert_eq!(catalog.context()[0].content(), "main instructions");
    fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[test]
fn linked_worktree_boundaries_distinguish_nested_and_sibling_layouts() {
    let root = std::env::temp_dir().join(format!(
        "coding-context-worktree-boundaries-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let main = root.join("main");
    let nested = main.join("worktrees/nested");
    let sibling = root.join("sibling");
    link_worktree(&main, &nested, "nested");
    link_worktree(&main, &sibling, "sibling");
    let nested_workspace = nested.join("src");
    let sibling_workspace = sibling.join("src");
    fs::create_dir_all(&nested_workspace).unwrap();
    fs::create_dir_all(&sibling_workspace).unwrap();

    assert_eq!(
        project_instruction_boundary(&nested_workspace).unwrap(),
        fs::canonicalize(&main).unwrap()
    );
    assert_eq!(
        project_instruction_boundary(&sibling_workspace).unwrap(),
        fs::canonicalize(&sibling).unwrap()
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn submodule_pointer_keeps_the_submodule_as_its_instruction_boundary() {
    let root = std::env::temp_dir().join(format!(
        "coding-context-submodule-boundary-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let parent = root.join("parent");
    let submodule = parent.join("vendor/library");
    let workspace = submodule.join("src");
    let git_dir = parent.join(".git/modules/vendor/library");
    fs::create_dir_all(&git_dir).unwrap();
    fs::create_dir_all(&workspace).unwrap();
    fs::write(parent.join(".git/HEAD"), "ref: refs/heads/main\n").unwrap();
    fs::write(git_dir.join("HEAD"), "ref: refs/heads/main\n").unwrap();
    fs::write(
        submodule.join(".git"),
        format!("gitdir: {}\n", git_dir.display()),
    )
    .unwrap();

    assert_eq!(
        project_instruction_boundary(&workspace).unwrap(),
        fs::canonicalize(&submodule).unwrap()
    );
    fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
fn link_worktree(main: &std::path::Path, worktree: &std::path::Path, name: &str) {
    let git_dir = main.join(format!(".git/worktrees/{name}"));
    fs::create_dir_all(&git_dir).unwrap();
    fs::create_dir_all(worktree).unwrap();
    fs::write(main.join(".git/HEAD"), "ref: refs/heads/main\n").unwrap();
    fs::write(git_dir.join("HEAD"), "ref: refs/heads/feature\n").unwrap();
    fs::write(git_dir.join("commondir"), "../..\n").unwrap();
    fs::write(
        worktree.join(".git"),
        format!("gitdir: {}\n", git_dir.display()),
    )
    .unwrap();
}

#[test]
fn explicit_context_uses_the_nested_logical_workspace_locator() {
    let root = std::env::temp_dir().join(format!(
        "coding-context-explicit-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let workspace_path = root.join("repo/sub");
    fs::create_dir_all(&workspace_path).unwrap();
    fs::write(workspace_path.join("CONTEXT.md"), "explicit rules\n").unwrap();
    let workspace = WorkspaceRoot::new(&workspace_path).unwrap();
    let mut catalog = ResourceCatalog::discover(
        &root,
        &workspace_path,
        ProjectAccess::Trusted,
        &[],
        &[],
        None,
        None,
    )
    .unwrap();

    catalog
        .add_explicit_context_files(&workspace, &["CONTEXT.md".to_owned()])
        .unwrap();

    assert_eq!(catalog.logical_workspace(), "<workspace>/repo/sub");
    assert_eq!(catalog.context().len(), 1);
    assert_eq!(catalog.context()[0].locator(), "repo/sub/CONTEXT.md");
    fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[test]
fn trusted_context_symlink_cannot_escape_boundary() {
    use std::os::unix::fs::symlink;

    let root = std::env::temp_dir().join(format!(
        "coding-context-symlink-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let outside = root.with_extension("outside");
    fs::create_dir_all(&root).unwrap();
    fs::write(&outside, "host secret").unwrap();
    symlink(&outside, root.join("AGENTS.md")).unwrap();
    assert!(
        ResourceCatalog::discover(&root, &root, ProjectAccess::Trusted, &[], &[], None, None,)
            .is_err()
    );
    fs::remove_dir_all(root).unwrap();
    fs::remove_file(outside).unwrap();
}
