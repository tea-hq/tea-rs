use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

use tea_coding::ProjectAccess;
use tea_coding::resources::{CodingPromptResourceRoots, ResourceCatalog};
use tea_context::TrustLevel;

static ID: AtomicU64 = AtomicU64::new(0);

fn fixture(name: &str) -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
    let root = std::env::temp_dir().join(format!(
        "coding-system-resources-{name}-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let global = root.join("global");
    let workspace = root.join("workspace");
    fs::create_dir_all(&global).unwrap();
    fs::create_dir_all(workspace.join(".tea")).unwrap();
    (root, global, workspace)
}

fn discover(
    global: &std::path::Path,
    workspace: &std::path::Path,
    access: ProjectAccess,
) -> ResourceCatalog {
    let roots = CodingPromptResourceRoots::new(global)
        .unwrap()
        .with_project_root(workspace.join(".tea"))
        .unwrap();
    ResourceCatalog::discover_complete(workspace, workspace, access, &[], None, None, Some(&roots))
        .unwrap()
}

#[test]
fn project_system_resources_override_global_with_logical_provenance() {
    let (root, global, workspace) = fixture("precedence");
    fs::write(global.join("SYSTEM.md"), "global behavior").unwrap();
    fs::write(global.join("APPEND_SYSTEM.md"), "global append").unwrap();
    fs::write(workspace.join(".tea/SYSTEM.md"), "project behavior").unwrap();
    fs::write(workspace.join(".tea/APPEND_SYSTEM.md"), "project append").unwrap();

    let catalog = discover(&global, &workspace, ProjectAccess::Trusted);
    let behavior = catalog.system_prompt().unwrap();
    assert_eq!(behavior.content(), "project behavior");
    assert_eq!(behavior.locator(), "<workspace>/.tea/SYSTEM.md");
    assert_eq!(behavior.trust(), TrustLevel::Delegated);
    let append = catalog.append_system_prompt().unwrap();
    assert_eq!(append.content(), "project append");
    assert_eq!(append.locator(), "<workspace>/.tea/APPEND_SYSTEM.md");
    assert_eq!(append.trust(), TrustLevel::Delegated);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn untrusted_project_resources_are_not_read_and_global_resources_remain_available() {
    let (root, global, workspace) = fixture("untrusted");
    fs::write(global.join("SYSTEM.md"), "global behavior").unwrap();
    fs::write(global.join("APPEND_SYSTEM.md"), "global append").unwrap();
    fs::write(workspace.join(".tea/SYSTEM.md"), [0xff]).unwrap();
    fs::write(workspace.join(".tea/APPEND_SYSTEM.md"), [0xff]).unwrap();

    let catalog = discover(&global, &workspace, ProjectAccess::Ignored);
    assert_eq!(
        catalog.system_prompt().unwrap().content(),
        "global behavior"
    );
    assert_eq!(
        catalog.system_prompt().unwrap().locator(),
        "<global>/SYSTEM.md"
    );
    assert_eq!(
        catalog.append_system_prompt().unwrap().content(),
        "global append"
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn blank_system_resources_are_treated_as_missing() {
    let (root, global, workspace) = fixture("blank");
    fs::write(global.join("SYSTEM.md"), "").unwrap();
    fs::write(global.join("APPEND_SYSTEM.md"), "").unwrap();

    let catalog = discover(&global, &workspace, ProjectAccess::Trusted);
    assert!(catalog.system_prompt().is_none());
    assert!(catalog.append_system_prompt().is_none());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn invalid_selected_system_resource_fails_without_fallback() {
    let (root, global, workspace) = fixture("invalid");
    fs::write(global.join("SYSTEM.md"), "global behavior").unwrap();
    fs::write(workspace.join(".tea/SYSTEM.md"), [0xff]).unwrap();
    let roots = CodingPromptResourceRoots::new(&global)
        .unwrap()
        .with_project_root(workspace.join(".tea"))
        .unwrap();

    assert!(
        ResourceCatalog::discover_complete(
            &workspace,
            &workspace,
            ProjectAccess::Trusted,
            &[],
            None,
            None,
            Some(&roots),
        )
        .is_err()
    );
    fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[test]
fn selected_system_resource_symlink_cannot_escape_its_root() {
    use std::os::unix::fs::symlink;

    let (root, global, workspace) = fixture("escape");
    let outside = root.join("outside.md");
    fs::write(&outside, "outside").unwrap();
    symlink(&outside, workspace.join(".tea/SYSTEM.md")).unwrap();
    let roots = CodingPromptResourceRoots::new(&global)
        .unwrap()
        .with_project_root(workspace.join(".tea"))
        .unwrap();

    assert!(
        ResourceCatalog::discover_complete(
            &workspace,
            &workspace,
            ProjectAccess::Trusted,
            &[],
            None,
            None,
            Some(&roots),
        )
        .is_err()
    );
    fs::remove_dir_all(root).unwrap();
}
