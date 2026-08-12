use std::collections::BTreeMap;
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

use tea_coding::ProjectAccess;
use tea_coding::resources::ResourceCatalog;

static ID: AtomicU64 = AtomicU64::new(0);

#[test]
fn project_template_overrides_global_and_expansion_is_single_pass() {
    let root = std::env::temp_dir().join(format!(
        "coding-prompts-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let workspace = root.join("workspace");
    let global = root.join("global");
    let project = root.join("project");
    for path in [&workspace, &global, &project] {
        fs::create_dir_all(path).unwrap();
    }
    fs::write(
        global.join("review.md"),
        "---\nname: review\ndescription: global\ndefault_tone: concise\n---\nGlobal $1 ${tone}\n",
    )
    .unwrap();
    fs::write(
        project.join("review.md"),
        "---\nname: review\ndescription: project\ndefault_tone: concise\n---\nProject $1 ${tone}\n",
    )
    .unwrap();
    let catalog = ResourceCatalog::discover(
        &root,
        &workspace,
        ProjectAccess::Trusted,
        &[],
        &[],
        Some(&global),
        Some(&project),
    )
    .unwrap();
    let template = &catalog.prompts()[0];
    assert_eq!(template.description(), "project");
    let output = template.expand(&["${tone}".to_owned()], &BTreeMap::new());
    assert_eq!(output, "Project ${tone} concise\n");
    let ignored = ResourceCatalog::discover(
        &root,
        &workspace,
        ProjectAccess::Ignored,
        &[],
        &[],
        Some(&global),
        Some(&project),
    )
    .unwrap();
    assert_eq!(ignored.prompts()[0].description(), "global");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn frontmatter_accepts_crlf_and_quoted_colons() {
    let root = std::env::temp_dir().join(format!(
        "coding-prompts-frontmatter-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let workspace = root.join("workspace");
    let global = root.join("global");
    fs::create_dir_all(&workspace).unwrap();
    fs::create_dir_all(&global).unwrap();
    fs::write(
        global.join("review.md"),
        "---\r\nname: review\r\ndescription: \"Review: Rust\"\r\n---\r\nBody\r\n",
    )
    .unwrap();

    let catalog = ResourceCatalog::discover(
        &root,
        &workspace,
        ProjectAccess::Ignored,
        &[],
        &[],
        Some(&global),
        None,
    )
    .unwrap();
    assert_eq!(catalog.prompts()[0].description(), "Review: Rust");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn frontmatter_rejects_non_string_required_fields() {
    let root = std::env::temp_dir().join(format!(
        "coding-prompts-frontmatter-invalid-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let workspace = root.join("workspace");
    let global = root.join("global");
    fs::create_dir_all(&workspace).unwrap();
    fs::create_dir_all(&global).unwrap();
    fs::write(
        global.join("review.md"),
        "---\nname: review\ndescription: true\n---\nBody\n",
    )
    .unwrap();

    assert!(
        ResourceCatalog::discover(
            &root,
            &workspace,
            ProjectAccess::Ignored,
            &[],
            &[],
            Some(&global),
            None,
        )
        .is_err()
    );
    fs::remove_dir_all(root).unwrap();
}
