use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};

use tea_coding::ProjectAccess;
use tea_coding::resources::{ResourceCatalog, SkillRoot, SkillSource};
use tea_coding_tools::MAX_READ_BYTES;
use tea_context::SkillCommand;

static ID: AtomicU64 = AtomicU64::new(0);

#[test]
fn skill_roots_and_discovered_skills_expose_bounded_provenance() {
    let root = std::env::temp_dir().join(format!(
        "coding-skills-provenance-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let workspace = root.join("workspace");
    let skills = root.join("skills");
    let skill = skills.join("review");
    fs::create_dir_all(&workspace).unwrap();
    fs::create_dir_all(&skill).unwrap();
    fs::write(
        skill.join("SKILL.md"),
        "---\nname: review\ndescription: Review code safely\n---\nbody\n",
    )
    .unwrap();

    let skill_root = SkillRoot::new(&skills, SkillSource::ProjectAgents).unwrap();
    assert_eq!(skill_root.path(), skills.as_path());
    assert_eq!(skill_root.source(), SkillSource::ProjectAgents);
    assert!(SkillRoot::new("relative/skills", SkillSource::Explicit).is_err());

    let catalog = ResourceCatalog::discover_with_skill_roots(
        &root,
        &workspace,
        ProjectAccess::Trusted,
        &[skill_root],
        None,
        None,
    )
    .unwrap();
    let discovered = &catalog.skills()[0];
    assert_eq!(discovered.source(), SkillSource::ProjectAgents);
    assert!(discovered.model_invocable());
    assert_eq!(
        discovered.manifest_path(),
        fs::canonicalize(skill.join("SKILL.md")).unwrap()
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn skill_diagnostics_expose_source_without_absolute_paths() {
    let root = std::env::temp_dir().join(format!(
        "coding-skills-diagnostic-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let workspace = root.join("workspace");
    let skills = root.join("skills");
    let skill = skills.join("broken");
    fs::create_dir_all(&workspace).unwrap();
    fs::create_dir_all(&skill).unwrap();
    fs::write(skill.join("SKILL.md"), "not frontmatter\n").unwrap();

    let catalog = ResourceCatalog::discover_with_skill_roots(
        &root,
        &workspace,
        ProjectAccess::Ignored,
        &[SkillRoot::new(&skills, SkillSource::UserAgents).unwrap()],
        None,
        None,
    )
    .unwrap();
    let diagnostic = catalog
        .diagnostics()
        .iter()
        .find(|diagnostic| diagnostic.code() == "skill_invalid")
        .unwrap();
    assert_eq!(diagnostic.source(), Some(SkillSource::UserAgents));
    assert!(!diagnostic.subject().contains(root.to_str().unwrap()));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn directory_roots_discover_nested_skills_with_ignores_and_boundaries() {
    let root = std::env::temp_dir().join(format!(
        "coding-skills-recursive-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let workspace = root.join("workspace");
    let skills = root.join("skills");
    fs::create_dir_all(&workspace).unwrap();
    for directory in [
        "team/review/nested",
        "release",
        ".hidden",
        "node_modules/vendor",
        "ignored-git",
        "ignored-ignore",
        "ignored-fd",
    ] {
        fs::create_dir_all(skills.join(directory)).unwrap();
    }
    fs::write(skills.join(".gitignore"), "ignored-git/\n").unwrap();
    fs::write(skills.join(".ignore"), "ignored-ignore/\n").unwrap();
    fs::write(skills.join(".fdignore"), "ignored-fd/\n").unwrap();
    fs::write(skills.join("root-note.md"), "not a skill\n").unwrap();
    for (directory, name) in [
        ("team/review", "review"),
        ("team/review/nested", "nested"),
        ("release", "release"),
        (".hidden", "hidden"),
        ("node_modules/vendor", "vendor"),
        ("ignored-git", "git-ignored"),
        ("ignored-ignore", "ignore-ignored"),
        ("ignored-fd", "fd-ignored"),
    ] {
        fs::write(
            skills.join(directory).join("SKILL.md"),
            format!("---\nname: {name}\ndescription: {name}\n---\nbody\n"),
        )
        .unwrap();
    }

    let catalog = ResourceCatalog::discover_with_skill_roots(
        &root,
        &workspace,
        ProjectAccess::Ignored,
        &[SkillRoot::new(&skills, SkillSource::UserTea).unwrap()],
        None,
        None,
    )
    .unwrap();
    let ids = catalog
        .skills()
        .iter()
        .map(|skill| skill.metadata().id().as_str())
        .collect::<Vec<_>>();
    assert_eq!(ids, ["release", "review"]);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn explicit_manifest_files_are_supported_and_other_files_are_rejected() {
    let root = std::env::temp_dir().join(format!(
        "coding-skills-explicit-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let workspace = root.join("workspace");
    fs::create_dir_all(&workspace).unwrap();
    let manifest = root.join("single.md");
    fs::write(
        &manifest,
        "---\nname: single\ndescription: Explicit\n---\nbody\n",
    )
    .unwrap();
    let catalog = ResourceCatalog::discover_with_skill_roots(
        &root,
        &workspace,
        ProjectAccess::Ignored,
        &[SkillRoot::new(&manifest, SkillSource::Explicit).unwrap()],
        None,
        None,
    )
    .unwrap();
    assert_eq!(catalog.skills()[0].metadata().id().as_str(), "single");

    let text = root.join("single.txt");
    fs::write(&text, "body\n").unwrap();
    assert!(
        ResourceCatalog::discover_with_skill_roots(
            &root,
            &workspace,
            ProjectAccess::Ignored,
            &[SkillRoot::new(&text, SkillSource::Explicit).unwrap()],
            None,
            None,
        )
        .is_err()
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn malformed_candidates_do_not_hide_valid_siblings() {
    let root = std::env::temp_dir().join(format!(
        "coding-skills-fault-isolation-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let workspace = root.join("workspace");
    let skills = root.join("skills");
    fs::create_dir_all(&workspace).unwrap();
    for directory in ["good", "bad-yaml", "missing-name", "bad-id", "too-large"] {
        fs::create_dir_all(skills.join(directory)).unwrap();
    }
    fs::write(
        skills.join("good/SKILL.md"),
        "---\nname: good\ndescription: good\n---\nbody\n",
    )
    .unwrap();
    fs::write(skills.join("bad-yaml/SKILL.md"), "not yaml\n").unwrap();
    fs::write(
        skills.join("missing-name/SKILL.md"),
        "---\ndescription: missing\n---\nbody\n",
    )
    .unwrap();
    fs::write(
        skills.join("bad-id/SKILL.md"),
        "---\nname: Bad\ndescription: bad\n---\nbody\n",
    )
    .unwrap();
    fs::write(
        skills.join("too-large/SKILL.md"),
        format!(
            "---\nname: too-large\ndescription: large\n---\n{}",
            "x".repeat(128 * 1024)
        ),
    )
    .unwrap();

    let catalog = ResourceCatalog::discover_with_skill_roots(
        &root,
        &workspace,
        ProjectAccess::Ignored,
        &[SkillRoot::new(&skills, SkillSource::UserTea).unwrap()],
        None,
        None,
    )
    .unwrap();
    assert_eq!(catalog.skills().len(), 1);
    assert_eq!(catalog.skills()[0].metadata().id().as_str(), "good");
    assert!(
        catalog
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.code() == "skill_invalid")
            .count()
            >= 4
    );

    #[cfg(unix)]
    {
        use std::os::unix::fs::symlink;
        let outside = root.join("outside.md");
        fs::write(&outside, "outside\n").unwrap();
        fs::create_dir_all(skills.join("escape")).unwrap();
        symlink(&outside, skills.join("escape/SKILL.md")).unwrap();
        let catalog = ResourceCatalog::discover_with_skill_roots(
            &root,
            &workspace,
            ProjectAccess::Ignored,
            &[SkillRoot::new(&skills, SkillSource::UserTea).unwrap()],
            None,
            None,
        )
        .unwrap();
        assert!(
            catalog
                .diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.code() == "skill_invalid")
        );
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn skill_roots_merge_by_source_precedence_and_keep_sorted_winners() {
    let root = std::env::temp_dir().join(format!(
        "coding-skills-precedence-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let workspace = root.join("workspace");
    fs::create_dir_all(&workspace).unwrap();
    let sources = [
        (SkillSource::Explicit, "explicit"),
        (SkillSource::ProjectTea, "project-tea"),
        (SkillSource::ProjectAgents, "project-agents"),
        (SkillSource::UserTea, "user-tea"),
        (SkillSource::UserAgents, "user-agents"),
        (SkillSource::TeaData, "tea-data"),
    ];
    let mut roots = Vec::new();
    for (source, label) in sources {
        let directory = root.join(label);
        fs::create_dir_all(directory.join("shared")).unwrap();
        fs::write(
            directory.join("shared/SKILL.md"),
            format!("---\nname: shared\ndescription: {label}\n---\n{label} body\n"),
        )
        .unwrap();
        fs::create_dir_all(directory.join("unique")).unwrap();
        fs::write(
            directory.join("unique/SKILL.md"),
            format!("---\nname: {label}\ndescription: unique\n---\nbody\n"),
        )
        .unwrap();
        roots.push(SkillRoot::new(directory, source).unwrap());
    }
    roots.reverse();
    let catalog = ResourceCatalog::discover_with_skill_roots(
        &root,
        &workspace,
        ProjectAccess::Trusted,
        &roots,
        None,
        None,
    )
    .unwrap();
    let ids = catalog
        .skills()
        .iter()
        .map(|skill| skill.metadata().id().as_str())
        .collect::<Vec<_>>();
    let mut sorted = ids.clone();
    sorted.sort_unstable();
    assert_eq!(ids, sorted);
    let shared = catalog
        .skills()
        .iter()
        .find(|skill| skill.metadata().id().as_str() == "shared")
        .unwrap();
    assert_eq!(shared.source(), SkillSource::Explicit);
    assert_eq!(shared.metadata().description(), "explicit");
    assert_eq!(
        catalog.invoke_skill("/skill:shared").unwrap().content(),
        "explicit body\n"
    );
    assert_eq!(
        catalog
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.code() == "skill_shadowed")
            .count(),
        5
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn untrusted_project_roots_are_not_touched_before_trust_filtering() {
    let root = std::env::temp_dir().join(format!(
        "coding-skills-trust-filter-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let workspace = root.join("workspace");
    fs::create_dir_all(&workspace).unwrap();
    let project_tea = root.join("project-tea");
    let project_agents = root.join("project-agents");
    fs::write(&project_tea, "not a directory").unwrap();
    fs::write(&project_agents, "not a directory").unwrap();
    let roots = [
        SkillRoot::new(&project_tea, SkillSource::ProjectTea).unwrap(),
        SkillRoot::new(&project_agents, SkillSource::ProjectAgents).unwrap(),
    ];
    let ignored = ResourceCatalog::discover_with_skill_roots(
        &root,
        &workspace,
        ProjectAccess::Ignored,
        &roots,
        None,
        None,
    )
    .unwrap();
    assert!(ignored.skills().is_empty());
    assert!(ignored.diagnostics().is_empty());
    let trusted = ResourceCatalog::discover_with_skill_roots(
        &root,
        &workspace,
        ProjectAccess::Trusted,
        &roots,
        None,
        None,
    )
    .unwrap();
    assert_eq!(
        trusted
            .diagnostics()
            .iter()
            .filter(|diagnostic| diagnostic.code() == "skill_root_invalid")
            .count(),
        2
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn hidden_skills_are_explicitly_loadable_but_not_model_visible() {
    let root = std::env::temp_dir().join(format!(
        "coding-skills-hidden-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let workspace = root.join("workspace");
    let skills = root.join("skills");
    let skill = skills.join("manual");
    fs::create_dir_all(&workspace).unwrap();
    fs::create_dir_all(&skill).unwrap();
    fs::write(
        skill.join("SKILL.md"),
        "---\nname: manual-only\ndescription: Explicit invocation only\ndisable-model-invocation: true\n---\nRead references/checklist.md.\n",
    )
    .unwrap();
    let catalog = ResourceCatalog::discover_with_skill_roots(
        &root,
        &workspace,
        ProjectAccess::Ignored,
        &[SkillRoot::new(&skills, SkillSource::UserTea).unwrap()],
        None,
        None,
    )
    .unwrap();
    assert_eq!(catalog.skills().len(), 1);
    assert!(catalog.skill_metadata().is_empty());
    let command = "/skill:manual-only src/lib.rs"
        .parse::<SkillCommand>()
        .unwrap();
    let loaded = catalog.load_skill(&command).unwrap();
    assert_eq!(loaded.arguments(), "src/lib.rs");
    let loaded_without_args = catalog
        .load_skill(&"/skill:manual-only".parse::<SkillCommand>().unwrap())
        .unwrap();
    assert_eq!(
        loaded_without_args.prompt_fragment(),
        "Read references/checklist.md.\n"
    );
    assert_eq!(
        loaded.prompt_fragment(),
        "Read references/checklist.md.\n\n\nArguments: src/lib.rs"
    );
    assert_eq!(
        catalog
            .invoke_skill("/skill:manual-only src/lib.rs")
            .unwrap(),
        loaded
    );

    fs::write(
        skill.join("SKILL.md"),
        "---\nname: manual-only\ndescription: Changed\ndisable-model-invocation: true\n---\nbody\n",
    )
    .unwrap();
    assert!(catalog.load_skill(&command).is_err());
    fs::remove_dir_all(root).unwrap();
}

#[cfg(unix)]
#[test]
fn canonical_manifest_aliases_are_deduplicated_before_name_merge() {
    use std::os::unix::fs::symlink;

    let root = std::env::temp_dir().join(format!(
        "coding-skills-canonical-dedupe-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let workspace = root.join("workspace");
    let real = root.join("real");
    let alias = root.join("alias");
    fs::create_dir_all(&workspace).unwrap();
    fs::create_dir_all(real.join("review")).unwrap();
    fs::write(
        real.join("review/SKILL.md"),
        "---\nname: review\ndescription: one\n---\nbody\n",
    )
    .unwrap();
    symlink(&real, &alias).unwrap();

    let roots = [
        SkillRoot::new(&real, SkillSource::UserTea).unwrap(),
        SkillRoot::new(&alias, SkillSource::UserAgents).unwrap(),
    ];
    let catalog = ResourceCatalog::discover_with_skill_roots(
        &root,
        &workspace,
        ProjectAccess::Ignored,
        &roots,
        None,
        None,
    )
    .unwrap();
    assert_eq!(catalog.skills().len(), 1);
    assert!(
        !catalog
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.code() == "skill_shadowed")
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn skill_metadata_is_eager_body_is_explicit_and_references_are_confined() {
    let root = std::env::temp_dir().join(format!(
        "coding-skills-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let workspace = root.join("workspace");
    let skill = root.join("global-skills/review");
    fs::create_dir_all(&workspace).unwrap();
    fs::create_dir_all(&skill).unwrap();
    fs::write(skill.join("guide.md"), "guide").unwrap();
    fs::write(
        skill.join("SKILL.md"),
        "---\nname: review\ndescription: Review code safely\n---\nFollow guide.md.\n",
    )
    .unwrap();
    let catalog = ResourceCatalog::discover(
        &root,
        &workspace,
        ProjectAccess::Ignored,
        &[root.join("global-skills")],
        &[],
        None,
        None,
    )
    .unwrap();
    assert_eq!(
        catalog.skill_metadata()[0].description(),
        "Review code safely"
    );
    let loaded = catalog.invoke_skill("/skill:review src/lib.rs").unwrap();
    assert_eq!(loaded.arguments(), "src/lib.rs");
    assert!(catalog.invoke_skill("/skill:reviewer").is_err());
    assert!(loaded.content().contains("guide.md"));
    assert_eq!(
        loaded.resolve_reference("guide.md").unwrap(),
        fs::canonicalize(skill.join("guide.md")).unwrap()
    );
    assert!(loaded.resolve_reference("../outside").is_err());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn skill_resources_are_bounded_line_reads_from_the_winning_skill() {
    let root = std::env::temp_dir().join(format!(
        "coding-skill-resources-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let workspace = root.join("workspace");
    let skills = root.join("skills/review");
    fs::create_dir_all(&workspace).unwrap();
    fs::create_dir_all(skills.join("references")).unwrap();
    fs::write(
        skills.join("SKILL.md"),
        "---\nname: review\ndescription: Review code\n---\nbody\n",
    )
    .unwrap();
    fs::write(
        skills.join("references/checklist.md"),
        "zero\none\ntwo\nthree\n",
    )
    .unwrap();
    fs::create_dir(skills.join("references/directory")).unwrap();
    fs::write(skills.join("binary"), [0, 1, 2]).unwrap();
    fs::write(skills.join("large"), vec![b'x'; MAX_READ_BYTES + 1]).unwrap();

    let catalog = ResourceCatalog::discover_with_skill_roots(
        &root,
        &workspace,
        ProjectAccess::Ignored,
        &[SkillRoot::new(root.join("skills"), SkillSource::UserTea).unwrap()],
        None,
        None,
    )
    .unwrap();
    let skill_id = "review".parse().unwrap();
    let result = catalog
        .read_skill_resource(&skill_id, "references/checklist.md", Some(2), Some(3))
        .unwrap();
    assert_eq!(result.skill_id().as_str(), "review");
    assert_eq!(result.path(), "references/checklist.md");
    assert_eq!(result.content(), "one\ntwo\nthree\n");
    assert_eq!(result.start_line(), 2);
    assert_eq!(result.end_line(), 4);
    assert_eq!(result.total_lines(), 4);
    assert!(result.truncated());

    for path in ["", "../outside", "/etc/passwd", "missing.md"] {
        assert!(
            catalog
                .read_skill_resource(&skill_id, path, None, None)
                .is_err()
        );
    }
    for path in ["references/directory", "binary", "large"] {
        assert!(
            catalog
                .read_skill_resource(&skill_id, path, None, None)
                .is_err()
        );
    }
    let unknown = "unknown".parse().unwrap();
    assert!(
        catalog
            .read_skill_resource(&unknown, "references/checklist.md", None, None)
            .is_err()
    );
    assert!(
        catalog
            .read_skill_resource(&skill_id, "references/checklist.md", Some(0), None)
            .is_err()
    );

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn duplicate_skill_names_use_first_wins_and_diagnostic() {
    let root = std::env::temp_dir().join(format!(
        "coding-skills-dup-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    let workspace = root.join("workspace");
    for path in [
        root.join("skills/a"),
        root.join("skills/b"),
        workspace.clone(),
    ] {
        fs::create_dir_all(path).unwrap();
    }
    for (dir, description) in [("a", "first"), ("b", "second")] {
        fs::write(
            root.join(format!("skills/{dir}/SKILL.md")),
            format!("---\nname: same\ndescription: {description}\n---\n{description}\n"),
        )
        .unwrap();
    }
    let catalog = ResourceCatalog::discover(
        &root,
        &workspace,
        ProjectAccess::Ignored,
        &[root.join("skills")],
        &[],
        None,
        None,
    )
    .unwrap();
    assert_eq!(catalog.skills().len(), 1);
    assert_eq!(catalog.skills()[0].metadata().description(), "first");
    assert_eq!(
        catalog.invoke_skill("/skill:same").unwrap().content(),
        "first\n"
    );
    assert!(
        catalog
            .diagnostics()
            .iter()
            .any(|diagnostic| diagnostic.code() == "skill_shadowed")
    );
    fs::remove_dir_all(root).unwrap();
}
