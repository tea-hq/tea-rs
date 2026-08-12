use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use clap::Parser as _;
use tea_cli::args::CliArgs;
use tea_cli::{BootstrapEnvironment, CliBootstrap, ExitCategory};
use tea_coding::resources::SkillSource;

fn temp_root(label: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "tea-cli-skills-{label}-{}",
        uuid::Uuid::now_v7().hyphenated()
    ));
    fs::create_dir_all(&root).unwrap();
    root
}

fn write_skill(root: &Path, id: &str, description: &str) {
    let directory = root.join(id);
    fs::create_dir_all(&directory).unwrap();
    fs::write(
        directory.join("SKILL.md"),
        format!("---\nname: {id}\ndescription: {description}\n---\n{id} body\n"),
    )
    .unwrap();
}

fn args(workspace: &Path, config: &Path, state: &Path, data: &Path, trust: &str) -> CliArgs {
    CliArgs::try_parse_from([
        "tea",
        "--no-session",
        "--provider",
        "openai",
        "--model",
        "gpt-4o-mini",
        "--trust",
        trust,
        "--cwd",
        workspace.to_str().unwrap(),
        "--config-dir",
        config.to_str().unwrap(),
        "--state-dir",
        state.to_str().unwrap(),
        "--data-dir",
        data.to_str().unwrap(),
    ])
    .unwrap()
}

fn bootstrap(workspace: &Path, home: &Path) -> CliBootstrap {
    CliBootstrap::new(BootstrapEnvironment::new(
        workspace,
        Some(home.to_path_buf()),
        [("TEA_OPENAI_API_KEY".to_owned(), "test-key".to_owned())]
            .into_iter()
            .collect::<BTreeMap<_, _>>(),
    ))
}

#[test]
fn bootstrap_discovers_all_six_skill_sources_and_keeps_precedence() {
    let root = temp_root("all-sources");
    let workspace = root.join("workspace");
    let home = root.join("home");
    let config = root.join("config");
    let state = root.join("state");
    let data = root.join("data");
    fs::create_dir_all(&workspace).unwrap();
    fs::create_dir_all(config.join("skills")).unwrap();
    fs::create_dir_all(home.join(".agents/skills")).unwrap();
    fs::create_dir_all(data.join("skills")).unwrap();

    fs::write(
        workspace.join("explicit-skill.md"),
        "---\nname: explicit\ndescription: explicit\n---\nexplicit body\n",
    )
    .unwrap();
    write_skill(&workspace.join(".tea/skills"), "project-tea", "project tea");
    write_skill(
        &workspace.join(".agents/skills"),
        "project-agents",
        "project agents",
    );
    write_skill(&config.join("skills"), "user-tea", "user tea");
    write_skill(&home.join(".agents/skills"), "user-agents", "user agents");
    write_skill(&data.join("skills"), "tea-data", "tea data");
    write_skill(&workspace, "shared", "explicit shared");
    write_skill(
        &workspace.join(".tea/skills"),
        "shared-project-tea",
        "project",
    );
    fs::write(
        workspace.join("explicit-shared.md"),
        "---\nname: shared\ndescription: explicit shared\n---\nshared body\n",
    )
    .unwrap();
    fs::write(
        config.join("settings.json"),
        r#"{"schemaVersion":1,"resources":{"skillPaths":["explicit-shared.md"]}}"#,
    )
    .unwrap();

    let (service, _) = bootstrap(&workspace, &home)
        .build(&args(&workspace, &config, &state, &data, "once"))
        .unwrap();
    let skills = service.resources().skills();
    let sources = skills
        .iter()
        .map(tea_coding::resources::DiscoveredSkill::source)
        .collect::<Vec<_>>();
    for source in [
        SkillSource::Explicit,
        SkillSource::ProjectTea,
        SkillSource::ProjectAgents,
        SkillSource::UserTea,
        SkillSource::UserAgents,
        SkillSource::TeaData,
    ] {
        assert!(sources.contains(&source), "missing {source:?}");
    }
    let shared = skills
        .iter()
        .find(|skill| skill.metadata().id().as_str() == "shared")
        .unwrap();
    assert_eq!(shared.source(), SkillSource::Explicit);

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn ignored_project_skill_roots_are_not_read_or_diagnosed() {
    let root = temp_root("ignore");
    let workspace = root.join("workspace");
    let home = root.join("home");
    let config = root.join("config");
    let state = root.join("state");
    let data = root.join("data");
    fs::create_dir_all(workspace.join(".tea/skills/broken")).unwrap();
    fs::create_dir_all(workspace.join(".agents/skills/broken")).unwrap();
    fs::create_dir_all(config.join("skills")).unwrap();
    fs::create_dir_all(home.join(".agents/skills")).unwrap();
    fs::create_dir_all(data.join("skills")).unwrap();
    fs::write(
        workspace.join(".tea/skills/broken/SKILL.md"),
        "not frontmatter",
    )
    .unwrap();
    fs::write(
        workspace.join(".agents/skills/broken/SKILL.md"),
        "not frontmatter",
    )
    .unwrap();
    write_skill(&config.join("skills"), "user-tea", "user tea");
    write_skill(&home.join(".agents/skills"), "user-agents", "user agents");
    write_skill(&data.join("skills"), "tea-data", "tea data");

    let (service, _) = bootstrap(&workspace, &home)
        .build(&args(&workspace, &config, &state, &data, "ignore"))
        .unwrap();
    assert!(service.resources().skills().iter().all(|skill| !matches!(
        skill.source(),
        SkillSource::ProjectTea | SkillSource::ProjectAgents
    )));
    assert!(service.resources().diagnostics().iter().all(|diagnostic| {
        !matches!(
            diagnostic.source(),
            Some(SkillSource::ProjectTea | SkillSource::ProjectAgents)
        )
    }));

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn project_agents_root_requests_trust_in_default_mode() {
    let root = temp_root("agents-trust");
    let workspace = root.join("workspace");
    let config = root.join("config");
    let state = root.join("state");
    let data = root.join("data");
    fs::create_dir_all(workspace.join(".agents/skills")).unwrap();

    let error = bootstrap(&workspace, &root.join("home"))
        .build(&args(&workspace, &config, &state, &data, "default"))
        .unwrap_err();
    assert_eq!(error.category(), ExitCategory::TrustOrConfig);

    fs::remove_dir_all(root).unwrap();
}
