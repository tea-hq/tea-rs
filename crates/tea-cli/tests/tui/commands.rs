use tea_cli::tui::{CommandCatalog, CommandCompletionItem, CommandCompletionKind, SlashCommand};
use tea_protocol::ReasoningEffort;

#[test]
fn builtins_templates_and_skills_parse_without_prefix_guessing() {
    let catalog = CommandCatalog::new(["review"], ["rust-check"]).unwrap();
    assert_eq!(catalog.parse("/new").unwrap(), SlashCommand::New);
    assert_eq!(catalog.parse("/mcp").unwrap(), SlashCommand::Mcp);
    assert_eq!(
        catalog.parse("/mcp reconnect fixture").unwrap(),
        SlashCommand::McpReconnect("fixture".parse().unwrap())
    );
    assert_eq!(catalog.parse("/skills").unwrap(), SlashCommand::Skills);
    assert_eq!(
        catalog.parse("/model fake/model").unwrap(),
        SlashCommand::Model(Some("fake/model".parse().unwrap()))
    );
    assert_eq!(
        catalog.parse("/reasoning").unwrap(),
        SlashCommand::Reasoning(None)
    );
    for effort in ReasoningEffort::ALL {
        assert_eq!(
            catalog
                .parse(&format!("/reasoning {}", effort.as_str()))
                .unwrap(),
            SlashCommand::Reasoning(Some(effort))
        );
    }
    assert_eq!(
        catalog.parse("/review src/lib.rs").unwrap(),
        SlashCommand::Template {
            name: "review".to_owned(),
            arguments: vec!["src/lib.rs".to_owned()],
        }
    );
    assert_eq!(
        catalog.parse("/skill:rust-check --all").unwrap(),
        SlashCommand::Skill("/skill:rust-check --all".parse().unwrap())
    );
    assert_eq!(
        catalog.parse("/rust-check --all").unwrap(),
        SlashCommand::Skill("/skill:rust-check --all".parse().unwrap())
    );
    assert_eq!(
        catalog.parse("/rust-check --all  src").unwrap(),
        SlashCommand::Skill("/skill:rust-check --all  src".parse().unwrap())
    );
    assert!(catalog.parse("/unknown").is_err());
    assert!(catalog.parse("/mcp reconnect").is_err());
    assert!(catalog.parse("/skills extra").is_err());
    assert!(catalog.parse("/reasoning extreme").is_err());
    assert!(catalog.parse("/reasoning low high").is_err());
    for input in [
        "/skill:",
        "/skill:unknown",
        "/skill:bad.id?",
        "/skill:rust-check\0",
    ] {
        assert!(catalog.parse(input).is_err(), "accepted {input:?}");
    }
    assert!(
        catalog
            .parse(&format!("/skill:rust-check {}", "x".repeat(16 * 1024 + 1)))
            .is_err()
    );
}

#[test]
fn completion_is_sorted_bounded_and_includes_declarative_resources() {
    let catalog = CommandCatalog::new(
        ["review", "release"],
        [
            "rust-check",
            "frontend-design",
            "design-taste-frontend",
            "fast-node-debug",
        ],
    )
    .unwrap();
    assert_eq!(
        catalog.complete("/re", 8),
        [
            completion("/reasoning", CommandCompletionKind::Command),
            completion("/release", CommandCompletionKind::Prompt),
            completion("/resume", CommandCompletionKind::Command),
            completion("/review", CommandCompletionKind::Prompt),
            completion("/design-taste-frontend", CommandCompletionKind::Skill),
            completion("/frontend-design", CommandCompletionKind::Skill),
            completion("/rust-check", CommandCompletionKind::Skill),
        ]
    );
    assert_eq!(
        catalog.complete("/front", 8),
        [
            completion("/frontend-design", CommandCompletionKind::Skill),
            completion("/design-taste-frontend", CommandCompletionKind::Skill),
        ]
    );
    assert_eq!(
        catalog.complete("$front", 8),
        [
            completion("$frontend-design", CommandCompletionKind::Skill),
            completion("$design-taste-frontend", CommandCompletionKind::Skill),
        ]
    );
    assert_eq!(
        catalog.complete("$fnd", 8),
        [
            completion("$design-taste-frontend", CommandCompletionKind::Skill),
            completion("$fast-node-debug", CommandCompletionKind::Skill),
            completion("$frontend-design", CommandCompletionKind::Skill),
        ]
    );
    assert_eq!(
        catalog.complete("$RUST", 8),
        [completion("$rust-check", CommandCompletionKind::Skill)]
    );
    assert_eq!(
        catalog.complete("$", 8),
        [
            completion("$design-taste-frontend", CommandCompletionKind::Skill),
            completion("$fast-node-debug", CommandCompletionKind::Skill),
            completion("$frontend-design", CommandCompletionKind::Skill),
            completion("$rust-check", CommandCompletionKind::Skill),
        ]
    );
    assert!(catalog.complete("$missing", 8).is_empty());
    assert_eq!(
        catalog.complete("/mc", 8),
        [completion("/mcp", CommandCompletionKind::Command)]
    );

    let hidden = CommandCatalog::new(Vec::<String>::new(), ["manual-only", "skills"]).unwrap();
    assert_eq!(
        hidden.complete("/skill", 8),
        [
            completion("/skills", CommandCompletionKind::Command),
            completion("/skills", CommandCompletionKind::Skill),
        ]
    );
    assert_eq!(
        hidden.parse("/skill:skills").unwrap(),
        SlashCommand::Skill("/skill:skills".parse().unwrap())
    );
}

#[test]
fn skill_completions_carry_normalized_display_only_descriptions() {
    let catalog = CommandCatalog::new(Vec::<String>::new(), ["frontend-design"])
        .unwrap()
        .with_skill_descriptions([
            (
                "frontend-design",
                "  Create distinctive,\nproduction-grade\tinterfaces.\u{1b}  ",
            ),
            ("unknown", "must be ignored"),
        ]);

    for query in ["/front", "$front"] {
        let candidates = catalog.complete(query, 8);
        assert_eq!(candidates.len(), 1);
        assert_eq!(
            candidates[0].description(),
            Some("Create distinctive, production-grade interfaces.")
        );
        assert_eq!(
            catalog.completion_draft(&candidates[0]),
            "$frontend-design "
        );
    }
}

#[test]
fn selected_skill_completions_create_mentions_and_validate_arguments() {
    let catalog = CommandCatalog::new(Vec::<String>::new(), ["rust-check"]).unwrap();
    assert_eq!(
        catalog.completion_draft(&CommandCompletionItem::new(
            "/rust-check",
            CommandCompletionKind::Skill,
        )),
        "$rust-check "
    );
    assert_eq!(
        catalog.completion_draft(&CommandCompletionItem::new(
            "$rust-check",
            CommandCompletionKind::Skill,
        )),
        "$rust-check "
    );
    assert_eq!(
        catalog.completion_draft(&CommandCompletionItem::new(
            "/reasoning",
            CommandCompletionKind::Command,
        )),
        "/reasoning"
    );
    let mention = catalog
        .parse_skill_mention("$rust-check --all")
        .unwrap()
        .expect("registered skill mention must parse");
    assert_eq!(mention.to_string(), "/skill:rust-check --all");
    assert_eq!(
        catalog.skill_mention_token("$rust-check --all"),
        Some("$rust-check".to_owned())
    );
    assert!(
        catalog
            .parse_skill_mention("$unknown --all")
            .unwrap()
            .is_none()
    );
    let oversized = format!("$rust-check {}", "x".repeat(16 * 1024 + 1));
    assert!(catalog.parse_skill_mention(&oversized).is_err());
    assert_eq!(
        catalog.skill_mention_token(&oversized),
        Some("$rust-check".to_owned())
    );
}

fn completion(value: &str, kind: CommandCompletionKind) -> CommandCompletionItem {
    CommandCompletionItem::new(value, kind)
}

#[test]
fn skills_builtin_reserves_prompt_template_name_but_not_skill_id() {
    assert!(CommandCatalog::new(["skills"], Vec::<String>::new()).is_err());
    assert!(CommandCatalog::new(Vec::<String>::new(), ["skills"]).is_ok());
}

#[test]
fn image_commands_preserve_paths_and_reject_unbounded_arguments() {
    let catalog = CommandCatalog::new(Vec::<String>::new(), Vec::<String>::new()).unwrap();

    assert_eq!(
        catalog.parse("/image fixtures/my image.png").unwrap(),
        SlashCommand::Image("fixtures/my image.png".to_owned())
    );
    assert_eq!(
        catalog.parse("/image remove 4").unwrap(),
        SlashCommand::ImageRemove(4)
    );
    assert_eq!(
        catalog.parse("/image clear").unwrap(),
        SlashCommand::ImageClear
    );

    for input in [
        "/image",
        "/image remove",
        "/image remove 0",
        "/image remove 5",
        "/image remove 1 extra",
        "/image clear extra",
        "/image bad\npath.png",
    ] {
        assert!(catalog.parse(input).is_err(), "accepted {input:?}");
    }
    assert!(
        catalog
            .parse(&format!("/image {}", "x".repeat(4097)))
            .is_err()
    );
}
