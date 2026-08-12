use tea_cli::tui::{CommandCompletion, CommandCompletionItem, CommandCompletionKind};

#[test]
fn command_completion_is_bounded_and_accepts_only_an_explicit_selection() {
    let mut completion = CommandCompletion::new([
        "/compact".to_owned(),
        "/model".to_owned(),
        "/session".to_owned(),
    ]);

    assert_eq!(
        completion.selected().map(CommandCompletionItem::value),
        Some("/compact")
    );
    completion.move_next();
    assert_eq!(
        completion.selected().map(CommandCompletionItem::value),
        Some("/model")
    );
    completion.move_previous();
    assert_eq!(
        completion.selected().map(CommandCompletionItem::value),
        Some("/compact")
    );
    assert_eq!(
        completion
            .options()
            .iter()
            .map(CommandCompletionItem::value)
            .collect::<Vec<_>>(),
        ["/compact", "/model", "/session"]
    );
}

#[test]
fn command_completion_accepts_composer_facing_skill_options() {
    let completion = CommandCompletion::new([
        "$frontend-design".to_owned(),
        "frontend-design".to_owned(),
        "$design-taste-frontend".to_owned(),
    ]);

    assert_eq!(
        completion
            .options()
            .iter()
            .map(CommandCompletionItem::value)
            .collect::<Vec<_>>(),
        ["$frontend-design", "$design-taste-frontend"]
    );
    assert_eq!(
        completion.selected().map(CommandCompletionItem::value),
        Some("$frontend-design")
    );
    assert!(
        completion
            .options()
            .iter()
            .all(|item| item.kind() == CommandCompletionKind::Skill)
    );
}
