use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Color;
use tea_cli::tui::{
    Action, CommandCompletion, CommandCompletionItem, CommandCompletionKind, ComposerAttachment,
    Overlay, Renderer, Theme, TuiState, reduce,
};
use tea_protocol::{AgentEvent, EventDelta, ReasoningEffort, RunStatus};
use unicode_width::UnicodeWidthStr as _;

use crate::common;

#[tokio::test(flavor = "current_thread")]
async fn render_snapshots_are_deterministic_width_safe_and_control_free() {
    let snapshot = common::pending_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());
    let _ = reduce(
        &mut state,
        Action::SetEditor(
            "修复 emoji 👩🏽‍💻 and a-very-long-unbroken-token-0123456789\n```rust\nfn incomplete("
                .to_owned(),
        ),
    );
    let _ = reduce(
        &mut state,
        Action::QueueSteering("检查 CJK 宽度".to_owned()),
    );
    let _ = reduce(
        &mut state,
        Action::Notify("control \u{1b}[31m must be visible, not active".to_owned()),
    );

    let theme = Theme::default();
    let mut renderer = Renderer::new();
    for width in [20_u16, 40, 80, 160] {
        let first = renderer.lines(&state, width, &theme);
        let second = renderer.lines(&state, width, &theme);
        assert_eq!(first, second, "width {width} must be deterministic");
        assert!(!first.is_empty());
        for line in &first {
            assert!(
                line.text().width() <= usize::from(width),
                "line exceeded {width} cells: {:?}",
                line.text()
            );
            assert!(!line.text().contains('\u{1b}'));
        }
        let snapshot_text = first
            .iter()
            .map(tea_cli::tui::RenderedLine::text)
            .collect::<Vec<_>>()
            .join("\n");
        let compact = snapshot_text
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect::<String>();
        assert!(compact.contains("approval"));
        assert!(compact.contains("write_text_file"));
        assert!(compact.contains("👩🏽‍💻") || width < 8);
        assert!(snapshot_text.contains("must be visible, not active") || width < 160);
    }
}

#[tokio::test(flavor = "current_thread")]
async fn hosted_search_sources_and_lifecycle_remain_width_safe_without_opaque_state() {
    let snapshot = common::hosted_search_snapshot().await;
    let state = TuiState::from_snapshot(&snapshot, common::startup());
    let theme = Theme::default();
    let mut renderer = Renderer::new();

    for width in [20_u16, 40, 80] {
        let first = renderer.lines(&state, width, &theme);
        let second = renderer.lines(&state, width, &theme);
        assert_eq!(first, second, "hosted projection changed at width {width}");
        assert!(
            first
                .iter()
                .all(|line| line.text().width() <= usize::from(width)),
            "hosted projection overflowed at width {width}: {first:?}"
        );
        let frame = first
            .iter()
            .map(tea_cli::tui::RenderedLine::text)
            .collect::<Vec<_>>()
            .join("\n");
        let compact = frame
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect::<String>();
        assert!(compact.contains("Searchedweb"), "width {width}: {frame:?}");
        assert!(compact.contains("Sources"), "width {width}: {frame:?}");
        assert!(
            compact.contains("Hostedsearcharchitecture"),
            "width {width}: {frame:?}"
        );
        assert!(
            compact.contains("Providersearchreference"),
            "width {width}: {frame:?}"
        );
        assert!(!frame.contains("CONTINUATION_MUST_NOT_RENDER"));
        assert!(!frame.contains("CITATION_CONTINUATION_MUST_NOT_RENDER"));
        assert!(!frame.contains("https://"));
    }
}

#[tokio::test(flavor = "current_thread")]
async fn persisted_web_fetch_metadata_and_body_remain_width_safe_and_control_free() {
    let snapshot = common::web_fetch_snapshot().await;
    let state = TuiState::from_snapshot(&snapshot, common::startup());
    let theme = Theme::default();
    let mut renderer = Renderer::new();

    for width in [20_u16, 40, 80] {
        let first = renderer.lines(&state, width, &theme);
        let second = renderer.lines(&state, width, &theme);
        assert_eq!(
            first, second,
            "web fetch projection changed at width {width}"
        );
        assert!(
            first
                .iter()
                .all(|line| line.text().width() <= usize::from(width)),
            "web fetch projection overflowed at width {width}: {first:?}"
        );
        let frame = first
            .iter()
            .map(tea_cli::tui::RenderedLine::text)
            .collect::<Vec<_>>()
            .join("\n");
        let compact = frame
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect::<String>();
        assert!(compact.contains("Fetched"), "width {width}: {frame:?}");
        assert!(compact.contains("web_fetch"), "width {width}: {frame:?}");
        assert!(
            compact.contains("example.com/final"),
            "width {width}: {frame:?}"
        );
        assert!(compact.contains("text/html"), "width {width}: {frame:?}");
        assert!(
            compact.contains("Truncated:bodycharacters"),
            "width {width}: {frame:?}"
        );
        assert!(!frame.contains('\u{1b}'));
        assert!(!frame.contains("MODEL_RESULT_MUST_NOT_RENDER"));
        assert!(!frame.contains("continuation"));
    }
}

#[tokio::test(flavor = "current_thread")]
async fn persisted_code_change_reflows_with_line_numbers_and_safe_diff_markers() {
    let snapshot = common::diff_snapshot().await;
    let state = TuiState::from_snapshot(&snapshot, common::startup());
    let theme = Theme::default();
    let mut renderer = Renderer::new();
    for width in [16_u16, 32, 80] {
        let lines = renderer.lines(&state, width, &theme);
        assert!(
            lines
                .iter()
                .all(|line| line.text().width() <= usize::from(width))
        );
        assert!(lines.iter().all(|line| !line.text().contains('\u{1b}')));
        let output = lines
            .iter()
            .map(tea_cli::tui::RenderedLine::text)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(output.contains("Updated") || width < 8);
        assert!(output.contains("@@") || width < 8);
        assert!(output.contains('+') || width < 8);
        assert!(output.contains('-') || width < 8);
    }

    let lines = renderer.lines(&state, 80, &theme);
    let header = lines
        .iter()
        .find(|line| line.text().contains("Updated"))
        .expect("diff action header must be visible");
    assert!(
        header.text().starts_with(" • Updated"),
        "diff action header must align with other tool status rows"
    );
    assert!(
        lines
            .iter()
            .any(|line| line.text().starts_with("@@ -1,3 +1,3 @@")),
        "hunk header must start at the terminal edge"
    );

    let context = lines
        .iter()
        .find(|line| line.text().contains("pub fn answer"))
        .expect("context line must be visible");
    let deletion = lines
        .iter()
        .find(|line| line.text().starts_with("2 -"))
        .expect("deletion must use its old line number");
    let addition = lines
        .iter()
        .find(|line| line.text().starts_with("2 +"))
        .expect("addition must use its new line number");

    assert!(context.text().starts_with("1  pub fn answer"));
    assert_eq!(deletion.text().width(), 80);
    assert_eq!(addition.text().width(), 80);
    assert_eq!(deletion.style().bg, Some(Color::Rgb(74, 34, 29)));
    assert_eq!(addition.style().bg, Some(Color::Rgb(33, 58, 43)));
    assert!(
        [&context, &deletion, &addition]
            .iter()
            .all(|line| !line.text().contains("├ ") && !line.text().contains("└ "))
    );

    let area = Rect::new(0, 0, 80, 40);
    let mut buffer = Buffer::empty(area);
    renderer.render(&state, area, &mut buffer, &theme);
    let rows = buffer
        .content
        .chunks(usize::from(area.width))
        .collect::<Vec<_>>();
    for (prefix, background) in [
        ("2 -", Color::Rgb(74, 34, 29)),
        ("2 +", Color::Rgb(33, 58, 43)),
    ] {
        let row = rows
            .iter()
            .find(|row| {
                row.iter()
                    .map(ratatui::buffer::Cell::symbol)
                    .collect::<String>()
                    .starts_with(prefix)
            })
            .expect("styled diff row must reach the terminal buffer");
        assert!(
            row.iter().all(|cell| cell.style().bg == Some(background)),
            "the diff background must fill the complete terminal row"
        );
    }
}

#[tokio::test(flavor = "current_thread")]
async fn image_attachments_have_compact_width_safe_composer_snapshots() {
    let snapshot = common::archive_snapshot().await;
    let attachments = [
        ComposerAttachment::inline("image/png", "iVBORw0KGgo=", "first.png").unwrap(),
        ComposerAttachment::inline("image/jpeg", "/9j/", "b.jpg").unwrap(),
    ];
    let theme = Theme::default();

    for attachment_count in 1..=attachments.len() {
        let mut state = TuiState::from_snapshot(&snapshot, common::startup());
        for attachment in attachments.iter().take(attachment_count).cloned() {
            let _ = reduce(&mut state, Action::AddAttachment(attachment));
        }
        let mut renderer = Renderer::new();
        for editor in ["", "first line\nsecond line"] {
            let _ = reduce(&mut state, Action::SetEditor(editor.to_owned()));
            for width in [20_u16, 40, 80, 160] {
                let lines = renderer.lines(&state, width, &theme);
                let attachment_rows = lines
                    .iter()
                    .map(tea_cli::tui::RenderedLine::text)
                    .filter(|line| line.contains("image/png") || line.contains("image/jpeg"))
                    .map(str::trim_end)
                    .collect::<Vec<_>>();
                let expected = if width == 20 {
                    vec!["1. image/png 8 B", "2. image/jpeg 3 B"]
                } else {
                    vec![
                        "1. first.png · image/png · 8 B",
                        "2. b.jpg · image/jpeg · 3 B",
                    ]
                };
                assert_eq!(
                    attachment_rows,
                    expected[..attachment_count],
                    "attachments {attachment_count}, editor {editor:?}, width {width}"
                );
                assert!(
                    lines
                        .iter()
                        .all(|line| line.text().width() <= usize::from(width)),
                    "attachments {attachment_count}, editor {editor:?}, width {width}"
                );
                let (column, row) = renderer
                    .cursor_position(&state, width, 24, editor.len())
                    .expect("composer cursor remains visible");
                assert!(column < width);
                assert!(row < 24);
            }
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn persisted_images_render_metadata_without_payloads_or_references() {
    let snapshot = common::image_snapshot().await;
    let state = TuiState::from_snapshot(&snapshot, common::startup());
    let theme = Theme::default();
    let mut renderer = Renderer::new();

    for width in [20_u16, 40, 80, 160] {
        let first = renderer.lines(&state, width, &theme);
        let second = renderer.lines(&state, width, &theme);
        assert_eq!(first, second, "cache replay at width {width}");
        let frame = first
            .iter()
            .map(tea_cli::tui::RenderedLine::text)
            .collect::<Vec<_>>()
            .join("\n");
        let compact = frame
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect::<String>();
        assert!(
            compact.contains("[imageimage/png·8B]"),
            "width {width}: {frame:?}"
        );
        assert!(compact.contains("[imageimage/jpeg]"));
        assert!(!frame.contains("iVBORw0KGgo="));
        assert!(!frame.contains("private:artifact-image"));
    }
}

#[tokio::test(flavor = "current_thread")]
async fn streaming_resize_and_collapse_invalidate_cached_lines() {
    let snapshot = common::pending_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());
    let mut renderer = Renderer::new();
    let theme = Theme::default();

    let initial = renderer.lines(&state, 40, &theme);
    let next = state.event_cursor().get() + 1;
    let _ = reduce(
        &mut state,
        Action::Event(Box::new(common::event(next, 80, AgentEvent::RunStarted {}))),
    );
    let _ = reduce(
        &mut state,
        Action::Event(Box::new(common::event(
            next + 1,
            81,
            AgentEvent::MessageDelta {
                message_id: common::message_id(),
                content_index: 0,
                delta: EventDelta::ThinkingDelta {
                    text: "incomplete **reasoning".to_owned(),
                },
            },
        ))),
    );
    let _ = reduce(&mut state, Action::SetEditor("new content".to_owned()));
    let changed = renderer.lines(&state, 40, &theme);
    assert_ne!(initial, changed);
    let narrow = renderer.lines(&state, 20, &theme);
    assert_ne!(changed, narrow);

    let _ = reduce(&mut state, Action::ToggleThinking);
    let collapsed = renderer.lines(&state, 20, &theme);
    assert_ne!(narrow, collapsed);
    assert!(
        collapsed
            .iter()
            .flat_map(|line| line.text().chars())
            .filter(|character| !character.is_whitespace())
            .collect::<String>()
            .contains("collapsed")
    );
    assert_ne!(state.generation(), 0);
    assert_eq!(collapsed, renderer.lines(&state, 20, &theme));
    let wide_collapsed = renderer.lines(&state, 40, &theme);
    assert_ne!(collapsed, wide_collapsed);
    assert!(wide_collapsed.iter().all(|line| line.text().width() <= 40));
    assert_eq!(wide_collapsed, renderer.lines(&state, 40, &theme));

    let tool_id = *state.tools().keys().next().unwrap();
    let _ = reduce(&mut state, Action::ToggleTool(tool_id));
    let tool_collapsed = renderer.lines(&state, 20, &theme);
    assert_eq!(
        collapsed, tool_collapsed,
        "pending approval tool details stay redacted in both collapse states"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn tool_metadata_is_retained_but_only_rendered_after_explicit_disclosure() {
    let snapshot = common::archive_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());
    let tool_id = *state.tools().keys().next().expect("fixture has a tool");
    let theme = Theme::default();
    let mut renderer = Renderer::new();

    let compact = renderer
        .lines(&state, 80, &theme)
        .iter()
        .map(tea_cli::tui::RenderedLine::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!compact.contains("arguments {"), "compact: {compact:?}");

    let _ = reduce(&mut state, Action::ToggleTool(tool_id));
    let expanded = renderer
        .lines(&state, 80, &theme)
        .iter()
        .map(tea_cli::tui::RenderedLine::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        expanded.contains("└ arguments {") || expanded.contains("├ arguments {"),
        "expanded: {expanded:?}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn rich_markdown_streams_wrap_safely_at_narrow_width() {
    let snapshot = common::archive_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());
    let next = state.event_cursor().get() + 1;
    let _ = reduce(
        &mut state,
        Action::Event(Box::new(common::event(next, 90, AgentEvent::RunStarted {}))),
    );
    let _ = reduce(
        &mut state,
        Action::Event(Box::new(common::event(
            next + 1,
            91,
            AgentEvent::MessageDelta {
                message_id: common::message_id(),
                content_index: 0,
                delta: EventDelta::TextDelta {
                    text: "# Result\n\n- bullet one\n  - nested bullet\n\n3. first\n4. second\n   1. nested\n\n> quoted\n\nRead **Tea** *docs* at [the guide](https://example.test/tea), then run `tea-cli`; ~~ignore tea-old~~.\n\n```rust\nlet value = 1;\n```\n\u{1b}[31mnot a terminal command".to_owned(),
                },
            },
        ))),
    );

    let theme = Theme::default();
    let mut renderer = Renderer::new();
    for width in [20_u16, 80] {
        let lines = renderer.lines(&state, width, &theme);
        let overlong = lines
            .iter()
            .map(tea_cli::tui::RenderedLine::text)
            .filter(|line| line.width() > usize::from(width))
            .collect::<Vec<_>>();
        if width == 20 {
            assert_eq!(
                overlong
                    .iter()
                    .map(|line| line.trim_start())
                    .collect::<Vec<_>>(),
                ["(https://example.test/tea),"]
            );
        } else {
            assert!(overlong.is_empty(), "width {width}: {overlong:?}");
        }
        let frame = lines
            .iter()
            .map(tea_cli::tui::RenderedLine::text)
            .collect::<Vec<_>>()
            .join("\n");
        let compact = frame
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect::<String>();
        assert!(compact.contains("3.first"), "width {width}: {frame:?}");
        assert!(compact.contains("4.second"), "width {width}: {frame:?}");
        assert!(compact.contains("1.nested"), "width {width}: {frame:?}");
        assert!(compact.contains("-bulletone"), "width {width}: {frame:?}");
        assert!(
            compact.contains("-nestedbullet"),
            "width {width}: {frame:?}"
        );
        assert!(compact.contains("Result"), "width {width}: {frame:?}");
        assert!(compact.contains("quoted"), "width {width}: {frame:?}");
        assert!(frame.contains("# Result"), "width {width}: {frame:?}");
        assert!(frame.contains("> quoted"), "width {width}: {frame:?}");
        assert!(
            compact.contains(
                "ReadTeadocsattheguide(https://example.test/tea),thenruntea-cli;ignoretea-old."
            ),
            "width {width}: {frame:?}"
        );
        assert!(!frame.contains('`'), "inline code marker leaked: {frame:?}");
        assert!(frame.contains("let value = 1;"), "width {width}: {frame:?}");
        for leaked_label in ["you:", "assistant:", "heading:", "quote:", "code:", "rust:"] {
            assert!(
                !frame.contains(leaked_label),
                "Markdown metadata {leaked_label:?} leaked at width {width}: {frame:?}"
            );
        }
        assert!(!frame.contains('\u{1b}'));
        assert!(
            compact.contains("␛[31mnotaterminalcommand"),
            "width {width}: {frame:?}"
        );
        if width == 80 {
            assert!(
                frame.contains("    - nested bullet"),
                "nested unordered list must use Codex alignment: {frame:?}"
            );
            assert!(
                frame.contains("    1. nested"),
                "nested ordered list must use Codex alignment: {frame:?}"
            );
            assert!(
                frame.contains("• Ran write_text_file /workspace/notes.txt"),
                "operational actions must remain visible: {frame:?}"
            );
            assert!(
                frame.contains("• Returned write_text_file\n   └ wrote notes"),
                "tool results must use the Codex-aligned tree grammar: {frame:?}"
            );
        }
    }
}

#[tokio::test(flavor = "current_thread")]
async fn streaming_table_reflows_between_record_and_grid_layouts() {
    let snapshot = common::archive_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());
    let next = state.event_cursor().get() + 1;
    let _ = reduce(
        &mut state,
        Action::Event(Box::new(common::event(next, 92, AgentEvent::RunStarted {}))),
    );
    let _ = reduce(
        &mut state,
        Action::Event(Box::new(common::event(
            next + 1,
            93,
            AgentEvent::MessageDelta {
                message_id: common::message_id(),
                content_index: 0,
                delta: EventDelta::TextDelta {
                    text: "| Name | Status | Extra | More |\n|---|---|---|---|\n| Tea | Ready | Stable | Safe |".to_owned(),
                },
            },
        ))),
    );

    let theme = Theme::default();
    let mut renderer = Renderer::new();
    let narrow = renderer.lines(&state, 20, &theme);
    let wide = renderer.lines(&state, 80, &theme);
    let narrow_text = narrow
        .iter()
        .map(tea_cli::tui::RenderedLine::text)
        .collect::<Vec<_>>()
        .join("\n");
    let wide_text = wide
        .iter()
        .map(tea_cli::tui::RenderedLine::text)
        .collect::<Vec<_>>()
        .join("\n");

    for expected in ["Name", "Tea", "Status", "Ready"] {
        assert!(narrow_text.contains(expected), "narrow: {narrow_text:?}");
        assert!(wide_text.contains(expected), "wide: {wide_text:?}");
    }
    assert!(narrow.iter().all(|line| line.text().width() <= 20));
    assert!(wide.iter().all(|line| line.text().width() <= 80));
    assert!(!narrow_text.contains('━'));
    assert!(wide_text.contains('━'));
}

#[tokio::test(flavor = "current_thread")]
async fn durable_and_streaming_assistant_messages_share_one_representation() {
    let snapshot = common::archive_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());
    let theme = Theme::default();
    let mut renderer = Renderer::new();
    let durable = renderer.lines(&state, 80, &theme);
    let durable_line = durable
        .iter()
        .find(|line| line.text().trim_start() == "I will write the notes.")
        .expect("fixture assistant message must be visible")
        .clone();

    let next = state.event_cursor().get() + 1;
    let _ = reduce(
        &mut state,
        Action::Event(Box::new(common::event(next, 92, AgentEvent::RunStarted {}))),
    );
    let _ = reduce(
        &mut state,
        Action::Event(Box::new(common::event(
            next + 1,
            93,
            AgentEvent::MessageDelta {
                message_id: common::message_id(),
                content_index: 0,
                delta: EventDelta::TextDelta {
                    text: "I will write the notes.".to_owned(),
                },
            },
        ))),
    );

    let matching = renderer
        .lines(&state, 80, &theme)
        .into_iter()
        .filter(|line| line.text() == durable_line.text())
        .collect::<Vec<_>>();
    assert_eq!(matching.len(), 2);
    assert!(
        matching
            .iter()
            .all(|line| line.style() == durable_line.style())
    );
    assert!(
        matching
            .iter()
            .all(|line| !line.text().contains("assistant:"))
    );
}

#[tokio::test(flavor = "current_thread")]
async fn unicode_editor_cursor_is_visible_and_cell_bounded() {
    let snapshot = common::archive_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());
    let editor = "中🙂\ntext";
    let _ = reduce(&mut state, Action::SetEditor(editor.to_owned()));
    let mut renderer = Renderer::new();
    let (column, row) = renderer
        .cursor_position(&state, 20, 8, "中🙂".len())
        .unwrap();
    assert!(column < 20);
    assert!(row < 8);
}

#[tokio::test(flavor = "current_thread")]
async fn composer_surface_reaches_terminal_edges_at_narrow_and_wide_sizes() {
    let snapshot = common::archive_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());
    let _ = reduce(&mut state, Action::SetEditor("draft".to_owned()));
    let theme = Theme::default();
    let mut renderer = Renderer::new();

    for width in [20_u16, 80] {
        let area = Rect::new(0, 0, width, 12);
        let mut buffer = Buffer::empty(area);
        renderer.render(&state, area, &mut buffer, &theme);
        let rows = buffer
            .content
            .chunks(usize::from(width))
            .collect::<Vec<_>>();
        let input = rows
            .iter()
            .find(|row| {
                row.iter()
                    .map(ratatui::buffer::Cell::symbol)
                    .collect::<String>()
                    .contains("› draft")
            })
            .expect("editor line must be rendered");
        assert_eq!(input.first().unwrap().symbol(), "›");
        assert_eq!(input.last().unwrap().symbol(), " ");
        assert!(
            input
                .iter()
                .all(|cell| cell.style().bg == theme.composer.bg),
            "the edge-to-edge editor surface must keep a consistent background"
        );
        let text = input
            .iter()
            .map(ratatui::buffer::Cell::symbol)
            .collect::<String>();
        assert!(
            !text.contains('│') && !text.contains('╭') && !text.contains('╰'),
            "the editor surface must not reintroduce a box frame"
        );
    }
}

#[tokio::test(flavor = "current_thread")]
async fn selected_skill_mention_is_colored_inside_the_composer_surface() {
    let snapshot = common::archive_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());
    let _ = reduce(
        &mut state,
        Action::SetComposer {
            text: "$review help me".to_owned(),
            skill_mention: Some("$review".to_owned()),
        },
    );
    let theme = Theme::default();
    let area = Rect::new(0, 0, 40, 12);
    let mut buffer = Buffer::empty(area);
    Renderer::new().render(&state, area, &mut buffer, &theme);
    let row = buffer
        .content
        .chunks(usize::from(area.width))
        .find(|row| {
            row.iter()
                .map(ratatui::buffer::Cell::symbol)
                .collect::<String>()
                .contains("$review help me")
        })
        .expect("selected skill draft must be rendered");
    let start = row
        .windows("$review".len())
        .position(|cells| {
            cells
                .iter()
                .map(ratatui::buffer::Cell::symbol)
                .collect::<String>()
                == "$review"
        })
        .expect("skill mention cells must be contiguous");

    assert!(
        row[start..start + "$review".len()]
            .iter()
            .all(|cell| cell.style().fg == theme.editor.fg),
        "mention styles: {:?}; expected {:?}",
        row[start..start + "$review".len()]
            .iter()
            .map(|cell| (cell.symbol(), cell.style()))
            .collect::<Vec<_>>(),
        theme.editor
    );
    assert_eq!(row[start + "$review".len()].style().fg, theme.composer.fg);
    assert!(row.iter().all(|cell| cell.style().bg == theme.composer.bg));
}

#[tokio::test(flavor = "current_thread")]
async fn idle_composer_uses_a_muted_prompt_surface_without_a_banner_or_divider() {
    let snapshot = common::archive_snapshot().await;
    let state = TuiState::from_snapshot(&snapshot, common::startup());
    let area = Rect::new(0, 0, 80, 16);
    let mut buffer = Buffer::empty(area);
    let mut renderer = Renderer::new();
    let theme = Theme::default();
    renderer.render(&state, area, &mut buffer, &theme);

    let rows = buffer
        .content
        .chunks(usize::from(area.width))
        .map(|row| {
            row.iter()
                .map(ratatui::buffer::Cell::symbol)
                .collect::<String>()
        })
        .collect::<Vec<_>>();
    let frame = rows.join("\n");
    assert!(frame.contains("› Ask Tea to do anything"), "rows: {rows:?}");
    assert!(
        frame.contains("default · default · workspace/demo"),
        "rows: {rows:?}"
    );
    assert!(!frame.contains("? for shortcuts"), "rows: {rows:?}");
    assert!(!frame.contains("tokens"), "rows: {rows:?}");
    assert!(!frame.contains("tea: workspace/demo"), "rows: {rows:?}");
    let placeholder_row = rows
        .iter()
        .position(|row| row.contains("› Ask Tea to do anything"))
        .unwrap();
    let footer_row = rows
        .iter()
        .position(|row| row.contains("default · default · workspace/demo"))
        .unwrap();
    let placeholder_text_column = rows[placeholder_row].split_once("Ask").unwrap().0.width();
    let footer_text_column = rows[footer_row].split_once("default").unwrap().0.width();
    assert_eq!(
        placeholder_text_column, footer_text_column,
        "footer metadata must align with the composer text column"
    );
    assert_eq!(footer_row, placeholder_row + 2, "rows: {rows:?}");
    assert!(rows[footer_row - 1].trim().is_empty(), "rows: {rows:?}");
    assert!(
        buffer.content
            [usize::from(area.width) * (footer_row - 1)..usize::from(area.width) * footer_row]
            .iter()
            .all(|cell| cell.style().bg == theme.composer.bg),
        "composer bottom padding must remain intact without an extra footer gap"
    );
    let placeholder_column = rows[placeholder_row].find('›').unwrap();
    let placeholder = &buffer[(
        u16::try_from(placeholder_column).unwrap(),
        u16::try_from(placeholder_row).unwrap(),
    )];
    assert_eq!(Some(placeholder.fg), theme.footer.fg);
    assert_eq!(Some(placeholder.bg), theme.composer.bg);
    assert_ne!(Some(placeholder.fg), theme.composer.fg);
    assert!(
        !rows
            .iter()
            .any(|row| row.chars().all(|character| character == '─')),
        "the composer must not be separated by a heavy divider: {rows:?}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn footer_uses_durable_and_pending_reasoning_projection() {
    let snapshot = common::reasoning_snapshot("high").await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());
    let theme = Theme::default();
    let mut renderer = Renderer::new();

    for width in [24_u16, 80] {
        let durable = renderer.lines(&state, width, &theme);
        assert!(
            durable
                .iter()
                .all(|line| line.text().width() <= usize::from(width))
        );
        let durable = durable
            .iter()
            .map(tea_cli::tui::RenderedLine::text)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(durable.contains("fake/model · high"), "{width}: {durable}");
    }

    let _ = reduce(
        &mut state,
        Action::SetPendingReasoningEffort(Some(ReasoningEffort::Maximum)),
    );
    for width in [24_u16, 80] {
        let pending = renderer.lines(&state, width, &theme);
        assert!(
            pending
                .iter()
                .all(|line| line.text().width() <= usize::from(width))
        );
        let pending = pending
            .iter()
            .map(tea_cli::tui::RenderedLine::text)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(pending.contains("fake/model · max"), "{width}: {pending}");
        assert!(!pending.contains("fake/model · high"), "{width}: {pending}");
    }
}

#[tokio::test(flavor = "current_thread")]
async fn modal_surfaces_render_above_the_composer_with_approval_priority() {
    let snapshot = common::archive_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());
    let _ = reduce(&mut state, Action::SetEditor("preserved draft".to_owned()));
    let _ = reduce(
        &mut state,
        Action::SetOverlay(Some(Overlay::CommandCompletion(CommandCompletion::new([
            CommandCompletionItem::new("/help", CommandCompletionKind::Command),
            CommandCompletionItem::new("/review", CommandCompletionKind::Prompt),
            CommandCompletionItem::new("/frontend-design", CommandCompletionKind::Skill),
        ])))),
    );
    let area = Rect::new(0, 0, 80, 24);
    let theme = Theme::default();
    let mut renderer = Renderer::new();
    let mut buffer = Buffer::empty(area);
    renderer.render(&state, area, &mut buffer, &theme);
    let rows = buffer
        .content
        .chunks(usize::from(area.width))
        .map(|row| {
            row.iter()
                .map(ratatui::buffer::Cell::symbol)
                .collect::<String>()
        })
        .collect::<Vec<_>>();
    let modal_row = rows
        .iter()
        .position(|row| row.contains("completions: select item"))
        .expect("command completion must be visible");
    let composer_row = rows
        .iter()
        .position(|row| row.contains("› preserved draft"))
        .expect("draft must remain visible");
    assert!(modal_row < composer_row, "rows: {rows:?}");
    let completion_rows = rows
        .iter()
        .filter(|row| {
            row.contains("[command]") || row.contains("[prompt]") || row.contains("[skill]")
        })
        .collect::<Vec<_>>();
    assert_eq!(completion_rows.len(), 3, "completion rows: {rows:?}");
    let type_columns = completion_rows
        .iter()
        .filter_map(|row| row.find('[').map(|index| row[..index].width()))
        .collect::<Vec<_>>();
    assert!(type_columns.windows(2).all(|pair| pair[0] == pair[1]));

    let pending = common::pending_snapshot().await;
    let mut approval_state = TuiState::from_snapshot(&pending, common::startup());
    let _ = reduce(
        &mut approval_state,
        Action::SetOverlay(Some(Overlay::CommandCompletion(CommandCompletion::new([
            "/help".to_owned(),
        ])))),
    );
    let mut approval_buffer = Buffer::empty(area);
    renderer.render(&approval_state, area, &mut approval_buffer, &theme);
    let approval_frame = approval_buffer
        .content
        .chunks(usize::from(area.width))
        .map(|row| {
            row.iter()
                .map(ratatui::buffer::Cell::symbol)
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(approval_frame.contains("approval required"));
    assert!(!approval_frame.contains("completions: select item"));
    assert!(approval_frame.contains("› Ask Tea to do anything"));
}

#[tokio::test(flavor = "current_thread")]
async fn skill_completion_names_align_with_descriptions_and_stay_single_line() {
    let snapshot = common::archive_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());
    let _ = reduce(
        &mut state,
        Action::SetOverlay(Some(Overlay::CommandCompletion(CommandCompletion::new([
            CommandCompletionItem::new("$frontend-design", CommandCompletionKind::Skill)
                .with_description("Create distinctive,\nproduction-grade frontend interfaces."),
            CommandCompletionItem::new("$design-taste-frontend", CommandCompletionKind::Skill)
                .with_description("Anti-slop frontend skill for landing pages and redesigns."),
        ])))),
    );
    let theme = Theme::default();
    let mut renderer = Renderer::new();
    let area = Rect::new(0, 0, 80, 16);
    let mut buffer = Buffer::empty(area);
    renderer.render(&state, area, &mut buffer, &theme);
    let rows = buffer
        .content
        .chunks(usize::from(area.width))
        .map(|row| {
            row.iter()
                .map(ratatui::buffer::Cell::symbol)
                .collect::<String>()
        })
        .collect::<Vec<_>>();
    let frame = rows.join("\n");
    assert!(frame.contains("skills: select skill"));
    assert!(frame.contains("[skill] Create distinctive, production-grade"));
    assert!(frame.contains("[skill] Anti-slop frontend skill"));
    let type_columns = rows
        .iter()
        .filter_map(|row| row.find("[skill]").map(|index| row[..index].width()))
        .collect::<Vec<_>>();
    assert_eq!(type_columns.len(), 2, "skill rows: {rows:?}");
    assert_eq!(type_columns[0], type_columns[1]);

    let narrow_area = Rect::new(0, 0, 28, 16);
    let mut narrow_buffer = Buffer::empty(narrow_area);
    renderer.render(&state, narrow_area, &mut narrow_buffer, &theme);
    let narrow_rows = narrow_buffer
        .content
        .chunks(usize::from(narrow_area.width))
        .map(|row| {
            row.iter()
                .map(ratatui::buffer::Cell::symbol)
                .collect::<String>()
        })
        .collect::<Vec<_>>();
    let option_rows = narrow_rows
        .iter()
        .enumerate()
        .filter_map(|(index, row)| {
            (row.contains("$frontend") || row.contains("$design")).then_some(index)
        })
        .collect::<Vec<_>>();
    assert_eq!(option_rows.len(), 2, "narrow rows: {narrow_rows:?}");
    assert_eq!(
        option_rows[1],
        option_rows[0] + 1,
        "descriptions must not wrap into extra menu rows: {narrow_rows:?}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn user_messages_use_the_composer_surface_without_editor_chrome() {
    let snapshot = common::archive_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());
    let _ = reduce(
        &mut state,
        Action::ShowPendingUserPrompt("pending 中文内容需要换行".to_owned()),
    );

    let width = 16_u16;
    let theme = Theme::default();
    let lines = Renderer::new().lines(&state, width, &theme);

    for first_content_row in [
        lines
            .iter()
            .position(|line| line.text().starts_with("› Inspect"))
            .expect("historical user input must use the composer prompt prefix"),
        lines
            .iter()
            .position(|line| line.text().starts_with("› pending"))
            .expect("pending user input must use the composer prompt prefix"),
    ] {
        let bottom_padding_row = lines[first_content_row..]
            .iter()
            .position(|line| line.text().trim().is_empty())
            .map(|offset| first_content_row + offset)
            .expect("read-only composer must have bottom padding");
        let surface = &lines[first_content_row - 1..=bottom_padding_row];

        assert!(surface.first().unwrap().text().trim().is_empty());
        assert!(surface.last().unwrap().text().trim().is_empty());
        assert!(
            surface
                .iter()
                .all(|line| line.style().bg == theme.composer.bg
                    && line.text().width() == usize::from(width)),
            "the read-only composer must fill every row with the composer background: {surface:?}"
        );
        assert!(
            surface[2..surface.len() - 1]
                .iter()
                .all(|line| line.text().starts_with("  ")),
            "wrapped rows must use the composer continuation prefix: {surface:?}"
        );
        assert!(surface.iter().all(|line| {
            !line.text().contains("Ask Tea to do anything")
                && !line.text().contains("workspace/demo")
        }));
    }

    let composer = &lines[lines.len() - 4..];
    assert!(composer[0].text().trim().is_empty());
    assert!(composer[1].text().starts_with("› Ask Tea"));
    assert!(composer[2].text().trim().is_empty());
    assert_eq!(composer[2].style(), theme.composer);
    assert_eq!(composer[3].style(), theme.footer);
}

#[tokio::test(flavor = "current_thread")]
async fn frame_matrix_keeps_submitted_prompt_composer_and_cursor_visible() {
    let snapshot = common::archive_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());
    let next = state.event_cursor().get() + 1;
    let _ = reduce(
        &mut state,
        Action::Event(Box::new(common::event(
            next,
            490,
            AgentEvent::RunStarted {},
        ))),
    );
    let _ = reduce(
        &mut state,
        Action::ShowPendingUserPrompt("submitted before model output".to_owned()),
    );
    let draft = "next instruction";
    let _ = reduce(&mut state, Action::SetEditor(draft.to_owned()));

    let theme = Theme::default();
    let mut renderer = Renderer::new();
    for area in [
        Rect::new(0, 0, 40, 16),
        Rect::new(0, 0, 80, 24),
        Rect::new(0, 0, 120, 32),
    ] {
        let mut buffer = Buffer::empty(area);
        renderer.render(&state, area, &mut buffer, &theme);
        let frame = buffer
            .content
            .chunks(usize::from(area.width))
            .map(|row| {
                row.iter()
                    .map(ratatui::buffer::Cell::symbol)
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            frame.contains("submitted before model output"),
            "submitted prompts must not wait for model output at {area:?}: {frame:?}"
        );
        assert!(
            !frame.contains("you: submitted before model output"),
            "submitted prompts must render as message content at {area:?}: {frame:?}"
        );
        assert!(
            frame.contains("› next instruction"),
            "the draft must remain available at {area:?}: {frame:?}"
        );
        let (column, row) = renderer
            .cursor_position(&state, area.width, area.height, draft.len())
            .expect("editor cursor must have a visible position");
        assert!(column < area.width, "cursor column at {area:?}");
        assert!(row < area.height, "cursor row at {area:?}");
    }
}

#[tokio::test(flavor = "current_thread")]
async fn activity_stage_and_elapsed_time_survive_wrapping() {
    let snapshot = common::archive_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());
    let next = state.event_cursor().get() + 1;
    let _ = reduce(
        &mut state,
        Action::Event(Box::new(common::event(
            next,
            480,
            AgentEvent::RunStarted {},
        ))),
    );
    for _ in 0..125 {
        let _ = reduce(&mut state, Action::AdvanceRunElapsed(1));
    }
    let theme = Theme::default();
    let mut renderer = Renderer::new();

    for width in [20_u16, 80] {
        let frame_text = renderer
            .lines(&state, width, &theme)
            .iter()
            .map(tea_cli::tui::RenderedLine::text)
            .collect::<Vec<_>>()
            .join("\n");
        let compact = frame_text
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect::<String>();
        assert!(
            compact.contains("Working(2m05s,esctointerrupt)"),
            "width {width}: {frame_text:?}"
        );
        assert!(compact.contains("2m05s"), "width {width}: {frame_text:?}");
    }
}

#[tokio::test(flavor = "current_thread")]
async fn completed_run_uses_the_codex_worked_for_separator_and_keeps_composer_height() {
    let snapshot = common::archive_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());
    let next = state.event_cursor().get() + 1;
    let _ = reduce(
        &mut state,
        Action::Event(Box::new(common::event(
            next,
            481,
            AgentEvent::RunStarted {},
        ))),
    );
    for _ in 0..125 {
        let _ = reduce(&mut state, Action::AdvanceRunElapsed(1));
    }
    let _ = reduce(
        &mut state,
        Action::Event(Box::new(common::event(
            next + 1,
            482,
            AgentEvent::RunFinished {
                status: RunStatus::Completed,
                usage: Some(
                    tea_protocol::Usage::new(
                        tea_protocol::TokenCount::new(1_200).unwrap(),
                        tea_protocol::TokenCount::new(340).unwrap(),
                    )
                    .with_cache_read(tea_protocol::TokenCount::new(200).unwrap()),
                ),
                cost: None,
            },
        ))),
    );
    let _ = reduce(
        &mut state,
        Action::SnapshotLoaded(Box::new(snapshot.clone())),
    );

    let theme = Theme::default();
    let lines = Renderer::new().lines(&state, 40, &theme);
    let worked_row = lines
        .iter()
        .position(|line| line.text().contains("Worked for 2m 05s"))
        .expect("completed run must retain and render elapsed time");
    let input_row = lines
        .iter()
        .position(|line| line.text().contains("Ask Tea to do anything"))
        .expect("composer must remain visible");

    assert!(
        lines[worked_row]
            .text()
            .starts_with("─ Worked for 2m 05s ─")
    );
    assert_eq!(lines[worked_row].text().width(), 40);
    assert!(lines[worked_row - 1].text().is_empty());
    assert_eq!(
        lines[worked_row + 1].text(),
        "  └ 1,340 tokens (+ 200 cached)"
    );
    assert!(lines[worked_row + 2].text().is_empty());
    assert_eq!(input_row, worked_row + 4);
    assert_eq!(lines.len() - (worked_row + 3), 4);

    let _ = reduce(&mut state, Action::StartRunActivity);
    let restarted = Renderer::new()
        .lines(&state, 40, &theme)
        .iter()
        .map(tea_cli::tui::RenderedLine::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(!restarted.contains("Worked for"));
    assert!(restarted.contains("Working (0s, esc to interrupt)"));
}

#[tokio::test(flavor = "current_thread")]
async fn status_surface_stays_visible_during_streams_and_reports_tool_progress() {
    let snapshot = common::archive_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());
    let next = state.event_cursor().get() + 1;
    let _ = reduce(
        &mut state,
        Action::Event(Box::new(common::event(
            next,
            495,
            AgentEvent::RunStarted {},
        ))),
    );
    let theme = Theme::default();
    let mut renderer = Renderer::new();
    assert!(
        renderer
            .lines(&state, 80, &theme)
            .iter()
            .any(|line| line.text().contains("* Working (0s, esc to interrupt)"))
    );

    let _ = reduce(
        &mut state,
        Action::Event(Box::new(common::event(
            next + 1,
            496,
            AgentEvent::MessageDelta {
                message_id: common::message_id(),
                content_index: 0,
                delta: EventDelta::TextDelta {
                    text: "visible response".to_owned(),
                },
            },
        ))),
    );
    let streamed = renderer
        .lines(&state, 80, &theme)
        .iter()
        .map(tea_cli::tui::RenderedLine::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(streamed.contains("visible response"));
    assert!(
        streamed.contains("* Working (0s, esc to interrupt)"),
        "the fixed status row must remain visible during streaming: {streamed:?}"
    );

    let _ = reduce(
        &mut state,
        Action::Event(Box::new(common::event(
            next + 2,
            497,
            AgentEvent::ToolCallRequested {
                tool_call_id: common::tool_call_id(),
                tool_name: "index_workspace".to_owned(),
                arguments: serde_json::json!({"path": "."}),
            },
        ))),
    );
    let _ = reduce(
        &mut state,
        Action::Event(Box::new(common::event(
            next + 3,
            498,
            AgentEvent::ToolExecutionProgress {
                tool_call_id: common::tool_call_id(),
                message: "2 files indexed".to_owned(),
                completed_units: 2,
                total_units: Some(5),
            },
        ))),
    );
    let tool_frame = renderer
        .lines(&state, 80, &theme)
        .iter()
        .map(tea_cli::tui::RenderedLine::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(tool_frame.contains("* Running index_workspace"));
    assert!(tool_frame.contains("2/5 2 files indexed"));
}

#[tokio::test(flavor = "current_thread")]
async fn reasoning_only_stream_keeps_truthful_status_visible() {
    let snapshot = common::archive_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());
    let next = state.event_cursor().get() + 1;
    let _ = reduce(
        &mut state,
        Action::Event(Box::new(common::event(
            next,
            499,
            AgentEvent::RunStarted {},
        ))),
    );
    let _ = reduce(
        &mut state,
        Action::Event(Box::new(common::event(
            next + 1,
            500,
            AgentEvent::MessageDelta {
                message_id: common::message_id(),
                content_index: 0,
                delta: EventDelta::ThinkingDelta {
                    text: "checking workspace context".to_owned(),
                },
            },
        ))),
    );

    let theme = Theme::default();
    let frame = Renderer::new()
        .lines(&state, 80, &theme)
        .iter()
        .map(tea_cli::tui::RenderedLine::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(frame.contains("checking workspace context"));
    assert!(
        frame.contains("* Working (0s, esc to interrupt)"),
        "reasoning is distinct from visible assistant response text: {frame:?}"
    );
}

#[tokio::test(flavor = "current_thread")]
async fn mcp_health_rows_render_as_bounded_notices() {
    let snapshot = common::archive_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());
    let _ = reduce(
        &mut state,
        Action::SetMcpHealth(vec![
            "docs.search · ready".to_owned(),
            "files.read · disconnected".to_owned(),
        ]),
    );

    let theme = Theme::default();
    let frame = Renderer::new()
        .lines(&state, 40, &theme)
        .iter()
        .map(tea_cli::tui::RenderedLine::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(frame.contains("docs.search · ready"));
    assert!(frame.contains("files.read · disconnected"));
    assert!(frame.contains("•"));
}

#[tokio::test(flavor = "current_thread")]
async fn skill_catalog_rows_render_without_manifest_paths() {
    let snapshot = common::archive_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());
    let manifest_path = "/Users/example/.agents/skills/manual/SKILL.md";
    let _ = reduce(
        &mut state,
        Action::SetSkillCatalog(vec![
            "manual-only [project-agents; explicit-only] - Explicit invocation only".to_owned(),
            "review [user-tea] - Review code safely".to_owned(),
        ]),
    );

    let theme = Theme::default();
    let frame = Renderer::new()
        .lines(&state, 80, &theme)
        .iter()
        .map(tea_cli::tui::RenderedLine::text)
        .collect::<Vec<_>>()
        .join("\n");
    assert!(frame.contains("manual-only [project-agents; explicit-only]"));
    assert!(frame.contains("review [user-tea] - Review code safely"));
    assert!(!frame.contains(manifest_path));
}

#[tokio::test(flavor = "current_thread")]
async fn composer_stays_in_a_reserved_bottom_region_during_streaming() {
    let snapshot = common::archive_snapshot().await;
    let mut state = TuiState::from_snapshot(&snapshot, common::startup());
    let next = state.event_cursor().get() + 1;
    let _ = reduce(
        &mut state,
        Action::Event(Box::new(common::event(
            next,
            501,
            AgentEvent::RunStarted {},
        ))),
    );
    let _ = reduce(
        &mut state,
        Action::Event(Box::new(common::event(
            next + 1,
            502,
            AgentEvent::MessageDelta {
                message_id: common::message_id(),
                content_index: 0,
                delta: EventDelta::TextDelta {
                    text: "active work above the composer".to_owned(),
                },
            },
        ))),
    );
    let _ = reduce(
        &mut state,
        Action::SetEditor("draft stays visible".to_owned()),
    );

    let area = Rect::new(0, 0, 40, 10);
    let mut buffer = Buffer::empty(area);
    let mut renderer = Renderer::new();
    let theme = Theme::default();
    renderer.render(&state, area, &mut buffer, &theme);

    let rows = buffer
        .content
        .chunks(usize::from(area.width))
        .map(|row| {
            row.iter()
                .map(ratatui::buffer::Cell::symbol)
                .collect::<String>()
        })
        .collect::<Vec<_>>();
    let status_row = rows
        .iter()
        .position(|row| row.contains("Working (0s, esc to interrupt)"))
        .expect("working status must remain visible");
    let input_row = rows
        .iter()
        .position(|row| row.contains("› draft stays visible"))
        .expect("composer must remain visible");
    assert!(status_row > 0, "rows: {rows:?}");
    assert!(rows[status_row - 1].trim().is_empty(), "rows: {rows:?}");
    assert!(rows[status_row + 1].trim().is_empty(), "rows: {rows:?}");
    assert_eq!(input_row, status_row + 3, "rows: {rows:?}");
    assert!(
        buffer.content[usize::from(area.width) * (status_row + 2) + 1
            ..usize::from(area.width) * (status_row + 3) - 1]
            .iter()
            .all(|cell| cell.style().bg == theme.composer.bg),
        "the original composer top padding must remain intact: {rows:?}"
    );
    assert!(
        rows.iter()
            .any(|row| row.contains("default · default · workspace/demo")),
        "rows: {rows:?}"
    );
    assert!(
        !rows[4..]
            .iter()
            .any(|row| row.contains('╭') || row.contains('╰') || row.contains('│')),
        "composer must not use a box frame: {rows:?}"
    );
    let input_cells = &buffer.content
        [usize::from(area.width) * input_row..usize::from(area.width) * (input_row + 1)];
    assert!(
        input_cells
            .iter()
            .all(|cell| cell.style().bg == theme.composer.bg),
        "the input row needs a consistent low-contrast surface"
    );
}
