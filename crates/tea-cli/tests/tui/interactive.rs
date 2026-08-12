use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::str::FromStr as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use clap::Parser as _;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use tea_cli::args::{CliArgs, SessionSelection};
use tea_cli::tui::{
    CellNode, Clipboard, FrameSink, InputEvent, MemoryClipboard, Renderer, Selector, SelectorItem,
    SelectorValue, Theme, TuiState, run_with_channels, run_with_channels_with_settings_path,
};
use tea_cli::{BootstrapEnvironment, CliBootstrap};
use tea_coding::CodingAgentService;
use tea_model::{
    ModelCapabilities, ModelDisplayName, ModelEvent, ModelResponseInfo, ModelSpec, ProviderId,
    ReasoningProfile, Utf8Delta,
};
use tea_protocol::{
    CanonicalMessage, ContentBlock, ImageSource, ModelId, ModelRef, ReasoningEffort, TokenCount,
};
use tea_testkit::{ScriptStep, ScriptedModelProvider, ScriptedModelResponse};

fn model_ref(model_id: &str) -> ModelRef {
    ModelRef::new("fake".parse().unwrap(), model_id.parse().unwrap())
}

fn reasoning_model(
    provider_id: &ProviderId,
    model_id: &str,
    default_effort: ReasoningEffort,
    supported_efforts: impl IntoIterator<Item = ReasoningEffort>,
) -> ModelSpec {
    ModelSpec::new(
        ModelId::from_str(model_id).unwrap(),
        provider_id.clone(),
        ModelDisplayName::from_str(model_id).unwrap(),
        TokenCount::new(32_000).unwrap(),
        TokenCount::new(4_000).unwrap(),
        ModelCapabilities::text().with_tools(true),
    )
    .unwrap()
    .with_reasoning_profile(ReasoningProfile::new(default_effort, supported_efforts).unwrap())
}

fn reasoning_service(
    root: &Path,
    models: Vec<ModelSpec>,
    scripts: Vec<ScriptedModelResponse>,
    initial_model: &str,
) -> (
    Arc<ScriptedModelProvider>,
    CodingAgentService,
    SessionSelection,
) {
    let provider = Arc::new(ScriptedModelProvider::new(
        ProviderId::from_str("fake").unwrap(),
        models,
        scripts,
    ));
    let args = CliArgs::try_parse_from([
        "tea".to_owned(),
        "--no-session".to_owned(),
        "--provider".to_owned(),
        "fake".to_owned(),
        "--model".to_owned(),
        initial_model.to_owned(),
        "--trust".to_owned(),
        "ignore".to_owned(),
        "--cwd".to_owned(),
        root.display().to_string(),
        "--config-dir".to_owned(),
        root.join("config").display().to_string(),
        "--state-dir".to_owned(),
        root.join("state").display().to_string(),
        "--data-dir".to_owned(),
        root.join("data").display().to_string(),
    ])
    .unwrap();
    let bootstrap = CliBootstrap::new(BootstrapEnvironment::new(
        root,
        Some(root.to_path_buf()),
        BTreeMap::new(),
    ))
    .with_provider(provider.clone());
    let (service, selection) = bootstrap.build(&args).unwrap();
    (provider, service, selection)
}

#[test]
fn selectors_filter_move_and_accept_typed_values() {
    let mut selector = Selector::new(
        "models",
        [
            SelectorItem::new("Alpha", SelectorValue::Model(model_ref("fake/alpha"))),
            SelectorItem::new("Beta", SelectorValue::Model(model_ref("fake/beta"))),
        ],
    )
    .unwrap();
    selector.set_query("be");
    assert_eq!(selector.visible_items().len(), 1);
    assert_eq!(
        selector.accept(),
        Some(SelectorValue::Model(model_ref("fake/beta")))
    );
    selector.set_query("");
    selector.move_next();
    assert_eq!(selector.selected_label(), Some("Beta"));
    selector.move_previous();
    assert_eq!(selector.selected_label(), Some("Alpha"));

    selector.set_query(&"界".repeat(100));
    assert!(selector.query().len() <= 256);
    assert!(selector.query().is_char_boundary(selector.query().len()));

    let mut reasoning = Selector::new(
        "reasoning effort",
        ReasoningEffort::ALL
            .into_iter()
            .map(|effort| SelectorItem::new(effort.as_str(), SelectorValue::Reasoning(effort))),
    )
    .unwrap();
    reasoning.select_value(&SelectorValue::Reasoning(ReasoningEffort::Maximum));
    assert_eq!(reasoning.selected_label(), Some("max"));
    assert_eq!(
        reasoning.accept(),
        Some(SelectorValue::Reasoning(ReasoningEffort::Maximum))
    );
}

#[test]
fn clipboard_is_an_explicit_injected_boundary() {
    let mut clipboard = MemoryClipboard::default();
    clipboard.copy("bounded text").unwrap();
    assert_eq!(clipboard.contents(), Some("bounded text"));
    assert!(clipboard.copy("contains\0nul").is_err());
    assert_eq!(clipboard.contents(), Some("bounded text"));
}

#[derive(Default)]
struct Frames {
    count: usize,
    renderer: Renderer,
    saw_prompt_before_response: bool,
    saw_activity_before_response: bool,
    saw_running: bool,
    run_stopped: Arc<AtomicBool>,
    inserted_history: Vec<String>,
    history_replacements: usize,
}

impl FrameSink for Frames {
    fn insert_history_cells(&mut self, cells: &[CellNode]) -> std::io::Result<()> {
        self.inserted_history
            .extend(cells.iter().map(CellNode::raw_text));
        Ok(())
    }

    fn replace_history_cells(&mut self, cells: &[CellNode]) -> std::io::Result<()> {
        self.history_replacements += 1;
        self.inserted_history = cells.iter().map(CellNode::raw_text).collect();
        Ok(())
    }

    fn render(&mut self, state: &TuiState, _cursor_byte: usize) -> std::io::Result<()> {
        self.count += 1;
        if state.is_running() {
            self.saw_running = true;
        } else if self.saw_running {
            self.run_stopped.store(true, Ordering::Release);
        }
        let frame = self
            .renderer
            .lines(state, 80, &Theme::default())
            .iter()
            .map(tea_cli::tui::RenderedLine::text)
            .collect::<Vec<_>>()
            .join("\n");
        self.saw_prompt_before_response |= frame.contains("initial")
            && !frame.contains("you: initial")
            && !frame.contains("partial");
        self.saw_activity_before_response |=
            frame.contains("* Working (0s, esc to interrupt)") && !frame.contains("partial");
        Ok(())
    }
}

#[derive(Default)]
struct ReasoningFrames {
    labels: Vec<String>,
    selected: Option<String>,
}

impl FrameSink for ReasoningFrames {
    fn render(&mut self, state: &TuiState, _cursor_byte: usize) -> std::io::Result<()> {
        let Some(selector) = state
            .selector()
            .filter(|selector| selector.title() == "reasoning effort")
        else {
            return Ok(());
        };
        self.labels = selector
            .visible_items()
            .into_iter()
            .map(|item| item.label().to_owned())
            .collect();
        self.selected = selector.selected_label().map(str::to_owned);
        Ok(())
    }
}

#[derive(Default)]
struct AttachmentFrames {
    saw_idle_attachment: bool,
    saw_running_without_attachment: bool,
}

impl FrameSink for AttachmentFrames {
    fn render(&mut self, state: &TuiState, _cursor_byte: usize) -> std::io::Result<()> {
        self.saw_idle_attachment |= !state.is_running() && state.attachments().len() == 1;
        self.saw_running_without_attachment |= state.is_running() && state.attachments().is_empty();
        Ok(())
    }
}

async fn assert_image_command_failure(
    label: &str,
    capabilities: ModelCapabilities,
    bytes: &[u8],
    expected_notice: &str,
) {
    let root = std::env::temp_dir().join(format!(
        "tea-cli-image-{label}-{}",
        uuid::Uuid::now_v7().hyphenated()
    ));
    fs::create_dir_all(&root).unwrap();
    let path = root.join("fixture image.png");
    fs::write(&path, bytes).unwrap();
    let provider_id = ProviderId::from_str("fake").unwrap();
    let provider = Arc::new(ScriptedModelProvider::new(
        provider_id.clone(),
        vec![
            ModelSpec::new(
                ModelId::from_str("fake/model").unwrap(),
                provider_id,
                ModelDisplayName::from_str("Fake Model").unwrap(),
                TokenCount::new(32_000).unwrap(),
                TokenCount::new(4_000).unwrap(),
                capabilities,
            )
            .unwrap(),
        ],
        [],
    ));
    let args = CliArgs::try_parse_from([
        "tea",
        "--no-session",
        "--provider",
        "fake",
        "--model",
        "fake/model",
        "--trust",
        "ignore",
        "--cwd",
        root.to_str().unwrap(),
        "--config-dir",
        root.join("config").to_str().unwrap(),
        "--state-dir",
        root.join("state").to_str().unwrap(),
        "--data-dir",
        root.join("data").to_str().unwrap(),
    ])
    .unwrap();
    let bootstrap = CliBootstrap::new(BootstrapEnvironment::new(
        &root,
        Some(root.clone()),
        BTreeMap::new(),
    ))
    .with_provider(provider.clone());
    let (service, selection) = bootstrap.build(&args).unwrap();
    let command = format!("/image {}", path.display());
    let (sender, receiver) = tokio::sync::mpsc::channel(8);
    for input in [
        InputEvent::Paste(command.clone()),
        InputEvent::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
        InputEvent::Key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL)),
    ] {
        sender.send(input).await.unwrap();
    }
    drop(sender);

    let mut frames = AttachmentFrames::default();
    let mut clipboard = MemoryClipboard::default();
    let state = Box::pin(run_with_channels(
        &service,
        selection,
        receiver,
        &mut frames,
        &mut clipboard,
        None,
    ))
    .await
    .unwrap();
    let rendered = Renderer::new()
        .lines(&state, 80, &Theme::default())
        .iter()
        .map(tea_cli::tui::RenderedLine::text)
        .collect::<Vec<_>>()
        .join("\n");

    assert_eq!(state.editor(), command);
    assert!(state.attachments().is_empty());
    assert!(provider.captured_requests().unwrap().is_empty());
    assert!(rendered.contains(expected_notice), "{rendered}");
    service.shutdown().await;
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn unsupported_models_and_invalid_files_preserve_the_image_command_draft() {
    Box::pin(assert_image_command_failure(
        "unsupported",
        ModelCapabilities::text().with_tools(true),
        b"\x89PNG\r\n\x1a\nbody",
        "selected model does not support image input",
    ))
    .await;
    Box::pin(assert_image_command_failure(
        "invalid",
        ModelCapabilities::text()
            .with_image_input()
            .with_tools(true),
        b"not an image",
        "image format is unsupported",
    ))
    .await;
}

#[tokio::test(flavor = "current_thread")]
#[allow(clippy::too_many_lines)]
async fn model_change_after_attach_is_rechecked_without_consuming_the_composer() {
    let root = std::env::temp_dir().join(format!(
        "tea-cli-image-model-change-{}",
        uuid::Uuid::now_v7().hyphenated()
    ));
    fs::create_dir_all(&root).unwrap();
    let path = root.join("fixture.png");
    fs::write(&path, b"\x89PNG\r\n\x1a\nprivate-image").unwrap();
    let provider_id = ProviderId::from_str("fake").unwrap();
    let provider = Arc::new(ScriptedModelProvider::new(
        provider_id.clone(),
        vec![
            ModelSpec::new(
                ModelId::from_str("fake/image").unwrap(),
                provider_id.clone(),
                ModelDisplayName::from_str("Fake Image").unwrap(),
                TokenCount::new(32_000).unwrap(),
                TokenCount::new(4_000).unwrap(),
                ModelCapabilities::text()
                    .with_image_input()
                    .with_tools(true),
            )
            .unwrap(),
            ModelSpec::new(
                ModelId::from_str("fake/text").unwrap(),
                provider_id,
                ModelDisplayName::from_str("Fake Text").unwrap(),
                TokenCount::new(32_000).unwrap(),
                TokenCount::new(4_000).unwrap(),
                ModelCapabilities::text().with_tools(true),
            )
            .unwrap(),
        ],
        [],
    ));
    let args = CliArgs::try_parse_from([
        "tea",
        "--no-session",
        "--provider",
        "fake",
        "--model",
        "fake/image",
        "--trust",
        "ignore",
        "--cwd",
        root.to_str().unwrap(),
        "--config-dir",
        root.join("config").to_str().unwrap(),
        "--state-dir",
        root.join("state").to_str().unwrap(),
        "--data-dir",
        root.join("data").to_str().unwrap(),
    ])
    .unwrap();
    let bootstrap = CliBootstrap::new(BootstrapEnvironment::new(
        &root,
        Some(root.clone()),
        BTreeMap::new(),
    ))
    .with_provider(provider.clone());
    let (service, selection) = bootstrap.build(&args).unwrap();
    let (sender, receiver) = tokio::sync::mpsc::channel(12);
    for input in [
        InputEvent::Paste(format!("/image {}", path.display())),
        InputEvent::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
        InputEvent::Paste("/model fake/text".to_owned()),
        InputEvent::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
        InputEvent::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
        InputEvent::Paste("keep after model change".to_owned()),
        InputEvent::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
        InputEvent::Key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL)),
    ] {
        sender.send(input).await.unwrap();
    }
    drop(sender);

    let mut frames = Frames::default();
    let mut clipboard = MemoryClipboard::default();
    let state = Box::pin(run_with_channels(
        &service,
        selection,
        receiver,
        &mut frames,
        &mut clipboard,
        None,
    ))
    .await
    .unwrap();

    assert_eq!(state.model_id().map(ModelId::as_str), Some("fake/text"));
    assert_eq!(state.editor(), "keep after model change");
    assert_eq!(state.attachments().len(), 1);
    assert!(provider.captured_requests().unwrap().is_empty());
    service.shutdown().await;
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
#[allow(clippy::too_many_lines)]
async fn typed_image_submission_clears_on_acceptance_and_active_retry_is_retained() {
    let root = std::env::temp_dir().join(format!(
        "tea-cli-image-submit-{}",
        uuid::Uuid::now_v7().hyphenated()
    ));
    fs::create_dir_all(&root).unwrap();
    let path = root.join("fixture image.png");
    fs::write(&path, b"\x89PNG\r\n\x1a\nprivate-image").unwrap();
    let provider_id = ProviderId::from_str("fake").unwrap();
    let provider = Arc::new(ScriptedModelProvider::new(
        provider_id.clone(),
        vec![
            ModelSpec::new(
                ModelId::from_str("fake/model").unwrap(),
                provider_id,
                ModelDisplayName::from_str("Fake Model").unwrap(),
                TokenCount::new(32_000).unwrap(),
                TokenCount::new(4_000).unwrap(),
                ModelCapabilities::text()
                    .with_image_input()
                    .with_tools(true),
            )
            .unwrap(),
        ],
        [ScriptedModelResponse::await_cancellation()],
    ));
    let args = CliArgs::try_parse_from([
        "tea",
        "--no-session",
        "--provider",
        "fake",
        "--model",
        "fake/model",
        "--trust",
        "ignore",
        "--cwd",
        root.to_str().unwrap(),
        "--config-dir",
        root.join("config").to_str().unwrap(),
        "--state-dir",
        root.join("state").to_str().unwrap(),
        "--data-dir",
        root.join("data").to_str().unwrap(),
    ])
    .unwrap();
    let bootstrap = CliBootstrap::new(BootstrapEnvironment::new(
        &root,
        Some(root.clone()),
        BTreeMap::new(),
    ))
    .with_provider(provider.clone());
    let (service, selection) = bootstrap.build(&args).unwrap();
    let command = format!("/image {}", path.display());
    let (sender, receiver) = tokio::sync::mpsc::channel(16);
    let observed_provider = provider.clone();
    let repeated_command = command.clone();
    let producer = tokio::spawn(async move {
        for input in [
            InputEvent::Paste(command),
            InputEvent::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            InputEvent::Paste("describe this image".to_owned()),
            InputEvent::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
        ] {
            sender.send(input).await.unwrap();
        }
        while observed_provider.captured_requests().unwrap().is_empty() {
            tokio::task::yield_now().await;
        }
        for input in [
            InputEvent::Paste(repeated_command),
            InputEvent::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            InputEvent::Paste("keep this draft".to_owned()),
            InputEvent::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            InputEvent::Key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL)),
        ] {
            sender.send(input).await.unwrap();
        }
    });

    let mut frames = AttachmentFrames::default();
    let mut clipboard = MemoryClipboard::default();
    let state = Box::pin(run_with_channels(
        &service,
        selection,
        receiver,
        &mut frames,
        &mut clipboard,
        None,
    ))
    .await
    .unwrap();
    producer.await.unwrap();

    assert!(frames.saw_idle_attachment);
    assert!(frames.saw_running_without_attachment);
    assert_eq!(state.editor(), "keep this draft");
    assert_eq!(state.attachments().len(), 1);
    let requests = provider.captured_requests().unwrap();
    assert_eq!(requests.len(), 1);
    assert!(matches!(
        requests[0].messages().last(),
        Some(CanonicalMessage::User { content, .. })
            if matches!(content.as_slice(), [
                ContentBlock::Text { text } ,
                ContentBlock::Image {
                    mime_type,
                    source: ImageSource::InlineBase64 { .. },
                },
            ] if text == "describe this image" && mime_type == "image/png")
    ));

    service.shutdown().await;
    fs::remove_dir_all(root).unwrap();
}

#[derive(Default)]
struct CompletionFrames {
    renderer: Renderer,
    saw_menu: bool,
    saw_skill_menu: bool,
    saw_typed_skill: bool,
}

impl FrameSink for CompletionFrames {
    fn render(&mut self, state: &TuiState, _cursor_byte: usize) -> std::io::Result<()> {
        self.saw_menu = self.saw_menu
            || self
                .renderer
                .lines(state, 80, &Theme::default())
                .iter()
                .any(|line| line.text().contains("completions: select item"));
        self.saw_skill_menu = self.saw_skill_menu
            || self
                .renderer
                .lines(state, 80, &Theme::default())
                .iter()
                .any(|line| line.text().contains("skills: select skill"));
        self.saw_typed_skill = self.saw_typed_skill
            || self
                .renderer
                .lines(state, 80, &Theme::default())
                .iter()
                .any(|line| line.text().contains("/frontend-design  [skill]"));
        Ok(())
    }
}

#[derive(Default)]
struct SkillCatalogFrames {
    renderer: Renderer,
    last_frame: String,
}

#[derive(Default)]
struct SkillInvocationFrames {
    renderer: Renderer,
    saw_selected_skill_draft: Arc<AtomicBool>,
    saw_visible_invocation_without_body: Arc<AtomicBool>,
    run_stopped: Arc<AtomicBool>,
    saw_running: bool,
    inserted_history: Vec<String>,
}

impl FrameSink for SkillInvocationFrames {
    fn insert_history_cells(&mut self, cells: &[CellNode]) -> std::io::Result<()> {
        self.inserted_history
            .extend(cells.iter().map(CellNode::raw_text));
        Ok(())
    }

    fn replace_history_cells(&mut self, cells: &[CellNode]) -> std::io::Result<()> {
        self.inserted_history = cells.iter().map(CellNode::raw_text).collect();
        Ok(())
    }

    fn render(&mut self, state: &TuiState, _cursor_byte: usize) -> std::io::Result<()> {
        if state.editor() == "$review " && state.editor_skill_mention() == Some("$review") {
            self.saw_selected_skill_draft.store(true, Ordering::Release);
        }
        if state.is_running() {
            self.saw_running = true;
        } else if self.saw_running {
            self.run_stopped.store(true, Ordering::Release);
        }
        let frame = self
            .renderer
            .lines(state, 80, &Theme::default())
            .iter()
            .map(tea_cli::tui::RenderedLine::text)
            .collect::<Vec<_>>()
            .join("\n");
        if frame.contains("$review inspect src") && !frame.contains("PRIVATE SKILL BODY") {
            self.saw_visible_invocation_without_body
                .store(true, Ordering::Release);
        }
        Ok(())
    }
}

#[tokio::test(flavor = "current_thread")]
#[allow(clippy::too_many_lines)]
async fn selected_skill_is_visible_before_response_without_rendering_its_body() {
    let root = std::env::temp_dir().join(format!(
        "tea-cli-skill-invocation-{}",
        uuid::Uuid::now_v7().hyphenated()
    ));
    let skill = root.join("config/skills/review");
    fs::create_dir_all(&skill).unwrap();
    fs::write(
        skill.join("SKILL.md"),
        "---\nname: review\ndescription: Review code safely\n---\nPRIVATE SKILL BODY\n",
    )
    .unwrap();

    let provider_id = ProviderId::from_str("fake").unwrap();
    let provider = Arc::new(ScriptedModelProvider::new(
        provider_id.clone(),
        vec![reasoning_model(
            &provider_id,
            "fake/model",
            ReasoningEffort::Low,
            [ReasoningEffort::Low],
        )],
        [ScriptedModelResponse::await_cancellation()],
    ));
    let args = CliArgs::try_parse_from([
        "tea",
        "--no-session",
        "--provider",
        "fake",
        "--model",
        "fake/model",
        "--trust",
        "ignore",
        "--cwd",
        root.to_str().unwrap(),
        "--config-dir",
        root.join("config").to_str().unwrap(),
        "--state-dir",
        root.join("state").to_str().unwrap(),
        "--data-dir",
        root.join("data").to_str().unwrap(),
    ])
    .unwrap();
    let bootstrap = CliBootstrap::new(BootstrapEnvironment::new(
        &root,
        Some(root.clone()),
        BTreeMap::new(),
    ))
    .with_provider(provider.clone());
    let (service, selection) = bootstrap.build(&args).unwrap();
    let (sender, receiver) = tokio::sync::mpsc::channel(8);
    let mut frames = SkillInvocationFrames::default();
    let saw_selected = Arc::clone(&frames.saw_selected_skill_draft);
    let saw_visible = Arc::clone(&frames.saw_visible_invocation_without_body);
    let run_stopped = Arc::clone(&frames.run_stopped);
    let observed_provider = Arc::clone(&provider);
    let producer = tokio::spawn(async move {
        sender
            .send(InputEvent::Paste("/rev".to_owned()))
            .await
            .unwrap();
        sender
            .send(InputEvent::Key(KeyEvent::new(
                KeyCode::Tab,
                KeyModifiers::NONE,
            )))
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while !saw_selected.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("selected skill did not become a styled composer mention");
        sender
            .send(InputEvent::Paste("inspect src".to_owned()))
            .await
            .unwrap();
        sender
            .send(InputEvent::Key(KeyEvent::new(
                KeyCode::Enter,
                KeyModifiers::NONE,
            )))
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while !saw_visible.load(Ordering::Acquire)
                || observed_provider.captured_requests().unwrap().is_empty()
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("skill invocation was not rendered before the model response");
        sender
            .send(InputEvent::Key(KeyEvent::new(
                KeyCode::Esc,
                KeyModifiers::NONE,
            )))
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while !run_stopped.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("skill run did not stop after cancellation");
        sender
            .send(InputEvent::Key(KeyEvent::new(
                KeyCode::Char('d'),
                KeyModifiers::CONTROL,
            )))
            .await
            .unwrap();
    });
    let mut clipboard = MemoryClipboard::default();
    let state = Box::pin(run_with_channels(
        &service,
        selection,
        receiver,
        &mut frames,
        &mut clipboard,
        None,
    ))
    .await
    .unwrap();
    producer.await.unwrap();

    let requests = provider.captured_requests().unwrap();
    assert!(requests[0].messages().iter().any(|message| matches!(
        message,
        CanonicalMessage::User { content, .. }
            if content.iter().any(|block| matches!(
                block,
                ContentBlock::ContextualText { text } if text.contains("PRIVATE SKILL BODY")
            ))
    )));
    assert!(state.messages().iter().any(|message| matches!(
        message,
        CanonicalMessage::User { content, .. }
            if content.iter().any(|block| matches!(
                block,
                ContentBlock::Text { text } if text == "$review inspect src"
            ))
    )));
    assert!(
        frames
            .inserted_history
            .iter()
            .any(|text| text == "$review inspect src")
    );
    assert!(
        frames
            .inserted_history
            .iter()
            .all(|text| !text.contains("PRIVATE SKILL BODY"))
    );

    service.shutdown().await;
    fs::remove_dir_all(root).unwrap();
}

impl FrameSink for SkillCatalogFrames {
    fn render(&mut self, state: &TuiState, _cursor_byte: usize) -> std::io::Result<()> {
        self.last_frame = self
            .renderer
            .lines(state, 80, &Theme::default())
            .iter()
            .map(tea_cli::tui::RenderedLine::text)
            .collect::<Vec<_>>()
            .join("\n");
        Ok(())
    }
}

#[tokio::test(flavor = "current_thread")]
async fn skills_command_projects_sorted_frozen_rows_without_filesystem_paths() {
    let root = std::env::temp_dir().join(format!(
        "tea-cli-skills-command-{}",
        uuid::Uuid::now_v7().hyphenated()
    ));
    let review = root.join("config/skills/review");
    let manual = root.join("config/skills/manual");
    fs::create_dir_all(&review).unwrap();
    fs::create_dir_all(&manual).unwrap();
    fs::write(
        review.join("SKILL.md"),
        "---\nname: review\ndescription: Review code safely\n---\nReview.\n",
    )
    .unwrap();
    fs::write(
        manual.join("SKILL.md"),
        "---\nname: manual-only\ndescription: Explicit invocation only\ndisable-model-invocation: true\n---\nManual.\n",
    )
    .unwrap();

    let provider_id = ProviderId::from_str("fake").unwrap();
    let (_provider, service, selection) = reasoning_service(
        &root,
        vec![reasoning_model(
            &provider_id,
            "fake/model",
            ReasoningEffort::Low,
            [ReasoningEffort::Low],
        )],
        Vec::new(),
        "fake/model",
    );
    let (sender, receiver) = tokio::sync::mpsc::channel(16);
    for character in "/skills".chars() {
        sender
            .send(InputEvent::Key(KeyEvent::new(
                KeyCode::Char(character),
                KeyModifiers::NONE,
            )))
            .await
            .unwrap();
    }
    sender
        .send(InputEvent::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        )))
        .await
        .unwrap();
    sender
        .send(InputEvent::Key(KeyEvent::new(
            KeyCode::Char('d'),
            KeyModifiers::CONTROL,
        )))
        .await
        .unwrap();
    drop(sender);

    let mut frames = SkillCatalogFrames::default();
    let mut clipboard = MemoryClipboard::default();
    let state = Box::pin(run_with_channels(
        &service,
        selection,
        receiver,
        &mut frames,
        &mut clipboard,
        None,
    ))
    .await
    .unwrap();

    assert_eq!(
        state.skill_catalog(),
        [
            "manual-only [user-tea; explicit-only] - Explicit invocation only",
            "review [user-tea] - Review code safely",
        ]
    );
    assert!(
        frames
            .last_frame
            .contains("manual-only [user-tea; explicit-only]")
    );
    assert!(
        frames
            .last_frame
            .contains("review [user-tea] - Review code safely")
    );
    assert!(!frames.last_frame.contains("SKILL.md"));
    assert!(!frames.last_frame.contains(root.to_str().unwrap()));

    service.shutdown().await;
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn skills_command_reports_empty_catalog_without_mutating_the_session() {
    let root = std::env::temp_dir().join(format!(
        "tea-cli-empty-skills-command-{}",
        uuid::Uuid::now_v7().hyphenated()
    ));
    fs::create_dir_all(&root).unwrap();
    let provider_id = ProviderId::from_str("fake").unwrap();
    let (_provider, service, selection) = reasoning_service(
        &root,
        vec![reasoning_model(
            &provider_id,
            "fake/model",
            ReasoningEffort::Low,
            [ReasoningEffort::Low],
        )],
        Vec::new(),
        "fake/model",
    );
    let (sender, receiver) = tokio::sync::mpsc::channel(16);
    for input in [
        InputEvent::Key(KeyEvent::new(KeyCode::Char('/'), KeyModifiers::NONE)),
        InputEvent::Key(KeyEvent::new(KeyCode::Char('s'), KeyModifiers::NONE)),
        InputEvent::Key(KeyEvent::new(KeyCode::Char('k'), KeyModifiers::NONE)),
        InputEvent::Key(KeyEvent::new(KeyCode::Char('i'), KeyModifiers::NONE)),
        InputEvent::Key(KeyEvent::new(KeyCode::Char('l'), KeyModifiers::NONE)),
        InputEvent::Key(KeyEvent::new(KeyCode::Char('l'), KeyModifiers::NONE)),
        InputEvent::Key(KeyEvent::new(KeyCode::Char('s'), KeyModifiers::NONE)),
        InputEvent::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
        InputEvent::Key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL)),
    ] {
        sender.send(input).await.unwrap();
    }
    drop(sender);

    let mut frames = SkillCatalogFrames::default();
    let mut clipboard = MemoryClipboard::default();
    let state = Box::pin(run_with_channels(
        &service,
        selection,
        receiver,
        &mut frames,
        &mut clipboard,
        None,
    ))
    .await
    .unwrap();

    assert_eq!(state.skill_catalog(), ["no skills loaded"]);
    assert!(frames.last_frame.contains("no skills loaded"));

    service.shutdown().await;
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn command_completion_edits_a_draft_without_executing_or_losing_it_on_escape() {
    let root = std::env::temp_dir().join(format!(
        "tea-cli-command-completion-{}",
        uuid::Uuid::now_v7().hyphenated()
    ));
    fs::create_dir_all(&root).unwrap();
    let provider_id = ProviderId::from_str("fake").unwrap();
    let provider = Arc::new(ScriptedModelProvider::new(
        provider_id.clone(),
        vec![
            ModelSpec::new(
                ModelId::from_str("fake/model").unwrap(),
                provider_id,
                ModelDisplayName::from_str("Fake Model").unwrap(),
                TokenCount::new(32_000).unwrap(),
                TokenCount::new(4_000).unwrap(),
                ModelCapabilities::text().with_tools(true),
            )
            .unwrap(),
        ],
        Vec::<ScriptedModelResponse>::new(),
    ));
    let args = CliArgs::try_parse_from([
        "tea",
        "--no-session",
        "--provider",
        "fake",
        "--model",
        "fake/model",
        "--trust",
        "ignore",
        "--cwd",
        root.to_str().unwrap(),
        "--config-dir",
        root.join("config").to_str().unwrap(),
        "--state-dir",
        root.join("state").to_str().unwrap(),
        "--data-dir",
        root.join("data").to_str().unwrap(),
    ])
    .unwrap();
    let bootstrap = CliBootstrap::new(BootstrapEnvironment::new(
        &root,
        Some(root.clone()),
        BTreeMap::new(),
    ))
    .with_provider(provider);
    let (service, selection) = bootstrap.build(&args).unwrap();
    let (sender, receiver) = tokio::sync::mpsc::channel(16);
    for input in [
        InputEvent::Key(KeyEvent::new(KeyCode::Char('/'), KeyModifiers::NONE)),
        InputEvent::Key(KeyEvent::new(KeyCode::Char('n'), KeyModifiers::NONE)),
        InputEvent::Key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE)),
        InputEvent::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
        InputEvent::Key(KeyEvent::new(KeyCode::Backspace, KeyModifiers::NONE)),
        InputEvent::Key(KeyEvent::new(KeyCode::Char('n'), KeyModifiers::NONE)),
        InputEvent::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
        InputEvent::Key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL)),
    ] {
        sender.send(input).await.unwrap();
    }
    drop(sender);

    let mut frames = CompletionFrames::default();
    let mut clipboard = MemoryClipboard::default();
    let state = Box::pin(run_with_channels(
        &service,
        selection,
        receiver,
        &mut frames,
        &mut clipboard,
        None,
    ))
    .await
    .unwrap();
    let rendered = Renderer::new()
        .lines(&state, 80, &Theme::default())
        .iter()
        .map(tea_cli::tui::RenderedLine::text)
        .collect::<Vec<_>>()
        .join("\n");

    assert!(frames.saw_menu);
    assert_eq!(state.editor(), "/name");
    assert!(!state.is_running());
    assert!(state.messages().is_empty());
    assert!(!rendered.contains("session name updated"));
    service.shutdown().await;
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn slash_skill_completion_opens_and_escape_preserves_the_draft() {
    let root = std::env::temp_dir().join(format!(
        "tea-cli-skill-completion-{}",
        uuid::Uuid::now_v7().hyphenated()
    ));
    let skill = root.join("config/skills/frontend-design");
    fs::create_dir_all(&skill).unwrap();
    fs::write(
        skill.join("SKILL.md"),
        "---\nname: frontend-design\ndescription: Design frontend interfaces\n---\nInstructions\n",
    )
    .unwrap();
    let provider_id = ProviderId::from_str("fake").unwrap();
    let provider = Arc::new(ScriptedModelProvider::new(
        provider_id.clone(),
        vec![reasoning_model(
            &provider_id,
            "fake/model",
            ReasoningEffort::Low,
            [ReasoningEffort::Low],
        )],
        Vec::<ScriptedModelResponse>::new(),
    ));
    let args = CliArgs::try_parse_from([
        "tea",
        "--no-session",
        "--provider",
        "fake",
        "--model",
        "fake/model",
        "--trust",
        "ignore",
        "--cwd",
        root.to_str().unwrap(),
        "--config-dir",
        root.join("config").to_str().unwrap(),
        "--state-dir",
        root.join("state").to_str().unwrap(),
        "--data-dir",
        root.join("data").to_str().unwrap(),
    ])
    .unwrap();
    let bootstrap = CliBootstrap::new(BootstrapEnvironment::new(
        &root,
        Some(root.clone()),
        BTreeMap::new(),
    ))
    .with_provider(provider);
    let (service, selection) = bootstrap.build(&args).unwrap();
    let (sender, receiver) = tokio::sync::mpsc::channel(8);
    for input in [
        InputEvent::Paste("/front".to_owned()),
        InputEvent::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
        InputEvent::Key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL)),
    ] {
        sender.send(input).await.unwrap();
    }
    drop(sender);

    let mut frames = CompletionFrames::default();
    let mut clipboard = MemoryClipboard::default();
    let state = Box::pin(run_with_channels(
        &service,
        selection,
        receiver,
        &mut frames,
        &mut clipboard,
        None,
    ))
    .await
    .unwrap();

    assert!(frames.saw_menu);
    assert!(frames.saw_typed_skill);
    assert_eq!(state.editor(), "/front");
    assert!(state.messages().is_empty());
    service.shutdown().await;
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
#[allow(clippy::too_many_lines)]
async fn fake_backed_loop_steers_queues_aborts_restores_and_exits() {
    let root = std::env::temp_dir().join(format!(
        "tea-cli-interactive-{}",
        uuid::Uuid::now_v7().hyphenated()
    ));
    fs::create_dir_all(&root).unwrap();
    let provider_id = ProviderId::from_str("fake").unwrap();
    let provider = Arc::new(ScriptedModelProvider::new(
        provider_id.clone(),
        vec![
            ModelSpec::new(
                ModelId::from_str("fake/model").unwrap(),
                provider_id.clone(),
                ModelDisplayName::from_str("Fake Model").unwrap(),
                TokenCount::new(32_000).unwrap(),
                TokenCount::new(4_000).unwrap(),
                ModelCapabilities::text().with_tools(true),
            )
            .unwrap(),
            ModelSpec::new(
                ModelId::from_str("fake/beta").unwrap(),
                provider_id,
                ModelDisplayName::from_str("Fake Beta").unwrap(),
                TokenCount::new(32_000).unwrap(),
                TokenCount::new(4_000).unwrap(),
                ModelCapabilities::text().with_tools(true),
            )
            .unwrap(),
        ],
        [ScriptedModelResponse::new([
            ScriptStep::event(ModelEvent::Started(ModelResponseInfo::new())),
            ScriptStep::event(ModelEvent::TextDelta(Utf8Delta::new("partial").unwrap())),
            ScriptStep::AwaitCancellation,
        ])],
    ));
    let args = CliArgs::try_parse_from([
        "tea",
        "--no-session",
        "--provider",
        "fake",
        "--model",
        "fake/model",
        "--trust",
        "ignore",
        "--cwd",
        root.to_str().unwrap(),
        "--config-dir",
        root.join("config").to_str().unwrap(),
        "--state-dir",
        root.join("state").to_str().unwrap(),
        "--data-dir",
        root.join("data").to_str().unwrap(),
    ])
    .unwrap();
    let bootstrap = CliBootstrap::new(BootstrapEnvironment::new(
        &root,
        Some(root.clone()),
        BTreeMap::new(),
    ))
    .with_provider(provider.clone());
    let (service, selection) = bootstrap.build(&args).unwrap();
    assert_eq!(selection, SessionSelection::NoSession);
    let (sender, receiver) = tokio::sync::mpsc::channel(16);
    let mut frames = Frames::default();
    let run_stopped = Arc::clone(&frames.run_stopped);
    let producer = tokio::spawn(async move {
        sender
            .send(InputEvent::Key(KeyEvent::new(
                KeyCode::Char('o'),
                KeyModifiers::CONTROL,
            )))
            .await
            .unwrap();
        sender
            .send(InputEvent::Key(KeyEvent::new(
                KeyCode::Down,
                KeyModifiers::NONE,
            )))
            .await
            .unwrap();
        sender
            .send(InputEvent::Key(KeyEvent::new(
                KeyCode::Enter,
                KeyModifiers::NONE,
            )))
            .await
            .unwrap();
        sender
            .send(InputEvent::Key(KeyEvent::new(
                KeyCode::Enter,
                KeyModifiers::NONE,
            )))
            .await
            .unwrap();
        sender
            .send(InputEvent::Paste("initial".to_owned()))
            .await
            .unwrap();
        sender
            .send(InputEvent::Key(KeyEvent::new(
                KeyCode::Enter,
                KeyModifiers::NONE,
            )))
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while provider.captured_requests().unwrap().is_empty() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("interactive run did not reach the model provider");
        sender
            .send(InputEvent::Paste("steer".to_owned()))
            .await
            .unwrap();
        sender
            .send(InputEvent::Key(KeyEvent::new(
                KeyCode::Enter,
                KeyModifiers::NONE,
            )))
            .await
            .unwrap();
        sender
            .send(InputEvent::Paste("later".to_owned()))
            .await
            .unwrap();
        sender
            .send(InputEvent::Key(KeyEvent::new(
                KeyCode::Enter,
                KeyModifiers::ALT,
            )))
            .await
            .unwrap();
        sender
            .send(InputEvent::Key(KeyEvent::new(
                KeyCode::Esc,
                KeyModifiers::NONE,
            )))
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while !run_stopped.load(Ordering::Acquire) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("interactive projection did not observe the aborted run");
        sender
            .send(InputEvent::Key(KeyEvent::new(
                KeyCode::Char('d'),
                KeyModifiers::CONTROL,
            )))
            .await
            .unwrap();
    });
    let mut clipboard = MemoryClipboard::default();
    let state = Box::pin(run_with_channels(
        &service,
        selection,
        receiver,
        &mut frames,
        &mut clipboard,
        None,
    ))
    .await
    .unwrap();
    producer.await.unwrap();
    assert!(!state.is_running());
    assert_eq!(state.model_id().map(ModelId::as_str), Some("fake/beta"));
    assert!(state.editor().contains("steer"));
    assert!(state.editor().contains("later"));
    assert!(frames.count > 3);
    assert!(frames.saw_prompt_before_response);
    assert!(frames.saw_activity_before_response);
    assert_eq!(
        frames
            .inserted_history
            .iter()
            .filter(|text| text.as_str() == "initial")
            .count(),
        1,
        "a finalized cell must be handed to terminal history exactly once"
    );
    assert_eq!(frames.history_replacements, 0);
    assert_eq!(clipboard.contents(), None);
    service.shutdown().await;
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn reasoning_selector_includes_extended_levels_and_preselects_the_model_default() {
    let root = std::env::temp_dir().join(format!(
        "tea-cli-reasoning-selector-{}",
        uuid::Uuid::now_v7().hyphenated()
    ));
    fs::create_dir_all(&root).unwrap();
    let provider_id = ProviderId::from_str("fake").unwrap();
    let model = reasoning_model(
        &provider_id,
        "fake/reasoning",
        ReasoningEffort::Medium,
        ReasoningEffort::ALL,
    );
    let (_, service, selection) = reasoning_service(&root, vec![model], vec![], "fake/reasoning");
    let (sender, receiver) = tokio::sync::mpsc::channel(4);
    sender
        .send(InputEvent::Paste("/reasoning".to_owned()))
        .await
        .unwrap();
    sender
        .send(InputEvent::Key(KeyEvent::new(
            KeyCode::Enter,
            KeyModifiers::NONE,
        )))
        .await
        .unwrap();
    drop(sender);

    let mut frames = ReasoningFrames::default();
    let mut clipboard = MemoryClipboard::default();
    let state = run_with_channels(
        &service,
        selection,
        receiver,
        &mut frames,
        &mut clipboard,
        None,
    )
    .await
    .unwrap();

    assert_eq!(
        frames.labels,
        ["off", "minimal", "low", "medium", "high", "xhigh", "max"]
    );
    assert_eq!(frames.selected.as_deref(), Some("medium"));
    assert_eq!(state.selector().unwrap().selected_label(), Some("medium"));
    service.shutdown().await;
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn reasoning_command_updates_only_the_durable_session() {
    let root = std::env::temp_dir().join(format!(
        "tea-cli-reasoning-session-{}",
        uuid::Uuid::now_v7().hyphenated()
    ));
    fs::create_dir_all(&root).unwrap();
    let provider_id = ProviderId::from_str("fake").unwrap();
    let model = reasoning_model(
        &provider_id,
        "fake/reasoning",
        ReasoningEffort::Medium,
        ReasoningEffort::ALL,
    );
    let (_, service, selection) = reasoning_service(&root, vec![model], vec![], "fake/reasoning");
    let (sender, receiver) = tokio::sync::mpsc::channel(4);
    for input in [
        InputEvent::Paste("/reasoning high".to_owned()),
        InputEvent::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
        InputEvent::Key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL)),
    ] {
        sender.send(input).await.unwrap();
    }
    drop(sender);

    let mut frames = Frames::default();
    let mut clipboard = MemoryClipboard::default();
    let state = run_with_channels(
        &service,
        selection,
        receiver,
        &mut frames,
        &mut clipboard,
        None,
    )
    .await
    .unwrap();

    assert_eq!(state.reasoning_effort(), Some(ReasoningEffort::High));
    assert_eq!(state.displayed_reasoning_effort(), "high");
    assert!(!root.join("config/settings.json").exists());
    service.shutdown().await;
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn shift_tab_cycles_shortcut_levels_without_extended_efforts_or_persistence() {
    let root = std::env::temp_dir().join(format!(
        "tea-cli-reasoning-shortcut-{}",
        uuid::Uuid::now_v7().hyphenated()
    ));
    fs::create_dir_all(&root).unwrap();
    let provider_id = ProviderId::from_str("fake").unwrap();
    let model = reasoning_model(
        &provider_id,
        "fake/reasoning",
        ReasoningEffort::Medium,
        ReasoningEffort::ALL,
    );
    let (_, service, selection) = reasoning_service(&root, vec![model], vec![], "fake/reasoning");
    let (sender, receiver) = tokio::sync::mpsc::channel(4);
    for input in [
        InputEvent::Key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT)),
        InputEvent::Key(KeyEvent::new(KeyCode::BackTab, KeyModifiers::SHIFT)),
        InputEvent::Key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL)),
    ] {
        sender.send(input).await.unwrap();
    }
    drop(sender);

    let mut frames = Frames::default();
    let mut clipboard = MemoryClipboard::default();
    let state = run_with_channels(
        &service,
        selection,
        receiver,
        &mut frames,
        &mut clipboard,
        None,
    )
    .await
    .unwrap();

    assert_eq!(state.reasoning_effort(), Some(ReasoningEffort::Off));
    assert!(!root.join("config/settings.json").exists());
    service.shutdown().await;
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn unsupported_direct_reasoning_is_clamped_with_a_notice() {
    let root = std::env::temp_dir().join(format!(
        "tea-cli-reasoning-clamp-{}",
        uuid::Uuid::now_v7().hyphenated()
    ));
    fs::create_dir_all(&root).unwrap();
    let provider_id = ProviderId::from_str("fake").unwrap();
    let model = reasoning_model(
        &provider_id,
        "fake/reasoning",
        ReasoningEffort::Low,
        [
            ReasoningEffort::Off,
            ReasoningEffort::Low,
            ReasoningEffort::High,
        ],
    );
    let (_, service, selection) = reasoning_service(&root, vec![model], vec![], "fake/reasoning");
    let (sender, receiver) = tokio::sync::mpsc::channel(4);
    for input in [
        InputEvent::Paste("/reasoning max".to_owned()),
        InputEvent::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
        InputEvent::Key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL)),
    ] {
        sender.send(input).await.unwrap();
    }
    drop(sender);

    let mut frames = Frames::default();
    let mut clipboard = MemoryClipboard::default();
    let state = run_with_channels(
        &service,
        selection,
        receiver,
        &mut frames,
        &mut clipboard,
        None,
    )
    .await
    .unwrap();
    let rendered = Renderer::new()
        .lines(&state, 80, &Theme::default())
        .iter()
        .map(tea_cli::tui::RenderedLine::text)
        .collect::<Vec<_>>()
        .join("\n");

    assert_eq!(state.reasoning_effort(), Some(ReasoningEffort::High));
    assert!(
        rendered.contains("reasoning max adjusted to high for the selected model"),
        "{rendered}"
    );
    service.shutdown().await;
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
#[allow(clippy::too_many_lines)]
async fn active_run_reasoning_change_is_frozen_then_applied_to_the_next_turn() {
    let root = std::env::temp_dir().join(format!(
        "tea-cli-reasoning-pending-{}",
        uuid::Uuid::now_v7().hyphenated()
    ));
    fs::create_dir_all(&root).unwrap();
    let provider_id = ProviderId::from_str("fake").unwrap();
    let model = reasoning_model(
        &provider_id,
        "fake/reasoning",
        ReasoningEffort::Medium,
        ReasoningEffort::ALL,
    );
    let (provider, service, _) = reasoning_service(
        &root,
        vec![model],
        vec![
            ScriptedModelResponse::await_cancellation(),
            ScriptedModelResponse::text(["done"]),
        ],
        "fake/reasoning",
    );
    let session_id = service.create_session().await.unwrap();
    let (sender, receiver) = tokio::sync::mpsc::channel(12);
    let producer = async {
        for input in [
            InputEvent::Paste("first".to_owned()),
            InputEvent::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
        ] {
            sender.send(input).await.unwrap();
        }
        while provider.captured_requests().unwrap().is_empty() {
            tokio::task::yield_now().await;
        }
        for input in [
            InputEvent::Paste("/reasoning high".to_owned()),
            InputEvent::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            InputEvent::Key(KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE)),
        ] {
            sender.send(input).await.unwrap();
        }
        let mut pending_applied = false;
        for _ in 0..4096 {
            let snapshot = service.session_snapshot(session_id).await.unwrap();
            if snapshot.state().configuration().reasoning_effort() == Some(ReasoningEffort::High)
                && !service.snapshot(session_id).await.unwrap().is_running()
            {
                pending_applied = true;
                break;
            }
            tokio::task::yield_now().await;
        }
        if pending_applied {
            for input in [
                InputEvent::Paste("second".to_owned()),
                InputEvent::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
            ] {
                sender.send(input).await.unwrap();
            }
            while provider.captured_requests().unwrap().len() < 2 {
                tokio::task::yield_now().await;
            }
            while service.snapshot(session_id).await.unwrap().is_running() {
                tokio::task::yield_now().await;
            }
        }
        sender
            .send(InputEvent::Key(KeyEvent::new(
                KeyCode::Char('d'),
                KeyModifiers::CONTROL,
            )))
            .await
            .unwrap();
        drop(sender);
        pending_applied
    };
    let mut frames = Frames::default();
    let mut clipboard = MemoryClipboard::default();
    let interactive = run_with_channels(
        &service,
        SessionSelection::Existing(session_id),
        receiver,
        &mut frames,
        &mut clipboard,
        None,
    );
    let (state, pending_applied) = tokio::join!(interactive, producer);
    let state = state.unwrap();
    let requests = provider.captured_requests().unwrap();

    assert!(
        pending_applied,
        "queued effort was not applied: running={}, pending={:?}, durable={:?}, requests={}",
        state.is_running(),
        state.pending_reasoning_effort(),
        state.reasoning_effort(),
        requests.len()
    );
    assert_eq!(requests.len(), 2);
    assert_eq!(
        requests[0]
            .reasoning()
            .map(tea_model::ReasoningOptions::effort),
        Some(ReasoningEffort::Medium)
    );
    assert_eq!(
        requests[1]
            .reasoning()
            .map(tea_model::ReasoningOptions::effort),
        Some(ReasoningEffort::High)
    );
    assert_eq!(state.reasoning_effort(), Some(ReasoningEffort::High));
    assert_eq!(state.pending_reasoning_effort(), None);
    service.shutdown().await;
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn model_then_reasoning_persists_sparse_global_defaults() {
    let root = std::env::temp_dir().join(format!(
        "tea-cli-reasoning-persist-{}",
        uuid::Uuid::now_v7().hyphenated()
    ));
    fs::create_dir_all(&root).unwrap();
    let provider_id = ProviderId::from_str("fake").unwrap();
    let models = ["fake/alpha", "fake/beta"]
        .into_iter()
        .map(|model_id| {
            reasoning_model(
                &provider_id,
                model_id,
                ReasoningEffort::Medium,
                ReasoningEffort::ALL,
            )
        })
        .collect();
    let (_, service, selection) = reasoning_service(&root, models, vec![], "fake/alpha");
    let settings_path = root.join("config/settings.json");
    let (sender, receiver) = tokio::sync::mpsc::channel(8);
    for input in [
        InputEvent::Paste("/model fake/beta".to_owned()),
        InputEvent::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
        InputEvent::Key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE)),
        InputEvent::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
        InputEvent::Key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL)),
    ] {
        sender.send(input).await.unwrap();
    }
    drop(sender);

    let mut frames = Frames::default();
    let mut clipboard = MemoryClipboard::default();
    let state = run_with_channels_with_settings_path(
        &service,
        selection,
        receiver,
        &mut frames,
        &mut clipboard,
        None,
        settings_path.clone(),
    )
    .await
    .unwrap();
    let persisted: serde_json::Value =
        serde_json::from_slice(&fs::read(&settings_path).unwrap()).unwrap();

    assert_eq!(state.model_id().map(ModelId::as_str), Some("fake/beta"));
    assert_eq!(state.reasoning_effort(), Some(ReasoningEffort::High));
    assert_eq!(
        persisted,
        serde_json::json!({
            "schemaVersion": 1,
            "provider": "fake",
            "model": "fake/beta",
            "thinking": "high"
        })
    );
    service.shutdown().await;
    fs::remove_dir_all(root).unwrap();
}

#[tokio::test(flavor = "current_thread")]
async fn model_defaults_write_failure_keeps_the_session_change_and_warns_once() {
    let root = std::env::temp_dir().join(format!(
        "tea-cli-reasoning-persist-failure-{}",
        uuid::Uuid::now_v7().hyphenated()
    ));
    fs::create_dir_all(&root).unwrap();
    let provider_id = ProviderId::from_str("fake").unwrap();
    let models = ["fake/alpha", "fake/beta"]
        .into_iter()
        .map(|model_id| {
            reasoning_model(
                &provider_id,
                model_id,
                ReasoningEffort::Medium,
                ReasoningEffort::ALL,
            )
        })
        .collect();
    let (_, service, selection) = reasoning_service(&root, models, vec![], "fake/alpha");
    let unavailable_path = root.join("missing/config/settings.json");
    let (sender, receiver) = tokio::sync::mpsc::channel(8);
    for input in [
        InputEvent::Paste("/model fake/beta".to_owned()),
        InputEvent::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
        InputEvent::Key(KeyEvent::new(KeyCode::Down, KeyModifiers::NONE)),
        InputEvent::Key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE)),
        InputEvent::Key(KeyEvent::new(KeyCode::Char('d'), KeyModifiers::CONTROL)),
    ] {
        sender.send(input).await.unwrap();
    }
    drop(sender);

    let mut frames = Frames::default();
    let mut clipboard = MemoryClipboard::default();
    let state = run_with_channels_with_settings_path(
        &service,
        selection,
        receiver,
        &mut frames,
        &mut clipboard,
        None,
        unavailable_path.clone(),
    )
    .await
    .unwrap();
    let rendered = Renderer::new()
        .lines(&state, 80, &Theme::default())
        .iter()
        .map(tea_cli::tui::RenderedLine::text)
        .collect::<Vec<_>>()
        .join("\n");
    let warning = "session updated, but global model defaults could not be saved";

    assert_eq!(state.model_id().map(ModelId::as_str), Some("fake/beta"));
    assert_eq!(state.reasoning_effort(), Some(ReasoningEffort::High));
    assert!(!unavailable_path.exists());
    assert_eq!(rendered.matches(warning).count(), 1, "{rendered}");
    service.shutdown().await;
    fs::remove_dir_all(root).unwrap();
}
