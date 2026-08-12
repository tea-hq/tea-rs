use std::collections::BTreeMap;
use std::path::Path;

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::style::Style;
use tea_protocol::{ModelId, ToolCallId};
use unicode_width::UnicodeWidthStr as _;

use super::attachment::{format_byte_count, format_compact_byte_count};
use super::bottom_pane::BottomPane;
#[cfg(test)]
use super::cells::MAX_VISIBLE_PLAN_BODY_ROWS;
use super::cells::{CellBlock, CellContext, CellHost, CellList};
use super::components::{
    Component, Element, RenderContext, Text, truncate_cells, truncate_cells_with_ellipsis,
};
use super::hyperlink::HyperlinkBuffer;
use super::layout::{
    CursorState, Renderable, VerticalLayout, draw_lines, draw_lines_with_hyperlinks,
};
use super::overlay::Overlay;
use super::presentation::{CellContent, CellId, CellNode, Presentation};
#[cfg(test)]
use super::presentation::{
    LifecycleStatus, OutputFormat, PlanStep, PlanStepStatus, TimelineDetail, TimelineDetailKind,
    TimelineSource,
};
use super::render_output::{RenderedLine, RenderedSpan};
use super::state::{ApprovalChoice, TuiState};
use super::status::StatusIndicator;
use super::theme::Theme;
use super::transcript::Transcript;

const MAX_CELL_RENDER_CACHE_ENTRIES: usize = 256;
const MAX_APPROVAL_ARGUMENT_CELLS: usize = 480;
const COMPOSER_VERTICAL_PADDING_ROWS: usize = 1;
const COMPOSER_PROMPT_PREFIX: &str = "› ";
const CONTENT_HORIZONTAL_MARGIN: u16 = 1;
const CONTENT_VERTICAL_MARGIN: u16 = 1;
const MIN_INLINE_VIEWPORT_HEIGHT: u16 = 8;
const MAX_COMPOSER_EDITOR_ROWS: usize = 6;
const MAX_BOTTOM_MODAL_OPTIONS: usize = 7;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct CellCacheKey {
    id: CellId,
    width: u16,
    theme_generation: u64,
    stylesheet_generation: u64,
}

impl CellCacheKey {
    fn from_cell(cell: &CellNode, width: u16, theme: &Theme, stylesheet_generation: u64) -> Self {
        Self {
            id: cell.id(),
            width,
            theme_generation: theme.generation,
            stylesheet_generation,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CellCacheEntry {
    content: CellContent,
    tool_call_id: Option<ToolCallId>,
    block: CellBlock,
}

#[derive(Debug, Default)]
struct CellRenderCache {
    entries: BTreeMap<CellCacheKey, CellCacheEntry>,
    #[cfg(test)]
    hits: usize,
    #[cfg(test)]
    misses: usize,
}

impl CellRenderCache {
    fn get_or_render(
        &mut self,
        key: CellCacheKey,
        content: &CellContent,
        tool_call_id: Option<ToolCallId>,
        render: impl FnOnce() -> CellBlock,
    ) -> CellBlock {
        if let Some(entry) = self.entries.get(&key)
            && entry.content == *content
            && entry.tool_call_id == tool_call_id
        {
            #[cfg(test)]
            {
                self.hits = self.hits.saturating_add(1);
            }
            return entry.block.clone();
        }
        #[cfg(test)]
        {
            self.misses = self.misses.saturating_add(1);
        }
        let block = render();
        if !self.entries.contains_key(&key)
            && self.entries.len() >= MAX_CELL_RENDER_CACHE_ENTRIES
            && let Some(first) = self.entries.keys().next().cloned()
        {
            self.entries.remove(&first);
        }
        self.entries.insert(
            key,
            CellCacheEntry {
                content: content.clone(),
                tool_call_id,
                block: block.clone(),
            },
        );
        block
    }

    fn clear(&mut self) {
        self.entries.clear();
        #[cfg(test)]
        {
            self.hits = 0;
            self.misses = 0;
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
struct RenderSections {
    timeline: Vec<RenderedLine>,
    modal: Vec<RenderedLine>,
}

impl RenderSections {
    fn all_lines(&self, composer: Vec<RenderedLine>) -> Vec<RenderedLine> {
        let mut lines = self.timeline.clone();
        lines.extend(self.modal.clone());
        lines.extend(composer);
        lines
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ComposerFrame {
    lines: Vec<RenderedLine>,
    cursor: Option<CursorState>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BottomPaneLines {
    lines: Vec<RenderedLine>,
    cursor: Option<CursorState>,
}

fn bottom_pane_lines(
    status: Option<&StatusIndicator>,
    modal: &[RenderedLine],
    composer: ComposerFrame,
) -> BottomPaneLines {
    let status_lines = status.map_or(&[][..], StatusIndicator::lines);
    let modal_len = modal.len();
    let mut lines = Vec::with_capacity(
        status_lines
            .len()
            .saturating_add(modal_len)
            .saturating_add(composer.lines.len()),
    );
    lines.extend_from_slice(modal);
    lines.extend_from_slice(status_lines);
    let cursor = composer.cursor.map(|mut cursor| {
        cursor.y = cursor.y.saturating_add(
            u16::try_from(status_lines.len().saturating_add(modal_len)).unwrap_or(u16::MAX),
        );
        cursor
    });
    lines.extend(composer.lines);
    BottomPaneLines { lines, cursor }
}

fn axis_margin(length: u16, preferred: u16) -> u16 {
    preferred.min(length.saturating_sub(1) / 2)
}

fn content_area(area: Rect) -> Rect {
    let vertical = if area.height >= MIN_INLINE_VIEWPORT_HEIGHT {
        axis_margin(area.height, CONTENT_VERTICAL_MARGIN)
    } else {
        0
    };
    Rect::new(
        area.x,
        area.y.saturating_add(vertical),
        area.width,
        area.height.saturating_sub(vertical.saturating_mul(2)),
    )
}

fn content_width(width: u16) -> u16 {
    let horizontal = axis_margin(width, CONTENT_HORIZONTAL_MARGIN);
    width.saturating_sub(horizontal.saturating_mul(2)).max(1)
}

fn modal_surface_in_viewport(
    state: &TuiState,
    screen_width: usize,
    theme: &Theme,
) -> Vec<RenderedLine> {
    let screen_width = u16::try_from(screen_width).unwrap_or(u16::MAX).max(1);
    let margin = axis_margin(screen_width, CONTENT_HORIZONTAL_MARGIN);
    modal_surface(state, usize::from(content_width(screen_width)), theme)
        .into_iter()
        .map(|line| line.with_left_columns(usize::from(margin)))
        .collect()
}

/// Deterministic terminal renderer with a bounded width/theme-aware cell cache.
#[derive(Debug, Default)]
pub struct Renderer {
    cell_host: CellHost,
    cache: CellRenderCache,
}

impl Renderer {
    /// Creates an empty renderer cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(test)]
    fn with_cell_host(mut self, cell_host: CellHost) -> Self {
        self.cell_host = cell_host;
        self
    }

    /// Projects state into width-safe rows without mutating application state.
    #[must_use]
    pub fn lines(&mut self, state: &TuiState, width: u16, theme: &Theme) -> Vec<RenderedLine> {
        let width = width.max(1);
        let sections = self.sections(state, width, theme);
        let composer = composer_frame(
            state,
            usize::from(width),
            usize::MAX,
            state.editor.len(),
            theme,
        );
        let status = StatusIndicator::from_state(state, usize::from(width), theme);
        let bottom_pane = bottom_pane_lines(status.as_ref(), &sections.modal, composer);
        sections.all_lines(bottom_pane.lines)
    }

    fn sections(&mut self, state: &TuiState, width: u16, theme: &Theme) -> RenderSections {
        let width = width.max(1);
        self.render_sections(state, usize::from(width), theme)
    }

    /// Renders one complete clipped frame into a Ratatui buffer with the cursor at the editor end.
    pub fn render(&mut self, state: &TuiState, area: Rect, buffer: &mut Buffer, theme: &Theme) {
        self.render_with_cursor(state, area, buffer, theme, state.editor.len());
    }

    /// Renders one complete clipped frame into a Ratatui buffer.
    pub fn render_with_cursor(
        &mut self,
        state: &TuiState,
        area: Rect,
        buffer: &mut Buffer,
        theme: &Theme,
        cursor_byte: usize,
    ) {
        self.render_with_cursor_and_hyperlinks(state, area, buffer, None, theme, cursor_byte);
    }

    pub(crate) fn render_with_cursor_and_hyperlinks(
        &mut self,
        state: &TuiState,
        area: Rect,
        buffer: &mut Buffer,
        hyperlinks: Option<&mut HyperlinkBuffer>,
        theme: &Theme,
        cursor_byte: usize,
    ) {
        if area.width == 0 || area.height == 0 {
            return;
        }
        let cursor_byte =
            if cursor_byte <= state.editor.len() && state.editor.is_char_boundary(cursor_byte) {
                cursor_byte
            } else {
                state.editor.len()
            };
        let area = content_area(area);
        let width = usize::from(area.width);
        let presentation = Presentation::from_state(state);
        let timeline = self.render_timeline(&presentation, width, theme);
        let modal = modal_surface_in_viewport(state, width, theme);
        let composer = composer_frame(state, width, usize::from(area.height), cursor_byte, theme);
        let status = StatusIndicator::from_state(state, width, theme);
        let pane_lines = bottom_pane_lines(status.as_ref(), &modal, composer);
        let bottom_pane = BottomPane::new(&pane_lines.lines, pane_lines.cursor);
        let layout = VerticalLayout::with_bottom_pane(area, bottom_pane.desired_height(area.width));
        let transcript = Transcript::new(&timeline, state.transcript_viewport(), theme);
        if let Some(hyperlinks) = hyperlinks {
            transcript.render_with_hyperlinks(layout.transcript, buffer, hyperlinks);
        } else {
            transcript.render(layout.transcript, buffer);
        }
        bottom_pane.render(layout.bottom_pane, buffer);
    }

    /// Returns the dynamic Codex-style inline viewport height.
    pub(crate) fn inline_height(
        &mut self,
        state: &TuiState,
        width: u16,
        max_height: u16,
        theme: &Theme,
        cursor_byte: usize,
    ) -> u16 {
        let max_height = max_height.max(1);
        let outer_area = Rect::new(0, 0, width.max(1), max_height);
        let area = content_area(outer_area);
        let width = usize::from(area.width);
        let presentation = Presentation::from_state(state);
        let timeline = self.render_live_timeline(&presentation, width, theme);
        let modal = modal_surface_in_viewport(state, width, theme);
        let composer = composer_frame(state, width, usize::from(area.height), cursor_byte, theme);
        let status = StatusIndicator::from_state(state, width, theme);
        let pane_lines = bottom_pane_lines(status.as_ref(), &modal, composer);
        let bottom_pane = BottomPane::new(&pane_lines.lines, pane_lines.cursor);
        let bottom_height = bottom_pane.desired_height(u16::try_from(width).unwrap_or(u16::MAX));
        let timeline_height = u16::try_from(timeline.len()).unwrap_or(u16::MAX);
        let margins = max_height.saturating_sub(area.height);
        timeline_height
            .saturating_add(bottom_height)
            .saturating_add(margins)
            .clamp(MIN_INLINE_VIEWPORT_HEIGHT.min(max_height), max_height)
    }

    /// Renders only active content and the bottom pane into the inline viewport.
    pub(crate) fn render_inline_with_cursor(
        &mut self,
        state: &TuiState,
        area: Rect,
        buffer: &mut Buffer,
        hyperlinks: Option<&mut HyperlinkBuffer>,
        theme: &Theme,
        cursor_byte: usize,
    ) {
        if area.width == 0 || area.height == 0 {
            return;
        }
        let area = content_area(area);
        let width = usize::from(area.width);
        let presentation = Presentation::from_state(state);
        let timeline = self.render_live_timeline(&presentation, width, theme);
        let modal = modal_surface_in_viewport(state, width, theme);
        let composer = composer_frame(state, width, usize::from(area.height), cursor_byte, theme);
        let status = StatusIndicator::from_state(state, width, theme);
        let pane_lines = bottom_pane_lines(status.as_ref(), &modal, composer);
        let bottom_pane = BottomPane::new(&pane_lines.lines, pane_lines.cursor);
        let layout = VerticalLayout::with_bottom_pane(area, bottom_pane.desired_height(area.width));
        let visible_start = timeline
            .len()
            .saturating_sub(usize::from(layout.transcript.height));
        if let Some(hyperlinks) = hyperlinks {
            draw_lines_with_hyperlinks(
                &timeline[visible_start..],
                layout.transcript,
                buffer,
                hyperlinks,
            );
        } else {
            draw_lines(&timeline[visible_start..], layout.transcript, buffer);
        }
        bottom_pane.render(layout.bottom_pane, buffer);
    }

    /// Renders finalized source cells into pre-wrapped terminal rows.
    pub(crate) fn history_lines(
        &mut self,
        cells: &[CellNode],
        width: u16,
        theme: &Theme,
    ) -> Vec<RenderedLine> {
        self.render_cell_nodes(cells.iter(), width, theme)
    }

    fn render_cell_nodes<'a>(
        &mut self,
        cells: impl IntoIterator<Item = &'a CellNode>,
        width: u16,
        theme: &Theme,
    ) -> Vec<RenderedLine> {
        let width = width.max(1);
        let context = CellContext { theme };
        let stylesheet_generation = self.cell_host.stylesheet_generation();
        let host = &self.cell_host;
        let cache = &mut self.cache;
        let blocks = cells
            .into_iter()
            .map(|cell| {
                let key = CellCacheKey::from_cell(cell, width, theme, stylesheet_generation);
                cache.get_or_render(key, cell.content(), cell.tool_call_id(), || {
                    host.render(cell.content().spec(), &context, width)
                })
            })
            .collect();
        CellList::new(blocks).render().into_lines()
    }

    /// Returns the visible display-cell cursor position for the editor.
    #[must_use]
    pub fn cursor_position(
        &mut self,
        state: &TuiState,
        width: u16,
        height: u16,
        cursor_byte: usize,
    ) -> Option<(u16, u16)> {
        let area = content_area(Rect::new(0, 0, width.max(1), height.max(1)));
        let width = usize::from(area.width);
        let height = usize::from(area.height);
        if cursor_byte > state.editor.len() || !state.editor.is_char_boundary(cursor_byte) {
            return None;
        }
        let theme = Theme::default();
        let sections = self.sections(state, u16::try_from(width).unwrap_or(u16::MAX), &theme);
        let composer = composer_frame(state, width, height, cursor_byte, &theme);
        let status = StatusIndicator::from_state(state, width, &theme);
        let pane_lines = bottom_pane_lines(status.as_ref(), &sections.modal, composer);
        let bottom_pane = BottomPane::new(&pane_lines.lines, pane_lines.cursor);
        let layout = VerticalLayout::with_bottom_pane(
            area,
            bottom_pane.desired_height(u16::try_from(width).unwrap_or(u16::MAX)),
        );
        bottom_pane
            .cursor(layout.bottom_pane)
            .map(|cursor| (cursor.x, cursor.y))
    }

    /// Removes all cached wrapped rows, for example after terminal reprobe.
    pub fn invalidate(&mut self) {
        self.cache.clear();
    }

    #[allow(clippy::too_many_lines)] // Keep semantic regions visibly ordered.
    fn render_sections(&mut self, state: &TuiState, width: usize, theme: &Theme) -> RenderSections {
        let presentation = Presentation::from_state(state);
        RenderSections {
            timeline: self.render_timeline(&presentation, width, theme),
            modal: modal_surface_in_viewport(state, width, theme),
        }
    }

    fn render_timeline(
        &mut self,
        presentation: &Presentation,
        width: usize,
        theme: &Theme,
    ) -> Vec<RenderedLine> {
        let cells = presentation
            .history()
            .iter()
            .chain(
                presentation
                    .active()
                    .iter()
                    .filter(|cell| cell.content().is_live_timeline_visible()),
            )
            .chain(presentation.notifications().iter());
        self.render_cell_nodes(cells, u16::try_from(width).unwrap_or(u16::MAX), theme)
    }

    fn render_live_timeline(
        &mut self,
        presentation: &Presentation,
        width: usize,
        theme: &Theme,
    ) -> Vec<RenderedLine> {
        let cells = presentation
            .active()
            .iter()
            .filter(|cell| cell.content().is_live_timeline_visible())
            .chain(presentation.notifications().iter());
        self.render_cell_nodes(cells, u16::try_from(width).unwrap_or(u16::MAX), theme)
    }
}

fn modal_surface(state: &TuiState, width: usize, theme: &Theme) -> Vec<RenderedLine> {
    if let Some(approval) = approval_panel(state, width, theme) {
        return approval;
    }
    let Some(overlay) = &state.overlay else {
        return Vec::new();
    };
    match overlay {
        Overlay::Selector(selector) => {
            let mut rows = Vec::new();
            let query = selector.query();
            let heading = if query.is_empty() {
                selector.title().to_owned()
            } else {
                format!("{} · {query}", selector.title())
            };
            push_wrapped(&mut rows, "select", &heading, width, theme.footer, theme);
            let items = selector.visible_items();
            let selected = selector
                .selected_label()
                .and_then(|label| items.iter().position(|item| item.label() == label))
                .unwrap_or(0);
            let range = modal_option_range(items.len(), selected);
            for item in &items[range] {
                let marker = if selector.selected_label() == Some(item.label()) {
                    "›"
                } else {
                    " "
                };
                push_wrapped(&mut rows, marker, item.label(), width, theme.editor, theme);
            }
            rows
        }
        Overlay::CommandCompletion(completion) => {
            let mut rows = Vec::new();
            let skill_completion = completion
                .options()
                .first()
                .is_some_and(|option| option.value().starts_with('$'));
            push_wrapped(
                &mut rows,
                if skill_completion {
                    "skills"
                } else {
                    "completions"
                },
                if skill_completion {
                    "select skill"
                } else {
                    "select item"
                },
                width,
                theme.footer,
                theme,
            );
            let selected = completion
                .selected()
                .and_then(|selected| {
                    completion
                        .options()
                        .iter()
                        .position(|option| option == selected)
                })
                .unwrap_or(0);
            let range = modal_option_range(completion.options().len(), selected);
            let visible_options = &completion.options()[range];
            let name_width = visible_options
                .iter()
                .map(|option| option.value().width())
                .max()
                .unwrap_or(0);
            for option in visible_options {
                let marker = if completion.selected() == Some(option) {
                    "›"
                } else {
                    " "
                };
                let padding = " ".repeat(name_width.saturating_sub(option.value().width()));
                let description = option
                    .description()
                    .map_or_else(String::new, |description| format!(" {description}"));
                let label = format!(
                    "{}{padding}  [{}]{description}",
                    option.value(),
                    option.kind().label()
                );
                push_truncated(&mut rows, marker, &label, width, theme.editor, theme);
            }
            rows
        }
    }
}

fn modal_option_range(length: usize, selected: usize) -> std::ops::Range<usize> {
    let start = selected
        .saturating_sub(MAX_BOTTOM_MODAL_OPTIONS / 2)
        .min(length.saturating_sub(MAX_BOTTOM_MODAL_OPTIONS));
    start..start.saturating_add(MAX_BOTTOM_MODAL_OPTIONS).min(length)
}

fn composer_frame(
    state: &TuiState,
    width: usize,
    frame_height: usize,
    cursor_byte: usize,
    theme: &Theme,
) -> ComposerFrame {
    let width = width.max(1);
    let cursor_byte = cursor_byte.min(state.editor.len());
    let attachment_lines = composer_attachment_lines(state, width, theme);
    let attachment_rows = attachment_lines.len();
    let editor_lines = composer_content_lines(
        &state.editor,
        state.editor_skill_mention(),
        composer_content_width(width),
        theme.composer,
        theme,
    );
    let cursor_lines = composer_content_lines(
        &state.editor[..cursor_byte],
        None,
        composer_content_width(width),
        theme.composer,
        theme,
    );
    let cursor_row = cursor_lines.len().saturating_sub(1);
    let max_editor_rows = max_composer_editor_rows(frame_height, attachment_rows);
    let start = cursor_row
        .saturating_add(1)
        .saturating_sub(max_editor_rows)
        .min(editor_lines.len().saturating_sub(max_editor_rows));
    let end = start
        .saturating_add(max_editor_rows)
        .min(editor_lines.len());
    let mut lines = Vec::with_capacity(
        COMPOSER_VERTICAL_PADDING_ROWS
            .saturating_mul(2)
            .saturating_add(attachment_rows)
            .saturating_add(end.saturating_sub(start))
            .saturating_add(1),
    );
    lines.extend(attachment_lines);
    for _ in 0..COMPOSER_VERTICAL_PADDING_ROWS {
        lines.push(composer_blank_line(width, theme.composer));
    }
    if state.editor.is_empty() {
        lines.push(composer_placeholder_line(
            width,
            theme.composer.patch(theme.footer),
        ));
    } else {
        lines.extend(
            editor_lines[start..end]
                .iter()
                .map(|line| fill_composer_line(line, width)),
        );
    }
    for _ in 0..COMPOSER_VERTICAL_PADDING_ROWS {
        lines.push(composer_blank_line(width, theme.composer));
    }
    lines.push(footer_line(state, width, theme.footer));

    let column = cursor_lines
        .last()
        .map_or(0, |line| line.text.width())
        .min(composer_content_width(width));
    ComposerFrame {
        lines,
        cursor: Some(CursorState {
            x: u16::try_from(column.min(width.saturating_sub(1))).unwrap_or(u16::MAX),
            y: u16::try_from(
                attachment_rows
                    .saturating_add(COMPOSER_VERTICAL_PADDING_ROWS)
                    .saturating_add(cursor_row.saturating_sub(start)),
            )
            .unwrap_or(u16::MAX),
        }),
    }
}

fn max_composer_editor_rows(frame_height: usize, attachment_rows: usize) -> usize {
    let frame_limit = frame_height.saturating_sub(
        COMPOSER_VERTICAL_PADDING_ROWS
            .saturating_mul(2)
            .saturating_add(attachment_rows)
            .saturating_add(1),
    );
    frame_limit.clamp(1, MAX_COMPOSER_EDITOR_ROWS)
}

fn composer_attachment_lines(state: &TuiState, width: usize, theme: &Theme) -> Vec<RenderedLine> {
    state
        .attachments()
        .iter()
        .enumerate()
        .map(|(index, attachment)| {
            let index = index + 1;
            let size = format_byte_count(attachment.decoded_bytes());
            let full = format!(
                "{index}. {} · {} · {size}",
                attachment.display_name(),
                attachment.mime_type()
            );
            let core = format!("{index}. {} {size}", attachment.mime_type());
            let compact = format!(
                "{index} {} {}",
                attachment.mime_type(),
                format_compact_byte_count(attachment.decoded_bytes())
            );
            let text = if full.width() <= width {
                full
            } else if core.width() <= width {
                core
            } else {
                compact
            };
            fill_composer_line(
                &RenderedLine::new(
                    truncate_cells(&text, width),
                    theme.composer.patch(theme.footer),
                ),
                width,
            )
        })
        .collect()
}

fn composer_blank_line(width: usize, style: Style) -> RenderedLine {
    RenderedLine::new(" ".repeat(width.max(1)), style)
}

fn composer_placeholder_line(width: usize, style: Style) -> RenderedLine {
    fill_composer_line(
        &RenderedLine::new(
            truncate_cells(
                &format!("{COMPOSER_PROMPT_PREFIX}Ask Tea to do anything"),
                width,
            ),
            style,
        ),
        width,
    )
}

fn fill_composer_line(line: &RenderedLine, width: usize) -> RenderedLine {
    let padding = " ".repeat(width.saturating_sub(line.text.width()));
    if line.rendered_spans().is_empty() {
        return RenderedLine::new(format!("{}{padding}", line.text), line.style);
    }
    let mut spans = line.rendered_spans().to_vec();
    if !padding.is_empty() {
        spans.push(RenderedSpan::new(padding, line.style));
    }
    RenderedLine::from_spans(line.style, spans)
}

fn composer_content_lines(
    text: &str,
    skill_mention: Option<&str>,
    width: usize,
    style: Style,
    theme: &Theme,
) -> Vec<RenderedLine> {
    let mut lines = component_lines(
        Text::new(text, style).with_prefixes(COMPOSER_PROMPT_PREFIX, "  "),
        width,
        theme,
    );
    let Some(mention) = skill_mention.filter(|mention| text.starts_with(*mention)) else {
        return lines;
    };
    let mention_style = style.patch(theme.editor);
    let mut remaining = mention.len();
    for (index, line) in lines.iter_mut().enumerate() {
        if remaining == 0 {
            break;
        }
        let prefix = if index == 0 {
            COMPOSER_PROMPT_PREFIX
        } else {
            "  "
        };
        let Some(content) = line.text().strip_prefix(prefix) else {
            break;
        };
        let highlighted_bytes = remaining.min(content.len());
        let highlighted = &content[..highlighted_bytes];
        let rest = &content[highlighted_bytes..];
        let mut spans = vec![
            RenderedSpan::new(prefix.to_owned(), style),
            RenderedSpan::new(highlighted.to_owned(), mention_style),
        ];
        if !rest.is_empty() {
            spans.push(RenderedSpan::new(rest.to_owned(), style));
        }
        *line = RenderedLine::from_spans(style, spans);
        remaining -= highlighted_bytes;
    }
    lines
}

const fn composer_content_width(width: usize) -> usize {
    width
}

#[allow(clippy::too_many_lines)] // Approval details remain ordered for policy review.
fn approval_panel(state: &TuiState, width: usize, theme: &Theme) -> Option<Vec<RenderedLine>> {
    let approval = state.approval.as_ref()?;
    if width < 8 {
        let mut rows = Vec::new();
        push_wrapped(
            &mut rows,
            "approval",
            &format!("{} [{}]", approval.tool_name, approval.target),
            width,
            theme.approval,
            theme,
        );
        return Some(rows);
    }

    let inner_width = width.saturating_sub(4);
    let mut content = Vec::new();
    push_truncated(
        &mut content,
        "approval",
        "approval required",
        inner_width,
        theme.approval,
        theme,
    );
    push_wrapped(
        &mut content,
        "tool",
        &approval.tool_name,
        inner_width,
        theme.approval,
        theme,
    );
    push_wrapped(
        &mut content,
        "reason",
        &approval.reason,
        inner_width,
        theme.approval,
        theme,
    );
    push_wrapped(
        &mut content,
        "target",
        &approval.target,
        inner_width,
        theme.approval,
        theme,
    );
    push_wrapped(
        &mut content,
        "effects",
        &approval.effects.join(", "),
        inner_width,
        theme.approval,
        theme,
    );
    push_wrapped(
        &mut content,
        "resources",
        &approval.resources.join(", "),
        inner_width,
        theme.approval,
        theme,
    );
    push_wrapped(
        &mut content,
        "expires",
        &approval.expires_at.to_string(),
        inner_width,
        theme.approval,
        theme,
    );
    let arguments = truncate_cells_with_ellipsis(
        &approval.arguments,
        MAX_APPROVAL_ARGUMENT_CELLS.min(inner_width.saturating_mul(3)),
    );
    push_wrapped(
        &mut content,
        "arguments",
        &arguments,
        inner_width,
        theme.approval,
        theme,
    );
    if state.approval_submitting {
        push_wrapped(
            &mut content,
            "decision",
            &format!("{} [submitting]", state.approval_choice.label()),
            inner_width,
            theme.approval,
            theme,
        );
    } else {
        for choice in [
            ApprovalChoice::AllowOnce,
            ApprovalChoice::AllowSession,
            ApprovalChoice::Deny,
        ] {
            let marker = if choice == state.approval_choice {
                ">"
            } else {
                " "
            };
            push_wrapped(
                &mut content,
                marker,
                &format!("{} ({})", choice.label(), approval_scope(choice)),
                inner_width,
                theme.approval,
                theme,
            );
        }
    }

    Some(box_panel(content, width, theme.approval))
}

pub(crate) fn workspace_trust_panel(
    workspace: &Path,
    width: usize,
    trust_selected: bool,
    theme: &Theme,
) -> Vec<RenderedLine> {
    let inner_width = if width < 8 {
        width
    } else {
        width.saturating_sub(4)
    };
    let mut content = Vec::new();
    push_truncated(
        &mut content,
        "trust",
        "workspace trust required",
        inner_width,
        theme.approval,
        theme,
    );
    push_wrapped(
        &mut content,
        "folder",
        &workspace.display().to_string(),
        inner_width,
        theme.approval,
        theme,
    );
    push_wrapped(
        &mut content,
        "security",
        "Tea will be able to read, edit, and execute files in this folder.",
        inner_width,
        theme.approval,
        theme,
    );
    push_wrapped(
        &mut content,
        "",
        "Only continue if you trust the files and their authors.",
        inner_width,
        theme.approval,
        theme,
    );
    for (is_trust, label) in [(true, "Yes, trust this folder"), (false, "No, exit")] {
        let marker = if is_trust == trust_selected { ">" } else { " " };
        push_wrapped(
            &mut content,
            marker,
            label,
            inner_width,
            theme.approval,
            theme,
        );
    }
    box_panel(content, width, theme.approval)
}

const fn approval_scope(choice: ApprovalChoice) -> &'static str {
    match choice {
        ApprovalChoice::AllowOnce | ApprovalChoice::Deny => "this call",
        ApprovalChoice::AllowSession => "matching resources this session",
    }
}

fn box_panel(content: Vec<RenderedLine>, width: usize, style: Style) -> Vec<RenderedLine> {
    if width < 8 {
        return content;
    }
    let inner_width = width - 4;
    let mut panel = Vec::with_capacity(content.len() + 2);
    panel.push(RenderedLine::new(
        format!("+{}+", "-".repeat(width - 2)),
        style,
    ));
    for line in content {
        let padding = " ".repeat(inner_width.saturating_sub(line.text.width()));
        panel.push(RenderedLine::new(
            format!("| {}{padding} |", line.text),
            line.style,
        ));
    }
    panel.push(RenderedLine::new(
        format!("+{}+", "-".repeat(width - 2)),
        style,
    ));
    panel
}

fn footer_line(state: &TuiState, width: usize, style: Style) -> RenderedLine {
    let model = state.model_id().map_or("default", ModelIdText::text);
    let text = format!(
        "{model} · {} · {}",
        state.displayed_reasoning_effort(),
        state.startup.workspace()
    );
    let margin = COMPOSER_PROMPT_PREFIX.width().min(width.saturating_sub(1));
    let text = truncate_cells(&text, width.saturating_sub(margin).max(1));
    RenderedLine::new(format!("{}{text}", " ".repeat(margin)), style)
}

trait ModelIdText {
    fn text(&self) -> &str;
}

impl ModelIdText for ModelId {
    fn text(&self) -> &str {
        self.as_str()
    }
}

fn component_lines(component: impl Component, width: usize, theme: &Theme) -> Vec<RenderedLine> {
    let context = RenderContext { theme };
    Element::new(component)
        .render(&context, u16::try_from(width).unwrap_or(u16::MAX).max(1))
        .into_lines()
}

fn push_wrapped(
    output: &mut Vec<RenderedLine>,
    label: &str,
    text: &str,
    width: usize,
    style: Style,
    theme: &Theme,
) {
    output.extend(component_lines(
        Text::new(text, style).with_label(label),
        width,
        theme,
    ));
}

fn push_truncated(
    output: &mut Vec<RenderedLine>,
    label: &str,
    text: &str,
    width: usize,
    style: Style,
    theme: &Theme,
) {
    output.extend(component_lines(
        Text::truncated(text, style).with_label(label),
        width,
        theme,
    ));
}

#[cfg(test)]
mod tests {
    use std::fmt::Write as _;

    use super::{
        CellCacheKey, CellRenderCache, LifecycleStatus, OutputFormat, PlanStep, PlanStepStatus,
        RenderedLine, RenderedSpan, Renderer, TimelineDetail, TimelineDetailKind, TimelineSource,
        component_lines,
    };
    use crate::tui::cells::{
        CellBlock, CellContext, CellHost, CellList, CellStylePatch, CellStyleSheet, InsetsPatch,
    };
    use crate::tui::components::Markdown;
    use crate::tui::components::style::Insets;
    use crate::tui::hyperlink::HyperlinkBuffer;
    use crate::tui::layout::draw_lines_with_hyperlinks;
    use crate::tui::presentation::{
        CellContent, CellId, CellLane, CellNode, DecisionCell, DecisionStatus, DiffCell,
        LifecycleCell, LifecycleKind, MessageAuthor, MessageCell, NoticeCell, NoticeKind,
        NoticeSeverity, PlanCell, QueuedInputCell, QueuedInputKind, ReasoningCell, ResultCell,
        SourcesCell,
    };
    use crate::tui::{TerminalCapabilities, Theme};
    use ratatui::buffer::Buffer;
    use ratatui::layout::{Position, Rect};
    use ratatui::style::{Color, Modifier, Style};
    use tea_protocol::{CodeChange, CodeChangeKind};
    use unicode_width::UnicodeWidthStr as _;

    fn cache_key(index: u16) -> CellCacheKey {
        CellCacheKey {
            id: CellId::Synthetic {
                lane: CellLane::History,
                index,
            },
            width: 80,
            theme_generation: 0,
            stylesheet_generation: 0,
        }
    }

    fn test_cell(content: CellContent) -> CellNode {
        test_cell_at(0, content)
    }

    fn test_cell_at(index: u16, content: CellContent) -> CellNode {
        CellNode::new(
            CellId::Synthetic {
                lane: CellLane::History,
                index,
            },
            content,
            None,
        )
    }

    fn test_cell_with_owner(content: CellContent, owner: tea_protocol::ToolCallId) -> CellNode {
        test_cell_at_with_owner(0, content, owner)
    }

    fn test_cell_at_with_owner(
        index: u16,
        content: CellContent,
        owner: tea_protocol::ToolCallId,
    ) -> CellNode {
        CellNode::new(
            CellId::Synthetic {
                lane: CellLane::History,
                index,
            },
            content,
            Some(owner),
        )
    }

    fn render_test_node(cell: &CellNode, width: usize, theme: &Theme) -> Vec<RenderedLine> {
        let host = CellHost::new(CellStyleSheet::default().with_global_patch(CellStylePatch {
            margin: InsetsPatch {
                right: Some(0),
                left: Some(0),
                ..InsetsPatch::default()
            },
            ..CellStylePatch::default()
        }));
        let context = CellContext { theme };
        let block = host.render(
            cell.content().spec(),
            &context,
            u16::try_from(width).unwrap_or(u16::MAX).max(1),
        );
        CellList::new(vec![block]).render().into_lines()
    }

    fn message(author: MessageAuthor, source: &str, format: OutputFormat) -> CellContent {
        CellContent::Message(MessageCell::new(author, source, format))
    }

    fn result(
        action: &str,
        source_name: Option<&str>,
        content: &str,
        format: OutputFormat,
        is_error: bool,
    ) -> CellContent {
        CellContent::Result(ResultCell::new(
            action,
            source_name,
            content,
            format,
            is_error,
        ))
    }

    #[test]
    fn stable_cell_cache_reuses_matching_blocks_and_stays_bounded() {
        let mut cache = CellRenderCache::default();
        let key = cache_key(0);
        let content = message(
            MessageAuthor::Assistant,
            "stable response",
            OutputFormat::Markdown,
        );
        let expected = CellBlock::new(
            vec![RenderedLine::new(
                "stable response".to_owned(),
                Style::default(),
            )],
            Insets::default(),
        );

        assert_eq!(
            cache.get_or_render(key.clone(), &content, None, || expected.clone()),
            expected
        );
        assert_eq!(
            cache.get_or_render(key, &content, None, || {
                panic!("matching cell must use cached rows")
            }),
            expected
        );
        assert_eq!(cache.hits, 1);
        assert_eq!(cache.misses, 1);

        for index in 0..=super::MAX_CELL_RENDER_CACHE_ENTRIES {
            let text = index.to_string();
            let content = message(MessageAuthor::Assistant, &text, OutputFormat::Markdown);
            let _ = cache.get_or_render(
                cache_key(u16::try_from(index).unwrap_or(u16::MAX)),
                &content,
                None,
                CellBlock::empty,
            );
        }
        assert!(cache.entries.len() <= super::MAX_CELL_RENDER_CACHE_ENTRIES);
    }

    #[test]
    fn stable_cell_cache_keys_width_theme_and_stylesheet_generations() {
        let theme = Theme::default();
        let cell = test_cell(message(
            MessageAuthor::Assistant,
            "A stable response that wraps differently when the viewport becomes narrow.",
            OutputFormat::Plain,
        ));
        let mut renderer = Renderer::new();

        let wide = renderer.history_lines(std::slice::from_ref(&cell), 80, &theme);
        let cached = renderer.history_lines(std::slice::from_ref(&cell), 80, &theme);
        assert_eq!(wide, cached);
        assert_eq!((renderer.cache.hits, renderer.cache.misses), (1, 1));

        let narrow = renderer.history_lines(std::slice::from_ref(&cell), 20, &theme);
        assert_ne!(wide, narrow);
        assert_eq!((renderer.cache.hits, renderer.cache.misses), (1, 2));

        let mut changed_theme = theme;
        changed_theme.generation = changed_theme.generation.saturating_add(100);
        let themed = renderer.history_lines(std::slice::from_ref(&cell), 20, &changed_theme);
        assert_eq!(narrow, themed);
        assert_eq!((renderer.cache.hits, renderer.cache.misses), (1, 3));

        let spaced_host =
            CellHost::new(CellStyleSheet::default().with_global_patch(CellStylePatch {
                margin: InsetsPatch {
                    top: Some(1),
                    ..InsetsPatch::default()
                },
                ..CellStylePatch::default()
            }));
        let stylesheet_generation = spaced_host.stylesheet_generation();
        renderer = renderer.with_cell_host(spaced_host);
        let spaced = renderer.history_lines(std::slice::from_ref(&cell), 20, &changed_theme);

        assert_ne!(themed, spaced);
        assert_eq!((renderer.cache.hits, renderer.cache.misses), (1, 4));
        assert!(
            renderer
                .cache
                .entries
                .keys()
                .any(|key| key.stylesheet_generation == stylesheet_generation)
        );
        assert_eq!(
            spaced,
            renderer.history_lines(std::slice::from_ref(&cell), 20, &changed_theme)
        );
        assert_eq!((renderer.cache.hits, renderer.cache.misses), (2, 4));
    }

    #[test]
    fn stable_cell_cache_guards_content_owner_role_and_state() {
        let theme = Theme::default();
        let owner_a = "0195a0b1-7e00-7000-8000-000000000001".parse().unwrap();
        let owner_b = "0195a0b1-7e00-7000-8000-000000000002".parse().unwrap();
        let original = test_cell_at_with_owner(
            4,
            message(MessageAuthor::Assistant, "same", OutputFormat::Plain),
            owner_a,
        );
        let mut renderer = Renderer::new();

        let first = renderer.history_lines(std::slice::from_ref(&original), 40, &theme);
        assert_eq!(
            first,
            renderer.history_lines(std::slice::from_ref(&original), 40, &theme)
        );

        let changed_content = test_cell_at_with_owner(
            4,
            message(MessageAuthor::Assistant, "changed", OutputFormat::Plain),
            owner_a,
        );
        let content_output =
            renderer.history_lines(std::slice::from_ref(&changed_content), 40, &theme);
        assert_ne!(first, content_output);

        let changed_owner = test_cell_at_with_owner(
            4,
            message(MessageAuthor::Assistant, "changed", OutputFormat::Plain),
            owner_b,
        );
        let owner_output = renderer.history_lines(std::slice::from_ref(&changed_owner), 40, &theme);

        let changed_role = test_cell_at_with_owner(
            4,
            CellContent::Notice(NoticeCell::new(
                NoticeKind::General,
                NoticeSeverity::Information,
                "changed",
                None,
            )),
            owner_b,
        );
        let role_output = renderer.history_lines(std::slice::from_ref(&changed_role), 40, &theme);
        assert_ne!(owner_output, role_output);

        let changed_state = test_cell_at_with_owner(
            4,
            CellContent::Notice(NoticeCell::new(
                NoticeKind::General,
                NoticeSeverity::Warning,
                "changed",
                None,
            )),
            owner_b,
        );
        let state_output = renderer.history_lines(std::slice::from_ref(&changed_state), 40, &theme);
        assert_ne!(role_output, state_output);
        assert_eq!(renderer.cache.entries.len(), 1);
        assert_eq!((renderer.cache.hits, renderer.cache.misses), (1, 5));

        assert_eq!(
            state_output,
            renderer.history_lines(std::slice::from_ref(&changed_state), 40, &theme)
        );
        assert_eq!((renderer.cache.hits, renderer.cache.misses), (2, 5));
    }

    #[test]
    fn stable_cell_cache_rejects_active_tick_expansion_and_content_changes() {
        let theme = Theme::default();
        let lifecycle = |action: &str, expanded: bool, tick: u64| {
            test_cell_at(
                7,
                CellContent::Lifecycle(LifecycleCell::new(
                    LifecycleKind::ToolCall,
                    action,
                    Some("src/lib.rs"),
                    LifecycleStatus::Running,
                    vec![TimelineDetail::new(
                        TimelineDetailKind::Metadata,
                        Some("phase"),
                        "compiling",
                    )],
                    expanded,
                    tick,
                )),
            )
        };
        let mut renderer = Renderer::new();

        let tick_zero = lifecycle("Building", false, 0);
        let first = renderer.history_lines(std::slice::from_ref(&tick_zero), 40, &theme);
        let tick_one = lifecycle("Building", false, 1);
        let tick_output = renderer.history_lines(std::slice::from_ref(&tick_one), 40, &theme);
        assert_ne!(first, tick_output);

        let expanded = lifecycle("Building", true, 1);
        let expanded_output = renderer.history_lines(std::slice::from_ref(&expanded), 40, &theme);
        assert_ne!(tick_output, expanded_output);
        assert!(
            expanded_output
                .iter()
                .any(|line| line.text().contains("compiling"))
        );

        let changed = lifecycle("Linking", true, 1);
        let changed_output = renderer.history_lines(std::slice::from_ref(&changed), 40, &theme);
        assert_ne!(expanded_output, changed_output);
        assert_eq!((renderer.cache.hits, renderer.cache.misses), (0, 4));
        assert_eq!(renderer.cache.entries.len(), 1);

        assert_eq!(
            changed_output,
            renderer.history_lines(std::slice::from_ref(&changed), 40, &theme)
        );
        assert_eq!((renderer.cache.hits, renderer.cache.misses), (1, 4));
    }

    #[test]
    fn stable_cell_cache_never_aliases_distinct_ids_with_identical_text() {
        let theme = Theme::default();
        let cells = vec![
            test_cell_at(
                8,
                message(MessageAuthor::Assistant, "identical", OutputFormat::Plain),
            ),
            test_cell_at(
                9,
                message(MessageAuthor::Assistant, "identical", OutputFormat::Plain),
            ),
        ];
        let mut renderer = Renderer::new();

        let first = renderer.history_lines(&cells, 40, &theme);
        assert_eq!(renderer.cache.entries.len(), 2);
        assert_eq!((renderer.cache.hits, renderer.cache.misses), (0, 2));
        assert_eq!(first, renderer.history_lines(&cells, 40, &theme));
        assert_eq!((renderer.cache.hits, renderer.cache.misses), (2, 2));
    }

    #[test]
    fn message_cells_end_with_one_default_spacing_row_while_tool_rows_stay_compact() {
        let theme = Theme::default();
        let render_message = |author, source: &str| {
            render_test_node(
                &test_cell(message(author, source, OutputFormat::Plain)),
                24,
                &theme,
            )
        };

        let user = render_message(MessageAuthor::User, "first request");
        let assistant = render_message(MessageAuthor::Assistant, "first response");

        for lines in [&user, &assistant] {
            assert_eq!(lines.last().unwrap().text(), "");
            assert_eq!(lines.last().unwrap().style(), Style::default());
            assert_ne!(lines[lines.len() - 2].text(), "");
            assert!(lines.iter().all(|line| line.text().width() <= 24));
        }

        let tool = render_test_node(
            &test_cell(CellContent::Lifecycle(LifecycleCell::new(
                LifecycleKind::ToolCall,
                "Read",
                Some("README.md"),
                LifecycleStatus::Succeeded,
                Vec::new(),
                false,
                0,
            ))),
            24,
            &theme,
        );
        assert_ne!(tool.last().unwrap().text(), "");
    }

    #[test]
    #[allow(clippy::too_many_lines)]
    fn mixed_cell_list_uses_the_component_pipeline() {
        let theme = Theme::default();
        let width = 24;
        let cells = [
            test_cell_at(
                0,
                message(MessageAuthor::User, "ship it", OutputFormat::Plain),
            ),
            test_cell_at(
                1,
                message(MessageAuthor::Assistant, "", OutputFormat::Plain),
            ),
            test_cell_at(
                2,
                message(
                    MessageAuthor::Assistant,
                    "See [docs](https://x.test).",
                    OutputFormat::Markdown,
                ),
            ),
            test_cell_at(
                3,
                CellContent::Lifecycle(LifecycleCell::new(
                    LifecycleKind::ToolCall,
                    "Read",
                    Some("README.md"),
                    LifecycleStatus::Succeeded,
                    Vec::new(),
                    false,
                    0,
                )),
            ),
            test_cell_at(
                4,
                result(
                    "Returned",
                    Some("cargo test"),
                    "1 passed",
                    OutputFormat::Plain,
                    false,
                ),
            ),
            test_cell_at(
                5,
                CellContent::Diff(DiffCell::new(
                    "Updated",
                    CodeChange::new(
                        "src/lib.rs",
                        CodeChangeKind::Update,
                        Vec::new(),
                        false,
                        None,
                        None,
                        None,
                    )
                    .unwrap(),
                )),
            ),
        ];
        let mut renderer = Renderer::new();

        let output = renderer.render_cell_nodes(cells.iter(), width, &theme);
        let user = output
            .iter()
            .find(|line| line.text().contains("› ship it"))
            .unwrap();
        let assistant = output
            .iter()
            .find(|line| line.text().contains("docs"))
            .unwrap();
        let lifecycle = output
            .iter()
            .find(|line| line.text().contains("Read README.md"))
            .unwrap();
        let result = output
            .iter()
            .find(|line| line.text().contains("Returned cargo test"))
            .unwrap();
        let diff_body = output
            .iter()
            .find(|line| line.text().contains("(no textual change)"))
            .unwrap();
        let links = output
            .iter()
            .flat_map(super::super::render_output::RenderedLine::rendered_spans)
            .filter_map(|span| span.link())
            .collect::<Vec<_>>();

        assert!(user.text().starts_with('›'));
        assert_eq!(user.text().width(), usize::from(width));
        assert!(assistant.text().starts_with(' '));
        assert!(lifecycle.text().starts_with(' '));
        assert!(result.text().starts_with(' '));
        assert_eq!(diff_body.text(), "(no textual change)");
        assert_eq!(
            output.iter().filter(|line| line.text().is_empty()).count(),
            2
        );
        assert!(
            output
                .windows(2)
                .all(|rows| !(rows[0].text().is_empty() && rows[1].text().is_empty()))
        );
        assert!(!links.is_empty());
        assert!(
            links
                .iter()
                .all(|destination| *destination == "https://x.test/")
        );
        assert!(
            output
                .iter()
                .all(|line| line.text().width() <= usize::from(width)),
            "{:?}",
            output
                .iter()
                .map(|line| (line.text(), line.text().width()))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn markdown_structure_survives_no_color_without_becoming_a_row_identity() {
        let theme = Theme::for_capabilities(TerminalCapabilities::from_environment(
            Some("dumb"),
            None,
            false,
            false,
        ));
        let output = component_lines(
            Markdown::new("## Result\n\n> quoted\n\n```rust\nlet ready = true;\n```"),
            80,
            &theme,
        );
        let text = output.iter().map(RenderedLine::text).collect::<Vec<_>>();

        assert_eq!(text, ["## Result", "", "> quoted", "", "let ready = true;"]);
        assert!(output[0].style().add_modifier.contains(Modifier::BOLD));
        assert_eq!(output[2].style(), theme.markdown_quote);
        assert_eq!(output[4].style(), theme.markdown_code);
    }

    #[test]
    fn fenced_syntax_highlighting_is_color_gated_and_cache_stable() {
        let source = "```rust\nfn main() { let answer = 42; }\n```";
        let render = |theme: &Theme| component_lines(Markdown::new(source), 80, theme);
        let color_theme = Theme::default();
        let colored = render(&color_theme);
        let no_color_theme = Theme::for_capabilities(TerminalCapabilities::from_environment(
            Some("dumb"),
            None,
            false,
            false,
        ));
        let plain = render(&no_color_theme);

        assert_eq!(colored[0].text(), "fn main() { let answer = 42; }");
        assert_eq!(colored[0].text(), plain[0].text());
        assert!(
            colored[0]
                .spans
                .iter()
                .any(|span| span.style != color_theme.markdown_code)
        );
        assert!(
            plain[0]
                .spans
                .iter()
                .all(|span| span.style == no_color_theme.markdown_code)
        );
        let ansi16_theme = Theme::for_capabilities(TerminalCapabilities::from_environment(
            Some("xterm"),
            None,
            false,
            false,
        ));
        assert!(
            render(&ansi16_theme)[0]
                .spans
                .iter()
                .all(|span| { !matches!(span.style.fg, Some(Color::Indexed(_) | Color::Rgb(..))) })
        );
        let ansi256_theme = Theme::for_capabilities(TerminalCapabilities::from_environment(
            Some("xterm-256color"),
            None,
            false,
            false,
        ));
        let ansi256 = render(&ansi256_theme);
        assert!(
            ansi256[0]
                .spans
                .iter()
                .all(|span| !matches!(span.style.fg, Some(Color::Rgb(..))))
        );
        assert!(
            ansi256[0]
                .spans
                .iter()
                .any(|span| matches!(span.style.fg, Some(Color::Indexed(_))))
        );

        let cell = test_cell(message(
            MessageAuthor::Assistant,
            source,
            OutputFormat::Markdown,
        ));
        let mut renderer = Renderer::new();
        let first = renderer.history_lines(std::slice::from_ref(&cell), 80, &color_theme);
        let cached = renderer.history_lines(&[cell], 80, &color_theme);
        assert_eq!(first, cached);
        assert_eq!(renderer.cache.hits, 1);
    }

    #[test]
    fn inline_markdown_styles_reach_terminal_cells_without_source_markers() {
        let theme = Theme::default();
        let output = component_lines(
            Markdown::new("**bold** *italic* ~~old~~ `tea-cli` [docs](https://e.test)"),
            80,
            &theme,
        );
        assert_eq!(
            output[0].text(),
            "bold italic old tea-cli docs (https://e.test/)"
        );
        assert!(
            !output[0]
                .text()
                .chars()
                .any(|character| matches!(character, '`' | '*'))
        );

        let area = Rect::new(0, 0, 80, 1);
        let mut buffer = Buffer::empty(area);
        crate::tui::layout::draw_lines(&output, area, &mut buffer);
        assert!(buffer[(0, 0)].modifier.contains(Modifier::BOLD));
        assert!(buffer[(5, 0)].modifier.contains(Modifier::ITALIC));
        assert!(buffer[(12, 0)].modifier.contains(Modifier::CROSSED_OUT));
        assert_eq!(Some(buffer[(16, 0)].fg), theme.markdown_inline_code.fg);
        assert!(buffer[(24, 0)].modifier.contains(Modifier::UNDERLINED));
    }

    #[test]
    fn validated_link_metadata_reaches_rendered_rows_but_not_ratatui_text() {
        let theme = Theme::default();
        let output = component_lines(
            Markdown::new("Open [Tea](HTTPS://Example.COM:443/a/../docs) now."),
            24,
            &theme,
        );

        let links = output
            .iter()
            .flat_map(|line| &line.spans)
            .filter_map(|span| span.link.as_deref())
            .collect::<Vec<_>>();
        assert!(!links.is_empty());
        assert!(links.iter().all(|link| *link == "https://example.com/docs"));
        let terminal_text = output
            .iter()
            .flat_map(|line| line.as_ratatui_line().spans)
            .map(|span| span.content.into_owned())
            .collect::<String>();
        assert!(!terminal_text.contains('\u{1b}'));
        assert!(!terminal_text.contains("]8;"));
    }

    #[test]
    fn hyperlink_sidecar_tracks_wide_cells_and_clips_with_the_visible_line() {
        let theme = Theme::default();
        let output = component_lines(Markdown::new("[界A](https://example.test/)"), 80, &theme);
        let area = Rect::new(3, 4, 3, 1);
        let mut buffer = Buffer::empty(area);
        let mut hyperlinks = HyperlinkBuffer::empty(area);

        draw_lines_with_hyperlinks(&output, area, &mut buffer, &mut hyperlinks);

        assert_eq!(buffer[(3, 4)].symbol(), "界");
        assert_eq!(buffer[(5, 4)].symbol(), "A");
        for x in 3..=5 {
            assert_eq!(
                hyperlinks.get(Position::new(x, 4)).map(AsRef::as_ref),
                Some("https://example.test/")
            );
        }
        assert_eq!(hyperlinks.get(Position::new(6, 4)), None);
    }

    #[test]
    fn hosted_sources_wrap_labels_and_retain_normalized_hyperlink_sidecar_metadata() {
        let source = tea_protocol::ExternalSource::new(
            "https://Example.COM:443/a/../docs?section=hosted-search",
        )
        .unwrap()
        .with_title("Hosted search architecture reference")
        .unwrap();
        let cell = test_cell(CellContent::Sources(SourcesCell::new(vec![
            TimelineSource::from_external(&source),
        ])));
        let theme = Theme::default();
        let output = render_test_node(&cell, 20, &theme);

        assert!(output.iter().all(|line| line.text().width() <= 20));
        let links = output
            .iter()
            .flat_map(|line| line.spans.iter())
            .filter_map(|span| span.link.as_deref())
            .collect::<Vec<_>>();
        assert!(!links.is_empty());
        assert!(links.iter().all(|destination| {
            *destination == "https://example.com/docs?section=hosted-search"
        }));

        let area = Rect::new(0, 0, 20, u16::try_from(output.len()).unwrap());
        let mut buffer = Buffer::empty(area);
        let mut hyperlinks = HyperlinkBuffer::empty(area);
        draw_lines_with_hyperlinks(&output, area, &mut buffer, &mut hyperlinks);
        assert!((0..area.height).any(|y| {
            (0..area.width).any(|x| {
                hyperlinks
                    .get(Position::new(x, y))
                    .is_some_and(|destination| {
                        destination.as_ref() == "https://example.com/docs?section=hosted-search"
                    })
            })
        }));
    }

    #[test]
    fn rendered_link_metadata_survives_cell_cache_hits() {
        let theme = Theme::default();
        let cell = test_cell(message(
            MessageAuthor::Assistant,
            "See [Tea docs](https://example.com/docs).",
            OutputFormat::Markdown,
        ));
        let mut renderer = Renderer::new();

        let first = renderer.history_lines(std::slice::from_ref(&cell), 80, &theme);
        let second = renderer.history_lines(&[cell], 80, &theme);
        let links = |lines: &[RenderedLine]| {
            lines
                .iter()
                .flat_map(|line| &line.spans)
                .filter_map(|span| span.link.clone())
                .collect::<Vec<_>>()
        };

        assert_eq!(links(&first), links(&second));
        assert!(!links(&second).is_empty());
        let linked_line = second
            .iter()
            .find(|line| {
                line.rendered_spans()
                    .iter()
                    .any(|span| span.link().is_some())
            })
            .unwrap();
        assert!(linked_line.text().starts_with(' '));
        assert_eq!(linked_line.rendered_spans()[0].link(), None);
        assert_eq!(renderer.cache.hits, 1);
    }

    #[test]
    fn rendered_table_reflows_by_width_and_survives_cell_cache_hits() {
        let theme = Theme::default();
        let source =
            "| Key | Value | Extra | More |\n|---|---|---|---|\n| item | alpha | beta | gamma |";
        let cell = test_cell(message(
            MessageAuthor::Assistant,
            source,
            OutputFormat::Markdown,
        ));
        let mut renderer = Renderer::new();

        let wide = renderer.history_lines(std::slice::from_ref(&cell), 80, &theme);
        let cached = renderer.history_lines(std::slice::from_ref(&cell), 80, &theme);
        let narrow = renderer.history_lines(&[cell], 16, &theme);

        assert_eq!(wide, cached);
        assert!(wide.iter().any(|line| line.text().contains('━')));
        assert!(narrow.iter().any(|line| line.text().trim() == "Key"));
        assert!(narrow.iter().any(|line| line.text().trim() == "item"));
        assert!(narrow.iter().all(|line| line.text().width() <= 16));
        assert_eq!(renderer.cache.hits, 1);
    }

    #[test]
    fn adjacent_rendered_spans_do_not_merge_across_link_destinations() {
        let first = RenderedSpan::with_link(
            "one".to_owned(),
            Style::default(),
            "https://one.example/".to_owned(),
        );
        let second = RenderedSpan::with_link(
            "two".to_owned(),
            Style::default(),
            "https://two.example/".to_owned(),
        );
        let line = RenderedLine::from_spans(Style::default(), vec![first, second]);

        assert_eq!(line.spans.len(), 2);
        assert_ne!(line.spans[0].link, line.spans[1].link);
    }

    #[test]
    fn terminal_result_previews_stdout_and_stderr_separately() {
        let theme = Theme::default();
        let output = render_test_node(
            &test_cell_with_owner(
                result(
                    "Returned",
                    Some("cargo test"),
                    "exit code: Some(0)\nstdout:\n39 passed\nstderr:\nwarning: retrying",
                    OutputFormat::Terminal,
                    false,
                ),
                "0195a0b1-7e00-7000-8000-000000000002".parse().unwrap(),
            ),
            80,
            &theme,
        );

        let text = output.iter().map(RenderedLine::text).collect::<Vec<_>>();
        assert_eq!(
            text,
            [
                "• Returned cargo test",
                "  └ exit code: Some(0)",
                "    39 passed",
                "    warning: retrying",
            ]
        );
        assert_eq!(output[1].style(), theme.footer);
        assert_eq!(output[2].style(), theme.normal);
        assert_eq!(output[3].style(), theme.error);
    }

    #[test]
    fn terminal_result_omits_only_generated_empty_stream_markers() {
        let render = |content: &str| {
            let output = render_test_node(
                &test_cell(result(
                    "Returned",
                    Some("bash"),
                    content,
                    OutputFormat::Terminal,
                    false,
                )),
                80,
                &Theme::default(),
            );
            output
                .iter()
                .map(|line| line.text().to_owned())
                .collect::<Vec<_>>()
        };

        assert_eq!(
            render("exit code: Some(0)\nstdout:\nlisting\nstderr:\n(empty)"),
            ["• Returned bash", "  └ exit code: Some(0)", "    listing",]
        );
        assert_eq!(
            render("exit code: Some(0)\nstdout:\n(empty)\nstderr:\n(empty)"),
            ["• Returned bash", "  └ exit code: Some(0)"]
        );
        assert_eq!(render("(empty)"), ["• Returned bash", "  └ (empty)"]);
    }

    #[test]
    fn shell_command_result_keeps_exit_and_duration_visible() {
        let theme = Theme::default();
        let output = render_test_node(
            &test_cell_with_owner(
                result(
                    "Ran",
                    Some("cargo test -p tea-cli --test tui"),
                    "stdout: 39 passed\nexit 0 in 1.42s",
                    OutputFormat::Terminal,
                    false,
                ),
                "0195a0b1-7e00-7000-8000-000000000002".parse().unwrap(),
            ),
            80,
            &theme,
        );

        let text = output.iter().map(RenderedLine::text).collect::<Vec<_>>();
        assert_eq!(
            text,
            [
                "• Ran cargo test -p tea-cli --test tui",
                "  └ 39 passed",
                "    exit 0 in 1.42s",
            ]
        );
        assert!(
            output[0]
                .spans
                .iter()
                .any(|span| span.style.add_modifier.contains(Modifier::BOLD))
        );
    }

    #[test]
    fn terminal_result_keeps_bounded_head_and_tail_at_narrow_widths() {
        let theme = Theme::default();
        let content = (1..=14)
            .map(|line| {
                if line == 1 {
                    format!("line {line} {}", "long-token-".repeat(8))
                } else {
                    format!("line {line}")
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        let output = render_test_node(
            &test_cell(result(
                "Returned",
                Some("process"),
                &content,
                OutputFormat::Terminal,
                false,
            )),
            20,
            &theme,
        );

        let text = output.iter().map(RenderedLine::text).collect::<Vec<_>>();
        assert_eq!(text.len(), 12);
        assert!(text[1].starts_with("  └ line 1"));
        assert_eq!(text[5], "    line 5");
        assert_eq!(text[6], "    ... +4 lines");
        assert_eq!(text[7], "    line 10");
        assert_eq!(text[11], "    line 14");
        assert!(text.iter().all(|line| line.width() <= 20));
    }

    #[test]
    fn proposed_plan_reflows_markdown_and_retains_raw_source() {
        let theme = Theme::default();
        let source = "## Renderer migration\n\n1. Preserve narrow CJK 宽度 and a long continuation\n2. Run tests";
        let cell = test_cell(CellContent::Plan(PlanCell::new(
            "Proposed Plan",
            Some(source),
            Vec::new(),
            None,
        )));
        let output = render_test_node(&cell, 24, &theme);

        assert_eq!(cell.raw_text(), source);
        assert_eq!(output[0].text(), "○ Proposed Plan");
        assert!(output[1].text().starts_with("  └ ## Renderer"));
        assert!(output.len() > 4, "proposal must reflow at narrow width");
        assert!(output.iter().all(|line| line.text().width() <= 24));
        assert!(output.len() <= 1 + super::MAX_VISIBLE_PLAN_BODY_ROWS);
    }

    #[test]
    fn plan_progress_is_bounded_and_semantic_without_color() {
        let cell = test_cell(CellContent::Plan(PlanCell::new(
            "Updated Plan",
            None,
            std::iter::once(PlanStep::new(PlanStepStatus::Completed, "Freeze gallery"))
                .chain(std::iter::once(PlanStep::new(
                    PlanStepStatus::InProgress,
                    "Port typed plan cells",
                )))
                .chain(std::iter::repeat_n(
                    PlanStep::new(PlanStepStatus::Pending, "Run regression matrix"),
                    14,
                ))
                .collect(),
            Some("Protocol projection remains disabled."),
        )));
        let themes = [
            Theme::default(),
            Theme::for_capabilities(TerminalCapabilities::from_environment(
                Some("dumb"),
                None,
                false,
                false,
            )),
        ];

        for theme in themes {
            let output = render_test_node(&cell, 32, &theme);
            assert_eq!(output[0].text(), "• Updated Plan");
            assert!(output[1].text().starts_with("  [x] Freeze gallery"));
            assert!(output[2].text().starts_with("  [>] Port typed plan cells"));
            assert!(output[3].text().starts_with("  [ ] Run regression matrix"));
            assert!(
                !output[1].spans[0]
                    .style
                    .add_modifier
                    .contains(Modifier::CROSSED_OUT)
            );
            assert!(
                output[1].spans[1]
                    .style
                    .add_modifier
                    .contains(Modifier::CROSSED_OUT)
            );
            assert!(
                output[2].spans[1]
                    .style
                    .add_modifier
                    .contains(Modifier::BOLD)
            );
            assert!(output.last().unwrap().text().contains("plan rows"));
            assert!(output.iter().all(|line| line.text().width() <= 32));
            assert!(output.len() <= 1 + super::MAX_VISIBLE_PLAN_BODY_ROWS);
        }
        assert!(cell.raw_text().contains("[x] Freeze gallery"));
        assert!(cell.raw_text().contains("[>] Port typed plan cells"));
        assert!(cell.raw_text().contains("[ ] Run regression matrix"));
    }

    #[test]
    fn queued_input_uses_a_narrow_tree_continuation() {
        let output = render_test_node(
            &test_cell(CellContent::QueuedInput(QueuedInputCell::new(
                QueuedInputKind::FollowUp,
                "verify CJK 宽度 after the active run",
            ))),
            20,
            &Theme::default(),
        );

        assert_eq!(output[0].text(), "○ Queued follow-up");
        assert!(output[1].text().starts_with("  └ verify"));
        assert!(output.iter().all(|line| line.text().width() <= 20));
    }

    #[test]
    fn notice_severities_keep_distinct_markers_safe_hints_and_width_bounds() {
        let theme = Theme::default();
        for (severity, marker, expected_style) in [
            (NoticeSeverity::Information, "•", theme.information),
            (NoticeSeverity::Warning, "!", theme.warning),
            (NoticeSeverity::Error, "■", theme.error),
        ] {
            let output = render_test_node(
                &test_cell(CellContent::Notice(NoticeCell::new(
                    NoticeKind::General,
                    severity,
                    "diagnostic\u{1b}[31m CJK 宽度内容",
                    Some("bounded supporting hint"),
                ))),
                16,
                &theme,
            );

            assert!(output[0].text().starts_with(marker));
            assert_eq!(output[0].style(), expected_style);
            assert!(output.iter().all(|line| line.text().width() <= 16));
            assert!(
                output
                    .iter()
                    .all(|line| { !line.text().chars().any(char::is_control) })
            );
            assert!(output.iter().any(|line| line.text().starts_with("  └ ")));
            assert_eq!(output.last().unwrap().style(), theme.footer);
        }
    }

    struct GalleryCase {
        name: &'static str,
        readiness: &'static str,
        cell: CellNode,
    }

    struct GalleryProfile {
        name: &'static str,
        width: usize,
        height: usize,
        theme: Theme,
    }

    #[test]
    fn codex_aligned_output_gallery_matches_all_terminal_profiles() {
        let profiles = gallery_profiles();
        let cases = gallery_cases();
        let mut actual = String::from(
            "# Tea TUI output gallery\n# Codex reference: 1836ae0612052137d0cabaff7807ff8314cee940\n",
        );

        for profile in profiles {
            writeln!(
                actual,
                "\n## profile {} {}x{} theme-generation={}",
                profile.name, profile.width, profile.height, profile.theme.generation
            )
            .unwrap();
            for case in &cases {
                let lines = render_test_node(&case.cell, profile.width, &profile.theme);
                assert!(
                    lines.len() <= profile.height,
                    "{} exceeds {} rows in {}",
                    case.name,
                    profile.height,
                    profile.name
                );
                assert!(
                    lines
                        .iter()
                        .all(|line| line.text().width() <= profile.width),
                    "{} exceeds {} columns in {}: {:?}",
                    case.name,
                    profile.width,
                    profile.name,
                    lines.iter().map(RenderedLine::text).collect::<Vec<_>>()
                );
                writeln!(actual, "\n### {} [{}]", case.name, case.readiness).unwrap();
                writeln!(actual, "raw |{}|", snapshot_text(&case.cell.raw_text())).unwrap();
                let display = lines
                    .iter()
                    .map(RenderedLine::text)
                    .collect::<Vec<_>>()
                    .join("\n");
                writeln!(actual, "display |{}|", snapshot_text(&display)).unwrap();
                let styles = lines
                    .iter()
                    .map(line_style_signature)
                    .collect::<Vec<_>>()
                    .join(" || ");
                writeln!(actual, "styles |{styles}|").unwrap();
            }
        }

        let snapshot_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/tui/output-gallery.snap"
        );
        if std::env::var_os("TEA_UPDATE_TUI_SNAPSHOTS").is_some() {
            std::fs::write(snapshot_path, &actual).unwrap();
        }
        let expected = std::fs::read_to_string(snapshot_path).unwrap();
        assert_eq!(actual, expected);
    }

    #[test]
    fn output_gallery_freezes_the_full_regression_sign_off_matrix() {
        let cases = gallery_cases();
        let names = cases.iter().map(|case| case.name).collect::<Vec<_>>();
        for expected in [
            "lifecycle-proposed",
            "lifecycle-approval-pending",
            "lifecycle-failed",
            "lifecycle-interrupted",
            "lifecycle-uncertain",
            "lifecycle-queued",
            "empty-tool-result",
            "malformed-tool-detail",
            "hostile-cjk-emoji",
            "long-process-output",
        ] {
            assert!(
                names.contains(&expected),
                "regression gallery is missing {expected}"
            );
        }

        let mut statuses = cases
            .iter()
            .filter_map(|case| match case.cell.content() {
                CellContent::Lifecycle(lifecycle) => Some(lifecycle.status()),
                _ => None,
            })
            .collect::<Vec<_>>();
        statuses.sort_unstable();
        statuses.dedup();
        assert_eq!(
            statuses,
            vec![
                LifecycleStatus::Proposed,
                LifecycleStatus::Requested,
                LifecycleStatus::ApprovalPending,
                LifecycleStatus::Running,
                LifecycleStatus::Succeeded,
                LifecycleStatus::Failed,
                LifecycleStatus::Interrupted,
                LifecycleStatus::Uncertain,
                LifecycleStatus::Queued,
            ]
        );

        for case in cases {
            let raw = case.cell.raw_text();
            if case.name == "empty-tool-result" {
                assert!(raw.is_empty(), "empty canonical output must stay empty");
            } else {
                assert!(!raw.is_empty(), "{} has no raw fallback", case.name);
            }
            assert!(
                raw.chars()
                    .all(|character| character == '\n' || !character.is_control()),
                "{} raw fallback contains a terminal control: {raw:?}",
                case.name
            );
        }
    }

    fn gallery_profiles() -> Vec<GalleryProfile> {
        let mut profiles = Vec::new();
        for (width, height) in [(40, 16), (80, 24), (120, 32)] {
            profiles.push(GalleryProfile {
                name: "default",
                width,
                height,
                theme: Theme::for_capabilities(TerminalCapabilities::from_environment(
                    Some("xterm-truecolor"),
                    Some("truecolor"),
                    false,
                    false,
                )),
            });
            profiles.push(GalleryProfile {
                name: "no-color",
                width,
                height,
                theme: Theme::for_capabilities(TerminalCapabilities::from_environment(
                    Some("xterm"),
                    None,
                    true,
                    false,
                )),
            });
            profiles.push(GalleryProfile {
                name: "reduced-motion",
                width,
                height,
                theme: Theme::for_capabilities(TerminalCapabilities::from_environment(
                    Some("xterm-truecolor"),
                    Some("truecolor"),
                    false,
                    true,
                )),
            });
        }
        profiles
    }

    #[allow(clippy::too_many_lines)]
    fn gallery_cases() -> Vec<GalleryCase> {
        vec![
            gallery(
                "user-prompt",
                "supported",
                message(
                    MessageAuthor::User,
                    "Refactor the renderer and preserve CJK 宽度.",
                    OutputFormat::Plain,
                ),
            ),
            gallery(
                "assistant-response",
                "supported",
                message(
                    MessageAuthor::Assistant,
                    "## Result\n\n- Added typed cells\n- Preserved **Markdown** output",
                    OutputFormat::Markdown,
                ),
            ),
            gallery(
                "thinking-reasoning",
                "supported-observable-events-only",
                CellContent::Reasoning(ReasoningCell::new(
                    "Checking the projection and terminal width constraints.",
                    false,
                )),
            ),
            gallery(
                "run-activity",
                "supported",
                lifecycle(
                    LifecycleKind::RunActivity,
                    "Working",
                    Some("renderer migration"),
                    LifecycleStatus::Running,
                    vec![detail(TimelineDetailKind::Progress, Some("elapsed"), "12s")],
                    2,
                ),
            ),
            gallery(
                "generic-tool-call",
                "supported",
                lifecycle(
                    LifecycleKind::ToolCall,
                    "Requested",
                    Some("custom_tool workspace/index"),
                    LifecycleStatus::Requested,
                    Vec::new(),
                    0,
                ),
            ),
            gallery(
                "lifecycle-proposed",
                "supported",
                lifecycle(
                    LifecycleKind::ToolCall,
                    "Proposed",
                    Some("custom_tool workspace/index"),
                    LifecycleStatus::Proposed,
                    Vec::new(),
                    0,
                ),
            ),
            gallery(
                "lifecycle-approval-pending",
                "supported",
                lifecycle(
                    LifecycleKind::ToolCall,
                    "Waiting for approval",
                    Some("write_text_file /workspace/notes.txt"),
                    LifecycleStatus::ApprovalPending,
                    Vec::new(),
                    0,
                ),
            ),
            gallery(
                "tool-arguments-progress-result",
                "supported-generic-fallback",
                lifecycle(
                    LifecycleKind::ToolCall,
                    "Running",
                    Some("custom_tool workspace/index"),
                    LifecycleStatus::Running,
                    vec![
                        detail(
                            TimelineDetailKind::Metadata,
                            Some("arguments"),
                            r#"{"path":"src","depth":3}"#,
                        ),
                        detail(
                            TimelineDetailKind::Progress,
                            Some("progress"),
                            "24/80 files indexed",
                        ),
                    ],
                    2,
                ),
            ),
            gallery(
                "lifecycle-failed",
                "supported",
                lifecycle(
                    LifecycleKind::ToolCall,
                    "Failed",
                    Some("cargo test -p tea-cli --test tui"),
                    LifecycleStatus::Failed,
                    vec![detail(
                        TimelineDetailKind::Error,
                        Some("exit"),
                        "101 after 1.42s",
                    )],
                    0,
                ),
            ),
            gallery(
                "lifecycle-interrupted",
                "supported",
                lifecycle(
                    LifecycleKind::ToolCall,
                    "Interrupted",
                    Some("cargo test"),
                    LifecycleStatus::Interrupted,
                    vec![detail(
                        TimelineDetailKind::Metadata,
                        Some("reason"),
                        "cancelled by user",
                    )],
                    0,
                ),
            ),
            gallery(
                "lifecycle-uncertain",
                "supported",
                lifecycle(
                    LifecycleKind::ToolCall,
                    "Interrupted",
                    Some("remote deploy"),
                    LifecycleStatus::Uncertain,
                    vec![detail(
                        TimelineDetailKind::Metadata,
                        Some("outcome"),
                        "unknown after disconnect",
                    )],
                    0,
                ),
            ),
            gallery(
                "lifecycle-queued",
                "supported",
                lifecycle(
                    LifecycleKind::ToolCall,
                    "Queued",
                    Some("follow-up validation"),
                    LifecycleStatus::Queued,
                    Vec::new(),
                    0,
                ),
            ),
            gallery(
                "empty-tool-result",
                "supported",
                result(
                    "Returned",
                    Some("custom_tool"),
                    "",
                    OutputFormat::Plain,
                    false,
                ),
            ),
            gallery(
                "malformed-tool-detail",
                "supported-generic-fallback",
                lifecycle(
                    LifecycleKind::ToolCall,
                    "Ran",
                    Some("custom_tool"),
                    LifecycleStatus::Succeeded,
                    vec![detail(
                        TimelineDetailKind::Metadata,
                        Some("arguments"),
                        "{\"path\":\u{1b}[31m",
                    )],
                    0,
                ),
            ),
            gallery(
                "hostile-cjk-emoji",
                "supported",
                notice(
                    NoticeSeverity::Warning,
                    "控制字符 \u{1b}[31m 保持可见，emoji 👩🏽‍💻 不破坏宽度",
                    Some("a-very-long-unbroken-token-0123456789-abcdefghijklmnopqrstuvwxyz"),
                ),
            ),
            gallery(
                "long-process-output",
                "supported-bounded-preview",
                result(
                    "Returned",
                    Some("long-running-command"),
                    &(1..=14)
                        .map(|line| format!("line {line} {}", "wide-output-".repeat(4)))
                        .collect::<Vec<_>>()
                        .join("\n"),
                    OutputFormat::Terminal,
                    false,
                ),
            ),
            gallery(
                "shell-command",
                "target-generic-until-command-metadata",
                lifecycle(
                    LifecycleKind::ToolCall,
                    "Ran",
                    Some("cargo test -p tea-cli --test tui"),
                    LifecycleStatus::Succeeded,
                    vec![
                        detail(TimelineDetailKind::Output, None, "39 passed; 0 failed"),
                        detail(TimelineDetailKind::Metadata, Some("exit"), "0 in 1.42s"),
                    ],
                    0,
                ),
            ),
            gallery(
                "background-terminal",
                "protocol-gated",
                lifecycle(
                    LifecycleKind::ToolCall,
                    "Waited",
                    Some("cargo test"),
                    LifecycleStatus::Running,
                    vec![detail(
                        TimelineDetailKind::Metadata,
                        Some("stdin"),
                        "waiting for process output",
                    )],
                    2,
                ),
            ),
            gallery(
                "file-search-list",
                "protocol-gated-generic-fallback",
                lifecycle(
                    LifecycleKind::ToolCall,
                    "Read",
                    Some("crates/tea-cli/src/tui"),
                    LifecycleStatus::Succeeded,
                    vec![detail(
                        TimelineDetailKind::Output,
                        None,
                        "presentation.rs, render.rs, theme.rs",
                    )],
                    0,
                ),
            ),
            gallery(
                "web-search",
                "protocol-gated-generic-fallback",
                lifecycle(
                    LifecycleKind::ToolCall,
                    "Searched the web for",
                    Some("ratatui styled wrapping"),
                    LifecycleStatus::Succeeded,
                    Vec::new(),
                    0,
                ),
            ),
            gallery(
                "mcp-lifecycle",
                "target-generic-until-mcp-call-events",
                lifecycle(
                    LifecycleKind::ToolCall,
                    "Called",
                    Some("docs.search query=ratatui"),
                    LifecycleStatus::Succeeded,
                    vec![detail(
                        TimelineDetailKind::Output,
                        None,
                        "Found styling guidance in styles.md",
                    )],
                    0,
                ),
            ),
            gallery(
                "fenced-code",
                "supported-semantic-style",
                message(
                    MessageAuthor::Assistant,
                    "```rust\nfn render() -> bool { true }\n```",
                    OutputFormat::Markdown,
                ),
            ),
            gallery(
                "process-code-output",
                "supported-generic-fallback",
                result(
                    "Returned",
                    Some("cargo test"),
                    "stdout: 39 passed\nstderr: warning: retrying",
                    OutputFormat::Terminal,
                    false,
                ),
            ),
            gallery(
                "plan-proposal",
                "renderer-ready-protocol-gated",
                CellContent::Plan(PlanCell::new(
                    "Proposed Plan",
                    Some(
                        "## Renderer migration\n\n1. Introduce typed cells\n2. Port output grammar\n3. Run PTY tests",
                    ),
                    Vec::new(),
                    None,
                )),
            ),
            gallery(
                "plan-progress",
                "renderer-ready-protocol-gated",
                CellContent::Plan(PlanCell::new(
                    "Updated Plan",
                    None,
                    vec![
                        PlanStep::new(PlanStepStatus::Completed, "Freeze gallery"),
                        PlanStep::new(PlanStepStatus::InProgress, "Port tool cells"),
                        PlanStep::new(PlanStepStatus::Pending, "Run PTY tests"),
                    ],
                    Some("Protocol projection remains disabled."),
                )),
            ),
            gallery(
                "patch-diff",
                "protocol-gated-generic-fallback",
                result(
                    "Updated",
                    Some("src/render.rs"),
                    "- label: text\n+ • Ran tool\n+   └ output",
                    OutputFormat::Terminal,
                    false,
                ),
            ),
            gallery(
                "approval",
                "supported-modal-and-durable-fallback",
                decision(
                    "Approval required",
                    "write_text_file",
                    DecisionStatus::Pending,
                    vec![
                        detail(TimelineDetailKind::Metadata, Some("target"), "native"),
                        detail(TimelineDetailKind::Metadata, Some("effects"), "fs.write"),
                    ],
                ),
            ),
            gallery(
                "error",
                "supported",
                notice(
                    NoticeSeverity::Error,
                    "Tool execution failed",
                    Some("permission denied"),
                ),
            ),
            gallery(
                "warning",
                "supported-typed-callers",
                notice(
                    NoticeSeverity::Warning,
                    "Session is approaching its context limit",
                    None,
                ),
            ),
            gallery(
                "information",
                "supported",
                notice(
                    NoticeSeverity::Information,
                    "Connection restored",
                    Some("session state reloaded"),
                ),
            ),
            gallery(
                "queued-steering-follow-up",
                "supported",
                CellContent::QueuedInput(QueuedInputCell::new(
                    QueuedInputKind::Steering,
                    "also update the narrow-terminal tests",
                )),
            ),
            gallery(
                "image-artifact",
                "protocol-gated-textual-fallback",
                result(
                    "Produced",
                    Some("screenshot.png"),
                    "image/png, 1280x720",
                    OutputFormat::Plain,
                    false,
                ),
            ),
            gallery(
                "hook",
                "protocol-gated-generic-fallback",
                lifecycle(
                    LifecycleKind::ToolCall,
                    "Ran hook",
                    Some("post-format"),
                    LifecycleStatus::Succeeded,
                    vec![detail(
                        TimelineDetailKind::Output,
                        None,
                        "cargo fmt --check",
                    )],
                    0,
                ),
            ),
            gallery(
                "multi-agent",
                "runtime-gated-generic-fallback",
                lifecycle(
                    LifecycleKind::ToolCall,
                    "Completed task",
                    Some("agent/review renderer audit"),
                    LifecycleStatus::Succeeded,
                    Vec::new(),
                    0,
                ),
            ),
        ]
    }

    fn gallery(name: &'static str, readiness: &'static str, content: CellContent) -> GalleryCase {
        GalleryCase {
            name,
            readiness,
            cell: test_cell(content),
        }
    }

    fn lifecycle(
        kind: LifecycleKind,
        action: &str,
        target: Option<&str>,
        status: LifecycleStatus,
        details: Vec<TimelineDetail>,
        tick: u64,
    ) -> CellContent {
        CellContent::Lifecycle(LifecycleCell::new(
            kind, action, target, status, details, true, tick,
        ))
    }

    fn decision(
        action: &str,
        subject: &str,
        status: DecisionStatus,
        details: Vec<TimelineDetail>,
    ) -> CellContent {
        CellContent::Decision(DecisionCell::new(action, subject, status, details))
    }

    fn notice(severity: NoticeSeverity, message: &str, hint: Option<&str>) -> CellContent {
        CellContent::Notice(NoticeCell::new(
            NoticeKind::General,
            severity,
            message,
            hint,
        ))
    }

    fn detail(kind: TimelineDetailKind, label: Option<&str>, text: &str) -> TimelineDetail {
        TimelineDetail::new(kind, label, text)
    }

    fn snapshot_text(text: &str) -> String {
        text.replace('\\', "\\\\")
            .replace('|', "\\|")
            .replace('\n', "\\n")
    }

    fn style_signature(style: Style) -> String {
        let foreground = style
            .fg
            .map_or_else(|| "-".to_owned(), |color| format!("{color:?}"));
        let background = style
            .bg
            .map_or_else(|| "-".to_owned(), |color| format!("{color:?}"));
        format!(
            "{foreground}/{background}/{:?}/{:?}",
            style.add_modifier, style.sub_modifier
        )
    }

    fn line_style_signature(line: &RenderedLine) -> String {
        let mut signature = style_signature(line.style());
        if !line.spans.is_empty() {
            let spans = line
                .spans
                .iter()
                .map(|span| style_signature(span.style))
                .collect::<Vec<_>>()
                .join(", ");
            write!(signature, " spans=[{spans}]").unwrap();
        }
        signature
    }
}
