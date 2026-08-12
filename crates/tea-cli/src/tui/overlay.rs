use super::commands::CommandCompletionItem;
use super::selectors::Selector;

const MAX_COMMAND_COMPLETIONS: usize = 16;

/// One bounded command or skill completion menu owned by the local TUI.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandCompletion {
    options: Vec<CommandCompletionItem>,
    selected: usize,
}

impl CommandCompletion {
    /// Creates a bounded deterministic completion list.
    #[must_use]
    pub fn new<I, T>(options: I) -> Self
    where
        I: IntoIterator<Item = T>,
        T: Into<CommandCompletionItem>,
    {
        let options = options
            .into_iter()
            .map(Into::into)
            .filter(|option| {
                matches!(option.value().as_bytes().first(), Some(b'/' | b'$'))
                    && option.value().len() <= 512
                    && !option.value().chars().any(char::is_control)
            })
            .take(MAX_COMMAND_COMPLETIONS)
            .collect();
        Self {
            options,
            selected: 0,
        }
    }

    /// Returns all visible completion options in stable order.
    #[must_use]
    pub fn options(&self) -> &[CommandCompletionItem] {
        &self.options
    }

    /// Returns the explicit option currently selected by the user.
    #[must_use]
    pub fn selected(&self) -> Option<&CommandCompletionItem> {
        self.options.get(self.selected)
    }

    /// Moves to the next option with wraparound.
    pub fn move_next(&mut self) {
        if !self.options.is_empty() {
            self.selected = (self.selected + 1) % self.options.len();
        }
    }

    /// Moves to the previous option with wraparound.
    pub fn move_previous(&mut self) {
        if !self.options.is_empty() {
            self.selected = self
                .selected
                .checked_sub(1)
                .unwrap_or(self.options.len() - 1);
        }
    }

    pub(crate) fn should_show(&self, text: &str) -> bool {
        self.selected().is_some() && !self.options.iter().any(|option| option.value() == text)
    }
}

/// The one top-priority local interaction view currently visible above composer input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Overlay {
    /// A typed selector backed by host queries.
    Selector(Selector),
    /// A command or skill completion menu that edits the composer draft.
    CommandCompletion(CommandCompletion),
}

impl Overlay {
    /// Returns the selector when this overlay owns a selector view.
    #[must_use]
    pub const fn selector(&self) -> Option<&Selector> {
        match self {
            Self::Selector(selector) => Some(selector),
            Self::CommandCompletion(_) => None,
        }
    }

    /// Returns mutable selector state when this overlay owns it.
    pub(crate) fn selector_mut(&mut self) -> Option<&mut Selector> {
        match self {
            Self::Selector(selector) => Some(selector),
            Self::CommandCompletion(_) => None,
        }
    }

    /// Returns the command menu when this overlay owns it.
    #[must_use]
    pub const fn command_completion(&self) -> Option<&CommandCompletion> {
        match self {
            Self::Selector(_) => None,
            Self::CommandCompletion(completion) => Some(completion),
        }
    }

    /// Returns mutable command-menu state when this overlay owns it.
    pub(crate) fn command_completion_mut(&mut self) -> Option<&mut CommandCompletion> {
        match self {
            Self::Selector(_) => None,
            Self::CommandCompletion(completion) => Some(completion),
        }
    }
}
