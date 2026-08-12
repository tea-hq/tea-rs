use std::collections::{BTreeMap, BTreeSet};
use std::str::FromStr as _;

use tea::context::SkillCommand;
use tea_mcp::McpServerId;
use tea_protocol::{MessageId, ModelId, ReasoningEffort, SessionId};

use super::attachment::MAX_COMPOSER_ATTACHMENTS;

const BUILTINS: [&str; 15] = [
    "compact",
    "copy",
    "fork",
    "help",
    "image",
    "mcp",
    "model",
    "name",
    "new",
    "quit",
    "resume",
    "reasoning",
    "session",
    "skills",
    "tree",
];
const MAX_COMMANDS: usize = 512;
const MAX_COMPLETION_DESCRIPTION_BYTES: usize = 512;
const MAX_IMAGE_PATH_BYTES: usize = 4096;

/// Parsed interactive slash command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlashCommand {
    /// Create and switch to a new session.
    New,
    /// Resume an explicit session or open the session selector.
    Resume(Option<SessionId>),
    /// Open the session selector.
    Session,
    /// Set or clear a session display name.
    Name(Option<String>),
    /// Set a model or open the model selector.
    Model(Option<ModelId>),
    /// Set reasoning effort or open the reasoning selector.
    Reasoning(Option<ReasoningEffort>),
    /// Compact the active session.
    Compact,
    /// Open the branch tree selector.
    Tree,
    /// Fork from one durable message.
    Fork(MessageId),
    /// Copy the last assistant response.
    Copy,
    /// Load one explicit local image path.
    Image(String),
    /// Remove one image by its one-based composer index.
    ImageRemove(usize),
    /// Remove every image from the active session composer.
    ImageClear,
    /// Display safe MCP server health and frozen aliases.
    Mcp,
    /// Display the frozen discovered skill catalog.
    Skills,
    /// Reconnect one MCP server only against its frozen discovery snapshot.
    McpReconnect(McpServerId),
    /// Show command help.
    Help,
    /// Exit the application.
    Quit,
    /// Expand one trusted declarative prompt template.
    Template {
        /// Canonical template name.
        name: String,
        /// Positional template arguments.
        arguments: Vec<String>,
    },
    /// Load one explicit trusted skill invocation.
    Skill(SkillCommand),
}

/// Invalid command catalog or invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum CommandError {
    /// The command name, arguments, or identifier is invalid.
    #[error("slash command is invalid")]
    Invalid,
    /// A declarative resource conflicts with another command.
    #[error("slash command catalog has a duplicate")]
    Duplicate,
    /// The command catalog exceeds its fixed bound.
    #[error("slash command catalog is too large")]
    TooMany,
}

/// Semantic type shown beside one completion candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommandCompletionKind {
    /// Built-in TUI command.
    Command,
    /// Declarative prompt template.
    Prompt,
    /// Discovered skill.
    Skill,
}

impl CommandCompletionKind {
    /// Returns the compact user-facing type label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Command => "command",
            Self::Prompt => "prompt",
            Self::Skill => "skill",
        }
    }

    const fn precedence(self) -> u8 {
        match self {
            Self::Command => 0,
            Self::Prompt => 1,
            Self::Skill => 2,
        }
    }
}

/// One typed completion candidate and the text it inserts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandCompletionItem {
    value: String,
    kind: CommandCompletionKind,
    description: Option<String>,
}

impl CommandCompletionItem {
    /// Creates one completion candidate.
    #[must_use]
    pub fn new(value: impl Into<String>, kind: CommandCompletionKind) -> Self {
        Self {
            value: value.into(),
            kind,
            description: None,
        }
    }

    /// Adds display-only metadata that never becomes composer text.
    #[must_use]
    pub fn with_description(mut self, description: impl AsRef<str>) -> Self {
        self.description = normalize_completion_description(description.as_ref());
        self
    }

    /// Returns the composer-facing candidate text.
    #[must_use]
    pub fn value(&self) -> &str {
        &self.value
    }

    /// Returns the candidate's semantic type.
    #[must_use]
    pub const fn kind(&self) -> CommandCompletionKind {
        self.kind
    }

    /// Returns the normalized single-line display description, when present.
    #[must_use]
    pub fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }
}

impl From<String> for CommandCompletionItem {
    fn from(value: String) -> Self {
        let kind = if value.starts_with('$') {
            CommandCompletionKind::Skill
        } else {
            CommandCompletionKind::Command
        };
        Self::new(value, kind)
    }
}

/// Deterministic built-in and declarative slash command catalog.
#[derive(Debug, Clone)]
pub struct CommandCatalog {
    templates: BTreeSet<String>,
    skills: BTreeSet<String>,
    skill_descriptions: BTreeMap<String, String>,
    completions: Vec<CommandCompletionItem>,
}

impl CommandCatalog {
    /// Builds a bounded catalog from trusted resource names.
    ///
    /// # Errors
    ///
    /// Rejects invalid, duplicate, conflicting, or oversized names.
    pub fn new<T, S, TI, SI>(templates: TI, skills: SI) -> Result<Self, CommandError>
    where
        T: AsRef<str>,
        S: AsRef<str>,
        TI: IntoIterator<Item = T>,
        SI: IntoIterator<Item = S>,
    {
        let templates = collect_names(templates)?;
        let skills = collect_names(skills)?;
        if templates
            .iter()
            .any(|name| BUILTINS.contains(&name.as_str()))
        {
            return Err(CommandError::Duplicate);
        }
        if templates.len().saturating_add(skills.len()) + BUILTINS.len() > MAX_COMMANDS {
            return Err(CommandError::TooMany);
        }
        let completions = BUILTINS
            .into_iter()
            .map(|name| {
                CommandCompletionItem::new(format!("/{name}"), CommandCompletionKind::Command)
            })
            .chain(templates.iter().map(|name| {
                CommandCompletionItem::new(format!("/{name}"), CommandCompletionKind::Prompt)
            }))
            .chain(skills.iter().map(|name| {
                CommandCompletionItem::new(format!("/{name}"), CommandCompletionKind::Skill)
            }))
            .collect::<Vec<_>>();
        Ok(Self {
            templates,
            skills,
            skill_descriptions: BTreeMap::new(),
            completions,
        })
    }

    /// Adds display-only descriptions for registered skills.
    #[must_use]
    pub fn with_skill_descriptions<K, V, I>(mut self, descriptions: I) -> Self
    where
        K: AsRef<str>,
        V: AsRef<str>,
        I: IntoIterator<Item = (K, V)>,
    {
        for (name, description) in descriptions {
            let name = name.as_ref();
            if !self.skills.contains(name) {
                continue;
            }
            if let Some(description) = normalize_completion_description(description.as_ref()) {
                self.skill_descriptions.insert(name.to_owned(), description);
            }
        }
        for completion in &mut self.completions {
            let Some(name) = completion
                .value
                .strip_prefix('/')
                .filter(|_| completion.kind == CommandCompletionKind::Skill)
            else {
                continue;
            };
            completion.description = self.skill_descriptions.get(name).cloned();
        }
        self
    }

    /// Parses one complete command line.
    ///
    /// # Errors
    ///
    /// Rejects unknown commands, malformed identifiers, or extra arguments.
    pub fn parse(&self, input: &str) -> Result<SlashCommand, CommandError> {
        if input.is_empty() || input.contains('\0') || input.len() > crate::tui::MAX_EDITOR_BYTES {
            return Err(CommandError::Invalid);
        }
        if let Some(skill) = input.strip_prefix("/skill:") {
            let command = format!("/skill:{skill}")
                .parse::<SkillCommand>()
                .map_err(|_| CommandError::Invalid)?;
            if self.skills.contains(command.skill_id().as_str()) {
                return Ok(SlashCommand::Skill(command));
            }
            return Err(CommandError::Invalid);
        }
        if input == "/image" || input.starts_with("/image ") {
            return parse_image(input);
        }
        let command_line = input.strip_prefix('/').ok_or(CommandError::Invalid)?;
        let separator = command_line.find(char::is_whitespace);
        let (command, argument_text) = separator.map_or((command_line, ""), |index| {
            (&command_line[..index], command_line[index..].trim_start())
        });
        let arguments = argument_text
            .split_whitespace()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        match command {
            "new" if arguments.is_empty() => Ok(SlashCommand::New),
            "resume" if arguments.len() <= 1 => Ok(SlashCommand::Resume(
                arguments
                    .first()
                    .map(|value| SessionId::from_str(value))
                    .transpose()
                    .map_err(|_| CommandError::Invalid)?,
            )),
            "session" if arguments.is_empty() => Ok(SlashCommand::Session),
            "name" => Ok(SlashCommand::Name(
                (!arguments.is_empty()).then(|| arguments.join(" ")),
            )),
            "model" if arguments.len() <= 1 => Ok(SlashCommand::Model(
                arguments
                    .first()
                    .map(|value| ModelId::from_str(value))
                    .transpose()
                    .map_err(|_| CommandError::Invalid)?,
            )),
            "reasoning" if arguments.len() <= 1 => Ok(SlashCommand::Reasoning(
                arguments
                    .first()
                    .map(|value| ReasoningEffort::from_str(value))
                    .transpose()
                    .map_err(|_| CommandError::Invalid)?,
            )),
            "compact" if arguments.is_empty() => Ok(SlashCommand::Compact),
            "tree" if arguments.is_empty() => Ok(SlashCommand::Tree),
            "fork" if arguments.len() == 1 => MessageId::from_str(&arguments[0])
                .map(SlashCommand::Fork)
                .map_err(|_| CommandError::Invalid),
            "copy" if arguments.is_empty() => Ok(SlashCommand::Copy),
            "mcp" if arguments.is_empty() => Ok(SlashCommand::Mcp),
            "skills" if arguments.is_empty() => Ok(SlashCommand::Skills),
            "mcp" if arguments.len() == 2 && arguments[0] == "reconnect" => {
                McpServerId::from_str(&arguments[1])
                    .map(SlashCommand::McpReconnect)
                    .map_err(|_| CommandError::Invalid)
            }
            "help" if arguments.is_empty() => Ok(SlashCommand::Help),
            "quit" if arguments.is_empty() => Ok(SlashCommand::Quit),
            name if self.templates.contains(name) => Ok(SlashCommand::Template {
                name: name.to_owned(),
                arguments,
            }),
            name if self.skills.contains(name) && !BUILTINS.contains(&name) => {
                let skill_id = name.parse().map_err(|_| CommandError::Invalid)?;
                SkillCommand::new(skill_id, argument_text)
                    .map(SlashCommand::Skill)
                    .map_err(|_| CommandError::Invalid)
            }
            _ => Err(CommandError::Invalid),
        }
    }

    /// Converts a selected completion into its composer-facing draft.
    ///
    /// Skill commands use a compact mention while all other completions retain
    /// their slash-command spelling.
    #[must_use]
    pub fn completion_draft(&self, completion: &CommandCompletionItem) -> String {
        if completion.kind == CommandCompletionKind::Skill {
            return completion
                .value
                .strip_prefix('/')
                .or_else(|| completion.value.strip_prefix('$'))
                .filter(|name| self.skills.contains(*name))
                .map_or_else(|| completion.value.clone(), |name| format!("${name} "));
        }
        completion.value.clone()
    }

    /// Parses a registered composer-facing skill mention.
    ///
    /// The mention is accepted only at the start of the draft and preserves
    /// all following text as uninterpreted skill arguments.
    ///
    /// # Errors
    ///
    /// Returns an error when a registered mention has invalid or oversized
    /// arguments.
    pub fn parse_skill_mention(&self, input: &str) -> Result<Option<SkillCommand>, CommandError> {
        let Some((name, arguments)) = split_skill_mention(input) else {
            return Ok(None);
        };
        if !self.skills.contains(name) {
            return Ok(None);
        }
        let skill_id = name.parse().map_err(|_| CommandError::Invalid)?;
        SkillCommand::new(skill_id, arguments)
            .map(Some)
            .map_err(|_| CommandError::Invalid)
    }

    /// Returns the styled token for a registered composer-facing skill mention.
    #[must_use]
    pub fn skill_mention_token(&self, input: &str) -> Option<String> {
        let (name, _) = split_skill_mention(input)?;
        self.skills.contains(name).then(|| format!("${name}"))
    }

    /// Returns deterministic command or skill completions up to the caller's bound.
    ///
    /// Slash commands retain prefix matching. Dollar-prefixed skill mentions
    /// rank exact, prefix, substring, and subsequence matches in that order.
    #[must_use]
    pub fn complete(&self, query: &str, limit: usize) -> Vec<CommandCompletionItem> {
        if let Some(query) = query.strip_prefix('$') {
            let query = query.to_ascii_lowercase();
            let mut matches = self
                .skills
                .iter()
                .filter_map(|name| {
                    skill_match_rank(&name.to_ascii_lowercase(), &query).map(|rank| {
                        let mut item = CommandCompletionItem::new(
                            format!("${name}"),
                            CommandCompletionKind::Skill,
                        );
                        item.description = self.skill_descriptions.get(name).cloned();
                        (rank, item)
                    })
                })
                .collect::<Vec<_>>();
            sort_completion_matches(&mut matches);
            return matches
                .into_iter()
                .take(limit)
                .map(|(_, item)| item)
                .collect();
        }
        let Some(query) = query.strip_prefix('/') else {
            return Vec::new();
        };
        let query = query.to_ascii_lowercase();
        let mut matches = self
            .completions
            .iter()
            .filter_map(|item| {
                let name = item.value.strip_prefix('/')?.to_ascii_lowercase();
                let rank = if item.kind == CommandCompletionKind::Skill {
                    skill_match_rank(&name, &query)
                } else {
                    command_match_rank(&name, &query)
                }?;
                Some((rank, item.clone()))
            })
            .collect::<Vec<_>>();
        sort_completion_matches(&mut matches);
        matches
            .into_iter()
            .take(limit)
            .map(|(_, item)| item)
            .collect()
    }
}

fn normalize_completion_description(description: &str) -> Option<String> {
    let mut normalized = String::new();
    let mut pending_space = false;
    for character in description.chars() {
        if character.is_whitespace() || character.is_control() {
            pending_space = !normalized.is_empty();
            continue;
        }
        let separator_bytes = usize::from(pending_space);
        if normalized
            .len()
            .saturating_add(separator_bytes)
            .saturating_add(character.len_utf8())
            > MAX_COMPLETION_DESCRIPTION_BYTES
        {
            break;
        }
        if pending_space {
            normalized.push(' ');
            pending_space = false;
        }
        normalized.push(character);
    }
    (!normalized.is_empty()).then_some(normalized)
}

fn sort_completion_matches(matches: &mut [(u8, CommandCompletionItem)]) {
    matches.sort_by(|(left_rank, left), (right_rank, right)| {
        left_rank
            .cmp(right_rank)
            .then_with(|| left.value.cmp(&right.value))
            .then_with(|| left.kind.precedence().cmp(&right.kind.precedence()))
    });
}

fn command_match_rank(name: &str, query: &str) -> Option<u8> {
    if name == query {
        Some(0)
    } else if name.starts_with(query) {
        Some(1)
    } else {
        None
    }
}

fn skill_match_rank(name: &str, query: &str) -> Option<u8> {
    if name == query {
        return Some(0);
    }
    if name.starts_with(query) {
        return Some(1);
    }
    if name.contains(query) {
        return Some(2);
    }
    is_subsequence(name, query).then_some(3)
}

fn is_subsequence(name: &str, query: &str) -> bool {
    if query.is_empty() {
        return true;
    }
    let mut query = query.bytes();
    let Some(mut expected) = query.next() else {
        return true;
    };
    for byte in name.bytes() {
        if byte != expected {
            continue;
        }
        let Some(next) = query.next() else {
            return true;
        };
        expected = next;
    }
    false
}

fn split_skill_mention(input: &str) -> Option<(&str, &str)> {
    let remaining = input.strip_prefix('$')?;
    let separator = remaining.find(char::is_whitespace);
    Some(separator.map_or((remaining, ""), |index| {
        (&remaining[..index], remaining[index..].trim_start())
    }))
}

fn parse_image(input: &str) -> Result<SlashCommand, CommandError> {
    let argument = input
        .strip_prefix("/image ")
        .map(str::trim)
        .filter(|argument| !argument.is_empty())
        .ok_or(CommandError::Invalid)?;
    if argument.len() > MAX_IMAGE_PATH_BYTES || argument.chars().any(char::is_control) {
        return Err(CommandError::Invalid);
    }

    let mut parts = argument.split_whitespace();
    match parts.next() {
        Some("clear") if parts.next().is_none() => Ok(SlashCommand::ImageClear),
        Some("clear") | None => Err(CommandError::Invalid),
        Some("remove") => {
            let index = parts
                .next()
                .and_then(|value| value.parse::<usize>().ok())
                .filter(|index| (1..=MAX_COMPOSER_ATTACHMENTS).contains(index))
                .ok_or(CommandError::Invalid)?;
            if parts.next().is_some() {
                return Err(CommandError::Invalid);
            }
            Ok(SlashCommand::ImageRemove(index))
        }
        Some(_) => Ok(SlashCommand::Image(argument.to_owned())),
    }
}

fn collect_names<T, I>(values: I) -> Result<BTreeSet<String>, CommandError>
where
    T: AsRef<str>,
    I: IntoIterator<Item = T>,
{
    let mut names = BTreeSet::new();
    for value in values {
        let value = value.as_ref();
        if !valid_name(value) {
            return Err(CommandError::Invalid);
        }
        if !names.insert(value.to_owned()) {
            return Err(CommandError::Duplicate);
        }
        if names.len() > MAX_COMMANDS {
            return Err(CommandError::TooMany);
        }
    }
    Ok(names)
}

fn valid_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 128
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
}
