use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::resources::frontmatter::parse;
use crate::{CodingError, CodingErrorCode};

const MAX_TEMPLATES: usize = 128;

#[derive(Debug, Deserialize)]
struct PromptFrontmatter {
    name: String,
    description: String,
    #[serde(flatten)]
    defaults: BTreeMap<String, String>,
}

/// Validated non-executable Markdown prompt template.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PromptTemplate {
    name: String,
    description: String,
    body: String,
    defaults: BTreeMap<String, String>,
}

impl PromptTemplate {
    /// Returns the canonical template name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }
    /// Returns metadata shown by command discovery.
    #[must_use]
    pub fn description(&self) -> &str {
        &self.description
    }
    /// Expands `$1`..`$9` and `${name}` once without recursion or evaluation.
    #[must_use]
    pub fn expand(&self, positional: &[String], named: &BTreeMap<String, String>) -> String {
        let mut output = String::with_capacity(self.body.len());
        let mut remaining = self.body.as_str();
        while let Some(marker) = remaining.find('$') {
            output.push_str(&remaining[..marker]);
            remaining = &remaining[marker..];
            if let Some(index) = remaining
                .as_bytes()
                .get(1)
                .copied()
                .filter(u8::is_ascii_digit)
                .map(|digit| usize::from(digit - b'0'))
                .filter(|index| (1..=9).contains(index))
            {
                output.push_str(positional.get(index - 1).map_or("", String::as_str));
                remaining = &remaining[2..];
            } else if let Some(after_open) = remaining.strip_prefix("${") {
                if let Some(end) = after_open.find('}') {
                    let name = &after_open[..end];
                    if let Some(default) = self.defaults.get(name) {
                        output.push_str(named.get(name).map_or(default, String::as_ref));
                        remaining = &after_open[end + 1..];
                    } else {
                        output.push('$');
                        remaining = &remaining[1..];
                    }
                } else {
                    output.push('$');
                    remaining = &remaining[1..];
                }
            } else {
                output.push('$');
                remaining = &remaining[1..];
            }
        }
        output.push_str(remaining);
        output
    }
}

pub(crate) fn discover(
    global: Option<&Path>,
    trusted_project: Option<&Path>,
) -> Result<Vec<PromptTemplate>, CodingError> {
    let mut templates = BTreeMap::new();
    for root in [global, trusted_project].into_iter().flatten() {
        for template in load_layer(root)? {
            templates.insert(template.name.clone(), template);
        }
    }
    if templates.len() > MAX_TEMPLATES {
        return Err(invalid());
    }
    Ok(templates.into_values().collect())
}

fn load_layer(root: &Path) -> Result<Vec<PromptTemplate>, CodingError> {
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(_) => return Err(not_found()),
    };
    let mut paths = entries
        .filter_map(Result::ok)
        .filter(|entry| {
            entry.file_type().is_ok_and(|kind| kind.is_file())
                && entry
                    .path()
                    .extension()
                    .is_some_and(|extension| extension == "md")
        })
        .map(|entry| entry.path())
        .collect::<Vec<PathBuf>>();
    paths.sort();
    let mut layer = BTreeMap::new();
    for path in paths {
        let document =
            parse::<PromptFrontmatter>(&fs::read_to_string(path).map_err(|_| not_found())?)?;
        let PromptFrontmatter {
            name,
            description,
            defaults,
        } = document.metadata;
        if !valid_name(&name) {
            return Err(invalid());
        }
        let defaults = defaults
            .into_iter()
            .filter_map(|(key, value)| {
                key.strip_prefix("default_")
                    .map(|name| (name.to_owned(), value))
            })
            .collect();
        let template = PromptTemplate {
            name: name.clone(),
            description,
            body: document.body,
            defaults,
        };
        if layer.insert(name, template).is_some() {
            return Err(CodingError::new(
                CodingErrorCode::InvalidInput,
                "duplicate prompt template name in one layer",
            ));
        }
    }
    Ok(layer.into_values().collect())
}

fn valid_name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'-' | b'_')
        })
}
fn invalid() -> CodingError {
    CodingError::new(CodingErrorCode::InvalidInput, "prompt template is invalid")
}
fn not_found() -> CodingError {
    CodingError::new(
        CodingErrorCode::NotFound,
        "prompt template directory is missing",
    )
}
