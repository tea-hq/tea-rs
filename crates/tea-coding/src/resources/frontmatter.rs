use crate::{CodingError, CodingErrorCode};
use serde::de::DeserializeOwned;

pub(crate) const MAX_RESOURCE_BYTES: usize = 128 * 1024;
const MAX_FRONTMATTER_BYTES: usize = 16 * 1024;

pub(crate) struct FrontmatterDocument<T> {
    pub(crate) metadata: T,
    pub(crate) body: String,
}

pub(crate) fn parse<T: DeserializeOwned>(
    source: &str,
) -> Result<FrontmatterDocument<T>, CodingError> {
    if source.len() > MAX_RESOURCE_BYTES {
        return Err(invalid());
    }
    let opening_len = if source.starts_with("---\r\n") {
        5
    } else if source.starts_with("---\n") {
        4
    } else {
        return Err(invalid());
    };
    let mut cursor = opening_len;
    let mut closing = None;
    while cursor < source.len() {
        let line_end = source[cursor..]
            .find('\n')
            .map_or(source.len(), |offset| cursor + offset + 1);
        let line = source[cursor..line_end]
            .strip_suffix('\n')
            .unwrap_or(&source[cursor..line_end])
            .strip_suffix('\r')
            .unwrap_or_else(|| {
                source[cursor..line_end]
                    .strip_suffix('\n')
                    .unwrap_or(&source[cursor..line_end])
            });
        if line == "---" {
            closing = Some((cursor, line_end));
            break;
        }
        cursor = line_end;
    }
    let (frontmatter_end, body_start) = closing.ok_or_else(invalid)?;
    if frontmatter_end - opening_len > MAX_FRONTMATTER_BYTES {
        return Err(invalid());
    }
    let frontmatter = &source[opening_len..frontmatter_end];
    let value = serde_yaml::from_str::<serde_yaml::Value>(frontmatter).map_err(|_| invalid())?;
    if !value.is_mapping() {
        return Err(invalid());
    }
    let metadata = serde_yaml::from_value::<T>(value).map_err(|_| invalid())?;
    let body = source[body_start..].to_owned();
    if body.is_empty() || body.contains('\0') {
        return Err(invalid());
    }
    Ok(FrontmatterDocument { metadata, body })
}

pub(crate) fn invalid() -> CodingError {
    CodingError::new(
        CodingErrorCode::InvalidInput,
        "declarative resource is invalid",
    )
}
