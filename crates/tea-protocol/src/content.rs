use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use thiserror::Error;

use crate::ToolCallId;
use crate::external::{ExternalContentError, HostedToolActivity, SourceCitation};
use crate::metadata::{ProtocolMetadataError, validate_json_bounds};

/// Maximum UTF-8 bytes in one text or thinking content block.
pub const MAX_TEXT_BLOCK_BYTES: usize = 1024 * 1024;
/// Maximum encoded bytes in one inline Base64 image.
pub const MAX_INLINE_IMAGE_BASE64_BYTES: usize = 6 * 1024 * 1024;
/// Maximum encoded JSON bytes for tool arguments.
pub const MAX_TOOL_ARGUMENT_BYTES: usize = 256 * 1024;
/// Maximum nesting depth for tool arguments.
pub const MAX_TOOL_ARGUMENT_DEPTH: usize = 32;
/// Maximum UTF-8 bytes in one opaque provider tool-call identifier.
pub const MAX_PROVIDER_TOOL_CALL_ID_BYTES: usize = 256;

/// A provider-neutral message content block.
#[derive(Debug, Clone, PartialEq)]
pub enum ContentBlock {
    /// Visible text content.
    Text {
        /// UTF-8 text.
        text: String,
    },
    /// Model-visible contextual text hidden by presentation surfaces.
    ContextualText {
        /// UTF-8 contextual text.
        text: String,
    },
    /// Model reasoning content that products may choose to hide.
    Thinking {
        /// UTF-8 reasoning text.
        text: String,
    },
    /// Image content with a MIME type and source.
    Image {
        /// Image MIME type.
        mime_type: String,
        /// Inline or referenced image source.
        source: ImageSource,
    },
    /// A request to invoke a named tool.
    ToolCall {
        /// Canonical tool-call identifier.
        tool_call_id: ToolCallId,
        /// Opaque provider identifier used to continue the model conversation.
        provider_call_id: Option<String>,
        /// Registered tool name.
        tool_name: String,
        /// Provider-neutral JSON arguments.
        arguments: Value,
    },
    /// One complete provider-hosted activity, including normalized sources.
    HostedTool {
        /// Validated activity and opaque same-provider continuation state.
        activity: HostedToolActivity,
    },
    /// A normalized citation associated with assistant text.
    Citation {
        /// Validated cited source and optional provider continuation state.
        citation: SourceCitation,
    },
}

impl ContentBlock {
    /// Creates a validated visible text block.
    ///
    /// # Errors
    ///
    /// Returns [`ContentValidationError::InvalidText`] when the text is empty,
    /// contains a null character, or exceeds [`MAX_TEXT_BLOCK_BYTES`].
    pub fn text(text: impl Into<String>) -> Result<Self, ContentValidationError> {
        let text = text.into();
        validate_text(&text)?;
        Ok(Self::Text { text })
    }

    /// Creates a validated model-visible contextual text block.
    ///
    /// Presentation surfaces must not render this block as user-authored text.
    ///
    /// # Errors
    ///
    /// Returns [`ContentValidationError::InvalidText`] when the text is empty,
    /// contains a null character, or exceeds [`MAX_TEXT_BLOCK_BYTES`].
    pub fn contextual_text(text: impl Into<String>) -> Result<Self, ContentValidationError> {
        let text = text.into();
        validate_text(&text)?;
        Ok(Self::ContextualText { text })
    }

    /// Creates a validated thinking block.
    ///
    /// # Errors
    ///
    /// Returns [`ContentValidationError::InvalidText`] when the text is empty,
    /// contains a null character, or exceeds [`MAX_TEXT_BLOCK_BYTES`].
    pub fn thinking(text: impl Into<String>) -> Result<Self, ContentValidationError> {
        let text = text.into();
        validate_text(&text)?;
        Ok(Self::Thinking { text })
    }

    /// Creates a validated inline Base64 image block.
    ///
    /// # Errors
    ///
    /// Returns an error when the MIME type is invalid or the data is empty,
    /// oversized, or not valid standard Base64.
    pub fn inline_image(
        mime_type: impl Into<String>,
        data: impl Into<String>,
    ) -> Result<Self, ContentValidationError> {
        let mime_type = mime_type.into();
        let data = data.into();
        validate_mime_type(&mime_type)?;
        if data.is_empty() || data.len() > MAX_INLINE_IMAGE_BASE64_BYTES {
            return Err(ContentValidationError::InvalidImageData);
        }
        STANDARD
            .decode(data.as_bytes())
            .map_err(|_| ContentValidationError::InvalidImageData)?;
        Ok(Self::Image {
            mime_type,
            source: ImageSource::InlineBase64 { data },
        })
    }

    /// Creates a validated referenced image block.
    ///
    /// # Errors
    ///
    /// Returns an error when the MIME type or bounded reference is invalid.
    pub fn image_reference(
        mime_type: impl Into<String>,
        reference: impl Into<String>,
    ) -> Result<Self, ContentValidationError> {
        let mime_type = mime_type.into();
        let reference = reference.into();
        validate_mime_type(&mime_type)?;
        if reference.is_empty() || reference.len() > 1024 || reference.chars().any(char::is_control)
        {
            return Err(ContentValidationError::InvalidImageReference);
        }
        Ok(Self::Image {
            mime_type,
            source: ImageSource::Reference { reference },
        })
    }

    /// Creates a validated tool-call block.
    ///
    /// # Errors
    ///
    /// Returns an error when the tool name is invalid, arguments are not an
    /// object, or arguments exceed the byte or nesting limits.
    pub fn tool_call(
        tool_call_id: ToolCallId,
        tool_name: impl Into<String>,
        arguments: Value,
    ) -> Result<Self, ContentValidationError> {
        Self::tool_call_inner(tool_call_id, None, tool_name.into(), arguments)
    }

    /// Creates a validated tool-call block that retains its provider identifier.
    ///
    /// # Errors
    ///
    /// Returns an error when the provider identifier, tool name, or arguments
    /// violate their protocol bounds.
    pub fn tool_call_with_provider_id(
        tool_call_id: ToolCallId,
        provider_call_id: impl Into<String>,
        tool_name: impl Into<String>,
        arguments: Value,
    ) -> Result<Self, ContentValidationError> {
        Self::tool_call_inner(
            tool_call_id,
            Some(provider_call_id.into()),
            tool_name.into(),
            arguments,
        )
    }

    fn tool_call_inner(
        tool_call_id: ToolCallId,
        provider_call_id: Option<String>,
        tool_name: String,
        arguments: Value,
    ) -> Result<Self, ContentValidationError> {
        if let Some(provider_call_id) = provider_call_id.as_deref() {
            validate_provider_tool_call_id(provider_call_id)?;
        }
        validate_tool_name(&tool_name)?;
        if !arguments.is_object() {
            return Err(ContentValidationError::ToolArgumentsMustBeObject);
        }
        validate_json_bounds(&arguments, MAX_TOOL_ARGUMENT_BYTES, MAX_TOOL_ARGUMENT_DEPTH)?;
        Ok(Self::ToolCall {
            tool_call_id,
            provider_call_id,
            tool_name,
            arguments,
        })
    }

    /// Wraps one validated provider-hosted tool activity.
    #[must_use]
    pub fn hosted_tool(activity: HostedToolActivity) -> Self {
        Self::HostedTool { activity }
    }

    /// Wraps one validated external source citation.
    #[must_use]
    pub fn citation(citation: SourceCitation) -> Self {
        Self::Citation { citation }
    }

    /// Returns the opaque provider identifier for a tool-call block.
    #[must_use]
    pub fn provider_call_id(&self) -> Option<&str> {
        match self {
            Self::ToolCall {
                provider_call_id, ..
            } => provider_call_id.as_deref(),
            Self::HostedTool { activity } => Some(activity.provider_call_id()),
            _ => None,
        }
    }

    pub(crate) fn validate(&self) -> Result<(), ContentValidationError> {
        match self {
            Self::Text { text } | Self::ContextualText { text } | Self::Thinking { text } => {
                validate_text(text)
            }
            Self::Image { mime_type, source } => {
                validate_mime_type(mime_type)?;
                match source {
                    ImageSource::InlineBase64 { data } => {
                        if data.is_empty() || data.len() > MAX_INLINE_IMAGE_BASE64_BYTES {
                            return Err(ContentValidationError::InvalidImageData);
                        }
                        STANDARD
                            .decode(data.as_bytes())
                            .map_err(|_| ContentValidationError::InvalidImageData)?;
                    }
                    ImageSource::Reference { reference } => {
                        if reference.is_empty()
                            || reference.len() > 1024
                            || reference.chars().any(char::is_control)
                        {
                            return Err(ContentValidationError::InvalidImageReference);
                        }
                    }
                }
                Ok(())
            }
            Self::ToolCall {
                provider_call_id,
                tool_name,
                arguments,
                ..
            } => {
                if let Some(provider_call_id) = provider_call_id.as_deref() {
                    validate_provider_tool_call_id(provider_call_id)?;
                }
                validate_tool_name(tool_name)?;
                if !arguments.is_object() {
                    return Err(ContentValidationError::ToolArgumentsMustBeObject);
                }
                validate_json_bounds(arguments, MAX_TOOL_ARGUMENT_BYTES, MAX_TOOL_ARGUMENT_DEPTH)?;
                Ok(())
            }
            Self::HostedTool { activity } => activity.validate().map_err(Into::into),
            Self::Citation { citation } => citation.validate().map_err(Into::into),
        }
    }

    pub(crate) const fn valid_for_user(&self) -> bool {
        matches!(
            self,
            Self::Text { .. } | Self::ContextualText { .. } | Self::Image { .. }
        )
    }

    pub(crate) const fn valid_for_assistant(&self) -> bool {
        matches!(
            self,
            Self::Text { .. }
                | Self::Thinking { .. }
                | Self::ToolCall { .. }
                | Self::HostedTool { .. }
                | Self::Citation { .. }
        )
    }

    pub(crate) const fn valid_for_tool_result(&self) -> bool {
        matches!(self, Self::Text { .. } | Self::Image { .. })
    }
}

impl Serialize for ContentBlock {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.validate().map_err(serde::ser::Error::custom)?;
        SerializableContentBlock::from(self).serialize(serializer)
    }
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SerializableContentBlock<'a> {
    Text {
        text: &'a str,
    },
    ContextualText {
        text: &'a str,
    },
    Thinking {
        text: &'a str,
    },
    Image {
        #[serde(rename = "mimeType")]
        mime_type: &'a str,
        source: &'a ImageSource,
    },
    ToolCall {
        #[serde(rename = "toolCallId")]
        tool_call_id: &'a ToolCallId,
        #[serde(rename = "providerCallId", skip_serializing_if = "Option::is_none")]
        provider_call_id: Option<&'a str>,
        #[serde(rename = "toolName")]
        tool_name: &'a str,
        arguments: &'a Value,
    },
    HostedTool {
        activity: &'a HostedToolActivity,
    },
    Citation {
        citation: &'a SourceCitation,
    },
}

impl<'a> From<&'a ContentBlock> for SerializableContentBlock<'a> {
    fn from(value: &'a ContentBlock) -> Self {
        match value {
            ContentBlock::Text { text } => Self::Text { text },
            ContentBlock::ContextualText { text } => Self::ContextualText { text },
            ContentBlock::Thinking { text } => Self::Thinking { text },
            ContentBlock::Image { mime_type, source } => Self::Image { mime_type, source },
            ContentBlock::ToolCall {
                tool_call_id,
                provider_call_id,
                tool_name,
                arguments,
            } => Self::ToolCall {
                tool_call_id,
                provider_call_id: provider_call_id.as_deref(),
                tool_name,
                arguments,
            },
            ContentBlock::HostedTool { activity } => Self::HostedTool { activity },
            ContentBlock::Citation { citation } => Self::Citation { citation },
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum RawContentBlock {
    Text {
        text: String,
    },
    ContextualText {
        text: String,
    },
    Thinking {
        text: String,
    },
    Image {
        #[serde(rename = "mimeType")]
        mime_type: String,
        source: ImageSource,
    },
    ToolCall {
        #[serde(rename = "toolCallId")]
        tool_call_id: ToolCallId,
        #[serde(rename = "providerCallId", default)]
        provider_call_id: Option<String>,
        #[serde(rename = "toolName")]
        tool_name: String,
        arguments: Value,
    },
    HostedTool {
        activity: HostedToolActivity,
    },
    Citation {
        citation: SourceCitation,
    },
}

impl<'de> Deserialize<'de> for ContentBlock {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawContentBlock::deserialize(deserializer)?;
        let result = match raw {
            RawContentBlock::Text { text } => Self::text(text),
            RawContentBlock::ContextualText { text } => Self::contextual_text(text),
            RawContentBlock::Thinking { text } => Self::thinking(text),
            RawContentBlock::Image { mime_type, source } => match source {
                ImageSource::InlineBase64 { data } => Self::inline_image(mime_type, data),
                ImageSource::Reference { reference } => Self::image_reference(mime_type, reference),
            },
            RawContentBlock::ToolCall {
                tool_call_id,
                provider_call_id,
                tool_name,
                arguments,
            } => Self::tool_call_inner(tool_call_id, provider_call_id, tool_name, arguments),
            RawContentBlock::HostedTool { activity } => activity
                .validate()
                .map_err(ContentValidationError::from)
                .map(|()| Self::hosted_tool(activity)),
            RawContentBlock::Citation { citation } => citation
                .validate()
                .map_err(ContentValidationError::from)
                .map(|()| Self::citation(citation)),
        };
        result.map_err(serde::de::Error::custom)
    }
}

/// Source of an image content block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSource {
    /// Base64 data embedded directly in the protocol value.
    InlineBase64 {
        /// Standard padded Base64 text.
        data: String,
    },
    /// Stable reference resolved by an adapter or artifact store.
    Reference {
        /// Bounded opaque reference string.
        reference: String,
    },
}

/// Error returned when validating content blocks.
#[derive(Debug, Error)]
pub enum ContentValidationError {
    /// Text is empty, too large, or contains a null character.
    #[error("text content is empty, too large, or contains a null character")]
    InvalidText,
    /// The MIME type is not a supported canonical image type.
    #[error("image MIME type must use canonical image/type syntax")]
    InvalidMimeType,
    /// Inline image data is empty, too large, or not valid standard Base64.
    #[error("inline image data is invalid")]
    InvalidImageData,
    /// An image reference is empty, too large, or contains controls.
    #[error("image reference is invalid")]
    InvalidImageReference,
    /// The tool name is not canonical.
    #[error(
        "tool name must start with a lowercase letter and contain lowercase ASCII, digits, '_', '-', or '.'"
    )]
    InvalidToolName,
    /// The provider tool-call identifier is empty, oversized, or contains controls.
    #[error("provider tool-call identifier is invalid")]
    InvalidProviderToolCallId,
    /// Tool arguments must be a JSON object.
    #[error("tool arguments must be a JSON object")]
    ToolArgumentsMustBeObject,
    /// Tool arguments exceed JSON byte or nesting limits.
    #[error("tool arguments exceed protocol bounds: {0}")]
    ToolArgumentsOutOfBounds(#[from] ProtocolMetadataError),
    /// Hosted activity, source, citation, or continuation content is invalid.
    #[error("external content is invalid: {0}")]
    InvalidExternalContent(#[from] ExternalContentError),
}

fn validate_text(text: &str) -> Result<(), ContentValidationError> {
    if text.is_empty() || text.len() > MAX_TEXT_BLOCK_BYTES || text.contains('\0') {
        Err(ContentValidationError::InvalidText)
    } else {
        Ok(())
    }
}

fn validate_mime_type(value: &str) -> Result<(), ContentValidationError> {
    let subtype = value
        .strip_prefix("image/")
        .ok_or(ContentValidationError::InvalidMimeType)?;
    if subtype.is_empty()
        || !subtype.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'+' | b'-' | b'.')
        })
    {
        return Err(ContentValidationError::InvalidMimeType);
    }
    Ok(())
}

pub(crate) fn validate_tool_name(value: &str) -> Result<(), ContentValidationError> {
    let mut bytes = value.bytes();
    if !bytes.next().is_some_and(|byte| byte.is_ascii_lowercase())
        || value.len() > 128
        || !bytes.all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'-' | b'.')
        })
    {
        return Err(ContentValidationError::InvalidToolName);
    }
    Ok(())
}

pub(crate) fn validate_provider_tool_call_id(value: &str) -> Result<(), ContentValidationError> {
    if value.is_empty()
        || value.len() > MAX_PROVIDER_TOOL_CALL_ID_BYTES
        || value.chars().any(char::is_control)
    {
        Err(ContentValidationError::InvalidProviderToolCallId)
    } else {
        Ok(())
    }
}
