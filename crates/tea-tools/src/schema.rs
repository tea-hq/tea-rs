use std::fmt;
use std::sync::Arc;

use jsonschema::Validator;
use serde_json::Value;
use thiserror::Error;

/// Maximum encoded JSON bytes in one tool schema or validated value.
pub const MAX_TOOL_VALUE_BYTES: usize = 256 * 1024;
/// Maximum JSON nesting depth in one tool schema or validated value.
pub const MAX_TOOL_VALUE_DEPTH: usize = 32;
/// Maximum normalized validation errors returned for one value.
pub const MAX_SCHEMA_ERRORS: usize = 16;
const MAX_SCHEMA_ERROR_MESSAGE_BYTES: usize = 4096;

/// Offline compiled Draft 2020-12 tool schema.
#[derive(Clone)]
pub struct CompiledToolSchema {
    source: Value,
    validator: Arc<Validator>,
}

impl CompiledToolSchema {
    /// Compiles a bounded, self-contained Draft 2020-12 schema.
    ///
    /// # Errors
    ///
    /// Rejects oversized/deep schemas, external references, and invalid schema
    /// syntax. HTTP and file retrieval are not enabled.
    pub fn compile(source: Value) -> Result<Self, SchemaCompilationError> {
        validate_json_bounds(&source).map_err(|()| SchemaCompilationError::SchemaOutOfBounds)?;
        if contains_external_reference(&source) {
            return Err(SchemaCompilationError::ExternalReference);
        }
        let validator = jsonschema::draft202012::options()
            .build(&source)
            .map_err(|_| SchemaCompilationError::InvalidSchema)?;
        Ok(Self {
            source,
            validator: Arc::new(validator),
        })
    }

    /// Validates a bounded JSON value.
    ///
    /// # Errors
    ///
    /// Returns bounded deterministic diagnostics or a value-bounds failure.
    pub fn validate(&self, value: &Value) -> Result<(), SchemaValidationFailure> {
        validate_json_bounds(value).map_err(|()| SchemaValidationFailure::ValueOutOfBounds)?;
        self.validate_prebounded(value)
    }

    /// Validates a JSON value after the caller has enforced its own resource
    /// bounds.
    ///
    /// # Errors
    ///
    /// Returns bounded deterministic diagnostics for schema violations.
    pub fn validate_prebounded(&self, value: &Value) -> Result<(), SchemaValidationFailure> {
        let mut errors = self
            .validator
            .iter_errors(value)
            .map(|error| {
                let instance_path = error.instance_path.to_string();
                let schema_path = error.schema_path.to_string();
                let code = schema_keyword(&schema_path);
                let message = truncate_utf8(&error.to_string(), MAX_SCHEMA_ERROR_MESSAGE_BYTES);
                SchemaValidationError {
                    code,
                    instance_path,
                    schema_path,
                    message,
                }
            })
            .take(MAX_SCHEMA_ERRORS)
            .collect::<Vec<_>>();
        errors.sort_by(|left, right| {
            (
                left.instance_path.as_str(),
                left.schema_path.as_str(),
                left.message.as_str(),
            )
                .cmp(&(
                    right.instance_path.as_str(),
                    right.schema_path.as_str(),
                    right.message.as_str(),
                ))
        });
        if errors.is_empty() {
            Ok(())
        } else {
            Err(SchemaValidationFailure::Invalid { errors })
        }
    }

    /// Returns the immutable source schema.
    #[must_use]
    pub const fn source(&self) -> &Value {
        &self.source
    }
}

impl fmt::Debug for CompiledToolSchema {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompiledToolSchema")
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

/// Error compiling a tool schema.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum SchemaCompilationError {
    /// Schema exceeds encoded byte or nesting limits.
    #[error("tool schema exceeds supported bounds")]
    SchemaOutOfBounds,
    /// Schema contains a non-local reference while retrieval is disabled.
    #[error("tool schema contains an external reference")]
    ExternalReference,
    /// Schema is not valid Draft 2020-12 syntax.
    #[error("tool schema is invalid")]
    InvalidSchema,
}

/// Normalized bounded schema validation error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaValidationError {
    code: String,
    instance_path: String,
    schema_path: String,
    message: String,
}

impl SchemaValidationError {
    /// Returns the schema keyword or stable validation code.
    #[must_use]
    pub fn code(&self) -> &str {
        &self.code
    }

    /// Returns the JSON Pointer into the rejected instance.
    #[must_use]
    pub fn instance_path(&self) -> &str {
        &self.instance_path
    }

    /// Returns the JSON Pointer into the schema.
    #[must_use]
    pub fn schema_path(&self) -> &str {
        &self.schema_path
    }

    /// Returns the bounded English technical message.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

/// Failure validating one tool argument or output value.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SchemaValidationFailure {
    /// Value exceeds encoded byte or nesting limits.
    #[error("tool value exceeds supported bounds")]
    ValueOutOfBounds,
    /// Value violates one or more schema constraints.
    #[error("tool value violates its JSON Schema")]
    Invalid {
        /// Sorted bounded validation diagnostics.
        errors: Vec<SchemaValidationError>,
    },
}

impl SchemaValidationFailure {
    /// Returns normalized diagnostics; bounds failures have no per-path errors.
    #[must_use]
    pub fn errors(&self) -> &[SchemaValidationError] {
        match self {
            Self::ValueOutOfBounds => &[],
            Self::Invalid { errors } => errors,
        }
    }
}

fn validate_json_bounds(value: &Value) -> Result<(), ()> {
    if exceeds_json_depth(value, 1)
        || serde_json::to_vec(value).map_err(|_| ())?.len() > MAX_TOOL_VALUE_BYTES
    {
        Err(())
    } else {
        Ok(())
    }
}

fn contains_external_reference(value: &Value) -> bool {
    match value {
        Value::Array(values) => values.iter().any(contains_external_reference),
        Value::Object(values) => values.iter().any(|(key, value)| {
            (key == "$ref"
                && value
                    .as_str()
                    .is_some_and(|reference| !reference.starts_with('#')))
                || contains_external_reference(value)
        }),
        _ => false,
    }
}

fn exceeds_json_depth(value: &Value, depth: usize) -> bool {
    if depth > MAX_TOOL_VALUE_DEPTH {
        return true;
    }
    match value {
        Value::Array(values) => values
            .iter()
            .any(|value| exceeds_json_depth(value, depth + 1)),
        Value::Object(values) => values
            .values()
            .any(|value| exceeds_json_depth(value, depth + 1)),
        _ => false,
    }
}

fn schema_keyword(schema_path: &str) -> String {
    schema_path
        .rsplit('/')
        .find(|segment| !segment.is_empty() && !segment.bytes().all(|byte| byte.is_ascii_digit()))
        .unwrap_or("schema_validation")
        .replace('~', "_")
}

fn truncate_utf8(value: &str, max_bytes: usize) -> String {
    if value.len() <= max_bytes {
        return value.to_owned();
    }
    let mut end = max_bytes;
    while !value.is_char_boundary(end) {
        end -= 1;
    }
    value[..end].to_owned()
}
