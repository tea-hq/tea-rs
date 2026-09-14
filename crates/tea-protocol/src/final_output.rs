use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use thiserror::Error;

/// Maximum encoded JSON bytes in a final-output schema.
pub const MAX_FINAL_OUTPUT_SCHEMA_BYTES: usize = 256 * 1024;
/// Maximum JSON nesting depth in a final-output schema.
pub const MAX_FINAL_OUTPUT_SCHEMA_DEPTH: usize = 32;
/// Deterministic model instruction required by JSON-object wire protocols.
pub const FINAL_JSON_OBJECT_INSTRUCTION: &str =
    "Return the final response as a JSON object. Do not use Markdown.";

/// Provider-neutral contract for the final visible assistant output of one run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FinalOutputFormat {
    /// Require the final assistant text to decode as one JSON object.
    JsonObject,
    /// Require the final assistant text to conform to the supplied JSON Schema.
    JsonSchema {
        /// Provider-neutral JSON Schema passed unchanged to capable providers.
        schema: Value,
    },
}

impl FinalOutputFormat {
    /// Validates the schema representation, resource bounds, and Draft 2020-12
    /// meta-schema semantics.
    ///
    /// # Errors
    ///
    /// Returns an error when a JSON Schema is not an object, exceeds the
    /// encoded byte or nesting-depth limit, or is not valid Draft 2020-12.
    pub fn validate(&self) -> Result<(), FinalOutputFormatError> {
        let Self::JsonSchema { schema } = self else {
            return Ok(());
        };
        if !schema.is_object() {
            return Err(FinalOutputFormatError::SchemaMustBeObject);
        }
        if exceeds_json_depth(schema, 1)
            || serde_json::to_vec(schema)
                .map_err(|_| FinalOutputFormatError::SchemaOutOfBounds)?
                .len()
                > MAX_FINAL_OUTPUT_SCHEMA_BYTES
        {
            return Err(FinalOutputFormatError::SchemaOutOfBounds);
        }
        jsonschema::draft202012::options()
            .build(schema)
            .map_err(|_| FinalOutputFormatError::InvalidSchema)?;
        Ok(())
    }
}

#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum FinalOutputFormatRef<'a> {
    JsonObject {},
    JsonSchema { schema: &'a Value },
}

impl Serialize for FinalOutputFormat {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.validate().map_err(serde::ser::Error::custom)?;
        match self {
            Self::JsonObject => FinalOutputFormatRef::JsonObject {}.serialize(serializer),
            Self::JsonSchema { schema } => {
                FinalOutputFormatRef::JsonSchema { schema }.serialize(serializer)
            }
        }
    }
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum RawFinalOutputFormat {
    JsonObject {},
    JsonSchema { schema: Value },
}

impl<'de> Deserialize<'de> for FinalOutputFormat {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let format = match RawFinalOutputFormat::deserialize(deserializer)? {
            RawFinalOutputFormat::JsonObject {} => Self::JsonObject,
            RawFinalOutputFormat::JsonSchema { schema } => Self::JsonSchema { schema },
        };
        format.validate().map_err(serde::de::Error::custom)?;
        Ok(format)
    }
}

/// Invalid final-output contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum FinalOutputFormatError {
    /// A JSON Schema must be represented by a JSON object.
    #[error("final-output JSON Schema must be an object")]
    SchemaMustBeObject,
    /// A JSON Schema exceeds its encoded byte or nesting-depth limit.
    #[error("final-output JSON Schema exceeds supported bounds")]
    SchemaOutOfBounds,
    /// A JSON Schema violates the Draft 2020-12 meta-schema.
    #[error("final-output JSON Schema is not valid Draft 2020-12")]
    InvalidSchema,
}

fn exceeds_json_depth(value: &Value, depth: usize) -> bool {
    if depth > MAX_FINAL_OUTPUT_SCHEMA_DEPTH {
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
