use serde_json::json;
use tea_tools::{
    CompiledToolSchema, MAX_SCHEMA_ERRORS, SchemaCompilationError, SchemaValidationFailure,
};

#[test]
fn draft_2020_12_object_schema_compiles_and_validates() {
    let schema = json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "$defs": {
            "path": { "type": "string", "minLength": 1 }
        },
        "properties": {
            "path": { "$ref": "#/$defs/path" },
            "count": { "type": "integer", "minimum": 1 }
        },
        "required": ["path", "count"],
        "additionalProperties": false
    });
    let compiled = CompiledToolSchema::compile(schema.clone()).unwrap();
    assert_eq!(compiled.source(), &schema);
    compiled
        .validate(&json!({"path":"notes.txt", "count": 1}))
        .unwrap();

    let errors = compiled
        .validate(&json!({"path":"", "extra":true}))
        .unwrap_err();
    assert!(!errors.errors().is_empty());
    assert!(errors.errors().len() <= MAX_SCHEMA_ERRORS);
    assert!(errors.errors().iter().all(|error| {
        !error.code().is_empty()
            && error.message().len() <= 4096
            && error.instance_path().starts_with('/')
            || error.instance_path().is_empty()
    }));
}

#[test]
fn invalid_schema_and_external_references_fail_offline() {
    assert_eq!(
        CompiledToolSchema::compile(json!({"type": 42})).unwrap_err(),
        SchemaCompilationError::InvalidSchema
    );
    assert_eq!(
        CompiledToolSchema::compile(json!({
            "type":"object",
            "properties":{"x":{"$ref":"https://example.com/schema.json"}}
        }))
        .unwrap_err(),
        SchemaCompilationError::ExternalReference
    );
    assert_eq!(
        CompiledToolSchema::compile(json!({
            "type":"object",
            "properties":{"x":{"$ref":"file:///tmp/schema.json"}}
        }))
        .unwrap_err(),
        SchemaCompilationError::ExternalReference
    );
}

#[test]
fn schema_and_instance_bounds_are_enforced() {
    let oversized = json!({"type":"object", "description":"x".repeat(256 * 1024)});
    assert_eq!(
        CompiledToolSchema::compile(oversized).unwrap_err(),
        SchemaCompilationError::SchemaOutOfBounds
    );

    let mut nested = json!({"type":"object"});
    for _ in 0..40 {
        nested = json!({"type":"object", "properties":{"next":nested}});
    }
    assert_eq!(
        CompiledToolSchema::compile(nested).unwrap_err(),
        SchemaCompilationError::SchemaOutOfBounds
    );

    let compiled = CompiledToolSchema::compile(json!({"type":"object"})).unwrap();
    assert!(matches!(
        compiled.validate(&json!({"data":"x".repeat(256 * 1024)})),
        Err(SchemaValidationFailure::ValueOutOfBounds)
    ));

    compiled
        .validate_prebounded(&json!({"data":"x".repeat(300 * 1024)}))
        .unwrap();

    let mut deep_value = json!(null);
    for _ in 0..40 {
        deep_value = json!({"next":deep_value});
    }
    compiled.validate_prebounded(&deep_value).unwrap();
}

#[test]
fn validation_errors_are_deterministically_sorted_and_capped() {
    let properties = (0..32)
        .map(|index| (format!("field_{index:02}"), json!({"type":"string"})))
        .collect::<serde_json::Map<_, _>>();
    let required = (0..32)
        .map(|index| json!(format!("field_{index:02}")))
        .collect::<Vec<_>>();
    let compiled = CompiledToolSchema::compile(json!({
        "type":"object",
        "properties":properties,
        "required":required
    }))
    .unwrap();
    let failure = compiled.validate(&json!({})).unwrap_err();
    let errors = failure.errors();
    assert_eq!(errors.len(), MAX_SCHEMA_ERRORS);
    assert!(errors.windows(2).all(|pair| {
        (
            pair[0].instance_path(),
            pair[0].schema_path(),
            pair[0].message(),
        ) <= (
            pair[1].instance_path(),
            pair[1].schema_path(),
            pair[1].message(),
        )
    }));
}
