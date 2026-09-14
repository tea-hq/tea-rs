use std::str::FromStr;

use serde_json::json;
use tea_model::{
    HostedToolKind, HostedToolOptions, ModelCapabilities, ModelDisplayName, ModelRequest,
    ModelRequestError, ModelSpec, ModelToolDefinition, ProviderId, ReasoningEffort,
    ReasoningOptions, ReasoningProfile, WebSearchLocation, WebSearchOptions,
};
use tea_protocol::{
    CanonicalMessage, ContentBlock, FinalOutputFormat, MessageId, ModelId, ProtocolMetadata,
    ProtocolTimestamp, TokenCount,
};

const MESSAGE_ID: &str = "0195a0b1-5e3d-73de-b461-0aa7aa000004";
const TIMESTAMP: &str = "2026-07-23T09:30:12.123Z";

fn tokens(value: u64) -> TokenCount {
    TokenCount::new(value).unwrap()
}

fn user_message(content: Vec<ContentBlock>) -> CanonicalMessage {
    CanonicalMessage::user(
        MessageId::from_str(MESSAGE_ID).unwrap(),
        content,
        ProtocolTimestamp::from_str(TIMESTAMP).unwrap(),
    )
    .unwrap()
}

fn model(capabilities: ModelCapabilities) -> ModelSpec {
    ModelSpec::new(
        ModelId::from_str("test/model").unwrap(),
        ProviderId::from_str("test-provider").unwrap(),
        ModelDisplayName::from_str("Test Model").unwrap(),
        tokens(32_000),
        tokens(8_000),
        capabilities,
    )
    .unwrap()
}

fn tool(name: &str) -> ModelToolDefinition {
    ModelToolDefinition::new(
        name,
        "Writes a bounded text file.",
        json!({
            "type": "object",
            "properties": {
                "path": { "type": "string" },
                "content": { "type": "string" }
            },
            "required": ["path", "content"]
        }),
    )
    .unwrap()
}

fn hosted_web_search(options: WebSearchOptions) -> ModelToolDefinition {
    ModelToolDefinition::hosted(
        "Searches the public web and returns cited sources.",
        json!({
            "type": "object",
            "properties": {
                "query": { "type": "string" }
            },
            "required": ["query"]
        }),
        HostedToolOptions::WebSearch(options),
    )
    .unwrap()
}

#[test]
fn request_preserves_provider_neutral_turn_snapshot() {
    let message = user_message(vec![ContentBlock::text("Create notes").unwrap()]);
    let metadata = ProtocolMetadata::from_entries([(
        "com.example.request".to_owned(),
        json!({"trace": "safe"}),
    )])
    .unwrap();
    let reasoning = ReasoningOptions::new(ReasoningEffort::Medium).with_budget(tokens(2_000));
    let final_output_format = FinalOutputFormat::JsonSchema {
        schema: json!({
            "type":"object",
            "properties":{"answer":{"type":"string"}},
            "required":["answer"],
            "additionalProperties":false
        }),
    };

    let request = ModelRequest::new(
        ModelId::from_str("test/model").unwrap(),
        vec![message.clone()],
    )
    .unwrap()
    .with_system_prompt("You are a careful assistant.")
    .unwrap()
    .with_tools(vec![tool("write_text_file")], true)
    .unwrap()
    .with_reasoning(reasoning)
    .with_final_output_format(final_output_format.clone())
    .with_max_output_tokens(tokens(4_000))
    .with_metadata(metadata.clone());

    assert_eq!(request.model_id().as_str(), "test/model");
    assert_eq!(
        request.system_prompt(),
        Some("You are a careful assistant.")
    );
    assert_eq!(request.messages(), &[message]);
    assert_eq!(request.tools()[0].name(), "write_text_file");
    assert!(request.allow_parallel_tool_calls());
    assert_eq!(request.reasoning(), Some(reasoning));
    assert_eq!(request.final_output_format(), Some(&final_output_format));
    assert_eq!(request.max_output_tokens(), Some(tokens(4_000)));
    assert_eq!(request.metadata(), &metadata);

    request
        .validate_for(&model(
            ModelCapabilities::text()
                .with_reasoning()
                .with_tools(true)
                .with_final_json_schema_with_tools(),
        ))
        .unwrap();
}

#[test]
fn final_output_formats_fail_closed_against_independent_capabilities() {
    let base = ModelRequest::new(
        ModelId::from_str("test/model").unwrap(),
        vec![user_message(vec![
            ContentBlock::text("Return JSON").unwrap(),
        ])],
    )
    .unwrap();
    let object_request = base
        .clone()
        .with_final_output_format(FinalOutputFormat::JsonObject);
    let schema_request = base.with_final_output_format(FinalOutputFormat::JsonSchema {
        schema: json!({"type":"object"}),
    });

    let neither = ModelCapabilities::text();
    let object_only = ModelCapabilities::text().with_final_json_object();
    let schema_only = ModelCapabilities::text().with_final_json_schema();
    let both = ModelCapabilities::text()
        .with_final_json_object()
        .with_final_json_schema();
    let cases = [
        (
            &object_request,
            neither,
            Err(ModelRequestError::FinalJsonObjectUnsupported),
        ),
        (
            &schema_request,
            neither,
            Err(ModelRequestError::FinalJsonSchemaUnsupported),
        ),
        (&object_request, object_only, Ok(())),
        (
            &schema_request,
            object_only,
            Err(ModelRequestError::FinalJsonSchemaUnsupported),
        ),
        (
            &object_request,
            schema_only,
            Err(ModelRequestError::FinalJsonObjectUnsupported),
        ),
        (&schema_request, schema_only, Ok(())),
        (&object_request, both, Ok(())),
        (&schema_request, both, Ok(())),
    ];

    for (request, capabilities, expected) in cases {
        assert_eq!(request.validate_for(&model(capabilities)), expected);
    }
}

#[test]
fn final_output_format_is_validated_for_direct_model_requests() {
    let request = ModelRequest::new(
        ModelId::from_str("test/model").unwrap(),
        vec![user_message(vec![
            ContentBlock::text("Return JSON").unwrap(),
        ])],
    )
    .unwrap()
    .with_final_output_format(FinalOutputFormat::JsonSchema {
        schema: json!({"type": 7}),
    });

    assert_eq!(
        request
            .validate_for(&model(ModelCapabilities::text().with_final_json_schema()))
            .unwrap_err(),
        ModelRequestError::InvalidFinalOutputFormat
    );
}

#[test]
fn final_json_schema_with_tools_requires_the_combination_capability() {
    let base = ModelRequest::new(
        ModelId::from_str("test/model").unwrap(),
        vec![user_message(vec![
            ContentBlock::text("Return JSON").unwrap(),
        ])],
    )
    .unwrap()
    .with_tools(vec![tool("write_text_file")], false)
    .unwrap();
    let schema_request = base
        .clone()
        .with_final_output_format(FinalOutputFormat::JsonSchema {
            schema: json!({"type":"object"}),
        });
    let schema_only = ModelCapabilities::text()
        .with_tools(false)
        .with_final_json_schema();

    let error = schema_request
        .validate_for(&model(schema_only))
        .unwrap_err();
    assert_eq!(
        error,
        ModelRequestError::FinalJsonSchemaWithToolsUnsupported
    );
    assert_eq!(
        error.to_string(),
        "model does not support JSON-Schema final output with tools"
    );

    let schema_with_tools = ModelCapabilities::text()
        .with_tools(false)
        .with_final_json_schema_with_tools();
    assert!(schema_with_tools.supports_final_json_schema());
    schema_request
        .validate_for(&model(schema_with_tools))
        .unwrap();

    base.with_final_output_format(FinalOutputFormat::JsonObject)
        .validate_for(&model(
            ModelCapabilities::text()
                .with_tools(false)
                .with_final_json_object(),
        ))
        .unwrap();
}

#[test]
fn tools_require_object_schemas_and_unique_names() {
    assert_eq!(
        ModelToolDefinition::new("bad tool", "description", json!({"type":"object"})).unwrap_err(),
        ModelRequestError::InvalidToolName
    );
    assert_eq!(
        ModelToolDefinition::new("read_file", "description", json!("string")).unwrap_err(),
        ModelRequestError::ToolSchemaMustBeObject
    );
    assert_eq!(
        ModelToolDefinition::new("read_file", "description", json!({"properties":{}})).unwrap_err(),
        ModelRequestError::ToolSchemaMustDeclareObject
    );

    let request = ModelRequest::new(
        ModelId::from_str("test/model").unwrap(),
        vec![user_message(vec![ContentBlock::text("Read").unwrap()])],
    )
    .unwrap();
    assert_eq!(
        request
            .with_tools(vec![tool("read_file"), tool("read_file")], false)
            .unwrap_err(),
        ModelRequestError::DuplicateToolName
    );
}

#[test]
fn hosted_web_search_is_typed_and_preserves_function_compatibility() {
    let function = tool("read_file");
    assert!(function.as_function().is_some());
    assert!(function.as_hosted().is_none());
    assert_eq!(function.hosted_kind(), None);

    let location = WebSearchLocation::new()
        .with_country("US")
        .unwrap()
        .with_city("San Francisco")
        .unwrap()
        .with_region("California")
        .unwrap()
        .with_timezone("America/Los_Angeles")
        .unwrap();
    let options = WebSearchOptions::new()
        .with_allowed_domains(["docs.rs", "www.rust-lang.org"])
        .unwrap()
        .with_location(location);
    let hosted = hosted_web_search(options);

    assert_eq!(hosted.name(), "web_search");
    assert_eq!(hosted.hosted_kind(), Some(HostedToolKind::WebSearch));
    assert!(hosted.as_function().is_none());
    let hosted_definition = hosted.as_hosted().unwrap();
    assert_eq!(
        hosted_definition.options().web_search().allowed_domains(),
        &["docs.rs", "www.rust-lang.org"]
    );
    assert_eq!(
        hosted_definition
            .options()
            .web_search()
            .location()
            .unwrap()
            .country(),
        Some("US")
    );
    assert_eq!(hosted.input_schema()["required"], json!(["query"]));
}

#[test]
fn hosted_web_search_options_fail_closed() {
    assert_eq!(
        WebSearchOptions::new()
            .with_allowed_domains(["https://example.com"])
            .unwrap_err(),
        ModelRequestError::InvalidWebSearchDomain
    );
    assert_eq!(
        WebSearchOptions::new()
            .with_allowed_domains(["example.com"])
            .unwrap()
            .with_blocked_domains(["blocked.example"])
            .unwrap_err(),
        ModelRequestError::ConflictingWebSearchDomainFilters
    );
    assert_eq!(
        WebSearchLocation::new().with_country("USA").unwrap_err(),
        ModelRequestError::InvalidWebSearchLocation
    );
    assert_eq!(
        WebSearchLocation::new()
            .with_timezone("not a timezone")
            .unwrap_err(),
        ModelRequestError::InvalidWebSearchLocation
    );
}

#[test]
fn duplicate_names_include_hosted_and_function_definitions() {
    let request = ModelRequest::new(
        ModelId::from_str("test/model").unwrap(),
        vec![user_message(vec![ContentBlock::text("Search").unwrap()])],
    )
    .unwrap();

    assert_eq!(
        request
            .with_tools(
                vec![
                    tool("web_search"),
                    hosted_web_search(WebSearchOptions::new())
                ],
                false,
            )
            .unwrap_err(),
        ModelRequestError::DuplicateToolName
    );
}

#[test]
fn request_bounds_messages_prompt_tools_and_schema() {
    let model_id = ModelId::from_str("test/model").unwrap();
    assert_eq!(
        ModelRequest::new(model_id.clone(), vec![]).unwrap_err(),
        ModelRequestError::EmptyMessages
    );

    let request = ModelRequest::new(
        model_id,
        vec![user_message(vec![ContentBlock::text("Hello").unwrap()])],
    )
    .unwrap();
    assert_eq!(
        request.clone().with_system_prompt("").unwrap_err(),
        ModelRequestError::InvalidSystemPrompt
    );
    assert_eq!(
        request
            .clone()
            .with_system_prompt("bad\0prompt")
            .unwrap_err(),
        ModelRequestError::InvalidSystemPrompt
    );
    assert_eq!(
        ModelToolDefinition::new("read_file", "", json!({"type":"object"})).unwrap_err(),
        ModelRequestError::InvalidToolDescription
    );

    let mut nested = json!({"type":"object"});
    for _ in 0..40 {
        nested = json!({"type":"object", "properties":{"next":nested}});
    }
    assert_eq!(
        ModelToolDefinition::new("read_file", "description", nested).unwrap_err(),
        ModelRequestError::ToolSchemaOutOfBounds
    );
}

#[test]
fn request_capabilities_fail_closed_before_transport() {
    let text_request = ModelRequest::new(
        ModelId::from_str("test/model").unwrap(),
        vec![user_message(vec![ContentBlock::text("Hello").unwrap()])],
    )
    .unwrap();
    let wrong_model = model(ModelCapabilities::text());
    let other_model = ModelSpec::new(
        ModelId::from_str("other/model").unwrap(),
        ProviderId::from_str("test-provider").unwrap(),
        ModelDisplayName::from_str("Other").unwrap(),
        tokens(32_000),
        tokens(8_000),
        ModelCapabilities::text(),
    )
    .unwrap();

    assert_eq!(
        text_request.validate_for(&other_model).unwrap_err(),
        ModelRequestError::ModelMismatch
    );
    assert_eq!(
        text_request
            .clone()
            .with_max_output_tokens(tokens(8_001))
            .validate_for(&wrong_model)
            .unwrap_err(),
        ModelRequestError::OutputLimitUnsupported
    );
    assert_eq!(
        text_request
            .clone()
            .with_reasoning(ReasoningOptions::new(ReasoningEffort::High))
            .validate_for(&wrong_model)
            .unwrap_err(),
        ModelRequestError::ReasoningUnsupported
    );
    assert_eq!(
        text_request
            .clone()
            .with_tools(vec![tool("read_file")], false)
            .unwrap()
            .validate_for(&wrong_model)
            .unwrap_err(),
        ModelRequestError::ToolsUnsupported
    );
    assert_eq!(
        text_request
            .with_tools(vec![tool("read_file")], true)
            .unwrap()
            .validate_for(&model(ModelCapabilities::text().with_tools(false)))
            .unwrap_err(),
        ModelRequestError::ParallelToolsUnsupported
    );
}

#[test]
fn hosted_tools_require_model_level_capability() {
    let request = ModelRequest::new(
        ModelId::from_str("test/model").unwrap(),
        vec![user_message(vec![ContentBlock::text("Search").unwrap()])],
    )
    .unwrap()
    .with_tools(vec![hosted_web_search(WebSearchOptions::new())], false)
    .unwrap();

    assert_eq!(
        request
            .validate_for(&model(ModelCapabilities::text().with_tools(true)))
            .unwrap_err(),
        ModelRequestError::HostedToolUnsupported
    );
    request
        .validate_for(&model(
            ModelCapabilities::text().with_hosted_tool(HostedToolKind::WebSearch),
        ))
        .unwrap();
}

#[test]
fn reasoning_budget_and_request_collection_limits_fail_closed() {
    let base = ModelRequest::new(
        ModelId::from_str("test/model").unwrap(),
        vec![user_message(vec![ContentBlock::text("Hello").unwrap()])],
    )
    .unwrap();
    assert_eq!(
        base.clone()
            .with_reasoning(
                ReasoningOptions::new(ReasoningEffort::Medium).with_budget(tokens(4_001)),
            )
            .with_max_output_tokens(tokens(4_000))
            .validate_for(&model(ModelCapabilities::text().with_reasoning()))
            .unwrap_err(),
        ModelRequestError::ReasoningBudgetUnsupported
    );
    assert_eq!(
        base.clone()
            .with_reasoning(ReasoningOptions::new(ReasoningEffort::Medium).with_budget(tokens(0)),)
            .validate_for(&model(ModelCapabilities::text().with_reasoning()))
            .unwrap_err(),
        ModelRequestError::ReasoningBudgetUnsupported
    );

    let messages = vec![user_message(vec![ContentBlock::text("Hello").unwrap()]); 4097];
    assert_eq!(
        ModelRequest::new(ModelId::from_str("test/model").unwrap(), messages).unwrap_err(),
        ModelRequestError::TooManyMessages
    );

    let tools = (0..257)
        .map(|index| tool(&format!("tool_{index}")))
        .collect();
    assert_eq!(
        base.with_tools(tools, false).unwrap_err(),
        ModelRequestError::TooManyTools
    );
}

#[test]
fn tool_schema_encoded_size_is_bounded() {
    let schema = json!({
        "type": "object",
        "description": "x".repeat(256 * 1024),
    });
    assert_eq!(
        ModelToolDefinition::new("read_file", "description", schema).unwrap_err(),
        ModelRequestError::ToolSchemaOutOfBounds
    );
}

#[test]
fn image_requests_require_image_capability() {
    let request = ModelRequest::new(
        ModelId::from_str("test/model").unwrap(),
        vec![user_message(vec![
            ContentBlock::text("Inspect").unwrap(),
            ContentBlock::image_reference("image/png", "artifact://image-1").unwrap(),
        ])],
    )
    .unwrap();

    assert_eq!(
        request
            .validate_for(&model(ModelCapabilities::text()))
            .unwrap_err(),
        ModelRequestError::ImageInputUnsupported
    );
    request
        .validate_for(&model(ModelCapabilities::text().with_image_input()))
        .unwrap();
}

#[test]
fn request_rejects_an_effort_not_resolved_for_the_model() {
    let request = ModelRequest::new(
        ModelId::from_str("test/model").unwrap(),
        vec![user_message(vec![ContentBlock::text("Think").unwrap()])],
    )
    .unwrap()
    .with_reasoning(ReasoningOptions::new(ReasoningEffort::High));
    let profile = ReasoningProfile::new(
        ReasoningEffort::Medium,
        [ReasoningEffort::Low, ReasoningEffort::Medium],
    )
    .unwrap();
    let model = model(ModelCapabilities::text()).with_reasoning_profile(profile);

    assert_eq!(
        request.validate_for(&model).unwrap_err(),
        ModelRequestError::ReasoningEffortUnsupported
    );
}
