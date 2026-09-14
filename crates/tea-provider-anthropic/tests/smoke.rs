//! Opt-in live smoke coverage for native Anthropic Messages structured output.
//!
//! Run explicitly with process credentials:
//! `cargo test -p tea-provider-anthropic --features live --test integration smoke::structured_output -- --ignored --nocapture`

#![cfg(feature = "live")]

use std::str::FromStr;

use futures_util::StreamExt;
use serde_json::{Value, json};
use tea_control::CancellationScope;
use tea_model::{
    ModelEvent, ModelProvider, ModelRequest, ModelStreamValidator, ModelToolDefinition,
};
use tea_protocol::{
    CanonicalMessage, ContentBlock, FinalOutputFormat, MessageId, ProtocolTimestamp, StopReason,
};
use tea_provider_anthropic::{AnthropicProviderBuilder, CredentialResolver, EnvCredentialResolver};

fn configured_provider() -> Option<tea_provider_anthropic::AnthropicProvider> {
    let api_key = std::env::var("TEA_ANTHROPIC_API_KEY").ok();
    let model = std::env::var("TEA_ANTHROPIC_MODEL").ok();
    let schema_enabled = std::env::var("TEA_ANTHROPIC_FINAL_JSON_SCHEMA")
        .ok()
        .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"));
    if api_key.is_none_or(|value| value.is_empty())
        || model.is_none_or(|value| value.is_empty())
        || !schema_enabled
    {
        return None;
    }

    let config = EnvCredentialResolver::new()
        .resolve()
        .expect("configured Anthropic live-smoke environment must be valid");
    Some(
        AnthropicProviderBuilder::new()
            .with_config(std::sync::Arc::new(config))
            .build()
            .expect("configured Anthropic live-smoke provider must build"),
    )
}

fn output_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "answer": {"type": "string"}
        },
        "required": ["answer"],
        "additionalProperties": false
    })
}

fn user_message() -> CanonicalMessage {
    CanonicalMessage::user(
        MessageId::from_str("0195a0b1-5e52-74b2-8c25-0aa7aa000032").unwrap(),
        vec![ContentBlock::text("Return a short greeting in the answer field.").unwrap()],
        ProtocolTimestamp::from_str("2026-07-23T09:30:12.125Z").unwrap(),
    )
    .unwrap()
}

async fn run_structured_output_smoke(with_function_tool: bool) {
    let Some(provider) = configured_provider() else {
        eprintln!(
            "skipping Anthropic structured-output live smoke: set non-empty \
             TEA_ANTHROPIC_API_KEY and TEA_ANTHROPIC_MODEL plus \
             TEA_ANTHROPIC_FINAL_JSON_SCHEMA=true"
        );
        return;
    };
    let schema = output_schema();
    let mut request = ModelRequest::new(
        provider.models()[0].model_id().clone(),
        vec![user_message()],
    )
    .unwrap()
    .with_final_output_format(FinalOutputFormat::JsonSchema {
        schema: schema.clone(),
    });
    if with_function_tool {
        request = request
            .with_tools(
                vec![
                    ModelToolDefinition::new(
                        "unused_lookup",
                        "An unused lookup tool. Do not call it for this request.",
                        json!({
                            "type": "object",
                            "properties": {"query": {"type": "string"}},
                            "required": ["query"],
                            "additionalProperties": false
                        }),
                    )
                    .unwrap(),
                ],
                false,
            )
            .unwrap();
    }
    let mut stream = provider.stream(request, CancellationScope::new());
    let mut stream_validator = ModelStreamValidator::new();
    let mut text = String::new();
    let mut stop_reason = None;

    while let Some(event) = stream.next().await {
        stream_validator
            .observe(&event)
            .expect("Anthropic live stream must follow normalized event grammar");
        match event {
            ModelEvent::TextDelta(delta) => text.push_str(delta.as_str()),
            ModelEvent::Completed(completion) => {
                stop_reason = Some(completion.stop_reason().clone());
            }
            ModelEvent::Failed(failure) => panic!("Anthropic live stream failed: {failure:?}"),
            _ => {}
        }
    }

    stream_validator
        .finish()
        .expect("Anthropic live stream must contain exactly one terminal event");
    assert_eq!(stop_reason, Some(StopReason::Completed));
    let value: Value = serde_json::from_str(&text)
        .expect("Anthropic structured-output text must decode as one JSON value");
    let validator = jsonschema::draft202012::options()
        .build(&schema)
        .expect("live-smoke schema must compile as Draft 2020-12");
    assert!(
        validator.is_valid(&value),
        "Anthropic output did not satisfy the requested schema: {value}"
    );
}

#[tokio::test]
#[ignore = "requires native Anthropic Messages credentials, schema support, and a live network"]
async fn structured_output_native_messages_json_schema() {
    run_structured_output_smoke(false).await;
}

#[tokio::test]
#[ignore = "requires native Anthropic Messages schema-plus-tools support and a live network"]
async fn structured_output_native_messages_json_schema_with_function_tool() {
    run_structured_output_smoke(true).await;
}
