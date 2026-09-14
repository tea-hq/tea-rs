//! Live smoke tests for the `OpenAI` provider.
//!
//! These tests are `#[ignore]`d and never run in CI. Run them locally:
//!   cargo test -p tea-provider-openai --features live --test integration `smoke::` -- --ignored
//! They load `.env` (committed template at `.env.example`) via the dependency-
//! free loader and skip with a message when required vars are unset.

#![cfg(feature = "live")]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::str::FromStr;

use futures_util::StreamExt;
use serde_json::{Value, json};
use tea_control::CancellationScope;
use tea_model::{
    HostedToolKind, HostedToolOptions, ModelCapabilities, ModelDisplayName, ModelEvent,
    ModelProvider, ModelSpec, ModelStreamValidator, ModelToolDefinition, WebSearchOptions,
};
use tea_protocol::{
    CanonicalMessage, ContentBlock, FinalOutputFormat, MessageId, ModelId, ProtocolMetadata,
    ProtocolTimestamp, StopReason, TokenCount,
};
use tea_provider_openai::{
    OpenAiApiMode, OpenAiCompatibilityProfile, OpenAiProvider, OpenAiProviderBuilder,
    credential::{CredentialResolver, MapCredentialResolver},
    env_file::load_env_file,
};

const OPENAI_ENV_KEYS: &[&str] = &[
    "TEA_OPENAI_BASE_URL",
    "TEA_OPENAI_API_KEY",
    "TEA_OPENAI_MODEL",
    "TEA_OPENAI_API_KEY_HEADER",
    "TEA_OPENAI_API_KEY_PREFIX",
    "TEA_OPENAI_API_MODE",
    "TEA_OPENAI_COMPATIBILITY_PROFILE",
    "TEA_OPENAI_ORG_ID",
    "TEA_OPENAI_PROJECT_ID",
    "TEA_OPENAI_REASONING_EFFORT",
    "TEA_OPENAI_VISION",
    "TEA_OPENAI_FINAL_JSON_OBJECT",
    "TEA_OPENAI_FINAL_JSON_SCHEMA",
    "TEA_OPENAI_FINAL_JSON_SCHEMA_WITH_TOOLS",
    "TEA_OPENAI_HOSTED_WEB_SEARCH",
    "TEA_OPENAI_REQUEST_TIMEOUT_MS",
];

fn load_env_map() -> BTreeMap<String, String> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.env");
    let mut values = if path.exists() {
        load_env_file(&path).unwrap_or_default()
    } else {
        BTreeMap::new()
    };
    for key in OPENAI_ENV_KEYS {
        if let Ok(value) = std::env::var(key) {
            values.insert((*key).to_owned(), value);
        }
    }
    values
}

#[allow(dead_code)]
fn smoke_provider(api_mode: Option<OpenAiApiMode>) -> Option<OpenAiProviderBuilder> {
    smoke_provider_with_capabilities(api_mode, ModelCapabilities::text().with_tools(true))
}

fn smoke_provider_with_capabilities(
    api_mode: Option<OpenAiApiMode>,
    capabilities: ModelCapabilities,
) -> Option<OpenAiProviderBuilder> {
    let map = load_env_map();
    let api_key = map.get("TEA_OPENAI_API_KEY").filter(|v| !v.is_empty())?;
    let model_text = map
        .get("TEA_OPENAI_MODEL")
        .filter(|v| !v.is_empty())?
        .clone();
    let _ = api_key;
    let resolver = MapCredentialResolver::new(map);
    let mut config = resolver.resolve().ok()?;
    if let Some(api_mode) = api_mode {
        config = config.with_api_mode(api_mode);
    }
    let provider_id = config.provider_id().clone();
    let model_id = ModelId::from_str(&model_text).ok()?;
    let spec = ModelSpec::new(
        model_id,
        provider_id,
        ModelDisplayName::from_str("Smoke Model").unwrap(),
        TokenCount::new(128_000).unwrap(),
        TokenCount::new(4_000).unwrap(),
        capabilities,
    )
    .unwrap();
    Some(
        OpenAiProviderBuilder::new()
            .with_config(std::sync::Arc::new(config))
            .with_catalog(vec![spec]),
    )
}

fn web_search_tool() -> ModelToolDefinition {
    ModelToolDefinition::hosted(
        "Searches the public web and returns cited sources.",
        serde_json::json!({
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }),
        HostedToolOptions::WebSearch(WebSearchOptions::new()),
    )
    .unwrap()
}

fn user_message(text: &str) -> CanonicalMessage {
    CanonicalMessage::user(
        MessageId::from_str("0195a0b1-5e52-74b2-8c25-0aa7aa000031").unwrap(),
        vec![ContentBlock::text(text).unwrap()],
        ProtocolTimestamp::from_str("2026-07-23T09:30:12.125Z").unwrap(),
    )
    .unwrap()
}

#[derive(Debug, Clone, Copy)]
enum StructuredSmokeFormat {
    JsonObject,
    JsonSchema,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StructuredSmokeTool {
    None,
    Function,
    HostedWebSearch,
}

#[derive(Debug, Clone, Copy)]
struct StructuredSmokeTuple {
    profile: OpenAiCompatibilityProfile,
    api_mode: OpenAiApiMode,
    format: StructuredSmokeFormat,
    tool: StructuredSmokeTool,
}

fn structured_output_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "answer": {"type": "string"}
        },
        "required": ["answer"],
        "additionalProperties": false
    })
}

fn assert_structured_output(
    tuple: StructuredSmokeTuple,
    schema: &Value,
    text: &str,
    hosted_successes: usize,
    source_count: usize,
) {
    let value: Value = serde_json::from_str(text)
        .expect("structured-output text must decode as exactly one JSON value");
    assert!(
        value.is_object(),
        "structured output must be a JSON object: {value}"
    );
    if matches!(tuple.format, StructuredSmokeFormat::JsonSchema) {
        let schema_validator = jsonschema::draft202012::options()
            .build(schema)
            .expect("live-smoke schema must compile as Draft 2020-12");
        assert!(
            schema_validator.is_valid(&value),
            "structured output did not satisfy the requested schema: {value}"
        );
    }
    if tuple.tool == StructuredSmokeTool::HostedWebSearch {
        assert!(hosted_successes > 0, "hosted search must report success");
        assert!(source_count > 0, "hosted search must return sources");
    }
}

fn configured_structured_provider(tuple: StructuredSmokeTuple) -> Option<OpenAiProvider> {
    let values = load_env_map();
    for key in [
        "TEA_OPENAI_BASE_URL",
        "TEA_OPENAI_API_KEY",
        "TEA_OPENAI_MODEL",
        "TEA_OPENAI_API_MODE",
        "TEA_OPENAI_COMPATIBILITY_PROFILE",
    ] {
        if values.get(key).is_none_or(String::is_empty) {
            return None;
        }
    }

    let config = MapCredentialResolver::new(values)
        .resolve()
        .expect("configured OpenAI-compatible live-smoke environment must be valid");
    if config.compatibility_profile() != Some(tuple.profile) || config.api_mode() != tuple.api_mode
    {
        return None;
    }
    let capability_enabled = match tuple.format {
        StructuredSmokeFormat::JsonObject => config.final_json_object(),
        StructuredSmokeFormat::JsonSchema => config.final_json_schema(),
    };
    if !capability_enabled {
        return None;
    }

    if tuple.tool != StructuredSmokeTool::None && !config.final_json_schema_with_tools() {
        return None;
    }
    if tuple.tool == StructuredSmokeTool::HostedWebSearch && !config.hosted_web_search() {
        return None;
    }
    let provider = OpenAiProviderBuilder::new()
        .with_config(std::sync::Arc::new(config))
        .build()
        .expect("configured OpenAI-compatible structured-output provider must build");
    if tuple.tool != StructuredSmokeTool::None
        && !provider.models()[0]
            .capabilities()
            .supports_final_json_schema_with_tools()
    {
        return None;
    }
    if tuple.tool == StructuredSmokeTool::HostedWebSearch
        && !provider.models()[0]
            .capabilities()
            .supports_hosted_tool(HostedToolKind::WebSearch)
    {
        return None;
    }
    Some(provider)
}

async fn run_structured_output_smoke(tuple: StructuredSmokeTuple) {
    let Some(provider) = configured_structured_provider(tuple) else {
        eprintln!(
            "skipping OpenAI-compatible structured-output live smoke for {tuple:?}: set explicit \
             TEA_OPENAI_BASE_URL, TEA_OPENAI_API_KEY, TEA_OPENAI_MODEL, \
             TEA_OPENAI_API_MODE, TEA_OPENAI_COMPATIBILITY_PROFILE, and the matching \
             TEA_OPENAI_FINAL_JSON_* capability flag; hosted-search tuples also require \
             TEA_OPENAI_HOSTED_WEB_SEARCH=true"
        );
        return;
    };
    let schema = structured_output_schema();
    let format = match tuple.format {
        StructuredSmokeFormat::JsonObject => FinalOutputFormat::JsonObject,
        StructuredSmokeFormat::JsonSchema => FinalOutputFormat::JsonSchema {
            schema: schema.clone(),
        },
    };
    let prompt = match tuple.tool {
        StructuredSmokeTool::HostedWebSearch => {
            "Use web_search to find the current Rust stable release. Return exactly one JSON object with the answer in the answer field. Do not use Markdown."
        }
        StructuredSmokeTool::None | StructuredSmokeTool::Function => {
            "Return exactly one JSON object with a short greeting in the answer field. Do not use Markdown."
        }
    };
    let mut request = tea_model::ModelRequest::new(
        provider.models()[0].model_id().clone(),
        vec![user_message(prompt)],
    )
    .unwrap()
    .with_final_output_format(format);
    request = match tuple.tool {
        StructuredSmokeTool::None => request,
        StructuredSmokeTool::Function => request
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
            .unwrap(),
        StructuredSmokeTool::HostedWebSearch => {
            request.with_tools(vec![web_search_tool()], false).unwrap()
        }
    };

    let mut stream = provider.stream(request, CancellationScope::new());
    let mut validator = ModelStreamValidator::new();
    let mut text = String::new();
    let mut stop_reason = None;
    let mut hosted_successes = 0;
    let mut source_count = 0;
    while let Some(event) = stream.next().await {
        validator
            .observe(&event)
            .expect("live stream must follow normalized event grammar");
        match event {
            ModelEvent::TextDelta(delta) => text.push_str(delta.as_str()),
            ModelEvent::Completed(completion) => {
                stop_reason = Some(completion.stop_reason().clone());
            }
            ModelEvent::HostedToolCompleted(activity) => {
                hosted_successes += usize::from(matches!(
                    activity.outcome(),
                    tea_protocol::HostedToolOutcome::Success
                ));
                source_count += activity.sources().len();
            }
            ModelEvent::Failed(failure) => {
                panic!("OpenAI-compatible structured-output live stream failed: {failure:?}")
            }
            _ => {}
        }
    }
    validator
        .finish()
        .expect("live stream must contain exactly one terminal event");
    assert_eq!(stop_reason, Some(StopReason::Completed));

    assert_structured_output(tuple, &schema, &text, hosted_successes, source_count);
}

macro_rules! structured_output_smoke {
    ($name:ident, $profile:expr, $api_mode:expr, $format:expr) => {
        #[tokio::test]
        #[ignore = "requires an explicitly configured compatible model, endpoint, and live network"]
        async fn $name() {
            run_structured_output_smoke(StructuredSmokeTuple {
                profile: $profile,
                api_mode: $api_mode,
                format: $format,
                tool: StructuredSmokeTool::None,
            })
            .await;
        }
    };
}

macro_rules! structured_output_with_function_tool_smoke {
    ($name:ident, $profile:expr, $api_mode:expr) => {
        #[tokio::test]
        #[ignore = "requires an explicitly configured model supporting JSON Schema with function tools and a live network"]
        async fn $name() {
            run_structured_output_smoke(StructuredSmokeTuple {
                profile: $profile,
                api_mode: $api_mode,
                format: StructuredSmokeFormat::JsonSchema,
                tool: StructuredSmokeTool::Function,
            })
            .await;
        }
    };
}

structured_output_smoke!(
    structured_output_open_ai_chat_json_object,
    OpenAiCompatibilityProfile::OpenAi,
    OpenAiApiMode::ChatCompletions,
    StructuredSmokeFormat::JsonObject
);
structured_output_smoke!(
    structured_output_open_ai_chat_json_schema,
    OpenAiCompatibilityProfile::OpenAi,
    OpenAiApiMode::ChatCompletions,
    StructuredSmokeFormat::JsonSchema
);
structured_output_smoke!(
    structured_output_open_ai_responses_json_object,
    OpenAiCompatibilityProfile::OpenAi,
    OpenAiApiMode::Responses,
    StructuredSmokeFormat::JsonObject
);
structured_output_smoke!(
    structured_output_open_ai_responses_json_schema,
    OpenAiCompatibilityProfile::OpenAi,
    OpenAiApiMode::Responses,
    StructuredSmokeFormat::JsonSchema
);
structured_output_smoke!(
    structured_output_azure_open_ai_chat_json_object,
    OpenAiCompatibilityProfile::AzureOpenAi,
    OpenAiApiMode::ChatCompletions,
    StructuredSmokeFormat::JsonObject
);
structured_output_smoke!(
    structured_output_azure_open_ai_chat_json_schema,
    OpenAiCompatibilityProfile::AzureOpenAi,
    OpenAiApiMode::ChatCompletions,
    StructuredSmokeFormat::JsonSchema
);
structured_output_smoke!(
    structured_output_azure_open_ai_responses_json_object,
    OpenAiCompatibilityProfile::AzureOpenAi,
    OpenAiApiMode::Responses,
    StructuredSmokeFormat::JsonObject
);
structured_output_smoke!(
    structured_output_azure_open_ai_responses_json_schema,
    OpenAiCompatibilityProfile::AzureOpenAi,
    OpenAiApiMode::Responses,
    StructuredSmokeFormat::JsonSchema
);
structured_output_smoke!(
    structured_output_xai_chat_json_object,
    OpenAiCompatibilityProfile::Xai,
    OpenAiApiMode::ChatCompletions,
    StructuredSmokeFormat::JsonObject
);
structured_output_smoke!(
    structured_output_xai_chat_json_schema,
    OpenAiCompatibilityProfile::Xai,
    OpenAiApiMode::ChatCompletions,
    StructuredSmokeFormat::JsonSchema
);
structured_output_smoke!(
    structured_output_xai_responses_json_object,
    OpenAiCompatibilityProfile::Xai,
    OpenAiApiMode::Responses,
    StructuredSmokeFormat::JsonObject
);
structured_output_smoke!(
    structured_output_xai_responses_json_schema,
    OpenAiCompatibilityProfile::Xai,
    OpenAiApiMode::Responses,
    StructuredSmokeFormat::JsonSchema
);
structured_output_smoke!(
    structured_output_deep_seek_chat_json_object,
    OpenAiCompatibilityProfile::DeepSeek,
    OpenAiApiMode::ChatCompletions,
    StructuredSmokeFormat::JsonObject
);
structured_output_smoke!(
    structured_output_deep_seek_responses_json_object,
    OpenAiCompatibilityProfile::DeepSeek,
    OpenAiApiMode::Responses,
    StructuredSmokeFormat::JsonObject
);
structured_output_smoke!(
    structured_output_deep_seek_responses_json_schema,
    OpenAiCompatibilityProfile::DeepSeek,
    OpenAiApiMode::Responses,
    StructuredSmokeFormat::JsonSchema
);
structured_output_smoke!(
    structured_output_gemini_open_ai_chat_json_object,
    OpenAiCompatibilityProfile::GeminiOpenAi,
    OpenAiApiMode::ChatCompletions,
    StructuredSmokeFormat::JsonObject
);
structured_output_smoke!(
    structured_output_gemini_open_ai_chat_json_schema,
    OpenAiCompatibilityProfile::GeminiOpenAi,
    OpenAiApiMode::ChatCompletions,
    StructuredSmokeFormat::JsonSchema
);
structured_output_smoke!(
    structured_output_ollama_local_chat_json_object,
    OpenAiCompatibilityProfile::OllamaLocal,
    OpenAiApiMode::ChatCompletions,
    StructuredSmokeFormat::JsonObject
);
structured_output_smoke!(
    structured_output_ollama_local_chat_json_schema,
    OpenAiCompatibilityProfile::OllamaLocal,
    OpenAiApiMode::ChatCompletions,
    StructuredSmokeFormat::JsonSchema
);
structured_output_smoke!(
    structured_output_open_router_chat_json_object,
    OpenAiCompatibilityProfile::OpenRouter,
    OpenAiApiMode::ChatCompletions,
    StructuredSmokeFormat::JsonObject
);
structured_output_smoke!(
    structured_output_open_router_chat_json_schema,
    OpenAiCompatibilityProfile::OpenRouter,
    OpenAiApiMode::ChatCompletions,
    StructuredSmokeFormat::JsonSchema
);
structured_output_smoke!(
    structured_output_groq_chat_json_object,
    OpenAiCompatibilityProfile::Groq,
    OpenAiApiMode::ChatCompletions,
    StructuredSmokeFormat::JsonObject
);
structured_output_smoke!(
    structured_output_groq_chat_json_schema,
    OpenAiCompatibilityProfile::Groq,
    OpenAiApiMode::ChatCompletions,
    StructuredSmokeFormat::JsonSchema
);
structured_output_smoke!(
    structured_output_mistral_chat_json_object,
    OpenAiCompatibilityProfile::Mistral,
    OpenAiApiMode::ChatCompletions,
    StructuredSmokeFormat::JsonObject
);
structured_output_smoke!(
    structured_output_mistral_chat_json_schema,
    OpenAiCompatibilityProfile::Mistral,
    OpenAiApiMode::ChatCompletions,
    StructuredSmokeFormat::JsonSchema
);
structured_output_smoke!(
    structured_output_together_chat_json_object,
    OpenAiCompatibilityProfile::Together,
    OpenAiApiMode::ChatCompletions,
    StructuredSmokeFormat::JsonObject
);
structured_output_smoke!(
    structured_output_together_chat_json_schema,
    OpenAiCompatibilityProfile::Together,
    OpenAiApiMode::ChatCompletions,
    StructuredSmokeFormat::JsonSchema
);
structured_output_smoke!(
    structured_output_vllm_chat_json_object,
    OpenAiCompatibilityProfile::Vllm,
    OpenAiApiMode::ChatCompletions,
    StructuredSmokeFormat::JsonObject
);
structured_output_smoke!(
    structured_output_vllm_chat_json_schema,
    OpenAiCompatibilityProfile::Vllm,
    OpenAiApiMode::ChatCompletions,
    StructuredSmokeFormat::JsonSchema
);
structured_output_smoke!(
    structured_output_vllm_responses_json_object,
    OpenAiCompatibilityProfile::Vllm,
    OpenAiApiMode::Responses,
    StructuredSmokeFormat::JsonObject
);
structured_output_smoke!(
    structured_output_vllm_responses_json_schema,
    OpenAiCompatibilityProfile::Vllm,
    OpenAiApiMode::Responses,
    StructuredSmokeFormat::JsonSchema
);

structured_output_with_function_tool_smoke!(
    structured_output_open_ai_chat_schema_with_function_tool,
    OpenAiCompatibilityProfile::OpenAi,
    OpenAiApiMode::ChatCompletions
);
structured_output_with_function_tool_smoke!(
    structured_output_open_ai_responses_schema_with_function_tool,
    OpenAiCompatibilityProfile::OpenAi,
    OpenAiApiMode::Responses
);
structured_output_with_function_tool_smoke!(
    structured_output_deep_seek_responses_schema_with_function_tool,
    OpenAiCompatibilityProfile::DeepSeek,
    OpenAiApiMode::Responses
);
structured_output_with_function_tool_smoke!(
    structured_output_gemini_open_ai_chat_schema_with_function_tool,
    OpenAiCompatibilityProfile::GeminiOpenAi,
    OpenAiApiMode::ChatCompletions
);
structured_output_with_function_tool_smoke!(
    structured_output_ollama_local_chat_schema_with_function_tool,
    OpenAiCompatibilityProfile::OllamaLocal,
    OpenAiApiMode::ChatCompletions
);
structured_output_with_function_tool_smoke!(
    structured_output_open_router_chat_schema_with_function_tool,
    OpenAiCompatibilityProfile::OpenRouter,
    OpenAiApiMode::ChatCompletions
);
structured_output_with_function_tool_smoke!(
    structured_output_mistral_chat_schema_with_function_tool,
    OpenAiCompatibilityProfile::Mistral,
    OpenAiApiMode::ChatCompletions
);
structured_output_with_function_tool_smoke!(
    structured_output_together_chat_schema_with_function_tool,
    OpenAiCompatibilityProfile::Together,
    OpenAiApiMode::ChatCompletions
);
structured_output_with_function_tool_smoke!(
    structured_output_vllm_chat_schema_with_function_tool,
    OpenAiCompatibilityProfile::Vllm,
    OpenAiApiMode::ChatCompletions
);

#[tokio::test]
#[ignore = "requires Azure OpenAI structured output, tools, and a live network"]
async fn structured_output_azure_open_ai_schema_with_parallel_tools_disabled() {
    for api_mode in [OpenAiApiMode::ChatCompletions, OpenAiApiMode::Responses] {
        run_structured_output_smoke(StructuredSmokeTuple {
            profile: OpenAiCompatibilityProfile::AzureOpenAi,
            api_mode,
            format: StructuredSmokeFormat::JsonSchema,
            tool: StructuredSmokeTool::Function,
        })
        .await;
    }
}

#[tokio::test]
#[ignore = "requires an xAI model supporting structured output with tools and a live network"]
async fn structured_output_xai_schema_with_function_tool() {
    for api_mode in [OpenAiApiMode::ChatCompletions, OpenAiApiMode::Responses] {
        run_structured_output_smoke(StructuredSmokeTuple {
            profile: OpenAiCompatibilityProfile::Xai,
            api_mode,
            format: StructuredSmokeFormat::JsonSchema,
            tool: StructuredSmokeTool::Function,
        })
        .await;
    }
}

#[tokio::test]
#[ignore = "requires OpenAI Responses structured output with hosted web search and a live network"]
async fn structured_output_open_ai_responses_schema_with_hosted_web_search() {
    run_structured_output_smoke(StructuredSmokeTuple {
        profile: OpenAiCompatibilityProfile::OpenAi,
        api_mode: OpenAiApiMode::Responses,
        format: StructuredSmokeFormat::JsonSchema,
        tool: StructuredSmokeTool::HostedWebSearch,
    })
    .await;
}

#[tokio::test]
#[ignore = "requires xAI Responses structured output with hosted web search and a live network"]
async fn structured_output_xai_responses_schema_with_hosted_web_search() {
    run_structured_output_smoke(StructuredSmokeTuple {
        profile: OpenAiCompatibilityProfile::Xai,
        api_mode: OpenAiApiMode::Responses,
        format: StructuredSmokeFormat::JsonSchema,
        tool: StructuredSmokeTool::HostedWebSearch,
    })
    .await;
}

#[tokio::test]
#[ignore = "requires TEA_OPENAI_* env vars and a live network"]
async fn live_text_stream_completes() {
    let Some(builder) = smoke_provider(None) else {
        eprintln!("skipping live smoke: TEA_OPENAI_* not configured");
        return;
    };
    let provider = builder.build().unwrap();
    let model_id = provider.models().first().unwrap().model_id().clone();
    let request = tea_model::ModelRequest::new(
        model_id,
        vec![user_message("Say hello in one short sentence.")],
    )
    .unwrap();
    let mut stream = provider.stream(request, CancellationScope::new());
    let mut text = String::new();
    let mut completed = false;
    while let Some(event) = stream.next().await {
        match event {
            ModelEvent::TextDelta(delta) => text.push_str(delta.as_str()),
            ModelEvent::Completed(completion) => {
                assert!(completion.usage().is_some(), "usage should be reported");
                completed = true;
                break;
            }
            ModelEvent::Failed(failure) => panic!("live stream failed: {failure:?}"),
            _ => {}
        }
    }
    assert!(completed, "stream must reach a terminal completion");
    assert!(!text.is_empty(), "stream must produce visible text");
}

#[tokio::test]
#[ignore = "requires TEA_OPENAI_* env vars, Responses API support, and a live network"]
async fn live_responses_text_stream_completes() {
    let Some(builder) = smoke_provider(Some(OpenAiApiMode::Responses)) else {
        eprintln!("skipping live Responses smoke: TEA_OPENAI_* not configured");
        return;
    };
    let provider = builder.build().unwrap();
    let model_id = provider.models().first().unwrap().model_id().clone();
    let request = tea_model::ModelRequest::new(
        model_id,
        vec![user_message("Say hello in one short sentence.")],
    )
    .unwrap();
    let mut stream = provider.stream(request, CancellationScope::new());
    let mut text = String::new();
    let mut completed = false;
    while let Some(event) = stream.next().await {
        match event {
            ModelEvent::TextDelta(delta) => text.push_str(delta.as_str()),
            ModelEvent::Completed(completion) => {
                assert!(completion.usage().is_some(), "usage should be reported");
                completed = true;
                break;
            }
            ModelEvent::Failed(failure) => panic!("live Responses stream failed: {failure:?}"),
            _ => {}
        }
    }
    assert!(
        completed,
        "Responses stream must reach a terminal completion"
    );
    assert!(
        !text.is_empty(),
        "Responses stream must produce visible text"
    );
}

#[tokio::test]
#[ignore = "requires TEA_OPENAI_* env vars, hosted web search, and a live network"]
async fn live_responses_hosted_web_search_completes_with_sources() {
    let capabilities = ModelCapabilities::text()
        .with_tools(true)
        .with_hosted_tool(HostedToolKind::WebSearch);
    let Some(builder) =
        smoke_provider_with_capabilities(Some(OpenAiApiMode::Responses), capabilities)
    else {
        eprintln!("skipping live Responses hosted-search smoke: TEA_OPENAI_* not configured");
        return;
    };
    let provider = builder.build().unwrap();
    let model_id = provider.models().first().unwrap().model_id().clone();
    let request = tea_model::ModelRequest::new(
        model_id,
        vec![user_message(
            "Use web_search to find the latest official Rust blog post. Reply with its title, date, and source URL.",
        )],
    )
    .unwrap()
    .with_tools(vec![web_search_tool()], false)
    .unwrap();
    let mut stream = provider.stream(request, CancellationScope::new());
    let mut text = String::new();
    let mut replayed_full_text = false;
    let mut hosted_successes = 0;
    let mut source_count = 0;
    let mut completed = false;
    while let Some(event) = stream.next().await {
        match event {
            ModelEvent::TextDelta(delta) => {
                replayed_full_text |= !text.is_empty() && delta.as_str() == text;
                text.push_str(delta.as_str());
            }
            ModelEvent::HostedToolCompleted(activity) => {
                hosted_successes += usize::from(matches!(
                    activity.outcome(),
                    tea_protocol::HostedToolOutcome::Success
                ));
                source_count += activity.sources().len();
            }
            ModelEvent::Completed(_) => {
                completed = true;
                break;
            }
            ModelEvent::Failed(failure) => {
                panic!("live Responses hosted-search stream failed: {failure:?}")
            }
            _ => {}
        }
    }
    assert!(completed, "hosted-search stream must complete");
    assert!(hosted_successes > 0, "hosted search must report success");
    assert!(source_count > 0, "hosted search must return sources");
    assert!(!text.is_empty(), "hosted search must produce visible text");
    assert!(
        !replayed_full_text,
        "terminal snapshot must not replay the complete streamed answer"
    );
}

#[tokio::test]
#[ignore = "requires TEA_OPENAI_* env vars and a live network"]
async fn live_cancellation_closes_stream() {
    let Some(builder) = smoke_provider(None) else {
        eprintln!("skipping live smoke: TEA_OPENAI_* not configured");
        return;
    };
    let provider = builder.build().unwrap();
    let model_id = provider.models().first().unwrap().model_id().clone();
    let request = tea_model::ModelRequest::new(
        model_id,
        vec![user_message("Write a long essay about the ocean.")],
    )
    .unwrap();
    let cancellation = CancellationScope::new();
    let mut stream = provider.stream(request, cancellation.clone());
    cancellation.cancel();
    while let Some(event) = stream.next().await {
        if matches!(event, ModelEvent::Failed(_)) {
            break;
        }
    }
    let _ = ProtocolMetadata::default();
}
