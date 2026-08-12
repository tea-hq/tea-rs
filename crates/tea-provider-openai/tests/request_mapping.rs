//! Request mapping: `ModelRequest` -> `OpenAI` Chat Completions body.

use std::collections::BTreeMap;
use std::str::FromStr;

use serde_json::{Value, json};
use tea_model::{
    HostedToolOptions, ModelRequest, ModelToolDefinition, ReasoningEffort, ReasoningOptions,
    WebSearchLocation, WebSearchOptions,
};
use tea_protocol::{
    CanonicalMessage, ContentBlock, ExternalSource, MessageId, ModelId, ProtocolMetadata,
    ProtocolTimestamp, SourceCitation, ToolCallId,
};
use tea_provider_openai::{
    OpenAiReasoningEffortMap,
    credential::{CredentialResolver, MapCredentialResolver, OpenAiApiMode, OpenAiConfig},
    request::{
        build_chat_completions_body, build_chat_completions_body_with_reasoning_map,
        chat_completions_url, request_headers,
    },
    responses::{build_responses_body, build_responses_body_with_reasoning_map, responses_url},
};

fn now() -> ProtocolTimestamp {
    ProtocolTimestamp::from_str("2026-07-23T09:30:12.125Z").unwrap()
}
fn config() -> OpenAiConfig {
    let map = env_map(&[
        ("TEA_OPENAI_BASE_URL", "https://api.example.test/v1"),
        ("TEA_OPENAI_API_KEY", "sk-test-key"),
        ("TEA_OPENAI_MODEL", "gpt-4o-mini"),
    ]);
    MapCredentialResolver::new(map).resolve().unwrap()
}
fn responses_config() -> OpenAiConfig {
    let map = env_map(&[
        ("TEA_OPENAI_BASE_URL", "https://api.example.test/v1"),
        ("TEA_OPENAI_API_KEY", "sk-test-key"),
        ("TEA_OPENAI_MODEL", "gpt-4o-mini"),
        ("TEA_OPENAI_API_MODE", "responses"),
    ]);
    MapCredentialResolver::new(map).resolve().unwrap()
}
fn legacy_reasoning_config() -> OpenAiConfig {
    let map = env_map(&[
        ("TEA_OPENAI_BASE_URL", "https://api.example.test/v1"),
        ("TEA_OPENAI_API_KEY", "sk-test-key"),
        ("TEA_OPENAI_MODEL", "gpt-4o-mini"),
        ("TEA_OPENAI_REASONING_EFFORT", "high"),
    ]);
    MapCredentialResolver::new(map).resolve().unwrap()
}
fn env_map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect()
}

fn user(text: &str) -> CanonicalMessage {
    CanonicalMessage::user(
        MessageId::from_str("0195a0b1-5e52-74b2-8c25-0aa7aa000025").unwrap(),
        vec![ContentBlock::text(text).unwrap()],
        now(),
    )
    .unwrap()
}
fn user_with_context(text: &str, context: &str) -> CanonicalMessage {
    CanonicalMessage::user(
        MessageId::from_str("0195a0b1-5e52-74b2-8c25-0aa7aa000025").unwrap(),
        vec![
            ContentBlock::text(text).unwrap(),
            ContentBlock::contextual_text(context).unwrap(),
        ],
        now(),
    )
    .unwrap()
}
fn assistant_text(text: &str) -> CanonicalMessage {
    CanonicalMessage::assistant(
        MessageId::from_str("0195a0b1-5e53-74b2-8c25-0aa7aa000026").unwrap(),
        vec![ContentBlock::text(text).unwrap()],
        tea_protocol::StopReason::Completed,
        now(),
    )
    .unwrap()
}
fn assistant_tool_call(call_id: &str, name: &str, args: Value) -> CanonicalMessage {
    CanonicalMessage::assistant(
        MessageId::from_str("0195a0b1-5e54-74b2-8c25-0aa7aa000027").unwrap(),
        vec![
            ContentBlock::tool_call(
                ToolCallId::from_str(call_id).unwrap(),
                name.to_owned(),
                args,
            )
            .unwrap(),
        ],
        tea_protocol::StopReason::ToolUse,
        now(),
    )
    .unwrap()
}
fn assistant_provider_tool_call(
    call_id: &str,
    provider_call_id: &str,
    name: &str,
    args: Value,
) -> CanonicalMessage {
    CanonicalMessage::assistant(
        MessageId::from_str("0195a0b1-5e54-74b2-8c25-0aa7aa000027").unwrap(),
        vec![
            ContentBlock::tool_call_with_provider_id(
                ToolCallId::from_str(call_id).unwrap(),
                provider_call_id,
                name,
                args,
            )
            .unwrap(),
        ],
        tea_protocol::StopReason::ToolUse,
        now(),
    )
    .unwrap()
}
fn tool_result(call_id: &str, name: &str, text: &str) -> CanonicalMessage {
    CanonicalMessage::tool_result_success(
        MessageId::from_str("0195a0b1-5e55-74b2-8c25-0aa7aa000028").unwrap(),
        ToolCallId::from_str(call_id).unwrap(),
        name,
        vec![ContentBlock::text(text).unwrap()],
        now(),
    )
    .unwrap()
}

fn base_request() -> ModelRequest {
    ModelRequest::new(
        ModelId::from_str("gpt-4o-mini").unwrap(),
        vec![user("hello"), assistant_text("hi"), user("how are you")],
    )
    .unwrap()
}

fn web_search(options: WebSearchOptions) -> ModelToolDefinition {
    ModelToolDefinition::hosted(
        "Searches the public web.",
        json!({
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }),
        HostedToolOptions::WebSearch(options),
    )
    .unwrap()
}

#[test]
fn maps_messages_roles_and_model() {
    let body = build_chat_completions_body(&base_request(), &config()).unwrap();
    assert_eq!(body["model"], "gpt-4o-mini");
    assert_eq!(body["stream"], true);
    assert_eq!(body["stream_options"]["include_usage"], true);
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[0]["content"], "hello");
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[1]["content"], "hi");
    assert_eq!(messages[2]["role"], "user");
}

#[test]
fn maps_system_prompt_as_first_message() {
    let request = base_request()
        .with_system_prompt("you are helpful")
        .unwrap();
    let body = build_chat_completions_body(&request, &config()).unwrap();
    let messages = body["messages"].as_array().unwrap();
    assert_eq!(messages[0]["role"], "system");
    assert_eq!(messages[0]["content"], "you are helpful");
}

#[test]
fn maps_contextual_user_text_to_both_openai_request_shapes() {
    let request = ModelRequest::new(
        ModelId::from_str("gpt-4o-mini").unwrap(),
        vec![user_with_context(
            "/skill:review src",
            "private review instructions",
        )],
    )
    .unwrap();

    let chat = build_chat_completions_body(&request, &config()).unwrap();
    assert_eq!(
        chat["messages"][0]["content"],
        "/skill:review srcprivate review instructions"
    );

    let responses = build_responses_body(&request, &responses_config()).unwrap();
    assert_eq!(
        responses["input"][0]["content"][0]["text"],
        "/skill:review src"
    );
    assert_eq!(
        responses["input"][0]["content"][1]["text"],
        "private review instructions"
    );
}

#[test]
fn maps_assistant_tool_calls_with_string_arguments() {
    let request = ModelRequest::new(
        ModelId::from_str("gpt-4o-mini").unwrap(),
        vec![assistant_tool_call(
            "0195a0b1-5e60-7000-8000-0aa7aa000001",
            "read_file",
            json!({"path":"/notes.txt"}),
        )],
    )
    .unwrap();
    let body = build_chat_completions_body(&request, &config()).unwrap();
    let tool_calls = &body["messages"][0]["tool_calls"];
    assert_eq!(tool_calls[0]["type"], "function");
    assert_eq!(tool_calls[0]["function"]["name"], "read_file");
    // Arguments must be serialized to a JSON string, not an object.
    assert_eq!(
        tool_calls[0]["function"]["arguments"].as_str().unwrap(),
        r#"{"path":"/notes.txt"}"#
    );
    assert!(tool_calls[0]["index"].is_number());
}

#[test]
fn chat_completions_restores_provider_tool_call_ids() {
    let call_id = "0195a0b1-5e60-7000-8000-0aa7aa000001";
    let provider_call_id = "call_provider_123";
    let request = ModelRequest::new(
        ModelId::from_str("gpt-4o-mini").unwrap(),
        vec![
            assistant_provider_tool_call(
                call_id,
                provider_call_id,
                "read_file",
                json!({"path":"/notes.txt"}),
            ),
            tool_result(call_id, "read_file", "contents"),
        ],
    )
    .unwrap();
    let body = build_chat_completions_body(&request, &config()).unwrap();
    assert_eq!(body["messages"][0]["tool_calls"][0]["id"], provider_call_id);
    assert_eq!(body["messages"][1]["tool_call_id"], provider_call_id);
}

#[test]
fn maps_tools_and_parallel_flag() {
    let request = base_request()
        .with_tools(
            vec![tea_model::ModelToolDefinition::new(
                "read_file",
                "Reads one file.",
                json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}),
            )
            .unwrap()],
            true,
        )
        .unwrap();
    let body = build_chat_completions_body(&request, &config()).unwrap();
    assert_eq!(body["tools"][0]["type"], "function");
    assert_eq!(body["tools"][0]["function"]["name"], "read_file");
    assert_eq!(body["parallel_tool_calls"], true);
}

#[test]
fn maps_reasoning_effort_and_completion_tokens() {
    let request = base_request()
        .with_reasoning(ReasoningOptions::new(ReasoningEffort::Medium))
        .with_max_output_tokens(tea_protocol::TokenCount::new(1024).unwrap());
    let body = build_chat_completions_body(&request, &config()).unwrap();
    assert_eq!(body["reasoning_effort"], "medium");
    // Reasoning models use max_completion_tokens, not max_tokens.
    assert_eq!(body["max_completion_tokens"], 1024);
    assert!(body.get("max_tokens").is_none());
}

#[test]
fn canonical_reasoning_levels_map_identically_and_off_is_omitted() {
    for effort in ReasoningEffort::ALL {
        let request = base_request().with_reasoning(ReasoningOptions::new(effort));
        let chat = build_chat_completions_body(&request, &config()).unwrap();
        let responses = build_responses_body(&request, &responses_config()).unwrap();
        if effort == ReasoningEffort::Off {
            assert!(chat.get("reasoning_effort").is_none());
            assert!(responses.get("reasoning").is_none());
        } else {
            assert_eq!(chat["reasoning_effort"], effort.as_str());
            assert_eq!(responses["reasoning"]["effort"], effort.as_str());
        }
    }
}

#[test]
fn model_level_reasoning_map_can_translate_provider_wire_values() {
    let map = OpenAiReasoningEffortMap::new([
        (ReasoningEffort::Minimal, "low".to_owned()),
        (ReasoningEffort::Maximum, "high".to_owned()),
    ])
    .unwrap();
    let minimal = base_request().with_reasoning(ReasoningOptions::new(ReasoningEffort::Minimal));
    let maximum = base_request().with_reasoning(ReasoningOptions::new(ReasoningEffort::Maximum));

    let chat =
        build_chat_completions_body_with_reasoning_map(&minimal, &config(), Some(&map)).unwrap();
    let responses =
        build_responses_body_with_reasoning_map(&maximum, &responses_config(), Some(&map)).unwrap();
    assert_eq!(chat["reasoning_effort"], "low");
    assert_eq!(responses["reasoning"]["effort"], "high");
}

#[test]
fn legacy_connection_effort_cannot_override_request_intent() {
    let request = base_request().with_reasoning(ReasoningOptions::new(ReasoningEffort::Low));
    let body = build_chat_completions_body(&request, &legacy_reasoning_config()).unwrap();
    assert_eq!(body["reasoning_effort"], "low");

    let off = base_request().with_reasoning(ReasoningOptions::new(ReasoningEffort::Off));
    let body = build_chat_completions_body(&off, &legacy_reasoning_config()).unwrap();
    assert!(body.get("reasoning_effort").is_none());
}

#[test]
fn missing_model_level_reasoning_mapping_fails_closed() {
    let map =
        OpenAiReasoningEffortMap::new([(ReasoningEffort::Minimal, "low".to_owned())]).unwrap();
    let request = base_request().with_reasoning(ReasoningOptions::new(ReasoningEffort::High));
    assert!(
        build_chat_completions_body_with_reasoning_map(&request, &config(), Some(&map)).is_err()
    );
}

#[test]
fn non_reasoning_uses_max_tokens() {
    let request =
        base_request().with_max_output_tokens(tea_protocol::TokenCount::new(512).unwrap());
    let body = build_chat_completions_body(&request, &config()).unwrap();
    assert_eq!(body["max_tokens"], 512);
    assert!(body.get("max_completion_tokens").is_none());
}

#[test]
fn endpoint_url_and_headers() {
    let config = config();
    assert_eq!(
        chat_completions_url(&config),
        "https://api.example.test/v1/chat/completions",
    );
    let headers = request_headers(&config);
    assert!(
        headers
            .iter()
            .any(|(k, v)| k == "Authorization" && v == "Bearer sk-test-key")
    );
    // reqwest's .json() sets Content-Type; the adapter must not duplicate it.
    assert!(
        !headers
            .iter()
            .any(|(k, _)| k.eq_ignore_ascii_case("content-type"))
    );
    let _ = ProtocolMetadata::default();
}

#[test]
fn maps_responses_messages_tools_and_results() {
    let call_id = "0195a0b1-5e60-7000-8000-0aa7aa000001";
    let provider_call_id = "call_provider_123";
    let request = ModelRequest::new(
        ModelId::from_str("gpt-4o-mini").unwrap(),
        vec![
            user("hello"),
            assistant_text("I will read it."),
            assistant_provider_tool_call(
                call_id,
                provider_call_id,
                "read_file",
                json!({"path":"/notes.txt"}),
            ),
            tool_result(call_id, "read_file", "contents"),
        ],
    )
    .unwrap()
    .with_system_prompt("you are helpful")
    .unwrap()
    .with_tools(
        vec![tea_model::ModelToolDefinition::new(
            "read_file",
            "Reads one file.",
            json!({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}),
        )
        .unwrap()],
        true,
    )
    .unwrap();

    let config = responses_config();
    assert_eq!(config.api_mode(), OpenAiApiMode::Responses);
    let body = build_responses_body(&request, &config).unwrap();
    assert_eq!(body["model"], "gpt-4o-mini");
    assert_eq!(body["instructions"], "you are helpful");
    assert_eq!(body["store"], false);
    assert_eq!(body["stream"], true);
    assert_eq!(body["parallel_tool_calls"], true);
    assert_eq!(body["tool_choice"], "auto");
    assert_eq!(body["tools"][0]["type"], "function");
    assert_eq!(body["tools"][0]["name"], "read_file");
    assert_eq!(body["tools"][0]["strict"], false);
    assert!(body["tools"][0].get("function").is_none());

    let input = body["input"].as_array().unwrap();
    assert_eq!(input[0]["type"], "message");
    assert_eq!(input[0]["role"], "user");
    assert_eq!(input[0]["content"][0]["type"], "input_text");
    assert_eq!(input[1]["role"], "assistant");
    assert_eq!(input[1]["content"][0]["type"], "output_text");
    assert_eq!(input[2]["type"], "function_call");
    assert_eq!(input[2]["call_id"], provider_call_id);
    assert_eq!(input[2]["arguments"], r#"{"path":"/notes.txt"}"#);
    assert_eq!(input[3]["type"], "function_call_output");
    assert_eq!(input[3]["call_id"], provider_call_id);
    assert_eq!(input[3]["output"], "contents");
}

#[test]
fn maps_responses_reasoning_and_output_limit() {
    let request = base_request()
        .with_reasoning(ReasoningOptions::new(ReasoningEffort::Medium))
        .with_max_output_tokens(tea_protocol::TokenCount::new(1024).unwrap());
    let body = build_responses_body(&request, &responses_config()).unwrap();
    assert_eq!(body["reasoning"]["effort"], "medium");
    assert_eq!(body["reasoning"]["summary"], "auto");
    assert_eq!(body["include"][0], "reasoning.encrypted_content");
    assert_eq!(body["max_output_tokens"], 1024);
}

#[test]
fn responses_endpoint_uses_parallel_path() {
    assert_eq!(
        responses_url(&responses_config()),
        "https://api.example.test/v1/responses"
    );
}

#[test]
fn maps_responses_hosted_web_search_options_and_sources_include() {
    let location = WebSearchLocation::new()
        .with_country("US")
        .unwrap()
        .with_city("Minneapolis")
        .unwrap()
        .with_region("Minnesota")
        .unwrap()
        .with_timezone("America/Chicago")
        .unwrap();
    let options = WebSearchOptions::new()
        .with_allowed_domains(["docs.rs", "example.com"])
        .unwrap()
        .with_location(location);
    let request = base_request()
        .with_tools(vec![web_search(options)], false)
        .unwrap();

    let body = build_responses_body(&request, &responses_config()).unwrap();

    assert_eq!(body["tools"].as_array().unwrap().len(), 1);
    assert_eq!(body["tools"][0]["type"], "web_search");
    assert_eq!(
        body["tools"][0]["filters"]["allowed_domains"],
        json!(["docs.rs", "example.com"])
    );
    assert!(body["tools"][0]["filters"].get("blocked_domains").is_none());
    assert_eq!(body["tools"][0]["user_location"]["type"], "approximate");
    assert_eq!(body["tools"][0]["user_location"]["country"], "US");
    assert_eq!(
        body["tools"][0]["user_location"]["timezone"],
        "America/Chicago"
    );
    assert!(
        body["include"]
            .as_array()
            .unwrap()
            .contains(&json!("web_search_call.action.sources"))
    );
    assert!(
        !serde_json::to_string(&body)
            .unwrap()
            .contains("web_search_preview")
    );
}

#[test]
fn maps_responses_hosted_web_search_blocked_domains() {
    let options = WebSearchOptions::new()
        .with_blocked_domains(["example.net", "spam.example"])
        .unwrap();
    let request = base_request()
        .with_tools(vec![web_search(options)], false)
        .unwrap();

    let body = build_responses_body(&request, &responses_config()).unwrap();

    assert_eq!(
        body["tools"][0]["filters"]["blocked_domains"],
        json!(["example.net", "spam.example"])
    );
    assert!(body["tools"][0]["filters"].get("allowed_domains").is_none());
}

#[test]
fn chat_completions_rejects_hosted_tool_definitions() {
    let request = base_request()
        .with_tools(vec![web_search(WebSearchOptions::new())], false)
        .unwrap();

    let error = build_chat_completions_body(&request, &config()).unwrap_err();

    assert_eq!(
        error.code(),
        tea_provider_openai::OpenAiErrorCode::InvalidRequest
    );
    assert!(error.message().contains("Responses"));
}

#[test]
fn responses_maps_canonical_utf8_citation_ranges_to_character_offsets() {
    let citation = SourceCitation::new(ExternalSource::new("https://example.com/unicode").unwrap())
        .with_range(6, 9)
        .unwrap();
    let assistant = CanonicalMessage::assistant(
        MessageId::from_str("0195a0b1-5e53-74b2-8c25-0aa7aa000088").unwrap(),
        vec![
            ContentBlock::text("茶和茶").unwrap(),
            ContentBlock::citation(citation),
        ],
        tea_protocol::StopReason::Completed,
        now(),
    )
    .unwrap();
    let request = ModelRequest::new(
        ModelId::from_str("gpt-4.1").unwrap(),
        vec![assistant, user("continue")],
    )
    .unwrap();

    let body = build_responses_body(&request, &responses_config()).unwrap();
    let annotation = &body["input"][0]["content"][0]["annotations"][0];

    assert_eq!(annotation["start_index"], 2);
    assert_eq!(annotation["end_index"], 3);
}
