use std::collections::BTreeMap;
use std::str::FromStr;

use serde_json::json;
use tea_model::{
    HostedToolOptions, ModelRequest, ModelToolDefinition, WebSearchLocation, WebSearchOptions,
};
use tea_protocol::{
    CanonicalMessage, ContentBlock, ExternalSource, HostedToolActivity, HostedToolOutcome,
    MessageId, ModelId, ProtocolTimestamp, ProviderContinuation, SourceCitation, StopReason,
    ToolCallId,
};
use tea_provider_anthropic::{
    credential::{CredentialResolver, MapCredentialResolver},
    request::{build_messages_body, messages_url, request_headers},
};

fn now() -> ProtocolTimestamp {
    ProtocolTimestamp::from_str("2026-07-30T09:30:12.125Z").unwrap()
}

fn config() -> tea_provider_anthropic::AnthropicConfig {
    MapCredentialResolver::new(
        [
            ("TEA_ANTHROPIC_API_KEY", "sk-ant-test"),
            ("TEA_ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            ("TEA_ANTHROPIC_BASE_URL", "https://api.example.test"),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_owned(), value.to_owned()))
        .collect::<BTreeMap<_, _>>(),
    )
    .resolve()
    .unwrap()
}

fn configured_web_search() -> tea_provider_anthropic::AnthropicConfig {
    MapCredentialResolver::new(
        [
            ("TEA_ANTHROPIC_API_KEY", "sk-ant-test"),
            ("TEA_ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            ("TEA_ANTHROPIC_BASE_URL", "https://api.example.test"),
            ("TEA_ANTHROPIC_WEB_SEARCH_TOOL_TYPE", "web_search_20260101"),
            ("TEA_ANTHROPIC_WEB_SEARCH_MAX_USES", "7"),
        ]
        .into_iter()
        .map(|(key, value)| (key.to_owned(), value.to_owned()))
        .collect::<BTreeMap<_, _>>(),
    )
    .resolve()
    .unwrap()
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
fn maps_messages_tools_and_results_to_anthropic_shape() {
    let call_id = "0195a0b1-5e60-7000-8000-0aa7aa000001";
    let provider_call_id = "toolu_provider_123";
    let assistant = CanonicalMessage::assistant(
        MessageId::from_str("0195a0b1-5e53-74b2-8c25-0aa7aa000026").unwrap(),
        vec![
            ContentBlock::tool_call_with_provider_id(
                ToolCallId::from_str(call_id).unwrap(),
                provider_call_id,
                "read_file",
                json!({"path":"notes.txt"}),
            )
            .unwrap(),
        ],
        StopReason::ToolUse,
        now(),
    )
    .unwrap();
    let result = CanonicalMessage::tool_result_success(
        MessageId::from_str("0195a0b1-5e54-74b2-8c25-0aa7aa000027").unwrap(),
        ToolCallId::from_str(call_id).unwrap(),
        "read_file",
        vec![ContentBlock::text("contents").unwrap()],
        now(),
    )
    .unwrap();
    let request = ModelRequest::new(
        ModelId::from_str("claude-sonnet-4-20250514").unwrap(),
        vec![user("hello"), assistant, result],
    )
    .unwrap()
    .with_system_prompt("be concise")
    .unwrap()
    .with_tools(
        vec![
            ModelToolDefinition::new(
                "read_file",
                "Reads one file.",
                json!({"type":"object","properties":{"path":{"type":"string"}}}),
            )
            .unwrap(),
        ],
        true,
    )
    .unwrap();

    let body = build_messages_body(&request, &config()).unwrap();
    assert_eq!(body["model"], "claude-sonnet-4-20250514");
    assert_eq!(body["stream"], true);
    assert_eq!(body["max_tokens"], 4096);
    assert_eq!(body["system"], "be concise");
    assert_eq!(body["messages"][0]["role"], "user");
    assert_eq!(body["messages"][0]["content"][0]["text"], "hello");
    assert_eq!(body["messages"][1]["content"][0]["type"], "tool_use");
    assert_eq!(body["messages"][1]["content"][0]["id"], provider_call_id);
    assert_eq!(
        body["messages"][1]["content"][0]["input"]["path"],
        "notes.txt"
    );
    assert_eq!(body["messages"][2]["content"][0]["type"], "tool_result");
    assert_eq!(
        body["messages"][2]["content"][0]["tool_use_id"],
        provider_call_id
    );
    assert_eq!(body["tools"][0]["input_schema"]["type"], "object");
}

#[test]
fn maps_contextual_user_text_as_model_visible_anthropic_content() {
    let request = ModelRequest::new(
        ModelId::from_str("claude-sonnet-4-20250514").unwrap(),
        vec![user_with_context(
            "/skill:review src",
            "private review instructions",
        )],
    )
    .unwrap();

    let body = build_messages_body(&request, &config()).unwrap();
    assert_eq!(
        body["messages"][0]["content"][0]["text"],
        "/skill:review src"
    );
    assert_eq!(
        body["messages"][0]["content"][1]["text"],
        "private review instructions"
    );
}

#[test]
fn coalesces_parallel_tool_results_into_one_user_turn() {
    let first_call_id = "0195a0b1-5e60-7000-8000-0aa7aa000001";
    let second_call_id = "0195a0b1-5e60-7000-8000-0aa7aa000002";
    let assistant = CanonicalMessage::assistant(
        MessageId::from_str("0195a0b1-5e53-74b2-8c25-0aa7aa000026").unwrap(),
        vec![
            ContentBlock::tool_call_with_provider_id(
                ToolCallId::from_str(first_call_id).unwrap(),
                "toolu_provider_123",
                "read_file",
                json!({"path":"Cargo.toml"}),
            )
            .unwrap(),
            ContentBlock::tool_call_with_provider_id(
                ToolCallId::from_str(second_call_id).unwrap(),
                "toolu_provider_456",
                "read_file",
                json!({"path":"src/lib.rs"}),
            )
            .unwrap(),
        ],
        StopReason::ToolUse,
        now(),
    )
    .unwrap();
    let first_result = CanonicalMessage::tool_result_success(
        MessageId::from_str("0195a0b1-5e54-74b2-8c25-0aa7aa000027").unwrap(),
        ToolCallId::from_str(first_call_id).unwrap(),
        "read_file",
        vec![ContentBlock::text("manifest").unwrap()],
        now(),
    )
    .unwrap();
    let second_result = CanonicalMessage::tool_result_success(
        MessageId::from_str("0195a0b1-5e54-74b2-8c25-0aa7aa000028").unwrap(),
        ToolCallId::from_str(second_call_id).unwrap(),
        "read_file",
        vec![ContentBlock::text("source").unwrap()],
        now(),
    )
    .unwrap();
    let request = ModelRequest::new(
        ModelId::from_str("claude-sonnet-4-20250514").unwrap(),
        vec![
            user("inspect both files"),
            assistant,
            first_result,
            second_result,
        ],
    )
    .unwrap();

    let body = build_messages_body(&request, &config()).unwrap();

    assert_eq!(body["messages"].as_array().unwrap().len(), 3);
    assert_eq!(body["messages"][2]["role"], "user");
    assert_eq!(body["messages"][2]["content"].as_array().unwrap().len(), 2);
    assert_eq!(
        body["messages"][2]["content"][0]["tool_use_id"],
        "toolu_provider_123"
    );
    assert_eq!(
        body["messages"][2]["content"][1]["tool_use_id"],
        "toolu_provider_456"
    );
}

#[test]
fn endpoint_and_headers_use_messages_contract() {
    let config = config();
    assert_eq!(
        messages_url(&config),
        "https://api.example.test/v1/messages"
    );
    let headers = request_headers(&config);
    assert!(
        headers
            .iter()
            .any(|(key, value)| key == "x-api-key" && value == "sk-ant-test")
    );
    assert!(
        headers
            .iter()
            .any(|(key, value)| key == "anthropic-version" && value == "2023-06-01")
    );
}

#[test]
fn maps_versioned_web_search_controls_domains_and_location() {
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
    let request = ModelRequest::new(
        ModelId::from_str("claude-sonnet-4-20250514").unwrap(),
        vec![user("search")],
    )
    .unwrap()
    .with_tools(vec![web_search(options)], false)
    .unwrap();

    let body = build_messages_body(&request, &configured_web_search()).unwrap();

    assert_eq!(body["tools"][0]["type"], "web_search_20260101");
    assert_eq!(body["tools"][0]["name"], "web_search");
    assert_eq!(body["tools"][0]["max_uses"], 7);
    assert_eq!(
        body["tools"][0]["allowed_domains"],
        json!(["docs.rs", "example.com"])
    );
    assert!(body["tools"][0].get("blocked_domains").is_none());
    assert_eq!(body["tools"][0]["user_location"]["type"], "approximate");
    assert_eq!(body["tools"][0]["user_location"]["country"], "US");
    assert_eq!(
        body["tools"][0]["user_location"]["timezone"],
        "America/Chicago"
    );
}

#[test]
fn maps_blocked_domains_without_an_allowed_domain_field() {
    let options = WebSearchOptions::new()
        .with_blocked_domains(["spam.example", "tracking.example"])
        .unwrap();
    let request = ModelRequest::new(
        ModelId::from_str("claude-sonnet-4-20250514").unwrap(),
        vec![user("search")],
    )
    .unwrap()
    .with_tools(vec![web_search(options)], false)
    .unwrap();

    let body = build_messages_body(&request, &config()).unwrap();

    assert_eq!(
        body["tools"][0]["blocked_domains"],
        json!(["spam.example", "tracking.example"])
    );
    assert!(body["tools"][0].get("allowed_domains").is_none());
}

#[test]
fn hosted_and_citation_continuations_round_trip_exactly() {
    let server_block = json!({
        "type":"server_tool_use",
        "id":"srvtoolu_replay",
        "name":"web_search",
        "input":{"query":"tea-rs"}
    });
    let result_block = json!({
        "type":"web_search_tool_result",
        "tool_use_id":"srvtoolu_replay",
        "content":[{
            "type":"web_search_result",
            "url":"https://example.com/tea",
            "title":"Tea",
            "encrypted_content":"enc_result"
        }]
    });
    let hosted_continuation = ProviderContinuation::new(
        "anthropic",
        "anthropic.messages.web_search.v1",
        json!({"content_blocks":[server_block.clone(), result_block.clone()]}),
    )
    .unwrap();
    let source = ExternalSource::new("https://example.com/tea")
        .unwrap()
        .with_title("Tea")
        .unwrap();
    let activity = HostedToolActivity::new(
        ToolCallId::from_str("0195a0b1-5e60-7000-8000-0aa7aa000099").unwrap(),
        "srvtoolu_replay",
        "web_search",
        json!({"query":"tea-rs"}),
        HostedToolOutcome::Success,
        vec![source.clone()],
        Some(hosted_continuation),
    )
    .unwrap();
    let raw_citation = json!({
        "type":"web_search_result_location",
        "url":"https://example.com/tea",
        "title":"Tea",
        "encrypted_index":"enc_index",
        "cited_text":"Tea"
    });
    let citation_continuation = ProviderContinuation::new(
        "anthropic",
        "anthropic.messages.web_search.v1",
        json!({"citation":raw_citation.clone()}),
    )
    .unwrap();
    let citation = SourceCitation::new(source)
        .with_cited_text("Tea")
        .unwrap()
        .with_continuation(citation_continuation);
    let assistant = CanonicalMessage::assistant(
        MessageId::from_str("0195a0b1-5e53-74b2-8c25-0aa7aa000099").unwrap(),
        vec![
            ContentBlock::hosted_tool(activity),
            ContentBlock::text("Tea is documented.").unwrap(),
            ContentBlock::citation(citation),
        ],
        StopReason::PauseTurn,
        now(),
    )
    .unwrap();
    let request = ModelRequest::new(
        ModelId::from_str("claude-sonnet-4-20250514").unwrap(),
        vec![assistant, user("continue")],
    )
    .unwrap();

    let body = build_messages_body(&request, &config()).unwrap();
    let replay = body["messages"][0]["content"].as_array().unwrap();

    assert_eq!(replay[0], server_block);
    assert_eq!(replay[1], result_block);
    assert_eq!(replay[2]["type"], "text");
    assert_eq!(replay[2]["text"], "Tea is documented.");
    assert_eq!(replay[2]["citations"][0], raw_citation);
    assert_eq!(body["messages"][1]["role"], "user");
}
