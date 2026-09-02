//! `SSE` parsing and `ModelEvent` mapping over recorded fixtures.

use std::path::Path;
use std::str::FromStr;

use tea_model::{ModelEvent, ModelRequest, ModelStreamValidator};
use tea_protocol::{
    CanonicalMessage, ContentBlock, ExternalSource, HostedToolActivity, HostedToolOutcome,
    MessageId, ModelId, ProtocolTimestamp, ProviderContinuation, SourceCitation, StopReason,
    ToolCallId,
};
use tea_provider_openai::{
    credential::{CredentialResolver, MapCredentialResolver},
    responses::build_responses_body,
    responses_stream::ResponsesReducer,
    sse::SseParser,
    stream::{ChunkReducer, map_http_failure, map_stream_error},
};

#[test]
fn provider_failures_include_bounded_http_diagnostics() {
    let failure = map_http_failure(
        403,
        r#"{"error":{"message":"blocked by gateway WAF","api_key":"sk-secret"}}"#,
    );
    assert_eq!(failure.message(), "HTTP 403: blocked by gateway WAF");
    assert!(!failure.message().contains("sk-secret"));
    assert!(failure.is_safe_diagnostic());

    let stream_failure = map_stream_error(&serde_json::json!({
        "type": "server_error",
        "message": "upstream overloaded",
    }));
    assert_eq!(stream_failure.message(), "upstream overloaded");
}

fn fixture(name: &str) -> String {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(manifest.join("tests/fixtures").join(name)).unwrap()
}

fn events_from_fixture(name: &str) -> Vec<ModelEvent> {
    let bytes = fixture(name);
    let mut parser = SseParser::new();
    let mut reducer = ChunkReducer::new();
    let mut events = Vec::new();
    for chunk in bytes.as_bytes().chunks(13) {
        for sse in parser.feed(chunk) {
            match sse {
                tea_provider_openai::sse::SseEvent::Data(payload) => {
                    let value: serde_json::Value = serde_json::from_str(&payload).unwrap();
                    events.extend(reducer.map_chunk(&value).unwrap());
                }
                tea_provider_openai::sse::SseEvent::Done => {
                    if let Some(terminal) = reducer.finish().unwrap() {
                        events.push(terminal);
                    }
                }
            }
        }
    }
    for sse in parser.finish() {
        if let tea_provider_openai::sse::SseEvent::Data(payload) = sse {
            let value: serde_json::Value = serde_json::from_str(&payload).unwrap();
            events.extend(reducer.map_chunk(&value).unwrap());
        }
    }
    if let Some(terminal) = reducer.finish().unwrap() {
        events.push(terminal);
    }
    events
}

fn responses_events_from_fixture(name: &str) -> Vec<ModelEvent> {
    let bytes = fixture(name);
    let mut parser = SseParser::new();
    let mut reducer = ResponsesReducer::new();
    let mut events = Vec::new();
    for chunk in bytes.as_bytes().chunks(17) {
        for sse in parser.feed(chunk) {
            match sse {
                tea_provider_openai::sse::SseEvent::Data(payload) => {
                    let value: serde_json::Value = serde_json::from_str(&payload).unwrap();
                    events.extend(reducer.map_chunk(&value).unwrap());
                }
                tea_provider_openai::sse::SseEvent::Done => {
                    if let Some(terminal) = reducer.finish().unwrap() {
                        events.push(terminal);
                    }
                }
            }
        }
    }
    for sse in parser.finish() {
        if let tea_provider_openai::sse::SseEvent::Data(payload) = sse {
            let value: serde_json::Value = serde_json::from_str(&payload).unwrap();
            events.extend(reducer.map_chunk(&value).unwrap());
        }
    }
    if let Some(terminal) = reducer.finish().unwrap() {
        events.push(terminal);
    }
    events
}

fn web_search_added(output_index: u16, id: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "response.output_item.added",
        "output_index": output_index,
        "item": {"id": id, "type": "web_search_call"}
    })
}

fn web_search_item(id: &str, url: &str) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "type": "web_search_call",
        "status": "completed",
        "action": {
            "type": "search",
            "queries": ["tea"],
            "sources": [{"type": "url", "url": url}]
        }
    })
}

fn web_search_done(output_index: u16, id: &str, url: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "response.output_item.done",
        "output_index": output_index,
        "item": web_search_item(id, url)
    })
}

fn message_added(output_index: u16, id: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "response.output_item.added",
        "output_index": output_index,
        "item": {"id": id, "type": "message", "content": []}
    })
}

fn output_text_delta(output_index: u16, content_index: u16, text: &str) -> serde_json::Value {
    serde_json::json!({
        "type": "response.output_text.delta",
        "output_index": output_index,
        "content_index": content_index,
        "delta": text
    })
}

fn citation_annotation(url: &str, start: u16, end: u16) -> serde_json::Value {
    serde_json::json!({
        "type": "url_citation",
        "start_index": start,
        "end_index": end,
        "url": url,
        "title": "Source"
    })
}

fn citation_added(
    output_index: u16,
    content_index: u16,
    annotation_index: u16,
    url: &str,
    start: u16,
    end: u16,
) -> serde_json::Value {
    serde_json::json!({
        "type": "response.output_text.annotation.added",
        "output_index": output_index,
        "content_index": content_index,
        "annotation_index": annotation_index,
        "annotation": citation_annotation(url, start, end)
    })
}

fn message_item(id: &str, text: &str, annotations: &[serde_json::Value]) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "type": "message",
        "content": [{
            "type": "output_text",
            "text": text,
            "annotations": annotations
        }]
    })
}

fn reducer_with_pending_search(text: &str) -> ResponsesReducer {
    let mut reducer = ResponsesReducer::new();
    for value in [
        web_search_added(0, "ws_pending"),
        message_added(1, "msg_pending"),
        output_text_delta(1, 0, text),
    ] {
        reducer.map_chunk(&value).unwrap();
    }
    reducer
}

#[test]
fn text_fixture_maps_to_started_deltas_and_completion() {
    let events = events_from_fixture("text.sse");
    assert!(matches!(events.first(), Some(ModelEvent::Started(_))));
    let text: String = events
        .iter()
        .filter_map(|event| event.as_text_delta().map(str::to_owned))
        .collect();
    assert_eq!(text, "Hello world");
    let completion = events
        .iter()
        .find_map(|event| match event {
            ModelEvent::Completed(completion) => Some(completion),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        completion.stop_reason(),
        &tea_protocol::StopReason::Completed
    );
    assert!(completion.usage().is_some());
    assert!(events_validator_accepts(&events));
}

#[test]
fn tool_call_fixture_maps_started_delta_and_completed() {
    let events = events_from_fixture("tool_call.sse");
    assert!(
        events
            .iter()
            .any(|event| matches!(event, ModelEvent::ToolCallStarted(_)))
    );
    let deltas = events
        .iter()
        .filter(|event| matches!(event, ModelEvent::ToolArgumentsDelta(_)))
        .count();
    assert_eq!(deltas, 2, "two argument fragments");
    let completed = events
        .iter()
        .find_map(|event| match event {
            ModelEvent::ToolCallCompleted(call) => Some(call),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        completed.arguments().get("path").unwrap().as_str().unwrap(),
        "/notes.txt"
    );
    assert!(matches!(events.last(), Some(ModelEvent::Completed(_))));
    assert!(events_validator_accepts(&events));
}

#[test]
fn reasoning_fixture_emits_thinking_then_text() {
    let events = events_from_fixture("reasoning.sse");
    let thinking = events
        .iter()
        .find_map(|event| event.as_thinking_delta())
        .unwrap();
    assert_eq!(thinking, "Let me think.");
    assert!(
        events
            .iter()
            .any(|event| event.as_text_delta() == Some("Answer."))
    );
    let usage = events
        .iter()
        .find_map(|event| match event {
            ModelEvent::Completed(completion) => completion.usage(),
            _ => None,
        })
        .expect("the terminal usage chunk must be preserved");
    assert_eq!(usage.input_tokens().get(), 1);
    assert_eq!(usage.output_tokens().get(), 4);
    assert_eq!(usage.cache_read_tokens().unwrap().get(), 3);
    assert_eq!(usage.cache_write_tokens().unwrap().get(), 1);
    assert_eq!(usage.reasoning_tokens().unwrap().get(), 2);
    assert!(events_validator_accepts(&events));
}

#[test]
fn midstream_error_fixture_maps_to_failed_terminal() {
    let events = events_from_fixture("midstream_error.sse");
    let failure = events
        .iter()
        .find_map(|event| match event {
            ModelEvent::Failed(failure) => Some(failure),
            _ => None,
        })
        .unwrap();
    assert_eq!(failure.code(), tea_model::ModelFailureCode::Unavailable);
    assert_eq!(failure.retry(), tea_protocol::RetryClass::AfterBackoff);
}

#[test]
fn responses_fixture_maps_text_reasoning_tools_and_usage() {
    let events = responses_events_from_fixture("responses.sse");
    let response_info = match events.first() {
        Some(ModelEvent::Started(info)) => info,
        other => panic!("expected response start, got {other:?}"),
    };
    assert_eq!(
        response_info.response_id().unwrap().as_str(),
        "resp_tea_test"
    );
    let thinking: String = events
        .iter()
        .filter_map(|event| event.as_thinking_delta().map(str::to_owned))
        .collect();
    assert_eq!(thinking, "Need the file.");
    let text: String = events
        .iter()
        .filter_map(|event| event.as_text_delta().map(str::to_owned))
        .collect();
    assert_eq!(text, "I will read it.");
    let argument_deltas: String = events
        .iter()
        .filter_map(|event| {
            event
                .as_tool_arguments_delta()
                .map(|delta| delta.delta().to_owned())
        })
        .collect();
    assert_eq!(argument_deltas, r#"{"path":"/notes.txt"}"#);
    let completed_call = events
        .iter()
        .find_map(|event| match event {
            ModelEvent::ToolCallCompleted(call) => Some(call),
            _ => None,
        })
        .unwrap();
    assert_eq!(completed_call.tool_name(), "read_file");
    assert_eq!(completed_call.arguments()["path"], "/notes.txt");
    let completion = events
        .iter()
        .find_map(|event| match event {
            ModelEvent::Completed(completion) => Some(completion),
            _ => None,
        })
        .unwrap();
    assert_eq!(completion.stop_reason(), &tea_protocol::StopReason::ToolUse);
    let usage = completion.usage().unwrap();
    assert_eq!(usage.input_tokens().get(), 8);
    assert_eq!(usage.output_tokens().get(), 4);
    assert_eq!(usage.cache_read_tokens().unwrap().get(), 3);
    assert_eq!(usage.cache_write_tokens().unwrap().get(), 2);
    assert_eq!(usage.reasoning_tokens().unwrap().get(), 2);
    assert!(events_validator_accepts(&events));
}

#[test]
fn responses_failed_fixture_maps_to_failed_terminal() {
    let events = responses_events_from_fixture("responses_failed.sse");
    let failure = events
        .iter()
        .find_map(|event| match event {
            ModelEvent::Failed(failure) => Some(failure),
            _ => None,
        })
        .unwrap();
    assert_eq!(failure.code(), tea_model::ModelFailureCode::RateLimited);
    assert_eq!(failure.retry(), tea_protocol::RetryClass::AfterBackoff);
    assert!(events_validator_accepts(&events));
}

#[test]
fn responses_incomplete_is_a_terminal_failure() {
    let mut reducer = ResponsesReducer::new();
    let events = reducer
        .map_chunk(&serde_json::json!({
            "type": "response.incomplete",
            "response": {
                "id": "resp_incomplete",
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"}
            }
        }))
        .unwrap();
    let failure = events
        .iter()
        .find_map(|event| match event {
            ModelEvent::Failed(failure) => Some(failure),
            _ => None,
        })
        .unwrap();
    assert_eq!(failure.code(), tea_model::ModelFailureCode::ContextOverflow);
    assert!(events_validator_accepts(&events));
}

#[test]
fn responses_web_search_fixture_maps_activity_sources_citations_and_text() {
    let events = responses_events_from_fixture("responses_web_search.sse");

    let started = events
        .iter()
        .find_map(ModelEvent::as_hosted_tool_started)
        .expect("hosted search must start");
    assert_eq!(started.provider_call_id().as_str(), "ws_tea_search");
    assert_eq!(started.tool_name(), "web_search");

    let completed = events
        .iter()
        .find_map(ModelEvent::as_hosted_tool_completed)
        .expect("hosted search must complete");
    assert_eq!(completed.arguments()["type"], "search");
    assert_eq!(
        completed.arguments()["queries"],
        serde_json::json!(["tea-rs hosted tools"])
    );
    assert!(completed.arguments().get("sources").is_none());
    assert_eq!(completed.sources().len(), 2);
    assert_eq!(completed.sources()[0].url(), "https://example.com/guide");
    let continuation = completed
        .continuation()
        .expect("the raw output item is needed for stateless replay");
    assert_eq!(continuation.provider(), "openai");
    assert_eq!(continuation.format(), "openai.responses.web_search.v1");
    assert_eq!(continuation.payload()["type"], "web_search_call");
    assert_eq!(
        continuation.payload()["action"]["sources"]
            .as_array()
            .unwrap()
            .len(),
        2
    );

    let text: String = events
        .iter()
        .filter_map(|event| event.as_text_delta().map(str::to_owned))
        .collect();
    assert_eq!(text, "Tea uses sources.");
    let citation = events
        .iter()
        .find_map(ModelEvent::as_source_citation)
        .expect("URL annotation must become a normalized citation");
    assert_eq!(
        citation.provider_call_id().unwrap().as_str(),
        "ws_tea_search"
    );
    assert_eq!(
        citation.citation().source().url(),
        "https://example.com/guide"
    );
    assert_eq!(citation.citation().source().title(), Some("Tea guide"));
    assert_eq!(citation.citation().range(), Some((9, 16)));
    assert_eq!(citation.citation().cited_text(), Some("sources"));
    assert_eq!(
        citation.citation().continuation().unwrap().format(),
        "openai.responses.url_citation.v1"
    );

    assert!(matches!(events.last(), Some(ModelEvent::Completed(_))));
    assert!(events_validator_accepts(&events));
}

#[test]
fn responses_web_search_preserves_provider_specific_sources_without_exposing_them_as_urls() {
    let api_source = serde_json::json!({"type": "api", "name": "weather"});
    let item = serde_json::json!({
        "id": "ws_weather",
        "type": "web_search_call",
        "status": "completed",
        "action": {
            "type": "search",
            "query": "Hangzhou weather today",
            "sources": [api_source.clone()]
        }
    });
    let mut reducer = ResponsesReducer::new();
    let mut events = reducer
        .map_chunk(&web_search_added(0, "ws_weather"))
        .unwrap();
    events.extend(
        reducer
            .map_chunk(&serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": item.clone()
            }))
            .unwrap(),
    );
    events.extend(
        reducer
            .map_chunk(&serde_json::json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_weather",
                    "status": "completed",
                    "output": [item]
                }
            }))
            .unwrap(),
    );

    let completed = events
        .iter()
        .find_map(ModelEvent::as_hosted_tool_completed)
        .expect("hosted weather search must complete");
    assert!(completed.sources().is_empty());
    assert_eq!(
        completed.continuation().unwrap().payload()["action"]["sources"][0],
        api_source
    );
    assert!(matches!(events.last(), Some(ModelEvent::Completed(_))));
    assert!(events_validator_accepts(&events));
}

#[test]
fn responses_web_search_still_rejects_malformed_url_sources() {
    let cases = [
        (serde_json::json!([{"type": "url"}]), "source URL missing"),
        (
            serde_json::json!([{"type": "url", "url": "not a URL"}]),
            "source URL invalid",
        ),
        (
            serde_json::json!({"type": "url", "url": "https://example.com"}),
            "sources are not an array",
        ),
    ];

    for (sources, expected_message) in cases {
        let mut reducer = ResponsesReducer::new();
        reducer
            .map_chunk(&web_search_added(0, "ws_invalid_source"))
            .unwrap();
        let mut item = web_search_item("ws_invalid_source", "https://example.com");
        item["action"]["sources"] = sources;
        let error = reducer
            .map_chunk(&serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": item
            }))
            .unwrap_err();

        assert_eq!(
            error.code(),
            tea_provider_openai::OpenAiErrorCode::MalformedResponse
        );
        assert!(error.message().contains(expected_message));
    }
}

#[test]
fn responses_web_search_continuation_replays_without_provider_state() {
    let events = responses_events_from_fixture("responses_web_search.sse");
    let completed = events
        .iter()
        .find_map(ModelEvent::as_hosted_tool_completed)
        .unwrap();
    let citation = events
        .iter()
        .find_map(ModelEvent::as_source_citation)
        .unwrap();
    let activity = HostedToolActivity::new(
        ToolCallId::from_str("0195a0b1-5e60-7000-8000-0aa7aa000099").unwrap(),
        completed.provider_call_id().as_str(),
        completed.tool_name(),
        completed.arguments().clone(),
        completed.outcome().clone(),
        completed.sources().to_vec(),
        completed.continuation().cloned(),
    )
    .unwrap();
    let timestamp = ProtocolTimestamp::from_str("2026-07-23T09:30:12.125Z").unwrap();
    let prior = CanonicalMessage::assistant(
        MessageId::from_str("0195a0b1-5e53-74b2-8c25-0aa7aa000099").unwrap(),
        vec![
            ContentBlock::hosted_tool(activity),
            ContentBlock::text("Tea uses sources.").unwrap(),
            ContentBlock::citation(citation.citation().clone()),
        ],
        StopReason::Completed,
        timestamp,
    )
    .unwrap();
    let follow_up = CanonicalMessage::user(
        MessageId::from_str("0195a0b1-5e52-74b2-8c25-0aa7aa000098").unwrap(),
        vec![ContentBlock::text("Summarize that source.").unwrap()],
        timestamp,
    )
    .unwrap();
    let request = ModelRequest::new(
        ModelId::from_str("gpt-4.1").unwrap(),
        vec![prior, follow_up],
    )
    .unwrap();
    let config = MapCredentialResolver::new(std::collections::BTreeMap::from([
        ("TEA_OPENAI_API_KEY".to_owned(), "sk-test-key".to_owned()),
        ("TEA_OPENAI_MODEL".to_owned(), "gpt-4.1".to_owned()),
        ("TEA_OPENAI_API_MODE".to_owned(), "responses".to_owned()),
    ]))
    .resolve()
    .unwrap();

    let body = build_responses_body(&request, &config).unwrap();
    let input = body["input"].as_array().unwrap();

    assert_eq!(input[0]["type"], "web_search_call");
    assert_eq!(input[0]["id"], "ws_tea_search");
    assert_eq!(input[0]["action"]["sources"].as_array().unwrap().len(), 2);
    assert_eq!(input[1]["type"], "message");
    assert_eq!(input[1]["content"][0]["text"], "Tea uses sources.");
    assert_eq!(
        input[1]["content"][0]["annotations"][0]["type"],
        "url_citation"
    );
    assert_eq!(
        input[1]["content"][0]["annotations"][0]["url"],
        "https://example.com/guide"
    );
    assert_eq!(input[2]["role"], "user");
}

#[test]
fn responses_web_search_replay_preserves_interleaved_output_order() {
    let continuation = ProviderContinuation::new(
        "openai",
        "openai.responses.web_search.v1",
        serde_json::json!({
            "id": "ws_interleaved",
            "type": "web_search_call",
            "status": "completed",
            "action": {"type": "search", "queries": ["tea-rs"]}
        }),
    )
    .unwrap();
    let activity = HostedToolActivity::new(
        ToolCallId::from_str("0195a0b1-5e60-7000-8000-0aa7aa000100").unwrap(),
        "ws_interleaved",
        "web_search",
        serde_json::json!({"type": "search", "queries": ["tea-rs"]}),
        HostedToolOutcome::Success,
        Vec::new(),
        Some(continuation),
    )
    .unwrap();
    let timestamp = ProtocolTimestamp::from_str("2026-07-23T09:30:12.125Z").unwrap();
    let prior = CanonicalMessage::assistant(
        MessageId::from_str("0195a0b1-5e53-74b2-8c25-0aa7aa000100").unwrap(),
        vec![
            ContentBlock::text("I will check.").unwrap(),
            ContentBlock::hosted_tool(activity),
            ContentBlock::text("The source confirms it.").unwrap(),
        ],
        StopReason::Completed,
        timestamp,
    )
    .unwrap();
    let request = ModelRequest::new(ModelId::from_str("gpt-4.1").unwrap(), vec![prior]).unwrap();
    let config = MapCredentialResolver::new(std::collections::BTreeMap::from([
        ("TEA_OPENAI_API_KEY".to_owned(), "sk-test-key".to_owned()),
        ("TEA_OPENAI_MODEL".to_owned(), "gpt-4.1".to_owned()),
        ("TEA_OPENAI_API_MODE".to_owned(), "responses".to_owned()),
    ]))
    .resolve()
    .unwrap();

    let body = build_responses_body(&request, &config).unwrap();
    let input = body["input"].as_array().unwrap();

    assert_eq!(input.len(), 3);
    assert_eq!(input[0]["type"], "message");
    assert_eq!(input[0]["content"][0]["text"], "I will check.");
    assert_eq!(input[1]["type"], "web_search_call");
    assert_eq!(input[1]["id"], "ws_interleaved");
    assert_eq!(input[2]["type"], "message");
    assert_eq!(input[2]["content"][0]["text"], "The source confirms it.");
}

#[test]
fn responses_rejects_semantically_invalid_web_search_continuation() {
    let continuation = ProviderContinuation::new(
        "openai",
        "openai.responses.web_search.v1",
        serde_json::json!({
            "id": "ws_incomplete",
            "type": "web_search_call",
            "status": "in_progress",
            "action": {"type": "search", "queries": ["tea-rs"]}
        }),
    )
    .unwrap();
    let activity = HostedToolActivity::new(
        ToolCallId::from_str("0195a0b1-5e60-7000-8000-0aa7aa000101").unwrap(),
        "ws_incomplete",
        "web_search",
        serde_json::json!({"type": "search", "queries": ["tea-rs"]}),
        HostedToolOutcome::Success,
        Vec::new(),
        Some(continuation),
    )
    .unwrap();
    let prior = CanonicalMessage::assistant(
        MessageId::from_str("0195a0b1-5e53-74b2-8c25-0aa7aa000101").unwrap(),
        vec![ContentBlock::hosted_tool(activity)],
        StopReason::Completed,
        ProtocolTimestamp::from_str("2026-07-23T09:30:12.125Z").unwrap(),
    )
    .unwrap();
    let request = ModelRequest::new(ModelId::from_str("gpt-4.1").unwrap(), vec![prior]).unwrap();
    let config = MapCredentialResolver::new(std::collections::BTreeMap::from([
        ("TEA_OPENAI_API_KEY".to_owned(), "sk-test-key".to_owned()),
        ("TEA_OPENAI_MODEL".to_owned(), "gpt-4.1".to_owned()),
        ("TEA_OPENAI_API_MODE".to_owned(), "responses".to_owned()),
    ]))
    .resolve()
    .unwrap();

    let error = build_responses_body(&request, &config).unwrap_err();

    assert_eq!(
        error.code(),
        tea_provider_openai::OpenAiErrorCode::InvalidRequest
    );
    assert!(error.message().contains("continuation payload is invalid"));
}

#[test]
fn responses_rejects_semantically_invalid_url_citation_continuation() {
    let url = "https://example.com/citation";
    let continuation = ProviderContinuation::new(
        "openai",
        "openai.responses.url_citation.v1",
        serde_json::json!({
            "type": "url_citation",
            "start_index": 0,
            "end_index": 99,
            "url": url,
            "title": "Source"
        }),
    )
    .unwrap();
    let source = ExternalSource::new(url)
        .unwrap()
        .with_title("Source")
        .unwrap();
    let citation = SourceCitation::new(source)
        .with_range(0, 3)
        .and_then(|citation| citation.with_cited_text("Tea"))
        .map(|citation| citation.with_continuation(continuation))
        .unwrap();
    let assistant = CanonicalMessage::assistant(
        MessageId::from_str("0195a0b1-5e53-74b2-8c25-0aa7aa000102").unwrap(),
        vec![
            ContentBlock::text("Tea").unwrap(),
            ContentBlock::citation(citation),
        ],
        StopReason::Completed,
        ProtocolTimestamp::from_str("2026-07-23T09:30:12.125Z").unwrap(),
    )
    .unwrap();
    let request =
        ModelRequest::new(ModelId::from_str("gpt-4.1").unwrap(), vec![assistant]).unwrap();
    let config = MapCredentialResolver::new(std::collections::BTreeMap::from([
        ("TEA_OPENAI_API_KEY".to_owned(), "sk-test-key".to_owned()),
        ("TEA_OPENAI_MODEL".to_owned(), "gpt-4.1".to_owned()),
        ("TEA_OPENAI_API_MODE".to_owned(), "responses".to_owned()),
    ]))
    .resolve()
    .unwrap();

    let error = build_responses_body(&request, &config).unwrap_err();

    assert_eq!(
        error.code(),
        tea_provider_openai::OpenAiErrorCode::InvalidRequest
    );
    assert!(
        error
            .message()
            .contains("URL-citation continuation payload is invalid")
    );
}

#[test]
fn responses_normalizes_part_local_citation_offsets_for_replay() {
    let url = "https://example.com/citation";
    let continuation = ProviderContinuation::new(
        "openai",
        "openai.responses.url_citation.v1",
        citation_annotation(url, 0, 4),
    )
    .unwrap();
    let citation = SourceCitation::new(
        ExternalSource::new(url)
            .unwrap()
            .with_title("Source")
            .unwrap(),
    )
    .with_range(6, 10)
    .and_then(|citation| citation.with_cited_text("Beta"))
    .map(|citation| citation.with_continuation(continuation))
    .unwrap();
    let assistant = CanonicalMessage::assistant(
        MessageId::from_str("0195a0b1-5e53-74b2-8c25-0aa7aa000103").unwrap(),
        vec![
            ContentBlock::text("Alpha Beta").unwrap(),
            ContentBlock::citation(citation),
        ],
        StopReason::Completed,
        ProtocolTimestamp::from_str("2026-07-23T09:30:12.125Z").unwrap(),
    )
    .unwrap();
    let request =
        ModelRequest::new(ModelId::from_str("gpt-4.1").unwrap(), vec![assistant]).unwrap();
    let config = MapCredentialResolver::new(std::collections::BTreeMap::from([
        ("TEA_OPENAI_API_KEY".to_owned(), "sk-test-key".to_owned()),
        ("TEA_OPENAI_MODEL".to_owned(), "gpt-4.1".to_owned()),
        ("TEA_OPENAI_API_MODE".to_owned(), "responses".to_owned()),
    ]))
    .resolve()
    .unwrap();

    let body = build_responses_body(&request, &config).unwrap();
    let annotation = &body["input"][0]["content"][0]["annotations"][0];

    assert_eq!(annotation["start_index"], 6);
    assert_eq!(annotation["end_index"], 10);
}

#[test]
fn responses_url_citation_character_offsets_become_utf8_byte_ranges() {
    let mut reducer = ResponsesReducer::new();
    let mut events = Vec::new();
    for value in [
        serde_json::json!({
            "type": "response.created",
            "response": {"id": "resp_unicode", "model": "gpt-4.1"}
        }),
        serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"id": "ws_unicode", "type": "web_search_call"}
        }),
        serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": "ws_unicode",
                "type": "web_search_call",
                "status": "completed",
                "action": {
                    "type": "search",
                    "queries": ["unicode"],
                    "sources": [{"type": "url", "url": "https://example.com/unicode"}]
                }
            }
        }),
        serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {"id": "msg_unicode", "type": "message", "content": []}
        }),
        serde_json::json!({
            "type": "response.output_text.delta",
            "output_index": 1,
            "content_index": 0,
            "delta": "茶和茶"
        }),
        serde_json::json!({
            "type": "response.output_text.annotation.added",
            "output_index": 1,
            "content_index": 0,
            "annotation_index": 0,
            "item_id": "msg_unicode",
            "annotation": {
                "type": "url_citation",
                "start_index": 2,
                "end_index": 3,
                "url": "https://example.com/unicode",
                "title": "Unicode source"
            }
        }),
    ] {
        events.extend(reducer.map_chunk(&value).unwrap());
    }

    let citation = events
        .iter()
        .find_map(ModelEvent::as_source_citation)
        .unwrap();
    assert_eq!(citation.citation().range(), Some((6, 9)));
    assert_eq!(citation.citation().cited_text(), Some("茶"));
}

#[test]
fn responses_failed_web_search_is_a_hosted_outcome_not_transport_failure() {
    let mut reducer = ResponsesReducer::new();
    let mut events = reducer
        .map_chunk(&serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"id": "ws_failed", "type": "web_search_call"}
        }))
        .unwrap();
    events.extend(
        reducer
            .map_chunk(&serde_json::json!({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": "ws_failed",
                    "type": "web_search_call",
                    "status": "failed",
                    "action": null
                }
            }))
            .unwrap(),
    );

    let completed = events
        .iter()
        .find_map(ModelEvent::as_hosted_tool_completed)
        .unwrap();
    let HostedToolOutcome::Error(error) = completed.outcome() else {
        panic!("failed provider call must remain a hosted tool error");
    };
    assert_eq!(error.code(), "provider_failed");
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, ModelEvent::Failed(_)))
    );
}

#[test]
fn responses_completed_rejects_unfinished_web_search() {
    let mut reducer = ResponsesReducer::new();
    reducer
        .map_chunk(&serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"id": "ws_unfinished", "type": "web_search_call"}
        }))
        .unwrap();

    let error = reducer
        .map_chunk(&serde_json::json!({
            "type": "response.completed",
            "response": {
                "id": "resp_unfinished",
                "status": "completed",
                "output": []
            }
        }))
        .unwrap_err();

    assert_eq!(
        error.code(),
        tea_provider_openai::OpenAiErrorCode::MalformedResponse
    );
    assert!(error.message().contains("without completion"));
}

#[test]
fn responses_completed_rejects_unfinished_function_call() {
    let mut reducer = ResponsesReducer::new();
    reducer
        .map_chunk(&serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": "call_unfinished",
                "name": "read_file",
                "arguments": ""
            }
        }))
        .unwrap();

    let error = reducer
        .map_chunk(&serde_json::json!({
            "type": "response.completed",
            "response": {
                "id": "resp_unfinished",
                "status": "completed",
                "output": []
            }
        }))
        .unwrap_err();

    assert_eq!(
        error.code(),
        tea_provider_openai::OpenAiErrorCode::MalformedResponse
    );
    assert!(error.message().contains("function call without completion"));
}

#[test]
fn responses_rejects_output_index_type_changes() {
    let mut reducer = ResponsesReducer::new();
    reducer
        .map_chunk(&serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"id": "ws_changed", "type": "web_search_call"}
        }))
        .unwrap();

    let error = reducer
        .map_chunk(&serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": "call_changed",
                "name": "read_file",
                "arguments": "{}"
            }
        }))
        .unwrap_err();

    assert_eq!(
        error.code(),
        tea_provider_openai::OpenAiErrorCode::MalformedResponse
    );
    assert!(error.message().contains("changed type"));

    let mut reducer = ResponsesReducer::new();
    reducer
        .map_chunk(&serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": "call_changed",
                "name": "read_file",
                "arguments": ""
            }
        }))
        .unwrap();
    let error = reducer
        .map_chunk(&serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": "ws_changed",
                "type": "web_search_call",
                "status": "completed",
                "action": {"type": "search", "queries": ["tea"]}
            }
        }))
        .unwrap_err();
    assert!(error.message().contains("changed type"));
}

#[test]
fn responses_resolves_reference_events_by_declared_item_identity() {
    let mut reducer = ResponsesReducer::new();
    for value in [
        serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"id": "reasoning_0", "type": "reasoning"}
        }),
        serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"id": "message_1", "type": "message", "content": []}
        }),
    ] {
        reducer.map_chunk(&value).unwrap();
    }

    let events = reducer
        .map_chunk(&serde_json::json!({
            "type": "response.output_text.delta",
            "output_index": 0,
            "item_id": "message_1",
            "content_index": 0,
            "delta": "identity wins"
        }))
        .unwrap();

    assert!(matches!(
        events.as_slice(),
        [ModelEvent::TextDelta(delta)] if delta.as_str() == "identity wins"
    ));
}

#[test]
fn responses_allows_reasoning_and_message_streams_to_reuse_a_non_executable_index() {
    let mut reducer = ResponsesReducer::new();
    reducer
        .map_chunk(&serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"type": "reasoning"}
        }))
        .unwrap();

    let events = reducer
        .map_chunk(&serde_json::json!({
            "type": "response.output_text.delta",
            "output_index": 0,
            "content_index": 0,
            "delta": "visible answer"
        }))
        .unwrap();

    assert!(matches!(
        events.as_slice(),
        [ModelEvent::TextDelta(delta)] if delta.as_str() == "visible answer"
    ));

    let text_done_events = reducer
        .map_chunk(&serde_json::json!({
            "type": "response.output_text.done",
            "output_index": 1,
            "content_index": 0,
            "text": "visible answer"
        }))
        .unwrap();
    assert!(text_done_events.is_empty());

    let terminal_item_events = reducer
        .map_chunk(&serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {
                "id": "message_terminal",
                "type": "message",
                "content": [{
                    "type": "output_text",
                    "text": "visible answer",
                    "annotations": []
                }]
            }
        }))
        .unwrap();
    assert!(terminal_item_events.is_empty());
}

#[test]
fn responses_reasoning_keeps_multiple_summary_parts_independent() {
    let mut reducer = ResponsesReducer::new();
    let streamed = [
        serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"id": "reasoning_multi", "type": "reasoning", "summary": []}
        }),
        serde_json::json!({
            "type": "response.reasoning_summary_text.delta",
            "output_index": 0,
            "item_id": "reasoning_multi",
            "summary_index": 0,
            "delta": "First part."
        }),
        serde_json::json!({
            "type": "response.reasoning_summary_text.delta",
            "output_index": 0,
            "item_id": "reasoning_multi",
            "summary_index": 1,
            "delta": "Second part."
        }),
        serde_json::json!({
            "type": "response.reasoning_summary_text.done",
            "output_index": 0,
            "item_id": "reasoning_multi",
            "summary_index": 0,
            "text": "First part."
        }),
        serde_json::json!({
            "type": "response.reasoning_summary_text.done",
            "output_index": 0,
            "item_id": "reasoning_multi",
            "summary_index": 1,
            "text": "Second part."
        }),
        serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": "reasoning_multi",
                "type": "reasoning",
                "summary": [
                    {"type": "summary_text", "text": "First part."},
                    {"type": "summary_text", "text": "Second part."}
                ]
            }
        }),
    ];
    let mut events = Vec::new();
    for value in streamed {
        events.extend(reducer.map_chunk(&value).unwrap());
    }

    events.extend(
        reducer
            .map_chunk(&serde_json::json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_reasoning_multi",
                    "status": "completed",
                    "output": [{
                        "id": "reasoning_multi",
                        "type": "reasoning",
                        "summary": [
                            {"type": "summary_text", "text": "First part."},
                            {"type": "summary_text", "text": "Second part."}
                        ]
                    }]
                }
            }))
            .unwrap(),
    );

    let thinking = events
        .iter()
        .filter_map(ModelEvent::as_thinking_delta)
        .collect::<String>();
    assert_eq!(thinking, "First part.Second part.");
    assert!(matches!(events.last(), Some(ModelEvent::Completed(_))));
    assert!(events_validator_accepts(&events));
}

#[test]
fn responses_rejects_conflicting_duplicate_web_search_completion() {
    let url = "https://example.com/duplicate";
    let mut reducer = ResponsesReducer::new();
    reducer
        .map_chunk(&web_search_added(0, "ws_duplicate"))
        .unwrap();
    let completed = web_search_done(0, "ws_duplicate", url);
    reducer.map_chunk(&completed).unwrap();
    assert!(reducer.map_chunk(&completed).unwrap().is_empty());

    let mut conflict = completed;
    conflict["item"]["action"]["queries"][0] = serde_json::json!("changed");
    let error = reducer.map_chunk(&conflict).unwrap_err();
    assert!(error.message().contains("changed payload"));
}

#[test]
fn responses_rejects_conflicting_web_search_in_completed_response_snapshot() {
    let url = "https://example.com/duplicate";
    let mut reducer = ResponsesReducer::new();
    reducer
        .map_chunk(&web_search_added(0, "ws_duplicate"))
        .unwrap();
    reducer
        .map_chunk(&web_search_done(0, "ws_duplicate", url))
        .unwrap();

    let mut conflict = web_search_item("ws_duplicate", url);
    conflict["action"]["queries"][0] = serde_json::json!("changed");
    let error = reducer
        .map_chunk(&serde_json::json!({
            "type": "response.completed",
            "response": {
                "id": "resp_duplicate",
                "status": "completed",
                "output": [conflict]
            }
        }))
        .unwrap_err();

    assert!(error.message().contains("changed payload"));
}

#[test]
fn responses_completed_snapshot_can_supply_the_hosted_search_lifecycle() {
    let mut reducer = ResponsesReducer::new();
    let events = reducer
        .map_chunk(&serde_json::json!({
            "type": "response.completed",
            "response": {
                "id": "resp_snapshot_only",
                "status": "completed",
                "output": [web_search_item(
                    "ws_snapshot_only",
                    "https://example.com/snapshot-only"
                )]
            }
        }))
        .unwrap();

    assert!(matches!(events.first(), Some(ModelEvent::Started(_))));
    assert!(
        events
            .iter()
            .any(|event| matches!(event, ModelEvent::HostedToolStarted(_)))
    );
    assert!(
        events
            .iter()
            .any(|event| matches!(event, ModelEvent::HostedToolCompleted(_)))
    );
    assert!(matches!(events.last(), Some(ModelEvent::Completed(_))));
    assert!(events_validator_accepts(&events));
}

#[test]
fn responses_rejects_conflicting_function_call_in_completed_response_snapshot() {
    let mut reducer = ResponsesReducer::new();
    for value in [
        serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": "call_snapshot",
                "name": "read_file",
                "arguments": "{}"
            }
        }),
        serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "call_id": "call_snapshot",
                "name": "read_file",
                "arguments": "{}"
            }
        }),
    ] {
        reducer.map_chunk(&value).unwrap();
    }

    let error = reducer
        .map_chunk(&serde_json::json!({
            "type": "response.completed",
            "response": {
                "id": "resp_function_snapshot",
                "status": "completed",
                "output": [{
                    "type": "function_call",
                    "call_id": "call_snapshot",
                    "name": "read_file",
                    "arguments": "{\"path\":\"changed\"}"
                }]
            }
        }))
        .unwrap_err();

    assert!(error.message().contains("arguments"));
}

#[test]
#[allow(clippy::too_many_lines)]
fn responses_completed_does_not_replay_items_when_terminal_ids_change() {
    let mut reducer = ResponsesReducer::new();
    let streamed = [
        serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {"id": "reasoning_stream", "type": "reasoning", "summary": []}
        }),
        serde_json::json!({
            "type": "response.reasoning_summary_text.delta",
            "output_index": 0,
            "item_id": "reasoning_stream",
            "delta": "Need the file."
        }),
        serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "id": "reasoning_stream",
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Need the file."}]
            }
        }),
        serde_json::json!({
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {
                "id": "function_stream",
                "type": "function_call",
                "call_id": "call_read",
                "name": "read",
                "arguments": ""
            }
        }),
        serde_json::json!({
            "type": "response.function_call_arguments.done",
            "output_index": 1,
            "item_id": "function_stream",
            "arguments": "{\"path\":\"README.md\"}"
        }),
        serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {
                "id": "function_stream",
                "type": "function_call",
                "call_id": "call_read",
                "name": "read",
                "arguments": "{\"path\":\"README.md\"}"
            }
        }),
    ];
    let mut events = Vec::new();
    for value in streamed {
        events.extend(reducer.map_chunk(&value).unwrap());
    }

    events.extend(
        reducer
            .map_chunk(&serde_json::json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_changed_ids",
                    "status": "completed",
                    "output": [
                        {
                            "id": "reasoning_terminal",
                            "type": "reasoning",
                            "summary": [{"type": "summary_text", "text": "Need the file."}]
                        },
                        {
                            "id": "function_terminal",
                            "type": "function_call",
                            "call_id": "call_read",
                            "name": "read",
                            "arguments": "{\"path\":\"README.md\"}"
                        }
                    ]
                }
            }))
            .unwrap(),
    );

    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, ModelEvent::ToolCallStarted(_)))
            .count(),
        1
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| matches!(event, ModelEvent::ToolCallCompleted(_)))
            .count(),
        1
    );
    let thinking = events
        .iter()
        .filter_map(ModelEvent::as_thinking_delta)
        .collect::<String>();
    assert_eq!(thinking, "Need the file.");
    assert!(matches!(events.last(), Some(ModelEvent::Completed(_))));
    assert!(events_validator_accepts(&events));
}

#[test]
fn responses_rejects_known_output_index_changing_to_unknown_type() {
    let mut reducer = ResponsesReducer::new();
    reducer
        .map_chunk(&web_search_added(0, "ws_changed"))
        .unwrap();

    let error = reducer
        .map_chunk(&serde_json::json!({
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {"id": "ws_changed", "type": "future_tool_call"}
        }))
        .unwrap_err();

    assert!(error.message().contains("changed type"));
}

#[test]
fn responses_rejects_conflicting_duplicate_citation() {
    let url = "https://example.com/duplicate";
    let mut reducer = ResponsesReducer::new();
    for value in [
        web_search_added(0, "ws_duplicate"),
        web_search_done(0, "ws_duplicate", url),
        message_added(1, "msg_duplicate"),
        output_text_delta(1, 0, "Tea."),
    ] {
        reducer.map_chunk(&value).unwrap();
    }
    let citation = citation_added(1, 0, 0, url, 0, 3);
    assert_eq!(reducer.map_chunk(&citation).unwrap().len(), 1);
    assert!(reducer.map_chunk(&citation).unwrap().is_empty());

    let mut conflict = citation;
    conflict["annotation"]["title"] = serde_json::json!("Changed");
    let error = reducer.map_chunk(&conflict).unwrap_err();
    assert!(error.message().contains("changed payload"));
}

#[test]
fn responses_defers_citation_identity_until_search_completion() {
    let mut reducer = ResponsesReducer::new();
    let mut events = Vec::new();
    for value in [
        web_search_added(0, "ws_deferred"),
        message_added(1, "msg_deferred"),
        output_text_delta(1, 0, "Tea."),
    ] {
        events.extend(reducer.map_chunk(&value).unwrap());
    }
    let annotation_events = reducer
        .map_chunk(&citation_added(
            1,
            0,
            0,
            "https://example.com/deferred",
            0,
            3,
        ))
        .unwrap();
    assert!(
        annotation_events
            .iter()
            .all(|event| event.as_source_citation().is_none())
    );
    events.extend(annotation_events);

    let completion_events = reducer
        .map_chunk(&web_search_done(
            0,
            "ws_deferred",
            "https://example.com/deferred",
        ))
        .unwrap();
    let citation = completion_events
        .iter()
        .find_map(ModelEvent::as_source_citation)
        .expect("citation must be emitted once source identity is stable");
    assert_eq!(citation.provider_call_id().unwrap().as_str(), "ws_deferred");
    let completed_position = completion_events
        .iter()
        .position(|event| event.as_hosted_tool_completed().is_some())
        .unwrap();
    let citation_position = completion_events
        .iter()
        .position(|event| event.as_source_citation().is_some())
        .unwrap();
    assert!(completed_position < citation_position);
    events.extend(completion_events);

    events.extend(
        reducer
            .map_chunk(&serde_json::json!({
                "type": "response.completed",
                "response": {
                    "id": "resp_deferred",
                    "status": "completed",
                    "output": [
                        web_search_item("ws_deferred", "https://example.com/deferred"),
                        message_item(
                            "msg_deferred",
                            "Tea.",
                            &[citation_annotation(
                                "https://example.com/deferred",
                                0,
                                3,
                            )],
                        )
                    ]
                }
            }))
            .unwrap(),
    );
    assert_eq!(
        events
            .iter()
            .filter(|event| event.as_source_citation().is_some())
            .count(),
        1
    );
    assert!(events_validator_accepts(&events));
}

#[test]
fn responses_waits_for_all_searches_before_resolving_citation_identity() {
    let url = "https://example.com/shared";
    let mut reducer = ResponsesReducer::new();
    let mut events = Vec::new();
    for value in [
        web_search_added(0, "ws_first"),
        web_search_done(0, "ws_first", url),
        web_search_added(1, "ws_second"),
        message_added(2, "msg_shared"),
        output_text_delta(2, 0, "Tea."),
    ] {
        events.extend(reducer.map_chunk(&value).unwrap());
    }

    let annotation_events = reducer
        .map_chunk(&citation_added(2, 0, 0, url, 0, 3))
        .unwrap();
    assert!(
        annotation_events
            .iter()
            .all(|event| event.as_source_citation().is_none())
    );
    events.extend(annotation_events);
    events.extend(
        reducer
            .map_chunk(&web_search_done(1, "ws_second", url))
            .unwrap(),
    );
    assert!(
        events
            .iter()
            .all(|event| event.as_source_citation().is_none())
    );
    events.extend(
        reducer
            .map_chunk(&serde_json::json!({
                "type": "response.completed",
                "response": {"id": "resp_shared", "status": "completed", "output": []}
            }))
            .unwrap(),
    );

    let citations = events
        .iter()
        .filter_map(ModelEvent::as_source_citation)
        .collect::<Vec<_>>();
    assert_eq!(citations.len(), 1);
    assert!(citations[0].provider_call_id().is_none());
    assert!(events_validator_accepts(&events));
}

#[test]
fn responses_bounds_pending_citation_count_and_bytes() {
    let url = "https://example.com/pending";
    let mut reducer = reducer_with_pending_search("x");
    for annotation_index in 0..256 {
        let events = reducer
            .map_chunk(&citation_added(1, 0, annotation_index, url, 0, 1))
            .unwrap();
        assert!(events.is_empty());
    }
    let error = reducer
        .map_chunk(&citation_added(1, 0, 256, url, 0, 1))
        .unwrap_err();
    assert!(
        error
            .message()
            .contains("pending URL citations exceeded bounds")
    );

    let mut reducer = reducer_with_pending_search("x");
    let mut first = citation_added(1, 0, 0, url, 0, 1);
    first["annotation"]["opaque"] = serde_json::Value::String("x".repeat(2_200_000));
    assert!(reducer.map_chunk(&first).unwrap().is_empty());
    let mut second = citation_added(1, 0, 1, url, 0, 1);
    second["annotation"]["opaque"] = serde_json::Value::String("x".repeat(2_200_000));
    let error = reducer.map_chunk(&second).unwrap_err();
    assert!(
        error
            .message()
            .contains("pending URL citations exceeded bounds")
    );
}

#[test]
fn responses_citation_offsets_are_local_to_the_output_content_part() {
    let url = "https://example.com/multipart";
    let mut reducer = ResponsesReducer::new();
    let mut events = Vec::new();
    for value in [
        web_search_added(0, "ws_multipart"),
        web_search_done(0, "ws_multipart", url),
        message_added(1, "msg_multipart"),
        output_text_delta(1, 0, "Alpha "),
        output_text_delta(1, 1, "Beta"),
        citation_added(1, 1, 0, url, 0, 4),
    ] {
        events.extend(reducer.map_chunk(&value).unwrap());
    }

    let citation = events
        .iter()
        .find_map(ModelEvent::as_source_citation)
        .unwrap();
    assert_eq!(citation.citation().range(), Some((6, 10)));
    assert_eq!(citation.citation().cited_text(), Some("Beta"));
}

fn events_validator_accepts(events: &[ModelEvent]) -> bool {
    let mut validator = ModelStreamValidator::new();
    for event in events {
        if validator.observe(event).is_err() {
            return false;
        }
    }
    validator.finish().is_ok()
}
