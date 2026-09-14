use tea_model::{ModelEvent, ModelStreamValidator};
use tea_protocol::{HostedToolOutcome, StopReason};
use tea_provider_anthropic::AnthropicErrorCode;
use tea_provider_anthropic::sse::{SseEvent, SseParser};
use tea_provider_anthropic::stream::{AnthropicReducer, map_http_failure, map_stream_error};

fn fixture_events(fixture: &str) -> Vec<ModelEvent> {
    let mut parser = SseParser::new();
    let mut parsed = parser.feed(fixture.as_bytes());
    parsed.extend(parser.finish());
    let mut reducer = AnthropicReducer::new();
    parsed
        .into_iter()
        .flat_map(|event| match event {
            SseEvent::Data(payload) => {
                let value = serde_json::from_str(&payload).unwrap();
                reducer.map_chunk(&value).unwrap()
            }
            SseEvent::Done => reducer.finish().unwrap(),
        })
        .collect()
}

fn assert_valid_stream(events: &[ModelEvent]) {
    let mut validator = ModelStreamValidator::new();
    for event in events {
        validator.observe(event).unwrap();
    }
    validator.finish().unwrap();
}

#[test]
fn maps_text_tools_and_usage_from_messages_events() {
    let mut reducer = AnthropicReducer::new();
    let chunks = [
        serde_json::json!({
            "type":"message_start",
            "message":{"id":"msg_123","model":"claude-sonnet-4-20250514","usage":{"input_tokens":10,"cache_read_input_tokens":2}}
        }),
        serde_json::json!({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}),
        serde_json::json!({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}),
        serde_json::json!({"type":"content_block_stop","index":0}),
        serde_json::json!({"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_123","name":"read_file","input":{}}}),
        serde_json::json!({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"notes.txt\"}"}}),
        serde_json::json!({"type":"content_block_stop","index":1}),
        serde_json::json!({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":3}}),
        serde_json::json!({"type":"message_stop"}),
    ];
    let events = chunks
        .iter()
        .flat_map(|chunk| reducer.map_chunk(chunk).unwrap())
        .collect::<Vec<_>>();

    assert!(matches!(events.first(), Some(ModelEvent::Started(_))));
    assert!(
        events.iter().any(
            |event| matches!(event, ModelEvent::TextDelta(delta) if delta.as_str() == "Hello")
        )
    );
    assert!(events.iter().any(|event| matches!(event, ModelEvent::ToolCallStarted(call) if call.tool_name() == "read_file")));
    assert!(events.iter().any(|event| matches!(event, ModelEvent::ToolCallCompleted(call) if call.arguments()["path"] == "notes.txt")));
    let completion = events
        .iter()
        .find_map(|event| match event {
            ModelEvent::Completed(completion) => Some(completion),
            _ => None,
        })
        .unwrap();
    assert_eq!(completion.usage().unwrap().input_tokens().get(), 10);
    assert_eq!(completion.usage().unwrap().output_tokens().get(), 3);
    assert_eq!(
        completion
            .usage()
            .unwrap()
            .cache_read_tokens()
            .unwrap()
            .get(),
        2
    );
}

#[test]
fn refusal_fixture_preserves_visible_text_and_terminal_reason() {
    let events = fixture_events(include_str!("fixtures/refusal.sse"));
    let text = events
        .iter()
        .filter_map(ModelEvent::as_text_delta)
        .collect::<String>();

    assert_eq!(text, "I cannot comply.");
    assert!(matches!(
        events.last(),
        Some(ModelEvent::Completed(completion))
            if completion.stop_reason() == &StopReason::Refusal
    ));
    assert_valid_stream(&events);
}

#[test]
fn unknown_stop_reason_is_not_downgraded_to_success() {
    let mut reducer = AnthropicReducer::new();
    let events = [
        serde_json::json!({
            "type":"message_start",
            "message":{"id":"msg_future","model":"claude-test","usage":{"input_tokens":1}}
        }),
        serde_json::json!({
            "type":"message_delta",
            "delta":{"stop_reason":"future_stop_reason"},
            "usage":{"output_tokens":1}
        }),
        serde_json::json!({"type":"message_stop"}),
    ]
    .iter()
    .flat_map(|chunk| reducer.map_chunk(chunk).unwrap())
    .collect::<Vec<_>>();

    assert!(matches!(
        events.last(),
        Some(ModelEvent::Completed(completion))
            if completion.stop_reason()
                == &StopReason::Unknown("future_stop_reason".to_owned())
    ));
    assert_valid_stream(&events);
}

#[test]
fn nullable_and_missing_stop_reason_deltas_are_nonterminal() {
    let mut reducer = AnthropicReducer::new();
    let mut events = reducer
        .map_chunk(&serde_json::json!({
            "type":"message_start",
            "message":{"id":"msg_missing_stop","model":"claude-test","usage":{"input_tokens":1}}
        }))
        .unwrap();
    events.extend(
        reducer
            .map_chunk(&serde_json::json!({
                "type":"message_delta",
                "delta":{"stop_reason":null},
                "usage":{"output_tokens":1}
            }))
            .unwrap(),
    );
    events.extend(
        reducer
            .map_chunk(&serde_json::json!({
                "type":"message_delta",
                "delta":{},
                "usage":{"output_tokens":2}
            }))
            .unwrap(),
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, ModelEvent::Completed(_) | ModelEvent::Failed(_)))
    );
    assert!(!reducer.terminal_emitted());

    events.extend(
        reducer
            .map_chunk(&serde_json::json!({
                "type":"message_delta",
                "delta":{"stop_reason":"end_turn"},
                "usage":{"output_tokens":3}
            }))
            .unwrap(),
    );
    assert!(
        !events
            .iter()
            .any(|event| matches!(event, ModelEvent::Completed(_) | ModelEvent::Failed(_)))
    );
    events.extend(
        reducer
            .map_chunk(&serde_json::json!({"type":"message_stop"}))
            .unwrap(),
    );

    let Some(ModelEvent::Completed(completion)) = events.last() else {
        panic!("message_stop must emit completion");
    };
    assert_eq!(completion.stop_reason(), &StopReason::Completed);
    assert_eq!(completion.usage().unwrap().output_tokens().get(), 3);
    assert_valid_stream(&events);
}

#[test]
fn provider_failures_use_stable_categories() {
    let http_failure = map_http_failure(
        403,
        r#"{"error":{"message":"blocked by gateway WAF","api_key":"sk-secret"}}"#,
    );
    assert!(matches!(
        http_failure.code(),
        tea_model::ModelFailureCode::PermissionDenied
    ));
    assert_eq!(http_failure.message(), "HTTP 403: blocked by gateway WAF");
    assert!(!http_failure.message().contains("sk-secret"));
    assert!(http_failure.is_safe_diagnostic());
    assert!(matches!(
        map_http_failure(429, "slow down").code(),
        tea_model::ModelFailureCode::RateLimited
    ));
    assert!(matches!(
        map_stream_error(Some(
            &serde_json::json!({"type":"overloaded_error","message":"busy"})
        ))
        .code(),
        tea_model::ModelFailureCode::Unavailable
    ));
}

#[test]
fn message_stop_without_a_terminal_reason_is_malformed() {
    let mut reducer = AnthropicReducer::new();
    reducer
        .map_chunk(&serde_json::json!({"type":"message_start","message":{"id":"msg_123"}}))
        .unwrap();

    let error = reducer
        .map_chunk(&serde_json::json!({"type":"message_stop"}))
        .unwrap_err();

    assert_eq!(error.code(), AnthropicErrorCode::MalformedResponse);
    assert!(!reducer.terminal_emitted());
}

#[test]
fn end_of_stream_without_a_terminal_reason_is_malformed() {
    let mut reducer = AnthropicReducer::new();
    reducer
        .map_chunk(&serde_json::json!({
            "type":"message_start",
            "message":{"id":"msg_truncated","model":"claude-test","usage":{"input_tokens":1}}
        }))
        .unwrap();

    let error = reducer.finish().unwrap_err();

    assert_eq!(error.code(), AnthropicErrorCode::MalformedResponse);
    assert!(!reducer.terminal_emitted());
}

#[test]
fn end_of_stream_after_stop_reason_before_message_stop_is_malformed() {
    let mut reducer = AnthropicReducer::new();
    reducer
        .map_chunk(&serde_json::json!({
            "type":"message_start",
            "message":{"id":"msg_truncated_after_reason"}
        }))
        .unwrap();
    reducer
        .map_chunk(&serde_json::json!({
            "type":"message_delta",
            "delta":{"stop_reason":"end_turn"},
            "usage":{"output_tokens":1}
        }))
        .unwrap();

    let error = reducer.finish().unwrap_err();

    assert_eq!(error.code(), AnthropicErrorCode::MalformedResponse);
    assert!(!reducer.terminal_emitted());
}

#[test]
fn stop_reason_closes_the_content_phase() {
    let reducer_after_stop_reason = |open_text_block: bool| {
        let mut reducer = AnthropicReducer::new();
        reducer
            .map_chunk(&serde_json::json!({
                "type":"message_start",
                "message":{"id":"msg_pending_stop"}
            }))
            .unwrap();
        if open_text_block {
            reducer
                .map_chunk(&serde_json::json!({
                    "type":"content_block_start",
                    "index":0,
                    "content_block":{"type":"text","text":""}
                }))
                .unwrap();
        }
        reducer
            .map_chunk(&serde_json::json!({
                "type":"message_delta",
                "delta":{"stop_reason":"end_turn"}
            }))
            .unwrap();
        reducer
    };
    let cases = [
        (
            "content block start",
            false,
            serde_json::json!({
                "type":"content_block_start",
                "index":1,
                "content_block":{"type":"text","text":"late"}
            }),
        ),
        (
            "content block delta",
            true,
            serde_json::json!({
                "type":"content_block_delta",
                "index":0,
                "delta":{"type":"text_delta","text":"late"}
            }),
        ),
        (
            "content block stop",
            true,
            serde_json::json!({"type":"content_block_stop","index":0}),
        ),
        (
            "second message delta",
            false,
            serde_json::json!({"type":"message_delta","delta":{}}),
        ),
        (
            "stream error",
            false,
            serde_json::json!({
                "type":"error",
                "error":{"type":"api_error","message":"late"}
            }),
        ),
        (
            "unknown data event",
            false,
            serde_json::json!({"type":"future_event","future_field":true}),
        ),
    ];

    for (name, open_text_block, event) in cases {
        let error = reducer_after_stop_reason(open_text_block)
            .map_chunk(&event)
            .unwrap_err();
        assert_eq!(
            error.code(),
            AnthropicErrorCode::MalformedResponse,
            "{name}"
        );
    }

    let mut reducer = reducer_after_stop_reason(false);
    assert!(
        reducer
            .map_chunk(&serde_json::json!({"type":"ping"}))
            .unwrap()
            .is_empty()
    );
    assert!(matches!(
        reducer
            .map_chunk(&serde_json::json!({"type":"message_stop"}))
            .unwrap()
            .last(),
        Some(ModelEvent::Completed(_))
    ));
}

#[test]
fn duplicate_message_stop_is_malformed() {
    let mut reducer = AnthropicReducer::new();
    reducer
        .map_chunk(&serde_json::json!({
            "type":"message_start",
            "message":{"id":"msg_duplicate_stop"}
        }))
        .unwrap();
    reducer
        .map_chunk(&serde_json::json!({
            "type":"message_delta",
            "delta":{"stop_reason":"end_turn"}
        }))
        .unwrap();
    let events = reducer
        .map_chunk(&serde_json::json!({"type":"message_stop"}))
        .unwrap();
    assert!(matches!(events.last(), Some(ModelEvent::Completed(_))));

    let error = reducer
        .map_chunk(&serde_json::json!({"type":"message_stop"}))
        .unwrap_err();
    assert_eq!(error.code(), AnthropicErrorCode::MalformedResponse);
}

#[test]
fn duplicate_message_start_is_malformed() {
    let mut reducer = AnthropicReducer::new();
    reducer
        .map_chunk(&serde_json::json!({
            "type":"message_start",
            "message":{"id":"msg_duplicate_start","usage":{"input_tokens":1}}
        }))
        .unwrap();

    let error = reducer
        .map_chunk(&serde_json::json!({
            "type":"message_start",
            "message":{"id":"msg_duplicate_start","usage":{"input_tokens":1}}
        }))
        .unwrap_err();

    assert_eq!(error.code(), AnthropicErrorCode::MalformedResponse);
    assert!(!reducer.terminal_emitted());
}

#[test]
fn terminal_rejects_every_non_ping_event() {
    let terminal_reducer = || {
        let mut reducer = AnthropicReducer::new();
        reducer
            .map_chunk(&serde_json::json!({
                "type":"message_start",
                "message":{"id":"msg_terminal"}
            }))
            .unwrap();
        reducer
            .map_chunk(&serde_json::json!({
                "type":"message_delta",
                "delta":{"stop_reason":"end_turn"}
            }))
            .unwrap();
        reducer
            .map_chunk(&serde_json::json!({"type":"message_stop"}))
            .unwrap();
        reducer
    };
    let cases = [
        serde_json::json!({
            "type":"content_block_start",
            "index":0,
            "content_block":{"type":"text","text":""}
        }),
        serde_json::json!({"type":"future_event","future_field":true}),
        serde_json::json!({"type":"error","error":{"type":"api_error","message":"late"}}),
    ];

    for event in cases {
        let error = terminal_reducer().map_chunk(&event).unwrap_err();
        assert_eq!(error.code(), AnthropicErrorCode::MalformedResponse);
    }

    assert!(
        terminal_reducer()
            .map_chunk(&serde_json::json!({"type":"ping"}))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn unknown_events_before_terminal_remain_forward_compatible() {
    let mut reducer = AnthropicReducer::new();
    reducer
        .map_chunk(&serde_json::json!({
            "type":"message_start",
            "message":{"id":"msg_future_event"}
        }))
        .unwrap();

    assert!(
        reducer
            .map_chunk(&serde_json::json!({
                "type":"future_event",
                "future_field":{"nested":true}
            }))
            .unwrap()
            .is_empty()
    );
    reducer
        .map_chunk(&serde_json::json!({
            "type":"message_delta",
            "delta":{"stop_reason":"end_turn"}
        }))
        .unwrap();
    assert!(matches!(
        reducer
            .map_chunk(&serde_json::json!({"type":"message_stop"}))
            .unwrap()
            .last(),
        Some(ModelEvent::Completed(_))
    ));
}

#[test]
fn invalid_known_usage_fields_are_malformed() {
    let start_usage_cases = [
        serde_json::json!({"input_tokens":"1"}),
        serde_json::json!({"cache_read_input_tokens":-1}),
        serde_json::json!({"cache_creation_input_tokens":1.5}),
        serde_json::json!({"output_tokens":9_007_199_254_740_992_u64}),
    ];
    for usage in start_usage_cases {
        let error = AnthropicReducer::new()
            .map_chunk(&serde_json::json!({
                "type":"message_start",
                "message":{"id":"msg_invalid_start_usage","usage":usage}
            }))
            .unwrap_err();
        assert_eq!(error.code(), AnthropicErrorCode::MalformedResponse);
    }

    let mut reducer = AnthropicReducer::new();
    reducer
        .map_chunk(&serde_json::json!({
            "type":"message_start",
            "message":{"id":"msg_invalid_delta_usage","usage":{"input_tokens":1}}
        }))
        .unwrap();
    let error = reducer
        .map_chunk(&serde_json::json!({
            "type":"message_delta",
            "delta":{"stop_reason":"end_turn"},
            "usage":{"output_tokens":"1"}
        }))
        .unwrap_err();
    assert_eq!(error.code(), AnthropicErrorCode::MalformedResponse);
}

#[test]
fn rejects_invalid_content_block_lifecycles() {
    let text_start = || {
        serde_json::json!({
            "type":"content_block_start",
            "index":0,
            "content_block":{"type":"text","text":""}
        })
    };
    let text_delta = || {
        serde_json::json!({
            "type":"content_block_delta",
            "index":0,
            "delta":{"type":"text_delta","text":"hello"}
        })
    };
    let stop = || serde_json::json!({"type":"content_block_stop","index":0});
    let cases = [
        ("delta before start", vec![text_delta()]),
        ("stop before start", vec![stop()]),
        ("duplicate start", vec![text_start(), text_start()]),
        ("duplicate stop", vec![text_start(), stop(), stop()]),
        ("delta after stop", vec![text_start(), stop(), text_delta()]),
        (
            "delta kind mismatch",
            vec![
                serde_json::json!({
                    "type":"content_block_start",
                    "index":0,
                    "content_block":{"type":"tool_use","id":"toolu_123","name":"read_file","input":{}}
                }),
                text_delta(),
            ],
        ),
        (
            "known delta on unknown block",
            vec![
                serde_json::json!({
                    "type":"content_block_start",
                    "index":0,
                    "content_block":{"type":"future_block"}
                }),
                text_delta(),
            ],
        ),
        (
            "terminal with open block",
            vec![
                text_start(),
                serde_json::json!({
                    "type":"message_delta",
                    "delta":{"stop_reason":"end_turn"},
                    "usage":{"output_tokens":1}
                }),
                serde_json::json!({"type":"message_stop"}),
            ],
        ),
    ];

    for (name, chunks) in cases {
        let mut reducer = AnthropicReducer::new();
        for chunk in &chunks[..chunks.len() - 1] {
            reducer
                .map_chunk(chunk)
                .unwrap_or_else(|error| panic!("{name} setup failed: {error}"));
        }
        let error = reducer.map_chunk(chunks.last().unwrap()).unwrap_err();

        assert_eq!(
            error.code(),
            AnthropicErrorCode::MalformedResponse,
            "{name}"
        );
        assert!(!reducer.terminal_emitted(), "{name}");
    }
}

#[test]
fn web_search_fixture_maps_activity_sources_citations_and_opaque_fields() {
    let events = fixture_events(include_str!("fixtures/web_search.sse"));

    let started = events
        .iter()
        .find_map(ModelEvent::as_hosted_tool_started)
        .expect("hosted search must start");
    assert_eq!(started.provider_call_id().as_str(), "srvtoolu_tea_search");
    assert_eq!(started.tool_name(), "web_search");

    let completed = events
        .iter()
        .find_map(ModelEvent::as_hosted_tool_completed)
        .expect("hosted search must complete");
    assert_eq!(completed.arguments()["query"], "tea-rs hosted tools");
    assert!(matches!(completed.outcome(), HostedToolOutcome::Success));
    assert_eq!(completed.sources().len(), 2);
    assert_eq!(completed.sources()[0].url(), "https://example.com/guide");
    assert_eq!(completed.sources()[0].title(), Some("Tea guide"));
    let continuation = completed
        .continuation()
        .expect("encrypted search results are required for replay");
    assert_eq!(continuation.provider(), "anthropic");
    assert_eq!(continuation.format(), "anthropic.messages.web_search.v1");
    assert_eq!(
        continuation.payload()["content_blocks"][1]["content"][0]["encrypted_content"],
        "enc_guide"
    );

    let text = events
        .iter()
        .filter_map(ModelEvent::as_text_delta)
        .collect::<String>();
    assert_eq!(text, "Tea uses sources.");
    let citation = events
        .iter()
        .find_map(ModelEvent::as_source_citation)
        .expect("citation delta must be normalized");
    assert_eq!(
        citation.provider_call_id().unwrap().as_str(),
        "srvtoolu_tea_search"
    );
    assert_eq!(
        citation.citation().source().url(),
        "https://example.com/guide"
    );
    assert_eq!(citation.citation().cited_text(), Some("sources"));
    assert_eq!(
        citation.citation().continuation().unwrap().payload()["citation"]["encrypted_index"],
        "enc_index_guide"
    );
    assert!(matches!(
        events.last(),
        Some(ModelEvent::Completed(completion))
            if *completion.stop_reason() == StopReason::Completed
    ));
    assert_valid_stream(&events);
}

#[test]
fn web_search_error_inside_http_success_is_a_hosted_error() {
    let chunks = [
        serde_json::json!({
            "type":"content_block_start",
            "index":0,
            "content_block":{
                "type":"server_tool_use",
                "id":"srvtoolu_search_error",
                "name":"web_search",
                "input":{"query":"too much"}
            }
        }),
        serde_json::json!({"type":"content_block_stop","index":0}),
        serde_json::json!({
            "type":"content_block_start",
            "index":1,
            "content_block":{
                "type":"web_search_tool_result",
                "tool_use_id":"srvtoolu_search_error",
                "content":{
                    "type":"web_search_tool_result_error",
                    "error_code":"max_uses_exceeded"
                }
            }
        }),
        serde_json::json!({"type":"content_block_stop","index":1}),
        serde_json::json!({
            "type":"message_delta",
            "delta":{"stop_reason":"end_turn"},
            "usage":{"output_tokens":2}
        }),
        serde_json::json!({"type":"message_stop"}),
    ];
    let mut reducer = AnthropicReducer::new();
    let events = chunks
        .iter()
        .flat_map(|chunk| reducer.map_chunk(chunk).unwrap())
        .collect::<Vec<_>>();

    let completed = events
        .iter()
        .find_map(ModelEvent::as_hosted_tool_completed)
        .unwrap();
    let HostedToolOutcome::Error(error) = completed.outcome() else {
        panic!("provider tool error must not become a model transport failure");
    };
    assert_eq!(error.code(), "max_uses_exceeded");
    assert!(completed.sources().is_empty());
    assert_eq!(
        completed.continuation().unwrap().payload()["content_blocks"][1]["content"]["error_code"],
        "max_uses_exceeded"
    );
    assert_valid_stream(&events);
}

#[test]
fn pause_turn_fixture_preserves_nonterminal_stop_reason() {
    let events = fixture_events(include_str!("fixtures/web_search_pause_turn.sse"));

    assert!(
        events
            .iter()
            .any(|event| event.as_hosted_tool_completed().is_some())
    );
    assert!(matches!(
        events.last(),
        Some(ModelEvent::Completed(completion))
            if *completion.stop_reason() == StopReason::PauseTurn
    ));
    assert_valid_stream(&events);
}
