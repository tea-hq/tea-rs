use std::str::FromStr;

use serde_json::json;
use tea_model::{
    HostedToolCompleted, HostedToolStarted, ModelCompletion, ModelEvent, ModelFailure,
    ModelFailureCode, ModelResponseInfo, ModelSourceCitation, ModelStreamIndex,
    ModelStreamValueError, ProviderResponseId, ProviderToolCallId, ToolArgumentsDelta,
    ToolCallCompleted, ToolCallStarted, Utf8Delta,
};
use tea_protocol::{
    CurrencyCode, DecimalAmount, ExactCost, ExternalSource, HostedToolOutcome, ModelId,
    ProtocolMetadata, ProviderContinuation, RetryClass, SourceCitation, StopReason, TokenCount,
    Usage,
};

fn metadata() -> ProtocolMetadata {
    ProtocolMetadata::from_entries([("com.example.provider".to_owned(), json!({"region":"test"}))])
        .unwrap()
}

#[test]
fn normalized_stream_events_cover_text_reasoning_and_tools() {
    let started = ModelEvent::Started(
        ModelResponseInfo::new()
            .with_response_id(ProviderResponseId::from_str("response_123").unwrap())
            .with_response_model(ModelId::from_str("provider/concrete-model").unwrap())
            .with_metadata(metadata()),
    );
    let text = ModelEvent::TextDelta(Utf8Delta::new("Hello ").unwrap());
    let thinking = ModelEvent::ThinkingDelta(Utf8Delta::new("Inspecting").unwrap());
    let call_id = ProviderToolCallId::from_str("call_123").unwrap();
    let index = ModelStreamIndex::new(0).unwrap();
    let tool_started = ModelEvent::ToolCallStarted(
        ToolCallStarted::new(index, call_id.clone(), "read_file").unwrap(),
    );
    let tool_delta = ModelEvent::ToolArgumentsDelta(
        ToolArgumentsDelta::new(index, call_id.clone(), r#"{"path":"/tmp"#).unwrap(),
    );
    let tool_completed = ModelEvent::ToolCallCompleted(
        ToolCallCompleted::new(
            index,
            call_id.clone(),
            "read_file",
            json!({"path":"/tmp/notes.txt"}),
        )
        .unwrap(),
    );

    assert!(matches!(started, ModelEvent::Started(_)));
    assert_eq!(text.as_text_delta(), Some("Hello "));
    assert_eq!(thinking.as_thinking_delta(), Some("Inspecting"));
    assert_eq!(
        tool_started
            .as_tool_call_started()
            .unwrap()
            .provider_call_id(),
        &call_id
    );
    assert_eq!(
        tool_delta.as_tool_arguments_delta().unwrap().delta(),
        r#"{"path":"/tmp"#
    );
    assert_eq!(
        tool_completed.as_tool_call_completed().unwrap().arguments(),
        &json!({"path":"/tmp/notes.txt"})
    );
}

#[test]
fn normalized_stream_events_cover_hosted_tools_and_citations() {
    let index = ModelStreamIndex::new(2).unwrap();
    let provider_call_id = ProviderToolCallId::from_str("ws_123").unwrap();
    let source = ExternalSource::new("https://example.com/result")
        .unwrap()
        .with_title("Result")
        .unwrap();
    let continuation = ProviderContinuation::new(
        "openai",
        "openai.responses.web_search.v1",
        json!({"type":"web_search_call","id":"ws_123"}),
    )
    .unwrap();
    let started = ModelEvent::HostedToolStarted(
        HostedToolStarted::new(index, provider_call_id.clone(), "web_search").unwrap(),
    );
    let completed = ModelEvent::HostedToolCompleted(
        HostedToolCompleted::new(
            index,
            provider_call_id.clone(),
            "web_search",
            json!({"query":"example"}),
            HostedToolOutcome::Success,
            vec![source.clone()],
            Some(continuation.clone()),
        )
        .unwrap(),
    );
    let citation = ModelEvent::SourceCitation(
        ModelSourceCitation::new(
            Some(provider_call_id.clone()),
            SourceCitation::new(source)
                .with_range(0, 6)
                .unwrap()
                .with_continuation(continuation),
        )
        .unwrap(),
    );

    assert_eq!(
        started.as_hosted_tool_started().unwrap().provider_call_id(),
        &provider_call_id
    );
    assert_eq!(
        completed.as_hosted_tool_completed().unwrap().sources()[0].title(),
        Some("Result")
    );
    assert_eq!(
        citation.as_source_citation().unwrap().provider_call_id(),
        Some(&provider_call_id)
    );
}

#[test]
fn completed_event_normalizes_stop_usage_and_exact_cost() {
    let usage = Usage::new(TokenCount::new(10).unwrap(), TokenCount::new(4).unwrap());
    let cost = ExactCost::new(
        DecimalAmount::from_str("0.00014").unwrap(),
        CurrencyCode::from_str("USD").unwrap(),
    );
    let completion = ModelCompletion::new(StopReason::Completed)
        .unwrap()
        .with_usage(usage.clone())
        .with_cost(cost.clone())
        .with_metadata(metadata());
    let event = ModelEvent::Completed(completion.clone());

    assert_eq!(completion.stop_reason(), &StopReason::Completed);
    assert_eq!(completion.usage(), Some(&usage));
    assert_eq!(completion.cost(), Some(&cost));
    assert_eq!(completion.metadata(), &metadata());
    assert!(matches!(event, ModelEvent::Completed(_)));

    let refusal = ModelCompletion::new(StopReason::Refusal).unwrap();
    assert_eq!(refusal.stop_reason(), &StopReason::Refusal);
    let future_reason = StopReason::Unknown("future_provider_reason".to_owned());
    let unknown = ModelCompletion::new(future_reason.clone()).unwrap();
    assert_eq!(unknown.stop_reason(), &future_reason);
    assert!(!unknown.stop_reason().is_success());

    assert_eq!(
        ModelCompletion::new(StopReason::Cancelled).unwrap_err(),
        ModelStreamValueError::InvalidCompletionReason
    );
    assert_eq!(
        ModelCompletion::new(StopReason::Error).unwrap_err(),
        ModelStreamValueError::InvalidCompletionReason
    );
}

#[test]
fn provider_failures_are_typed_safe_terminal_values() {
    let failure = ModelFailure::new(
        ModelFailureCode::RateLimited,
        "provider rate limit exceeded",
        RetryClass::AfterBackoff,
    )
    .unwrap()
    .with_metadata(metadata());

    assert_eq!(failure.code(), ModelFailureCode::RateLimited);
    assert_eq!(failure.message(), "provider rate limit exceeded");
    assert_eq!(failure.retry(), RetryClass::AfterBackoff);
    assert_eq!(failure.metadata(), &metadata());
    assert!(matches!(ModelEvent::Failed(failure), ModelEvent::Failed(_)));

    assert_eq!(
        ModelFailure::new(ModelFailureCode::Internal, "", RetryClass::Never).unwrap_err(),
        ModelStreamValueError::InvalidFailureMessage
    );
    assert_eq!(
        ModelFailure::new(
            ModelFailureCode::Internal,
            "bad\0message",
            RetryClass::Never
        )
        .unwrap_err(),
        ModelStreamValueError::InvalidFailureMessage
    );
}

#[test]
fn identifiers_deltas_indexes_and_tool_arguments_are_bounded() {
    assert!(ProviderResponseId::from_str("").is_err());
    assert!(ProviderResponseId::from_str("bad\nid").is_err());
    assert!(ProviderToolCallId::from_str(&"x".repeat(257)).is_err());
    assert_eq!(
        ModelStreamIndex::new(1024).unwrap_err(),
        ModelStreamValueError::InvalidStreamIndex
    );
    assert_eq!(
        Utf8Delta::new("").unwrap_err(),
        ModelStreamValueError::InvalidDelta
    );
    assert_eq!(
        Utf8Delta::new("bad\0delta").unwrap_err(),
        ModelStreamValueError::InvalidDelta
    );

    let index = ModelStreamIndex::new(1).unwrap();
    let call_id = ProviderToolCallId::from_str("call_1").unwrap();
    assert_eq!(
        ToolCallCompleted::new(index, call_id.clone(), "read_file", json!("not-object"))
            .unwrap_err(),
        ModelStreamValueError::ToolArgumentsMustBeObject
    );
    assert_eq!(
        ToolCallStarted::new(index, call_id, "Bad Tool").unwrap_err(),
        ModelStreamValueError::InvalidToolName
    );
}

#[test]
fn stream_values_reject_oversized_diagnostics_deltas_and_arguments() {
    assert_eq!(
        Utf8Delta::new("x".repeat(64 * 1024 + 1)).unwrap_err(),
        ModelStreamValueError::InvalidDelta
    );
    assert_eq!(
        ModelFailure::new(
            ModelFailureCode::Internal,
            "x".repeat(4097),
            RetryClass::Never,
        )
        .unwrap_err(),
        ModelStreamValueError::InvalidFailureMessage
    );

    let mut nested = json!({});
    for _ in 0..40 {
        nested = json!({"next": nested});
    }
    assert_eq!(
        ToolCallCompleted::new(
            ModelStreamIndex::new(0).unwrap(),
            ProviderToolCallId::from_str("call_deep").unwrap(),
            "read_file",
            nested,
        )
        .unwrap_err(),
        ModelStreamValueError::ToolArgumentsOutOfBounds
    );
}

#[test]
fn all_failure_codes_are_explicit() {
    assert_eq!(ModelFailureCode::ALL.len(), 10);
    assert!(ModelFailureCode::ALL.contains(&ModelFailureCode::InvalidRequest));
    assert!(ModelFailureCode::ALL.contains(&ModelFailureCode::ContextOverflow));
    assert!(ModelFailureCode::ALL.contains(&ModelFailureCode::Authentication));
    assert!(ModelFailureCode::ALL.contains(&ModelFailureCode::PermissionDenied));
    assert!(ModelFailureCode::ALL.contains(&ModelFailureCode::RateLimited));
    assert!(ModelFailureCode::ALL.contains(&ModelFailureCode::Unavailable));
    assert!(ModelFailureCode::ALL.contains(&ModelFailureCode::Transport));
    assert!(ModelFailureCode::ALL.contains(&ModelFailureCode::MalformedResponse));
    assert!(ModelFailureCode::ALL.contains(&ModelFailureCode::Cancelled));
    assert!(ModelFailureCode::ALL.contains(&ModelFailureCode::Internal));
}
