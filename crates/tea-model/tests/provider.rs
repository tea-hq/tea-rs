use std::str::FromStr;

use futures_util::{StreamExt, stream};
use tea_model::{
    BoxModelStream, ModelCancellation, ModelCapabilities, ModelCompletion, ModelDisplayName,
    ModelEvent, ModelFailureCode, ModelProvider, ModelRequest, ModelResponseInfo, ModelSpec,
    ProviderId, Utf8Delta,
};
use tea_protocol::{
    CanonicalMessage, ContentBlock, FinalOutputFormat, MessageId, ModelId, ProtocolTimestamp,
    RetryClass, StopReason, TokenCount,
};

#[derive(Debug)]
struct InMemoryProvider {
    provider_id: ProviderId,
    models: Vec<ModelSpec>,
}

impl ModelProvider for InMemoryProvider {
    fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }

    fn models(&self) -> &[ModelSpec] {
        &self.models
    }

    fn stream(&self, _request: ModelRequest, _cancellation: ModelCancellation) -> BoxModelStream {
        Box::pin(stream::iter([
            ModelEvent::Started(ModelResponseInfo::new()),
            ModelEvent::TextDelta(Utf8Delta::new("hello").unwrap()),
            ModelEvent::Completed(ModelCompletion::new(StopReason::Completed).unwrap()),
        ]))
    }
}

fn provider() -> InMemoryProvider {
    InMemoryProvider {
        provider_id: ProviderId::from_str("memory").unwrap(),
        models: vec![
            ModelSpec::new(
                ModelId::from_str("memory/test").unwrap(),
                ProviderId::from_str("memory").unwrap(),
                ModelDisplayName::from_str("Memory Test").unwrap(),
                TokenCount::new(8_000).unwrap(),
                TokenCount::new(2_000).unwrap(),
                ModelCapabilities::text(),
            )
            .unwrap(),
        ],
    }
}

fn request() -> ModelRequest {
    ModelRequest::new(
        ModelId::from_str("memory/test").unwrap(),
        vec![
            CanonicalMessage::user(
                MessageId::from_str("0195a0b1-5e3d-73de-b461-0aa7aa000004").unwrap(),
                vec![ContentBlock::text("hello").unwrap()],
                ProtocolTimestamp::from_str("2026-07-23T09:30:12.123Z").unwrap(),
            )
            .unwrap(),
        ],
    )
    .unwrap()
}

#[tokio::test(flavor = "current_thread")]
async fn provider_is_object_safe_and_returns_normalized_stream() {
    let provider: Box<dyn ModelProvider> = Box::new(provider());
    assert_eq!(provider.provider_id().as_str(), "memory");
    assert_eq!(provider.models().len(), 1);
    assert_eq!(
        provider
            .model(&ModelId::from_str("memory/test").unwrap())
            .unwrap()
            .display_name()
            .as_str(),
        "Memory Test"
    );
    assert!(
        provider
            .model(&ModelId::from_str("memory/missing").unwrap())
            .is_none()
    );

    let events = provider
        .stream(request(), ModelCancellation::new())
        .collect::<Vec<_>>()
        .await;
    assert_eq!(events.len(), 3);
    assert!(matches!(events.first(), Some(ModelEvent::Started(_))));
    assert!(matches!(events.last(), Some(ModelEvent::Completed(_))));
}

#[test]
fn provider_and_stream_contracts_are_send_sync_as_required() {
    fn assert_provider<T: ModelProvider>() {}
    fn assert_stream<T: tea_model::ModelStream>() {}
    assert_provider::<InMemoryProvider>();
    assert_stream::<futures_util::stream::Iter<std::array::IntoIter<ModelEvent, 0>>>();
}

#[test]
fn default_provider_preflight_checks_catalog_and_model_capabilities() {
    let provider = provider();

    provider.validate_request(&request()).unwrap();

    let missing_model = ModelRequest::new(
        ModelId::from_str("memory/missing").unwrap(),
        request().messages().to_vec(),
    )
    .unwrap();
    let missing = provider.validate_request(&missing_model).unwrap_err();
    assert_eq!(missing.code(), ModelFailureCode::InvalidRequest);
    assert_eq!(missing.retry(), RetryClass::Never);
    assert_eq!(
        missing.message(),
        "model request selects a model not advertised by this provider"
    );

    let unsupported = request().with_final_output_format(FinalOutputFormat::JsonObject);
    let unsupported = provider.validate_request(&unsupported).unwrap_err();
    assert_eq!(unsupported.code(), ModelFailureCode::InvalidRequest);
    assert_eq!(unsupported.retry(), RetryClass::Never);
    assert_eq!(
        unsupported.message(),
        "model does not support JSON-object final output"
    );
}
