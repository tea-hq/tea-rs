//! Live Anthropic Messages `ModelProvider` backed by `reqwest`.

use std::sync::Arc;
use std::time::{Duration, SystemTime};

use futures_util::StreamExt;
use tea_control::CancellationScope;
use tea_model::{
    ModelEvent, ModelFailure, ModelFailureCode, ModelProvider, ModelRequest, ModelResponseInfo,
    ModelSpec, ProviderId,
};
use tea_protocol::RetryClass;
use tea_provider_http::{
    ProviderHttpConfig, UserAgent, read_bounded_error_body, retry_after_delay,
};

use crate::catalog::default_catalog;
use crate::credential::{AnthropicConfig, CredentialResolver, EnvCredentialResolver};
use crate::error::{AnthropicError, AnthropicErrorCode};
use crate::request::{build_messages_body, messages_url, request_headers};
use crate::sse::{SseEvent, SseParser};
use crate::stream::{AnthropicReducer, map_http_failure};

/// Anthropic Messages streaming provider adapter.
#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    config: Arc<AnthropicConfig>,
    client: reqwest::Client,
    catalog: Vec<ModelSpec>,
}

impl AnthropicProvider {
    /// Creates a provider from an immutable connection config and model catalog.
    ///
    /// # Errors
    ///
    /// Returns an error when the HTTP client cannot be constructed.
    pub fn new(
        config: Arc<AnthropicConfig>,
        catalog: Vec<ModelSpec>,
    ) -> Result<Self, AnthropicError> {
        Self::new_with_http_config(config, catalog, &ProviderHttpConfig::new())
    }

    fn new_with_http_config(
        config: Arc<AnthropicConfig>,
        catalog: Vec<ModelSpec>,
        http_config: &ProviderHttpConfig,
    ) -> Result<Self, AnthropicError> {
        validate_catalog_structured_output(&config, &catalog)?;
        let client = http_config
            .build_client(Duration::from_millis(config.timeout_millis()))
            .map_err(|_| AnthropicError::new(AnthropicErrorCode::Transport, "client failed"))?;
        Ok(Self {
            config,
            client,
            catalog,
        })
    }

    /// Returns the connection configuration.
    #[must_use]
    pub fn config(&self) -> &AnthropicConfig {
        &self.config
    }

    fn validated_request_body(
        &self,
        request: &ModelRequest,
    ) -> Result<serde_json::Value, ModelFailure> {
        let model = self.model(request.model_id()).ok_or_else(|| {
            invalid_request("model request selects a model not advertised by this provider")
        })?;
        request
            .validate_for(model)
            .map_err(|error| invalid_request(error.to_string()))?;
        build_messages_body(request, &self.config).map_err(|error| adapter_failure(&error))
    }
}

impl ModelProvider for AnthropicProvider {
    fn provider_id(&self) -> &ProviderId {
        self.config.provider_id()
    }

    fn models(&self) -> &[ModelSpec] {
        &self.catalog
    }

    fn validate_request(&self, request: &ModelRequest) -> Result<(), ModelFailure> {
        self.validated_request_body(request).map(|_| ())
    }

    #[allow(clippy::too_many_lines)]
    fn stream(
        &self,
        request: ModelRequest,
        cancellation: CancellationScope,
    ) -> tea_model::BoxModelStream {
        let (config, client) = (Arc::clone(&self.config), self.client.clone());
        let body_result = self.validated_request_body(&request);
        Box::pin(async_stream::stream! {
            let mut started_emitted = false;
            let body = match body_result {
                Ok(body) => body,
                Err(failure) => {
                    yield ModelEvent::Started(ModelResponseInfo::new());
                    yield ModelEvent::Failed(failure);
                    return;
                }
            };
            let mut request_builder = client.post(messages_url(&config)).json(&body);
            for (header, value) in request_headers(&config) {
                request_builder = request_builder.header(header, value);
            }
            let response = tokio::select! {
                biased;
                () = cancellation.cancelled() => {
                    yield ModelEvent::Started(ModelResponseInfo::new());
                    yield ModelEvent::Failed(ModelFailure::new(
                        ModelFailureCode::Cancelled,
                        "model request was cancelled",
                        RetryClass::Never,
                    ).unwrap_or_else(|_| ModelFailure::internal_adapter_failure()));
                    return;
                }
                response = request_builder.send() => match response {
                    Ok(response) => response,
                    Err(error) => {
                        yield ModelEvent::Started(ModelResponseInfo::new());
                        yield ModelEvent::Failed(transport_failure(&error));
                        return;
                    }
                },
            };
            if !response.status().is_success() {
                let status = response.status().as_u16();
                let retry_after = retry_after_delay(response.headers(), SystemTime::now());
                let body = read_bounded_error_body(response).await;
                let mut failure = map_http_failure(status, &body);
                if let Some(retry_after) = retry_after {
                    failure = failure.with_retry_after(retry_after);
                }
                yield ModelEvent::Started(ModelResponseInfo::new());
                yield ModelEvent::Failed(failure);
                return;
            }
            let mut bytes = response.bytes_stream();
            let mut parser = SseParser::new();
            let mut reducer = AnthropicReducer::new();
            let mut pending_completion = None;
            loop {
                let chunk = tokio::select! {
                    biased;
                    () = cancellation.cancelled() => {
                        if !started_emitted {
                            yield ModelEvent::Started(ModelResponseInfo::new());
                        }
                        yield ModelEvent::Failed(ModelFailure::new(
                            ModelFailureCode::Cancelled,
                            "model request was cancelled",
                            RetryClass::Never,
                        ).unwrap_or_else(|_| ModelFailure::internal_adapter_failure()));
                        return;
                    }
                    chunk = bytes.next() => match chunk {
                        Some(Ok(bytes)) => bytes,
                        None => break,
                        Some(Err(error)) => {
                            if !started_emitted {
                                yield ModelEvent::Started(ModelResponseInfo::new());
                            }
                            yield ModelEvent::Failed(transport_failure(&error));
                            return;
                        }
                    },
                };
                for event in parser.feed(&chunk) {
                    match map_sse_event(&mut reducer, event) {
                        Ok(events) => {
                            for event in events {
                                match event {
                                    ModelEvent::Started(info) => {
                                        started_emitted = true;
                                        yield ModelEvent::Started(info);
                                    }
                                    ModelEvent::Completed(completion) => {
                                        pending_completion = Some(completion);
                                    }
                                    ModelEvent::Failed(failure) => {
                                        if !started_emitted {
                                            yield ModelEvent::Started(ModelResponseInfo::new());
                                        }
                                        yield ModelEvent::Failed(failure);
                                        return;
                                    }
                                    event => yield event,
                                }
                            }
                        }
                        Err(error) => {
                            if !started_emitted {
                                yield ModelEvent::Started(ModelResponseInfo::new());
                            }
                            yield ModelEvent::Failed(adapter_failure(&error));
                            return;
                        }
                    }
                }
            }
            for event in parser.finish() {
                    match map_sse_event(&mut reducer, event) {
                    Ok(events) => {
                        for event in events {
                            match event {
                                ModelEvent::Started(info) => {
                                    started_emitted = true;
                                    yield ModelEvent::Started(info);
                                }
                                ModelEvent::Completed(completion) => {
                                    pending_completion = Some(completion);
                                }
                                ModelEvent::Failed(failure) => {
                                    if !started_emitted {
                                        yield ModelEvent::Started(ModelResponseInfo::new());
                                    }
                                    yield ModelEvent::Failed(failure);
                                    return;
                                }
                                event => yield event,
                            }
                        }
                    }
                    Err(error) => {
                        if !started_emitted {
                            yield ModelEvent::Started(ModelResponseInfo::new());
                        }
                        yield ModelEvent::Failed(adapter_failure(&error));
                        return;
                    }
                }
            }
            match reducer.finish() {
                Ok(events) => {
                    for event in events {
                        match event {
                            ModelEvent::Started(info) => {
                                started_emitted = true;
                                yield ModelEvent::Started(info);
                            }
                            ModelEvent::Completed(completion) => {
                                pending_completion = Some(completion);
                            }
                            ModelEvent::Failed(failure) => {
                                if !started_emitted {
                                    yield ModelEvent::Started(ModelResponseInfo::new());
                                }
                                yield ModelEvent::Failed(failure);
                                return;
                            }
                            event => yield event,
                        }
                    }
                }
                Err(error) => {
                    if !started_emitted {
                        yield ModelEvent::Started(ModelResponseInfo::new());
                    }
                    yield ModelEvent::Failed(adapter_failure(&error));
                    return;
                }
            }
            if let Some(completion) = pending_completion {
                if !started_emitted {
                    yield ModelEvent::Started(ModelResponseInfo::new());
                }
                yield ModelEvent::Completed(completion);
            }
        })
    }
}

fn validate_catalog_structured_output(
    config: &AnthropicConfig,
    catalog: &[ModelSpec],
) -> Result<(), AnthropicError> {
    for model in catalog {
        let capabilities = model.capabilities();
        if capabilities.supports_final_json_object() {
            return Err(AnthropicError::new(
                AnthropicErrorCode::InvalidRequest,
                "Anthropic model catalog cannot advertise final JSON Object output",
            ));
        }
        if capabilities.supports_final_json_schema() && !config.supports_final_json_schema() {
            return Err(AnthropicError::new(
                AnthropicErrorCode::InvalidRequest,
                "Anthropic model catalog cannot advertise final JSON Schema output without endpoint opt-in",
            ));
        }
    }
    Ok(())
}

/// Builder for [`AnthropicProvider`] backed by the environment contract.
#[derive(Debug, Default)]
pub struct AnthropicProviderBuilder {
    config: Option<Arc<AnthropicConfig>>,
    catalog: Option<Vec<ModelSpec>>,
    resolver: Option<Arc<dyn CredentialResolver>>,
    http_config: ProviderHttpConfig,
}

impl AnthropicProviderBuilder {
    /// Creates an empty builder that resolves config from the environment.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a builder pre-populated from the `TEA_ANTHROPIC_*` environment contract.
    ///
    /// # Errors
    ///
    /// Returns an error when required environment values are missing or invalid.
    pub fn from_env() -> Result<Self, AnthropicError> {
        let config = Arc::new(EnvCredentialResolver::new().resolve()?);
        Ok(Self {
            config: Some(config),
            catalog: None,
            resolver: None,
            http_config: ProviderHttpConfig::new(),
        })
    }

    /// Overrides the connection configuration.
    #[must_use]
    pub fn with_config(mut self, config: Arc<AnthropicConfig>) -> Self {
        self.config = Some(config);
        self
    }

    /// Overrides the credential resolver.
    #[must_use]
    pub fn with_resolver(mut self, resolver: Arc<dyn CredentialResolver>) -> Self {
        self.resolver = Some(resolver);
        self
    }

    /// Overrides the advertised model catalog.
    #[must_use]
    pub fn with_catalog(mut self, catalog: Vec<ModelSpec>) -> Self {
        self.catalog = Some(catalog);
        self
    }

    /// Sets the shared HTTP client policy used by model requests.
    #[must_use]
    pub fn with_http_config(mut self, http_config: ProviderHttpConfig) -> Self {
        self.http_config = http_config;
        self
    }

    /// Sets the application identity sent with model requests.
    #[must_use]
    pub fn with_user_agent(mut self, user_agent: UserAgent) -> Self {
        self.http_config = self.http_config.with_user_agent(user_agent);
        self
    }

    /// Builds the provider.
    ///
    /// # Errors
    ///
    /// Returns an error when config resolution, catalog construction, or client
    /// construction fails.
    pub fn build(self) -> Result<AnthropicProvider, AnthropicError> {
        let config = if let Some(config) = self.config {
            config
        } else {
            let resolver = self
                .resolver
                .unwrap_or_else(|| Arc::new(EnvCredentialResolver::new()));
            Arc::new(resolver.resolve()?)
        };
        let catalog = self.catalog.map_or_else(|| default_catalog(&config), Ok)?;
        AnthropicProvider::new_with_http_config(config, catalog, &self.http_config)
    }
}

fn map_sse_event(
    reducer: &mut AnthropicReducer,
    event: SseEvent,
) -> Result<Vec<ModelEvent>, AnthropicError> {
    match event {
        SseEvent::Data(payload) => serde_json::from_str(&payload)
            .map_err(|_| {
                AnthropicError::new(AnthropicErrorCode::MalformedResponse, "invalid sse json")
            })
            .and_then(|value| reducer.map_chunk(&value)),
        SseEvent::Done => reducer.finish(),
    }
}

fn adapter_failure(error: &AnthropicError) -> ModelFailure {
    ModelFailure::new(
        error.code().into_model_failure_code(),
        error.message(),
        RetryClass::Never,
    )
    .unwrap_or_else(|_| ModelFailure::internal_adapter_failure())
}

fn invalid_request(message: impl Into<String>) -> ModelFailure {
    ModelFailure::new(ModelFailureCode::InvalidRequest, message, RetryClass::Never)
        .unwrap_or_else(|_| ModelFailure::internal_adapter_failure())
}

fn transport_failure(error: &reqwest::Error) -> ModelFailure {
    let code = if error.is_timeout() || error.is_connect() || error.is_request() {
        ModelFailureCode::Transport
    } else {
        ModelFailureCode::Internal
    };
    ModelFailure::new(code, "anthropic transport error", RetryClass::Immediate)
        .unwrap_or_else(|_| ModelFailure::internal_adapter_failure())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::str::FromStr;
    use std::sync::atomic::{AtomicBool, Ordering};

    use tea_model::{ModelCapabilities, ModelDisplayName, ModelToolDefinition};
    use tea_protocol::{
        CanonicalMessage, ContentBlock, ExternalSource, FinalOutputFormat, MessageId, ModelId,
        ProtocolTimestamp, SourceCitation, StopReason, TokenCount,
    };
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
    use tokio::net::TcpListener;

    use super::*;
    use crate::credential::MapCredentialResolver;

    fn provider(user_agent: Option<UserAgent>) -> AnthropicProvider {
        let config = MapCredentialResolver::new(BTreeMap::from([
            ("TEA_ANTHROPIC_API_KEY".to_owned(), "sk-ant-test".to_owned()),
            (
                "TEA_ANTHROPIC_MODEL".to_owned(),
                "claude-sonnet-4-20250514".to_owned(),
            ),
        ]))
        .resolve()
        .unwrap();
        let mut builder = AnthropicProviderBuilder::new().with_config(Arc::new(config));
        if let Some(user_agent) = user_agent {
            builder = builder.with_user_agent(user_agent);
        }
        builder.build().unwrap()
    }

    fn provider_for_base_url(base_url: String) -> AnthropicProvider {
        let config = MapCredentialResolver::new(BTreeMap::from([
            (
                "TEA_ANTHROPIC_API_KEY".to_owned(),
                "sk-ant-test-key".to_owned(),
            ),
            (
                "TEA_ANTHROPIC_MODEL".to_owned(),
                "claude-sonnet-4-20250514".to_owned(),
            ),
            ("TEA_ANTHROPIC_BASE_URL".to_owned(), base_url),
        ]))
        .resolve()
        .unwrap();
        AnthropicProviderBuilder::new()
            .with_config(Arc::new(config))
            .build()
            .unwrap()
    }

    fn text_request() -> ModelRequest {
        let message = CanonicalMessage::user(
            MessageId::from_str("0195a0b1-5e52-74b2-8c25-0aa7aa000045").unwrap(),
            vec![ContentBlock::text("hello").unwrap()],
            ProtocolTimestamp::from_str("2026-07-23T09:30:12.125Z").unwrap(),
        )
        .unwrap();
        ModelRequest::new(
            ModelId::from_str("claude-sonnet-4-20250514").unwrap(),
            vec![message],
        )
        .unwrap()
    }

    fn assert_conformant_failure(events: &[ModelEvent], code: ModelFailureCode) {
        assert!(matches!(events.first(), Some(ModelEvent::Started(_))));
        assert!(matches!(
            events.last(),
            Some(ModelEvent::Failed(failure)) if failure.code() == code
        ));
        let mut validator = tea_model::ModelStreamValidator::new();
        for event in events {
            validator.observe(event).unwrap();
        }
        assert!(!validator.finish().unwrap().succeeded());
    }

    fn structured_output_config(enabled: bool) -> Arc<AnthropicConfig> {
        let mut values = BTreeMap::from([
            ("TEA_ANTHROPIC_API_KEY".to_owned(), "sk-ant-test".to_owned()),
            (
                "TEA_ANTHROPIC_MODEL".to_owned(),
                "claude-sonnet-4-20250514".to_owned(),
            ),
        ]);
        if enabled {
            values.insert(
                "TEA_ANTHROPIC_FINAL_JSON_SCHEMA".to_owned(),
                "true".to_owned(),
            );
        }
        Arc::new(MapCredentialResolver::new(values).resolve().unwrap())
    }

    fn model_with_capabilities(
        config: &AnthropicConfig,
        capabilities: ModelCapabilities,
    ) -> ModelSpec {
        ModelSpec::new(
            config.model_id().clone(),
            config.provider_id().clone(),
            ModelDisplayName::from_str("Claude test").unwrap(),
            TokenCount::new(128_000).unwrap(),
            TokenCount::new(4_000).unwrap(),
            capabilities,
        )
        .unwrap()
    }

    #[test]
    fn custom_catalog_rejects_json_object_capability() {
        let config = structured_output_config(true);
        let model =
            model_with_capabilities(&config, ModelCapabilities::text().with_final_json_object());

        let error = AnthropicProviderBuilder::new()
            .with_config(config)
            .with_catalog(vec![model])
            .build()
            .unwrap_err();

        assert_eq!(error.code(), AnthropicErrorCode::InvalidRequest);
        assert_eq!(
            error.message(),
            "Anthropic model catalog cannot advertise final JSON Object output"
        );
    }

    #[test]
    fn custom_catalog_rejects_schema_capability_without_endpoint_opt_in() {
        let config = structured_output_config(false);
        let model =
            model_with_capabilities(&config, ModelCapabilities::text().with_final_json_schema());

        let error = AnthropicProviderBuilder::new()
            .with_config(config)
            .with_catalog(vec![model])
            .build()
            .unwrap_err();

        assert_eq!(error.code(), AnthropicErrorCode::InvalidRequest);
        assert_eq!(
            error.message(),
            "Anthropic model catalog cannot advertise final JSON Schema output without endpoint opt-in"
        );
    }

    #[test]
    fn custom_catalog_may_keep_schema_disabled_after_endpoint_opt_in() {
        let config = structured_output_config(true);
        let model = model_with_capabilities(&config, ModelCapabilities::text());

        let provider = AnthropicProviderBuilder::new()
            .with_config(config)
            .with_catalog(vec![model])
            .build()
            .unwrap();

        assert!(
            !provider.models()[0]
                .capabilities()
                .supports_final_json_schema()
        );
    }

    #[test]
    fn provider_preflight_rejects_schema_with_historical_citation() {
        let config = structured_output_config(true);
        let model =
            model_with_capabilities(&config, ModelCapabilities::text().with_final_json_schema());
        let provider = AnthropicProviderBuilder::new()
            .with_config(config)
            .with_catalog(vec![model])
            .build()
            .unwrap();
        let citation = SourceCitation::new(
            ExternalSource::new("https://example.com/structured-output").unwrap(),
        );
        let assistant = CanonicalMessage::assistant(
            MessageId::from_str("0195a0b1-5e53-74b2-8c25-0aa7aa000098").unwrap(),
            vec![ContentBlock::citation(citation)],
            StopReason::Completed,
            ProtocolTimestamp::from_str("2026-07-23T09:30:12.124Z").unwrap(),
        )
        .unwrap();
        let request = ModelRequest::new(
            ModelId::from_str("claude-sonnet-4-20250514").unwrap(),
            vec![assistant, text_request().messages()[0].clone()],
        )
        .unwrap()
        .with_final_output_format(FinalOutputFormat::JsonSchema {
            schema: serde_json::json!({"type": "object"}),
        });

        let failure = provider.validate_request(&request).unwrap_err();

        assert_eq!(failure.code(), ModelFailureCode::InvalidRequest);
        assert_eq!(
            failure.message(),
            "Anthropic final JSON Schema output does not support citation content"
        );
    }

    async fn events_from_sse(sse: &str) -> Vec<ModelEvent> {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let sse = sse.to_owned();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            let _ = stream.read(&mut request).await.unwrap();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{sse}",
                sse.len()
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        });
        let provider = provider_for_base_url(format!("http://{address}"));
        let events = provider
            .stream(text_request(), CancellationScope::new())
            .collect::<Vec<_>>()
            .await;
        server.await.unwrap();
        events
    }

    async fn events_from_split_sse(first: &str, second: &str) -> Vec<ModelEvent> {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let first = first.to_owned();
        let second = second.to_owned();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            let _ = stream.read(&mut request).await.unwrap();
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
                )
                .await
                .unwrap();
            stream
                .write_all(format!("{:X}\r\n{first}\r\n", first.len()).as_bytes())
                .await
                .unwrap();
            release_rx.await.unwrap();
            stream
                .write_all(format!("{:X}\r\n{second}\r\n0\r\n\r\n", second.len()).as_bytes())
                .await
                .unwrap();
        });
        let provider = provider_for_base_url(format!("http://{address}"));
        let mut stream = provider.stream(text_request(), CancellationScope::new());
        let first_event = stream.next().await.unwrap();
        release_tx.send(()).unwrap();
        let mut events = vec![first_event];
        events.extend(stream.collect::<Vec<_>>().await);
        server.await.unwrap();
        events
    }

    #[tokio::test]
    async fn cancellation_interrupts_waiting_for_response_headers() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (request_seen_tx, request_seen_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            let _ = stream.read(&mut request).await.unwrap();
            request_seen_tx.send(()).unwrap();
            std::future::pending::<()>().await;
        });
        let provider = provider_for_base_url(format!("http://{address}"));
        let cancellation = CancellationScope::new();
        let stream = provider.stream(text_request(), cancellation.clone());
        let collector = tokio::spawn(async move { stream.collect::<Vec<_>>().await });

        tokio::time::timeout(Duration::from_secs(1), request_seen_rx)
            .await
            .expect("provider did not send request headers")
            .unwrap();
        cancellation.cancel();
        let events = tokio::time::timeout(Duration::from_secs(1), collector)
            .await
            .expect("cancellation did not interrupt response-header wait")
            .unwrap();
        server.abort();
        let _ = server.await;

        assert_conformant_failure(&events, ModelFailureCode::Cancelled);
    }

    async fn captured_headers(provider: &AnthropicProvider) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 1024];
            loop {
                let read = stream.read(&mut buffer).await.unwrap();
                assert_ne!(read, 0, "request closed before headers were sent");
                request.extend_from_slice(&buffer[..read]);
                if request.windows(4).any(|window| window == b"\r\n\r\n") {
                    break;
                }
            }
            stream
                .write_all(
                    b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .await
                .unwrap();
            String::from_utf8(request).unwrap()
        });
        let response = provider
            .client
            .post(format!("http://{address}/v1/messages"))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::NO_CONTENT);
        server.await.unwrap()
    }

    #[tokio::test]
    async fn client_sends_configured_user_agent_and_omits_it_by_default() {
        let headers =
            captured_headers(&provider(Some(UserAgent::new("tea-cli/0.1.0").unwrap()))).await;
        assert!(
            headers
                .lines()
                .any(|header| header.eq_ignore_ascii_case("user-agent: tea-cli/0.1.0"))
        );

        let headers = captured_headers(&provider(None)).await;
        assert!(
            !headers
                .lines()
                .any(|header| header.to_ascii_lowercase().starts_with("user-agent:"))
        );
    }

    #[tokio::test]
    async fn unadvertised_structured_output_combinations_fail_before_transport() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let http_seen = Arc::new(AtomicBool::new(false));
        let server_seen = Arc::clone(&http_seen);
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            server_seen.store(true, Ordering::SeqCst);
            let mut request = [0_u8; 4096];
            let _ = stream.read(&mut request).await.unwrap();
            stream
                .write_all(
                    b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .await
                .unwrap();
        });
        let model_id = ModelId::from_str("claude-sonnet-4-20250514").unwrap();
        let config = Arc::new(
            MapCredentialResolver::new(BTreeMap::from([
                ("TEA_ANTHROPIC_API_KEY".to_owned(), "sk-ant-test".to_owned()),
                (
                    "TEA_ANTHROPIC_MODEL".to_owned(),
                    model_id.as_str().to_owned(),
                ),
                (
                    "TEA_ANTHROPIC_BASE_URL".to_owned(),
                    format!("http://{address}"),
                ),
                (
                    "TEA_ANTHROPIC_FINAL_JSON_SCHEMA".to_owned(),
                    "true".to_owned(),
                ),
            ]))
            .resolve()
            .unwrap(),
        );
        let model = model_with_capabilities(&config, ModelCapabilities::text());
        let provider = AnthropicProviderBuilder::new()
            .with_config(Arc::clone(&config))
            .with_catalog(vec![model])
            .build()
            .unwrap();
        let request = text_request().with_final_output_format(FinalOutputFormat::JsonSchema {
            schema: serde_json::json!({"type": "object"}),
        });

        assert_eq!(
            provider.validate_request(&request).unwrap_err().code(),
            ModelFailureCode::InvalidRequest
        );

        let events = provider
            .stream(request, CancellationScope::new())
            .collect::<Vec<_>>()
            .await;
        assert_conformant_failure(&events, ModelFailureCode::InvalidRequest);
        assert!(!http_seen.load(Ordering::SeqCst));

        let model = model_with_capabilities(
            &config,
            ModelCapabilities::text()
                .with_tools(true)
                .with_final_json_schema(),
        );
        let provider = AnthropicProviderBuilder::new()
            .with_config(config)
            .with_catalog(vec![model])
            .build()
            .unwrap();
        let tool = ModelToolDefinition::new(
            "lookup",
            "Looks up a value.",
            serde_json::json!({"type": "object"}),
        )
        .unwrap();
        let request = text_request()
            .with_tools(vec![tool], false)
            .unwrap()
            .with_final_output_format(FinalOutputFormat::JsonSchema {
                schema: serde_json::json!({"type": "object"}),
            });

        assert_eq!(
            provider.validate_request(&request).unwrap_err().code(),
            ModelFailureCode::InvalidRequest
        );

        let events = provider
            .stream(request, CancellationScope::new())
            .collect::<Vec<_>>()
            .await;
        assert_conformant_failure(&events, ModelFailureCode::InvalidRequest);
        assert!(!http_seen.load(Ordering::SeqCst));

        server.abort();
        let _ = server.await;
    }

    #[tokio::test]
    async fn http_failure_propagates_retry_after_hint() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0_u8; 4096];
            let _ = stream.read(&mut request).await.unwrap();
            let body = r#"{"type":"error","error":{"type":"overloaded_error","message":"busy"}}"#;
            let response = format!(
                "HTTP/1.1 503 Service Unavailable\r\nRetry-After: 11\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        });
        let config = MapCredentialResolver::new(BTreeMap::from([
            (
                "TEA_ANTHROPIC_API_KEY".to_owned(),
                "sk-ant-test-key".to_owned(),
            ),
            (
                "TEA_ANTHROPIC_MODEL".to_owned(),
                "claude-sonnet-4-20250514".to_owned(),
            ),
            (
                "TEA_ANTHROPIC_BASE_URL".to_owned(),
                format!("http://{address}"),
            ),
        ]))
        .resolve()
        .unwrap();
        let provider = AnthropicProviderBuilder::new()
            .with_config(Arc::new(config))
            .build()
            .unwrap();
        let message = CanonicalMessage::user(
            MessageId::from_str("0195a0b1-5e52-74b2-8c25-0aa7aa000045").unwrap(),
            vec![ContentBlock::text("hello").unwrap()],
            ProtocolTimestamp::from_str("2026-07-23T09:30:12.125Z").unwrap(),
        )
        .unwrap();
        let request = ModelRequest::new(
            ModelId::from_str("claude-sonnet-4-20250514").unwrap(),
            vec![message],
        )
        .unwrap();
        let events = provider
            .stream(request, CancellationScope::new())
            .collect::<Vec<_>>()
            .await;
        let failure = events
            .iter()
            .find_map(|event| match event {
                ModelEvent::Failed(failure) => Some(failure),
                _ => None,
            })
            .unwrap();
        assert_eq!(failure.retry_after(), Some(Duration::from_secs(11)));
        assert_conformant_failure(&events, ModelFailureCode::Unavailable);
        server.await.unwrap();
    }

    #[tokio::test]
    async fn malformed_first_sse_event_is_a_conformant_failed_stream() {
        let events = events_from_sse("data: not-json\n\n").await;

        assert_conformant_failure(&events, ModelFailureCode::MalformedResponse);
    }

    #[tokio::test]
    async fn truncated_sse_stream_is_a_conformant_failed_stream() {
        let events = events_from_sse(concat!(
            "data: {\"type\":\"message_start\",\"message\":{",
            "\"id\":\"msg_truncated\",\"model\":\"claude-sonnet-4-20250514\",",
            "\"usage\":{\"input_tokens\":1}}}\n\n",
        ))
        .await;

        assert_conformant_failure(&events, ModelFailureCode::MalformedResponse);
    }

    #[tokio::test]
    async fn late_event_in_terminal_parser_batch_replaces_success_with_failure() {
        let events = events_from_sse(concat!(
            "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_late_batch\"}}\n\n",
            "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
            "data: {\"type\":\"content_block_start\",\"index\":0,",
            "\"content_block\":{\"type\":\"text\",\"text\":\"late\"}}\n\n",
        ))
        .await;

        assert!(
            !events
                .iter()
                .any(|event| matches!(event, ModelEvent::Completed(_)))
        );
        assert_conformant_failure(&events, ModelFailureCode::MalformedResponse);
    }

    #[tokio::test]
    async fn late_event_in_later_http_chunk_replaces_success_with_failure() {
        let events = events_from_split_sse(
            concat!(
                "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_late_chunk\"}}\n\n",
                "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
                "data: {\"type\":\"message_stop\"}\n\n",
            ),
            concat!(
                "data: {\"type\":\"content_block_start\",\"index\":0,",
                "\"content_block\":{\"type\":\"text\",\"text\":\"late\"}}\n\n",
            ),
        )
        .await;

        assert!(
            !events
                .iter()
                .any(|event| matches!(event, ModelEvent::Completed(_)))
        );
        assert_conformant_failure(&events, ModelFailureCode::MalformedResponse);
    }
}
