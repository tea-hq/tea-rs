//! Live OpenAI-compatible `ModelProvider` backed by `reqwest`.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use futures_util::StreamExt;
use tea_control::CancellationScope;
use tea_model::{
    HostedToolKind, ModelCapabilities, ModelEvent, ModelFailure, ModelFailureCode, ModelProvider,
    ModelRequest, ModelResponseInfo, ModelSpec, ProviderId,
};
use tea_protocol::{ModelId, ReasoningEffort, RetryClass};
use tea_provider_http::{
    ProviderHttpConfig, UserAgent, read_bounded_error_body, retry_after_delay,
};

use crate::catalog::default_catalog;
use crate::credential::{CredentialResolver, EnvCredentialResolver, OpenAiApiMode, OpenAiConfig};
use crate::error::{OpenAiError, OpenAiErrorCode};
use crate::reasoning::OpenAiReasoningEffortMap;
use crate::request::{chat_completions_url, request_headers};
use crate::responses::{build_responses_body_with_reasoning_map, responses_url};
use crate::responses_stream::ResponsesReducer;
use crate::sse::{SseEvent, SseParser};
use crate::stream::{ChunkReducer, map_http_failure};

#[derive(Debug)]
enum ApiReducer {
    ChatCompletions(ChunkReducer),
    Responses(Box<ResponsesReducer>),
}

impl ApiReducer {
    fn new(mode: OpenAiApiMode, provider_id: &ProviderId) -> Self {
        match mode {
            OpenAiApiMode::ChatCompletions => Self::ChatCompletions(ChunkReducer::new()),
            OpenAiApiMode::Responses => Self::Responses(Box::new(ResponsesReducer::for_provider(
                provider_id.as_str(),
            ))),
        }
    }

    fn terminal_emitted(&self) -> bool {
        match self {
            Self::ChatCompletions(reducer) => reducer.terminal_emitted(),
            Self::Responses(reducer) => reducer.terminal_emitted(),
        }
    }

    fn map_chunk(&mut self, value: &serde_json::Value) -> Result<Vec<ModelEvent>, OpenAiError> {
        match self {
            Self::ChatCompletions(reducer) => reducer.map_chunk(value),
            Self::Responses(reducer) => reducer.map_chunk(value),
        }
    }

    fn finish(&mut self) -> Result<Option<ModelEvent>, OpenAiError> {
        match self {
            Self::ChatCompletions(reducer) => reducer.finish(),
            Self::Responses(reducer) => reducer.finish(),
        }
    }
}

/// OpenAI-compatible streaming provider adapter.
#[derive(Debug, Clone)]
pub struct OpenAiProvider {
    config: Arc<OpenAiConfig>,
    client: reqwest::Client,
    catalog: Vec<ModelSpec>,
    reasoning_effort_maps: BTreeMap<ModelId, OpenAiReasoningEffortMap>,
}

impl OpenAiProvider {
    /// Creates a provider from a connection config and model catalog.
    ///
    /// # Errors
    ///
    /// Returns an error when the HTTP client cannot be built.
    pub fn new(config: Arc<OpenAiConfig>, catalog: Vec<ModelSpec>) -> Result<Self, OpenAiError> {
        Self::new_with_http_config(config, catalog, BTreeMap::new(), &ProviderHttpConfig::new())
    }

    fn new_with_http_config(
        config: Arc<OpenAiConfig>,
        catalog: Vec<ModelSpec>,
        reasoning_effort_maps: BTreeMap<ModelId, OpenAiReasoningEffortMap>,
        http_config: &ProviderHttpConfig,
    ) -> Result<Self, OpenAiError> {
        let client = http_config
            .build_client(Duration::from_millis(config.timeout_millis()))
            .map_err(|error| OpenAiError::new(OpenAiErrorCode::Transport, error.to_string()))?;
        let catalog = normalize_catalog(&config, catalog)?;
        let reasoning_effort_maps =
            normalize_reasoning_effort_maps(&catalog, reasoning_effort_maps)?;
        Ok(Self {
            config,
            client,
            catalog,
            reasoning_effort_maps,
        })
    }

    /// Returns the connection configuration.
    #[must_use]
    pub fn config(&self) -> &OpenAiConfig {
        &self.config
    }
}

impl ModelProvider for OpenAiProvider {
    fn provider_id(&self) -> &ProviderId {
        self.config.provider_id()
    }

    fn models(&self) -> &[ModelSpec] {
        &self.catalog
    }

    /// # Errors
    ///
    /// Returns an error when the request body cannot be built.
    #[allow(clippy::too_many_lines)]
    fn stream(
        &self,
        request: ModelRequest,
        cancellation: CancellationScope,
    ) -> tea_model::BoxModelStream {
        let config = Arc::clone(&self.config);
        let client = self.client.clone();
        let reasoning_map = self.reasoning_effort_maps.get(request.model_id()).cloned();
        Box::pin(async_stream::stream! {
            let api_mode = config.api_mode();
            let body_result = match api_mode {
                OpenAiApiMode::ChatCompletions => crate::request::build_chat_completions_body_with_reasoning_map(
                    &request,
                    &config,
                    reasoning_map.as_ref(),
                ),
                OpenAiApiMode::Responses => build_responses_body_with_reasoning_map(
                    &request,
                    &config,
                    reasoning_map.as_ref(),
                ),
            };
            let body = match body_result {
                Ok(body) => body,
                Err(error) => {
                    yield ModelEvent::Started(ModelResponseInfo::new());
                    yield ModelEvent::Failed(ModelFailure::new(
                        error.code().into_model_failure_code(),
                        error.message(),
                        RetryClass::Never,
                    ).unwrap_or_else(|_| ModelFailure::internal_adapter_failure()));
                    return;
                }
            };
            let url = match api_mode {
                OpenAiApiMode::ChatCompletions => chat_completions_url(&config),
                OpenAiApiMode::Responses => responses_url(&config),
            };
            let mut request_builder = client.post(&url).json(&body);
            for (header, value) in request_headers(&config) {
                request_builder = request_builder.header(header, value);
            }
            let response = match request_builder.send().await {
                Ok(response) => response,
                Err(error) => {
                    yield ModelEvent::Started(ModelResponseInfo::new());
                    yield ModelEvent::Failed(transport_failure(&error));
                    return;
                }
            };
            if !response.status().is_success() {
                let status = response.status().as_u16();
                let retry_after = retry_after_delay(response.headers(), SystemTime::now());
                let body_text = read_bounded_error_body(response).await;
                yield ModelEvent::Started(ModelResponseInfo::new());
                let mut failure = map_http_failure(status, &body_text);
                if let Some(retry_after) = retry_after {
                    failure = failure.with_retry_after(retry_after);
                }
                yield ModelEvent::Failed(failure);
                return;
            }
            let mut bytes = response.bytes_stream();
            let mut parser = SseParser::new();
            let mut reducer = ApiReducer::new(api_mode, config.provider_id());
            let mut started_emitted = false;
            while !reducer.terminal_emitted() {
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
                for sse in parser.feed(&chunk) {
                    match sse {
                        SseEvent::Data(payload) => {
                            let Ok(value) = serde_json::from_str::<serde_json::Value>(&payload) else {
                                continue;
                            };
                            match reducer.map_chunk(&value) {
                                Ok(events) => {
                                    for event in events {
                                        let started = matches!(event, ModelEvent::Started(_));
                                        let terminal = matches!(event, ModelEvent::Completed(_) | ModelEvent::Failed(_));
                                        if terminal && !started_emitted {
                                            yield ModelEvent::Started(ModelResponseInfo::new());
                                        }
                                        if started {
                                            started_emitted = true;
                                        }
                                        yield event;
                                        if terminal {
                                            return;
                                        }
                                    }
                                }
                                Err(error) => {
                                    if !started_emitted {
                                        yield ModelEvent::Started(ModelResponseInfo::new());
                                    }
                                    yield ModelEvent::Failed(ModelFailure::new(
                                        error.code().into_model_failure_code(),
                                        error.message(),
                                        RetryClass::Never,
                                    ).unwrap_or_else(|_| ModelFailure::internal_adapter_failure()));
                                    return;
                                }
                            }
                        }
                        SseEvent::Done => {
                            match reducer.finish() {
                                Ok(Some(terminal)) => {
                                    if !started_emitted {
                                        yield ModelEvent::Started(ModelResponseInfo::new());
                                    }
                                    yield terminal;
                                }
                                Ok(None) => {}
                                Err(error) => {
                                    if !started_emitted {
                                        yield ModelEvent::Started(ModelResponseInfo::new());
                                    }
                                    yield ModelEvent::Failed(adapter_failure(&error));
                                }
                            }
                            return;
                        }
                    }
                }
            }
            for sse in parser.finish() {
                if let SseEvent::Data(payload) = sse
                    && let Ok(value) = serde_json::from_str::<serde_json::Value>(&payload)
                {
                    match reducer.map_chunk(&value) {
                        Ok(events) => {
                            for event in events {
                                let started = matches!(event, ModelEvent::Started(_));
                                let terminal = matches!(event, ModelEvent::Completed(_) | ModelEvent::Failed(_));
                                if terminal && !started_emitted {
                                    yield ModelEvent::Started(ModelResponseInfo::new());
                                }
                                if started {
                                    started_emitted = true;
                                }
                                yield event;
                                if terminal {
                                    return;
                                }
                            }
                        }
                        Err(error) => {
                            if !started_emitted {
                                yield ModelEvent::Started(ModelResponseInfo::new());
                            }
                            yield ModelEvent::Failed(ModelFailure::new(
                                error.code().into_model_failure_code(),
                                error.message(),
                                RetryClass::Never,
                            ).unwrap_or_else(|_| ModelFailure::internal_adapter_failure()));
                            return;
                        }
                    }
                    if reducer.terminal_emitted() {
                        return;
                    }
                }
            }
            match reducer.finish() {
                Ok(Some(terminal)) => {
                    if !started_emitted {
                        yield ModelEvent::Started(ModelResponseInfo::new());
                    }
                    yield terminal;
                }
                Ok(None) => {}
                Err(error) => {
                    if !started_emitted {
                        yield ModelEvent::Started(ModelResponseInfo::new());
                    }
                    yield ModelEvent::Failed(adapter_failure(&error));
                }
            }
        })
    }
}

fn normalize_catalog(
    config: &OpenAiConfig,
    catalog: Vec<ModelSpec>,
) -> Result<Vec<ModelSpec>, OpenAiError> {
    catalog
        .into_iter()
        .map(|spec| {
            let advertised = spec.capabilities();
            let mut capabilities = ModelCapabilities::text();
            if advertised.accepts_images() {
                capabilities = capabilities.with_image_input();
            }
            if advertised.supports_reasoning() {
                capabilities = capabilities.with_reasoning();
            }
            if advertised.supports_tools() {
                capabilities = capabilities.with_tools(advertised.supports_parallel_tool_calls());
            }
            if advertised.reports_usage() {
                capabilities = capabilities.with_usage_reporting();
            }
            let hosted_search = config.api_mode() == OpenAiApiMode::Responses
                && if is_official_openai_endpoint(config.base_url()) {
                    supports_openai_web_search_model(spec.model_id().as_str())
                } else {
                    advertised.supports_hosted_tool(HostedToolKind::WebSearch)
                };
            if hosted_search {
                capabilities = capabilities.with_hosted_tool(HostedToolKind::WebSearch);
            }
            let normalized = ModelSpec::new(
                spec.model_id().clone(),
                spec.provider_id().clone(),
                spec.display_name().clone(),
                spec.context_window_tokens(),
                spec.max_output_tokens(),
                capabilities,
            )
            .map_err(|error| {
                OpenAiError::new(
                    OpenAiErrorCode::Internal,
                    format!("OpenAI model catalog normalization failed: {error}"),
                )
            })?;
            Ok(spec
                .reasoning_profile()
                .cloned()
                .map_or(normalized.clone(), |profile| {
                    normalized.with_reasoning_profile(profile)
                }))
        })
        .collect()
}

fn normalize_reasoning_effort_maps(
    catalog: &[ModelSpec],
    mut configured: BTreeMap<ModelId, OpenAiReasoningEffortMap>,
) -> Result<BTreeMap<ModelId, OpenAiReasoningEffortMap>, OpenAiError> {
    let mut normalized = BTreeMap::new();
    for model in catalog {
        let Some(profile) = model.reasoning_profile() else {
            if configured.remove(model.model_id()).is_some() {
                return Err(OpenAiError::new(
                    OpenAiErrorCode::InvalidRequest,
                    "non-reasoning model has a reasoning effort wire map",
                ));
            }
            continue;
        };
        let map = configured
            .remove(model.model_id())
            .map_or_else(|| OpenAiReasoningEffortMap::for_profile(profile), Ok)?;
        let supported = profile
            .supported_efforts()
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        if map.efforts().any(|effort| !supported.contains(&effort))
            || supported
                .iter()
                .copied()
                .filter(|effort| *effort != ReasoningEffort::Off)
                .any(|effort| map.wire_effort(effort).is_err())
        {
            return Err(OpenAiError::new(
                OpenAiErrorCode::InvalidRequest,
                "reasoning effort wire map does not match the model profile",
            ));
        }
        normalized.insert(model.model_id().clone(), map);
    }
    if !configured.is_empty() {
        return Err(OpenAiError::new(
            OpenAiErrorCode::InvalidRequest,
            "reasoning effort wire map references an unknown model",
        ));
    }
    Ok(normalized)
}

fn is_official_openai_endpoint(base_url: &str) -> bool {
    url::Url::parse(base_url)
        .is_ok_and(|url| url.scheme() == "https" && url.host_str() == Some("api.openai.com"))
}

fn supports_openai_web_search_model(model_id: &str) -> bool {
    let family = strip_date_snapshot(model_id);
    family == "gpt-4.1"
        || family == "gpt-4.1-mini"
        || family == "o4-mini"
        || family == "gpt-5"
        || family.strip_prefix("gpt-5.").is_some_and(|version| {
            !version.is_empty()
                && version
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || byte == b'.')
        })
}

fn strip_date_snapshot(model_id: &str) -> &str {
    let Some((family, date)) = model_id.rsplit_once('-') else {
        return model_id;
    };
    if date.len() != 2 || !date.bytes().all(|byte| byte.is_ascii_digit()) {
        return model_id;
    }
    let Some((family, month)) = family.rsplit_once('-') else {
        return model_id;
    };
    if month.len() != 2 || !month.bytes().all(|byte| byte.is_ascii_digit()) {
        return model_id;
    }
    let Some((family, year)) = family.rsplit_once('-') else {
        return model_id;
    };
    if year.len() == 4 && year.bytes().all(|byte| byte.is_ascii_digit()) {
        family
    } else {
        model_id
    }
}

/// Builder for [`OpenAiProvider`] backed by the env contract.
#[derive(Debug, Default)]
pub struct OpenAiProviderBuilder {
    config: Option<Arc<OpenAiConfig>>,
    catalog: Option<Vec<ModelSpec>>,
    reasoning_effort_maps: BTreeMap<ModelId, OpenAiReasoningEffortMap>,
    resolver: Option<Arc<dyn CredentialResolver>>,
    http_config: ProviderHttpConfig,
}

impl OpenAiProviderBuilder {
    /// Creates an empty builder that resolves config from the environment.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a builder pre-populated from the `TEA_OPENAI_*` env contract.
    ///
    /// # Errors
    ///
    /// Returns an error when required env values are missing or invalid.
    pub fn from_env() -> Result<Self, OpenAiError> {
        let config = Arc::new(EnvCredentialResolver::new().resolve()?);
        Ok(Self {
            config: Some(config),
            catalog: None,
            reasoning_effort_maps: BTreeMap::new(),
            resolver: None,
            http_config: ProviderHttpConfig::new(),
        })
    }

    /// Overrides the connection configuration.
    #[must_use]
    pub fn with_config(mut self, config: Arc<OpenAiConfig>) -> Self {
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

    /// Overrides model-level mappings from canonical reasoning efforts to wire values.
    #[must_use]
    pub fn with_reasoning_effort_maps(
        mut self,
        maps: BTreeMap<ModelId, OpenAiReasoningEffortMap>,
    ) -> Self {
        self.reasoning_effort_maps = maps;
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
    /// Returns an error when config resolution, catalog construction, or the
    /// HTTP client build fails.
    pub fn build(self) -> Result<OpenAiProvider, OpenAiError> {
        let config = if let Some(config) = self.config {
            config
        } else {
            let resolver = self
                .resolver
                .unwrap_or_else(|| Arc::new(EnvCredentialResolver::new()));
            Arc::new(resolver.resolve()?)
        };
        let catalog = match self.catalog {
            Some(catalog) => catalog,
            None => default_catalog(&config)?,
        };
        OpenAiProvider::new_with_http_config(
            config,
            catalog,
            self.reasoning_effort_maps,
            &self.http_config,
        )
    }
}

impl OpenAiErrorCode {
    pub(crate) fn into_model_failure_code(self) -> ModelFailureCode {
        match self {
            Self::Authentication => ModelFailureCode::Authentication,
            Self::PermissionDenied => ModelFailureCode::PermissionDenied,
            Self::RateLimited => ModelFailureCode::RateLimited,
            Self::Unavailable => ModelFailureCode::Unavailable,
            Self::Transport => ModelFailureCode::Transport,
            Self::InvalidRequest => ModelFailureCode::InvalidRequest,
            Self::MalformedResponse => ModelFailureCode::MalformedResponse,
            Self::ContextOverflow => ModelFailureCode::ContextOverflow,
            Self::Cancelled => ModelFailureCode::Cancelled,
            Self::Internal => ModelFailureCode::Internal,
        }
    }
}

fn transport_failure(error: &reqwest::Error) -> ModelFailure {
    let code = if error.is_timeout() || error.is_connect() || error.is_request() {
        ModelFailureCode::Transport
    } else {
        ModelFailureCode::Internal
    };
    ModelFailure::new(code, "openai transport error", RetryClass::Immediate)
        .unwrap_or_else(|_| ModelFailure::internal_adapter_failure())
}

fn adapter_failure(error: &OpenAiError) -> ModelFailure {
    ModelFailure::new(
        error.code().into_model_failure_code(),
        error.message(),
        RetryClass::Never,
    )
    .unwrap_or_else(|_| ModelFailure::internal_adapter_failure())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::str::FromStr;

    use futures_util::StreamExt as _;
    use tea_model::{
        HostedToolOptions, ModelCapabilities, ModelDisplayName, ModelStreamValidator,
        ModelToolDefinition, ReasoningEffort, ReasoningProfile, WebSearchOptions,
    };
    use tea_protocol::{
        CanonicalMessage, ContentBlock, MessageId, ModelId, ProtocolTimestamp, TokenCount,
    };
    use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
    use tokio::net::TcpListener;

    use super::*;
    use crate::credential::MapCredentialResolver;

    fn provider(user_agent: Option<UserAgent>) -> OpenAiProvider {
        let config = MapCredentialResolver::new(BTreeMap::from([
            ("TEA_OPENAI_API_KEY".to_owned(), "sk-test-key".to_owned()),
            ("TEA_OPENAI_MODEL".to_owned(), "gpt-4o-mini".to_owned()),
        ]))
        .resolve()
        .unwrap();
        let mut builder = OpenAiProviderBuilder::new()
            .with_config(Arc::new(config))
            .with_catalog(Vec::new());
        if let Some(user_agent) = user_agent {
            builder = builder.with_user_agent(user_agent);
        }
        builder.build().unwrap()
    }

    fn model_spec(id: &str, capabilities: ModelCapabilities) -> ModelSpec {
        ModelSpec::new(
            ModelId::from_str(id).unwrap(),
            ProviderId::from_str("openai").unwrap(),
            ModelDisplayName::from_str(id).unwrap(),
            TokenCount::new(128_000).unwrap(),
            TokenCount::new(16_384).unwrap(),
            capabilities,
        )
        .unwrap()
    }

    #[test]
    fn adapter_error_codes_preserve_provider_neutral_failure_classification() {
        let cases = [
            (
                OpenAiErrorCode::InvalidRequest,
                ModelFailureCode::InvalidRequest,
            ),
            (
                OpenAiErrorCode::Authentication,
                ModelFailureCode::Authentication,
            ),
            (
                OpenAiErrorCode::PermissionDenied,
                ModelFailureCode::PermissionDenied,
            ),
            (OpenAiErrorCode::RateLimited, ModelFailureCode::RateLimited),
            (OpenAiErrorCode::Unavailable, ModelFailureCode::Unavailable),
            (OpenAiErrorCode::Transport, ModelFailureCode::Transport),
            (
                OpenAiErrorCode::MalformedResponse,
                ModelFailureCode::MalformedResponse,
            ),
            (
                OpenAiErrorCode::ContextOverflow,
                ModelFailureCode::ContextOverflow,
            ),
            (OpenAiErrorCode::Cancelled, ModelFailureCode::Cancelled),
            (OpenAiErrorCode::Internal, ModelFailureCode::Internal),
        ];

        for (adapter, expected) in cases {
            assert_eq!(adapter.into_model_failure_code(), expected);
        }
    }

    async fn captured_headers(provider: &OpenAiProvider) -> String {
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
            .post(format!("http://{address}/v1/chat/completions"))
            .send()
            .await
            .unwrap();
        assert_eq!(response.status(), reqwest::StatusCode::NO_CONTENT);
        server.await.unwrap()
    }

    async fn read_http_request(stream: &mut tokio::net::TcpStream) -> Vec<u8> {
        let mut request = Vec::new();
        let mut buffer = [0_u8; 1024];
        loop {
            let read = stream.read(&mut buffer).await.unwrap();
            assert_ne!(read, 0, "request closed before body was sent");
            request.extend_from_slice(&buffer[..read]);
            let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n")
            else {
                continue;
            };
            let headers = String::from_utf8_lossy(&request[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    line.split_once(':').and_then(|(name, value)| {
                        name.eq_ignore_ascii_case("content-length")
                            .then(|| value.trim().parse::<usize>().ok())
                            .flatten()
                    })
                })
                .unwrap_or(0);
            if request.len() >= header_end + 4 + content_length {
                return request;
            }
        }
    }

    #[test]
    fn user_agent_rejects_empty_and_invalid_header_values() {
        assert!(UserAgent::new("").is_err());
        assert!(UserAgent::new("tea-cli/1.0\r\ninvalid").is_err());
    }

    #[test]
    fn hosted_search_capability_is_scoped_to_responses_model_and_endpoint() {
        let responses_config = MapCredentialResolver::new(BTreeMap::from([
            ("TEA_OPENAI_API_KEY".to_owned(), "sk-test-key".to_owned()),
            ("TEA_OPENAI_MODEL".to_owned(), "gpt-4.1".to_owned()),
            ("TEA_OPENAI_API_MODE".to_owned(), "responses".to_owned()),
        ]))
        .resolve()
        .unwrap();
        let provider = OpenAiProviderBuilder::new()
            .with_config(Arc::new(responses_config))
            .with_catalog(vec![
                model_spec("gpt-4.1", ModelCapabilities::text()),
                model_spec(
                    "gpt-4o-mini",
                    ModelCapabilities::text().with_hosted_tool(HostedToolKind::WebSearch),
                ),
            ])
            .build()
            .unwrap();
        assert!(
            provider.models()[0]
                .capabilities()
                .supports_hosted_tool(HostedToolKind::WebSearch)
        );
        assert!(
            !provider.models()[1]
                .capabilities()
                .supports_hosted_tool(HostedToolKind::WebSearch)
        );

        let chat_config = provider
            .config()
            .clone()
            .with_api_mode(OpenAiApiMode::ChatCompletions);
        let provider = OpenAiProviderBuilder::new()
            .with_config(Arc::new(chat_config))
            .with_catalog(vec![model_spec(
                "gpt-4.1",
                ModelCapabilities::text().with_hosted_tool(HostedToolKind::WebSearch),
            )])
            .build()
            .unwrap();
        assert!(
            !provider.models()[0]
                .capabilities()
                .supports_hosted_tool(HostedToolKind::WebSearch)
        );
    }

    #[test]
    fn custom_responses_endpoint_requires_explicit_hosted_capability() {
        let config = MapCredentialResolver::new(BTreeMap::from([
            (
                "TEA_OPENAI_BASE_URL".to_owned(),
                "https://gateway.example.test/v1".to_owned(),
            ),
            ("TEA_OPENAI_API_KEY".to_owned(), "sk-test-key".to_owned()),
            ("TEA_OPENAI_MODEL".to_owned(), "gpt-4.1".to_owned()),
            ("TEA_OPENAI_API_MODE".to_owned(), "responses".to_owned()),
        ]))
        .resolve()
        .unwrap();
        let provider = OpenAiProviderBuilder::new()
            .with_config(Arc::new(config))
            .with_catalog(vec![
                model_spec("gpt-4.1", ModelCapabilities::text()),
                model_spec(
                    "gateway-search-model",
                    ModelCapabilities::text().with_hosted_tool(HostedToolKind::WebSearch),
                ),
            ])
            .build()
            .unwrap();

        assert!(
            !provider.models()[0]
                .capabilities()
                .supports_hosted_tool(HostedToolKind::WebSearch)
        );
        assert!(
            provider.models()[1]
                .capabilities()
                .supports_hosted_tool(HostedToolKind::WebSearch)
        );
    }

    #[test]
    fn catalog_normalization_preserves_reasoning_profile_and_validates_wire_map() {
        let config = MapCredentialResolver::new(BTreeMap::from([
            ("TEA_OPENAI_API_KEY".to_owned(), "sk-test-key".to_owned()),
            ("TEA_OPENAI_MODEL".to_owned(), "gpt-5".to_owned()),
        ]))
        .resolve()
        .unwrap();
        let profile = ReasoningProfile::new(
            ReasoningEffort::Medium,
            [
                ReasoningEffort::Minimal,
                ReasoningEffort::Medium,
                ReasoningEffort::ExtraHigh,
            ],
        )
        .unwrap();
        let spec = model_spec(
            "gpt-5",
            ModelCapabilities::text().with_tools(true).with_reasoning(),
        )
        .with_reasoning_profile(profile.clone());
        let map = OpenAiReasoningEffortMap::new([
            (ReasoningEffort::Minimal, "minimal".to_owned()),
            (ReasoningEffort::Medium, "medium".to_owned()),
            (ReasoningEffort::ExtraHigh, "xhigh".to_owned()),
        ])
        .unwrap();
        let provider = OpenAiProviderBuilder::new()
            .with_config(Arc::new(config.clone()))
            .with_catalog(vec![spec.clone()])
            .with_reasoning_effort_maps(BTreeMap::from([(spec.model_id().clone(), map)]))
            .build()
            .unwrap();
        assert_eq!(provider.models()[0].reasoning_profile(), Some(&profile));

        let incomplete =
            OpenAiReasoningEffortMap::new([(ReasoningEffort::Minimal, "minimal".to_owned())])
                .unwrap();
        assert!(
            OpenAiProviderBuilder::new()
                .with_config(Arc::new(config))
                .with_catalog(vec![spec.clone()])
                .with_reasoning_effort_maps(BTreeMap::from([
                    (spec.model_id().clone(), incomplete,)
                ]))
                .build()
                .is_err()
        );
    }

    #[test]
    fn default_catalog_advertises_known_reasoning_families() {
        let config = MapCredentialResolver::new(BTreeMap::from([
            ("TEA_OPENAI_API_KEY".to_owned(), "sk-test-key".to_owned()),
            ("TEA_OPENAI_MODEL".to_owned(), "gpt-5".to_owned()),
        ]))
        .resolve()
        .unwrap();
        let provider = OpenAiProviderBuilder::new()
            .with_config(Arc::new(config))
            .build()
            .unwrap();
        let profile = provider.models()[0].reasoning_profile().unwrap();
        assert_eq!(profile.default_effort(), ReasoningEffort::Medium);
        assert!(
            profile
                .supported_efforts()
                .contains(&ReasoningEffort::Minimal)
        );
        assert!(profile.supported_efforts().contains(&ReasoningEffort::Off));
    }

    #[tokio::test]
    async fn request_mapping_failure_still_obeys_stream_grammar() {
        let provider = provider(None);
        let message = CanonicalMessage::user(
            MessageId::from_str("0195a0b1-5e52-74b2-8c25-0aa7aa000043").unwrap(),
            vec![ContentBlock::text("search").unwrap()],
            ProtocolTimestamp::from_str("2026-07-23T09:30:12.125Z").unwrap(),
        )
        .unwrap();
        let request = ModelRequest::new(ModelId::from_str("gpt-4o-mini").unwrap(), vec![message])
            .unwrap()
            .with_tools(
                vec![
                    ModelToolDefinition::hosted(
                        "Searches the web.",
                        serde_json::json!({"type": "object"}),
                        HostedToolOptions::WebSearch(WebSearchOptions::new()),
                    )
                    .unwrap(),
                ],
                false,
            )
            .unwrap();

        let events = provider
            .stream(request, CancellationScope::new())
            .collect::<Vec<_>>()
            .await;

        assert!(matches!(events.first(), Some(ModelEvent::Started(_))));
        assert!(matches!(events.last(), Some(ModelEvent::Failed(_))));
        let mut validator = ModelStreamValidator::new();
        for event in &events {
            validator.observe(event).unwrap();
        }
        validator.finish().unwrap();
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
    async fn http_failure_propagates_retry_after_hint() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let _ = read_http_request(&mut stream).await;
            let body = r#"{"error":{"message":"Service temporarily unavailable"}}"#;
            let response = format!(
                "HTTP/1.1 503 Service Unavailable\r\nRetry-After: 7\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        });
        let config = MapCredentialResolver::new(BTreeMap::from([
            ("TEA_OPENAI_API_KEY".to_owned(), "sk-test-key".to_owned()),
            ("TEA_OPENAI_MODEL".to_owned(), "gpt-4o-mini".to_owned()),
            (
                "TEA_OPENAI_BASE_URL".to_owned(),
                format!("http://{address}/v1"),
            ),
        ]))
        .resolve()
        .unwrap();
        let provider = OpenAiProviderBuilder::new()
            .with_config(Arc::new(config))
            .with_catalog(Vec::new())
            .build()
            .unwrap();
        let message = CanonicalMessage::user(
            MessageId::from_str("0195a0b1-5e52-74b2-8c25-0aa7aa000044").unwrap(),
            vec![ContentBlock::text("hello").unwrap()],
            ProtocolTimestamp::from_str("2026-07-23T09:30:12.125Z").unwrap(),
        )
        .unwrap();
        let request =
            ModelRequest::new(ModelId::from_str("gpt-4o-mini").unwrap(), vec![message]).unwrap();
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
        assert_eq!(failure.retry_after(), Some(Duration::from_secs(7)));
        server.await.unwrap();
    }

    #[tokio::test]
    async fn responses_mode_posts_to_responses_and_reduces_the_stream() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request = read_http_request(&mut stream).await;
            let sse = concat!(
                "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_http_test\",\"model\":\"gpt-4.1\"}}\n\n",
                "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_http_test\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[]}}\n\n",
                "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":\"hello\"}\n\n",
                "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"msg_http_test\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"hello\",\"annotations\":[]}]}}\n\n",
                "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_http_test\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n",
                "data: [DONE]\n\n",
            );
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{sse}",
                sse.len()
            );
            stream.write_all(response.as_bytes()).await.unwrap();
            request
        });

        let config = MapCredentialResolver::new(BTreeMap::from([
            (
                "TEA_OPENAI_BASE_URL".to_owned(),
                format!("http://{address}/v1"),
            ),
            ("TEA_OPENAI_API_KEY".to_owned(), "sk-test-key".to_owned()),
            ("TEA_OPENAI_MODEL".to_owned(), "gpt-4.1".to_owned()),
            ("TEA_OPENAI_API_MODE".to_owned(), "responses".to_owned()),
        ]))
        .resolve()
        .unwrap();
        let provider = OpenAiProviderBuilder::new()
            .with_config(Arc::new(config))
            .with_catalog(Vec::new())
            .build()
            .unwrap();
        let message = CanonicalMessage::user(
            MessageId::from_str("0195a0b1-5e52-74b2-8c25-0aa7aa000041").unwrap(),
            vec![ContentBlock::text("hi").unwrap()],
            ProtocolTimestamp::from_str("2026-07-23T09:30:12.125Z").unwrap(),
        )
        .unwrap();
        let request =
            ModelRequest::new(ModelId::from_str("gpt-4.1").unwrap(), vec![message]).unwrap();
        let mut model_stream = provider.stream(request, CancellationScope::new());
        let mut text = String::new();
        let mut completed = false;
        while let Some(event) = model_stream.next().await {
            match event {
                ModelEvent::TextDelta(delta) => text.push_str(delta.as_str()),
                ModelEvent::Completed(_) => completed = true,
                ModelEvent::Failed(failure) => panic!("unexpected failure: {failure:?}"),
                _ => {}
            }
        }
        assert_eq!(text, "hello");
        assert!(completed);

        let captured = server.await.unwrap();
        let captured = String::from_utf8(captured).unwrap();
        assert!(captured.starts_with("POST /v1/responses HTTP/1.1\r\n"));
        let body = captured.split_once("\r\n\r\n").unwrap().1;
        let body: serde_json::Value = serde_json::from_str(body).unwrap();
        assert_eq!(body["store"], false);
        assert_eq!(body["input"][0]["content"][0]["type"], "input_text");
    }

    #[tokio::test]
    async fn responses_mode_sends_and_reduces_hosted_web_search_contract() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request = read_http_request(&mut stream).await;
            let sse = concat!(
                "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_http_search\",\"model\":\"gpt-4.1\"}}\n\n",
                "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"ws_http_search\",\"type\":\"web_search_call\"}}\n\n",
                "data: {\"type\":\"response.output_item.done\",\"output_index\":0,\"item\":{\"id\":\"ws_http_search\",\"type\":\"web_search_call\",\"status\":\"completed\",\"action\":{\"type\":\"search\",\"queries\":[\"tea-rs\"],\"sources\":[{\"type\":\"url\",\"url\":\"https://example.com/search\"}]}}}\n\n",
                "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_http_search\",\"status\":\"completed\",\"output\":[],\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n",
                "data: [DONE]\n\n",
            );
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{sse}",
                sse.len()
            );
            stream.write_all(response.as_bytes()).await.unwrap();
            request
        });
        let config = MapCredentialResolver::new(BTreeMap::from([
            (
                "TEA_OPENAI_BASE_URL".to_owned(),
                format!("http://{address}/v1"),
            ),
            ("TEA_OPENAI_API_KEY".to_owned(), "sk-test-key".to_owned()),
            ("TEA_OPENAI_MODEL".to_owned(), "gpt-4.1".to_owned()),
            ("TEA_OPENAI_API_MODE".to_owned(), "responses".to_owned()),
        ]))
        .resolve()
        .unwrap();
        let provider = OpenAiProviderBuilder::new()
            .with_config(Arc::new(config))
            .with_catalog(vec![model_spec(
                "gpt-4.1",
                ModelCapabilities::text().with_hosted_tool(HostedToolKind::WebSearch),
            )])
            .build()
            .unwrap();
        let message = CanonicalMessage::user(
            MessageId::from_str("0195a0b1-5e52-74b2-8c25-0aa7aa000042").unwrap(),
            vec![ContentBlock::text("search").unwrap()],
            ProtocolTimestamp::from_str("2026-07-23T09:30:12.125Z").unwrap(),
        )
        .unwrap();
        let request = ModelRequest::new(ModelId::from_str("gpt-4.1").unwrap(), vec![message])
            .unwrap()
            .with_tools(
                vec![
                    ModelToolDefinition::hosted(
                        "Searches the web.",
                        serde_json::json!({"type":"object"}),
                        HostedToolOptions::WebSearch(WebSearchOptions::new()),
                    )
                    .unwrap(),
                ],
                false,
            )
            .unwrap();
        let mut model_stream = provider.stream(request, CancellationScope::new());
        let mut hosted_started = false;
        let mut hosted_completed = false;
        let mut completed = false;
        while let Some(event) = model_stream.next().await {
            match event {
                ModelEvent::HostedToolStarted(_) => hosted_started = true,
                ModelEvent::HostedToolCompleted(_) => hosted_completed = true,
                ModelEvent::Completed(_) => completed = true,
                ModelEvent::Failed(failure) => panic!("unexpected failure: {failure:?}"),
                _ => {}
            }
        }
        assert!(hosted_started);
        assert!(hosted_completed);
        assert!(completed);

        let captured = String::from_utf8(server.await.unwrap()).unwrap();
        let body: serde_json::Value =
            serde_json::from_str(captured.split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(body["tools"][0]["type"], "web_search");
        assert!(
            body["include"]
                .as_array()
                .unwrap()
                .contains(&serde_json::json!("web_search_call.action.sources"))
        );
    }
}
