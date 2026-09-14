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
use tea_protocol::{MAX_TEXT_BLOCK_BYTES, ModelId, ReasoningEffort, RetryClass};
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
use crate::stream::{ChunkReducer, map_chat_completion_response, map_http_failure};

const MAX_NON_STREAMING_RESPONSE_BODY_BYTES: usize = MAX_TEXT_BLOCK_BYTES * 8;

#[derive(Debug)]
enum ApiReducer {
    ChatCompletions(ChunkReducer),
    Responses(Box<ResponsesReducer>),
}

#[derive(Debug, Clone, Copy)]
enum StructuredOutputKind {
    JsonObject,
    JsonSchema,
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
        if config
            .compatibility_profile()
            .is_some_and(|profile| !profile.supports_api_mode(config.api_mode()))
        {
            return Err(OpenAiError::new(
                OpenAiErrorCode::InvalidRequest,
                "OpenAI compatibility profile does not support the configured API mode",
            ));
        }
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
        let reasoning_map = self.reasoning_effort_maps.get(request.model_id());
        match self.config.api_mode() {
            OpenAiApiMode::ChatCompletions => {
                crate::request::build_chat_completions_body_with_reasoning_map(
                    request,
                    &self.config,
                    reasoning_map,
                )
            }
            OpenAiApiMode::Responses => {
                build_responses_body_with_reasoning_map(request, &self.config, reasoning_map)
            }
        }
        .map_err(|error| adapter_failure(&error))
    }
}

impl ModelProvider for OpenAiProvider {
    fn provider_id(&self) -> &ProviderId {
        self.config.provider_id()
    }

    fn models(&self) -> &[ModelSpec] {
        &self.catalog
    }

    fn validate_request(&self, request: &ModelRequest) -> Result<(), ModelFailure> {
        self.validated_request_body(request).map(|_| ())
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
        let body_result = self.validated_request_body(&request);
        Box::pin(async_stream::stream! {
            let api_mode = config.api_mode();
            let body = match body_result {
                Ok(body) => body,
                Err(failure) => {
                    yield ModelEvent::Started(ModelResponseInfo::new());
                    yield ModelEvent::Failed(failure);
                    return;
                }
            };
            let Some(streaming_response) = body.get("stream").and_then(serde_json::Value::as_bool) else {
                yield ModelEvent::Started(ModelResponseInfo::new());
                yield ModelEvent::Failed(ModelFailure::new(
                    ModelFailureCode::Internal,
                    "OpenAI request body omitted the stream mode",
                    RetryClass::Never,
                ).unwrap_or_else(|_| ModelFailure::internal_adapter_failure()));
                return;
            };
            let url = match api_mode {
                OpenAiApiMode::ChatCompletions => chat_completions_url(&config),
                OpenAiApiMode::Responses => responses_url(&config),
            };
            let mut request_builder = client.post(&url).json(&body);
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
                let body_text = read_bounded_error_body(response).await;
                yield ModelEvent::Started(ModelResponseInfo::new());
                let mut failure = map_http_failure(status, &body_text);
                if let Some(retry_after) = retry_after {
                    failure = failure.with_retry_after(retry_after);
                }
                yield ModelEvent::Failed(failure);
                return;
            }
            if !streaming_response {
                if api_mode != OpenAiApiMode::ChatCompletions {
                    yield ModelEvent::Started(ModelResponseInfo::new());
                    yield ModelEvent::Failed(ModelFailure::new(
                        ModelFailureCode::Internal,
                        "non-streaming OpenAI response used an unsupported API mode",
                        RetryClass::Never,
                    ).unwrap_or_else(|_| ModelFailure::internal_adapter_failure()));
                    return;
                }
                let response_body = match read_bounded_success_body(response, &cancellation).await {
                    Ok(body) => body,
                    Err(failure) => {
                        yield ModelEvent::Started(ModelResponseInfo::new());
                        yield ModelEvent::Failed(failure);
                        return;
                    }
                };
                let Ok(value) = serde_json::from_slice(&response_body) else {
                    yield ModelEvent::Started(ModelResponseInfo::new());
                    yield ModelEvent::Failed(adapter_failure(&OpenAiError::new(
                        OpenAiErrorCode::MalformedResponse,
                        "invalid non-streaming Chat Completions JSON",
                    )));
                    return;
                };
                match map_chat_completion_response(&value) {
                    Ok(events) => {
                        for event in events {
                            yield event;
                        }
                    }
                    Err(error) => {
                        yield ModelEvent::Started(ModelResponseInfo::new());
                        yield ModelEvent::Failed(adapter_failure(&error));
                    }
                }
                return;
            }
            let mut bytes = response.bytes_stream();
            let mut parser = SseParser::new();
            let mut reducer = ApiReducer::new(api_mode, config.provider_id());
            let mut started_emitted = false;
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
                let mut saw_done = false;
                for sse in parser.feed(&chunk) {
                    if saw_done {
                        if !started_emitted {
                            yield ModelEvent::Started(ModelResponseInfo::new());
                        }
                        yield ModelEvent::Failed(adapter_failure(&OpenAiError::new(
                            OpenAiErrorCode::MalformedResponse,
                            "SSE event arrived after [DONE]",
                        )));
                        return;
                    }
                    match sse {
                        SseEvent::Data(payload) => {
                            let value = match parse_sse_payload(&payload) {
                                Ok(value) => value,
                                Err(error) => {
                                    if !started_emitted {
                                        yield ModelEvent::Started(ModelResponseInfo::new());
                                    }
                                    yield ModelEvent::Failed(adapter_failure(&error));
                                    return;
                                }
                            };
                            match reducer.map_chunk(&value) {
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
                                Ok(Some(ModelEvent::Completed(completion))) => {
                                    pending_completion = Some(completion);
                                }
                                Ok(Some(ModelEvent::Failed(failure))) => {
                                    if !started_emitted {
                                        yield ModelEvent::Started(ModelResponseInfo::new());
                                    }
                                    yield ModelEvent::Failed(failure);
                                    return;
                                }
                                Ok(Some(event)) => yield event,
                                Ok(None) => {}
                                Err(error) => {
                                    if !started_emitted {
                                        yield ModelEvent::Started(ModelResponseInfo::new());
                                    }
                                    yield ModelEvent::Failed(adapter_failure(&error));
                                    return;
                                }
                            }
                            saw_done = true;
                        }
                    }
                }
                if saw_done {
                    if let Some(completion) = pending_completion {
                        if !started_emitted {
                            yield ModelEvent::Started(ModelResponseInfo::new());
                        }
                        yield ModelEvent::Completed(completion);
                    }
                    return;
                }
            }
            for sse in parser.finish() {
                if let SseEvent::Data(payload) = sse {
                    let value = match parse_sse_payload(&payload) {
                        Ok(value) => value,
                        Err(error) => {
                            if !started_emitted {
                                yield ModelEvent::Started(ModelResponseInfo::new());
                            }
                            yield ModelEvent::Failed(adapter_failure(&error));
                            return;
                        }
                    };
                    match reducer.map_chunk(&value) {
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
                            yield ModelEvent::Failed(ModelFailure::new(
                                error.code().into_model_failure_code(),
                                error.message(),
                                RetryClass::Never,
                            ).unwrap_or_else(|_| ModelFailure::internal_adapter_failure()));
                            return;
                        }
                    }
                }
            }
            match reducer.finish() {
                Ok(Some(ModelEvent::Completed(completion))) => {
                    pending_completion = Some(completion);
                }
                Ok(Some(ModelEvent::Failed(failure))) => {
                    if !started_emitted {
                        yield ModelEvent::Started(ModelResponseInfo::new());
                    }
                    yield ModelEvent::Failed(failure);
                    return;
                }
                Ok(Some(event)) => yield event,
                Ok(None) => {}
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

async fn read_bounded_success_body(
    response: reqwest::Response,
    cancellation: &CancellationScope,
) -> Result<Vec<u8>, ModelFailure> {
    let mut body = Vec::new();
    let mut bytes = response.bytes_stream();
    loop {
        let chunk = tokio::select! {
            biased;
            () = cancellation.cancelled() => {
                return Err(ModelFailure::new(
                    ModelFailureCode::Cancelled,
                    "model request was cancelled",
                    RetryClass::Never,
                ).unwrap_or_else(|_| ModelFailure::internal_adapter_failure()));
            }
            chunk = bytes.next() => chunk,
        };
        let Some(chunk) = chunk else { return Ok(body) };
        let chunk = chunk.map_err(|error| transport_failure(&error))?;
        let Some(new_len) = body.len().checked_add(chunk.len()) else {
            return Err(malformed_success_body());
        };
        if new_len > MAX_NON_STREAMING_RESPONSE_BODY_BYTES {
            return Err(malformed_success_body());
        }
        body.extend_from_slice(&chunk);
    }
}

fn malformed_success_body() -> ModelFailure {
    ModelFailure::new(
        ModelFailureCode::MalformedResponse,
        "non-streaming Chat Completions response body exceeds the supported bound",
        RetryClass::Never,
    )
    .unwrap_or_else(|_| ModelFailure::internal_adapter_failure())
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
            if advertised.supports_final_json_object() {
                validate_advertised_structured_output(config, StructuredOutputKind::JsonObject)?;
                capabilities = capabilities.with_final_json_object();
            }
            if advertised.supports_final_json_schema() {
                validate_advertised_structured_output(config, StructuredOutputKind::JsonSchema)?;
                capabilities = capabilities.with_final_json_schema();
            }
            if advertised.supports_final_json_schema_with_tools() {
                let Some(profile) = config.compatibility_profile() else {
                    return Err(OpenAiError::new(
                        OpenAiErrorCode::InvalidRequest,
                        "JSON Schema with tools capability requires an explicit compatibility profile",
                    ));
                };
                if !profile.supports_final_json_schema_with_tools(config.api_mode()) {
                    return Err(OpenAiError::new(
                        OpenAiErrorCode::InvalidRequest,
                        "JSON Schema with tools capability is incompatible with the profile and API mode",
                    ));
                }
                capabilities = capabilities.with_final_json_schema_with_tools();
            }
            if advertised.supports_hosted_tool(HostedToolKind::WebSearch) {
                if config.api_mode() != OpenAiApiMode::Responses {
                    return Err(OpenAiError::new(
                        OpenAiErrorCode::InvalidRequest,
                        "hosted web-search capability requires the Responses API mode",
                    ));
                }
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

fn validate_advertised_structured_output(
    config: &OpenAiConfig,
    format: StructuredOutputKind,
) -> Result<(), OpenAiError> {
    let Some(profile) = config.compatibility_profile() else {
        return Err(OpenAiError::new(
            OpenAiErrorCode::InvalidRequest,
            "structured-output model capability requires an explicit compatibility profile",
        ));
    };
    let supported = match format {
        StructuredOutputKind::JsonObject => profile.supports_final_json_object(config.api_mode()),
        StructuredOutputKind::JsonSchema => profile.supports_final_json_schema(config.api_mode()),
    };
    if !supported {
        return Err(OpenAiError::new(
            OpenAiErrorCode::InvalidRequest,
            "structured-output model capability is incompatible with the profile and API mode",
        ));
    }
    Ok(())
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

fn invalid_request(message: impl Into<String>) -> ModelFailure {
    ModelFailure::new(ModelFailureCode::InvalidRequest, message, RetryClass::Never)
        .unwrap_or_else(|_| ModelFailure::internal_adapter_failure())
}

fn parse_sse_payload(payload: &str) -> Result<serde_json::Value, OpenAiError> {
    serde_json::from_str(payload).map_err(|_| {
        OpenAiError::new(
            OpenAiErrorCode::MalformedResponse,
            "invalid OpenAI SSE JSON",
        )
    })
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
        CanonicalMessage, ContentBlock, FinalOutputFormat, MessageId, ModelId, ProtocolTimestamp,
        TokenCount,
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
        let mut builder = OpenAiProviderBuilder::new().with_config(Arc::new(config));
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
    fn structured_output_catalog_requires_a_compatible_profile() {
        let openai = MapCredentialResolver::new(BTreeMap::from([
            ("TEA_OPENAI_API_KEY".to_owned(), "sk-test-key".to_owned()),
            ("TEA_OPENAI_MODEL".to_owned(), "gpt-test".to_owned()),
        ]))
        .resolve()
        .unwrap();
        let capabilities = ModelCapabilities::text()
            .with_final_json_object()
            .with_final_json_schema_with_tools();
        let provider = OpenAiProviderBuilder::new()
            .with_config(Arc::new(openai))
            .with_catalog(vec![model_spec("gpt-test", capabilities)])
            .build()
            .unwrap();
        assert!(
            provider.models()[0]
                .capabilities()
                .supports_final_json_object()
        );
        assert!(
            provider.models()[0]
                .capabilities()
                .supports_final_json_schema()
        );
        assert!(
            provider.models()[0]
                .capabilities()
                .supports_final_json_schema_with_tools()
        );

        let unprofiled = MapCredentialResolver::for_provider(
            ProviderId::from_str("custom-gateway").unwrap(),
            BTreeMap::from([
                ("TEA_OPENAI_API_KEY".to_owned(), "sk-test-key".to_owned()),
                ("TEA_OPENAI_MODEL".to_owned(), "custom-model".to_owned()),
            ]),
        )
        .resolve()
        .unwrap();
        let error = OpenAiProviderBuilder::new()
            .with_config(Arc::new(unprofiled))
            .with_catalog(vec![model_spec(
                "custom-model",
                ModelCapabilities::text().with_final_json_object(),
            )])
            .build()
            .unwrap_err();
        assert_eq!(error.code(), OpenAiErrorCode::InvalidRequest);

        let groq = MapCredentialResolver::new(BTreeMap::from([
            ("TEA_OPENAI_API_KEY".to_owned(), "sk-test-key".to_owned()),
            ("TEA_OPENAI_MODEL".to_owned(), "groq-model".to_owned()),
        ]))
        .resolve()
        .unwrap()
        .with_compatibility_profile(crate::OpenAiCompatibilityProfile::Groq);
        let error = OpenAiProviderBuilder::new()
            .with_config(Arc::new(groq))
            .with_catalog(vec![model_spec(
                "groq-model",
                ModelCapabilities::text().with_final_json_schema_with_tools(),
            )])
            .build()
            .unwrap_err();
        assert_eq!(error.code(), OpenAiErrorCode::InvalidRequest);

        let deepseek_chat = MapCredentialResolver::for_provider(
            ProviderId::from_str("deepseek").unwrap(),
            BTreeMap::from([
                ("TEA_OPENAI_API_KEY".to_owned(), "sk-test-key".to_owned()),
                ("TEA_OPENAI_MODEL".to_owned(), "deepseek-chat".to_owned()),
            ]),
        )
        .resolve()
        .unwrap()
        .with_compatibility_profile(crate::OpenAiCompatibilityProfile::DeepSeek);
        let error = OpenAiProviderBuilder::new()
            .with_config(Arc::new(deepseek_chat))
            .with_catalog(vec![model_spec(
                "deepseek-chat",
                ModelCapabilities::text().with_final_json_schema(),
            )])
            .build()
            .unwrap_err();
        assert_eq!(error.code(), OpenAiErrorCode::InvalidRequest);
    }

    #[test]
    fn provider_revalidates_profile_and_mode_after_config_setters() {
        let config = MapCredentialResolver::new(BTreeMap::from([
            ("TEA_OPENAI_API_KEY".to_owned(), "sk-test-key".to_owned()),
            ("TEA_OPENAI_MODEL".to_owned(), "router-model".to_owned()),
        ]))
        .resolve()
        .unwrap()
        .with_api_mode(OpenAiApiMode::Responses)
        .with_compatibility_profile(crate::OpenAiCompatibilityProfile::OpenRouter);

        let error = OpenAiProvider::new(Arc::new(config), Vec::new()).unwrap_err();

        assert_eq!(error.code(), OpenAiErrorCode::InvalidRequest);
    }

    #[tokio::test]
    async fn unadvertised_structured_output_fails_before_transport() {
        let config = MapCredentialResolver::new(BTreeMap::from([
            (
                "TEA_OPENAI_BASE_URL".to_owned(),
                "http://127.0.0.1:9/v1".to_owned(),
            ),
            ("TEA_OPENAI_API_KEY".to_owned(), "sk-test-key".to_owned()),
            ("TEA_OPENAI_MODEL".to_owned(), "gpt-test".to_owned()),
        ]))
        .resolve()
        .unwrap();
        let provider = OpenAiProviderBuilder::new()
            .with_config(Arc::new(config))
            .with_catalog(vec![model_spec("gpt-test", ModelCapabilities::text())])
            .build()
            .unwrap();
        let message = CanonicalMessage::user(
            MessageId::from_str("0195a0b1-5e52-74b2-8c25-0aa7aa000055").unwrap(),
            vec![ContentBlock::text("answer as JSON").unwrap()],
            ProtocolTimestamp::from_str("2026-07-23T09:30:12.125Z").unwrap(),
        )
        .unwrap();
        let request = ModelRequest::new(ModelId::from_str("gpt-test").unwrap(), vec![message])
            .unwrap()
            .with_final_output_format(FinalOutputFormat::JsonObject);

        assert_eq!(
            provider.validate_request(&request).unwrap_err().code(),
            ModelFailureCode::InvalidRequest
        );

        let events = provider
            .stream(request, CancellationScope::new())
            .collect::<Vec<_>>()
            .await;

        assert!(matches!(events.first(), Some(ModelEvent::Started(_))));
        assert!(matches!(
            events.last(),
            Some(ModelEvent::Failed(failure))
                if failure.code() == ModelFailureCode::InvalidRequest
        ));
    }

    #[tokio::test]
    async fn schema_with_tools_requires_combined_capability_before_transport() {
        let config = MapCredentialResolver::new(BTreeMap::from([
            (
                "TEA_OPENAI_BASE_URL".to_owned(),
                "http://127.0.0.1:9/v1".to_owned(),
            ),
            ("TEA_OPENAI_API_KEY".to_owned(), "sk-test-key".to_owned()),
            ("TEA_OPENAI_MODEL".to_owned(), "gpt-test".to_owned()),
            (
                "TEA_OPENAI_COMPATIBILITY_PROFILE".to_owned(),
                "open-ai".to_owned(),
            ),
        ]))
        .resolve()
        .unwrap();
        let provider = OpenAiProviderBuilder::new()
            .with_config(Arc::new(config))
            .with_catalog(vec![model_spec(
                "gpt-test",
                ModelCapabilities::text()
                    .with_tools(false)
                    .with_final_json_schema(),
            )])
            .build()
            .unwrap();
        let message = CanonicalMessage::user(
            MessageId::from_str("0195a0b1-5e52-74b2-8c25-0aa7aa000061").unwrap(),
            vec![ContentBlock::text("read a file and answer as JSON").unwrap()],
            ProtocolTimestamp::from_str("2026-07-23T09:30:12.125Z").unwrap(),
        )
        .unwrap();
        let request = ModelRequest::new(ModelId::from_str("gpt-test").unwrap(), vec![message])
            .unwrap()
            .with_tools(
                vec![
                    ModelToolDefinition::new(
                        "read_file",
                        "Reads one file.",
                        serde_json::json!({"type": "object"}),
                    )
                    .unwrap(),
                ],
                false,
            )
            .unwrap()
            .with_final_output_format(FinalOutputFormat::JsonSchema {
                schema: serde_json::json!({
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": false
                }),
            });

        let preflight = provider.validate_request(&request).unwrap_err();
        assert_eq!(preflight.code(), ModelFailureCode::InvalidRequest);
        assert_eq!(
            preflight.message(),
            "model does not support JSON-Schema final output with tools"
        );

        let events = provider
            .stream(request, CancellationScope::new())
            .collect::<Vec<_>>()
            .await;

        assert_conformant_failure(&events, ModelFailureCode::InvalidRequest);
        assert!(matches!(
            events.last(),
            Some(ModelEvent::Failed(failure))
                if failure.message() == "model does not support JSON-Schema final output with tools"
        ));
    }

    #[test]
    fn schema_with_hosted_search_requires_both_hosted_and_combination_capabilities() {
        let config = MapCredentialResolver::new(BTreeMap::from([
            (
                "TEA_OPENAI_BASE_URL".to_owned(),
                "https://api.x.ai/v1".to_owned(),
            ),
            ("TEA_OPENAI_API_KEY".to_owned(), "xai-test-key".to_owned()),
            ("TEA_OPENAI_MODEL".to_owned(), "grok-test".to_owned()),
        ]))
        .resolve()
        .unwrap()
        .with_api_mode(OpenAiApiMode::Responses)
        .with_compatibility_profile(crate::OpenAiCompatibilityProfile::Xai);
        let message = CanonicalMessage::user(
            MessageId::from_str("0195a0b1-5e52-74b2-8c25-0aa7aa000063").unwrap(),
            vec![ContentBlock::text("search and answer as JSON").unwrap()],
            ProtocolTimestamp::from_str("2026-07-23T09:30:12.125Z").unwrap(),
        )
        .unwrap();
        let request = ModelRequest::new(ModelId::from_str("grok-test").unwrap(), vec![message])
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
            .unwrap()
            .with_final_output_format(FinalOutputFormat::JsonSchema {
                schema: serde_json::json!({
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": false
                }),
            });

        let without_hosted_search = OpenAiProviderBuilder::new()
            .with_config(Arc::new(config.clone()))
            .with_catalog(vec![model_spec(
                "grok-test",
                ModelCapabilities::text().with_final_json_schema_with_tools(),
            )])
            .build()
            .unwrap();
        let error = without_hosted_search
            .validate_request(&request)
            .unwrap_err();
        assert_eq!(error.code(), ModelFailureCode::InvalidRequest);
        assert_eq!(
            error.message(),
            "model does not support a requested hosted tool"
        );

        let without_combination = OpenAiProviderBuilder::new()
            .with_config(Arc::new(config.clone()))
            .with_catalog(vec![model_spec(
                "grok-test",
                ModelCapabilities::text()
                    .with_hosted_tool(HostedToolKind::WebSearch)
                    .with_final_json_schema(),
            )])
            .build()
            .unwrap();
        let error = without_combination.validate_request(&request).unwrap_err();
        assert_eq!(error.code(), ModelFailureCode::InvalidRequest);
        assert_eq!(
            error.message(),
            "model does not support JSON-Schema final output with tools"
        );

        let with_both = OpenAiProviderBuilder::new()
            .with_config(Arc::new(config))
            .with_catalog(vec![model_spec(
                "grok-test",
                ModelCapabilities::text()
                    .with_hosted_tool(HostedToolKind::WebSearch)
                    .with_final_json_schema_with_tools(),
            )])
            .build()
            .unwrap();
        with_both.validate_request(&request).unwrap();
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

    fn provider_for_base_url(base_url: String) -> OpenAiProvider {
        let config = MapCredentialResolver::new(BTreeMap::from([
            ("TEA_OPENAI_BASE_URL".to_owned(), base_url),
            ("TEA_OPENAI_API_KEY".to_owned(), "sk-test-key".to_owned()),
            ("TEA_OPENAI_MODEL".to_owned(), "gpt-4o-mini".to_owned()),
        ]))
        .resolve()
        .unwrap();
        OpenAiProviderBuilder::new()
            .with_config(Arc::new(config))
            .build()
            .unwrap()
    }

    fn text_request() -> ModelRequest {
        let message = CanonicalMessage::user(
            MessageId::from_str("0195a0b1-5e52-74b2-8c25-0aa7aa000056").unwrap(),
            vec![ContentBlock::text("hello").unwrap()],
            ProtocolTimestamp::from_str("2026-07-23T09:30:12.125Z").unwrap(),
        )
        .unwrap();
        ModelRequest::new(ModelId::from_str("gpt-4o-mini").unwrap(), vec![message]).unwrap()
    }

    fn assert_conformant_failure(events: &[ModelEvent], code: ModelFailureCode) {
        assert!(matches!(events.first(), Some(ModelEvent::Started(_))));
        assert!(matches!(
            events.last(),
            Some(ModelEvent::Failed(failure)) if failure.code() == code
        ));
        let mut validator = ModelStreamValidator::new();
        for event in events {
            validator.observe(event).unwrap();
        }
        assert!(!validator.finish().unwrap().succeeded());
    }

    async fn chat_events_from_sse(sse: &str) -> Vec<ModelEvent> {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let sse = sse.to_owned();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let _ = read_http_request(&mut stream).await;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{sse}",
                sse.len()
            );
            stream.write_all(response.as_bytes()).await.unwrap();
        });
        let provider = provider_for_base_url(format!("http://{address}/v1"));
        let events = provider
            .stream(text_request(), CancellationScope::new())
            .collect::<Vec<_>>()
            .await;
        server.await.unwrap();
        events
    }

    async fn responses_events_from_sse(sse: &str) -> Vec<ModelEvent> {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let sse = sse.to_owned();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let _ = read_http_request(&mut stream).await;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{sse}",
                sse.len()
            );
            stream.write_all(response.as_bytes()).await.unwrap();
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
            .build()
            .unwrap();
        let message = CanonicalMessage::user(
            MessageId::from_str("0195a0b1-5e52-74b2-8c25-0aa7aa000064").unwrap(),
            vec![ContentBlock::text("hello").unwrap()],
            ProtocolTimestamp::from_str("2026-07-23T09:30:12.125Z").unwrap(),
        )
        .unwrap();
        let request =
            ModelRequest::new(ModelId::from_str("gpt-4.1").unwrap(), vec![message]).unwrap();
        let events = provider
            .stream(request, CancellationScope::new())
            .collect::<Vec<_>>()
            .await;
        server.await.unwrap();
        events
    }

    async fn responses_events_from_split_sse(first: &str, second: &str) -> Vec<ModelEvent> {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let first = first.to_owned();
        let second = second.to_owned();
        let (release_tx, release_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let _ = read_http_request(&mut stream).await;
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
        let config = MapCredentialResolver::new(BTreeMap::from([
            (
                "TEA_OPENAI_BASE_URL".to_owned(),
                format!("http://{address}/v1"),
            ),
            ("TEA_OPENAI_API_KEY".to_owned(), "sk-test-key".to_owned()),
            ("TEA_OPENAI_MODEL".to_owned(), "gpt-4o-mini".to_owned()),
            ("TEA_OPENAI_API_MODE".to_owned(), "responses".to_owned()),
        ]))
        .resolve()
        .unwrap();
        let provider = OpenAiProviderBuilder::new()
            .with_config(Arc::new(config))
            .build()
            .unwrap();
        let mut model_stream = provider.stream(text_request(), CancellationScope::new());
        let first_event = model_stream.next().await.unwrap();
        release_tx.send(()).unwrap();
        let mut events = vec![first_event];
        events.extend(model_stream.collect::<Vec<_>>().await);
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
            let _ = read_http_request(&mut stream).await;
            request_seen_tx.send(()).unwrap();
            std::future::pending::<()>().await;
        });
        let provider = provider_for_base_url(format!("http://{address}/v1"));
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

    #[tokio::test]
    async fn cancellation_interrupts_non_streaming_response_body_read() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (body_started_tx, body_started_rx) = tokio::sync::oneshot::channel();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let _ = read_http_request(&mut stream).await;
            stream
                .write_all(
                    b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 1024\r\nConnection: close\r\n\r\n{",
                )
                .await
                .unwrap();
            body_started_tx.send(()).unwrap();
            std::future::pending::<()>().await;
        });
        let config = MapCredentialResolver::new(BTreeMap::from([
            (
                "TEA_OPENAI_BASE_URL".to_owned(),
                format!("http://{address}/v1"),
            ),
            ("TEA_OPENAI_API_KEY".to_owned(), "sk-test-key".to_owned()),
            ("TEA_OPENAI_MODEL".to_owned(), "groq-model".to_owned()),
        ]))
        .resolve()
        .unwrap()
        .with_compatibility_profile(crate::OpenAiCompatibilityProfile::Groq)
        .with_final_json_schema(true);
        let provider = OpenAiProviderBuilder::new()
            .with_config(Arc::new(config))
            .build()
            .unwrap();
        let message = CanonicalMessage::user(
            MessageId::from_str("0195a0b1-5e52-74b2-8c25-0aa7aa000062").unwrap(),
            vec![ContentBlock::text("answer as JSON").unwrap()],
            ProtocolTimestamp::from_str("2026-07-23T09:30:12.125Z").unwrap(),
        )
        .unwrap();
        let request = ModelRequest::new(ModelId::from_str("groq-model").unwrap(), vec![message])
            .unwrap()
            .with_final_output_format(FinalOutputFormat::JsonSchema {
                schema: serde_json::json!({
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": false
                }),
            });
        let cancellation = CancellationScope::new();
        let model_stream = provider.stream(request, cancellation.clone());
        let collector = tokio::spawn(async move { model_stream.collect::<Vec<_>>().await });

        tokio::time::timeout(Duration::from_secs(1), body_started_rx)
            .await
            .expect("provider did not begin reading the non-streaming body")
            .unwrap();
        cancellation.cancel();
        let events = tokio::time::timeout(Duration::from_secs(1), collector)
            .await
            .expect("cancellation did not interrupt non-streaming body read")
            .unwrap();
        server.abort();
        let _ = server.await;

        assert_conformant_failure(&events, ModelFailureCode::Cancelled);
    }

    #[tokio::test]
    async fn malformed_first_sse_event_is_a_conformant_failed_stream() {
        let events = chat_events_from_sse("data: not-json\n\n").await;

        assert_conformant_failure(&events, ModelFailureCode::MalformedResponse);
    }

    #[tokio::test]
    async fn chat_stream_without_finish_reason_is_a_conformant_failure() {
        let events = chat_events_from_sse(concat!(
            "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},",
            "\"finish_reason\":null}]}\n\n",
            "data: [DONE]\n\n",
        ))
        .await;

        assert_conformant_failure(&events, ModelFailureCode::MalformedResponse);
    }

    #[tokio::test]
    async fn responses_stream_rejects_data_after_completed_before_done() {
        let events = responses_events_from_sse(concat!(
            "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_terminal\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[]}}\n\n",
            "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":\"{}\"}\n\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_terminal\",\"status\":\"completed\",\"error\":null,\"incomplete_details\":null,\"output\":[{\"id\":\"msg_terminal\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"{}\"}]}}}\n\n",
            "data: {\"type\":\"response.incomplete\",\"response\":{\"id\":\"resp_terminal\",\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}\n\n",
            "data: [DONE]\n\n",
        ))
        .await;

        assert_conformant_failure(&events, ModelFailureCode::MalformedResponse);
        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, ModelEvent::Failed(_)))
                .count(),
            1
        );
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, ModelEvent::Completed(_)))
        );
    }

    #[tokio::test]
    async fn responses_stream_rejects_data_after_completed_in_later_http_chunk() {
        let events = responses_events_from_split_sse(
            concat!(
                "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"id\":\"msg_terminal\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[]}}\n\n",
                "data: {\"type\":\"response.output_text.delta\",\"output_index\":0,\"content_index\":0,\"delta\":\"{}\"}\n\n",
                "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_terminal\",\"status\":\"completed\",\"error\":null,\"incomplete_details\":null,\"output\":[{\"id\":\"msg_terminal\",\"type\":\"message\",\"role\":\"assistant\",\"status\":\"completed\",\"content\":[{\"type\":\"output_text\",\"text\":\"{}\"}]}}}\n\n",
            ),
            concat!(
                "data: {\"type\":\"response.incomplete\",\"response\":{\"id\":\"resp_terminal\",\"status\":\"incomplete\",\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}\n\n",
                "data: [DONE]\n\n",
            ),
        )
        .await;

        assert_conformant_failure(&events, ModelFailureCode::MalformedResponse);
        assert!(
            !events
                .iter()
                .any(|event| matches!(event, ModelEvent::Completed(_)))
        );
    }

    #[test]
    fn user_agent_rejects_empty_and_invalid_header_values() {
        assert!(UserAgent::new("").is_err());
        assert!(UserAgent::new("tea-cli/1.0\r\ninvalid").is_err());
    }

    #[test]
    fn hosted_search_capability_requires_explicit_responses_advertisement() {
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
            !provider.models()[0]
                .capabilities()
                .supports_hosted_tool(HostedToolKind::WebSearch)
        );
        assert!(
            provider.models()[1]
                .capabilities()
                .supports_hosted_tool(HostedToolKind::WebSearch)
        );

        let chat_config = provider
            .config()
            .clone()
            .with_api_mode(OpenAiApiMode::ChatCompletions);
        let error = OpenAiProviderBuilder::new()
            .with_config(Arc::new(chat_config))
            .with_catalog(vec![model_spec(
                "gpt-4.1",
                ModelCapabilities::text().with_hosted_tool(HostedToolKind::WebSearch),
            )])
            .build()
            .unwrap_err();
        assert_eq!(error.code(), OpenAiErrorCode::InvalidRequest);
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
    async fn groq_schema_posts_non_streaming_chat_and_normalizes_full_response() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let request = read_http_request(&mut stream).await;
            let body = serde_json::to_string(&serde_json::json!({
                "id": "chatcmpl-groq-http",
                "object": "chat.completion",
                "model": "groq-model",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "{\"answer\":\"ok\"}",
                        "refusal": null
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 5,
                    "total_tokens": 13
                }
            }))
            .unwrap();
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
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
            ("TEA_OPENAI_MODEL".to_owned(), "groq-model".to_owned()),
        ]))
        .resolve()
        .unwrap()
        .with_compatibility_profile(crate::OpenAiCompatibilityProfile::Groq)
        .with_final_json_schema(true);
        let provider = OpenAiProviderBuilder::new()
            .with_config(Arc::new(config))
            .build()
            .unwrap();
        let message = CanonicalMessage::user(
            MessageId::from_str("0195a0b1-5e52-74b2-8c25-0aa7aa000060").unwrap(),
            vec![ContentBlock::text("answer as JSON").unwrap()],
            ProtocolTimestamp::from_str("2026-07-23T09:30:12.125Z").unwrap(),
        )
        .unwrap();
        let request = ModelRequest::new(ModelId::from_str("groq-model").unwrap(), vec![message])
            .unwrap()
            .with_final_output_format(FinalOutputFormat::JsonSchema {
                schema: serde_json::json!({
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": false
                }),
            });
        let events = provider
            .stream(request, CancellationScope::new())
            .collect::<Vec<_>>()
            .await;

        let mut validator = ModelStreamValidator::new();
        for event in &events {
            validator.observe(event).unwrap();
        }
        assert!(validator.finish().unwrap().succeeded());
        assert_eq!(
            events
                .iter()
                .filter_map(ModelEvent::as_text_delta)
                .collect::<String>(),
            "{\"answer\":\"ok\"}"
        );
        let captured = String::from_utf8(server.await.unwrap()).unwrap();
        assert!(captured.starts_with("POST /v1/chat/completions HTTP/1.1\r\n"));
        let body: serde_json::Value =
            serde_json::from_str(captured.split_once("\r\n\r\n").unwrap().1).unwrap();
        assert_eq!(body["stream"], false);
        assert!(body.get("stream_options").is_none());
        assert_eq!(body["response_format"]["type"], "json_schema");
        assert_eq!(body["response_format"]["json_schema"]["strict"], true);
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
                "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_http_test\",\"status\":\"completed\",\"output\":[{\"id\":\"msg_http_test\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"hello\",\"annotations\":[]}]}],\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n",
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
                "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_http_search\",\"status\":\"completed\",\"output\":[{\"id\":\"ws_http_search\",\"type\":\"web_search_call\",\"status\":\"completed\",\"action\":{\"type\":\"search\",\"queries\":[\"tea-rs\"],\"sources\":[{\"type\":\"url\",\"url\":\"https://example.com/search\"}]}}],\"usage\":{\"input_tokens\":1,\"output_tokens\":1,\"total_tokens\":2}}}\n\n",
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
