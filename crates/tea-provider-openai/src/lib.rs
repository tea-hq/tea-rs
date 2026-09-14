#![forbid(unsafe_code)]
#![deny(missing_docs)]

//! OpenAI-compatible streaming provider adapter for `tea-rs`.
//!
//! Implements `tea_model::ModelProvider` over `OpenAI` Chat Completions and
//! Responses APIs with `Server-Sent-Events` streaming. The adapter is
//! gateway-agnostic: real `OpenAI` (`api.openai.com/v1`) and `OpenAI`-compatible
//! gateways are served by changing `TEA_OPENAI_BASE_URL` and optional
//! headers. No secret is stored by the adapter; a `CredentialResolver` returns
//! configuration at request time.

pub mod catalog;
pub mod credential;
pub mod env_file;
pub mod error;
pub mod provider;
pub mod reasoning;
pub mod request;
pub mod responses;
mod responses_model;
pub mod responses_stream;
pub mod sse;
pub mod stream;

pub use credential::{
    ApiKey, CredentialResolver, EnvCredentialResolver, MapCredentialResolver, OpenAiApiMode,
    OpenAiCompatibilityProfile, OpenAiConfig, PROVIDER_ID,
};
pub use error::{OpenAiError, OpenAiErrorCode};
pub use provider::{OpenAiProvider, OpenAiProviderBuilder};
pub use reasoning::OpenAiReasoningEffortMap;
pub use tea_provider_http::{ProviderHttpConfig, UserAgent};
