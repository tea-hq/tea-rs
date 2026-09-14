//! Contract tests for the `OpenAI` provider errors, credential resolver, and env loader.

use std::collections::BTreeMap;
use std::str::FromStr as _;

use tea_model::{HostedToolKind, ProviderId};
use tea_provider_openai::{
    OpenAiCompatibilityProfile,
    catalog::default_catalog,
    credential::{ApiKey, CredentialResolver, MapCredentialResolver, OpenAiApiMode, OpenAiConfig},
    env_file::load_env_file,
    error::{OpenAiError, OpenAiErrorCode},
};

fn env_map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
        .collect()
}

#[test]
fn error_codes_are_stable() {
    for code in [
        OpenAiErrorCode::InvalidRequest,
        OpenAiErrorCode::Authentication,
        OpenAiErrorCode::PermissionDenied,
        OpenAiErrorCode::RateLimited,
        OpenAiErrorCode::Unavailable,
        OpenAiErrorCode::Transport,
        OpenAiErrorCode::MalformedResponse,
        OpenAiErrorCode::ContextOverflow,
        OpenAiErrorCode::Cancelled,
        OpenAiErrorCode::Internal,
    ] {
        let error = OpenAiError::new(code, "example");
        assert_eq!(error.code(), code);
        assert!(!error.message().is_empty());
        assert!(!error.message().contains('\0'));
    }
}

#[test]
fn api_key_is_bounded_and_redacts_in_debug() {
    let key = ApiKey::new("sk-test-1234567890").unwrap();
    assert_eq!(key.as_str(), "sk-test-1234567890");
    assert!(!format!("{key:?}").contains("1234567890"));
}

#[test]
fn explicit_config_uses_safe_connection_defaults() {
    let config = OpenAiConfig::new(
        "example-model".parse().unwrap(),
        "https://gateway.example.test/v1",
        ApiKey::new("test-key").unwrap(),
    )
    .unwrap();

    assert_eq!(config.provider_id().as_str(), "openai");
    assert_eq!(config.model_id().as_str(), "example-model");
    assert_eq!(config.base_url(), "https://gateway.example.test/v1");
    assert_eq!(config.api_key_header(), "Authorization");
    assert_eq!(config.api_key_prefix(), "Bearer ");
    assert_eq!(config.api_mode(), OpenAiApiMode::ChatCompletions);
    assert_eq!(config.compatibility_profile(), None);
    assert!(!config.final_json_object());
    assert!(!config.final_json_schema());
    assert!(!config.final_json_schema_with_tools());
    assert_eq!(config.timeout_millis(), 60_000);
}

#[test]
fn explicit_config_requires_an_explicit_structured_output_profile() {
    let config = OpenAiConfig::new(
        "example-model".parse().unwrap(),
        "https://gateway.example.test/v1",
        ApiKey::new("test-key").unwrap(),
    )
    .unwrap();
    assert_eq!(config.compatibility_profile(), None);

    let config = config.with_compatibility_profile(OpenAiCompatibilityProfile::OpenAi);
    assert_eq!(
        config.compatibility_profile(),
        Some(OpenAiCompatibilityProfile::OpenAi)
    );
}

#[test]
fn structured_output_capability_setters_preserve_combination_invariant() {
    let config = OpenAiConfig::new(
        "example-model".parse().unwrap(),
        "https://gateway.example.test/v1",
        ApiKey::new("test-key").unwrap(),
    )
    .unwrap()
    .with_final_json_schema_with_tools(true);
    assert!(config.final_json_schema());
    assert!(config.final_json_schema_with_tools());

    let config = config.with_final_json_schema(false);
    assert!(!config.final_json_schema());
    assert!(!config.final_json_schema_with_tools());
}

#[test]
fn compatibility_profile_matrix_is_explicit() {
    let profiles = [
        OpenAiCompatibilityProfile::OpenAi,
        OpenAiCompatibilityProfile::AzureOpenAi,
        OpenAiCompatibilityProfile::Xai,
        OpenAiCompatibilityProfile::DeepSeek,
        OpenAiCompatibilityProfile::GeminiOpenAi,
        OpenAiCompatibilityProfile::OllamaLocal,
        OpenAiCompatibilityProfile::OpenRouter,
        OpenAiCompatibilityProfile::Groq,
        OpenAiCompatibilityProfile::Mistral,
        OpenAiCompatibilityProfile::Together,
        OpenAiCompatibilityProfile::Vllm,
    ];
    for profile in profiles {
        assert!(profile.supports_api_mode(OpenAiApiMode::ChatCompletions));
        assert!(profile.supports_final_json_object(OpenAiApiMode::ChatCompletions));
        assert_eq!(
            profile.supports_final_json_schema(OpenAiApiMode::ChatCompletions),
            profile != OpenAiCompatibilityProfile::DeepSeek
        );
        let responses = matches!(
            profile,
            OpenAiCompatibilityProfile::OpenAi
                | OpenAiCompatibilityProfile::AzureOpenAi
                | OpenAiCompatibilityProfile::Xai
                | OpenAiCompatibilityProfile::DeepSeek
                | OpenAiCompatibilityProfile::Groq
                | OpenAiCompatibilityProfile::Vllm
        );
        assert_eq!(
            profile.supports_api_mode(OpenAiApiMode::Responses),
            responses
        );
        assert_eq!(
            profile.supports_final_json_object(OpenAiApiMode::Responses),
            responses && profile != OpenAiCompatibilityProfile::Groq
        );
        assert_eq!(
            profile.supports_final_json_schema(OpenAiApiMode::Responses),
            responses && profile != OpenAiCompatibilityProfile::Groq
        );
        for api_mode in [OpenAiApiMode::ChatCompletions, OpenAiApiMode::Responses] {
            let schema = profile.supports_final_json_schema(api_mode);
            assert_eq!(
                profile.supports_final_json_schema_with_tools(api_mode),
                schema
                    && profile != OpenAiCompatibilityProfile::Groq
                    && !(profile == OpenAiCompatibilityProfile::Vllm
                        && api_mode == OpenAiApiMode::Responses)
            );
            assert_eq!(
                profile.supports_streaming_final_json_schema(api_mode),
                schema && profile != OpenAiCompatibilityProfile::Groq
            );
        }
        assert_eq!(
            profile.requires_parameter_support(),
            profile == OpenAiCompatibilityProfile::OpenRouter
        );
        assert_eq!(
            profile.supports_parallel_tools_with_json_schema(),
            !matches!(
                profile,
                OpenAiCompatibilityProfile::AzureOpenAi | OpenAiCompatibilityProfile::Groq
            )
        );
    }
}

#[test]
fn compatibility_profile_names_round_trip_exactly() {
    for (profile, name) in [
        (OpenAiCompatibilityProfile::OpenAi, "open-ai"),
        (OpenAiCompatibilityProfile::AzureOpenAi, "azure-open-ai"),
        (OpenAiCompatibilityProfile::Xai, "xai"),
        (OpenAiCompatibilityProfile::DeepSeek, "deep-seek"),
        (OpenAiCompatibilityProfile::GeminiOpenAi, "gemini-open-ai"),
        (OpenAiCompatibilityProfile::OllamaLocal, "ollama-local"),
        (OpenAiCompatibilityProfile::OpenRouter, "open-router"),
        (OpenAiCompatibilityProfile::Groq, "groq"),
        (OpenAiCompatibilityProfile::Mistral, "mistral"),
        (OpenAiCompatibilityProfile::Together, "together"),
        (OpenAiCompatibilityProfile::Vllm, "vllm"),
    ] {
        let encoded = serde_json::to_string(&profile).unwrap();
        assert_eq!(encoded, format!("\"{name}\""));
        assert_eq!(
            serde_json::from_str::<OpenAiCompatibilityProfile>(&encoded).unwrap(),
            profile
        );
        assert_eq!(name.parse::<OpenAiCompatibilityProfile>().unwrap(), profile);
    }
}

#[test]
fn explicit_config_rejects_an_empty_base_url() {
    let error = OpenAiConfig::new(
        "example-model".parse().unwrap(),
        "",
        ApiKey::new("test-key").unwrap(),
    )
    .unwrap_err();

    assert_eq!(error.code(), OpenAiErrorCode::InvalidRequest);
}

#[test]
fn map_resolver_reads_contract() {
    let map = env_map(&[
        ("TEA_OPENAI_BASE_URL", "https://gateway.example.test/v1"),
        ("TEA_OPENAI_API_KEY", "sk-test-key"),
        ("TEA_OPENAI_MODEL", "gpt-4o-mini"),
        ("TEA_OPENAI_API_KEY_HEADER", "x-api-key"),
        ("TEA_OPENAI_API_KEY_PREFIX", "__NONE__"),
    ]);
    let config = MapCredentialResolver::new(map).resolve().unwrap();
    assert_eq!(config.base_url(), "https://gateway.example.test/v1");
    assert_eq!(config.api_key().as_str(), "sk-test-key");
    assert_eq!(config.api_key_header(), "x-api-key");
    assert_eq!(config.api_key_prefix(), "");
    assert_eq!(config.api_mode(), OpenAiApiMode::ChatCompletions);
    assert_eq!(config.model_id().as_str(), "gpt-4o-mini");
    assert_eq!(config.compatibility_profile(), None);
}

#[test]
fn map_resolver_selects_responses_api() {
    let map = env_map(&[
        ("TEA_OPENAI_API_KEY", "sk-key"),
        ("TEA_OPENAI_MODEL", "gpt-4o"),
        ("TEA_OPENAI_API_MODE", "responses"),
    ]);
    let config = MapCredentialResolver::new(map).resolve().unwrap();
    assert_eq!(config.api_mode(), OpenAiApiMode::Responses);
}

#[test]
fn map_resolver_reads_explicit_profile_and_model_capability_opt_ins() {
    let map = env_map(&[
        ("TEA_OPENAI_API_KEY", "sk-key"),
        ("TEA_OPENAI_MODEL", "router-model"),
        ("TEA_OPENAI_COMPATIBILITY_PROFILE", "open-router"),
        ("TEA_OPENAI_FINAL_JSON_OBJECT", "1"),
        ("TEA_OPENAI_FINAL_JSON_SCHEMA", "TrUe"),
        ("TEA_OPENAI_FINAL_JSON_SCHEMA_WITH_TOOLS", "true"),
    ]);
    let config = MapCredentialResolver::new(map).resolve().unwrap();

    assert_eq!(
        config.compatibility_profile(),
        Some(OpenAiCompatibilityProfile::OpenRouter)
    );
    assert!(config.final_json_object());
    assert!(config.final_json_schema());
    assert!(config.final_json_schema_with_tools());
    let model = default_catalog(&config).unwrap().remove(0);
    assert!(model.capabilities().supports_final_json_object());
    assert!(model.capabilities().supports_final_json_schema());
    assert!(model.capabilities().supports_final_json_schema_with_tools());
}

#[test]
fn map_resolver_reads_explicit_hosted_web_search_opt_in() {
    let map = env_map(&[
        ("TEA_OPENAI_API_KEY", "sk-key"),
        ("TEA_OPENAI_MODEL", "search-model"),
        ("TEA_OPENAI_API_MODE", "responses"),
        ("TEA_OPENAI_HOSTED_WEB_SEARCH", "true"),
    ]);
    let config = MapCredentialResolver::new(map).resolve().unwrap();

    assert!(config.hosted_web_search());
    let model = default_catalog(&config).unwrap().remove(0);
    assert!(
        model
            .capabilities()
            .supports_hosted_tool(HostedToolKind::WebSearch)
    );
}

#[test]
fn model_capability_opt_ins_fail_closed_for_every_other_value() {
    for value in ["", "0", "false", "yes", "on", "2", " true "] {
        let map = env_map(&[
            ("TEA_OPENAI_API_KEY", "sk-key"),
            ("TEA_OPENAI_MODEL", "gpt-4o-mini"),
            ("TEA_OPENAI_FINAL_JSON_OBJECT", value),
            ("TEA_OPENAI_FINAL_JSON_SCHEMA", value),
            ("TEA_OPENAI_FINAL_JSON_SCHEMA_WITH_TOOLS", value),
            ("TEA_OPENAI_HOSTED_WEB_SEARCH", value),
        ]);
        let config = MapCredentialResolver::new(map).resolve().unwrap();

        assert!(
            !config.final_json_object(),
            "unexpected object opt-in: {value:?}"
        );
        assert!(
            !config.final_json_schema(),
            "unexpected schema opt-in: {value:?}"
        );
        assert!(
            !config.final_json_schema_with_tools(),
            "unexpected schema-with-tools opt-in: {value:?}"
        );
        assert!(
            !config.hosted_web_search(),
            "unexpected hosted-search opt-in: {value:?}"
        );
        let model = default_catalog(&config).unwrap().remove(0);
        assert!(!model.capabilities().supports_final_json_object());
        assert!(!model.capabilities().supports_final_json_schema());
        assert!(!model.capabilities().supports_final_json_schema_with_tools());
    }
}

#[test]
fn map_resolver_rejects_unknown_or_mode_incompatible_profile() {
    for (profile, mode) in [
        ("unknown", "chat-completions"),
        ("anthropic", "chat-completions"),
        ("open-router", "responses"),
        ("mistral", "responses"),
        ("together", "responses"),
    ] {
        let map = env_map(&[
            ("TEA_OPENAI_API_KEY", "sk-key"),
            ("TEA_OPENAI_MODEL", "test-model"),
            ("TEA_OPENAI_COMPATIBILITY_PROFILE", profile),
            ("TEA_OPENAI_API_MODE", mode),
        ]);
        let error = MapCredentialResolver::new(map).resolve().unwrap_err();
        assert_eq!(error.code(), OpenAiErrorCode::InvalidRequest);
    }
}

#[test]
fn map_resolver_rejects_unknown_api_mode() {
    let map = env_map(&[
        ("TEA_OPENAI_API_KEY", "sk-key"),
        ("TEA_OPENAI_MODEL", "gpt-4o"),
        ("TEA_OPENAI_API_MODE", "legacy"),
    ]);
    let error = MapCredentialResolver::new(map).resolve().unwrap_err();
    assert_eq!(error.code(), OpenAiErrorCode::InvalidRequest);
}

#[test]
fn map_resolver_defaults_openai_shape() {
    let map = env_map(&[
        ("TEA_OPENAI_API_KEY", "sk-key"),
        ("TEA_OPENAI_MODEL", "gpt-4o"),
    ]);
    let config = MapCredentialResolver::new(map).resolve().unwrap();
    assert_eq!(config.base_url(), "https://api.openai.com/v1");
    assert_eq!(config.api_key_header(), "Authorization");
    assert_eq!(config.api_key_prefix(), "Bearer ");
    assert_eq!(
        config.compatibility_profile(),
        Some(OpenAiCompatibilityProfile::OpenAi)
    );
    assert_eq!(config.timeout_millis(), 60_000);
}

#[test]
fn map_resolver_supports_custom_provider_identity() {
    let map = env_map(&[
        ("TEA_OPENAI_API_KEY", "custom-key"),
        ("TEA_OPENAI_MODEL", "custom-model"),
    ]);
    let config =
        MapCredentialResolver::for_provider(ProviderId::from_str("deepseek").unwrap(), map)
            .resolve()
            .unwrap();
    assert_eq!(config.provider_id().as_str(), "deepseek");
    assert!(config.compatibility_profile().is_none());
}

#[test]
fn map_resolver_requires_key_and_model() {
    let map = env_map(&[("TEA_OPENAI_BASE_URL", "https://x.test/v1")]);
    let err = MapCredentialResolver::new(map).resolve().unwrap_err();
    assert_eq!(err.code(), OpenAiErrorCode::Authentication);
}

#[test]
fn load_env_file_parses_key_value_lines() {
    let path = std::env::temp_dir().join("tea-openai-env-contract.env");
    std::fs::write(
        &path,
        "# a comment\n\nTEA_OPENAI_BASE_URL=https://example.test/v1\nTEA_OPENAI_API_KEY=\"sk-quoted\"\nTEA_OPENAI_MODEL=gpt-4o-mini\n",
    )
    .unwrap();
    let map = load_env_file(&path).unwrap();
    assert_eq!(
        map.get("TEA_OPENAI_BASE_URL"),
        Some(&"https://example.test/v1".to_owned())
    );
    assert_eq!(map.get("TEA_OPENAI_API_KEY"), Some(&"sk-quoted".to_owned()));
    assert_eq!(map.get("TEA_OPENAI_MODEL"), Some(&"gpt-4o-mini".to_owned()));
    let _ = std::fs::remove_file(&path);
}

#[allow(dead_code)]
fn _send_sync<R: CredentialResolver + Send + Sync>(_r: R) {}
