use std::collections::BTreeMap;

use tea_model::HostedToolKind;
use tea_provider_anthropic::{
    AnthropicError, AnthropicErrorCode,
    catalog::default_catalog,
    credential::{
        ApiKey, CredentialResolver, DEFAULT_WEB_SEARCH_MAX_USES, DEFAULT_WEB_SEARCH_TOOL_TYPE,
        MapCredentialResolver,
    },
};

fn env_map(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| ((*key).to_owned(), (*value).to_owned()))
        .collect()
}

#[test]
fn credential_contract_is_bounded_redacted_and_configured() {
    let key = ApiKey::new("sk-ant-test").unwrap();
    assert_eq!(key.as_str(), "sk-ant-test");
    assert!(!format!("{key:?}").contains("ant-test"));

    let config = MapCredentialResolver::new(env_map(&[
        ("TEA_ANTHROPIC_API_KEY", "sk-ant-test"),
        ("TEA_ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        ("TEA_ANTHROPIC_BASE_URL", "https://gateway.example.test"),
        ("TEA_ANTHROPIC_API_VERSION", "2023-06-01"),
    ]))
    .resolve()
    .unwrap();
    assert_eq!(config.provider_id().as_str(), "anthropic");
    assert_eq!(config.base_url(), "https://gateway.example.test");
    assert_eq!(config.model_id().as_str(), "claude-sonnet-4-20250514");
    assert_eq!(config.timeout_millis(), 60_000);
    assert_eq!(
        config.web_search().tool_type(),
        DEFAULT_WEB_SEARCH_TOOL_TYPE
    );
    assert_eq!(config.web_search().max_uses(), DEFAULT_WEB_SEARCH_MAX_USES);
    assert!(!config.supports_final_json_schema());
    assert!(
        default_catalog(&config).unwrap()[0]
            .capabilities()
            .supports_hosted_tool(HostedToolKind::WebSearch)
    );
    assert!(
        !default_catalog(&config).unwrap()[0]
            .capabilities()
            .supports_final_json_schema()
    );
    assert!(
        !default_catalog(&config).unwrap()[0]
            .capabilities()
            .supports_final_json_schema_with_tools()
    );

    let error = MapCredentialResolver::new(BTreeMap::new())
        .resolve()
        .unwrap_err();
    assert_eq!(error.code(), AnthropicErrorCode::Authentication);
}

#[test]
fn final_json_schema_capability_requires_explicit_model_endpoint_opt_in() {
    for value in ["1", "true", "TRUE"] {
        let config = MapCredentialResolver::new(env_map(&[
            ("TEA_ANTHROPIC_API_KEY", "sk-ant-test"),
            ("TEA_ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            ("TEA_ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
            ("TEA_ANTHROPIC_FINAL_JSON_SCHEMA", value),
        ]))
        .resolve()
        .unwrap();

        assert!(config.supports_final_json_schema());
        assert!(
            default_catalog(&config).unwrap()[0]
                .capabilities()
                .supports_final_json_schema()
        );
        assert!(
            default_catalog(&config).unwrap()[0]
                .capabilities()
                .supports_final_json_schema_with_tools()
        );
    }

    for value in [None, Some("0"), Some("false"), Some("yes")] {
        let mut values = env_map(&[
            ("TEA_ANTHROPIC_API_KEY", "sk-ant-test"),
            ("TEA_ANTHROPIC_MODEL", "deepseek-chat"),
            (
                "TEA_ANTHROPIC_BASE_URL",
                "https://api.deepseek.example/anthropic",
            ),
        ]);
        if let Some(value) = value {
            values.insert(
                "TEA_ANTHROPIC_FINAL_JSON_SCHEMA".to_owned(),
                value.to_owned(),
            );
        }
        let config = MapCredentialResolver::new(values).resolve().unwrap();

        assert!(!config.supports_final_json_schema());
        assert!(
            !default_catalog(&config).unwrap()[0]
                .capabilities()
                .supports_final_json_schema()
        );
        assert!(
            !default_catalog(&config).unwrap()[0]
                .capabilities()
                .supports_final_json_schema_with_tools()
        );
    }
}

#[test]
fn invalid_explicit_web_search_configuration_fails_closed() {
    for (key, value) in [
        ("TEA_ANTHROPIC_WEB_SEARCH_TOOL_TYPE", "web_search_latest"),
        ("TEA_ANTHROPIC_WEB_SEARCH_MAX_USES", "not-a-number"),
        ("TEA_ANTHROPIC_WEB_SEARCH_MAX_USES", "0"),
        ("TEA_ANTHROPIC_WEB_SEARCH_MAX_USES", "101"),
    ] {
        let mut values = env_map(&[
            ("TEA_ANTHROPIC_API_KEY", "sk-ant-test"),
            ("TEA_ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
        ]);
        values.insert(key.to_owned(), value.to_owned());

        let error = MapCredentialResolver::new(values).resolve().unwrap_err();

        assert_eq!(error.code(), AnthropicErrorCode::InvalidRequest);
    }
}

#[test]
fn errors_remain_safe_and_classified() {
    for code in [
        AnthropicErrorCode::InvalidRequest,
        AnthropicErrorCode::Authentication,
        AnthropicErrorCode::PermissionDenied,
        AnthropicErrorCode::RateLimited,
        AnthropicErrorCode::Unavailable,
        AnthropicErrorCode::Transport,
        AnthropicErrorCode::MalformedResponse,
        AnthropicErrorCode::ContextOverflow,
        AnthropicErrorCode::Cancelled,
        AnthropicErrorCode::Internal,
    ] {
        let error = AnthropicError::new(code, "example\0error");
        assert_eq!(error.code(), code);
        assert!(!error.message().contains('\0'));
    }
}
