use std::collections::BTreeMap;
use std::ffi::OsString;
use std::sync::Arc;

use tea_coding::config::{
    CodingSettings, ProviderConfig, SettingsLayer, merge_settings,
    resolve_openai_compatible_provider,
};
use tea_coding::mcp_config::{
    McpEnvironmentErrorCode, McpLifecycleSettings, McpLimitsSettings, McpServerSettings,
    McpTransportSettings, StaticMcpEnvironmentResolver, resolve_mcp_environment,
};
use tea_coding::{CodingCredentialResolver, CodingErrorCode, McpEnvironmentResolver};
use tea_model::{HostedToolKind, ModelProvider as _};
use tea_provider_openai::{MapCredentialResolver, OpenAiApiMode, OpenAiCompatibilityProfile};

#[test]
fn credential_debug_and_settings_serialization_never_expose_secrets() {
    let secret = "sk-super-secret-value";
    let resolver = MapCredentialResolver::new(BTreeMap::from([
        ("TEA_OPENAI_API_KEY".to_owned(), secret.to_owned()),
        ("TEA_OPENAI_MODEL".to_owned(), "gpt-test".to_owned()),
    ]));
    assert!(!format!("{resolver:?}").contains(secret));
    let coding = CodingCredentialResolver::new(Arc::new(resolver));
    assert!(!format!("{coding:?}").contains(secret));
    let config = coding.resolve().unwrap();
    assert_eq!(config.api_key().as_str(), secret);
    assert!(!format!("{config:?}").contains(secret));
}

#[test]
fn builtin_openai_custom_base_url_does_not_inherit_the_openai_profile() {
    let resolved = resolve_openai_compatible_provider(
        "openai",
        "test-model",
        None,
        None,
        BTreeMap::from([
            (
                "TEA_OPENAI_BASE_URL".to_owned(),
                "https://gateway.example.test/v1".to_owned(),
            ),
            ("TEA_OPENAI_API_KEY".to_owned(), "test-key".to_owned()),
        ]),
    )
    .unwrap();
    let provider = resolved.provider_builder().build().unwrap();

    assert_eq!(provider.config().compatibility_profile(), None);
}

#[test]
fn builtin_openai_hosted_web_search_opt_in_reaches_the_default_catalog() {
    let resolved = resolve_openai_compatible_provider(
        "openai",
        "search-model",
        None,
        None,
        BTreeMap::from([
            ("TEA_OPENAI_API_KEY".to_owned(), "test-key".to_owned()),
            ("TEA_OPENAI_API_MODE".to_owned(), "responses".to_owned()),
            ("TEA_OPENAI_HOSTED_WEB_SEARCH".to_owned(), "true".to_owned()),
        ]),
    )
    .unwrap();
    let provider = resolved.provider_builder().build().unwrap();

    assert!(
        provider.models()[0]
            .capabilities()
            .supports_hosted_tool(HostedToolKind::WebSearch)
    );
}

fn structured_provider(
    profile: Option<&str>,
    api_mode: &str,
    final_json_object: bool,
    final_json_schema: bool,
    final_json_schema_with_tools: bool,
) -> ProviderConfig {
    serde_json::from_value(serde_json::json!({
        "base_url": "https://compatible.example/v1",
        "api_key": "test-key",
        "api_mode": api_mode,
        "compatibility_profile": profile,
        "models": [{
            "id": "test-model",
            "capabilities": {
                "final_json_object": final_json_object,
                "final_json_schema": final_json_schema,
                "final_json_schema_with_tools": final_json_schema_with_tools
            }
        }]
    }))
    .unwrap()
}

type StructuredOutputProfileCase = (
    &'static str,
    OpenAiCompatibilityProfile,
    &'static str,
    bool,
    bool,
);

const STRUCTURED_OUTPUT_PROFILE_CASES: &[StructuredOutputProfileCase] = &[
    (
        "open-ai",
        OpenAiCompatibilityProfile::OpenAi,
        "chat-completions",
        true,
        true,
    ),
    (
        "open-ai",
        OpenAiCompatibilityProfile::OpenAi,
        "responses",
        true,
        true,
    ),
    (
        "azure-open-ai",
        OpenAiCompatibilityProfile::AzureOpenAi,
        "chat-completions",
        true,
        true,
    ),
    (
        "azure-open-ai",
        OpenAiCompatibilityProfile::AzureOpenAi,
        "responses",
        true,
        true,
    ),
    (
        "deep-seek",
        OpenAiCompatibilityProfile::DeepSeek,
        "chat-completions",
        true,
        false,
    ),
    (
        "deep-seek",
        OpenAiCompatibilityProfile::DeepSeek,
        "responses",
        true,
        true,
    ),
    (
        "gemini-open-ai",
        OpenAiCompatibilityProfile::GeminiOpenAi,
        "chat-completions",
        true,
        true,
    ),
    (
        "ollama-local",
        OpenAiCompatibilityProfile::OllamaLocal,
        "chat-completions",
        true,
        true,
    ),
    (
        "open-router",
        OpenAiCompatibilityProfile::OpenRouter,
        "chat-completions",
        true,
        true,
    ),
    (
        "vllm",
        OpenAiCompatibilityProfile::Vllm,
        "chat-completions",
        true,
        true,
    ),
    (
        "vllm",
        OpenAiCompatibilityProfile::Vllm,
        "responses",
        true,
        true,
    ),
];

#[test]
fn structured_output_profile_matrix_is_projected_into_model_capabilities() {
    for &(profile_name, expected_profile, api_mode, object, schema) in
        STRUCTURED_OUTPUT_PROFILE_CASES
    {
        let configured = structured_provider(Some(profile_name), api_mode, object, schema, false);
        let resolved = resolve_openai_compatible_provider(
            "compatible",
            "test-model",
            Some(&configured),
            None,
            BTreeMap::new(),
        )
        .unwrap();
        let provider = resolved.provider_builder().build().unwrap();
        assert_eq!(
            provider.config().compatibility_profile(),
            Some(expected_profile),
            "profile={profile_name}, mode={api_mode}"
        );
        assert_eq!(
            provider.config().api_mode(),
            if api_mode == "responses" {
                OpenAiApiMode::Responses
            } else {
                OpenAiApiMode::ChatCompletions
            }
        );
        let capabilities = provider.models()[0].capabilities();
        assert_eq!(
            capabilities.supports_final_json_object(),
            object,
            "profile={profile_name}, mode={api_mode}"
        );
        assert_eq!(
            capabilities.supports_final_json_schema(),
            schema,
            "profile={profile_name}, mode={api_mode}"
        );
    }
}

#[test]
fn custom_provider_without_profile_keeps_structured_output_disabled() {
    let configured = structured_provider(None, "responses", false, false, false);
    let resolved = resolve_openai_compatible_provider(
        "openai",
        "test-model",
        Some(&configured),
        None,
        BTreeMap::new(),
    )
    .unwrap();
    let provider = resolved.provider_builder().build().unwrap();
    assert_eq!(provider.config().compatibility_profile(), None);
    let capabilities = provider.models()[0].capabilities();
    assert!(!capabilities.supports_final_json_object());
    assert!(!capabilities.supports_final_json_schema());
}

#[test]
fn custom_provider_profile_does_not_imply_model_capabilities() {
    for (profile, api_mode) in [
        ("open-ai", "chat-completions"),
        ("open-ai", "responses"),
        ("open-router", "chat-completions"),
    ] {
        let configured = structured_provider(Some(profile), api_mode, false, false, false);
        let resolved = resolve_openai_compatible_provider(
            "compatible",
            "test-model",
            Some(&configured),
            None,
            BTreeMap::new(),
        )
        .unwrap();
        let provider = resolved.provider_builder().build().unwrap();
        let capabilities = provider.models()[0].capabilities();

        assert!(!capabilities.supports_final_json_object());
        assert!(!capabilities.supports_final_json_schema());
    }
}

#[test]
fn structured_output_profile_mode_and_capability_mismatches_fail_closed() {
    let cases = [
        (Some("deep-seek"), "chat-completions", false, true),
        (Some("gemini-open-ai"), "responses", true, false),
        (Some("ollama-local"), "responses", true, false),
        (Some("open-router"), "responses", true, false),
        (None, "chat-completions", true, false),
        (None, "responses", false, true),
    ];

    for (profile, api_mode, object, schema) in cases {
        let configured = structured_provider(profile, api_mode, object, schema, false);
        let error = resolve_openai_compatible_provider(
            "compatible",
            "test-model",
            Some(&configured),
            None,
            BTreeMap::new(),
        )
        .expect_err("invalid profile/mode/capability combination");
        assert_eq!(
            error.code(),
            CodingErrorCode::InvalidInput,
            "profile={profile:?}, mode={api_mode}"
        );
    }

    let configured = structured_provider(None, "chat-completions", false, true, false);
    let error = resolve_openai_compatible_provider(
        "openai",
        "test-model",
        Some(&configured),
        None,
        BTreeMap::new(),
    )
    .expect_err("providers.json must not inherit the built-in OpenAI profile");
    assert_eq!(error.code(), CodingErrorCode::InvalidInput);
}

#[test]
fn configured_schema_with_tools_capability_is_explicit_and_consistent() {
    let configured = structured_provider(Some("open-ai"), "chat-completions", false, true, true);
    let resolved = resolve_openai_compatible_provider(
        "compatible",
        "test-model",
        Some(&configured),
        None,
        BTreeMap::new(),
    )
    .unwrap();
    let capabilities = resolved.provider_builder().build().unwrap().models()[0].capabilities();
    assert!(capabilities.supports_final_json_schema());
    assert!(capabilities.supports_final_json_schema_with_tools());

    let invalid = structured_provider(Some("open-ai"), "chat-completions", false, false, true);
    let error = resolve_openai_compatible_provider(
        "compatible",
        "test-model",
        Some(&invalid),
        None,
        BTreeMap::new(),
    )
    .expect_err("schema-with-tools requires base schema support");
    assert_eq!(error.code(), CodingErrorCode::InvalidInput);

    for (profile, api_mode) in [("groq", "chat-completions"), ("vllm", "responses")] {
        let incompatible = structured_provider(Some(profile), api_mode, false, true, true);
        let error = resolve_openai_compatible_provider(
            "compatible",
            "test-model",
            Some(&incompatible),
            None,
            BTreeMap::new(),
        )
        .expect_err("profile must support schema with tools");
        assert_eq!(error.code(), CodingErrorCode::InvalidInput);
    }
}

fn environment_server(names: Vec<String>) -> tea_mcp::McpServerConfig {
    let layer = SettingsLayer {
        mcp_servers: Some(vec![McpServerSettings {
            id: "environment".to_owned(),
            transport: McpTransportSettings::Stdio {
                executable: "/usr/bin/env".into(),
                arguments: Vec::new(),
            },
            inherited_environment: names,
            tools: Vec::new(),
            limits: McpLimitsSettings::default(),
            lifecycle: McpLifecycleSettings::default(),
            reconnect: None,
        }]),
        ..Default::default()
    };
    merge_settings(CodingSettings::default(), Some(&layer), None, None, None)
        .unwrap()
        .mcp_servers
        .remove(0)
}

#[test]
fn mcp_environment_values_are_late_bounded_and_redacting() {
    let secret = "mcp-secret-value-that-must-never-persist";
    let resolver = StaticMcpEnvironmentResolver::new(BTreeMap::from([
        ("ALLOWED_TOKEN".to_owned(), OsString::from(secret)),
        (
            "UNRELATED_SECRET".to_owned(),
            OsString::from("must-not-be-inherited"),
        ),
    ]))
    .unwrap();
    assert!(!format!("{resolver:?}").contains(secret));

    let server = environment_server(vec!["ALLOWED_TOKEN".to_owned()]);
    assert!(!format!("{server:?}").contains(secret));
    let environment = resolve_mcp_environment(&server, &resolver).unwrap();
    assert_eq!(environment.names().collect::<Vec<_>>(), ["ALLOWED_TOKEN"]);
    assert!(!format!("{environment:?}").contains(secret));

    #[cfg(unix)]
    {
        let mut command = std::process::Command::new("/usr/bin/env");
        command.env("AMBIENT_SECRET", "must-be-cleared");
        environment.apply_to(&mut command);
        let output = command.output().unwrap();
        assert!(output.status.success());
        let stdout = String::from_utf8(output.stdout).unwrap();
        assert!(stdout.contains(&format!("ALLOWED_TOKEN={secret}")));
        assert!(!stdout.contains("UNRELATED_SECRET"));
        assert!(!stdout.contains("AMBIENT_SECRET"));
    }
}

#[test]
fn mcp_environment_missing_errors_expose_only_name_and_code() {
    let secret = "missing-secret-must-not-appear";
    let resolver = StaticMcpEnvironmentResolver::new(BTreeMap::from([(
        "OTHER".to_owned(),
        OsString::from(secret),
    )]))
    .unwrap();
    let server = environment_server(vec!["REQUIRED_TOKEN".to_owned()]);
    let error = resolve_mcp_environment(&server, &resolver).unwrap_err();

    assert_eq!(error.code(), McpEnvironmentErrorCode::MissingVariable);
    assert_eq!(error.name(), "REQUIRED_TOKEN");
    assert!(format!("{error:?}").contains("REQUIRED_TOKEN"));
    assert!(!format!("{error:?}").contains(secret));
    let object: &dyn McpEnvironmentResolver = &resolver;
    assert!(object.resolve("REQUIRED_TOKEN").unwrap().is_none());
}
