use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use tea_coding::AppPaths;
use tea_coding::config::{
    ClientWebSearchSettingsLayer, CodingSettings, HostedToolCapability, ProvidersConfigLoadError,
    ResourceSettingsLayer, SettingsLayer, TuiSettingsLayer, WebFetchCacheSettingsLayer,
    WebFetchSettingsLayer, WebSearchLocationSettingsLayer, WebSearchRoutePreference,
    WebSearchSettingsLayer, load_providers_file, load_settings_file, merge_settings,
    persist_global_model_settings,
};
use tea_coding::mcp_config::{
    MAX_CONFIGURED_MCP_SERVERS, McpLifecycleSettings, McpLimitsSettings, McpServerSettings,
    McpToolSettings, McpTransportSettings,
};
use tea_protocol::{ModelRef, ReasoningEffort};
use tea_provider_openai::OpenAiCompatibilityProfile;

static ID: AtomicU64 = AtomicU64::new(0);

fn mcp_server(id: &str, executable: &str, alias: Option<&str>) -> McpServerSettings {
    McpServerSettings {
        id: id.to_owned(),
        transport: McpTransportSettings::Stdio {
            executable: PathBuf::from(executable),
            arguments: Vec::new(),
        },
        inherited_environment: Vec::new(),
        tools: vec![McpToolSettings {
            remote_name: "ping".to_owned(),
            alias: alias.map(str::to_owned),
            declaration: None,
        }],
        limits: McpLimitsSettings::default(),
        lifecycle: McpLifecycleSettings::default(),
        reconnect: None,
    }
}

#[test]
fn precedence_and_nested_merge_are_deterministic_and_secret_free() {
    let global = SettingsLayer {
        model: Some("global/model".to_owned()),
        tui: Some(TuiSettingsLayer {
            viewport: Some("inline".to_owned()),
            ..Default::default()
        }),
        resources: Some(ResourceSettingsLayer {
            context_files: Some(false),
            ..Default::default()
        }),
        ..Default::default()
    };
    let project = SettingsLayer {
        model: Some("project/model".to_owned()),
        tui: Some(TuiSettingsLayer {
            collapse_thinking: Some(true),
            reduced_motion: Some(true),
            ..Default::default()
        }),
        ..Default::default()
    };
    let environment = SettingsLayer {
        model: Some("env/model".to_owned()),
        ..Default::default()
    };
    let cli = SettingsLayer {
        model: Some("cli/model".to_owned()),
        max_retries: Some(4),
        retry_base_delay_ms: Some(750),
        retry_max_delay_ms: Some(12_000),
        ..Default::default()
    };
    let resolved = merge_settings(
        CodingSettings::default(),
        Some(&global),
        Some(&project),
        Some(&environment),
        Some(&cli),
    )
    .unwrap();
    assert_eq!(resolved.model, "cli/model");
    assert_eq!(resolved.tui.viewport, "inline");
    assert!(resolved.tui.collapse_thinking);
    assert!(resolved.tui.reduced_motion);
    assert!(!resolved.resources.context_files);
    assert_eq!(resolved.max_retries, 4);
    assert_eq!(resolved.retry_base_delay_ms, 750);
    assert_eq!(resolved.retry_max_delay_ms, 12_000);
    let json = serde_json::to_string(&resolved).unwrap();
    assert!(!json.contains("\"apiKey\":"));
    assert!(!json.contains("\"secret\":"));
    assert!(
        merge_settings(
            CodingSettings::default(),
            None,
            None,
            None,
            Some(&SettingsLayer {
                max_retries: Some(8),
                ..Default::default()
            }),
        )
        .is_err()
    );
    assert!(
        merge_settings(
            CodingSettings::default(),
            None,
            None,
            None,
            Some(&SettingsLayer {
                tui: Some(TuiSettingsLayer {
                    submit_key: Some("not+a+key".to_owned()),
                    ..Default::default()
                }),
                ..Default::default()
            }),
        )
        .is_err()
    );
}

#[test]
fn retry_defaults_and_delay_bounds_are_pi_style_and_validated() {
    let defaults = CodingSettings::default();
    assert_eq!(defaults.max_retries, 3);
    assert_eq!(defaults.retry_base_delay_ms, 2_000);
    assert_eq!(defaults.retry_max_delay_ms, 60_000);

    for layer in [
        SettingsLayer {
            retry_base_delay_ms: Some(0),
            ..Default::default()
        },
        SettingsLayer {
            retry_base_delay_ms: Some(4_000),
            retry_max_delay_ms: Some(2_000),
            ..Default::default()
        },
        SettingsLayer {
            retry_base_delay_ms: Some(30_001),
            ..Default::default()
        },
        SettingsLayer {
            retry_max_delay_ms: Some(300_001),
            ..Default::default()
        },
    ] {
        assert!(merge_settings(CodingSettings::default(), None, None, None, Some(&layer)).is_err());
    }
}

#[test]
fn search_tools_are_valid_active_tool_configuration() {
    let resolved = merge_settings(
        CodingSettings::default(),
        None,
        None,
        None,
        Some(&SettingsLayer {
            active_tools: Some(vec!["grep".to_owned(), "find".to_owned(), "ls".to_owned()]),
            ..Default::default()
        }),
    )
    .unwrap();
    assert_eq!(resolved.active_tools, ["grep", "find", "ls"]);
}

#[test]
fn web_search_defaults_do_not_expand_the_active_tool_allowlist() {
    let defaults = CodingSettings::default();
    assert_eq!(defaults.active_tools, ["read", "write", "edit", "bash"]);
    assert!(!defaults.web_search.client.enabled);
    assert_eq!(
        serde_json::to_value(defaults.web_search.client.backend).unwrap(),
        "tavily"
    );
    assert_eq!(
        defaults.web_search.route_preference,
        WebSearchRoutePreference::PreferHosted
    );

    let resolved = merge_settings(
        defaults,
        None,
        None,
        None,
        Some(&SettingsLayer {
            web_search: Some(WebSearchSettingsLayer {
                client: Some(ClientWebSearchSettingsLayer {
                    enabled: Some(true),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            ..Default::default()
        }),
    )
    .unwrap();
    assert!(resolved.web_search.client.enabled);
    assert_eq!(resolved.active_tools, ["read", "write", "edit", "bash"]);
}

#[test]
fn web_fetch_defaults_and_backend_configuration_do_not_expand_active_tools() {
    let defaults = CodingSettings::default();
    assert_eq!(defaults.active_tools, ["read", "write", "edit", "bash"]);
    assert!(!defaults.web_fetch.enabled);
    assert_eq!(
        serde_json::to_value(defaults.web_fetch.backend).unwrap(),
        "http"
    );

    let resolved = merge_settings(
        defaults,
        None,
        None,
        None,
        Some(&SettingsLayer {
            web_fetch: Some(WebFetchSettingsLayer {
                enabled: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        }),
    )
    .unwrap();
    assert!(resolved.web_fetch.enabled);
    assert_eq!(resolved.active_tools, ["read", "write", "edit", "bash"]);
}

#[test]
fn web_fetch_layers_merge_bounded_cache_controls() {
    let global = SettingsLayer {
        web_fetch: Some(WebFetchSettingsLayer {
            enabled: Some(true),
            cache: Some(WebFetchCacheSettingsLayer {
                ttl_seconds: Some(120),
                max_entries: Some(12),
                ..Default::default()
            }),
            ..Default::default()
        }),
        ..Default::default()
    };
    let project = SettingsLayer {
        web_fetch: Some(WebFetchSettingsLayer {
            cache: Some(WebFetchCacheSettingsLayer {
                max_total_bytes: Some(256 * 1024),
                ..Default::default()
            }),
            ..Default::default()
        }),
        ..Default::default()
    };
    let cli = SettingsLayer {
        web_fetch: Some(WebFetchSettingsLayer {
            cache: Some(WebFetchCacheSettingsLayer {
                max_entry_bytes: Some(64 * 1024),
                ..Default::default()
            }),
            ..Default::default()
        }),
        ..Default::default()
    };

    let resolved = merge_settings(
        CodingSettings::default(),
        Some(&global),
        Some(&project),
        None,
        Some(&cli),
    )
    .unwrap();
    assert!(resolved.web_fetch.enabled);
    assert_eq!(resolved.web_fetch.cache.ttl_seconds, 120);
    assert_eq!(resolved.web_fetch.cache.max_entries, 12);
    assert_eq!(resolved.web_fetch.cache.max_total_bytes, 256 * 1024);
    assert_eq!(resolved.web_fetch.cache.max_entry_bytes, 64 * 1024);
}

#[test]
fn web_fetch_configuration_rejects_invalid_cache_or_backend_values() {
    let invalid_cache = |cache| {
        merge_settings(
            CodingSettings::default(),
            None,
            None,
            None,
            Some(&SettingsLayer {
                web_fetch: Some(WebFetchSettingsLayer {
                    cache: Some(cache),
                    ..Default::default()
                }),
                ..Default::default()
            }),
        )
        .is_err()
    };
    assert!(invalid_cache(WebFetchCacheSettingsLayer {
        ttl_seconds: Some(0),
        ..Default::default()
    }));
    assert!(invalid_cache(WebFetchCacheSettingsLayer {
        max_entries: Some(0),
        ..Default::default()
    }));
    assert!(invalid_cache(WebFetchCacheSettingsLayer {
        max_total_bytes: Some(1024),
        max_entry_bytes: Some(2048),
        ..Default::default()
    }));
    assert!(
        serde_json::from_str::<SettingsLayer>(
            r#"{"webFetch":{"enabled":true,"backend":"unknown"}}"#
        )
        .is_err()
    );
}

#[test]
fn web_search_layers_merge_nested_controls_without_credentials() {
    let global = SettingsLayer {
        web_search: Some(WebSearchSettingsLayer {
            allowed_domains: Some(vec![
                "docs.example.com".to_owned(),
                "example.com".to_owned(),
            ]),
            location: Some(WebSearchLocationSettingsLayer {
                country: Some("CN".to_owned()),
                region: Some("Zhejiang".to_owned()),
                ..Default::default()
            }),
            client: Some(ClientWebSearchSettingsLayer {
                enabled: Some(true),
                api_key_environment: Some("SEARCH_API_KEY".to_owned()),
                ..Default::default()
            }),
            ..Default::default()
        }),
        ..Default::default()
    };
    let project = SettingsLayer {
        web_search: Some(WebSearchSettingsLayer {
            location: Some(WebSearchLocationSettingsLayer {
                city: Some("Hangzhou".to_owned()),
                timezone: Some("Asia/Shanghai".to_owned()),
                ..Default::default()
            }),
            ..Default::default()
        }),
        ..Default::default()
    };
    let environment = SettingsLayer {
        web_search: Some(WebSearchSettingsLayer {
            client: Some(ClientWebSearchSettingsLayer {
                timeout_millis: Some(12_000),
                ..Default::default()
            }),
            ..Default::default()
        }),
        ..Default::default()
    };
    let cli = SettingsLayer {
        web_search: Some(WebSearchSettingsLayer {
            route_preference: Some(WebSearchRoutePreference::ForceClient),
            ..Default::default()
        }),
        ..Default::default()
    };

    let resolved = merge_settings(
        CodingSettings::default(),
        Some(&global),
        Some(&project),
        Some(&environment),
        Some(&cli),
    )
    .unwrap();
    assert_eq!(
        resolved.web_search.allowed_domains,
        ["docs.example.com", "example.com"]
    );
    let location = resolved.web_search.location.as_ref().unwrap();
    assert_eq!(location.country.as_deref(), Some("CN"));
    assert_eq!(location.city.as_deref(), Some("Hangzhou"));
    assert_eq!(location.region.as_deref(), Some("Zhejiang"));
    assert_eq!(location.timezone.as_deref(), Some("Asia/Shanghai"));
    assert_eq!(resolved.web_search.client.timeout_millis, 12_000);
    assert_eq!(encoded_backend(&resolved), serde_json::json!("tavily"));
    assert_eq!(
        resolved.web_search.route_preference,
        WebSearchRoutePreference::ForceClient
    );

    let encoded = serde_json::to_value(&resolved).unwrap();
    assert_eq!(
        encoded["webSearch"]["client"]["apiKeyEnvironment"],
        "SEARCH_API_KEY"
    );
    assert!(encoded["webSearch"]["client"].get("apiKey").is_none());
}

fn encoded_backend(settings: &CodingSettings) -> serde_json::Value {
    serde_json::to_value(settings).unwrap()["webSearch"]["client"]["backend"].clone()
}

#[test]
fn web_search_configuration_rejects_unsafe_or_ambiguous_controls() {
    let invalid = |web_search| {
        merge_settings(
            CodingSettings::default(),
            Some(&SettingsLayer {
                web_search: Some(web_search),
                ..Default::default()
            }),
            None,
            None,
            None,
        )
        .is_err()
    };

    assert!(invalid(WebSearchSettingsLayer {
        allowed_domains: Some(vec!["Example.COM".to_owned()]),
        ..Default::default()
    }));
    assert!(invalid(WebSearchSettingsLayer {
        allowed_domains: Some(vec!["example.com".to_owned()]),
        blocked_domains: Some(vec!["blocked.example".to_owned()]),
        ..Default::default()
    }));
    assert!(invalid(WebSearchSettingsLayer {
        location: Some(WebSearchLocationSettingsLayer {
            country: Some("cn".to_owned()),
            ..Default::default()
        }),
        ..Default::default()
    }));
    assert!(invalid(WebSearchSettingsLayer {
        location: Some(WebSearchLocationSettingsLayer {
            timezone: Some("Asia/Shanghai?token=secret".to_owned()),
            ..Default::default()
        }),
        ..Default::default()
    }));
    assert!(invalid(WebSearchSettingsLayer {
        client: Some(ClientWebSearchSettingsLayer {
            endpoint: Some("http://search.example.com/query".to_owned()),
            ..Default::default()
        }),
        ..Default::default()
    }));
    assert!(invalid(WebSearchSettingsLayer {
        client: Some(ClientWebSearchSettingsLayer {
            endpoint: Some("https://user:secret@search.example.com/query".to_owned()),
            ..Default::default()
        }),
        ..Default::default()
    }));
    assert!(invalid(WebSearchSettingsLayer {
        client: Some(ClientWebSearchSettingsLayer {
            api_key_environment: Some("SEARCH-API-KEY".to_owned()),
            ..Default::default()
        }),
        ..Default::default()
    }));
    assert!(invalid(WebSearchSettingsLayer {
        client: Some(ClientWebSearchSettingsLayer {
            timeout_millis: Some(0),
            ..Default::default()
        }),
        ..Default::default()
    }));
    assert!(invalid(WebSearchSettingsLayer {
        client: Some(ClientWebSearchSettingsLayer {
            timeout_millis: Some(60_001),
            ..Default::default()
        }),
        ..Default::default()
    }));
    assert!(invalid(WebSearchSettingsLayer {
        route_preference: Some(WebSearchRoutePreference::ForceClient),
        ..Default::default()
    }));
}

#[test]
fn resolved_tui_settings_fill_new_keybindings_for_v1_documents() {
    let settings: tea_coding::config::TuiSettings = serde_json::from_str(
        r#"{"viewport":"inline","collapseThinking":true,"submitKey":"enter"}"#,
    )
    .unwrap();
    assert_eq!(settings.newline_key, "shift+enter");
    assert_eq!(settings.follow_up_key, "alt+enter");
    assert_eq!(settings.retrieve_queued_key, "ctrl+r");
    assert!(!settings.reduced_motion);
}

#[test]
fn untrusted_project_layer_has_no_effect_when_omitted() {
    let project = SettingsLayer {
        model: Some("project/model".to_owned()),
        ..Default::default()
    };
    let untrusted = merge_settings(CodingSettings::default(), None, None, None, None).unwrap();
    let trusted =
        merge_settings(CodingSettings::default(), None, Some(&project), None, None).unwrap();
    assert_ne!(untrusted.model, trusted.model);
    assert_eq!(trusted.model, "project/model");
}

#[test]
fn strict_settings_files_and_injected_paths_never_require_home() {
    let root = std::env::temp_dir().join(format!(
        "tea-coding-config-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&root).unwrap();
    let paths = AppPaths::new(root.join("config"), root.join("state"), root.join("data")).unwrap();
    fs::create_dir_all(paths.config_dir()).unwrap();
    fs::write(
        paths.settings_file(),
        br#"{"schemaVersion":1,"model":"file/model","tui":{"viewport":"inline"}}"#,
    )
    .unwrap();
    let layer = load_settings_file(&paths.settings_file()).unwrap().unwrap();
    assert_eq!(layer.model.as_deref(), Some("file/model"));
    assert_eq!(paths.providers_file(), root.join("config/providers.json"));
    fs::write(paths.settings_file(), br#"{"unknown":true}"#).unwrap();
    assert!(load_settings_file(&paths.settings_file()).is_err());
    fs::write(
        paths.settings_file(),
        br#"{"webSearch":{"client":{"backend":"unknown"}}}"#,
    )
    .unwrap();
    assert!(load_settings_file(&paths.settings_file()).is_err());
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn global_model_persistence_preserves_sparse_unrelated_settings() {
    let root = std::env::temp_dir().join(format!(
        "tea-coding-global-settings-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&root).unwrap();
    let path = root.join("settings.json");
    fs::write(
        &path,
        br#"{
            "schemaVersion": 1,
            "provider": "compatible",
            "model": "old-model",
            "thinking": "low",
            "activeTools": ["read"],
            "maxRetries": 4,
            "tui": {"reducedMotion": true}
        }"#,
    )
    .unwrap();

    persist_global_model_settings(
        &path,
        &ModelRef::new("compatible".parse().unwrap(), "new-model".parse().unwrap()),
        ReasoningEffort::ExtraHigh,
    )
    .unwrap();

    let document: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    assert_eq!(document["schemaVersion"], 1);
    assert_eq!(document["provider"], "compatible");
    assert_eq!(document["model"], "new-model");
    assert_eq!(document["thinking"], "xhigh");
    assert_eq!(document["activeTools"], serde_json::json!(["read"]));
    assert_eq!(document["maxRetries"], 4);
    assert_eq!(document["tui"]["reducedMotion"], true);
    assert_eq!(document.as_object().unwrap().len(), 7);
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn global_model_persistence_creates_a_private_versioned_file() {
    let root = std::env::temp_dir().join(format!(
        "tea-coding-new-global-settings-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&root).unwrap();
    let path = root.join("settings.json");
    persist_global_model_settings(
        &path,
        &ModelRef::new(
            "provider".parse().unwrap(),
            "provider/model".parse().unwrap(),
        ),
        ReasoningEffort::Maximum,
    )
    .unwrap();

    let document: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    assert_eq!(document["schemaVersion"], 1);
    assert_eq!(document["provider"], "provider");
    assert_eq!(document["model"], "provider/model");
    assert_eq!(document["thinking"], "max");
    assert_eq!(document.as_object().unwrap().len(), 4);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;

        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn failed_global_model_persistence_leaves_old_file_intact() {
    let root = std::env::temp_dir().join(format!(
        "tea-coding-bounded-global-settings-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&root).unwrap();
    let path = root.join("settings.json");
    let padding = "x".repeat(tea_coding::config::MAX_SETTINGS_FILE_BYTES - 64);
    let original = format!(r#"{{"schemaVersion":1,"sessionDatabase":"{padding}"}}"#).into_bytes();
    assert!(original.len() <= tea_coding::config::MAX_SETTINGS_FILE_BYTES);
    fs::write(&path, &original).unwrap();

    assert!(
        persist_global_model_settings(
            &path,
            &ModelRef::new(
                "provider".parse().unwrap(),
                "provider/model".parse().unwrap(),
            ),
            ReasoningEffort::High,
        )
        .is_err()
    );
    assert_eq!(fs::read(&path).unwrap(), original);
    assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
    fs::remove_dir_all(root).unwrap();
}

#[test]
#[allow(clippy::too_many_lines)]
fn provider_files_are_strict_soft_failing_and_field_merged() {
    let root = std::env::temp_dir().join(format!(
        "tea-coding-providers-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&root).unwrap();
    let global_path = root.join("global.json");
    let project_path = root.join("project.json");
    fs::write(
        &global_path,
        br#"{
            "providers": {
                "deepseek": {
                    "base_url": "https://api.deepseek.test/v1",
                    "api_key": "$DEEPSEEK_API_KEY",
                    "compatibility_profile": "deep-seek",
                    "models": [{
                        "id": "deepseek-chat",
                        "capabilities": {
                            "final_json_object": true,
                            "hosted_tools": ["web_search"]
                        }
                    }]
                }
            }
        }"#,
    )
    .unwrap();
    fs::write(
        &project_path,
        br#"{
            "providers": {
                "deepseek": {
                    "api_mode": "responses",
                    "compatibility_profile": "open-ai",
                    "timeout_millis": 30000
                }
            }
        }"#,
    )
    .unwrap();
    let global = load_providers_file(&global_path);
    let project = load_providers_file(&project_path);
    assert_eq!(global.error, None);
    assert_eq!(project.error, None);
    let merged = global.config.merged(project.config);
    let provider = &merged.providers["deepseek"];
    assert_eq!(
        provider.base_url.as_deref(),
        Some("https://api.deepseek.test/v1")
    );
    assert_eq!(provider.api_mode.as_deref(), Some("responses"));
    assert_eq!(
        provider.compatibility_profile,
        Some(OpenAiCompatibilityProfile::OpenAi)
    );
    assert_eq!(provider.models[0].id, "deepseek-chat");
    assert!(provider.models[0].capabilities.final_json_object);
    assert!(!provider.models[0].capabilities.final_json_schema);
    assert!(!provider.models[0].capabilities.final_json_schema_with_tools);
    assert_eq!(
        provider.models[0].capabilities.hosted_tools,
        [HostedToolCapability::WebSearch]
    );
    assert!(!format!("{merged:?}").contains("DEEPSEEK_API_KEY"));
    assert!(format!("{merged:?}").contains("OpenAi"));

    fs::write(&global_path, br#"{"providers":{},"unknown":true}"#).unwrap();
    let invalid = load_providers_file(&global_path);
    assert_eq!(invalid.error, Some(ProvidersConfigLoadError::Invalid));
    assert!(invalid.config.providers.is_empty());
    fs::write(
        &global_path,
        br#"{"providers":{"deepseek":{"compatibility_profile":"unknown"}}}"#,
    )
    .unwrap();
    assert_eq!(
        load_providers_file(&global_path).error,
        Some(ProvidersConfigLoadError::Invalid)
    );
    fs::write(&global_path, vec![b' '; 256 * 1024 + 1]).unwrap();
    assert_eq!(
        load_providers_file(&global_path).error,
        Some(ProvidersConfigLoadError::TooLarge)
    );

    fs::write(
        &global_path,
        br#"{"providers":{"deepseek":{"models":[{"id":"deepseek-chat","capabilities":{"hosted_tools":["unknown"]}}]}}}"#,
    )
    .unwrap();
    assert_eq!(
        load_providers_file(&global_path).error,
        Some(ProvidersConfigLoadError::Invalid)
    );
    fs::write(
        &global_path,
        br#"{"providers":{"deepseek":{"models":[{"id":"deepseek-chat","capabilities":{"hosted_tools":["web_search","web_search"]}}]}}}"#,
    )
    .unwrap();
    assert_eq!(
        load_providers_file(&global_path).error,
        Some(ProvidersConfigLoadError::Invalid)
    );
    fs::write(
        &global_path,
        br#"{"providers":{"compatible":{"compatibility_profile":"open-ai","models":[{"id":"test","capabilities":{"final_json_schema_with_tools":true}}]}}}"#,
    )
    .unwrap();
    assert_eq!(
        load_providers_file(&global_path).error,
        Some(ProvidersConfigLoadError::Invalid)
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn model_reasoning_profiles_resolve_identity_custom_and_disabled_levels() {
    let root = std::env::temp_dir().join(format!(
        "tea-coding-model-reasoning-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&root).unwrap();
    let path = root.join("providers.json");
    fs::write(
        &path,
        br#"{
            "providers": {
                "compatible": {
                    "reasoning_effort": "max",
                    "models": [
                        {
                            "id": "identity",
                            "capabilities": {
                                "reasoning": {"default_effort": "medium"}
                            }
                        },
                        {
                            "id": "custom",
                            "capabilities": {
                                "reasoning": {
                                    "default_effort": "medium",
                                    "effort_map": {
                                        "minimal": "tiny",
                                        "high": null,
                                        "xhigh": "ultra",
                                        "max": null
                                    }
                                }
                            }
                        }
                    ]
                }
            }
        }"#,
    )
    .unwrap();

    let loaded = load_providers_file(&path);
    assert_eq!(loaded.error, None);
    let provider = &loaded.config.providers["compatible"];
    assert_eq!(provider.reasoning_effort.as_deref(), Some("max"));

    let (identity_profile, identity_map) = provider.models[0]
        .capabilities
        .reasoning
        .as_ref()
        .unwrap()
        .resolved()
        .unwrap();
    assert_eq!(identity_profile.default_effort(), ReasoningEffort::Medium);
    assert_eq!(
        identity_profile.supported_efforts(),
        &[
            ReasoningEffort::Off,
            ReasoningEffort::Minimal,
            ReasoningEffort::Low,
            ReasoningEffort::Medium,
            ReasoningEffort::High,
        ]
    );
    assert_eq!(identity_map[&ReasoningEffort::Minimal], "minimal");
    assert_eq!(identity_map[&ReasoningEffort::High], "high");
    assert!(!identity_map.contains_key(&ReasoningEffort::ExtraHigh));
    assert!(!identity_map.contains_key(&ReasoningEffort::Maximum));

    let (custom_profile, custom_map) = provider.models[1]
        .capabilities
        .reasoning
        .as_ref()
        .unwrap()
        .resolved()
        .unwrap();
    assert_eq!(
        custom_profile.supported_efforts(),
        &[
            ReasoningEffort::Off,
            ReasoningEffort::Minimal,
            ReasoningEffort::Low,
            ReasoningEffort::Medium,
            ReasoningEffort::ExtraHigh,
        ]
    );
    assert_eq!(custom_map[&ReasoningEffort::Minimal], "tiny");
    assert_eq!(custom_map[&ReasoningEffort::ExtraHigh], "ultra");
    assert!(!custom_map.contains_key(&ReasoningEffort::High));
    assert!(!custom_map.contains_key(&ReasoningEffort::Maximum));
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn invalid_model_reasoning_profiles_fail_closed() {
    let root = std::env::temp_dir().join(format!(
        "tea-coding-invalid-model-reasoning-{}-{}",
        std::process::id(),
        ID.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&root).unwrap();
    let path = root.join("providers.json");
    let invalid_reasoning = [
        r#"{"default_effort":"xhigh"}"#,
        r#"{"default_effort":"high","effort_map":{"high":null}}"#,
        r#"{"default_effort":"medium","effort_map":{"off":"disabled"}}"#,
        r#"{"default_effort":"medium","effort_map":{"extreme":"extreme"}}"#,
        r#"{"default_effort":"medium","effort_map":{"high":"not valid"}}"#,
    ];
    for reasoning in invalid_reasoning {
        let document = format!(
            r#"{{"providers":{{"compatible":{{"models":[{{"id":"test","capabilities":{{"reasoning":{reasoning}}}}}]}}}}}}"#
        );
        fs::write(&path, document).unwrap();
        assert_eq!(
            load_providers_file(&path).error,
            Some(ProvidersConfigLoadError::Invalid),
            "unexpectedly accepted {reasoning}"
        );
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn mcp_servers_merge_by_id_across_all_precedence_layers() {
    let global = SettingsLayer {
        mcp_servers: Some(vec![
            mcp_server("alpha", "/global/alpha", None),
            mcp_server("global-only", "/global/only", None),
        ]),
        ..Default::default()
    };
    let project = SettingsLayer {
        mcp_servers: Some(vec![mcp_server("alpha", "/project/alpha", None)]),
        ..Default::default()
    };
    let environment = SettingsLayer {
        mcp_servers: Some(vec![mcp_server("alpha", "/environment/alpha", None)]),
        ..Default::default()
    };
    let cli = SettingsLayer {
        mcp_servers: Some(vec![mcp_server("alpha", "/cli/alpha", None)]),
        ..Default::default()
    };

    let resolved = merge_settings(
        CodingSettings::default(),
        Some(&global),
        Some(&project),
        Some(&environment),
        Some(&cli),
    )
    .unwrap();

    assert_eq!(resolved.mcp_servers.len(), 2);
    assert_eq!(resolved.mcp_servers[0].id().as_str(), "alpha");
    assert_eq!(
        resolved.mcp_servers[0].transport().as_stdio().executable(),
        std::path::Path::new("/cli/alpha")
    );
    assert_eq!(resolved.mcp_servers[1].id().as_str(), "global-only");
}

#[test]
fn mcp_settings_fail_closed_on_alias_collisions_paths_and_bounds() {
    let duplicate_aliases = SettingsLayer {
        mcp_servers: Some(vec![
            mcp_server("alpha", "/servers/alpha", Some("shared.alias")),
            mcp_server("beta", "/servers/beta", Some("shared.alias")),
        ]),
        ..Default::default()
    };
    assert!(
        merge_settings(
            CodingSettings::default(),
            Some(&duplicate_aliases),
            None,
            None,
            None,
        )
        .is_err()
    );

    let duplicate_ids = SettingsLayer {
        mcp_servers: Some(vec![
            mcp_server("duplicate", "/servers/first", None),
            mcp_server("duplicate", "/servers/second", None),
        ]),
        ..Default::default()
    };
    assert!(
        merge_settings(
            CodingSettings::default(),
            Some(&duplicate_ids),
            None,
            None,
            None,
        )
        .is_err()
    );

    let mut invalid_environment = mcp_server("environment", "/servers/environment", None);
    invalid_environment.inherited_environment = vec!["TOKEN".to_owned(), "token".to_owned()];
    let invalid_environment = SettingsLayer {
        mcp_servers: Some(vec![invalid_environment]),
        ..Default::default()
    };
    assert!(
        merge_settings(
            CodingSettings::default(),
            Some(&invalid_environment),
            None,
            None,
            None,
        )
        .is_err()
    );

    let relative = SettingsLayer {
        mcp_servers: Some(vec![mcp_server("relative", "server", None)]),
        ..Default::default()
    };
    assert!(merge_settings(CodingSettings::default(), Some(&relative), None, None, None).is_err());

    let oversized = SettingsLayer {
        mcp_servers: Some(
            (0..=MAX_CONFIGURED_MCP_SERVERS)
                .map(|index| mcp_server(&format!("server-{index}"), "/server", None))
                .collect(),
        ),
        ..Default::default()
    };
    assert!(
        merge_settings(
            CodingSettings::default(),
            Some(&oversized),
            None,
            None,
            None,
        )
        .is_err()
    );
}

#[test]
fn omitted_untrusted_project_cannot_discover_mcp_servers() {
    let project = SettingsLayer {
        mcp_servers: Some(vec![mcp_server("project", "/project/server", None)]),
        ..Default::default()
    };
    let ignored = merge_settings(CodingSettings::default(), None, None, None, None).unwrap();
    let trusted =
        merge_settings(CodingSettings::default(), None, Some(&project), None, None).unwrap();

    assert!(ignored.mcp_servers.is_empty());
    assert_eq!(trusted.mcp_servers.len(), 1);
}
