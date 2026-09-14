use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use clap::{CommandFactory as _, Parser as _};
use tea_cli::args::{CliArgs, SessionSelection, TrustArg};
use tea_cli::{BootstrapEnvironment, CliBootstrap, ExitCategory};
use tea_protocol::{
    FinalOutputFormat, MAX_FINAL_OUTPUT_SCHEMA_BYTES, MAX_FINAL_OUTPUT_SCHEMA_DEPTH,
    ReasoningEffort,
};

#[test]
fn shared_flags_parse_and_resolve_session_selection() {
    let args = CliArgs::try_parse_from([
        "tea",
        "--print",
        "--cwd",
        "/tmp",
        "--provider",
        "openai",
        "--model",
        "openai/test",
        "--reasoning-effort",
        "xhigh",
        "--api-key",
        "sk-cli-test",
        "--profile",
        "coding-agent",
        "--tools",
        "read,edit",
        "--context-file",
        "CONTEXT.md",
        "--no-session",
        "--trust",
        "ignore",
        "-vv",
        "hello",
    ])
    .unwrap();
    assert!(args.print);
    assert!(!args.json);
    assert!(!args.rpc);
    assert_eq!(args.tools, ["read", "edit"]);
    assert_eq!(args.context_files, ["CONTEXT.md"]);
    assert_eq!(
        args.reasoning_effort.map(ReasoningEffort::as_str),
        Some("xhigh")
    );
    assert_eq!(args.api_key.as_ref().unwrap().as_str(), "sk-cli-test");
    assert!(!format!("{args:?}").contains("sk-cli-test"));
    assert_eq!(args.trust, TrustArg::Ignore);
    assert_eq!(args.verbose, 2);
    assert_eq!(args.prompt, ["hello"]);
    assert_eq!(
        args.session_selection().unwrap(),
        SessionSelection::NoSession
    );
}

#[test]
fn session_flags_conflict_and_explicit_id_is_validated() {
    assert!(CliArgs::try_parse_from(["tea", "--new", "--continue"]).is_err());
    assert!(CliArgs::try_parse_from(["tea", "--print", "--json"]).is_err());
    assert!(CliArgs::try_parse_from(["tea", "--rpc", "--json"]).is_err());
    let args = CliArgs::try_parse_from(["tea", "--session", "not-an-id"]).unwrap();
    assert!(args.session_selection().is_err());
    assert!(CliArgs::try_parse_from(["tea", "--reasoning-effort", "extreme"]).is_err());
}

#[test]
fn documented_cli_modes_are_available_without_credentials() {
    let help = CliArgs::command().render_long_help().to_string();

    assert!(help.contains("--rpc"));
    assert!(help.contains("--json"));
    assert!(help.contains("--print"));
}

#[test]
fn final_output_flags_are_explicit_and_mutually_exclusive() {
    let object =
        CliArgs::try_parse_from(["tea", "--print", "--output-format", "json-object", "answer"])
            .unwrap();
    assert_eq!(object.output_format.as_deref(), Some("json-object"));
    assert!(object.output_schema.is_none());

    let schema = CliArgs::try_parse_from([
        "tea",
        "--print",
        "--output-schema",
        "answer.schema.json",
        "answer",
    ])
    .unwrap();
    assert_eq!(
        schema.output_schema.as_deref(),
        Some(std::path::Path::new("answer.schema.json"))
    );
    assert!(
        CliArgs::try_parse_from([
            "tea",
            "--output-format",
            "json-object",
            "--output-schema",
            "answer.schema.json",
        ])
        .is_err()
    );

    for args in [
        ["tea", "--output-format", "json-object"],
        ["tea", "--output-schema", "answer.schema.json"],
    ] {
        assert!(
            CliArgs::try_parse_from(args).is_err(),
            "structured output must select --print or --json"
        );
    }
}

fn output_schema_root(label: &str) -> PathBuf {
    let root = std::env::temp_dir().join(format!(
        "tea-output-schema-{label}-{}",
        uuid::Uuid::now_v7().hyphenated()
    ));
    fs::create_dir_all(&root).unwrap();
    root
}

fn output_schema_args(root: &Path, path: &Path) -> CliArgs {
    CliArgs::try_parse_from([
        "tea",
        "--print",
        "--cwd",
        root.to_str().unwrap(),
        "--output-schema",
        path.to_str().unwrap(),
        "answer",
    ])
    .unwrap()
}

fn output_schema_bootstrap(root: &Path) -> CliBootstrap {
    CliBootstrap::new(BootstrapEnvironment::new(
        root,
        Some(root.to_path_buf()),
        BTreeMap::new(),
    ))
}

fn assert_output_schema_error(bootstrap: &CliBootstrap, args: &CliArgs, message: &str) {
    let error = bootstrap.final_output_format(args).unwrap_err();
    assert_eq!(error.category(), ExitCategory::Usage);
    assert_eq!(error.message(), message);
}

#[test]
fn bootstrap_preflight_loads_valid_final_output_formats() {
    let root = output_schema_root("valid");
    let schema = serde_json::json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "$defs": {"answer": {"type": "string", "minLength": 1}},
        "properties": {"answer": {"$ref": "#/$defs/answer"}},
        "required": ["answer"],
        "additionalProperties": false
    });
    fs::write(
        root.join("answer.schema.json"),
        serde_json::to_vec(&schema).unwrap(),
    )
    .unwrap();
    let bootstrap = output_schema_bootstrap(&root);
    let schema_args = output_schema_args(&root, Path::new("answer.schema.json"));
    assert_eq!(
        bootstrap.final_output_format(&schema_args).unwrap(),
        Some(FinalOutputFormat::JsonSchema { schema })
    );

    let object_args =
        CliArgs::try_parse_from(["tea", "--print", "--output-format", "json-object", "answer"])
            .unwrap();
    assert_eq!(
        bootstrap.final_output_format(&object_args).unwrap(),
        Some(FinalOutputFormat::JsonObject)
    );
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn bootstrap_preflight_rejects_malformed_schema_json() {
    let root = output_schema_root("malformed");
    fs::write(root.join("answer.schema.json"), b"{not-json").unwrap();
    let bootstrap = output_schema_bootstrap(&root);
    let args = output_schema_args(&root, Path::new("answer.schema.json"));

    assert_output_schema_error(&bootstrap, &args, "output schema file is not valid JSON");
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn bootstrap_preflight_rejects_invalid_draft_syntax_and_external_refs() {
    let root = output_schema_root("invalid-draft");
    let bootstrap = output_schema_bootstrap(&root);
    let cases = [
        ("non-object.schema.json", serde_json::json!([])),
        ("invalid.schema.json", serde_json::json!({"type": 42})),
        (
            "external.schema.json",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "answer": {"$ref": "https://example.com/answer.schema.json"}
                }
            }),
        ),
    ];
    for (file_name, schema) in cases {
        fs::write(root.join(file_name), serde_json::to_vec(&schema).unwrap()).unwrap();
        let args = output_schema_args(&root, Path::new(file_name));
        assert_output_schema_error(
            &bootstrap,
            &args,
            "output schema is not valid Draft 2020-12 JSON Schema",
        );
    }
    fs::remove_dir_all(root).unwrap();
}

#[test]
fn bootstrap_preflight_enforces_schema_bounds_and_workspace_paths() {
    let root = output_schema_root("bounds");
    let bootstrap = output_schema_bootstrap(&root);

    let mut deep_schema = serde_json::json!({"type": "object"});
    for _ in 0..=MAX_FINAL_OUTPUT_SCHEMA_DEPTH {
        deep_schema = serde_json::json!({"allOf": [deep_schema]});
    }
    fs::write(
        root.join("deep.schema.json"),
        serde_json::to_vec(&deep_schema).unwrap(),
    )
    .unwrap();
    let deep_args = output_schema_args(&root, Path::new("deep.schema.json"));
    assert_output_schema_error(
        &bootstrap,
        &deep_args,
        "output schema exceeds supported bounds",
    );

    let oversized = serde_json::json!({
        "type": "object",
        "description": "x".repeat(MAX_FINAL_OUTPUT_SCHEMA_BYTES)
    });
    fs::write(
        root.join("oversized.schema.json"),
        serde_json::to_vec(&oversized).unwrap(),
    )
    .unwrap();
    let oversized_args = output_schema_args(&root, Path::new("oversized.schema.json"));
    assert_output_schema_error(
        &bootstrap,
        &oversized_args,
        "output schema file is invalid or unreadable",
    );

    let outside = root.with_extension("outside.schema.json");
    fs::write(&outside, br#"{"type":"object"}"#).unwrap();
    let outside_args = output_schema_args(&root, &outside);
    assert_output_schema_error(
        &bootstrap,
        &outside_args,
        "output schema file is invalid or unreadable",
    );

    fs::remove_file(outside).unwrap();
    fs::remove_dir_all(root).unwrap();
}
