use std::time::Duration;

use tea::{
    AgentSession,
    model::{
        ModelCapabilities, ModelCompletion, ModelEvent, ModelFailure, ModelFailureCode,
        ModelProvider, ModelResponseInfo, ModelStreamIndex, ProviderToolCallId, ToolCallCompleted,
        ToolCallStarted, Utf8Delta,
    },
    protocol::{FinalOutputFormat, RetryClass, StopReason},
};
use tea_coding_tools::{WorkspaceRoot, read_only_workspace_tools};
use tea_testkit::ScriptedModelResponse;

use crate::common::{provider_with, provider_with_capabilities};

fn tool_script(name: &str, arguments: serde_json::Value, opaque_id: &str) -> ScriptedModelResponse {
    let index = ModelStreamIndex::new(0).unwrap();
    let provider_id: ProviderToolCallId = opaque_id.parse().unwrap();
    ScriptedModelResponse::events([
        ModelEvent::Started(ModelResponseInfo::new()),
        ModelEvent::ToolCallStarted(
            ToolCallStarted::new(index, provider_id.clone(), name).unwrap(),
        ),
        ModelEvent::ToolCallCompleted(
            ToolCallCompleted::new(index, provider_id, name, arguments).unwrap(),
        ),
        ModelEvent::Completed(ModelCompletion::new(StopReason::ToolUse).unwrap()),
    ])
}

fn tool_script_with_preamble(
    preamble: &str,
    name: &str,
    arguments: serde_json::Value,
    opaque_id: &str,
) -> ScriptedModelResponse {
    let index = ModelStreamIndex::new(0).unwrap();
    let provider_id: ProviderToolCallId = opaque_id.parse().unwrap();
    ScriptedModelResponse::events([
        ModelEvent::Started(ModelResponseInfo::new()),
        ModelEvent::TextDelta(Utf8Delta::new(preamble).unwrap()),
        ModelEvent::ToolCallStarted(
            ToolCallStarted::new(index, provider_id.clone(), name).unwrap(),
        ),
        ModelEvent::ToolCallCompleted(
            ToolCallCompleted::new(index, provider_id, name, arguments).unwrap(),
        ),
        ModelEvent::Completed(ModelCompletion::new(StopReason::ToolUse).unwrap()),
    ])
}

#[tokio::test]
async fn prompt_returns_aggregated_assistant_text() {
    let provider = provider_with([ScriptedModelResponse::text(["hello", " from tea"])]);
    let model = provider.models()[0].model_ref().clone();
    let session = AgentSession::builder(provider, model)
        .build()
        .await
        .unwrap();

    let response = session.prompt("Say hello.", None).await.unwrap();

    assert_eq!(response.text(), "hello from tea");
}

#[tokio::test]
async fn default_policy_executes_a_read_only_workspace_tool() {
    let provider = provider_with([
        tool_script("read", serde_json::json!({"path":"Cargo.toml"}), "read-1"),
        ScriptedModelResponse::text(["read completed"]),
    ]);
    let model = provider.models()[0].model_ref().clone();
    let workspace = WorkspaceRoot::new(std::env::current_dir().unwrap()).unwrap();
    let tools = read_only_workspace_tools(&workspace).unwrap();

    let session = AgentSession::builder(provider, model)
        .tools(tools)
        .build()
        .await
        .unwrap();

    let response = session.prompt("Read Cargo.toml.", None).await.unwrap();

    assert_eq!(response.text(), "read completed");
}

#[tokio::test]
async fn prompt_preserves_and_validates_a_structured_final_output() {
    let provider = provider_with_capabilities(
        [ScriptedModelResponse::text([r#"{"answer":"tea"}"#])],
        ModelCapabilities::text().with_final_json_object(),
    );
    let model = provider.models()[0].model_ref().clone();
    let session = AgentSession::builder(provider.clone(), model)
        .build()
        .await
        .unwrap();

    let response = session
        .prompt(
            "Return an answer object.",
            Some(FinalOutputFormat::JsonObject),
        )
        .await
        .unwrap();

    assert_eq!(response.text(), r#"{"answer":"tea"}"#);
    assert_eq!(
        provider.captured_requests().unwrap()[0].final_output_format(),
        Some(&FinalOutputFormat::JsonObject)
    );
}

#[tokio::test]
async fn prompt_discards_partial_text_from_a_retried_structured_output() {
    let failure = ModelFailure::safe(
        ModelFailureCode::Unavailable,
        "provider temporarily unavailable",
        RetryClass::Immediate,
    )
    .unwrap()
    .with_retry_after(Duration::from_millis(1));
    let provider = provider_with_capabilities(
        [
            ScriptedModelResponse::events([
                ModelEvent::Started(ModelResponseInfo::new()),
                ModelEvent::TextDelta(Utf8Delta::new(r#"{"stale":"partial"#).unwrap()),
                ModelEvent::Failed(failure),
            ]),
            ScriptedModelResponse::text([r#"{"answer":"tea"}"#]),
        ],
        ModelCapabilities::text().with_final_json_object(),
    );
    let model = provider.models()[0].model_ref().clone();
    let session = AgentSession::builder(provider.clone(), model)
        .build()
        .await
        .unwrap();

    let response = session
        .prompt(
            "Return an answer object.",
            Some(FinalOutputFormat::JsonObject),
        )
        .await
        .unwrap();

    assert_eq!(response.text(), r#"{"answer":"tea"}"#);
    assert_eq!(provider.captured_requests().unwrap().len(), 2);
}

#[tokio::test]
async fn prompt_returns_only_the_final_structured_output_after_a_tool_turn() {
    let provider = provider_with_capabilities(
        [
            tool_script_with_preamble(
                "I will inspect the workspace first.",
                "read",
                serde_json::json!({"path":"Cargo.toml"}),
                "read-with-preamble-1",
            ),
            ScriptedModelResponse::text([r#"{"answer":"read completed"}"#]),
        ],
        ModelCapabilities::text()
            .with_tools(true)
            .with_final_json_object(),
    );
    let model = provider.models()[0].model_ref().clone();
    let workspace = WorkspaceRoot::new(std::env::current_dir().unwrap()).unwrap();
    let tools = read_only_workspace_tools(&workspace).unwrap();
    let session = AgentSession::builder(provider, model)
        .tools(tools)
        .build()
        .await
        .unwrap();

    let response = session
        .prompt(
            "Inspect Cargo.toml and return an answer object.",
            Some(FinalOutputFormat::JsonObject),
        )
        .await
        .unwrap();

    assert_eq!(response.text(), r#"{"answer":"read completed"}"#);
}
