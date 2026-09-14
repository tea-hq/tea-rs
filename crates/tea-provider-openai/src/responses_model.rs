use std::fmt;

use serde::Serialize;
use serde_json::Value;

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ResponsesContentItem {
    InputText {
        text: String,
    },
    OutputText {
        text: String,
        annotations: Vec<Value>,
    },
    InputImage {
        image_url: String,
        detail: String,
    },
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(untagged)]
pub(crate) enum FunctionCallOutputPayload {
    Text(String),
    Content(Vec<ResponsesContentItem>),
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub(crate) struct ReasoningSummaryItem {
    pub(crate) r#type: String,
    pub(crate) text: String,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ResponseItem {
    Message {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        role: String,
        content: Vec<ResponsesContentItem>,
        #[serde(skip_serializing_if = "Option::is_none")]
        status: Option<String>,
    },
    FunctionCall {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        call_id: String,
        name: String,
        arguments: String,
    },
    FunctionCallOutput {
        call_id: String,
        output: FunctionCallOutputPayload,
    },
    Reasoning {
        #[serde(skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        summary: Vec<ReasoningSummaryItem>,
        #[serde(skip_serializing_if = "Option::is_none")]
        encrypted_content: Option<String>,
    },
}

#[derive(Clone, Serialize, PartialEq)]
#[serde(untagged)]
pub(crate) enum ResponsesInputItem {
    Canonical(ResponseItem),
    ProviderContinuation(Value),
}

impl fmt::Debug for ResponsesInputItem {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Canonical(_) => formatter.write_str("Canonical(<redacted>)"),
            Self::ProviderContinuation(_) => {
                formatter.write_str("ProviderContinuation(<redacted>)")
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub(crate) enum ResponsesApiTool {
    Function {
        name: String,
        description: String,
        parameters: Value,
        strict: bool,
    },
    WebSearch {
        #[serde(skip_serializing_if = "Option::is_none")]
        filters: Option<ResponsesWebSearchFilters>,
        #[serde(skip_serializing_if = "Option::is_none")]
        user_location: Option<ResponsesWebSearchLocation>,
    },
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub(crate) struct ResponsesWebSearchFilters {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) allowed_domains: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) blocked_domains: Vec<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub(crate) struct ResponsesWebSearchLocation {
    pub(crate) r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) country: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) city: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) region: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) timezone: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub(crate) struct ResponsesReasoning {
    pub(crate) effort: String,
    pub(crate) summary: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub(crate) struct ResponsesText {
    pub(crate) format: ResponsesTextFormat,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub(crate) struct ResponsesTextFormat {
    pub(crate) r#type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) strict: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) schema: Option<Value>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub(crate) struct ResponsesApiRequest {
    pub(crate) model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) instructions: Option<String>,
    pub(crate) input: Vec<ResponsesInputItem>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) tools: Vec<ResponsesApiTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) reasoning: Option<ResponsesReasoning>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) text: Option<ResponsesText>,
    pub(crate) store: bool,
    pub(crate) stream: bool,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) include: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) max_output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prompt_cache_key: Option<String>,
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    const OPAQUE_SENTINEL: &str = "opaque-continuation-must-not-appear-in-debug";

    #[test]
    fn responses_request_debug_redacts_all_input_payloads() {
        let canonical = ResponsesInputItem::Canonical(ResponseItem::Message {
            id: None,
            role: "assistant".to_owned(),
            content: vec![ResponsesContentItem::OutputText {
                text: "answer".to_owned(),
                annotations: vec![json!({"opaque": OPAQUE_SENTINEL})],
            }],
            status: None,
        });
        let continuation =
            ResponsesInputItem::ProviderContinuation(json!({"opaque": OPAQUE_SENTINEL}));
        let request = ResponsesApiRequest {
            model: "gpt-test".to_owned(),
            instructions: None,
            input: vec![canonical, continuation],
            tools: Vec::new(),
            tool_choice: None,
            parallel_tool_calls: None,
            reasoning: None,
            text: None,
            store: false,
            stream: true,
            include: Vec::new(),
            max_output_tokens: None,
            service_tier: None,
            prompt_cache_key: None,
        };

        let debug = format!("{request:?}");
        assert!(!debug.contains(OPAQUE_SENTINEL), "{debug}");
    }
}
