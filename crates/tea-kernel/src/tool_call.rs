use std::str::FromStr;

use tea_protocol::ProtocolMetadata;
use tea_tools::{
    ToolInvocation, ToolName, ToolRegistry, ToolRegistryError, ValidatedToolInvocation,
};

use crate::model_turn::CompletedToolCall;

pub(crate) enum PreparedToolCall {
    Valid(ValidatedToolInvocation),
    Rejected {
        code: &'static str,
        message: &'static str,
    },
}

pub(crate) fn prepare(
    tools: &ToolRegistry,
    call: &CompletedToolCall,
    metadata: ProtocolMetadata,
) -> PreparedToolCall {
    let Ok(name) = ToolName::from_str(&call.tool_name) else {
        return PreparedToolCall::Rejected {
            code: "unknown_tool",
            message: "model requested an unknown tool",
        };
    };
    let Ok(invocation) =
        ToolInvocation::new(call.tool_call_id, name, call.arguments.clone(), metadata)
    else {
        return PreparedToolCall::Rejected {
            code: "invalid_tool_arguments",
            message: "model supplied invalid tool arguments",
        };
    };
    match tools.validate(invocation) {
        Ok(validated) => PreparedToolCall::Valid(validated),
        Err(ToolRegistryError::UnknownTool | ToolRegistryError::HostedToolNotClientExecutable) => {
            PreparedToolCall::Rejected {
                code: "unknown_tool",
                message: "model requested an unknown tool",
            }
        }
        Err(ToolRegistryError::InvalidArguments(_) | ToolRegistryError::Resources(_)) => {
            PreparedToolCall::Rejected {
                code: "invalid_tool_arguments",
                message: "model supplied invalid tool arguments",
            }
        }
        Err(
            ToolRegistryError::DuplicateTool
            | ToolRegistryError::VersionConflict
            | ToolRegistryError::HostedToolNameMismatch
            | ToolRegistryError::NoSupportedToolRoute { .. }
            | ToolRegistryError::ModelProjection(_)
            | ToolRegistryError::Schema(_),
        ) => PreparedToolCall::Rejected {
            code: "tool_registry_failure",
            message: "tool registry could not validate the invocation",
        },
    }
}
