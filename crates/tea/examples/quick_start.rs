//! Minimal real-provider agent session with read-only workspace tools.
//!
//! Set `TEA_API_KEY`, `TEA_MODEL`, and `TEA_BASE_URL`, then run:
//! `cargo run --example quick_start -p tea -- "Summarize this project."`

#![forbid(unsafe_code)]

use std::env;
use std::error::Error;
use std::io;
use std::sync::Arc;

use tea::{AgentSession, model::ModelProvider};
use tea_coding_tools::{WorkspaceRoot, read_only_workspace_tools};
use tea_provider_openai::{ApiKey, OpenAiConfig, OpenAiProviderBuilder};

const DEFAULT_MESSAGE: &str = "What files are in the current directory?";

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn Error>> {
    let api_key = required_env("TEA_API_KEY")?;
    let model = required_env("TEA_MODEL")?;
    let base_url = required_env("TEA_BASE_URL")?;
    let message = env::args()
        .nth(1)
        .unwrap_or_else(|| DEFAULT_MESSAGE.to_owned());
    let config = OpenAiConfig::new(model.parse()?, base_url, ApiKey::new(api_key)?)?;
    let provider = OpenAiProviderBuilder::new()
        .with_config(Arc::new(config))
        .build()?;
    let model = provider.models()[0].model_ref().clone();
    let workspace = WorkspaceRoot::new(env::current_dir()?)?;

    let session = AgentSession::builder(Arc::new(provider), model)
        .system_prompt("You are a concise assistant. Use the read-only tools when helpful.")
        .tools(read_only_workspace_tools(&workspace)?)
        .build()
        .await?;

    println!("{}", session.prompt(message, None).await?.text());
    Ok(())
}

fn required_env(name: &str) -> Result<String, io::Error> {
    match env::var(name) {
        Ok(value) if !value.is_empty() => Ok(value),
        Ok(_) | Err(env::VarError::NotPresent) => Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("{name} is not set"),
        )),
        Err(env::VarError::NotUnicode(_)) => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{name} is not valid Unicode"),
        )),
    }
}
