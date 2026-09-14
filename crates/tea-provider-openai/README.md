# tea-provider-openai

OpenAI-compatible model provider for `tea-rs`.

The package is `tea-provider-openai`; Rust code imports it as
`tea_provider_openai`. It implements the provider-neutral `tea_model::ModelProvider`
contract for OpenAI Chat Completions and Responses APIs, including streaming
text, the non-streaming Groq strict-schema path, reasoning, images, function
tools, Responses hosted web search, usage, and provider continuation data.

## Configuration

`OpenAiConfig` can be constructed directly for an injected credential/configuration
source. `EnvCredentialResolver` provides the process environment contract for
the CLI and examples:

```text
TEA_OPENAI_API_KEY
TEA_OPENAI_MODEL
TEA_OPENAI_BASE_URL       (optional; defaults to https://api.openai.com/v1)
TEA_OPENAI_API_MODE       (optional; chat-completions or responses)
TEA_OPENAI_COMPATIBILITY_PROFILE
TEA_OPENAI_FINAL_JSON_OBJECT  (optional model/endpoint assertion)
TEA_OPENAI_FINAL_JSON_SCHEMA  (optional model/endpoint assertion)
TEA_OPENAI_FINAL_JSON_SCHEMA_WITH_TOOLS (optional combination assertion)
TEA_OPENAI_HOSTED_WEB_SEARCH  (optional Responses model/endpoint assertion)
TEA_OPENAI_REQUEST_TIMEOUT_MS (optional)
```

`TEA_OPENAI_COMPATIBILITY_PROFILE` accepts `open-ai`, `azure-open-ai`, `xai`,
`deep-seek`, `gemini-open-ai`, `ollama-local`, `open-router`, `groq`, `mistral`,
`together`, or `vllm`.
The built-in OpenAI resolver defaults to `open-ai` only when the resolved base
URL is exactly `https://api.openai.com/v1`. Overriding the base URL requires an
explicit profile, including when using the built-in provider identity. A custom
provider resolver and the general `OpenAiConfig::new` constructor have no
default profile. Direct configuration callers must use
`with_compatibility_profile` before advertising structured-output capabilities.
Tea never infers a dialect from a provider id, model id, display name, or custom
base URL.

The capability flags are explicit. `1` and case-insensitive `true` enable a
flag; every other value leaves it disabled. The schema-with-tools flag also
enables the base schema capability, but is rejected when the selected profile
does not support that combination. `TEA_OPENAI_HOSTED_WEB_SEARCH` advertises the
hosted tool only in Responses mode; enabling the capability with Chat
Completions is a configuration error. Neither the base URL nor a model-name
pattern grants hosted search. This is necessary because protocol support does
not prove that the exact configured model and endpoint offer constrained
decoding or hosted tools. Credentials are not stored in model requests, events,
or session records.

## Structured final output

`FinalOutputFormat::JsonObject` requires one parseable JSON object but does not
constrain its fields. `FinalOutputFormat::JsonSchema` requests provider-side
constrained decoding against an application schema. They are distinct contracts:
Tea never silently downgrades schema mode to JSON-object mode or emulates it with
a tool call.

The currently documented OpenAI-compatible matrix is:

| Compatibility profile | Chat object | Chat schema | Chat schema + tools | Responses object | Responses schema | Responses schema + tools |
|---|---:|---:|---:|---:|---:|---:|
| `open-ai` | yes | yes | opt-in | yes | yes | opt-in |
| `azure-open-ai` | yes | yes | opt-in, parallel off | yes | yes | opt-in, parallel off |
| `xai` | yes | yes | opt-in | yes | yes | opt-in |
| `deep-seek` | yes | no | no | yes | yes | opt-in |
| `gemini-open-ai` | yes | yes | opt-in | no | no | no |
| `ollama-local` | yes | yes | opt-in | no | no | no |
| `open-router` | yes | yes | opt-in | no | no | no |
| `groq` | yes | yes, non-streaming | no | no | no | no |
| `mistral` | yes | yes | opt-in | no | no | no |
| `together` | yes | yes | opt-in | no | no | no |
| `vllm` | yes | yes | opt-in | yes | yes | no |

`opt-in` means the profile permits the wire combination only when the exact
model/endpoint catalog entry advertises `final_json_schema_with_tools`. It is
not inferred from independent schema and tool support. Hosted tools additionally
require their own model capability, supplied by
`TEA_OPENAI_HOSTED_WEB_SEARCH=true` for the environment-backed catalog or by an
explicit custom model catalog entry. The ignored Responses live matrix covers
OpenAI and xAI schema plus hosted web search explicitly.

Unsupported profile/mode/format combinations are rejected before HTTP
transport. In particular, DeepSeek Chat does not advertise JSON Schema;
Mistral and Together are Chat-only profiles; and Gemini, Ollama, OpenRouter,
Groq, Mistral, and Together do not advertise structured output over Responses.
Groq's beta Responses API remains usable without a final-output format. vLLM
Responses object and schema modes use the standard `text.format` envelope, but
schema plus tools is rejected because the released implementation and tests do
not establish that combination. `ollama-local` means a local Ollama server;
Ollama Cloud structured output is not enabled by this profile.

Chat Completions maps the contract to `response_format`; Responses maps it to
`text.format`. Every supported schema dialect sends `strict: true` except the
documented DeepSeek Responses shape, which omits `strict`. OpenRouter structured
requests also send `provider.require_parameters: true`, preventing selection of
a route that ignores the parameter. Groq strict schema is sent only to Chat
Completions with `stream: false`; schema with tools and every Groq Responses
structured-output request fail before HTTP. The complete Groq JSON response is
bounded and normalized into the same event grammar as streamed providers.

Together schema requests append a fixed JSON-only instruction plus the exact
serialized schema to the system prompt while preserving caller-supplied system
text. Azure JSON Schema requests with parallel function calls enabled are
rejected. Every schema-plus-tools request also requires the exact model catalog
entry to advertise `final_json_schema_with_tools`; protocol support alone never
enables the combination.

JSON-object mode requires the model context to mention JSON on OpenAI-family
endpoints. Both request mappers append one fixed, explicit instruction to every
JSON-object wire request. They do not use substring heuristics over arbitrary
conversation text, and they do not mutate the provider-neutral `ModelRequest`.

Tea validates the provider-neutral schema representation and bounds before
mapping it. Providers implement smaller and different JSON-Schema subsets, so a
schema accepted by Tea's local Draft 2020-12 validator may still be rejected by
the selected service. A refusal is preserved as `StopReason::Refusal`, and token
truncation remains `StopReason::Length`. Chat `content_filter`, future finish
reasons, and a stream ending without any finish reason remain non-successful
`StopReason::Unknown` outcomes. None of these is a schema-conforming success.
Callers must require `StopReason::Completed`, parse the accumulated text, and
perform final local validation before accepting the result.

## Live matrix

Ignored live tests exist for every supported object/schema tuple above. Every
`opt-in` function-tool tuple has a dedicated test; OpenAI and xAI Responses also
have exact schema-plus-hosted-web-search tests, and Azure tests both API modes
with parallel calls disabled. Combination tests require
`TEA_OPENAI_FINAL_JSON_SCHEMA_WITH_TOOLS=true`. Each test requires explicit base
URL, model, API mode, compatibility profile, and matching capability flag;
otherwise it prints a skip message and performs no network request. Values may
come from the repository `.env` file or process environment, with process
values taking precedence.

```bash
TEA_OPENAI_API_MODE=chat-completions \
TEA_OPENAI_COMPATIBILITY_PROFILE=open-ai \
TEA_OPENAI_FINAL_JSON_SCHEMA=true \
cargo test -p tea-provider-openai --features live --test integration \
  smoke::structured_output_open_ai_chat_json_schema -- --ignored --nocapture
```

Every live structured-output test accumulates the normalized text events, requires
`StopReason::Completed`, parses one JSON object, and locally validates schema
mode against the same Draft 2020-12 schema. Real credentials remain outside the
repository.

## Integration

Use `OpenAiProviderBuilder` to create a provider and advertise its model
catalog. The adapter owns HTTP request mapping, SSE/full-response parsing,
capability validation, retry classification, and normalization into `tea_model` events;
runtime policy, session storage, and tool execution remain host-selected.

See the public [Tea integration documentation](https://github.com/tea-hq/tea-docs)
for provider selection and application setup.
