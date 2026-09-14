# tea-provider-anthropic

Streaming Anthropic Messages API adapter for tea-rs.

Configure the adapter through an injected credential resolver or the process
environment contract used by the CLI:

```bash
export TEA_PROVIDER=anthropic
export TEA_ANTHROPIC_API_KEY='...'
export TEA_ANTHROPIC_MODEL='<anthropic-model-id>'
tea --provider anthropic --model "$TEA_ANTHROPIC_MODEL"
```

`TEA_ANTHROPIC_BASE_URL` defaults to `https://api.anthropic.com`,
`TEA_ANTHROPIC_API_VERSION` defaults to `2023-06-01`, and
`TEA_ANTHROPIC_REQUEST_TIMEOUT_MS` defaults to `60000`. The adapter supports
text, images, function tools, parallel tool calls, usage reporting, and the
Anthropic hosted web-search tool. Hosted web-search options are configured with
`TEA_ANTHROPIC_WEB_SEARCH_TOOL_TYPE` and
`TEA_ANTHROPIC_WEB_SEARCH_MAX_USES` when the host activates that tool.

## Structured final output

Native Anthropic Messages supports `FinalOutputFormat::JsonSchema` through the
GA `output_config.format` request field. Because the selected model and endpoint
are configurable, Tea advertises this capability only when the operator makes
an explicit model/endpoint assertion:

```bash
export TEA_ANTHROPIC_FINAL_JSON_SCHEMA=true
```

`1` and case-insensitive `true` enable it; every other value leaves it disabled.
The adapter does not infer support from an Anthropic-looking model name or base
URL. In particular, DeepSeek's Anthropic-compatible endpoint does not document
`output_config.format` and must remain disabled. A custom gateway should enable
the flag only when it implements the native Anthropic GA wire contract.

Anthropic has no schema-less JSON-object output mode, so
`FinalOutputFormat::JsonObject` is rejected before HTTP transport. Schema mode
uses exactly:

```json
{"output_config":{"format":{"type":"json_schema","schema":{}}}}
```

Tea does not send the obsolete `output_format` field or the former
`anthropic-beta: structured-outputs-2025-11-13` header. Provider constrained
decoding supports a subset of JSON Schema; a schema accepted by Tea's local
Draft 2020-12 validator can still be rejected by a particular Anthropic model.

Anthropic JSON outputs are incompatible with its document-citations feature and
with assistant-message prefilling. Tea rejects assistant-prefill schema requests
locally and does not enable document citations in this adapter. Function tools
remain compatible; strict function arguments are a separate provider feature.

`stop_reason: "refusal"` is normalized as `StopReason::Refusal`, while
`stop_reason: "max_tokens"` becomes `StopReason::Length`. Neither is a
schema-conforming success, even when visible text happens to parse as JSON;
callers must require `StopReason::Completed` before accepting and locally
validating final output.

The ignored native live smokes exercise both schema-only and
schema-plus-function-tool streaming requests, parse the accumulated text, and
validate it against the same schema locally. They skip without making a network
request unless the API key, model, and capability flag are all present:

```bash
cargo test -p tea-provider-anthropic --features live --test integration \
  smoke::structured_output -- --ignored --nocapture
```

Extended thinking is not currently supported. Provider-specific request and
stream types remain behind this adapter; hosts consume normalized
`tea_model` events.
