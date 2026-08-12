# tea-cli

Reference Coding CLI for `tea-rs`.

The package is `tea-cli`; Rust code imports its support library as `tea_cli`; the executable is `tea`. It provides interactive terminal, script-safe print, canonical JSON event, and strict JSONL/RPC modes over the same mode-neutral `tea-coding` service.

> **Security:** tools execute directly as the current OS user. Approval is not a
> sandbox, and repository instructions or skills may contain prompt injection.
> Read the public [TUI guide](https://github.com/tea-hq/tea-docs/blob/main/src/content/docs/cli/tui.md)
> and [security boundary](https://github.com/tea-hq/tea-docs/blob/main/src/content/docs/safety/security.md)
> before trusting a workspace.

## Interactive mode

```bash
export TEA_OPENAI_API_KEY='...'
tea --model gpt-5.4 --trust once
tea --continue --trust once "inspect the current changes"
```

Interactive mode is selected when stdin and stdout are terminals; piped stdin selects print mode automatically. Its bounded Unicode multiline editor supports grapheme-safe movement/deletion, atomic bracketed paste, undo, local history, command completion, streaming updates, steering, follow-up queues, cancellation, and model/session/tree selectors. Enter submits while idle and steers while running, Alt+Enter queues a follow-up, Escape aborts, Shift+Enter inserts a newline, and Ctrl+D exits. All bindings are configurable in nested TUI settings and ambiguous configurations fail closed.

Built-ins are `/new`, `/resume`, `/session`, `/name`, `/model`, `/reasoning`,
`/compact`, `/tree`, `/fork`, `/image`, `/copy`, `/mcp`, `/skills`, `/help`, and
`/quit`. Trusted prompt templates are exposed as slash commands and trusted
skills as `/skill:<id> [args]`. `/skills` is a read-only view of the immutable
startup catalog; it shows every winning skill, including
`disable-model-invocation: true` skills, with source provenance and no manifest
paths. Clipboard writes use an explicit host adapter and never enter
an inward runtime crate. Session and tree selectors are projections of the
catalog and canonical append-only branch records. Switching sessions replaces
the event subscription and rebuilds durable state; steering/follow-up display
queues are retained per session only while the same service process remains
alive and are empty after restart.

Pending approvals offer `allow once`, `allow for session`, and `deny`. Session authorization is restricted to the persisted actor, profile, session, tool/version, effects, resource locator prefix, and access mode; it is not a workspace-wide bypass. The request's full locator is used as the narrowest prefix supported by the policy contract. The UI renders the persisted redacted approval presentation and disables further selection/submission after the service accepts a decision. Closing the UI before resolution leaves the request pending in SQLite, so reopening shows the same request. Once execution has started, cancellation records an uncertain interruption and never automatically replays the tool.

## Print mode

```bash
export TEA_OPENAI_API_KEY='...'
tea --print --model gpt-5.4 --trust once "inspect this workspace"
printf '%s' 'summarize the repository' | tea --print --trust ignore
tea --print --trust once @prompt.md
```

Prompt arguments, workspace-confined `@file` content, and piped UTF-8 stdin are joined in order under one 256 KiB bound. Print mode subscribes before prompt submission and drains runtime events internally, but stdout receives exactly the final assistant text plus one LF. Diagnostics use stderr. It never prints startup banners, progress, ANSI styling, thinking text, tool output, or credentials to stdout.

Shared options cover cwd, provider/model/profile, an invocation-local redacted `--api-key`, tools, explicit context files, new/continue/explicit/in-memory sessions, application state paths, project trust, and verbosity. Non-interactive default trust fails closed only when project-local resources exist and no saved decision applies; `--trust once`, `persist`, `reject`, and `ignore` are explicit alternatives. Project settings and declarative resources remain workspace-confined even when trusted.

Stable non-zero process categories are: usage `2`, trust/config `3`, provider `4`, policy/approval `5`, cancellation `6`, and internal/persistence `70`. `SIGINT` cooperatively cancels the owned run and leaves stdout empty.

## Skill resources

The catalog considers these roots from highest to lowest precedence:

1. explicitly configured `resources.skillPaths`, in configured order;
2. the trusted workspace `.tea/skills` directory;
3. the trusted workspace `.agents/skills` directory;
4. the user Tea config `skills` directory, normally `~/.tea/skills`;
5. the user `.agents/skills` directory; and
6. the Tea data `skills` directory.

Directory roots recursively discover `SKILL.md`, honoring `.gitignore`,
`.ignore`, and `.fdignore`, skipping hidden directories and `node_modules`, and
stopping below a directory that already contains a skill manifest. Ordinary
root-level `.md` files are accepted only through an explicit
`resources.skillPaths` entry. Both project roots are trust-gated, and only the
workspace-root `.agents/skills` is scanned; ancestor directories are not
searched.

`read_skill_resource` is the dedicated read-only capability for files beneath
the winning skill directory. It enforces relative-path containment, bounded
UTF-8 reads, and race-checked file identity. It cannot write, install, or
execute a skill's scripts; the existing `bash` capability remains
workspace-rooted. The catalog is a startup snapshot, so filesystem changes are
observed after rebuilding or restarting the service.

Select built-in `openai` or `anthropic`, or an OpenAI-compatible provider from `<config-dir>/providers.json` or trusted `.tea/providers.json`, with `--provider` or `TEA_PROVIDER`; `--model` takes precedence over environment/global/project/default configuration. OpenAI-compatible connections use configured fields with `TEA_OPENAI_*` fallback; Anthropic Messages connections use `TEA_ANTHROPIC_API_KEY` and optional `TEA_ANTHROPIC_BASE_URL`. API keys are resolved through redacting credential ports and are not persisted. See the public [configuration guide](https://github.com/tea-hq/tea-docs/blob/main/src/content/docs/configuration/settings.md) for the settings precedence, trust boundary, and credential references.

## JSON event mode

```bash
tea --json --model gpt-5.4 --trust once "inspect and explain"
```

JSON mode emits strict compact JSON with exactly one LF delimiter per value. The first line is a versioned `tea_event_stream` header containing `modeVersion`, canonical `protocolVersion`, `sessionId`, and the path-independent `workspaceId`. Every later line is an unchanged canonical `tea_protocol::EventEnvelope`; each line parses independently. Thinking/text deltas, tool requests/progress, approval, checkpoints, and terminal run observations remain protocol events rather than presentation text. Human diagnostics, ANSI styling, banners, and secrets never enter stdout.

A dedicated blocking writer owns stdout behind a 32-slot queue. Every line is capped at 1 MiB and flushed with a 500 ms enqueue/write-ack deadline. A slow or broken pipe cancels the owned run, returns exit category `6`, and cannot deadlock runtime shutdown; already written bytes remain complete independently parseable lines.

## JSONL/RPC mode

```bash
tea --rpc --continue --model gpt-5.4 --trust once
```

RPC stdin and stdout contain compact JSON values separated by byte `LF`; a `CR` immediately before `LF` is accepted. Unicode line and paragraph separators inside JSON strings are ordinary content. The first output is a versioned `ready` frame with `sessionId` and path-independent `workspaceId`. Every request includes `rpcVersion: "1.0"`, an optional bounded string `id`, a `type`, and a typed `payload`:

```json
{"rpcVersion":"1.0","id":"p1","type":"prompt","payload":{"text":"inspect the changes"}}
{"rpcVersion":"1.0","id":"s1","type":"query_snapshot","payload":{"afterSequence":"12","limit":32}}
```

Prompt and approval requests receive a correlated `command_accepted` response and later an asynchronous `command_finished` frame. Runtime observations are emitted as `event` frames whose payload is an unchanged canonical `EventEnvelope`. Host queries cover state, paginated durable snapshots, statistics, branch tree, sessions, and models; mutations cover new/open/name, prompt, steer, follow-up, abort, approval, model, compact, and fork. Session rebinds and subscription replacement require a snapshot query rather than inferring durable state from deltas.

Input and output frames are capped at 1 MiB. Snapshot pages contain at most 64 canonical records. The output writer has 32 slots and a 500 ms enqueue/write/flush deadline. A complete malformed frame receives one `parse_error` response and the next LF frame is still processed. Oversized or unterminated input, I/O failure, EOF, signal, disconnect, or a slow writer ends the connection; the owning mode then cancels and awaits service work. RPC stdout never contains diagnostics, ANSI escapes, or banners.

## Terminal projection boundary

Ratatui and Crossterm are confined to this outward product crate. Fullscreen is the default; fullscreen and inline modes both pass the committed PTY/virtual-screen matrix. `tui::TuiState` rebuilds from a complete canonical session snapshot and then reduces bounded transient runtime observations in monotonic sequence order; reconnects and inconsistent state-bearing events request a fresh snapshot rather than guessing durable state. Rendering is pure, Unicode display-width bounded, and cached only by content generation, width, collapse state, and theme generation.

The product-owned bounded Action/Effect loop keeps async snapshot/frame work outside reducers. `TerminalGuard` records every enabled terminal mode and tears it down in reverse order on normal exit, partial setup failure, panic, suspend, or foreground-child handoff. No terminal dependency or render projection enters an inward runtime crate.

Maintained end-user documentation lives in
[`tea-hq/tea-docs`](https://github.com/tea-hq/tea-docs), organized around the
[TUI workflow](https://github.com/tea-hq/tea-docs/blob/main/src/content/docs/cli/tui.md)
and [CLI modes](https://github.com/tea-hq/tea-docs/blob/main/src/content/docs/get-started/cli-modes.md).
