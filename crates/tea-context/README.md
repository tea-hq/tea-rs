# tea-context

Tokio-free prompt modules, context-provider contracts, deterministic compilation, provenance, trust labels, budgets, diagnostics, and inspection for `tea-rs`.

The package is `tea-context`; Rust code imports it as `tea_context`. Its normal dependency graph contains no model provider, tool executor, JSON Schema compiler, filesystem, process, network, database, or async runtime.

## Prompt modules and segments

Each `PromptSegment` carries a canonical segment ID, non-empty bounded content, `PromptProvenance`, a `TrustLevel`, a `CacheScope`, an optional `ConflictClaim`, and a `BudgetBehavior` (`Required`, `Truncate`, or `Omit`). Segments are grouped into non-empty `PromptModule` values with a canonical module ID, a fixed `PromptAuthority`, and a bounded numeric `PromptPriority`.

`PromptAuthority` fixes high-to-low precedence: Kernel, Organization, Product, Workspace, Tool, Skill, Session, UserAddition. Numeric priority only breaks ties within one authority.

## Provider contract

`ContextProvider` is object-safe and returns a project-owned boxed standard Future. It receives an immutable `ContextRequest` snapshot of the active profile, session, run, canonical active tools, and bounded metadata. Built-in providers perform no I/O: `ToolHintProvider` renders active `ToolSpec::prompt_hint()` guidance, `WorkspaceInstructionProvider` carries caller-supplied documents, `SessionSummaryProvider` inserts an optional durable summary, and `SkillMetadataProvider` exposes skill metadata with canonical `/skill:<id> [args]` commands. The legacy `@skill <id>` parser remains available as a compatibility library API; this pure crate does not load skill files or execute skills.

## Compiler

`PromptCompiler` is pure and synchronous. It flattens modules, sorts by authority, descending priority, module ID, and source order, deduplicates exact segment duplicates, resolves conflict keys by precedence, enforces byte and conservative token budgets (including separators), and renders accepted segments joined with `\n\n`.

- Exact duplicates across modules are emitted once with a diagnostic.
- Reusing a segment ID with different content is a compile error.
- A lower-precedence claim never overrides a protected winner; replaceable winners produce shadow diagnostics.
- Equal-precedence conflicts with different content fail closed.
- `Required` overflow fails; `Omit` emits a diagnostic; `Truncate` uses a UTF-8 boundary and a fixed `[truncated]` marker only when it fits.
- Fully optional prompts may compile to an empty string with diagnostics.

## Inspection

`CompiledPrompt` exposes exact text, byte count, conservative estimated tokens, ordered diagnostics, and one `PromptInspectionEntry` per unique input segment with authority, provenance, trust, cache scope, disposition, exact byte range, rendered bytes, and estimated tokens. Byte ranges slice exact output content; every output byte maps to an included segment or a separator.

`CompiledPrompt::inspection_snapshot` creates a serializable, content-free
`PromptInspection` for host query APIs. It retains module and segment IDs,
authority, prompt-safe provenance, trust, cache scope, disposition, sizes, and
compiler diagnostics while excluding prompt text and byte ranges.

Trust labels describe origin for inspection and downstream policy; they do not sanitize content or claim prompt-injection prevention.
