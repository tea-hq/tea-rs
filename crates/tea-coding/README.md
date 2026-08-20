# tea-coding

Mode-neutral Coding CLI product assembly for `tea-rs`.

The package is `tea-coding`; Rust code imports it as `tea_coding`. It composes versioned secret-free settings, injected application paths, canonical project trust, trusted declarative resources, a modular privacy-safe system prompt, coding profile/policy, live or fake provider, native workspace tools (`read`, `grep`, `find`, `ls`, `write`, `edit`, and `bash`), optional client or hosted web tools, one SQLite store/catalog, and the mode-neutral `CodingAgentService`.

`CodingAgentService` exposes session lifecycle and query operations plus prompt, steering, follow-up, abort, approval, model/profile, compaction, fork, and naming commands. `prompt_inspection` returns content-free metadata for the last prompt compiled in the live process. Prompt and approval-continuation acceptance are returned before their owned tasks complete so a caller can subscribe first and stream through the runtime's bounded event channel; `wait` and `shutdown` await task ownership. SQLite remains authoritative across approval pause and process rebuild.

Embedding products select their policy surface through
`CodingAgentBuilder::execution_surface`. The builder defaults to `Cli` for the
reference command-line product; desktop and IDE hosts must select `Desktop` or
`Ide` explicitly so policy observations retain the real outward boundary.

Interactive, print, JSON event, and JSONL/RPC modes must all call this same service. This crate does not depend on Ratatui, Crossterm, a clipboard implementation, or another UI framework.

## System prompt privacy

`CodingSystemPromptBuilder` produces deterministic `PromptModule` values for
the coding behavior contract, logical working directory, project instructions,
active-tool guidance, and skill metadata. It accepts prompt-safe resource
snapshots rather than host paths. The logical resource tree uses `<workspace>`
as its root and `<workspace>/<relative-path>` for files; a nested working
directory is represented as `<workspace>/<relative-directory>`.

Canonical host paths remain process-local metadata used by resource discovery
and execution capabilities. They are not copied into prompt content or
provenance. Project instruction bodies remain byte-identical; Tea-generated
source-label segments provide logical attribution separately. Skills are
advertised by stable IDs and loaded through the bounded `read_skill_resource`
tool without exposing manifest or resource-root paths.

The CLI discovers global `SYSTEM.md` and `APPEND_SYSTEM.md` in the injected Tea
data directory and trusted project versions under `<workspace>/.tea`. Project
files win. `SYSTEM.md` replaces only the default coding-behavior segment;
workspace identity, active tools, project instructions, skills, privacy checks,
and compiler budgets remain active. `APPEND_SYSTEM.md` is a separate
`UserAddition` module with logical provenance.

Project instructions use the first existing candidate in each directory:
`AGENTS.override.md`, `AGENTS.md`, `AGENTS.MD`, `CLAUDE.md`, then `CLAUDE.MD`.
Instructions are ordered from the global layer and trusted workspace boundary
to the active directory. Canonical targets are loaded once, so a compatibility
`CLAUDE.md` symlink to `AGENTS.md` does not duplicate the instructions.

Embedders add request-sensitive prompt content with `ContextProvider`. It runs
before every compilation and returns typed `PromptModule` values, preserving
authority, trust, provenance, conflict, and budget enforcement. The runtime's
last-successful inspection contains module/segment metadata, sizes, disposition,
and diagnostics but no prompt text or byte ranges, and it is never persisted.

These SDK additions are backward compatible: the generic runtime keeps its
built-in tool guidance enabled by default, and existing coding resource and
provider contracts remain valid. The coding profile opts into the new composed
provider without adding a persistence migration or provider-specific fallback.

## Skills and scoped resources

`ResourceCatalog` discovers a deterministic six-level skill catalog in this
order: explicit `resources.skillPaths`, trusted workspace `.tea/skills`,
trusted workspace `.agents/skills`, user Tea config `skills`, user
`.agents/skills`, and Tea data `skills`. The first valid manifest for an ID
wins. Directory roots recursively find `SKILL.md`, honor `.gitignore`,
`.ignore`, and `.fdignore`, skip hidden directories and `node_modules`, and
stop below a directory containing a manifest. Ordinary `.md` files are
accepted only when an explicit configured path selects the file. Project roots
are filtered before canonicalization or traversal unless the workspace is
trusted; only the workspace-root `.agents/skills` participates.

Discovery retains bounded metadata and provenance, not skill bodies. The
typed `/skill:<id> [args]` command loads a winning skill after revalidating its
manifest, while the legacy `@skill <id>` parser remains available in
`tea-context`. `disable-model-invocation: true` removes metadata from the model
prompt but does not remove explicit invocation or catalog visibility.

The catalog also owns `read_skill_resource`, a read-only, race-checked,
bounded UTF-8 read beneath the winning skill directory. It rejects traversal,
absolute paths, symlink escapes, directories, binary/oversized files, and
changed identities. It does not execute scripts or broaden workspace tools.
The catalog is immutable for the lifetime of the service; restart or rebuild
the service to observe filesystem changes. `CodingAgentBuilder` and
`CodingAgentService` share the same `Arc<ResourceCatalog>` snapshot.
