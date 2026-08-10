# tea-coding

Mode-neutral Coding CLI product assembly for `tea-rs`.

The package is `tea-coding`; Rust code imports it as `tea_coding`. It composes versioned secret-free settings, injected application paths, canonical project trust, trusted declarative resources, a concise product identity prompt, coding profile/policy, live or fake provider, native workspace tools (`read`, `grep`, `find`, `ls`, `write`, `edit`, and `bash`), optional client or hosted web tools, one SQLite store/catalog, and the mode-neutral `CodingAgentService`.

`CodingAgentService` exposes session lifecycle and query operations plus prompt, steering, follow-up, abort, approval, model/profile, compaction, fork, and naming commands. Prompt and approval-continuation acceptance are returned before their owned tasks complete so a caller can subscribe first and stream through the runtime's bounded event channel; `wait` and `shutdown` await task ownership. SQLite remains authoritative across approval pause and process rebuild.

Embedding products select their policy surface through
`CodingAgentBuilder::execution_surface`. The builder defaults to `Cli` for the
reference command-line product; desktop and IDE hosts must select `Desktop` or
`Ide` explicitly so policy observations retain the real outward boundary.

Interactive, print, JSON event, and JSONL/RPC modes must all call this same service. This crate does not depend on Ratatui, Crossterm, a clipboard implementation, or another UI framework.
