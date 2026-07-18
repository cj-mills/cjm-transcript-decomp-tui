# cjm-transcript-decomp-tui

<!-- generated from the context graph by `cjm-context-graph readme` — do not edit by hand; edit the graph (the urge to hand-edit = move it on-graph) -->

Decomp-batch TUI driver for the transcript-decomposition workflow: browse the transcription core's run manifests, queue an ordered multi-run batch with per-run authoritative-text (text-from) picks, preview how the batch folds into headless invocations, then hand off to cjm-transcript-decomp-core's batch runner — one loaded capability stack per text-from group, every run byte-identical to a hand-launched one. Includes a read-only decomp-results view over the decomp core's own manifests.

## Modules

- **`cjm_transcript_decomp_tui.__init__`**
- **`cjm_transcript_decomp_tui.app`**
- **`cjm_transcript_decomp_tui.cli`**
- **`cjm_transcript_decomp_tui.discovery`**
- **`cjm_transcript_decomp_tui.runs`**
- **`cjm_transcript_decomp_tui.state`**

## API

### `cjm_transcript_decomp_tui.app`

- `DecompApp` _class_ — Decomp-batch setup, v0 thinnest slice: one selection stage over the

### `cjm_transcript_decomp_tui.cli`

- `batch_argv` _function_ — Render one text-from group as headless decomp-core argv.
- `build_parser` _function_ — The TUI driver's argument surface (batch-setup options + core passthrough).
- `main` _function_ — Resolve settings (flags > persisted state > manifest discovery), run the

### `cjm_transcript_decomp_tui.discovery`

- `discover_capability` _function_ — Pick a DEFAULT capability for a role by surface match.
- `manifests_with_method` _function_ — Enumerate installed capabilities whose structural surface lists `method`.

### `cjm_transcript_decomp_tui.runs`

- `DecompIndex` _class_ — Decomp-core run manifests read back: coverage chips for the batch stage
- `SourceRunIndex` _class_ — Transcription-core run manifests — the decomp workflow's SOURCES — plus
- `group_by_text_from` _function_ — Fold an ordered batch selection into headless hand-off groups.

### `cjm_transcript_decomp_tui.state`

- `load_state` _function_ — Read this project's persisted TUI state.
- `save_state` _function_ — Merge updates into the persisted state and write it back (best-effort:
- `state_path` _function_ — Where this project's TUI state lives.

## Dependencies

**Depends on:** `cjm-substrate-tui-kit`, `cjm-transcript-decomp-core`, `textual`
