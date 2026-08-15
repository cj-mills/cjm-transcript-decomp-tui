# cjm-transcript-decomp-tui

<!-- generated from the context graph by `cjm-context-graph readme` — do not edit by hand; edit the graph (the urge to hand-edit = move it on-graph) -->

Decomp-batch TUI driver for the transcript-decomposition workflow: browse the transcription core's run manifests, queue an ordered multi-run batch with per-run authoritative-text (text-from) picks, preview how the batch folds into headless invocations, then hand off to cjm-transcript-decomp-core's batch runner — one loaded capability stack per text-from group, every run byte-identical to a hand-launched one. Includes a read-only decomp-results view over the decomp core's own manifests.

## Modules

- **`cjm_transcript_decomp_tui.__init__`**
- **`cjm_transcript_decomp_tui.app`** — The decomp-batch TUI: pick transcription runs into an ordered batch, then a
- **`cjm_transcript_decomp_tui.cli`** — The console-script driver: run the batch-selection TUI, then hand each
- **`cjm_transcript_decomp_tui.discovery`** — Capability-role discovery by manifest surface match — the journaling-by-
- **`cjm_transcript_decomp_tui.runs`** — Run-manifest indexes for the decomp-batch TUI (work item 0ff6bf0f): the
- **`cjm_transcript_decomp_tui.segments`** — Fine-segment inspection for the decomp TUI (work item 166dd2b8, half a):
- **`cjm_transcript_decomp_tui.state`** — Sidecar TUI state: last-used batch settings persisted across sessions (the

## API

### `cjm_transcript_decomp_tui.app`

- `DecompApp` _class_ — Decomp-batch setup, v0 thinnest slice: one selection stage over the

### `cjm_transcript_decomp_tui.cli`

- `batch_argv` _function_ — Render one hand-off group as headless decomp-core argv.
- `build_parser` _function_ — The TUI driver's argument surface (batch-setup options + core passthrough).
- `event_split_batch_error` _function_ — One propset carves ONE source: the core applies --event-propset to every
- `hand_off` _function_ — The shared driver tail: persist the confirmed choices, adopt the in-app
- `main` _function_ — Resolve the shared setup surface, run the batch app, hand off — the
- `resolve_settings` _function_ — Resolve the batch-setup settings every shell shares (flags > persisted
- `resolve_split_flags` _function_ — Resolve the post-parse split-flag contract (pure; main() calls it

### `cjm_transcript_decomp_tui.discovery`

- `discover_capability` _function_ — Pick a DEFAULT capability for a role by surface match.
- `manifests_with_method` _function_ — Enumerate installed capabilities whose structural surface lists `method`.

### `cjm_transcript_decomp_tui.runs`

- `DecompIndex` _class_ — Decomp-core run manifests read back: coverage chips for the batch stage
- `PropsetIndex` _class_ — Proposal-set manifests under the workspace proposals/ dir — the model's
- `SourceRunIndex` _class_ — Transcription-core run manifests — the decomp workflow's SOURCES — plus
- `TrainingRunIndex` _class_ — Training-run manifests under the workspace training-runs/ dir — the
- `group_batches` _function_ — Fold an ordered batch selection into headless hand-off groups.

### `cjm_transcript_decomp_tui.segments`

- `SegmentStack` _class_ — One lazily-opened READ-ONLY graph seat, keyed by db path.
- `aseg_index_for` _function_ — Which coarse AudioSegment a source-coordinate time falls in (pure;
- `build_display` _function_ — Interleave gap markers into the paintable entry list (pure).
- `capability_config_schema` _function_ — A capability's config_schema off its installed manifest (json read, pure).
- `find_gaps` _function_ — Uncovered timestamp spans between committed segments (pure).
- `fmt_ts` _function_ — Source-coordinate timestamp for listing rows (pure).
- `locate_span` _function_ — Resolve a source-coordinate span onto its owning coarse WAV (pure).
- `ordered_rows` _function_ — Re-impose the manifest's spine order on fetched rows (pure).
- `predicted_rows` _function_ — Synthesize segment-shaped rows for a probe's predicted skeleton (pure).
- `probe_compare` _function_ — Compare a VAD probe against the committed skeleton (pure).
- `realign_rows` _function_ — Re-run the decomp pipeline's own text fold over a PROBE skeleton (pure).
- `split_predicted` _function_ — Run the decomp pipeline's own SENTENCE-SPLIT stage over a probe skeleton
- `vad_summary` _function_ — The decomp run's recorded VAD config, one status line (pure).

### `cjm_transcript_decomp_tui.state`

- `load_state` _function_ — Read this project's persisted TUI state.
- `save_state` _function_ — Merge updates into the persisted state and write it back (best-effort:
- `state_path` _function_ — Where this project's TUI state lives.

## Dependencies

**Depends on:** `cjm-substrate-tui-kit`, `cjm-transcript-decomp-core`, `textual`
**Used by:** `cjm-transcript-decomp-qt`
