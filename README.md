# cjm-transcript-decomp-tui

<!-- generated from the context graph by `cjm-context-graph readme` — do not edit by hand; edit the graph (the urge to hand-edit = move it on-graph) -->

Decomp-batch TUI driver for the transcript-decomposition workflow: browse the transcription core's run manifests, queue an ordered multi-run batch with per-run authoritative-text (text-from) picks, preview how the batch folds into headless invocations, then hand off to cjm-transcript-decomp-core's batch runner — one loaded capability stack per text-from group, every run byte-identical to a hand-launched one. Includes a read-only decomp-results view over the decomp core's own manifests.

## Modules

- **`cjm_transcript_decomp_tui.__init__`**
- **`cjm_transcript_decomp_tui.app`** — The decomp-batch TUI: pick transcription runs into an ordered batch, then a
- **`cjm_transcript_decomp_tui.cli`** — The console-script driver: run the batch-selection TUI, then hand each

## API

### `cjm_transcript_decomp_tui.app`

- `DecompApp` _class_ — Decomp-batch setup, v0 thinnest slice: one selection stage over the

### `cjm_transcript_decomp_tui.cli`

- `main` _function_ — Resolve the shared setup surface, run the batch app, hand off — the

## Dependencies

**Depends on:** `cjm-substrate-tui-kit`, `cjm-transcript-decomp-core`, `textual`
