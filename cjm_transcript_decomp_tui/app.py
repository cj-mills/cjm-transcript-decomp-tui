"""The decomp-batch TUI: pick transcription runs into an ordered batch, then a
headless hand-off to the decomp core's batch runner (work item 0ff6bf0f).

v0 is deliberately the analogy's THIN slice: the felt pain was queueing — the
decomp CLI took one transcription-run manifest per invocation — so the app is
one selection stage over the run-manifest corpus plus a read-only results
view; which comparison/inspection views decomp actually needs stays
demand-driven from real decomp sessions, not mirrored speculatively from the
transcription TUI. Presentation lessons carried: spans-only Rich styling (base
styles bleed), no markup parsing of content strings (bare [/] would
MarkupError), AUTO_FOCUS None so bindings own the keys, one-line listing rows,
coalesced repaints (kit RepaintThrottle)."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from cjm_substrate.core.manager import CapabilityManager
from cjm_substrate_tui_kit.audio import ChunkPlayer, load_chunk
from cjm_substrate_tui_kit.form import ConfigForm
from cjm_substrate_tui_kit.repaint import RepaintThrottle
from cjm_substrate_tui_kit.viewport import tail, visible_slice
from cjm_transcript_decomp_core.cli import load_capabilities
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Input, Static

from .runs import DecompIndex, group_batches, PropsetIndex, SourceRunIndex, TrainingRunIndex
from .segments import (aseg_index_for, build_display, capability_config_schema, fmt_ts, locate_span,
                       predicted_rows, probe_compare, realign_rows, SegmentStack, split_predicted,
                       vad_summary)


class DecompApp(App):
    """Decomp-batch setup, v0 thinnest slice: one selection stage over the
    transcription-run corpus (RUNS), a read-only decomp-results view (RESULTS),
    then exit with a grouped batch plan.

    RUNS lists the transcription core's own run manifests newest-first; enter
    toggles a run into the ORDERED batch (the row shows its queue position),
    t cycles a multi-transcriber run's authoritative text_from (default: the
    accuracy slot), and the batch panel below previews exactly how many
    headless invocations the selection folds into — one per distinct
    text_from, each riding ONE loaded capability stack. The app itself never
    loads a capability: confirm exits with the plan and the driver hands off
    to the decomp core's batch CLI, so TUI-queued runs are byte-identical to
    hand-launched ones. RESULTS reads the decomp core's own manifests back
    (list -> per-source drill); segment TEXTS live in the graph, not the
    manifest, so deeper inspection waits for real-session demand.
    """

    AUTO_FOCUS = None

    CSS = """
    #main { height: 1fr; }
    #status { dock: bottom; height: 1; }
    """

    BINDINGS = [
        Binding("j", "move(1)", "down"),
        Binding("down", "move(1)", "down", show=False),
        Binding("k", "move(-1)", "up"),
        Binding("up", "move(-1)", "up", show=False),
        Binding("enter", "select", "pick/open"),
        Binding("space", "select", "pick", show=False),
        Binding("t", "cycle_text_from", "text-from"),
        Binding("v", "results", "decomp runs"),
        Binding("r", "reload", "reload"),
        Binding("n", "confirm", "confirm batch"),
        Binding("b", "back", "back"),
        Binding("c", "vad_config", "vad config"),
        Binding("p", "probe", "probe"),
        Binding("s", "toggle_split", "split preview", show=False),
        Binding("R", "toggle_respine", "respine", show=False),
        Binding("e", "toggle_event_split", "event-split", show=False),
        Binding("E", "cycle_propset", "propset", show=False),
        Binding("P", "propose_missing", "propose", show=False),
        Binding("left_square_bracket", "speed_down", "slower", show=False,
                key_display="["),
        Binding("right_square_bracket", "speed_up", "faster", show=False,
                key_display="]"),
        Binding("escape", "stop_audio", "stop", show=False, priority=True),
        Binding("q", "quit_app", "quit"),
    ]

    REPAINT_INTERVAL = 1 / 30  # Coalescing window: at most ~30 full repaints/s

    def __init__(self, manifests_dir: str,          # Capability manifests directory
                 *, runs_dir: str = "runs",         # Both cores' cwd-relative manifest dir
                 sysmon_capability: Optional[str] = None,  # Monitor for GPU attribution (CR-7)
                 graph_capability: str = "cjm-capability-graph-sqlite",  # Extension target
                 graph_db_path: Optional[str] = None,      # Caller-wins graph db override
                 gap_threshold: float = 2.0,               # Seconds of uncovered span that paints a gap row
                 sentence_split: bool = True,              # Seed the batch-level split toggle (DEFAULT-ON, DEC 552bde8d; s toggles)
                 respine: bool = False,                    # Seed the batch-level respine toggle (fresh spine, same config — DEC 9241564f; R toggles)
                 proposals_dir: str = "proposals",         # Workspace proposals/ — the propset picker's discovery root
                 event_split: bool = False,                # Seed the batch-level event-carve toggle (e toggles; DEC ae450551)
                 propset_pin: Optional[str] = None,        # Explicit --event-propset (pins EVERY group; per-run resolution off)
                 training_runs_dir: str = "training-runs", # Workspace training-runs/ — P propose-now model discovery
                 training_run_pin: Optional[str] = None,   # Marker/flag pin naming the CURRENT model (data, not code)
                 event_capability: str = "cjm-capability-pyannote"):  # Capability serving the audio_event_detection task
        super().__init__()
        self.respine = respine
        self.manifests_dir = manifests_dir
        self.sysmon_capability = sysmon_capability
        self.graph_capability = graph_capability
        self.graph_db_path = graph_db_path
        self.src_index = SourceRunIndex(runs_dir)
        self.dec_index = DecompIndex(runs_dir)
        # Event-carve batch state (DEC ae450551, the hub-launch path's way in):
        # per-run propsets resolve latest-by-source from the workspace
        # proposals/ dir; E cycles a source's older sets; an explicit CLI pin
        # (scripted path) short-circuits resolution entirely.
        self.propset_index = PropsetIndex(proposals_dir)
        self.event_split = event_split
        self.propset_pin = propset_pin
        self.propset_pick: Dict[str, str] = {}  # _path -> explicit E-cycle propset override
        # P propose-now pre-stage (DEC 1cfe6d0f): manifest-driven model
        # discovery + the armed two-step confirm; the capability seat opens
        # lazily per batch and tears down after.
        self.trainrun_index = TrainingRunIndex(training_runs_dir)
        self.training_run_pin = training_run_pin
        self.event_capability = event_capability
        self.propose_armed: Optional[Dict[str, Any]] = None  # {"run": manifest, "targets": [run manifests]}
        self.propose_busy = False
        self.stage = "runs"
        self.cursor = 0
        # The batch is keyed by manifest _path (stable across r-reloads, where
        # a list index would silently re-target after a re-sort).
        self.picked: List[str] = []
        self.text_from: Dict[str, str] = {}  # _path -> explicit t-cycle override
        self._decomp_counts: Dict[str, int] = {}
        self.results_run: Optional[int] = None  # None = decomp list; else drilled index
        self.results_cursor = 0
        self.results_src = 0
        self.gap_threshold = gap_threshold
        # Segments drill (166dd2b8 half a): a lazy READ-ONLY graph seat reads
        # the drilled source's committed fine spine back; entries interleave
        # focusable gap rows (de994164 — the miss class lives BETWEEN rows).
        self._seg_stack: Optional[SegmentStack] = None
        self.seg_rows: List[Dict[str, Any]] = []
        self.seg_display: List[Tuple[str, int, float]] = []
        self.seg_cursor = 0
        self.seg_title = ""
        self.seg_vad: Optional[str] = None
        self.seg_missing = 0
        self.seg_source_id = ""
        self.seg_vad_id = "cjm-capability-silero-vad"
        self.seg_vad_cfg: Dict[str, Any] = {}
        self.seg_fa_id = "cjm-capability-qwen3-forced-aligner"
        self.seg_fa_cfg: Dict[str, Any] = {}
        self.seg_seg_id = "cjm-capability-pysbd"
        self.seg_seg_cfg: Dict[str, Any] = {}
        self.seg_text_from = ""
        self.seg_asegs: List[Dict[str, Any]] = []
        # Audio + probe (drive feedback 2026-07-22): r auditions the focused
        # span — gaps INCLUDED, their audio exists only in the coarse WAVs —
        # and the c-form probes VAD configs against the committed skeleton
        # (166dd2b8 half b, probe-only per ee4a4b9c).
        self.player: Optional[ChunkPlayer] = None
        self.speed = 1.0
        self.vad_form: Optional[ConfigForm] = None
        self.form_cursor = 0
        self.form_editing = False
        self.probe_aseg: Optional[int] = None
        self.probe_result: Optional[Dict[str, Any]] = None
        self.probe_busy = False
        # Sentence-split preview (DEC f1024568 deliverable d): s toggles the
        # pipeline's own split stage over the probed skeleton — the view shows
        # what a --sentence-split re-decomposition would commit, pre-commit.
        self.probe_split = False
        # Batch-level sentence-split (s in RUNS): rides the confirmed plan into
        # the core hand-off as --sentence-split — the hub-launch path's only
        # way to enable the split stage (no flags survive the hub's exec).
        self.sentence_split = sentence_split
        self._probe_ctx: Optional[Tuple[Dict[str, Any], List[Tuple[float, float]],
                                        Optional[str],
                                        Optional[List[Tuple[str, float, float]]],
                                        Optional[List[Tuple[int, int]]]]] = None
        # Probe-skeleton view (drive feedback: counts alone cannot answer
        # "worth a re-decomposition?" — the predicted skeleton must WALK).
        self.probe_view_rows: List[Dict[str, Any]] = []
        self.probe_view_display: List[Tuple[str, int, float]] = []
        self.probe_view_cursor = 0
        self.error: Optional[str] = None
        self.notice: Optional[str] = None
        self._throttle = RepaintThrottle(self._paint_now, self.set_timer,
                                         self.REPAINT_INTERVAL)

    def compose(self) -> ComposeResult:
        yield Static(id="main")
        yield Static(id="status")
        # Transient value-entry Input for the VAD form's open fields
        # (transcription-TUI escape-hatch precedent); hidden until opened.
        editor = Input(id="editor")
        editor.display = False
        yield editor

    def on_mount(self) -> None:
        self._reload_indexes()
        self._paint()

    def on_resize(self, event) -> None:
        self._paint()

    # ---- painting (spans only — a base style on a composed row bleeds) ----

    def _paint(self) -> None:
        """Request a repaint, coalescing bursts (kit RepaintThrottle)."""
        self._throttle.request()

    def _paint_now(self) -> None:
        pane = {"runs": self._paint_runs,
                "results": self._paint_results,
                "segments": self._paint_segments,
                "vadform": self._paint_vadform,
                "probeview": self._paint_probeview}[self.stage]()
        self.query_one("#main", Static).update(pane)
        status = Text()
        # Decomp ALWAYS journals — extension IS graph writing — so the chip
        # shows the target, never a NOT-JOURNALED state (that state cannot
        # exist for this workflow).
        status.append(f" graph→{self.graph_capability} ", style="green")
        if self.graph_db_path:
            status.append(f"@{tail(self.graph_db_path, 28)} ", style="dim")
        if self.error:
            status.append(f" {self.error} ", style="bold red")
        elif self.notice:
            status.append(f" {self.notice} ", style="cyan")
        else:
            hints = {
                "runs": "enter/space pick · t text-from · s split · R respine · e event · E propset · P propose · v decomp runs · r reload · n confirm · q quit",
                "results": ("enter open run · j/k walk · b back · q quit"
                            if self.results_run is None
                            else "enter segments · j/k source · b decomp list · q quit"),
                "segments": "j/k walk · r play · [ ] speed · c vad config · b back · q quit",
                "vadform": "j/k field · enter cycle/edit · p probe · s split · v skeleton · b back · q quit",
                "probeview": "j/k walk · r play · s split · b form · q quit",
            }[self.stage]
            status.append(f" {self.stage.upper()}  ·  {hints}", style="dim")
        status.truncate(max(20, self.size.width), overflow="ellipsis")
        self.query_one("#status", Static).update(status)

    def _paint_runs(self) -> Text:
        # One screen line per listing row (row discipline: wrapped rows eat the
        # windowing budget); the batch panel keeps a fixed tail below the list.
        width = max(20, self.size.width)
        out = Text()
        runs = self.src_index.runs
        out.append(f" Transcription runs ({len(runs)})  ·  {self.src_index.runs_dir}/\n\n",
                   style="bold")
        if not runs:
            out.append("   (no transcription run manifests found — confirmed "
                       "transcription runs land here)\n", style="dim")
        batches = self._batches()
        panel = (2 + len(batches)) if self.picked else 0
        self.cursor = max(0, min(self.cursor, max(0, len(runs) - 1)))
        # Detail region (614dd647): up to 4 source rows + the graph-db line.
        detail = (min(4, len(runs[self.cursor].get("sources") or [])) + 3) if runs else 0
        budget = max(3, max(4, self.size.height - 1) - 4 - panel - detail)
        start, end, above, below = visible_slice(len(runs), self.cursor, budget)
        if above:
            out.append(f"   … {above} above\n", style="dim")
        picked_set = set(self.picked)
        for i in range(start, end):
            m = runs[i]
            focus = (i == self.cursor)
            key = m["_path"]
            line = Text()
            line.append(" > " if focus else "   ", style="bold cyan" if focus else "dim")
            if key in picked_set:
                # Queue position, not just [x]: the batch is ORDERED.
                line.append(f"[{self.picked.index(key) + 1}] ", style="green")
            else:
                line.append("[ ] ", style="dim")
            when = time.strftime("%Y-%m-%d %H:%M",
                                 time.localtime(float(m.get("created_at") or 0)))
            line.append(str(m["run_id"]), style="bold" if focus else "")
            line.append(f"  {when}  {len(m.get('sources') or [])} src · "
                        f"{SourceRunIndex.segment_count(m)} seg", style="dim")
            names = [t.removeprefix("cjm-capability-")
                     for t in SourceRunIndex.transcribers(m)]
            if names:
                line.append("  " + "+".join(names), style="dim")
            tf = self._resolved_text_from(m)
            if len(names) > 1 and tf:
                line.append(f"  tf={tf.removeprefix('cjm-capability-')}",
                            style="yellow" if key in self.text_from else "dim cyan")
            n = self._decomp_counts.get(str(Path(key).resolve()), 0)
            if n:
                line.append(f"  ·decomp×{n}", style="dim cyan")
            if self.event_split:
                # Event-armed rows carry their resolved propset (DEC ae450551):
                # the carve input must be visible BEFORE confirm, per run.
                ps = self._resolved_propset(m)
                if ps is None:
                    line.append("  ⚡no propset", style="bold red")
                else:
                    chosen = next((s for s in self._propsets_for(m)
                                   if s["_path"] == ps), None)
                    label = (PropsetIndex.summary(chosen) if chosen
                             else tail(ps, 20))
                    line.append(f"  ⚡{label}",
                                style="yellow" if key in self.propset_pick
                                else "dim magenta")
            line.truncate(width, overflow="ellipsis")
            out.append_text(line)
            out.append("\n")
        if below:
            out.append(f"   … {below} below\n", style="dim")
        if runs:
            # Focused-run detail (614dd647): WHAT was transcribed + WHICH graph
            # it landed in, in-pane — identity must not need a round-trip to
            # the transcription TUI.
            focused = runs[self.cursor]
            out.append("\n")
            srcs = focused.get("sources") or []
            for s in srcs[:4]:
                nm = Path(str(s.get("source_path") or "?")).stem
                row = Text()
                row.append(f"   · {nm}", style="bold")
                row.append(f"  {len(s.get('segments') or [])} seg", style="dim")
                row.truncate(width, overflow="ellipsis")
                out.append_text(row)
                out.append("\n")
            if len(srcs) > 4:
                out.append(f"   … {len(srcs) - 4} more source(s)\n", style="dim")
            db = self._resolved_graph_db(focused)
            dbline = Text()
            if db is None:
                dbline.append("   ⚠ no graph db recorded in this manifest",
                              style="bold red")
            elif not Path(db).exists():
                dbline.append(f"   ⚠ recorded graph db missing on disk: {tail(db, 40)}",
                              style="bold red")
            else:
                dbline.append(f"   graph db: {tail(db, 48)}", style="dim cyan")
            dbline.truncate(width, overflow="ellipsis")
            out.append_text(dbline)
            out.append("\n")
        if self.picked:
            out.append(f"\n Batch ({len(self.picked)} run(s) -> {len(batches)} "
                       f"invocation(s))", style="bold")
            if self.sentence_split:
                out.append("  ✂ sentence-split ON", style="bold magenta")
            if self.respine:
                out.append("  ⟳ respine ON (fresh spine, same config)", style="bold cyan")
            if self.event_split:
                out.append("  ⚡ event-split ON (model cuts, ad963c57)",
                           style="bold magenta")
            out.append(":\n", style="bold")
            for bi, ((tf, db, ps), paths) in enumerate(batches):
                short = (tf or "?").removeprefix("cjm-capability-")
                line = Text()
                line.append(f"   {bi + 1}. text-from {short}", style="green")
                if db is None:
                    line.append("  ⚠ no graph db recorded", style="bold red")
                elif not Path(db).exists():
                    line.append(f"  ⚠ db missing: {tail(db, 24)}", style="bold red")
                else:
                    line.append(f"  db {tail(db, 24)}", style="dim")
                if self.event_split:
                    if ps is None:
                        line.append("  ⚡ no propset", style="bold red")
                    else:
                        line.append(f"  ⚡ {Path(ps).parent.name[-12:]}",
                                    style="dim magenta")
                line.append(": ", style="green")
                line.append(", ".join(self._run_id_for(p) for p in paths), style="dim")
                line.truncate(width, overflow="ellipsis")
                out.append_text(line)
                out.append("\n")
        return out

    def _paint_results(self) -> Text:
        width = max(20, self.size.width)
        out = Text()
        runs = self.dec_index.runs
        if self.results_run is None:
            out.append(f" Decomp runs ({len(runs)})  ·  {self.dec_index.runs_dir}/\n\n",
                       style="bold")
            if not runs:
                out.append("   (no decomp manifests found — batch hand-offs land here)\n",
                           style="dim")
            budget = max(3, max(4, self.size.height - 1) - 4)
            start, end, above, below = visible_slice(len(runs), self.results_cursor,
                                                     budget)
            if above:
                out.append(f"   … {above} above\n", style="dim")
            for i in range(start, end):
                m = runs[i]
                focus = (i == self.results_cursor)
                line = Text()
                line.append(" > " if focus else "   ",
                            style="bold cyan" if focus else "dim")
                when = time.strftime("%Y-%m-%d %H:%M",
                                     time.localtime(float(m.get("created_at") or 0)))
                srcs = m.get("sources") or []
                segs = sum(int(s.get("segment_count") or 0) for s in srcs)
                line.append(str(m["run_id"]), style="bold" if focus else "")
                line.append(f"  {when}  {len(srcs)} source(s) · {segs} seg", style="dim")
                # Source titles in-row (614dd647 extended to the decomp list —
                # identity must not need a drill now the segments view is live).
                names = [str(s.get("title") or "?") for s in srcs]
                if names:
                    extra = f" +{len(names) - 2}" if len(names) > 2 else ""
                    line.append("  " + ", ".join(names[:2]) + extra, style="dim cyan")
                tf = (m.get("config") or {}).get("text_from")
                if tf:
                    line.append(f"  tf={str(tf).removeprefix('cjm-capability-')}",
                                style="dim cyan")
                line.truncate(width, overflow="ellipsis")
                out.append_text(line)
                out.append("\n")
            if below:
                out.append(f"   … {below} below\n", style="dim")
            return out
        m = runs[self.results_run]
        header = Text()
        header.append(f" {m['run_id']}", style="bold")
        src_manifest = m.get("source_manifest")
        if src_manifest:
            header.append(f"   <- {tail(str(src_manifest), 48)}", style="dim")
        header.truncate(width, overflow="ellipsis")
        out.append_text(header)
        out.append("\n\n")
        srcs = m.get("sources") or []
        if not srcs:
            out.append("   (no sources extended — the run failed or was aborted "
                       "before its first commit)\n", style="dim")
        budget = max(3, max(4, self.size.height - 1) - 4)
        self.results_src = max(0, min(self.results_src, max(0, len(srcs) - 1)))
        start, end, above, below = visible_slice(len(srcs), self.results_src, budget)
        if above:
            out.append(f"   … {above} above\n", style="dim")
        for i in range(start, end):
            s = srcs[i]
            focus = (i == self.results_src)
            line = Text()
            line.append(" > " if focus else "   ",
                        style="bold cyan" if focus else "dim")
            line.append(str(s.get("title") or "?"), style="bold" if focus else "")
            line.append(f"  {int(s.get('segment_count') or 0)} fine segment(s)",
                        style="dim")
            node = str(s.get("source_node_id") or "")
            if node:
                line.append(f"  Source {node[:8]}…", style="dim cyan")
            line.truncate(width, overflow="ellipsis")
            out.append_text(line)
            out.append("\n")
        if below:
            out.append(f"   … {below} below\n", style="dim")
        return out

    def _paint_segments(self) -> Text:
        """The drilled source's committed fine spine + interleaved gap rows.

        Gap rows are FOCUSABLE (build_display): the gap is the inspection
        target here — its detail names the exact uncovered span to audition
        against the source (de994164)."""
        width = max(20, self.size.width)
        out = Text()
        head = Text()
        head.append(f" {self.seg_title}", style="bold")
        head.append(f"  ·  {len(self.seg_rows)} fine segment(s)", style="dim")
        gaps = sum(1 for kind, _, _ in self.seg_display if kind == "gap")
        if gaps:
            head.append(f"  ·  {gaps} gap(s) ≥{self.gap_threshold:g}s", style="bold yellow")
        if self.seg_missing:
            head.append(f"  ·  {self.seg_missing} id(s) not in graph", style="bold red")
        head.truncate(width, overflow="ellipsis")
        out.append_text(head)
        out.append("\n")
        if self.seg_vad:
            vline = Text()
            vline.append(f"   {self.seg_vad}", style="dim cyan")
            vline.truncate(width, overflow="ellipsis")
            out.append_text(vline)
            out.append("\n")
        out.append("\n")
        entries = self.seg_display
        if not entries:
            out.append("   (no committed segments found in the graph)\n", style="dim")
            return out
        self.seg_cursor = max(0, min(self.seg_cursor, len(entries) - 1))
        # 4 fixed detail lines below the list (focused-entry expansion).
        budget = max(3, max(4, self.size.height - 1) - 8)
        start, end, above, below = visible_slice(len(entries), self.seg_cursor, budget)
        if above:
            out.append(f"   … {above} above\n", style="dim")
        for i in range(start, end):
            kind, ri, secs = entries[i]
            focus = (i == self.seg_cursor)
            line = Text()
            line.append(" > " if focus else "   ", style="bold cyan" if focus else "dim")
            r = self.seg_rows[ri]
            if kind == "gap":
                s = float(r.get("start_time") or 0.0)
                line.append(f"⚠ {secs:.1f}s gap", style="bold yellow")
                line.append(f"  {fmt_ts(s - secs)} → {fmt_ts(s)}  no segment was cut here",
                            style="yellow" if focus else "dim")
            else:
                s, e = r.get("start_time"), r.get("end_time")
                span = (f"{fmt_ts(float(s))}–{fmt_ts(float(e))}"
                        if s is not None and e is not None else "?–?")
                line.append(f"{int(r.get('index') or ri):>4} ", style="bold" if focus else "dim")
                line.append(f" {span}  ", style="dim")
                line.append(str(r.get("text") or "").replace("\n", " "))
            line.truncate(width, overflow="ellipsis")
            out.append_text(line)
            out.append("\n")
        if below:
            out.append(f"   … {below} below\n", style="dim")
        # Focused-entry detail: the full text a one-line row truncated, or a
        # gap's exact uncovered span (the check-the-source pointer).
        kind, ri, secs = entries[self.seg_cursor]
        r = self.seg_rows[ri]
        out.append("\n")
        if kind == "gap":
            s = float(r.get("start_time") or 0.0)
            out.append(f"   uncovered span {fmt_ts(s - secs)} → {fmt_ts(s)} "
                       f"({secs:.1f}s) — audition the source audio here\n",
                       style="yellow")
        else:
            s, e = r.get("start_time"), r.get("end_time")
            if s is not None and e is not None:
                out.append(f"   {fmt_ts(float(s))} → {fmt_ts(float(e))} · "
                           f"{float(e) - float(s):.1f}s\n", style="dim cyan")
            text = str(r.get("text") or "").replace("\n", " ")
            w = max(10, width - 4)
            for j in range(0, min(len(text), 3 * w), w):
                out.append(f"   {text[j:j + w]}\n")
        return out

    def _paint_vadform(self) -> Text:
        """The VAD config form + probe readout (166dd2b8 half b, probe-only).

        Rows come from the capability's manifest config_schema (kit ConfigForm,
        the f4a9d253 keystone) seeded with the RUN's recorded config; p sweeps
        the frozen target chunk and paints what the config WOULD cut vs the
        committed skeleton — recovered uncovered spans are the de994164 answer."""
        width = max(20, self.size.width)
        out = Text()
        head = Text()
        head.append(" VAD probe · ", style="bold")
        head.append(self.seg_vad_id.removeprefix("cjm-capability-"), style="bold cyan")
        head.append(f"  ·  {self.seg_title}", style="dim")
        head.truncate(width, overflow="ellipsis")
        out.append_text(head)
        out.append("\n")
        out.append("   probe-only: nothing commits — a promising sweep argues an "
                   "explicit re-decomposition\n", style="dim")
        out.append("   sentence-split preview: ", style="dim")
        out.append("ON" if self.probe_split else "off",
                   style="bold magenta" if self.probe_split else "dim")
        out.append("  (s toggles — the pipeline's post-FA split stage over the "
                   "predicted skeleton)\n\n", style="dim")
        fields = self.vad_form.fields if self.vad_form is not None else []
        for i, f in enumerate(fields):
            focus = (i == self.form_cursor)
            line = Text()
            line.append(" > " if focus else "   ", style="bold cyan" if focus else "dim")
            line.append(f"{f.title:<26}", style="bold" if focus else "")
            base = self.seg_vad_cfg.get(f.key, f.default)
            line.append(f.render(), style="yellow" if f.value != base else "")
            if f.value != base:
                line.append(f"  run: {base}", style="dim cyan")
            line.truncate(width, overflow="ellipsis")
            out.append_text(line)
            out.append("\n")
        out.append("\n")
        if self.probe_aseg is not None and self.seg_asegs:
            a = self.seg_asegs[self.probe_aseg]
            tgt = Text()
            tgt.append(f"   target: coarse chunk {self.probe_aseg + 1}/"
                       f"{len(self.seg_asegs)} · {fmt_ts(a['start'])} → "
                       f"{fmt_ts(a['end'])}", style="dim")
            if not a.get("wav"):
                tgt.append("  ⚠ no model-input WAV on disk", style="bold red")
            tgt.truncate(width, overflow="ellipsis")
            out.append_text(tgt)
            out.append("\n")
        if self.probe_busy:
            out.append("   probing …\n", style="cyan")
        elif self.probe_result is not None:
            r = self.probe_result
            out.append(f"   predicted {r['predicted']} chunk(s) vs {r['committed']} "
                       f"committed · {len(r['recovered'])} recovered span(s)\n",
                       style="bold")
            for s, e in r["recovered"][:8]:
                out.append(f"     + {fmt_ts(s)} → {fmt_ts(e)}  ({e - s:.1f}s) — "
                           "was uncovered\n", style="green")
            if len(r["recovered"]) > 8:
                out.append(f"     … {len(r['recovered']) - 8} more\n", style="dim")
        return out

    def _paint_probeview(self) -> Text:
        """WALK the last probe's predicted skeleton (v from the form).

        Painted with the SEGMENTS grammar so the comparison is direct: same
        rows, same interleaved gap markers (holes the predicted skeleton would
        STILL leave), text borrowed from overlapping committed segments for
        orientation, recovered chunks green — and r auditions any of them
        before a re-decomposition is ever committed (probe-only, ee4a4b9c)."""
        width = max(20, self.size.width)
        out = Text()
        head = Text()
        head.append(" predicted skeleton", style="bold")
        diffs = []
        for f in (self.vad_form.fields if self.vad_form is not None else []):
            base = self.seg_vad_cfg.get(f.key, f.default)
            if f.value != base:
                diffs.append(f"{f.key}={f.render()}")
        head.append("  ·  " + ("; ".join(diffs) if diffs else "run config"),
                    style="yellow" if diffs else "dim")
        if self.probe_aseg is not None and self.seg_asegs:
            a = self.seg_asegs[self.probe_aseg]
            head.append(f"  ·  chunk {self.probe_aseg + 1}: "
                        f"{fmt_ts(a['start'])} → {fmt_ts(a['end'])}", style="dim")
        head.truncate(width, overflow="ellipsis")
        out.append_text(head)
        out.append("\n")
        if self.probe_result is not None:
            r = self.probe_result
            realigned = bool(self.probe_view_rows
                             and self.probe_view_rows[0].get("realigned"))
            tag = ("text FA-realigned — what a re-decomposition would commit"
                   if realigned else
                   "text borrowed from committed rows (FA realign unavailable)")
            out.append(f"   predicted {r['predicted']} vs {r['committed']} committed "
                       f"· {len(r['recovered'])} recovered · {tag}\n", style="dim")
        out.append("\n")
        entries = self.probe_view_display
        if not entries:
            out.append("   (probe predicted no chunks)\n", style="dim")
            return out
        self.probe_view_cursor = max(0, min(self.probe_view_cursor, len(entries) - 1))
        budget = max(3, max(4, self.size.height - 1) - 8)
        start, end, above, below = visible_slice(len(entries), self.probe_view_cursor,
                                                 budget)
        if above:
            out.append(f"   … {above} above\n", style="dim")
        for i in range(start, end):
            kind, ri, secs = entries[i]
            focus = (i == self.probe_view_cursor)
            line = Text()
            line.append(" > " if focus else "   ", style="bold cyan" if focus else "dim")
            row = self.probe_view_rows[ri]
            if kind == "gap":
                s = float(row.get("start_time") or 0.0)
                line.append(f"⚠ {secs:.1f}s gap", style="bold yellow")
                line.append(f"  {fmt_ts(s - secs)} → {fmt_ts(s)}  STILL uncut at "
                            "this config", style="yellow" if focus else "dim")
            else:
                s, e = row.get("start_time"), row.get("end_time")
                span = (f"{fmt_ts(float(s))}–{fmt_ts(float(e))}"
                        if s is not None and e is not None else "?–?")
                rec = bool(row.get("recovered"))
                line.append(f"{int(row.get('index') or ri):>4} ",
                            style="bold" if focus else "dim")
                line.append(f" {span}  ", style="green" if rec else "dim")
                if rec:
                    line.append("+ ", style="bold green")
                if row.get("split"):
                    line.append("✂ ", style="bold magenta")
                text = str(row.get("text") or "").replace("\n", " ")
                empty = ("(no words assigned — audition with r)"
                         if row.get("realigned") else
                         "(no committed text — audition with r)")
                line.append(text if text else empty, style="" if text else "green")
            line.truncate(width, overflow="ellipsis")
            out.append_text(line)
            out.append("\n")
        if below:
            out.append(f"   … {below} below\n", style="dim")
        kind, ri, secs = entries[self.probe_view_cursor]
        row = self.probe_view_rows[ri]
        out.append("\n")
        if kind == "gap":
            s = float(row.get("start_time") or 0.0)
            out.append(f"   still uncovered at this config: {fmt_ts(s - secs)} → "
                       f"{fmt_ts(s)} ({secs:.1f}s)\n", style="yellow")
        else:
            s, e = row.get("start_time"), row.get("end_time")
            if s is not None and e is not None:
                out.append(f"   {fmt_ts(float(s))} → {fmt_ts(float(e))} · "
                           f"{float(e) - float(s):.1f}s"
                           + ("  · recovered span" if row.get("recovered") else "")
                           + "\n", style="green" if row.get("recovered") else "dim cyan")
            text = str(row.get("text") or "").replace("\n", " ")
            w = max(10, width - 4)
            for j in range(0, min(len(text), 3 * w), w):
                out.append(f"   {text[j:j + w]}\n")
        return out

    # ---- selection state helpers (pure reads over the indexes) ----

    def _run_by_path(self, key: str) -> Optional[Dict[str, Any]]:
        """The loaded transcription manifest at a _path key (None after eviction)."""
        for m in self.src_index.runs:
            if m["_path"] == key:
                return m
        return None

    def _run_id_for(self, key: str) -> str:
        m = self._run_by_path(key)
        return str(m["run_id"]) if m else Path(key).stem

    def _resolved_text_from(self, m: Dict[str, Any]) -> Optional[str]:
        """Effective authority pick: the operator's t-cycle override, else the
        convention default (sole transcriber / the accuracy slot)."""
        return (self.text_from.get(m["_path"])
                or SourceRunIndex.default_text_from(m))

    def _resolved_graph_db(self, m: Dict[str, Any]) -> Optional[str]:
        """Effective graph db for one run: the explicit override (flag/state)
        wins, else the db the run RECORDED writing to (e087d059 — provenance-
        following, never the decomp stack's own configured default)."""
        return (self.graph_db_path
                or SourceRunIndex.recorded_graph_db(m, self.graph_capability))

    def _batches(self) -> List[Tuple[Tuple[Optional[str], Optional[str], Optional[str]], List[str]]]:
        """The hand-off fold: key = (text_from, graph_db_path, event_propset) — everything
        that applies invocation-wide on the core CLI. The propset element is
        per-SOURCE (DEC ae450551): distinct sources resolve distinct sets, so
        each event-armed run naturally becomes its own core invocation with
        the right --event-propset; unarmed it is None uniformly and the fold
        keeps its old shape."""
        picks: List[Tuple[str, Tuple[Optional[str], Optional[str], Optional[str]]]] = []
        for key in self.picked:
            m = self._run_by_path(key)
            if m is not None:
                picks.append((key, (self._resolved_text_from(m),
                                    self._resolved_graph_db(m),
                                    self._resolved_propset(m) if self.event_split
                                    else None)))
        return group_batches(picks)

    def _reload_indexes(self) -> None:
        self.src_index.load()
        self.dec_index.load()
        self.propset_index.load()
        self.trainrun_index.load()
        self.propose_armed = None  # any reload invalidates an armed P confirm
        self._decomp_counts = self.dec_index.counts_by_source_manifest()
        # A reload may evict manifests the batch still names; drop those picks
        # rather than hand off paths the core would refuse.
        self.picked = [p for p in self.picked if self._run_by_path(p) is not None]

    def _propsets_for(self, m: Dict[str, Any]) -> List[Dict[str, Any]]:
        """The focused run's candidate proposal sets, newest first (single-
        source runs only — a propset binds ONE source, so a multi-source run
        resolves to nothing and the confirm guard names it)."""
        srcs = m.get("sources") or []
        if len(srcs) != 1:
            return []
        s = srcs[0]
        return self.propset_index.for_source(
            content_hash=str(s.get("content_hash") or "") or None,
            source_path=str(s.get("source_path") or "") or None)

    def _resolved_propset(self, m: Dict[str, Any]) -> Optional[str]:
        """The run's effective --event-propset pointer: CLI pin > E-cycle
        override > latest-by-source (the correction lane's join rule)."""
        if self.propset_pin:
            return self.propset_pin
        pick = self.propset_pick.get(m["_path"])
        if pick:
            return pick
        sets = self._propsets_for(m)
        return sets[0]["_path"] if sets else None

    def _propose_targets(self) -> List[Dict[str, Any]]:
        """Picked single-source runs whose source resolves NO propset — the
        exact set the event-armed confirm guard would refuse (one target per
        source: a propset binds the source, not the run)."""
        out: List[Dict[str, Any]] = []
        seen: set = set()
        for p in self.picked:
            m = self._run_by_path(p)
            if m is None or self._resolved_propset(m) is not None:
                continue
            srcs = m.get("sources") or []
            if len(srcs) != 1:
                continue
            key = str(srcs[0].get("content_hash") or srcs[0].get("source_path") or p)
            if key in seen:
                continue
            seen.add(key)
            out.append(m)
        return out

    async def action_propose_missing(self) -> None:
        """P (DEC 1cfe6d0f): the batch pre-stage — propose over every picked
        event-armed source with no propset, through the capability task
        channel. TWO-STEP: the first press names the resolved model and the
        second press runs it — a wrong model burns a walk session
        (b9717422), so the human stays between resolution and inference."""
        if self.stage != "runs" or self.propose_busy:
            return
        if not self.event_split:
            self.error = "arm event-split (e) first — P proposes for event-armed runs"
            self._paint()
            return
        if self.propose_armed is not None:
            armed, self.propose_armed = self.propose_armed, None
            self.error = None
            self.run_worker(self._run_propose_batch(armed["run"], armed["targets"]),
                            exclusive=True)
            return
        targets = self._propose_targets()
        if not targets:
            self.error = ("every picked run already resolves a propset"
                          if self.picked else
                          "pick runs first — P proposes for the picked batch")
            self._paint()
            return
        self.trainrun_index.load()
        run = self.trainrun_index.resolve(self.training_run_pin)
        if run is None:
            self.error = (f"training-run pin {self.training_run_pin!r} matches nothing "
                          f"under {self.trainrun_index.training_runs_dir}"
                          if self.training_run_pin else
                          f"no training runs under {self.trainrun_index.training_runs_dir}")
            self._paint()
            return
        self.propose_armed = {"run": run, "targets": targets}
        self.error = None
        self.notice = (f"P again: propose {len(targets)} source(s) with "
                       f"{TrainingRunIndex.summary(run)}"
                       + ("" if self.training_run_pin else " — newest run, no pin"))
        self._paint()

    async def _run_propose_batch(self, run_m: Dict[str, Any],
                                 targets: List[Dict[str, Any]]) -> None:
        """The propose seat: load the event capability, run the
        audio_event_detection task per source, tear down, re-index so the
        ⚡chips flip. The SECOND sanctioned exception to the batch app's
        loads-no-capability posture (after the segments drill) — a third
        exception is the signal for a real capability-seat design (82c463fe)."""
        self.propose_busy = True
        manager = None
        landed = 0
        try:
            self.notice = "propose: opening capability seat…"
            self._paint()
            manager = CapabilityManager(search_paths=[Path(self.manifests_dir)])
            load_capabilities(manager, [self.event_capability])
            for i, m in enumerate(targets, 1):
                src = (m.get("sources") or [{}])[0]
                src_path = str(src.get("source_path") or "")
                self.notice = (f"propose {i}/{len(targets)}: {Path(src_path).name} … "
                               "(batch pre-stage)")
                self._paint()
                await manager.execute_capability_task_async(
                    self.event_capability, "audio_event_detection", "propose",
                    training_run=run_m["_path"], source=src_path)
                landed += 1
            self.error = None
            self.notice = (f"⚡ {landed} propset(s) landed with "
                           f"{TrainingRunIndex.summary(run_m)}")
        except (Exception, SystemExit) as e:
            # SystemExit included: load_capabilities exits on a missing
            # capability, and a library exit must paint, not kill the TUI.
            self.error = f"propose failed after {landed} set(s): {e}"
        finally:
            if manager is not None:
                try:
                    manager.unload_capability(self.event_capability)
                except Exception:
                    self.log.error("propose seat teardown failed", exc_info=True)
            self.propose_busy = False
        self._reload_indexes()
        self._paint()

    async def _open_segments(self) -> None:
        """Drill the focused results source into its committed fine spine.

        Opens the lazy READ-ONLY graph seat against the db this decomp run
        RECORDED (e087d059 reused; explicit --graph-db-path still wins), reads
        the manifest's segment_ids back in spine order, and interleaves the
        gap rows (de994164: a chunk VAD never cut is invisible downstream —
        the uncovered span between committed rows is where it shows)."""
        m = self.dec_index.runs[self.results_run]
        srcs = m.get("sources") or []
        if not srcs:
            return
        src = srcs[self.results_src]
        ids = [str(i) for i in (src.get("segment_ids") or [])]
        if not ids:
            self.error = (f"{m['run_id']}: no segment_ids recorded for this "
                          "source (pre-segment_ids manifest?)")
            self._paint()
            return
        db = self.graph_db_path or SourceRunIndex.recorded_graph_db(m, self.graph_capability)
        if not db or not Path(db).exists():
            self.error = ("no graph db recorded for this decomp run — pass --graph-db-path"
                          if not db else f"recorded graph db missing on disk: {tail(db, 40)}")
            self._paint()
            return
        self.notice = f"opening graph (read-only) · {tail(db, 32)} …"
        self._paint_now()
        try:
            if self._seg_stack is None:
                self._seg_stack = SegmentStack(self.manifests_dir, self.graph_capability)
            await self._seg_stack.open(db)
            rows = await self._seg_stack.read_segments(ids)
            rends = sorted({str(r.get("rendition_id"))
                            for r in rows if r.get("rendition_id")})
            src_node = str(src.get("source_node_id") or "")
            asegs = (await self._seg_stack.read_audio_join(src_node, rends)
                     if src_node and rends else [])
        except (Exception, SystemExit) as e:
            # SystemExit included: load_capabilities exits on a missing
            # manifest, and an async action must never take the app with it.
            self.notice = None
            self.error = f"graph open failed: {e}"
            self._paint()
            return
        self.seg_rows = rows
        self.seg_missing = len(ids) - len(rows)
        self.seg_title = str(src.get("title") or "?")
        self.seg_vad = vad_summary(m)
        self.seg_source_id = src_node
        self.seg_asegs = asegs
        self.seg_vad_id = str((m.get("config") or {}).get("vad_capability")
                              or "cjm-capability-silero-vad")
        vcap = (m.get("capabilities") or {}).get(self.seg_vad_id) or {}
        self.seg_vad_cfg = dict(vcap.get("config") or {})
        self.seg_fa_id = str((m.get("config") or {}).get("fa_capability")
                             or "cjm-capability-qwen3-forced-aligner")
        fcap = (m.get("capabilities") or {}).get(self.seg_fa_id) or {}
        self.seg_fa_cfg = dict(fcap.get("config") or {})
        # B.5 probe parity: the split preview segments with the RUN's segmenter
        # when the manifest recorded one, else the pipeline default.
        self.seg_seg_id = str((m.get("config") or {}).get("seg_capability")
                              or "cjm-capability-pysbd")
        scap = (m.get("capabilities") or {}).get(self.seg_seg_id) or {}
        self.seg_seg_cfg = dict(scap.get("config") or {})
        self.seg_text_from = str((m.get("config") or {}).get("text_from") or "")
        self.seg_display = build_display(rows, self.gap_threshold)
        self.seg_cursor = 0
        self.stage = "segments"
        self.notice = None
        self.error = None
        self._paint()

    async def _close_segments(self) -> None:
        """Tear down the read-only graph seat (quit/confirm paths)."""
        if self._seg_stack is not None:
            try:
                await self._seg_stack.close()
            except Exception:
                pass
            self._seg_stack = None

    def _play_focused(self) -> None:
        """Play the focused entry's audio span (r) — gaps INCLUDED.

        A gap's audio exists in NO fine chunk (it was never cut), but the
        coarse model-input WAV spans it, so the audition reads from the owning
        AudioSegment's WAV via locate_span (the 6beaa0e4 demand, served here
        where the gap rows live)."""
        if self.stage == "segments" and self.seg_display:
            entries, rows, cur = self.seg_display, self.seg_rows, self.seg_cursor
        elif self.stage == "probeview" and self.probe_view_display:
            entries, rows, cur = (self.probe_view_display, self.probe_view_rows,
                                  self.probe_view_cursor)
        else:
            return
        kind, ri, secs = entries[cur]
        r = rows[ri]
        if kind == "gap":
            s = float(r.get("start_time") or 0.0)
            span = (s - secs, s)
        else:
            if r.get("start_time") is None or r.get("end_time") is None:
                return
            span = (float(r["start_time"]), float(r["end_time"]))
        loc = locate_span(self.seg_asegs, span[0], span[1])
        if loc is None:
            self.error = "no model-input WAV covers this span"
            self._paint()
            return
        wav, ls, le = loc
        try:
            if self.player is None:
                self.player = ChunkPlayer()
            self.player.play(load_chunk(wav, ls, le, speed=self.speed))
            self.notice = f"▶ {fmt_ts(span[0])} → {fmt_ts(span[1])} ×{self.speed:g}"
            self.error = None
        except Exception as e:
            self.error = f"audio: {e}"
        self._paint()

    def action_stop_audio(self) -> None:
        """Escape: close the transient editor, else stop playback."""
        if self.form_editing:
            self._close_field_editor()
            self._paint()
            return
        if self.player is not None:
            self.player.stop()
            self.notice = None
            self._paint()

    def action_speed_down(self) -> None:
        self._nudge_speed(-0.25)

    def action_speed_up(self) -> None:
        self._nudge_speed(0.25)

    def _nudge_speed(self, delta: float) -> None:
        """[ / ] playback-speed ladder 0.5-3.0 (kit WSOLA — pitch survives)."""
        if self.stage not in ("segments", "probeview"):
            return
        self.speed = max(0.5, min(3.0, self.speed + delta))
        self.notice = f"speed ×{self.speed:g}"
        self._paint()

    def action_vad_config(self) -> None:
        """Open the VAD probe form (c in segments; 166dd2b8 half b).

        Form rows come from the manifest config_schema, seeded with the RUN's
        recorded config; the probe target freezes to the coarse chunk holding
        the focused entry (gap start for gap rows)."""
        if self.stage != "segments":
            return
        schema = capability_config_schema(self.manifests_dir, self.seg_vad_id)
        form = ConfigForm.from_schema(schema)
        if not form.fields:
            self.error = f"{self.seg_vad_id}: no config_schema in {self.manifests_dir}"
            self._paint()
            return
        form.apply(self.seg_vad_cfg)
        self.vad_form = form
        self.form_cursor = 0
        self.probe_result = None
        self._probe_ctx = None
        self.probe_view_rows = []
        self.probe_view_display = []
        self.probe_view_cursor = 0
        t = None
        if self.seg_display:
            kind, ri, secs = self.seg_display[self.seg_cursor]
            r = self.seg_rows[ri]
            if r.get("start_time") is not None:
                t = float(r["start_time"]) - (secs if kind == "gap" else 0.0)
        self.probe_aseg = aseg_index_for(self.seg_asegs, t) if t is not None else None
        self.error = None
        self.stage = "vadform"
        self._paint()

    async def action_probe(self) -> None:
        """Run the form's config over the target coarse WAV (p; probe-only)."""
        if (self.stage != "vadform" or self.vad_form is None or self.probe_busy
                or self._seg_stack is None):
            return
        if self.probe_aseg is None or not self.seg_asegs:
            self.error = "no probe target — reopen the form from a timed entry"
            self._paint()
            return
        a = self.seg_asegs[self.probe_aseg]
        if not a.get("wav"):
            self.error = "target coarse chunk has no model-input WAV on disk"
            self._paint()
            return
        cfg = {f.key: f.value for f in self.vad_form.fields}
        self.probe_busy = True
        self.error = None
        self._paint_now()
        try:
            local = await self._seg_stack.probe_vad(self.seg_vad_id, cfg, a["wav"])
        except (Exception, SystemExit) as e:
            self.probe_busy = False
            self.error = f"probe failed: {e}"
            self._paint()
            return
        # FA leg: realign the coarse Transcript's text over the predicted
        # skeleton — the pipeline's own fold, so the walk shows the text a
        # re-decomposition would COMMIT (drive feedback: borrowed text
        # repeated a split chunk's whole parent). Failure degrades to borrow.
        fa_text: Optional[str] = None
        fa_words: Optional[List[Tuple[str, float, float]]] = None
        seg_spans: Optional[List[Tuple[int, int]]] = None
        rend = a.get("rendition")
        if rend and self.seg_text_from:
            try:
                text = await self._seg_stack.read_transcript_text(
                    rend, self.seg_text_from)
                if text:
                    self.notice = f"aligning (FA · {len(local)} chunk(s)) …"
                    self._paint_now()
                    fa_words = await self._seg_stack.probe_fa(
                        self.seg_fa_id, self.seg_fa_cfg, a["wav"], text)
                    fa_text = text
            except (Exception, SystemExit) as e:
                self.notice = f"FA realign unavailable ({e}) — text borrowed"
        if fa_text:
            # B.5 probe parity: sentence spans from the segmentation CAPABILITY
            # (same segmenter a --sentence-split re-decomposition would run).
            # Failure degrades to no split preview, never a wedged probe.
            try:
                seg_spans = await self._seg_stack.probe_segment(
                    self.seg_seg_id, self.seg_seg_cfg, fa_text)
            except (Exception, SystemExit) as e:
                self.notice = f"sentence-segmentation unavailable ({e}) — split preview off"
        self.probe_busy = False
        self._probe_ctx = (a, local, fa_text, fa_words, seg_spans)
        self._rebuild_probe_view()
        self._paint()

    def _rebuild_probe_view(self) -> None:
        """Derive the probe readout + walkable predicted rows from the last probe.

        Re-derivable when the s split-preview toggles — no re-probe needed: the
        VAD + FA outputs are unchanged, only the skeleton refinement differs
        (split_predicted runs the pipeline's OWN stage, then the fold re-runs
        over the refined chunks — DEC f1024568 deliverable d)."""
        if self._probe_ctx is None:
            return
        a, local, fa_text, fa_words, seg_spans = self._probe_ctx
        chunks_local = list(local)
        realigned = None
        n_cuts = 0
        if fa_text and fa_words is not None:
            if self.probe_split and seg_spans is not None:
                chunks_local = split_predicted(fa_text, fa_words, local, seg_spans)
                n_cuts = len(chunks_local) - len(local)
            realigned = realign_rows(fa_text, fa_words, chunks_local)
        predicted = [(s + a["start"], e + a["start"]) for s, e in chunks_local]
        committed = [(float(r["start_time"]), float(r["end_time"]))
                     for r in self.seg_rows
                     if r.get("start_time") is not None
                     and r.get("end_time") is not None
                     and a["start"] <= float(r["start_time"]) < a["end"]]
        gaps = []
        for kind, ri, secs in self.seg_display:
            if kind != "gap":
                continue
            gs = float(self.seg_rows[ri].get("start_time") or 0.0) - secs
            if a["start"] <= gs < a["end"]:
                gaps.append((gs, gs + secs))
        self.probe_result = probe_compare(committed, predicted, gaps)
        self.probe_view_rows = predicted_rows(predicted, self.seg_rows, gaps,
                                              realigned=realigned)
        if n_cuts > 0:
            # Mark the rows the split stage minted (spans absent from the raw
            # VAD prediction) — the ✂ is the judgment target.
            originals = {(round(s, 3), round(e, 3)) for s, e in local}
            for row, (ls, le) in zip(self.probe_view_rows, chunks_local):
                row["split"] = (round(ls, 3), round(le, 3)) not in originals
        self.probe_view_display = build_display(self.probe_view_rows,
                                                self.gap_threshold)
        self.probe_view_cursor = 0
        if fa_words is not None:
            split_tag = f" · sentence-split ×{n_cuts}" if self.probe_split else ""
            self.notice = f"v walks the predicted skeleton (FA-realigned{split_tag})"
        elif self.probe_split:
            self.notice = "split preview needs FA — realign unavailable, raw skeleton shown"
        elif not (self.notice or "").startswith("FA realign unavailable"):
            self.notice = "v walks the predicted skeleton (text borrowed)"

    def action_toggle_respine(self) -> None:
        """R: toggle respine — fresh spine under the SAME config (DEC 9241564f).

        The recovery for the e8458f6e verify-collide: a post-upgrade re-run of
        an already-decomposed source refuses (drifted text-slice provenance
        cannot re-emit the same skeleton ids, correctly) — respine mints a
        DISTINCT skeleton hash so the re-run coexists instead of colliding.
        Rides the plan into the hand-off argv as --respine (the split-toggle
        precedent: what the TUI decided is always visible in the printed
        command)."""
        if self.stage != "runs":
            return
        self.respine = not self.respine
        self.notice = ("batch respine ON — confirmed groups run --respine "
                       "(fresh spine, same config; coexists with the old one)"
                       if self.respine else "batch respine off")
        self._paint()

    def action_toggle_event_split(self) -> None:
        """e: toggle the event-carve stage batch-wide (DEC ae450551, the
        hub-launch path's way in — the verdict ad963c57 made model cuts the
        split authority). Arming flips sentence-split OFF (the ratified
        respine seam default; s re-enables it explicitly — stages stay
        composable per 6cc10fb7). Each run's propset resolves latest-by-source
        and paints on its row; E cycles a source's older sets."""
        if self.stage != "runs":
            return
        self.event_split = not self.event_split
        if self.event_split:
            self.sentence_split = False
            self.notice = ("batch event-split ON (sentence-split off — the "
                           "respine seam default; s re-enables) — rows show "
                           "each run's resolved propset")
        else:
            self.notice = "batch event-split off"
        self._paint()

    def action_cycle_propset(self) -> None:
        """E: cycle the focused run's proposal set among its source's sets
        (newest first — older sets are earlier model generations; the t-cycle
        precedent: the default is a convention, the override one keypress)."""
        if self.stage != "runs" or not self.event_split or not self.src_index.runs:
            return
        m = self.src_index.runs[self.cursor]
        if self.propset_pin:
            self.notice = "propset pinned by --event-propset — E-cycle disabled"
            self._paint()
            return
        sets = self._propsets_for(m)
        if len(sets) < 2:
            self.notice = ("no alternate proposal sets for this source"
                           if sets else "no proposal set matches this source")
            self._paint()
            return
        paths = [s["_path"] for s in sets]
        current = self.propset_pick.get(m["_path"]) or paths[0]
        nxt = paths[(paths.index(current) + 1) % len(paths)] \
            if current in paths else paths[0]
        self.propset_pick[m["_path"]] = nxt
        chosen = next(s for s in sets if s["_path"] == nxt)
        self.notice = f"propset → {PropsetIndex.summary(chosen)}"
        self._paint()

    def action_toggle_split(self) -> None:
        """s: toggle sentence-split — the BATCH flag in RUNS (rides the plan
        into the core hand-off as --sentence-split), the probe PREVIEW in the
        form/probe views (re-derives in place; pure post-FA refinement)."""
        if self.stage == "runs":
            self.sentence_split = not self.sentence_split
            self.notice = ("batch sentence-split ON — confirmed groups run "
                           "--sentence-split (parallel spine, new skeleton hash)"
                           if self.sentence_split else "batch sentence-split off")
            self._paint()
            return
        if self.stage not in ("vadform", "probeview"):
            return
        self.probe_split = not self.probe_split
        if self._probe_ctx is not None:
            self._rebuild_probe_view()
        else:
            self.notice = f"sentence-split preview {'ON' if self.probe_split else 'off'}"
        self._paint()

    def _open_field_editor(self, field) -> None:
        """Show the transient Input primed with a field's current value."""
        editor = self.query_one("#editor", Input)
        editor.value = field.render()
        editor.display = True
        editor.focus()
        self.form_editing = True

    def _close_field_editor(self) -> None:
        editor = self.query_one("#editor", Input)
        editor.display = False
        editor.value = ""
        self.set_focus(None)
        self.form_editing = False

    async def on_input_submitted(self, event) -> None:
        """Apply a typed value to the focused form field (enter in the Input)."""
        if not self.form_editing or self.vad_form is None:
            return
        field = self.vad_form.fields[self.form_cursor]
        try:
            field.parse(event.value)
            self.error = None
        except ValueError as e:
            self.error = f"{field.key}: {e}"
        self._close_field_editor()
        self._paint()

    # ---- stage actions (single key vocabulary, stage-dispatched) ----

    def action_move(self, delta: int) -> None:
        if self.stage == "runs":
            if self.src_index.runs:
                self.cursor = max(0, min(self.cursor + delta,
                                         len(self.src_index.runs) - 1))
        elif self.stage == "segments":
            if self.seg_display:
                self.seg_cursor = max(0, min(self.seg_cursor + delta,
                                             len(self.seg_display) - 1))
        elif self.stage == "vadform":
            if self.vad_form is not None and self.vad_form.fields:
                self.form_cursor = max(0, min(self.form_cursor + delta,
                                              len(self.vad_form.fields) - 1))
        elif self.stage == "probeview":
            if self.probe_view_display:
                self.probe_view_cursor = max(0, min(self.probe_view_cursor + delta,
                                                    len(self.probe_view_display) - 1))
        elif self.results_run is None:
            if self.dec_index.runs:
                self.results_cursor = max(0, min(self.results_cursor + delta,
                                                 len(self.dec_index.runs) - 1))
        else:
            srcs = self.dec_index.runs[self.results_run].get("sources") or []
            if srcs:
                self.results_src = max(0, min(self.results_src + delta,
                                              len(srcs) - 1))
        self._paint()

    def on_mouse_scroll_down(self, event) -> None:
        """Wheel = the j/k cursor walk (transcription-TUI drive-2 lesson)."""
        self.action_move(1)

    def on_mouse_scroll_up(self, event) -> None:
        self.action_move(-1)

    async def action_select(self) -> None:
        if self.stage == "runs":
            runs = self.src_index.runs
            if not runs:
                return
            m = runs[self.cursor]
            if not SourceRunIndex.transcribers(m):
                # The core would refuse it at run time ("no transcribers");
                # surfacing that at PICK time keeps the batch hand-off clean.
                self.error = f"{m['run_id']}: manifest lists no transcribers"
                self._paint()
                return
            key = m["_path"]
            if key in self.picked:
                self.picked.remove(key)
            else:
                self.picked.append(key)
            self.error = None
        elif self.stage == "results" and self.results_run is None and self.dec_index.runs:
            self.results_run = self.results_cursor
            self.results_src = 0
        elif self.stage == "results" and self.results_run is not None:
            await self._open_segments()
            return
        elif self.stage == "vadform":
            # Closed sets (enum/bool) cycle in place; open kinds hand off to
            # the transient Input (the value-typing escape hatch).
            if (self.vad_form is not None and self.vad_form.fields
                    and not self.form_editing):
                field = self.vad_form.fields[self.form_cursor]
                if not field.cycle():
                    self._open_field_editor(field)
        self._paint()

    def action_cycle_text_from(self) -> None:
        """Cycle the focused run's authoritative transcriber (t).

        Only multi-transcriber runs have a choice to make — the layer-0 text
        the fine spine commits comes from exactly one of them (--text-from),
        so the pick belongs HERE, per run, not as one flag over the batch;
        grouping folds equal picks back into shared invocations."""
        if self.stage != "runs" or not self.src_index.runs:
            return
        m = self.src_index.runs[self.cursor]
        ts = SourceRunIndex.transcribers(m)
        if len(ts) < 2:
            self.notice = "single-transcriber run — text-from is its sole transcriber"
            self._paint()
            return
        cur = self._resolved_text_from(m)
        nxt = ts[(ts.index(cur) + 1) % len(ts)] if cur in ts else ts[-1]
        self.text_from[m["_path"]] = nxt
        self.notice = None
        self._paint()

    def action_results(self) -> None:
        """Open the decomp-runs view (v): re-reads the manifests so a batch
        that just finished in another terminal shows without a restart.
        In the VAD form, v walks the last probe's predicted skeleton."""
        if self.stage == "vadform":
            if self.probe_view_display:
                self.stage = "probeview"
                self.notice = None
                self._paint()
            else:
                self.error = "no probe yet — p runs the form's config first"
                self._paint()
            return
        if self.stage != "runs":
            return
        self.notice = None
        self.error = None
        self._reload_indexes()
        self.results_run = None
        self.results_cursor = 0
        self.stage = "results"
        self._paint()

    def action_reload(self) -> None:
        if self.stage in ("segments", "probeview"):
            self._play_focused()   # r = play focused (the correction-TUI replay key)
            return
        if self.stage != "runs":
            return
        self._reload_indexes()
        self.notice = (f"reloaded: {len(self.src_index.runs)} transcription / "
                       f"{len(self.dec_index.runs)} decomp run(s)")
        self._paint()

    def action_back(self) -> None:
        if self.stage == "probeview":
            self.stage = "vadform"
        elif self.stage == "vadform":
            if self.form_editing:
                self._close_field_editor()
            self.stage = "segments"
        elif self.stage == "segments":
            self.stage = "results"  # seat stays open — b is a browse, not a teardown
        elif self.stage == "results":
            if self.results_run is not None:
                self.results_run = None  # drilled run -> back to the decomp list
            else:
                self.stage = "runs"
        self._paint()

    async def action_confirm(self) -> None:
        if self.stage != "runs":
            return
        if not self.picked:
            self.error = "pick at least one transcription run first"
            self._paint()
            return
        for (tf, db, ps), paths in self._batches():
            if db is None:
                # Proceeding without a db is the guaranteed-wrong-graph failure
                # the first drive hit (e087d059) — block, don't warn.
                self.error = (f"no graph db recorded for {self._run_id_for(paths[0])} "
                              "— pass --graph-db-path to override")
                self._paint()
                return
            if self.event_split and ps is None:
                # The db guard's twin (DEC ae450551): an event-armed run with
                # no source-matched propset would carve NOTHING silently (or
                # worse, a pin would carve the WRONG source) — block, name it.
                self.error = (f"event-split armed but no proposal set matches "
                              f"{self._run_id_for(paths[0])} — propose over its "
                              "source first (or e disarms)")
                self._paint()
                return
        if self.player is not None:
            self.player.close()
        await self._close_segments()
        self.exit({
            "batches": [{"text_from": tf, "graph_db_path": db,
                         "event_propset": ps, "manifests": list(paths)}
                        for (tf, db, ps), paths in self._batches()],
            "runs_dir": str(self.src_index.runs_dir),
            "manifests_dir": self.manifests_dir,
            "sysmon_capability": self.sysmon_capability,
            "graph_capability": self.graph_capability,
            "graph_db_path": self.graph_db_path,
            "sentence_split": self.sentence_split,
            "respine": self.respine,
            "event_split": self.event_split,
        })

    async def action_quit_app(self) -> None:
        if self.player is not None:
            self.player.close()
        await self._close_segments()
        self.exit(None)
