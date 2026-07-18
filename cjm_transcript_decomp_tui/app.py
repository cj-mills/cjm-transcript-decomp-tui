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

from cjm_substrate_tui_kit.repaint import RepaintThrottle
from cjm_substrate_tui_kit.viewport import tail, visible_slice
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Static

from .runs import DecompIndex, group_batches, SourceRunIndex


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
        Binding("q", "quit_app", "quit"),
    ]

    REPAINT_INTERVAL = 1 / 30  # Coalescing window: at most ~30 full repaints/s

    def __init__(self, manifests_dir: str,          # Capability manifests directory
                 *, runs_dir: str = "runs",         # Both cores' cwd-relative manifest dir
                 sysmon_capability: Optional[str] = None,  # Monitor for GPU attribution (CR-7)
                 graph_capability: str = "cjm-capability-graph-sqlite",  # Extension target
                 graph_db_path: Optional[str] = None):     # Caller-wins graph db override
        super().__init__()
        self.manifests_dir = manifests_dir
        self.sysmon_capability = sysmon_capability
        self.graph_capability = graph_capability
        self.graph_db_path = graph_db_path
        self.src_index = SourceRunIndex(runs_dir)
        self.dec_index = DecompIndex(runs_dir)
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
        self.error: Optional[str] = None
        self.notice: Optional[str] = None
        self._throttle = RepaintThrottle(self._paint_now, self.set_timer,
                                         self.REPAINT_INTERVAL)

    def compose(self) -> ComposeResult:
        yield Static(id="main")
        yield Static(id="status")

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
                "results": self._paint_results}[self.stage]()
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
                "runs": "enter/space pick · t text-from · v decomp runs · r reload · n confirm · q quit",
                "results": ("enter open run · j/k walk · b back · q quit"
                            if self.results_run is None
                            else "j/k source · b decomp list · q quit"),
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
                       f"invocation(s)):\n", style="bold")
            for bi, ((tf, db), paths) in enumerate(batches):
                short = (tf or "?").removeprefix("cjm-capability-")
                line = Text()
                line.append(f"   {bi + 1}. text-from {short}", style="green")
                if db is None:
                    line.append("  ⚠ no graph db recorded", style="bold red")
                elif not Path(db).exists():
                    line.append(f"  ⚠ db missing: {tail(db, 24)}", style="bold red")
                else:
                    line.append(f"  db {tail(db, 24)}", style="dim")
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

    def _batches(self) -> List[Tuple[Tuple[Optional[str], Optional[str]], List[str]]]:
        """The hand-off fold: key = (text_from, graph_db_path) — everything
        that applies invocation-wide on the core CLI."""
        picks: List[Tuple[str, Tuple[Optional[str], Optional[str]]]] = []
        for key in self.picked:
            m = self._run_by_path(key)
            if m is not None:
                picks.append((key, (self._resolved_text_from(m),
                                    self._resolved_graph_db(m))))
        return group_batches(picks)

    def _reload_indexes(self) -> None:
        self.src_index.load()
        self.dec_index.load()
        self._decomp_counts = self.dec_index.counts_by_source_manifest()
        # A reload may evict manifests the batch still names; drop those picks
        # rather than hand off paths the core would refuse.
        self.picked = [p for p in self.picked if self._run_by_path(p) is not None]

    # ---- stage actions (single key vocabulary, stage-dispatched) ----

    def action_move(self, delta: int) -> None:
        if self.stage == "runs":
            if self.src_index.runs:
                self.cursor = max(0, min(self.cursor + delta,
                                         len(self.src_index.runs) - 1))
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

    def action_select(self) -> None:
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
        elif self.results_run is None and self.dec_index.runs:
            self.results_run = self.results_cursor
            self.results_src = 0
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
        that just finished in another terminal shows without a restart."""
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
        if self.stage != "runs":
            return
        self._reload_indexes()
        self.notice = (f"reloaded: {len(self.src_index.runs)} transcription / "
                       f"{len(self.dec_index.runs)} decomp run(s)")
        self._paint()

    def action_back(self) -> None:
        if self.stage == "results":
            if self.results_run is not None:
                self.results_run = None  # drilled run -> back to the decomp list
            else:
                self.stage = "runs"
        self._paint()

    def action_confirm(self) -> None:
        if self.stage != "runs":
            return
        if not self.picked:
            self.error = "pick at least one transcription run first"
            self._paint()
            return
        for (tf, db), paths in self._batches():
            if db is None:
                # Proceeding without a db is the guaranteed-wrong-graph failure
                # the first drive hit (e087d059) — block, don't warn.
                self.error = (f"no graph db recorded for {self._run_id_for(paths[0])} "
                              "— pass --graph-db-path to override")
                self._paint()
                return
        self.exit({
            "batches": [{"text_from": tf, "graph_db_path": db,
                         "manifests": list(paths)}
                        for (tf, db), paths in self._batches()],
            "runs_dir": str(self.src_index.runs_dir),
            "manifests_dir": self.manifests_dir,
            "sysmon_capability": self.sysmon_capability,
            "graph_capability": self.graph_capability,
            "graph_db_path": self.graph_db_path,
        })

    def action_quit_app(self) -> None:
        self.exit(None)
