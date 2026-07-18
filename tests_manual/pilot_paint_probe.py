"""Headless paint-path pilot for the decomp-batch TUI (67335f7d: pilot probe,
not pytest, is the verification layer for TUI paint strings — pytest can't see
a MarkupError or a style bleed). Drives the REAL app over synthetic run
manifests with Textual's run_test pilot and reads the painted Statics; the
hand-off itself is the headless core's own surface, so the pilot stops at the
confirmed plan.

    python tests_manual/pilot_paint_probe.py
"""

import asyncio
import json
import tempfile
from pathlib import Path
from textual.widgets import Static
from cjm_transcript_decomp_tui.app import DecompApp


def write_corpus(runs_dir: Path) -> dict:
    """Stage a synthetic runs dir: a multi-transcriber run (newest), a
    single-transcriber run, and one decomp manifest consuming the multi run
    (the coverage-chip case). Returns the paths keyed for assertions."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    # A REAL on-disk db file: the recorded-db provenance (e087d059) paints
    # warn-free and rides the confirmed plan only when the path exists.
    db = runs_dir / "context_graph.db"
    db.write_bytes(b"")
    caps = {"cjm-capability-graph-sqlite": {"db_path": str(db)}}
    multi = runs_dir / "run_multi.json"
    multi.write_text(json.dumps({
        "format": "cjm-transcription-core/run-manifest", "version": "0.2.0",
        "run_id": "run_multi", "created_at": 200.0,
        "config": {"transcriber_capabilities": ["cjm-capability-solo",
                                                "cjm-capability-acc"]},
        "capabilities": caps,
        "sources": [{"source_path": "/tmp/a.mp3",
                     "segments": [{"index": 0}, {"index": 1}]}]}))
    single = runs_dir / "run_single.json"
    single.write_text(json.dumps({
        "format": "cjm-transcription-core/run-manifest", "version": "0.2.0",
        "run_id": "run_single", "created_at": 100.0,
        "config": {"transcriber_capabilities": ["cjm-capability-solo"]},
        "capabilities": caps,
        "sources": [{"source_path": "/tmp/b.mp3", "segments": [{"index": 0}]}]}))
    (runs_dir / "decomp_x.json").write_text(json.dumps({
        "format": "cjm-transcript-decomp-core/run-manifest", "version": "0.2.1",
        "run_id": "decomp_x", "created_at": 300.0,
        "config": {"text_from": "cjm-capability-acc"},
        "source_manifest": str(multi),
        "sources": [{"source_node_id": "abc12345ffff", "source_path": "/tmp/a.mp3",
                     "title": "a", "segment_count": 5, "segment_ids": []}]}))
    return {"multi": str(multi), "single": str(single), "db": str(db)}


async def drive_batch(runs_dir: Path, paths: dict) -> None:
    """Walk pick -> t-cycle -> grouping fold -> results drill -> confirm,
    asserting stage + painted strings at every step."""
    app = DecompApp(str(runs_dir / "no-manifests"), runs_dir=str(runs_dir))
    async with app.run_test() as pilot:
        def paint() -> str:
            # Repaints coalesce — flush before reading so the assertion never
            # races the trailing tick (transcription-pilot precedent).
            app._paint_now()
            return str(app.query_one("#main", Static).render())

        assert app.stage == "runs"
        body = paint()
        assert "Transcription runs (2)" in body, body[:400]
        assert "·decomp×1" in body, body               # coverage chip (multi run)
        assert "tf=acc" in body, body                  # default = accuracy slot
        chip = str(app.query_one("#status", Static).render())
        assert "graph→cjm-capability-graph-sqlite" in chip, chip

        # Focused-run detail (614dd647): source names + the recorded graph db.
        assert "· a" in body and "graph db:" in body, body
        assert "⚠" not in body, body                   # provenance complete: no warns

        await pilot.press("enter")                     # pick run_multi (newest first)
        assert app.picked == [paths["multi"]], app.picked
        body = paint()
        assert "[1]" in body and "1 invocation(s)" in body, body
        assert "text-from acc" in body and "run_multi" in body, body
        assert "db " in body, body                     # the group's db paints (e087d059)

        await pilot.press("j")
        await pilot.press("enter")                     # pick run_single too
        body = paint()
        assert "2 invocation(s)" in body, body         # acc-group + solo-group

        await pilot.press("k")
        await pilot.press("t")                         # cycle multi: acc -> solo
        body = paint()
        assert "tf=solo" in body, body
        assert "1 invocation(s)" in body, body         # groups folded together

        await pilot.press("j")
        await pilot.press("t")                         # t on a single-transcriber run
        chip = str(app.query_one("#status", Static).render())
        # The one-line dock ellipsizes — assert on the notice HEAD, which
        # survives truncation at any sane width.
        assert "single-transcriber run" in chip, chip

        await pilot.press("v")                         # -> decomp results
        assert app.stage == "results", app.stage
        body = paint()
        assert "Decomp runs (1)" in body and "decomp_x" in body, body[:400]
        assert "5 seg" in body and "tf=acc" in body, body
        await pilot.press("enter")                     # drill
        body = paint()
        assert "5 fine segment(s)" in body, body[:400]
        assert "Source abc12345" in body, body
        await pilot.press("b")                         # drilled -> list
        assert app.results_run is None
        await pilot.press("b")                         # list -> runs
        assert app.stage == "runs", app.stage

        await pilot.press("n")                         # confirm the batch
    plan = app.return_value
    assert plan is not None, "confirm returned no plan"
    assert plan["batches"] == [{"text_from": "cjm-capability-solo",
                                "graph_db_path": paths["db"],
                                "manifests": [paths["multi"], paths["single"]]}], plan
    print("pilot OK: pick order, t-cycle + grouping fold, coverage chip, "
          "results drill, confirmed plan")


async def drive_windowing(runs_dir: Path) -> None:
    """A 60-manifest corpus must window around the cursor (one-line rows, no
    horizontal overflow) and quit without a plan."""
    for i in range(60):
        (runs_dir / f"r{i:03d}.json").write_text(json.dumps({
            "format": "cjm-transcription-core/run-manifest", "version": "0.2.0",
            "run_id": f"run_{i:03d}" + "x" * 80,  # long ids must ellipsize, not wrap
            "created_at": float(i),
            "config": {"transcriber_capabilities": ["cjm-capability-solo"]},
            "sources": [{"source_path": f"/tmp/{i}.mp3", "segments": []}]}))
    app = DecompApp(str(runs_dir / "no-manifests"), runs_dir=str(runs_dir))
    async with app.run_test() as pilot:
        def paint() -> str:
            app._paint_now()
            return str(app.query_one("#main", Static).render())

        body = paint()
        assert "below" in body, body[:400]      # tail hidden behind the indicator
        assert "run_059" in body                # newest first, near-cursor rows painted
        assert "run_000" not in body            # far tail is NOT painted
        assert "…" in body                      # long ids ellipsized
        assert max(len(ln) for ln in body.splitlines()) <= 80, \
            max(body.splitlines(), key=len)
        for _ in range(59):                     # held-j to the end (coalesced)
            app.action_move(1)
        assert app.cursor == 59
        body = paint()
        assert "run_000" in body and "above" in body, body[:400]
        app.on_mouse_scroll_up(None)            # wheel = the j/k cursor walk
        assert app.cursor == 58
        await pilot.press("q")
    assert app.return_value is None             # quit without a confirmed plan
    print("pilot OK: viewport windowing, one-line rows, wheel scroll, quit-no-plan")


def main() -> None:
    """Stage throwaway corpora, then drive the app (no project state touched:
    the app itself never writes the sidecar — the driver does, post-confirm)."""
    with tempfile.TemporaryDirectory() as td:
        runs = Path(td) / "runs"
        paths = write_corpus(runs)
        asyncio.run(drive_batch(runs, paths))
    with tempfile.TemporaryDirectory() as td:
        runs = Path(td) / "runs"
        runs.mkdir()
        asyncio.run(drive_windowing(runs))


# Entry-point dispatch — LAST region on purpose (regions append in order, and
# the __main__ call must follow every driver it names; ba810a2a).
if __name__ == "__main__":
    main()
