"""The console-script driver: run the batch-selection TUI, then hand each
text-from group to the HEADLESS decomp core CLI in-process (terminal restored
first). Every group's equivalent cjm-transcript-decomp-core command prints
before execution — TUI-queued batches stay reproducible by copy-paste — and
--plan-only stops at the printout."""

import argparse
import shlex
from typing import Any, Dict, List, Optional

from cjm_transcript_decomp_core.cli import main as core_main

from .app import DecompApp
from .discovery import discover_capability
from .state import load_state, save_state


def build_parser() -> argparse.ArgumentParser:  # Configured CLI parser
    """The TUI driver's argument surface (batch-setup options + core passthrough)."""
    p = argparse.ArgumentParser(
        prog="cjm-transcript-decomp-tui",
        description="Batch-setup TUI for the headless decomp pipeline: pick "
                    "transcription runs into an ordered batch (with per-run "
                    "authoritative-text choices), then hand off to "
                    "cjm-transcript-decomp-core's batch runner — one loaded "
                    "capability stack per text-from group.")
    p.add_argument("--runs-dir", default="runs",
                   help="Run-manifest directory (BOTH cores' cwd-relative default)")
    p.add_argument("--manifests-dir", default=".cjm/manifests",
                   help="Capability manifests directory")
    p.add_argument("--vad-capability", default="cjm-capability-silero-vad",
                   help="VAD capability name (forwarded to the core)")
    p.add_argument("--fa-capability", default="cjm-capability-qwen3-forced-aligner",
                   help="Forced-alignment capability name (forwarded to the core)")
    p.add_argument("--graph-capability", default="cjm-capability-graph-sqlite",
                   help="Graph-storage capability the fine spine extends "
                        "(decomp REQUIRES one — there is no unjournaled decomp)")
    p.add_argument("--graph-db-path", default=None,
                   help="Explicit graph db path (default: last-used, else the "
                        "capability's configured db_path)")
    p.add_argument("--sysmon-capability", default=None,
                   help="monitor capability for GPU attribution (default: last-used, "
                        "else auto-discovered from manifests)")
    p.add_argument("--no-sysmon", action="store_true",
                   help="Explicitly disable the monitor (overrides state + discovery)")
    p.add_argument("--language", default="English",
                   help="Forced-alignment language (forwarded to the core)")
    p.add_argument("--force", action="store_true",
                   help="Bypass capability-side caches (forwarded to the core)")
    p.add_argument("--actor", default=None,
                   help="Forwarded journal attribution (default: cli:<user>)")
    p.add_argument("--plan-only", action="store_true",
                   help="Print each group's equivalent headless command and exit "
                        "WITHOUT running anything")
    return p


def batch_argv(
    batch: Dict[str, Any],       # One confirmed group: {"text_from", "manifests"}
    args: argparse.Namespace,    # The TUI's parsed args (passthrough run options)
    sysmon: Optional[str],       # Resolved monitor capability (None = disabled)
    graph_db_path: Optional[str],  # Resolved graph db override (None = capability default)
) -> List[str]:  # cjm-transcript-decomp-core argv (the reproducibility contract)
    """Render one text-from group as headless decomp-core argv.

    Everything the TUI decided (the ordered member manifests, the group's
    authoritative transcriber) plus everything it merely passes through
    (capabilities, language, force, sysmon, actor) lands in ONE argv — printed
    before execution so any TUI-queued batch can be replayed by hand.
    """
    argv = ["run", *batch["manifests"], "--yes",
            "--manifests-dir", args.manifests_dir,
            "--vad-capability", args.vad_capability,
            "--fa-capability", args.fa_capability,
            "--graph-capability", args.graph_capability,
            "--language", args.language]
    if batch.get("text_from"):
        argv += ["--text-from", batch["text_from"]]
    if graph_db_path:
        argv += ["--graph-db-path", graph_db_path]
    if sysmon:
        argv += ["--sysmon-capability", sysmon]
    if args.force:
        argv += ["--force"]
    if args.actor:
        argv += ["--actor", args.actor]
    return argv


def main() -> int:  # Console-script entry point (cjm-transcript-decomp-tui)
    """Resolve settings (flags > persisted state > manifest discovery), run the
    batch app, persist the confirmed choices, then print + run each group."""
    args = build_parser().parse_args()
    state = load_state(args.manifests_dir)
    sysmon = None if args.no_sysmon else (
        args.sysmon_capability or state.get("sysmon_capability")
        or discover_capability(args.manifests_dir, "get_system_status"))
    graph_db_path = args.graph_db_path or state.get("graph_db_path")
    app = DecompApp(args.manifests_dir, runs_dir=args.runs_dir,
                    sysmon_capability=sysmon,
                    graph_capability=args.graph_capability,
                    graph_db_path=graph_db_path)
    plan = app.run()
    if not plan:
        print("no batch confirmed")
        return 0
    save_state(args.manifests_dir,
               sysmon_capability=plan["sysmon_capability"],
               graph_db_path=plan["graph_db_path"])
    rc = 0
    for i, batch in enumerate(plan["batches"]):
        argv = batch_argv(batch, args, plan["sysmon_capability"],
                          plan["graph_db_path"])
        print(f"batch {i + 1}/{len(plan['batches'])}: "
              + shlex.join(["cjm-transcript-decomp-core"] + argv))
        if args.plan_only:
            continue
        # Sequential by design: groups share the GPU; the core loads one stack
        # per group and its exit code aggregates that group's members.
        rc = max(rc, int(core_main(argv)))
    return rc
