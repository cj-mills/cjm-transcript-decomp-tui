"""The console-script driver: run the batch-selection TUI, then hand each
text-from group to the HEADLESS decomp core CLI in-process (terminal restored
first). The whole launch surface — build_parser, resolve_split_flags,
resolve_settings, hand_off — lives in cjm_transcript_decomp_core.launch
(spine absorption 12f342f1) and is imported here, so this shell and the Qt
shell cannot drift on the reproducibility contract; only the app in the
middle differs."""

from cjm_transcript_decomp_core.launch import (build_parser, hand_off, resolve_settings,
                                               resolve_split_flags)

from .app import DecompApp


def main() -> int:  # Console-script entry point (cjm-transcript-decomp-tui)
    """Resolve the shared setup surface, run the batch app, hand off — the
    build_parser -> resolve_split_flags -> resolve_settings -> app -> hand_off
    ladder both shells share (only the app in the middle differs)."""
    args = resolve_split_flags(build_parser().parse_args())
    s = resolve_settings(args)
    app = DecompApp(s["manifests_dir"], runs_dir=s["runs_dir"],
                    sysmon_capability=s["sysmon_capability"],
                    graph_capability=args.graph_capability,
                    graph_db_path=s["graph_db_path"],
                    gap_threshold=args.gap_threshold,
                    sentence_split=args.sentence_split,
                    respine=args.respine,
                    proposals_dir=s["proposals_dir"],
                    event_split=args.event_split,
                    propset_pin=args.event_propset,
                    training_runs_dir=s["training_runs_dir"],
                    training_run_pin=s["training_run_pin"])
    return hand_off(app.run(), args)
