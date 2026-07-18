"""Run-manifest indexes for the decomp-batch TUI (work item 0ff6bf0f): the
transcription core's own runs/*.json read into a selectable batch (decomp
consumes RUN MANIFESTS, not media files), the decomp core's own manifests read
back as coverage chips + results rows, and the pure grouping fold the confirm
hand-off uses. Pure logic, Textual-free (the transcription TUI's results.py
precedent: everything below the paint path tests directly; the app only paints
it). Both cores share the cwd-relative runs/ default, so the manifest FORMAT
tag — the manifest-as-interchange contract (CR-20) — is what separates
transcription runs from decomp runs living in one directory."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_manifests(
    runs_dir: Path,   # Directory holding both cores' run manifests
    format_tag: str,  # Substring the manifest's format tag must carry
) -> List[Dict[str, Any]]:  # Matching manifest dicts, newest first (+ "_path")
    """Read every readable manifest in runs_dir whose format tag matches.

    The RunIndex forgiveness contract (transcription TUI results.py): the runs
    dir is shared ground — unreadable/foreign jsons are skipped, never raised,
    because one corrupt file must not hide the rest. The format-tag filter is
    load-bearing here where RunIndex could skip it: BOTH cores write manifests
    with run_id + sources into the same directory, so shape alone no longer
    separates them."""
    rows: List[Dict[str, Any]] = []
    try:
        files = sorted(runs_dir.glob("*.json"))
    except OSError:
        files = []
    for f in files:
        try:
            m = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if not (isinstance(m, dict) and m.get("run_id")
                and format_tag in str(m.get("format", ""))
                and isinstance(m.get("sources"), list)):
            continue
        m["_path"] = str(f)
        rows.append(m)
    rows.sort(key=lambda m: float(m.get("created_at") or 0.0), reverse=True)
    return rows


class SourceRunIndex:
    """Transcription-core run manifests — the decomp workflow's SOURCES — plus
    the per-run facts the batch stage paints: transcriber lists, segment
    totals, and the authoritative-text default. Reads what the transcription
    runs wrote, no parallel record (results-layer principle)."""

    FORMAT_TAG = "transcription-core"

    def __init__(self, runs_dir: str = "runs"):  # Both cores' cwd-relative default
        self.runs_dir = Path(runs_dir)
        self.runs: List[Dict[str, Any]] = []  # Manifest dicts, newest first (+ "_path")

    def load(self) -> int:  # Number of manifests loaded
        """(Re)read every readable transcription-run manifest, newest first."""
        self.runs = _load_manifests(self.runs_dir, self.FORMAT_TAG)
        return len(self.runs)

    @staticmethod
    def transcribers(m: Dict[str, Any]) -> List[str]:  # Transcriber instance ids, manifest order
        """The run's transcriber ids (config snapshot; pre-0.2.0 single-key
        manifests fold to a one-element list — the pipeline's own tolerance)."""
        cfg = m.get("config") or {}
        ids = cfg.get("transcriber_capabilities") or []
        out = [str(i) for i in ids] if isinstance(ids, list) else []
        if not out and cfg.get("transcriber_capability"):
            out = [str(cfg["transcriber_capability"])]
        return out

    @staticmethod
    def segment_count(m: Dict[str, Any]) -> int:  # Pipeline segments across all sources
        """Total pipeline segments in the run (the batch-size signal a row paints)."""
        return sum(len(s.get("segments") or []) for s in m.get("sources") or [])

    @classmethod
    def default_text_from(cls, m: Dict[str, Any]) -> Optional[str]:  # Pre-picked authoritative transcriber
        """The default --text-from pick: the sole transcriber, else the LAST.

        The transcription TUI's confirmed pair lands [lightweight, accuracy],
        so last = the accuracy model — the natural layer-0 authority. A
        CONVENTION default only: the row paints it and t cycles it, so a
        hand-built manifest with a different order is one keypress away."""
        t = cls.transcribers(m)
        return t[-1] if t else None


class DecompIndex:
    """Decomp-core run manifests read back: coverage chips for the batch stage
    (which transcription runs already have a decomp run) and the results
    view's rows. v0 stops at the manifest's own facts — segment TEXTS live in
    the graph, not the manifest, so deeper inspection views wait for real
    decomp-session demand (the work item's deliberate unshaping)."""

    FORMAT_TAG = "transcript-decomp-core"

    def __init__(self, runs_dir: str = "runs"):  # Both cores' cwd-relative default
        self.runs_dir = Path(runs_dir)
        self.runs: List[Dict[str, Any]] = []  # Manifest dicts, newest first (+ "_path")

    def load(self) -> int:  # Number of manifests loaded
        """(Re)read every readable decomp-run manifest, newest first."""
        self.runs = _load_manifests(self.runs_dir, self.FORMAT_TAG)
        return len(self.runs)

    def counts_by_source_manifest(self) -> Dict[str, int]:
        """resolved transcription-manifest path -> decomp runs that consumed it.

        Path-keyed and hash-free (the prior-run-chip pattern: browse-time chips
        must stay cheap); the decomp core records source_manifest RESOLVED, so
        resolving the browse key on lookup matches regardless of how the runs
        dir was spelled."""
        counts: Dict[str, int] = {}
        for m in self.runs:
            p = m.get("source_manifest")
            if p:
                key = str(Path(p).resolve())
                counts[key] = counts.get(key, 0) + 1
        return counts


def group_by_text_from(
    picks: List[Tuple[str, Optional[str]]],  # Ordered (manifest_path, resolved text_from) selection
) -> List[Tuple[Optional[str], List[str]]]:  # (text_from, member paths) per hand-off invocation
    """Fold an ordered batch selection into headless hand-off groups.

    --text-from applies invocation-wide, so one core invocation per DISTINCT
    text_from is the finest split that keeps the whole ergonomic win: every
    manifest in a group rides the SAME loaded capability stack. Sole-transcriber
    runs resolve to their sole transcriber (always valid against the core's
    membership check), so they merge into a pair-run's group whenever the
    authority model matches. First-seen order of groups and members preserves
    the operator's queueing order."""
    order: List[Optional[str]] = []
    groups: Dict[Optional[str], List[str]] = {}
    for path, tf in picks:
        if tf not in groups:
            groups[tf] = []
            order.append(tf)
        groups[tf].append(path)
    return [(tf, groups[tf]) for tf in order]
