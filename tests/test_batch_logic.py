"""Tests for the decomp-batch TUI's pure logic: manifest-format filtering, the
text-from default + grouping fold, coverage counts, and the hand-off argv
(everything below the paint path; the paint layer's verification is the
tests_manual pilot probe, per the TUI craft register)."""

import json
from pathlib import Path
from cjm_transcript_decomp_tui.cli import batch_argv, build_parser
from cjm_transcript_decomp_tui.runs import (DecompIndex, SourceRunIndex,
                                            group_by_text_from)


def _src_manifest(run_id, created, transcribers):
    # Minimal but format-faithful transcription-core manifest (0.2.0 shape).
    return {"format": "cjm-transcription-core/run-manifest", "version": "0.2.0",
            "run_id": run_id, "created_at": created,
            "config": {"transcriber_capabilities": transcribers},
            "sources": [{"source_path": f"/tmp/{run_id}.mp3",
                         "segments": [{"index": 0}, {"index": 1}]}]}


def _dec_manifest(run_id, created, source_manifest):
    # Minimal but format-faithful decomp-core manifest (0.2.1 shape).
    return {"format": "cjm-transcript-decomp-core/run-manifest", "version": "0.2.1",
            "run_id": run_id, "created_at": created,
            "config": {"text_from": None},
            "source_manifest": source_manifest,
            "sources": [{"source_node_id": "abc123", "source_path": "/tmp/x.mp3",
                         "title": "x", "segment_count": 5, "segment_ids": []}]}


def test_format_filtering_and_order(tmp_path):
    # Both cores' manifests + junk share one runs dir; the format tag separates.
    (tmp_path / "a.json").write_text(json.dumps(_src_manifest("run_a", 100.0, ["w"])))
    (tmp_path / "b.json").write_text(json.dumps(_src_manifest("run_b", 200.0, ["w"])))
    (tmp_path / "d.json").write_text(
        json.dumps(_dec_manifest("decomp_x", 300.0, str(tmp_path / "a.json"))))
    (tmp_path / "junk.json").write_text("not json")
    (tmp_path / "foreign.json").write_text(json.dumps({"run_id": "x", "sources": []}))
    src = SourceRunIndex(str(tmp_path))
    assert src.load() == 2
    assert [m["run_id"] for m in src.runs] == ["run_b", "run_a"]  # newest first
    assert SourceRunIndex.segment_count(src.runs[0]) == 2
    dec = DecompIndex(str(tmp_path))
    assert dec.load() == 1
    assert dec.runs[0]["run_id"] == "decomp_x"


def test_decomp_coverage_counts(tmp_path):
    a = tmp_path / "a.json"
    a.write_text(json.dumps(_src_manifest("run_a", 1.0, ["w"])))
    (tmp_path / "d1.json").write_text(json.dumps(_dec_manifest("decomp_1", 2.0, str(a))))
    (tmp_path / "d2.json").write_text(json.dumps(_dec_manifest("decomp_2", 3.0, str(a))))
    dec = DecompIndex(str(tmp_path))
    dec.load()
    assert dec.counts_by_source_manifest() == {str(a.resolve()): 2}


def test_transcribers_and_default_text_from():
    multi = _src_manifest("m", 1.0, ["cap-light", "cap-acc"])
    assert SourceRunIndex.transcribers(multi) == ["cap-light", "cap-acc"]
    # The confirmed pair lands [lightweight, accuracy] -> last = accuracy slot.
    assert SourceRunIndex.default_text_from(multi) == "cap-acc"
    legacy = {"config": {"transcriber_capability": "cap-solo"}}
    assert SourceRunIndex.transcribers(legacy) == ["cap-solo"]
    assert SourceRunIndex.default_text_from(legacy) == "cap-solo"
    assert SourceRunIndex.default_text_from({"config": {}}) is None


def test_group_by_text_from_preserves_order():
    groups = group_by_text_from([("m1", "acc"), ("m2", "solo"), ("m3", "acc")])
    assert groups == [("acc", ["m1", "m3"]), ("solo", ["m2"])]
    assert group_by_text_from([]) == []


def test_batch_argv_renders_the_handoff():
    args = build_parser().parse_args(["--force", "--actor", "tui:test"])
    argv = batch_argv({"text_from": "cap-acc", "manifests": ["a.json", "b.json"]},
                      args, "cjm-capability-monitor-nvidia", "/tmp/g.db")
    assert argv[:3] == ["run", "a.json", "b.json"]
    assert "--yes" in argv
    assert argv[argv.index("--text-from") + 1] == "cap-acc"
    assert argv[argv.index("--graph-db-path") + 1] == "/tmp/g.db"
    assert argv[argv.index("--sysmon-capability") + 1] == "cjm-capability-monitor-nvidia"
    assert "--force" in argv and "--actor" in argv
    # Optionals stay OUT when unset (a copy-pasteable minimal command).
    bare = batch_argv({"text_from": None, "manifests": ["a.json"]},
                      build_parser().parse_args([]), None, None)
    for flag in ("--text-from", "--graph-db-path", "--sysmon-capability",
                 "--force", "--actor"):
        assert flag not in bare


def test_parser_defaults():
    args = build_parser().parse_args([])
    assert args.runs_dir == "runs"
    assert args.fa_capability == "cjm-capability-qwen3-forced-aligner"
    assert args.graph_capability == "cjm-capability-graph-sqlite"
    assert not args.plan_only and not args.no_sysmon and not args.force
