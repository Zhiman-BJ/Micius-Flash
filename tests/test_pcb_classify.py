import hashlib
import json

import pytest

from micius_eval.pcb_agent.verify import SCHEMA, classify, inventory


def _write(path, value):
    path.write_text(json.dumps(value))


def _setup(tmp_path, structure="PASS"):
    snapshot = tmp_path / "snapshot"
    (snapshot / "output").mkdir(parents=True)
    (snapshot / "output" / "board.kicad_sch").write_text("(kicad_sch)")
    work = tmp_path / "work"
    work.mkdir()
    (work / "erc.json").write_text("{}")
    erc_sha = hashlib.sha256(b"{}").hexdigest()
    before = inventory(snapshot)
    binding = {"case_id": "Q1"}
    reviewer = {"model": "m", "reasoning_effort": "high"}
    axes = {"answer": {"status": "PASS", "reason": "r"}, "structure": {"status": structure, "reason": "r"},
            "completion": {"status": "FAIL", "reason": "r"}}
    common = {"schema": SCHEMA, "binding": binding, "reviewer": reviewer}
    _write(work / "review.json", {**common, "axes": axes, "final_candidate_files": [
        {"path": "/retired/output/board.kicad_sch", "sha256": before["output/board.kicad_sch"]["sha256"]}]})
    _write(work / "review-evidence.json", {**common, "executions": [], "checks": [
        {"axis": "answer", "kind": "delivery", "status": "PASS", "evidence": "files exist", "files": []},
        {"axis": "structure", "kind": "native_static", "status": "PASS", "evidence": "ERC 0/0",
         "files": [{"path": "erc.json", "sha256": erc_sha}]}]})
    _write(work / "review-completion.json", {**common, "axes": axes, "required_checks_complete": True})
    return work, binding, before, snapshot, reviewer


def test_classify_accepts_bound_evidence(tmp_path):
    work, binding, before, snapshot, reviewer = _setup(tmp_path)
    out = classify(work, binding, before, snapshot, reviewer)
    assert out["task_success"] is False and out["axes"]["structure"]["status"] == "PASS"


def test_classify_rejects_tampered_snapshot(tmp_path):
    work, binding, before, snapshot, reviewer = _setup(tmp_path)
    (snapshot / "output" / "board.kicad_sch").write_text("changed")
    with pytest.raises(ValueError):
        classify(work, binding, before, snapshot, reviewer)


def test_classify_rejects_wrong_reviewer(tmp_path):
    work, binding, before, snapshot, _ = _setup(tmp_path)
    with pytest.raises(ValueError):
        classify(work, binding, before, snapshot, {"model": "other", "reasoning_effort": "high"})
