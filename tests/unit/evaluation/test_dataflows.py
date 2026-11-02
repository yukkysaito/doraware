"""Dataflow files: composition (extends / remove / rewire), wiring checks, recorder."""
from pathlib import Path

import pytest
import yaml

from doraware.core.config import REPO_ROOT
from doraware.evaluation.suite.runner import check_dataflow, compose_dataflow, render_dataflow

FLOWS = sorted((REPO_ROOT / "dataflows").glob("*.yml"))


@pytest.mark.parametrize("path", FLOWS, ids=lambda p: p.name)
def test_every_dataflow_composes_and_is_wired(path):
    flow = yaml.safe_load(render_dataflow(True, path))
    check_dataflow(flow)
    rec = [n for n in flow["nodes"] if n["id"] == "recorder"][0]
    outs = sum(len(n.get("outputs") or []) for n in flow["nodes"] if n["id"] != "recorder")
    assert len(rec["inputs"]) == outs                       # the recorder sees every output


def test_trajectory_planner_variant_is_a_rewired_closed_loop():
    flow = compose_dataflow(REPO_ROOT / "dataflows" / "trajectory_planner.yml")
    ids = [n["id"] for n in flow["nodes"]]
    assert "trajectory_planner" in ids and "planner" not in ids and "trajectory_generator" not in ids
    ctrl = [n for n in flow["nodes"] if n["id"] == "controller"][0]
    assert ctrl["inputs"]["trajectory"]["source"] == "trajectory_planner/trajectory"


def test_wiring_errors_are_reported(tmp_path):
    bad = {"nodes": [{"id": "a", "outputs": ["x"]}, {"id": "b", "inputs": {"y": {"source": "a/z"}}}]}
    with pytest.raises(ValueError, match="b/y reads 'a/z'"):
        check_dataflow(bad)
    with pytest.raises(ValueError, match="duplicate node ids"):
        check_dataflow({"nodes": [{"id": "a"}, {"id": "a"}]})
    base = tmp_path / "base.yml"
    base.write_text(yaml.safe_dump({"nodes": [{"id": "a", "outputs": ["x"]},
                                              {"id": "b", "inputs": {"y": "a/x"}, "outputs": []}]}))
    variant = tmp_path / "v.yml"
    variant.write_text(yaml.safe_dump({"extends": "base.yml", "remove": ["a"], "rewire": {"a": "c"},
                                       "nodes": [{"id": "c", "outputs": ["x"]}]}))
    flow = compose_dataflow(Path(variant))
    assert [n["id"] for n in flow["nodes"]] == ["b", "c"] and flow["nodes"][0]["inputs"]["y"] == "c/x"
    check_dataflow(flow)
