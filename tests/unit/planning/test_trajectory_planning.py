"""Trajectory planner plugins (planning/trajectory_planning): registry and the
path_pipeline reference plugin against the separate planner + generator pipeline."""
import numpy as np
import pytest

from doraware.core.config import load_config
from doraware.interfaces import Header, Pose, VehicleState, to_ns
from doraware.planning.path_planning.registry import make_planner
from doraware.planning.trajectory_generation.generator import TrajectoryGenerator
from doraware.planning.trajectory_planning.plugins.path_pipeline.planner import PathPipeline
from doraware.planning.trajectory_planning.registry import TRAJECTORY_PLANNERS, make_trajectory_planner

from helpers.scenario_request import request_of

CFG = load_config()


def test_registry_builds_the_configured_plugin():
    assert "path_pipeline" in TRAJECTORY_PLANNERS.names()
    assert isinstance(make_trajectory_planner(CFG), PathPipeline)
    cfg = load_config(overrides=["planning.trajectory_planner.type=nope"])
    with pytest.raises(ValueError, match="unknown trajectory planner 'nope'"):
        make_trajectory_planner(cfg)


def test_path_pipeline_equals_planner_plus_generator():
    req = request_of("B2")
    state = VehicleState(header=Header(stamp=0), pose=Pose.from_xyyaw(req.start.x, req.start.y, req.start.yaw))
    pipe = make_trajectory_planner(CFG)
    res = pipe.plan(req, state, 0)
    assert res.success and res.request_id == req.request_id
    info = pipe.diagnostics()
    assert info["trajectory_ok"] and info["plan_start_time"] == 0.0 and info["plan_duration"] > 10.0

    ref = make_planner(CFG).plan(req)
    gen = TrajectoryGenerator(CFG["planning"]["trajectory_generation"], CFG["timing"], req.vehicle)
    gen.on_planning_result(ref, req.map, plan_start_ns=0, initial_steering=0.0)
    for t in (0.0, 3.0, 12.5):
        a, b = pipe.trajectory_at(state, to_ns(t)), gen.trajectory_at(to_ns(t))
        assert len(a) > 0 and len(a) == len(b)
        for f in ("x", "y", "yaw", "longitudinal_velocity", "gear"):
            assert np.array_equal(getattr(a, f), getattr(b, f)), f


def test_path_pipeline_without_a_plan_stops():
    pipe = make_trajectory_planner(CFG)
    traj = pipe.trajectory_at(VehicleState(), to_ns(1.0))
    assert len(traj) == 0 and traj.valid_until > traj.start_time      # empty = stop
