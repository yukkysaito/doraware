"""State Lattice planner adapter: interfaces <-> planner, status mapping, GeometricPath."""
import json
import math
from dataclasses import fields

import numpy as np
import pytest

from doraware.interfaces import GeometricPath, Gear, OccupancyGrid, PlanningStatus
from doraware.planning.path_planning.plugins.state_lattice.adapter import (StateLatticeAdapter, parse_planner_config, to_planner_grid)

from helpers.scenario_request import request_of


@pytest.fixture(scope="module")
def adapter():
    return StateLatticeAdapter()


@pytest.fixture(scope="module")
def b2(adapter):
    return adapter.plan(request_of("B2"))


def gear_changes(path: GeometricPath) -> int:
    g = np.asarray(path.gear)[1:]
    return int(np.count_nonzero(g[1:] != g[:-1]))


def test_b2_success_one_gear_change(b2):
    assert b2.status == PlanningStatus.SUCCESS and b2.success
    p = b2.path
    assert len(p) > 100 and gear_changes(p) == 1
    assert set(np.unique(p.gear)) == {int(Gear.DRIVE), int(Gear.REVERSE)}
    assert np.all(np.isfinite(p.curvature)) and np.max(np.abs(p.curvature)) <= math.tan(0.55) / 2.79 + 1e-9
    # starts at the start, ends at the goal (within the planner's tolerance)
    req = request_of("B2")
    assert (p.x[0], p.y[0]) == pytest.approx((req.start.x, req.start.y), abs=1e-6)
    assert math.hypot(p.x[-1] - req.goal.x, p.y[-1] - req.goal.y) <= 0.1
    # statistics only in details
    d = json.loads(b2.details)
    assert d["direction_changes"] == 1 and d["raw_status"] == "success" and d["expanded_nodes"] > 0
    assert b2.planning_time > 0


def test_geometric_path_is_planner_agnostic(b2):
    assert [f.name for f in fields(GeometricPath)] == ["x", "y", "yaw", "curvature", "gear"]
    for f in fields(GeometricPath):
        assert isinstance(getattr(b2.path, f.name), np.ndarray)


def test_yaw_wrapped(b2):
    assert np.all((b2.path.yaw >= -math.pi) & (b2.path.yaw < math.pi))


def test_g4_start_in_collision(adapter):
    assert adapter.plan(request_of("G4")).status == PlanningStatus.START_IN_COLLISION


def test_g6_invalid_vehicle(adapter):
    res = adapter.plan(request_of("G6"))
    assert res.status == PlanningStatus.INVALID_INPUT and "wheel_base" in res.message
    assert len(res.path) == 0


def test_g9_unnormalised_yaw(adapter, b2):
    req = request_of("G9")
    assert req.goal.yaw > 2 * math.pi
    res = adapter.plan(req)
    assert res.status == PlanningStatus.SUCCESS
    assert np.all((res.path.yaw >= -math.pi) & (res.path.yaw < math.pi))
    assert -math.pi <= res.goal.yaw < math.pi
    assert gear_changes(res.path) == gear_changes(b2.path)


def test_tuple_override_sweep_scenario(adapter):
    """J-len-s: planner_overrides with a tuple (length_classes) round-trips through JSON."""
    req = request_of("J-len-s")
    assert json.loads(req.planner_config) == {"length_classes": ["short"]}
    res = adapter.plan(req)
    assert res.status == PlanningStatus.SUCCESS


def test_planner_config_parsing():
    name, cfg = parse_planner_config('{"length_classes": ["short", "medium"], "planner": "forward", '
                                     '"max_planning_time": Infinity}')
    assert name == "forward" and cfg.length_classes == ("short", "medium") and math.isinf(cfg.max_planning_time)
    hash(cfg)                                   # hashable -> usable as a cache key
    assert parse_planner_config("")[0] == "bidirectional"


@pytest.mark.parametrize("cfg", ['{"no_such_field": 1}', '{"planner": "rrt"}', "not json", "[1, 2]",
                                 '{"heuristic_weight": 0.5}'])
def test_invalid_planner_config(adapter, cfg):
    req = request_of("G8")
    req.planner_config = cfg
    res = adapter.plan(req)
    assert res.status == PlanningStatus.INVALID_INPUT and res.message


def test_forward_planner_selectable(adapter):
    res = adapter.plan(request_of("B2", planner="forward"))
    assert res.status == PlanningStatus.SUCCESS and json.loads(res.details)["planner"] == "forward"


def test_grid_conversion_row_major():
    g = OccupancyGrid(resolution=0.2, width=5, height=3, origin_x=1.0, origin_y=-2.0,
                      data=np.zeros(15, dtype=np.uint8))
    g.data[1 * 5 + 3] = 1                      # (ix=3, iy=1)
    pg = to_planner_grid(g)
    assert pg.data.shape == (3, 5) and pg.data[1, 3] == 1 and pg.data.sum() == 1
    assert pg.origin == (1.0, -2.0) and pg.resolution == 0.2
    assert bool(pg.is_occupied(1.0 + 3.5 * 0.2, -2.0 + 1.5 * 0.2))


def test_bad_grid_is_invalid_input(adapter):
    req = request_of("G8")
    req.map.width += 1
    assert adapter.plan(req).status == PlanningStatus.INVALID_INPUT


def test_start_equals_goal(adapter):
    res = adapter.plan(request_of("G8"))
    assert res.status == PlanningStatus.SUCCESS and len(res.path) >= 1
