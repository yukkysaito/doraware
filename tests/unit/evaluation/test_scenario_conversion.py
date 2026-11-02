import numpy as np
import pytest

from doraware.core.occupancy import boxes_to_grid, grid_to_boxes
from doraware.evaluation.scenarios.parking import (FEASIBLE, INFEASIBLE, INVALID_INPUT, PARAMETER, RANDOMIZED,
                                                   build_bundle, fixed_scenario_ids)


def test_grid_to_boxes_exact_random():
    rng = np.random.default_rng(0)
    g = (rng.random((40, 60)) > 0.7).astype(np.uint8)
    boxes = grid_to_boxes(g, 0.1, (1.0, -2.0))
    assert (boxes_to_grid(boxes, g.shape, 0.1, (1.0, -2.0)) == g).all()


@pytest.mark.parametrize("sid", ["B2", "C2", "E10", "N-003"])
def test_scene_matches_planner_grid(sid):
    b = build_bundle(sid)
    s = b.scene
    boxes = np.stack([s.obstacles_cx - s.obstacles_length / 2, s.obstacles_cy - s.obstacles_width / 2,
                      s.obstacles_cx + s.obstacles_length / 2, s.obstacles_cy + s.obstacles_width / 2], 1)
    m = b.request.map
    assert (boxes_to_grid(boxes, (m.height, m.width), m.resolution, (m.origin_x, m.origin_y)) == m.as_2d()).all()
    assert (s.initial_pose.x, s.initial_pose.y) == (b.request.start.x, b.request.start.y)


def test_classification_covers_all_fixed():
    classes = {sid: build_bundle(sid).expected.scenario_class for sid in fixed_scenario_ids()}
    assert classes["B2"] == FEASIBLE and classes["G1"] == INFEASIBLE and classes["C3"] == INFEASIBLE
    assert classes["G4"] == INVALID_INPUT and classes["G6"] == INVALID_INPUT and classes["G7"] == INVALID_INPUT
    assert classes["I-xy0.05"] == PARAMETER and classes["G8"] == FEASIBLE
    assert build_bundle("N-000").expected.scenario_class == RANDOMIZED
    assert len(classes) == 107


def test_invalid_vehicle_reaches_planner_unchanged():
    b = build_bundle("G6")
    assert b.request.vehicle.wheel_base == 0.0 and b.scene.vehicle.wheel_base > 0 and not b.planner_vehicle_valid
