"""Collision revalidation (exact footprint rectangle vs occupied cells)."""
import math

import numpy as np

from doraware.core.config import load_config
from doraware.interfaces import OccupancyGrid, PlanningResult, PlanningStatus
from doraware.planning.trajectory_generation.generator import TrajectoryGenerator, TrajectoryStatus
from doraware.planning.trajectory_generation.revalidation import FootprintChecker, revalidate, sweep_bound

from helpers.path_fixtures import VEHICLE, arc_path, free_grid

CFG = load_config()
V = VEHICLE
FRONT = V.wheel_base + V.front_overhang
HALF_W = V.wheel_tread / 2 + V.left_overhang


def grid_with(cells, res=0.1, w=200, h=100) -> OccupancyGrid:
    g = free_grid(w * res, h * res, res)
    d = g.as_2d().copy()
    for ix, iy in cells:
        d[iy, ix] = 1
    g.data = d.ravel()
    return g


def test_exact_rectangle_vs_cell():
    # vehicle at (5, 5, 0): front face at x = 5 + FRONT = 8.69, left side at y = 5 + 0.95
    x0, y0 = 5.0, 5.0
    front = x0 + FRONT
    ix_front = int(math.floor(front / 0.1))
    hit = FootprintChecker(grid_with([(ix_front, 50)]), V)       # the cell containing the front face
    assert hit.collides(x0, y0, 0.0)
    miss = FootprintChecker(grid_with([(ix_front + 1, 50)]), V)  # next cell: 1 cm+ in front
    assert not miss.collides(x0, y0, 0.0)
    # rotated: a cell beyond the axis-aligned box of the corner but outside the rotated rectangle
    g = grid_with([(int((x0 + FRONT) / 0.1), int((y0 + HALF_W) / 0.1))])   # near the front-left corner
    c = FootprintChecker(g, V)
    assert c.collides(x0, y0, 0.0)
    assert not c.collides(x0, y0, math.radians(-10.0))       # corner rotated away from the cell
    # margin grows the rectangle
    assert FootprintChecker(grid_with([(ix_front + 1, 50)]), V, margin=0.15).collides(x0, y0, 0.0)


def test_outside_map_is_collision():
    c = FootprintChecker(free_grid(20.0, 10.0), V)
    assert not c.collides(5.0, 5.0, 0.0)
    assert c.collides(0.5, 5.0, 0.0)          # rear overhang beyond x = 0
    assert c.collides(5.0, 9.5, 0.3)


def test_revalidation_catches_obstacle_on_path():
    path = arc_path([(+1, 0.0, 6.0), (-1, 0.3, 3.0)], start=(5.0, 10.0, 0.0))
    gen = TrajectoryGenerator(CFG["planning"]["trajectory_generation"], CFG["timing"], V)
    free = free_grid(40.0, 20.0)
    assert gen.on_planning_result(PlanningResult(status=PlanningStatus.SUCCESS, path=path), free) == \
        TrajectoryStatus.OK
    blocked = free_grid(40.0, 20.0)
    d = blocked.as_2d().copy()
    d[100, 130] = 1                           # one cell at (13.05, 10.05): on the straight part
    blocked.data = d.ravel()
    st = gen.on_planning_result(PlanningResult(status=PlanningStatus.SUCCESS, path=path), blocked)
    assert st == TrajectoryStatus.REVALIDATION_FAILED and "collides" in gen.message
    assert len(gen.trajectory_at(0)) == 0     # stop
    assert gen.planning_status == PlanningStatus.SUCCESS


def test_sweep_bound_applied():
    path = arc_path([(+1, 0.5, 3.0)], start=(10.0, 10.0, 0.0))
    poses = np.column_stack([path.x, path.y, path.yaw])
    k = math.tan(0.5) / V.wheel_base
    res = revalidate([poses], free_grid(), V, 0.05, k)
    assert res.ok and abs(res.check_margin - (0.05 + sweep_bound(V, 0.05, 0.025, k))) < 1e-6
    assert 0.0 < res.check_margin - 0.05 < 0.05
