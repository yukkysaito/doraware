"""Collision revalidation of the trajectory footprints against the OccupancyGrid.

Exact test of the oriented footprint rectangle (base_link frame
[-rear, wheel_base + front] x [-(tread/2 + right), tread/2 + left], inflated by
``margin``) against every occupied cell (axis-aligned square), by the
separating axis theorem on the 4 candidate axes (map x, map y, vehicle
x, vehicle y). Touching is not a collision. Any part of the footprint outside
the grid is a collision (unknown space is not drivable).

Continuous motion: the reference poses are the path points, linearly
interpolated. Between two poses p0, p1 (distance ds, yaw change |d yaw| <=
k_max ds) a body point at radius r moves at most ds (1 + k_max r), so every
point touched in between lies within half of that of one of the two sampled
footprints. Checking the samples with ``margin + sweep``,
sweep = 0.5 ds_max (1 + k_max r_max), therefore guarantees ``margin`` for the
whole continuous motion.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from doraware.interfaces import OccupancyGrid, VehicleParameters


@dataclass
class RevalidationResult:
    ok: bool
    n_checked: int
    check_margin: float               # margin + sweep bound actually applied [m]
    collision_pose: Optional[tuple] = None
    message: str = ""


def footprint_extents(vehicle: VehicleParameters, margin: float):
    """(front, rear, left, right) distances from base_link (all >= 0)."""
    return (vehicle.wheel_base + vehicle.front_overhang + margin, vehicle.rear_overhang + margin,
            vehicle.wheel_tread / 2.0 + vehicle.left_overhang + margin,
            vehicle.wheel_tread / 2.0 + vehicle.right_overhang + margin)


class FootprintChecker:
    def __init__(self, grid: OccupancyGrid, vehicle: VehicleParameters, margin: float = 0.0):
        self.cells = grid.as_2d()
        self.res = float(grid.resolution)
        self.ox, self.oy = float(grid.origin_x), float(grid.origin_y)
        self.vehicle = vehicle
        self.margin = margin
        self.ext = footprint_extents(vehicle, margin)

    def collides(self, x: float, y: float, yaw: float) -> bool:
        front, rear, left, right = self.ext
        c, s = math.cos(yaw), math.sin(yaw)
        local = np.array([[front, -right], [front, left], [-rear, left], [-rear, -right]])
        wx = x + c * local[:, 0] - s * local[:, 1]
        wy = y + s * local[:, 0] + c * local[:, 1]
        r = self.res
        h, w = self.cells.shape
        # map-axis separation -> candidate cell window; outside the grid = collision
        fx0, fx1 = (wx.min() - self.ox) / r, (wx.max() - self.ox) / r
        fy0, fy1 = (wy.min() - self.oy) / r, (wy.max() - self.oy) / r
        if fx0 < 0.0 or fy0 < 0.0 or fx1 > w or fy1 > h:
            return True
        ix0, ix1 = int(math.floor(fx0)), min(int(math.ceil(fx1)), w)
        iy0, iy1 = int(math.floor(fy0)), min(int(math.ceil(fy1)), h)
        iy, ix = np.nonzero(self.cells[iy0:iy1, ix0:ix1])
        if len(ix) == 0:
            return False
        cx = self.ox + (ix + ix0 + 0.5) * r - x
        cy = self.oy + (iy + iy0 + 0.5) * r - y
        # strict overlap on the map axes (cells merely touching the AABB are not hits)
        eps = 1e-9
        hit = ((cx + x + 0.5 * r > wx.min() + eps) & (cx + x - 0.5 * r < wx.max() - eps)
               & (cy + y + 0.5 * r > wy.min() + eps) & (cy + y - 0.5 * r < wy.max() - eps))
        # vehicle axes: project the cell centre, half-extent of a square on a unit axis = r/2 (|c| + |s|)
        e = 0.5 * r * (abs(c) + abs(s))
        pu = c * cx + s * cy
        pn = -s * cx + c * cy
        hit &= (pu + e > -rear + eps) & (pu - e < front - eps) & (pn + e > -right + eps) & (pn - e < left - eps)
        return bool(np.any(hit))


def sweep_bound(vehicle: VehicleParameters, margin: float, ds_max: float, k_max: float) -> float:
    front, rear, left, right = footprint_extents(vehicle, margin)
    r_max = max(math.hypot(front, max(left, right)), math.hypot(rear, max(left, right)))
    return 0.5 * ds_max * (1.0 + k_max * r_max)


def revalidate(poses_list: Sequence[np.ndarray], grid: OccupancyGrid, vehicle: VehicleParameters,
               margin: float, k_max: float) -> RevalidationResult:
    """``poses_list``: (n, 3) pose arrays of the consecutive motion pieces."""
    ds_max = max((float(np.max(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1])))) for p in poses_list if len(p) > 1),
                 default=0.0)
    m = margin + sweep_bound(vehicle, margin, ds_max, k_max)
    checker = FootprintChecker(grid, vehicle, m)
    n = 0
    for poses in poses_list:
        for x, y, yaw in poses:
            n += 1
            if checker.collides(float(x), float(y), float(yaw)):
                return RevalidationResult(False, n, m, (float(x), float(y), float(yaw)),
                                          f"footprint (margin {m:.3f} m) collides at "
                                          f"({x:.2f}, {y:.2f}, {math.degrees(yaw):.1f} deg)")
    return RevalidationResult(True, n, m, None, "collision free")
