"""Final stop-point generation: place the last stop on the goal.

Search-based planners accept a goal tolerance (the State Lattice planner:
0.10 m / 2°), so the path may end up to that far from the requested goal. That
residual would consume the whole system goal-accuracy budget before any
tracking error. When the residual is purely *longitudinal* along the path's
final heading (the planner stopped short of / past the goal on the same line),
the stop point can be moved onto the goal's projection without changing the
path shape: truncate the final segment (overshoot) or continue it along its
final curvature (undershoot). Lateral and yaw residuals cannot be removed by
moving along the path and are left untouched.

Safety: an extension is accepted only if every new pose passes the exact
footprint check with ``margin`` (the planner's safety margin); a truncation
stays on the planner's validated path.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from doraware.core.geometry import world_to_body
from doraware.interfaces import GeometricPath, OccupancyGrid, Pose2D, VehicleParameters
from doraware.planning.trajectory_generation.revalidation import FootprintChecker


@dataclass
class StopPointResult:
    path: GeometricPath
    applied: bool
    shift: float = 0.0          # [m] signed change of the final segment length (+ extend)
    reason: str = ""


def _last_segment_start(gear: np.ndarray) -> int:
    i = len(gear) - 1
    while i > 1 and gear[i - 1] == gear[-1]:
        i -= 1
    return max(i - 1, 0)


def snap_stop_to_goal(path: GeometricPath, goal: Optional[Pose2D], grid: Optional[OccupancyGrid],
                      vehicle: VehicleParameters, max_shift: float = 0.15, lateral_tol: float = 0.01,
                      margin: float = 0.1, ds: float = 0.01, min_shift: float = 0.005) -> StopPointResult:
    n = len(path)
    if goal is None or n < 2 or not all(map(math.isfinite, (goal.x, goal.y, goal.yaw))):
        return StopPointResult(path, False, reason="no goal")
    if not all(np.isfinite(np.asarray(a, dtype=float)).all() for a in (path.x, path.y, path.yaw, path.curvature)):
        return StopPointResult(path, False, reason="non-finite path (left to path validation)")
    ex, ey, eyaw = float(path.x[-1]), float(path.y[-1]), float(path.yaw[-1])
    lon, lat = world_to_body(ex, ey, eyaw, goal.x, goal.y)   # goal in the final-pose frame
    if abs(lat) > lateral_tol:
        return StopPointResult(path, False, reason=f"lateral residual {lat:+.3f} m")
    if abs(lon) < min_shift:
        return StopPointResult(path, False, reason="already at the goal")
    if abs(lon) > max_shift:
        return StopPointResult(path, False, reason=f"longitudinal residual {lon:+.3f} m too large")
    direction = 1 if int(path.gear[-1]) == 4 else -1        # Gear.DRIVE = 4
    shift = lon * direction                                 # + : travel further along the motion
    seg0 = _last_segment_start(np.asarray(path.gear))
    xs, ys, yaws = np.asarray(path.x), np.asarray(path.y), np.asarray(path.yaw)
    if shift < 0:  # overshoot: truncate the final segment
        s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(xs[seg0:]), np.diff(ys[seg0:])))])
        keep = s[-1] + shift
        if keep < 0.3:
            return StopPointResult(path, False, reason="final segment too short to truncate")
        j = int(np.searchsorted(s, keep))            # first index with s >= keep
        w = (keep - s[j - 1]) / max(s[j] - s[j - 1], 1e-12)
        i0, i1 = seg0 + j - 1, seg0 + j
        dyaw = math.remainder(yaws[i1] - yaws[i0], 2 * math.pi)
        new = GeometricPath(
            x=np.append(xs[:i1], xs[i0] + w * (xs[i1] - xs[i0])),
            y=np.append(ys[:i1], ys[i0] + w * (ys[i1] - ys[i0])),
            yaw=np.append(yaws[:i1], yaws[i0] + w * dyaw),
            curvature=np.append(np.asarray(path.curvature)[:i1], path.curvature[i1]),
            gear=np.append(np.asarray(path.gear)[:i1], path.gear[i1]).astype(np.int8))
        return StopPointResult(new, True, shift)
    # undershoot: continue along the final curvature (exact arc integration)
    k = float(path.curvature[-1])
    m = max(1, int(math.ceil(shift / ds)))
    d = np.arange(1, m + 1) * (shift / m) * direction     # signed travel along the heading
    if abs(k) < 1e-9:
        nx, ny, nyaw = ex + d * math.cos(eyaw), ey + d * math.sin(eyaw), np.full(m, eyaw)
    else:
        nyaw = eyaw + k * d
        nx = ex + (np.sin(nyaw) - math.sin(eyaw)) / k
        ny = ey - (np.cos(nyaw) - math.cos(eyaw)) / k
    if grid is not None:
        chk = FootprintChecker(grid, vehicle, margin)
        if any(chk.collides(float(a), float(b), float(c)) for a, b, c in zip(nx, ny, nyaw)):
            return StopPointResult(path, False, reason="extension violates the safety margin")
    new = GeometricPath(x=np.append(xs, nx), y=np.append(ys, ny), yaw=np.append(yaws, nyaw),
                        curvature=np.append(path.curvature, np.full(m, k)),
                        gear=np.append(path.gear, np.full(m, path.gear[-1])).astype(np.int8))
    return StopPointResult(new, True, shift)
