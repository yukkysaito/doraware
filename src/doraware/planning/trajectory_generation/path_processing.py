"""GeometricPath -> direction segments with a followable steering reference.

Conventions of :class:`GeometricPath` (see doraware.interfaces.planning): ``gear[i]`` and
``curvature[i]`` describe the motion that *arrives* at point ``i``; so the
interval ``i -> i+1`` has direction ``gear[i+1]`` and curvature
``curvature[i+1]``. ``curvature`` is the steering curvature
kappa = tan(delta) / L (yaw rate = signed v * kappa, also when reversing).

Geometry is never re-integrated: the poses of a segment are the planner's
points (linear interpolation of x, y and of the unwrapped yaw in between), so
the planner's collision guarantee still applies to the reference poses.

Steering ramp
-------------
Planner steering is piecewise constant; a step cannot be followed (finite
steering rate). The steering *reference* uses the box-filtered curvature

    kappa_f(s) = (K(s + h) - K(s - h)) / (2 h),   K(s) = integral_0^s kappa,
    h(s) = min(w / 2, s, S - s),

with w = ``steer_transition_length``: an isolated step of size dk becomes a
linear ramp of width w centred on the step (overlapping ramps superpose). A
vehicle driving kappa_f instead of the step deviates laterally from the
planner path by at most (small-angle, double integral of the curvature error)

    e_max = |dk| * w^2 / 24          (per step; steps add up at worst),

e.g. dk = 2 * 0.22 1/m (full left -> full right), w = 0.6 m: 6.6 mm, well inside
the planner's 0.1 m safety margin. Near a segment end h shrinks to 0 so the
steering at the stop point equals the planner's (the remaining step, if
any, is done at standstill during the dwell). The speed profile then limits
v so that |d delta/dt| = |d delta/ds| * |v| <= max_steer_rate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from doraware.core.geometry import wrap_angle
from doraware.interfaces import GeometricPath, Gear

MIN_INTERVAL = 1e-9       # [m] duplicate points are dropped
MIN_SEGMENT_LENGTH = 1e-4 # [m] shorter direction segments are ignored (no motion)


class InvalidPathError(ValueError):
    pass


@dataclass
class PathSegment:
    """One direction segment (cusp to cusp). Arrays of the original points:
    ``s`` (cumulative chord length from 0), ``x``, ``y``, ``yaw`` (unwrapped),
    ``kappa_int`` (curvature of each interval, len n-1). Speed-profile grid:
    ``grid_s`` (uniform, ends exactly at ``length``), ``grid_delta`` (ramped
    front-wheel angle) and ``grid_kappa`` = tan(grid_delta) / L."""
    direction: int
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    kappa_int: np.ndarray
    grid_s: np.ndarray
    grid_delta: np.ndarray
    grid_kappa: np.ndarray

    @property
    def length(self) -> float:
        return float(self.s[-1])

    @property
    def gear(self) -> Gear:
        return Gear.from_direction(self.direction)

    def pose_at(self, s: float) -> Tuple[float, float, float]:
        return (float(np.interp(s, self.s, self.x)), float(np.interp(s, self.s, self.y)),
                wrap_angle(float(np.interp(s, self.s, self.yaw))))

    def steering_at(self, s: float) -> float:
        """Front-wheel angle, linear in s between grid nodes."""
        return float(np.interp(s, self.grid_s, self.grid_delta))

    def poses(self) -> np.ndarray:
        return np.column_stack([self.x, self.y, self.yaw])

    def ramp_deviation_bound(self, w: float) -> float:
        """Sum over curvature steps of |dk| * w^2 / 24 (see module doc)."""
        return float(np.abs(np.diff(self.kappa_int)).sum() * w * w / 24.0)


def split_at_cusps(path: GeometricPath) -> List[Tuple[int, np.ndarray]]:
    """[(direction, point indices)] per direction segment; neighbours share the cusp point."""
    n = len(path)
    if n == 0:
        raise InvalidPathError("empty path")
    arrays = (path.x, path.y, path.yaw, path.curvature, path.gear)
    if any(len(a) != n for a in arrays):
        raise InvalidPathError("path arrays have different lengths")
    if not all(np.all(np.isfinite(np.asarray(a, dtype=float))) for a in arrays[:4]):
        raise InvalidPathError("path contains non-finite values")
    dirs = np.array([Gear(int(g)).direction for g in path.gear])
    if n == 1:
        return []
    d_int = dirs[1:]                     # direction of interval i -> i+1
    if np.any(d_int == 0):
        raise InvalidPathError("path gear must be DRIVE or REVERSE")
    out, start = [], 0
    for j in range(1, n - 1):            # interval j-1 vs interval j: cusp at point j
        if d_int[j] != d_int[j - 1]:
            out.append((int(d_int[j - 1]), np.arange(start, j + 1)))
            start = j
    out.append((int(d_int[-1]), np.arange(start, n)))
    return out


def build_segment(path: GeometricPath, direction: int, idx: np.ndarray, wheel_base: float,
                  max_steer: float, ds: float, w: float) -> PathSegment | None:
    """Segment from point indices ``idx`` (None if it has no length)."""
    x, y = np.asarray(path.x, float)[idx], np.asarray(path.y, float)[idx]
    yaw = np.unwrap(np.asarray(path.yaw, float)[idx])
    kappa = np.asarray(path.curvature, float)[idx[1:]]          # interval curvature (arrival convention)
    step = np.hypot(np.diff(x), np.diff(y))
    keep = step > MIN_INTERVAL
    if not np.any(keep):
        return None
    pts = np.concatenate([[True], keep])
    x, y, yaw, step, kappa = x[pts], y[pts], yaw[pts], step[keep], kappa[keep]
    s = np.concatenate([[0.0], np.cumsum(step)])
    length = float(s[-1])
    if length < MIN_SEGMENT_LENGTH:
        return None
    n = max(2, int(math.ceil(length / ds - 1e-9)))
    grid_s = np.linspace(0.0, length, n + 1)
    kf = filtered_curvature(s, kappa, grid_s, w)
    delta = np.clip(np.arctan(wheel_base * kf), -max_steer, max_steer)
    return PathSegment(direction, s, x, y, yaw, kappa, grid_s, delta, np.tan(delta) / wheel_base)


def filtered_curvature(s: np.ndarray, kappa_int: np.ndarray, query: np.ndarray, w: float) -> np.ndarray:
    """Box filter of the piecewise-constant curvature with the shrinking
    half-width h(s) = min(w/2, s, S - s), evaluated exactly through the
    piecewise-linear integral K(s)."""
    K = np.concatenate([[0.0], np.cumsum(kappa_int * np.diff(s))])
    S = s[-1]
    h = np.minimum(np.minimum(0.5 * w, query), S - query)
    out = np.empty_like(query)
    big = h > 1e-6
    out[big] = (np.interp(query[big] + h[big], s, K) - np.interp(query[big] - h[big], s, K)) / (2.0 * h[big])
    # h -> 0 (segment ends): the curvature of the interval at that end
    j = np.clip(np.searchsorted(s, query[~big], side="right") - 1, 0, len(kappa_int) - 1)
    out[~big] = kappa_int[j]
    return out


def build_segments(path: GeometricPath, wheel_base: float, max_steer: float, ds: float,
                   w: float) -> List[PathSegment]:
    segs = []
    for direction, idx in split_at_cusps(path):
        seg = build_segment(path, direction, idx, wheel_base, max_steer, ds, w)
        if seg is not None:
            segs.append(seg)
    return segs
