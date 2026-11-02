"""Route -> dense centreline path (no Dora)."""
from __future__ import annotations

from dataclasses import dataclass

import math

import numpy as np

from doraware.core.geometry import wrap_angles
from doraware.interfaces import GeometricPath, Gear, LaneMap, LaneRelation, Pose2D, Route


class RouteError(ValueError):
    """The route cannot be turned into a path (unknown lanes, goal not on the route, ...)."""


@dataclass
class DensePath:
    s: np.ndarray        # arc length [m]
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    curvature: np.ndarray
    speed_limit: float

    def project(self, x: float, y: float, lo: int = 0, hi: int = -1) -> int:
        """Index of the nearest point (optionally within [lo, hi))."""
        hi = len(self.s) if hi < 0 else hi
        d = np.hypot(self.x[lo:hi] - x, self.y[lo:hi] - y)
        return lo + int(np.argmin(d))

    def project_s(self, x: float, y: float, lo: int = 0, hi: int = -1) -> float:
        """Arc length of the exact projection of (x, y) onto the polyline near
        its nearest point (optionally searched within [lo, hi))."""
        i = self.project(x, y, lo, hi)
        best = float(self.s[i])
        best_d = math.hypot(self.x[i] - x, self.y[i] - y)
        for j in (i - 1, i):
            if j < 0 or j + 1 >= len(self.s):
                continue
            ax, ay = self.x[j], self.y[j]
            dx, dy = self.x[j + 1] - ax, self.y[j + 1] - ay
            L2 = dx * dx + dy * dy
            u = min(max(((x - ax) * dx + (y - ay) * dy) / L2, 0.0), 1.0) if L2 > 0 else 0.0
            px, py = ax + u * dx, ay + u * dy
            d = math.hypot(px - x, py - y)
            if d < best_d:
                best, best_d = float(self.s[j] + u * (self.s[j + 1] - self.s[j])), d
        return best

    def cut(self, s0: float, s1: float, min_gap: float = 0.05) -> "DensePath":
        """The part between arc lengths s0 and s1, with exact end points."""
        inner = (self.s > s0 + min_gap) & (self.s < s1 - min_gap)
        ss = np.concatenate([[s0], self.s[inner], [s1]])
        x, y, yaw, k = self.interp(ss)
        return DensePath(ss - s0, x, y, yaw, k, self.speed_limit)

    def interp(self, s: np.ndarray):
        s = np.clip(s, self.s[0], self.s[-1])
        yaw = np.unwrap(self.yaw)
        return (np.interp(s, self.s, self.x), np.interp(s, self.s, self.y),
                wrap_angles(np.interp(s, self.s, yaw)), np.interp(s, self.s, self.curvature))

    def geometric(self, i0: int, i1: int) -> GeometricPath:
        sl = slice(i0, i1 + 1)
        n = i1 + 1 - i0
        return GeometricPath(x=self.x[sl].copy(), y=self.y[sl].copy(), yaw=self.yaw[sl].copy(),
                             curvature=self.curvature[sl].copy(), gear=np.full(n, int(Gear.DRIVE), np.int8))


def route_centerline(lanes: LaneMap, route: Route, start: Pose2D, goal: Pose2D, ds: float = 0.25) -> DensePath:
    """Centreline of the route lanes, resampled at ``ds`` and cut to the stretch
    between the projections of the start and the goal."""
    if len(route.lane_ids) == 0:
        raise RouteError("empty route")
    pts = []
    limits = []
    for lid in route.lane_ids:
        try:
            i = lanes.index_of(int(lid))
        except KeyError as exc:
            raise RouteError(str(exc)) from None
        c = lanes.centerline(i)
        if pts and len(c) and np.hypot(*(c[0] - pts[-1][-1])) < 1e-6:
            c = c[1:]
        pts.append(c)
        limits.append(float(lanes.speed_limit[i]))
    p = np.concatenate(pts)
    seg = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))
    keep = np.concatenate([[True], seg > 1e-9])
    p = p[keep]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1])))])
    sd = np.arange(0.0, s[-1] + 1e-9, ds)
    x, y = np.interp(sd, s, p[:, 0]), np.interp(sd, s, p[:, 1])
    yaw = np.arctan2(np.gradient(y), np.gradient(x))
    k = np.gradient(np.unwrap(yaw), sd)
    full = DensePath(sd, x, y, wrap_angles(yaw), k, min(limits))
    s0, s1 = full.project_s(start.x, start.y), full.project_s(goal.x, goal.y)
    gx, gy, _, _ = full.interp(np.array([s1]))
    if math.hypot(gx[0] - goal.x, gy[0] - goal.y) > 0.5:
        raise RouteError("the goal is not on the route centreline")
    if s1 <= s0 + 1.0:
        raise RouteError("the goal is not ahead of the start on the route")
    return full.cut(s0, s1)


def neighbor_offset(lanes: LaneMap, route: Route, path: DensePath, side: LaneRelation) -> np.ndarray:
    """Signed lateral offset (+ = left of the path) of the same-direction neighbour
    lane (``side``: LEFT_NEIGHBOR / RIGHT_NEIGHBOR) at every path point; NaN where
    the route lane there has no such neighbour."""
    out = np.full(len(path.s), np.nan)
    rel_from, rel_to, rel_type = (np.asarray(a) for a in (lanes.relation_from, lanes.relation_to, lanes.relation_type))
    for lid in route.lane_ids:
        i = lanes.index_of(int(lid))
        hits = rel_to[(rel_from == i) & (rel_type == int(side))]
        if len(hits) == 0:
            continue
        own, nb = lanes.centerline(i), lanes.centerline(int(hits[0]))
        if len(own) < 2 or len(nb) < 2:
            continue
        for k in range(len(path.s)):
            px, py = path.x[k], path.y[k]
            j = int(np.argmin(np.hypot(own[:, 0] - px, own[:, 1] - py)))
            if np.hypot(own[j, 0] - px, own[j, 1] - py) > 0.5:
                continue                                    # this path point is not on route lane i
            m = int(np.argmin(np.hypot(nb[:, 0] - px, nb[:, 1] - py)))
            nx, ny = -math.sin(path.yaw[k]), math.cos(path.yaw[k])
            dx, dy = nb[m, 0] - px, nb[m, 1] - py
            along = dx * math.cos(path.yaw[k]) + dy * math.sin(path.yaw[k])
            if abs(along) < 0.75:                            # the neighbour runs alongside here
                out[k] = dx * nx + dy * ny
    return out


def shifted(path: DensePath, offset: np.ndarray) -> DensePath:
    """The path moved sideways by ``offset`` (+ = left) at every point; arc length,
    heading and curvature recomputed."""
    nx, ny = -np.sin(path.yaw), np.cos(path.yaw)
    x, y = path.x + nx * offset, path.y + ny * offset
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    yaw = np.unwrap(np.arctan2(np.gradient(y), np.gradient(x)))
    k = np.gradient(yaw, s)
    return DensePath(s, x, y, wrap_angles(yaw), k, path.speed_limit)


def lane_change_profile(s: np.ndarray, s_out: float, s_back: float, length: float, width: float) -> np.ndarray:
    """Lateral offset along s: 0, a smooth (cosine) ramp to ``width`` over ``length``
    starting at ``s_out``, held, and back to 0 over ``length`` starting at ``s_back``."""
    def ramp(u):
        u = np.clip(u, 0.0, 1.0)
        return 0.5 - 0.5 * np.cos(math.pi * u)
    return width * (ramp((s - s_out) / length) - ramp((s - s_back) / length))
