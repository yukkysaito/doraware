"""Time-based trajectory queries (shared by controller, evaluation, viz).

``sample(traj, t)`` returns the desired state at sim time ``t`` [s] by linear
interpolation between fixed-step points (yaw interpolated on the circle).
Before ``start_time`` the first point is returned; after the last point the
last point is *held* only if it is stationary (|v| < eps), else ``None``
(undefined -- the caller must not extrapolate a moving reference).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from doraware.core.geometry import angle_diff, wrap_angle
from doraware.interfaces import Gear, Trajectory, to_sec

STATIONARY_EPS = 1e-3


@dataclass
class ReferencePoint:
    t: float
    x: float
    y: float
    yaw: float
    velocity: float
    acceleration: float
    curvature: float
    steering: float
    gear: Gear
    held: bool = False          # True if beyond the end (stationary hold)


def _point(traj: Trajectory, i: int, t: float, held: bool = False) -> ReferencePoint:
    return ReferencePoint(t, float(traj.x[i]), float(traj.y[i]), float(traj.yaw[i]),
                          0.0 if held else float(traj.longitudinal_velocity[i]),
                          0.0 if held else float(traj.longitudinal_acceleration[i]),
                          float(traj.curvature[i]), float(traj.front_wheel_angle[i]), Gear(int(traj.gear[i])), held)


def is_stationary_end(traj: Trajectory) -> bool:
    return len(traj) > 0 and abs(float(traj.longitudinal_velocity[-1])) < STATIONARY_EPS


def sample(traj: Trajectory, t: float) -> Optional[ReferencePoint]:
    n = len(traj)
    if n == 0:
        return None
    u = (t - to_sec(traj.start_time)) / traj.time_step
    if u <= 0.0:
        return _point(traj, 0, t)
    if u >= n - 1:
        if u - (n - 1) < 1e-9:
            return _point(traj, n - 1, t)
        return _point(traj, n - 1, t, held=True) if is_stationary_end(traj) else None
    i = int(np.floor(u))
    w = u - i
    j = i + 1

    def lerp(a):
        return float(a[i] + w * (a[j] - a[i]))

    gear = Gear(int(traj.gear[i] if w < 0.5 else traj.gear[j]))
    return ReferencePoint(t, lerp(traj.x), lerp(traj.y),
                          wrap_angle(float(traj.yaw[i]) + w * angle_diff(float(traj.yaw[j]), float(traj.yaw[i]))),
                          lerp(traj.longitudinal_velocity), lerp(traj.longitudinal_acceleration),
                          lerp(traj.curvature), lerp(traj.front_wheel_angle), gear)
