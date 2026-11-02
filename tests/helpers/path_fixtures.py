"""Test helper: synthetic GeometricPaths made of constant-steering pieces
(the shape a lattice / Reeds-Shepp planner produces) and limit checks."""
from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np

from doraware.core.geometry import wrap_angles
from doraware.interfaces import GeometricPath, Gear, OccupancyGrid, VehicleParameters

VEHICLE = VehicleParameters()


def arc_path(pieces: Iterable[Tuple[int, float, float]], start=(5.0, 5.0, 0.0), ds: float = 0.025,
             wheel_base: float = VEHICLE.wheel_base) -> GeometricPath:
    """``pieces``: (direction +1/-1, steering [rad], length [m]). Per point the
    gear/curvature of the motion arriving at it (first point: first piece)."""
    x, y, yaw = start
    pts, curv, gear = [(x, y, yaw)], [], []
    for d, steer, length in pieces:
        k = math.tan(steer) / wheel_base
        n = max(1, int(math.ceil(length / ds)))
        step = length / n
        for _ in range(n):
            if abs(k) < 1e-12:
                x, y = x + d * step * math.cos(yaw), y + d * step * math.sin(yaw)
            else:
                yaw1 = yaw + d * step * k
                x, y = x + (math.sin(yaw1) - math.sin(yaw)) / k, y - (math.cos(yaw1) - math.cos(yaw)) / k
                yaw = yaw1
            pts.append((x, y, yaw))
            curv.append(k)
            gear.append(int(Gear.from_direction(d)))
    curv.insert(0, curv[0])
    gear.insert(0, gear[0])
    p = np.array(pts)
    return GeometricPath(x=p[:, 0].copy(), y=p[:, 1].copy(), yaw=wrap_angles(p[:, 2]), curvature=np.array(curv),
                         gear=np.array(gear, dtype=np.int8))


def free_grid(width_m: float = 40.0, height_m: float = 30.0, res: float = 0.1) -> OccupancyGrid:
    w, h = int(round(width_m / res)), int(round(height_m / res))
    return OccupancyGrid(resolution=res, width=w, height=h, data=np.zeros(w * h, dtype=np.uint8))


def sample_profile(profile, dt: float = 0.01, tail: float = 1.0):
    ts = np.arange(-0.5, profile.duration + tail, dt)
    s = [profile.sample(float(t)) for t in ts]
    arr = {k: np.array([getattr(p, k) for p in s], dtype=float)
           for k in ("x", "y", "yaw", "velocity", "acceleration", "curvature", "steering")}
    arr["gear"] = np.array([int(p.gear) for p in s])
    arr["t"] = ts
    return arr


def check_limits(arr, lim, dt: float = 0.01, tol: float = 1e-6) -> dict:
    """Numerical limit check of a profile sampled at ``dt``; returns the measured maxima."""
    v, a, d, k = arr["velocity"], arr["acceleration"], arr["steering"], arr["curvature"]
    fwd, rev = v[v > 0], v[v < 0]
    m = {
        "v_fwd": float(fwd.max()) if len(fwd) else 0.0,
        "v_rev": float(-rev.min()) if len(rev) else 0.0,
        "accel_fd": float(np.max(np.abs(np.diff(v)) / dt)),     # |dv/dt| from the velocity samples
        "accel": float(np.max(np.abs(a))),
        "jerk": float(np.max(np.abs(np.diff(a)) / dt)),
        "steer_rate": float(np.max(np.abs(np.diff(d)) / dt)),
        "lat": float(np.max(v * v * np.abs(k))),
    }
    speeding_up = np.diff(np.abs(v)) / dt
    m["acc_up"] = float(speeding_up.max())
    m["acc_down"] = float(-speeding_up.min())
    assert m["v_fwd"] <= lim.max_forward_speed + tol
    assert m["v_rev"] <= lim.max_reverse_speed + tol
    assert m["acc_up"] <= lim.max_acceleration + tol
    assert m["acc_down"] <= lim.max_deceleration + tol
    assert m["accel"] <= max(lim.max_acceleration, lim.max_deceleration) + tol
    assert m["jerk"] <= lim.max_jerk * (1 + 1e-6) + tol
    assert m["steer_rate"] <= lim.max_steer_rate * (1 + 1e-6) + tol
    assert m["lat"] <= lim.max_lateral_acceleration + tol
    return m
