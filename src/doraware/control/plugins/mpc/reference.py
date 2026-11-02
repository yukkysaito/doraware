"""Horizon reference built from a time-based Trajectory (docs/mpc.md §5).

Reference sample k is the trajectory state at sim time t0 + k*dt, where
t0 = t_now + prediction delay. Rules:

* Times beyond a *moving* end are undefined (``sampling.sample`` -> None):
  those steps get ``valid = False`` (zero tracking weight), never extrapolated.
* Yaw is unwrapped along the horizon and shifted by 2*pi*n so that sample 0
  is within pi of the current yaw (continuous error for the QP).
* Direction consistency: the MPC runs in the vehicle's current gear. From the
  first sample whose gear direction differs (a cusp ahead) the reference is
  clamped to the last same-direction pose with v = a = 0 (stop at the cusp),
  while the steering reference keeps following the trajectory (the steering
  can be prepared during the stop).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from doraware.core.geometry import unwrap_near
from doraware.interfaces import Gear, Trajectory, to_sec
from doraware.core.trajectory import STATIONARY_EPS, ReferencePoint, sample


@dataclass
class HorizonReference:
    t: np.ndarray          # (N+1,) sim times [s]
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray        # unwrapped near the current yaw
    v: np.ndarray
    a: np.ndarray
    delta: np.ndarray
    valid: np.ndarray      # (N+1,) bool
    clamped: np.ndarray    # (N+1,) bool, True after a direction change (cusp hold)


def trajectory_usable(traj: Optional[Trajectory], t_now: float) -> bool:
    """A trajectory may be used at ``t_now`` [s] iff it is unexpired and defined there."""
    if traj is None or len(traj) == 0:
        return False
    if t_now > to_sec(traj.valid_until):
        return False
    return sample(traj, t_now) is not None


def build_reference(traj: Trajectory, t0: float, n: int, dt: float, yaw_now: float,
                    gear_dir: int) -> HorizonReference:
    times = t0 + dt * np.arange(n + 1)
    pts = [sample(traj, float(t)) for t in times]
    valid = np.array([p is not None for p in pts])
    # beyond a moving end everything is undefined (monotone in time)
    if not valid.all():
        first_bad = int(np.argmin(valid))
        valid[first_bad:] = False
    last = next((p for p in reversed(pts) if p is not None), None)
    filled = [p if p is not None else last for p in pts]
    if last is None:
        filled = [ReferencePoint(float(t), 0.0, 0.0, yaw_now, 0.0, 0.0, 0.0, 0.0, Gear.NONE) for t in times]

    def arr(name: str) -> np.ndarray:
        return np.array([getattr(p, name) for p in filled], dtype=float)

    x, y, v, a, delta = arr("x"), arr("y"), arr("velocity"), arr("acceleration"), arr("steering")
    yaw = unwrap_near(arr("yaw"), yaw_now)
    clamped = np.zeros(n + 1, dtype=bool)
    if gear_dir != 0:
        for k, p in enumerate(filled):
            d = p.gear.direction if p is not None else 0
            if valid[k] and d != 0 and d != gear_dir:
                j = max(k - 1, 0)
                x[k:], y[k:], yaw[k:] = x[j], y[j], yaw[j]
                v[k:], a[k:] = 0.0, 0.0
                clamped[k:] = True
                break
    # a reference moving against the gear (should not happen) is clipped to zero speed
    if gear_dir > 0:
        v = np.maximum(v, 0.0)
    elif gear_dir < 0:
        v = np.minimum(v, 0.0)
    return HorizonReference(times, x, y, yaw, v, a, delta, valid, clamped)


def is_stationary_window(traj: Trajectory, t_from: float, t_to: float, step: float = 0.05) -> bool:
    """True if the reference does not move in [t_from, t_to] (used for standstill hold)."""
    t = t_from
    while True:
        p = sample(traj, t)
        if p is None or abs(p.velocity) >= STATIONARY_EPS or abs(p.acceleration) >= 1e-3:
            return False
        if t >= t_to:
            return True
        t = min(t + step, t_to)
