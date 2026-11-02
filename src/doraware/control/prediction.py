"""Delay compensation: command history + state prediction (docs/mpc.md §7).

The actuator of channel c (steering, acceleration) is modelled for prediction
as a pure delay D_c = dead_time_c (+ time_constant_c with lag compensation,
the first-order lag 1/(tau s + 1) ~ e^{-tau s} at low frequency). The
command issued now therefore acts from t_now + D, D = max(D_delta, D_a), and
the MPC initial state z_0 is the measured state propagated over [t_now, t_now + D]
with the *already issued* commands c(s - D_c) (zero-order hold).
"""
from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np


class CommandHistory:
    def __init__(self, keep: float = 2.0):
        self.keep = keep
        self.items: Deque[Tuple[float, float, float]] = deque()   # (t, delta_cmd, a_cmd)

    def push(self, t: float, delta: float, accel: float) -> None:
        self.items.append((t, delta, accel))
        while self.items and self.items[0][0] < t - self.keep:
            self.items.popleft()

    def clear(self) -> None:
        self.items.clear()

    def at(self, t: float) -> Optional[Tuple[float, float]]:
        """Command in force at time t (last issued with stamp <= t)."""
        out = None
        for (ti, d, a) in self.items:
            if ti <= t + 1e-9:
                out = (d, a)
            else:
                break
        return out


def predict_pose(x: float, y: float, yaw: float, v: float, delta_meas: float, t_now: float,
                 horizon: float, d_delta: float, d_acc: float, hist: CommandHistory,
                 gear_dir: int, wheel_base: float, step: float = 0.01) -> np.ndarray:
    """Propagate (x, y, yaw, v) over ``horizon`` seconds (rear-axle kinematic
    bicycle, Euler-midpoint sub-steps). Braking never reverses the motion
    (plant semantics: a request against the gear stops the car)."""
    if horizon <= 0.0:
        return np.array([x, y, yaw, v])
    n = max(int(math.ceil(horizon / step)), 1)
    h = horizon / n
    last = hist.items[-1] if hist.items else None
    for i in range(n):
        s = t_now + (i + 0.5) * h
        cd = hist.at(s - d_delta)
        ca = hist.at(s - d_acc)
        delta = cd[0] if (cd is not None and s - d_delta < t_now) else (last[1] if last else delta_meas)
        acc = ca[1] if (ca is not None and s - d_acc < t_now) else (last[2] if last else 0.0)
        v_new = v + acc * h
        if gear_dir > 0:
            v_new = max(v_new, 0.0)
        elif gear_dir < 0:
            v_new = min(v_new, 0.0)
        vm = 0.5 * (v + v_new)
        yaw_m = yaw + 0.5 * h * vm * math.tan(delta) / wheel_base
        x += h * vm * math.cos(yaw_m)
        y += h * vm * math.sin(yaw_m)
        yaw += h * vm * math.tan(delta) / wheel_base
        v = v_new
    return np.array([x, y, yaw, v])
