"""Fixed-time-step Trajectory windows cut from a TimedProfile.

Point i of the window is the profile state at absolute sim time
``t_now + i * time_step`` (relative time t_rel = (t_now - plan_start) [s] + i
time_step). Every window is a sampling of the same deterministic function of
absolute time, so windows produced at different ``t_now`` agree wherever they
overlap (no discontinuity between consecutive publications).
"""
from __future__ import annotations

import numpy as np

from doraware.interfaces import ClockDomain, Header, Trajectory, to_ns, to_sec
from doraware.planning.trajectory_generation.profile import TimedProfile


def window_size(time_step: float, horizon: float) -> int:
    """N = round(horizon / time_step) + 1 points (both ends included)."""
    if time_step <= 0.0 or horizon < 0.0:
        raise ValueError("time_step must be > 0 and horizon >= 0")
    return int(round(horizon / time_step)) + 1


def make_trajectory(profile: TimedProfile, plan_start_ns: int, t_now_ns: int, time_step: float, horizon: float,
                    validity: float, header_seq: int = 0, frame_id: str = "map") -> Trajectory:
    n = window_size(time_step, horizon)
    t0 = to_sec(t_now_ns - plan_start_ns)          # integer ns difference first: no precision loss
    cols = {k: np.empty(n) for k in ("x", "y", "yaw", "v", "a", "k", "d")}
    gear = np.empty(n, dtype=np.int8)
    for i in range(n):
        p = profile.sample(t0 + i * time_step)
        cols["x"][i], cols["y"][i], cols["yaw"][i] = p.x, p.y, p.yaw
        cols["v"][i], cols["a"][i], cols["k"][i], cols["d"][i] = p.velocity, p.acceleration, p.curvature, p.steering
        gear[i] = int(p.gear)
    return Trajectory(header=Header(stamp=t_now_ns, frame_id=frame_id, sequence_id=header_seq, clock=ClockDomain.SIM),
                      start_time=t_now_ns, time_step=float(time_step), valid_until=t_now_ns + to_ns(validity),
                      x=cols["x"], y=cols["y"], yaw=cols["yaw"], longitudinal_velocity=cols["v"],
                      longitudinal_acceleration=cols["a"], curvature=cols["k"], front_wheel_angle=cols["d"],
                      gear=gear)


def empty_trajectory(t_now_ns: int, time_step: float, validity: float, header_seq: int = 0,
                     frame_id: str = "map") -> Trajectory:
    """No points = stop (the controller brakes / holds)."""
    return Trajectory(header=Header(stamp=t_now_ns, frame_id=frame_id, sequence_id=header_seq, clock=ClockDomain.SIM),
                      start_time=t_now_ns, time_step=float(time_step), valid_until=t_now_ns + to_ns(validity))
