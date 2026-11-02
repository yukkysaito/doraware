"""Time-parameterised motion profile built from a GeometricPath.

The profile is a sequence of phases on the relative time axis t_rel >= 0:

    [start hold] [motion seg 0] [dwell] [motion seg 1] ... [motion seg n-1]  (then: final hold)

* start hold: v = 0 for ``start_stop_duration`` (gear engagement) and at
  least the time to rotate the steering from ``initial_steering`` to the first
  segment's start angle at ``max_steer_rate``.
* motion: exact constant-acceleration intervals of the segment's speed
  profile (``speed_profile.py``): s(tau) = s_k + v_k tau + a_k tau^2 / 2.
* dwell (gear change at a cusp): v = 0, a = 0, gear = the *next* segment's
  gear (the controller shifts while stopped), steering moves at
  ``max_steer_rate`` from the end angle of one segment to the start angle of
  the next; duration = max(stop_duration, |d delta| / max_steer_rate).
* after ``duration``: the final pose is held with v = 0 (stationary end).

``sample(t_rel)`` returns the signed velocity / acceleration (negative when
reversing) and the explicit gear.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from doraware.core.geometry import wrap_angle
from doraware.interfaces import GeometricPath, Gear
from doraware.planning.trajectory_generation.limits import ProfileLimits
from doraware.planning.trajectory_generation.path_processing import PathSegment, build_segments, split_at_cusps
from doraware.planning.trajectory_generation.speed_profile import SpeedProfile, optimize_speed


@dataclass
class ProfileSample:
    x: float
    y: float
    yaw: float              # wrapped to [-pi, pi)
    velocity: float         # signed longitudinal velocity [m/s]
    acceleration: float     # signed longitudinal acceleration [m/s^2]
    curvature: float        # tan(delta) / L [1/m]
    steering: float         # front-wheel angle delta [rad]
    gear: Gear


@dataclass
class _Hold:
    duration: float
    pose: tuple
    delta_from: float
    delta_to: float
    gear: Gear
    rate: float

    def sample(self, tau: float, wheel_base: float) -> ProfileSample:
        d = self.delta_to - self.delta_from
        delta = self.delta_from + math.copysign(min(abs(d), max(tau, 0.0) * self.rate), d)
        return ProfileSample(self.pose[0], self.pose[1], wrap_angle(self.pose[2]), 0.0, 0.0,
                             math.tan(delta) / wheel_base, delta, self.gear)


@dataclass
class _Motion:
    seg: PathSegment
    speed: SpeedProfile

    @property
    def duration(self) -> float:
        return self.speed.duration

    def sample(self, tau: float, wheel_base: float) -> ProfileSample:
        sp = self.speed
        n_int = len(sp.dt)
        tau = min(max(tau, 0.0), sp.duration)
        k = min(max(bisect.bisect_right(sp.t, tau) - 1, 0), n_int - 1)
        dt = tau - sp.t[k]
        v = max(sp.v[k] + sp.a[k] * dt, 0.0)
        s = min(sp.s[k] + sp.v[k] * dt + 0.5 * sp.a[k] * dt * dt, sp.s[k + 1])
        acc = self._accel(tau, k)
        delta = self.seg.steering_at(s)
        x, y, yaw = self.seg.pose_at(s)
        d = self.seg.direction
        return ProfileSample(x, y, yaw, d * v, d * acc, math.tan(delta) / wheel_base, delta, self.seg.gear)

    def _accel(self, tau: float, k: int) -> float:
        """Piecewise-linear acceleration through the interval mid-times (0 at both ends)."""
        sp = self.speed
        mid = sp.t[k] + 0.5 * sp.dt[k]
        if tau >= mid:
            t1, a1 = (sp.t[k + 1] + 0.5 * sp.dt[k + 1], sp.a[k + 1]) if k + 1 < len(sp.dt) else (sp.t[-1], 0.0)
            t0, a0 = mid, sp.a[k]
        else:
            t0, a0 = (sp.t[k - 1] + 0.5 * sp.dt[k - 1], sp.a[k - 1]) if k > 0 else (0.0, 0.0)
            t1, a1 = mid, sp.a[k]
        w = 0.0 if t1 <= t0 else (tau - t0) / (t1 - t0)
        return float(a0 + w * (a1 - a0))


class TimedProfile:
    """Deterministic function of the relative time ``t_rel`` [s]."""

    def __init__(self, phases: List[object], wheel_base: float, segments: List[PathSegment]):
        self.phases = phases
        self.wheel_base = wheel_base
        self.segments = segments
        self.starts = np.concatenate([[0.0], np.cumsum([p.duration for p in phases])]) if phases else np.zeros(1)

    @property
    def duration(self) -> float:
        return float(self.starts[-1])

    def sample(self, t_rel: float) -> ProfileSample:
        if not self.phases:
            raise ValueError("empty profile")
        if t_rel >= self.duration:                  # stationary end: hold the final pose
            last = self.phases[-1].sample(self.phases[-1].duration, self.wheel_base)
            last.velocity = 0.0
            last.acceleration = 0.0
            return last
        i = min(max(bisect.bisect_right(self.starts, t_rel) - 1, 0), len(self.phases) - 1)
        return self.phases[i].sample(t_rel - self.starts[i], self.wheel_base)

    def summary(self) -> dict:
        motions = [p for p in self.phases if isinstance(p, _Motion)]
        holds = [p for p in self.phases if isinstance(p, _Hold)]
        return {
            "duration": self.duration,
            "segments": len(motions),
            "gears": [int(m.seg.gear) for m in motions],
            "lengths": [m.seg.length for m in motions],
            "segment_durations": [m.duration for m in motions],
            "max_abs_velocity": max((float(np.max(m.speed.v)) for m in motions), default=0.0),
            "hold_durations": [h.duration for h in holds],
            "solver_status": [m.speed.solver_status for m in motions],
        }


def build_profile(path: GeometricPath, wheel_base: float, max_steer: float, lim: ProfileLimits,
                  initial_steering: Optional[float] = None) -> TimedProfile:
    """GeometricPath -> TimedProfile. ``initial_steering``: the vehicle's
    current front-wheel angle (None = assume it already matches the path)."""
    lim.validate()
    segs = build_segments(path, wheel_base, max_steer, lim.path_resample_ds, lim.steer_transition_length)
    rate = lim.max_steer_rate
    if not segs:      # start == goal (or no motion): stationary at the first point
        gear = Gear.from_direction(Gear(int(path.gear[0])).direction) if len(path) else Gear.PARK
        pose = (float(path.x[0]), float(path.y[0]), float(path.yaw[0]))
        d0 = 0.0 if initial_steering is None else float(initial_steering)
        return TimedProfile([_Hold(0.0, pose, d0, d0, gear, rate)], wheel_base, [])
    phases: List[object] = []
    first = segs[0]
    d_start = float(first.grid_delta[0])
    d_init = d_start if initial_steering is None else float(np.clip(initial_steering, -max_steer, max_steer))
    phases.append(_Hold(max(lim.start_stop_duration, abs(d_start - d_init) / rate),
                        (first.x[0], first.y[0], first.yaw[0]), d_init, d_start, first.gear, rate))
    for i, seg in enumerate(segs):
        v_max = lim.max_forward_speed if seg.direction > 0 else lim.max_reverse_speed
        phases.append(_Motion(seg, optimize_speed(seg.grid_s, seg.grid_kappa, seg.grid_delta, v_max, lim)))
        if i + 1 < len(segs):
            nxt = segs[i + 1]
            d0, d1 = float(seg.grid_delta[-1]), float(nxt.grid_delta[0])
            phases.append(_Hold(max(lim.stop_duration, abs(d1 - d0) / rate),
                                (seg.x[-1], seg.y[-1], seg.yaw[-1]), d0, d1, nxt.gear, rate))
    return TimedProfile(phases, wheel_base, segs)


__all__ = ["TimedProfile", "ProfileSample", "build_profile", "split_at_cusps"]
