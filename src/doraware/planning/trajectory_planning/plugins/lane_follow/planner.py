"""``lane_follow``: an online lane-following reference trajectory planner.

Plan: the route centreline from the start to the goal (``route.py``), with the
curvature smoothed over ``curvature_smoothing`` metres (roads built from
straights and arcs have curvature steps that no steering can follow).

Every planner tick it builds the speed profile again, from the state of the
previous plan at this time (or the measured state if they disagree), as the
largest speed that respects, along the path ahead,

* the speed limit (lanes, ``max_forward_speed``), the lateral acceleration and
  the steering-rate limits (``planning.trajectory_generation``),
* stopping exactly at the goal,
* objects in the corridor of the path (now, or ``prediction_time`` ahead at
  their velocity): a vehicle moving along the path is **followed** (its speed
  at a gap of ``stop_margin + time_gap * v``), anything else is **stopped
  for** ``stop_margin`` before it; the vehicle continues once the corridor has
  been clear for ``clear_time``,

all braked for at ``comfort_deceleration`` (a backward pass), then integrates it
forward with the acceleration limit. Sudden obstacles may need harder
braking, up to the vehicle's ``max_deceleration``.

A reference for the trajectory planner contract (lanes, route and objects in,
Trajectory out, recomputed every tick), not a production planner: no lane
changes, no yielding rules, no prediction beyond constant velocity.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from doraware.core.trajectory import stop_trajectory
from doraware.interfaces import (Gear, PerceivedObjects, PlanningRequest, PlanningResult, PlanningStatus,
                                 Trajectory, VehicleState, to_ns, to_sec)
from doraware.planning.trajectory_planning.plugins.lane_follow.route import DensePath, RouteError, route_centerline

STOP_SPEED = 0.05     # [m/s] below this, with a stop point right ahead, the profile stops

PARAMS = ("stop_margin", "corridor_margin", "prediction_time", "clear_time", "comfort_deceleration", "lookahead",
          "time_gap", "lead_min_speed", "curvature_smoothing")


@dataclass
class _Window:
    t0: float              # [s] time of the first point
    dt: float
    s: np.ndarray
    v: np.ndarray
    a: np.ndarray


class LaneFollow:
    inputs = {"objects": PerceivedObjects}

    def __init__(self, params: Dict[str, Any], cfg: Dict[str, Any]):
        unknown, missing = set(params) - set(PARAMS), set(PARAMS) - set(params)
        if unknown or missing:     # values come from planning.trajectory_planner.lane_follow
            raise ValueError(f"lane_follow: unknown parameters {sorted(unknown)}, missing {sorted(missing)}")
        self.p = {k: float(params[k]) for k in PARAMS}
        self.timing = cfg["timing"]
        self.lim = dict(cfg["planning"]["trajectory_generation"])
        self.objects: Optional[PerceivedObjects] = None
        self.path: Optional[DensePath] = None
        self.request: Optional[PlanningRequest] = None
        self.window: Optional[_Window] = None
        self.hold_s: Optional[float] = None       # stop point kept until the corridor is clear
        self.clear_since: Optional[float] = None
        self.i_now = 0

    # ---------------------------------------------------------------- inputs
    def on_input(self, name: str, msg: Any) -> None:
        self.objects = msg

    def plan(self, request: PlanningRequest, state: VehicleState, t_ns: int) -> Optional[PlanningResult]:
        self.request, self.window, self.hold_s, self.clear_since, self.i_now = request, None, None, None, 0
        try:
            self.path = route_centerline(request.lanes, request.route, request.start, request.goal)
        except RouteError as exc:
            self.path = None
            return PlanningResult(request_id=request.request_id, status=PlanningStatus.INVALID_INPUT,
                                  message=str(exc), goal=request.goal)
        self.path.curvature = _smooth(self.path.curvature, self.path.s, self.p["curvature_smoothing"])
        vp = request.vehicle
        self.v_max = min(float(self.lim["max_forward_speed"]), self.path.speed_limit)
        self.a_acc = min(float(self.lim["max_acceleration"]), vp.max_acceleration)
        self.a_dec = min(self.p["comfort_deceleration"], vp.max_deceleration)
        self.a_hard = vp.max_deceleration
        delta = np.arctan(vp.wheel_base * self.path.curvature)
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.abs(np.gradient(delta, self.path.s))
            v_steer = np.where(slope > 1e-9, float(self.lim["max_steer_rate"]) / slope, np.inf)
            v_lat = np.where(np.abs(self.path.curvature) > 1e-9,
                             np.sqrt(float(self.lim["max_lateral_acceleration"]) / np.abs(self.path.curvature)), np.inf)
        self.v_static = np.minimum(np.minimum(v_steer, v_lat), self.v_max)       # per path point
        details = {"path_length": float(self.path.s[-1]), "speed_limit": self.path.speed_limit}
        return PlanningResult(request_id=request.request_id, status=PlanningStatus.SUCCESS, goal=request.goal,
                              path=self.path.geometric(0, len(self.path.s) - 1), details=json.dumps(details))

    def diagnostics(self) -> Dict[str, Any]:
        # no plan timing: the end depends on traffic, so the evaluator waits until the
        # trajectory comes to rest at the goal
        return {}

    # ------------------------------------------------------------- per tick
    def trajectory_at(self, state: VehicleState, t_ns: int) -> Trajectory:
        if self.path is None:
            return stop_trajectory(t_ns, self.timing)
        t = to_sec(t_ns)
        x, y, _ = state.pose.xyyaw()
        lo, hi = max(self.i_now - 8, 0), min(self.i_now + 80, len(self.path.s))
        i = self.path.project(x, y, lo, hi)
        if math.hypot(self.path.x[i] - x, self.path.y[i] - y) > 1.0:     # far from the last position
            lo, hi = 0, len(self.path.s)
        self.i_now = self.path.project(x, y, lo, hi)
        s_meas, v_meas = self.path.project_s(x, y, lo, hi), max(float(state.speed), 0.0)
        s0, v0 = s_meas, v_meas
        w = self.window
        if w is not None:                            # continue the previous plan if the car is on it
            k = int(round((t - w.t0) / w.dt))
            if 0 <= k < len(w.s) and abs(w.s[k] - s_meas) < 0.3 and abs(w.v[k] - v_meas) < 0.5:
                s0, v0 = float(w.s[k]), float(w.v[k])
        envelope, moving = self._envelope(s0, v0, t)
        dt = float(self.timing["trajectory_time_step"])
        n = int(round(float(self.timing["trajectory_horizon"]) / dt)) + 1
        s, v, a = np.zeros(n), np.zeros(n), np.zeros(n)
        s_goal = float(self.path.s[-1])
        si, vi = s0, v0
        for i in range(n):
            s[i], v[i] = si, vi
            ahead = si + vi * dt
            cap = envelope(ahead)
            for s_block, v_lead in moving:                   # a lead vehicle keeps moving meanwhile
                gap = s_block + v_lead * (i + 1) * dt - ahead
                cap = min(cap, math.sqrt(v_lead * v_lead + 2.0 * self.a_dec * max(gap, 0.0)))
            v_next = min(vi + self.a_acc * dt, cap)
            v_next = max(v_next, vi - self.a_hard * dt, 0.0)
            if v_next < STOP_SPEED and cap < STOP_SPEED:     # arrived at a stop point (goal or object)
                v_next = 0.0
            a[i] = (v_next - vi) / dt
            si = min(si + 0.5 * (vi + v_next) * dt, s_goal)
            vi = 0.0 if s_goal - si < 1e-3 else v_next
        a[-1] = 0.0 if v[-1] == 0.0 else a[-1]
        self.window = _Window(t, dt, s, v, a)
        px, py, pyaw, pk = self.path.interp(s)
        return Trajectory(start_time=t_ns, time_step=dt,
                          valid_until=t_ns + to_ns(float(self.timing["trajectory_validity"])),
                          x=px, y=py, yaw=pyaw, longitudinal_velocity=v, longitudinal_acceleration=a, curvature=pk,
                          front_wheel_angle=np.arctan(self.request.vehicle.wheel_base * pk),
                          gear=np.full(n, int(Gear.DRIVE), np.int8))

    # --------------------------------------------------------------- helpers
    def _envelope(self, s0: float, v0: float, t: float):
        """Admissible speed as a function of arc length (rear axle), and the moving
        lead vehicles (s_block, v_lead). Speed limits from the path (curvature,
        steering rate) are braked for on the path grid; stop points (the goal,
        objects) are exact, so a stop lands on them, not on a grid point."""
        path, a = self.path, self.a_dec
        env = self.v_static.copy()                           # backward pass: brake in time
        ds = np.diff(path.s)
        for j in range(len(env) - 2, -1, -1):
            env[j] = min(env[j], math.sqrt(env[j + 1] ** 2 + 2.0 * a * ds[j]))
        stops = [float(path.s[-1])]
        moving = []
        for s_block, v_lead in self._blockers(s0, v0, t):
            (moving.append((s_block, v_lead)) if v_lead > 0.0 else stops.append(s_block))

        def cap(s: float) -> float:
            c = float(np.interp(s, path.s, env))
            for sb in stops:
                c = min(c, math.sqrt(2.0 * a * max(sb - s, 0.0)))
            return c
        return cap, moving

    def _blockers(self, s0: float, v0: float, t: float) -> List[tuple]:
        """(rear-axle arc length to stay behind, speed there) for every object in
        the corridor ahead; a stop point is kept until the corridor has been clear
        for clear_time."""
        out = []
        vp = self.request.vehicle
        front = vp.wheel_base + vp.front_overhang
        for s_back, v_along in self._corridor_objects(s0, v0):
            moving = v_along >= self.p["lead_min_speed"]
            v_lead = v_along if moving else 0.0
            gap = self.p["stop_margin"] + (self.p["time_gap"] * v_lead if moving else 0.0)
            out.append((s_back - front - gap, v_lead))
        stops = [s for s, v in out if v == 0.0]
        if stops:
            self.hold_s, self.clear_since = min(stops), None
        elif self.hold_s is not None:
            self.clear_since = t if self.clear_since is None else self.clear_since
            if t - self.clear_since >= self.p["clear_time"]:
                self.hold_s = None
            else:
                out.append((self.hold_s, 0.0))
        return out

    def _corridor_objects(self, s0: float, v0: float) -> List[tuple]:
        """(arc length of the object's rear end, its speed along the path) for objects
        whose box (now or after prediction_time) intrudes the corridor ahead."""
        o = self.objects
        if o is None or len(o) == 0:
            return []
        vp = self.request.vehicle
        half = vp.vehicle_width / 2.0
        reach = s0 + vp.wheel_base + vp.front_overhang + v0 * v0 / (2.0 * self.a_dec) + self.p["stop_margin"] + \
            self.p["lookahead"]
        lo = self.i_now
        hi = int(np.searchsorted(self.path.s, min(reach, self.path.s[-1] + 5.0)))
        if hi <= lo + 1:
            return []
        out = []
        for k in range(len(o)):
            best = None
            for dt in (0.0, self.p["prediction_time"]):
                ox, oy = o.x[k] + o.velocity_x[k] * dt, o.y[k] + o.velocity_y[k] * dt
                i = self.path.project(ox, oy, lo, hi)
                yaw_i = float(self.path.yaw[i])
                rel = float(o.yaw[k]) - yaw_i
                across = abs(math.sin(rel)) * o.length[k] / 2 + abs(math.cos(rel)) * o.width[k] / 2
                along = abs(math.cos(rel)) * o.length[k] / 2 + abs(math.sin(rel)) * o.width[k] / 2
                nx, ny = -math.sin(yaw_i), math.cos(yaw_i)
                if abs((ox - self.path.x[i]) * nx + (oy - self.path.y[i]) * ny) > half + across + \
                        self.p["corridor_margin"]:
                    continue
                s_back = float(self.path.s[i]) - along
                if s_back < s0:                                  # behind the rear axle: not ahead
                    continue
                v_along = o.velocity_x[k] * math.cos(yaw_i) + o.velocity_y[k] * math.sin(yaw_i)
                if best is None or s_back < best[0]:
                    best = (s_back, max(float(v_along), 0.0))
            if best is not None:
                out.append(best)
        return out


def _smooth(k: np.ndarray, s: np.ndarray, length: float) -> np.ndarray:
    """Moving average of the curvature over ``length`` metres (curvature steps -> ramps)."""
    if length <= 0 or len(s) < 3:
        return k
    ds = float(np.median(np.diff(s)))
    w = max(int(round(length / ds)), 1) | 1
    pad = np.pad(k, w // 2, mode="edge")
    return np.convolve(pad, np.ones(w) / w, mode="valid")
