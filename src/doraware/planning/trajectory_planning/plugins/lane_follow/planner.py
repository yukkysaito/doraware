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

Lane changes (passing a stopped obstacle): when a stationary vehicle or
obstacle (not a pedestrian, cyclist or animal) blocks the lane ahead, a
same-direction neighbour lane (``LaneMap`` LEFT/RIGHT_NEIGHBOR) runs alongside
for the whole manoeuvre, that lane is free of objects there, and there is
still room to move over (``lane_change_length`` before the obstacle), the
path is shifted into the neighbour lane with a smooth (cosine) ramp, held
past the obstacle and shifted back. Otherwise the vehicle stops and waits.

A reference for the trajectory planner contract (lanes, route and objects in,
Trajectory out, recomputed every tick), not a production planner: no
discretionary lane changes, no yielding rules, no prediction beyond constant
velocity, and traffic in the neighbour lane is only checked at its current
and predicted position.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from doraware.core.trajectory import stop_trajectory
from doraware.interfaces import (Gear, LaneRelation, ObjectClass, PerceivedObjects, PlanningRequest, PlanningResult,
                                 PlanningStatus, Trajectory, VehicleState, to_ns, to_sec)
from doraware.planning.trajectory_planning.plugins.lane_follow.route import (DensePath, RouteError,
                                                                              lane_change_profile, neighbor_offset,
                                                                              route_centerline, shifted)

STOP_SPEED = 0.05     # [m/s] below this, with a stop point right ahead, the profile stops

PARAMS = ("stop_margin", "corridor_margin", "prediction_time", "clear_time", "comfort_deceleration", "lookahead",
          "time_gap", "lead_min_speed", "curvature_smoothing", "lane_change_length", "lane_change_margin")
PASSABLE = {ObjectClass.UNKNOWN, ObjectClass.CAR, ObjectClass.TRUCK, ObjectClass.BUS, ObjectClass.STATIC_OBSTACLE}


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
        self.base: Optional[DensePath] = None      # the route centreline (own lane)
        self.neighbors: Dict[LaneRelation, np.ndarray] = {}
        self.lane_change_end: Optional[float] = None   # base-path arc length where the current one ends
        self.lane_changes: List[dict] = []

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
        self.base, self.lane_change_end, self.lane_changes = self.path, None, []
        self.neighbors = {side: neighbor_offset(request.lanes, request.route, self.base, side)
                          for side in (LaneRelation.LEFT_NEIGHBOR, LaneRelation.RIGHT_NEIGHBOR)}
        vp = request.vehicle
        self.v_max = min(float(self.lim["max_forward_speed"]), self.path.speed_limit)
        self.a_acc = min(float(self.lim["max_acceleration"]), vp.max_acceleration)
        self.a_dec = min(self.p["comfort_deceleration"], vp.max_deceleration)
        self.a_hard = vp.max_deceleration
        self._set_path(self.path)
        details = {"path_length": float(self.path.s[-1]), "speed_limit": self.path.speed_limit}
        return PlanningResult(request_id=request.request_id, status=PlanningStatus.SUCCESS, goal=request.goal,
                              path=self.path.geometric(0, len(self.path.s) - 1), details=json.dumps(details))

    def _set_path(self, path: DensePath) -> None:
        """Use ``path``: smoothed curvature and the static speed limits along it."""
        path.curvature = _smooth(path.curvature, path.s, self.p["curvature_smoothing"])
        self.path = path
        delta = np.arctan(self.request.vehicle.wheel_base * path.curvature)
        with np.errstate(divide="ignore", invalid="ignore"):
            slope = np.abs(np.gradient(delta, path.s))
            v_steer = np.where(slope > 1e-9, float(self.lim["max_steer_rate"]) / slope, np.inf)
            v_lat = np.where(np.abs(path.curvature) > 1e-9,
                             np.sqrt(float(self.lim["max_lateral_acceleration"]) / np.abs(path.curvature)), np.inf)
        self.v_static = np.minimum(np.minimum(v_steer, v_lat), self.v_max)       # per path point

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
        if self._maybe_change_lane(x, y, v_meas):
            self.i_now = self.path.project(x, y)
            s_meas = self.path.project_s(x, y)
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

    # ------------------------------------------------------------ lane change
    def _maybe_change_lane(self, x: float, y: float, v: float) -> bool:
        """Start passing a stationary obstacle through a free neighbour lane if
        possible; returns True when the path was replaced."""
        base = self.base
        sb_now = base.project_s(x, y)
        if self.lane_change_end is not None:
            if sb_now < self.lane_change_end:
                return False                                # still in a lane change
            self.lane_change_end = None
        vp = self.request.vehicle
        front = vp.wheel_base + vp.front_overhang
        L = self.p["lane_change_length"]
        for ob in self._corridor_objects(sb_now, v, path=base, lo=base.project(x, y)):
            if ob["v_along"] >= self.p["lead_min_speed"] or ob["cls"] not in PASSABLE:
                continue
            s_out = sb_now + 1.0
            if s_out + L > ob["s_back"] - front - 1.0:
                continue                                    # too close: stop and wait instead
            s_back = ob["s_front"] + vp.rear_overhang + 1.0
            if s_back + L > float(base.s[-1]) - 0.5:
                continue                                    # no room to return before the goal
            sel = (base.s >= s_out - 1e-9) & (base.s <= s_back + L + 1e-9)
            for side in (LaneRelation.LEFT_NEIGHBOR, LaneRelation.RIGHT_NEIGHBOR):
                off = self.neighbors[side][sel]
                if len(off) == 0 or not np.all(np.isfinite(off)):
                    continue
                width = float(np.mean(off))
                if not self._lane_free(width, s_out - 5.0, s_back + L + 5.0):
                    continue
                profile = lane_change_profile(base.s, s_out, s_back, L, width)
                self._set_path(shifted(base, profile))
                self.window, self.hold_s, self.clear_since = None, None, None
                self.lane_change_end = s_back + L
                self.lane_changes.append({"side": side.name, "s_out": s_out, "s_back": s_back,
                                          "object": int(self.objects.object_id[ob["k"]])})
                return True
        return False

    def _lane_free(self, offset: float, s_lo: float, s_hi: float) -> bool:
        """No object (now or after prediction_time) in the lane ``offset`` metres to
        the side of the base path between arc lengths s_lo and s_hi."""
        o, base = self.objects, self.base
        half = self.request.vehicle.vehicle_width / 2.0
        for k in range(len(o)):
            for dt in (0.0, self.p["prediction_time"]):
                ox, oy = o.x[k] + o.velocity_x[k] * dt, o.y[k] + o.velocity_y[k] * dt
                i = base.project(ox, oy)
                if not s_lo <= base.s[i] <= s_hi:
                    continue
                nx, ny = -math.sin(base.yaw[i]), math.cos(base.yaw[i])
                lateral = (ox - base.x[i]) * nx + (oy - base.y[i]) * ny
                rel = float(o.yaw[k] - base.yaw[i])
                across = abs(math.sin(rel)) * o.length[k] / 2 + abs(math.cos(rel)) * o.width[k] / 2
                if abs(lateral - offset) < half + across + self.p["lane_change_margin"]:
                    return False
        return True

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
        for ob in self._corridor_objects(s0, v0):
            s_back, v_along = ob["s_back"], ob["v_along"]
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

    def _corridor_objects(self, s0: float, v0: float, path: Optional[DensePath] = None,
                          lo: Optional[int] = None) -> List[dict]:
        """Objects whose box (now or after prediction_time) intrudes the corridor of
        ``path`` (default: the current path) ahead of s0: arc length of their rear
        (``s_back``) and front end, speed along the path, class and index."""
        o = self.objects
        if o is None or len(o) == 0:
            return []
        path = self.path if path is None else path
        vp = self.request.vehicle
        half = vp.vehicle_width / 2.0
        reach = s0 + vp.wheel_base + vp.front_overhang + v0 * v0 / (2.0 * self.a_dec) + self.p["stop_margin"] + \
            self.p["lookahead"]
        lo = self.i_now if lo is None else lo
        hi = int(np.searchsorted(path.s, min(reach, path.s[-1] + 5.0)))
        if hi <= lo + 1:
            return []
        out = []
        for k in range(len(o)):
            best = None
            for dt in (0.0, self.p["prediction_time"]):
                ox, oy = o.x[k] + o.velocity_x[k] * dt, o.y[k] + o.velocity_y[k] * dt
                i = path.project(ox, oy, lo, hi)
                yaw_i = float(path.yaw[i])
                rel = float(o.yaw[k]) - yaw_i
                across = abs(math.sin(rel)) * o.length[k] / 2 + abs(math.cos(rel)) * o.width[k] / 2
                along = abs(math.cos(rel)) * o.length[k] / 2 + abs(math.sin(rel)) * o.width[k] / 2
                nx, ny = -math.sin(yaw_i), math.cos(yaw_i)
                if abs((ox - path.x[i]) * nx + (oy - path.y[i]) * ny) > half + across + self.p["corridor_margin"]:
                    continue
                s_back = float(path.s[i]) - along
                if s_back < s0:                                  # behind the rear axle: not ahead
                    continue
                v_along = o.velocity_x[k] * math.cos(yaw_i) + o.velocity_y[k] * math.sin(yaw_i)
                if best is None or s_back < best["s_back"]:
                    best = {"s_back": s_back, "s_front": float(path.s[i]) + along,
                            "v_along": max(float(v_along), 0.0), "cls": ObjectClass(int(o.classification[k])),
                            "k": k}
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
