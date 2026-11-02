"""``lane_follow``: a lane-following reference trajectory planner.

* Plan: the route centreline from the start to the goal (``route.py``), timed by
  the trajectory generation with the route's speed limit.
* Every planner tick it checks the declared ``objects`` input: an object whose
  box (now, or ``prediction_time`` ahead at its velocity) lies in the corridor
  of the path ahead blocks it. The vehicle then brakes (at least
  ``comfort_deceleration``, at most the vehicle limit) to stop
  ``stop_margin`` before the object, waits until the corridor has been clear
  for ``clear_time``, and continues along the rest of the path from standstill.

It is a reference for the trajectory planner contract (lanes, route and
objects in, Trajectory out), not a production planner: no lane changes, no
yielding rules, no following of moving traffic.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, Optional

import numpy as np

from doraware.interfaces import (Gear, PerceivedObjects, PlanningRequest, PlanningResult, PlanningStatus,
                                 Trajectory, VehicleState, to_ns, to_sec)
from doraware.planning.trajectory_generation.generator import TrajectoryGenerator
from doraware.planning.trajectory_planning.plugins.lane_follow.route import DensePath, RouteError, route_centerline

DEFAULTS = {"stop_margin": 2.0, "corridor_margin": 0.4, "prediction_time": 1.0, "clear_time": 1.0,
            "comfort_deceleration": 1.0, "lookahead": 12.0}


class LaneFollow:
    inputs = {"objects": PerceivedObjects}

    def __init__(self, params: Dict[str, Any], cfg: Dict[str, Any]):
        self.p = {**DEFAULTS, **{k: float(v) for k, v in params.items()}}
        unknown = set(params) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"lane_follow: unknown parameters {sorted(unknown)}")
        self.timing = cfg["timing"]
        self.gen_cfg = dict(cfg["planning"]["trajectory_generation"])
        self.objects: Optional[PerceivedObjects] = None
        self.path: Optional[DensePath] = None
        self.request: Optional[PlanningRequest] = None
        self.gen: Optional[TrajectoryGenerator] = None
        self.mode = "idle"                    # idle | drive | stop
        self.stop = None                      # (t0 [s], s0, v0, a, s_stop)
        self.clear_since: Optional[float] = None
        self.i_now = 0

    # ---------------------------------------------------------------- inputs
    def on_input(self, name: str, msg: Any) -> None:
        self.objects = msg

    def plan(self, request: PlanningRequest, state: VehicleState, t_ns: int) -> Optional[PlanningResult]:
        self.request, self.mode = request, "idle"
        try:
            self.path = route_centerline(request.lanes, request.route, request.start, request.goal)
        except RouteError as exc:
            self.path = None
            return PlanningResult(request_id=request.request_id, status=PlanningStatus.INVALID_INPUT,
                                  message=str(exc), goal=request.goal)
        self.gen_cfg["max_forward_speed"] = min(float(self.gen_cfg["max_forward_speed"]), self.path.speed_limit)
        self.i_now = 0
        self._drive_from(0.0, state, t_ns)
        details = {"path_length": float(self.path.s[-1]), "speed_limit": self.path.speed_limit}
        return PlanningResult(request_id=request.request_id, status=PlanningStatus.SUCCESS, goal=request.goal,
                              path=self.path.geometric(0, len(self.path.s) - 1), details=json.dumps(details))

    # ------------------------------------------------------------- per tick
    def trajectory_at(self, state: VehicleState, t_ns: int) -> Trajectory:
        if self.path is None or self.gen is None:
            return self._empty(t_ns)
        t = to_sec(t_ns)
        x, y, _ = state.pose.xyyaw()
        lo, hi = max(self.i_now - 8, 0), min(self.i_now + 80, len(self.path.s))
        self.i_now = self.path.project(x, y, lo, hi)
        s_now = self.path.project_s(x, y, lo, hi)
        v = abs(float(state.speed))
        gap = self._free_distance(s_now, v)
        if self.mode == "drive":
            # keep driving until braking at comfort_deceleration (plus one planner period of
            # travel) would just stop stop_margin before the object
            tick = 1.0 / float(self.timing["planner_hz"])
            braking = v * v / (2.0 * self.p["comfort_deceleration"]) + v * tick
            if gap is not None and gap - self.p["stop_margin"] <= braking:
                self._start_stop(t, s_now, v, gap)
            else:
                return self.gen.trajectory_at(t_ns) or self._empty(t_ns)
        # stop mode: brake / wait until the corridor has been clear for clear_time, then continue
        if gap is None:
            self.clear_since = t if self.clear_since is None else self.clear_since
            if t - self.clear_since >= self.p["clear_time"] and v < 0.05:
                self._drive_from(s_now, state, t_ns)
                return self.gen.trajectory_at(t_ns) or self._empty(t_ns)
        else:
            self.clear_since = None
        return self._stop_window(t_ns)

    def diagnostics(self) -> Dict[str, Any]:
        # no plan timing: stops for objects make the end time unknown in advance, so the
        # evaluator waits until the trajectory comes to rest at the goal
        return {} if self.gen is None else {"trajectory_ok": self.gen.ok, "message": self.gen.message}

    # --------------------------------------------------------------- helpers
    def _drive_from(self, s0: float, state: VehicleState, t_ns: int) -> None:
        """Drive the rest of the path from arc length s0 (vehicle at rest there)."""
        self.mode, self.stop, self.clear_since = "drive", None, None
        self.gen = TrajectoryGenerator(self.gen_cfg, self.timing, self.request.vehicle)
        rest_path = self.path.cut(s0, float(self.path.s[-1]))
        rest = PlanningResult(status=PlanningStatus.SUCCESS, path=rest_path.geometric(0, len(rest_path.s) - 1))
        self.gen.on_planning_result(rest, None, plan_start_ns=t_ns, initial_steering=float(state.steering_angle))

    def _free_distance(self, s_now: float, v: float) -> Optional[float]:
        """Distance from the vehicle front to the nearest object blocking the path
        ahead (within the lookahead), or None if the corridor is clear."""
        o = self.objects
        if o is None or len(o) == 0:
            return None
        vp = self.request.vehicle
        half = vp.vehicle_width / 2.0
        s_front = s_now + vp.wheel_base + vp.front_overhang
        reach = s_front + v * v / (2.0 * self.p["comfort_deceleration"]) + self.p["stop_margin"] + \
            self.p["lookahead"]
        lo = self.i_now
        hi = int(np.searchsorted(self.path.s, min(reach, self.path.s[-1] + 5.0)))
        if hi <= lo + 1:
            return None
        best = None
        for k in range(len(o)):
            for dt in (0.0, self.p["prediction_time"]):
                ox, oy = o.x[k] + o.velocity_x[k] * dt, o.y[k] + o.velocity_y[k] * dt
                i = self.path.project(ox, oy, lo, hi)
                rel = float(o.yaw[k] - self.path.yaw[i])
                across = abs(math.sin(rel)) * o.length[k] / 2 + abs(math.cos(rel)) * o.width[k] / 2
                along = abs(math.cos(rel)) * o.length[k] / 2 + abs(math.sin(rel)) * o.width[k] / 2
                nx, ny = -math.sin(self.path.yaw[i]), math.cos(self.path.yaw[i])
                lateral = abs((ox - self.path.x[i]) * nx + (oy - self.path.y[i]) * ny)
                if lateral > half + across + self.p["corridor_margin"]:
                    continue
                gap = float(self.path.s[i]) - along - s_front
                if gap > -vp.vehicle_length:            # ahead of (or next to) the vehicle front
                    best = gap if best is None else min(best, gap)
        return best

    def _start_stop(self, t: float, s_now: float, v: float, gap: float) -> None:
        dist = max(gap - self.p["stop_margin"], 0.05)
        a = min(max(self.p["comfort_deceleration"], v * v / (2.0 * dist)), self.request.vehicle.max_deceleration)
        self.mode, self.clear_since = "stop", None
        self.stop = (t, s_now, v, a, s_now + v * v / (2.0 * a))

    def _stop_window(self, t_ns: int) -> Trajectory:
        t0, s0, v0, a, s_stop = self.stop
        dt = float(self.timing["trajectory_time_step"])
        n = int(round(float(self.timing["trajectory_horizon"]) / dt)) + 1
        tau = to_sec(t_ns) - t0 + np.arange(n) * dt
        t_end = v0 / a if a > 0 else 0.0
        tc = np.minimum(tau, t_end)
        s = s0 + v0 * tc - 0.5 * a * tc * tc
        v = np.maximum(v0 - a * tc, 0.0)
        acc = np.where(tau < t_end, -a, 0.0)
        x, y, yaw, k = self.path.interp(np.minimum(s, s_stop))
        wb = self.request.vehicle.wheel_base
        return Trajectory(start_time=t_ns, time_step=dt, valid_until=t_ns + to_ns(float(self.timing["trajectory_validity"])),
                          x=x, y=y, yaw=yaw, longitudinal_velocity=v, longitudinal_acceleration=acc, curvature=k,
                          front_wheel_angle=np.arctan(wb * k), gear=np.full(n, int(Gear.DRIVE), np.int8))

    def _empty(self, t_ns: int) -> Trajectory:
        return Trajectory(start_time=t_ns, time_step=float(self.timing["trajectory_time_step"]),
                          valid_until=t_ns + to_ns(float(self.timing["trajectory_validity"])))
