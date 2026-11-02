"""Closed-loop metrics computed from the *simulator's* vehicle states.

The planned path / trajectory is used only as the reference for tracking
errors; goal accuracy, safety and constraint checks use measured states and
issued commands. Pure Python (no Dora), fed message by message.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from shapely import STRtree, box as shp_box
from shapely.geometry import Polygon

from doraware.core.geometry import angle_diff, footprint_corners, path_errors
from doraware.interfaces import (ControlCommand, ControlDebug, GeometricPath, Gear, OccupancyGrid,
                                   PlanningResult, SimulationStatus, Trajectory, TurnIndicator, VehicleParameters,
                                   VehicleSignalStatus, VehicleState, to_sec)
from doraware.core.trajectory import sample
from doraware.core.occupancy import grid_to_boxes


class ObstacleField:
    """Exact distance from a vehicle footprint to the occupied cells."""

    def __init__(self, grid: OccupancyGrid, vehicle: VehicleParameters):
        boxes = grid_to_boxes(grid.as_2d(), grid.resolution, (grid.origin_x, grid.origin_y))
        self.geoms = [shp_box(*b) for b in boxes]
        self.tree = STRtree(self.geoms) if self.geoms else None
        self.extent = (grid.origin_x, grid.origin_x + grid.width * grid.resolution,
                       grid.origin_y, grid.origin_y + grid.height * grid.resolution)
        v = vehicle
        self.dims = (v.wheel_base + v.front_overhang, v.rear_overhang, v.wheel_tread / 2 + v.left_overhang,
                     v.wheel_tread / 2 + v.right_overhang)

    def footprint(self, x: float, y: float, yaw: float) -> np.ndarray:
        return footprint_corners(x, y, yaw, *self.dims)

    def clearance(self, x: float, y: float, yaw: float, search: float = 3.0) -> float:
        """Distance footprint -> nearest obstacle (0 if overlapping); inf if none within ``search``."""
        if self.tree is None:
            return math.inf
        poly = Polygon(self.footprint(x, y, yaw))
        idx = self.tree.query(poly.buffer(search))
        if len(idx) == 0:
            return math.inf
        return float(min(poly.distance(self.geoms[i]) for i in idx))

    def overlaps(self, x: float, y: float, yaw: float) -> bool:
        if self.tree is None:
            return False
        poly = Polygon(self.footprint(x, y, yaw))
        return any(poly.intersection(self.geoms[i]).area > 1e-6 for i in self.tree.query(poly))

    def out_of_bounds(self, x: float, y: float, yaw: float) -> bool:
        c = self.footprint(x, y, yaw)
        xmin, xmax, ymin, ymax = self.extent
        return bool((c[:, 0] < xmin).any() or (c[:, 0] > xmax).any() or (c[:, 1] < ymin).any()
                    or (c[:, 1] > ymax).any())


@dataclass
class _Stats:
    values: List[float] = field(default_factory=list)

    def add(self, v: float) -> None:
        if v is not None and math.isfinite(v):
            self.values.append(float(v))

    def summary(self, prefix: str) -> Dict[str, float]:
        a = np.abs(np.asarray(self.values)) if self.values else np.zeros(1)
        return {f"{prefix}_max": float(a.max()), f"{prefix}_rms": float(np.sqrt(np.mean(a ** 2))),
                f"{prefix}_mean": float(a.mean())}


def count_gear_changes(gears: np.ndarray) -> int:
    g = np.asarray(gears)
    return int(np.count_nonzero(g[1:] != g[:-1])) if len(g) > 1 else 0


class SignalStats:
    """Turn indicator / hazard light usage from VehicleSignalStatus messages.
    Durations hold each status until the next one (sim time)."""

    def __init__(self):
        self.n = 0
        self.last: Optional[VehicleSignalStatus] = None
        self.time = {"hazard": 0.0, "left": 0.0, "right": 0.0}
        self.activations = {"hazard": 0, "left": 0, "right": 0}
        self.first_hazard_time: Optional[float] = None
        self.lamp_inconsistencies = 0   # a lamp lit although no function requests that side

    def add(self, s: VehicleSignalStatus) -> None:
        t = to_sec(s.header.stamp)
        on = {"hazard": s.hazard_lights, "left": s.turn_indicator == TurnIndicator.LEFT,
              "right": s.turn_indicator == TurnIndicator.RIGHT}
        if self.last is not None:
            prev = {"hazard": self.last.hazard_lights, "left": self.last.turn_indicator == TurnIndicator.LEFT,
                    "right": self.last.turn_indicator == TurnIndicator.RIGHT}
            dt = t - to_sec(self.last.header.stamp)
            for k in prev:
                self.time[k] += dt if prev[k] else 0.0
                self.activations[k] += int(on[k] and not prev[k])
        else:
            for k in on:
                self.activations[k] += int(on[k])
        if s.hazard_lights and self.first_hazard_time is None:
            self.first_hazard_time = t
        self.lamp_inconsistencies += int((s.left_lamp_on and not (on["hazard"] or on["left"]))
                                         or (s.right_lamp_on and not (on["hazard"] or on["right"])))
        self.last = s
        self.n += 1

    def summary(self) -> Dict:
        if self.last is None:
            return {}
        return {"signal_hazard_time": self.time["hazard"], "signal_left_time": self.time["left"],
                "signal_right_time": self.time["right"], "signal_hazard_activations": self.activations["hazard"],
                "signal_left_activations": self.activations["left"],
                "signal_right_activations": self.activations["right"],
                "signal_first_hazard_time": self.first_hazard_time,
                "signal_final_turn_indicator": TurnIndicator(self.last.turn_indicator).name,
                "signal_final_hazard_lights": bool(self.last.hazard_lights),
                "signal_lamp_inconsistencies": self.lamp_inconsistencies}


class ClosedLoopMetrics:
    def __init__(self, grid: OccupancyGrid, sim_vehicle: VehicleParameters, goal, safety_margin: float = 0.1,
                 clearance_every: int = 1):
        self.field = ObstacleField(grid, sim_vehicle)
        self.vehicle = sim_vehicle
        self.goal = goal
        self.safety_margin = safety_margin
        self.clearance_every = max(1, clearance_every)
        self.path: Optional[GeometricPath] = None
        self.traj: Optional[Trajectory] = None
        self.planned_min_clearance = math.inf
        # histories
        self.t: List[float] = []
        self.pose: List[tuple] = []
        self.speed: List[float] = []
        self.accel: List[float] = []
        self.steer: List[float] = []
        self.gear: List[int] = []
        self.cmd_t: List[float] = []
        self.cmd_acc: List[float] = []
        self.cmd_steer: List[float] = []
        self.cmd_gear: List[int] = []
        # errors
        self.err_lat = _Stats()
        self.err_lon = _Stats()
        self.err_head = _Stats()
        self.err_vel = _Stats()
        self.cte = _Stats()
        self.min_clearance = math.inf
        self.margin_violations = 0
        self.geometric_collisions = 0
        self.out_of_bounds = 0
        self.physics_contacts = 0
        self.contact_objects: set = set()
        self.command_timeouts = 0
        self.solver_ok = 0
        self.solver_total = 0
        self.solve_times: List[float] = []
        self.compute_times: List[float] = []
        self.cmd_propulsion: List[float] = []
        self.cmd_braking: List[float] = []
        self.modes: Dict[str, int] = {}
        self.first_collision_time: Optional[float] = None
        self._n_states = 0
        self.initial_overlap: Optional[bool] = None   # the scenario itself starts in collision (e.g. G4)
        self.signals = SignalStats()

    # ------------------------------------------------------------- inputs
    def on_plan(self, result: PlanningResult) -> None:
        if result.success and len(result.path):
            self.path = result.path
            p = result.path
            step = max(1, len(p) // 400)
            idx = list(range(0, len(p), step)) + [len(p) - 1]
            self.planned_min_clearance = min(self.field.clearance(p.x[i], p.y[i], p.yaw[i]) for i in idx)

    def on_trajectory(self, traj: Trajectory) -> None:
        if self.traj is None or traj.header.stamp >= self.traj.header.stamp:
            self.traj = traj

    def on_state(self, s: VehicleState) -> None:
        t = to_sec(s.header.stamp)
        x, y, yaw = s.pose.xyyaw()
        self.t.append(t)
        self.pose.append((x, y, yaw))
        self.speed.append(s.twist.linear.x)
        self.accel.append(s.acceleration.x)
        self.steer.append(s.steering_angle)
        self.gear.append(int(s.gear))
        if self.initial_overlap is None:
            self.initial_overlap = self.field.overlaps(x, y, yaw)
        if self._n_states % self.clearance_every == 0 or s.twist.linear.x != 0.0:
            c = self.field.clearance(x, y, yaw)
            self.min_clearance = min(self.min_clearance, c)
            if c < self.safety_margin - 1e-9:
                self.margin_violations += 1
            if c <= 0.0 and self.field.overlaps(x, y, yaw):
                self.geometric_collisions += 1
                if self.first_collision_time is None:
                    self.first_collision_time = t
        if self.field.out_of_bounds(x, y, yaw):
            self.out_of_bounds += 1
        self._n_states += 1
        if self.traj is not None:
            ref = sample(self.traj, t)
            if ref is not None:
                lon, lat, head = path_errors(x, y, yaw, ref.x, ref.y, ref.yaw)
                self.err_lon.add(lon)
                self.err_lat.add(lat)
                self.err_head.add(head)
                self.err_vel.add(s.twist.linear.x - ref.velocity)
                self.cte.add(self._cross_track(x, y, ref))

    def _cross_track(self, x: float, y: float, ref) -> Optional[float]:
        """Distance to the geometric path near the time reference (same gear)."""
        p = self.path
        if p is None or len(p) < 2:
            return None
        d_ref = np.hypot(p.x - ref.x, p.y - ref.y)
        near = (d_ref < 1.5) & (p.gear == int(ref.gear))
        if not near.any():
            near = d_ref < 1.5
        if not near.any():
            return None
        idx = np.flatnonzero(near)
        return float(np.min(np.hypot(p.x[idx] - x, p.y[idx] - y)))

    def on_command(self, c: ControlCommand) -> None:
        self.cmd_t.append(to_sec(c.header.stamp))
        self.cmd_acc.append(c.target_acceleration)
        self.cmd_steer.append(c.target_steering_angle)
        self.cmd_gear.append(int(c.gear_request))
        # sign relative to the engaged gear: + propels in the gear's direction, - brakes
        d = Gear(self.gear[-1]).direction if self.gear else 0
        if d:
            (self.cmd_propulsion if c.target_acceleration * d >= 0 else self.cmd_braking).append(
                abs(c.target_acceleration))

    def on_debug(self, d: ControlDebug) -> None:
        self.modes[d.mode] = self.modes.get(d.mode, 0) + 1
        if d.compute_time > 0:
            self.compute_times.append(d.compute_time)
        if d.solver_status in ("", "no_valid_trajectory"):
            return   # no QP was solved in this step (hold / gear change / fallback without trajectory)
        self.solver_total += 1
        if d.solver_status in ("solved", "solved inaccurate", "solved_inaccurate"):
            self.solver_ok += 1
        if d.solve_time > 0:
            self.solve_times.append(d.solve_time)

    def on_sim_status(self, s: SimulationStatus) -> None:
        if s.collision_count > self.physics_contacts and self.first_collision_time is None:
            self.first_collision_time = to_sec(s.header.stamp)
        self.physics_contacts = max(self.physics_contacts, s.collision_count)
        if s.collision_objects:
            self.contact_objects.update(x for x in s.collision_objects.split(",") if x)
        self.command_timeouts = max(self.command_timeouts, s.command_timeouts)

    def on_signal_status(self, s: VehicleSignalStatus) -> None:
        self.signals.add(s)

    # ------------------------------------------------------------- queries
    def pre_existing_collision_only(self, max_motion: float = 0.01) -> bool:
        """Contacts/overlap exist only because the scenario starts in collision
        and the vehicle never moved: not caused by the system."""
        return bool(self.initial_overlap) and self.max_displacement() <= max_motion

    @property
    def collided(self) -> bool:
        if self.pre_existing_collision_only():
            return False
        return self.physics_contacts > 0 or self.geometric_collisions > 0

    def displacement_since(self, t0: float) -> float:
        if not self.pose:
            return 0.0
        i0 = int(np.searchsorted(self.t, t0))
        if i0 >= len(self.pose):
            return 0.0
        p = np.asarray(self.pose[i0:])
        return float(np.max(np.hypot(p[:, 0] - p[0, 0], p[:, 1] - p[0, 1])))

    def max_displacement(self) -> float:
        return self.displacement_since(-1.0)

    def goal_errors(self) -> Dict[str, float]:
        if not self.pose:
            return {}
        x, y, yaw = self.pose[-1]
        gx, gy, gyaw = self.goal
        out = {"final_position_error": math.hypot(x - gx, y - gy),
               "final_yaw_error_deg": abs(math.degrees(angle_diff(yaw, gyaw))),
               "final_speed": abs(self.speed[-1])}
        if self.path is not None and len(self.path):
            px, py, pyaw = self.path.x[-1], self.path.y[-1], self.path.yaw[-1]
            out["path_end_position_error"] = math.hypot(x - px, y - py)
            out["path_end_yaw_error_deg"] = abs(math.degrees(angle_diff(yaw, pyaw)))
            # error of the plan itself (planner tolerance), to separate it from tracking error
            out["planned_goal_position_error"] = math.hypot(px - gx, py - gy)
            out["planned_goal_yaw_error_deg"] = abs(math.degrees(angle_diff(pyaw, gyaw)))
        return out

    def summary(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        out.update(self.goal_errors())
        out.update(self.err_lat.summary("lateral_error"))
        out.update(self.err_lon.summary("longitudinal_error"))
        out.update(self.err_head.summary("heading_error_rad"))
        out.update(self.err_vel.summary("velocity_error"))
        out.update(self.cte.summary("cross_track_error"))
        t = np.asarray(self.t)
        if len(t) > 2:
            dt = np.diff(t)
            acc = np.asarray(self.accel)
            st = np.asarray(self.steer)
            out["max_acceleration"] = float(acc.max())
            out["max_deceleration"] = float(-acc.min())
            out["max_abs_acceleration"] = float(np.abs(acc).max())
            out["max_jerk"] = float(np.max(np.abs(np.diff(acc) / np.maximum(dt, 1e-9))))
            out["max_steering_angle"] = float(np.abs(st).max())
            out["max_steering_rate"] = float(np.max(np.abs(np.diff(st) / np.maximum(dt, 1e-9))))
            out["max_speed"] = float(np.abs(self.speed).max())
            out["gear_changes"] = count_gear_changes(np.asarray(self.gear))
            moving = np.abs(np.asarray(self.speed)) > 0.05
            out["driven_distance"] = float(np.sum(np.hypot(np.diff(np.asarray(self.pose)[:, 0]),
                                                           np.diff(np.asarray(self.pose)[:, 1]))))
            # a gear change while moving is a defect (should be impossible in the simulator)
            g = np.asarray(self.gear)
            out["gear_changes_while_moving"] = int(np.count_nonzero((g[1:] != g[:-1]) & moving[1:]))
            ds = np.hypot(np.diff(np.asarray(self.pose)[:, 0]), np.diff(np.asarray(self.pose)[:, 1]))
            sp = np.asarray(self.speed)[1:]
            out["driven_forward_distance"] = float(ds[sp > 0.02].sum())
            out["driven_reverse_distance"] = float(ds[sp < -0.02].sum())
        ct = np.asarray(self.cmd_t)
        if len(ct) > 2:
            ca, cs = np.asarray(self.cmd_acc), np.asarray(self.cmd_steer)
            dct = np.maximum(np.diff(ct), 1e-9)
            out["cmd_max_acceleration"] = float(ca.max())
            out["cmd_min_acceleration"] = float(ca.min())
            out["cmd_max_jerk"] = float(np.max(np.abs(np.diff(ca)) / dct))
            out["cmd_max_steering_angle"] = float(np.abs(cs).max())
            out["cmd_max_steering_rate"] = float(np.max(np.abs(np.diff(cs)) / dct))
            span = ct[-1] - ct[0]
            out["controller_frequency"] = float((len(ct) - 1) / span) if span > 0 else 0.0
        out["min_clearance_actual"] = self.min_clearance
        out["min_clearance_planned"] = self.planned_min_clearance
        out["safety_margin_violations"] = self.margin_violations
        out["geometric_collisions"] = self.geometric_collisions
        out["physics_contacts"] = self.physics_contacts
        out["collision_count"] = int(self.collided)
        out["initial_overlap"] = bool(self.initial_overlap)
        out["contact_objects"] = ",".join(sorted(self.contact_objects))
        out["out_of_bounds_events"] = self.out_of_bounds
        out["command_timeouts"] = self.command_timeouts
        out["mpc_solver_success_rate"] = self.solver_ok / self.solver_total if self.solver_total else float("nan")
        st = np.asarray(self.solve_times) if self.solve_times else np.zeros(1)
        out["mpc_solve_time_mean_ms"] = float(st.mean() * 1e3)
        out["mpc_solve_time_p95_ms"] = float(np.percentile(st, 95) * 1e3)
        out["mpc_solve_time_max_ms"] = float(st.max() * 1e3)
        ct = np.asarray(self.compute_times) if self.compute_times else np.zeros(1)
        out["controller_compute_time_p95_ms"] = float(np.percentile(ct, 95) * 1e3)
        out["controller_compute_time_max_ms"] = float(ct.max() * 1e3)
        out["cmd_max_propulsion"] = max(self.cmd_propulsion, default=0.0)
        out["cmd_max_braking"] = max(self.cmd_braking, default=0.0)
        out["controller_modes"] = dict(self.modes)
        out["sim_duration"] = float(self.t[-1]) if self.t else 0.0
        out.update(self.signals.summary())
        return out
