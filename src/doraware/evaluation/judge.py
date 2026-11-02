"""Scenario outcome judge: terminal conditions + pass/fail against the
expected behaviour (pure logic, unit-testable, no Dora).

Acceptance (docs/testing.md):
* plan produced → the vehicle must come to rest at the goal:
  position error ≤ max(0.10 m, scenario goal tolerance), yaw ≤ max(2°, …),
  |v| ≤ 0.02 m/s, terminal error w.r.t. the planned path end ≤ 0.10 m / 2°,
  no physics contact, no footprint overlap with the map, no out-of-bounds,
  controller ≥ 20 Hz, and the original planner expectation (gear changes,
  directions) must hold;
* plan rejected → the rejection must be expected (status matches when the
  original test names one) and the vehicle must stay still and collision-free.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from doraware.core.trajectory import STATIONARY_EPS
from doraware.interfaces import Gear, PlanningResult, PlanningStatus, to_sec
from doraware.evaluation.metrics import ClosedLoopMetrics, count_gear_changes

STATUS_NAMES = {  # PlanningStatus -> planner status strings of the original tests
    PlanningStatus.NO_PATH: ("no_path", "max_expansions"),
    PlanningStatus.UNREACHABLE: ("unreachable",),
    PlanningStatus.START_IN_COLLISION: ("start_in_collision",),
    PlanningStatus.GOAL_IN_COLLISION: ("goal_in_collision",),
    PlanningStatus.INVALID_INPUT: ("invalid_input", "invalid_vehicle"),
    PlanningStatus.TIMEOUT: ("timeout",),
    PlanningStatus.ERROR: ("error",),
}


GOAL_REST_RADIUS = 0.5   # [m] an online planner's final stop this close to the goal ends the run


@dataclass
class Verdict:
    done: bool = False
    outcome: str = ""
    passed: bool = False
    reasons: List[str] = field(default_factory=list)


class ScenarioJudge:
    def __init__(self, expected: Dict, eval_cfg: Dict, metrics: ClosedLoopMetrics):
        self.exp = expected
        self.cfg = eval_cfg
        self.m = metrics
        self.plan: Optional[PlanningResult] = None
        self.plan_time: Optional[float] = None      # sim time the plan outcome became known
        self.plan_end_time: Optional[float] = None  # sim time the trajectory reaches the goal
        self.settle_start: Optional[float] = None
        self.verdict = Verdict()
        self.trajectory_failure: Optional[str] = None

    # ------------------------------------------------------------- inputs
    def on_plan(self, res: PlanningResult, t_sim: float) -> None:
        self.plan = res
        self.plan_time = t_sim

    def on_trajectory_failure(self, message: str, t_sim: float) -> None:
        """The path could not be turned into a trajectory (e.g. revalidation):
        the system must stop safely; judged like a rejection that was not expected."""
        self.trajectory_failure = message
        self.plan_time = t_sim

    def on_plan_timing(self, plan_start: float, duration: float) -> None:
        self.plan_end_time = plan_start + duration

    # ------------------------------------------------------------- update
    def update(self, t: float) -> Verdict:
        """Call after each vehicle state; returns the verdict once terminal."""
        if self.verdict.done:
            return self.verdict
        m, c = self.m, self.cfg
        if t >= c["max_sim_time"]:
            return self._finish("timeout", ["absolute sim time cap reached"])
        if m.collided and m.first_collision_time is not None and t >= m.first_collision_time + 0.5:
            return self._finish("collision", [])
        if self.plan is None:
            return self.verdict
        if not self.plan.success or self.trajectory_failure:
            if t >= self.plan_time + c["rejection_hold_time"]:
                return self._finish("rejected", [])
            return self.verdict
        if self.plan_end_time is not None and t > self.plan_end_time + c["timeout_margin"]:
            return self._finish("timeout", ["did not settle at the goal in time"])
        stopped = bool(m.speed) and abs(m.speed[-1]) <= c["final_speed_tolerance"]
        if self._motion_over(t) and stopped:
            if self.settle_start is None:
                self.settle_start = t
            if t - self.settle_start >= c["settle_time"]:
                return self._finish("stopped", [])
        else:
            self.settle_start = None
        return self.verdict

    def _motion_over(self, t: float) -> bool:
        """The planned motion is over: past the announced plan end, or, for planners
        that announce none (online trajectory planners), the newest trajectory has no
        moving point from ``t`` on and comes to rest at the goal (within
        ``GOAL_REST_RADIUS``; the acceptance criteria are checked afterwards). A stop
        elsewhere, e.g. waiting for a pedestrian, is not the end of the run."""
        if self.plan_end_time is not None:
            return t >= self.plan_end_time
        traj = self.m.traj
        if traj is None or len(traj) == 0:
            return False
        i0 = max(int(np.ceil((t - to_sec(traj.start_time)) / traj.time_step - 1e-9)), 0)
        v = np.abs(np.asarray(traj.longitudinal_velocity[i0:]))
        if len(v) == 0 or not np.all(v < STATIONARY_EPS):
            return False
        gx, gy, _ = self.m.goal
        return math.hypot(float(traj.x[-1]) - gx, float(traj.y[-1]) - gy) <= GOAL_REST_RADIUS

    # ------------------------------------------------------------- verdict
    def _finish(self, outcome: str, reasons: List[str]) -> Verdict:
        reasons = list(reasons)
        reasons += self._safety_reasons()
        if self.plan is None:
            reasons.append("no planning result")
        else:
            limit = self.exp.get("max_planning_time")
            if limit is not None and self.plan.planning_time > limit:
                reasons.append(f"planning took {self.plan.planning_time:.1f} s > {limit:g} s")
        if self.plan is None:
            pass
        elif self.plan.success:
            reasons += self._planned_reasons(outcome)
        else:
            reasons += self._rejected_reasons()
        if outcome == "stopped" and not reasons:
            outcome = "goal_reached"
        self.verdict = Verdict(True, outcome, not reasons, reasons)
        return self.verdict

    def _safety_reasons(self) -> List[str]:
        m, r = self.m, []
        if m.pre_existing_collision_only(self.cfg["rejection_max_motion"]):
            return r   # the start pose itself overlaps the map (input scenario); nothing moved
        if m.physics_contacts:
            r.append(f"physics contact x{m.physics_contacts} ({','.join(sorted(m.contact_objects))})")
        if m.geometric_collisions:
            r.append(f"footprint overlapped the map {m.geometric_collisions} times")
        if self.cfg.get("out_of_bounds_check", True) and m.out_of_bounds:
            r.append(f"out of map bounds {m.out_of_bounds} times")
        return r

    def _planned_reasons(self, outcome: str) -> List[str]:
        e, c, r = self.exp, self.cfg, []
        if e.get("plan_expected") is False:
            r.append(f"planner succeeded but failure was expected ({e.get('expected_status')})")
        path = self.plan.path
        gears = np.asarray(path.gear)
        changes = count_gear_changes(gears)
        if e.get("max_direction_changes") is not None and changes > e["max_direction_changes"]:
            r.append(f"{changes} gear changes > {e['max_direction_changes']}")
        if e.get("min_direction_changes") is not None and changes < e["min_direction_changes"]:
            r.append(f"{changes} gear changes < {e['min_direction_changes']}")
        if e.get("allowed_directions") == "forward" and (gears[1:] == int(Gear.REVERSE)).any():
            r.append("used reverse")
        if e.get("allowed_directions") == "reverse" and (gears[1:] == int(Gear.DRIVE)).any():
            r.append("used forward")
        if self.trajectory_failure:
            r.append(f"trajectory generation failed: {self.trajectory_failure[:150]}")
            moved = self.m.max_displacement()
            if moved > c["rejection_max_motion"]:
                r.append(f"vehicle moved {moved:.3f} m without a valid trajectory")
            return r
        if outcome != "stopped":
            r.append(f"run ended with {outcome}")
            return r
        g = self.m.goal_errors()
        pos_tol = max(c["goal_position_tolerance"], float(e.get("position_tolerance") or 0.0))
        yaw_tol = max(c["goal_yaw_tolerance_deg"], math.degrees(float(e.get("yaw_tolerance") or 0.0)))
        if g["final_position_error"] > pos_tol + 1e-9:
            r.append(f"goal position error {g['final_position_error']:.3f} m > {pos_tol:.2f}")
        if g["final_yaw_error_deg"] > yaw_tol + 1e-9:
            r.append(f"goal yaw error {g['final_yaw_error_deg']:.2f} deg > {yaw_tol:.1f}")
        if g["final_speed"] > c["final_speed_tolerance"] + 1e-9:
            r.append(f"final speed {g['final_speed']:.3f} m/s")
        if g.get("path_end_position_error", 0.0) > c["goal_position_tolerance"] + 1e-9:
            r.append(f"terminal tracking error {g['path_end_position_error']:.3f} m")
        if g.get("path_end_yaw_error_deg", 0.0) > c["goal_yaw_tolerance_deg"] + 1e-9:
            r.append(f"terminal yaw tracking error {g['path_end_yaw_error_deg']:.2f} deg")
        s = self.m.summary()
        r += self._constraint_reasons(s)
        if s.get("command_timeouts", 0):
            r.append(f"{s['command_timeouts']} control command timeouts (emergency braking / lockstep broken)")
        if s.get("gear_changes_while_moving", 0):
            r.append(f"{s['gear_changes_while_moving']} gear changes while moving")
        if e.get("allowed_directions") == "forward" and s.get("driven_reverse_distance", 0.0) > 0.05:
            r.append(f"vehicle reversed {s['driven_reverse_distance']:.2f} m (forward only)")
        if e.get("allowed_directions") == "reverse" and s.get("driven_forward_distance", 0.0) > 0.05:
            r.append(f"vehicle drove forward {s['driven_forward_distance']:.2f} m (reverse only)")
        budget_ms = 1e3 / c["min_controller_hz"]
        if s.get("controller_compute_time_p95_ms", 0.0) > budget_ms:
            r.append(f"controller p95 compute {s['controller_compute_time_p95_ms']:.1f} ms > {budget_ms:.0f} ms "
                     f"(not real-time capable at {c['min_controller_hz']:g} Hz)")
        if s.get("controller_frequency", 0.0) < c["min_controller_hz"]:
            r.append(f"controller frequency {s.get('controller_frequency', 0):.1f} Hz")
        return r

    def _constraint_reasons(self, s: Dict) -> List[str]:
        """Issued commands must respect the configured vehicle limits."""
        v, r, eps = self.m.vehicle, [], 1e-6
        checks = [("cmd_max_steering_angle", v.max_steer_angle), ("cmd_max_steering_rate", v.max_steer_rate),
                  ("cmd_max_jerk", v.max_jerk), ("cmd_max_propulsion", v.max_acceleration),
                  ("cmd_max_braking", v.max_deceleration)]
        for key, lim in checks:
            if s.get(key, 0.0) > lim + eps:
                r.append(f"{key} {s[key]:.3f} > limit {lim:.3f}")
        if -s.get("cmd_min_acceleration", 0.0) > max(v.max_acceleration, v.max_deceleration) + eps:
            r.append(f"cmd_min_acceleration {s['cmd_min_acceleration']:.3f} beyond limit")
        return r

    def _rejected_reasons(self) -> List[str]:
        e, c, r = self.exp, self.cfg, []
        st = self.plan.status
        try:
            raw = json.loads(self.plan.details or "{}").get("raw_status")
        except (ValueError, AttributeError):
            raw = None
        names = (raw,) if raw else STATUS_NAMES.get(st, (st.name.lower(),))
        if e.get("plan_expected") is True:
            r.append(f"planning failed ({st.name}: {self.plan.message[:120]})")
        elif e.get("expected_status") and e["expected_status"] not in names:
            r.append(f"expected {e['expected_status']}, got {names[0]}")
        elif e.get("scenario_class") == "randomized" and not e.get("rejection_allowed"):
            r.append(f"random scene not solved ({names[0]}) and rejection is not allowed by the spec")
        elif e.get("scenario_class") == "randomized" and names[0] not in ("no_path", "max_expansions"):
            # upstream spec_check: a rejection counts only as an answer within the search budget
            r.append(f"random scene rejected by {names[0]}, not within the expansion budget")
        if st == PlanningStatus.ERROR:
            r.append(f"planner error: {self.plan.message[:120]}")
        moved = self.m.max_displacement()
        if moved > c["rejection_max_motion"]:
            r.append(f"vehicle moved {moved:.3f} m after a rejection")
        return r
