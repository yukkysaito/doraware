"""TrajectoryFollower: the controller application (no middleware).

Mode priority per step (docs/mpc.md §8-§9):
    fallback     no usable (unexpired, defined) trajectory, or the solver kept failing
    gear_change  the vehicle gear differs from the desired gear (stop -> shift)
    hold         stationary reference reached: brake at stop_hold_acceleration
    track        LTV-MPC (``track_reuse`` = solver failed, shifted previous plan reused)

Every command passes the same output stage: steering-rate and jerk limits per
control period, steering-angle and gear-dependent acceleration saturation.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from doraware.core.geometry import path_errors
from doraware.interfaces import (ControlCommand, ControlDebug, Gear, Header, Trajectory,
                                   VehicleParameters, VehicleState, to_ns, to_sec)
from doraware.core.trajectory import sample
from doraware.control.gear import GearManager, GearPhase
from doraware.control.plugins.mpc.model import IA, IDELTA
from doraware.control.plugins.mpc.controller import LtvMpc, MpcSolution, accel_bounds
from doraware.control.plugins.mpc.params import ControllerParams
from doraware.control.prediction import CommandHistory, predict_pose
from doraware.control.plugins.mpc.reference import build_reference, is_stationary_window, trajectory_usable

STANDSTILL_SPEED = 0.005   # [m/s] below this the car is considered stopped (brake release before shifting)


class TrajectoryFollower:
    def __init__(self, config: Dict[str, Any], vehicle_params: VehicleParameters):
        self.params = ControllerParams.from_config(config, vehicle_params)
        self.vehicle = vehicle_params
        self.mpc = LtvMpc(self.params.mpc)
        self.gear = GearManager(self.params.gear_change_speed)
        self.history = CommandHistory()
        self.trajectory: Optional[Trajectory] = None
        self.prev_delta: Optional[float] = None
        self.prev_accel: float = 0.0
        self.prev_t: Optional[float] = None
        self.failures = 0
        self.holding = False
        self.seq = 0
        self.last_gear: Optional[Gear] = None
        self.last_solution: Optional[MpcSolution] = None

    # ------------------------------------------------------------- input
    def on_trajectory(self, traj: Trajectory) -> bool:
        """Keep the newest trajectory (by header stamp). Returns False if ignored
        (stale / out of order / malformed)."""
        n = len(traj)   # n == 0 is valid: an empty trajectory means "stop" (planning failed)
        if traj.time_step <= 0.0 or any(len(a) != n for a in (
                traj.y, traj.yaw, traj.longitudinal_velocity, traj.longitudinal_acceleration,
                traj.curvature, traj.front_wheel_angle, traj.gear)):
            return False
        if self.trajectory is not None and traj.header.stamp <= self.trajectory.header.stamp:
            return False
        self.trajectory = traj
        return True

    # -------------------------------------------------------------- step
    def step(self, state: VehicleState, t_now_ns: int) -> Tuple[ControlCommand, ControlDebug]:
        p = self.params
        t = to_sec(t_now_ns)
        v = float(state.speed)
        if self.prev_delta is None:
            self.prev_delta = float(state.steering_angle)
        if self.last_gear is not None and state.gear != self.last_gear:
            self.mpc.reset()          # direction changed: the old plan is meaningless
        self.last_gear = state.gear
        dbg = ControlDebug(header=Header(stamp=t_now_ns))
        request = Gear.NONE
        traj = self.trajectory if trajectory_usable(self.trajectory, t) else None

        if traj is None:
            mode, (d_t, a_t) = "fallback", self._fallback_target(state)
            dbg.solver_status = "no_valid_trajectory"
            self.holding = False
        else:
            ref_now = sample(traj, t)
            lon, lat, hdg = path_errors(*state.pose.xyyaw(), ref_now.x, ref_now.y, ref_now.yaw)
            dbg.longitudinal_error, dbg.lateral_error, dbg.heading_error = lon, lat, hdg
            dbg.velocity_error = v - ref_now.velocity
            dbg.reference_time = t_now_ns
            decision = self.gear.decide(traj, t, state.gear, v)
            gdir = state.gear.direction
            if decision.phase == GearPhase.SHIFTING or (decision.phase == GearPhase.STOPPING and gdir == 0):
                mode = "gear_change"
                request = decision.request
                self.holding = False
                self.mpc.reset()
                a_t = -math.copysign(p.stop_hold_acceleration, v) if abs(v) > STANDSTILL_SPEED else 0.0
                d_t = ref_now.steering
            elif decision.phase == GearPhase.ENGAGED and self._hold(traj, t, v, lon, gdir):
                mode = "hold"
                self.mpc.reset()
                a_t = -gdir * p.stop_hold_acceleration if gdir != 0 else 0.0
                d_t = ref_now.steering
            elif decision.phase == GearPhase.ENGAGED and self._release(traj, t, v, lon):
                # the reference departs within hold_release_lookahead: release the
                # brake (jerk-limited) so the MPC starts from a ~ 0 at departure
                mode = "hold_release"
                self.mpc.reset()
                a_t, d_t = 0.0, ref_now.steering
            else:
                # ENGAGED (track) or STOPPING in the old gear: the reference is
                # cusp-clamped so the MPC itself brings the car to rest at the cusp.
                mode, d_t, a_t = self._track(state, traj, t, dbg)
                if decision.phase == GearPhase.STOPPING:
                    mode = "gear_change"

        cmd_delta, cmd_acc = self._limit(d_t, a_t, state.gear.direction, t)
        self.history.push(t, cmd_delta, cmd_acc)
        self.prev_delta, self.prev_accel = cmd_delta, cmd_acc
        self.seq += 1
        dbg.mode = mode
        cmd = ControlCommand(header=Header(stamp=t_now_ns, sequence_id=self.seq),
                             target_acceleration=cmd_acc, target_steering_angle=cmd_delta,
                             gear_request=request, valid_until=t_now_ns + to_ns(p.command_validity))
        return cmd, dbg

    # ------------------------------------------------------------- modes
    def _hold(self, traj: Trajectory, t: float, v: float, lon: float, gdir: int) -> bool:
        """Standstill hold with hysteresis: enter at |e_lon| < tol, leave at > 2 tol."""
        p = self.params
        tol = p.stop_position_tolerance * (2.0 if self.holding else 1.0)
        ok = (abs(v) < p.stop_speed and abs(lon) < tol
              and is_stationary_window(traj, t, t + p.hold_release_lookahead))
        self.holding = ok
        return ok

    def _release(self, traj: Trajectory, t: float, v: float, lon: float) -> bool:
        p = self.params
        ref = sample(traj, t)
        return (self.prev_accel * self.last_gear.direction < 0.0 and abs(v) < p.stop_speed
                and abs(lon) < 2.0 * p.stop_position_tolerance
                and abs(ref.velocity) < 1e-3 and abs(ref.acceleration) < 1e-3)

    def _fallback_target(self, state: VehicleState) -> Tuple[float, float]:
        """Brake at fallback_deceleration (jerk-limited by the output stage), hold the steering."""
        p, v = self.params, float(state.speed)
        if abs(v) > STANDSTILL_SPEED:
            a = -math.copysign(p.fallback_deceleration, v)
        else:
            a = -state.gear.direction * p.stop_hold_acceleration
        self.mpc.reset()
        return self.prev_delta, a

    def _initial_state(self, state: VehicleState, t: float) -> Tuple[np.ndarray, float]:
        p = self.params
        x, y, yaw = state.pose.xyyaw()
        d = max(p.steering_delay, p.acceleration_delay) if p.delay_compensation else 0.0
        xyv = predict_pose(x, y, yaw, float(state.speed), float(state.steering_angle), t, d,
                           p.steering_delay, p.acceleration_delay, self.history, state.gear.direction,
                           p.mpc.wheel_base)
        lo, hi = accel_bounds(state.gear.direction, p.mpc.max_acceleration, p.mpc.max_deceleration)
        a0 = float(np.clip(self.prev_accel, lo, hi))
        if abs(xyv[3]) < STANDSTILL_SPEED and a0 * state.gear.direction < 0.0:
            a0 = 0.0      # braking at standstill produces no motion (plant semantics)
        z0 = np.array([xyv[0], xyv[1], xyv[2], xyv[3],
                       float(np.clip(self.prev_delta, -p.mpc.max_steer_angle, p.mpc.max_steer_angle)), a0])
        return z0, t + d

    def _track(self, state: VehicleState, traj: Trajectory, t: float,
               dbg: ControlDebug) -> Tuple[str, float, float]:
        p = self.params
        z0, t0 = self._initial_state(state, t)
        ref = build_reference(traj, t0, p.mpc.horizon_steps, p.mpc.dt, z0[2], state.gear.direction)
        forced = p.solver_failure_after is not None and t >= p.solver_failure_after
        sol = self.mpc.solve(z0, ref, state.gear.direction, t0, force_failure=forced)
        dbg.solver_status, dbg.iterations, dbg.solve_time = sol.status, sol.iterations, sol.solve_time
        dbg.cost = sol.cost if sol.ok else 0.0
        dbg.reference_time = to_ns(t0)
        T = p.control_period
        if sol.ok:
            self.failures = 0
            self.last_solution = sol
            dbg.predicted_x, dbg.predicted_y = sol.z[:, 0].copy(), sol.z[:, 1].copy()
            dbg.predicted_yaw, dbg.predicted_velocity = sol.z[:, 2].copy(), sol.z[:, 3].copy()
            # command = state of the plan one control period after z_0 (u_0 is constant on [0, dt))
            return "track", z0[IDELTA] + sol.u[0, 0] * T, z0[IA] + sol.u[0, 1] * T
        self.failures += 1
        u_prev = self.mpc.shifted_previous(t0)
        if self.failures <= p.max_consecutive_failures and u_prev is not None:
            return "track_reuse", z0[IDELTA] + u_prev[0] * T, z0[IA] + u_prev[1] * T
        d_t, a_t = self._fallback_target(state)
        return "fallback", d_t, a_t

    # -------------------------------------------------------- output stage
    def _limit(self, delta: float, accel: float, gear_dir: int, t: float) -> Tuple[float, float]:
        p, m = self.params, self.params.mpc
        # rate/jerk limits over the time actually elapsed since the previous command
        # (controller ticks are not equidistant when physics_hz / controller_hz is not an integer)
        T = p.control_period if self.prev_t is None else max(t - self.prev_t, 0.0)
        self.prev_t = t
        dd = m.max_steer_rate * T
        delta = float(np.clip(delta, self.prev_delta - dd, self.prev_delta + dd))
        delta = float(np.clip(delta, -m.max_steer_angle, m.max_steer_angle))
        da = m.max_jerk * T
        accel = float(np.clip(accel, self.prev_accel - da, self.prev_accel + da))
        lo, hi = accel_bounds(gear_dir, m.max_acceleration, m.max_deceleration)
        accel = float(np.clip(accel, lo, hi))
        return delta, accel
