"""Controller parameters (parsed once from the system config + vehicle parameters).

Everything the controller needs is collected here so the algorithm modules
never read raw dictionaries. See docs/mpc.md, section "Tuning parameters".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from doraware.interfaces import VehicleParameters


@dataclass
class MpcWeights:
    longitudinal: float = 20.0
    lateral: float = 60.0
    yaw: float = 40.0
    velocity: float = 4.0
    steering: float = 0.2
    acceleration: float = 0.5
    steering_rate: float = 2.0
    jerk: float = 0.5
    terminal_scale: float = 5.0
    velocity_slack: float = 1.0e4


@dataclass
class MpcParams:
    horizon_steps: int = 30
    dt: float = 0.1
    weights: MpcWeights = field(default_factory=MpcWeights)
    max_iterations: int = 4000
    eps_abs: float = 1e-4
    eps_rel: float = 1e-4
    time_limit: float = 0.02
    max_speed: float = 3.0          # [m/s] |v| bound (soft) in either gear
    # vehicle / actuation limits (copied from VehicleParameters)
    wheel_base: float = 2.79
    max_steer_angle: float = 0.55
    max_steer_rate: float = 0.6
    max_acceleration: float = 1.5
    max_deceleration: float = 3.0
    max_jerk: float = 3.0


@dataclass
class ControllerParams:
    mpc: MpcParams = field(default_factory=MpcParams)
    control_period: float = 0.02            # [s] 1 / timing.controller_hz
    command_validity: float = 0.1           # [s]
    gear_change_speed: float = 0.03         # [m/s]
    stop_hold_acceleration: float = 0.8     # [m/s^2]
    fallback_deceleration: float = 2.0      # [m/s^2]
    max_consecutive_failures: int = 3
    delay_compensation: bool = True
    lag_compensation: bool = True           # treat first-order lags tau as extra dead time
    stop_position_tolerance: float = 0.03   # [m] enter standstill hold below this |e_lon|
    stop_speed: float = 0.05                # [m/s] |v| below which hold may be entered
    hold_release_lookahead: float = 0.3     # [s] leave hold this early before the reference moves
    steering_dead_time: float = 0.0         # [s] actuator model used for state prediction
    steering_time_constant: float = 0.0
    acceleration_dead_time: float = 0.0
    acceleration_time_constant: float = 0.0
    solver_failure_after: Optional[float] = None  # [s, sim] fault injection

    @property
    def steering_delay(self) -> float:
        return self.steering_dead_time + (self.steering_time_constant if self.lag_compensation else 0.0)

    @property
    def acceleration_delay(self) -> float:
        return self.acceleration_dead_time + (self.acceleration_time_constant if self.lag_compensation else 0.0)

    @staticmethod
    def from_config(cfg: Dict[str, Any], vehicle: VehicleParameters) -> "ControllerParams":
        """``cfg`` is the full system config (configs/default.yaml layout); the
        controller's own parameters are ``control.controller.mpc``."""
        c = m = cfg["control"]["controller"].get("mpc") or {}
        w = MpcWeights(**{k: float(v) for k, v in (m.get("weights", {}) or {}).items()})
        mpc = MpcParams(
            horizon_steps=int(m.get("horizon_steps", 30)), dt=float(m.get("dt", 0.1)), weights=w,
            max_iterations=int(m.get("max_iterations", 4000)), eps_abs=float(m.get("eps_abs", 1e-4)),
            eps_rel=float(m.get("eps_rel", 1e-4)), time_limit=float(m.get("time_limit", 0.02)),
            max_speed=float(m.get("max_speed", 3.0)),
            wheel_base=float(vehicle.wheel_base), max_steer_angle=float(vehicle.max_steer_angle),
            max_steer_rate=float(vehicle.max_steer_rate), max_acceleration=float(vehicle.max_acceleration),
            max_deceleration=float(vehicle.max_deceleration), max_jerk=float(vehicle.max_jerk))
        timing = cfg.get("timing", {}) or {}
        act = cfg["vehicle"].get("actuator") or {}
        fi = cfg.get("fault_injection", {}) or {}
        sfa = fi.get("solver_failure_after")
        return ControllerParams(
            mpc=mpc,
            control_period=1.0 / float(timing.get("controller_hz", 50.0)),
            command_validity=float(timing.get("command_validity", 0.1)),
            gear_change_speed=float(c.get("gear_change_speed", 0.03)),
            stop_hold_acceleration=float(c.get("stop_hold_acceleration", 0.8)),
            fallback_deceleration=float(c.get("fallback_deceleration", 2.0)),
            max_consecutive_failures=int(c.get("max_consecutive_failures", 3)),
            delay_compensation=bool(c.get("delay_compensation", True)),
            lag_compensation=bool(c.get("lag_compensation", True)),
            stop_position_tolerance=float(c.get("stop_position_tolerance", 0.03)),
            stop_speed=float(c.get("stop_speed", 0.05)),
            hold_release_lookahead=float(c.get("hold_release_lookahead", 0.3)),
            steering_dead_time=float(act.get("steering_dead_time", 0.0) or 0.0),
            steering_time_constant=float(act.get("steering_time_constant", 0.0) or 0.0),
            acceleration_dead_time=float(act.get("acceleration_dead_time", 0.0) or 0.0),
            acceleration_time_constant=float(act.get("acceleration_time_constant", 0.0) or 0.0),
            solver_failure_after=None if sfa is None else float(sfa),
        )
