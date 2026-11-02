"""Planning limits of the trajectory generator (from ``configs/default.yaml``)."""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Mapping, Optional

from doraware.interfaces import VehicleParameters


@dataclass(frozen=True)
class ProfileLimits:
    """All values positive, SI units. Kinematic limits are the planning limits
    of ``trajectory_generator`` clipped to the vehicle's actuator limits."""
    max_forward_speed: float = 1.5          # [m/s]
    max_reverse_speed: float = 1.0          # [m/s]
    max_acceleration: float = 0.6           # [m/s^2] speeding up (|v| increasing)
    max_deceleration: float = 0.6           # [m/s^2] slowing down (|v| decreasing)
    max_jerk: float = 1.0                   # [m/s^3]
    max_lateral_acceleration: float = 0.6   # [m/s^2]  v^2 |kappa|
    max_steer_rate: float = 0.35            # [rad/s]  |d delta / dt|
    steer_transition_length: float = 0.6    # [m] width w of the curvature ramp (box filter)
    stop_duration: float = 1.0              # [s] minimum standstill at a gear change
    start_stop_duration: float = 0.5        # [s] standstill before the first motion (gear engage)
    path_resample_ds: float = 0.05          # [m] speed-profile grid spacing
    revalidation_margin: float = 0.0        # [m] footprint inflation of the collision revalidation

    @staticmethod
    def from_config(cfg: Mapping[str, Any], vehicle: Optional[VehicleParameters] = None) -> "ProfileLimits":
        """``cfg`` = the ``trajectory_generator`` section (unknown keys are ignored)."""
        names = {f.name for f in fields(ProfileLimits)}
        lim = ProfileLimits(**{k: float(v) for k, v in cfg.items() if k in names})
        if vehicle is None:
            return lim
        return ProfileLimits(**{**lim.__dict__,
                                "max_acceleration": min(lim.max_acceleration, vehicle.max_acceleration),
                                "max_deceleration": min(lim.max_deceleration, vehicle.max_deceleration),
                                "max_jerk": min(lim.max_jerk, vehicle.max_jerk),
                                "max_steer_rate": min(lim.max_steer_rate, vehicle.max_steer_rate)})

    def validate(self) -> None:
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name in ("revalidation_margin", "stop_duration", "start_stop_duration",
                          "steer_transition_length"):
                if not v >= 0.0:
                    raise ValueError(f"{f.name} must be >= 0 (got {v})")
            elif not v > 0.0:
                raise ValueError(f"{f.name} must be > 0 (got {v})")
