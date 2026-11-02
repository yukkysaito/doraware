"""Control messages: actuation commands and controller debug output."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from doraware.interfaces.common import F64Array, Header
from doraware.interfaces.vehicle import Gear


# ================================================================= control
@dataclass
class ControlCommand:
    """Actuation request. ``target_acceleration`` is the signed longitudinal
    acceleration along base_link x [m/s^2] (in REVERSE, speeding up backwards
    is negative). ``target_steering_angle`` is the equivalent front-wheel
    angle [rad], +left. The command must not be applied after ``valid_until``.
    """
    header: Header = field(default_factory=Header)
    target_acceleration: float = 0.0
    target_steering_angle: float = 0.0
    gear_request: Gear = Gear.NONE
    valid_until: int = 0
    emergency_stop: bool = False


@dataclass
class ControlDebug:
    """Controller internals for visualization / evaluation (not for control)."""
    header: Header = field(default_factory=Header)
    mode: str = ""                          # e.g. "track", "gear_change", "hold", "fallback"
    solver_status: str = ""
    solve_time: float = 0.0                 # [s, wall] QP solve only
    compute_time: float = 0.0               # [s, wall] whole controller step
    iterations: int = 0
    cost: float = 0.0
    reference_time: int = 0                 # [ns, sim] trajectory time used for this step
    lateral_error: float = 0.0
    heading_error: float = 0.0
    velocity_error: float = 0.0
    longitudinal_error: float = 0.0
    predicted_x: F64Array = field(default_factory=lambda: np.zeros(0))
    predicted_y: F64Array = field(default_factory=lambda: np.zeros(0))
    predicted_yaw: F64Array = field(default_factory=lambda: np.zeros(0))
    predicted_velocity: F64Array = field(default_factory=lambda: np.zeros(0))
