"""Vehicle messages: state, parameters, gear, turn indicators and hazard lights."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from doraware.interfaces.common import Header, Pose, Twist, Vector3


class Gear(IntEnum):
    """Transmission state / request. Direction of travel is explicit here
    because a signed velocity alone is ambiguous at standstill."""
    NONE = 0     # request: keep current gear
    PARK = 1
    REVERSE = 2
    NEUTRAL = 3
    DRIVE = 4

    @property
    def direction(self) -> int:
        """+1 forward, -1 reverse, 0 none."""
        return 1 if self == Gear.DRIVE else -1 if self == Gear.REVERSE else 0

    @staticmethod
    def from_direction(direction: int) -> "Gear":
        return Gear.DRIVE if direction > 0 else Gear.REVERSE if direction < 0 else Gear.PARK


# ============================================================ vehicle state
@dataclass
class VehicleState:
    """Measured vehicle state.

    * ``pose``: base_link (rear axle center) in ``header.frame_id`` (map).
    * ``twist``: velocity of the base_link origin expressed in base_link
      (linear.x = signed longitudinal speed, linear.y = lateral speed,
      angular.z = yaw rate).
    * ``acceleration``: acceleration of the base_link origin in base_link.
    * ``steering_angle``: equivalent front-wheel (bicycle) angle [rad], +left.
    """
    header: Header = field(default_factory=Header)
    pose: Pose = field(default_factory=Pose)
    twist: Twist = field(default_factory=Twist)
    acceleration: Vector3 = field(default_factory=Vector3)
    steering_angle: float = 0.0
    gear: Gear = Gear.PARK

    @property
    def speed(self) -> float:
        return self.twist.linear.x


# ======================================================= vehicle parameters
@dataclass
class VehicleParameters:
    """Geometry (Autoware VehicleInfo-compatible) + dynamics + actuation limits."""
    name: str = "default"
    wheel_base: float = 2.79
    wheel_tread: float = 1.60
    front_overhang: float = 0.90
    rear_overhang: float = 1.00
    left_overhang: float = 0.15
    right_overhang: float = 0.15
    vehicle_height: float = 1.5
    max_steer_angle: float = 0.55           # front-wheel equivalent [rad]
    max_steer_rate: float = 0.6             # [rad/s]
    max_acceleration: float = 1.5           # [m/s^2]
    max_deceleration: float = 3.0           # [m/s^2] (positive number)
    max_jerk: float = 3.0                   # [m/s^3]
    mass: float = 1500.0                    # [kg]
    cog_x: float = 1.30                     # CoG ahead of the rear axle [m]
    cog_height: float = 0.55                # [m]
    yaw_inertia: float = 2500.0             # [kg m^2]
    tire_friction: float = 0.9              # mu
    cornering_stiffness: float = 60000.0    # per tire [N/rad]

    @property
    def vehicle_length(self) -> float:
        return self.front_overhang + self.wheel_base + self.rear_overhang

    @property
    def vehicle_width(self) -> float:
        return self.wheel_tread + self.left_overhang + self.right_overhang

    @property
    def footprint_center(self) -> tuple:
        """Footprint rectangle center in base_link."""
        return ((self.wheel_base + self.front_overhang - self.rear_overhang) / 2.0,
                (self.left_overhang - self.right_overhang) / 2.0)


# ========================================================= vehicle signals
class TurnIndicator(IntEnum):
    """Turn indicator function. A value is a complete state, never a delta."""
    OFF = 0
    LEFT = 1
    RIGHT = 2


@dataclass
class VehicleSignalCommand:
    """Requested signalling state of the vehicle: turn indicators and hazard lights.

    * **Complete state.** Every command carries the full desired state (there is
      no "keep" value), so a command is idempotent and the newest one wins.
    * **Latched.** The vehicle keeps the last command; lamps do not switch off
      when commands stop arriving. Publishers send at a fixed rate (sim clock)
      so late subscribers and recordings always see the current request.
    * **Independent requests.** ``turn_indicator`` and ``hazard_lights`` are
      requested separately. On the lamps, hazard lights take priority (both
      sides flash); the requested turn indicator resumes when they go off.
    * Signals never influence vehicle motion. Flashing is done by the vehicle,
      not by the publisher.
    """
    header: Header = field(default_factory=Header)
    turn_indicator: TurnIndicator = TurnIndicator.OFF
    hazard_lights: bool = False


@dataclass
class VehicleSignalStatus:
    """Signalling state reported by the vehicle at ``header.stamp``.

    * ``turn_indicator`` / ``hazard_lights``: the functions the vehicle is
      executing (the last command it accepted), independent of the flash phase.
    * ``left_lamp_on`` / ``right_lamp_on``: the instantaneous lamp outputs, i.e.
      the flash phase of the active function (both follow the hazard lights
      when they are on).
    """
    header: Header = field(default_factory=Header)
    turn_indicator: TurnIndicator = TurnIndicator.OFF
    hazard_lights: bool = False
    left_lamp_on: bool = False
    right_lamp_on: bool = False
