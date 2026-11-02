"""Planning messages: requests, results, geometric paths and time-based trajectories."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from doraware.interfaces.common import F64Array, Header, I8Array, Pose2D, to_ns
from doraware.interfaces.map import OccupancyGrid
from doraware.interfaces.vehicle import VehicleParameters


class PlanningStatus(IntEnum):
    UNKNOWN = 0
    SUCCESS = 1
    NO_PATH = 2              # search space exhausted / budget used up without a path
    UNREACHABLE = 3          # proven unreachable
    START_IN_COLLISION = 4
    GOAL_IN_COLLISION = 5
    INVALID_INPUT = 6        # malformed request (e.g. invalid vehicle parameters)
    TIMEOUT = 7
    ERROR = 8                # internal planner failure


# ============================================================ planning
@dataclass
class PlanningRequest:
    """``planner_config`` is an opaque planner-specific JSON object string."""
    header: Header = field(default_factory=Header)
    request_id: str = ""
    start: Pose2D = field(default_factory=Pose2D)
    goal: Pose2D = field(default_factory=Pose2D)
    map: OccupancyGrid = field(default_factory=OccupancyGrid)
    vehicle: VehicleParameters = field(default_factory=VehicleParameters)
    planner_config: str = "{}"


@dataclass
class GeometricPath:
    """Planner-agnostic geometric path of base_link. ``gear`` per point is the
    direction of the motion that *arrives* at that point; a gear change happens
    at a cusp (stop point)."""
    x: F64Array = field(default_factory=lambda: np.zeros(0))
    y: F64Array = field(default_factory=lambda: np.zeros(0))
    yaw: F64Array = field(default_factory=lambda: np.zeros(0))
    curvature: F64Array = field(default_factory=lambda: np.zeros(0))
    gear: I8Array = field(default_factory=lambda: np.zeros(0, dtype=np.int8))

    def __len__(self) -> int:
        return int(len(self.x))


@dataclass
class PlanningResult:
    header: Header = field(default_factory=Header)
    request_id: str = ""
    status: PlanningStatus = PlanningStatus.UNKNOWN
    message: str = ""
    planning_time: float = 0.0              # [s, wall]
    path: GeometricPath = field(default_factory=GeometricPath)
    goal: Pose2D = field(default_factory=Pose2D)
    details: str = "{}"                     # planner-specific JSON (diagnostics only)

    @property
    def success(self) -> bool:
        return self.status == PlanningStatus.SUCCESS


# ============================================================== trajectory
@dataclass
class Trajectory:
    """Time-based trajectory sampled at a fixed ``time_step``.

    Point ``i`` describes the desired state at
    ``start_time + i * time_step`` (sim clock, ns / s). All arrays have the
    same length. Pose = base_link in ``header.frame_id``.

    * ``longitudinal_velocity`` is signed (negative = reverse) and
      ``gear`` gives the direction explicitly (also when stopped).
    * ``valid_until``: the trajectory must not be used after this instant.
    * Beyond the last point the trajectory may only be *held* (stay at the
      last pose) if the last point is stationary; otherwise it is undefined.
    """
    header: Header = field(default_factory=Header)
    start_time: int = 0                      # [ns, sim]
    time_step: float = 0.1                   # [s]
    valid_until: int = 0                     # [ns, sim]
    x: F64Array = field(default_factory=lambda: np.zeros(0))
    y: F64Array = field(default_factory=lambda: np.zeros(0))
    yaw: F64Array = field(default_factory=lambda: np.zeros(0))
    longitudinal_velocity: F64Array = field(default_factory=lambda: np.zeros(0))
    longitudinal_acceleration: F64Array = field(default_factory=lambda: np.zeros(0))
    curvature: F64Array = field(default_factory=lambda: np.zeros(0))
    front_wheel_angle: F64Array = field(default_factory=lambda: np.zeros(0))
    gear: I8Array = field(default_factory=lambda: np.zeros(0, dtype=np.int8))

    def __len__(self) -> int:
        return int(len(self.x))

    @property
    def end_time(self) -> int:
        """Time of the last point [ns]."""
        return self.start_time + to_ns(max(len(self) - 1, 0) * self.time_step)

    def time_of(self, i: int) -> int:
        return self.start_time + to_ns(i * self.time_step)
