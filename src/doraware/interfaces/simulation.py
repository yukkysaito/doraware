"""Simulation messages: scene description and simulator status."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from doraware.interfaces.common import F64Array, Header, Pose2D
from doraware.interfaces.vehicle import VehicleParameters


class SimulationState(IntEnum):
    INITIALIZING = 0
    RUNNING = 1
    PAUSED = 2
    STOPPED = 3


# ======================================================== simulation
@dataclass
class SimulationStatus:
    header: Header = field(default_factory=Header)
    state: SimulationState = SimulationState.INITIALIZING
    step_count: int = 0
    physics_dt: float = 0.01
    collision_count: int = 0                # cumulative physics contacts with obstacles
    collision_objects: str = ""             # comma separated names of contacted bodies
    step_wall_time: float = 0.0             # wall time of the last physics chunk [s]
    command_timeouts: int = 0               # cumulative expired-command events
    max_lockstep_wait: float = 0.0          # [s, wall] longest wait for a producer so far (stall indicator)


@dataclass
class SimulationScene:
    """Everything the simulator needs to (re)set a scenario; no planner data."""
    header: Header = field(default_factory=Header)
    scenario_id: str = ""
    map_xmin: float = 0.0
    map_xmax: float = 0.0
    map_ymin: float = 0.0
    map_ymax: float = 0.0
    obstacles_cx: F64Array = field(default_factory=lambda: np.zeros(0))
    obstacles_cy: F64Array = field(default_factory=lambda: np.zeros(0))
    obstacles_yaw: F64Array = field(default_factory=lambda: np.zeros(0))
    obstacles_length: F64Array = field(default_factory=lambda: np.zeros(0))
    obstacles_width: F64Array = field(default_factory=lambda: np.zeros(0))
    obstacle_height: float = 1.5
    initial_pose: Pose2D = field(default_factory=Pose2D)
    goal_pose: Pose2D = field(default_factory=Pose2D)
    vehicle: VehicleParameters = field(default_factory=VehicleParameters)
