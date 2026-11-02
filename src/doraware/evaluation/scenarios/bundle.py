"""Scenario bundle: what a scenario source produces for one run (no Dora).

A bundle holds everything a closed-loop run needs: the simulator scene, the
planning request and the expected behaviour that the evaluator judges.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

from doraware.interfaces import PlanningRequest, SimulationScene

# ------------------------------------------------------------ classification
FEASIBLE = "feasible"            # a plan is expected; the vehicle must reach the goal
INFEASIBLE = "infeasible"        # the planner must fail; the vehicle must stay stopped
INVALID_INPUT = "invalid_input"  # malformed input must be detected; vehicle stays stopped
PARAMETER = "parameter"          # sweep / open-ended: no planner expectation (closed loop must be safe)
RANDOMIZED = "randomized"        # seeded random scene with a product spec


@dataclass
class ExpectedBehavior:
    """What the closed-loop run must show (derived from the original Expectation)."""
    scenario_class: str
    plan_expected: Optional[bool]            # True / False / None (don't care)
    expected_status: Optional[str] = None    # planner status string of the original test (e.g. "no_path")
    max_direction_changes: Optional[int] = None
    min_direction_changes: Optional[int] = None
    allowed_directions: Optional[str] = None
    rejection_allowed: bool = False          # random/spec scenes: a fast rejection is acceptable
    spec_max_switches: Optional[int] = None
    max_planning_time: Optional[float] = None
    position_tolerance: float = 0.10         # planner goal tolerance of this scenario [m]
    yaw_tolerance: float = math.radians(2.0)
    notes: str = ""


@dataclass
class ScenarioBundle:
    scenario_id: str
    name: str
    category: str
    description: str
    scene: SimulationScene
    request: PlanningRequest
    expected: ExpectedBehavior
    tags: Tuple[str, ...] = ()
    difficulty: str = ""
    vehicle_name: str = ""
    planner_vehicle_valid: bool = True
    extra: Dict = field(default_factory=dict)
