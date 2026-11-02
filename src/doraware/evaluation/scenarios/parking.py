"""Parking scenario source: converts the State Lattice planner repository's test
scenarios into the system's generic messages, without modifying them.

This is the only module (besides the planner adapter) that imports the
planner repository. Start, goal, occupancy grid, vehicle geometry, planner
overrides and expectations are taken verbatim from ``scenarios.py``.

Simulator obstacles are generated from the planner's occupancy grid (merged
occupied cells), so the simulated world equals the planner's map exactly.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict
from typing import List, Optional

import numpy as np

from doraware.core.occupancy import grid_to_boxes
from doraware.evaluation.scenarios.bundle import (FEASIBLE, INFEASIBLE, INVALID_INPUT, PARAMETER,  # noqa: F401
                                                  RANDOMIZED, ExpectedBehavior, ScenarioBundle)
from doraware.core.third_party import ensure_planner_importable
from doraware.interfaces import (Header, OccupancyGrid, PlanningRequest, Pose2D, SimulationScene,
                                   VehicleParameters)

def classify(sc) -> ExpectedBehavior:
    e = sc.expectation
    cfg = sc.make_config()
    common = dict(max_direction_changes=e.max_direction_changes, min_direction_changes=e.min_direction_changes,
                  allowed_directions=e.allowed_directions, spec_max_switches=sc.spec_max_switches,
                  rejection_allowed=bool(sc.spec_reject_ok), max_planning_time=e.max_time,
                  position_tolerance=cfg.position_tolerance, yaw_tolerance=cfg.yaw_tolerance, notes=e.notes)
    if sc.category == "random":
        return ExpectedBehavior(RANDOMIZED, None, **common)
    if e.expect_exception:
        return ExpectedBehavior(INVALID_INPUT, False, expected_status="invalid_input", **common)
    if e.expected_status in ("start_in_collision", "goal_in_collision"):
        return ExpectedBehavior(INVALID_INPUT, False, expected_status=e.expected_status, **common)
    if e.success is True:
        return ExpectedBehavior(FEASIBLE, True, **common)
    if e.success is False:
        return ExpectedBehavior(INFEASIBLE, False, expected_status=e.expected_status, **common)
    return ExpectedBehavior(PARAMETER, None, **common)


# ------------------------------------------------------------ grid -> boxes
# ------------------------------------------------------------ conversion
def vehicle_parameters(params: dict, dynamics: dict) -> VehicleParameters:
    """Scenario geometry (planner VehicleInfo kwargs) + configured dynamics.
    No validation here: invalid inputs must reach the planner unchanged."""
    v = VehicleParameters()
    fields = asdict(v)
    for k, val in params.items():
        if k in fields:
            setattr(v, k, val)
    for k, val in (dynamics or {}).items():
        if k in fields and k not in params:
            setattr(v, k, val)
    return v


def _json_default(o):
    if isinstance(o, tuple):
        return list(o)
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def build_bundle(scenario_id: str, seed: int = 42, dynamics: Optional[dict] = None,
                 obstacle_height: float = 1.5) -> ScenarioBundle:
    ensure_planner_importable()
    from scenarios import build_scenario  # planner repository
    from vehicle import InvalidVehicleParameterError, default_vehicle  # planner repository
    from scenarios import vehicle_params_of

    sc = build_scenario(scenario_id, seed=seed)
    grid = sc.grid
    header = Header(stamp=0, frame_id="map")
    valid = True
    try:
        sc.make_vehicle()
    except (InvalidVehicleParameterError, ValueError):
        valid = False
    planner_vehicle = vehicle_parameters(sc.vehicle_params, dynamics or {})
    # The simulated car must be physical: for an invalid-parameter test (G6)
    # only the planner input is corrupted, the simulated vehicle is the default one.
    sim_vehicle = planner_vehicle if valid else vehicle_parameters(vehicle_params_of(default_vehicle()),
                                                                     dynamics or {})
    margin = float((dynamics or {}).get("steering_control_margin", 0.0))
    if margin:
        sim_vehicle = dataclasses.replace(sim_vehicle, max_steer_angle=sim_vehicle.max_steer_angle + margin)
    xmin, xmax, ymin, ymax = grid.extent
    boxes = grid_to_boxes(grid.data, grid.resolution, grid.origin)
    scene = SimulationScene(
        header=header, scenario_id=sc.id, map_xmin=xmin, map_xmax=xmax, map_ymin=ymin, map_ymax=ymax,
        obstacles_cx=(boxes[:, 0] + boxes[:, 2]) / 2, obstacles_cy=(boxes[:, 1] + boxes[:, 3]) / 2,
        obstacles_yaw=np.zeros(len(boxes)), obstacles_length=boxes[:, 2] - boxes[:, 0],
        obstacles_width=boxes[:, 3] - boxes[:, 1], obstacle_height=obstacle_height,
        initial_pose=Pose2D(*map(float, sc.start)), goal_pose=Pose2D(*map(float, sc.goal)), vehicle=sim_vehicle)
    occ = OccupancyGrid(header=header, resolution=grid.resolution, width=grid.width, height=grid.height,
                        origin_x=grid.origin[0], origin_y=grid.origin[1],
                        data=np.ascontiguousarray(grid.data, dtype=np.uint8).reshape(-1))
    request = PlanningRequest(header=header, request_id=f"{sc.id}/0", start=Pose2D(*map(float, sc.start)),
                              goal=Pose2D(*map(float, sc.goal)), map=occ, vehicle=planner_vehicle,
                              planner_config=json.dumps(sc.planner_overrides or {}, default=_json_default))
    expected = classify(sc)
    expected.safety_margin = float((sc.planner_overrides or {}).get("safety_margin", 0.1))   # planner's margin
    return ScenarioBundle(sc.id, sc.name, sc.category, sc.description, scene, request, expected,
                          tags=tuple(sc.tags), difficulty=sc.difficulty, vehicle_name=planner_vehicle.name,
                          planner_vehicle_valid=valid, extra={"meta": dict(sc.meta)})


def fixed_scenario_ids() -> List[str]:
    ensure_planner_importable()
    from scenarios import scenario_ids
    return scenario_ids()


def random_scenario_ids(n: int) -> List[str]:
    return [f"N-{i:03d}" for i in range(n)]


class ParkingScenarios:
    """Scenario source ``parking`` (evaluation/scenarios/registry.py).

    Suites: ``fixed`` (the planner repository's test patterns), ``random``
    (``N-000`` ... seeded random scenes) and ``all``."""

    def suite(self, name: str, n_random: int = 0) -> List[str]:
        if name not in ("fixed", "random", "all"):
            raise ValueError(f"unknown parking suite {name!r} (fixed, random, all)")
        out = fixed_scenario_ids() if name in ("fixed", "all") else []
        if name in ("random", "all") or n_random:
            out += random_scenario_ids(n_random or 100)
        return out

    def build(self, scenario_id: str, seed: int, cfg: dict) -> ScenarioBundle:
        dyn = cfg["vehicle"]
        return build_bundle(scenario_id, seed=seed, dynamics=dyn, obstacle_height=float(dyn.get("vehicle_height", 1.5)))
