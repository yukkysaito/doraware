"""Test helper: planner scenario (scenarios.build_scenario) -> common interfaces."""
from __future__ import annotations

import json

import numpy as np

from doraware.interfaces import OccupancyGrid, PlanningRequest, Pose2D, VehicleParameters
from doraware.core.third_party import ensure_planner_importable

ensure_planner_importable()
from scenarios import build_scenario  # noqa: E402

_GEOMETRY = ("wheel_base", "wheel_tread", "front_overhang", "rear_overhang", "left_overhang", "right_overhang",
             "max_steer_angle", "name")


def grid_of(sc_grid) -> OccupancyGrid:
    h, w = sc_grid.data.shape
    return OccupancyGrid(resolution=sc_grid.resolution, width=w, height=h, origin_x=sc_grid.origin[0],
                         origin_y=sc_grid.origin[1], data=np.ascontiguousarray(sc_grid.data, dtype=np.uint8).ravel())


def vehicle_of(params: dict) -> VehicleParameters:
    """Geometry from the scenario's raw VehicleInfo kwargs (no validation: G6 stays invalid)."""
    return VehicleParameters(**{k: params[k] for k in _GEOMETRY if k in params})


def request_of(sid: str, **extra_config) -> PlanningRequest:
    sc = build_scenario(sid)
    cfg = dict(sc.planner_overrides)
    cfg.update(extra_config)
    return PlanningRequest(request_id=sid, start=Pose2D(*sc.start), goal=Pose2D(*sc.goal), map=grid_of(sc.grid),
                           vehicle=vehicle_of(sc.vehicle_params), planner_config=json.dumps(cfg))
