"""Adapter: common :class:`PlanningRequest` -> State Lattice planner -> :class:`PlanningResult`.

This is the only module that knows the planner's data types. It

* converts the interfaces into planner types (grid ``data[iy*w+ix]`` ->
  planner ``data[y, x]``, :class:`VehicleParameters` geometry -> ``VehicleInfo``,
  ``planner_config`` JSON -> ``PlannerConfig().with_changes(**overrides)``),
* runs the selected planner (``"planner": "bidirectional" | "forward"`` in the
  config JSON, default bidirectional),
* maps the planner status to :class:`PlanningStatus`, and
* converts the planner trajectory into a planner-agnostic
  :class:`GeometricPath` (x, y, wrapped yaw, curvature, gear).

Search statistics (expanded nodes, which search won, number of direction
changes, raw status string) go only into ``PlanningResult.details`` (JSON).
:meth:`StateLatticeAdapter.plan` never raises.
"""
from __future__ import annotations

import json
import math
import time
from collections import OrderedDict
from typing import Any, Dict, Tuple

import numpy as np

from doraware.core.geometry import wrap_angle, wrap_angles
from doraware.interfaces import (GeometricPath, Gear, Header, OccupancyGrid, PlanningRequest, PlanningResult,
                                   PlanningStatus, Pose2D, VehicleParameters)
from doraware.core.third_party import ensure_planner_importable

ensure_planner_importable()
from occupancy_grid import OccupancyGrid as PlannerGrid  # noqa: E402
from parking_planner import PLANNERS  # noqa: E402
from state_lattice import PlannerConfig, PlanStatus  # noqa: E402
from vehicle import InvalidVehicleParameterError, VehicleInfo  # noqa: E402

DEFAULT_PLANNER = "bidirectional"
PLANNER_KEY = "planner"          # reserved key of the planner_config JSON (planner selection)

_STATUS_MAP = {
    PlanStatus.SUCCESS: PlanningStatus.SUCCESS,
    PlanStatus.NO_PATH: PlanningStatus.NO_PATH,
    PlanStatus.MAX_EXPANSIONS: PlanningStatus.NO_PATH,
    PlanStatus.UNREACHABLE: PlanningStatus.UNREACHABLE,
    PlanStatus.TIMEOUT: PlanningStatus.TIMEOUT,
    PlanStatus.START_IN_COLLISION: PlanningStatus.START_IN_COLLISION,
    PlanStatus.GOAL_IN_COLLISION: PlanningStatus.GOAL_IN_COLLISION,
    PlanStatus.INVALID_INPUT: PlanningStatus.INVALID_INPUT,
    PlanStatus.ERROR: PlanningStatus.ERROR,
}


class InvalidRequestError(ValueError):
    """The request cannot be converted into a planner problem."""


# ------------------------------------------------------------ conversions
def to_planner_grid(grid: OccupancyGrid) -> PlannerGrid:
    """Row-major ``data[iy * width + ix]`` -> planner ``data[y, x]`` (same cells, same origin)."""
    data = np.asarray(grid.data)
    if grid.width <= 0 or grid.height <= 0 or data.size != grid.width * grid.height:
        raise InvalidRequestError(f"occupancy grid size mismatch: {grid.width}x{grid.height} vs {data.size} cells")
    if not (math.isfinite(grid.resolution) and grid.resolution > 0):
        raise InvalidRequestError(f"invalid grid resolution {grid.resolution!r}")
    return PlannerGrid(data.reshape(grid.height, grid.width), grid.resolution, (grid.origin_x, grid.origin_y))


def to_vehicle_info(vp: VehicleParameters) -> VehicleInfo:
    """Geometry part of :class:`VehicleParameters` (raises InvalidVehicleParameterError)."""
    return VehicleInfo(wheel_base=vp.wheel_base, wheel_tread=vp.wheel_tread, front_overhang=vp.front_overhang,
                       rear_overhang=vp.rear_overhang, left_overhang=vp.left_overhang,
                       right_overhang=vp.right_overhang, max_steer_angle=vp.max_steer_angle, name=vp.name)


def parse_planner_config(text: str) -> Tuple[str, PlannerConfig]:
    """``planner_config`` JSON -> (planner name, PlannerConfig).

    JSON has no tuples: list values are converted back to tuples (the
    tuple-typed fields such as ``length_classes`` must stay hashable, they are
    part of the planner's cache keys). Non-finite floats (``NaN``,
    ``Infinity``) are accepted (Python's json default).
    """
    try:
        overrides = json.loads(text) if text and text.strip() else {}
    except json.JSONDecodeError as exc:
        raise InvalidRequestError(f"planner_config is not valid JSON: {exc}") from exc
    if not isinstance(overrides, dict):
        raise InvalidRequestError("planner_config must be a JSON object")
    name = overrides.pop(PLANNER_KEY, DEFAULT_PLANNER)
    if name not in PLANNERS:
        raise InvalidRequestError(f"unknown planner {name!r} (available: {sorted(PLANNERS)})")
    overrides = {k: _tuplify(v) for k, v in overrides.items()}
    try:
        cfg = PlannerConfig().with_changes(**overrides)
        cfg.validate()
    except (TypeError, ValueError) as exc:
        raise InvalidRequestError(f"invalid planner_config: {exc}") from exc
    return name, cfg


def _tuplify(v: Any) -> Any:
    return tuple(_tuplify(x) for x in v) if isinstance(v, list) else v


def to_geometric_path(traj) -> GeometricPath:
    """Planner Trajectory -> GeometricPath. Per point, ``curvature`` and ``gear``
    describe the motion that arrives at the point (planner convention; the
    first point carries the first motion's values). ``curvature`` is the
    steering curvature tan(delta)/L, i.e. yaw rate = signed velocity * curvature."""
    pts = traj.points
    return GeometricPath(
        x=np.array([p.x for p in pts], dtype=np.float64),
        y=np.array([p.y for p in pts], dtype=np.float64),
        yaw=wrap_angles(np.array([p.yaw for p in pts], dtype=np.float64)),
        curvature=np.array([p.curvature for p in pts], dtype=np.float64),
        gear=np.array([int(Gear.from_direction(p.direction)) for p in pts], dtype=np.int8),
    )


# ---------------------------------------------------------------- adapter
class StateLatticeAdapter:
    """Planner-node algorithm (no Dora). Planner instances are cached per
    (planner, vehicle geometry, config) because building one loads/generates
    the primitive set and heuristic tables."""

    def __init__(self, cache_size: int = 4):
        self._cache: "OrderedDict[tuple, Any]" = OrderedDict()
        self._cache_size = cache_size

    def plan(self, request: PlanningRequest) -> PlanningResult:
        t0 = time.perf_counter()
        goal = Pose2D(request.goal.x, request.goal.y,
                      wrap_angle(request.goal.yaw) if math.isfinite(request.goal.yaw) else request.goal.yaw)
        out = PlanningResult(header=Header(stamp=request.header.stamp, frame_id=request.header.frame_id,
                                           sequence_id=request.header.sequence_id, clock=request.header.clock),
                             request_id=request.request_id, goal=goal)
        details: Dict[str, Any] = {}
        try:
            name, cfg = parse_planner_config(request.planner_config)
            details["planner"] = name
            vehicle = to_vehicle_info(request.vehicle)
            grid = to_planner_grid(request.map)
            planner = self._planner(name, vehicle, cfg)
            start = (request.start.x, request.start.y, request.start.yaw)
            res = planner.plan(grid, start, (request.goal.x, request.goal.y, request.goal.yaw))
        except (InvalidVehicleParameterError, InvalidRequestError) as exc:
            return self._finish(out, PlanningStatus.INVALID_INPUT, str(exc), t0, details)
        except Exception as exc:  # the planner must never take the node down
            return self._finish(out, PlanningStatus.ERROR, f"planner failure: {exc!r}", t0, details)

        status = _STATUS_MAP.get(res.status, PlanningStatus.ERROR)
        details.update(raw_status=res.status.value, search=res.search, expanded_nodes=int(res.expanded_nodes),
                       generated_nodes=int(res.generated_nodes), planner_time=float(res.planning_time))
        if status == PlanningStatus.SUCCESS:
            if res.trajectory is None or len(res.trajectory) == 0:
                return self._finish(out, PlanningStatus.ERROR, "planner returned an empty path", t0, details)
            out.path = to_geometric_path(res.trajectory)
            details.update(direction_changes=int(res.trajectory.n_direction_changes),
                           path_length=float(res.trajectory.length), cost=float(res.cost))
        return self._finish(out, status, res.message, t0, details)

    # ---------------------------------------------------------- helpers
    def _planner(self, name: str, vehicle: VehicleInfo, cfg: PlannerConfig):
        key = (name, vehicle.key(), cfg)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        planner = PLANNERS[name](vehicle, cfg)
        self._cache[key] = planner
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return planner

    @staticmethod
    def _finish(out: PlanningResult, status: PlanningStatus, message: str, t0: float,
                details: Dict[str, Any]) -> PlanningResult:
        out.status = status
        out.message = message or ""
        out.planning_time = time.perf_counter() - t0
        out.details = json.dumps(details, default=str)
        return out


__all__ = ["StateLatticeAdapter", "parse_planner_config", "to_planner_grid", "to_vehicle_info",
           "to_geometric_path", "InvalidRequestError", "DEFAULT_PLANNER"]
