"""``path_pipeline``: the configured path planner (``planning.path_planner``) followed
by trajectory generation (``planning.trajectory_generation``), in one process.

It is the reference trajectory planner: it produces exactly what the default
dataflow's ``planner`` + ``trajectory_generator`` nodes produce (an e2e test
checks the closed loop is bit-identical), so it validates the trajectory
planner contract and the ``trajectory_planner`` dataflow.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from doraware.core.trajectory import stop_trajectory
from doraware.interfaces import (PlanningRequest, PlanningResult, PlanningStatus, Trajectory, VehicleState)
from doraware.planning.path_planning.registry import make_planner
from doraware.planning.trajectory_generation.generator import TrajectoryGenerator


class PathPipeline:
    def __init__(self, params: Dict[str, Any], cfg: Dict[str, Any]):
        self.cfg = cfg
        self.timing = cfg["timing"]
        self.planner = make_planner(cfg)
        self.gen: Optional[TrajectoryGenerator] = None

    def plan(self, request: PlanningRequest, state: VehicleState, t_ns: int) -> Optional[PlanningResult]:
        try:
            res = self.planner.plan(request)
        except Exception as exc:  # report, never crash silently
            res = PlanningResult(status=PlanningStatus.ERROR, message=f"{type(exc).__name__}: {exc}",
                                 goal=request.goal)
        res.request_id = request.request_id
        self.gen = TrajectoryGenerator(self.cfg["planning"]["trajectory_generation"], self.timing, request.vehicle)
        self.gen.on_planning_result(res, request.map, plan_start_ns=t_ns, initial_steering=float(state.steering_angle))
        return res

    def trajectory_at(self, state: VehicleState, t_ns: int) -> Trajectory:
        traj = self.gen.trajectory_at(t_ns) if self.gen is not None else None
        return traj or stop_trajectory(t_ns, self.timing)

    def diagnostics(self) -> Dict[str, Any]:
        return {} if self.gen is None else self.gen.status_report()
