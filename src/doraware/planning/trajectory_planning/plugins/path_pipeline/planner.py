"""``path_pipeline``: the configured path planner (``planning.planner``) followed
by trajectory generation (``planning.trajectory_generation``), in one process.

It is the reference trajectory planner: it produces exactly what the default
dataflow's ``planner`` + ``trajectory_generator`` nodes produce (an e2e test
checks the closed loop is bit-identical), so it validates the trajectory
planner contract and the ``trajectory_planner`` dataflow.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from doraware.interfaces import (PlanningRequest, PlanningResult, PlanningStatus, Trajectory, VehicleState,
                                 to_ns)
from doraware.planning.registry import make_planner
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
        if traj is None:
            traj = Trajectory(start_time=t_ns, time_step=float(self.timing["trajectory_time_step"]),
                              valid_until=t_ns + to_ns(float(self.timing["trajectory_validity"])))
        return traj

    def diagnostics(self) -> Dict[str, Any]:
        if self.gen is None:
            return {}
        info = self.gen.diagnostics()
        info["trajectory_ok"] = self.gen.ok
        if self.gen.profile is not None:
            info["plan_start_time"] = self.gen.plan_start_ns * 1e-9
            info["plan_duration"] = self.gen.profile.duration
        return info
