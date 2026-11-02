"""Trajectory planner plugins, selected by ``planning.trajectory_planner.type``.

A trajectory planner runs on the planner ticks of the sim clock and returns
the executable :class:`Trajectory` itself (contract in docs/extending.md). It
is hosted by ``planning/trajectory_planning/node.py`` and replaces the
``planner`` + ``trajectory_generator`` pair of the default dataflow
(``dataflows/trajectory_planner.yml``).

A plugin is a factory ``(params, cfg) -> TrajectoryPlanner``: ``params`` is
``planning.trajectory_planner.<type>`` and ``cfg`` the full system config.
Built-in plugins live in ``plugins/<name>/``; separately installed packages
register under the ``doraware.trajectory_planners`` entry-point group.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

from doraware.core.registry import Registry
from doraware.interfaces import PlanningRequest, PlanningResult, Trajectory, VehicleState


class TrajectoryPlanner(Protocol):
    #: Extra node inputs the planner needs (input id -> message class), e.g.
    #: ``{"objects": PerceivedObjects}``. Optional; the node delivers each through
    #: ``on_input`` and, in lockstep, waits for the message of the current tick.
    inputs: Dict[str, type]

    def on_input(self, name: str, msg: Any) -> None:
        """A message of a declared extra input (optional when there are none)."""

    def plan(self, request: PlanningRequest, state: VehicleState, t_ns: int) -> Optional[PlanningResult]:
        """Take a new request at planner tick ``t_ns`` (may block: planning is wall time).
        Return the outcome for evaluation and visualization (status, and a path if
        there is one), or None if the planner has no discrete planning step."""

    def trajectory_at(self, state: VehicleState, t_ns: int) -> Trajectory:
        """The trajectory window for planner tick ``t_ns``. ``start_time`` and the
        sampling are set by the planner; an empty trajectory means "stop".
        The node sets the header."""

    def diagnostics(self) -> Dict[str, Any]:
        """Values for the ``trajectory_status`` diagnostics after a plan. The keys
        ``plan_start_time`` and ``plan_duration`` [s, sim] tell the evaluator when the
        motion ends; without them it waits until the trajectory stops moving."""


TRAJECTORY_PLANNERS: Registry[TrajectoryPlanner] = Registry("trajectory planner", "doraware.trajectory_planners")


@TRAJECTORY_PLANNERS.register("path_pipeline")
def _path_pipeline(params: Dict[str, Any], cfg: Dict[str, Any]) -> TrajectoryPlanner:
    from doraware.planning.trajectory_planning.plugins.path_pipeline.planner import PathPipeline
    return PathPipeline(params, cfg)


def make_trajectory_planner(cfg: Dict[str, Any]) -> TrajectoryPlanner:
    section = cfg["planning"]["trajectory_planner"]
    name = section["type"]
    return TRAJECTORY_PLANNERS.create(name, dict(section.get(name) or {}), cfg)
