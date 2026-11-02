"""Planner plugins, selected by ``planning.path_planner.type``.

A planner plugin is a factory ``(params, cfg) -> PathPlanner``: ``params`` is
the plugin's own config section ``planning.path_planner.<type>`` (``{}`` if absent)
and ``cfg`` the full system config. Built-in plugins live in
``planning/path_planning/plugins/<name>/`` and are registered below; separately installed
packages register under the ``doraware.planners`` entry-point group.

Planners that output a time-based trajectory directly (for example learned
planners) are trajectory planner plugins instead
(``planning/trajectory_planning/registry.py``).
"""
from __future__ import annotations

from typing import Any, Dict, Protocol

from doraware.core.registry import Registry
from doraware.interfaces import PlanningRequest, PlanningResult


class PathPlanner(Protocol):
    """Request-driven planner: one PlanningRequest in, one PlanningResult with a
    GeometricPath out. ``plan`` must not raise; failures are a status."""

    def plan(self, request: PlanningRequest) -> PlanningResult: ...


PLANNERS: Registry[PathPlanner] = Registry("planner", "doraware.planners")


@PLANNERS.register("state_lattice")
def _state_lattice(params: Dict[str, Any], cfg: Dict[str, Any]) -> PathPlanner:
    from doraware.planning.path_planning.plugins.state_lattice.planner import StateLatticePlanner
    return StateLatticePlanner(params)


def make_planner(cfg: Dict[str, Any]) -> PathPlanner:
    """Instantiate the planner selected by ``cfg["planning"]["path_planner"]["type"]``."""
    return PLANNERS.create_from_section(cfg["planning"]["path_planner"], cfg)
