"""Planner plugins, selected by ``planning.planner.type``.

A planner plugin is a factory ``(params, cfg) -> PathPlanner``: ``params`` is
the plugin's own config section ``planning.planner.<type>`` (``{}`` if absent)
and ``cfg`` the full system config. Built-in plugins live in
``planning/plugins/<name>/`` and are registered below; separately installed
packages register under the ``doraware.planners`` entry-point group.

Planners that output a time-based trajectory directly (for example learned
planners) are not plugins of this node; they are their own planning node that
publishes ``Trajectory`` (see docs/extending.md).
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
    from doraware.planning.plugins.state_lattice.planner import StateLatticePlanner
    return StateLatticePlanner(params)


def make_planner(cfg: Dict[str, Any]) -> PathPlanner:
    """Instantiate the planner selected by ``cfg["planning"]["planner"]["type"]``."""
    section = cfg["planning"]["planner"]
    name = section["type"]
    return PLANNERS.create(name, dict(section.get(name) or {}), cfg)
