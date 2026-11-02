"""State Lattice planner plugin: the adapter plus the request policy.

``params`` is the ``planning.path_planner.state_lattice`` config section:

* ``base_config``: planner parameters applied under the scenario's own
  ``planner_config`` (the scenario wins),
* ``accuracy_fallback_keys``: keys dropped for one re-plan when the tighter
  setting ran out of budget (see ``policy.py``).
"""
from __future__ import annotations

from typing import Any, Dict

from doraware.interfaces import PlanningRequest, PlanningResult
from doraware.planning.path_planning.plugins.state_lattice.adapter import StateLatticeAdapter
from doraware.planning.path_planning.plugins.state_lattice.policy import plan_with_policy


class StateLatticePlanner:
    def __init__(self, params: Dict[str, Any]):
        self.adapter = StateLatticeAdapter()
        self.base_config = dict(params.get("base_config") or {})
        self.fallback_keys = tuple(params.get("accuracy_fallback_keys") or ())

    def plan(self, request: PlanningRequest) -> PlanningResult:
        return plan_with_policy(self.adapter, request, self.base_config, self.fallback_keys)
