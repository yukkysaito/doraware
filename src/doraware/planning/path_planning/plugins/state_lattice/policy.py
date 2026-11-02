"""Request policy of the State Lattice plugin (no Dora import).

* ``base_config`` parameters are applied under the scenario's own planner
  parameters (the scenario wins).
* Accuracy fallback: the base config may request a tighter goal tolerance than
  the planner's default (goal-accuracy budget allocation, docs/architecture.md
  §12). If planning then runs out of budget (timeout / expansion limit; not a
  proof and not an exhausted search space), the request
  is re-planned once without those keys, so the planner is never less capable
  than with its own defaults.
"""
from __future__ import annotations

import copy
import json
from typing import Dict, Iterable

from doraware.interfaces import PlanningRequest, PlanningResult, PlanningStatus

RETRYABLE_RAW = frozenset({"timeout", "max_expansions"})   # budget exhausted (not a proof, not exhaustive)


def _budget_exhausted(res: PlanningResult) -> bool:
    if res.status == PlanningStatus.TIMEOUT:
        return True
    try:
        raw = json.loads(res.details or "{}").get("raw_status")
    except ValueError:
        raw = None
    return res.status == PlanningStatus.NO_PATH and raw in RETRYABLE_RAW


def plan_with_policy(planner, request: PlanningRequest, base_config: Dict,
                     fallback_keys: Iterable[str] = ()) -> PlanningResult:
    own = json.loads(request.planner_config or "{}")
    first = copy.copy(request)
    first.planner_config = json.dumps({**base_config, **own})
    res = planner.plan(first)
    tightened = [k for k in fallback_keys if k in base_config and k not in own]
    if res.success or not _budget_exhausted(res) or not tightened:
        return res
    relaxed = copy.copy(request)
    relaxed.planner_config = json.dumps({**{k: v for k, v in base_config.items() if k not in tightened}, **own})
    res2 = planner.plan(relaxed)
    details = json.loads(res2.details or "{}")
    details.update(tolerance_fallback=True, first_attempt_status=res.status.name,
                   first_attempt_time=res.planning_time, relaxed_keys=tightened)
    res2.details = json.dumps(details)
    res2.planning_time += res.planning_time        # the system waited for both attempts
    return res2
