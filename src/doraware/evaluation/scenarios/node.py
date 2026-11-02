"""Dora node: publishes one scenario (scene, planning request, expectation)
from the scenario source selected by ``evaluation.scenario_source``.

Env: DORAWARE_SCENARIO (id, e.g. B2 or N-007), DORAWARE_SEED (random scenes).
"""
from __future__ import annotations

import dataclasses
import json
import os

from doraware.evaluation.scenarios.registry import make_scenario_source
from doraware.interfaces import Diagnostics, PlanningResult, ScenarioDone, ScenarioInfo
from doraware.core.node import NodeRuntime


def main() -> None:
    rt = NodeRuntime("scenario_manager", {"scenario_done": ScenarioDone, "planner_diagnostics": Diagnostics,
                                          "planning_result": PlanningResult})
    sid = os.environ.get("DORAWARE_SCENARIO", "B2")
    seed = int(os.environ.get("DORAWARE_SEED", "42"))
    b = make_scenario_source(rt.cfg).build(sid, seed, rt.cfg)
    info = {"scenario_id": b.scenario_id, "name": b.name, "category": b.category, "description": b.description,
            "difficulty": b.difficulty, "tags": list(b.tags), "seed": seed,
            "planner_vehicle_valid": b.planner_vehicle_valid, "expected": dataclasses.asdict(b.expected),
            "planner_config": json.loads(b.request.planner_config), **b.extra}
    rt.write_json("scenario.json", info)
    rt.send("scenario_info", ScenarioInfo(header=rt.header(0, "scenario_info"), scenario_id=b.scenario_id,
                                          description=json.dumps(info, default=str)))
    rt.send("scene", b.scene)
    rt.send("planning_request", b.request)
    rt.log.info("scenario %s (%s, %s) published", b.scenario_id, b.name, b.expected.scenario_class)
    # Stay alive until the run is over: Dora's shared-memory payloads must
    # outlive the sender until every receiver has mapped them.
    answered = False
    for ev in rt.events():
        if ev["type"] != "INPUT" or ev["id"] == "heartbeat":
            continue
        if ev["id"] == "planning_result":
            answered = True
        elif ev["id"] == "planner_diagnostics":
            d = rt.decode(ev)
            if json.loads(d.values or "{}").get("restarted") and not answered:
                rt.log.warning("planner restarted: re-sending the outstanding planning request")
                rt.send("planning_request", b.request)
        elif ev["id"] == "scenario_done":
            done = rt.decode(ev)
            rt.log.info("scenario %s finished: %s", done.scenario_id, done.outcome)
            break


if __name__ == "__main__":
    main()
