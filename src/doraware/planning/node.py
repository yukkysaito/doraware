"""Dora node: PlanningRequest -> PlanningResult (+ diagnostics).

The planning algorithm is the plugin selected by ``planning.planner.type``
(planning/registry.py); the node itself is planner-agnostic.
"""
from __future__ import annotations

import os
import time

from doraware.interfaces import DiagnosticLevel, PlanningRequest, PlanningResult, PlanningStatus
from doraware.planning.registry import make_planner
from doraware.core.node import NodeRuntime


def main() -> None:
    rt = NodeRuntime("planner", {"planning_request": PlanningRequest})
    planner = make_planner(rt.cfg)
    crash_once = bool(rt.cfg["fault_injection"].get("planner_crash_once"))
    if rt.node.is_restart():
        # Dora restarted this node after a failure; inputs are not redelivered, so announce
        # the restart and let the requester re-send its outstanding request.
        rt.diagnose(0, DiagnosticLevel.WARN, "planner restarted", restart_count=rt.node.restart_count(),
                    restarted=True)
    for ev in rt.events():
        if ev["type"] != "INPUT":
            continue
        req: PlanningRequest = rt.decode(ev)
        if crash_once and not rt.node.is_restart():
            rt.log.error("fault injection: planner crash")
            os._exit(3)
        t0 = time.perf_counter()
        try:
            res: PlanningResult = planner.plan(req)
        except Exception as exc:  # report, never crash silently
            rt.log.exception("planner crashed")
            res = PlanningResult(request_id=req.request_id, status=PlanningStatus.ERROR,
                                 message=f"{type(exc).__name__}: {exc}", goal=req.goal)
        res.header = rt.header(req.header.stamp, "planning_result")
        res.request_id = req.request_id
        wall = time.perf_counter() - t0
        rt.send("planning_result", res)
        level = DiagnosticLevel.OK if res.success else DiagnosticLevel.WARN
        rt.diagnose(req.header.stamp, level, f"planning {res.status.name}: {res.message[:200]}",
                    status=res.status.name, planning_time=res.planning_time, wall_time=wall, points=len(res.path))


if __name__ == "__main__":
    main()
