"""Dora node: hosts a trajectory planner plugin (``planning.trajectory_planner.type``).

Outputs on every planner tick of the sim clock (derived from VehicleState
stamps) one Trajectory stamped at the tick, which is the contract the
controller, the signal decider and the lockstep rely on. A new
PlanningRequest is handed to the plugin at the next planner tick, with that
tick's vehicle state; at the first tick the node waits for the request
(planning is wall time, not sim time). After each plan it publishes the
plugin's PlanningResult (if any) and ``trajectory_status`` diagnostics.
"""
from __future__ import annotations

import time
from typing import Optional

from doraware.core.clock import RateClock
from doraware.core.node import NodeRuntime
from doraware.interfaces import DiagnosticLevel, PlanningRequest, PlanningStatus, VehicleState, to_sec
from doraware.planning.trajectory_planning.registry import make_trajectory_planner

INPUTS = {"planning_request": PlanningRequest, "vehicle_state": VehicleState}


def main() -> None:
    rt = NodeRuntime("trajectory_planner", INPUTS)
    timing = rt.cfg["timing"]
    ticks = RateClock(timing, "planner_hz")
    silence_after = rt.cfg["fault_injection"].get("planner_silence_after")
    planner = make_trajectory_planner(rt.cfg)
    pending: Optional[PlanningRequest] = None
    planned = False
    gave_up = False
    if rt.node.is_restart():
        rt.diagnose(0, DiagnosticLevel.WARN, "trajectory planner restarted", restarted=True,
                    restart_count=rt.node.restart_count())

    for ev in rt.events():
        if ev["type"] != "INPUT":
            continue
        if ev["id"] == "planning_request":
            pending = rt.decode(ev)
            continue
        state: VehicleState = rt.decode(ev)
        t_ns = state.header.stamp
        if not ticks.is_tick(t_ns):
            continue
        if silence_after is not None and to_sec(t_ns) >= float(silence_after):
            continue  # fault injection: the planning pipeline went silent
        if not planned and pending is None and not gave_up:
            got, timed_out = rt.wait_for(lambda e: e["id"] == "planning_request",
                                         float(timing["planning_wall_timeout"]))
            if got is None:
                rt.diagnose(t_ns, DiagnosticLevel.ERROR, "no planning request (timeout)" if timed_out else "closed")
                gave_up = True
            else:
                pending = rt.decode(got)
        if pending is not None:
            w0 = time.perf_counter()
            res = planner.plan(pending, state, t_ns)
            wall = time.perf_counter() - w0
            if res is not None:
                res.header = rt.header(pending.header.stamp, "planning_result")
                rt.send("planning_result", res)
                level = DiagnosticLevel.OK if res.status == PlanningStatus.SUCCESS else DiagnosticLevel.WARN
                rt.diagnose(t_ns, level, f"planning {res.status.name}: {res.message[:200]}",
                            status=res.status.name, planning_time=res.planning_time, wall_time=wall)
            info = planner.diagnostics()
            ok = bool(info.get("trajectory_ok", True))
            rt.diagnose(t_ns, DiagnosticLevel.OK if ok or res is None or not res.success else DiagnosticLevel.ERROR,
                        str(info.get("message", "")), output="trajectory_status",
                        **{k: v for k, v in info.items() if k != "message"})
            pending, planned = None, True
        traj = planner.trajectory_at(state, t_ns)
        traj.header = rt.header(t_ns, "trajectory")
        rt.send("trajectory", traj)


if __name__ == "__main__":
    main()
