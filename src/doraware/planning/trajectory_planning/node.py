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

from doraware.core.trajectory import stop_trajectory
from doraware.core.clock import RateClock
from doraware.core.lockstep import InputBarrier
from doraware.core.node import NodeRuntime
from doraware.interfaces import (DiagnosticLevel, PlanningRequest, PlanningResult, PlanningStatus, Trajectory,
                                 VehicleState, to_sec)
from doraware.planning.trajectory_planning.registry import make_trajectory_planner

INPUTS = {"planning_request": PlanningRequest, "vehicle_state": VehicleState}


def main() -> None:
    rt = NodeRuntime("trajectory_planner", INPUTS)
    timing = rt.cfg["timing"]
    ticks = RateClock(timing, "planner_hz")
    silence_after = rt.cfg["fault_injection"].get("planner_silence_after")
    planner = make_trajectory_planner(rt.cfg)
    # extra inputs declared by the plugin (e.g. objects): delivered in lockstep, others ignored
    extra = dict(getattr(planner, "inputs", None) or {})
    rt.inputs.update(extra)
    barriers = [InputBarrier(rt, name, lambda msg, n=name: planner.on_input(n, msg)) for name in extra]
    pending: Optional[PlanningRequest] = None
    planned = False
    failed = False          # the plugin raised: publish "stop" (empty trajectories) from then on
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
        if ev["id"] != "vehicle_state":
            for b in barriers:
                if b.accepts(ev):
                    b.take(ev)
            continue       # inputs the plugin did not declare are ignored
        state: VehicleState = rt.decode(ev)
        t_ns = state.header.stamp
        if not ticks.is_tick(t_ns):
            continue
        if silence_after is not None and to_sec(t_ns) >= float(silence_after):
            continue  # fault injection: the planning pipeline went silent
        for b in barriers:
            b.wait(t_ns, t_ns)        # the plugin sees the inputs of this tick
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
            try:
                res = planner.plan(pending, state, t_ns)
            except Exception as exc:  # a plugin failure is a planning failure, never a silent crash
                rt.log.exception("trajectory planner plan() failed")
                res = PlanningResult(request_id=pending.request_id, status=PlanningStatus.ERROR,
                                     message=f"{type(exc).__name__}: {exc}", goal=pending.goal)
                failed = True
            wall = time.perf_counter() - w0
            if res is not None:
                res.header = rt.header(pending.header.stamp, "planning_result")
                rt.send("planning_result", res)
                level = DiagnosticLevel.OK if res.status == PlanningStatus.SUCCESS else DiagnosticLevel.WARN
                rt.diagnose(t_ns, level, f"planning {res.status.name}: {res.message[:200]}",
                            status=res.status.name, planning_time=res.planning_time, wall_time=wall)
            info = {} if failed else planner.diagnostics()
            ok = bool(info.get("trajectory_ok", True))
            level = DiagnosticLevel.OK if ok or res is None or not res.success else DiagnosticLevel.ERROR
            if level == DiagnosticLevel.OK and info.get("speed_profile_fallback"):
                level = DiagnosticLevel.WARN       # slow conservative speed profile: report it
            rt.diagnose(t_ns, level,
                        str(info.get("message", "")), output="trajectory_status",
                        **{k: v for k, v in info.items() if k != "message"})
            pending, planned = None, True
        try:
            traj = Trajectory() if failed else planner.trajectory_at(state, t_ns)
        except Exception as exc:  # stop safely (empty trajectory) and report
            if not failed:
                rt.log.exception("trajectory planner trajectory_at() failed")
                rt.diagnose(t_ns, DiagnosticLevel.ERROR, f"trajectory planner failed: {type(exc).__name__}: {exc}")
            failed, traj = True, Trajectory()
        if len(traj) == 0:
            traj = stop_trajectory(t_ns, timing)
        traj.header = rt.header(t_ns, "trajectory")
        rt.send("trajectory", traj)


if __name__ == "__main__":
    main()
