"""Dora node: GeometricPath -> time-based Trajectory windows at planner_hz.

Lockstep: on every planner tick (sim clock, derived from VehicleState
stamps) it publishes the window starting at that tick. At the first tick it
waits for the PlanningResult (planning is not sim time).

A PlanningResult is applied only at a planner tick, with that tick as the
plan start and the vehicle state of that tick (measured steering angle) as
the initial condition. It never depends on whether the result arrived before
or after a VehicleState, so the trajectory is deterministic."""
from __future__ import annotations

from typing import Optional

from doraware.core.clock import RateClock
from doraware.core.node import NodeRuntime
from doraware.interfaces import (DiagnosticLevel, PlanningRequest, PlanningResult, Trajectory, VehicleState,
                                 to_ns, to_sec)
from doraware.planning.trajectory_generation.generator import TrajectoryGenerator

INPUTS = {"planning_request": PlanningRequest, "planning_result": PlanningResult, "vehicle_state": VehicleState}


def main() -> None:
    rt = NodeRuntime("trajectory_generator", INPUTS)
    timing = rt.cfg["timing"]
    ticks = RateClock(timing, "planner_hz")
    silence_after = rt.cfg["fault_injection"].get("planner_silence_after")
    plan_wait = float(timing["planning_wall_timeout"])
    request: Optional[PlanningRequest] = None
    pending: Optional[PlanningResult] = None     # received, applied at the next planner tick
    gen: Optional[TrajectoryGenerator] = None
    status_sent = False
    gave_up = False

    def on_event(ev) -> None:
        nonlocal request, gen, pending
        msg = rt.decode(ev)
        if ev["id"] == "planning_request":
            request = msg
            gen = TrajectoryGenerator(rt.cfg["planning"]["trajectory_generation"], timing, msg.vehicle)
        elif ev["id"] == "planning_result":
            pending = msg

    for ev in rt.events():
        if ev["type"] != "INPUT":
            continue
        if ev["id"] != "vehicle_state":
            on_event(ev)
            continue
        state: VehicleState = rt.decode(ev)
        t_ns = state.header.stamp
        if not ticks.is_tick(t_ns):
            continue
        if silence_after is not None and to_sec(t_ns) >= float(silence_after):
            continue  # fault injection: the planning pipeline went silent
        while (gen is None or not gen.has_result) and pending is None and not gave_up:
            got, timed_out = rt.wait_for(lambda e: e["id"] in ("planning_request", "planning_result"), plan_wait)
            if got is None:
                rt.diagnose(t_ns, DiagnosticLevel.ERROR, "no planning result (timeout)" if timed_out else "closed",
                            output="trajectory_status")
                gave_up = True   # do not block every later tick; the controller stops safely
                break
            on_event(got)
        if pending is not None:
            if gen is None:
                rt.log.error("planning result without a planning request: ignored")
            else:
                gen.on_planning_result(pending, request.map, plan_start_ns=t_ns,
                                       initial_steering=float(state.steering_angle))
                status_sent = False
            pending = None
        if gen is None:
            continue
        traj = gen.trajectory_at(t_ns)
        if traj is None:
            traj = Trajectory(start_time=t_ns, time_step=float(timing["trajectory_time_step"]),
                              valid_until=t_ns + to_ns(float(timing["trajectory_validity"])))
        traj.header = rt.header(t_ns, "trajectory")
        rt.send("trajectory", traj)
        if not status_sent and gen.has_result:
            info = gen.diagnostics()
            info["trajectory_ok"] = gen.ok
            if gen.profile is not None:
                info["plan_start_time"] = gen.plan_start_ns * 1e-9
                info["plan_duration"] = gen.profile.duration
            level = DiagnosticLevel.OK if gen.ok or not gen.planning_status == 1 else DiagnosticLevel.ERROR
            rt.diagnose(t_ns, level, str(info.get("message", "")), output="trajectory_status",
                        **{k: v for k, v in info.items() if k != "message"})
            status_sent = True


if __name__ == "__main__":
    main()
