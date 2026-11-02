"""Dora node: trajectory following at controller_hz (sim clock).

The control law is the plugin selected by ``control.controller.type``
(control/registry.py).

Lockstep barrier: before computing the command of tick t it waits for the
trajectory of the latest planner tick <= t, so the result does not depend on
message arrival order."""
from __future__ import annotations

import time

from doraware.control.registry import make_controller
from doraware.core.clock import RateClock
from doraware.core.lockstep import InputBarrier
from doraware.core.node import NodeRuntime
from doraware.interfaces import DiagnosticLevel, SimulationScene, Trajectory, VehicleState, to_sec

INPUTS = {"scene": SimulationScene, "vehicle_state": VehicleState, "trajectory": Trajectory}


def main() -> None:
    rt = NodeRuntime("controller", INPUTS)
    timing = rt.cfg["timing"]
    control_ticks = RateClock(timing, "controller_hz")
    planner_ticks = RateClock(timing, "planner_hz")
    silence_after = rt.cfg["fault_injection"].get("controller_silence_after")
    controller = None
    pending = []          # trajectories that arrive before the scene (vehicle parameters)

    def on_trajectory(traj: Trajectory) -> None:
        if controller is None:
            pending.append(traj)
        else:
            controller.on_trajectory(traj)

    trajectories = InputBarrier(rt, "trajectory", on_trajectory, timing)

    def on_scene(scene: SimulationScene) -> None:
        nonlocal controller
        controller = make_controller(rt.cfg, scene.vehicle)
        for traj in pending:
            controller.on_trajectory(traj)
        pending.clear()

    for ev in rt.events():
        if ev["type"] != "INPUT":
            continue
        if trajectories.accepts(ev):
            trajectories.take(ev)
            continue
        if ev["id"] == "scene":
            on_scene(rt.decode(ev))
            continue
        state: VehicleState = rt.decode(ev)
        t_ns = state.header.stamp
        if not control_ticks.is_tick(t_ns):
            continue
        if controller is None:
            got, _ = rt.wait_for(lambda e: e["id"] == "scene", float(timing["lockstep_wall_timeout"]))
            if got is None:
                rt.diagnose(t_ns, DiagnosticLevel.ERROR, "no scene / vehicle parameters")
                continue
            on_scene(rt.decode(got))
        trajectories.wait(planner_ticks.last_tick_ns(t_ns), t_ns)
        if silence_after is not None and to_sec(t_ns) >= float(silence_after):
            continue  # fault injection: controller silent
        w0 = time.perf_counter()
        cmd, dbg = controller.step(state, t_ns)
        wall = time.perf_counter() - w0
        cmd.header = rt.header(t_ns, "control_command", frame_id="base_link")
        dbg.header = rt.header(t_ns, "control_debug")
        dbg.compute_time = wall
        rt.send("control_command", cmd)
        rt.send("control_debug", dbg)
        if wall > 1.0 / control_ticks.rate_hz:
            rt.diagnose(t_ns, DiagnosticLevel.WARN, "controller step exceeded its period (wall)", wall_time=wall)


if __name__ == "__main__":
    main()
