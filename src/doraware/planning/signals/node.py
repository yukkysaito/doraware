"""Dora node: VehicleSignalCommand (turn indicators, hazard lights) at planner_hz.

It runs on the planner ticks of the sim clock (derived from VehicleState
stamps). Lockstep: like the controller, it first waits for the trajectory of
the tick, so the decision does not depend on message arrival order. The sim
bridge waits for this command in turn."""
from __future__ import annotations

from doraware.core.clock import RateClock
from doraware.core.lockstep import InputBarrier
from doraware.core.node import NodeRuntime
from doraware.interfaces import DiagnosticLevel, Trajectory, VehicleSignalCommand, VehicleState
from doraware.planning.signals.decider import SignalDecider, SignalParams

INPUTS = {"vehicle_state": VehicleState, "trajectory": Trajectory}


def main() -> None:
    rt = NodeRuntime("signal_decider", INPUTS)
    ticks = RateClock(rt.cfg["timing"], "planner_hz")
    decider = SignalDecider(SignalParams.from_config(rt.cfg["planning"]["signals"]))
    trajectories = InputBarrier(rt, "trajectory", decider.on_trajectory)
    last = None

    for ev in rt.events():
        if ev["type"] != "INPUT":
            continue
        if trajectories.accepts(ev):
            trajectories.take(ev)
            continue
        state: VehicleState = rt.decode(ev)
        t_ns = state.header.stamp
        if not ticks.is_tick(t_ns):
            continue
        trajectories.wait(t_ns, t_ns)    # starved: decide from what we have (expired -> hazard lights)
        d = decider.step(state, t_ns)
        rt.send("vehicle_signal_command", VehicleSignalCommand(
            header=rt.header(t_ns, "vehicle_signal_command", frame_id="base_link"),
            turn_indicator=d.turn_indicator, hazard_lights=d.hazard_lights))
        if (d.turn_indicator, d.hazard_lights) != last:
            last = (d.turn_indicator, d.hazard_lights)
            rt.diagnose(t_ns, DiagnosticLevel.OK,
                        f"signals: turn {d.turn_indicator.name}, hazard {'on' if d.hazard_lights else 'off'}",
                        turn_indicator=d.turn_indicator.name, hazard_lights=d.hazard_lights,
                        hazard_reason=d.hazard_reason, lateral_offset=d.lateral_offset)


if __name__ == "__main__":
    main()
