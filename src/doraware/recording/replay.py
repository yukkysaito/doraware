"""Replay of recorded runs (MCAP written by the recorder node).

Two clearly separated operations:

* ``states``  — *playback* of the recorded VehicleState stream: no physics is
  run. The recorded states are re-evaluated offline with the same metrics as
  the online evaluator and plotted (planned path vs. recorded driven path).
* ``commands`` — *re-simulation*: the recorded ControlCommands are re-injected
  into a fresh simulator backend (same scene, actuator config and seed,
  no planner/controller), together with the recorded VehicleSignalCommands.
  The re-simulated vehicle states and signal states are compared with the
  recorded ones; with the deterministic simulator they must match exactly.

    doraware replay states   results/runs/x/B2
    doraware replay commands results/runs/x/B2 [--gui]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml
from mcap.reader import make_reader

from doraware.core.clock import ns_to_step
from doraware.interfaces import codec
from doraware.interfaces import (ControlCommand, PlanningRequest, PlanningResult, SimulationScene,
                                   Trajectory, VehicleSignalCommand, VehicleSignalStatus, VehicleState)
from doraware.evaluation.metrics import ClosedLoopMetrics


def read_topics(run_dir: Path, topics: Dict[str, type]) -> Dict[str, List]:
    out = {t: [] for t in topics}
    with open(Path(run_dir) / "recording.mcap", "rb") as f:
        for _, ch, msg in make_reader(f).iter_messages(topics=list(topics)):
            out[ch.topic].append(codec.from_json_dict(topics[ch.topic], json.loads(msg.data)))
    return out


def replay_states(run_dir: Path, plot: bool = True) -> Dict:
    """Playback: offline metrics + plot from the recorded states (no physics)."""
    d = read_topics(run_dir, {"vehicle_state": VehicleState, "scene": SimulationScene,
                              "planning_request": PlanningRequest, "planning_result": PlanningResult,
                              "control_command": ControlCommand, "trajectory": Trajectory})
    req, scene = d["planning_request"][0], d["scene"][0]
    m = ClosedLoopMetrics(req.map, scene.vehicle, (req.goal.x, req.goal.y, req.goal.yaw))
    for r in d["planning_result"]:
        m.on_plan(r)
    # chronological merge; at equal stamps the trajectory window precedes the state
    events = [(t.header.stamp, 0, "t", t) for t in d["trajectory"]] + \
             [(s.header.stamp, 1, "s", s) for s in d["vehicle_state"]] + \
             [(c.header.stamp, 2, "c", c) for c in d["control_command"]]
    for _, _, kind, msg in sorted(events, key=lambda e: (e[0], e[1])):
        {"t": m.on_trajectory, "s": m.on_state, "c": m.on_command}[kind](msg)
    summary = {k: v for k, v in m.summary().items() if isinstance(v, (int, float))}
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        g = req.map
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(g.as_2d(), origin="lower", cmap="Greys", extent=(g.origin_x, g.origin_x + g.width * g.resolution,
                                                                   g.origin_y, g.origin_y + g.height * g.resolution))
        if d["planning_result"] and len(d["planning_result"][0].path):
            p = d["planning_result"][0].path
            ax.plot(p.x, p.y, "-", color="tab:orange", lw=1.5, label="planned path")
        xy = np.array([s.pose.xyyaw()[:2] for s in d["vehicle_state"]])
        ax.plot(xy[:, 0], xy[:, 1], "--", color="tab:blue", lw=1.2, label="recorded vehicle (rear axle)")
        ax.plot(req.goal.x, req.goal.y, "g*", ms=12, label="goal")
        ax.set_aspect("equal")
        ax.legend(loc="upper right")
        ax.set_title(f"state playback: {scene.scenario_id}")
        fig.savefig(Path(run_dir) / "replay_states.png", dpi=110)
        plt.close(fig)
    return summary


def replay_commands(run_dir: Path, gui: bool = False) -> Dict:
    """Re-simulation: recorded commands -> fresh simulator physics; compare with the recording.

    The commands are applied exactly as the sim bridge applies them: a missing or
    expired command means emergency braking with the current steering angle."""
    from doraware.simulation.registry import make_simulator

    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    seed = int(json.loads((run_dir / "scenario.json").read_text()).get("seed", 42)) \
        if (run_dir / "scenario.json").exists() else 42
    d = read_topics(run_dir, {"vehicle_state": VehicleState, "scene": SimulationScene,
                              "control_command": ControlCommand, "vehicle_signal_command": VehicleSignalCommand,
                              "vehicle_signal_status": VehicleSignalStatus})
    scene, cmds, rec = d["scene"][0], d["control_command"], d["vehicle_state"]
    sig_cmds = d["vehicle_signal_command"]
    hz = float(cfg["timing"]["physics_hz"])
    sim = make_simulator(cfg, physics_hz=hz, gui=gui, log_path=str(run_dir / "simulator_replay.log"))
    try:
        frame = sim.reset(scene, seed)
        rec_by_step = {ns_to_step(s.header.stamp, hz): s for s in rec}
        sig_by_step = {ns_to_step(s.header.stamp, hz): s for s in d["vehicle_signal_status"]}
        cmd_steps = [ns_to_step(c.header.stamp, hz) for c in cmds]
        sig_steps = [ns_to_step(c.header.stamp, hz) for c in sig_cmds]
        last_rec_step = max(rec_by_step)
        step, k, ks, current = 0, 0, 0, None
        worst = 0.0
        compared = 0
        sig_compared = sig_mismatches = 0
        while step < last_rec_step:
            while k < len(cmds) and cmd_steps[k] <= step:
                current, k = cmds[k], k + 1
            signal = None
            while ks < len(sig_cmds) and sig_steps[ks] <= step:
                signal, ks = sig_cmds[ks], ks + 1
            nxt = min(cmd_steps[k:k + 1] + sig_steps[ks:ks + 1] + [last_rec_step])
            n = max(1, nxt - step)
            if current is None or step * 1e9 / hz > current.valid_until:
                cmd = ControlCommand(target_steering_angle=frame.vehicle_state.steering_angle, emergency_stop=True)
            else:
                cmd = current
            frame = sim.step(n, cmd, signal)
            step += n
            if step in rec_by_step:
                a = frame.vehicle_state.pose.xyyaw()
                b = rec_by_step[step].pose.xyyaw()
                worst = max(worst, float(np.hypot(a[0] - b[0], a[1] - b[1])))
                compared += 1
            if step in sig_by_step:
                a, b = frame.signal_status, sig_by_step[step]
                sig_compared += 1
                sig_mismatches += int((a.turn_indicator, a.hazard_lights, a.left_lamp_on, a.right_lamp_on) !=
                                      (b.turn_indicator, b.hazard_lights, b.left_lamp_on, b.right_lamp_on))
    finally:
        sim.close()
    return {"compared_states": compared, "max_position_deviation": worst,
            "compared_signal_states": sig_compared, "signal_mismatches": sig_mismatches,
            "identical": worst == 0.0 and sig_mismatches == 0}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="doraware replay", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["states", "commands"])
    ap.add_argument("run_dir")
    ap.add_argument("--gui", action="store_true")
    a = ap.parse_args(argv)
    out = replay_states(Path(a.run_dir)) if a.mode == "states" else replay_commands(Path(a.run_dir), a.gui)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
