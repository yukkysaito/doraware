"""Dora node: the sim bridge, i.e. simulation coordinator + simulator backend
(docs/architecture.md §6/§7).

Owns the sim clock. In lockstep mode it waits for the control command of each
controller tick (and the vehicle signal command of each planner tick) before
advancing physics, which makes closed-loop runs deterministic and independent
of host load and rendering. The backend (``simulation.backend``) only advances
the world; it is created through simulation/registry.py."""
from __future__ import annotations

import dataclasses
import os
import time
from typing import Callable, Dict, Optional

from doraware.core.node import NodeRuntime
from doraware.interfaces import (ControlCommand, ControlDebug, DiagnosticLevel, Gear, PlanningResult,
                                 ScenarioDone, SimulationScene, SimulationState, SimulationStatus, Trajectory,
                                 PerceivedObjects, PlanningRequest, VehicleSignalCommand)
from doraware.simulation.backend import SimFrame, SimulatorBackend, VizFrame
from doraware.simulation.coordinator import StepCoordinator
from doraware.simulation.ground_truth import scene_objects
from doraware.simulation.registry import make_simulator

CAMERAS = {"top": 0, "chase": 1, "free": 2, "overview": 3}    # GUI cameras for screenshots
INPUTS = {"scene": SimulationScene, "control_command": ControlCommand, "scenario_done": ScenarioDone,
          "trajectory": Trajectory, "control_debug": ControlDebug, "planning_result": PlanningResult,
          "vehicle_signal_command": VehicleSignalCommand, "objects": PerceivedObjects,
          "planning_request": PlanningRequest}


class Bridge:
    def __init__(self, rt: NodeRuntime):
        self.rt = rt
        cfg = rt.cfg
        self.timing = cfg["timing"]
        self.coord = StepCoordinator(self.timing)
        self.lockstep = bool(self.timing["lockstep"])
        self.wall_timeout = float(self.timing["lockstep_wall_timeout"])
        self.stall_warning = float(self.timing["lockstep_stall_warning"])
        self.max_wait = 0.0       # [s, wall] longest lockstep wait after the first command (stalls)
        self.rtf = float(self.timing.get("realtime_factor") or 0.0)
        self.max_sim_time = float(cfg["evaluation"]["max_sim_time"]) + 5.0
        self.gui = os.environ.get("DORAWARE_GUI") == "1"
        self.seed = int(os.environ.get("DORAWARE_SEED", "0") or 0)
        self.sim: Optional[SimulatorBackend] = None
        self.frame: Optional[SimFrame] = None
        self.step = 0
        self.done = False
        self.scene: Optional[SimulationScene] = None
        # control commands (keyed by stamp); a silent controller is not waited for again
        self.commands: Dict[int, ControlCommand] = {}
        self.cmd: Optional[ControlCommand] = None
        self.controller_silent = False
        self.expired = False
        self.timeouts = 0
        self.last_seen: Dict[str, tuple] = {}   # input -> (latest stamp, wall time received): timeout reports
        # vehicle signals: latched like on a real vehicle; in lockstep the command of every
        # planner tick is awaited (only if a signal source is wired into this dataflow)
        self.signals_wired = rt.input_connected("vehicle_signal_command")
        self.signal_cmds: Dict[int, VehicleSignalCommand] = {}
        self.signal: Optional[VehicleSignalCommand] = None
        self.signals_silent = False
        # visualization inputs (GUI only)
        self.traj: Optional[Trajectory] = None
        self.debug: Optional[ControlDebug] = None
        self.plan: Optional[PlanningResult] = None
        self.plan_sent = False
        self.objects: Optional[PerceivedObjects] = None
        self.request: Optional[PlanningRequest] = None
        self.lanes_sent = False
        self.shots = sorted(float(x) for x in (cfg["simulation"].get("screenshot_times") or []))
        self.shot_cameras = list(cfg["simulation"].get("screenshot_cameras") or ["top", "chase"])
        unknown = [c for c in self.shot_cameras if c not in CAMERAS]
        if unknown:
            raise ValueError(f"simulation.screenshot_cameras: unknown {unknown}; use {sorted(CAMERAS)}")

    # ---------------------------------------------------------- events
    def handle(self, ev) -> None:
        if ev["type"] != "INPUT":
            if ev["type"] == "STOP":
                self.done = True
            return
        iid = ev["id"]
        msg = self.rt.decode(ev)
        stamp = getattr(getattr(msg, "header", None), "stamp", None)
        if stamp is not None:
            self.last_seen[iid] = (int(stamp), time.monotonic())
        if iid == "control_command":
            self.commands[msg.header.stamp] = msg
        elif iid == "vehicle_signal_command":
            self.signal_cmds[msg.header.stamp] = msg
        elif iid == "scenario_done":
            self.done = True
        elif iid == "trajectory":
            self.traj = msg
        elif iid == "control_debug":
            self.debug = msg
        elif iid == "planning_result":
            self.plan, self.plan_sent = msg, False
        elif iid == "scene":
            self.scene = msg
        elif iid == "planning_request":
            self.request, self.lanes_sent = msg, False   # GUI: lane map and route
        elif iid == "objects":
            self.objects = msg       # perception output (optional input): drawn in the GUI

    def pump(self, until: Callable[[], bool], timeout: float) -> bool:
        """Process events until ``until()`` is true. False on wall timeout."""
        deadline = time.monotonic() + timeout
        while not until() and not self.done:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            ev = self.rt.next(timeout=remaining)
            if ev is None:
                self.done = True
                break
            if ev["type"] == "TIMEOUT":
                return False
            self.handle(ev)
        return True

    def drain(self) -> None:
        while True:
            ev = self.rt.next(timeout=0.0)
            if ev is None or ev["type"] == "TIMEOUT":
                return
            self.handle(ev)

    def await_lockstep(self, t: int, received: Callable[[], bool], first: bool, silent: bool, what: str) -> bool:
        """Wait (lockstep) for a producer's message of tick ``t``. The first one waits
        for planning (wall time, not sim time); a producer already declared silent
        is not waited for again. Returns False (and reports once) on a timeout."""
        limit = float(self.timing["planning_wall_timeout"]) if first else self.wall_timeout
        if silent:
            limit = 0.05
        w0 = time.monotonic()
        ok = self.pump(received, limit)
        waited = time.monotonic() - w0
        if not first:      # the first wait includes planning (wall time by design)
            self.max_wait = max(self.max_wait, waited)
            if ok and waited > self.stall_warning:
                self.rt.diagnose(t, DiagnosticLevel.WARN, f"{what} late by {waited:.1f} s (wall): lockstep stall",
                                 step=self.step, waited=waited)
        if not ok and not silent:
            now = time.monotonic()
            seen = {k: {"stamp": v[0], "age_wall": round(now - v[1], 3)} for k, v in sorted(self.last_seen.items())}
            self.rt.diagnose(t, DiagnosticLevel.ERROR, f"{what} timeout (lockstep)", step=self.step, last_seen=seen)
        return ok

    # ---------------------------------------------------------- stepping
    def publish(self) -> None:
        t = self.coord.time_ns(self.step)
        f = self.frame
        self.rt.send("vehicle_state", dataclasses.replace(f.vehicle_state, header=self.rt.header(t, "vehicle_state")))
        self.rt.send("vehicle_signal_status", dataclasses.replace(
            f.signal_status, header=self.rt.header(t, "vehicle_signal_status", frame_id="base_link")))
        if self.coord.planner.is_tick(self.step):     # ground truth in the perception contract
            self.rt.send("ground_truth_objects", scene_objects(
                self.scene, self.rt.header(t, "ground_truth_objects")))
        self.rt.send("sim_status", SimulationStatus(
            header=self.rt.header(t, "sim_status"), state=SimulationState.RUNNING, step_count=self.step,
            physics_dt=self.coord.dt, collision_count=f.contacts, collision_objects=",".join(f.contact_names),
            step_wall_time=f.step_wall_time, command_timeouts=self.timeouts, max_lockstep_wait=self.max_wait))

    def select_command(self) -> ControlCommand:
        t = self.coord.time_ns(self.step)
        if self.coord.needs_command(self.step):
            if self.lockstep:
                ok = self.await_lockstep(t, lambda: t in self.commands, self.cmd is None, self.controller_silent,
                                         "control command")
                if not ok:
                    self.timeouts += 1   # any late/missing command breaks lockstep: reported, fails nominal runs
                self.controller_silent = not ok
            else:
                self.drain()
            if t in self.commands:
                self.cmd = self.commands.pop(t)
            elif not self.lockstep and self.commands:
                self.cmd = self.commands.pop(max(self.commands))
            self.commands = {k: v for k, v in self.commands.items() if k > t}
        if self.cmd is None or t > self.cmd.valid_until:
            if self.cmd is not None and not self.expired:
                self.timeouts += 1
                self.rt.diagnose(t, DiagnosticLevel.ERROR, "control command expired: emergency braking")
            self.expired = True
            return ControlCommand(target_acceleration=0.0, target_steering_angle=self.frame.vehicle_state.steering_angle,
                                  gear_request=Gear.NONE, emergency_stop=True)
        self.expired = False
        return self.cmd

    def select_signals(self) -> Optional[VehicleSignalCommand]:
        """Signal command to latch before stepping (None = keep). In lockstep, the
        command of each planner tick is awaited like a control command."""
        t = self.coord.time_ns(self.step)
        if self.signals_wired and self.lockstep and self.coord.planner.is_tick(self.step):
            ok = self.await_lockstep(t, lambda: t in self.signal_cmds, self.signal is None, self.signals_silent,
                                     "vehicle signal command")
            self.signals_silent = not ok
        due = [k for k in self.signal_cmds if k <= t or not self.lockstep]
        if not due:
            return None
        cmd = self.signal_cmds[max(due)]
        self.signal_cmds = {k: v for k, v in self.signal_cmds.items() if k not in due}
        changed = self.signal is None or (cmd.turn_indicator, cmd.hazard_lights) != (
            self.signal.turn_indicator, self.signal.hazard_lights)
        self.signal = cmd
        return cmd if changed else None

    # ---------------------------------------------------------- GUI
    def send_viz(self) -> None:
        if self.sim.headless:
            return
        info: Dict[str, object] = {"scenario": self.scene.scenario_id if self.scene else ""}
        path, path_gear = None, ()
        if self.plan is not None:
            info["planning"] = self.plan.status.name
            if not self.plan_sent:
                path = list(zip(self.plan.path.x.tolist(), self.plan.path.y.tolist()))
                path_gear = self.plan.path.gear.tolist()
                self.plan_sent = True
        c = self.cmd
        if c is not None:
            info.update(cmd_accel=c.target_acceleration, cmd_steer=c.target_steering_angle,
                        gear_request=int(c.gear_request), emergency=bool(c.emergency_stop) or self.expired)
        d = self.debug
        if d is not None:
            info.update(mode=d.mode, solver=d.solver_status, solve_ms=d.solve_time * 1e3,
                        lateral_error=d.lateral_error, heading_error=d.heading_error)
        lanes = None
        if self.request is not None and not self.lanes_sent:
            lanes, self.lanes_sent = self.request.lanes, True
        self.sim.send_viz(VizFrame(
            lanes=lanes, route_lane_ids=self.request.route.lane_ids.tolist() if self.request is not None else [],
            planned=list(zip(self.traj.x.tolist(), self.traj.y.tolist())) if self.traj is not None else [],
            predicted=list(zip(d.predicted_x.tolist(), d.predicted_y.tolist())) if d is not None else [],
            path=path, path_gear=path_gear, info=info, objects=self.objects))
        self.objects = None

    def maybe_screenshot(self) -> None:
        if self.sim.headless:
            return
        t = self.step * self.coord.dt
        while self.shots and t >= self.shots[0]:
            ts = self.shots.pop(0)
            for name in self.shot_cameras:
                path = os.path.join(self.rt.run_dir or ".", f"screenshot_{ts:05.1f}s_{name}.png")
                self.sim.screenshot(path, CAMERAS[name])

    # ---------------------------------------------------------- main loop
    def run(self) -> None:
        self.pump(lambda: self.scene is not None, 600.0)
        if self.scene is None:
            self.rt.log.error("no scene received")
            return
        delay = float(self.rt.cfg["fault_injection"].get("simulator_start_delay") or 0.0)
        if delay > 0:
            time.sleep(delay)    # fault injection: a slow simulator start (e.g. GUI) must not change results
        self.sim = make_simulator(self.rt.cfg, physics_hz=self.coord.physics_hz, gui=self.gui,
                                  log_path=os.path.join(self.rt.run_dir or ".", "simulator.log"))
        try:
            self.frame = self.sim.reset(self.scene, self.seed)
            self.publish()
            wall0 = time.monotonic()
            while not self.done:
                cmd = self.select_command()
                signals = self.select_signals()
                if self.done:
                    break
                n = self.coord.chunk(self.step)
                self.frame = self.sim.step(n, cmd, signals)
                self.step += n
                if self.coord.publishes_state(self.step):
                    self.publish()
                if self.coord.sends_viz(self.step):
                    self.send_viz()
                    self.maybe_screenshot()
                if self.rtf > 0:
                    lag = self.step * self.coord.dt / self.rtf - (time.monotonic() - wall0)
                    if lag > 0:
                        time.sleep(lag)
                if self.step * self.coord.dt > self.max_sim_time:
                    self.rt.diagnose(self.coord.time_ns(self.step), DiagnosticLevel.ERROR, "max sim time reached")
                    break
        finally:
            self.sim.close()
            self.rt.log.info("simulation stopped at step %d (t=%.2f s)", self.step, self.step * self.coord.dt)


def main() -> None:
    rt = NodeRuntime("sim_bridge", INPUTS)
    Bridge(rt).run()


if __name__ == "__main__":
    main()
