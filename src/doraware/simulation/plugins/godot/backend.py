"""Godot backend: :class:`SimulatorBackend` on top of the Godot process client.

Converts between interface types (map frame) and the Godot protocol
(docs/architecture.md §7) with the functions of ``frame.py``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import doraware.simulation.plugins.godot.frame as gf
from doraware.interfaces import ControlCommand, SimulationScene, VehicleSignalCommand
from doraware.simulation.backend import SimFrame, VizFrame
from doraware.simulation.plugins.godot.client import GodotSimulator


def to_frame(state: Dict[str, Any]) -> SimFrame:
    """Godot ``state`` reply -> :class:`SimFrame`."""
    return SimFrame(time=float(state["time"]), vehicle_state=gf.state_to_vehicle_state(state),
                    signal_status=gf.state_to_signal_status(state), contacts=int(state.get("contacts", 0)),
                    contact_names=list(state.get("contact_names", [])),
                    step_wall_time=float(state.get("step_wall_time", 0.0)))


class GodotBackend:
    def __init__(self, params: Dict[str, Any], cfg: Dict[str, Any], *, physics_hz: float,
                 log_path: Optional[str] = None, gui: bool = False):
        sim_cfg = dict(params)
        if gui:
            sim_cfg["headless"] = False
        self.physics_hz = float(physics_hz)
        self.actuator = dict(cfg["vehicle"].get("actuator") or {})
        self.sim = GodotSimulator(sim_cfg, log_path=log_path, physics_hz=self.physics_hz)
        self.headless = self.sim.headless

    def reset(self, scene: SimulationScene, seed: int) -> SimFrame:
        return to_frame(self.sim.reset(scene, self.actuator, self.physics_hz, seed))

    def step(self, n: int, command: ControlCommand, signals: Optional[VehicleSignalCommand] = None) -> SimFrame:
        return to_frame(self.sim.step(n, command.target_acceleration, command.target_steering_angle,
                                      int(command.gear_request), bool(command.emergency_stop), signals))

    def send_viz(self, viz: VizFrame) -> None:
        if not self.headless:
            self.sim.send_viz(planned=viz.planned, predicted=viz.predicted, path=viz.path, info=viz.info,
                              path_gear=viz.path_gear)

    def screenshot(self, path: str, camera: int = 0) -> None:
        if not self.headless:
            self.sim.screenshot(path, camera)

    def close(self) -> None:
        self.sim.close()
