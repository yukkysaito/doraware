"""Simulator backends, selected by ``simulation.backend``.

A backend factory is ``(params, cfg, *, physics_hz, log_path, gui) ->
SimulatorBackend`` (simulation/backend.py): ``params`` is the backend's own
config section ``simulation.<backend>`` and ``cfg`` the full system config.
Built-in backends live in ``simulation/plugins/<name>/``; separately installed
packages register under the ``doraware.simulators`` entry-point group.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from doraware.core.registry import Registry
from doraware.simulation.backend import SimulatorBackend

SIMULATORS: Registry[SimulatorBackend] = Registry("simulator backend", "doraware.simulators")


@SIMULATORS.register("godot")
def _godot(params: Dict[str, Any], cfg: Dict[str, Any], **kwargs) -> SimulatorBackend:
    from doraware.simulation.plugins.godot.backend import GodotBackend
    return GodotBackend(params, cfg, **kwargs)


def make_simulator(cfg: Dict[str, Any], *, physics_hz: float, log_path: Optional[str] = None,
                   gui: bool = False) -> SimulatorBackend:
    return SIMULATORS.create_from_section(cfg["simulation"], cfg, key="backend", physics_hz=physics_hz,
                                          log_path=log_path, gui=gui)
