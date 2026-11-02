"""Simulator backend contract used by the sim bridge (no Dora import).

The sim bridge owns sim time and the lockstep; a backend only advances the
physical world. A backend works in interface types and the map frame, so the
bridge, the replay and the tests never see backend-specific data. Backends
live in ``simulation/plugins/<name>/`` and are selected by
``simulation.backend`` (simulation/registry.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

from doraware.interfaces import (ControlCommand, SimulationScene, VehicleSignalCommand, VehicleSignalStatus,
                                 VehicleState)

Polyline = Sequence[Tuple[float, float]]


@dataclass
class SimFrame:
    """World state after a reset or a step. Message headers carry only the
    sim stamp; the bridge fills sequence ids when it publishes them."""
    time: float                              # [s, sim] step count * physics dt
    vehicle_state: VehicleState
    signal_status: VehicleSignalStatus
    contacts: int = 0                        # cumulative contacts with obstacles
    contact_names: List[str] = field(default_factory=list)
    step_wall_time: float = 0.0              # [s, wall] of the last request


@dataclass
class VizFrame:
    """Visualization overlay for GUI backends (map frame). ``path`` is sent only
    when it changed; ``info`` holds labelled values for the HUD."""
    planned: Polyline = ()                   # current trajectory window
    predicted: Polyline = ()                 # controller prediction
    path: Optional[Polyline] = None          # full planned path
    path_gear: Sequence[int] = ()            # Gear per path point (direction colouring)
    info: Dict[str, object] = field(default_factory=dict)


class SimulatorBackend(Protocol):
    headless: bool

    def reset(self, scene: SimulationScene, seed: int) -> SimFrame:
        """Rebuild the world for ``scene`` (deterministically for a given seed)."""

    def step(self, n: int, command: ControlCommand, signals: Optional[VehicleSignalCommand] = None) -> SimFrame:
        """Apply ``command`` for ``n`` physics steps; ``signals`` is latched (None = keep)."""

    def send_viz(self, viz: VizFrame) -> None:
        """Update the GUI overlay (no-op when headless)."""

    def screenshot(self, path: str, camera: int = 0) -> None:
        """GUI only: save a rendered frame."""

    def close(self) -> None: ...
