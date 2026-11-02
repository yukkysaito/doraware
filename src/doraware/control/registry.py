"""Controller plugins, selected by ``control.controller.type``.

A controller plugin is a factory ``(params, cfg, vehicle) -> Controller``:
``params`` is its own config section ``control.controller.<type>``, ``cfg``
the full system config and ``vehicle`` the scenario's VehicleParameters.
Built-in plugins live in ``control/plugins/<name>/``; separately installed
packages register under the ``doraware.controllers`` entry-point group.
"""
from __future__ import annotations

from typing import Any, Dict, Protocol, Tuple

from doraware.core.registry import Registry
from doraware.interfaces import ControlCommand, ControlDebug, Trajectory, VehicleParameters, VehicleState


class Controller(Protocol):
    def on_trajectory(self, traj: Trajectory) -> bool:
        """Offer a new trajectory; False if it was ignored (stale or malformed)."""

    def step(self, state: VehicleState, t_now_ns: int) -> Tuple[ControlCommand, ControlDebug]:
        """Command for sim time ``t_now_ns`` (one controller tick)."""


CONTROLLERS: Registry[Controller] = Registry("controller", "doraware.controllers")


@CONTROLLERS.register("mpc")
def _mpc(params: Dict[str, Any], cfg: Dict[str, Any], vehicle: VehicleParameters) -> Controller:
    from doraware.control.plugins.mpc.follower import TrajectoryFollower
    return TrajectoryFollower(cfg, vehicle, params)


def make_controller(cfg: Dict[str, Any], vehicle: VehicleParameters) -> Controller:
    return CONTROLLERS.create_from_section(cfg["control"]["controller"], cfg, vehicle)
