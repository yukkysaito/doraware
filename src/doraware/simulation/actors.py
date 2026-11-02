"""Scripted traffic participants (actors) of a SimulationScene (no Dora).

Motion model (the contract of ``SimulationScene``, implemented identically in
the Godot backend): position and yaw are linearly interpolated between
consecutive waypoints by sim time; yaw takes the shorter way round. Before
the first waypoint the actor stands at it, after the last one it stands still.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from doraware.core.geometry import angle_diff, wrap_angle
from doraware.interfaces import SimulationScene


@dataclass
class ActorState:
    x: float
    y: float
    yaw: float
    velocity_x: float
    velocity_y: float
    yaw_rate: float


def actor_waypoints(scene: SimulationScene, i: int) -> np.ndarray:
    """Waypoints of actor ``i`` as rows (t, x, y, yaw), sorted by time."""
    sel = np.nonzero(np.asarray(scene.waypoint_actor) == i)[0]
    if len(sel) == 0:
        raise ValueError(f"actor {i} has no waypoints")
    w = np.stack([np.asarray(scene.waypoint_time)[sel], np.asarray(scene.waypoint_x)[sel],
                  np.asarray(scene.waypoint_y)[sel], np.asarray(scene.waypoint_yaw)[sel]], axis=1)
    return w[np.argsort(w[:, 0], kind="stable")]


def actor_state(waypoints: np.ndarray, t: float) -> ActorState:
    w = waypoints
    if len(w) == 1 or t <= w[0, 0]:
        return ActorState(w[0, 1], w[0, 2], w[0, 3], 0.0, 0.0, 0.0)
    if t >= w[-1, 0]:
        return ActorState(w[-1, 1], w[-1, 2], w[-1, 3], 0.0, 0.0, 0.0)
    k = int(np.searchsorted(w[:, 0], t, side="right")) - 1
    (t0, x0, y0, a0), (t1, x1, y1, a1) = w[k], w[k + 1]
    dt = t1 - t0
    s = (t - t0) / dt
    da = float(angle_diff(a1, a0))
    return ActorState(x0 + s * (x1 - x0), y0 + s * (y1 - y0), float(wrap_angle(a0 + s * da)),
                      (x1 - x0) / dt, (y1 - y0) / dt, da / dt)


def actor_states(scene: SimulationScene, t: float) -> List[ActorState]:
    return [actor_state(actor_waypoints(scene, i), t) for i in range(len(scene.actor_id))]
