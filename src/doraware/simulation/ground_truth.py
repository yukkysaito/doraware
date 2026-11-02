"""Ground truth of the simulated world in interface types (no Dora).

Until perception exists, the simulator provides the objects it simulates with
the same contract a perception component will use (``PerceivedObjects``), so
consumers can switch from ground truth to perception by rewiring the dataflow.
"""
from __future__ import annotations

import numpy as np

from doraware.interfaces import Header, ObjectClass, PerceivedObjects, SimulationScene, to_sec
from doraware.simulation.actors import actor_states

ACTOR_ID_OFFSET = 1_000_000   # ground-truth id of actor k = ACTOR_ID_OFFSET + actor_id[k]


def scene_objects(scene: SimulationScene, header: Header) -> PerceivedObjects:
    """Objects of the scene at ``header.stamp``: the static obstacles (ids =
    obstacle index, class STATIC_OBSTACLE, box bottom on the ground) and the
    actors at their scripted pose and velocity (ids = ACTOR_ID_OFFSET + actor id).
    All tracked, with probability 1."""
    n = len(scene.obstacles_cx)
    h = float(scene.obstacle_height)
    states = actor_states(scene, to_sec(header.stamp))
    m = len(states)
    a = lambda f: np.array([getattr(s, f) for s in states], dtype=float)   # noqa: E731
    return PerceivedObjects(
        header=header,
        object_id=np.concatenate([np.arange(n), ACTOR_ID_OFFSET + np.asarray(scene.actor_id)]).astype(np.int64),
        classification=np.concatenate([np.full(n, int(ObjectClass.STATIC_OBSTACLE)),
                                       np.asarray(scene.actor_class)]).astype(np.int8),
        class_probability=np.ones(n + m), existence_probability=np.ones(n + m),
        x=np.concatenate([np.asarray(scene.obstacles_cx, float), a("x")]),
        y=np.concatenate([np.asarray(scene.obstacles_cy, float), a("y")]),
        z=np.concatenate([np.full(n, h / 2.0), np.asarray(scene.actor_height, float) / 2.0]),
        yaw=np.concatenate([np.asarray(scene.obstacles_yaw, float), a("yaw")]),
        length=np.concatenate([np.asarray(scene.obstacles_length, float), np.asarray(scene.actor_length, float)]),
        width=np.concatenate([np.asarray(scene.obstacles_width, float), np.asarray(scene.actor_width, float)]),
        height=np.concatenate([np.full(n, h), np.asarray(scene.actor_height, float)]),
        velocity_x=np.concatenate([np.zeros(n), a("velocity_x")]),
        velocity_y=np.concatenate([np.zeros(n), a("velocity_y")]),
        yaw_rate=np.concatenate([np.zeros(n), a("yaw_rate")]))
