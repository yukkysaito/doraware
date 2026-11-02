"""Ground truth of the simulated world in interface types (no Dora).

Until perception exists, the simulator provides the objects it simulates with
the same contract a perception component will use (``PerceivedObjects``), so
consumers can switch from ground truth to perception by rewiring the dataflow.
"""
from __future__ import annotations

import numpy as np

from doraware.interfaces import Header, ObjectClass, PerceivedObjects, SimulationScene


def scene_objects(scene: SimulationScene, header: Header) -> PerceivedObjects:
    """The scene's obstacles as tracked static objects (ids = obstacle index,
    probability 1, box bottom on the ground)."""
    n = len(scene.obstacles_cx)
    h = float(scene.obstacle_height)
    zeros = np.zeros(n)
    return PerceivedObjects(
        header=header, object_id=np.arange(n, dtype=np.int64),
        classification=np.full(n, int(ObjectClass.STATIC_OBSTACLE), dtype=np.int8),
        class_probability=np.ones(n), existence_probability=np.ones(n),
        x=np.asarray(scene.obstacles_cx, float), y=np.asarray(scene.obstacles_cy, float), z=np.full(n, h / 2.0),
        yaw=np.asarray(scene.obstacles_yaw, float), length=np.asarray(scene.obstacles_length, float),
        width=np.asarray(scene.obstacles_width, float), height=np.full(n, h),
        velocity_x=zeros.copy(), velocity_y=zeros.copy(), yaw_rate=zeros.copy())
