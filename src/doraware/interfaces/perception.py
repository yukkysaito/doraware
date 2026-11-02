"""Perception messages: objects around the ego vehicle."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from doraware.interfaces.common import F64Array, Header, I8Array, I64Array

UNTRACKED = -1   # object_id of an object that is detected but not tracked


class ObjectClass(IntEnum):
    UNKNOWN = 0
    CAR = 1
    TRUCK = 2
    BUS = 3
    MOTORCYCLE = 4
    BICYCLE = 5
    PEDESTRIAN = 6
    ANIMAL = 7
    STATIC_OBSTACLE = 8      # anything that does not move (walls, parked objects, debris)


_ARRAYS = ("object_id", "classification", "class_probability", "existence_probability", "x", "y", "z", "yaw",
           "length", "width", "height", "velocity_x", "velocity_y", "yaw_rate")


@dataclass
class PerceivedObjects:
    """Objects around the vehicle at ``header.stamp``, in ``header.frame_id`` (map).

    One entry per object, in parallel arrays of equal length (structure of
    arrays: compact, zero-copy Arrow lists). The same message serves detection
    (``object_id = UNTRACKED``) and tracking (``object_id`` stable while the
    object is tracked). Predicted motion is a separate message.

    * ``classification`` (:class:`ObjectClass`) with ``class_probability``;
      ``existence_probability``: confidence that the object exists, [0, 1].
    * Bounding box: centre ``x``, ``y``, ``z`` [m] (``z`` = height of the box
      centre above the ground), heading ``yaw`` [rad] of the box x axis,
      ``length`` (along x), ``width``, ``height`` [m].
    * Motion in the map frame: ``velocity_x``, ``velocity_y`` [m/s],
      ``yaw_rate`` [rad/s]. Static objects have zeros.

    The simulator publishes ground truth with this contract
    (``sim_bridge/ground_truth_objects``), so a perception component can replace
    it without changing its consumers.
    """
    header: Header = field(default_factory=Header)
    object_id: I64Array = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    classification: I8Array = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    class_probability: F64Array = field(default_factory=lambda: np.zeros(0))
    existence_probability: F64Array = field(default_factory=lambda: np.zeros(0))
    x: F64Array = field(default_factory=lambda: np.zeros(0))
    y: F64Array = field(default_factory=lambda: np.zeros(0))
    z: F64Array = field(default_factory=lambda: np.zeros(0))
    yaw: F64Array = field(default_factory=lambda: np.zeros(0))
    length: F64Array = field(default_factory=lambda: np.zeros(0))
    width: F64Array = field(default_factory=lambda: np.zeros(0))
    height: F64Array = field(default_factory=lambda: np.zeros(0))
    velocity_x: F64Array = field(default_factory=lambda: np.zeros(0))
    velocity_y: F64Array = field(default_factory=lambda: np.zeros(0))
    yaw_rate: F64Array = field(default_factory=lambda: np.zeros(0))

    def __len__(self) -> int:
        return int(len(self.x))

    def validate(self) -> None:
        """Raise ValueError unless all per-object arrays have the same length."""
        n = len(self)
        bad = [name for name in _ARRAYS if len(getattr(self, name)) != n]
        if bad:
            raise ValueError(f"PerceivedObjects arrays differ in length from x ({n}): {bad}")
