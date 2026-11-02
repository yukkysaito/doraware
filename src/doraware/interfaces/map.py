"""Map messages: occupancy grids and vectorized lane maps."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np

from doraware.interfaces.common import F64Array, Header, I8Array, I64Array, U8Array


# ================================================================= map
@dataclass
class OccupancyGrid:
    """Row-major grid ``data[iy * width + ix]`` (0 free, 1 occupied); cell
    (ix, iy) covers [origin + ix*res, origin + (ix+1)*res) in the map frame.
    Space outside the grid is unknown and treated as not drivable."""
    header: Header = field(default_factory=Header)
    resolution: float = 0.1
    width: int = 0
    height: int = 0
    origin_x: float = 0.0
    origin_y: float = 0.0
    data: U8Array = field(default_factory=lambda: np.zeros(0, dtype=np.uint8))

    def as_2d(self) -> np.ndarray:
        return np.asarray(self.data, dtype=np.uint8).reshape(self.height, self.width)


# ============================================================ lane map
class LaneRelation(IntEnum):
    SUCCESSOR = 0          # driving continues from `from` into `to`
    LEFT_NEIGHBOR = 1      # `to` is the adjacent lane on the left of `from` (same direction)
    RIGHT_NEIGHBOR = 2     # `to` is the adjacent lane on the right of `from`


_LANE_ARRAYS = ("lane_id", "speed_limit")
_POINT_ARRAYS = ("point_lane", "x", "y", "left_x", "left_y", "right_x", "right_y")
_RELATION_ARRAYS = ("relation_from", "relation_to", "relation_type")


@dataclass
class LaneMap:
    """Vectorized lane network in the map frame, independent of the map format
    (Lanelet2, OpenDRIVE, ... are converted into it by a loader).

    * **Lanes** (one entry per lane): stable ``lane_id`` and ``speed_limit`` [m/s].
    * **Points** (one row per centreline sample, grouped by lane, in driving
      order): ``point_lane`` is the lane's index in the lane arrays; ``x``, ``y``
      the centreline; ``left_*`` / ``right_*`` the lane boundary at the same
      station (left / right in the driving direction).
    * **Relations**: lane ``relation_to`` is the ``LaneRelation`` of lane
      ``relation_from`` (indices into the lane arrays).

    Learned planners consume exactly this kind of polyline map; an empty map
    means "no lane information" (free-space planning, e.g. parking).
    """
    header: Header = field(default_factory=Header)
    lane_id: I64Array = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    speed_limit: F64Array = field(default_factory=lambda: np.zeros(0))
    point_lane: I64Array = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    x: F64Array = field(default_factory=lambda: np.zeros(0))
    y: F64Array = field(default_factory=lambda: np.zeros(0))
    left_x: F64Array = field(default_factory=lambda: np.zeros(0))
    left_y: F64Array = field(default_factory=lambda: np.zeros(0))
    right_x: F64Array = field(default_factory=lambda: np.zeros(0))
    right_y: F64Array = field(default_factory=lambda: np.zeros(0))
    relation_from: I64Array = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    relation_to: I64Array = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    relation_type: I8Array = field(default_factory=lambda: np.zeros(0, dtype=np.int8))

    def __len__(self) -> int:
        return int(len(self.lane_id))

    def index_of(self, lane_id: int) -> int:
        hits = np.nonzero(np.asarray(self.lane_id) == lane_id)[0]
        if len(hits) != 1:
            raise KeyError(f"lane {lane_id} not in the map")
        return int(hits[0])

    def centerline(self, i: int) -> np.ndarray:
        """Centreline of lane index ``i`` as an (n, 2) array."""
        sel = np.asarray(self.point_lane) == i
        return np.stack([np.asarray(self.x)[sel], np.asarray(self.y)[sel]], axis=1)

    def validate(self) -> None:
        """Raise ValueError if array groups differ in length or indices are out of range."""
        for group in (_LANE_ARRAYS, _POINT_ARRAYS, _RELATION_ARRAYS):
            n = len(getattr(self, group[0]))
            bad = [name for name in group if len(getattr(self, name)) != n]
            if bad:
                raise ValueError(f"LaneMap arrays differ in length from {group[0]}: {bad}")
        for name in ("point_lane", "relation_from", "relation_to"):
            idx = np.asarray(getattr(self, name))
            if len(idx) and (idx.min() < 0 or idx.max() >= len(self)):
                raise ValueError(f"LaneMap.{name} has indices outside the lane arrays")
