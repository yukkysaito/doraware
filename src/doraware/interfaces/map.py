"""Map messages."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from doraware.interfaces.common import Header, U8Array


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
