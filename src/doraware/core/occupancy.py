"""Occupancy grid <-> axis-aligned box conversion (exact, used for simulator obstacles and clearance)."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def grid_to_boxes(data: np.ndarray, resolution: float, origin: Tuple[float, float]) -> np.ndarray:
    """Cover the occupied cells of ``data[y, x]`` exactly with axis-aligned
    rectangles: horizontal runs per row, merged with identical runs of the
    following rows. Returns (N, 4) [xmin, ymin, xmax, ymax] in world coords."""
    occ = np.asarray(data) > 0
    h, w = occ.shape
    open_runs: Dict[Tuple[int, int], int] = {}   # (x0, x1) -> start row
    rects: List[Tuple[int, int, int, int]] = []
    for iy in range(h + 1):
        runs = set()
        if iy < h:
            row = occ[iy]
            if row.any():
                padded = np.concatenate([[False], row, [False]])
                d = np.diff(padded.astype(np.int8))
                starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
                runs = set(zip(starts.tolist(), ends.tolist()))
        for run in list(open_runs):
            if run not in runs:
                rects.append((run[0], open_runs.pop(run), run[1], iy))
        for run in runs:
            open_runs.setdefault(run, iy)
    out = np.array([[origin[0] + x0 * resolution, origin[1] + y0 * resolution,
                     origin[0] + x1 * resolution, origin[1] + y1 * resolution]
                    for x0, y0, x1, y1 in rects], dtype=float).reshape(-1, 4)
    return out


def boxes_to_grid(boxes: np.ndarray, shape: Tuple[int, int], resolution: float,
                  origin: Tuple[float, float]) -> np.ndarray:
    """Inverse of :func:`grid_to_boxes` (used to verify the conversion)."""
    g = np.zeros(shape, dtype=np.uint8)
    for xmin, ymin, xmax, ymax in boxes:
        x0, x1 = int(round((xmin - origin[0]) / resolution)), int(round((xmax - origin[0]) / resolution))
        y0, y1 = int(round((ymin - origin[1]) / resolution)), int(round((ymax - origin[1]) / resolution))
        g[y0:y1, x0:x1] = 1
    return g
