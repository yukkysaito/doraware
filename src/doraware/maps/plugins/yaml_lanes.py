"""``yaml``: a small, documented lane map file format (reader and writer).

    lanes:
      - id: 100
        speed_limit: 3.0                    # [m/s]
        width: 3.5                          # [m] constant width, or give left/right explicitly
        centerline: [[0.0, 0.0], [45.0, 0.0]]
        # left:  [[x, y], ...]              # optional boundaries, same number of points
        # right: [[x, y], ...]
        successors: [101]                   # optional lane ids
        left_neighbor: 102                  # optional lane id (same direction)
        right_neighbor: null

Coordinates are map-frame metres. Centrelines are resampled every
``resample`` metres (0 keeps the given points); constant-width boundaries are
offset perpendicular to the centreline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import yaml

from doraware.interfaces import LaneMap, LaneRelation


class YamlLaneLoader:
    def __init__(self, resample: float = 0.5):
        self.resample = float(resample)

    def load(self, path: str) -> LaneMap:
        doc = yaml.safe_load(Path(path).read_text()) or {}
        return lanes_from_dict(doc, self.resample)


def _resample(c: np.ndarray, ds: float) -> np.ndarray:
    if ds <= 0 or len(c) < 2:
        return c
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(c, axis=0).T))])
    sd = np.append(np.arange(0.0, s[-1], ds), s[-1])
    return np.stack([np.interp(sd, s, c[:, 0]), np.interp(sd, s, c[:, 1])], axis=1)


def lanes_from_dict(doc: Dict[str, Any], resample: float = 0.5) -> LaneMap:
    lanes: List[Dict[str, Any]] = list(doc.get("lanes") or [])
    if not lanes:
        raise ValueError("lane map: no lanes")
    ids = [int(lane["id"]) for lane in lanes]
    if len(set(ids)) != len(ids):
        raise ValueError("lane map: duplicate lane ids")
    index = {lid: i for i, lid in enumerate(ids)}
    cols = {k: [] for k in ("point_lane", "x", "y", "left_x", "left_y", "right_x", "right_y")}
    rel_from, rel_to, rel_type = [], [], []
    for i, lane in enumerate(lanes):
        c = np.asarray(lane["centerline"], dtype=float).reshape(-1, 2)
        if "left" in lane or "right" in lane:
            left = np.asarray(lane["left"], dtype=float).reshape(-1, 2)
            right = np.asarray(lane["right"], dtype=float).reshape(-1, 2)
            if not len(left) == len(right) == len(c):
                raise ValueError(f"lane {ids[i]}: centerline, left and right need the same number of points")
        else:
            c = _resample(c, resample)
            d = np.gradient(c, axis=0)
            yaw = np.arctan2(d[:, 1], d[:, 0])
            half = float(lane.get("width", 3.5)) / 2.0
            n = np.stack([-np.sin(yaw), np.cos(yaw)], axis=1) * half
            left, right = c + n, c - n
        for k, v in (("x", c[:, 0]), ("y", c[:, 1]), ("left_x", left[:, 0]), ("left_y", left[:, 1]),
                     ("right_x", right[:, 0]), ("right_y", right[:, 1])):
            cols[k].append(v)
        cols["point_lane"].append(np.full(len(c), i))
        for key, kind in (("successors", LaneRelation.SUCCESSOR),):
            for to in lane.get(key) or []:
                rel_from.append(i), rel_to.append(index[int(to)]), rel_type.append(int(kind))
        for key, kind in (("left_neighbor", LaneRelation.LEFT_NEIGHBOR), ("right_neighbor", LaneRelation.RIGHT_NEIGHBOR)):
            if lane.get(key) is not None:
                rel_from.append(i), rel_to.append(index[int(lane[key])]), rel_type.append(int(kind))
    m = LaneMap(lane_id=np.array(ids, dtype=np.int64),
                speed_limit=np.array([float(lane["speed_limit"]) for lane in lanes]),
                point_lane=np.concatenate(cols["point_lane"]).astype(np.int64),
                **{k: np.concatenate(v) for k, v in cols.items() if k != "point_lane"},
                relation_from=np.array(rel_from, dtype=np.int64), relation_to=np.array(rel_to, dtype=np.int64),
                relation_type=np.array(rel_type, dtype=np.int8))
    m.validate()
    return m


def lanes_to_dict(m: LaneMap) -> Dict[str, Any]:
    """Inverse of :func:`lanes_from_dict` with explicit boundaries (lossless)."""
    out = []
    for i in range(len(m)):
        sel = np.asarray(m.point_lane) == i
        pts = lambda xs, ys: np.stack([np.asarray(xs)[sel], np.asarray(ys)[sel]], axis=1).tolist()  # noqa: E731
        lane: Dict[str, Any] = {"id": int(m.lane_id[i]), "speed_limit": float(m.speed_limit[i]),
                                "centerline": pts(m.x, m.y), "left": pts(m.left_x, m.left_y),
                                "right": pts(m.right_x, m.right_y)}
        for f, t, k in zip(m.relation_from, m.relation_to, m.relation_type):
            if int(f) != i:
                continue
            to = int(m.lane_id[int(t)])
            if k == LaneRelation.SUCCESSOR:
                lane.setdefault("successors", []).append(to)
            elif k == LaneRelation.LEFT_NEIGHBOR:
                lane["left_neighbor"] = to
            else:
                lane["right_neighbor"] = to
        out.append(lane)
    return {"lanes": out}
