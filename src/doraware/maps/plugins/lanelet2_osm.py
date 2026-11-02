"""``lanelet2``: Lanelet2 maps (OSM XML) -> ``LaneMap``, without the lanelet2 library.

* Coordinates: the node tags ``local_x`` / ``local_y`` (Autoware maps) when all
  nodes have them; otherwise latitude / longitude projected onto a local plane
  around ``origin_lat`` / ``origin_lon`` (default: the first node), which is
  accurate to centimetres over a few kilometres.
* Lanes: every relation with ``type=lanelet`` and a road subtype (``road``,
  ``highway`` or no subtype). Its ``left`` / ``right`` ways are the boundaries;
  both are resampled to the same number of stations along their length and the
  centreline is their mean. ``speed_limit`` (km/h) becomes m/s, else
  ``default_speed_limit``.
* Relations: B is a successor of A if B's boundaries start where A's end
  (shared nodes or within ``join_tolerance``); a lane that shares A's left
  (right) boundary way and runs in the same direction is A's left (right)
  neighbour.

Lane ids are the lanelet relation ids.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import numpy as np

from doraware.interfaces import LaneMap, LaneRelation

ROAD_SUBTYPES = {"road", "highway", ""}
EARTH_RADIUS = 6378137.0


class Lanelet2Loader:
    def __init__(self, resample: float = 0.5, default_speed_limit: float = 50.0 / 3.6,
                 join_tolerance: float = 0.1, origin_lat: Optional[float] = None,
                 origin_lon: Optional[float] = None):
        self.resample = float(resample)
        self.default_speed_limit = float(default_speed_limit)
        self.join_tolerance = float(join_tolerance)
        self.origin = None if origin_lat is None or origin_lon is None else (float(origin_lat), float(origin_lon))

    def load(self, path: str) -> LaneMap:
        return self.parse(ET.parse(path).getroot())

    # ------------------------------------------------------------------ parsing
    def parse(self, root: ET.Element) -> LaneMap:
        nodes = self._nodes(root)
        ways = {int(w.get("id")): [int(nd.get("ref")) for nd in w.findall("nd")] for w in root.findall("way")}
        lanelets = []
        for rel in root.findall("relation"):
            tags = {t.get("k"): t.get("v") for t in rel.findall("tag")}
            if tags.get("type") != "lanelet" or tags.get("subtype", "") not in ROAD_SUBTYPES:
                continue
            members = {m.get("role"): int(m.get("ref")) for m in rel.findall("member") if m.get("type") == "way"}
            if "left" not in members or "right" not in members:
                raise ValueError(f"lanelet {rel.get('id')}: needs a left and a right way")
            left = np.array([nodes[n] for n in ways[members["left"]]])
            right = np.array([nodes[n] for n in ways[members["right"]]])
            limit = tags.get("speed_limit")
            lanelets.append({"id": int(rel.get("id")), "left_way": members["left"], "right_way": members["right"],
                             "left": left, "right": right,
                             "speed": float(limit) / 3.6 if limit else self.default_speed_limit})
        if not lanelets:
            raise ValueError("Lanelet2 map: no road lanelets")
        return self._lane_map(lanelets)

    def _nodes(self, root: ET.Element) -> Dict[int, Tuple[float, float]]:
        raw = []
        for n in root.findall("node"):
            tags = {t.get("k"): t.get("v") for t in n.findall("tag")}
            raw.append((int(n.get("id")), tags, float(n.get("lat", "nan")), float(n.get("lon", "nan"))))
        if not raw:
            return {}
        if all("local_x" in t and "local_y" in t for _, t, _, _ in raw):
            return {i: (float(t["local_x"]), float(t["local_y"])) for i, t, _, _ in raw}
        lat0, lon0 = self.origin or (raw[0][2], raw[0][3])
        k = math.cos(math.radians(lat0))
        return {i: (math.radians(lon - lon0) * EARTH_RADIUS * k, math.radians(lat - lat0) * EARTH_RADIUS)
                for i, _, lat, lon in raw}

    def _lane_map(self, lanelets: List[dict]) -> LaneMap:
        cols = {k: [] for k in ("x", "y", "left_x", "left_y", "right_x", "right_y", "point_lane")}
        for i, ll in enumerate(lanelets):
            n = max(int(max(_length(ll["left"]), _length(ll["right"])) / self.resample), 1) + 1
            left, right = _resample_n(ll["left"], n), _resample_n(ll["right"], n)
            center = (left + right) / 2.0
            for k, v in (("x", center[:, 0]), ("y", center[:, 1]), ("left_x", left[:, 0]), ("left_y", left[:, 1]),
                         ("right_x", right[:, 0]), ("right_y", right[:, 1])):
                cols[k].append(v)
            cols["point_lane"].append(np.full(n, i))
            ll["start"], ll["end"] = center[0], center[-1]
            ll["heading"] = math.atan2(*(center[-1] - center[0])[::-1])
        rel = []
        for a, A in enumerate(lanelets):
            for b, B in enumerate(lanelets):
                if a == b:
                    continue
                if np.hypot(*(A["end"] - B["start"])) <= self.join_tolerance:
                    rel.append((a, b, LaneRelation.SUCCESSOR))
                same_dir = math.cos(A["heading"] - B["heading"]) > 0.0
                if same_dir and A["left_way"] == B["right_way"]:
                    rel.append((a, b, LaneRelation.LEFT_NEIGHBOR))
                if same_dir and A["right_way"] == B["left_way"]:
                    rel.append((a, b, LaneRelation.RIGHT_NEIGHBOR))
        r = np.array([(a, b, int(k)) for a, b, k in rel], dtype=np.int64).reshape(-1, 3)
        m = LaneMap(lane_id=np.array([ll["id"] for ll in lanelets], dtype=np.int64),
                    speed_limit=np.array([ll["speed"] for ll in lanelets]),
                    point_lane=np.concatenate(cols["point_lane"]).astype(np.int64),
                    **{k: np.concatenate(v) for k, v in cols.items() if k != "point_lane"},
                    relation_from=r[:, 0], relation_to=r[:, 1], relation_type=r[:, 2].astype(np.int8))
        m.validate()
        return m


def _length(p: np.ndarray) -> float:
    return float(np.sum(np.hypot(*np.diff(p, axis=0).T))) if len(p) > 1 else 0.0


def _resample_n(p: np.ndarray, n: int) -> np.ndarray:
    """``n`` points equally spaced along the polyline (by arc length)."""
    s = np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))])
    if s[-1] <= 0.0:
        return np.repeat(p[:1], n, axis=0)
    sd = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(sd, s, p[:, 0]), np.interp(sd, s, p[:, 1])], axis=1)
