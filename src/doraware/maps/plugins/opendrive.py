"""``opendrive``: OpenDRIVE (.xodr) -> ``LaneMap`` (a documented subset, no extra library).

Supported:

* reference line (``planView``): ``line``, ``arc`` and ``spiral`` geometries
  (spirals integrated numerically), sampled every ``resample`` metres;
* ``laneOffset`` and lane ``width`` polynomials (a + b ds + c ds^2 + d ds^3);
* every ``laneSection`` of a road; lanes of type ``driving`` become lanes.
  Right lanes (negative ids) run along the reference line, left lanes
  (positive ids) against it, so their points are reversed into driving order;
* relations: successor lanes in the next section of the same road (lane
  ``link/successor``), and across roads through the road ``link`` (road to
  road; junction connections are not followed); left / right neighbours are
  adjacent driving lanes of the same section and direction.

Not supported: ``poly3`` / ``paramPoly3`` geometries (an error names them),
elevation, superelevation, lane heights, junction connecting roads, objects
and signals. Lane ids are ``road_id * 1000 + section_index * 100 + 50 + lane_id``
(``lane_id`` in -49..49; a non-numeric road id is hashed with CRC32). Speed limits come from the road ``type/speed``
(converted from km/h or mph), else ``default_speed_limit``.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from doraware.interfaces import LaneMap, LaneRelation

SPEED_UNITS = {"km/h": 1 / 3.6, "mph": 0.44704, "m/s": 1.0}


@dataclass
class _Lane:
    uid: int
    road: str
    section: int
    lane: int                       # OpenDRIVE lane id (negative = right)
    speed: float
    center: np.ndarray
    left: np.ndarray                # left / right boundary in driving direction
    right: np.ndarray
    successors: List[int] = field(default_factory=list)   # OpenDRIVE lane ids in the next section


class OpenDriveLoader:
    def __init__(self, resample: float = 0.5, default_speed_limit: float = 50.0 / 3.6):
        self.ds = float(resample)
        self.default_speed_limit = float(default_speed_limit)

    def load(self, path: str) -> LaneMap:
        return self.parse(ET.parse(path).getroot())

    def parse(self, root: ET.Element) -> LaneMap:
        lanes: List[_Lane] = []
        road_links: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        for road in root.findall("road"):
            rid = road.get("id")
            link = road.find("link")
            succ = link.find("successor") if link is not None else None
            road_links[rid] = (succ.get("elementId") if succ is not None and succ.get("elementType") == "road"
                               else None, succ.get("contactPoint") if succ is not None else None)
            lanes += self._road(road)
        if not lanes:
            raise ValueError("OpenDRIVE: no driving lanes")
        return self._lane_map(lanes, road_links)

    # ------------------------------------------------------------- one road
    def _road(self, road: ET.Element) -> List[_Lane]:
        rid = road.get("id")
        length = float(road.get("length"))
        s = np.append(np.arange(0.0, length, self.ds), length)
        x, y, hdg = self._reference_line(road, s)
        offset = _poly_at(road.findall("lanes/laneOffset"), s)
        speed = self._speed(road)
        sections = road.findall("lanes/laneSection")
        s_starts = [float(sec.get("s")) for sec in sections] + [length]
        out = []
        for k, sec in enumerate(sections):
            sel = (s >= s_starts[k] - 1e-9) & (s <= s_starts[k + 1] + 1e-9)
            ds_local = s[sel] - s_starts[k]
            nx, ny = -np.sin(hdg[sel]), np.cos(hdg[sel])          # left normal of the reference line
            base = offset[sel]
            for side, sign in (("right", -1), ("left", 1)):
                group = sec.find(side)
                if group is None:
                    continue
                lanes = sorted(group.findall("lane"), key=lambda e: abs(int(e.get("id"))))
                inner = base.copy()                                 # boundary nearer the reference line
                for ln in lanes:
                    lid = int(ln.get("id"))
                    width = _poly_at(ln.findall("width"), ds_local, absolute=False)
                    outer = inner + sign * width
                    if ln.get("type") == "driving":
                        mid = (inner + outer) / 2.0
                        pts = lambda t: np.stack([x[sel] + t * nx, y[sel] + t * ny], axis=1)  # noqa: E731
                        center, a, b = pts(mid), pts(inner), pts(outer)
                        if sign < 0:          # right lane: drives along s; inner boundary is on its left
                            left, right = a, b
                        else:                 # left lane: drives against s
                            center, left, right = center[::-1], a[::-1], b[::-1]
                        succ = [int(e.get("id")) for e in ln.findall("link/successor")]
                        base_id = int(rid) if rid.lstrip("-").isdigit() else zlib.crc32(rid.encode()) % 10 ** 6
                        uid = base_id * 1000 + k * 100 + 50 + lid
                        out.append(_Lane(uid, rid, k, lid, speed, center, left, right, succ))
                    inner = outer
        return out

    def _reference_line(self, road: ET.Element, s: np.ndarray):
        geoms = road.findall("planView/geometry")
        x, y, h = np.zeros_like(s), np.zeros_like(s), np.zeros_like(s)
        starts = [float(g.get("s")) for g in geoms] + [math.inf]
        for i, g in enumerate(geoms):
            sel = (s >= starts[i] - 1e-9) & (s < starts[i + 1] - 1e-9) if i + 1 < len(geoms) else s >= starts[i] - 1e-9
            u = s[sel] - starts[i]
            x0, y0, h0 = float(g.get("x")), float(g.get("y")), float(g.get("hdg"))
            kind = list(g)[0]
            if kind.tag == "line":
                xs, ys, hs = x0 + u * math.cos(h0), y0 + u * math.sin(h0), np.full_like(u, h0)
            elif kind.tag == "arc":
                k = float(kind.get("curvature"))
                hs = h0 + k * u
                if abs(k) < 1e-12:
                    xs, ys = x0 + u * math.cos(h0), y0 + u * math.sin(h0)
                else:
                    xs, ys = x0 + (np.sin(hs) - math.sin(h0)) / k, y0 - (np.cos(hs) - math.cos(h0)) / k
            elif kind.tag == "spiral":
                k0, k1 = float(kind.get("curvStart")), float(kind.get("curvEnd"))
                L = float(g.get("length"))
                xs, ys, hs = _spiral(x0, y0, h0, k0, (k1 - k0) / L, u)
            else:
                raise ValueError(f"OpenDRIVE: geometry {kind.tag!r} of road {road.get('id')} is not supported")
            x[sel], y[sel], h[sel] = xs, ys, hs
        return x, y, h

    def _speed(self, road: ET.Element) -> float:
        sp = road.find("type/speed")
        if sp is None or sp.get("max") in (None, "no limit", "undefined"):
            return self.default_speed_limit
        return float(sp.get("max")) * SPEED_UNITS.get(sp.get("unit", "m/s"), 1.0)

    # ------------------------------------------------------------- the map
    def _lane_map(self, lanes: List[_Lane], road_links) -> LaneMap:
        index = {(ln.road, ln.section, ln.lane): i for i, ln in enumerate(lanes)}
        last_section = {}
        for ln in lanes:
            last_section[ln.road] = max(last_section.get(ln.road, 0), ln.section)
        rel = []
        for i, ln in enumerate(lanes):
            forward = ln.lane < 0
            for s_id in ln.successors:
                if not forward:
                    continue     # left lanes: their "successor" lies against the driving direction
                if ln.section < last_section[ln.road]:
                    j = index.get((ln.road, ln.section + 1, s_id))
                else:
                    nxt, contact = road_links.get(ln.road, (None, None))
                    j = index.get((nxt, 0, s_id)) if nxt is not None and contact == "start" else None
                if j is not None:
                    rel.append((i, j, LaneRelation.SUCCESSOR))
            for other, kind in ((ln.lane + (1 if forward else -1), LaneRelation.LEFT_NEIGHBOR),
                                (ln.lane - (1 if forward else -1), LaneRelation.RIGHT_NEIGHBOR)):
                j = index.get((ln.road, ln.section, other))
                if j is not None and (lanes[j].lane < 0) == forward:
                    rel.append((i, j, kind))
        r = np.array([(a, b, int(k)) for a, b, k in rel], dtype=np.int64).reshape(-1, 3)
        cat = lambda f: np.concatenate([getattr(ln, f) for ln in lanes])  # noqa: E731
        m = LaneMap(lane_id=np.array([ln.uid for ln in lanes], dtype=np.int64),
                    speed_limit=np.array([ln.speed for ln in lanes]),
                    point_lane=np.concatenate([np.full(len(ln.center), i) for i, ln in enumerate(lanes)]).astype(np.int64),
                    x=cat("center")[:, 0], y=cat("center")[:, 1], left_x=cat("left")[:, 0], left_y=cat("left")[:, 1],
                    right_x=cat("right")[:, 0], right_y=cat("right")[:, 1],
                    relation_from=r[:, 0], relation_to=r[:, 1], relation_type=r[:, 2].astype(np.int8))
        m.validate()
        return m


def _poly_at(records, s: np.ndarray, absolute: bool = True) -> np.ndarray:
    """Piecewise cubic a + b ds + c ds^2 + d ds^3 (records sorted by their start
    ``s`` / ``sOffset``); zero before the first record."""
    out = np.zeros_like(s, dtype=float)
    key = "s" if absolute else "sOffset"
    recs = sorted(records, key=lambda e: float(e.get(key, 0.0)))
    for i, e in enumerate(recs):
        s0 = float(e.get(key, 0.0))
        s1 = float(recs[i + 1].get(key, 0.0)) if i + 1 < len(recs) else math.inf
        sel = (s >= s0 - 1e-9) & (s < s1)
        d = s[sel] - s0
        a, b, c, dd = (float(e.get(k, 0.0)) for k in ("a", "b", "c", "d"))
        out[sel] = a + b * d + c * d * d + dd * d ** 3
    return out


def _spiral(x0: float, y0: float, h0: float, k0: float, dk: float, u: np.ndarray):
    """Clothoid with curvature k0 + dk * u, integrated in fine steps."""
    if len(u) == 0:
        return u.copy(), u.copy(), u.copy()
    n = max(int(math.ceil(u.max() / 0.01)), 1)
    t = np.linspace(0.0, u.max(), n + 1)
    h = h0 + k0 * t + 0.5 * dk * t * t
    dt = np.diff(t)
    hm = (h[:-1] + h[1:]) / 2.0
    xs = np.concatenate([[x0], x0 + np.cumsum(np.cos(hm) * dt)])
    ys = np.concatenate([[y0], y0 + np.cumsum(np.sin(hm) * dt)])
    return np.interp(u, t, xs), np.interp(u, t, ys), h0 + k0 * u + 0.5 * dk * u * u
