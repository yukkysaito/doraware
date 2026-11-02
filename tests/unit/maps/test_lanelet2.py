"""Lanelet2 (OSM XML) map loader."""
import math

import numpy as np
import pytest

from doraware.core.config import REPO_ROOT
from doraware.interfaces import LaneRelation
from doraware.maps.registry import load_lane_map

MAP = REPO_ROOT / "data" / "maps" / "two_lane_road.osm"


def _rels(m):
    ids = m.lane_id
    return {(int(ids[a]), int(ids[b]), LaneRelation(int(k)))
            for a, b, k in zip(m.relation_from, m.relation_to, m.relation_type)}


def test_two_lane_road():
    m = load_lane_map(MAP, "lanelet2", {"resample": 1.0})
    assert sorted(m.lane_id.tolist()) == [1001, 1002, 1003, 1004]
    i = m.index_of(1001)                                      # right lane, first segment
    c = m.centerline(i)
    assert c[0] == pytest.approx([0.0, 0.0]) and c[-1] == pytest.approx([25.0, 0.0])
    sel = np.asarray(m.point_lane) == i
    assert np.allclose(np.asarray(m.left_y)[sel], 1.75) and np.allclose(np.asarray(m.right_y)[sel], -1.75)
    assert m.speed_limit[i] == pytest.approx(30 / 3.6)
    r = _rels(m)
    assert (1001, 1003, LaneRelation.SUCCESSOR) in r and (1002, 1004, LaneRelation.SUCCESSOR) in r
    assert (1001, 1002, LaneRelation.LEFT_NEIGHBOR) in r and (1002, 1001, LaneRelation.RIGHT_NEIGHBOR) in r
    assert (1003, 1001, LaneRelation.SUCCESSOR) not in r


def test_latlon_projection(tmp_path):
    # same geometry without local_x / local_y: projected from lat / lon around the first node
    lat0, lon0 = 35.0, 139.0
    dlon = math.degrees(25.0 / (6378137.0 * math.cos(math.radians(lat0))))
    dlat = math.degrees(3.5 / 6378137.0)
    f = tmp_path / "ll.osm"
    f.write_text(f"""<osm>
  <node id="1" lat="{lat0}" lon="{lon0}"/><node id="2" lat="{lat0}" lon="{lon0 + dlon}"/>
  <node id="3" lat="{lat0 + dlat}" lon="{lon0}"/><node id="4" lat="{lat0 + dlat}" lon="{lon0 + dlon}"/>
  <way id="10"><nd ref="1"/><nd ref="2"/></way><way id="11"><nd ref="3"/><nd ref="4"/></way>
  <relation id="7"><member type="way" role="left" ref="11"/><member type="way" role="right" ref="10"/>
    <tag k="type" v="lanelet"/></relation>
</osm>""")
    m = load_lane_map(f, "lanelet2")
    c = m.centerline(0)
    assert c[0] == pytest.approx([0.0, 1.75], abs=1e-6) and c[-1] == pytest.approx([25.0, 1.75], abs=1e-3)


def test_errors(tmp_path):
    f = tmp_path / "empty.osm"
    f.write_text("<osm></osm>")
    with pytest.raises(ValueError, match="no road lanelets"):
        load_lane_map(f, "lanelet2")
