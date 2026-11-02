"""OpenDRIVE (.xodr) map loader."""
import math

import numpy as np
import pytest

from doraware.interfaces import LaneRelation
from doraware.maps.registry import load_lane_map

XODR = """<OpenDRIVE>
  <road id="1" length="50.0" junction="-1">
    <link><successor elementType="road" elementId="2" contactPoint="start"/></link>
    <type s="0.0" type="town"><speed max="36" unit="km/h"/></type>
    <planView><geometry s="0.0" x="0.0" y="0.0" hdg="0.0" length="50.0"><line/></geometry></planView>
    <lanes>
      <laneOffset s="0.0" a="0.0" b="0.0" c="0.0" d="0.0"/>
      <laneSection s="0.0">
        <left><lane id="1" type="driving"><width sOffset="0.0" a="3.5" b="0" c="0" d="0"/></lane></left>
        <right><lane id="-1" type="driving"><link><successor id="-1"/></link>
          <width sOffset="0.0" a="3.5" b="0" c="0" d="0"/></lane>
          <lane id="-2" type="shoulder"><width sOffset="0.0" a="1.0" b="0" c="0" d="0"/></lane></right>
      </laneSection>
      <laneSection s="25.0">
        <left><lane id="1" type="driving"><width sOffset="0.0" a="3.5" b="0" c="0" d="0"/></lane></left>
        <right><lane id="-1" type="driving"><link><successor id="-1"/></link>
          <width sOffset="0.0" a="3.5" b="0" c="0" d="0"/></lane></right>
      </laneSection>
    </lanes>
  </road>
  <road id="2" length="20.0" junction="-1">
    <planView><geometry s="0.0" x="50.0" y="0.0" hdg="0.0" length="20.0"><arc curvature="0.05"/></geometry></planView>
    <lanes><laneSection s="0.0">
      <right><lane id="-1" type="driving"><width sOffset="0.0" a="3.5" b="0" c="0" d="0"/></lane></right>
    </laneSection></lanes>
  </road>
</OpenDRIVE>"""


def _rels(m):
    return {(int(m.lane_id[a]), int(m.lane_id[b]), LaneRelation(int(k)))
            for a, b, k in zip(m.relation_from, m.relation_to, m.relation_type)}


def test_roads_sections_lanes_and_links(tmp_path):
    f = tmp_path / "town.xodr"
    f.write_text(XODR)
    m = load_lane_map(f, "opendrive", {"resample": 0.5})
    ids = sorted(m.lane_id.tolist())
    assert ids == [1049, 1051, 1149, 1151, 2049]                     # shoulder lanes are skipped
    right = m.index_of(1049)                                         # road 1, section 0, lane -1
    c = m.centerline(right)
    assert c[0] == pytest.approx([0.0, -1.75]) and c[-1] == pytest.approx([25.0, -1.75])
    sel = np.asarray(m.point_lane) == right
    assert np.allclose(np.asarray(m.left_y)[sel], 0.0) and np.allclose(np.asarray(m.right_y)[sel], -3.5)
    left = m.centerline(m.index_of(1051))                            # lane 1 drives against s
    assert left[0] == pytest.approx([25.0, 1.75]) and left[-1] == pytest.approx([0.0, 1.75])
    assert m.speed_limit[right] == pytest.approx(10.0)
    r = _rels(m)
    assert (1049, 1149, LaneRelation.SUCCESSOR) in r                 # next section of road 1
    assert (1149, 2049, LaneRelation.SUCCESSOR) in r                 # into road 2 through the road link
    assert not any(k == LaneRelation.LEFT_NEIGHBOR for _, _, k in r)  # lane 1 runs the other way
    arc = m.centerline(m.index_of(2049))                             # right lane of a left arc: radius 21.75
    centre = np.array([50.0, 20.0])
    assert np.allclose(np.hypot(*(arc - centre).T), 21.75, atol=1e-6)


def test_spiral_and_unsupported(tmp_path):
    f = tmp_path / "spiral.xodr"
    f.write_text("""<OpenDRIVE><road id="5" length="10.0">
      <planView><geometry s="0" x="0" y="0" hdg="0" length="10.0"><spiral curvStart="0.0" curvEnd="0.1"/></geometry></planView>
      <lanes><laneSection s="0"><right><lane id="-1" type="driving"><width sOffset="0" a="2.0" b="0" c="0" d="0"/></lane></right></laneSection></lanes>
    </road></OpenDRIVE>""")
    c = load_lane_map(f, "opendrive", {"resample": 0.01}).centerline(0)
    end_heading = math.atan2(*(c[-1] - c[-2])[::-1])                 # of the last 1 cm
    assert end_heading == pytest.approx(0.5, abs=2e-3)               # heading = (k1 - k0) * L / 2
    f.write_text(f.read_text().replace('<spiral curvStart="0.0" curvEnd="0.1"/>', '<poly3 a="0" b="0" c="0" d="0"/>'))
    with pytest.raises(ValueError, match="poly3"):
        load_lane_map(f, "opendrive")
