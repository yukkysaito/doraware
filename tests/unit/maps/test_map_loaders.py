"""Map loaders: registry, the yaml lane format (reader / writer)."""
import numpy as np
import pytest
import yaml

from doraware.core.config import load_config
from doraware.evaluation.scenarios.road import RoadScenarios
from doraware.interfaces import LaneRelation, codec
from doraware.maps.plugins.yaml_lanes import lanes_to_dict
from doraware.maps.registry import MAP_LOADERS, load_lane_map


def test_yaml_roundtrip_is_lossless(tmp_path):
    lanes = RoadScenarios().build("R2", 42, load_config()).request.lanes
    f = tmp_path / "r2.yaml"
    f.write_text(yaml.safe_dump(lanes_to_dict(lanes)))
    back = load_lane_map(f)
    assert codec.to_json_dict(back) == codec.to_json_dict(lanes)


def test_constant_width_lanes_and_relations(tmp_path):
    f = tmp_path / "two.yaml"
    f.write_text(yaml.safe_dump({"lanes": [
        {"id": 1, "speed_limit": 5.0, "width": 3.0, "centerline": [[0, 0], [10, 0]], "successors": [2],
         "left_neighbor": 3},
        {"id": 2, "speed_limit": 5.0, "width": 3.0, "centerline": [[10, 0], [20, 0]]},
        {"id": 3, "speed_limit": 5.0, "width": 3.0, "centerline": [[0, 3], [10, 3]]}]}))
    m = load_lane_map(f, "yaml", {"resample": 1.0})
    assert len(m) == 3 and np.allclose(m.centerline(0)[:, 0], np.arange(11.0))
    assert np.allclose(m.left_y[np.asarray(m.point_lane) == 0], 1.5)
    rels = set(zip(m.relation_from.tolist(), m.relation_to.tolist(), m.relation_type.tolist()))
    assert rels == {(0, 1, int(LaneRelation.SUCCESSOR)), (0, 2, int(LaneRelation.LEFT_NEIGHBOR))}


def test_errors(tmp_path):
    f = tmp_path / "bad.yaml"
    f.write_text("lanes: []\n")
    with pytest.raises(ValueError, match="no lanes"):
        load_lane_map(f)
    with pytest.raises(ValueError, match="unknown map loader"):
        load_lane_map(f, "lanelet2")
    assert "yaml" in MAP_LOADERS.names()
