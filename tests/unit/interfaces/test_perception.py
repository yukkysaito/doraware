"""Perception message contract and the simulator's ground truth in that contract."""
import math

import numpy as np
import pytest

from doraware.interfaces import (UNTRACKED, Header, MESSAGE_TYPES, ObjectClass, PerceivedObjects, Pose2D,
                                 SimulationScene, codec)
import doraware.simulation.plugins.godot.frame as gf
from doraware.simulation.ground_truth import scene_objects


def _scene():
    return SimulationScene(obstacles_cx=np.array([4.0, -2.0]), obstacles_cy=np.array([1.0, 3.0]),
                           obstacles_yaw=np.array([0.0, 0.5]), obstacles_length=np.array([2.0, 4.5]),
                           obstacles_width=np.array([1.0, 1.8]), obstacle_height=1.5, initial_pose=Pose2D())


def test_roundtrip_and_registration():
    objs = scene_objects(_scene(), Header(stamp=7))
    back = codec.decode(codec.encode(objs), PerceivedObjects)
    assert codec.to_json_dict(back) == codec.to_json_dict(objs)
    assert back.object_id.dtype == np.int64 and MESSAGE_TYPES["PerceivedObjects"] is PerceivedObjects
    assert codec.json_schema(PerceivedObjects)["properties"]["object_id"]["items"]["type"] == "integer"


def test_validate_requires_equal_lengths():
    PerceivedObjects().validate()                      # empty is valid
    objs = scene_objects(_scene(), Header())
    objs.validate()
    objs.width = np.zeros(1)
    with pytest.raises(ValueError, match="width"):
        objs.validate()


def test_ground_truth_of_the_scene():
    o = scene_objects(_scene(), Header(stamp=7))
    assert len(o) == 2 and o.header.stamp == 7 and list(o.object_id) == [0, 1]
    assert all(c == ObjectClass.STATIC_OBSTACLE for c in o.classification)
    assert np.allclose(o.z, 0.75) and np.allclose(o.height, 1.5)          # box bottom on the ground
    assert np.allclose(o.existence_probability, 1.0) and not np.any(o.velocity_x)
    assert UNTRACKED == -1


def test_object_boxes_in_the_godot_frame():
    o = scene_objects(_scene(), Header())
    boxes = gf.object_boxes(o)
    assert boxes[1]["x"] == -2.0 and boxes[1]["z"] == -3.0              # Z_g = -y
    assert math.isclose(boxes[1]["rot_y"], 0.5) and boxes[1]["cls"] == int(ObjectClass.STATIC_OBSTACLE)
    assert gf.viz_request(objects=o)["objects"] == boxes
    assert "objects" not in gf.viz_request()                            # None = keep what is drawn
