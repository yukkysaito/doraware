"""Scripted actors: motion model, ground truth and the Godot reset request."""
import math

import numpy as np
import pytest

from doraware.interfaces import Header, ObjectClass, Pose2D, SimulationScene, codec, to_ns
import doraware.simulation.plugins.godot.frame as gf
from doraware.simulation.actors import actor_state, actor_states, actor_waypoints
from doraware.simulation.ground_truth import ACTOR_ID_OFFSET, scene_objects


def scene_with_actors():
    # actor 0 (id 7, car): waits at (0, 5), drives to (10, 5) between t = 2 s and 4 s
    # actor 1 (id 9, pedestrian): one waypoint (stands still); waypoints given out of order
    return SimulationScene(
        obstacles_cx=np.array([20.0]), obstacles_cy=np.array([0.0]), obstacles_yaw=np.zeros(1),
        obstacles_length=np.ones(1), obstacles_width=np.ones(1), initial_pose=Pose2D(),
        actor_id=np.array([7, 9]), actor_class=np.array([ObjectClass.CAR, ObjectClass.PEDESTRIAN], np.int8),
        actor_length=np.array([4.5, 0.5]), actor_width=np.array([1.8, 0.5]), actor_height=np.array([1.5, 1.7]),
        waypoint_actor=np.array([0, 1, 0]), waypoint_time=np.array([4.0, 0.0, 2.0]),
        waypoint_x=np.array([10.0, -3.0, 0.0]), waypoint_y=np.array([5.0, -3.0, 5.0]),
        waypoint_yaw=np.array([0.0, 1.0, 0.0]))


def test_motion_model():
    s = scene_with_actors()
    w = actor_waypoints(s, 0)
    assert list(w[:, 0]) == [2.0, 4.0]                       # sorted by time
    assert actor_state(w, 0.0).x == 0.0 and actor_state(w, 0.0).velocity_x == 0.0   # waits at the first
    mid = actor_state(w, 3.0)
    assert mid.x == pytest.approx(5.0) and mid.velocity_x == pytest.approx(5.0) and mid.velocity_y == 0.0
    end = actor_state(w, 9.0)
    assert end.x == 10.0 and end.velocity_x == 0.0            # stands still after the last
    assert actor_states(s, 1.0)[1].yaw == 1.0
    with pytest.raises(ValueError):
        actor_waypoints(SimulationScene(actor_id=np.array([1])), 0)


def test_yaw_takes_the_short_way():
    w = np.array([[0.0, 0.0, 0.0, 3.0], [1.0, 0.0, 0.0, -3.0]])
    st = actor_state(w, 0.5)
    assert abs(abs(st.yaw) - math.pi) < 0.2 and st.yaw_rate == pytest.approx(2 * math.pi - 6.0)


def test_ground_truth_includes_actors():
    s = scene_with_actors()
    o = scene_objects(s, Header(stamp=to_ns(3.0)))
    o.validate()
    assert len(o) == 3 and list(o.object_id) == [0, ACTOR_ID_OFFSET + 7, ACTOR_ID_OFFSET + 9]
    assert o.classification[1] == ObjectClass.CAR and o.x[1] == pytest.approx(5.0)
    assert o.velocity_x[1] == pytest.approx(5.0) and o.z[2] == pytest.approx(0.85)


def test_scene_roundtrip_and_godot_actors():
    s = scene_with_actors()
    assert codec.to_json_dict(codec.decode(codec.encode(s), SimulationScene)) == codec.to_json_dict(s)
    actors = gf.reset_request(s, {}, 100.0, 0)["scene"]["actors"]
    assert [a["id"] for a in actors] == [7, 9] and actors[1]["cls"] == int(ObjectClass.PEDESTRIAN)
    assert actors[0]["waypoints"] == [[2.0, 0.0, -5.0, 0.0], [4.0, 10.0, -5.0, 0.0]]   # Z_g = -y
