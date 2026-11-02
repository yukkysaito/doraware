"""Lane map / route interfaces, the road scenario source and the lane_follow trajectory planner."""
import math

import numpy as np
import pytest

from doraware.core.config import load_config
from doraware.evaluation.scenarios.road import RoadScenarios
from doraware.interfaces import (Header, LaneMap, ObjectClass, PerceivedObjects, PlanningStatus, Pose, Pose2D, Route,
                                 VehicleState, codec, to_ns)
from doraware.planning.trajectory_planning.plugins.lane_follow.planner import LaneFollow
from doraware.planning.trajectory_planning.plugins.lane_follow.route import RouteError, route_centerline
from doraware.planning.trajectory_planning.registry import make_trajectory_planner
import doraware.simulation.plugins.godot.frame as gf

CFG = load_config()
ROAD = load_config(path="configs/road.yaml")
SRC = RoadScenarios()


def _state(x, y, yaw, t, v=0.0):
    from doraware.interfaces import Twist, Vector3
    return VehicleState(header=Header(stamp=to_ns(t)), pose=Pose.from_xyyaw(x, y, yaw),
                        twist=Twist(linear=Vector3(v, 0, 0)))


def test_lane_map_contract():
    b = SRC.build("R2", 42, CFG)
    lanes = b.request.lanes
    lanes.validate()
    assert len(lanes) == 3 and list(b.request.route.lane_ids) == [100, 101, 102]
    assert lanes.index_of(101) == 1 and lanes.centerline(1).shape[1] == 2
    back = codec.decode(codec.encode(b.request), type(b.request))
    assert codec.to_json_dict(back.lanes) == codec.to_json_dict(lanes)
    bad = LaneMap(lane_id=np.array([1]), speed_limit=np.array([3.0]), point_lane=np.array([5]), x=np.zeros(1),
                  y=np.zeros(1), left_x=np.zeros(1), left_y=np.zeros(1), right_x=np.zeros(1), right_y=np.zeros(1))
    with pytest.raises(ValueError, match="indices outside"):
        bad.validate()
    with pytest.raises(KeyError):
        lanes.index_of(7)


def test_route_centerline_is_exact_at_start_and_goal():
    b = SRC.build("R2", 42, CFG)
    r = b.request
    p = route_centerline(r.lanes, r.route, r.start, r.goal)
    assert (p.x[0], p.y[0]) == pytest.approx((r.start.x, r.start.y), abs=1e-9)
    assert (p.x[-1], p.y[-1]) == pytest.approx((r.goal.x, r.goal.y), abs=1e-9)
    assert p.yaw[-1] == pytest.approx(math.pi / 2, abs=1e-6)
    assert max(p.curvature) == pytest.approx(1 / 15.0, rel=0.05)           # the 15 m curve
    with pytest.raises(RouteError, match="empty route"):
        route_centerline(r.lanes, Route(), r.start, r.goal)
    with pytest.raises(RouteError, match="not on the route"):
        route_centerline(r.lanes, r.route, r.start, Pose2D(0.0, 30.0, 0.0))


def _obj(x, y, vx=0.0, cls=ObjectClass.PEDESTRIAN, size=(0.5, 0.5)):
    return PerceivedObjects(
        header=Header(), object_id=np.array([1]), classification=np.array([cls], np.int8),
        class_probability=np.ones(1), existence_probability=np.ones(1), x=np.array([x]), y=np.array([y]),
        z=np.array([0.85]), yaw=np.zeros(1), length=np.array([size[0]]), width=np.array([size[1]]),
        height=np.array([1.7]), velocity_x=np.array([vx]), velocity_y=np.zeros(1), yaw_rate=np.zeros(1))


def _planner():
    b = SRC.build("R1", 42, ROAD)
    lf = make_trajectory_planner(ROAD)
    assert isinstance(lf, LaneFollow)
    res = lf.plan(b.request, _state(3.0, 0.0, 0.0, 0.0), 0)
    assert res.status == PlanningStatus.SUCCESS and len(res.path) > 10
    return lf, b.request.vehicle


def test_lane_follow_drives_to_the_goal_and_stops_there():
    lf, _ = _planner()
    tr = lf.trajectory_at(_state(3.0, 0.0, 0.0, 0.0), 0)
    assert tr.longitudinal_velocity[0] == 0.0 and tr.longitudinal_velocity[-1] > 2.0     # accelerates
    assert np.all(np.diff(tr.longitudinal_velocity) <= 0.8 * tr.time_step + 1e-9)        # within max_acceleration
    near = lf.trajectory_at(_state(36.0, 0.0, 0.0, 30.0, 1.0), to_ns(30.0))
    assert near.longitudinal_velocity[-1] == 0.0 and near.x[-1] == pytest.approx(38.0, abs=2e-3)   # at the goal


def test_lane_follow_stops_before_an_object_and_continues_when_clear():
    lf, vp = _planner()
    lf.on_input("objects", _obj(20.0, 0.0))
    tr = lf.trajectory_at(_state(10.0, 0.0, 0.0, 5.0, 2.0), to_ns(5.0))
    assert tr.longitudinal_velocity[-1] == 0.0
    front = tr.x[-1] + vp.wheel_base + vp.front_overhang
    assert 20.0 - 0.25 - front == pytest.approx(2.0, abs=0.05)                             # stop_margin
    lf.on_input("objects", _obj(20.0, 6.0))                                                # it left the lane
    x_stop = float(tr.x[-1])
    lf.window = None
    assert lf.trajectory_at(_state(x_stop, 0.0, 0.0, 9.0), to_ns(9.0)).longitudinal_velocity.max() == 0.0
    assert lf.trajectory_at(_state(x_stop, 0.0, 0.0, 10.1), to_ns(10.1)).longitudinal_velocity.max() > 1.0


def test_lane_follow_follows_a_lead_vehicle():
    lf, vp = _planner()
    # the lead's rear is at the desired gap (stop_margin + time_gap * v = 3.8 m) ahead of our front
    lead_x = 10.0 + vp.wheel_base + vp.front_overhang + 3.8 + 2.25
    lf.on_input("objects", _obj(lead_x, 0.0, vx=1.2, cls=ObjectClass.CAR, size=(4.5, 1.8)))
    tr = lf.trajectory_at(_state(10.0, 0.0, 0.0, 5.0, 1.2), to_ns(5.0))
    assert tr.longitudinal_velocity.max() <= 1.2 + 0.02                                   # keeps the lead's speed
    assert tr.longitudinal_velocity[-1] == pytest.approx(1.2, abs=0.01)
    lf.window = None                                     # further away: it closes in faster, then matches
    lf.on_input("objects", _obj(lead_x + 6.0, 0.0, vx=1.2, cls=ObjectClass.CAR, size=(4.5, 1.8)))
    tr = lf.trajectory_at(_state(10.0, 0.0, 0.0, 5.05, 1.2), to_ns(5.05))
    v = tr.longitudinal_velocity
    assert v.max() > 1.5 and 1.2 - 0.01 <= v[-1] < v.max()           # closes in, then slows towards 1.2 m/s
    lf.on_input("objects", _obj(25.0, 0.0, vx=0.2, cls=ObjectClass.CAR, size=(4.5, 1.8)))   # too slow: a stop
    assert lf.trajectory_at(_state(10.0, 0.0, 0.0, 5.1, 1.2), to_ns(5.1)).longitudinal_velocity[-1] == 0.0


def test_lane_follow_rejects_requests_without_a_route():
    b = SRC.build("R1", 42, ROAD)
    b.request.route = Route()
    res = make_trajectory_planner(ROAD).plan(b.request, _state(3.0, 0.0, 0.0, 0.0), 0)
    assert res.status == PlanningStatus.INVALID_INPUT and "empty route" in res.message


def test_road_source_and_lane_polylines():
    assert SRC.suite("all") == ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]
    r5 = SRC.build("R5", 42, CFG)                                            # from the Lanelet2 file
    assert sorted(r5.request.lanes.lane_id.tolist()) == [1001, 1002, 1003, 1004]
    with pytest.raises(ValueError):
        SRC.suite("random", 10)
    b = SRC.build("R4", 42, CFG)
    assert len(b.scene.actor_id) == 1 and b.scene.actor_class[0] == ObjectClass.CAR
    lanes = gf.lane_polylines(b.request.lanes, b.request.route.lane_ids)
    assert [l["route"] for l in lanes] == [True, False]
    assert lanes[0]["left"][0] == pytest.approx([0.0, -1.75])                # map y +1.75 -> Godot z -1.75
