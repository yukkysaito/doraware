"""Pure map <-> Godot frame conversions (docs/architecture.md §5)."""
import math

import numpy as np
import pytest

from doraware.interfaces import (Gear, Pose2D, SimulationScene, TurnIndicator, VehicleParameters,
                                 VehicleSignalCommand)
import doraware.simulation.plugins.godot.frame as gf


def _godot_forward(rot_y):
    """Godot vehicle +X axis in world after rotation rot_y about +Y."""
    return gf.godot_vehicle_to_world(0.0, 0.0, rot_y, 1.0, 0.0)


def _godot_left(rot_y):
    """Godot vehicle left = local -Z."""
    return gf.godot_vehicle_to_world(0.0, 0.0, rot_y, 0.0, -1.0)


def test_point_roundtrip_and_axes():
    assert gf.map_to_godot_point(1.0, 2.0) == (1.0, -2.0)
    assert gf.godot_to_map_point(*gf.map_to_godot_point(3.5, -7.25)) == (3.5, -7.25)
    # map +y (north) is Godot -Z
    assert gf.map_to_godot_vector((0.0, 1.0, 0.0)) == (0.0, 0.0, -1.0)
    # map +z (up) is Godot +Y
    assert gf.map_to_godot_vector((0.0, 0.0, 1.0)) == (0.0, 1.0, 0.0)
    assert gf.godot_to_map_vector(gf.map_to_godot_vector((1.0, 2.0, 3.0))) == (1.0, 2.0, 3.0)


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2, -math.pi / 2, 2.5, -3.0])
def test_forward_direction_matches_map_yaw(yaw):
    pose = gf.map_to_godot_pose(0.0, 0.0, yaw)
    fx, fz = _godot_forward(pose["rot_y"])
    mx, my = gf.godot_to_map_point(fx, fz)
    assert mx == pytest.approx(math.cos(yaw), abs=1e-12)
    assert my == pytest.approx(math.sin(yaw), abs=1e-12)


def test_yaw_zero_forward_is_plus_x_and_yaw_90_is_plus_y():
    assert gf.godot_to_map_point(*_godot_forward(0.0)) == pytest.approx((1.0, 0.0))
    rot = gf.map_to_godot_pose(0, 0, math.pi / 2)["rot_y"]
    assert gf.godot_to_map_point(*_godot_forward(rot)) == pytest.approx((0.0, 1.0), abs=1e-12)


@pytest.mark.parametrize("yaw", [0.0, 1.0, -2.0])
def test_left_is_ccw_of_forward(yaw):
    rot = gf.map_to_godot_pose(0, 0, yaw)["rot_y"]
    lx, ly = gf.godot_to_map_point(*_godot_left(rot))
    assert (lx, ly) == pytest.approx((-math.sin(yaw), math.cos(yaw)), abs=1e-12)


def test_yaw_direction_positive_rot_y_is_ccw():
    # a small positive rotation about +Y_g turns map +x towards map +y (CCW, left)
    fx, fz = _godot_forward(0.1)
    mx, my = gf.godot_to_map_point(fx, fz)
    assert my > 0 and math.atan2(my, mx) == pytest.approx(0.1)


def test_pose_roundtrip_wraps_yaw():
    p = gf.map_to_godot_pose(1.0, -2.0, 3.0 * math.pi)
    x, y, yaw = gf.godot_to_map_pose(p["x"], p["z"], p["rot_y"])
    assert (x, y) == (1.0, -2.0)
    assert -math.pi <= yaw < math.pi and abs(abs(yaw) - math.pi) < 1e-12


def test_local_vectors_and_reverse():
    # base_link (x fwd, y left, z up) <-> Godot local (X fwd, Y up, -Z left)
    assert gf.base_link_to_godot_local((1.0, 2.0, 3.0)) == (1.0, 3.0, -2.0)
    assert gf.godot_local_to_base_link((1.0, 3.0, -2.0)) == (1.0, 2.0, 3.0)
    # reversing: negative local X velocity is negative base_link x
    assert gf.godot_local_to_base_link((-0.5, 0.0, 0.0))[0] == -0.5


def test_vehicle_to_map_and_godot_world_agree():
    x, y, yaw = 3.0, -1.0, 0.7
    for px, py in [(1.0, 0.0), (0.0, 1.0), (2.79, -0.8), (-1.0, 0.8)]:
        mx, my = gf.vehicle_to_map(x, y, yaw, px, py)
        pose = gf.map_to_godot_pose(x, y, yaw)
        lx, _, lz = gf.base_link_to_godot_local((px, py, 0.0))
        gx, gz = gf.godot_vehicle_to_world(pose["x"], pose["z"], pose["rot_y"], lx, lz)
        assert gf.godot_to_map_point(gx, gz) == pytest.approx((mx, my), abs=1e-12)


def test_rear_axle_footprint_center():
    vp = VehicleParameters()
    cx, cy = vp.footprint_center
    # the front axle (wheel_base ahead of base_link) lies inside the footprint, ahead of its center
    assert 0.0 < cx < vp.wheel_base
    assert cy == pytest.approx(0.0)
    mx, my = gf.vehicle_to_map(0.0, 0.0, math.pi / 2, vp.wheel_base, 0.0)
    assert (mx, my) == pytest.approx((0.0, vp.wheel_base), abs=1e-12)


def test_state_to_vehicle_state():
    state = {"type": "state", "step": 3, "time": 0.03, "x": 1.0, "z": -2.0, "rot_y": 0.5,
             "v_local": [1.5, 0.0, -0.1], "ang_vel_y": 0.2, "accel_local": [0.3, 0.0, 0.4],
             "steer": -0.1, "accel_actual": 0.3, "gear": 2, "contacts": 0, "contact_names": []}
    vs = gf.state_to_vehicle_state(state)
    assert vs.header.stamp == 30_000_000
    assert vs.pose.xyyaw() == pytest.approx((1.0, 2.0, 0.5))
    assert vs.twist.linear.x == 1.5 and vs.twist.linear.y == pytest.approx(0.1)  # -v_local.z
    assert vs.twist.angular.z == 0.2
    assert vs.acceleration.y == pytest.approx(-0.4)
    assert vs.steering_angle == -0.1  # steering sign unchanged (+left in both frames)
    assert vs.gear == Gear.REVERSE


def test_reset_request_scene_conversion():
    vp = VehicleParameters()
    scene = SimulationScene(
        map_xmin=-1.0, map_xmax=10.0, map_ymin=-3.0, map_ymax=5.0,
        obstacles_cx=np.array([4.0]), obstacles_cy=np.array([2.0]), obstacles_yaw=np.array([0.3]),
        obstacles_length=np.array([2.0]), obstacles_width=np.array([1.0]), obstacle_height=1.2,
        initial_pose=Pose2D(1.0, 1.0, math.pi / 2), goal_pose=Pose2D(8.0, -2.0, -1.0), vehicle=vp)
    act = {"steering_angle_limit": None, "steering_time_constant": 0.1}
    req = gf.reset_request(scene, act, 100.0, 7)
    assert req["scene"]["bounds"] == [-1.0, 10.0, -5.0, 3.0]
    ob = req["scene"]["obstacles"][0]
    assert (ob["x"], ob["z"], ob["rot_y"]) == (4.0, -2.0, 0.3)
    assert (ob["length"], ob["width"], ob["height"]) == (2.0, 1.0, 1.2)
    assert req["initial"] == {"x": 1.0, "z": -1.0, "rot_y": pytest.approx(math.pi / 2)}
    assert req["scene"]["goal"]["z"] == 2.0
    assert req["actuator"]["steering_angle_limit"] == vp.max_steer_angle
    assert req["vehicle"]["wheel_base"] == vp.wheel_base


def test_viz_and_step_requests():
    v = gf.viz_request(planned=[(1.0, 2.0)], predicted=np.array([[0.0, -1.0], [1.0, 1.0]]),
                       path=[(3.0, 4.0)], info={"mode": "track", "solve_ms": 0.4})
    assert v["planned"] == [[1.0, -2.0]] and v["info"] == {"mode": "track", "solve_ms": 0.4}
    assert v["predicted"] == [[0.0, 1.0], [1.0, -1.0]]
    assert v["path"] == [[3.0, -4.0]]
    s = gf.step_request(2, -0.5, 0.1, int(Gear.REVERSE), True)
    assert s == {"type": "step", "n": 2, "command": {"accel": -0.5, "steer": 0.1, "gear": 2,
                                                       "emergency": True}}


def test_step_request_signals_and_signal_status():
    cmd = VehicleSignalCommand(turn_indicator=TurnIndicator.RIGHT, hazard_lights=True)
    s = gf.step_request(1, 0.0, 0.0, 0, False, cmd)
    assert s["command"]["signals"] == {"turn_indicator": 2, "hazard_lights": True}
    assert "signals" not in gf.step_request(1, 0.0, 0.0, 0)["command"]   # None = keep the latched state
    st = gf.state_to_signal_status({"time": 0.5, "turn_indicator": 1, "hazard_lights": False,
                                    "left_lamp_on": True, "right_lamp_on": False})
    assert st.header.stamp == 500_000_000 and st.turn_indicator == TurnIndicator.LEFT
    assert (st.hazard_lights, st.left_lamp_on, st.right_lamp_on) == (False, True, False)
