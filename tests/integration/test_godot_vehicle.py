"""Godot vehicle simulator driven through GodotSimulator (no Dora).

All checks are made in the map frame through godot_frame conversions.
"""
import math
import os
from pathlib import Path

import numpy as np
import pytest

from doraware.core.config import REPO_ROOT, load_config
from doraware.interfaces import (Gear, Pose2D, SimulationScene, TurnIndicator, VehicleParameters,
                                 VehicleSignalCommand)
import doraware.simulation.plugins.godot.frame as gf
from doraware.simulation.plugins.godot.client import GodotSimulator

pytestmark = pytest.mark.godot

CFG = load_config()
HZ = 100.0
DT = 1.0 / HZ
VP = VehicleParameters()
LOG_DIR = Path(os.environ.get("DORAWARE_TEST_LOG_DIR", REPO_ROOT / "results" / "test_logs"))


def actuator(**over):
    a = dict(CFG["vehicle"]["actuator"])
    a.update(over)
    return a


def scene(x=0.0, y=0.0, yaw=0.0, obstacles=()):
    obs = np.array(obstacles, dtype=float).reshape(-1, 5)
    return SimulationScene(
        map_xmin=-30, map_xmax=30, map_ymin=-30, map_ymax=30,
        obstacles_cx=obs[:, 0], obstacles_cy=obs[:, 1], obstacles_yaw=obs[:, 2],
        obstacles_length=obs[:, 3], obstacles_width=obs[:, 4], obstacle_height=1.5,
        initial_pose=Pose2D(x, y, yaw), goal_pose=Pose2D(x, y, yaw), vehicle=VP)


def make_sim(name, headless=True):
    return GodotSimulator(CFG["simulation"]["godot"], str(LOG_DIR / f"{name}.log"), physics_hz=HZ, headless=headless)


def assert_clean_log(name):
    """Godot keeps running after a GDScript error; the log is the only witness."""
    log = (LOG_DIR / f"{name}.log").read_text(errors="ignore")
    errors = [line for line in log.splitlines() if "SCRIPT ERROR" in line or line.startswith("ERROR:")]
    assert not errors, "\n".join(errors[:20])


@pytest.fixture(scope="module")
def sim():
    s = make_sim("test_godot_vehicle")
    yield s
    s.close()
    assert s.returncode == 0
    assert_clean_log("test_godot_vehicle")


def pose(state):
    return gf.state_pose_map(state)


def vstate(state):
    return gf.state_to_vehicle_state(state)


def run(sim, seconds, accel, steer=0.0, gear=Gear.NONE, n=2, emergency=False):
    out = []
    for _ in range(int(round(seconds * HZ / n))):
        out.append(sim.step(n, accel, steer, int(gear), emergency))
    return out


def speed_control(sim, seconds, v_ref, steer=0.0, gear=Gear.NONE, n=2):
    s, out = None, []
    v = 0.0
    for _ in range(int(round(seconds * HZ / n))):
        s = sim.step(n, 1.5 * (v_ref - v), steer, int(gear))
        v = vstate(s).twist.linear.x
        out.append(s)
    return out


# ------------------------------------------------------------------ basics
def test_reset_reports_initial_pose(sim):
    s = sim.reset(scene(2.0, -3.0, 0.4), actuator(), HZ, 0)
    assert s["step"] == 0 and s["time"] == 0.0
    assert pose(s) == pytest.approx((2.0, -3.0, 0.4), abs=1e-6)
    assert s["gear"] == int(Gear.PARK)


def test_park_holds_and_initial_gear_shift_takes_shift_time(sim):
    sim.reset(scene(), actuator(), HZ, 0)
    states = run(sim, 0.5, 1.5, n=1)             # PARK: accel request must not move the car
    assert abs(pose(states[-1])[0]) < 1e-6
    shift_steps = int(round(CFG["vehicle"]["actuator"]["gear_shift_time"] / DT))
    states = [sim.step(1, 0.0, 0.0, int(Gear.DRIVE)) for _ in range(shift_steps + 2)]
    gears = [s["gear"] for s in states]
    assert gears[shift_steps - 2] == int(Gear.PARK)
    assert gears[shift_steps - 1] == int(Gear.DRIVE)   # engaged after exactly gear_shift_time


def test_forward_accel_at_yaw0_moves_plus_x(sim):
    sim.reset(scene(), actuator(), HZ, 0, initial_gear=Gear.DRIVE)
    s = run(sim, 2.0, 1.0)[-1]
    x, y, yaw = pose(s)
    assert x > 1.5 and abs(y) < 1e-4 and abs(yaw) < 1e-6
    assert vstate(s).twist.linear.x > 1.5
    assert vstate(s).acceleration.x == pytest.approx(1.0, abs=0.05)


def test_forward_at_yaw_90_moves_plus_y(sim):
    sim.reset(scene(yaw=math.pi / 2), actuator(), HZ, 0, initial_gear=Gear.DRIVE)
    x, y, yaw = pose(run(sim, 2.0, 1.0)[-1])
    assert y > 1.5 and abs(x) < 1e-4 and yaw == pytest.approx(math.pi / 2, abs=1e-6)


def test_left_steer_forward_turns_left(sim):
    sim.reset(scene(), actuator(), HZ, 0, initial_gear=Gear.DRIVE)
    states = speed_control(sim, 4.0, 1.0, steer=0.3)
    x, y, yaw = pose(states[-1])
    assert yaw > 0.3 and y > 0.3 and x > 1.0
    assert vstate(states[-1]).twist.angular.z > 0
    assert states[-1]["steer"] == pytest.approx(0.3, abs=1e-6)


def test_right_steer_forward_turns_right(sim):
    sim.reset(scene(), actuator(), HZ, 0, initial_gear=Gear.DRIVE)
    x, y, yaw = pose(speed_control(sim, 4.0, 1.0, steer=-0.3)[-1])
    assert yaw < -0.3 and y < -0.3


def test_reverse_gear_negative_accel_moves_backwards(sim):
    sim.reset(scene(), actuator(), HZ, 0, initial_gear=Gear.REVERSE)
    s = run(sim, 2.0, -0.8)[-1]
    x, y, yaw = pose(s)
    assert x < -1.0 and abs(y) < 1e-4
    assert vstate(s).twist.linear.x < -1.0
    assert s["gear"] == int(Gear.REVERSE)


def test_reverse_with_left_steer_yaw_decreases(sim):
    # backing up with the wheels turned left: yaw decreases, the rear axle moves
    # back along a circle whose center is on the left (+y)
    sim.reset(scene(), actuator(), HZ, 0, initial_gear=Gear.REVERSE)
    x, y, yaw = pose(speed_control(sim, 4.0, -1.0, steer=0.3)[-1])
    assert x < -1.0 and yaw < -0.3 and y > 0.3


# ------------------------------------------------------------------ kinematics
def _fit_circle(xy):
    x, y = xy[:, 0], xy[:, 1]
    a = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    c = np.linalg.lstsq(a, b, rcond=None)[0]
    cx, cy = c[0] / 2, c[1] / 2
    return cx, cy, math.sqrt(c[2] + cx * cx + cy * cy)


@pytest.mark.parametrize("steer_sign", [1.0, -1.0])
def test_full_lock_circle_radius_matches_kinematics(sim, steer_sign):
    delta = steer_sign * VP.max_steer_angle
    sim.reset(scene(), actuator(), HZ, 0, initial_gear=Gear.DRIVE)
    speed_control(sim, 6.0, 0.5, steer=delta)       # settle
    pts = np.array([pose(s)[:2] for s in speed_control(sim, 30.0, 0.5, steer=delta)])
    _, _, r = _fit_circle(pts)
    r_kin = VP.wheel_base / math.tan(VP.max_steer_angle)
    assert r == pytest.approx(r_kin, rel=0.02)


# ------------------------------------------------------------------ actuators
def test_steering_step_response_dead_time_and_lag(sim):
    td, tau, cmd = 0.1, 0.2, 0.1
    sim.reset(scene(), actuator(steering_dead_time=td, steering_time_constant=tau,
                                steering_rate_limit=100.0), HZ, 0)
    nd = int(round(td / DT))
    alpha = 1.0 - math.exp(-DT / tau)
    for k in range(1, 80):
        s = sim.step(1, 0.0, cmd, int(Gear.NONE))
        expected = 0.0 if k <= nd else cmd * (1.0 - (1.0 - alpha) ** (k - nd))
        assert s["steer"] == pytest.approx(expected, abs=1e-12), k
        if k == nd + int(round(tau / DT)):   # 63 % of the step at dead time + time constant
            assert s["steer"] == pytest.approx(cmd * (1 - math.exp(-1)), rel=1e-9)


def test_steering_rate_and_angle_limit(sim):
    sim.reset(scene(), actuator(steering_time_constant=0.0, steering_rate_limit=0.5,
                                steering_angle_limit=0.4), HZ, 0)
    s = sim.step(10, 0.0, 1.0, int(Gear.NONE))
    assert s["steer"] == pytest.approx(0.05, abs=1e-12)
    s = sim.step(100, 0.0, 1.0, int(Gear.NONE))
    assert s["steer"] == pytest.approx(0.4, abs=1e-12)


def test_acceleration_step_response_dead_time_and_lag(sim):
    td, tau, cmd = 0.05, 0.1, 0.5
    sim.reset(scene(), actuator(acceleration_dead_time=td, acceleration_time_constant=tau,
                                jerk_limit=1000.0), HZ, 0, initial_gear=Gear.DRIVE)
    nd = int(round(td / DT))
    alpha = 1.0 - math.exp(-DT / tau)
    for k in range(1, 60):
        s = sim.step(1, cmd, 0.0, int(Gear.NONE))
        expected = 0.0 if k <= nd else cmd * (1.0 - (1.0 - alpha) ** (k - nd))
        assert s["accel_actual"] == pytest.approx(expected, abs=1e-12), k
    # the vehicle follows the actuator output
    assert vstate(s).acceleration.x == pytest.approx(s["accel_actual"], abs=0.02)


def test_jerk_and_acceleration_limits(sim):
    sim.reset(scene(), actuator(acceleration_time_constant=0.0, jerk_limit=2.0,
                                acceleration_limit=1.0), HZ, 0, initial_gear=Gear.DRIVE)
    s = sim.step(10, 5.0, 0.0, int(Gear.NONE))
    assert s["accel_actual"] == pytest.approx(0.2, abs=1e-12)    # jerk 2 m/s^3 * 0.1 s
    s = sim.step(100, 5.0, 0.0, int(Gear.NONE))
    assert s["accel_actual"] == pytest.approx(1.0, abs=1e-12)    # acceleration_limit


# ------------------------------------------------------------------ brake / gear
def test_braking_stops_and_holds_never_reverses(sim):
    sim.reset(scene(), actuator(), HZ, 0, initial_gear=Gear.DRIVE)
    speed_control(sim, 3.0, 1.0)
    states = run(sim, 4.0, -2.0, n=1)
    v = np.array([vstate(s).twist.linear.x for s in states])
    x = np.array([pose(s)[0] for s in states])
    assert v.min() >= -1e-6
    assert abs(v[-1]) < 1e-6
    assert np.all(np.diff(x) >= -1e-6)
    # standing still with a braking request: no motion at all
    x_stop = x[-1]
    states = run(sim, 2.0, -3.0, n=2)
    assert abs(pose(states[-1])[0] - x_stop) < 1e-5


def test_brake_request_against_gear_at_standstill_does_not_push(sim):
    for gear, accel in [(Gear.DRIVE, -1.5), (Gear.REVERSE, +1.5)]:
        sim.reset(scene(), actuator(), HZ, 0, initial_gear=gear)
        s = run(sim, 2.0, accel)[-1]
        assert abs(pose(s)[0]) < 1e-6 and abs(vstate(s).twist.linear.x) < 1e-6


def test_reverse_brake_while_moving_backwards(sim):
    sim.reset(scene(), actuator(), HZ, 0, initial_gear=Gear.REVERSE)
    speed_control(sim, 3.0, -1.0)
    v = np.array([vstate(s).twist.linear.x for s in run(sim, 3.0, +2.0, n=1)])
    assert v.max() <= 1e-6 and abs(v[-1]) < 1e-6


def test_emergency_brake(sim):
    sim.reset(scene(), actuator(), HZ, 0, initial_gear=Gear.DRIVE)
    speed_control(sim, 4.0, 1.5)
    states = run(sim, 3.0, 1.0, emergency=True, n=1)
    v = np.array([vstate(s).twist.linear.x for s in states])
    assert v.min() >= -1e-6 and abs(v[-1]) < 1e-6
    assert min(s["accel_actual"] for s in states) == pytest.approx(-CFG["vehicle"]["actuator"]["deceleration_limit"])


def test_gear_change_refused_while_moving(sim):
    sim.reset(scene(), actuator(), HZ, 0, initial_gear=Gear.DRIVE)
    speed_control(sim, 3.0, 1.0)
    states = [sim.step(2, 0.0, 0.0, int(Gear.REVERSE)) for _ in range(50)]
    assert all(s["gear"] == int(Gear.DRIVE) for s in states)
    assert states[-1]["gear_refusals"] > 0
    assert vstate(states[-1]).twist.linear.x > 0.5          # still rolling forward
    # stop, then the same request is executed
    run(sim, 3.0, -2.0)
    states = [sim.step(2, 0.0, 0.0, int(Gear.REVERSE)) for _ in range(30)]
    assert states[-1]["gear"] == int(Gear.REVERSE)


# ------------------------------------------------------------------ collisions
def test_collision_with_box_is_reported(sim):
    # 1 m x 3 m box centered 8 m ahead
    s0 = sim.reset(scene(obstacles=[(8.0, 0.0, 0.0, 1.0, 3.0)]), actuator(), HZ, 0,
                   initial_gear=Gear.DRIVE)
    assert s0["contacts"] == 0
    states = speed_control(sim, 10.0, 1.5)
    assert states[-1]["contacts"] >= 1
    assert states[-1]["contact_names"] == ["obstacle_0"]
    # the front bumper stopped at the box face (7.5 m) and did not pass through
    x = pose(states[-1])[0]
    assert x + VP.wheel_base + VP.front_overhang == pytest.approx(7.5, abs=0.1)


# ------------------------------------------------------------------ determinism
def _scripted_run(sim, seed=3):
    sim.reset(scene(1.0, 2.0, 0.3, obstacles=[(9.0, 6.0, 0.4, 2.0, 2.0)]),
              actuator(steering_noise_std=0.002, acceleration_noise_std=0.05,
                       steering_dead_time=0.04, acceleration_dead_time=0.02), HZ, seed)
    out = []
    for k in range(600):
        t = k * 2 * DT
        gear = Gear.DRIVE if t < 6.0 else Gear.REVERSE
        accel = 1.0 if t < 3.0 else (-2.0 if t < 6.0 else -0.8)
        steer = 0.4 * math.sin(0.7 * t)
        s = sim.step(2, accel, steer, int(gear))
        s.pop("step_wall_time")
        out.append(s)
    return out


# ------------------------------------------------------------------ signal lamps
def _sig(turn=TurnIndicator.OFF, hazard=False):
    return VehicleSignalCommand(turn_indicator=turn, hazard_lights=hazard)


def _lamps(sim, seconds, signals=None):
    """Step one physics tick at a time; signals are sent with the first step only."""
    out = []
    for i in range(int(round(seconds * HZ))):
        out.append(gf.state_to_signal_status(sim.step(1, 0.0, 0.0, 0, False, signals if i == 0 else None)))
    return out


def test_signals_off_after_reset_and_latched(sim):
    s = gf.state_to_signal_status(sim.reset(scene(), actuator(), HZ, 0))
    assert (s.turn_indicator, s.hazard_lights, s.left_lamp_on, s.right_lamp_on) == (TurnIndicator.OFF, False,
                                                                                   False, False)
    st = _lamps(sim, 2.0, _sig(TurnIndicator.LEFT))
    assert all(x.turn_indicator == TurnIndicator.LEFT and not x.hazard_lights for x in st)   # latched
    assert not any(x.right_lamp_on for x in st)
    s = gf.state_to_signal_status(sim.reset(scene(), actuator(), HZ, 0))
    assert s.turn_indicator == TurnIndicator.OFF            # a reset clears the latched signals


def test_flasher_frequency_and_phase(sim):
    sim.reset(scene(), actuator(flasher_frequency=1.5), HZ, 0)
    lamp = np.array([x.left_lamp_on for x in _lamps(sim, 4.0, _sig(TurnIndicator.LEFT))])
    assert lamp[0]                                          # starts on
    edges = np.nonzero(lamp[1:] != lamp[:-1])[0] + 1        # step index of every switch
    half = 1.0 / (2 * 1.5)
    assert len(edges) == 11                                 # 4 s at 3 switches per second (minus the start)
    assert np.allclose(np.diff(edges) * DT, half, atol=DT)
    assert abs(lamp.mean() - 0.5) < 0.02


def test_hazard_has_priority_and_turn_resumes(sim):
    sim.reset(scene(), actuator(), HZ, 0)
    st = _lamps(sim, 1.0, _sig(TurnIndicator.RIGHT, hazard=True))
    assert all(x.left_lamp_on == x.right_lamp_on for x in st)          # both sides flash in sync
    assert any(x.left_lamp_on for x in st) and st[0].turn_indicator == TurnIndicator.RIGHT
    st = _lamps(sim, 1.0, _sig(TurnIndicator.RIGHT))
    assert not any(x.left_lamp_on for x in st) and any(x.right_lamp_on for x in st)
    st = _lamps(sim, 0.5, _sig())
    assert not any(x.left_lamp_on or x.right_lamp_on for x in st)


def test_invalid_turn_indicator_is_off(sim):
    sim.reset(scene(), actuator(), HZ, 0)
    raw = sim.request({"type": "step", "n": 1, "command": {"accel": 0.0, "steer": 0.0, "gear": 0,
                                                           "signals": {"turn_indicator": 7}}})
    s = gf.state_to_signal_status(raw)
    assert s.turn_indicator == TurnIndicator.OFF and not s.left_lamp_on and not s.right_lamp_on


def test_signals_do_not_affect_motion(sim):
    def drive(with_signals):
        sim.reset(scene(), actuator(), HZ, 0)
        out = []
        for k in range(300):
            sig = _sig(TurnIndicator.LEFT if k < 150 else TurnIndicator.RIGHT, hazard=k >= 200) \
                if with_signals and k % 25 == 0 else None
            out.append(pose(sim.step(2, 1.0 if k < 150 else -1.0, 0.3, int(Gear.DRIVE), False, sig)))
        return out
    assert drive(False) == drive(True)


def test_determinism_same_process_and_new_process(sim):
    a = _scripted_run(sim)
    b = _scripted_run(sim)
    assert a == b                                   # bit-identical (exact float equality)
    other = make_sim("test_godot_vehicle_determinism")
    try:
        c = _scripted_run(other)
    finally:
        other.close()
    assert a == c
    assert pose(a[-1]) != pose(_scripted_run(sim, seed=4)[-1])   # the seed matters (noise)


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="GUI mode needs $DISPLAY")
def test_gui_matches_headless(sim):
    a = _scripted_run(sim)
    gui = make_sim("test_godot_vehicle_gui", headless=False)
    try:
        gui.send_viz(planned=[(0, 0), (5, 1)], predicted=[(0, 0), (2, 0.5)], path=[(0, 0), (9, 9)],
                     info={"scenario": "gui test"})
        import time
        from doraware.interfaces import Header
        from doraware.simulation.ground_truth import scene_objects
        boxes = scene(-6.0, 0.0, 0.0, obstacles=[[4.0, 2.0, 0.3, 4.5, 1.8], [-1.0, -3.0, 0.0, 0.6, 0.6]])
        gui.reset(boxes, actuator(), HZ, 0)
        gui.send_viz(objects=scene_objects(boxes, Header()))            # perception overlay (GUI only)
        for cam in range(4):
            gui.screenshot(str(LOG_DIR / f"test_godot_gui_camera{cam}.png"), cam)
        time.sleep(2.0)                                                  # the GUI renders while idle
        c = _scripted_run(gui)
    finally:
        gui.close()
    assert a == c
    assert_clean_log("test_godot_vehicle_gui")


def test_disconnect_raises_clear_error():
    from doraware.simulation.plugins.godot.client import SimulatorDisconnected
    s = make_sim("test_godot_vehicle_kill")
    s.reset(scene(), actuator(), HZ, 0)
    s._proc.kill()
    s._proc.wait()
    with pytest.raises(SimulatorDisconnected):
        for _ in range(3):
            s.step(1, 0.0, 0.0, 0)
    s.close()
