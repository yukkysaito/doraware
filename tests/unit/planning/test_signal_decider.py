"""Turn indicator / hazard light decision (planning/signals/decider.py)."""
import math

import numpy as np
import pytest

from doraware.interfaces import Gear, Header, Pose, Trajectory, TurnIndicator, VehicleState, to_ns
from doraware.planning.signals.decider import SignalDecider, SignalParams, SignalPolicy, upcoming_motion_end

DT = 0.05


def _traj(xs, ys, vs, gears, start=0.0, stamp=None, validity=1.0):
    n = len(xs)
    yaw = np.zeros(n)
    t0 = to_ns(start)
    return Trajectory(header=Header(stamp=t0 if stamp is None else stamp), start_time=t0, time_step=DT,
                      valid_until=t0 + to_ns(validity), x=np.asarray(xs, float), y=np.asarray(ys, float), yaw=yaw,
                      longitudinal_velocity=np.asarray(vs, float), longitudinal_acceleration=np.zeros(n),
                      curvature=np.zeros(n), front_wheel_angle=np.zeros(n),
                      gear=np.asarray([int(g) for g in gears], dtype=np.int8))


def _arc(side: int, n=100, v=1.0, gear=Gear.DRIVE, start=0.0, radius=5.0):
    """Quarter-circle-ish motion from the origin (yaw 0) to the left (+1) or right (-1)."""
    s = np.arange(n) * abs(v) * DT
    ang = s / radius
    sign = 1.0 if gear == Gear.DRIVE else -1.0
    xs = sign * radius * np.sin(ang)
    ys = side * radius * (1.0 - np.cos(ang))
    return _traj(xs, ys, np.full(n, v), [gear] * n, start=start)


def _state(x=0.0, y=0.0, yaw=0.0, t=0.0):
    return VehicleState(header=Header(stamp=to_ns(t)), pose=Pose.from_xyyaw(x, y, yaw), gear=Gear.DRIVE)


def _decider(**kw):
    kw.setdefault("policy", SignalPolicy.TURN_INDICATORS)
    return SignalDecider(SignalParams(**kw))


def test_params_from_config_validates():
    p = SignalParams.from_config({"lookahead_distance": 4, "activation_offset": 0.6,
                                                      "deactivation_offset": 0.3, "min_hold_time": 0.5})
    assert p.lookahead_distance == 4.0 and p.min_hold_time == 0.5
    assert SignalParams.from_config({}) == SignalParams()
    assert SignalParams().policy == SignalPolicy.MANEUVER_HAZARD
    assert SignalParams.from_config({"policy": "turn_indicators"}).policy == \
        SignalPolicy.TURN_INDICATORS
    with pytest.raises(ValueError):
        SignalParams.from_config({"policy": "blink_everything"})
    with pytest.raises(ValueError):
        SignalParams.from_config({"activation_offset": 0.2, "deactivation_offset": 0.3})
    with pytest.raises(TypeError):
        SignalParams.from_config({"unknown_key": 1.0})


@pytest.mark.parametrize("side,expected", [(1, TurnIndicator.LEFT), (-1, TurnIndicator.RIGHT)])
def test_turn_forward(side, expected):
    d = _decider()
    d.on_trajectory(_arc(side))
    out = d.step(_state(), 0)
    assert out.turn_indicator == expected and not out.hazard_lights
    assert math.copysign(1.0, out.lateral_offset) == side


@pytest.mark.parametrize("side,expected", [(1, TurnIndicator.LEFT), (-1, TurnIndicator.RIGHT)])
def test_turn_reverse_uses_the_side_the_body_moves_to(side, expected):
    d = _decider()
    d.on_trajectory(_arc(side, v=-1.0, gear=Gear.REVERSE))
    assert d.step(_state(), 0).turn_indicator == expected


def test_straight_motion_is_off():
    n = 100
    d = _decider()
    d.on_trajectory(_traj(np.arange(n) * DT, np.zeros(n), np.ones(n), [Gear.DRIVE] * n))
    out = d.step(_state(), 0)
    assert out.turn_indicator == TurnIndicator.OFF and out.lateral_offset == pytest.approx(0.0)


def test_stationary_trajectory_is_off():
    n = 40
    d = _decider()
    d.on_trajectory(_traj(np.zeros(n), np.zeros(n), np.zeros(n), [Gear.DRIVE] * n))
    out = d.step(_state(), 0)
    assert out.turn_indicator == TurnIndicator.OFF and out.lateral_offset is None and not out.hazard_lights


def test_waiting_before_departure_signals_the_next_motion():
    still = 20                                      # 1 s standstill, then a left arc
    arc = _arc(1, n=80)
    xs = np.r_[np.zeros(still), arc.x]
    ys = np.r_[np.zeros(still), arc.y]
    vs = np.r_[np.zeros(still), arc.longitudinal_velocity]
    d = _decider()
    d.on_trajectory(_traj(xs, ys, vs, [Gear.DRIVE] * len(xs)))
    assert d.step(_state(), 0).turn_indicator == TurnIndicator.LEFT


def test_upcoming_motion_stops_at_gear_change():
    # forward straight 2 m, then reverse to the left: only the forward part counts while it lasts
    fwd = 40
    xs = np.r_[np.arange(fwd) * DT, np.full(40, fwd * DT)]
    ys = np.r_[np.zeros(fwd), np.linspace(0.0, 3.0, 40)]
    vs = np.r_[np.ones(fwd - 1), 0.0, -np.ones(40)]
    gears = [Gear.DRIVE] * fwd + [Gear.REVERSE] * 40
    tr = _traj(xs, ys, vs, gears)
    end = upcoming_motion_end(tr, 0.0, lookahead=10.0)
    assert end == fwd - 1
    d = _decider()
    d.on_trajectory(tr)
    assert d.step(_state(), 0).turn_indicator == TurnIndicator.OFF
    # at the cusp (stopped), the next motion (reverse, to the left) is signalled
    assert upcoming_motion_end(tr, (fwd - 1) * DT, lookahead=10.0) > fwd


def test_lookahead_limits_the_considered_motion():
    n = 200                                         # 10 m straight, then a sharp left offset
    xs = np.arange(n) * DT
    ys = np.where(xs > 8.0, 2.0, 0.0)
    tr = _traj(xs, ys, np.ones(n), [Gear.DRIVE] * n)
    d = _decider(lookahead_distance=6.0)
    d.on_trajectory(tr)
    assert d.step(_state(), 0).turn_indicator == TurnIndicator.OFF


def test_hysteresis_and_min_hold_time():
    p = dict(activation_offset=0.5, deactivation_offset=0.2, min_hold_time=1.0)
    d = _decider(**p)
    d.turn = TurnIndicator.LEFT
    assert d._turn_for(0.3) == TurnIndicator.LEFT        # inside the hysteresis band: stays on
    assert d._turn_for(0.1) == TurnIndicator.OFF
    assert d._turn_for(-0.6) == TurnIndicator.RIGHT       # beyond the other activation: switches
    d.turn = TurnIndicator.OFF
    assert d._turn_for(0.3) == TurnIndicator.OFF          # below activation: stays off

    # a change is held for min_hold_time before the next one
    d = _decider(**p)
    d.on_trajectory(_arc(1, start=0.0))
    assert d.step(_state(), 0).turn_indicator == TurnIndicator.LEFT
    n = 100
    d.on_trajectory(_traj(np.arange(n) * DT, np.zeros(n), np.ones(n), [Gear.DRIVE] * n, start=0.5))
    assert d.step(_state(t=0.5), to_ns(0.5)).turn_indicator == TurnIndicator.LEFT    # held
    d.on_trajectory(_traj(np.arange(n) * DT, np.zeros(n), np.ones(n), [Gear.DRIVE] * n, start=1.0))
    assert d.step(_state(t=1.0), to_ns(1.0)).turn_indicator == TurnIndicator.OFF


def test_hazard_reasons():
    d = _decider()
    out = d.step(_state(), 0)
    assert out.hazard_lights and out.hazard_reason == "no_trajectory"
    d = _decider()
    d.on_trajectory(_traj([], [], [], []))
    out = d.step(_state(), 0)
    assert out.hazard_lights and out.hazard_reason == "empty_trajectory"
    assert out.turn_indicator == TurnIndicator.OFF
    d = _decider()
    d.on_trajectory(_arc(1))
    assert not d.step(_state(), 0).hazard_lights
    out = d.step(_state(t=1.5), to_ns(1.5))
    assert out.hazard_lights and out.hazard_reason == "expired_trajectory" and out.lateral_offset is None


def test_hazard_on_is_immediate_and_off_is_held():
    n = 100
    d = _decider(min_hold_time=1.0)
    d.on_trajectory(_traj(np.arange(n) * DT, np.zeros(n), np.ones(n), [Gear.DRIVE] * n, start=0.0))
    assert not d.step(_state(), 0).hazard_lights
    d.on_trajectory(_traj([], [], [], [], start=0.1))
    assert d.step(_state(t=0.1), to_ns(0.1)).hazard_lights          # on at once
    d.on_trajectory(_traj(np.arange(n) * DT, np.zeros(n), np.ones(n), [Gear.DRIVE] * n, start=0.5))
    assert d.step(_state(t=0.5), to_ns(0.5)).hazard_lights          # held for min_hold_time
    d.on_trajectory(_traj(np.arange(n) * DT, np.zeros(n), np.ones(n), [Gear.DRIVE] * n, start=1.1))
    assert not d.step(_state(t=1.1), to_ns(1.1)).hazard_lights


def test_older_trajectories_are_ignored():
    d = _decider()
    assert d.on_trajectory(_arc(1, start=1.0))
    assert not d.on_trajectory(_arc(-1, start=0.5))
    assert not d.on_trajectory(_arc(-1, start=1.0))


def test_maneuver_hazard_policy():
    """Parking default: hazard lights for the whole manoeuvre (also standing at a cusp), no indicator."""
    still = 20
    arc = _arc(1, n=80)
    xs, ys = np.r_[np.zeros(still), arc.x, np.full(40, arc.x[-1])], np.r_[np.zeros(still), arc.y, np.full(40, arc.y[-1])]
    vs = np.r_[np.zeros(still), arc.longitudinal_velocity, np.zeros(40)]
    tr = _traj(xs, ys, vs, [Gear.DRIVE] * len(xs), validity=100.0)
    d = _decider(policy=SignalPolicy.MANEUVER_HAZARD)
    d.on_trajectory(tr)
    out = d.step(_state(), 0)                                      # before departure
    assert out.hazard_lights and out.hazard_reason == "maneuvering"
    assert out.turn_indicator == TurnIndicator.OFF                 # never an indicator in this policy
    end_t = (len(xs) - 1) * DT
    ex, ey = float(xs[-1]), float(ys[-1])
    out = d.step(_state(ex, ey, t=end_t), to_ns(end_t))            # plan complete: nothing moves any more
    assert not out.hazard_lights and out.turn_indicator == TurnIndicator.OFF
    d = _decider(policy=SignalPolicy.MANEUVER_HAZARD)
    d.on_trajectory(_traj([], [], [], []))
    assert d.step(_state(), 0).hazard_reason == "empty_trajectory"   # failures as in the other policy
