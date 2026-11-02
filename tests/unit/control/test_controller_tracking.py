"""Closed-loop tracking tests: TrajectoryFollower + Python plant (test_controller_sim)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from doraware.interfaces import Gear

from test_controller_sim import (PlantConfig, Seg, build_master, check_constraints, check_gear_changes,
                                 make_config, run_closed_loop)

LAT_TOL = 0.05        # [m] lateral tracking error while tracking
POS_TOL = 0.10        # [m] final position
YAW_TOL = 2.0         # [deg] final heading
V_TOL = 0.02          # [m/s] final speed


def ramped(points, ramp=2.0):
    """Curvature steps (s_i, kappa_i), each step ramped linearly over the ``ramp`` metres before s_i."""
    def f(s):
        k = points[0][1]
        for (s1, k1) in points[1:]:
            if s >= s1:
                k = k1
            elif s > s1 - ramp:
                return k + (k1 - k) * (s - (s1 - ramp)) / ramp
            else:
                return k
        return k
    return f


S_CURVE = ramped([(0.0, 0.0), (2.0, 0.15), (6.0, -0.15), (11.0, 0.0)])

SCENARIOS = {
    "straight_forward": dict(segs=[Seg(1, 8.0, 1.0)]),
    "straight_reverse": dict(segs=[Seg(-1, 6.0, 1.0)]),
    "arc_left": dict(segs=[Seg(1, 8.0, 1.0, curvature=lambda s: 0.15)]),
    "arc_right": dict(segs=[Seg(1, 8.0, 1.0, curvature=lambda s: -0.15)]),
    "reverse_arc_left": dict(segs=[Seg(-1, 6.0, 0.8, curvature=lambda s: 0.15)]),
    "reverse_arc_right": dict(segs=[Seg(-1, 6.0, 0.8, curvature=lambda s: -0.15)]),
    "s_curve": dict(segs=[Seg(1, 14.0, 1.2, curvature=S_CURVE)]),
    "cusp_forward_reverse": dict(segs=[Seg(1, 5.0, 1.0, curvature=lambda s: 0.1), Seg(dwell=2.0),
                                       Seg(-1, 5.0, 0.8, curvature=lambda s: -0.12)]),
    "yaw_wrap": dict(segs=[Seg(1, 8.0, 1.0, curvature=lambda s: 0.15)], yaw0=math.pi - 0.3),
}


def assert_tracking(res, lat_tol=LAT_TOL):
    log = res.log
    track = np.array([m.startswith("track") for m in log.mode])
    assert track.sum() > 50
    lat = np.abs(log.np("lat")[track])
    assert lat.max() < lat_tol, lat.max()
    pos, yaw, v = res.final_errors()
    assert pos < POS_TOL and yaw < YAW_TOL and v <= V_TOL, (pos, yaw, v)
    check_constraints(res)
    check_gear_changes(res)
    assert all(s in ("", "solved", "solved inaccurate") for s in log.status)
    return lat.max()


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenario_tracking(name):
    sc = SCENARIOS[name]
    m = build_master(sc["segs"], yaw0=sc.get("yaw0", 0.3))
    res = run_closed_loop(m)
    assert_tracking(res)
    # stable standstill at the end: hold, no motion, no chattering
    log = res.log
    tail = [i for i, t in enumerate(log.t) if t > m.duration + 1.5]
    assert all(log.mode[i] == "hold" for i in tail)
    assert np.ptp(log.np("acc")[tail]) < 1e-9
    assert max(abs(log.v[i]) for i in tail) < 1e-6


def test_cusp_gear_change_only_at_standstill():
    sc = SCENARIOS["cusp_forward_reverse"]
    m = build_master(sc["segs"], yaw0=0.3)
    res = run_closed_loop(m)
    changes = res.plant.gear_changes
    assert [(c[1], c[2]) for c in changes] == [(Gear.PARK, Gear.DRIVE), (Gear.DRIVE, Gear.REVERSE)]
    log = res.log
    # the command never asks for propulsion against the gear sign while shifting
    for i, md in enumerate(log.mode):
        if md == "gear_change" and abs(log.v[i]) < 0.03:
            assert abs(log.acc[i]) <= 0.8 + 1e-9
    # the shift happens at the planned cusp (first REVERSE point of the plan)
    k = int(np.argmax(m.gear == int(Gear.REVERSE)))
    _, _, _, _, gx, gy = changes[1]
    assert math.hypot(gx - m.x[k], gy - m.y[k]) < 0.05


@pytest.mark.parametrize("time_step", [0.02, 0.05, 0.1])
def test_independent_of_trajectory_time_step(time_step):
    m = build_master(SCENARIOS["s_curve"]["segs"])
    res = run_closed_loop(m, make_config(timing__trajectory_time_step=time_step))
    assert_tracking(res)


@pytest.mark.parametrize("horizon", [2.0, 5.0, 10.0])
def test_independent_of_trajectory_horizon(horizon):
    m = build_master(SCENARIOS["cusp_forward_reverse"]["segs"])
    res = run_closed_loop(m, make_config(timing__trajectory_horizon=horizon))
    assert_tracking(res)


@pytest.mark.parametrize("hz", [20.0, 50.0, 100.0])
def test_controller_rates(hz):
    m = build_master(SCENARIOS["s_curve"]["segs"])
    res = run_closed_loop(m, make_config(timing__controller_hz=hz))
    assert_tracking(res)


def test_actuator_delay_compensation():
    """Dead time 0.1 s + lag 0.1 s on both channels: compensation keeps tracking tight."""
    pc = PlantConfig(steering_dead_time=0.1, steering_time_constant=0.1,
                     acceleration_dead_time=0.1, acceleration_time_constant=0.1)
    over = dict(vehicle__actuator__steering_dead_time=0.1, vehicle__actuator__steering_time_constant=0.1,
                vehicle__actuator__acceleration_dead_time=0.1, vehicle__actuator__acceleration_time_constant=0.1)
    m = build_master(SCENARIOS["s_curve"]["segs"])
    on = run_closed_loop(m, make_config(**over), plant_cfg=pc)
    lat_on = assert_tracking(on)
    off = run_closed_loop(m, make_config(control__controller__mpc__delay_compensation=False, **over), plant_cfg=pc)
    track = np.array([md.startswith("track") for md in off.log.mode])
    lat_off = np.abs(off.log.np("lat")[track]).max()
    assert lat_on < lat_off


def test_longitudinal_error_at_stationary_reference_is_corrected():
    """Vehicle stopped 8 cm short of a stationary reference: it creeps there, then holds."""
    m = build_master([Seg(1, 4.0, 1.0, ramp_time=2.0)])
    end = (m.x[-1], m.y[-1], m.yaw[-1])
    yaw = end[2]
    x0 = (end[0] - 0.08 * math.cos(yaw), end[1] - 0.08 * math.sin(yaw), yaw)
    # reference = final stationary state only (time window starting after the end)
    from test_controller_sim import window
    cfg = make_config()
    tm = cfg["timing"]
    res = run_closed_loop(m, cfg, x0=x0, t_end=8.0,
                          publish=lambda t: window(m, m.duration + t, tm["trajectory_time_step"],
                                                   tm["trajectory_horizon"], tm["trajectory_validity"],
                                                   stamp=t))
    pos, yaw_e, v = res.final_errors()
    assert pos < 0.03 and v <= V_TOL
    assert res.log.mode[-1] == "hold"
    check_constraints(res)
