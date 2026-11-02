"""Fault handling and trajectory-update tests of the TrajectoryFollower."""
from __future__ import annotations

import numpy as np

from doraware.control.plugins.mpc.follower import TrajectoryFollower
from doraware.interfaces import Gear, VehicleParameters, to_ns

from test_controller_sim import (Seg, build_master, check_constraints, make_config, run_closed_loop,
                                 window)
from test_controller_tracking import S_CURVE

MASTER = build_master([Seg(1, 14.0, 1.2, curvature=S_CURVE)])


def _stops_safely(res, t_fault: float, decel: float):
    log = res.log
    after = [i for i, t in enumerate(log.t) if t >= t_fault]
    acc = log.np("acc")[after]
    assert np.all(np.abs(acc) <= decel + 1e-9)
    assert abs(res.plant.v) < 1e-6
    check_constraints(res)
    # steering held during the fallback
    fb = [i for i in after if log.mode[i] == "fallback"]
    assert np.ptp(log.np("steer")[fb]) < 1e-12


def test_solver_failure_reuses_previous_then_safe_stop():
    cfg = make_config(fault_injection__solver_failure_after=6.0)
    res = run_closed_loop(MASTER, cfg, t_end=14.0)
    log = res.log
    after = [log.mode[i] for i, t in enumerate(log.t) if t >= 6.0 and log.mode[i] != "hold"]
    n_reuse = cfg["control"]["controller"]["mpc"]["max_consecutive_failures"]
    assert after[:n_reuse] == ["track_reuse"] * n_reuse
    assert set(after[n_reuse:]) == {"fallback"}
    assert all(s == "forced_failure" for i, s in enumerate(log.status) if log.t[i] >= 6.0)
    _stops_safely(res, 6.0, cfg["control"]["controller"]["mpc"]["fallback_deceleration"])
    assert log.v[log.t.index(6.0)] > 0.5      # it was moving when the fault started


def test_expired_trajectory_safe_stop():
    cfg = make_config()
    tm = cfg["timing"]
    pub = lambda t: (window(MASTER, t, tm["trajectory_time_step"], tm["trajectory_horizon"],
                            tm["trajectory_validity"], stamp=t) if t <= 5.0 + 1e-9 else None)
    res = run_closed_loop(MASTER, cfg, publish=pub, t_end=14.0)
    log = res.log
    expiry = 5.0 + tm["trajectory_validity"]
    for i, t in enumerate(log.t):
        if 2.0 < t <= expiry - 1e-9:
            assert log.mode[i].startswith("track")         # temporarily stale is tolerated
        if t > expiry + 1e-9:
            assert log.mode[i] == "fallback"                # never follow an expired trajectory
    _stops_safely(res, expiry, cfg["control"]["controller"]["mpc"]["fallback_deceleration"])


def test_moving_end_without_update_safe_stop():
    """A short window that is never refreshed: beyond its moving end it is undefined."""
    cfg = make_config()
    once = {"done": False}

    def pub(t):
        if once["done"]:
            return None
        once["done"] = True
        return window(MASTER, 0.0, 0.05, 4.5, validity=100.0)

    res = run_closed_loop(MASTER, cfg, publish=pub, t_end=10.0)
    log = res.log
    assert all(log.mode[i] == "fallback" for i, t in enumerate(log.t) if t > 4.5 + 1e-9)
    _stops_safely(res, 4.5, cfg["control"]["controller"]["mpc"]["fallback_deceleration"])


def test_no_trajectory_stays_still():
    res = run_closed_loop(MASTER, make_config(), publish=lambda t: None, t_end=3.0)
    assert set(res.log.mode) == {"fallback"}
    assert all(g == int(Gear.NONE) for g in res.log.gear_cmd)
    assert res.plant.x == MASTER.x[0] and res.plant.v == 0.0


def _single_plan(t):
    return window(MASTER, 0.0, 0.05, MASTER.duration + 5.0, validity=1e3) if t == 0.0 else None


def test_republished_windows_cause_no_command_jumps():
    """10 Hz windows of the same plan give the same commands as one long trajectory."""
    base = run_closed_loop(MASTER, make_config(), publish=_single_plan)
    rep = run_closed_loop(MASTER, make_config())
    assert np.max(np.abs(base.log.np("acc") - rep.log.np("acc"))) < 0.02
    assert np.max(np.abs(base.log.np("steer") - rep.log.np("steer"))) < 0.005
    check_constraints(rep)


def test_late_trajectory_arrival_is_time_consistent():
    """Windows delivered 0.3 s late (stamped at generation) are sampled by sim
    time, not from point 0: commands stay equal to the on-time case."""
    cfg = make_config()
    tm = cfg["timing"]

    def late(t):
        tg = round(t - 0.3, 6)
        if tg < 0.0:
            return window(MASTER, 0.0, tm["trajectory_time_step"], tm["trajectory_horizon"], 1.5,
                          stamp=0.0) if t == 0.0 else None
        return window(MASTER, tg, tm["trajectory_time_step"], tm["trajectory_horizon"], 1.5, stamp=tg)

    base = run_closed_loop(MASTER, cfg)
    res = run_closed_loop(MASTER, cfg, publish=late)
    assert "fallback" not in res.log.mode
    assert np.max(np.abs(base.log.np("acc") - res.log.np("acc"))) < 0.02
    assert np.max(np.abs(base.log.np("steer") - res.log.np("steer"))) < 0.005


def test_stale_and_out_of_order_trajectories_are_ignored():
    f = TrajectoryFollower(make_config(), VehicleParameters())
    new = window(MASTER, 2.0, 0.05, 5.0, stamp=2.0)
    old = window(MASTER, 1.0, 0.05, 5.0, stamp=1.0)
    assert f.on_trajectory(new)
    assert not f.on_trajectory(old)
    assert not f.on_trajectory(window(MASTER, 2.0, 0.05, 5.0, stamp=2.0))   # same stamp
    bad = window(MASTER, 3.0, 0.05, 5.0, stamp=3.0)
    bad.y = bad.y[:-1]
    assert not f.on_trajectory(bad)
    assert f.trajectory is new


def test_command_header_and_validity():
    cfg = make_config()
    f = TrajectoryFollower(cfg, VehicleParameters())
    f.on_trajectory(window(MASTER, 0.0, 0.05, 5.0))
    from test_controller_sim import Plant
    st = Plant(MASTER.x[0], MASTER.y[0], MASTER.yaw[0]).state()
    cmd, dbg = f.step(st, to_ns(0.0))
    assert cmd.valid_until == to_ns(cfg["timing"]["command_validity"])
    assert cmd.gear_request == Gear.DRIVE and dbg.mode == "gear_change"
