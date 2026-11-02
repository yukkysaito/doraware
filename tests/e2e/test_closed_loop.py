"""End-to-end closed-loop tests: full Dora dataflow + MPC + Godot physics.

Slow (each case is a complete simulation). Run with ``pytest tests/e2e -m e2e``.
Results are kept under results/runs/e2e/<test>/ for inspection.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from doraware.core.config import REPO_ROOT
from doraware.evaluation.suite.runner import run_scenario

pytestmark = pytest.mark.e2e
OUT = REPO_ROOT / "results" / "runs" / "e2e"


def _run(name, sid="B2", overrides=(), record=True, seed=42):
    r = run_scenario(sid, OUT / name, list(overrides), False, seed, record)
    return r, r.get("metrics") or {}


def _assert_goal(r, m):
    assert r["passed"], r.get("reasons")
    assert r["outcome"] == "goal_reached"
    assert m["final_position_error"] <= 0.10 and m["final_yaw_error_deg"] <= 2.0 and m["final_speed"] <= 0.02
    assert m["collision_count"] == 0 and m["out_of_bounds_events"] == 0


def test_b2_default_rates():
    """Planner 10 Hz, controller 50 Hz, physics 100 Hz; reverse parking with a gear change."""
    r, m = _run("b2_default")
    _assert_goal(r, m)
    assert m["gear_changes"] >= 1 and m["gear_changes_while_moving"] == 0
    assert 49.0 <= m["controller_frequency"] <= 51.0
    assert (Path(r["run_dir"]) / "recording.mcap").stat().st_size > 0
    # default signal policy (maneuver_hazard): hazard lights for the whole manoeuvre, off once parked
    assert m["signal_hazard_activations"] == 1 and m["signal_first_hazard_time"] <= 0.1
    assert m["signal_final_hazard_lights"] is False
    assert m["signal_left_activations"] == 0 and m["signal_right_activations"] == 0
    assert m["signal_lamp_inconsistencies"] == 0


def test_turn_indicator_policy():
    """turn_indicators policy: the turns are indicated; a plan that is followed needs no hazard lights."""
    r, m = _run("b2_turn_indicators", overrides=["planning.signals.policy=turn_indicators"], record=False)
    _assert_goal(r, m)
    assert m["signal_left_activations"] >= 1 and m["signal_right_activations"] >= 1
    assert m["signal_hazard_activations"] == 0 and m["signal_final_turn_indicator"] == "OFF"
    assert m["signal_lamp_inconsistencies"] == 0


@pytest.mark.parametrize("dt", [0.02, 0.1])
def test_trajectory_time_step(dt):
    r, m = _run(f"b2_dt_{dt}", overrides=[f"timing.trajectory_time_step={dt}"], record=False)
    _assert_goal(r, m)


@pytest.mark.parametrize("horizon", [2.0, 10.0])
def test_trajectory_horizon(horizon):
    r, m = _run(f"b2_h_{horizon}", overrides=[f"timing.trajectory_horizon={horizon}"], record=False)
    _assert_goal(r, m)


@pytest.mark.parametrize("rates", [(20, 5), (100, 20), (30, 10)])
def test_independent_rates(rates):
    ctrl, plan = rates
    r, m = _run(f"b2_rates_{ctrl}_{plan}", overrides=[f"timing.controller_hz={ctrl}", f"timing.planner_hz={plan}",
                                                      f"timing.state_hz={ctrl}"], record=False)
    _assert_goal(r, m)


def test_actuator_delays():
    r, m = _run("b2_delays", overrides=["vehicle.actuator.steering_dead_time=0.1", "vehicle.actuator.steering_time_constant=0.2",
                                        "vehicle.actuator.acceleration_dead_time=0.1",
                                        "vehicle.actuator.acceleration_time_constant=0.3"], record=False)
    _assert_goal(r, m)


def _states(run_dir):
    from mcap.reader import make_reader
    out = []
    with open(Path(run_dir) / "recording.mcap", "rb") as f:
        for _, _ch, msg in make_reader(f).iter_messages(topics=["vehicle_state"]):
            d = json.loads(msg.data)
            p = d["pose"]["position"]
            out.append((d["header"]["stamp"], p["x"], p["y"], d["twist"]["linear"]["x"], d["steering_angle"]))
    return np.array(out)


def _signals(run_dir):
    from mcap.reader import make_reader
    out = []
    with open(Path(run_dir) / "recording.mcap", "rb") as f:
        for _, _ch, msg in make_reader(f).iter_messages(topics=["vehicle_signal_status"]):
            d = json.loads(msg.data)
            out.append((d["header"]["stamp"], d["turn_indicator"], d["hazard_lights"], d["left_lamp_on"],
                        d["right_lamp_on"]))
    return np.array(out, dtype=np.int64)


def test_determinism():
    """Same scenario and seed twice -> identical vehicle state sequences, also when the
    simulator starts late (the plan is then ready before the first vehicle state)."""
    states = _states
    r1, m1 = _run("det_1")
    r2, m2 = _run("det_2", overrides=["fault_injection.simulator_start_delay=5.0"])
    a, b = states(r1["run_dir"]), states(r2["run_dir"])
    # the evaluator's verdict time is deterministic; after it the bridge may run a few
    # extra (asynchronous) ticks before it receives scenario_done, so compare up to it
    assert r1["end_sim_time"] == r2["end_sim_time"]
    t_end = int(round(r1["end_sim_time"] * 1e9))
    a, b = a[a[:, 0] <= t_end], b[b[:, 0] <= t_end]
    assert len(a) == len(b) and len(a) > 100
    assert np.array_equal(a, b)
    sa, sb = _signals(r1["run_dir"]), _signals(r2["run_dir"])
    sa, sb = sa[sa[:, 0] <= t_end], sb[sb[:, 0] <= t_end]
    assert len(sa) > 100 and np.array_equal(sa, sb)      # signal decisions are part of the lockstep
    for k in ("final_position_error", "final_yaw_error_deg", "lateral_error_max", "min_clearance_actual"):
        assert m1[k] == m2[k]


def test_solver_failure_safe_stop():
    r, m = _run("fault_solver", overrides=["fault_injection.solver_failure_after=6.0"], record=False)
    assert r["outcome"] in ("timeout", "stopped") and m["collision_count"] == 0
    assert m.get("controller_modes", {}).get("fallback", 0) > 0
    assert m["final_speed"] <= 0.02


def test_planner_silence_safe_stop():
    r, m = _run("fault_planner", overrides=["fault_injection.planner_silence_after=6.0",
                                            "timing.lockstep_wall_timeout=2.0",
                                            "planning.signals.policy=turn_indicators"], record=False)
    assert m["collision_count"] == 0 and m["final_speed"] <= 0.02
    assert m.get("controller_modes", {}).get("fallback", 0) > 0
    # trajectories expired: hazard lights from then on
    assert m["signal_first_hazard_time"] is not None and m["signal_first_hazard_time"] > 6.0
    assert m["signal_final_hazard_lights"] is True


def test_controller_silence_emergency_brake():
    r, m = _run("fault_controller", overrides=["fault_injection.controller_silence_after=6.0",
                                               "timing.lockstep_wall_timeout=1.0", "evaluation.max_sim_time=20.0"],
                record=False)
    assert m["command_timeouts"] >= 1 and m["collision_count"] == 0 and m["final_speed"] <= 0.02


def test_infeasible_scenario_rejected():
    r, m = _run("g1", sid="G1", record=False)
    assert r["passed"] and r["outcome"] == "rejected", r.get("reasons")
    # no plan: hazard lights from the first tick, no turn indicator
    assert m["signal_first_hazard_time"] is not None and m["signal_first_hazard_time"] <= 0.1
    assert m["signal_final_hazard_lights"] is True
    assert m["signal_left_activations"] == 0 and m["signal_right_activations"] == 0


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="GUI needs a display")
def test_gui_matches_headless():
    """Rendering (GUI, real-time pacing) must not change the physics."""
    rh = run_scenario("B2", OUT / "gui_vs_headless" / "headless", [], False, 42, True)
    rg = run_scenario("B2", OUT / "gui_vs_headless" / "gui", ["timing.realtime_factor=4.0"], True, 42, True)
    a, b = _states(rh["run_dir"]), _states(rg["run_dir"])
    n = min(len(a), len(b))
    assert n > 100 and rh["passed"] and rg["passed"]
    assert np.array_equal(a[:n], b[:n])


def test_replay_state_playback_and_command_resimulation():
    """Recording/replay: (a) state playback reproduces the online metrics without physics;
    (b) re-injecting the recorded commands into fresh physics reproduces the recorded states."""
    from doraware.recording.replay import replay_commands, replay_states
    r, m = _run("replay_src")
    run_dir = Path(r["run_dir"])
    s = replay_states(run_dir, plot=True)
    for k in ("final_position_error", "final_yaw_error_deg", "lateral_error_max", "min_clearance_actual"):
        assert s[k] == pytest.approx(m[k], abs=1e-9), k
    c = replay_commands(run_dir)
    assert c["compared_states"] > 100 and c["compared_signal_states"] > 100 and c["identical"], c


def test_planner_crash_restart_recovers():
    """Dora restarts the crashed planner; the scenario manager re-sends the request; the run succeeds."""
    r, m = _run("fault_planner_crash", overrides=["fault_injection.planner_crash_once=true"], record=False)
    _assert_goal(r, m)
    log = (Path(r["run_dir"]) / "dora.log").read_text()
    assert "restarting node after failure" in log and "re-sending the outstanding planning request" in log
