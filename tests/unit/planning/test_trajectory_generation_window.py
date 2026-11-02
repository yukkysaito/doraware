"""Fixed-time-step Trajectory windows and the TrajectoryGenerator application class."""
import numpy as np
import pytest

from doraware.core.config import load_config
from doraware.interfaces import (GeometricPath, Gear, PlanningResult, PlanningStatus, to_ns)
from doraware.planning.trajectory_generation.generator import TrajectoryGenerator, TrajectoryStatus
from doraware.planning.trajectory_generation.limits import ProfileLimits
from doraware.planning.trajectory_generation.profile import build_profile
from doraware.core.trajectory import sample
from doraware.planning.trajectory_generation.window import make_trajectory, window_size

from helpers.path_fixtures import VEHICLE, arc_path, free_grid

CFG = load_config()
LIM = ProfileLimits.from_config(CFG["planning"]["trajectory_generation"], VEHICLE)
PATH = arc_path([(+1, 0.0, 2.0), (+1, 0.4, 2.0), (-1, -0.3, 2.5)])
PROFILE = build_profile(PATH, VEHICLE.wheel_base, VEHICLE.max_steer_angle, LIM)


@pytest.mark.parametrize("dt", [0.02, 0.05, 0.1])
@pytest.mark.parametrize("horizon", [2.0, 5.0, 10.0])
def test_fixed_step_and_start_time(dt, horizon):
    plan_start, t_now = to_ns(1.0), to_ns(3.3)
    tr = make_trajectory(PROFILE, plan_start, t_now, dt, horizon, validity=1.0, header_seq=7)
    n = int(round(horizon / dt)) + 1
    assert len(tr) == n == window_size(dt, horizon)
    assert tr.start_time == t_now and tr.header.stamp == t_now and tr.header.sequence_id == 7
    assert tr.time_step == dt and tr.valid_until == t_now + to_ns(1.0)
    assert tr.end_time == t_now + to_ns((n - 1) * dt)
    for arr in (tr.y, tr.yaw, tr.longitudinal_velocity, tr.longitudinal_acceleration, tr.curvature,
                tr.front_wheel_angle, tr.gear):
        assert len(arr) == n
    # point i is the profile state at start_time + i * dt
    for i in (0, n // 3, n - 1):
        p = PROFILE.sample(2.3 + i * dt)
        assert tr.x[i] == pytest.approx(p.x) and tr.longitudinal_velocity[i] == pytest.approx(p.velocity)
        assert tr.gear[i] == int(p.gear)
    assert np.all((tr.yaw >= -np.pi) & (tr.yaw < np.pi))


def test_windows_are_mutually_consistent():
    """Windows from different t_now agree at common absolute times (and with sampling.sample)."""
    w1 = make_trajectory(PROFILE, 0, to_ns(4.0), 0.05, 5.0, 1.0)
    w2 = make_trajectory(PROFILE, 0, to_ns(4.1), 0.05, 5.0, 1.0)
    # w2[i] is at 4.1 + 0.05 i = w1[i + 2]
    for f in ("x", "y", "yaw", "longitudinal_velocity", "longitudinal_acceleration", "front_wheel_angle", "gear"):
        assert np.allclose(getattr(w1, f)[2:], getattr(w2, f)[:-2], atol=1e-9), f
    # an off-grid t_now: interpolating w1 matches w3's exact samples closely
    w3 = make_trajectory(PROFILE, 0, to_ns(4.0123), 0.05, 4.0, 1.0)
    for i in range(0, len(w3), 7):
        r = sample(w1, 4.0123 + i * 0.05)
        assert r.x == pytest.approx(w3.x[i], abs=2e-3) and r.velocity == pytest.approx(
            w3.longitudinal_velocity[i], abs=2e-2)


def test_window_before_start_and_after_end():
    before = make_trajectory(PROFILE, to_ns(10.0), to_ns(8.0), 0.1, 1.0, 1.0)
    assert np.all(before.longitudinal_velocity == 0.0) and before.x[0] == pytest.approx(PATH.x[0])
    after = make_trajectory(PROFILE, 0, to_ns(PROFILE.duration + 5.0), 0.1, 2.0, 1.0)
    assert np.all(after.longitudinal_velocity == 0.0)
    assert after.x[-1] == pytest.approx(PATH.x[-1]) and after.y[-1] == pytest.approx(PATH.y[-1])
    assert np.all(after.gear == int(Gear.REVERSE))


def _ok_result(path=PATH):
    return PlanningResult(status=PlanningStatus.SUCCESS, path=path)


def test_generator_success_and_status():
    gen = TrajectoryGenerator(CFG["planning"]["trajectory_generation"], CFG["timing"], VEHICLE)
    assert gen.trajectory_at(0) is None and gen.status == TrajectoryStatus.WAITING
    assert gen.on_planning_result(_ok_result(), free_grid(), plan_start_ns=to_ns(0.5)) == TrajectoryStatus.OK
    tr = gen.trajectory_at(to_ns(0.5))
    assert len(tr) == int(round(CFG["timing"]["trajectory_horizon"] / CFG["timing"]["trajectory_time_step"])) + 1
    assert tr.time_step == CFG["timing"]["trajectory_time_step"]
    d = gen.diagnostics()
    assert d["status"] == "OK" and d["segments"] == 2 and d["revalidation_ok"]
    assert not gen.finished(to_ns(0.5)) and gen.finished(to_ns(0.5 + d["duration"]))
    assert gen.trajectory_at(to_ns(0.6)).header.sequence_id > tr.header.sequence_id


def test_generator_failed_plan_gives_empty_trajectory():
    gen = TrajectoryGenerator(CFG["planning"]["trajectory_generation"], CFG["timing"], VEHICLE)
    gen.on_planning_result(PlanningResult(status=PlanningStatus.NO_PATH, message="exhausted"), free_grid())
    tr = gen.trajectory_at(to_ns(1.0))
    assert tr is not None and len(tr) == 0 and tr.valid_until > tr.start_time
    assert gen.status == TrajectoryStatus.PLANNING_FAILED and gen.planning_status == PlanningStatus.NO_PATH
    assert "exhausted" in gen.message


def test_generator_invalid_path():
    gen = TrajectoryGenerator(CFG["planning"]["trajectory_generation"], CFG["timing"], VEHICLE)
    bad = GeometricPath(x=np.array([0.0, np.nan]), y=np.zeros(2), yaw=np.zeros(2), curvature=np.zeros(2),
                        gear=np.full(2, int(Gear.DRIVE), dtype=np.int8))
    assert gen.on_planning_result(_ok_result(bad), free_grid()) == TrajectoryStatus.INVALID_PATH
    assert len(gen.trajectory_at(0)) == 0


@pytest.mark.parametrize("dt,horizon", [(0.02, 2.0), (0.1, 10.0)])
def test_generator_timing_config(dt, horizon):
    timing = {**CFG["timing"], "trajectory_time_step": dt, "trajectory_horizon": horizon}
    gen = TrajectoryGenerator(CFG["planning"]["trajectory_generation"], timing, VEHICLE)
    gen.on_planning_result(_ok_result(), free_grid())
    tr = gen.trajectory_at(to_ns(2.0))
    assert len(tr) == int(round(horizon / dt)) + 1 and tr.time_step == dt
