"""TimedProfile: speed profile limits, cusps / dwell, signed velocity, end of path."""
import math
import time

import numpy as np
import pytest

from doraware.core.config import REPO_ROOT, load_config
from doraware.interfaces import Gear
from doraware.planning.trajectory_generation.limits import ProfileLimits
from doraware.planning.trajectory_generation.path_processing import build_segments, split_at_cusps
from doraware.planning.trajectory_generation.profile import build_profile

from helpers.path_fixtures import VEHICLE, arc_path, check_limits, sample_profile

CFG = load_config()
LIM = ProfileLimits.from_config(CFG["planning"]["trajectory_generation"], VEHICLE)
MAX_STEER = VEHICLE.max_steer_angle
L = VEHICLE.wheel_base

# forward with steering steps (straight -> full left -> full right -> straight), cusp, reverse arc
PARKING = [(+1, 0.0, 3.0), (+1, MAX_STEER, 2.5), (+1, -MAX_STEER, 1.5), (+1, 0.0, 2.0),
           (-1, -MAX_STEER, 3.0), (-1, 0.0, 2.0)]


def profile_of(pieces, lim=LIM, **kw):
    return build_profile(arc_path(pieces), L, MAX_STEER, lim, **kw)


def test_limits_respected_on_sampled_profile():
    prof = profile_of(PARKING)
    arr = sample_profile(prof)
    m = check_limits(arr, LIM)
    assert m["v_fwd"] > 0.5 and m["v_rev"] > 0.5          # not trivially slow


def test_signed_velocity_and_gear_on_reverse_segment():
    prof = profile_of(PARKING)
    arr = sample_profile(prof)
    v, g = arr["velocity"], arr["gear"]
    assert np.all(g[v < -1e-9] == int(Gear.REVERSE))
    assert np.all(g[v > 1e-9] == int(Gear.DRIVE))
    assert np.any(v < -0.3)
    # reversing: position moves against the heading
    i = int(np.argmin(v))
    dx, dy = arr["x"][i + 1] - arr["x"][i - 1], arr["y"][i + 1] - arr["y"][i - 1]
    assert dx * math.cos(arr["yaw"][i]) + dy * math.sin(arr["yaw"][i]) < 0
    # signed acceleration: speeding up backwards is negative
    seg_rev = (g == int(Gear.REVERSE)) & (v < -1e-3)
    first = np.argmax(seg_rev)
    assert arr["acceleration"][first + 5] < 0


def test_cusp_stop_and_dwell():
    prof = profile_of(PARKING)
    summ = prof.summary()
    assert summ["segments"] == 2 and summ["gears"] == [int(Gear.DRIVE), int(Gear.REVERSE)]
    seg0_end = prof.starts[2]                     # phases: start hold, motion 0, dwell, motion 1
    dwell = prof.phases[2]
    # steering rotation at standstill: 0 -> -max_steer
    assert dwell.duration == pytest.approx(max(LIM.stop_duration, MAX_STEER / LIM.max_steer_rate))
    for tau in np.linspace(0.0, dwell.duration - 1e-6, 25):
        p = prof.sample(seg0_end + tau)
        assert p.velocity == 0.0 and p.acceleration == 0.0
        assert p.gear == Gear.REVERSE                  # next segment's gear while stopped
    # arrives at the cusp with v -> 0, leaves it from v = 0
    assert abs(prof.sample(seg0_end - 1e-4).velocity) < 1e-2
    assert abs(prof.sample(prof.starts[3] + 1e-4).velocity) < 1e-2
    # steering moves during the dwell at the rate limit
    d0, d1 = prof.sample(seg0_end).steering, prof.sample(prof.starts[3]).steering
    assert d0 == pytest.approx(0.0, abs=1e-9) and d1 == pytest.approx(-MAX_STEER, abs=1e-9)


def test_stop_duration_dominates_when_steering_matches():
    prof = profile_of([(+1, 0.0, 3.0), (-1, 0.0, 3.0)])
    assert prof.phases[2].duration == pytest.approx(LIM.stop_duration)


def test_profile_ends_exactly_at_path_end_and_holds():
    path = arc_path(PARKING)
    prof = build_profile(path, L, MAX_STEER, LIM)
    end = prof.sample(prof.duration)
    assert (end.x, end.y) == pytest.approx((path.x[-1], path.y[-1]), abs=1e-9)
    assert math.cos(end.yaw - path.yaw[-1]) == pytest.approx(1.0, abs=1e-12)
    late = prof.sample(prof.duration + 100.0)
    assert late.velocity == 0.0 and late.acceleration == 0.0 and (late.x, late.y) == (end.x, end.y)
    assert late.gear == Gear.REVERSE
    # just before the end it is (almost) stopped and on the path
    assert abs(prof.sample(prof.duration - 1e-3).velocity) < 1e-2


def test_start_hold_and_initial_steering():
    prof = profile_of([(+1, MAX_STEER, 3.0)], initial_steering=-MAX_STEER)
    hold = prof.phases[0]
    assert hold.duration == pytest.approx(2 * MAX_STEER / LIM.max_steer_rate)
    assert prof.sample(0.0).steering == pytest.approx(-MAX_STEER)
    assert prof.sample(hold.duration).steering == pytest.approx(MAX_STEER)
    assert prof.sample(-1.0).velocity == 0.0
    check_limits(sample_profile(prof), LIM)


def test_split_at_cusps_shares_cusp_point():
    path = arc_path([(+1, 0.0, 1.0), (-1, 0.3, 1.0), (+1, 0.0, 1.0)])
    parts = split_at_cusps(path)
    assert [d for d, _ in parts] == [1, -1, 1]
    assert parts[0][1][-1] == parts[1][1][0] and parts[1][1][-1] == parts[2][1][0]


def test_steering_ramp_is_continuous_and_bounded():
    segs = build_segments(arc_path(PARKING), L, MAX_STEER, LIM.path_resample_ds, LIM.steer_transition_length)
    seg = segs[0]
    # linear curvature ramp: |d kappa/ds| <= dk / w, |d delta/ds| <= L |d kappa/ds| -> no jumps
    k_max = math.tan(MAX_STEER) / L
    slope = np.abs(np.diff(seg.grid_delta)) / np.diff(seg.grid_s)
    assert slope.max() <= L * 2 * k_max / LIM.steer_transition_length + 1e-6
    assert seg.grid_delta[0] == pytest.approx(0.0) and seg.grid_delta[-1] == pytest.approx(0.0)
    # lateral deviation of driving the ramped curvature vs the planner geometry (re-integrated)
    s = np.linspace(0, seg.length, 20001)
    k = np.tan(np.interp(s, seg.grid_s, seg.grid_delta)) / L
    yaw = seg.yaw[0] + np.concatenate([[0], np.cumsum(0.5 * (k[1:] + k[:-1]) * np.diff(s))])
    x = seg.x[0] + np.concatenate([[0], np.cumsum(np.cos(0.5 * (yaw[1:] + yaw[:-1])) * np.diff(s))])
    y = seg.y[0] + np.concatenate([[0], np.cumsum(np.sin(0.5 * (yaw[1:] + yaw[:-1])) * np.diff(s))])
    ref = np.column_stack([np.interp(s, seg.s, seg.x), np.interp(s, seg.s, seg.y)])
    dev = np.hypot(x - ref[:, 0], y - ref[:, 1]).max()
    bound = seg.ramp_deviation_bound(LIM.steer_transition_length)
    assert dev <= bound * 1.2 + 1e-3, (dev, bound)
    assert bound < 0.02


@pytest.mark.parametrize("cfg_change", [
    dict(max_jerk=0.5), dict(max_acceleration=0.3, max_deceleration=1.0), dict(max_steer_rate=0.2),
    dict(max_lateral_acceleration=0.3), dict(steer_transition_length=1.0), dict(path_resample_ds=0.1),
])
def test_limits_for_other_settings(cfg_change):
    lim = ProfileLimits(**{**LIM.__dict__, **cfg_change})
    check_limits(sample_profile(profile_of(PARKING, lim)), lim)


def test_short_segment_and_start_equals_goal():
    prof = profile_of([(+1, 0.0, 2.0), (-1, 0.0, 0.08), (+1, 0.0, 1.0)])
    check_limits(sample_profile(prof), LIM)
    assert prof.summary()["segments"] == 3
    from doraware.interfaces import GeometricPath
    single = GeometricPath(x=np.array([1.0]), y=np.array([2.0]), yaw=np.array([0.5]), curvature=np.zeros(1),
                           gear=np.array([int(Gear.DRIVE)], dtype=np.int8))
    p = build_profile(single, L, MAX_STEER, LIM)
    assert p.duration == 0.0 and p.sample(3.0).velocity == 0.0 and p.sample(3.0).x == 1.0


def test_generation_time():
    path = arc_path(PARKING * 2)
    t = time.perf_counter()
    build_profile(path, L, MAX_STEER, LIM)
    elapsed = time.perf_counter() - t
    print(f"profile generation ({sum(p[2] for p in PARKING) * 2:.0f} m path): {elapsed * 1e3:.1f} ms")
    assert elapsed < 1.0


def test_long_curved_road_path_solves_within_limits():
    """A 45 m road path with a 15 m curve (~900 nodes) exceeds the strict QP budget;
    the relaxed retry must solve it (no slow fallback) within all limits."""
    from doraware.evaluation.scenarios.road import RoadScenarios
    from doraware.interfaces import PlanningResult, PlanningStatus
    from doraware.planning.trajectory_generation.generator import TrajectoryGenerator
    from doraware.planning.trajectory_planning.plugins.lane_follow.route import route_centerline
    cfg = load_config(path=REPO_ROOT / "configs" / "road.yaml")
    req = RoadScenarios().build("R2", 42, cfg).request
    path = route_centerline(req.lanes, req.route, req.start, req.goal)
    gen = TrajectoryGenerator(cfg["planning"]["trajectory_generation"], cfg["timing"], req.vehicle)
    gen.on_planning_result(PlanningResult(status=PlanningStatus.SUCCESS, path=path.geometric(0, len(path.s) - 1)),
                           None, plan_start_ns=0, initial_steering=0.0)
    assert not gen.speed_profile_fallback, gen.profile.summary()["solver_status"]
    assert gen.profile.duration < 30.0
    check_limits(sample_profile(gen.profile), gen.limits)
