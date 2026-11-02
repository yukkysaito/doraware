"""End to end: B2 planned by the State Lattice planner -> profile -> Trajectory windows."""
import time

import numpy as np
import pytest

from doraware.core.config import load_config
from doraware.interfaces import Gear, to_ns
from doraware.planning.plugins.state_lattice.adapter import StateLatticeAdapter
from doraware.planning.trajectory_generation.generator import TrajectoryGenerator, TrajectoryStatus

from helpers.path_fixtures import check_limits, sample_profile
from helpers.scenario_request import request_of

CFG = load_config()


@pytest.fixture(scope="module")
def planned():
    req = request_of("B2")
    res = StateLatticeAdapter().plan(req)
    assert res.success
    return req, res


@pytest.fixture(scope="module")
def generator(planned):
    req, res = planned
    gen = TrajectoryGenerator(CFG["planning"]["trajectory_generation"], CFG["timing"], req.vehicle)
    t = time.perf_counter()
    assert gen.on_planning_result(res, req.map) == TrajectoryStatus.OK, gen.message
    gen.elapsed = time.perf_counter() - t
    return gen


def test_b2_profile(generator, planned):
    req, res = planned
    prof = generator.profile
    s = prof.summary()
    print(f"\nB2 profile: duration {s['duration']:.2f} s, segments {s['segments']} gears {s['gears']}, "
          f"lengths {np.round(s['lengths'], 2).tolist()} m, segment durations "
          f"{np.round(s['segment_durations'], 2).tolist()} s, max|v| {s['max_abs_velocity']:.2f} m/s, "
          f"holds {np.round(s['hold_durations'], 2).tolist()} s, generation {generator.elapsed * 1e3:.0f} ms")
    assert s["segments"] == 2 and s["gears"] == [int(Gear.DRIVE), int(Gear.REVERSE)]
    assert s["hold_durations"][1] >= CFG["planning"]["trajectory_generation"]["stop_duration"]
    assert generator.elapsed < 1.0
    lim = generator.limits
    m = check_limits(sample_profile(prof), lim)
    assert m["v_fwd"] > 0.5
    end = prof.sample(prof.duration + 1.0)
    assert (end.x, end.y) == pytest.approx((res.path.x[-1], res.path.y[-1]), abs=1e-9)
    assert end.velocity == 0.0
    assert generator.revalidation.ok


def test_b2_windows(generator):
    dur = generator.profile.duration
    prev = None
    for t in np.arange(0.0, dur + 2.0, 0.1):
        tr = generator.trajectory_at(to_ns(t))
        assert len(tr) == 101 and tr.start_time == to_ns(t)
        if prev is not None:
            assert np.allclose(prev.x[2:], tr.x[:-2], atol=1e-9)
            assert np.allclose(prev.longitudinal_velocity[2:], tr.longitudinal_velocity[:-2], atol=1e-9)
        prev = tr
    assert np.all(prev.longitudinal_velocity == 0.0)
