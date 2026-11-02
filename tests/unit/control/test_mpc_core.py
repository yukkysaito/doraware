"""Unit tests of the MPC building blocks: model, discretisation, QP, reference, gear manager."""
from __future__ import annotations

import math

import numpy as np
import scipy.linalg

from doraware.control.gear import GearManager, GearPhase
from doraware.control.plugins.mpc.model import NU, NZ, discretize, rk4_step, rollout
from doraware.control.plugins.mpc.controller import LtvMpc, accel_bounds
from doraware.control.plugins.mpc.params import ControllerParams
from doraware.control.plugins.mpc.reference import build_reference, trajectory_usable
from doraware.interfaces import Gear, VehicleParameters, to_ns

from test_controller_sim import Seg, build_master, make_config, window

L = 2.79


def _jac(z, u):
    psi, v, d = z[2], z[3], z[4]
    a = np.zeros((NZ, NZ))
    b = np.zeros((NZ, NU))
    a[0, 2], a[0, 3] = -v * math.sin(psi), math.cos(psi)
    a[1, 2], a[1, 3] = v * math.cos(psi), math.sin(psi)
    a[2, 3], a[2, 4] = math.tan(d) / L, v / (L * math.cos(d) ** 2)
    a[3, 5] = 1.0
    b[4, 0] = b[5, 1] = 1.0
    return a, b


def test_discretisation_equals_matrix_exponential():
    rng = np.random.default_rng(0)
    zs = rng.normal(size=(5, NZ)) * [3, 3, 2, 1, 0.3, 0.5]
    us = rng.normal(size=(5, NU)) * [0.3, 1.0]
    zn = np.array([rk4_step(z, u, 0.1, L) for z, u in zip(zs, us)])
    ad, bd, d = discretize(zs, us, zn, 0.1, L)
    for k in range(5):
        a, b = _jac(zs[k], us[k])
        m = np.zeros((NZ + NU, NZ + NU))
        m[:NZ, :NZ], m[:NZ, NZ:] = a, b
        assert np.allclose(np.linalg.matrix_power(m, 5), 0.0)      # nilpotent -> finite series exact
        e = scipy.linalg.expm(m * 0.1)
        assert np.allclose(ad[k], e[:NZ, :NZ], atol=1e-12)
        assert np.allclose(bd[k], e[:NZ, NZ:], atol=1e-12)
        # affine term: exact at the nominal point
        assert np.allclose(ad[k] @ zs[k] + bd[k] @ us[k] + d[k], zn[k])


def test_linearisation_first_order_accurate():
    z = np.array([1.0, 2.0, 0.4, 1.2, 0.2, 0.3])
    u = np.array([0.1, -0.5])
    ad, bd, d = discretize(z[None], u[None], rk4_step(z, u, 0.1, L)[None], 0.1, L)
    dz = np.array([0.01, -0.01, 0.01, 0.02, 0.01, 0.02])
    du = np.array([0.02, 0.05])
    lin = ad[0] @ (z + dz) + bd[0] @ (u + du) + d[0]
    true = rk4_step(z + dz, u + du, 0.1, L)
    assert np.linalg.norm(lin - true) < 5e-4


def test_rk4_constant_curvature_circle():
    delta = 0.2
    r = L / math.tan(delta)
    z = np.array([0, 0, 0, 1.0, delta, 0.0])
    zs = rollout(z, np.zeros((63, NU)), 0.1, L)
    # rear axle stays on the circle of radius r centred at (0, r)
    assert np.allclose(np.hypot(zs[:, 0], zs[:, 1] - r), r, atol=1e-6)


def test_accel_bounds_mirror_in_reverse():
    assert accel_bounds(1, 1.5, 3.0) == (-3.0, 1.5)
    assert accel_bounds(-1, 1.5, 3.0) == (-1.5, 3.0)


def _mpc(**kw) -> LtvMpc:
    cfg = make_config()
    p = ControllerParams.from_config(cfg, VehicleParameters()).mpc
    for k, v in kw.items():
        setattr(p, k, v)
    return LtvMpc(p)


def test_qp_consistent_reference_gives_zero_effort():
    m = build_master([Seg(1, 10.0, 1.0, curvature=lambda s: 0.05)])
    tr = window(m, 4.0, 0.05, 5.0)
    mpc = _mpc()
    ref = build_reference(tr, 4.0, 30, 0.1, float(np.interp(4.0, m.t, m.yaw)), 1)
    z0 = np.array([ref.x[0], ref.y[0], ref.yaw[0], ref.v[0], ref.delta[0], ref.a[0]])
    sol = mpc.solve(z0, ref, 1, 4.0)
    assert sol.ok
    e = np.hypot(sol.z[:, 0] - ref.x, sol.z[:, 1] - ref.y)
    assert e.max() < 0.02


def test_qp_respects_hard_and_soft_constraints():
    # far-off reference demands violent action: constraints must hold
    m = build_master([Seg(1, 10.0, 1.5, curvature=lambda s: 0.3)])
    tr = window(m, 3.0, 0.05, 5.0)
    mpc = _mpc()
    ref = build_reference(tr, 3.0, 30, 0.1, 0.0, 1)
    z0 = np.array([ref.x[0] - 1.0, ref.y[0] + 1.0, 0.5, 0.0, -0.3, 0.0])
    sol = mpc.solve(z0, ref, 1, 3.0)
    assert sol.ok
    p = mpc.p
    tol = 1e-3
    assert np.all(np.abs(sol.u[:, 0]) <= p.max_steer_rate + tol)
    assert np.all(np.abs(sol.u[:, 1]) <= p.max_jerk + tol)
    assert np.all(np.abs(sol.z[:, 4]) <= p.max_steer_angle + tol)
    assert np.all(sol.z[:, 5] <= p.max_acceleration + tol) and np.all(sol.z[:, 5] >= -p.max_deceleration - tol)
    assert np.all(sol.z[:, 3] >= -0.01)            # DRIVE: v >= 0 (soft, essentially exact)


def test_qp_reverse_keeps_velocity_nonpositive():
    m = build_master([Seg(-1, 6.0, 1.0)])
    tr = window(m, 3.0, 0.05, 5.0)
    mpc = _mpc()
    ref = build_reference(tr, 3.0, 30, 0.1, 0.0, -1)
    z0 = np.array([ref.x[0], ref.y[0], 0.0, ref.v[0], 0.0, ref.a[0]])
    sol = mpc.solve(z0, ref, -1, 3.0)
    assert sol.ok and np.all(sol.z[:, 3] <= 0.01)
    assert np.all(sol.z[:, 5] >= -1.5 - 1e-3) and np.all(sol.z[:, 5] <= 3.0 + 1e-3)


def test_warm_start_reduces_iterations_and_solve_time():
    m = build_master([Seg(1, 12.0, 1.0, curvature=lambda s: 0.08 * math.sin(s / 2))])
    mpc = _mpc()
    times, iters = [], []
    for k in range(200):
        t = 2.0 + 0.02 * k
        tr = window(m, math.floor(t * 10) / 10, 0.05, 5.0)
        yaw = float(np.interp(t, m.t, m.yaw))
        ref = build_reference(tr, t, 30, 0.1, yaw, 1)
        z0 = np.array([ref.x[0] + 0.02, ref.y[0] - 0.03, yaw, ref.v[0], ref.delta[0], ref.a[0]])
        sol = mpc.solve(z0, ref, 1, t)
        assert sol.ok
        times.append(sol.total_time)
        iters.append(sol.iterations)
    med = float(np.median(times))
    print(f"\nMPC total step time median {med * 1e3:.2f} ms, p95 {np.percentile(times, 95) * 1e3:.2f} ms, "
          f"iterations median {np.median(iters)}")
    assert med < 5e-3
    assert np.median(iters[1:]) < iters[0] or iters[0] <= 25


def test_reference_yaw_unwrap_and_cusp_clamp():
    m = build_master([Seg(1, 4.0, 1.0, ramp_time=2.0), Seg(dwell=2.0), Seg(-1, 4.0, 1.0, ramp_time=2.0)],
                     yaw0=math.pi - 0.01)
    tr = window(m, 0.0, 0.05, 20.0, validity=100.0)
    yaw_now = float(np.interp(7.0, m.t, m.yaw)) - 2 * math.pi   # same heading, other side of the cut
    ref = build_reference(tr, 7.0, 30, 0.1, yaw_now, 1)
    # unwrapped near the current yaw (which is on the other side of the +-pi cut)
    assert abs(ref.yaw[0] - yaw_now) < 0.05
    assert np.all(np.abs(np.diff(ref.yaw)) < 0.1)
    # reverse part ahead is clamped to the cusp: stationary, positions frozen
    assert ref.clamped.any()
    k = int(np.argmax(ref.clamped))
    assert np.all(ref.v[k:] == 0.0)
    assert np.allclose(ref.x[k:], ref.x[k])


def test_reference_undefined_beyond_moving_end():
    m = build_master([Seg(1, 20.0, 1.0)])
    tr = window(m, 0.0, 0.05, 4.0, validity=4.0)   # ends while moving
    ref = build_reference(tr, 2.0, 30, 0.1, 0.0, 1)
    assert ref.valid[:21].all() and not ref.valid[21:].any()
    assert trajectory_usable(tr, 3.9)
    assert not trajectory_usable(tr, 4.1)     # expired AND beyond moving end
    tr.valid_until = to_ns(100.0)
    assert not trajectory_usable(tr, 4.1)     # beyond a moving end is undefined
    tr = window(m, 0.0, 0.05, 4.0, validity=1.0)
    assert not trajectory_usable(tr, 1.01)    # expired


def test_gear_manager_lookahead_and_phases():
    m = build_master([Seg(1, 4.0, 1.0, ramp_time=2.0), Seg(dwell=2.0), Seg(-1, 4.0, 1.0, ramp_time=2.0)])
    tr = window(m, 0.0, 0.05, 30.0, validity=100.0)
    gm = GearManager(0.03)
    assert gm.desired_gear(tr, 0.5) == Gear.DRIVE          # start dwell: next moving point is DRIVE
    assert gm.desired_gear(tr, 3.0) == Gear.DRIVE
    t_dwell = 1.5 + 6.0 + 0.5                              # inside the cusp dwell (labelled DRIVE)
    assert gm.desired_gear(tr, t_dwell) == Gear.REVERSE    # look-ahead to the reverse motion
    assert gm.desired_gear(tr, m.duration + 1.0) == Gear.REVERSE
    d = gm.decide(tr, t_dwell, Gear.DRIVE, 0.2)
    assert d.phase == GearPhase.STOPPING and d.request == Gear.NONE
    d = gm.decide(tr, t_dwell, Gear.DRIVE, 0.01)
    assert d.phase == GearPhase.SHIFTING and d.request == Gear.REVERSE
    assert gm.decide(tr, t_dwell, Gear.REVERSE, 0.0).phase == GearPhase.ENGAGED
