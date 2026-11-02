"""LTV-MPC: linearise about a nominal trajectory, build the QP, solve (docs/mpc.md).

Nominal (linearisation) trajectory, step k = 0..N-1:
    u_bar_k = previous optimal input evaluated at the same absolute time
              (time-shifted, works for any controller period), or, without a
              previous solution, the reference rates
              ((delta_ref_{k+1}-delta_ref_k)/dt, (a_ref_{k+1}-a_ref_k)/dt);
    z_bar   = RK4 rollout of u_bar from the (delay-compensated) z_0.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from doraware.control.plugins.mpc.model import IA, IDELTA, IPSI, IV, IX, IY, NU, NZ, discretize, rollout
from doraware.control.plugins.mpc.params import MpcParams
from doraware.control.plugins.mpc.qp import MpcQp, QpResult
from doraware.control.plugins.mpc.reference import HorizonReference


@dataclass
class MpcSolution:
    ok: bool
    status: str
    iterations: int
    cost: float
    solve_time: float           # [s, wall] OSQP
    total_time: float           # [s, wall] linearisation + QP build + solve
    t0: float                   # [s, sim] time of z_0
    z: np.ndarray               # (N+1, NZ) predicted states (z_0 included)
    u: np.ndarray               # (N, NU)
    max_slack: float = 0.0


def accel_bounds(gear_dir: int, max_acc: float, max_dec: float) -> tuple:
    """Signed acceleration bounds: in REVERSE propulsion is negative and braking positive."""
    if gear_dir > 0:
        return -max_dec, max_acc
    if gear_dir < 0:
        return -max_acc, max_dec
    return -max_dec, max_dec


class LtvMpc:
    def __init__(self, p: MpcParams):
        self.p = p
        self.n = p.horizon_steps
        self.qp = MpcQp(self.n, p.max_iterations, p.eps_abs, p.eps_rel, p.time_limit)
        self.prev: Optional[MpcSolution] = None

    def reset(self) -> None:
        self.prev = None
        self.qp.reset()

    # ----------------------------------------------------------- nominal
    def _nominal_inputs(self, ref: HorizonReference, t0: float) -> np.ndarray:
        p, n = self.p, self.n
        if self.prev is not None and self.prev.ok:
            tm = t0 + (np.arange(n) + 0.5) * p.dt
            idx = np.floor((tm - self.prev.t0) / p.dt).astype(int)
            ub = np.zeros((n, NU))
            ok = (idx >= 0) & (idx < n)
            ub[ok] = self.prev.u[idx[ok]]
        else:
            ub = np.stack([np.diff(ref.delta), np.diff(ref.a)], axis=1) / p.dt
            ub[~(ref.valid[:-1] & ref.valid[1:])] = 0.0
        ub[:, 0] = np.clip(ub[:, 0], -p.max_steer_rate, p.max_steer_rate)
        ub[:, 1] = np.clip(ub[:, 1], -p.max_jerk, p.max_jerk)
        return ub

    # -------------------------------------------------------------- cost
    def _cost(self, ref: HorizonReference) -> tuple:
        n, w = self.n, self.p.weights
        scale = ref.valid[1:].astype(float)
        if scale.any():
            scale[np.nonzero(scale)[0][-1]] *= w.terminal_scale
        c, s = np.cos(ref.yaw[1:]), np.sin(ref.yaw[1:])
        # position: e_lon = c dx + s dy, e_lat = -s dx + c dy  ->  Q_pos = R diag(w_lon, w_lat) R'
        qxx = w.longitudinal * c * c + w.lateral * s * s
        qxy = (w.longitudinal - w.lateral) * c * s
        qyy = w.longitudinal * s * s + w.lateral * c * c
        diag = np.array([w.yaw, w.velocity, w.steering, w.acceleration])
        # J = sum (z-r)' Q (z-r)  ->  P = 2Q, q = -2 Q r
        pz = 2.0 * scale[:, None] * np.column_stack([qxx, qxy, qyy, np.tile(diag, (n, 1))])
        pu = np.tile(2.0 * np.array([w.steering_rate, w.jerk]), n)
        ps = np.full(n, 2.0 * w.velocity_slack)
        p_vals = np.concatenate([pz.ravel(), pu, ps])

        rx, ry = ref.x[1:], ref.y[1:]
        qz = np.zeros((n, NZ))
        qz[:, IX] = -(pz[:, 0] * rx + pz[:, 1] * ry)
        qz[:, IY] = -(pz[:, 1] * rx + pz[:, 2] * ry)
        qz[:, IPSI] = -pz[:, 3] * ref.yaw[1:]
        qz[:, IV] = -pz[:, 4] * ref.v[1:]
        qz[:, IDELTA] = -pz[:, 5] * ref.delta[1:]
        qz[:, IA] = -pz[:, 6] * ref.a[1:]
        q = np.concatenate([qz.ravel(), np.zeros(NU * n), np.full(n, w.velocity_slack)])
        return p_vals, q

    # -------------------------------------------------------------- solve
    def solve(self, z0: np.ndarray, ref: HorizonReference, gear_dir: int, t0: float,
              force_failure: bool = False) -> MpcSolution:
        tic = time.perf_counter()
        p, n = self.p, self.n
        ub = self._nominal_inputs(ref, t0)
        zb = rollout(z0, ub, p.dt, p.wheel_base)
        ad, bd, d = discretize(zb[:-1], ub, zb[1:], p.dt, p.wheel_base)
        p_vals, q = self._cost(ref)
        a_lo, a_hi = accel_bounds(gear_dir, p.max_acceleration, p.max_deceleration)
        v_lo = 0.0 if gear_dir > 0 else -p.max_speed
        v_hi = 0.0 if gear_dir < 0 else p.max_speed
        bounds = dict(u_lo=np.tile([-p.max_steer_rate, -p.max_jerk], (n, 1)),
                      u_hi=np.tile([p.max_steer_rate, p.max_jerk], (n, 1)),
                      steer=p.max_steer_angle, a_lo=np.full(n, a_lo), a_hi=np.full(n, a_hi),
                      v_lo=np.full(n, v_lo), v_hi=np.full(n, v_hi))
        x_warm = np.concatenate([zb[1:].ravel(), ub.ravel(), np.zeros(n)])
        shift = 0 if self.prev is None else int(round((t0 - self.prev.t0) / p.dt))
        if force_failure:
            res = QpResult(False, "forced_failure", 0, float("nan"), 0.0, zb[1:], ub, np.zeros(n))
            self.qp.reset()
        else:
            res = self.qp.solve(z0, ad, bd, d, p_vals, q, bounds, x_warm, shift)
        total = time.perf_counter() - tic
        z = np.vstack([z0, res.z]) if res.ok else zb
        sol = MpcSolution(res.ok, res.status, res.iterations, res.cost, res.solve_time, total, t0, z,
                          res.u if res.ok else ub,
                          float(res.slack.max()) if res.ok and len(res.slack) else 0.0)
        if sol.ok:
            self.prev = sol
        return sol

    def shifted_previous(self, t0: float) -> Optional[np.ndarray]:
        """Input u(t0) of the last successful plan (failure handling)."""
        if self.prev is None:
            return None
        k = int(np.floor((t0 - self.prev.t0) / self.p.dt + 1e-9))
        if k < 0 or k >= self.n:
            return None
        return self.prev.u[k]
