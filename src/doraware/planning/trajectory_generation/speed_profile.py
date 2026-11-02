"""Speed profile of one direction segment: a QP on b = v^2 over the s-grid.

Variables b_i = v_i^2 >= 0 at the grid nodes s_i = i h (uniform h). With
constant acceleration on each interval (v^2 is then linear in s)

    a_k = (b_{k+1} - b_k) / (2 h),     dt_k = 2 h / (v_k + v_{k+1})     (exact),

and inside an interval |v| is monotone, so a bound that holds at both nodes
holds on the whole interval. Constraints (|v| magnitudes, direction handled
by the caller):

* b_0 = b_N = 0 (stop at the segment ends: cusps and goal);
* b_i <= v_max^2 (forward / reverse speed limit);
* lateral acceleration v^2 |kappa| <= a_lat on both nodes of every interval
  (|kappa| is linear-in-delta between nodes, its max is at a node);
* steering rate |d delta / ds| |v| <= rate on both nodes of every interval;
* jerk-limited start / stop from rest (a(0) = 0, jerk <= j): the reachable
  speed and acceleration at distance s are v <= (6^(2/3)/2) j^(1/3) s^(2/3),
  a <= (6 s j^2)^(1/3) (s = distance from the start / to the end);
* -a_dec <= a_k <= a_acc;
* jerk: the acceleration reported along the trajectory is the piecewise
  linear interpolation of a_k at the interval mid-times (and 0 at both
  ends), so the jerk between intervals k-1 and k is
  |a_k - a_{k-1}| / ((dt_{k-1} + dt_k) / 2) and dt >= h / v. With an upper
  bound v_ref of v around node k the linear constraint
  |a_k - a_{k-1}| <= j h / v_ref  guarantees jerk <= j. v_ref is enforced
  as a cap (b <= v_ref^2): first the forward/backward-pass seed (bounds every
  feasible profile), then successive QPs refine it to GROWTH * the previous
  solution (less conservative), keeping the fastest result. For the first half
  interval (a: 0 -> a_0 within dt_0/2 = h / v_1, v_1 = sqrt(2 a_0 h)) the jerk
  is sqrt(2) a_0^(3/2) / sqrt(h) <= j  <=>  a_0 <= (j^2 h / 2)^(1/3)
  (symmetrically for the last interval).

Objective: maximise sum w_i b_i with w_i = b_ref,i^(-3/2) (linearisation of
the travel time sum h / sqrt(b) at the previous iterate) minus a small
quadratic penalty on the acceleration differences (smoothness). b = 0 is always
feasible, so the QP always has a solution. Solved with OSQP; the result is
clipped and passed through a final acceleration-limit pass (which can only
lower b) so the hard limits hold exactly despite the solver tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import osqp
import scipy.sparse as sp

from doraware.planning.trajectory_generation.limits import ProfileLimits

SMOOTHING_WEIGHT = 0.1       # weight of sum (a_k - a_{k-1})^2 relative to sum b_i / N
MAX_ITERATIONS = 8           # successive QPs refining the jerk linearisation speed v_ref
GROWTH = 1.3                 # v_ref of the next iteration = GROWTH * |v| of the current one
V_FLOOR = 0.1                # [m/s] floor of the speed used in the time weights
CONVERGENCE = 0.005          # stop when the duration improves by less than 0.5 %
RELAXED_EPS = 1e-5           # second attempt when the strict QP does not converge (long paths)
RELAXED_MAX_ITER = 200000
_C_JERK_V = 6.0 ** (2.0 / 3.0) / 2.0


@dataclass
class SpeedProfile:
    """|v| at the grid nodes, a_k and dt_k per interval, node times t."""
    s: np.ndarray
    v: np.ndarray
    a: np.ndarray
    dt: np.ndarray
    t: np.ndarray
    solver_status: str

    @property
    def duration(self) -> float:
        return float(self.t[-1])


def velocity_bounds(s: np.ndarray, kappa: np.ndarray, delta: np.ndarray, v_max: float,
                    lim: ProfileLimits) -> np.ndarray:
    """Upper bound of b = v^2 at each node (0 at both ends)."""
    h = s[1] - s[0]
    S = s[-1]
    bmax = np.full(len(s), v_max * v_max)
    k_int = np.maximum(np.abs(kappa[:-1]), np.abs(kappa[1:]))
    with np.errstate(divide="ignore"):
        b_lat = np.where(k_int > 1e-12, lim.max_lateral_acceleration / np.maximum(k_int, 1e-12), np.inf)
        slope = np.abs(np.diff(delta)) / h
        b_steer = np.where(slope > 1e-12, (lim.max_steer_rate / np.maximum(slope, 1e-12)) ** 2, np.inf)
    b_int = np.minimum(b_lat, b_steer)
    bmax[:-1] = np.minimum(bmax[:-1], b_int)
    bmax[1:] = np.minimum(bmax[1:], b_int)
    d = np.minimum(s, S - s)                       # distance from the nearer end (rest)
    bmax = np.minimum(bmax, (_C_JERK_V * lim.max_jerk ** (1.0 / 3.0) * d ** (2.0 / 3.0)) ** 2)
    bmax[0] = bmax[-1] = 0.0
    return bmax


def accel_bounds(s: np.ndarray, lim: ProfileLimits):
    """Per-interval (acc_k, dec_k) magnitudes incl. the jerk-limited start/stop."""
    h = s[1] - s[0]
    S = s[-1]
    j = lim.max_jerk
    acc = np.minimum(lim.max_acceleration, np.cbrt(6.0 * s[1:] * j * j))
    dec = np.minimum(lim.max_deceleration, np.cbrt(6.0 * (S - s[:-1]) * j * j))
    first = np.cbrt(j * j * h / 2.0)
    acc[0] = min(acc[0], first)
    dec[-1] = min(dec[-1], first)
    return acc, dec


def accel_pass(bmax: np.ndarray, acc: np.ndarray, dec: np.ndarray, h: float) -> np.ndarray:
    """Largest b <= bmax with b_{k+1} - b_k <= 2 h acc_k and b_k - b_{k+1} <= 2 h dec_k."""
    b = bmax.copy()
    for k in range(len(b) - 1):
        b[k + 1] = min(b[k + 1], b[k] + 2.0 * h * acc[k])
    for k in range(len(b) - 2, -1, -1):
        b[k] = min(b[k], b[k + 1] + 2.0 * h * dec[k])
    return b


def optimize_speed(s: np.ndarray, kappa: np.ndarray, delta: np.ndarray, v_max: float,
                   lim: ProfileLimits, iterations: int = MAX_ITERATIONS) -> SpeedProfile:
    """Successive QPs: iteration m uses v_ref = min(seed, GROWTH * v_{m-1}) in
    the jerk rows *and* caps b <= v_ref^2, so the jerk guarantee holds in every
    iteration; the fastest valid profile is kept."""
    n = len(s)
    h = float(s[1] - s[0])
    bmax = velocity_bounds(s, kappa, delta, v_max, lim)
    acc, dec = accel_bounds(s, lim)
    seed = accel_pass(bmax, acc, dec, h)
    D1 = sp.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n)) / (2.0 * h)   # a_k
    D2 = sp.diags([np.ones(n - 2), -2.0 * np.ones(n - 2), np.ones(n - 2)], [0, 1, 2],
                  shape=(n - 2, n)) / (2.0 * h)                                              # a_k - a_{k-1}
    A = sp.vstack([sp.identity(n, format="csc"), D1, D2], format="csc")
    P = sp.triu(sp.csc_matrix(SMOOTHING_WEIGHT * (D2.T @ D2)), format="csc")

    def bounds(cap: np.ndarray):
        v_cap = np.sqrt(cap)
        # jerk rows at interior nodes k = 1..n-2 (between intervals k-1 and k)
        v_ref = np.maximum(np.maximum(v_cap[:-2], v_cap[1:-1]), v_cap[2:])
        da = np.minimum(lim.max_jerk * h / np.maximum(v_ref, 1e-9), lim.max_acceleration + lim.max_deceleration)
        return np.concatenate([np.zeros(n), -dec, -da]), np.concatenate([cap, acc, da])

    def successive_qp(eps: float, max_iter: int):
        best_b, best_T, status = None, np.inf, "not solved"
        cap = seed.copy()
        try:
            solver = osqp.OSQP()
            lo, hi = bounds(cap)
            solver.setup(P=P, q=time_weights(seed), A=A, l=lo, u=hi, verbose=False, eps_abs=eps, eps_rel=eps,
                         max_iter=max_iter, polishing=True, warm_starting=True)
            for it in range(iterations):
                if it:
                    lo, hi = bounds(cap)
                    solver.update(q=time_weights(best_b if best_b is not None else seed), l=lo, u=hi)
                res = solver.solve(raise_error=False)
                status = str(res.info.status)
                if res.x is None or not np.all(np.isfinite(res.x)) or "solved" not in status:
                    break
                b = accel_pass(np.clip(np.asarray(res.x, dtype=float), 0.0, cap), acc, dec, h)
                b[0] = b[-1] = 0.0
                T = _duration(b, h)
                if T < best_T * (1.0 - CONVERGENCE):
                    best_b, best_T = b, T
                elif best_b is not None:
                    break
                cap = np.minimum(seed, GROWTH * GROWTH * b + 1e-4)
        except Exception as exc:  # pragma: no cover - solver failure -> next attempt / fallback
            status = f"osqp error: {exc!r}"
        return best_b, status

    # strict settings first (all parking paths solve here, results unchanged); long
    # paths (roads) can exceed the iteration budget, so retry with a looser tolerance
    best_b, status = successive_qp(1e-7, 20000)
    if best_b is None:
        best_b, status2 = successive_qp(RELAXED_EPS, RELAXED_MAX_ITER)
        status = f"{status}; relaxed: {status2}"
    b = best_b
    if b is None:
        # fallback (solver failure): a slow, scaled-down seed
        b = accel_pass(0.1 * seed, acc, dec, h)
        status += " (fallback)"
    b[0] = b[-1] = 0.0
    v = np.sqrt(b)
    if n > 2 and np.any(v[1:-1] <= 1e-6):
        raise RuntimeError("speed profile has an interior standstill")
    a = np.diff(b) / (2.0 * h)
    dt = 2.0 * h / (v[:-1] + v[1:])
    t = np.concatenate([[0.0], np.cumsum(dt)])
    return SpeedProfile(np.asarray(s, float), v, a, dt, t, status)


def time_weights(b_ref: np.ndarray) -> np.ndarray:
    """Linear objective q = -w approximating the travel time sum h / sqrt(b):
    d(b^-1/2)/db = -b^-3/2 / 2, linearised at b_ref (floored at V_FLOOR^2) and
    normalised to mean 1 / n. Slow nodes get large weights, so the solver
    never trades a standstill for a smoother profile."""
    w = np.maximum(b_ref, V_FLOOR * V_FLOOR) ** -1.5
    return -w / (w.sum())


def _duration(b: np.ndarray, h: float) -> float:
    v = np.sqrt(b)
    with np.errstate(divide="ignore"):
        return float(np.sum(2.0 * h / (v[:-1] + v[1:])))


def jerk_limit_check(sp_: SpeedProfile) -> float:
    """Max jerk of the mid-time-interpolated acceleration (diagnostics)."""
    tm = np.concatenate([[0.0], sp_.t[:-1] + 0.5 * sp_.dt, [sp_.t[-1]]])
    am = np.concatenate([[0.0], sp_.a, [0.0]])
    return float(np.max(np.abs(np.diff(am) / np.maximum(np.diff(tm), 1e-12))))


__all__ = ["SpeedProfile", "optimize_speed", "velocity_bounds", "accel_bounds", "jerk_limit_check"]
