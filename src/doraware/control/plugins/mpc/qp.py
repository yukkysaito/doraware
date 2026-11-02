"""Sparse QP of the LTV-MPC and its OSQP solver wrapper (docs/mpc.md §4-§6).

Decision vector  X = [z_1 .. z_N | u_0 .. u_{N-1} | s_1 .. s_N]
(states after the fixed initial state z_0, inputs, velocity slacks).

    min  1/2 X' P X + q' X      s.t.  l <= C X <= u

Row groups of C (each has one block per prediction step):
    dyn   z_{k+1} - A_k z_k - B_k u_k = d_k      (k = 0 uses A_0 z_0 on the rhs)
    inp   |delta_dot| <= rate_max, |jerk| <= jerk_max
    steer |delta_k| <= delta_max
    acc   a_lo(gear) <= a_k <= a_hi(gear)
    vlo   v_k + s_k >= v_lo(gear)          (soft lower speed bound)
    vhi   v_k - s_k <= v_hi(gear)          (soft upper speed bound)
    slk   s_k >= 0

The sparsity pattern of P and C is fixed at construction, so every control
step only updates the numerical values (OSQP ``update(Px=, Ax=, ...)``) and
warm-starts from the shifted previous primal/dual solution.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import osqp
import scipy.sparse as sp

from doraware.control.plugins.mpc.model import IA, IDELTA, IPSI, IV, IX, IY, NU, NZ

INF = 1e30


@dataclass
class QpResult:
    ok: bool
    status: str
    iterations: int
    cost: float
    solve_time: float           # [s, wall] OSQP update + solve
    z: np.ndarray               # (N, NZ) predicted z_1..z_N
    u: np.ndarray               # (N, NU)
    slack: np.ndarray           # (N,)


class _Pattern:
    """Fixed COO entry list -> CSC with a permutation for fast value updates."""

    def __init__(self, rows: np.ndarray, cols: np.ndarray, shape: tuple):
        tag = np.arange(1, len(rows) + 1, dtype=float)
        m = sp.csc_matrix((tag, (rows, cols)), shape=shape)
        m.sort_indices()
        self.perm = (m.data - 1).astype(np.int64)
        self.matrix = m

    def csc(self, values: np.ndarray) -> sp.csc_matrix:
        m = self.matrix.copy()
        m.data = values[self.perm]
        return m

    def data(self, values: np.ndarray) -> np.ndarray:
        return values[self.perm]


class MpcQp:
    def __init__(self, n: int, max_iter: int = 4000, eps_abs: float = 1e-4, eps_rel: float = 1e-4,
                 time_limit: float = 0.02):
        self.n = n
        self.settings = dict(verbose=False, warm_starting=True, polishing=False, max_iter=max_iter,
                             eps_abs=eps_abs, eps_rel=eps_rel, time_limit=time_limit,
                             check_termination=10, adaptive_rho=True)
        self.iz = lambda k: NZ * (k - 1)                  # column of z_k, k = 1..N
        self.iu = lambda k: NZ * n + NU * k               # column of u_k, k = 0..N-1
        self.isl = lambda k: (NZ + NU) * n + (k - 1)      # column of s_k, k = 1..N
        self.nx = (NZ + NU + 1) * n
        self._build_patterns()
        self.solver: Optional[osqp.OSQP] = None
        self.last_x: Optional[np.ndarray] = None
        self.last_y: Optional[np.ndarray] = None

    # ----------------------------------------------------------- structure
    def _build_patterns(self) -> None:
        n = self.n
        # ---- P (upper triangle): per state block (x,x)(x,y)(y,y) + diagonals
        pr, pc = [], []
        for k in range(1, n + 1):
            b = self.iz(k)
            for (i, j) in [(IX, IX), (IX, IY), (IY, IY), (IPSI, IPSI), (IV, IV), (IDELTA, IDELTA), (IA, IA)]:
                pr.append(b + i)
                pc.append(b + j)
        for k in range(n):
            for i in range(NU):
                pr.append(self.iu(k) + i)
                pc.append(self.iu(k) + i)
        for k in range(1, n + 1):
            pr.append(self.isl(k))
            pc.append(self.isl(k))
        self.p_pat = _Pattern(np.array(pr), np.array(pc), (self.nx, self.nx))

        # ---- C
        rows, cols = [], []
        r = 0
        self.groups = {}
        # dynamics: identity on z_{k+1}, -A_k (dense) on z_k (k>=1), -B_k (dense) on u_k
        self.groups["dyn"] = (r, NZ)
        for k in range(n):
            for i in range(NZ):
                rows.append(r + i)
                cols.append(self.iz(k + 1) + i)
            if k >= 1:
                for i in range(NZ):
                    for j in range(NZ):
                        rows.append(r + i)
                        cols.append(self.iz(k) + j)
            for i in range(NZ):
                for j in range(NU):
                    rows.append(r + i)
                    cols.append(self.iu(k) + j)
            r += NZ
        self.n_dyn_entries = len(rows)
        self.groups["inp"] = (r, NU)
        for k in range(n):
            for i in range(NU):
                rows.append(r)
                cols.append(self.iu(k) + i)
                r += 1
        for name, idx in (("steer", IDELTA), ("acc", IA)):
            self.groups[name] = (r, 1)
            for k in range(1, n + 1):
                rows.append(r)
                cols.append(self.iz(k) + idx)
                r += 1
        for name in ("vlo", "vhi"):
            self.groups[name] = (r, 1)
            for k in range(1, n + 1):
                rows += [r, r]
                cols += [self.iz(k) + IV, self.isl(k)]
                r += 1
        self.groups["slk"] = (r, 1)
        for k in range(1, n + 1):
            rows.append(r)
            cols.append(self.isl(k))
            r += 1
        self.nc = r
        self.c_pat = _Pattern(np.array(rows), np.array(cols), (self.nc, self.nx))
        # constant part of the C values (everything except -A_k, -B_k)
        n_static = len(rows) - self.n_dyn_entries
        static = np.ones(n_static)
        # vhi rows: v - s  -> the slack coefficient is -1 (every 2nd entry of the vhi group)
        off = NU * n + 2 * n            # inp + steer + acc entries
        vhi_start = off + 2 * n         # after vlo entries
        static[vhi_start + 1: vhi_start + 2 * n: 2] = -1.0
        self.c_static = static

    # -------------------------------------------------------------- values
    def _c_values(self, ad: np.ndarray, bd: np.ndarray) -> np.ndarray:
        n = self.n
        dyn = []
        eye = np.ones(NZ)
        for k in range(n):
            dyn.append(eye)
            if k >= 1:
                dyn.append(-ad[k].ravel())
            dyn.append(-bd[k].ravel())
        return np.concatenate(dyn + [self.c_static])

    def solve(self, z0: np.ndarray, ad: np.ndarray, bd: np.ndarray, d: np.ndarray,
              p_vals: np.ndarray, q: np.ndarray, bounds: dict,
              x_warm: Optional[np.ndarray], shift: int) -> QpResult:
        """``bounds``: dict with arrays u_lo/u_hi (N,NU), steer (scalar), a_lo/a_hi,
        v_lo/v_hi (N,) for k = 1..N."""
        n = self.n
        l = np.empty(self.nc)
        u = np.empty(self.nc)
        rhs = d.copy()
        rhs[0] += ad[0] @ z0
        r0, _ = self.groups["dyn"]
        l[r0:r0 + NZ * n] = rhs.ravel()
        u[r0:r0 + NZ * n] = rhs.ravel()
        r0, _ = self.groups["inp"]
        l[r0:r0 + NU * n] = bounds["u_lo"].ravel()
        u[r0:r0 + NU * n] = bounds["u_hi"].ravel()
        r0, _ = self.groups["steer"]
        l[r0:r0 + n] = -bounds["steer"]
        u[r0:r0 + n] = bounds["steer"]
        r0, _ = self.groups["acc"]
        l[r0:r0 + n] = bounds["a_lo"]
        u[r0:r0 + n] = bounds["a_hi"]
        r0, _ = self.groups["vlo"]
        l[r0:r0 + n] = bounds["v_lo"]
        u[r0:r0 + n] = INF
        r0, _ = self.groups["vhi"]
        l[r0:r0 + n] = -INF
        u[r0:r0 + n] = bounds["v_hi"]
        r0, _ = self.groups["slk"]
        l[r0:r0 + n] = 0.0
        u[r0:r0 + n] = INF

        c_vals = self._c_values(ad, bd)
        t0 = time.perf_counter()
        if self.solver is None:
            self.solver = osqp.OSQP()
            self.solver.setup(self.p_pat.csc(p_vals), q, self.c_pat.csc(c_vals), l, u, **self.settings)
        else:
            self.solver.update(q=q, l=l, u=u, Px=self.p_pat.data(p_vals), Ax=self.c_pat.data(c_vals))
        y_warm = self._shift_dual(shift)
        if x_warm is not None:
            self.solver.warm_start(x=x_warm, y=y_warm if y_warm is not None else np.zeros(self.nc))
        res = self.solver.solve(raise_error=False)
        solve_time = time.perf_counter() - t0
        st = res.info.status_val
        ok = st in (osqp.SolverStatus.OSQP_SOLVED, osqp.SolverStatus.OSQP_SOLVED_INACCURATE) \
            and res.x is not None and np.all(np.isfinite(res.x))
        if ok:
            self.last_x, self.last_y = res.x.copy(), res.y.copy()
            xs = res.x
        else:
            self.last_x = self.last_y = None
            xs = np.zeros(self.nx)
        return QpResult(ok, str(res.info.status), int(res.info.iter),
                        float(res.info.obj_val) if ok else float("nan"), solve_time,
                        xs[:NZ * n].reshape(n, NZ).copy(),
                        xs[NZ * n:(NZ + NU) * n].reshape(n, NU).copy(),
                        xs[(NZ + NU) * n:].copy())

    def _shift_dual(self, shift: int) -> Optional[np.ndarray]:
        """Previous dual shifted by ``shift`` prediction steps (last block repeated)."""
        if self.last_y is None:
            return None
        if shift <= 0:
            return self.last_y
        y = self.last_y.copy()
        n = self.n
        for r0, w in self.groups.values():
            blk = self.last_y[r0:r0 + w * n].reshape(n, w)
            s = min(shift, n - 1)
            nb = np.vstack([blk[s:], np.repeat(blk[-1:], s, axis=0)])
            y[r0:r0 + w * n] = nb.ravel()
        return y

    def reset(self) -> None:
        self.last_x = self.last_y = None
