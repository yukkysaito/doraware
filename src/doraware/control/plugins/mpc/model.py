"""Kinematic bicycle model about the rear axle (docs/mpc.md §1-§3).

State   z = [x, y, psi, v, delta, a]   (rear-axle pose, signed speed, front-wheel angle, accel)
Input   u = [delta_dot, jerk]

    x'     = v cos(psi)
    y'     = v sin(psi)
    psi'   = v tan(delta) / L
    v'     = a
    delta' = u0
    a'     = u1

Nominal propagation: classical RK4 with zero-order-hold input.
Linearisation: A_c = df/dz, B_c = df/du at the nominal (z_k, u_k).
Discretisation: exact ZOH of the linearised system,
    [A_d B_d; 0 I] = expm([A_c B_c; 0 0] dt).
The augmented matrix M is nilpotent (dependency chain j -> a -> v -> psi -> x
has 4 edges, hence M^5 = 0), so the exponential is *exactly* the finite sum
sum_{i=0..4} (M dt)^i / i!.  The affine term d_k = F(z_k, u_k) - A_d z_k - B_d u_k
makes the discrete model exact at the nominal point.
"""
from __future__ import annotations

import math

import numpy as np

NZ = 6
NU = 2
IX, IY, IPSI, IV, IDELTA, IA = range(NZ)


def rk4_step(z: np.ndarray, u: np.ndarray, dt: float, wheel_base: float) -> np.ndarray:
    """One RK4 step of the continuous model (scalar code: fast for 6 states)."""
    inv_l = 1.0 / wheel_base

    def f(s0, s1, s2, s3, s4, s5):
        return (s3 * math.cos(s2), s3 * math.sin(s2), s3 * math.tan(s4) * inv_l, s5, u[0], u[1])

    z0 = z.tolist()
    k1 = f(*z0)
    k2 = f(*[z0[i] + 0.5 * dt * k1[i] for i in range(NZ)])
    k3 = f(*[z0[i] + 0.5 * dt * k2[i] for i in range(NZ)])
    k4 = f(*[z0[i] + dt * k3[i] for i in range(NZ)])
    return np.array([z0[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) for i in range(NZ)])


def rollout(z0: np.ndarray, us: np.ndarray, dt: float, wheel_base: float) -> np.ndarray:
    """States z_0..z_N for inputs u_0..u_{N-1} (shape (N+1, NZ))."""
    zs = np.empty((len(us) + 1, NZ))
    zs[0] = z0
    for k in range(len(us)):
        zs[k + 1] = rk4_step(zs[k], us[k], dt, wheel_base)
    return zs


def discretize(zbar: np.ndarray, ubar: np.ndarray, znext: np.ndarray, dt: float,
               wheel_base: float) -> tuple:
    """Batched exact-ZOH discretisation of the Jacobians at N nominal points.

    zbar: (N, NZ) nominal states z_k, ubar: (N, NU), znext: (N, NZ) = RK4(z_k, u_k).
    Returns A_d (N, NZ, NZ), B_d (N, NZ, NU), d (N, NZ) with
        z_{k+1} = A_d z_k + B_d u_k + d_k.
    """
    n = len(zbar)
    psi, v, delta = zbar[:, IPSI], zbar[:, IV], zbar[:, IDELTA]
    c, s = np.cos(psi), np.sin(psi)
    cd = np.cos(delta)
    m = np.zeros((n, NZ + NU, NZ + NU))
    m[:, IX, IPSI] = -v * s
    m[:, IX, IV] = c
    m[:, IY, IPSI] = v * c
    m[:, IY, IV] = s
    m[:, IPSI, IV] = np.tan(delta) / wheel_base
    m[:, IPSI, IDELTA] = v / (wheel_base * cd * cd)
    m[:, IV, IA] = 1.0
    m[:, IDELTA, NZ + 0] = 1.0
    m[:, IA, NZ + 1] = 1.0
    m *= dt
    # expm(M) = I + M + M^2/2 + M^3/6 + M^4/24   (M^5 = 0, exact)
    e = np.broadcast_to(np.eye(NZ + NU), m.shape).copy()
    term = e.copy()
    for i in range(1, 5):
        term = term @ m / i
        e += term
    ad = e[:, :NZ, :NZ]
    bd = e[:, :NZ, NZ:]
    d = znext - np.einsum("kij,kj->ki", ad, zbar) - np.einsum("kij,kj->ki", bd, ubar)
    return np.ascontiguousarray(ad), np.ascontiguousarray(bd), d
