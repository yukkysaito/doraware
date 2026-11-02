# MPC trajectory follower

Implementation: the `mpc` controller plugin, `src/doraware/control/plugins/mpc/`
(selected by `control.controller.type: mpc`; no middleware imports), plus two
controller-independent helpers in `src/doraware/control/`.

| Module | Content |
|---|---|
| `plugins/mpc/params.py` | `ControllerParams.from_config(cfg, vehicle)`: all tunables |
| `plugins/mpc/model.py` | kinematic bicycle, RK4, Jacobians, exact ZOH discretisation |
| `plugins/mpc/reference.py` | horizon reference from the time-based `Trajectory` |
| `plugins/mpc/qp.py` | sparse QP (fixed sparsity), OSQP wrapper, warm start |
| `plugins/mpc/controller.py` | `LtvMpc`: nominal trajectory, linearisation, cost, bounds |
| `plugins/mpc/follower.py` | `TrajectoryFollower`: modes, failure handling, output stage |
| `prediction.py` | command history, delay compensation |
| `gear.py` | `GearManager` |
| `registry.py` | controller plugins (`Controller` protocol) |
| `node.py` | Dora node (`controller` in the dataflow) |

Node API:

```python
follower = TrajectoryFollower(config: dict, vehicle_params: VehicleParameters)
follower.on_trajectory(traj: Trajectory) -> bool            # keeps the newest by header.stamp
cmd, dbg = follower.step(state: VehicleState, t_now_ns: int)  # -> (ControlCommand, ControlDebug)
```

`config` is the full system configuration (`configs/default.yaml` layout);
the controller reads `control.controller.mpc`, `timing` (`controller_hz`,
`command_validity`), `vehicle.actuator` (dead times / time constants for the
prediction) and `fault_injection.solver_failure_after`. The controller node
creates it through `control/registry.py`. Vehicle limits come
from `VehicleParameters` (`wheel_base`, `max_steer_angle`, `max_steer_rate`,
`max_acceleration`, `max_deceleration`, `max_jerk`).

## 1. Model

Rear-axle (base_link) kinematic bicycle with wheelbase $L$:

$$
z = \begin{bmatrix} x & y & \psi & v & \delta & a \end{bmatrix}^\top,\qquad
u = \begin{bmatrix} \dot\delta & j \end{bmatrix}^\top
$$

$$
\dot x = v\cos\psi,\quad \dot y = v\sin\psi,\quad \dot\psi = \frac{v\tan\delta}{L},\quad
\dot v = a,\quad \dot\delta = u_1,\quad \dot a = u_2 .
$$

$v$ is the signed longitudinal speed of the rear-axle point (negative in
reverse; the same equations hold in both directions), $\delta$ the
equivalent front-wheel angle, $a$ the signed longitudinal acceleration.

*Relation to the actuators.* $\delta$ and $a$ are the **commanded**
steering angle and acceleration, i.e. exactly what is sent in
`ControlCommand`. The steering rate $\dot\delta$ and the jerk $j$ are the
decision variables, so steering-rate and jerk limits are plain input bounds
and the steering/acceleration states are continuous by construction. The
actuators (dead time $T_c$, first-order lag $\tau_c$) are not states of the
MPC; they are handled by delay compensation (§7).

## 2. Linearisation (LTV)

Prediction step $\Delta t$ (`control.controller.mpc.dt`, default 0.1 s), horizon
$N$ (`horizon_steps`, default 30). Both are independent of the controller
period $T = 1/f_{ctrl}$.

A nominal input sequence $\bar u_0..\bar u_{N-1}$ is chosen:

* **previous solution, time-shifted**: $\bar u_k = u^{prev}_{i}$ with
  $i = \lfloor (t_0 + (k+\tfrac12)\Delta t - t_0^{prev})/\Delta t\rfloor$
  ($0$ beyond the old horizon). This is a shift by the elapsed sim time, so
  it is correct for any controller rate (20/50/100 Hz) and $\Delta t$;
* **otherwise** (start, after hold / gear change / fallback): the reference
  rates $\bar u_k = ((\delta^r_{k+1}-\delta^r_k)/\Delta t,\ (a^r_{k+1}-a^r_k)/\Delta t)$.

$\bar u$ is clipped to the input bounds and rolled out from the initial state
with RK4: $\bar z_0 = z_0$, $\bar z_{k+1} = F(\bar z_k, \bar u_k)$. The
continuous Jacobians at $(\bar z_k,\bar u_k)$ are

$$
A_c=\begin{bmatrix}
0&0&-v\sin\psi&\cos\psi&0&0\\
0&0&v\cos\psi&\sin\psi&0&0\\
0&0&0&\tan\delta/L&v/(L\cos^2\delta)&0\\
0&0&0&0&0&1\\
0&0&0&0&0&0\\
0&0&0&0&0&0\end{bmatrix},\qquad
B_c=\begin{bmatrix}0&0\\0&0\\0&0\\0&0\\1&0\\0&1\end{bmatrix}.
$$

## 3. Discretisation

*Nominal propagation*: classical RK4 over $\Delta t$ with zero-order-hold
input, $F(z,u)$.

*Jacobians*: exact zero-order-hold discretisation of the linearised system

$$
\begin{bmatrix}A_d & B_d\\ 0 & I\end{bmatrix} = \exp\!\left(M\Delta t\right),\qquad
M = \begin{bmatrix}A_c & B_c\\ 0 & 0\end{bmatrix}.
$$

$M$ is nilpotent: the dependency graph $j\to a\to v\to\psi\to x,y$ and
$\dot\delta\to\delta\to\psi$ is acyclic with longest path of 4 edges, hence
$M^5 = 0$ and the exponential is **exactly** the finite series
$\sum_{i=0}^{4} (M\Delta t)^i/i!$ (no truncation error, batched for all
$k$; verified against `scipy.linalg.expm` in the tests).

*Affine term*: $d_k = F(\bar z_k,\bar u_k) - A_{d,k}\bar z_k - B_{d,k}\bar u_k$,
so that the prediction model

$$ z_{k+1} = A_{d,k} z_k + B_{d,k} u_k + d_k $$

is exact (to RK4 accuracy) at the nominal trajectory and first-order accurate
around it.

## 4. Cost

Decision vector $X = [z_1..z_N,\ u_0..u_{N-1},\ s_1..s_N]$ ($s$: velocity
slacks), $z_0$ fixed. With the reference $r_k$ (§5) and reference heading
$\psi^r_k$, the position error is split in the reference frame:

$$
e^{lon}_k = \cos\psi^r_k\,\Delta x_k + \sin\psi^r_k\,\Delta y_k,\qquad
e^{lat}_k = -\sin\psi^r_k\,\Delta x_k + \cos\psi^r_k\,\Delta y_k .
$$

$$
J = \sum_{k=1}^{N} \sigma_k\Big[w_{lon}(e^{lon}_k)^2 + w_{lat}(e^{lat}_k)^2 + w_\psi(\psi_k-\psi^r_k)^2
+ w_v(v_k-v^r_k)^2 + w_\delta(\delta_k-\delta^r_k)^2 + w_a(a_k-a^r_k)^2\Big]
+ \sum_{k=0}^{N-1}\big[w_{\dot\delta}\dot\delta_k^2 + w_j j_k^2\big]
+ \sum_{k=1}^{N} w_s\,(s_k^2 + s_k)
$$

* $\sigma_k = 1$ for defined reference samples, $0$ for samples beyond a
  moving trajectory end (undefined, never extrapolated), and the **terminal
  cost** $\sigma_{k_T} = $ `terminal_scale` on the last defined sample $k_T$.
* The position term is $\Delta p^\top Q^{pos}_k\Delta p$ with
  $Q^{pos} = R(\psi^r)\,\mathrm{diag}(w_{lon},w_{lat})\,R(\psi^r)^\top$, i.e.
  $Q_{xx}=w_{lon}c^2+w_{lat}s^2$, $Q_{xy}=(w_{lon}-w_{lat})cs$,
  $Q_{yy}=w_{lon}s^2+w_{lat}c^2$.
* The slack penalty $w_s(s^2+s)$ is an exact (L1) penalty plus a quadratic
  regulariser: the soft bound is only violated if it cannot be met.
* QP form: $J = \tfrac12 X^\top P X + q^\top X + \text{const}$, $P = 2Q$,
  $q = -2Qr$ (upper triangle stored).

Weights (`control.controller.mpc.weights`): `longitudinal`, `lateral`, `yaw`,
`velocity`, `steering`, `acceleration`, `steering_rate`, `jerk`,
`terminal_scale`, `velocity_slack`.

## 5. Reference

Sample $k$ is `sampling.sample(traj, t_0 + kΔt)` with $t_0 = t_{now} + D$
($D$: prediction delay, §7) — the reference is a function of **sim time**,
never of the point index, so a new window of the same plan yields the same
reference (no command jump on republish; a late window is used from the
current time, not from its point 0).

* Yaw: the sequence is unwrapped and shifted by $2\pi n$ so that
  $|\psi^r_0 - \psi_0| \le \pi$ (continuous errors across $\pm\pi$).
* Beyond the last point the reference is held only if that point is
  stationary; otherwise the samples are undefined ($\sigma_k=0$).
* Cusp: the MPC runs in the vehicle's current gear $g$. From the first sample
  whose gear direction differs from $g$, position/heading are frozen at the
  previous sample (the cusp) with $v^r=a^r=0$, while $\delta^r$ keeps
  following the trajectory (steering is prepared during the stop).
* A reference velocity of the wrong sign for $g$ is clipped to 0.

A trajectory is **usable** at $t_{now}$ iff $t_{now} \le$ `valid_until` and
`sample(traj, t_now)` is defined. `on_trajectory` keeps the trajectory with
the newest `header.stamp`; older/equal stamps and malformed messages are
ignored.

## 6. Constraints

For $k = 0..N-1$ (inputs) and $k=1..N$ (states), $g\in\{+1 (DRIVE), -1 (REVERSE)\}$:

| Constraint | Form | Type |
|---|---|---|
| dynamics | $z_{k+1} = A_{d,k}z_k + B_{d,k}u_k + d_k$ | equality |
| steering rate | $\lvert\dot\delta_k\rvert \le \dot\delta_{max}$ | hard |
| jerk | $\lvert j_k\rvert \le j_{max}$ | hard |
| steering angle | $\lvert\delta_k\rvert \le \delta_{max}$ | hard |
| acceleration | $g=+1$: $-a_{dec}\le a_k\le a_{acc}$; $g=-1$: $-a_{acc}\le a_k\le a_{dec}$ (propulsion is negative in reverse, braking positive) | hard |
| velocity | $v_k + s_k \ge v_{lo}(g)$, $v_k - s_k \le v_{hi}(g)$, $s_k\ge 0$; $g=+1$: $[0, v_{max}]$, $g=-1$: $[-v_{max}, 0]$ | soft |

Hard constraints are always feasible: $\delta_0, a_0$ are clipped into their
bounds and $u = 0$ keeps them there; only velocity can conflict, and it is
soft. Sparsity of $P$ and the constraint matrix is fixed at construction
(dense $6\times6$ / $6\times2$ dynamics blocks), so each step calls OSQP
`update(Px, Ax, q, l, u)` instead of a new setup.

*Warm start*: primal $= [\bar z_{1..N}, \bar u, 0]$ (the shifted previous
solution re-rolled from the new $z_0$, which satisfies the equalities), dual
$=$ previous dual shifted by $\mathrm{round}((t_0-t_0^{prev})/\Delta t)$ steps per
row group (last block repeated).

## 7. State prediction / delay compensation

Per channel $c\in\{\delta, a\}$ the actuator is approximated by a pure delay
$D_c = T_c + \lambda\,\tau_c$ (dead time + first-order time constant;
$e^{-\tau s}\approx 1/(1+\tau s)$ at low frequency, $\lambda$ =
`lag_compensation`). The command issued at $t_{now}$ acts from
$t_0 = t_{now} + D$, $D = \max(D_\delta, D_a)$. The measured
$(x,y,\psi,v)$ is propagated over $[t_{now}, t_0]$ (rear-axle bicycle,
10 ms midpoint sub-steps) with the already-issued commands
$c(s - D_c)$ from the command history (the last command for instants not
yet covered); braking never reverses the motion in the propagation. Then

$$ z_0 = [x(t_0), y(t_0), \psi(t_0), v(t_0), \delta^{cmd}_{prev}, a^{cmd}_{prev}] $$

with $\delta, a$ clipped into their bounds, and $a_0 := 0$ if the car is at
standstill while braking (a brake at rest produces no motion). Delay
parameters come from the `vehicle.actuator` section; `delay_compensation: false`
sets $D=0$.

*Command*: $u_0$ is constant on $[t_0, t_0+\Delta t)$, so the command for the
next controller period is $\delta^{cmd} = \delta_0 + \dot\delta_0 T$,
$a^{cmd} = a_0 + j_0 T$.

## 8. Gear handling and standstill

`GearManager.desired_gear(traj, t)`: the gear of `sample(traj, t)` if the
reference moves at $t$, else the gear of the next moving trajectory point
(so the shift happens during the stop at a cusp), else the current reference
gear. If it differs from the reported vehicle gear:

1. **stopping** ($|v| \ge$ `gear_change_speed`): the MPC keeps running in the
   old gear on the cusp-clamped reference, which brings the car to rest at
   the cusp (in PARK/NEUTRAL: plain brake).
2. **shifting** ($|v| <$ `gear_change_speed`): `gear_request` = desired gear;
   acceleration target $-\mathrm{sign}(v)\,a_{hold}$ while $|v| > 5$ mm/s, then
   $0$ (jerk-limited), so the command sign convention can flip with the gear
   without a propulsion pulse; steering tracks $\delta^r(t_{now})$ rate-limited.
3. MPC tracking resumes when the vehicle **reports** the new gear (the old
   plan is discarded).

**Standstill hold**: if the reference is stationary over
$[t_{now}, t_{now} + $`hold_release_lookahead`$]$, $|v| <$ `stop_speed` and
$|e^{lon}| <$ `stop_position_tolerance` (hysteresis: leave above twice the
tolerance), the command is $a = -g\,a_{hold}$ (`stop_hold_acceleration`,
brake in the gear-consistent sign) with the steering on the reference: no
solver in the loop, hence no chattering. A larger longitudinal error keeps
the MPC active, which creeps to the stop point. When the reference is about
to depart (`hold_release` mode) the brake is released (jerk-limited) so the
MPC starts from $a\approx 0$.

## 9. Failure handling and fallback

* OSQP `solved` / `solved inaccurate` with finite solution = success.
  Anything else (`max iter`, `time limit`, infeasible, or the fault
  injection `fault_injection.solver_failure_after` [s, sim]) is a failure.
* Failure $n \le$ `max_consecutive_failures`: the previous plan is reused,
  input at the current time $u^{prev}(t_0)$ (`track_reuse`).
* Further failures, no usable trajectory (expired, beyond a moving end,
  none received): **safe fallback** — brake at `fallback_deceleration`
  against the motion ($-\mathrm{sign}(v)\,a_{fb}$), then hold with
  $-g\,a_{hold}$ at rest; steering held; jerk-limited like every command.
  The controller returns to tracking as soon as a usable trajectory arrives /
  the solver succeeds.

**Output stage** (every mode): $|\delta^{cmd}-\delta^{cmd}_{prev}| \le \dot\delta_{max}T$,
$|\delta^{cmd}|\le\delta_{max}$, $|a^{cmd}-a^{cmd}_{prev}|\le j_{max}T$,
$a^{cmd}\in[a_{lo}(g), a_{hi}(g)]$; `valid_until = stamp + command_validity`.

## 10. Diagnostics (`ControlDebug`)

`mode` (`track`, `track_reuse`, `gear_change`, `hold`, `hold_release`,
`fallback`), `solver_status` (OSQP status string, `forced_failure`,
`no_valid_trajectory`), `solve_time` (OSQP update + solve, wall),
`iterations`, `cost` (QP objective without constant), `reference_time`
($t_0$), errors w.r.t. the reference at $t_{now}$ (longitudinal, lateral
+left, heading, velocity), predicted $x, y, \psi, v$ over the horizon
($z_0..z_N$).

## 11. Tuning parameters

| Key | Default | Meaning |
|---|---|---|
| `control.controller.mpc.horizon_steps`, `dt` | 30, 0.1 s | prediction horizon (3 s) |
| `control.controller.mpc.weights.*` | see §4 | cost weights |
| `control.controller.mpc.max_iterations`, `eps_abs`, `eps_rel`, `time_limit` | 4000, 1e-4, 1e-4, 0.02 s | OSQP |
| `control.controller.mpc.max_speed` | 3.0 m/s | soft speed magnitude bound |
| `control.controller.mpc.gear_change_speed` | 0.03 m/s | gear request threshold |
| `control.controller.mpc.stop_hold_acceleration` | 0.8 m/s² | brake at standstill / shifting |
| `control.controller.mpc.fallback_deceleration` | 2.0 m/s² | safe stop |
| `control.controller.mpc.max_consecutive_failures` | 3 | solution reuse budget |
| `control.controller.mpc.delay_compensation`, `lag_compensation` | true, true | §7 |
| `control.controller.mpc.stop_position_tolerance` | 0.03 m | hold entry (exit at 2×) |
| `control.controller.mpc.stop_speed` | 0.05 m/s | hold entry speed |
| `control.controller.mpc.hold_release_lookahead` | 0.3 s | brake release before departure |

## 12. Validation (unit tests, Python plant)

`tests/unit/control/test_controller_sim.py` (plant: kinematic bicycle + dead time,
first-order lag, rate/jerk limits, gear semantics of the simulator),
`test_mpc_core.py` (discretisation vs `expm`, linearisation accuracy, QP
constraints, reverse bounds, warm start / timing, reference unwrap / cusp
clamp / undefined end, gear manager), `test_controller_tracking.py`
(straight ±, arcs ±κ forward and reverse, S-curve, cusp with dwell, yaw wrap;
trajectory time step 0.02/0.05/0.1 s, window 2/5/10 s, controller
20/50/100 Hz, actuator delay compensation, stop-point creep),
`test_controller_faults.py` (solver failure, expiry, moving end, no
trajectory, republish without jumps, late windows, stale messages).

## 13. Known limitations

* Kinematic model (no tire slip): adequate at parking speeds only.
* The actuator lag is compensated as an equivalent delay, not modelled as a
  state; the first-order phase lag remains at higher frequencies.
* One linearisation per step (no SQP iterations).
* If the car comes to rest short of a cusp (speed below `gear_change_speed`),
  the shift is requested immediately; the remaining distance is not driven
  in the old gear (the next leg starts with that longitudinal offset).
* The command-level constraints are guaranteed by the output stage; the
  plant's own actuator limits (e.g. `vehicle.actuator.jerk_limit`) are not modelled
  in the MPC beyond the vehicle limits.
