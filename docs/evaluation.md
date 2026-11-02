# Evaluation

How the closed-loop system is evaluated, the latest reference results, and the known
limitations. Numbers come from executed runs with the default configuration
(`configs/default.yaml`: planner 10 Hz, trajectory 0.05 s × 5 s, controller 50 Hz,
physics 100 Hz, lockstep) and planner submodule commit `0444fe3`.
Reproduce them with the commands in [§5](#5-reproduction).

## 1. What is measured

Every planner test pattern is converted into a Godot scene and driven in closed loop
(planner → trajectory generator → MPC → Godot physics). The verdict is computed from the
**simulated vehicle state**, never from the plan. Scenario classes and acceptance
criteria are defined in [testing.md](testing.md).

## 2. Reference results (v0.1.0)

| Evaluation | Result |
|---|---|
| 107 fixed planner scenarios | 107 / 107 pass |
| 100 random scenarios, seed 42 | 100 / 100 pass (94 goal reached, 6 accepted rejections) |
| 100 random scenarios, seed 142 | 100 / 100 pass (89 goal reached, 11 accepted rejections) |
| pytest (148 unit, 23 Godot integration, 17 end-to-end) | 188 / 188 pass |
| closed-loop regression vs. baseline | passed |

Latest check (2026-11-03, after the plugin restructuring, the vehicle signals, the startup-race
fix and the lockstep liveness change; 8 parallel workers): pytest 171 unit, 28 integration and 18
end-to-end pass; regression passed; **207 / 207** scenarios pass (186 goal reached, median 7 mm /
0.19°, no collisions, no command timeouts, longest lockstep wait 0.07 s). An earlier run of the same
code before the liveness change failed N-038 once because of a > 10 s middleware stall (see §4).

By class (fixed scenarios + seed 42):

| Class | Runs | Pass | Outcome |
|---|---|---|---|
| feasible | 52 | 52 | goal reached |
| infeasible (C3, G1–G3) | 4 | 4 | rejected, vehicle still |
| invalid input (G4–G7) | 4 | 4 | rejected with the expected status |
| parameter / sweep (I, J, K, L, F-*, C4, D4, D5) | 47 | 47 | 40 goal reached, 7 rejected safely |
| randomized | 100 | 100 | 94 goal reached, 6 rejected (`no_path`, allowed by the planner spec) |

## 3. Closed-loop quality (186 goal-reaching runs)

| Metric | median | p95 | max | criterion |
|---|---|---|---|---|
| final position error vs. goal [m] | 0.007 | 0.028 | 0.082¹ | ≤ 0.10 |
| final yaw error vs. goal [deg] | 0.19 | 0.41 | 1.76 | ≤ 2.0 |
| terminal tracking error vs. plan end [m] | 0.007 | 0.026 | 0.050 | ≤ 0.10 |
| max lateral tracking error (rear axle) [m] | 0.008 | 0.011 | 0.015 | — |
| min actual clearance [m] | 0.155 | — | min 0.021² | > 0 |
| collisions / gear changes while moving | 0 | | 0 | 0 |
| OSQP solve time [ms] | | 0.47 | 0.69 | — |
| controller step, wall [ms] | | 1.9 | 2.1 | ≤ 50 |
| commanded jerk [m/s³] / steering rate [rad/s] | | | 3.0 / 0.60 | ≤ 3.0 / 0.6 |

¹ L-pos0.20 configures a 0.20 m planner goal tolerance. ² K-margin0.00 plans with zero safety margin.

## 4. Known limitations

* **Planner speed.** The Python planner needs up to ~60 s on hard scenes; D4, D5 and
  F-circles-E3 can be rejected by the wall-clock limit rather than the expansion budget
  (the vehicle stays still).
* **Physical jerk at standstill.** Commanded jerk respects the limit, but the simulated
  car's acceleration jumps when it comes to rest or the brake hold engages
  (finite-difference jerk up to ~25 m/s³). The brake model has no stop smoothing.
* **Safety margin on the physical car.** Actual clearance can be up to ~3 cm below the
  planned clearance (random scene N-091 ended 2 mm inside the planner's 0.1 m margin, without
  contact). A tracking-aware margin would be required to guarantee the planner's margin.
* **Goal tolerance allocation.** The planner is asked for 0.05 m / 1°; if that exhausts its
  budget it re-plans with its defaults (0.10 m / 2°), leaving little room for tracking error.
* **Determinism scope.** Physics, control and vehicle signals are bit-identical across runs, GUI/headless
  and simulator start-up timing;
  a recording may contain a few ticks after the verdict (asynchronous shutdown).
* **Dora 1.1 quirks** worked around in the nodes and the runner: SHM payload lifetime, ~120 s idle
  stop (heartbeat timer), occasional startup hang with many concurrent dataflows (watchdog
  with reported retries).
* **Message stalls under parallel load.** With 8 concurrent closed loops, one control command
  once arrived more than 10 s (wall) late (N-038 in the 2026-11-03 run). In lockstep, sim time is
  paused while a node waits, so such a delay cannot change results, but the former 10 s liveness
  timeout declared the controller dead and failed the run. The liveness timeout is now 30 s
  (`timing.lockstep_wall_timeout`), and every wait longer than `timing.lockstep_stall_warning`
  (2 s) is reported as a warning and as the metric `max_lockstep_wait`. A producer that really
  stops is still detected and fails the run (e2e fault-injection tests).
* The kinematic MPC model (no tire slip) is meant for parking speeds (≤ 1.5 m/s).

## 5. Reproduction

```bash
./scripts/setup.sh
./scripts/test_all.sh                                    # unit/integration + 107 fixed + 100 random
./scripts/test_regression.sh                             # regression set vs. tests/regression/baseline.json
python -m pytest tests/e2e -m e2e                        # system tests (≈ 4 min)
doraware run --suite random --random 100 --seed 142 --workers 8
```

Each suite writes `summary.json` / `summary.md` to its output directory under `results/runs/`.
When a change moves these numbers, update this page in the same pull request.
