# Testing

## Levels

| Level | Location | What | Runs |
|---|---|---|---|
| Unit | `tests/unit/<package>/` (mirrors `src/doraware/`; shared fixtures in `tests/helpers/`) | interfaces/codec, clock & coordinator, plugin registries (built-in and entry points), scenario conversion (grid ⇄ boxes exact), frame transforms, planner adapter, trajectory generator limits, signal decision (both policies), MPC/controller on a Python plant, judge/metrics | `pytest tests/unit` |
| Integration | `tests/integration` (`-m godot`) | Godot vehicle through `GodotSimulator` (no Dora): directions/signs, rear-axle radius, actuator responses, gear logic, signal latching / flasher / hazard priority / no effect on motion, scripted actors (same pose as the Python model at every step, contacts, determinism), collisions, determinism, GUI = headless (all cameras rendered), no Godot script errors | `pytest tests/integration` |
| End-to-end closed loop | `tests/e2e` (`-m e2e`) + runner | full Dora dataflow + Godot: B2, timing variants, fault injection, determinism (also with a late simulator start), vehicle signals, replay | `pytest tests/e2e -m e2e`, `doraware run` |
| Scenario suites | runner | all 107 fixed scenarios + 100 seeded random scenarios | `scripts/test_all.sh` |
| Regression | runner + baseline | regression set vs. `tests/regression/baseline.json` | `scripts/test_regression.sh` |
| Planner (upstream) | submodule | `pytest tests`, `run_tests.py --regression`, `--all --random 100` | as documented upstream |

All Python test commands should run from the repository's virtual environment
(`source .venv/bin/activate`). Results of e2e tests are kept under `results/runs/e2e/` for inspection.

## Scenario classification (from the original `Expectation`)

| Class | Rule | Closed-loop pass condition |
|---|---|---|
| feasible | `success=True` | plan found, original expectation holds (gear changes, directions, planning time), vehicle settles at the goal |
| infeasible | `success=False` (G1–G3, C3) | planner rejects (status matches if the original names one), vehicle stays still (< 1 cm) for 3 s, no contact |
| invalid_input | `expect_exception` (G6) or `expected_status ∈ {start,goal}_in_collision` (G4, G5, G7) | rejection with the expected status, vehicle stays still |
| parameter | no expectation (I/J/K/L sweeps, F-* model comparisons, C4/D4/D5) | if planned: goal reached safely; if rejected: vehicle stays still |
| randomized | `N-000…N-099`, seed 42 | if planned: goal reached; if rejected: only when the product spec allows a rejection |

Properties of the planner (symmetry M-*, collision-model F-*, parameter
sweeps) are additionally covered by the upstream unit tests (301 tests).

## Acceptance criteria (closed loop, measured on the simulator state)

* collision count 0 (physics contacts **and** exact footprint-vs-map overlap), no out-of-bounds;
* goal position error ≤ 0.10 m, yaw error ≤ 2.0°, final |v| ≤ 0.02 m/s after a 1 s standstill;
  scenarios that configure a looser planner goal tolerance (L-pos0.20, L-yaw3/5) are judged against that
  tolerance for the goal, but the terminal error w.r.t. the planned path end must still be ≤ 0.10 m / 2°
  (the controller cannot be more accurate than the plan it is given);
* controller frequency ≥ 20 Hz (sim time), solver statistics reported;
* commanded steering/acceleration/jerk within the configured limits (saturation), reported.

Geometric safety margin of the plan (planner guarantee, 0.1 m) and the actual minimum clearance of the
simulated vehicle are reported separately (`min_clearance_planned`, `min_clearance_actual`).

## System tests (`tests/e2e/test_closed_loop.py`)

1. planner 10 Hz / controller 50 Hz / physics 100 Hz (default config) — B2 e2e.
2. trajectory time step 0.02 / 0.05 / 0.1 s — e2e.
3. trajectory horizon 2 / 5 / 10 s — e2e.
4. trajectory update stability: windows republished at 10 Hz, command continuity (max jerk, steering rate) — e2e metrics + unit test.
5. reverse tracking — B2 (reverse into slot), unit tests.
6. stop at gear change — e2e check `gear_changes_while_moving == 0`, unit tests.
7. solver failure → safe stop — `fault_injection.solver_failure_after`.
8. planner timeout (silent trajectory generator) → safe stop — `fault_injection.planner_silence_after`.
9. reproducibility — two identical runs give identical state logs, also when the simulator starts 5 s
   late (`fault_injection.simulator_start_delay`, the plan is then ready before the first state).
10. GUI vs headless — identical physics results (integration test with display; e2e when available).
11. vehicle signals — default policy: hazard lights for the whole parking manoeuvre and off once parked;
    `turn_indicators` policy: both turns of B2 indicated, no hazard; rejection (G1) and silent planning →
    hazard lights; signal states are deterministic and reproduced by command re-simulation.
12. trajectory planner dataflow — `dataflows/trajectory_planner.yml` with the `path_pipeline`
    plugin gives the same closed loop as the default dataflow, bit for bit.
13. road scenarios — `configs/road.yaml` (lane_follow): R1–R4 reach the goal; in R3 the vehicle
    stands before the crossing pedestrian.

## Updating the regression baseline

`tests/regression/baseline.json` stores pass/fail and the goal/tracking errors of the
regression scenario set (`REGRESSION_IDS` in `src/doraware/evaluation/suite/regression.py`).
A run fails if a passing scenario now fails or if an error grows beyond the tolerance.
After an intentional, reviewed behaviour change:

```bash
doraware regression --update-baseline
```

and commit the new baseline together with the change, stating why the numbers moved.
Never loosen the acceptance criteria or the tolerances to make a run pass.
