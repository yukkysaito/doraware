# Maintainer notes (for AI-assisted maintenance)

Read [CONTRIBUTING.md](CONTRIBUTING.md) first. Its ground rules apply here too: never hide failures,
loosen thresholds, or write ideal states into the simulator.

## Environment

* Use the project venv: `source .venv/bin/activate`. Run Python with `env -u PYTHONPATH`, because the host
  may export a ROS `PYTHONPATH` that breaks pytest.
* `tools/godot` is a symlink to the Godot 4.7.2-stable binary (created by `scripts/setup.sh`).
* Run outputs go to `results/` (git-ignored). Put scratch files outside the repository.

## Verification ladder (cheapest first)

1. `ruff check .` and `python -m pytest tests/unit -q` (~30 s)
2. `python -m pytest tests/integration -q` (Godot, ~1 min)
3. `doraware run --scenario B2` (single closed loop)
4. `python -m pytest tests/e2e -m e2e` (~4 min) and `./scripts/test_regression.sh`
5. `./scripts/test_all.sh` (all 107 fixed + 100 random scenarios, ~1 h with 8 workers)

Run 4 for any change to planning, control, simulation, a `node.py` or a dataflow. Run 5 before a release
or when reference numbers may move. After that, update `docs/evaluation.md`.

## Where things live

* The layout and extension points are in `docs/architecture.md` §3 and `docs/extending.md`. Keep them,
  the README layout and "Extending" blocks, and the code in sync.
* Components: `src/doraware/<component>/`. Algorithms are plain Python (no Dora imports), unit-tested in
  `tests/unit/<component>/`. Every Dora node is the `node.py` of its (sub-)package; shared node helpers are
  `core/node.py` (runtime), `core/clock.py` (`RateClock`) and `core/lockstep.py` (`InputBarrier`).
* Plugins: `<component>/plugins/<name>/`, registered in `<component>/registry.py` (built on
  `core/registry.py`, which also loads entry points; create plugins with `create_from_section`). Today:
  planning/path_planning (state_lattice),
  planning/trajectory_planning (path_pipeline, lane_follow), control (mpc), simulation (godot),
  evaluation/scenarios (parking, road), maps (yaml). Setups are config overlays (`configs/road.yaml`,
  `doraware run --config`).
* Dataflows: `dataflows/closed_loop.yml` (default) and `dataflows/trajectory_planner.yml`; `evaluation/suite/runner.py` renders it per run and appends the
  recorder (subscribed to every output). Node ids must equal the `NodeRuntime` names.
* Messages: `src/doraware/interfaces/<domain>.py`. Every dataclass with a `header` is a topic message.
  Interfaces matter most: design them as documented contracts (complete-state commands, request vs.
  reported state, frames/units/stamps) and update architecture.md §4.
* Configuration: `configs/default.yaml` is the only source of parameters, one section per package;
  plugin parameters live under the plugin name.
* Simulator: `simulator/godot/scripts/sim/` is deterministic physics; `scripts/gui/` is display only and
  must never write simulation state. Integration tests fail on any Godot `SCRIPT ERROR`. Check GUI changes
  with screenshots (`--gui --set "simulation.screenshot_times=[…]" "simulation.screenshot_cameras=[…]"`).
* Regression baseline: `tests/regression/baseline.json`. Update it only for an intentional,
  explained change.
* Planner: git submodule `third_party/state_lattice_based_valet_parking`. Do not patch it in place;
  adapt in `planning/path_planning/plugins/state_lattice/` or contribute upstream.
* Platform direction: a modular AD platform (Diffusion Planner, localization and perception are planned).
  Do not add parking-specific logic outside `planning/path_planning/plugins/state_lattice/` and
  `evaluation/scenarios/parking.py`.
* No CHANGELOG (maintainer decision). Behaviour changes go into the docs and the commit messages.

## Known Dora 1.1 quirks (already handled; keep these workarounds)

* Shared-memory payloads are lost if the sender exits early, so the scenario manager stays alive until `scenario_done`.
* Idle dataflows stop after ~120 s, so the dataflow has a `dora/timer/secs/5` heartbeat.
* Restarted nodes do not get inputs redelivered, so the planner announces the restart and the request is re-sent.
* There is an occasional startup hang, handled by the runner watchdog with reported retries.

## Release

Update the version in `pyproject.toml`, run step 5, update
`docs/evaluation.md`, then tag `vX.Y.Z`.
