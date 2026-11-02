# Contributing

Thanks for your interest in doraware. Bug reports, scenario contributions, new planner
adapters and controller improvements are welcome.

## Development setup

```bash
git clone --recursive https://github.com/yukkysaito/doraware.git && cd doraware
GODOT_BIN=/path/to/Godot_v4.7.2-stable_linux.x86_64 ./scripts/setup.sh
source .venv/bin/activate
```

`setup.sh` creates `.venv` with the pinned dependencies (including the `dev` extra), links the
Godot binary to `tools/godot`, imports the Godot project and warms up the planner's caches.

If your shell exports a `PYTHONPATH` from another environment (for example ROS), unset it for this
project (`env -u PYTHONPATH ...`). The scripts in `scripts/` already do this.

## Checks before opening a pull request

```bash
ruff check .                                   # lint
python -m pytest tests -m "not e2e"            # unit + Godot integration (~1 min)
python -m pytest tests/e2e -m e2e              # closed-loop system tests (~4 min)
./scripts/test_regression.sh                   # closed-loop regression vs. tests/regression/baseline.json
```

For changes to the planner adapter, trajectory generation, control or simulation, also run the
full suite (`./scripts/test_all.sh`) and include its `summary.md` pass counts in the pull request.

CI runs lint and unit tests only. Godot and closed-loop tests must be run locally.

## Ground rules

* **Never hide a failure.** Do not loosen acceptance criteria, regression tolerances or collision
  checks to make a run pass. Do not write planned or ideal states back into the simulator.
  If a scenario fails, report it and its cause.
* **Physics is the source of truth.** Metrics and verdicts are computed from the simulated vehicle
  state.
* **Keep determinism.** Simulation time is owned by the sim bridge (lockstep). Do not use wall-clock
  time in control or physics paths, and do not use unseeded randomness.
* **Configuration over constants.** New parameters go into `configs/default.yaml` and are
  documented in the relevant `docs/` page.
* Update `docs/` together with behaviour changes. If reference numbers move,
  update `docs/evaluation.md`. If the regression baseline changes, explain why
  (see [docs/testing.md](docs/testing.md)).

## Code layout and conventions

The layout is described in [docs/architecture.md §3](docs/architecture.md), and how to add planners,
components, controllers, simulator backends and scenario sources in [docs/extending.md](docs/extending.md).
In short:

* One package per component under `src/doraware/<component>/`. Its Dora entry point is `node.py`. All
  other modules are plain Python and are unit-testable without Dora or Godot. Components never import
  each other's internals; they communicate through messages.
* Messages are dataclasses in `src/doraware/interfaces/<domain>.py`, re-exported from
  `doraware.interfaces`. The Arrow codec is derived from the type annotations, so add fields there
  instead of hand-writing serialization. Changing a message is an interface change: say so in the
  pull request description.
* Algorithms that a component can host in several variants are plugins: `<component>/plugins/<name>/`
  registered in `<component>/registry.py`, with their parameters under their name in the component's
  config section. Heavy dependencies go in an optional extra and are imported lazily, or live in a
  separate package that registers through an entry point.
* The Godot GUI (`simulator/godot/scripts/gui/`) only displays; physics lives in `scripts/sim/`.
* Tests mirror the package: `tests/unit/<component>/`. Shared fixtures are in `tests/helpers/`.
* Use absolute imports (`from doraware.core.geometry import ...`).
* Units are SI, angles are radians, timestamps are integer nanoseconds of simulation time, and
  poses are in the map frame at the rear axle unless a name says otherwise.
* Python ≥ 3.11, 120-column lines, `ruff` as configured in `pyproject.toml`.

## Commits and pull requests

* Keep commits small and focused, and write the message in the imperative mood ("Add …", "Fix …").
* Describe what changed, why, and how you verified it (which tests or suites ran, with their results).
