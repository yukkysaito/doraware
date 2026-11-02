# Extending doraware

doraware is built so that new algorithms and new components can be added without
editing the existing ones. This guide shows how to add each kind of package. The
design behind it is in [architecture.md](architecture.md).

## The building blocks

| Concept | Where | How it is selected |
|---|---|---|
| **Message** (the contract between components) | `src/doraware/interfaces/<domain>.py` | imported from `doraware.interfaces` |
| **Component** (one Dora node, or a few) | `src/doraware/<component>/`, entry point `node.py` | listed in a dataflow (`dataflows/*.yml`) |
| **Plugin** (an algorithm a component can host) | `src/doraware/<component>/plugins/<name>/` | `type` / `backend` key in `configs/default.yaml`, resolved by `<component>/registry.py` |
| **External plugin** (in its own pip package) | anywhere | Python entry point in the registry's group |
| **Parameters** | `configs/default.yaml`, one section per component; a plugin's own parameters under its name | `--set key.path=value` per run |

Plugin registries (all built on `doraware.core.registry.Registry`):

| Registry | Config key | Entry-point group | Factory signature | Contract |
|---|---|---|---|---|
| `planning/registry.py` | `planning.planner.type` | `doraware.planners` | `(params, cfg)` | `PathPlanner.plan(PlanningRequest) -> PlanningResult` |
| `control/registry.py` | `control.controller.type` | `doraware.controllers` | `(params, cfg, vehicle)` | `Controller.on_trajectory(Trajectory)`, `.step(VehicleState, t_ns) -> (ControlCommand, ControlDebug)` |
| `simulation/registry.py` | `simulation.backend` | `doraware.simulators` | `(params, cfg, *, physics_hz, log_path, gui)` | `SimulatorBackend` in `simulation/backend.py` |
| `evaluation/scenarios/registry.py` | `evaluation.scenario_source` | `doraware.scenario_sources` | `()` | `ScenarioSource.suite(name, n_random)`, `.build(id, seed, cfg) -> ScenarioBundle` |

`params` is the plugin's own config section (for example `planning.planner.my_planner`,
`{}` when absent) and `cfg` the full system config.

Rules that keep the platform modular:

* Only `node.py` files (and `core/node.py`) import Dora. Everything else is plain
  Python and is unit-tested without Dora or a simulator.
* Components talk only through messages. They never import each other's modules.
  Shared utilities belong in `core/`.
* Heavy or optional dependencies (PyTorch, CUDA, ROS, ...) are imported lazily inside
  the plugin and declared as an optional extra in `pyproject.toml`, or live in an
  external package.
* Parking-specific logic stays in the parking plugin and the parking scenario source.

## Add a path planner (search, sampling, optimisation)

A path planner answers one `PlanningRequest` (start, goal, occupancy grid, vehicle,
opaque `planner_config` JSON) with one `PlanningResult` carrying a planner-agnostic
`GeometricPath` (x, y, yaw, curvature and gear per point). The existing trajectory
generation, controller, signals and evaluation then work unchanged.

1. Write the plugin in `src/doraware/planning/plugins/my_planner/planner.py`:

   ```python
   from doraware.interfaces import GeometricPath, PlanningRequest, PlanningResult, PlanningStatus

   class MyPlanner:
       def __init__(self, params: dict):
           self.step = float(params.get("step", 0.5))

       def plan(self, request: PlanningRequest) -> PlanningResult:
           # Never raise: report failures through the status.
           ...
           return PlanningResult(request_id=request.request_id, status=PlanningStatus.SUCCESS,
                                 path=GeometricPath(x=..., y=..., yaw=..., curvature=..., gear=...),
                                 goal=request.goal, details='{"iterations": 42}')
   ```

   Rules for the result: `gear` per point is the direction of the motion that
   arrives at that point (`Gear.DRIVE` / `Gear.REVERSE`), and a direction change is
   a cusp. Planner-specific statistics go into `details` (JSON), never into new
   message fields.

2. Register it in `planning/registry.py`:

   ```python
   @PLANNERS.register("my_planner")
   def _my_planner(params, cfg):
       from doraware.planning.plugins.my_planner.planner import MyPlanner
       return MyPlanner(params)
   ```

3. Add its parameters to `configs/default.yaml` under `planning.planner.my_planner`
   and select it with `planning.planner.type: my_planner` (or per run:
   `doraware run --scenario B2 --set planning.planner.type=my_planner`).

4. Add unit tests in `tests/unit/planning/test_my_planner_*.py` and run the
   closed-loop suites (see [testing.md](testing.md)).

**As an external package**, skip step 2 and declare an entry point instead:

```toml
# pyproject.toml of your package
[project.entry-points."doraware.planners"]
my_planner = "my_package.planner:MyPlanner"
```

The entry point is called like a built-in factory, `(params, cfg)`.

## Add a trajectory-output planner (for example Diffusion Planner)

Learned planners usually output a time-parameterised trajectory directly and run
periodically rather than once per request. Such a planner replaces both the
`planner` and the `trajectory_generator` nodes. It is a node of its own, because
its inputs (ego history, objects, lanes, route) are not those of a `PathPlanner`.

The contract a trajectory producer must keep, so that the controller, the signal
decider and the lockstep keep working:

* On every planner tick `t` of the sim clock (`RateClock(timing, "planner_hz")` on
  the `VehicleState` stamps) publish exactly one `Trajectory` with
  `header.stamp = start_time = t`.
* Fixed `time_step` (`timing.trajectory_time_step`), signed velocity, `gear` per
  point, and `valid_until = t + timing.trajectory_validity`.
* An empty trajectory means "stop" (no plan). Consumers bring the vehicle to a safe
  stop and switch the hazard lights on.
* At `t = 0` the node may take as long as it needs (wall time); the consumers wait
  for the first trajectory up to `timing.planning_wall_timeout`.

A minimal node (`src/doraware/planning/diffusion/node.py`, with the model in plain
Python modules next to it):

```python
from doraware.core.clock import RateClock
from doraware.core.node import NodeRuntime
from doraware.interfaces import Trajectory, VehicleState, to_ns

def main():
    rt = NodeRuntime("trajectory_planner", {"vehicle_state": VehicleState})
    ticks = RateClock(rt.cfg["timing"], "planner_hz")
    model = load_model(rt.cfg["planning"]["planner"]["diffusion"])      # lazy heavy import
    for ev in rt.events():
        if ev["type"] != "INPUT":
            continue
        state = rt.decode(ev)
        t = state.header.stamp
        if not ticks.is_tick(t):
            continue
        traj: Trajectory = model.infer(state, t)                         # your code
        traj.header = rt.header(t, "trajectory")
        traj.valid_until = t + to_ns(rt.cfg["timing"]["trajectory_validity"])
        rt.send("trajectory", traj)
```

Then copy `dataflows/closed_loop.yml` to `dataflows/trajectory_planner.yml`, replace
the `planner` and `trajectory_generator` nodes by this node (output `trajectory`),
point the `controller`, `signal_decider`, `sim_bridge` and `evaluator` inputs at it,
and run it with `doraware run --dataflow dataflows/trajectory_planner.yml`. The
recorder picks up the new topics by itself.

Not available yet, and needed for a full learned-planner setup:

* object, lane and route messages (add them as described below),
* other traffic participants in the simulator,
* an evaluation verdict without a `PlanningResult`. The current judge decides
  "rejected" vs. "goal reached" from the path planner's result and the trajectory
  generator's timing.

## Add a component (perception, localization, prediction, ...)

1. **Messages.** Create `src/doraware/interfaces/<domain>.py` (for example
   `perception.py` with `DetectedObjects`) as dataclasses with a `header: Header`
   field. Use the existing field types (`float`, `int`, `str`, `bool`, `IntEnum`,
   nested dataclasses, `F64Array` / `I8Array` / `U8Array`). The Arrow codec and the
   MCAP JSON schema are derived from the annotations. In `interfaces/__init__.py`,
   add the star re-export and append the module to `_DOMAINS`. Every dataclass with
   a `header` is then a topic message (`MESSAGE_TYPES`) and is recorded automatically.
   Design them as contracts: document frames, units and stamps, separate requests
   from reported state, and keep algorithm-specific data out of them.
2. **Algorithms.** Put them in `src/doraware/<component>/` as plain Python. If there
   will be several algorithms, give the component a `registry.py` and a `plugins/`
   directory like `planning/`.
3. **Node.** `src/doraware/<component>/node.py` with a `NodeRuntime("<node id>", inputs)`.
   The name must equal the node id in the dataflow. If the node runs on the sim
   clock and its consumers must see its output deterministically, publish on the
   ticks of a `RateClock` and let consumers wait with `core.lockstep.InputBarrier`.
4. **Wiring.** Add the node to a dataflow in `dataflows/` (`args: -m doraware.<component>.node`)
   and connect its outputs. Recording is automatic.
5. **Config.** Add a `<component>:` section to `configs/default.yaml`.
6. **Tests.** Unit tests in `tests/unit/<component>/`; if it influences driving, an
   end-to-end case in `tests/e2e/`.

While a component does not exist, the simulator provides its output as ground truth
(today: `VehicleState` instead of localization, the scenario's occupancy grid
instead of perception). Replacing ground truth with the new component is a dataflow
change.

## Add a controller

Implement the `Controller` protocol of `control/registry.py` in
`control/plugins/<name>/`, register its factory, and put its parameters under
`control.controller.<name>`. Reuse `control/gear.py` (gear changes at standstill)
and `control/prediction.py` (actuator delay compensation) where they fit. The output
must respect the `ControlCommand` contract: signed acceleration along base_link x,
front-wheel angle, explicit gear requests, and `valid_until`.

## Add a simulator backend

Implement `SimulatorBackend` (`simulation/backend.py`) in
`simulation/plugins/<name>/`, register it in `simulation/registry.py`, and put its
settings under `simulation.<name>`. A backend works in interface types and the map
frame (convert frames inside the plugin), advances exactly `n` physics steps per
`step` call, latches vehicle signals, and is deterministic for a given seed. The sim
bridge keeps owning time and the lockstep.

## Add a scenario source

Implement `ScenarioSource` (`evaluation/scenarios/registry.py`). `build` returns a
`ScenarioBundle` (`evaluation/scenarios/bundle.py`): the `SimulationScene`, the
`PlanningRequest` and the `ExpectedBehavior` the judge checks. Register it and
select it with `evaluation.scenario_source`. `evaluation/scenarios/parking.py` is
the reference.

## Checklist for a pull request

* `ruff check .` and `pytest tests/unit` pass. CI runs both.
* Closed-loop tests: `pytest tests/e2e -m e2e` and `./scripts/test_regression.sh` for
  anything that can change driving.
* New parameters are in `configs/default.yaml` with units and a comment.
* New messages and plugins are documented: the message table in architecture.md §4,
  the layout in architecture.md §3 and in the README.
