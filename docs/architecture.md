# Architecture

doraware is a modular autonomous-driving platform built on Dora. This document
covers the platform's design principles, how components and plugins are
organised, the message contracts, frames, units and time, the deterministic
stepping scheme and the simulator protocol. MPC details are in [`mpc.md`](mpc.md),
the test strategy is in [`testing.md`](testing.md), and how to add packages is in
[`extending.md`](extending.md).

## 1. Design principles

1. **Interfaces are the only contract.** Components exchange typed messages
   (`doraware.interfaces`) and never import each other. Any component can be
   replaced, in any language Dora supports, as long as it speaks the same messages.
2. **Components are Dora nodes; algorithms are plain code.** Each component is a
   package whose `node.py` is the only file that imports Dora (decode → call →
   encode). The algorithms beside it are unit-testable without middleware.
3. **Algorithms are plugins.** A component that can host several algorithms
   (for example the planner) selects one from a registry by configuration, so
   new algorithms don't touch the rest of the stack.
4. **A system is a dataflow.** Which components run, and how they connect, is a
   Dora dataflow YAML (`dataflows/`). Configuration (`configs/`) only sets
   parameters.
5. **Closed-loop verification is built in.** A deterministic, lockstep physics
   simulator and an evaluator that judges the *simulated* vehicle let every
   change be checked end to end, reproducibly.

## 2. Components

```text
 scenario_manager ──scene──────────────────────────────────────────────┐
        │  └──planning_request──► planner ──planning_result──┐         │
        │                                                     ▼         ▼
        │                         trajectory_generator ◄── vehicle_state ── sim_bridge ◄──► simulator backend
        │                                │ trajectory            ▲   │          ▲           (Godot: physics,
        │                                ▼                       │   │          │            actuators, lamps,
        │                            controller ──control_command────────────────┘            collisions, GUI)
        │                                │ control_debug         │   └─sim_status
        └──scenario_info──► evaluator ◄──┴───────────────────────┘
                               │ scenario_done ──► sim_bridge (shutdown)
                  recorder (MCAP) ◄── every output (added by the runner)

 trajectory + vehicle_state ──► signal_decider ──vehicle_signal_command──► sim_bridge ──vehicle_signal_status──► evaluator
```

| Node (dataflow id) | Package | Responsibility |
|---|---|---|
| `planner` | `doraware.planning` | Runs the configured path planner plugin on a `PlanningRequest` and returns a `PlanningResult` with a planner-agnostic `GeometricPath`. |
| `trajectory_generator` | `doraware.planning.trajectory_generation` | Converts a `GeometricPath` into a time-parameterised profile (speed, jerk and steering-rate limits, stops at cusps), publishes fixed-step `Trajectory` windows at `planner_hz`, and revalidates collisions. |
| `signal_decider` | `doraware.planning.signals` | Decides the turn indicator and hazard lights from the executable `Trajectory` (planner-agnostic) at `planner_hz` and publishes `VehicleSignalCommand`. |
| `controller` | `doraware.control` | Runs the configured controller plugin (LTV-MPC) at `controller_hz`: tracks the `Trajectory`, manages gears and falls back safely. Outputs `ControlCommand` (acceleration, front-wheel angle, gear). |
| `sim_bridge` | `doraware.simulation` | Simulation coordinator: owns sim time, steps the configured simulator backend in lockstep, and publishes `VehicleState` / `VehicleSignalStatus` / `SimulationStatus`. |
| Godot | `simulator/godot` (backend `doraware.simulation.plugins.godot`) | Rigid-body vehicle physics (Jolt), tire forces, actuator dynamics, gear logic, signal lamps, collisions and the GUI. |
| `scenario_manager` | `doraware.evaluation.scenarios` | Builds the scenario from the configured scenario source (today the parking planner's test patterns): a `SimulationScene`, a `PlanningRequest` and the expected behaviour. |
| `evaluator` | `doraware.evaluation` | Online metrics, terminal conditions, pass/fail and result JSON. It is never in the control path. |
| `recorder` | `doraware.recording` | Writes every topic to MCAP (JSON encoding, JSON-Schema channels). The runner subscribes it to every node output. It is off the control path. |

Localization, perception and sensing are not implemented yet. The simulator
currently provides ground truth in the contracts those components will use:
`VehicleState`, and `PerceivedObjects` on `sim_bridge/ground_truth_objects`
(at `planner_hz`). The planner receives the map directly. A perception node's
`PerceivedObjects` wired into the sim bridge's optional `objects` input is
drawn in the GUI (boxes coloured by class, with ids). These components will be added as new packages with their own
messages and nodes, and the dataflow will route their outputs where ground truth
is used today ([extending.md](extending.md)).

## 3. Package layout and extension points

```text
src/doraware/
  interfaces/            message types per domain (common, vehicle, map, perception, planning, control, simulation, system) + Arrow codec
  core/                  config, clock, lockstep barrier, plugin registry, geometry, occupancy,
                         trajectory sampling, Dora node runtime (core/node.py)
  planning/              planner node + registry.py (path planner plugins)
    plugins/state_lattice/   State Lattice adapter and request policy
    trajectory_generation/   path -> Trajectory node and algorithms
    trajectory_planning/     trajectory planner node + registry, plugins/path_pipeline
    signals/                 turn indicator / hazard light decision node
  control/               controller node + registry.py, gear manager, delay compensation
    plugins/mpc/             LTV-MPC trajectory follower: model, reference, QP, modes
  simulation/            sim bridge node, lockstep coordinator, ground truth, backend contract + registry.py
    plugins/godot/           Godot backend: process, protocol, frame conversion
  evaluation/            evaluator node, metrics, judge
    scenarios/               scenario manager node, ScenarioBundle, registry.py, parking source
    suite/                   runner (`doraware run`), reports, closed-loop regression
  recording/             MCAP recorder node, replay
  cli.py                 `doraware run | regression | replay`
```

Conventions:

* A component's Dora entry point is always `<package>/node.py`, referenced from the
  dataflow as `-m doraware.<package>.node`. Only `node.py` and `core/node.py`
  import Dora. The `NodeRuntime` name is the node id in the dataflow.
* A component that hosts several algorithms has a `registry.py` and a
  `plugins/<name>/` directory. The plugin is chosen by a config key, its own
  parameters live under its name in the component's config section, and external
  packages can add plugins through Python entry points (`core/registry.py`).
* Configuration sections follow the packages: `planning`, `control`, `simulation`,
  `evaluation`, `recording`, plus the system-wide `timing` and `vehicle`.

| Extension | Registry / place | Selected by |
|---|---|---|
| path planner | `planning/registry.py` | `planning.planner.type` |
| trajectory-output planner | `planning/trajectory_planning/registry.py` | `planning.trajectory_planner.type` + `--dataflow dataflows/trajectory_planner.yml` |
| controller | `control/registry.py` | `control.controller.type` |
| simulator backend | `simulation/registry.py` | `simulation.backend` |
| scenario source | `evaluation/scenarios/registry.py` | `evaluation.scenario_source` |
| new component (perception, localization, ...) | `src/doraware/<component>/`, `interfaces/<domain>.py` | dataflow |

Step-by-step instructions and code skeletons are in [extending.md](extending.md).

## 4. Messages

The messages are defined in `src/doraware/interfaces/`, with one module per
domain, and are transported as typed Arrow structs (`interfaces/codec.py`).
The schema is derived from the dataclass annotations, and the metadata carries
`schema` and `schema_version`.

| Message | Key fields |
|---|---|
| `Header` | `stamp` [ns], `frame_id`, `sequence_id`, `clock` (`SIM`/`WALL`) |
| `VehicleState` | pose (base_link in map), twist/accel (base_link frame), `steering_angle` (front-wheel equivalent), `gear` |
| `Trajectory` | `start_time` [ns], `time_step` [s], `valid_until` [ns], per point: x, y, yaw, signed `longitudinal_velocity`, `longitudinal_acceleration`, `curvature`, `front_wheel_angle`, `gear` |
| `ControlCommand` | `target_acceleration` [m/s², signed along base_link x], `target_steering_angle` [rad, front wheel], `gear_request`, `valid_until`, `emergency_stop` |
| `ControlDebug` | MPC mode, solver status/time/iterations/cost, tracking errors, predicted trajectory |
| `VehicleSignalCommand` | `turn_indicator` (`TurnIndicator`: `OFF`/`LEFT`/`RIGHT`), `hazard_lights` |
| `VehicleSignalStatus` | active `turn_indicator` / `hazard_lights`, instantaneous `left_lamp_on` / `right_lamp_on` |
| `PerceivedObjects` | per object (parallel arrays): `object_id` (stable while tracked, −1 = untracked), `classification` (`ObjectClass`) + probability, existence probability, box centre / yaw / size, velocity and yaw rate in map |
| `VehicleParameters` | Autoware VehicleInfo geometry + mass, CoG, yaw inertia, friction, actuator limits |
| `OccupancyGrid` | row-major `data[iy*width+ix]`, resolution, origin |
| `PlanningRequest` | start, goal (`Pose2D`), map, vehicle, opaque `planner_config` JSON |
| `PlanningResult` | `status` (`PlanningStatus`), message, planning time, `GeometricPath` (x, y, yaw, curvature, gear) |
| `SimulationScene` | map bounds, obstacle boxes, initial pose, goal pose (for display), vehicle, scripted actors (id, `ObjectClass`, box size, timed waypoints; linear interpolation by sim time) |
| `SimulationStatus` | state, step count, physics dt, collision count/objects, command timeouts, longest lockstep wait (wall) |
| `Diagnostics` | node, level, message, JSON values |
| `ScenarioDone` | outcome, passed, JSON summary |

Rules:

* Planner-specific data (arc length, primitives, gear-change count, search
  statistics) never enters `Trajectory`; at most it goes into
  `PlanningResult.details` (diagnostics only).
* `Trajectory.gear` makes the direction explicit, also at standstill (signed
  velocity is ambiguous when v = 0).
* `Trajectory` point *i* is valid at `start_time + i * time_step`. Consumers
  query it by sim time (`doraware.core.trajectory.sample`), never by index.
* After the last point a trajectory may only be held if that point is
  stationary.
* Vehicle signals follow the same split as the gear: a request
  (`VehicleSignalCommand`) and what the vehicle actually does
  (`VehicleSignalStatus`). A command is the complete desired state (there is
  no "keep" value), so it is idempotent. The vehicle latches it, so it is
  published at a fixed rate. Turn indicator and hazard lights are independent
  requests. Hazard lights take priority on the lamps, and the requested
  indicator resumes after them. The vehicle does the flashing; publishers
  never toggle lamps. Signals never affect motion.

## 5. Frames, units and time

* SI units, angles in rad, yaw wrapped to [-π, π).
* `map`: x east, y north, z up, yaw CCW about +z.
* `base_link`: **rear axle center** on the ground, x forward, y left, z up.
  All vehicle poses (planner start/goal, path, trajectory, vehicle state) are
  base_link poses. The simulator body origin is placed at the rear axle; the
  center of gravity is at `cog_x` ahead of it. Velocities reported by the
  simulator are those of the rear-axle point (not the CoG):
  `v_ra = v_cog + ω × (r_ra − r_cog)`.
* Steering angle: equivalent bicycle front-wheel angle, + = left (CCW).
  The simulator maps it to Ackermann left/right wheel angles:
  `tan δ_{l,r} = L / (L / tan δ ∓ T/2)`.
* Godot frame (Y up): `X_g = x`, `Y_g = z`, `Z_g = −y`; rotation about +Y_g
  equals map yaw. Vehicle local axes in Godot: +X forward, +Y up, −Z left.
  The conversion lives only in `doraware/simulation/plugins/godot/frame.py`
  (unit-tested); the Godot side never sees map coordinates.
* Time: `int` nanoseconds. `ClockDomain.SIM` starts at 0 for every scenario
  and is advanced only by physics steps (`step * physics_dt`). Wall time is
  used only for latency / compute-time diagnostics and timeouts.

## 6. Deterministic lockstep (simulation coordination)

Rates are independent (`configs/default.yaml: timing`): physics (100 Hz),
controller (50 Hz), trajectory publication (10 Hz), state publication,
visualization. All of them are defined on the **sim clock**. The sim bridge
is the only clock source.

1. `sim_bridge` publishes `VehicleState(t)` at every state tick and at every
   controller/planner tick.
2. `trajectory_generator` publishes `Trajectory` stamped `t` when `t` is a
   planner tick (`t = k / planner_hz`). At `t = 0` it waits for the
   `PlanningResult` first (planning time is not sim time: the plan is
   available at t = 0 deterministically).
3. `controller`, on `VehicleState(t)` with `t` a controller tick, waits until
   it holds the trajectory of the latest planner tick `≤ t` (barrier), then
   publishes `ControlCommand` stamped `t`. `signal_decider` does the same on
   planner ticks and publishes `VehicleSignalCommand` stamped `t`.
4. `sim_bridge` steps Godot only after the command for the current controller
   tick (and, on planner ticks, the signal command, if a signal source is wired
   in the dataflow) has arrived, applies it at `t` and advances physics to the
   next tick.

Because each consumer waits for the producer's message of the same sim time,
the result does not depend on host load or rendering frame rate. Every
barrier has a wall-clock liveness timeout (`lockstep_wall_timeout`, 30 s). While
a node waits, sim time is paused, so a late message cannot change results; waits
longer than `lockstep_stall_warning` (2 s) are only reported (warning diagnostics,
`SimulationStatus.max_lockstep_wait`). When the timeout fires, the producer is
declared dead: the node proceeds with what it has and reports a `Diagnostics` error (fault
injection tests use this: a silent planner → expired trajectory → safe stop;
a silent controller → expired command → simulator emergency braking).

Execution order inside one controller period: command(t) applied → physics
steps t → t + Δ (n = physics_hz / controller_hz steps) → state(t + Δ).

## 7. Simulator backend and the Godot protocol

The sim bridge drives the simulator through the `SimulatorBackend` contract
(`simulation/backend.py`): `reset(scene, seed)`, `step(n, command, signals)`
and the GUI-only `send_viz` / `screenshot`, all in interface types and the
map frame. A step returns a `SimFrame` (`VehicleState`, `VehicleSignalStatus`,
contacts). The Godot backend (`simulation/plugins/godot/`) implements it with
the protocol below; `frame.py` is the only place where the map and Godot
frames meet.

TCP, the bridge listens, Godot connects (`--port` user argument).
Newline-delimited JSON, one request → at most one reply. All coordinates are
in the **Godot frame** (see §5), rear-axle reference.

| Request | Reply |
|---|---|
| `{"type":"reset","physics_hz":100,"seed":0,"vehicle":{…VehicleParameters…},"actuator":{…},"scene":{"bounds":[xmin,xmax,zmin,zmax],"obstacles":[{"x","z","rot_y","length","width","height"}],"goal":{"x","z","rot_y"}},"initial":{"x","z","rot_y"}}` | `state` |
| `{"type":"step","n":2,"command":{"accel","steer","gear","emergency","signals":{"turn_indicator","hazard_lights"}}}` | `state` after n physics steps |
| `{"type":"viz","planned":[[x,z],…],"predicted":[[x,z],…],"path":[[x,z],…],"path_gear":[…],"info":{…}}` | none |
| `{"type":"screenshot","path":"…","camera":0}` | none |
| `{"type":"quit"}` | none (Godot exits) |

`state`: `{"type":"state","step","time","x","z","rot_y","v_local":[vx,vy,vz],"ang_vel_y","accel_local":[ax,ay,az],"steer","accel_actual","gear","turn_indicator","hazard_lights","left_lamp_on","right_lamp_on","contacts","contact_names":[…],"step_wall_time"}`
(`v_local`, `accel_local`: rear-axle point in vehicle local axes).

Details fixed by the implementation (`simulator/godot`, `doraware.simulation.plugins.godot.client`):

* `gear` (command and state) is the integer `Gear` value (0 = keep current gear).
  `reset.initial` may carry an optional `gear` (default `PARK`).
* `signals` is optional. When present it is latched before the steps; when
  absent the latched signals are kept. A `reset` switches all signals off.
  `turn_indicator` is the integer `TurnIndicator` value (an invalid value is
  treated as `OFF`). The lamp fields report the lamp outputs during the last
  physics step.
* `contacts` is the cumulative number of new contacts (Jolt `body_entered`) with
  obstacle and actor bodies; `contact_names` the unique names (`obstacle_<i>`,
  index in the `obstacles` list, or `actor_<id>`) in order of first contact.
* `reset.scene.actors` lists the scripted actors (`id`, `cls`, box size,
  `waypoints: [[t, x, z, rot_y], …]`); `state.actors` reports `[[id, x, z, rot_y], …]`
  (float32 node transforms) for tests.
* `state` additionally carries `shifting` (gear engagement in progress) and
  `gear_refusals` (cumulative gear requests refused because the car was moving).
* `viz` (GUI only) updates the overlays: `planned` (trajectory window),
  `predicted` (controller prediction), and, only when it changed, `path` (the
  planned geometric path) with `path_gear` (Gear per point). `info` holds HUD
  values (scenario, planning status, controller mode and solver, tracking
  errors, command, emergency).
* `screenshot` (GUI only) queues a PNG of a rendered frame from camera 0 top,
  1 chase, 2 free or 3 overview (`simulation.screenshot_times` /
  `screenshot_cameras`).
* The Godot scripts are split into `scripts/sim/` (vehicle, actuators, tires:
  deterministic physics) and `scripts/gui/` (world, car model, overlays, HUD,
  cameras). The GUI only reads the simulation, so GUI and headless runs are
  bit-identical (integration and e2e tests).
* Godot runs with `--fixed-fps <physics_hz>` (no real-time sync). Headless, the
  physics tick blocks until the next request; in GUI mode a tick without a
  pending step deactivates the physics server so the window keeps rendering.
  Exactly one physics step is executed per requested step in both modes.

## 8. Vehicle physics (Godot)

* `RigidBody3D` chassis (Jolt), mass, yaw inertia and custom center of mass
  from `VehicleParameters`; planar motion (linear Y and angular X/Z locked,
  gravity handled analytically through static axle loads).
* Four wheels at the axle corners. Each wheel produces a tire force applied
  at its contact point: lateral force from the slip angle (linear cornering
  stiffness, low-speed regularised and limited to never reverse the wheel's
  lateral velocity within one step), longitudinal drive/brake force, and the
  pair is limited by the friction circle `μ F_z`.
* Front wheels steered with Ackermann geometry from the actuator's bicycle
  angle.
* Powertrain/brake: actuator acceleration `a` → force `m a` along the body x
  axis. In `DRIVE` propulsion acts forward, in `REVERSE` backward; a request
  against the gear direction is braking, which opposes the motion and holds
  the car at standstill (never pushes it the other way). `PARK` holds.
  Gear changes are executed only below `gear_shift_max_speed` and take
  `gear_shift_time`.
* Signal lamps: a latched turn indicator / hazard light state and a flasher at
  `vehicle.actuator.flasher_frequency` (50 % duty, counted in physics steps). A change
  of the active function restarts the flasher with the lamps on. Hazard lights
  flash both sides; otherwise only the indicated side flashes.
* Actuators: dead time (FIFO), first-order lag, rate/jerk limits, angle and
  acceleration limits, optional Gaussian noise (seeded).
* Actors (scripted traffic participants) are kinematic boxes (`scripts/sim/actor.gd`)
  moved to their scripted pose at the end of each physics step, before Jolt
  integrates; the motion model is the one of `doraware.simulation.actors`
  (an integration test compares them at every step). Contacts with them count as
  collisions. The ground truth (`ground_truth_objects`) includes them with their
  velocities (ids `1_000_000 + actor id`).
* Obstacles are static boxes built from the planner's occupancy grid (merged
  occupied cells), so the simulated world and the planner's map coincide
  exactly. Contacts are counted from the physics engine; the evaluator
  additionally checks the actual footprint against the grid.

## 9. Configuration

`configs/default.yaml` holds all tunables, in one section per package:

| Section | Content |
|---|---|
| `timing` | rates on the sim clock, trajectory/command validity, lockstep timeouts, real-time pacing |
| `vehicle` | dynamics and limits; `vehicle.actuator`: actuator behaviour (emulated by the simulator, compensated by the controller), flasher |
| `planning` | `planner` (`type` + per-plugin sections), `trajectory_generation`, `signals` |
| `control` | `controller` (`type` + per-plugin sections) |
| `simulation` | `backend` + per-backend sections, GUI screenshots |
| `evaluation` | acceptance criteria, terminal conditions, `scenario_source` |
| `recording`, `fault_injection` | MCAP on/off; failures injected by tests |

Nodes read the file named by `$DORAWARE_CONFIG`; runs overlay dotted overrides
(`--set timing.controller_hz=20`), and unknown keys are an error. Scenario-specific
data (vehicle geometry, map, start, goal, planner overrides) comes from the
scenario, not the config.

## 10. Recording and replay

The runner appends the recorder node to the dataflow and subscribes it to every output of every node, so new
components are recorded without configuration. It writes MCAP: topic = output name (`<node>_<output>` when
several nodes share the name, e.g. `controller_diagnostics`), schema name = message class,
encoding JSON with a JSON-Schema derived from the same dataclass as the Arrow schema, log time = header stamp
(sim clock). Replay (`doraware.recording.replay`) distinguishes **state playback** (recorded VehicleState →
offline metrics/plots, no physics) from **command re-simulation** (recorded ControlCommand → a fresh
simulator backend with the recorded scene, actuator config and seed, together with the recorded VehicleSignalCommand;
vehicle states and signal states are verified bit-identical to the recording).

## 11. Lifecycle, shutdown, restart and reset

* **Start:** all nodes start with the dataflow; the scenario manager publishes scene / request / info and
  stays alive (heartbeat) until `scenario_done`.
* **Graceful shutdown:** the evaluator publishes `scenario_done` → the bridge quits Godot (`quit`, then kill
  on timeout) and exits → downstream inputs close → every node leaves its event loop (`STOP`).
* **Timeouts:** every lockstep barrier has a wall-clock timeout with a diagnostic; trajectories and
  commands carry `valid_until` (sim time) and expire → controller safe stop / simulator emergency braking.
* **Restart:** the planner runs with Dora `restart_policy: on-failure` (`max_restarts: 2`). Dora does not
  redeliver inputs to a restarted node, so the restarted planner announces itself (`diagnostics`,
  `restarted: true`) and the scenario manager re-sends the outstanding request (e2e test with
  `fault_injection.planner_crash_once`).
* **Reset:** the simulator protocol `reset` rebuilds vehicle and scene deterministically (tested with
  repeated resets in one process); each closed-loop run is a fresh dataflow, so node state never leaks
  between scenarios.

## 12. Design decisions from closed-loop evaluation

* **Steering control margin.** The scenario's `max_steer_angle` is the planner's limit and is sent to the
  planner unchanged. The planner draws arcs at exactly that limit, so a tracking controller would have no
  authority left to correct a heading error on such an arc: the steering actuator's lag at a full-lock
  reversal (τ = 50 ms, rate limit) produced an unrecoverable heading deficit of ~0.03 rad and 12 cm lateral
  error in B1, although the Godot car's steady-state curvature equals the kinematic one within 0.3 %.
  The physical actuator limit (and the MPC's steering bound) is therefore `max_steer_angle +
  vehicle.steering_control_margin` (0.05 rad). Result on B1: lateral error 12.3 cm → 1.4 cm.
* **Planner wall-clock limit.** The planner's deterministic budget is its node-expansion limit (product
  spec). Its wall-clock limit (default 20 s) made results depend on host load when many closed loops ran in
  parallel; `planning.planner.state_lattice.base_config.max_planning_time` raises it to 120 s (the same knob as the upstream
  `run_tests.py --time-limit`). Scenario overrides still take precedence; the original `max_time`
  expectations are still checked.
* **Dora idle stop.** Dora 1.1 stops a dataflow after ~120 s without traffic; the scenario manager has a 5 s
  timer heartbeat so long plans do not end the run.
* **Shared-memory lifetime.** A Dora sender must stay alive until receivers map large (SHM) messages; the
  scenario manager therefore lives until `scenario_done`.
* **Goal-accuracy budget allocation.** The system requirement (0.10 m / 2.0°, measured on the simulated
  car) cannot be met if the planner may itself end 0.10 m / 2.0° from the goal — in the first full random run
  six scenes failed exactly that way (plan end error = tolerance, controller error vs. the plan end
  0.5–2.8 cm / 0.2–1.9°). The planner's terminal tolerance is therefore configured to 0.05 m / 1.0°
  (`planning.planner.state_lattice.base_config`), leaving 0.05 m / 1.0° for tracking. Acceptance thresholds are unchanged;
  scenarios with their own tolerance settings (L-*) keep them and are judged with the documented rule in
  testing.md. The upstream planner test-suite is still run with upstream defaults.
  **Accuracy fallback:** a tighter tolerance can make a scene unsolvable within the planner's budget (N-030:
  NO_PATH after 200k expansions at 1°, solved in 20 s at the default 2°). The planner node therefore re-plans
  once with the planner's default tolerance when the tight attempt ran out of budget (timeout /
  max_expansions) — never after a proof (unreachable, start/goal in collision, invalid input) or an
  exhausted search space. Planning capability is thus never below the upstream defaults; the result records
  `tolerance_fallback` and the summed planning time.
* **Signal policy.** The signal decision uses only the executable `Trajectory`, so it works for any planner,
  including future planners that output trajectories directly. Two policies exist
  (`planning.signals.policy`). `maneuver_hazard` is the default for the current low-speed parking use: hazard
  lights for the whole manoeuvre (also while standing at a cusp or before departure), off once nothing moves
  any more, and no turn indicator. `turn_indicators` is for driving among traffic: it indicates the side the
  vehicle is about to move to (end of the upcoming motion within `lookahead_distance`, with hysteresis), and
  hazard lights are used only when there is no plan to follow. In both policies the hazard lights are on
  while the newest trajectory is empty (planning failed) or has expired (pipeline silent). When a behaviour
  or mission layer exists, the policy should come from the current manoeuvre instead of configuration.
* **Random-scene rejections** are accepted only as upstream's spec accepts them (`no_path` /
  `max_expansions` within the budget, and only where the spec allows a rejection); a wall-clock timeout is
  a failure.
* **Final stop-point generation.** When the plan ends short of / past the goal on the same line (longitudinal
  residual only), the trajectory generator moves the final stop onto the goal's projection (truncate, or
  extend along the final curvature after an exact footprint check with the planner's 0.1 m margin).
