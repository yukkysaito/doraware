"""Judge: terminal conditions and pass/fail against expectations (synthetic data)."""
import numpy as np

from doraware.evaluation.judge import ScenarioJudge
from doraware.evaluation.metrics import ClosedLoopMetrics
from doraware.interfaces import (GeometricPath, Gear, Header, OccupancyGrid, PlanningResult, PlanningStatus,
                                          Pose, SimulationStatus, Twist, Vector3, VehicleParameters, VehicleState,
                                          to_ns)

CFG = dict(goal_position_tolerance=0.10, goal_yaw_tolerance_deg=2.0, final_speed_tolerance=0.02, settle_time=1.0,
           rejection_hold_time=3.0, rejection_max_motion=0.01, timeout_margin=30.0, max_sim_time=300.0,
           min_controller_hz=20.0, out_of_bounds_check=True)
GOAL = (15.0, 10.0, 0.0)


def _grid():
    d = np.zeros((200, 300), np.uint8)
    d[0:20, :] = 1          # wall along the bottom (y < 2)
    return OccupancyGrid(resolution=0.1, width=300, height=200, data=d.reshape(-1))


def _state(t, x, y=10.0, yaw=0.0, v=0.0):
    return VehicleState(header=Header(stamp=to_ns(t)), pose=Pose.from_xyyaw(x, y, yaw),
                        twist=Twist(linear=Vector3(v, 0, 0)), gear=Gear.DRIVE)


def _plan(ok=True, status=PlanningStatus.SUCCESS):
    p = GeometricPath(x=np.linspace(5, 15, 50), y=np.full(50, 10.0), yaw=np.zeros(50), curvature=np.zeros(50),
                      gear=np.full(50, int(Gear.DRIVE), np.int8))
    return PlanningResult(status=status if not ok else PlanningStatus.SUCCESS, path=p if ok else GeometricPath())


def _judge(expected):
    m = ClosedLoopMetrics(_grid(), VehicleParameters(), GOAL)
    return ScenarioJudge(expected, CFG, m), m


def _drive(j, m, x_end, t_end=12.0):
    for k in range(int(t_end * 50) + 1):
        t = k / 50
        x = min(5 + t, x_end)
        m.on_state(_state(t, x, v=0.0 if x >= x_end else 1.0))
        from doraware.interfaces import ControlCommand
        m.on_command(ControlCommand(header=Header(stamp=to_ns(t))))
        v = j.update(t)
        if v.done:
            return v
    return v


def test_goal_reached_passes():
    j, m = _judge({"scenario_class": "feasible", "plan_expected": True})
    p = _plan()
    m.on_plan(p)
    j.on_plan(p, 0.0)
    j.on_plan_timing(0.0, 10.0)
    v = _drive(j, m, 15.0)
    assert v.done and v.passed and v.outcome == "goal_reached", v.reasons


def test_goal_missed_fails():
    j, m = _judge({"scenario_class": "feasible", "plan_expected": True})
    p = _plan()
    m.on_plan(p)
    j.on_plan(p, 0.0)
    j.on_plan_timing(0.0, 10.0)
    v = _drive(j, m, 14.8)
    assert v.done and not v.passed and any("goal position" in r for r in v.reasons)


def test_expected_rejection_with_still_vehicle_passes():
    j, m = _judge({"scenario_class": "infeasible", "plan_expected": False, "expected_status": "no_path"})
    p = _plan(False, PlanningStatus.NO_PATH)
    j.on_plan(p, 0.0)
    for k in range(200):
        m.on_state(_state(k / 50, 5.0))
        v = j.update(k / 50)
        if v.done:
            break
    assert v.done and v.passed and v.outcome == "rejected"


def test_rejection_with_motion_or_wrong_status_fails():
    j, m = _judge({"scenario_class": "invalid_input", "plan_expected": False, "expected_status": "goal_in_collision"})
    j.on_plan(_plan(False, PlanningStatus.NO_PATH), 0.0)
    for k in range(200):
        m.on_state(_state(k / 50, 5.0 + 0.001 * k))
        v = j.update(k / 50)
        if v.done:
            break
    assert not v.passed and any("expected goal_in_collision" in r for r in v.reasons)
    assert any("moved" in r for r in v.reasons)


def test_collision_terminates_and_fails():
    j, m = _judge({"scenario_class": "feasible", "plan_expected": True})
    p = _plan()
    m.on_plan(p)
    j.on_plan(p, 0.0)
    j.on_plan_timing(0.0, 10.0)
    m.on_state(_state(0.0, 5.0))
    m.on_sim_status(SimulationStatus(header=Header(stamp=to_ns(0.0)), collision_count=1, collision_objects="wall"))
    m.on_state(_state(0.6, 5.0))
    v = j.update(0.6)
    assert v.done and not v.passed and v.outcome == "collision"


def test_footprint_overlap_detected():
    m = ClosedLoopMetrics(_grid(), VehicleParameters(), GOAL)
    m.on_state(_state(0.0, 10.0, y=2.5))   # half width ~0.95 -> overlaps the wall below y=2
    assert m.geometric_collisions == 1 and m.min_clearance == 0.0


def test_start_in_collision_without_motion_is_not_a_system_collision():
    j, m = _judge({"scenario_class": "invalid_input", "plan_expected": False,
                   "expected_status": "start_in_collision"})
    j.on_plan(_plan(False, PlanningStatus.START_IN_COLLISION), 0.0)
    for k in range(200):
        m.on_state(_state(k / 50, 10.0, y=2.5))   # overlapping from the first state, never moving
        m.on_sim_status(SimulationStatus(header=Header(stamp=to_ns(k / 50)), collision_count=1,
                                         collision_objects="wall"))
        v = j.update(k / 50)
        if v.done:
            break
    assert v.done and v.passed and v.outcome == "rejected", v.reasons


def _traj_at(t, moving, x=15.0):
    """Trajectory window at sim time t: still moving forward, or stationary at x (15 = the goal)."""
    from doraware.interfaces import Trajectory
    n = 20
    v = np.full(n, 1.0 if moving else 0.0)
    return Trajectory(header=Header(stamp=to_ns(t)), start_time=to_ns(t), time_step=0.1,
                      valid_until=to_ns(t + 1.0), x=np.full(n, x), y=np.full(n, 10.0), yaw=np.zeros(n),
                      longitudinal_velocity=v, longitudinal_acceleration=np.zeros(n), curvature=np.zeros(n),
                      front_wheel_angle=np.zeros(n), gear=np.full(n, int(Gear.DRIVE), np.int8))


def test_without_plan_timing_the_motion_ends_when_the_trajectory_stops():
    """Trajectory planners announce no plan end: the judge waits until the newest
    trajectory has no moving point any more, then for the settle time."""
    j, m = _judge({"scenario_class": "feasible", "plan_expected": True})
    p = _plan()
    m.on_plan(p)
    j.on_plan(p, 0.0)
    for k in range(100):                                  # stopped before the goal (waiting): not the end
        t = k / 50
        m.on_trajectory(_traj_at(t, moving=False, x=10.0))
        m.on_state(_state(t, 10.0))
        assert not j.update(t).done
    for k in range(100, 400):                             # parked at the goal, but the plan still moves
        t = k / 50
        m.on_trajectory(_traj_at(t, moving=True))
        m.on_state(_state(t, 15.0))
        assert not j.update(t).done
    for k in range(400, 600):                             # the trajectory stops: settle, then goal reached
        t = k / 50
        m.on_trajectory(_traj_at(t, moving=False))
        m.on_state(_state(t, 15.0))
        from doraware.interfaces import ControlCommand
        m.on_command(ControlCommand(header=Header(stamp=to_ns(t))))
        v = j.update(t)
        if v.done:
            break
    assert v.done and v.passed and v.outcome == "goal_reached", v.reasons
    assert 8.0 + 1.0 - 1e-9 <= t < 8.0 + 1.1
