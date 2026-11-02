import math

import numpy as np

from doraware.interfaces import GeometricPath, Gear, OccupancyGrid, Pose2D, VehicleParameters
from doraware.planning.trajectory_generation.stop_point import snap_stop_to_goal

V = VehicleParameters()


def _grid(blocked_x=None):
    d = np.zeros((200, 400), np.uint8)
    if blocked_x is not None:
        d[:, int(blocked_x / 0.1):] = 1
    return OccupancyGrid(resolution=0.1, width=400, height=200, data=d.reshape(-1))


def _line(x0, x1, gear=Gear.DRIVE, n=101):
    x = np.linspace(x0, x1, n)
    return GeometricPath(x=x, y=np.full(n, 10.0), yaw=np.zeros(n), curvature=np.zeros(n),
                         gear=np.full(n, int(gear), np.int8))


def test_undershoot_is_extended_forward():
    r = snap_stop_to_goal(_line(5, 15), Pose2D(15.08, 10.0, 0.0), _grid(), V)
    assert r.applied and abs(r.shift - 0.08) < 1e-9 and abs(r.path.x[-1] - 15.08) < 1e-9


def test_overshoot_is_truncated_in_reverse():
    p = _line(15, 5, Gear.REVERSE)
    p.yaw[:] = 0.0                      # driving backwards along -x with heading +x
    r = snap_stop_to_goal(p, Pose2D(5.06, 10.0, 0.0), _grid(), V)
    assert r.applied and r.shift < 0 and abs(r.path.x[-1] - 5.06) < 1e-6 and len(r.path) <= len(p)


def test_arc_extension_follows_curvature():
    k, n = 0.2, 101
    th = np.linspace(0, 0.5, n)
    p = GeometricPath(x=np.sin(th) / k + 10, y=(1 - np.cos(th)) / k + 5, yaw=th, curvature=np.full(n, k),
                      gear=np.full(n, int(Gear.DRIVE), np.int8))
    g_th = 0.5 + 0.05 * k
    goal = Pose2D(math.sin(g_th) / k + 10, (1 - math.cos(g_th)) / k + 5, g_th)
    r = snap_stop_to_goal(p, goal, _grid(), V, lateral_tol=0.01)
    assert r.applied and math.hypot(r.path.x[-1] - goal.x, r.path.y[-1] - goal.y) < 1e-3


def test_lateral_residual_is_not_touched():
    r = snap_stop_to_goal(_line(5, 15), Pose2D(15.0, 10.05, 0.0), _grid(), V)
    assert not r.applied and "lateral" in r.reason


def test_extension_respects_safety_margin():
    # vehicle front at x_end + wheel_base + front_overhang = 15 + 3.69; wall at 18.75 -> 6 cm left
    r = snap_stop_to_goal(_line(5, 15), Pose2D(15.08, 10.0, 0.0), _grid(blocked_x=18.75), V, margin=0.1)
    assert not r.applied and "margin" in r.reason
