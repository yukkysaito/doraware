import math

import numpy as np
import pyarrow as pa
import pytest

from doraware.interfaces import codec
from doraware.interfaces import (ControlCommand, Gear, Header, OccupancyGrid, PlanningRequest,
                                          PlanningResult, PlanningStatus, Pose, Quaternion, Trajectory,
                                          TurnIndicator, VehicleSignalCommand, VehicleSignalStatus, VehicleState,
                                          to_ns, to_sec)
from doraware.core.trajectory import sample


def _traj(n=11, dt=0.1, v=1.0, start=0):
    t = np.arange(n) * dt
    return Trajectory(header=Header(stamp=start), start_time=start, time_step=dt, valid_until=start + to_ns(1.0),
                      x=v * t, y=np.zeros(n), yaw=np.zeros(n), longitudinal_velocity=np.full(n, v),
                      longitudinal_acceleration=np.zeros(n), curvature=np.zeros(n), front_wheel_angle=np.zeros(n),
                      gear=np.full(n, int(Gear.DRIVE), dtype=np.int8))


def test_roundtrip_all_types():
    msgs = [_traj(), VehicleState(pose=Pose.from_xyyaw(1, 2, 3.0), gear=Gear.REVERSE),
            ControlCommand(target_acceleration=-0.5, gear_request=Gear.DRIVE, valid_until=5),
            PlanningRequest(map=OccupancyGrid(width=3, height=2, data=np.arange(6, dtype=np.uint8) % 2)),
            PlanningResult(status=PlanningStatus.NO_PATH, message="x"),
            VehicleSignalCommand(turn_indicator=TurnIndicator.LEFT, hazard_lights=True),
            VehicleSignalStatus(turn_indicator=TurnIndicator.RIGHT, hazard_lights=False, right_lamp_on=True)]
    for m in msgs:
        arr = codec.encode(m)
        assert isinstance(arr, pa.StructArray) and len(arr) == 1
        back = codec.decode(arr, type(m))
        assert codec.to_json_dict(back) == codec.to_json_dict(m)


def test_schema_mismatch_rejected():
    with pytest.raises(TypeError):
        codec.decode(codec.encode(ControlCommand()), VehicleState)
    with pytest.raises(TypeError):
        codec.check_metadata({"schema": "ControlCommand"}, VehicleState)


def test_quaternion_yaw():
    for yaw in (-3.1, -1.0, 0.0, 0.7, 3.1):
        assert math.isclose(Quaternion.from_yaw(yaw).yaw, yaw, abs_tol=1e-12)


def test_time_conversion():
    assert to_ns(0.1) == 100_000_000 and to_sec(to_ns(12.345)) == pytest.approx(12.345)


def test_sampling_time_based():
    tr = _traj(start=to_ns(2.0))
    p = sample(tr, 2.25)
    assert p.x == pytest.approx(0.25) and p.velocity == pytest.approx(1.0)
    assert sample(tr, 1.0).x == 0.0                     # before start: first point
    assert sample(tr, 100.0) is None                    # moving end: undefined beyond
    stop = _traj(v=0.0)
    assert sample(stop, 100.0).held                     # stationary end: hold


def test_sampling_yaw_wrap():
    tr = _traj(n=2)
    tr.yaw = np.array([math.pi - 0.1, -math.pi + 0.1])
    assert abs(abs(sample(tr, 0.05).yaw) - math.pi) < 1e-9


def test_signal_messages_are_registered_and_default_off():
    from doraware.interfaces import MESSAGE_TYPES
    assert MESSAGE_TYPES["VehicleSignalCommand"] is VehicleSignalCommand
    assert MESSAGE_TYPES["VehicleSignalStatus"] is VehicleSignalStatus
    c, s = VehicleSignalCommand(), VehicleSignalStatus()
    assert c.turn_indicator == TurnIndicator.OFF and not c.hazard_lights
    assert not (s.hazard_lights or s.left_lamp_on or s.right_lamp_on)
    assert [t.name for t in TurnIndicator] == ["OFF", "LEFT", "RIGHT"]
