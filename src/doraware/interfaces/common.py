"""Common message primitives: time, header, geometry.

The message dataclasses in :mod:`doraware.interfaces` are the *only* contract between nodes. They are planner-,
controller- and simulator-agnostic. Every message is a dataclass that the
codec (:mod:`doraware.interfaces.codec`) maps to a typed Apache Arrow struct,
which is what Dora transports.

Conventions (see docs/architecture.md, "Frames, units and time"):

* SI units everywhere (m, s, rad, m/s, m/s^2, kg).
* ``map`` frame: right-handed, x east, y north, z up; yaw is the CCW angle
  about +z from +x, wrapped to [-pi, pi).
* ``base_link`` frame: origin at the **rear axle center** projected on the
  ground, x forward, y left, z up. Every vehicle pose in the system is the pose
  of ``base_link`` in ``map``.
* Time is an ``int`` nanosecond count in an explicit clock domain
  (:class:`ClockDomain`). Simulation time and wall time are never mixed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Annotated

import numpy as np


# Array field annotations understood by the codec (dtype is part of the schema).
F64Array = Annotated[np.ndarray, "float64"]
I8Array = Annotated[np.ndarray, "int8"]
U8Array = Annotated[np.ndarray, "uint8"]

NS_PER_S = 1_000_000_000


def to_ns(seconds: float) -> int:
    """Seconds -> integer nanoseconds (rounded, not truncated)."""
    return int(round(seconds * NS_PER_S))


def to_sec(ns: int) -> float:
    return ns / NS_PER_S


# =================================================================== enums
class ClockDomain(IntEnum):
    SIM = 0     # simulation time, starts at 0 for every scenario, advanced by the physics stepper
    WALL = 1    # host wall clock (time.time_ns), only for diagnostics / latency measurement


# ================================================================= geometry
@dataclass
class Vector3:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Quaternion:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    w: float = 1.0

    @staticmethod
    def from_yaw(yaw: float) -> "Quaternion":
        return Quaternion(0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))

    @property
    def yaw(self) -> float:
        # general formula (valid for any rotation, returns the heading)
        siny = 2.0 * (self.w * self.z + self.x * self.y)
        cosy = 1.0 - 2.0 * (self.y * self.y + self.z * self.z)
        return math.atan2(siny, cosy)


@dataclass
class Pose:
    position: Vector3 = field(default_factory=Vector3)
    orientation: Quaternion = field(default_factory=Quaternion)

    @staticmethod
    def from_xyyaw(x: float, y: float, yaw: float) -> "Pose":
        return Pose(Vector3(x, y, 0.0), Quaternion.from_yaw(yaw))

    @property
    def yaw(self) -> float:
        return self.orientation.yaw

    def xyyaw(self) -> tuple:
        return (self.position.x, self.position.y, self.yaw)


@dataclass
class Twist:
    linear: Vector3 = field(default_factory=Vector3)
    angular: Vector3 = field(default_factory=Vector3)


# =================================================================== header
@dataclass
class Header:
    """``stamp``: when the message was generated, in ``clock`` domain [ns].
    Messages whose data refers to another instant carry an extra field
    (e.g. :attr:`Trajectory.start_time`)."""
    stamp: int = 0
    frame_id: str = "map"
    sequence_id: int = 0
    clock: ClockDomain = ClockDomain.SIM

    @property
    def stamp_sec(self) -> float:
        return to_sec(self.stamp)


@dataclass
class Pose2D:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
