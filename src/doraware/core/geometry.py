"""Planar geometry helpers shared by all components (pure functions, SI units)."""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def wrap_angle(a: float) -> float:
    """Wrap to [-pi, pi)."""
    r = (a + math.pi) % (2.0 * math.pi) - math.pi
    return r - 2.0 * math.pi if r >= math.pi else r


def wrap_angles(a: np.ndarray) -> np.ndarray:
    r = (np.asarray(a, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi
    return np.where(r >= np.pi, r - 2.0 * np.pi, r)


def angle_diff(a: float, b: float) -> float:
    """Smallest signed a - b."""
    return wrap_angle(a - b)


def unwrap_near(angles: np.ndarray, reference: float) -> np.ndarray:
    """Unwrap a sequence and shift it by 2*pi*k so its first element is within
    pi of ``reference`` (continuous yaw for linearisation / interpolation)."""
    u = np.unwrap(np.asarray(angles, dtype=float))
    if len(u) == 0:
        return u
    return u + 2.0 * math.pi * round((reference - u[0]) / (2.0 * math.pi))


def interp_angle(a0: float, a1: float, t: float) -> float:
    return wrap_angle(a0 + t * angle_diff(a1, a0))


def body_to_world(x: float, y: float, yaw: float, px: float, py: float) -> Tuple[float, float]:
    """Point (px, py) given in the body frame at pose (x, y, yaw) -> world."""
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * px - s * py, y + s * px + c * py


def world_to_body(x: float, y: float, yaw: float, wx: float, wy: float) -> Tuple[float, float]:
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = wx - x, wy - y
    return c * dx + s * dy, -s * dx + c * dy


def path_errors(x: float, y: float, yaw: float, rx: float, ry: float, ryaw: float) -> Tuple[float, float, float]:
    """(longitudinal, lateral, heading) error of pose w.r.t. a reference pose,
    expressed in the reference frame (lateral +left)."""
    lon, lat = world_to_body(rx, ry, ryaw, x, y)
    return lon, lat, angle_diff(yaw, ryaw)


def footprint_corners(x: float, y: float, yaw: float, front: float, rear: float, left: float,
                      right: float) -> np.ndarray:
    """4x2 CCW corners of the rectangle [−rear, front] x [−right, left] (body) at the pose."""
    local = np.array([[front, -right], [front, left], [-rear, left], [-rear, -right]])
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([x, y])
