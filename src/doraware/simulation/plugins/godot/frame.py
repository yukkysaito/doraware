"""Explicit map <-> Godot frame conversions (docs/architecture.md §5).

This module is the *only* place where the two frames meet.

* ``map``: x east, y north, z up, yaw CCW about +z.
* Godot (Y up): ``X_g = x``, ``Y_g = z``, ``Z_g = -y``; rotation about +Y_g
  (``rot_y``) equals map yaw.
* Vehicle local axes in Godot: +X forward, +Y up, -Z left. Hence a vector in
  base_link (x fwd, y left, z up) maps to Godot local ``(x, z, -y)``.
* Steering angle and yaw rate are scalars about the vertical axis with the
  same sign in both frames (+ = CCW / left).

Pure functions, no Godot or Dora imports.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from doraware.core.geometry import wrap_angle
from doraware.interfaces import (Gear, Header, PerceivedObjects, Pose, SimulationScene, TurnIndicator, Twist,
                                 Vector3,
                                 VehicleParameters, VehicleSignalCommand, VehicleSignalStatus, VehicleState,
                                 to_ns)


def _wrap(a: float) -> float:
    """Wrap to [-pi, pi), leaving in-range values bit-exact."""
    a = float(a)
    return a if -math.pi <= a < math.pi else float(wrap_angle(a))


# ------------------------------------------------------------------ points / poses
def map_to_godot_point(x: float, y: float) -> Tuple[float, float]:
    """Map (x, y) -> Godot (X, Z) on the ground plane."""
    return float(x), float(-y)


def godot_to_map_point(gx: float, gz: float) -> Tuple[float, float]:
    return float(gx), float(-gz)


def map_to_godot_pose(x: float, y: float, yaw: float) -> Dict[str, float]:
    """base_link pose in map -> Godot body pose ``{"x", "z", "rot_y"}`` (rear axle)."""
    gx, gz = map_to_godot_point(x, y)
    return {"x": gx, "z": gz, "rot_y": _wrap(yaw)}


def godot_to_map_pose(gx: float, gz: float, rot_y: float) -> Tuple[float, float, float]:
    x, y = godot_to_map_point(gx, gz)
    return x, y, _wrap(rot_y)


def map_to_godot_polyline(points: Iterable[Sequence[float]]) -> List[List[float]]:
    """[(x, y), ...] (map) -> [[X, Z], ...] (Godot)."""
    return [list(map_to_godot_point(p[0], p[1])) for p in points]


# ------------------------------------------------------------------ vectors
def base_link_to_godot_local(v: Sequence[float]) -> Tuple[float, float, float]:
    """Vector in base_link (x fwd, y left, z up) -> Godot vehicle local (X fwd, Y up, Z right)."""
    return float(v[0]), float(v[2]), float(-v[1])


def godot_local_to_base_link(v: Sequence[float]) -> Tuple[float, float, float]:
    """Godot vehicle-local vector (X fwd, Y up, -Z left) -> base_link (x, y, z)."""
    return float(v[0]), float(-v[2]), float(v[1])


def map_to_godot_vector(v: Sequence[float]) -> Tuple[float, float, float]:
    """World vector map (x, y, z) -> Godot world (X, Y, Z)."""
    return float(v[0]), float(v[2]) if len(v) > 2 else 0.0, float(-v[1])


def godot_to_map_vector(v: Sequence[float]) -> Tuple[float, float, float]:
    return float(v[0]), float(-v[2]), float(v[1])


def vehicle_to_map(x: float, y: float, yaw: float, px: float, py: float) -> Tuple[float, float]:
    """Point (px, py) in base_link of a vehicle at map pose (x, y, yaw) -> map."""
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * px - s * py, y + s * px + c * py


def godot_vehicle_to_world(gx: float, gz: float, rot_y: float, lx: float, lz: float) -> Tuple[float, float]:
    """Point (lx, lz) in Godot vehicle local axes -> Godot world (X, Z).
    Rotation about +Y by rot_y: X' = X cos + Z sin, Z' = -X sin + Z cos."""
    c, s = math.cos(rot_y), math.sin(rot_y)
    return gx + c * lx + s * lz, gz - s * lx + c * lz


# ------------------------------------------------------------------ scene
def obstacle_boxes(scene: SimulationScene) -> List[Dict[str, float]]:
    """Scene obstacle rectangles (map, center/yaw/length along yaw/width) -> Godot boxes."""
    out = []
    for cx, cy, yaw, length, width in zip(scene.obstacles_cx, scene.obstacles_cy, scene.obstacles_yaw,
                                          scene.obstacles_length, scene.obstacles_width):
        p = map_to_godot_pose(cx, cy, yaw)
        out.append({"x": p["x"], "z": p["z"], "rot_y": p["rot_y"], "length": float(length),
                    "width": float(width), "height": float(scene.obstacle_height)})
    return out


def actor_specs(scene: SimulationScene) -> List[Dict]:
    """Scene actors (map) -> Godot actors ``{id, cls, length, width, height,
    waypoints: [[t, X, Z, rot_y], ...]}`` sorted by time (the motion model of
    ``doraware.simulation.actors`` is applied on the Godot side)."""
    from doraware.simulation.actors import actor_waypoints
    out = []
    for i in range(len(scene.actor_id)):
        wps = []
        for t, x, y, yaw in actor_waypoints(scene, i):
            p = map_to_godot_pose(x, y, yaw)
            wps.append([float(t), p["x"], p["z"], p["rot_y"]])
        out.append({"id": int(scene.actor_id[i]), "cls": int(scene.actor_class[i]),
                    "length": float(scene.actor_length[i]), "width": float(scene.actor_width[i]),
                    "height": float(scene.actor_height[i]), "waypoints": wps})
    return out


def map_bounds_to_godot(xmin: float, xmax: float, ymin: float, ymax: float) -> List[float]:
    """Map bounds -> ``[xmin, xmax, zmin, zmax]`` in Godot (z = -y)."""
    return [float(xmin), float(xmax), float(-ymax), float(-ymin)]


def vehicle_params_dict(vp: VehicleParameters) -> Dict[str, float]:
    return {k: (v if isinstance(v, str) else float(v)) for k, v in dataclasses.asdict(vp).items()}


def actuator_dict(actuator: Dict, vp: VehicleParameters) -> Dict:
    """Actuator config with ``steering_angle_limit: null`` resolved to the
    vehicle's ``max_steer_angle`` (the Godot side also handles null)."""
    out = dict(actuator)
    if out.get("steering_angle_limit") is None:
        out["steering_angle_limit"] = float(vp.max_steer_angle)
    return out


def reset_request(scene: SimulationScene, actuator: Dict, physics_hz: float, seed: int,
                  initial_gear: Optional[int] = None) -> Dict:
    """Build the ``reset`` request (architecture §7) from a map-frame scene."""
    init = map_to_godot_pose(scene.initial_pose.x, scene.initial_pose.y, scene.initial_pose.yaw)
    if initial_gear is not None:
        init["gear"] = int(initial_gear)
    return {
        "type": "reset",
        "physics_hz": float(physics_hz),
        "seed": int(seed),
        "vehicle": vehicle_params_dict(scene.vehicle),
        "actuator": actuator_dict(actuator, scene.vehicle),
        "scene": {
            "bounds": map_bounds_to_godot(scene.map_xmin, scene.map_xmax, scene.map_ymin, scene.map_ymax),
            "obstacles": obstacle_boxes(scene),
            "actors": actor_specs(scene),
            "goal": map_to_godot_pose(scene.goal_pose.x, scene.goal_pose.y, scene.goal_pose.yaw),
        },
        "initial": init,
    }


# ------------------------------------------------------------------ state
def state_pose_map(state: Dict) -> Tuple[float, float, float]:
    """(x, y, yaw) of base_link in map from a ``state`` reply."""
    return godot_to_map_pose(state["x"], state["z"], state["rot_y"])


def state_to_vehicle_state(state: Dict, header: Optional[Header] = None) -> VehicleState:
    """Godot ``state`` reply -> :class:`VehicleState` (base_link in map).

    ``twist.linear``/``acceleration`` are those of the rear-axle point in
    base_link: ``linear.y = -v_local.z``; ``angular.z = ang_vel_y``.
    If ``header`` is None, a SIM-clock header stamped ``state.time`` is made.
    """
    if header is None:
        header = Header(stamp=to_ns(float(state["time"])), frame_id="map")
    x, y, yaw = state_pose_map(state)
    v = godot_local_to_base_link(state["v_local"])
    a = godot_local_to_base_link(state["accel_local"])
    return VehicleState(
        header=header,
        pose=Pose.from_xyyaw(x, y, yaw),
        twist=Twist(linear=Vector3(*v), angular=Vector3(0.0, 0.0, float(state["ang_vel_y"]))),
        acceleration=Vector3(*a),
        steering_angle=float(state["steer"]),
        gear=Gear(int(state["gear"])),
    )


def object_boxes(objects: PerceivedObjects) -> List[Dict[str, float]]:
    """Perceived objects (map) -> Godot boxes ``{x, z, rot_y, y, length, width, height, cls, id}``."""
    out = []
    for i in range(len(objects)):
        p = map_to_godot_pose(objects.x[i], objects.y[i], objects.yaw[i])
        out.append({"x": p["x"], "z": p["z"], "rot_y": p["rot_y"], "y": float(objects.z[i]),
                    "length": float(objects.length[i]), "width": float(objects.width[i]),
                    "height": float(objects.height[i]), "cls": int(objects.classification[i]),
                    "id": int(objects.object_id[i])})
    return out


def state_to_signal_status(state: Dict, header: Optional[Header] = None) -> VehicleSignalStatus:
    """Godot ``state`` reply -> :class:`VehicleSignalStatus` (no frame conversion needed)."""
    if header is None:
        header = Header(stamp=to_ns(float(state["time"])), frame_id="base_link")
    return VehicleSignalStatus(header=header, turn_indicator=TurnIndicator(int(state["turn_indicator"])),
                               hazard_lights=bool(state["hazard_lights"]),
                               left_lamp_on=bool(state["left_lamp_on"]), right_lamp_on=bool(state["right_lamp_on"]))


def step_request(n: int, accel: float, steer: float, gear: int, emergency: bool = False,
                 signals: Optional[VehicleSignalCommand] = None) -> Dict:
    """``step`` request. accel: signed along base_link x; steer: bicycle angle, +left
    (same sign in Godot); gear: :class:`Gear` integer (0 = keep). ``signals``:
    the signal command to latch before these steps (None = keep the latched one)."""
    command = {"accel": float(accel), "steer": float(steer), "gear": int(gear), "emergency": bool(emergency)}
    if signals is not None:
        command["signals"] = {"turn_indicator": int(signals.turn_indicator),
                              "hazard_lights": bool(signals.hazard_lights)}
    return {"type": "step", "n": int(n), "command": command}


def viz_request(planned=(), predicted=(), path=None, info: Optional[Dict] = None, path_gear=(),
                objects: Optional[PerceivedObjects] = None) -> Dict:
    """``viz`` request from map-frame (x, y) polylines and HUD values (``info``:
    label -> number / string / bool). ``path_gear``: Gear per path point.
    ``objects``: perceived objects to draw (boxes in the Godot frame)."""
    msg = {"type": "viz",
           "planned": map_to_godot_polyline(_as_points(planned)),
           "predicted": map_to_godot_polyline(_as_points(predicted)),
           "info": dict(info or {})}
    if path is not None:
        msg["path"] = map_to_godot_polyline(_as_points(path))
        msg["path_gear"] = [int(g) for g in path_gear]
    if objects is not None:
        msg["objects"] = object_boxes(objects)
    return msg


def _as_points(pts) -> List[Sequence[float]]:
    if isinstance(pts, np.ndarray):
        return pts.reshape(-1, 2).tolist()
    return list(pts)
