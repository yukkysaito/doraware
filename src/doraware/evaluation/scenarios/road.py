"""Road scenario source: lane-following scenarios with a vectorized lane map, a
route and scripted traffic (built in code, no external data).

    R1  straight lane, drive to the goal
    R2  straight - 90 degree left curve - straight
    R3  straight lane, a pedestrian crosses in front: stop, wait, continue
    R4  two-lane road, an oncoming car passes in the other lane
    R5  a Lanelet2 map file (data/maps/two_lane_road.osm): two lanelets in a row,
        a car parked in the neighbouring lane
    R7  a lead vehicle: the car drives ahead, stops for 4 s and moves on; the
        vehicle follows at a time gap, stops behind it and continues
    R6  an OpenDRIVE map file (data/maps/curve_road.xodr): straight - 90 degree
        left arc - straight over three linked roads

They exercise the lane / route / object interfaces end to end with the
``lane_follow`` trajectory planner (configs/road.yaml).
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from doraware.core.config import REPO_ROOT
from doraware.evaluation.scenarios.bundle import FEASIBLE, ExpectedBehavior, ScenarioBundle
from doraware.interfaces import (Header, LaneMap, LaneRelation, ObjectClass, OccupancyGrid, PlanningRequest, Pose2D,
                                 Route, SimulationScene, VehicleParameters)

LANE_WIDTH = 3.5          # [m]
SPEED_LIMIT = 3.0         # [m/s] within the domain of the kinematic MPC
DS = 0.5                  # [m] centreline sampling


@dataclass
class _Actor:
    cls: ObjectClass
    size: Tuple[float, float, float]                       # length, width, height
    waypoints: List[Tuple[float, float, float, float]]     # t, x, y, yaw


@dataclass
class _Road:
    name: str
    description: str
    lanes: List[np.ndarray]                                # centrelines (n, 2), driving order
    successors: List[Tuple[int, int]] = field(default_factory=list)
    route: List[int] = field(default_factory=list)          # lane indices
    start: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    goal: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    actors: List[_Actor] = field(default_factory=list)
    map_file: Optional[str] = None                         # (loader, path) instead of `lanes`
    map_loader: str = "yaml"
    route_ids: List[int] = field(default_factory=list)     # lane ids, with map_file


def straight(x0: float, y0: float, yaw: float, length: float) -> np.ndarray:
    s = np.arange(0.0, length + 1e-9, DS)
    return np.stack([x0 + s * math.cos(yaw), y0 + s * math.sin(yaw)], axis=1)


def arc(x0: float, y0: float, yaw0: float, radius: float, angle: float) -> np.ndarray:
    """Arc starting at (x0, y0) with heading yaw0; angle > 0 turns left."""
    n = max(int(abs(radius * angle) / DS), 1)
    a = np.linspace(0.0, angle, n + 1)
    sign = 1.0 if angle > 0 else -1.0
    cx, cy = x0 - sign * radius * math.sin(yaw0), y0 + sign * radius * math.cos(yaw0)
    th = yaw0 - sign * math.pi / 2 + a
    return np.stack([cx + radius * np.cos(th), cy + radius * np.sin(th)], axis=1)


def _roads() -> Dict[str, _Road]:
    r2a = straight(0.0, 0.0, 0.0, 12.0)
    r2b = arc(12.0, 0.0, 0.0, 15.0, math.pi / 2)
    r2c = straight(27.0, 15.0, math.pi / 2, 14.0)
    return {
        "R1": _Road("Straight lane", "Drive along a straight lane to the goal.",
                    [straight(0.0, 0.0, 0.0, 45.0)], route=[0], start=(3.0, 0.0, 0.0), goal=(38.0, 0.0, 0.0)),
        "R2": _Road("Left curve", "Straight, a 90 degree left curve of radius 15 m, straight.",
                    [r2a, r2b, r2c], successors=[(0, 1), (1, 2)], route=[0, 1, 2],
                    start=(2.0, 0.0, 0.0), goal=(27.0, 25.0, math.pi / 2)),
        "R3": _Road("Pedestrian crossing", "A pedestrian crosses 20 m ahead; the vehicle stops, waits and continues.",
                    [straight(0.0, 0.0, 0.0, 45.0)], route=[0], start=(3.0, 0.0, 0.0), goal=(38.0, 0.0, 0.0),
                    actors=[_Actor(ObjectClass.PEDESTRIAN, (0.5, 0.5, 1.7),
                                   [(5.0, 23.0, -5.0, math.pi / 2), (12.0, 23.0, 5.0, math.pi / 2)])]),
        "R4": _Road("Oncoming car", "Two-lane road; a car drives towards the vehicle in the other lane.",
                    [straight(0.0, 0.0, 0.0, 45.0), straight(45.0, LANE_WIDTH, math.pi, 45.0)], route=[0],
                    start=(3.0, 0.0, 0.0), goal=(38.0, 0.0, 0.0),
                    actors=[_Actor(ObjectClass.CAR, (4.5, 1.8, 1.5),
                                   [(0.0, 44.0, LANE_WIDTH, math.pi), (15.0, 1.0, LANE_WIDTH, math.pi)])]),
        "R5": _Road("Lanelet2 map", "A Lanelet2 map: two lanelets in a row, a car parked in the left lane.",
                    [], map_file="data/maps/two_lane_road.osm", map_loader="lanelet2", route_ids=[1001, 1003],
                    start=(3.0, 0.0, 0.0), goal=(45.0, 0.0, 0.0),
                    actors=[_Actor(ObjectClass.CAR, (4.5, 1.8, 1.5), [(0.0, 22.0, LANE_WIDTH, 0.0)])]),
        "R7": _Road("Lead vehicle", "Follow a car that drives, stops for 4 s and drives on.",
                    [straight(0.0, 0.0, 0.0, 70.0)], route=[0], start=(3.0, 0.0, 0.0), goal=(60.0, 0.0, 0.0),
                    actors=[_Actor(ObjectClass.CAR, (4.5, 1.8, 1.5),
                                   [(0.0, 16.0, 0.0, 0.0), (2.0, 16.0, 0.0, 0.0), (14.5, 31.0, 0.0, 0.0),
                                    (18.5, 31.0, 0.0, 0.0), (58.5, 91.0, 0.0, 0.0)])]),
        "R6": _Road("OpenDRIVE map", "An OpenDRIVE map: straight, a 90 degree left arc, straight (three roads).",
                    [], map_file="data/maps/curve_road.xodr", map_loader="opendrive",
                    route_ids=[1049, 2049, 3049], start=(2.0, 0.0, 0.0), goal=(31.75, 28.0, math.pi / 2)),
    }


def lane_map(road: _Road) -> LaneMap:
    xs, ys, lx, ly, rx, ry, lane_of = [], [], [], [], [], [], []
    for i, c in enumerate(road.lanes):
        d = np.gradient(c, axis=0)
        yaw = np.arctan2(d[:, 1], d[:, 0])
        nx, ny = -np.sin(yaw) * LANE_WIDTH / 2, np.cos(yaw) * LANE_WIDTH / 2
        xs.append(c[:, 0]); ys.append(c[:, 1])                                      # noqa: E702
        lx.append(c[:, 0] + nx); ly.append(c[:, 1] + ny)                            # noqa: E702
        rx.append(c[:, 0] - nx); ry.append(c[:, 1] - ny)                            # noqa: E702
        lane_of.append(np.full(len(c), i))
    rel = np.array(road.successors, dtype=np.int64).reshape(-1, 2)
    m = LaneMap(lane_id=np.arange(len(road.lanes), dtype=np.int64) + 100,
                speed_limit=np.full(len(road.lanes), SPEED_LIMIT),
                point_lane=np.concatenate(lane_of).astype(np.int64), x=np.concatenate(xs), y=np.concatenate(ys),
                left_x=np.concatenate(lx), left_y=np.concatenate(ly), right_x=np.concatenate(rx),
                right_y=np.concatenate(ry), relation_from=rel[:, 0], relation_to=rel[:, 1],
                relation_type=np.full(len(rel), int(LaneRelation.SUCCESSOR), dtype=np.int8))
    m.validate()
    return m


def _vehicle(cfg: dict) -> Tuple[VehicleParameters, VehicleParameters]:
    dyn = cfg["vehicle"]
    names = {f.name for f in dataclasses.fields(VehicleParameters)}
    planner = dataclasses.replace(VehicleParameters(), **{k: v for k, v in dyn.items() if k in names})
    sim = dataclasses.replace(planner, max_steer_angle=planner.max_steer_angle +
                              float(dyn.get("steering_control_margin", 0.0)))
    return planner, sim


class RoadScenarios:
    """Scenario source ``road``. Suites: ``fixed`` / ``all`` (R1-R4)."""

    def suite(self, name: str, n_random: int = 0) -> List[str]:
        if name not in ("fixed", "all") or n_random:
            raise ValueError("road scenarios: suites 'fixed' and 'all', no random scenes")
        return sorted(_roads())

    def build(self, scenario_id: str, seed: int, cfg: dict) -> ScenarioBundle:
        roads = _roads()
        if scenario_id not in roads:
            raise ValueError(f"unknown road scenario {scenario_id!r}; available: {sorted(roads)}")
        r = roads[scenario_id]
        if r.map_file:
            from doraware.maps.registry import load_lane_map
            lanes = load_lane_map(REPO_ROOT / r.map_file, r.map_loader)
            route_ids = np.asarray(r.route_ids, dtype=np.int64)
        else:
            lanes = lane_map(r)
            route_ids = np.asarray(lanes.lane_id)[r.route]
        pts = np.stack([lanes.left_x, lanes.left_y], axis=1).tolist() + \
            np.stack([lanes.right_x, lanes.right_y], axis=1).tolist()
        pts = np.asarray(pts)
        xmin, ymin = pts.min(axis=0) - 3.0
        xmax, ymax = pts.max(axis=0) + 3.0
        res = 0.2
        w, h = int(math.ceil((xmax - xmin) / res)), int(math.ceil((ymax - ymin) / res))
        grid = OccupancyGrid(header=Header(frame_id="map"), resolution=res, width=w, height=h,
                             origin_x=float(xmin), origin_y=float(ymin), data=np.zeros(w * h, dtype=np.uint8))
        planner_vehicle, sim_vehicle = _vehicle(cfg)
        acts = r.actors
        wp = [(k, *p) for k, a in enumerate(acts) for p in a.waypoints]
        scene = SimulationScene(
            header=Header(frame_id="map"), scenario_id=scenario_id, map_xmin=float(xmin), map_xmax=float(xmin + w * res),
            map_ymin=float(ymin), map_ymax=float(ymin + h * res), obstacle_height=1.5,
            initial_pose=Pose2D(*r.start), goal_pose=Pose2D(*r.goal), vehicle=sim_vehicle,
            actor_id=np.arange(len(acts), dtype=np.int64), actor_class=np.array([a.cls for a in acts], dtype=np.int8),
            actor_length=np.array([a.size[0] for a in acts], float), actor_width=np.array([a.size[1] for a in acts], float),
            actor_height=np.array([a.size[2] for a in acts], float),
            waypoint_actor=np.array([p[0] for p in wp], dtype=np.int64), waypoint_time=np.array([p[1] for p in wp], float),
            waypoint_x=np.array([p[2] for p in wp], float), waypoint_y=np.array([p[3] for p in wp], float),
            waypoint_yaw=np.array([p[4] for p in wp], float))
        request = PlanningRequest(header=Header(frame_id="map"), request_id=f"{scenario_id}/0", start=Pose2D(*r.start),
                                  goal=Pose2D(*r.goal), map=grid, vehicle=planner_vehicle, lanes=lanes,
                                  route=Route(lane_ids=route_ids))
        expected = ExpectedBehavior(FEASIBLE, True, max_direction_changes=0)
        return ScenarioBundle(scenario_id, r.name, "road", r.description, scene, request, expected, tags=("road",))
