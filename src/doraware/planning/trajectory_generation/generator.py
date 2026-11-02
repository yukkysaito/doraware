"""Trajectory generator application logic (called by the Dora node; no Dora import).

    gen = TrajectoryGenerator(cfg["planning"]["trajectory_generation"], cfg["timing"], vehicle)
    gen.on_planning_result(result, grid, plan_start_ns=t)   # once per PlanningResult
    traj = gen.trajectory_at(t_now_ns)                      # at every planner tick

A failed plan, an unusable path or a failed collision revalidation yields an
*empty* Trajectory (= stop) and the reason is kept in ``status`` /
``message`` / ``diagnostics()``.
"""
from __future__ import annotations

import time
from enum import IntEnum
from typing import Any, Dict, Mapping, Optional

import numpy as np

from doraware.interfaces import OccupancyGrid, PlanningResult, PlanningStatus, Trajectory, VehicleParameters
from doraware.planning.trajectory_generation.limits import ProfileLimits
from doraware.planning.trajectory_generation.path_processing import InvalidPathError
from doraware.planning.trajectory_generation.profile import TimedProfile, build_profile
from doraware.planning.trajectory_generation.stop_point import snap_stop_to_goal
from doraware.planning.trajectory_generation.revalidation import RevalidationResult, revalidate
from doraware.planning.trajectory_generation.window import empty_trajectory, make_trajectory


REVALIDATION_DS = 0.005   # [m] pose spacing of the revalidation (sweep bound ~ 5 mm)


def densify_poses(poses: np.ndarray, ds: float) -> np.ndarray:
    """Linear interpolation (yaw on the circle) so consecutive poses are <= ds apart;
    the original poses are kept."""
    p = np.asarray(poses, dtype=float)
    if len(p) < 2:
        return p
    out = [p[:1]]
    for a, b in zip(p[:-1], p[1:]):
        n = max(1, int(np.ceil(np.hypot(b[0] - a[0], b[1] - a[1]) / ds)))
        w = np.arange(1, n + 1)[:, None] / n
        dyaw = (b[2] - a[2] + np.pi) % (2 * np.pi) - np.pi
        out.append(np.column_stack([a[0] + w[:, 0] * (b[0] - a[0]), a[1] + w[:, 0] * (b[1] - a[1]),
                                    a[2] + w[:, 0] * dyaw]))
    return np.vstack(out)


def pose_curvature_bound(poses: np.ndarray) -> float:
    """max |d yaw| / ds between consecutive poses: the yaw rate per distance that the
    linear interpolation between them actually has (sound sweep bound, no reliance on
    the curvature array being consistent with the poses)."""
    p = np.asarray(poses, dtype=float)
    if len(p) < 2:
        return 0.0
    ds = np.hypot(np.diff(p[:, 0]), np.diff(p[:, 1]))
    dyaw = np.abs((np.diff(p[:, 2]) + np.pi) % (2 * np.pi) - np.pi)
    ok = ds > 1e-9
    return float(np.max(dyaw[ok] / ds[ok])) if ok.any() else 0.0


class TrajectoryStatus(IntEnum):
    WAITING = 0                 # no PlanningResult received yet
    OK = 1
    PLANNING_FAILED = 2         # PlanningResult status != SUCCESS (see planning_status)
    INVALID_PATH = 3            # the path could not be time-parameterised
    REVALIDATION_FAILED = 4     # trajectory footprint collides with the map


class TrajectoryGenerator:
    def __init__(self, generator_config: Mapping[str, Any], timing_config: Mapping[str, Any],
                 vehicle: VehicleParameters):
        self.limits = ProfileLimits.from_config(generator_config, vehicle)
        self.limits.validate()
        self.vehicle = vehicle
        self.time_step = float(timing_config.get("trajectory_time_step", 0.05))
        self.horizon = float(timing_config.get("trajectory_horizon", 5.0))
        self.validity = float(timing_config.get("trajectory_validity", 1.0))
        self.profile: Optional[TimedProfile] = None
        self.plan_start_ns = 0
        self.status = TrajectoryStatus.WAITING
        self.planning_status = PlanningStatus.UNKNOWN
        self.message = "waiting for a planning result"
        self.revalidation: Optional[RevalidationResult] = None
        self.generation_time = 0.0
        self._seq = 0
        self.snap_cfg = dict(generator_config.get("goal_stop_snapping") or {})
        self.stop_point = None

    # ------------------------------------------------------------ input
    def on_planning_result(self, result: PlanningResult, grid: Optional[OccupancyGrid], plan_start_ns: int = 0,
                           initial_steering: Optional[float] = None) -> TrajectoryStatus:
        """Build the profile of ``result`` starting at sim time ``plan_start_ns``.
        ``grid=None`` skips the collision revalidation (reported in diagnostics)."""
        t0 = time.perf_counter()
        self.profile, self.revalidation = None, None
        self.plan_start_ns = int(plan_start_ns)
        self.planning_status = result.status
        if result.status != PlanningStatus.SUCCESS:
            return self._set(TrajectoryStatus.PLANNING_FAILED,
                             f"planning failed ({result.status.name}): {result.message}", t0)
        path = result.path
        self.stop_point = None
        if self.snap_cfg.get("enabled", True):
            self.stop_point = snap_stop_to_goal(path, result.goal, grid, self.vehicle,
                                                max_shift=float(self.snap_cfg.get("max_shift", 0.15)),
                                                margin=float(self.snap_cfg.get("margin", 0.1)))
            path = self.stop_point.path
        try:
            profile = build_profile(path, self.vehicle.wheel_base, self.vehicle.max_steer_angle,
                                    self.limits, initial_steering)
        except (InvalidPathError, ValueError, RuntimeError) as exc:
            return self._set(TrajectoryStatus.INVALID_PATH, f"cannot time-parameterise the path: {exc}", t0)
        if grid is not None:
            dense = [densify_poses(s.poses(), REVALIDATION_DS) for s in profile.segments]
            k_max = max((pose_curvature_bound(p) for p in dense), default=0.0)
            self.revalidation = revalidate(dense, grid, self.vehicle, self.limits.revalidation_margin, k_max)
            if not self.revalidation.ok:
                return self._set(TrajectoryStatus.REVALIDATION_FAILED,
                                 f"collision revalidation failed: {self.revalidation.message}", t0)
        self.profile = profile
        return self._set(TrajectoryStatus.OK, "ok" if grid is not None else "ok (revalidation skipped: no map)", t0)

    # ----------------------------------------------------------- output
    def trajectory_at(self, t_now_ns: int) -> Optional[Trajectory]:
        """Window starting at ``t_now_ns``; None before any result; empty (= stop) after a failure."""
        if self.status == TrajectoryStatus.WAITING:
            return None
        self._seq += 1
        if self.profile is None:
            return empty_trajectory(t_now_ns, self.time_step, self.validity, self._seq)
        return make_trajectory(self.profile, self.plan_start_ns, t_now_ns, self.time_step, self.horizon,
                               self.validity, self._seq)

    @property
    def has_result(self) -> bool:
        """True once a PlanningResult has been applied (successfully or not)."""
        return self.status != TrajectoryStatus.WAITING

    @property
    def ok(self) -> bool:
        return self.status == TrajectoryStatus.OK

    def finished(self, t_now_ns: int) -> bool:
        """True once the profile (incl. the final stop) is over."""
        return self.profile is not None and (t_now_ns - self.plan_start_ns) * 1e-9 >= self.profile.duration

    def status_report(self) -> Dict[str, Any]:
        """The ``trajectory_status`` diagnostics values: :meth:`diagnostics` plus
        ``trajectory_ok`` and, with a profile, ``plan_start_time`` / ``plan_duration``
        [s, sim] (the evaluator ends the run after the plan)."""
        info = self.diagnostics()
        info["trajectory_ok"] = self.ok
        if self.profile is not None:
            info["plan_start_time"] = self.plan_start_ns * 1e-9
            info["plan_duration"] = self.profile.duration
        return info

    @property
    def unexpected_failure(self) -> bool:
        """The planner succeeded but no trajectory could be made from its path."""
        return not self.ok and self.planning_status == PlanningStatus.SUCCESS

    def diagnostics(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"status": self.status.name, "planning_status": self.planning_status.name,
                             "message": self.message, "generation_time": self.generation_time}
        if self.profile is not None:
            d.update(self.profile.summary())
        if self.stop_point is not None:
            d.update(stop_point_snapped=self.stop_point.applied, stop_point_shift=self.stop_point.shift,
                     stop_point_note=self.stop_point.reason)
        if self.revalidation is not None:
            d.update(revalidation_ok=self.revalidation.ok, revalidation_margin=self.revalidation.check_margin,
                     revalidated_poses=self.revalidation.n_checked)
        return d

    def _set(self, status: TrajectoryStatus, message: str, t0: float) -> TrajectoryStatus:
        self.status, self.message = status, message
        self.generation_time = time.perf_counter() - t0
        return status
