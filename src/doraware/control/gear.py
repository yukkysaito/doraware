"""Gear manager (docs/mpc.md §8).

The Trajectory carries the direction explicitly (``gear`` per point). The
desired gear at sim time t is the gear of the reference at t if the reference
moves there, otherwise the gear of the *next moving* reference point (so the
shift happens during the stop at a cusp, before the reference departs). If
nothing moves any more (final stop) the gear at t is kept.

Shift sequence when the vehicle gear differs from the desired gear:
  1. STOPPING  |v| >= gear_change_speed: the vehicle is brought to a stop
               (MPC towards the cusp-clamped reference, or a plain brake).
  2. SHIFTING  |v| <  gear_change_speed: the gear is requested; braking is
               released to zero at standstill (jerk-limited) so that the
               acceleration sign convention can flip without a propulsion
               pulse; the steering keeps tracking the reference.
  3. done      once the vehicle *reports* the new gear, MPC tracking resumes.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from doraware.interfaces import Gear, Trajectory, to_sec
from doraware.core.trajectory import STATIONARY_EPS, sample


class GearPhase(str, Enum):
    ENGAGED = "engaged"
    STOPPING = "stopping"
    SHIFTING = "shifting"


@dataclass
class GearDecision:
    phase: GearPhase
    desired: Gear
    request: Gear            # value for ControlCommand.gear_request


class GearManager:
    def __init__(self, gear_change_speed: float):
        self.gear_change_speed = gear_change_speed

    @staticmethod
    def desired_gear(traj: Trajectory, t: float) -> Gear:
        p = sample(traj, t)
        if p is None:
            return Gear.NONE
        if abs(p.velocity) >= STATIONARY_EPS and p.gear.direction != 0:
            return p.gear
        # stationary now: look for the next moving point ahead
        i0 = max(int(np.ceil((t - to_sec(traj.start_time)) / traj.time_step)), 0)
        v = np.asarray(traj.longitudinal_velocity[i0:])
        moving = np.nonzero(np.abs(v) >= STATIONARY_EPS)[0]
        for j in moving:
            g = Gear(int(traj.gear[i0 + j]))
            if g.direction != 0:
                return g
        return p.gear if p.gear.direction != 0 else Gear.NONE

    def decide(self, traj: Trajectory, t: float, vehicle_gear: Gear, speed: float) -> GearDecision:
        desired = self.desired_gear(traj, t)
        if desired == Gear.NONE or desired == vehicle_gear:
            return GearDecision(GearPhase.ENGAGED, desired, Gear.NONE)
        if abs(speed) >= self.gear_change_speed:
            return GearDecision(GearPhase.STOPPING, desired, Gear.NONE)
        return GearDecision(GearPhase.SHIFTING, desired, desired)
