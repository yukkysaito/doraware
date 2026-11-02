"""Turn-indicator and hazard-light decision (no Dora).

The decision is planner-agnostic: it uses only the executable
:class:`Trajectory` (whatever produced it) and the measured
:class:`VehicleState`.

Two policies (config ``planning.signals.policy``):

* ``maneuver_hazard`` (default; low-speed manoeuvres such as parking): the
  hazard lights are on for the whole manoeuvre, i.e. while the trajectory has
  an upcoming motion (also while stopped at a gear change or before
  departure), and the turn indicator stays OFF. When the plan is complete
  (nothing moves any more), the hazard lights go off.
* ``turn_indicators`` (driving among traffic): the turn indicator shows the
  upcoming turn as described below; hazard lights only on failures.

In both policies the hazard lights are on while the vehicle has no plan to follow:

* ``no_trajectory``: no trajectory has been received yet,
* ``empty_trajectory``: the newest trajectory is empty (planning or trajectory
  generation failed, so the vehicle stops),
* ``expired_trajectory``: the newest trajectory is past ``valid_until`` (the
  planning pipeline went silent).

With ``turn_indicators``, the turn indicator shows the side to which the vehicle is about to move. The
*upcoming motion* is the part of the trajectory from now on that belongs to the
current motion. While the vehicle is stopped before departure or at a gear
change, it is the next motion. It ends at the next gear change and is at most
``lookahead_distance`` of arc length. The lateral offset ``y`` of its end point
in the current base_link frame selects LEFT (``y > activation_offset``) or
RIGHT (``y < -activation_offset``). An active indicator stays on until
``|y| <= deactivation_offset`` (hysteresis). In reverse this is still the side
the vehicle body moves to. With no upcoming motion the indicator is OFF.

Turning the hazard lights on is immediate. Every other change of a signal waits
until the signal has kept its state for ``min_hold_time``, so the lamps do not
flicker.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np

from doraware.core.trajectory import STATIONARY_EPS
from doraware.interfaces import Trajectory, TurnIndicator, VehicleState, to_sec


class SignalPolicy(str, Enum):
    MANEUVER_HAZARD = "maneuver_hazard"
    TURN_INDICATORS = "turn_indicators"


@dataclass
class SignalParams:
    policy: SignalPolicy = SignalPolicy.MANEUVER_HAZARD
    lookahead_distance: float = 6.0     # [m] arc length of the upcoming motion considered
    activation_offset: float = 0.5      # [m] |lateral offset| that switches an indicator on
    deactivation_offset: float = 0.2    # [m] an active indicator switches off at or below this
    min_hold_time: float = 1.0          # [s, sim] minimum time between changes of one signal

    @staticmethod
    def from_config(section: Dict[str, Any]) -> "SignalParams":
        """``section``: the ``planning.signals`` config section."""
        raw = dict(section or {})
        policy = SignalPolicy(raw.pop("policy", SignalPolicy.MANEUVER_HAZARD))   # ValueError if unknown
        p = SignalParams(policy=policy, **{k: float(v) for k, v in raw.items()})
        if not 0.0 <= p.deactivation_offset < p.activation_offset:
            raise ValueError("planning.signals: need 0 <= deactivation_offset < activation_offset")
        if p.lookahead_distance <= 0.0 or p.min_hold_time < 0.0:
            raise ValueError("planning.signals: lookahead_distance must be > 0 and min_hold_time >= 0")
        return p


@dataclass
class SignalDecision:
    turn_indicator: TurnIndicator
    hazard_lights: bool
    hazard_reason: str = ""                 # "" or why the hazard lights are requested
    lateral_offset: Optional[float] = None  # [m] of the upcoming motion's end point; None = no motion


def upcoming_motion_end(traj: Trajectory, t: float, lookahead: float) -> Optional[int]:
    """Index of the end point of the upcoming motion at sim time ``t`` [s]
    (see the module docstring), or None if nothing moves from ``t`` on."""
    n = len(traj)
    if n == 0:
        return None
    i0 = min(max(int(math.ceil((t - to_sec(traj.start_time)) / traj.time_step - 1e-9)), 0), n - 1)
    moving = np.nonzero(np.abs(np.asarray(traj.longitudinal_velocity[i0:])) >= STATIONARY_EPS)[0]
    if len(moving) == 0:
        return None
    first = i0 + int(moving[0])
    gear = int(traj.gear[first])
    end, dist = first, 0.0
    for k in range(i0 + 1, n):
        if k > first and int(traj.gear[k]) != gear:
            break
        step = math.hypot(float(traj.x[k] - traj.x[k - 1]), float(traj.y[k] - traj.y[k - 1]))
        if dist + step > lookahead:
            break
        dist += step
        end = k
    return end


class SignalDecider:
    def __init__(self, params: SignalParams):
        self.p = params
        self.trajectory: Optional[Trajectory] = None
        self.turn = TurnIndicator.OFF
        self.hazard = False
        self._turn_since = -math.inf
        self._hazard_since = -math.inf

    def on_trajectory(self, traj: Trajectory) -> bool:
        """Keep the newest trajectory (by header stamp). False if it was ignored."""
        if self.trajectory is not None and traj.header.stamp <= self.trajectory.header.stamp:
            return False
        self.trajectory = traj
        return True

    def hazard_reason(self, t_ns: int) -> str:
        tr = self.trajectory
        if tr is None:
            return "no_trajectory"
        if t_ns > tr.valid_until:
            return "expired_trajectory"
        if len(tr) == 0:
            return "empty_trajectory"
        return ""

    def lateral_offset(self, state: VehicleState, t: float) -> Optional[float]:
        tr = self.trajectory
        if tr is None or len(tr) == 0:
            return None
        end = upcoming_motion_end(tr, t, self.p.lookahead_distance)
        if end is None:
            return None
        x, y, yaw = state.pose.xyyaw()
        dx, dy = float(tr.x[end]) - x, float(tr.y[end]) - y
        return -math.sin(yaw) * dx + math.cos(yaw) * dy

    def _turn_for(self, y: Optional[float]) -> TurnIndicator:
        if y is None:
            return TurnIndicator.OFF
        if y > self.p.activation_offset or (self.turn == TurnIndicator.LEFT and y > self.p.deactivation_offset):
            return TurnIndicator.LEFT
        if y < -self.p.activation_offset or (self.turn == TurnIndicator.RIGHT and y < -self.p.deactivation_offset):
            return TurnIndicator.RIGHT
        return TurnIndicator.OFF

    def step(self, state: VehicleState, t_ns: int) -> SignalDecision:
        t = to_sec(t_ns)
        reason = self.hazard_reason(t_ns)
        y = None if reason == "expired_trajectory" else self.lateral_offset(state, t)
        maneuver = self.p.policy == SignalPolicy.MANEUVER_HAZARD
        if maneuver and not reason and y is not None:
            reason = "maneuvering"          # an upcoming motion exists: the manoeuvre is in progress
        want_hazard = reason != ""
        if want_hazard != self.hazard and (want_hazard or t - self._hazard_since >= self.p.min_hold_time):
            self.hazard, self._hazard_since = want_hazard, t
        want_turn = TurnIndicator.OFF if maneuver else self._turn_for(y)
        if want_turn != self.turn and t - self._turn_since >= self.p.min_hold_time:
            self.turn, self._turn_since = want_turn, t
        return SignalDecision(self.turn, self.hazard, reason, y)
