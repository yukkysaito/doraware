"""Step-based simulation coordinator (pure logic, see architecture §6).

Decides, per physics step, whether the bridge must publish a state, wait for a
control command, or send visualization, and how many physics steps can be
executed in one chunk before the next such event."""
from __future__ import annotations

from typing import Dict

from doraware.core.clock import TickSchedule, step_to_ns


class StepCoordinator:
    def __init__(self, timing: Dict):
        self.physics_hz = float(timing["physics_hz"])
        self.controller = TickSchedule(self.physics_hz, float(timing["controller_hz"]))
        self.planner = TickSchedule(self.physics_hz, float(timing["planner_hz"]))
        self.state = TickSchedule(self.physics_hz, float(timing["state_hz"]))
        self.viz = TickSchedule(self.physics_hz, float(timing["visualization_hz"]))
        self.dt = 1.0 / self.physics_hz

    def time_ns(self, step: int) -> int:
        return step_to_ns(step, self.physics_hz)

    def needs_command(self, step: int) -> bool:
        return self.controller.is_tick(step)

    def publishes_state(self, step: int) -> bool:
        """Every tick some node depends on gets a state (= clock) message."""
        return self.state.is_tick(step) or self.controller.is_tick(step) or self.planner.is_tick(step)

    def sends_viz(self, step: int) -> bool:
        return self.viz.is_tick(step)

    def chunk(self, step: int) -> int:
        """Physics steps until the next step with any event (>= 1)."""
        nxt = min(s.next_tick(step) for s in (self.controller, self.planner, self.state, self.viz))
        return max(1, nxt - step)
