"""Lockstep barrier on one input (docs/architecture.md §6).

A consumer that runs on the sim clock must not act on tick ``t`` before the
producer's message for ``t`` has arrived, or its output would depend on
message arrival order. ``InputBarrier.wait(t)`` blocks until a message
stamped at or after ``t`` has been received on the input.

* The first message may take long (it waits for planning, which is not sim
  time): ``timing.planning_wall_timeout``. Later waits use
  ``timing.lockstep_wall_timeout``.
* On a timeout the barrier reports a Diagnostics error once and stops waiting
  for good ("starved"): the node continues with what it has, and the expired
  data leads to a safe reaction downstream.
* Without ``timing.lockstep`` the barrier never waits.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from doraware.core.node import Event, NodeRuntime
from doraware.interfaces import DiagnosticLevel


class InputBarrier:
    def __init__(self, rt: NodeRuntime, input_id: str, on_message: Callable[[Any], None],
                 timing: Optional[dict] = None):
        timing = timing if timing is not None else rt.cfg["timing"]
        self.rt = rt
        self.input_id = input_id
        self.on_message = on_message
        self.enabled = bool(timing["lockstep"])
        self.first_timeout = float(timing["planning_wall_timeout"])
        self.timeout = float(timing["lockstep_wall_timeout"])
        self.latest_stamp = -1
        self.starved = False

    def accepts(self, ev: Event) -> bool:
        return ev.get("type") == "INPUT" and ev.get("id") == self.input_id

    def take(self, ev: Event) -> Any:
        """Decode an event of this input, record its stamp and hand it to ``on_message``."""
        msg = self.rt.decode(ev)
        self.latest_stamp = max(self.latest_stamp, int(msg.header.stamp))
        self.on_message(msg)
        return msg

    def wait(self, need_ns: int, now_ns: int) -> bool:
        """Wait until a message stamped >= ``need_ns`` was taken. False if starved."""
        if not self.enabled:
            return True
        while not self.starved and self.latest_stamp < need_ns:
            first = self.latest_stamp < 0
            got, _ = self.rt.wait_for(self.accepts, self.first_timeout if first else self.timeout)
            if got is None:
                self.starved = True
                self.rt.diagnose(now_ns, DiagnosticLevel.ERROR, f"{self.input_id} barrier timeout: producer silent",
                                 required_stamp=need_ns)
                break
            self.take(got)
        return not self.starved
