"""Thin Dora adapter shared by all nodes: typed send/receive, sequence ids,
diagnostics, input decoding with schema checks, barrier waits with wall-clock
timeouts, and graceful shutdown. Algorithms never import this module."""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import defaultdict, deque
from typing import Any, Callable, Deque, Dict, Iterator, Optional, Tuple, Type

from dora import Node

from doraware.core.config import load_config
from doraware.interfaces import codec
from doraware.interfaces import ClockDomain, DiagnosticLevel, Diagnostics, Header

Event = Dict[str, Any]


def setup_logging(name: str) -> logging.Logger:
    logging.basicConfig(stream=sys.stderr, level=os.environ.get("DORAWARE_LOG", "INFO"),
                        format=f"%(asctime)s {name} %(levelname)s %(message)s")
    return logging.getLogger(name)


class NodeRuntime:
    """Wraps :class:`dora.Node`. ``name`` is the node id in the dataflow.
    ``inputs`` maps a Dora input id to the message class it carries; decoding
    errors are reported, not hidden."""

    def __init__(self, name: str, inputs: Dict[str, Type]):
        self.name = name
        self.log = setup_logging(name)
        self.cfg = load_config()
        self.node = Node()
        self.inputs = inputs
        self._seq: Dict[str, int] = defaultdict(int)
        self._pending: Deque[Event] = deque()
        self.closed = False
        self.run_dir = os.environ.get("DORAWARE_RUN_DIR", "")

    # ------------------------------------------------------------- output
    def header(self, stamp_ns: int, output: str, frame_id: str = "map",
               clock: ClockDomain = ClockDomain.SIM) -> Header:
        self._seq[output] += 1
        return Header(stamp=int(stamp_ns), frame_id=frame_id, sequence_id=self._seq[output], clock=clock)

    def send(self, output: str, msg) -> None:
        meta = codec.metadata(msg)
        meta["wall_send_ns"] = time.time_ns()
        self.node.send_output(output, codec.encode(msg), meta)

    def diagnose(self, stamp_ns: int, level: DiagnosticLevel, message: str, output: str = "diagnostics",
                 **values) -> None:
        if level >= DiagnosticLevel.WARN:
            self.log.warning("%s %s", message, values or "")
        try:
            self.send(output, Diagnostics(header=self.header(stamp_ns, output), node=self.name, level=level,
                                          message=message, values=json.dumps(values, default=str)))
        except Exception as exc:  # diagnostics must never take the node down
            self.log.error("cannot send diagnostics: %s", exc)

    # -------------------------------------------------------------- input
    def input_connected(self, input_id: str) -> bool:
        """True if this node's ``input_id`` is wired in the running dataflow.
        Raises if the dataflow descriptor cannot be read (never guesses)."""
        nodes = self.node.dataflow_descriptor().get("nodes") or []
        mine = [n for n in nodes if n.get("id") == self.name]
        if len(mine) != 1:
            raise RuntimeError(f"node {self.name!r} not found in the dataflow descriptor")
        return input_id in (mine[0].get("inputs") or {})

    def decode(self, event: Event):
        cls = self.inputs[event["id"]]
        codec.check_metadata(event.get("metadata") or {}, cls)
        return codec.decode(event["value"], cls)

    def next(self, timeout: Optional[float] = None) -> Optional[Event]:
        """Next event (pending first). Returns ``{"type": "TIMEOUT"}`` on timeout
        and ``None`` once the stream is closed."""
        if self._pending:
            return self._pending.popleft()
        if self.closed:
            return None
        ev = self.node.next(timeout=timeout) if timeout is not None else self.node.next()
        if ev is None:
            self.closed = True
            return None
        if ev["type"] == "ERROR" and "imeout" in str(ev.get("error", "")):
            return {"type": "TIMEOUT"}
        if ev["type"] == "STOP":
            self.closed = True
        return ev

    def events(self) -> Iterator[Event]:
        while True:
            ev = self.next()
            if ev is None:
                return
            yield ev
            if ev["type"] == "STOP":
                return

    def wait_for(self, predicate: Callable[[Event], bool], timeout: float) -> Tuple[Optional[Event], bool]:
        """Block until an INPUT event satisfies ``predicate``. Other events are
        kept (in order) for later processing. Returns (event, timed_out)."""
        deadline = time.monotonic() + timeout
        stash = []
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None, True
                if self._pending:
                    ev = self._pending.popleft()
                else:
                    ev = self.node.next(timeout=remaining) if not self.closed else None
                    if ev is None:
                        self.closed = True
                        return None, False
                    if ev["type"] == "ERROR" and "imeout" in str(ev.get("error", "")):
                        return None, True
                if ev["type"] == "INPUT" and predicate(ev):
                    return ev, False
                stash.append(ev)
                if ev["type"] == "STOP":
                    self.closed = True
                    return None, False
        finally:
            # put back what we skipped, ahead of anything still pending
            self._pending.extendleft(reversed(stash))

    def write_json(self, filename: str, data: Any) -> Optional[str]:
        if not self.run_dir:
            return None
        path = os.path.join(self.run_dir, filename)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        return path
