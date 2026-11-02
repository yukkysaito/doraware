"""Deterministic multi-rate scheduling on the simulation clock.

Sim time only advances in physics steps: ``t(step) = step / physics_hz``.
A component running at ``rate_hz`` has a tick at step ``s`` when
``floor(s * rate / physics) > floor((s - 1) * rate / physics)`` (and at
step 0). Exact rational arithmetic keeps every node in agreement for any
pair of rates (e.g. 100 Hz physics / 30 Hz controller).
"""
from __future__ import annotations

from fractions import Fraction

from doraware.interfaces import NS_PER_S


def _frac(hz: float) -> Fraction:
    return Fraction(hz).limit_denominator(10_000)


class TickSchedule:
    def __init__(self, physics_hz: float, rate_hz: float):
        if rate_hz <= 0 or physics_hz <= 0:
            raise ValueError("rates must be positive")
        self.ratio = _frac(rate_hz) / _frac(physics_hz)   # ticks per physics step
        if self.ratio > 1:
            raise ValueError(f"rate {rate_hz} Hz exceeds the physics rate {physics_hz} Hz")
        self.physics_hz = _frac(physics_hz)

    def _count(self, step: int) -> int:
        return int((step * self.ratio).__floor__())

    def is_tick(self, step: int) -> bool:
        return step == 0 or self._count(step) > self._count(step - 1)

    def last_tick(self, step: int) -> int:
        """Largest tick step <= ``step``."""
        k = self._count(step)
        s = int((Fraction(k) / self.ratio).__ceil__())
        return s

    def next_tick(self, step: int) -> int:
        """Smallest tick step > ``step``."""
        k = self._count(step) + 1
        return int((Fraction(k) / self.ratio).__ceil__())


def step_to_ns(step: int, physics_hz: float) -> int:
    return int(round(Fraction(step) / _frac(physics_hz) * NS_PER_S))


def ns_to_step(ns: int, physics_hz: float) -> int:
    return int(round(Fraction(ns, NS_PER_S) * _frac(physics_hz)))


class RateClock:
    """Ticks of one rate (``timing.<rate_key>``) on the sim clock, queried by
    message stamp [ns]. Nodes use it to decide whether a VehicleState (the
    clock message) is one of their ticks."""

    def __init__(self, timing, rate_key: str):
        self.physics_hz = float(timing["physics_hz"])
        self.rate_hz = float(timing[rate_key])
        self.ticks = TickSchedule(self.physics_hz, self.rate_hz)

    def is_tick(self, t_ns: int) -> bool:
        return self.ticks.is_tick(ns_to_step(t_ns, self.physics_hz))

    def last_tick_ns(self, t_ns: int) -> int:
        """Stamp of the latest tick at or before ``t_ns``."""
        return step_to_ns(self.ticks.last_tick(ns_to_step(t_ns, self.physics_hz)), self.physics_hz)
