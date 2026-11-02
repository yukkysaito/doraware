"""Scenario sources, selected by ``evaluation.scenario_source``.

A scenario source turns scenario ids into :class:`ScenarioBundle` objects and
names suites of ids. Built-in sources live in ``evaluation/scenarios/``;
separately installed packages register a factory ``() -> ScenarioSource`` under
the ``doraware.scenario_sources`` entry-point group.
"""
from __future__ import annotations

from typing import Any, Dict, List, Protocol

from doraware.core.registry import Registry
from doraware.evaluation.scenarios.bundle import ScenarioBundle


class ScenarioSource(Protocol):
    def suite(self, name: str, n_random: int = 0) -> List[str]:
        """Scenario ids of a named suite (``n_random`` adds that many seeded random scenes)."""

    def build(self, scenario_id: str, seed: int, cfg: Dict[str, Any]) -> ScenarioBundle:
        """The bundle of one scenario; ``cfg`` is the full system config."""


SOURCES: Registry[ScenarioSource] = Registry("scenario source", "doraware.scenario_sources")


@SOURCES.register("parking")
def _parking() -> ScenarioSource:
    from doraware.evaluation.scenarios.parking import ParkingScenarios
    return ParkingScenarios()


def make_scenario_source(cfg: Dict[str, Any]) -> ScenarioSource:
    return SOURCES.create(cfg["evaluation"]["scenario_source"])
