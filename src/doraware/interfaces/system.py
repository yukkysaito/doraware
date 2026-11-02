"""System messages: diagnostics and scenario lifecycle."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from doraware.interfaces.common import Header


class DiagnosticLevel(IntEnum):
    OK = 0
    WARN = 1
    ERROR = 2
    STALE = 3


# ======================================================== diagnostics
@dataclass
class Diagnostics:
    header: Header = field(default_factory=Header)
    node: str = ""
    level: DiagnosticLevel = DiagnosticLevel.OK
    message: str = ""
    values: str = "{}"                      # JSON object of key -> number / string


@dataclass
class ScenarioDone:
    """Published by the evaluator when a run reaches a terminal condition."""
    header: Header = field(default_factory=Header)
    scenario_id: str = ""
    outcome: str = ""                       # "goal_reached" | "rejected" | "collision" | "timeout" | ...
    passed: bool = False
    summary: str = "{}"                     # JSON metrics


@dataclass
class ScenarioInfo:
    """Scenario identity and expected behaviour (evaluation only, JSON payload)."""
    header: Header = field(default_factory=Header)
    scenario_id: str = ""
    description: str = "{}"                 # JSON: name, category, class, expectation, ...
