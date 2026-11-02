"""Third-party code vendored as git submodules under ``third_party/``.

The State Lattice planner (``third_party/state_lattice_based_valet_parking``)
is a flat collection of top-level modules (``state_lattice``, ``vehicle``,
``trajectory``, ...) that import each other by bare name, so its directory
has to be on ``sys.path``. Two packages use it, the planner plugin and the
parking scenario source (its test patterns), so the locator lives here in
``core`` rather than in either of them.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from doraware.core.config import REPO_ROOT
PLANNER_DIR = Path(os.environ.get("DORAWARE_PLANNER_DIR",
                                  REPO_ROOT / "third_party" / "state_lattice_based_valet_parking"))


def ensure_planner_importable() -> Path:
    """Put the planner directory first on ``sys.path`` (idempotent)."""
    if not (PLANNER_DIR / "state_lattice.py").is_file():
        raise ImportError(f"State Lattice planner not found in {PLANNER_DIR} "
                          "(git submodule update --init?)")
    p = str(PLANNER_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
    return PLANNER_DIR
