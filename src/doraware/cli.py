"""``doraware`` command line entry point.

    doraware run --scenario B2 [--gui] [--set key.path=value ...]
    doraware run --suite all --random 100 --workers 8
    doraware regression [--update-baseline]
    doraware replay states|commands <run_dir>

Each subcommand forwards its remaining arguments to the module's own ``main``
(``--help`` works per subcommand).
"""
from __future__ import annotations

import importlib
import sys

COMMANDS = {
    "run": ("doraware.evaluation.suite.runner", "run scenarios in closed loop (single, list or suites)"),
    "regression": ("doraware.evaluation.suite.regression", "closed-loop regression vs. tests/regression/baseline.json"),
    "replay": ("doraware.recording.replay", "replay a recorded run (state playback / command re-simulation)"),
}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in COMMANDS:
        print("usage: doraware {" + ",".join(COMMANDS) + "} [args...]\n")
        for name, (_, desc) in COMMANDS.items():
            print(f"  {name:<11} {desc}")
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    module, _ = COMMANDS[argv[0]]
    return int(importlib.import_module(module).main(argv[1:]) or 0)


if __name__ == "__main__":
    sys.exit(main())
