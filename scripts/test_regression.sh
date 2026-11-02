#!/usr/bin/env bash
# Regression: unit + integration tests, planner regression, and the closed-loop
# regression scenario set compared against tests/regression/baseline.json.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
env -u PYTHONPATH python -m pytest tests -q -m "not e2e"
(cd third_party/state_lattice_based_valet_parking && env -u PYTHONPATH python run_tests.py --regression --no-plots \
   --out ../../results/tmp/planner_regression)
env -u PYTHONPATH doraware regression "$@"
