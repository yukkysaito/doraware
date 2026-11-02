#!/usr/bin/env bash
# Full evaluation: unit/integration tests, planner tests, all fixed scenarios
# and 100 seeded random scenarios in closed loop.
#   WORKERS=6 ./scripts/test_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
WORKERS="${WORKERS:-6}"
OUT="${OUT:-results/runs/all_$(date +%Y%m%d_%H%M%S)}"
env -u PYTHONPATH python -m pytest tests -q -m "not e2e"
(cd third_party/state_lattice_based_valet_parking && env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
   python -m pytest tests -q -p no:cacheprovider)
env -u PYTHONPATH doraware run --suite all --random 100 --workers "$WORKERS" --out "$OUT" \
   --no-record "$@"
