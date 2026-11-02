#!/usr/bin/env bash
# One-time setup: Python env (pinned), planner submodule, Godot binary check,
# Godot project import and planner heuristic-table warm-up.
set -euo pipefail
cd "$(dirname "$0")/.."
GODOT_VERSION=4.7.2-stable

git submodule update --init --recursive
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/"; exit 1; }
[ -d .venv ] || uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -e ".[dev]"

mkdir -p tools
if [ ! -x tools/godot ]; then
  if [ -n "${GODOT_BIN:-}" ]; then
    ln -sf "$GODOT_BIN" tools/godot
  elif [ "${1:-}" = "--download-godot" ]; then
    url="https://github.com/godotengine/godot/releases/download/${GODOT_VERSION}/Godot_v${GODOT_VERSION}_linux.x86_64.zip"
    echo "downloading $url"
    curl -L -o tools/godot.zip "$url" && (cd tools && unzip -o godot.zip && rm godot.zip)
    ln -sf "Godot_v${GODOT_VERSION}_linux.x86_64" tools/godot
  else
    echo "Godot ${GODOT_VERSION} not found. Either export GODOT_BIN=/path/to/Godot_v${GODOT_VERSION}_linux.x86_64"
    echo "or rerun: ./scripts/setup.sh --download-godot"
    exit 1
  fi
fi
tools/godot --version
tools/godot --headless --path simulator/godot --import >/dev/null 2>&1 || true

# build the planner's heuristic tables once (cached in the submodule's .cache/)
(cd third_party/state_lattice_based_valet_parking && env -u PYTHONPATH python run_tests.py --regression --no-plots \
   --out ../../results/tmp/planner_warmup >/dev/null)
echo "setup complete"
