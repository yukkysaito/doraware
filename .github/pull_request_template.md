## What and why

## How it was verified
<!-- Which checks ran, with their results. Closed-loop changes: e2e and regression; release-relevant
changes: the full suite (scripts/test_all.sh) pass counts. -->

- [ ] `ruff check .` and `pytest tests/unit`
- [ ] `pytest tests/integration` (Godot) if the simulator or the GUI changed
- [ ] `pytest tests/e2e -m e2e` and `./scripts/test_regression.sh` if driving can change
- [ ] Interface (message) changes are described here and in docs/architecture.md §4
- [ ] New parameters are in configs/default.yaml with units; docs are updated
