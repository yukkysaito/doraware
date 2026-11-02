"""Closed-loop regression check.

    doraware regression [--update-baseline] [--workers 6]

Runs the regression scenario set and compares each result with
``tests/regression/baseline.json``: a scenario that passed must still pass,
and its goal / tracking errors must not grow beyond the tolerances below
(deterministic lockstep runs reproduce exactly; the tolerances only absorb
intentional, reviewed controller changes)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from doraware.core.config import REPO_ROOT
from doraware.evaluation.suite.report import write_reports
from doraware.evaluation.suite.runner import run_suite

REGRESSION_IDS = ["B2", "B3", "C2", "D3", "E5", "D2", "A3", "A5", "C1", "E1", "F1", "G1", "G4", "G6", "H3", "M1-L"]
BASELINE = REPO_ROOT / "tests" / "regression" / "baseline.json"
TOLERANCES = {"final_position_error": 0.03, "final_yaw_error_deg": 0.5, "lateral_error_max": 0.05}


def compare(results, baseline):
    problems = {}
    for r in results:
        sid = r["scenario_id"]
        b = baseline.get(sid)
        p = []
        if b is None:
            p.append("no baseline")
        elif b["passed"] and not r.get("passed"):
            p.append("was passing, now fails: " + "; ".join(r.get("reasons", [])))
        elif b["passed"]:
            m = r.get("metrics") or {}
            for k, tol in TOLERANCES.items():
                if k in m and k in b["metrics"] and m[k] > b["metrics"][k] + tol:
                    p.append(f"{k} {m[k]:.3f} > baseline {b['metrics'][k]:.3f} + {tol}")
        if p:
            problems[sid] = p
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="doraware regression", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out or REPO_ROOT / "results" / "runs" / f"regression_{time.strftime('%Y%m%d_%H%M%S')}")
    results = run_suite(REGRESSION_IDS, out, [], a.workers, 42, False)
    write_reports(results, out)
    if a.update_baseline:
        data = {r["scenario_id"]: {"passed": bool(r.get("passed")), "outcome": r.get("outcome"),
                                   "metrics": {k: (r.get("metrics") or {}).get(k) for k in TOLERANCES}}
                for r in results}
        BASELINE.write_text(json.dumps(data, indent=2))
        print(f"baseline written: {BASELINE}")
        return 0
    if not BASELINE.exists():
        print("no baseline; run with --update-baseline")
        return 2
    problems = compare(results, json.loads(BASELINE.read_text()))
    for sid, p in problems.items():
        print(f"REGRESSION {sid}: {'; '.join(p)}")
    print("CLOSED-LOOP REGRESSION " + ("PASSED" if not problems else "FAILED"))
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
