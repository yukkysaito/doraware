"""Aggregate closed-loop results into summary.json / summary.md."""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

METRIC_KEYS = ["final_position_error", "final_yaw_error_deg", "final_speed", "path_end_position_error",
               "path_end_yaw_error_deg", "planned_goal_position_error", "lateral_error_max", "lateral_error_rms",
               "heading_error_rad_max", "cross_track_error_max", "velocity_error_rms", "min_clearance_actual",
               "min_clearance_planned", "max_jerk", "max_steering_rate", "cmd_max_jerk", "cmd_max_steering_rate",
               "mpc_solve_time_mean_ms", "mpc_solve_time_p95_ms", "mpc_solver_success_rate", "controller_frequency",
               "sim_duration"]


def failure_cause(r: Dict) -> str:
    """Coarse root-cause class of a failed run, used to group failures in reports."""
    reasons = " ".join(r.get("reasons", [])).lower()
    outcome = r.get("outcome", "")
    m = r.get("metrics") or {}
    if outcome in ("no_result", "aborted", "runner_error") or "dataflow exited" in reasons:
        return "Infrastructure / Timing Error"
    if "planning failed" in reasons or "expected" in reasons and "succeeded" in reasons or "not solved" in reasons \
            or "planning took" in reasons or "planner error" in reasons:
        return "Planning Failure"
    if "gear changes" in reasons or "used reverse" in reasons or "used forward" in reasons:
        return "Planning Failure"
    if "contact" in reasons or "overlapped" in reasons or "out of map" in reasons:
        return "Collision / Safety Violation"
    if (m.get("mpc_solver_success_rate") or 1.0) < 0.95:
        return "Solver Failure"
    if m.get("gear_changes_while_moving"):
        return "Gear Change Failure"
    if "timeout" in outcome:
        return "Timing Error"
    return "MPC Tracking Error"


def _stats(values: List[float]) -> Dict[str, float]:
    v = np.asarray([x for x in values if x is not None and isinstance(x, (int, float)) and math.isfinite(x)])
    if len(v) == 0:
        return {}
    return {"mean": float(v.mean()), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "max": float(v.max()), "min": float(v.min()), "n": int(len(v))}


def summarize(results: List[Dict], overrides=()) -> Dict:
    totals = {"runs": len(results), "passed": sum(bool(r.get("passed")) for r in results)}
    totals["failed"] = totals["runs"] - totals["passed"]
    by_class = defaultdict(lambda: Counter())
    for r in results:
        c = r.get("scenario_class") or (r.get("expected") or {}).get("scenario_class") or "unknown"
        by_class[c]["runs"] += 1
        by_class[c]["passed"] += bool(r.get("passed"))
        by_class[c]["outcome:" + str(r.get("outcome"))] += 1
    reached = [r for r in results if r.get("outcome") == "goal_reached"]
    metrics = {k: _stats([(r.get("metrics") or {}).get(k) for r in reached]) for k in METRIC_KEYS}
    failures = [{"scenario_id": r.get("scenario_id"), "outcome": r.get("outcome"), "cause": failure_cause(r),
                 "reasons": r.get("reasons"), "run_dir": r.get("run_dir")} for r in results if not r.get("passed")]
    return {"totals": totals, "by_class": {k: dict(v) for k, v in by_class.items()},
            "goal_reached_metrics": metrics, "failures": failures,
            "failure_causes": dict(Counter(f["cause"] for f in failures)), "overrides": list(overrides),
            "scenarios": [{"scenario_id": r.get("scenario_id"), "class": r.get("scenario_class"),
                           "passed": r.get("passed"), "outcome": r.get("outcome"),
                           "planning_status": r.get("planning_status"),
                           **{k: (r.get("metrics") or {}).get(k) for k in METRIC_KEYS}} for r in results]}


def write_reports(results: List[Dict], out_dir: Path, overrides=()) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = summarize(results, overrides)
    (out_dir / "summary.json").write_text(json.dumps(s, indent=2, default=str))
    lines = [f"# Closed-loop results ({s['totals']['passed']}/{s['totals']['runs']} passed)", ""]
    lines += ["| class | runs | passed |", "|---|---|---|"]
    lines += [f"| {k} | {v.get('runs', 0)} | {v.get('passed', 0)} |" for k, v in sorted(s["by_class"].items())]
    lines += ["", "| scenario | class | result | outcome | plan | pos err [m] | yaw err [deg] | max lat [m] | "
              "min clr act/plan [m] | sim [s] |", "|---|---|---|---|---|---|---|---|---|---|"]

    def f(v, fmt="{:.3f}"):
        return "-" if v is None or (isinstance(v, float) and not math.isfinite(v)) else fmt.format(v)
    for r in s["scenarios"]:
        lines.append(f"| {r['scenario_id']} | {r['class']} | {'PASS' if r['passed'] else '**FAIL**'} | "
                     f"{r['outcome']} | {r['planning_status']} | {f(r['final_position_error'])} | "
                     f"{f(r['final_yaw_error_deg'], '{:.2f}')} | {f(r['lateral_error_max'])} | "
                     f"{f(r['min_clearance_actual'])}/{f(r['min_clearance_planned'])} | {f(r['sim_duration'], '{:.1f}')} |")
    if s["failures"]:
        lines += ["", "## Failures", ""]
        lines += [f"- **{x['scenario_id']}** ({x['cause']}, {x['outcome']}): {'; '.join(x['reasons'] or [])}"
                  for x in s["failures"]]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")
    return s
