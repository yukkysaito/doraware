"""Dora node: online closed-loop evaluation. Not part of the control path:
it only observes, and finally publishes ``scenario_done`` (shutdown request)
and writes ``result.json`` into $DORAWARE_RUN_DIR."""
from __future__ import annotations

import json
import time

from doraware.evaluation.judge import ScenarioJudge
from doraware.evaluation.metrics import ClosedLoopMetrics
from doraware.interfaces import (ControlCommand, ControlDebug, Diagnostics, PlanningRequest, PlanningResult,
                                   ScenarioDone, ScenarioInfo, SimulationScene, SimulationStatus, Trajectory,
                                   VehicleSignalStatus, VehicleState, to_sec)
from doraware.core.node import NodeRuntime

INPUTS = {"scenario_info": ScenarioInfo, "planning_request": PlanningRequest, "scene": SimulationScene,
          "planning_result": PlanningResult, "trajectory": Trajectory, "vehicle_state": VehicleState,
          "control_command": ControlCommand, "control_debug": ControlDebug, "sim_status": SimulationStatus,
          "trajectory_status": Diagnostics, "diagnostics": Diagnostics, "vehicle_signal_status": VehicleSignalStatus}


def main() -> None:
    rt = NodeRuntime("evaluator", INPUTS)
    ecfg = rt.cfg["evaluation"]
    info = request = scene = None
    metrics = judge = None
    pending = []          # messages that arrive before the scenario is known
    diagnostics = []
    wall0 = time.time()
    last_t = 0.0
    result_written = False

    def setup():
        nonlocal metrics, judge
        cfgp = json.loads(request.planner_config or "{}")
        margin = float(cfgp.get("safety_margin", 0.1))
        metrics = ClosedLoopMetrics(request.map, scene.vehicle, (request.goal.x, request.goal.y, request.goal.yaw),
                                    safety_margin=margin)
        judge = ScenarioJudge(info["expected"], ecfg, metrics)

    def handle(iid, msg):
        nonlocal last_t
        if iid == "planning_result":
            metrics.on_plan(msg)
            judge.on_plan(msg, last_t)
        elif iid == "trajectory":
            metrics.on_trajectory(msg)
        elif iid == "vehicle_state":
            last_t = to_sec(msg.header.stamp)
            metrics.on_state(msg)
        elif iid == "control_command":
            metrics.on_command(msg)
        elif iid == "control_debug":
            metrics.on_debug(msg)
        elif iid == "sim_status":
            metrics.on_sim_status(msg)
        elif iid == "vehicle_signal_status":
            metrics.on_signal_status(msg)
        elif iid == "trajectory_status":
            v = json.loads(msg.values or "{}")
            if "plan_start_time" in v and "plan_duration" in v:
                judge.on_plan_timing(float(v["plan_start_time"]), float(v["plan_duration"]))
            elif v.get("trajectory_ok") is False and v.get("planning_status") == "SUCCESS":
                judge.on_trajectory_failure(str(msg.message), last_t)
        if iid == "diagnostics" or (iid == "trajectory_status" and msg.level > 0):
            diagnostics.append({"t": to_sec(msg.header.stamp), "node": msg.node, "level": int(msg.level),
                                "message": msg.message})

    for ev in rt.events():
        if ev["type"] != "INPUT":
            continue
        iid = ev["id"]
        msg = rt.decode(ev)
        if iid == "scenario_info":
            info = json.loads(msg.description)
        elif iid == "planning_request":
            request = msg
        elif iid == "scene":
            scene = msg
        if metrics is None:
            if info is not None and request is not None and scene is not None:
                setup()
                for p in pending:
                    handle(*p)
                pending.clear()
            elif iid not in ("scenario_info", "planning_request", "scene"):
                pending.append((iid, msg))
            continue
        if iid in ("scenario_info", "planning_request", "scene"):
            continue
        handle(iid, msg)
        if iid != "vehicle_state":
            continue
        v = judge.update(last_t)
        if v.done and not result_written:
            summary = metrics.summary()
            plan = judge.plan
            result = {
                "scenario_id": info["scenario_id"], "name": info["name"], "category": info["category"],
                "scenario_class": info["expected"]["scenario_class"], "expected": info["expected"],
                "passed": v.passed, "outcome": v.outcome, "reasons": v.reasons,
                "planning_status": plan.status.name if plan else None,
                "planning_message": plan.message if plan else None,
                "planning_time": plan.planning_time if plan else None,
                "planning_details": json.loads(plan.details) if plan and plan.details else {},
                "plan_end_time": judge.plan_end_time, "end_sim_time": last_t,
                "wall_time": time.time() - wall0, "metrics": summary, "diagnostics": diagnostics[-50:],
            }
            rt.write_json("result.json", result)
            result_written = True
            rt.log.info("scenario %s: %s (%s) %s", info["scenario_id"], "PASS" if v.passed else "FAIL", v.outcome,
                        "; ".join(v.reasons))
            rt.send("scenario_done", ScenarioDone(header=rt.header(msg.header.stamp, "scenario_done"),
                                                  scenario_id=info["scenario_id"], outcome=v.outcome,
                                                  passed=v.passed, summary=json.dumps(summary, default=str)))
    if not result_written and info is not None:
        rt.write_json("result.json", {"scenario_id": info.get("scenario_id"), "passed": False,
                                      "outcome": "aborted", "reasons": ["dataflow ended before a verdict"],
                                      "end_sim_time": last_t, "diagnostics": diagnostics[-50:]})


if __name__ == "__main__":
    main()
