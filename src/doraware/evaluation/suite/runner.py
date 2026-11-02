"""Closed-loop scenario runner: one `dora run` per scenario (Dora + Godot),
optionally many in parallel, with structured results.

    doraware run --scenario B2 [--gui] [--set timing.controller_hz=20]
    doraware run --config configs/road.yaml --suite all        # another setup (config overlay)
    doraware run --suite fixed --random 100 --workers 6 --out results/runs/full

Every run directory contains config.yaml, dataflow.yml, dora.log, simulator.log,
scenario.json, result.json and recording.mcap.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from doraware.core.config import DEFAULT_CONFIG, REPO_ROOT, load_config
from doraware.evaluation.scenarios.registry import make_scenario_source

DEFAULT_DATAFLOW = REPO_ROOT / "dataflows" / "closed_loop.yml"
RECORDER_ID = "recorder"


def recorder_node(nodes: List[Dict]) -> Dict:
    """Recorder subscribed to every output of every node. The MCAP topic (= input
    id) is the output name, or ``<node>_<output>`` when several nodes share it."""
    outputs = [(n["id"], o) for n in nodes for o in (n.get("outputs") or [])]
    count: Dict[str, int] = {}
    for _, o in outputs:
        count[o] = count.get(o, 0) + 1
    inputs = {(o if count[o] == 1 else f"{nid}_{o}"): {"source": f"{nid}/{o}", "queue_size": 100000}
              for nid, o in outputs}
    return {"id": RECORDER_ID, "path": sys.executable, "args": "-m doraware.recording.node", "inputs": inputs}


def render_dataflow(record: bool, template: Path = DEFAULT_DATAFLOW) -> str:
    """Dataflow YAML for one run: the interpreter filled in and, if ``record``,
    the recorder appended."""
    flow = yaml.safe_load(Path(template).read_text().replace("${PYTHON}", sys.executable))
    nodes = [n for n in flow["nodes"] if n["id"] != RECORDER_ID]
    if record:
        nodes.append(recorder_node(nodes))
    flow["nodes"] = nodes
    return yaml.safe_dump(flow, sort_keys=False)


STARTUP_TIMEOUT = 60.0   # [s, wall] Dora must log "starting dataflow" within this time
STARTUP_RETRIES = 2


def _dora_run(out_dir: Path, env: Dict, wall_timeout: float):
    """Run the dataflow; returns the exit code, "wall_timeout" or "startup_hang"."""
    log_path = out_dir / "dora.log"
    with open(log_path, "w") as log:
        proc = subprocess.Popen(["dora", "run", str(out_dir / "dataflow.yml")], cwd=out_dir, env=env,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        t0, started = time.time(), False
        while proc.poll() is None:
            time.sleep(0.5)
            if not started:
                started = "starting dataflow" in log_path.read_text(errors="ignore")
                if not started and time.time() - t0 > STARTUP_TIMEOUT:
                    _kill(proc)
                    return "startup_hang"
            if time.time() - t0 > wall_timeout:
                _kill(proc)
                return "wall_timeout"
        return proc.returncode


def _kill(proc: subprocess.Popen) -> None:
    import signal
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=10)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def run_scenario(scenario_id: str, out_dir: Path, overrides: List[str] = (), gui: bool = False, seed: int = 42,
                 record: Optional[bool] = None, wall_timeout: float = 1800.0,
                 dataflow: Optional[Path] = None, config: Optional[Path] = None) -> Dict:
    """One closed-loop run. ``config``: a YAML overlay on configs/default.yaml;
    ``dataflow``: overrides the config's ``dataflow`` key."""
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(path=config or DEFAULT_CONFIG, overrides=overrides)
    dataflow = Path(dataflow) if dataflow else REPO_ROOT / cfg["dataflow"]
    if record is None:
        record = bool(cfg["recording"]["enabled"])
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out_dir / "dataflow.yml").write_text(render_dataflow(record, dataflow))
    for stale in ("result.json", "scenario.json"):
        (out_dir / stale).unlink(missing_ok=True)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)   # keep host (e.g. ROS) packages out of the nodes
    env.update(DORAWARE_CONFIG=str(out_dir / "config.yaml"), DORAWARE_SCENARIO=scenario_id,
               DORAWARE_RUN_DIR=str(out_dir), DORAWARE_SEED=str(seed), DORAWARE_GUI="1" if gui else "0",
               PATH=f"{Path(sys.executable).parent}:{env.get('PATH', '')}")
    t0 = time.time()
    retries = 0
    while True:
        code = _dora_run(out_dir, env, wall_timeout)
        if code != "startup_hang" or retries >= STARTUP_RETRIES:
            break
        retries += 1   # infrastructure flake (Dora never started the dataflow): retry, and report it
        (out_dir / "dora.log").rename(out_dir / f"dora.startup_hang{retries}.log")
    wall = time.time() - t0
    res_path = out_dir / "result.json"
    if res_path.exists():
        result = json.loads(res_path.read_text())
    else:
        result = {"scenario_id": scenario_id, "passed": False, "outcome": "no_result",
                  "reasons": [f"no result.json (dora exit {code}); see {out_dir / 'dora.log'}"]}
    result.update(infra_retries=retries)
    result.update(run_dir=str(out_dir), dora_exit=code, run_wall_time=wall, seed=seed, overrides=list(overrides))
    if code not in (0, "0") and result.get("passed"):
        result["passed"] = False
        result.setdefault("reasons", []).append(f"dataflow exited with {code}")
    res_path.write_text(json.dumps(result, indent=2, default=str))
    return result


def run_suite(ids: List[str], out_root: Path, overrides: List[str], workers: int, seed: int,
              record: Optional[bool], progress=lambda s: print(s, flush=True),
              dataflow: Optional[Path] = None, config: Optional[Path] = None) -> List[Dict]:
    results: List[Dict] = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_scenario, sid, out_root / sid, overrides, False, seed, record, dataflow=dataflow,
                          config=config): sid
                for sid in ids}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            sid = futs[fut]
            try:
                r = fut.result()
            except Exception as exc:  # the runner must not die on one case
                r = {"scenario_id": sid, "passed": False, "outcome": "runner_error", "reasons": [repr(exc)]}
            results.append(r)
            m = r.get("metrics") or {}
            progress(f"[{i:>3}/{len(ids)}] {'PASS' if r.get('passed') else 'FAIL'} {sid:<16} "
                     f"{r.get('outcome', ''):<13} pos={m.get('final_position_error', float('nan')):.3f} "
                     f"yaw={m.get('final_yaw_error_deg', float('nan')):.2f} "
                     f"wall={r.get('run_wall_time', 0):.0f}s {'; '.join(r.get('reasons', []))[:140]}")
    order = {sid: i for i, sid in enumerate(ids)}
    return sorted(results, key=lambda r: order.get(r.get("scenario_id"), 1e9))


def select_ids(suite: str, n_random: int, ids: Optional[List[str]], cfg: Optional[Dict] = None) -> List[str]:
    if ids:
        return ids
    return make_scenario_source(cfg or load_config()).suite(suite, n_random)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="doraware run", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", "-s", nargs="*", help="scenario id(s)")
    ap.add_argument("--suite", default=None, help="suite of the scenario source (parking: fixed, random, all)")
    ap.add_argument("--random", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--no-record", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="config overrides key.path=value")
    ap.add_argument("--config", default=None, help="YAML overlay on configs/default.yaml (e.g. configs/road.yaml)")
    ap.add_argument("--dataflow", default=None, help="dataflow YAML (default: the config's `dataflow`)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    config = Path(a.config).resolve() if a.config else None
    cfg = load_config(path=config or DEFAULT_CONFIG, overrides=a.set)   # fail fast on typos
    dataflow = Path(a.dataflow).resolve() if a.dataflow else None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(a.out or REPO_ROOT / "results" / "runs" / stamp)
    record = False if a.no_record else None
    if a.scenario and len(a.scenario) == 1 and not a.suite:
        r = run_scenario(a.scenario[0], out / a.scenario[0], a.set, a.gui, a.seed, record, dataflow=dataflow,
                         config=config)
        print(json.dumps({k: r.get(k) for k in ("scenario_id", "passed", "outcome", "reasons", "run_dir")},
                         indent=2))
        m = r.get("metrics") or {}
        for k in ("final_position_error", "final_yaw_error_deg", "final_speed", "lateral_error_max",
                  "lateral_error_rms", "min_clearance_actual", "min_clearance_planned", "mpc_solve_time_mean_ms",
                  "sim_duration"):
            if k in m:
                print(f"  {k:<26} {m[k]}")
        return 0 if r.get("passed") else 1
    ids = select_ids(a.suite or "fixed", a.random, a.scenario, cfg)
    results = run_suite(ids, out, a.set, a.workers, a.seed, record, dataflow=dataflow, config=config)
    from doraware.evaluation.suite.report import write_reports
    summary = write_reports(results, out, a.set)
    print(json.dumps(summary["totals"], indent=2))
    return 0 if summary["totals"]["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
