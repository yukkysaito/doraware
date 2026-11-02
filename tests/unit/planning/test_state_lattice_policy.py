import json

from doraware.interfaces import PlanningRequest, PlanningResult, PlanningStatus
from doraware.planning.plugins.state_lattice.policy import plan_with_policy


class FakePlanner:
    def __init__(self, outcomes):
        self.outcomes, self.configs = list(outcomes), []

    def plan(self, req):
        self.configs.append(json.loads(req.planner_config))
        st = self.outcomes.pop(0)
        if isinstance(st, tuple):
            st, raw = st
        else:
            raw = {PlanningStatus.NO_PATH: "max_expansions"}.get(st, st.name.lower())
        return PlanningResult(status=st, planning_time=1.0, details=json.dumps({"raw_status": raw}))


BASE = {"max_planning_time": 120.0, "yaw_tolerance": 0.017}
KEYS = ["yaw_tolerance"]


def test_success_first_try_uses_tight_tolerance():
    p = FakePlanner([PlanningStatus.SUCCESS])
    r = plan_with_policy(p, PlanningRequest(planner_config='{"n_steer": 7}'), BASE, KEYS)
    assert r.success and p.configs == [{"max_planning_time": 120.0, "yaw_tolerance": 0.017, "n_steer": 7}]


def test_fallback_after_no_path_drops_tightened_keys():
    p = FakePlanner([PlanningStatus.NO_PATH, PlanningStatus.SUCCESS])
    r = plan_with_policy(p, PlanningRequest(), BASE, KEYS)
    assert r.success and p.configs[1] == {"max_planning_time": 120.0}
    d = json.loads(r.details)
    assert d["tolerance_fallback"] and r.planning_time == 2.0


def test_no_fallback_on_proofs_or_scenario_tolerance():
    for st in (PlanningStatus.UNREACHABLE, PlanningStatus.START_IN_COLLISION, PlanningStatus.INVALID_INPUT):
        p = FakePlanner([st])
        assert plan_with_policy(p, PlanningRequest(), BASE, KEYS).status == st and len(p.configs) == 1
    p = FakePlanner([PlanningStatus.NO_PATH])   # the scenario sets its own tolerance: nothing to relax
    r = plan_with_policy(p, PlanningRequest(planner_config='{"yaw_tolerance": 0.05}'), BASE, KEYS)
    assert r.status == PlanningStatus.NO_PATH and len(p.configs) == 1


def test_no_fallback_when_search_space_exhausted():
    p = FakePlanner([(PlanningStatus.NO_PATH, "no_path")])
    assert plan_with_policy(p, PlanningRequest(), BASE, KEYS).status == PlanningStatus.NO_PATH and len(p.configs) == 1


def test_planner_registry_builds_the_configured_plugin():
    import pytest

    from doraware.core.config import load_config
    from doraware.planning.plugins.state_lattice.planner import StateLatticePlanner
    from doraware.planning.registry import PLANNERS, make_planner
    cfg = load_config()
    p = make_planner(cfg)
    assert isinstance(p, StateLatticePlanner) and "state_lattice" in PLANNERS.names()
    assert p.base_config == cfg["planning"]["planner"]["state_lattice"]["base_config"]
    cfg["planning"]["planner"]["type"] = "nope"
    with pytest.raises(ValueError, match="unknown planner 'nope'"):
        make_planner(cfg)
