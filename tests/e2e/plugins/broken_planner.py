"""Test plugin: a trajectory planner whose plan() raises (fault isolation)."""


class BrokenPlanner:
    def __init__(self, params, cfg):
        pass

    def plan(self, request, state, t_ns):
        raise RuntimeError("model checkpoint missing")

    def trajectory_at(self, state, t_ns):
        raise AssertionError("never called after a failed plan")

    def diagnostics(self):
        return {}
