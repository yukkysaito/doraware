"""Test plugin (loaded by file path): the path_pipeline trajectory planner plus a
declared ``objects`` input. It logs, per planner tick, the stamp and size of the
objects message it has seen, so the e2e test can check lockstep delivery."""
import json
import os

from doraware.interfaces import PerceivedObjects
from doraware.planning.trajectory_planning.plugins.path_pipeline.planner import PathPipeline


class ObjectsProbe(PathPipeline):
    inputs = {"objects": PerceivedObjects}

    def __init__(self, params, cfg):
        super().__init__(params, cfg)
        self.latest = None
        self.log = []
        self.path = os.path.join(os.environ.get("DORAWARE_RUN_DIR", "."), "objects_probe.json")

    def on_input(self, name, msg):
        assert name == "objects"
        msg.validate()
        self.latest = msg

    def trajectory_at(self, state, t_ns):
        seen = self.latest
        self.log.append([t_ns, None if seen is None else seen.header.stamp, -1 if seen is None else len(seen)])
        with open(self.path, "w") as f:
            json.dump(self.log, f)
        return super().trajectory_at(state, t_ns)
