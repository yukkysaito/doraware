"""Dora node: records every input to MCAP (JSON message encoding, JSON-Schema
channels named after the message type). Off the control path; MCAP
log_time = message header stamp (sim clock), publish_time = same.

The Dora input id is used as the MCAP topic, so the recorded topics map 1:1
to the dataflow edges (communication schema == recording schema)."""
from __future__ import annotations

import json
import os

from mcap.writer import Writer

from doraware.interfaces import codec
from doraware.interfaces import MESSAGE_TYPES
from doraware.core.node import NodeRuntime


def main() -> None:
    rt = NodeRuntime("recorder", {})
    path = os.path.join(rt.run_dir or ".", "recording.mcap")
    schemas, channels = {}, {}
    n = 0
    skipped = set()
    with open(path, "wb") as f:
        w = Writer(f)
        w.start(profile="", library="doraware")
        for ev in rt.events():
            if ev["type"] != "INPUT":
                continue
            meta = ev.get("metadata") or {}
            name = meta.get("schema")
            cls = MESSAGE_TYPES.get(name)
            if cls is None:
                if ev["id"] not in skipped:   # never drop silently: say once per topic what is not recorded
                    skipped.add(ev["id"])
                    rt.log.warning("not recording %s: unknown message schema %r", ev["id"], name)
                continue
            msg = codec.decode(ev["value"], cls)
            if name not in schemas:
                schemas[name] = w.register_schema(name=name, encoding="jsonschema",
                                                  data=json.dumps(codec.json_schema(cls)).encode())
            topic = ev["id"]
            if topic not in channels:
                channels[topic] = w.register_channel(topic=topic, message_encoding="json", schema_id=schemas[name])
            stamp = int(msg.header.stamp) if hasattr(msg, "header") else 0
            w.add_message(channels[topic], log_time=stamp, publish_time=stamp, data=codec.dumps_json(msg),
                          sequence=n)
            n += 1
        w.finish()
    rt.log.info("recorded %d messages to %s", n, path)


if __name__ == "__main__":
    main()
