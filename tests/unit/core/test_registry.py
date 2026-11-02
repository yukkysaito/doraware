"""Plugin registry: built-in factories, entry points, errors (core/registry.py)."""
from importlib import metadata

import pytest

from doraware.core.registry import Registry


def test_register_create_and_names():
    r = Registry("widget")

    @r.register("a")
    def make_a(x, y=0):
        return ("a", x, y)

    r.register("b", lambda: "b")
    assert r.names() == ["a", "b"]
    assert r.create("a", 1, y=2) == ("a", 1, 2) and r.create("b") == "b"
    with pytest.raises(ValueError, match="already registered"):
        r.register("a", lambda: None)
    with pytest.raises(ValueError, match="unknown widget 'c'; available: \\['a', 'b'\\]"):
        r.create("c")


def test_entry_points_are_discovered(monkeypatch):
    ep = metadata.EntryPoint(name="external", value="collections:OrderedDict", group="doraware.widgets")
    monkeypatch.setattr(metadata, "entry_points", lambda group=None: [ep] if group == "doraware.widgets" else [])
    r = Registry("widget", "doraware.widgets")
    r.register("builtin", dict)
    assert r.names() == ["builtin", "external"]
    from collections import OrderedDict
    assert isinstance(r.create("external"), OrderedDict)
    assert r.factory("builtin") is dict           # built-ins win over entry points with the same name


def test_plugins_can_be_named_by_location(tmp_path):
    f = tmp_path / "my_plugin.py"
    f.write_text("class Planner:\n    def __init__(self, x):\n        self.x = x\n")
    r = Registry("widget")
    assert r.create(f"{f}:Planner", 3).x == 3                     # file path
    from collections import OrderedDict
    assert isinstance(r.create("collections:OrderedDict"), OrderedDict)   # module path
    with pytest.raises(ValueError, match="cannot load widget"):
        r.create(f"{tmp_path / 'missing.py'}:Planner")
    with pytest.raises(ValueError, match="cannot load widget"):
        r.create("collections:NoSuchThing")
