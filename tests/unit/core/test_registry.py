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


def test_create_from_section_passes_the_plugin_parameters(tmp_path):
    r = Registry("widget")
    r.register("a", lambda params, x: (params, x))
    assert r.create_from_section({"type": "a", "a": {"k": 1}}, 5) == ({"k": 1}, 5)
    assert r.create_from_section({"backend": "a"}, 5, key="backend") == ({}, 5)
    f = tmp_path / "w.py"
    f.write_text("def make(params, x):\n    return params, x\n")
    # a plugin given by location takes its parameters from `params`
    assert r.create_from_section({"type": f"{f}:make", "params": {"k": 2}}, 7) == ({"k": 2}, 7)


def test_config_accepts_plugin_sections_but_rejects_typos(tmp_path):
    from doraware.core.config import apply_overrides, load_config
    cfg = load_config()
    out = apply_overrides(cfg, ["planning.path_planner.type=my_planner", "planning.path_planner.my_planner.step=0.5",
                                "planning.path_planner.params.a.b=1"])
    assert out["planning"]["path_planner"]["my_planner"] == {"step": 0.5}
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["planning.path_planner.other_planner.step=1"])     # not the selected plugin
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["timing.controler_hz=20"])                          # typo
    ok = tmp_path / "ok.yaml"
    ok.write_text("control:\n  controller:\n    type: pid\n    pid: {kp: 2.0}\n")
    assert load_config(path=ok)["control"]["controller"]["pid"] == {"kp": 2.0}
    bad = tmp_path / "bad.yaml"
    bad.write_text("timing:\n  plannr_hz: 5\n")
    with pytest.raises(KeyError, match="timing.plannr_hz"):
        load_config(path=bad)
