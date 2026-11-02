"""YAML configuration loading with dotted-key overrides (no logic here)."""
from __future__ import annotations

import ast
import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def deep_merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def parse_value(text: str) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError:
        return ast.literal_eval(text)


SELECTOR_KEYS = ("type", "backend", "scenario_source")   # keys that select a plugin by name


def _is_selector(node: Dict[str, Any]) -> bool:
    return any(k in node for k in SELECTOR_KEYS)


def _plugin_section_allowed(node: Dict[str, Any], key: str) -> bool:
    """A key absent from the defaults may only be a plugin's own section: under a
    section that selects a plugin, named after the selected plugin, or ``params``
    for a plugin given by location."""
    if not _is_selector(node):
        return False
    return key == "params" or key in {str(node[k]) for k in SELECTOR_KEYS if k in node}


def apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """``["timing.controller_hz=20", ...]`` -> new config. Unknown keys are an
    error (typos must not silently do nothing), except inside a plugin's own
    section (see :func:`_plugin_section_allowed`), whose contents the plugin
    validates itself."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override {item!r} is not key=value")
        node, free = cfg, False
        parts = key.strip().split(".")
        for i, p in enumerate(parts):
            last = i == len(parts) - 1
            if p not in node:
                if not (free or _plugin_section_allowed(node, p)):
                    raise KeyError(f"unknown config key {key!r}")
                if not last:
                    node[p] = {}
            if last:
                node[p] = parse_value(raw)
            else:
                if not isinstance(node[p], dict):
                    raise KeyError(f"unknown config key {key!r}")
                free = free or _is_selector(node)
                node = node[p]
    return cfg


def check_overlay(default: Dict[str, Any], overlay: Dict[str, Any], merged: Dict[str, Any],
                  where: str = "", free: bool = False) -> None:
    """Reject overlay keys that do not exist in the defaults (same rule as overrides)."""
    for k, v in overlay.items():
        path = f"{where}.{k}" if where else k
        known = isinstance(default, dict) and k in default
        if not (known or free or _plugin_section_allowed(merged, k)):
            raise KeyError(f"unknown config key {path!r} in the config overlay")
        if isinstance(v, dict):
            check_overlay(default.get(k, {}) if known else {}, v, merged[k], path,
                          free or not known or _is_selector(merged))


def load_config(path: Optional[os.PathLike] = None, overrides: Iterable[str] = ()) -> Dict[str, Any]:
    """Load the default config, overlay ``path`` (if any) and ``overrides``."""
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    path = path or os.environ.get("DORAWARE_CONFIG")
    if path and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        with open(path) as f:
            overlay = yaml.safe_load(f) or {}
        merged = deep_merge(cfg, overlay)
        check_overlay(cfg, overlay, merged)
        cfg = merged
    return apply_overrides(cfg, overrides)
