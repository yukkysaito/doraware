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


def apply_overrides(cfg: Dict[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """``["timing.controller_hz=20", ...]`` -> new config. Unknown keys are an
    error (typos must not silently do nothing)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        key, _, raw = item.partition("=")
        if not _:
            raise ValueError(f"override {item!r} is not key=value")
        node = cfg
        parts = key.strip().split(".")
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                raise KeyError(f"unknown config key {key!r}")
            node = node[p]
        if parts[-1] not in node:
            raise KeyError(f"unknown config key {key!r}")
        node[parts[-1]] = parse_value(raw)
    return cfg


def load_config(path: Optional[os.PathLike] = None, overrides: Iterable[str] = ()) -> Dict[str, Any]:
    """Load the default config, overlay ``path`` (if any) and ``overrides``."""
    with open(DEFAULT_CONFIG) as f:
        cfg = yaml.safe_load(f)
    path = path or os.environ.get("DORAWARE_CONFIG")
    if path and Path(path).resolve() != DEFAULT_CONFIG.resolve():
        with open(path) as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    return apply_overrides(cfg, overrides)
