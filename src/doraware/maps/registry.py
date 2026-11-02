"""Map loaders: read a lane map file of some format into a ``LaneMap``.

A loader is a factory ``(params) -> MapLoader``; built-in loaders live in
``maps/plugins/`` and are registered below, separately installed packages
(e.g. a Lanelet2 or OpenDRIVE converter) register under the
``doraware.map_loaders`` entry-point group. Scenario sources call
``load_lane_map(path, type)``.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Protocol

from doraware.core.registry import Registry
from doraware.interfaces import LaneMap


class MapLoader(Protocol):
    def load(self, path: str) -> LaneMap:
        """Read ``path`` and return a validated LaneMap in the map frame."""


MAP_LOADERS: Registry[MapLoader] = Registry("map loader", "doraware.map_loaders")


@MAP_LOADERS.register("yaml")
def _yaml(params: Dict[str, Any]) -> MapLoader:
    from doraware.maps.plugins.yaml_lanes import YamlLaneLoader
    return YamlLaneLoader(**params)


@MAP_LOADERS.register("lanelet2")
def _lanelet2(params: Dict[str, Any]) -> MapLoader:
    from doraware.maps.plugins.lanelet2_osm import Lanelet2Loader
    return Lanelet2Loader(**params)


@MAP_LOADERS.register("opendrive")
def _opendrive(params: Dict[str, Any]) -> MapLoader:
    from doraware.maps.plugins.opendrive import OpenDriveLoader
    return OpenDriveLoader(**params)


def load_lane_map(path: str, loader: str = "yaml", params: Optional[Dict[str, Any]] = None) -> LaneMap:
    lanes = MAP_LOADERS.create(loader, dict(params or {})).load(str(path))
    lanes.validate()
    return lanes
