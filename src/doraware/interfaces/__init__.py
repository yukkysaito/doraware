"""Typed message interfaces: the only contract between doraware components.

Messages are dataclasses grouped by domain (``common``, ``vehicle``, ``map``,
``planning``, ``control``, ``simulation``, ``system``) and re-exported here, so
components import them as ``from doraware.interfaces import Trajectory``.
:mod:`doraware.interfaces.codec` derives the Arrow schema Dora transports.

To add a domain, create ``<domain>.py`` and add it to ``_DOMAINS`` below.
"""
import dataclasses
from typing import Dict

from doraware.interfaces import common, control, map, planning, simulation, system, vehicle
from doraware.interfaces.common import *  # noqa: F401,F403
from doraware.interfaces.control import *  # noqa: F401,F403
from doraware.interfaces.map import *  # noqa: F401,F403
from doraware.interfaces.planning import *  # noqa: F401,F403
from doraware.interfaces.simulation import *  # noqa: F401,F403
from doraware.interfaces.system import *  # noqa: F401,F403
from doraware.interfaces.vehicle import *  # noqa: F401,F403
from doraware.interfaces.codec import check_metadata, decode, encode, metadata  # noqa: F401

_DOMAINS = (common, vehicle, map, planning, control, simulation, system)


def _topic_messages() -> Dict[str, type]:
    """Every message dataclass with a ``header``: the types that travel on Dora
    topics (used e.g. by the recorder). Nested types (Pose, GeometricPath, ...)
    have no header."""
    out: Dict[str, type] = {}
    for mod in _DOMAINS:
        for obj in vars(mod).values():
            if (isinstance(obj, type) and dataclasses.is_dataclass(obj) and obj.__module__ == mod.__name__
                    and any(f.name == "header" for f in dataclasses.fields(obj))):
                out[obj.__name__] = obj
    return out


MESSAGE_TYPES: Dict[str, type] = _topic_messages()
