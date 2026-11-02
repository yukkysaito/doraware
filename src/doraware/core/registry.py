"""Plugin registries: name -> factory, for built-in and third-party plugins.

A component that can host several algorithms (planners, controllers,
simulator backends, scenario sources, ...) owns one :class:`Registry`.
Built-in plugins register a factory under a name. Factories import their
module lazily, so heavy dependencies (for example PyTorch for a learned
planner) are loaded only when that plugin is selected.

Plugins from separately installed packages register through Python entry
points, so they never have to edit doraware::

    # pyproject.toml of the plugin package
    [project.entry-points."doraware.planners"]
    my_planner = "my_package.planner:MyPlanner"

The entry point may name a class or any callable. It is called with the same
arguments as a built-in factory (documented by each registry).
"""
from __future__ import annotations

from importlib import metadata
from typing import Callable, Dict, Generic, List, Optional, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str, entry_point_group: Optional[str] = None):
        self.kind = kind                                  # e.g. "planner", used in error messages
        self.group = entry_point_group                    # e.g. "doraware.planners"
        self._factories: Dict[str, Callable[..., T]] = {}

    def register(self, name: str, factory: Optional[Callable[..., T]] = None):
        """Register ``factory`` under ``name``; usable as a decorator without ``factory``."""
        def add(f: Callable[..., T]) -> Callable[..., T]:
            if name in self._factories:
                raise ValueError(f"{self.kind} {name!r} is already registered")
            self._factories[name] = f
            return f
        return add if factory is None else add(factory)

    def _entry_points(self) -> Dict[str, metadata.EntryPoint]:
        if not self.group:
            return {}
        return {ep.name: ep for ep in metadata.entry_points(group=self.group)}

    def names(self) -> List[str]:
        return sorted(set(self._factories) | set(self._entry_points()))

    def factory(self, name: str) -> Callable[..., T]:
        if name in self._factories:
            return self._factories[name]
        ep = self._entry_points().get(name)
        if ep is None:
            raise ValueError(f"unknown {self.kind} {name!r}; available: {self.names()}")
        return ep.load()

    def create(self, name: str, *args, **kwargs) -> T:
        return self.factory(name)(*args, **kwargs)
