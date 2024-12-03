from __future__ import annotations
import importlib.metadata as importlib_metadata
import os
import warnings
from typing import TYPE_CHECKING, Final, Iterable
if TYPE_CHECKING:
    from . import PydanticPluginProtocol
PYDANTIC_ENTRY_POINT_GROUP: Final[str] = 'pydantic'
_plugins: dict[str, PydanticPluginProtocol] | None = None
_loading_plugins: bool = False

def get_plugins() -> Iterable[PydanticPluginProtocol]:
    """Load plugins for Pydantic.

    Inspired by: https://github.com/pytest-dev/pluggy/blob/1.3.0/src/pluggy/_manager.py#L376-L402
    """
    global _plugins, _loading_plugins

    if _plugins is not None:
        yield from _plugins.values()
        return

    if _loading_plugins:
        return

    _loading_plugins = True
    try:
        _plugins = {}
        for entry_point in importlib_metadata.entry_points(group=PYDANTIC_ENTRY_POINT_GROUP):
            try:
                plugin = entry_point.load()
                if not isinstance(plugin, type):
                    raise TypeError(f"Plugin {entry_point.name} must be a class")
                _plugins[entry_point.name] = plugin()
            except Exception as e:
                warnings.warn(f"Failed to load plugin {entry_point.name}: {e}", RuntimeWarning)

        yield from _plugins.values()
    finally:
        _loading_plugins = False
