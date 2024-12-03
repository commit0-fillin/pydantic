"""Pluggable schema validator for pydantic."""
from __future__ import annotations
import functools
from typing import TYPE_CHECKING, Any, Callable, Iterable, TypeVar
from pydantic_core import CoreConfig, CoreSchema, SchemaValidator, ValidationError
from typing_extensions import Literal, ParamSpec
if TYPE_CHECKING:
    from . import BaseValidateHandlerProtocol, PydanticPluginProtocol, SchemaKind, SchemaTypePath
P = ParamSpec('P')
R = TypeVar('R')
Event = Literal['on_validate_python', 'on_validate_json', 'on_validate_strings']
events: list[Event] = list(Event.__args__)

def create_schema_validator(schema: CoreSchema, schema_type: Any, schema_type_module: str, schema_type_name: str, schema_kind: SchemaKind, config: CoreConfig | None=None, plugin_settings: dict[str, Any] | None=None) -> SchemaValidator | PluggableSchemaValidator:
    """Create a `SchemaValidator` or `PluggableSchemaValidator` if plugins are installed.

    Returns:
        If plugins are installed then return `PluggableSchemaValidator`, otherwise return `SchemaValidator`.
    """
    from ._loader import get_plugins

    plugins = list(get_plugins())
    if plugins:
        schema_type_path = SchemaTypePath(schema_type_module, schema_type_name)
        return PluggableSchemaValidator(schema, schema_type, schema_type_path, schema_kind, config, plugins, plugin_settings or {})
    else:
        return SchemaValidator(schema, config)

class PluggableSchemaValidator:
    """Pluggable schema validator."""
    __slots__ = ('_schema_validator', 'validate_json', 'validate_python', 'validate_strings')

    def __init__(self, schema: CoreSchema, schema_type: Any, schema_type_path: SchemaTypePath, schema_kind: SchemaKind, config: CoreConfig | None, plugins: Iterable[PydanticPluginProtocol], plugin_settings: dict[str, Any]) -> None:
        self._schema_validator = SchemaValidator(schema, config)
        python_event_handlers: list[BaseValidateHandlerProtocol] = []
        json_event_handlers: list[BaseValidateHandlerProtocol] = []
        strings_event_handlers: list[BaseValidateHandlerProtocol] = []
        for plugin in plugins:
            try:
                p, j, s = plugin.new_schema_validator(schema, schema_type, schema_type_path, schema_kind, config, plugin_settings)
            except TypeError as e:
                raise TypeError(f'Error using plugin `{plugin.__module__}:{plugin.__class__.__name__}`: {e}') from e
            if p is not None:
                python_event_handlers.append(p)
            if j is not None:
                json_event_handlers.append(j)
            if s is not None:
                strings_event_handlers.append(s)
        self.validate_python = build_wrapper(self._schema_validator.validate_python, python_event_handlers)
        self.validate_json = build_wrapper(self._schema_validator.validate_json, json_event_handlers)
        self.validate_strings = build_wrapper(self._schema_validator.validate_strings, strings_event_handlers)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._schema_validator, name)

def filter_handlers(handler_cls: BaseValidateHandlerProtocol, method_name: str) -> bool:
    """Filter out handler methods which are not implemented by the plugin directly - e.g. are missing
    or are inherited from the protocol.
    """
    method = getattr(handler_cls, method_name, None)
    if method is None:
        return False
    
    # Check if the method is defined in the handler_cls and not inherited from BaseValidateHandlerProtocol
    return method.__qualname__.split('.')[0] == handler_cls.__name__
