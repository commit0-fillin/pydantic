from __future__ import annotations
import json
import warnings
from typing import TYPE_CHECKING, Any, Callable, Type, TypeVar, Union
from typing_extensions import deprecated
from ..json_schema import DEFAULT_REF_TEMPLATE, GenerateJsonSchema
from ..type_adapter import TypeAdapter
from ..warnings import PydanticDeprecatedSince20
if not TYPE_CHECKING:
    DeprecationWarning = PydanticDeprecatedSince20
__all__ = ('parse_obj_as', 'schema_of', 'schema_json_of')
NameFactory = Union[str, Callable[[Type[Any]], str]]
T = TypeVar('T')

@deprecated('`schema_of` is deprecated. Use `pydantic.TypeAdapter.json_schema` instead.', category=None)
def schema_of(type_: Any, *, title: NameFactory | None=None, by_alias: bool=True, ref_template: str=DEFAULT_REF_TEMPLATE, schema_generator: type[GenerateJsonSchema]=GenerateJsonSchema) -> dict[str, Any]:
    """Generate a JSON schema (as dict) for the passed model or dynamically generated one."""
    adapter = TypeAdapter(type_)
    schema = adapter.json_schema(by_alias=by_alias, ref_template=ref_template, schema_generator=schema_generator)
    if title is not None:
        if callable(title):
            schema['title'] = title(type_)
        else:
            schema['title'] = title
    return schema

@deprecated('`schema_json_of` is deprecated. Use `pydantic.TypeAdapter.json_schema` instead.', category=None)
def schema_json_of(type_: Any, *, title: NameFactory | None=None, by_alias: bool=True, ref_template: str=DEFAULT_REF_TEMPLATE, schema_generator: type[GenerateJsonSchema]=GenerateJsonSchema, **dumps_kwargs: Any) -> str:
    """Generate a JSON schema (as JSON) for the passed model or dynamically generated one."""
    schema = schema_of(type_, title=title, by_alias=by_alias, ref_template=ref_template, schema_generator=schema_generator)
    return json.dumps(schema, **dumps_kwargs)
