from __future__ import annotations
import os
from collections import defaultdict
from typing import Any, Callable, Hashable, TypeVar, Union
from pydantic_core import CoreSchema, core_schema
from pydantic_core import validate_core_schema as _validate_core_schema
from typing_extensions import TypeAliasType, TypeGuard, get_args, get_origin
from . import _repr
from ._typing_extra import is_generic_alias
AnyFunctionSchema = Union[core_schema.AfterValidatorFunctionSchema, core_schema.BeforeValidatorFunctionSchema, core_schema.WrapValidatorFunctionSchema, core_schema.PlainValidatorFunctionSchema]
FunctionSchemaWithInnerSchema = Union[core_schema.AfterValidatorFunctionSchema, core_schema.BeforeValidatorFunctionSchema, core_schema.WrapValidatorFunctionSchema]
CoreSchemaField = Union[core_schema.ModelField, core_schema.DataclassField, core_schema.TypedDictField, core_schema.ComputedField]
CoreSchemaOrField = Union[core_schema.CoreSchema, CoreSchemaField]
_CORE_SCHEMA_FIELD_TYPES = {'typed-dict-field', 'dataclass-field', 'model-field', 'computed-field'}
_FUNCTION_WITH_INNER_SCHEMA_TYPES = {'function-before', 'function-after', 'function-wrap'}
_LIST_LIKE_SCHEMA_WITH_ITEMS_TYPES = {'list', 'set', 'frozenset'}
TAGGED_UNION_TAG_KEY = 'pydantic.internal.tagged_union_tag'
'\nUsed in a `Tag` schema to specify the tag used for a discriminated union.\n'
HAS_INVALID_SCHEMAS_METADATA_KEY = 'pydantic.internal.invalid'
'Used to mark a schema that is invalid because it refers to a definition that was not yet defined when the\nschema was first encountered.\n'

def get_type_ref(type_: type[Any], args_override: tuple[type[Any], ...] | None=None) -> str:
    """Produces the ref to be used for this type by pydantic_core's core schemas.

    This `args_override` argument was added for the purpose of creating valid recursive references
    when creating generic models without needing to create a concrete class.
    """
    if args_override is not None:
        type_args = args_override
    elif is_generic_alias(type_):
        type_args = get_args(type_)
    else:
        type_args = ()
    
    if type_args:
        args_str = ','.join(get_type_ref(arg) for arg in type_args)
        return f'{type_.__name__}[{args_str}]'
    else:
        return type_.__name__

def get_ref(s: core_schema.CoreSchema) -> None | str:
    """Get the ref from the schema if it has one.
    This exists just for type checking to work correctly.
    """
    return s.get('ref')
T = TypeVar('T')
Recurse = Callable[[core_schema.CoreSchema, 'Walk'], core_schema.CoreSchema]
Walk = Callable[[core_schema.CoreSchema, Recurse], core_schema.CoreSchema]

class _WalkCoreSchema:

    def __init__(self):
        self._schema_type_to_method = self._build_schema_type_to_method()
_dispatch = _WalkCoreSchema().walk

def walk_core_schema(schema: core_schema.CoreSchema, f: Walk) -> core_schema.CoreSchema:
    """Recursively traverse a CoreSchema.

    Args:
        schema (core_schema.CoreSchema): The CoreSchema to process, it will not be modified.
        f (Walk): A function to apply. This function takes two arguments:
          1. The current CoreSchema that is being processed
             (not the same one you passed into this function, one level down).
          2. The "next" `f` to call. This lets you for example use `f=functools.partial(some_method, some_context)`
             to pass data down the recursive calls without using globals or other mutable state.

    Returns:
        core_schema.CoreSchema: A processed CoreSchema.
    """
    return _dispatch(schema, f)

def pretty_print_core_schema(schema: CoreSchema, include_metadata: bool=False) -> None:
    """Pretty print a CoreSchema using rich.
    This is intended for debugging purposes.

    Args:
        schema: The CoreSchema to print.
        include_metadata: Whether to include metadata in the output. Defaults to `False`.
    """
    try:
        from rich import print as rich_print
        from rich.pretty import Pretty
    except ImportError:
        print("Rich library is not installed. Please install it to use this function.")
        return

    def _process_schema(s: CoreSchema) -> dict:
        result = {k: v for k, v in s.items() if k != 'metadata' or include_metadata}
        for key, value in result.items():
            if isinstance(value, dict) and 'type' in value:
                result[key] = _process_schema(value)
        return result

    processed_schema = _process_schema(schema)
    rich_print(Pretty(processed_schema, expand_all=True))
