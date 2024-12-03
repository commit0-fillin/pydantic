import sys
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, NamedTuple, Type
from pydantic.v1.fields import Required
from pydantic.v1.main import BaseModel, create_model
from pydantic.v1.typing import is_typeddict, is_typeddict_special

__all__ = ('create_model_from_typeddict', 'create_model_from_namedtuple')

if TYPE_CHECKING:
    from typing_extensions import TypedDict
if sys.version_info < (3, 11):

def create_model_from_typeddict(typeddict_cls: Type['TypedDict'], **kwargs: Any) -> Type['BaseModel']:
    """
    Create a `BaseModel` based on the fields of a `TypedDict`.
    Since `typing.TypedDict` in Python 3.8 does not store runtime information about optional keys,
    we raise an error if this happens (see https://bugs.python.org/issue38834).
    """
    if not is_typeddict(typeddict_cls):
        raise ValueError(f'{typeddict_cls} is not a TypedDict')

    fields = {}
    for field_name, field_type in typeddict_cls.__annotations__.items():
        if sys.version_info < (3, 9) and field_type == Any:
            raise ValueError(
                f'Field {field_name} has type Any; on Python 3.8 this means that it was likely defined '
                f'without a type annotation, and then annotated as Any at runtime. '
                f'This is not supported, please add type annotations to your TypedDict.'
            )
        default = ... if typeddict_cls.__total__ else None
        fields[field_name] = (field_type, default)

    model_name = typeddict_cls.__name__
    return create_model(model_name, __base__=BaseModel, **fields, **kwargs)

def create_model_from_namedtuple(namedtuple_cls: Type['NamedTuple'], **kwargs: Any) -> Type['BaseModel']:
    """
    Create a `BaseModel` based on the fields of a named tuple.
    A named tuple can be created with `typing.NamedTuple` and declared annotations
    but also with `collections.namedtuple`, in this case we consider all fields
    to have type `Any`.
    """
    if not is_namedtuple(namedtuple_cls):
        raise ValueError(f'{namedtuple_cls} is not a NamedTuple')

    fields = {}
    for field_name, field_type in namedtuple_cls.__annotations__.items():
        default = namedtuple_cls._field_defaults.get(field_name, ...)
        fields[field_name] = (field_type, default)

    # Handle fields without annotations (e.g., from collections.namedtuple)
    for field_name in namedtuple_cls._fields:
        if field_name not in fields:
            default = namedtuple_cls._field_defaults.get(field_name, ...)
            fields[field_name] = (Any, default)

    model_name = namedtuple_cls.__name__
    return create_model(model_name, __base__=BaseModel, **fields, **kwargs)
