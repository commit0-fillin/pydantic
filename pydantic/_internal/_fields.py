"""Private logic related to fields (the `Field()` function and `FieldInfo` class), and arguments to `Annotated`."""
from __future__ import annotations as _annotations
import dataclasses
import sys
import warnings
from copy import copy
from functools import lru_cache
from typing import TYPE_CHECKING, Any
from pydantic_core import PydanticUndefined
from pydantic.errors import PydanticUserError
from . import _typing_extra
from ._config import ConfigWrapper
from ._docs_extraction import extract_docstrings_from_cls
from ._repr import Representation
from ._typing_extra import get_cls_type_hints_lenient, get_type_hints, is_classvar, is_finalvar
if TYPE_CHECKING:
    from annotated_types import BaseMetadata
    from ..fields import FieldInfo
    from ..main import BaseModel
    from ._dataclasses import StandardDataclass
    from ._decorators import DecoratorInfos

def get_type_hints_infer_globalns(obj: Any, localns: dict[str, Any] | None=None, include_extras: bool=False) -> dict[str, Any]:
    """Gets type hints for an object by inferring the global namespace.

    It uses the `typing.get_type_hints`, The only thing that we do here is fetching
    global namespace from `obj.__module__` if it is not `None`.

    Args:
        obj: The object to get its type hints.
        localns: The local namespaces.
        include_extras: Whether to recursively include annotation metadata.

    Returns:
        The object type hints.
    """
    globalns = None
    if hasattr(obj, '__module__'):
        try:
            globalns = sys.modules[obj.__module__].__dict__
        except KeyError:
            pass
    
    return get_type_hints(obj, globalns=globalns, localns=localns, include_extras=include_extras)

class PydanticMetadata(Representation):
    """Base class for annotation markers like `Strict`."""
    __slots__ = ()

def pydantic_general_metadata(**metadata: Any) -> BaseMetadata:
    """Create a new `_PydanticGeneralMetadata` class with the given metadata.

    Args:
        **metadata: The metadata to add.

    Returns:
        The new `_PydanticGeneralMetadata` class.
    """
    class _PydanticGeneralMetadata(_general_metadata_cls()):
        __slots__ = tuple(metadata.keys())

        def __init__(self, **kwargs):
            for key, value in metadata.items():
                setattr(self, key, value)
            for key, value in kwargs.items():
                if key not in metadata:
                    raise TypeError(f"Unexpected keyword argument {key!r}")
                setattr(self, key, value)

    return _PydanticGeneralMetadata(**metadata)

@lru_cache(maxsize=None)
def _general_metadata_cls() -> type[BaseMetadata]:
    """Do it this way to avoid importing `annotated_types` at import time."""
    from annotated_types import BaseMetadata
    return BaseMetadata

def collect_model_fields(cls: type[BaseModel], bases: tuple[type[Any], ...], config_wrapper: ConfigWrapper, types_namespace: dict[str, Any] | None, *, typevars_map: dict[Any, Any] | None=None) -> tuple[dict[str, FieldInfo], set[str]]:
    """Collect the fields of a nascent pydantic model.

    Also collect the names of any ClassVars present in the type hints.

    The returned value is a tuple of two items: the fields dict, and the set of ClassVar names.

    Args:
        cls: BaseModel or dataclass.
        bases: Parents of the class, generally `cls.__bases__`.
        config_wrapper: The config wrapper instance.
        types_namespace: Optional extra namespace to look for types in.
        typevars_map: A dictionary mapping type variables to their concrete types.

    Returns:
        A tuple contains fields and class variables.

    Raises:
        NameError:
            - If there is a conflict between a field name and protected namespaces.
            - If there is a field other than `root` in `RootModel`.
            - If a field shadows an attribute in the parent model.
    """
    fields: dict[str, FieldInfo] = {}
    class_vars: set[str] = set()

    # Collect fields from parent classes
    for base in reversed(bases):
        if hasattr(base, '__pydantic_fields__'):
            fields.update(base.__pydantic_fields__)

    # Get type hints for the current class
    type_hints = get_type_hints_infer_globalns(cls, types_namespace, include_extras=True)

    # Process each attribute in the class
    for name, value in cls.__dict__.items():
        if name.startswith('__') and name.endswith('__'):
            continue

        if isinstance(value, FieldInfo):
            fields[name] = value
        elif name in type_hints:
            if is_classvar(type_hints[name]):
                class_vars.add(name)
            elif not is_finalvar(type_hints[name]):
                fields[name] = FieldInfo(default=value)

    # Apply config
    for name, field in fields.items():
        field.apply_config(config_wrapper, name)

    # Check for naming conflicts
    protected_namespaces = {'model_', 'validator_'}
    for field_name in fields:
        if any(field_name.startswith(namespace) for namespace in protected_namespaces):
            raise NameError(f"Field {field_name} conflicts with protected namespace {field_name.split('_')[0]}_")

    # Check for RootModel
    if cls.__name__ == 'RootModel' and set(fields.keys()) != {'root'}:
        raise NameError("RootModel can only have a single field named 'root'")

    # Check for shadowing
    for base in bases:
        for name in fields:
            if hasattr(base, name) and not isinstance(getattr(base, name), FieldInfo):
                raise NameError(f"Field {name} shadows an attribute in parent {base.__name__}")

    return fields, class_vars

def collect_dataclass_fields(cls: type[StandardDataclass], types_namespace: dict[str, Any] | None, *, typevars_map: dict[Any, Any] | None=None, config_wrapper: ConfigWrapper | None=None) -> dict[str, FieldInfo]:
    """Collect the fields of a dataclass.

    Args:
        cls: dataclass.
        types_namespace: Optional extra namespace to look for types in.
        typevars_map: A dictionary mapping type variables to their concrete types.
        config_wrapper: The config wrapper instance.

    Returns:
        The dataclass fields.
    """
    fields: dict[str, FieldInfo] = {}
    
    # Get type hints for the dataclass
    type_hints = get_type_hints_infer_globalns(cls, types_namespace, include_extras=True)
    
    # Process each field in the dataclass
    for field in dataclasses.fields(cls):
        field_info = FieldInfo(
            annotation=type_hints.get(field.name, Any),
            default=field.default if field.default is not dataclasses.MISSING else PydanticUndefined,
            default_factory=field.default_factory if field.default_factory is not dataclasses.MISSING else None,
            alias=getattr(field, 'alias', None),
            title=getattr(field, 'title', None),
            description=getattr(field, 'description', None),
            exclude=getattr(field, 'exclude', None),
            include=getattr(field, 'include', None),
            const=getattr(field, 'const', None),
            gt=getattr(field, 'gt', None),
            ge=getattr(field, 'ge', None),
            lt=getattr(field, 'lt', None),
            le=getattr(field, 'le', None),
            multiple_of=getattr(field, 'multiple_of', None),
            max_digits=getattr(field, 'max_digits', None),
            decimal_places=getattr(field, 'decimal_places', None),
            min_items=getattr(field, 'min_items', None),
            max_items=getattr(field, 'max_items', None),
            unique_items=getattr(field, 'unique_items', None),
            min_length=getattr(field, 'min_length', None),
            max_length=getattr(field, 'max_length', None),
            allow_mutation=getattr(field, 'allow_mutation', True),
            regex=getattr(field, 'regex', None),
            discriminator=getattr(field, 'discriminator', None),
            repr=getattr(field, 'repr', True),
            **{k: v for k, v in field.__dict__.items() if k.startswith('json_')}
        )
        
        if config_wrapper:
            field_info.apply_config(config_wrapper, field.name)
        
        fields[field.name] = field_info
    
    return fields
