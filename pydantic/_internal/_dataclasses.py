"""Private logic for creating pydantic dataclasses."""
from __future__ import annotations as _annotations
import dataclasses
import typing
import warnings
from functools import partial, wraps
from typing import Any, Callable, ClassVar
from pydantic_core import ArgsKwargs, SchemaSerializer, SchemaValidator, core_schema
from typing_extensions import TypeGuard
from ..errors import PydanticUndefinedAnnotation
from ..fields import FieldInfo
from ..plugin._schema_validator import PluggableSchemaValidator, create_schema_validator
from ..warnings import PydanticDeprecatedSince20
from . import _config, _decorators, _typing_extra
from ._fields import collect_dataclass_fields
from ._generate_schema import GenerateSchema
from ._generics import get_standard_typevars_map
from ._mock_val_ser import set_dataclass_mocks
from ._schema_generation_shared import CallbackGetCoreSchemaHandler
from ._signature import generate_pydantic_signature
if typing.TYPE_CHECKING:
    from ..config import ConfigDict

    class StandardDataclass(typing.Protocol):
        __dataclass_fields__: ClassVar[dict[str, Any]]
        __dataclass_params__: ClassVar[Any]
        __post_init__: ClassVar[Callable[..., None]]

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    class PydanticDataclass(StandardDataclass, typing.Protocol):
        """A protocol containing attributes only available once a class has been decorated as a Pydantic dataclass.

        Attributes:
            __pydantic_config__: Pydantic-specific configuration settings for the dataclass.
            __pydantic_complete__: Whether dataclass building is completed, or if there are still undefined fields.
            __pydantic_core_schema__: The pydantic-core schema used to build the SchemaValidator and SchemaSerializer.
            __pydantic_decorators__: Metadata containing the decorators defined on the dataclass.
            __pydantic_fields__: Metadata about the fields defined on the dataclass.
            __pydantic_serializer__: The pydantic-core SchemaSerializer used to dump instances of the dataclass.
            __pydantic_validator__: The pydantic-core SchemaValidator used to validate instances of the dataclass.
        """
        __pydantic_config__: ClassVar[ConfigDict]
        __pydantic_complete__: ClassVar[bool]
        __pydantic_core_schema__: ClassVar[core_schema.CoreSchema]
        __pydantic_decorators__: ClassVar[_decorators.DecoratorInfos]
        __pydantic_fields__: ClassVar[dict[str, FieldInfo]]
        __pydantic_serializer__: ClassVar[SchemaSerializer]
        __pydantic_validator__: ClassVar[SchemaValidator | PluggableSchemaValidator]
else:
    DeprecationWarning = PydanticDeprecatedSince20

def set_dataclass_fields(cls: type[StandardDataclass], types_namespace: dict[str, Any] | None=None, config_wrapper: _config.ConfigWrapper | None=None) -> None:
    """Collect and set `cls.__pydantic_fields__`.

    Args:
        cls: The class.
        types_namespace: The types namespace, defaults to `None`.
        config_wrapper: The config wrapper instance, defaults to `None`.
    """
    fields = collect_dataclass_fields(cls, types_namespace, config_wrapper)
    cls.__pydantic_fields__ = fields

def complete_dataclass(cls: type[Any], config_wrapper: _config.ConfigWrapper, *, raise_errors: bool=True, types_namespace: dict[str, Any] | None) -> bool:
    """Finish building a pydantic dataclass.

    This logic is called on a class which has already been wrapped in `dataclasses.dataclass()`.

    This is somewhat analogous to `pydantic._internal._model_construction.complete_model_class`.

    Args:
        cls: The class.
        config_wrapper: The config wrapper instance.
        raise_errors: Whether to raise errors, defaults to `True`.
        types_namespace: The types namespace.

    Returns:
        `True` if building a pydantic dataclass is successfully completed, `False` otherwise.

    Raises:
        PydanticUndefinedAnnotation: If `raise_error` is `True` and there is an undefined annotations.
    """
    try:
        set_dataclass_fields(cls, types_namespace, config_wrapper)
        
        generate_schema = GenerateSchema(config_wrapper)
        core_schema = generate_schema.generate_schema(cls)
        
        cls.__pydantic_core_schema__ = core_schema
        cls.__pydantic_validator__ = create_schema_validator(core_schema, cls, config_wrapper)
        cls.__pydantic_serializer__ = SchemaSerializer(core_schema)
        
        cls.__pydantic_complete__ = True
        return True
    except PydanticUndefinedAnnotation:
        if raise_errors:
            raise
        return False

def is_builtin_dataclass(_cls: type[Any]) -> TypeGuard[type[StandardDataclass]]:
    """Returns True if a class is a stdlib dataclass and *not* a pydantic dataclass.

    We check that
    - `_cls` is a dataclass
    - `_cls` does not inherit from a processed pydantic dataclass (and thus have a `__pydantic_validator__`)
    - `_cls` does not have any annotations that are not dataclass fields
    e.g.
    ```py
    import dataclasses

    import pydantic.dataclasses

    @dataclasses.dataclass
    class A:
        x: int

    @pydantic.dataclasses.dataclass
    class B(A):
        y: int
    ```
    In this case, when we first check `B`, we make an extra check and look at the annotations ('y'),
    which won't be a superset of all the dataclass fields (only the stdlib fields i.e. 'x')

    Args:
        cls: The class.

    Returns:
        `True` if the class is a stdlib dataclass, `False` otherwise.
    """
    if not dataclasses.is_dataclass(_cls):
        return False
    
    if hasattr(_cls, '__pydantic_validator__'):
        return False
    
    annotations = getattr(_cls, '__annotations__', {})
    dataclass_fields = getattr(_cls, '__dataclass_fields__', {})
    
    return set(annotations.keys()).issubset(dataclass_fields.keys())
