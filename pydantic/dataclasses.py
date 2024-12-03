"""Provide an enhanced dataclass that performs validation."""
from __future__ import annotations as _annotations
import dataclasses
import sys
import types
from typing import TYPE_CHECKING, Any, Callable, Generic, NoReturn, TypeVar, overload
from typing_extensions import Literal, TypeGuard, dataclass_transform
from ._internal import _config, _decorators, _typing_extra
from ._internal import _dataclasses as _pydantic_dataclasses
from ._migration import getattr_migration
from .config import ConfigDict
from .errors import PydanticUserError
from .fields import Field, FieldInfo, PrivateAttr
if TYPE_CHECKING:
    from ._internal._dataclasses import PydanticDataclass
__all__ = ('dataclass', 'rebuild_dataclass')
_T = TypeVar('_T')
if sys.version_info >= (3, 10):

@dataclass_transform(field_specifiers=(dataclasses.field, Field, PrivateAttr))
def dataclass(_cls: type[_T] | None=None, *, init: Literal[False]=False, repr: bool=True, eq: bool=True, order: bool=False, unsafe_hash: bool=False, frozen: bool=False, config: ConfigDict | type[object] | None=None, validate_on_init: bool | None=None, kw_only: bool=False, slots: bool=False) -> Callable[[type[_T]], type[PydanticDataclass]] | type[PydanticDataclass]:
    """Usage docs: https://docs.pydantic.dev/2.8/concepts/dataclasses/

    A decorator used to create a Pydantic-enhanced dataclass, similar to the standard Python `dataclass`,
    but with added validation.

    This function should be used similarly to `dataclasses.dataclass`.

    Args:
        _cls: The target `dataclass`.
        init: Included for signature compatibility with `dataclasses.dataclass`, and is passed through to
            `dataclasses.dataclass` when appropriate. If specified, must be set to `False`, as pydantic inserts its
            own  `__init__` function.
        repr: A boolean indicating whether to include the field in the `__repr__` output.
        eq: Determines if a `__eq__` method should be generated for the class.
        order: Determines if comparison magic methods should be generated, such as `__lt__`, but not `__eq__`.
        unsafe_hash: Determines if a `__hash__` method should be included in the class, as in `dataclasses.dataclass`.
        frozen: Determines if the generated class should be a 'frozen' `dataclass`, which does not allow its
            attributes to be modified after it has been initialized.
        config: The Pydantic config to use for the `dataclass`.
        validate_on_init: A deprecated parameter included for backwards compatibility; in V2, all Pydantic dataclasses
            are validated on init.
        kw_only: Determines if `__init__` method parameters must be specified by keyword only. Defaults to `False`.
        slots: Determines if the generated class should be a 'slots' `dataclass`, which does not allow the addition of
            new attributes after instantiation.

    Returns:
        A decorator that accepts a class as its argument and returns a Pydantic `dataclass`.

    Raises:
        AssertionError: Raised if `init` is not `False` or `validate_on_init` is `False`.
    """
    def wrap(cls: type[_T]) -> type[PydanticDataclass]:
        if validate_on_init is False:
            raise AssertionError("`validate_on_init=False` is not supported in v2")
        
        if init is not False:
            raise AssertionError("`init` must be False for pydantic dataclasses")

        cls_dict = dict(cls.__dict__)
        cls_dict['__pydantic_config__'] = _config.ConfigWrapper(config)
        cls_dict['__pydantic_decorators__'] = _decorators.DecoratorInfos.build(cls)
        cls_dict['__pydantic_complete__'] = False

        std_dataclass = dataclasses.dataclass(
            cls,
            init=False,
            repr=repr,
            eq=eq,
            order=order,
            unsafe_hash=unsafe_hash,
            frozen=frozen,
            kw_only=kw_only,
            slots=slots,
        )

        complete_dataclass(std_dataclass, cls_dict['__pydantic_config__'], raise_errors=False, types_namespace=None)
        return std_dataclass

    if _cls is None:
        return wrap

    return wrap(_cls)
__getattr__ = getattr_migration(__name__)
if (3, 8) <= sys.version_info < (3, 11):

    def _call_initvar(*args: Any, **kwargs: Any) -> NoReturn:
        """This function does nothing but raise an error that is as similar as possible to what you'd get
        if you were to try calling `InitVar[int]()` without this monkeypatch. The whole purpose is just
        to ensure typing._type_check does not error if the type hint evaluates to `InitVar[<parameter>]`.
        """
        raise TypeError("Cannot instantiate InitVar")
    dataclasses.InitVar.__call__ = _call_initvar

def rebuild_dataclass(cls: type[PydanticDataclass], *, force: bool=False, raise_errors: bool=True, _parent_namespace_depth: int=2, _types_namespace: dict[str, Any] | None=None) -> bool | None:
    """Try to rebuild the pydantic-core schema for the dataclass.

    This may be necessary when one of the annotations is a ForwardRef which could not be resolved during
    the initial attempt to build the schema, and automatic rebuilding fails.

    This is analogous to `BaseModel.model_rebuild`.

    Args:
        cls: The class to rebuild the pydantic-core schema for.
        force: Whether to force the rebuilding of the schema, defaults to `False`.
        raise_errors: Whether to raise errors, defaults to `True`.
        _parent_namespace_depth: The depth level of the parent namespace, defaults to 2.
        _types_namespace: The types namespace, defaults to `None`.

    Returns:
        Returns `None` if the schema is already "complete" and rebuilding was not required.
        If rebuilding _was_ required, returns `True` if rebuilding was successful, otherwise `False`.
    """
    if not force and cls.__pydantic_complete__:
        return None

    if _types_namespace is None:
        _types_namespace = _typing_extra.parent_frame_namespace(parent_depth=_parent_namespace_depth)

    config_wrapper = _config.ConfigWrapper(cls.__pydantic_config__)
    return complete_dataclass(cls, config_wrapper, raise_errors=raise_errors, types_namespace=_types_namespace)

def is_pydantic_dataclass(class_: type[Any], /) -> TypeGuard[type[PydanticDataclass]]:
    """Whether a class is a pydantic dataclass.

    Args:
        class_: The class.

    Returns:
        `True` if the class is a pydantic dataclass, `False` otherwise.
    """
    return hasattr(class_, '__pydantic_complete__')
