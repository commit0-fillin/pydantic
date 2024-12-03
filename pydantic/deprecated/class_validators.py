"""Old `@validator` and `@root_validator` function validators from V1."""
from __future__ import annotations as _annotations
from functools import partial, partialmethod
from types import FunctionType
from typing import TYPE_CHECKING, Any, Callable, TypeVar, Union, overload
from warnings import warn
from typing_extensions import Literal, Protocol, TypeAlias, deprecated
from .._internal import _decorators, _decorators_v1
from ..errors import PydanticUserError
from ..warnings import PydanticDeprecatedSince20
_ALLOW_REUSE_WARNING_MESSAGE = '`allow_reuse` is deprecated and will be ignored; it should no longer be necessary'
if TYPE_CHECKING:

    class _OnlyValueValidatorClsMethod(Protocol):

        def __call__(self, __cls: Any, __value: Any) -> Any:
            ...

    class _V1ValidatorWithValuesClsMethod(Protocol):

        def __call__(self, __cls: Any, __value: Any, values: dict[str, Any]) -> Any:
            ...

    class _V1ValidatorWithValuesKwOnlyClsMethod(Protocol):

        def __call__(self, __cls: Any, __value: Any, *, values: dict[str, Any]) -> Any:
            ...

    class _V1ValidatorWithKwargsClsMethod(Protocol):

        def __call__(self, __cls: Any, **kwargs: Any) -> Any:
            ...

    class _V1ValidatorWithValuesAndKwargsClsMethod(Protocol):

        def __call__(self, __cls: Any, values: dict[str, Any], **kwargs: Any) -> Any:
            ...

    class _V1RootValidatorClsMethod(Protocol):

        def __call__(self, __cls: Any, __values: _decorators_v1.RootValidatorValues) -> _decorators_v1.RootValidatorValues:
            ...
    V1Validator = Union[_OnlyValueValidatorClsMethod, _V1ValidatorWithValuesClsMethod, _V1ValidatorWithValuesKwOnlyClsMethod, _V1ValidatorWithKwargsClsMethod, _V1ValidatorWithValuesAndKwargsClsMethod, _decorators_v1.V1ValidatorWithValues, _decorators_v1.V1ValidatorWithValuesKwOnly, _decorators_v1.V1ValidatorWithKwargs, _decorators_v1.V1ValidatorWithValuesAndKwargs]
    V1RootValidator = Union[_V1RootValidatorClsMethod, _decorators_v1.V1RootValidatorFunction]
    _PartialClsOrStaticMethod: TypeAlias = Union[classmethod[Any, Any, Any], staticmethod[Any, Any], partialmethod[Any]]
    _V1ValidatorType = TypeVar('_V1ValidatorType', V1Validator, _PartialClsOrStaticMethod)
    _V1RootValidatorFunctionType = TypeVar('_V1RootValidatorFunctionType', _decorators_v1.V1RootValidatorFunction, _V1RootValidatorClsMethod, _PartialClsOrStaticMethod)
else:
    DeprecationWarning = PydanticDeprecatedSince20

@deprecated('Pydantic V1 style `@validator` validators are deprecated. You should migrate to Pydantic V2 style `@field_validator` validators, see the migration guide for more details', category=None)
def validator(__field: str, *fields: str, pre: bool=False, each_item: bool=False, always: bool=False, check_fields: bool | None=None, allow_reuse: bool=False) -> Callable[[_V1ValidatorType], _V1ValidatorType]:
    """Decorate methods on the class indicating that they should be used to validate fields.

    Args:
        __field (str): The first field the validator should be called on; this is separate
            from `fields` to ensure an error is raised if you don't pass at least one.
        *fields (str): Additional field(s) the validator should be called on.
        pre (bool, optional): Whether this validator should be called before the standard
            validators (else after). Defaults to False.
        each_item (bool, optional): For complex objects (sets, lists etc.) whether to validate
            individual elements rather than the whole object. Defaults to False.
        always (bool, optional): Whether this method and other validators should be called even if
            the value is missing. Defaults to False.
        check_fields (bool | None, optional): Whether to check that the fields actually exist on the model.
            Defaults to None.
        allow_reuse (bool, optional): Whether to track and raise an error if another validator refers to
            the decorated function. Defaults to False.

    Returns:
        Callable: A decorator that can be used to decorate a
            function to be used as a validator.
    """
    if allow_reuse:
        warn(_ALLOW_REUSE_WARNING_MESSAGE, DeprecationWarning, stacklevel=2)

    all_fields = (__field,) + fields
    def dec(_func: _V1ValidatorType) -> _V1ValidatorType:
        _func.__validator_config__ = ValidatorDecoratorInfo(  # type: ignore
            fields=all_fields,
            mode='before' if pre else 'after',
            each_item=each_item,
            always=always,
            check_fields=check_fields,
        )
        return _func

    return dec

@deprecated('Pydantic V1 style `@root_validator` validators are deprecated. You should migrate to Pydantic V2 style `@model_validator` validators, see the migration guide for more details', category=None)
def root_validator(*__args, pre: bool=False, skip_on_failure: bool=False, allow_reuse: bool=False) -> Any:
    """Decorate methods on a model indicating that they should be used to validate (and perhaps
    modify) data either before or after standard model parsing/validation is performed.

    Args:
        pre (bool, optional): Whether this validator should be called before the standard
            validators (else after). Defaults to False.
        skip_on_failure (bool, optional): Whether to stop validation and return as soon as a
            failure is encountered. Defaults to False.
        allow_reuse (bool, optional): Whether to track and raise an error if another validator
            refers to the decorated function. Defaults to False.

    Returns:
        Any: A decorator that can be used to decorate a function to be used as a root_validator.
    """
    if allow_reuse:
        warn(_ALLOW_REUSE_WARNING_MESSAGE, DeprecationWarning, stacklevel=2)

    if __args and callable(__args[0]):
        func = __args[0]
        func.__root_validator_config__ = RootValidatorDecoratorInfo(  # type: ignore
            mode='before' if pre else 'after'
        )
        return func

    def dec(_func: Any) -> Any:
        _func.__root_validator_config__ = RootValidatorDecoratorInfo(  # type: ignore
            mode='before' if pre else 'after'
        )
        return _func

    return dec
