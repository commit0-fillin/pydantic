"""Logic for V1 validators, e.g. `@validator` and `@root_validator`."""
from __future__ import annotations as _annotations
from inspect import Parameter, signature
from typing import Any, Dict, Tuple, Union, cast
from pydantic_core import core_schema
from typing_extensions import Protocol
from ..errors import PydanticUserError
from ._decorators import can_be_positional

class V1OnlyValueValidator(Protocol):
    """A simple validator, supported for V1 validators and V2 validators."""

    def __call__(self, __value: Any) -> Any:
        ...

class V1ValidatorWithValues(Protocol):
    """A validator with `values` argument, supported for V1 validators and V2 validators."""

    def __call__(self, __value: Any, values: dict[str, Any]) -> Any:
        ...

class V1ValidatorWithValuesKwOnly(Protocol):
    """A validator with keyword only `values` argument, supported for V1 validators and V2 validators."""

    def __call__(self, __value: Any, *, values: dict[str, Any]) -> Any:
        ...

class V1ValidatorWithKwargs(Protocol):
    """A validator with `kwargs` argument, supported for V1 validators and V2 validators."""

    def __call__(self, __value: Any, **kwargs: Any) -> Any:
        ...

class V1ValidatorWithValuesAndKwargs(Protocol):
    """A validator with `values` and `kwargs` arguments, supported for V1 validators and V2 validators."""

    def __call__(self, __value: Any, values: dict[str, Any], **kwargs: Any) -> Any:
        ...
V1Validator = Union[V1ValidatorWithValues, V1ValidatorWithValuesKwOnly, V1ValidatorWithKwargs, V1ValidatorWithValuesAndKwargs]

def make_generic_v1_field_validator(validator: V1Validator) -> core_schema.WithInfoValidatorFunction:
    """Wrap a V1 style field validator for V2 compatibility.

    Args:
        validator: The V1 style field validator.

    Returns:
        A wrapped V2 style field validator.

    Raises:
        PydanticUserError: If the signature is not supported or the parameters are
            not available in Pydantic V2.
    """
    sig = signature(validator)
    params = list(sig.parameters.values())

    def wrapped(value: Any, info: core_schema.ValidationInfo) -> Any:
        kwargs = {}
        if len(params) >= 2 and params[1].name == 'values':
            kwargs['values'] = info.data
        if len(params) >= 3 and params[2].name == 'config':
            kwargs['config'] = info.config
        elif len(params) >= 3 and params[2].name == 'field':
            kwargs['field'] = info.field_info
        elif len(params) >= 3:
            raise PydanticUserError(
                f"Unsupported signature for V1 validator: {sig}",
                code='validator-v1-signature'
            )

        return validator(value, **kwargs)

    return wrapped
RootValidatorValues = Dict[str, Any]
RootValidatorFieldsTuple = Tuple[Any, ...]

class V1RootValidatorFunction(Protocol):
    """A simple root validator, supported for V1 validators and V2 validators."""

    def __call__(self, __values: RootValidatorValues) -> RootValidatorValues:
        ...

class V2CoreBeforeRootValidator(Protocol):
    """V2 validator with mode='before'."""

    def __call__(self, __values: RootValidatorValues, __info: core_schema.ValidationInfo) -> RootValidatorValues:
        ...

class V2CoreAfterRootValidator(Protocol):
    """V2 validator with mode='after'."""

    def __call__(self, __fields_tuple: RootValidatorFieldsTuple, __info: core_schema.ValidationInfo) -> RootValidatorFieldsTuple:
        ...

def make_v1_generic_root_validator(validator: V1RootValidatorFunction, pre: bool) -> V2CoreBeforeRootValidator | V2CoreAfterRootValidator:
    """Wrap a V1 style root validator for V2 compatibility.

    Args:
        validator: The V1 style field validator.
        pre: Whether the validator is a pre validator.

    Returns:
        A wrapped V2 style validator.
    """
    if pre:
        def wrapped(values: RootValidatorValues, info: core_schema.ValidationInfo) -> RootValidatorValues:
            return validator(values)
        return wrapped
    else:
        def wrapped(fields_tuple: RootValidatorFieldsTuple, info: core_schema.ValidationInfo) -> RootValidatorFieldsTuple:
            values = dict(zip(info.field_names, fields_tuple))
            validated_values = validator(values)
            return tuple(validated_values[field] for field in info.field_names)
        return wrapped
