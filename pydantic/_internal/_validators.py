"""Validator functions for standard library types.

Import of this module is deferred since it contains imports of many standard library modules.
"""
from __future__ import annotations as _annotations
import math
import re
import typing
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from typing import Any, Callable
from pydantic_core import PydanticCustomError, core_schema
from pydantic_core._pydantic_core import PydanticKnownError

def sequence_validator(input_value: typing.Sequence[Any], /, validator: core_schema.ValidatorFunctionWrapHandler) -> typing.Sequence[Any]:
    """Validator for `Sequence` types, isinstance(v, Sequence) has already been called."""
    return [validator(item) for item in input_value]

def _import_string_logic(dotted_path: str) -> Any:
    """Inspired by uvicorn — dotted paths should include a colon before the final item if that item is not a module.
    (This is necessary to distinguish between a submodule and an attribute when there is a conflict.).

    If the dotted path does not include a colon and the final item is not a valid module, importing as an attribute
    rather than a submodule will be attempted automatically.

    So, for example, the following values of `dotted_path` result in the following returned values:
    * 'collections': <module 'collections'>
    * 'collections.abc': <module 'collections.abc'>
    * 'collections.abc:Mapping': <class 'collections.abc.Mapping'>
    * `collections.abc.Mapping`: <class 'collections.abc.Mapping'> (though this is a bit slower than the previous line)

    An error will be raised under any of the following scenarios:
    * `dotted_path` contains more than one colon (e.g., 'collections:abc:Mapping')
    * the substring of `dotted_path` before the colon is not a valid module in the environment (e.g., '123:Mapping')
    * the substring of `dotted_path` after the colon is not an attribute of the module (e.g., 'collections:abc123')
    """
    if dotted_path.count(':') > 1:
        raise ImportError(f"Invalid dotted path: '{dotted_path}'. Only one colon allowed.")

    try:
        if ':' in dotted_path:
            module_path, class_name = dotted_path.rsplit(':', 1)
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        else:
            return __import__(dotted_path)
    except ImportError:
        if '.' in dotted_path:
            module_path, class_name = dotted_path.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            return getattr(module, class_name)
        raise
PatternType = typing.TypeVar('PatternType', str, bytes)

def ip_v4_network_validator(input_value: Any, /) -> IPv4Network:
    """Assume IPv4Network initialised with a default `strict` argument.

    See more:
    https://docs.python.org/library/ipaddress.html#ipaddress.IPv4Network
    """
    try:
        return IPv4Network(input_value)
    except ValueError as e:
        raise PydanticCustomError('ip_v4_network', 'Invalid IPv4 network address') from e

def ip_v6_network_validator(input_value: Any, /) -> IPv6Network:
    """Assume IPv6Network initialised with a default `strict` argument.

    See more:
    https://docs.python.org/library/ipaddress.html#ipaddress.IPv6Network
    """
    try:
        return IPv6Network(input_value)
    except ValueError as e:
        raise PydanticCustomError('ip_v6_network', 'Invalid IPv6 network address') from e
_CONSTRAINT_TO_VALIDATOR_MAP: dict[str, Callable] = {'gt': greater_than_validator, 'ge': greater_than_or_equal_validator, 'lt': less_than_validator, 'le': less_than_or_equal_validator, 'multiple_of': multiple_of_validator, 'min_length': min_length_validator, 'max_length': max_length_validator}

def get_constraint_validator(constraint: str) -> Callable:
    """Fetch the validator function for the given constraint."""
    validator = _CONSTRAINT_TO_VALIDATOR_MAP.get(constraint)
    if validator is None:
        raise ValueError(f"Unknown constraint: {constraint}")
    return validator
