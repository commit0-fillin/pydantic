import math
import re
from collections import OrderedDict, deque
from collections.abc import Hashable as CollectionsHashable
from datetime import date, datetime, time, timedelta
from decimal import Decimal, DecimalException
from enum import Enum, IntEnum
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Deque, Dict, ForwardRef, FrozenSet, Generator, Hashable, List, NamedTuple, Pattern, Set, Tuple, Type, TypeVar, Union
from uuid import UUID
from pydantic.v1 import errors
from pydantic.v1.datetime_parse import parse_date, parse_datetime, parse_duration, parse_time
from pydantic.v1.typing import AnyCallable, all_literal_values, display_as_type, get_class, is_callable_type, is_literal_type, is_namedtuple, is_none_type, is_typeddict
from pydantic.v1.utils import almost_equal_floats, lenient_issubclass, sequence_like
if TYPE_CHECKING:
    from typing_extensions import Literal, TypedDict
    from pydantic.v1.config import BaseConfig
    from pydantic.v1.fields import ModelField
    from pydantic.v1.types import ConstrainedDecimal, ConstrainedFloat, ConstrainedInt
    ConstrainedNumber = Union[ConstrainedDecimal, ConstrainedFloat, ConstrainedInt]
    AnyOrderedDict = OrderedDict[Any, Any]
    Number = Union[int, float, Decimal]
    StrBytes = Union[str, bytes]
BOOL_FALSE = {0, '0', 'off', 'f', 'false', 'n', 'no'}
BOOL_TRUE = {1, '1', 'on', 't', 'true', 'y', 'yes'}
max_str_int = 4300

def constant_validator(v: 'Any', field: 'ModelField') -> 'Any':
    """Validate ``const`` fields.

    The value provided for a ``const`` field must be equal to the default value
    of the field. This is to support the keyword of the same name in JSON
    Schema.
    """
    if field.default is not None and v != field.default:
        raise errors_.WrongConstantError(given=v, permitted=[field.default])
    return v

def ip_v4_network_validator(v: Any) -> IPv4Network:
    """
    Assume IPv4Network initialised with a default ``strict`` argument

    See more:
    https://docs.python.org/library/ipaddress.html#ipaddress.IPv4Network
    """
    try:
        return IPv4Network(v)
    except (ValueError, TypeError) as e:
        raise errors_.IPv4NetworkError() from e

def ip_v6_network_validator(v: Any) -> IPv6Network:
    """
    Assume IPv6Network initialised with a default ``strict`` argument

    See more:
    https://docs.python.org/library/ipaddress.html#ipaddress.IPv6Network
    """
    try:
        return IPv6Network(v)
    except (ValueError, TypeError) as e:
        raise errors_.IPv6NetworkError() from e

def callable_validator(v: Any) -> AnyCallable:
    """
    Perform a simple check if the value is callable.

    Note: complete matching of argument type hints and return types is not performed
    """
    if not callable(v):
        raise errors_.CallableError(value=v)
    return v
T = TypeVar('T')
NamedTupleT = TypeVar('NamedTupleT', bound=NamedTuple)

class IfConfig:

    def __init__(self, validator: AnyCallable, *config_attr_names: str, ignored_value: Any=False) -> None:
        self.validator = validator
        self.config_attr_names = config_attr_names
        self.ignored_value = ignored_value
_VALIDATORS: List[Tuple[Type[Any], List[Any]]] = [(IntEnum, [int_validator, enum_member_validator]), (Enum, [enum_member_validator]), (str, [str_validator, IfConfig(anystr_strip_whitespace, 'anystr_strip_whitespace'), IfConfig(anystr_upper, 'anystr_upper'), IfConfig(anystr_lower, 'anystr_lower'), IfConfig(anystr_length_validator, 'min_anystr_length', 'max_anystr_length')]), (bytes, [bytes_validator, IfConfig(anystr_strip_whitespace, 'anystr_strip_whitespace'), IfConfig(anystr_upper, 'anystr_upper'), IfConfig(anystr_lower, 'anystr_lower'), IfConfig(anystr_length_validator, 'min_anystr_length', 'max_anystr_length')]), (bool, [bool_validator]), (int, [int_validator]), (float, [float_validator, IfConfig(float_finite_validator, 'allow_inf_nan', ignored_value=True)]), (Path, [path_validator]), (datetime, [parse_datetime]), (date, [parse_date]), (time, [parse_time]), (timedelta, [parse_duration]), (OrderedDict, [ordered_dict_validator]), (dict, [dict_validator]), (list, [list_validator]), (tuple, [tuple_validator]), (set, [set_validator]), (frozenset, [frozenset_validator]), (deque, [deque_validator]), (UUID, [uuid_validator]), (Decimal, [decimal_validator]), (IPv4Interface, [ip_v4_interface_validator]), (IPv6Interface, [ip_v6_interface_validator]), (IPv4Address, [ip_v4_address_validator]), (IPv6Address, [ip_v6_address_validator]), (IPv4Network, [ip_v4_network_validator]), (IPv6Network, [ip_v6_network_validator])]
