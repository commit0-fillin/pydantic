import datetime
import warnings
from collections import deque
from decimal import Decimal
from enum import Enum
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from pathlib import Path
from re import Pattern
from types import GeneratorType
from typing import TYPE_CHECKING, Any, Callable, Dict, Type, Union
from uuid import UUID
from typing_extensions import deprecated
from ..color import Color
from ..networks import NameEmail
from ..types import SecretBytes, SecretStr
from ..warnings import PydanticDeprecatedSince20
if not TYPE_CHECKING:
    DeprecationWarning = PydanticDeprecatedSince20
__all__ = ('pydantic_encoder', 'custom_pydantic_encoder', 'timedelta_isoformat')

def decimal_encoder(dec_value: Decimal) -> Union[int, float]:
    """Encodes a Decimal as int of there's no exponent, otherwise float.

    This is useful when we use ConstrainedDecimal to represent Numeric(x,0)
    where a integer (but not int typed) is used. Encoding this as a float
    results in failed round-tripping between encode and parse.
    Our Id type is a prime example of this.

    >>> decimal_encoder(Decimal("1.0"))
    1.0

    >>> decimal_encoder(Decimal("1"))
    1
    """
    if dec_value.as_tuple().exponent >= 0:
        return int(dec_value)
    else:
        return float(dec_value)
ENCODERS_BY_TYPE: Dict[Type[Any], Callable[[Any], Any]] = {bytes: lambda o: o.decode(), Color: str, datetime.date: isoformat, datetime.datetime: isoformat, datetime.time: isoformat, datetime.timedelta: lambda td: td.total_seconds(), Decimal: decimal_encoder, Enum: lambda o: o.value, frozenset: list, deque: list, GeneratorType: list, IPv4Address: str, IPv4Interface: str, IPv4Network: str, IPv6Address: str, IPv6Interface: str, IPv6Network: str, NameEmail: str, Path: str, Pattern: lambda o: o.pattern, SecretBytes: str, SecretStr: str, set: list, UUID: str}

@deprecated('`timedelta_isoformat` is deprecated.', category=None)
def timedelta_isoformat(td: datetime.timedelta) -> str:
    """ISO 8601 encoding for Python timedelta object."""
    total_seconds = td.total_seconds()
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    result = "P"
    if days:
        result += f"{int(days)}D"
    if hours or minutes or seconds:
        result += "T"
        if hours:
            result += f"{int(hours)}H"
        if minutes:
            result += f"{int(minutes)}M"
        if seconds:
            result += f"{seconds:02.3f}S"
    
    return result if result != "P" else "PT0S"
