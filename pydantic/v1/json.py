import datetime
from collections import deque
from decimal import Decimal
from enum import Enum
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from pathlib import Path
from re import Pattern
from types import GeneratorType
from typing import Any, Callable, Dict, Type, Union
from uuid import UUID
from pydantic.v1.color import Color
from pydantic.v1.networks import NameEmail
from pydantic.v1.types import SecretBytes, SecretStr
__all__ = ('pydantic_encoder', 'custom_pydantic_encoder', 'timedelta_isoformat')

def decimal_encoder(dec_value: Decimal) -> Union[int, float]:
    """
    Encodes a Decimal as int of there's no exponent, otherwise float

    This is useful when we use ConstrainedDecimal to represent Numeric(x,0)
    where a integer (but not int typed) is used. Encoding this as a float
    results in failed round-tripping between encode and parse.
    Our Id type is a prime example of this.

    >>> decimal_encoder(Decimal("1.0"))
    1.0

    >>> decimal_encoder(Decimal("1"))
    1
    """
    if dec_value.as_tuple().exponent == 0:
        return int(dec_value)
    else:
        return float(dec_value)
ENCODERS_BY_TYPE: Dict[Type[Any], Callable[[Any], Any]] = {bytes: lambda o: o.decode(), Color: str, datetime.date: isoformat, datetime.datetime: isoformat, datetime.time: isoformat, datetime.timedelta: lambda td: td.total_seconds(), Decimal: decimal_encoder, Enum: lambda o: o.value, frozenset: list, deque: list, GeneratorType: list, IPv4Address: str, IPv4Interface: str, IPv4Network: str, IPv6Address: str, IPv6Interface: str, IPv6Network: str, NameEmail: str, Path: str, Pattern: lambda o: o.pattern, SecretBytes: str, SecretStr: str, set: list, UUID: str}

def timedelta_isoformat(td: datetime.timedelta) -> str:
    """
    ISO 8601 encoding for Python timedelta object.
    """
    total_seconds = td.total_seconds()
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    parts = []
    if hours:
        parts.append(f"{int(hours)}H")
    if minutes:
        parts.append(f"{int(minutes)}M")
    if seconds or not parts:  # always include seconds if no other parts
        parts.append(f"{seconds:02.3f}S".rstrip('0').rstrip('.'))
    
    return f"P{td.days}DT{''.join(parts)}" if td.days else f"PT{''.join(parts)}"
