"""Decorator for validating function calls."""
from __future__ import annotations as _annotations
import functools
from typing import TYPE_CHECKING, Any, Callable, TypeVar, overload
from ._internal import _typing_extra, _validate_call
__all__ = ('validate_call',)
if TYPE_CHECKING:
    from .config import ConfigDict
    AnyCallableT = TypeVar('AnyCallableT', bound=Callable[..., Any])

def validate_call(func: AnyCallableT | None=None, /, *, config: ConfigDict | None=None, validate_return: bool=False) -> AnyCallableT | Callable[[AnyCallableT], AnyCallableT]:
    """Usage docs: https://docs.pydantic.dev/2.8/concepts/validation_decorator/

    Returns a decorated wrapper around the function that validates the arguments and, optionally, the return value.

    Usage may be either as a plain decorator `@validate_call` or with arguments `@validate_call(...)`.

    Args:
        func: The function to be decorated.
        config: The configuration dictionary.
        validate_return: Whether to validate the return value.

    Returns:
        The decorated function.
    """
    def decorator(f: AnyCallableT) -> AnyCallableT:
        return _validate_call(f, config, validate_return)

    if func:
        return decorator(func)
    return decorator
