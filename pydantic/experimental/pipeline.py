"""Experimental pipeline API functionality. Be careful with this API, it's subject to change."""
from __future__ import annotations
import datetime
import operator
import re
import sys
from collections import deque
from collections.abc import Container
from dataclasses import dataclass
from decimal import Decimal
from functools import cached_property, partial
from typing import TYPE_CHECKING, Any, Callable, Generic, Pattern, Protocol, TypeVar, Union, overload
import annotated_types
from typing_extensions import Annotated
if TYPE_CHECKING:
    from pydantic_core import core_schema as cs
    from pydantic import GetCoreSchemaHandler
from pydantic._internal._internal_dataclass import slots_true as _slots_true
if sys.version_info < (3, 10):
    EllipsisType = type(Ellipsis)
else:
    from types import EllipsisType
__all__ = ['validate_as', 'validate_as_deferred', 'transform']
_slots_frozen = {**_slots_true, 'frozen': True}

@dataclass(**_slots_frozen)
class _ValidateAs:
    tp: type[Any]
    strict: bool = False

@dataclass
class _ValidateAsDefer:
    func: Callable[[], type[Any]]

@dataclass(**_slots_frozen)
class _Transform:
    func: Callable[[Any], Any]

@dataclass(**_slots_frozen)
class _PipelineOr:
    left: _Pipeline[Any, Any]
    right: _Pipeline[Any, Any]

@dataclass(**_slots_frozen)
class _PipelineAnd:
    left: _Pipeline[Any, Any]
    right: _Pipeline[Any, Any]

@dataclass(**_slots_frozen)
class _Eq:
    value: Any

@dataclass(**_slots_frozen)
class _NotEq:
    value: Any

@dataclass(**_slots_frozen)
class _In:
    values: Container[Any]

@dataclass(**_slots_frozen)
class _NotIn:
    values: Container[Any]
_ConstraintAnnotation = Union[annotated_types.Le, annotated_types.Ge, annotated_types.Lt, annotated_types.Gt, annotated_types.Len, annotated_types.MultipleOf, annotated_types.Timezone, annotated_types.Interval, annotated_types.Predicate, _Eq, _NotEq, _In, _NotIn, Pattern[str]]

@dataclass(**_slots_frozen)
class _Constraint:
    constraint: _ConstraintAnnotation
_Step = Union[_ValidateAs, _ValidateAsDefer, _Transform, _PipelineOr, _PipelineAnd, _Constraint]
_InT = TypeVar('_InT')
_OutT = TypeVar('_OutT')
_NewOutT = TypeVar('_NewOutT')

class _FieldTypeMarker:
    pass

@dataclass(**_slots_true)
class _Pipeline(Generic[_InT, _OutT]):
    """Abstract representation of a chain of validation, transformation, and parsing steps."""
    _steps: tuple[_Step, ...]

    def transform(self, func: Callable[[_OutT], _NewOutT]) -> _Pipeline[_InT, _NewOutT]:
        """Transform the output of the previous step.

        If used as the first step in a pipeline, the type of the field is used.
        That is, the transformation is applied to after the value is parsed to the field's type.
        """
        return _Pipeline(self._steps + (_Transform(func),))

    def validate_as(self, tp: type[_NewOutT] | EllipsisType, *, strict: bool=False) -> _Pipeline[_InT, Any]:
        """Validate / parse the input into a new type.

        If no type is provided, the type of the field is used.

        Types are parsed in Pydantic's `lax` mode by default,
        but you can enable `strict` mode by passing `strict=True`.
        """
        return _Pipeline(self._steps + (_ValidateAs(tp, strict),))

    def validate_as_deferred(self, func: Callable[[], type[_NewOutT]]) -> _Pipeline[_InT, _NewOutT]:
        """Parse the input into a new type, deferring resolution of the type until the current class
        is fully defined.

        This is useful when you need to reference the class in it's own type annotations.
        """
        return _Pipeline(self._steps + (_ValidateAsDefer(func),))

    def constrain(self, constraint: _ConstraintAnnotation) -> _Pipeline[_InT, _OutT]:
        """Constrain a value to meet a certain condition.

        We support most conditions from `annotated_types`, as well as regular expressions.

        Most of the time you'll be calling a shortcut method like `gt`, `lt`, `len`, etc
        so you don't need to call this directly.
        """
        return _Pipeline(self._steps + (_Constraint(constraint),))

    def predicate(self: _Pipeline[_InT, _NewOutT], func: Callable[[_NewOutT], bool]) -> _Pipeline[_InT, _NewOutT]:
        """Constrain a value to meet a certain predicate."""
        return self.constrain(annotated_types.Predicate(func))

    def gt(self: _Pipeline[_InT, _NewOutGt], gt: _NewOutGt) -> _Pipeline[_InT, _NewOutGt]:
        """Constrain a value to be greater than a certain value."""
        return self.constrain(annotated_types.Gt(gt))

    def lt(self: _Pipeline[_InT, _NewOutLt], lt: _NewOutLt) -> _Pipeline[_InT, _NewOutLt]:
        """Constrain a value to be less than a certain value."""
        return self.constrain(annotated_types.Lt(lt))

    def ge(self: _Pipeline[_InT, _NewOutGe], ge: _NewOutGe) -> _Pipeline[_InT, _NewOutGe]:
        """Constrain a value to be greater than or equal to a certain value."""
        return self.constrain(annotated_types.Ge(ge))

    def le(self: _Pipeline[_InT, _NewOutLe], le: _NewOutLe) -> _Pipeline[_InT, _NewOutLe]:
        """Constrain a value to be less than or equal to a certain value."""
        return self.constrain(annotated_types.Le(le))

    def len(self: _Pipeline[_InT, _NewOutLen], min_len: int, max_len: int | None=None) -> _Pipeline[_InT, _NewOutLen]:
        """Constrain a value to have a certain length."""
        return self.constrain(annotated_types.Len(min_len, max_len))

    def multiple_of(self: _Pipeline[_InT, Any], multiple_of: Any) -> _Pipeline[_InT, Any]:
        """Constrain a value to be a multiple of a certain number."""
        return self.constrain(annotated_types.MultipleOf(multiple_of))

    def eq(self: _Pipeline[_InT, _OutT], value: _OutT) -> _Pipeline[_InT, _OutT]:
        """Constrain a value to be equal to a certain value."""
        return self.constrain(_Eq(value))

    def not_eq(self: _Pipeline[_InT, _OutT], value: _OutT) -> _Pipeline[_InT, _OutT]:
        """Constrain a value to not be equal to a certain value."""
        return self.constrain(_NotEq(value))

    def in_(self: _Pipeline[_InT, _OutT], values: Container[_OutT]) -> _Pipeline[_InT, _OutT]:
        """Constrain a value to be in a certain set."""
        return self.constrain(_In(values))

    def not_in(self: _Pipeline[_InT, _OutT], values: Container[_OutT]) -> _Pipeline[_InT, _OutT]:
        """Constrain a value to not be in a certain set."""
        return self.constrain(_NotIn(values))

    def otherwise(self, other: _Pipeline[_OtherIn, _OtherOut]) -> _Pipeline[_InT | _OtherIn, _OutT | _OtherOut]:
        """Combine two validation chains, returning the result of the first chain if it succeeds, and the second chain if it fails."""
        return _Pipeline((_PipelineOr(self, other),))
    __or__ = otherwise

    def then(self, other: _Pipeline[_OutT, _OtherOut]) -> _Pipeline[_InT, _OtherOut]:
        """Pipe the result of one validation chain into another."""
        return _Pipeline((_PipelineAnd(self, other),))
    __and__ = then

    def __get_pydantic_core_schema__(self, source_type: Any, handler: GetCoreSchemaHandler) -> cs.CoreSchema:
        from pydantic_core import core_schema as cs
        queue = deque(self._steps)
        s = None
        while queue:
            step = queue.popleft()
            s = _apply_step(step, s, handler, source_type)
        s = s or cs.any_schema()
        return s

    def __supports_type__(self, _: _OutT) -> bool:
        raise NotImplementedError
validate_as = _Pipeline[Any, Any](()).validate_as
validate_as_deferred = _Pipeline[Any, Any](()).validate_as_deferred
transform = _Pipeline[Any, Any]((_ValidateAs(_FieldTypeMarker),)).transform

def _apply_constraint(s: cs.CoreSchema | None, constraint: _ConstraintAnnotation) -> cs.CoreSchema:
    """Apply a single constraint to a schema."""
    from pydantic_core import core_schema as cs

    if s is None:
        s = cs.any_schema()

    if isinstance(constraint, (annotated_types.Gt, annotated_types.Ge, annotated_types.Lt, annotated_types.Le)):
        return cs.with_constraint(s, constraint)
    elif isinstance(constraint, annotated_types.Len):
        return cs.with_constraint(s, constraint)
    elif isinstance(constraint, annotated_types.MultipleOf):
        return cs.with_constraint(s, constraint)
    elif isinstance(constraint, annotated_types.Timezone):
        return cs.with_constraint(s, constraint)
    elif isinstance(constraint, annotated_types.Interval):
        return cs.with_constraint(s, constraint)
    elif isinstance(constraint, annotated_types.Predicate):
        return cs.with_constraint(s, constraint)
    elif isinstance(constraint, _Eq):
        return cs.with_constraint(s, lambda x: x == constraint.value)
    elif isinstance(constraint, _NotEq):
        return cs.with_constraint(s, lambda x: x != constraint.value)
    elif isinstance(constraint, _In):
        return cs.with_constraint(s, lambda x: x in constraint.values)
    elif isinstance(constraint, _NotIn):
        return cs.with_constraint(s, lambda x: x not in constraint.values)
    elif isinstance(constraint, Pattern):
        return cs.with_constraint(s, lambda x: bool(constraint.match(x)))
    else:
        raise ValueError(f"Unsupported constraint type: {type(constraint)}")

class _SupportsRange(annotated_types.SupportsLe, annotated_types.SupportsGe, Protocol):
    pass

class _SupportsLen(Protocol):

    def __len__(self) -> int:
        ...
_NewOutGt = TypeVar('_NewOutGt', bound=annotated_types.SupportsGt)
_NewOutGe = TypeVar('_NewOutGe', bound=annotated_types.SupportsGe)
_NewOutLt = TypeVar('_NewOutLt', bound=annotated_types.SupportsLt)
_NewOutLe = TypeVar('_NewOutLe', bound=annotated_types.SupportsLe)
_NewOutLen = TypeVar('_NewOutLen', bound=_SupportsLen)
_NewOutDiv = TypeVar('_NewOutDiv', bound=annotated_types.SupportsDiv)
_NewOutMod = TypeVar('_NewOutMod', bound=annotated_types.SupportsMod)
_NewOutDatetime = TypeVar('_NewOutDatetime', bound=datetime.datetime)
_NewOutInterval = TypeVar('_NewOutInterval', bound=_SupportsRange)
_OtherIn = TypeVar('_OtherIn')
_OtherOut = TypeVar('_OtherOut')
