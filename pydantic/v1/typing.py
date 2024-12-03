import sys
import typing
from collections.abc import Callable
from os import PathLike
from typing import TYPE_CHECKING, AbstractSet, Any, Callable as TypingCallable, ClassVar, Dict, ForwardRef, Generator, Iterable, List, Mapping, NewType, Optional, Sequence, Set, Tuple, Type, TypeVar, Union, _eval_type, cast, get_type_hints
from typing_extensions import Annotated, Final, Literal, NotRequired as TypedDictNotRequired, Required as TypedDictRequired
try:
    from typing import _TypingBase as typing_base
except ImportError:
    from typing import _Final as typing_base
try:
    from typing import GenericAlias as TypingGenericAlias
except ImportError:
    TypingGenericAlias = ()
try:
    from types import UnionType as TypesUnionType
except ImportError:
    TypesUnionType = ()
if sys.version_info < (3, 9):
if sys.version_info < (3, 9):
    get_all_type_hints = get_type_hints
_T = TypeVar('_T')
AnyCallable = TypingCallable[..., Any]
NoArgAnyCallable = TypingCallable[[], Any]
AnyArgTCallable = TypingCallable[..., _T]
AnnotatedTypeNames = {'AnnotatedMeta', '_AnnotatedAlias'}
LITERAL_TYPES: Set[Any] = {Literal}
if hasattr(typing, 'Literal'):
    LITERAL_TYPES.add(typing.Literal)
if sys.version_info < (3, 8):
else:
    from typing import get_origin as _typing_get_origin

    def get_origin(tp: Type[Any]) -> Optional[Type[Any]]:
        """
        We can't directly use `typing.get_origin` since we need a fallback to support
        custom generic classes like `ConstrainedList`
        It should be useless once https://github.com/cython/cython/issues/3537 is
        solved and https://github.com/pydantic/pydantic/pull/1753 is merged.
        """
        origin = _typing_get_origin(tp)
        if origin is None and hasattr(tp, '__origin__'):
            origin = tp.__origin__
        return origin
if sys.version_info < (3, 8):
    from typing import _GenericAlias

    def get_args(t: Type[Any]) -> Tuple[Any, ...]:
        """Compatibility version of get_args for python 3.7.

        Mostly compatible with the python 3.8 `typing` module version
        and able to handle almost all use cases.
        """
        pass
else:
    from typing import get_args as _typing_get_args

    def _generic_get_args(tp: Type[Any]) -> Tuple[Any, ...]:
        """
        In python 3.9, `typing.Dict`, `typing.List`, ...
        do have an empty `__args__` by default (instead of the generic ~T for example).
        In order to still support `Dict` for example and consider it as `Dict[Any, Any]`,
        we retrieve the `_nparams` value that tells us how many parameters it needs.
        """
        if hasattr(tp, '_nparams'):
            return (Any,) * tp._nparams
        return ()

    def get_args(tp: Type[Any]) -> Tuple[Any, ...]:
        """Get type arguments with all substitutions performed.

        For unions, basic simplifications used by Union constructor are performed.
        Examples::
            get_args(Dict[str, int]) == (str, int)
            get_args(int) == ()
            get_args(Union[int, Union[T, int], str][int]) == (int, str)
            get_args(Union[int, Tuple[T, int]][str]) == (int, Tuple[str, int])
            get_args(Callable[[], T][int]) == ([], int)
        """
        args = _typing_get_args(tp)
        if not args and hasattr(tp, '__args__'):
            args = tp.__args__
        return args if args else ()
if sys.version_info < (3, 9):

    def convert_generics(tp: Type[Any]) -> Type[Any]:
        """Python 3.9 and older only supports generics from `typing` module.
        They convert strings to ForwardRef automatically.

        Examples::
            typing.List['Hero'] == typing.List[ForwardRef('Hero')]
        """
        pass
else:
    from typing import _UnionGenericAlias
    from typing_extensions import _AnnotatedAlias

    def convert_generics(tp: Type[Any]) -> Type[Any]:
        """
        Recursively searches for `str` type hints and replaces them with ForwardRef.

        Examples::
            convert_generics(list['Hero']) == list[ForwardRef('Hero')]
            convert_generics(dict['Hero', 'Team']) == dict[ForwardRef('Hero'), ForwardRef('Team')]
            convert_generics(typing.Dict['Hero', 'Team']) == typing.Dict[ForwardRef('Hero'), ForwardRef('Team')]
            convert_generics(list[str | 'Hero'] | int) == list[str | ForwardRef('Hero')] | int
        """
        if isinstance(tp, str):
            return ForwardRef(tp)
    
        origin = get_origin(tp)
        if origin is None:
            return tp
    
        args = get_args(tp)
        if not args:
            return tp
    
        converted_args = tuple(convert_generics(arg) for arg in args)
        if converted_args == args:
            return tp
    
        return origin[converted_args]
if sys.version_info < (3, 10):
    WithArgsTypes = (TypingGenericAlias,)
else:
    import types
    import typing
    WithArgsTypes = (typing._GenericAlias, types.GenericAlias, types.UnionType)
StrPath = Union[str, PathLike]
if TYPE_CHECKING:
    from pydantic.v1.fields import ModelField
    TupleGenerator = Generator[Tuple[str, Any], None, None]
    DictStrAny = Dict[str, Any]
    DictAny = Dict[Any, Any]
    SetStr = Set[str]
    ListStr = List[str]
    IntStr = Union[int, str]
    AbstractSetIntStr = AbstractSet[IntStr]
    DictIntStrAny = Dict[IntStr, Any]
    MappingIntStrAny = Mapping[IntStr, Any]
    CallableGenerator = Generator[AnyCallable, None, None]
    ReprArgs = Sequence[Tuple[Optional[str], Any]]
    MYPY = False
    if MYPY:
        AnyClassMethod = classmethod[Any]
    else:
        AnyClassMethod = classmethod[Any, Any, Any]
__all__ = ('AnyCallable', 'NoArgAnyCallable', 'NoneType', 'is_none_type', 'display_as_type', 'resolve_annotations', 'is_callable_type', 'is_literal_type', 'all_literal_values', 'is_namedtuple', 'is_typeddict', 'is_typeddict_special', 'is_new_type', 'new_type_supertype', 'is_classvar', 'is_finalvar', 'update_field_forward_refs', 'update_model_forward_refs', 'TupleGenerator', 'DictStrAny', 'DictAny', 'SetStr', 'ListStr', 'IntStr', 'AbstractSetIntStr', 'DictIntStrAny', 'CallableGenerator', 'ReprArgs', 'AnyClassMethod', 'CallableGenerator', 'WithArgsTypes', 'get_args', 'get_origin', 'get_sub_types', 'typing_base', 'get_all_type_hints', 'is_union', 'StrPath', 'MappingIntStrAny')
NoneType = None.__class__
NONE_TYPES: Tuple[Any, Any, Any] = (None, NoneType, Literal[None])
if sys.version_info < (3, 8):
elif sys.version_info[:2] == (3, 8):

def resolve_annotations(raw_annotations: Dict[str, Type[Any]], module_name: Optional[str]) -> Dict[str, Type[Any]]:
    """
    Partially taken from typing.get_type_hints.

    Resolve string or ForwardRef annotations into type objects if possible.
    """
    resolved_annotations = {}
    for name, value in raw_annotations.items():
        if isinstance(value, str):
            try:
                value = ForwardRef(value)
            except TypeError:
                # TypeErrors can be raised when using Literal['a', 'b', ...]
                # Treat it as a string literal
                resolved_annotations[name] = value
                continue
        try:
            value = _eval_type(value, globals(), locals())
        except NameError:
            # If a NameError is raised, it's probably because the annotation
            # refers to a name that's not available in the module yet.
            # We'll leave it as a string.
            pass
        resolved_annotations[name] = value
    return resolved_annotations

def all_literal_values(type_: Type[Any]) -> Tuple[Any, ...]:
    """
    This method is used to retrieve all Literal values as
    Literal can be used recursively (see https://www.python.org/dev/peps/pep-0586)
    e.g. `Literal[Literal[Literal[1, 2, 3], "foo"], 5, None]`
    """
    if not is_literal_type(type_):
        raise ValueError(f"Expected Literal type, got {type_}")

    values = []
    for arg in get_args(type_):
        if is_literal_type(arg):
            values.extend(all_literal_values(arg))
        else:
            values.append(arg)
    return tuple(values)

def is_namedtuple(type_: Type[Any]) -> bool:
    """
    Check if a given class is a named tuple.
    It can be either a `typing.NamedTuple` or `collections.namedtuple`
    """
    return (
        isinstance(type_, type) and
        issubclass(type_, tuple) and
        hasattr(type_, '_fields') and
        hasattr(type_, '_field_defaults') and
        hasattr(type_, '_asdict')
    )

def is_typeddict(type_: Type[Any]) -> bool:
    """
    Check if a given class is a typed dict (from `typing` or `typing_extensions`)
    In 3.10, there will be a public method (https://docs.python.org/3.10/library/typing.html#typing.is_typeddict)
    """
    return hasattr(type_, '__annotations__') and hasattr(type_, '__total__')

def is_typeddict_special(type_: Any) -> bool:
    """
    Check if type is a TypedDict special form (Required or NotRequired).
    """
    return type_ in (TypedDictRequired, TypedDictNotRequired)
test_type = NewType('test_type', str)

def is_new_type(type_: Type[Any]) -> bool:
    """
    Check whether type_ was created using typing.NewType
    """
    return hasattr(type_, '__supertype__')

def _check_finalvar(v: Optional[Type[Any]]) -> bool:
    """
    Check if a given type is a `typing.Final` type.
    """
    return v is not None and get_origin(v) is Final

def update_field_forward_refs(field: 'ModelField', globalns: Any, localns: Any) -> None:
    """
    Try to update ForwardRefs on fields based on this ModelField, globalns and localns.
    """
    if field.type_.__class__ == ForwardRef:
        field.type_ = field.type_._evaluate(globalns, localns or None)
        field.prepare()
    if field.sub_fields:
        for sub_field in field.sub_fields:
            update_field_forward_refs(sub_field, globalns=globalns, localns=localns)

def update_model_forward_refs(model: Type[Any], fields: Iterable['ModelField'], json_encoders: Dict[Union[Type[Any], str, ForwardRef], AnyCallable], localns: 'DictStrAny', exc_to_suppress: Tuple[Type[BaseException], ...]=()) -> None:
    """
    Try to update model fields ForwardRefs based on model and localns.
    """
    try:
        model_name = model.__name__
        if model.__module__ != 'typing':
            localns = {**localns, model_name: model}
        globalns = sys.modules[model.__module__].__dict__
    except AttributeError:
        globalns = {}

    for field in fields:
        try:
            update_field_forward_refs(field, globalns=globalns, localns=localns)
        except exc_to_suppress:
            pass

    for key, json_encoder in json_encoders.items():
        if isinstance(key, (ForwardRef, str)):
            try:
                json_encoders[evaluate_forwardref(key, globalns, localns)] = json_encoder
            except exc_to_suppress:
                continue

def get_class(type_: Type[Any]) -> Union[None, bool, Type[Any]]:
    """
    Tries to get the class of a Type[T] annotation. Returns True if Type is used
    without brackets. Otherwise returns None.
    """
    if type_ is Type:
        return True
    if get_origin(type_) is Type:
        args = get_args(type_)
        if len(args) == 1:
            return args[0]
    return None

def get_sub_types(tp: Any) -> List[Any]:
    """
    Return all the types that are allowed by type `tp`
    `tp` can be a `Union` of allowed types or an `Annotated` type
    """
    origin = get_origin(tp)
    if origin is Union:
        return list(get_args(tp))
    elif origin is Annotated:
        return get_sub_types(get_args(tp)[0])
    return [tp]
