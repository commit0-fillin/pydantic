from __future__ import annotations
import sys
import types
import typing
from collections import ChainMap
from contextlib import contextmanager
from contextvars import ContextVar
from types import prepare_class
from typing import TYPE_CHECKING, Any, Iterator, List, Mapping, MutableMapping, Tuple, TypeVar
from weakref import WeakValueDictionary
import typing_extensions
from ._core_utils import get_type_ref
from ._forward_ref import PydanticRecursiveRef
from ._typing_extra import TypeVarType, typing_base
from ._utils import all_identical, is_model_class
if sys.version_info >= (3, 10):
    from typing import _UnionGenericAlias
if TYPE_CHECKING:
    from ..main import BaseModel
GenericTypesCacheKey = Tuple[Any, Any, Tuple[Any, ...]]
KT = TypeVar('KT')
VT = TypeVar('VT')
_LIMITED_DICT_SIZE = 100
if TYPE_CHECKING:

    class LimitedDict(dict, MutableMapping[KT, VT]):

        def __init__(self, size_limit: int=_LIMITED_DICT_SIZE):
            ...
else:

    class LimitedDict(dict):
        """Limit the size/length of a dict used for caching to avoid unlimited increase in memory usage.

        Since the dict is ordered, and we always remove elements from the beginning, this is effectively a FIFO cache.
        """

        def __init__(self, size_limit: int=_LIMITED_DICT_SIZE):
            self.size_limit = size_limit
            super().__init__()

        def __setitem__(self, key: Any, value: Any, /) -> None:
            super().__setitem__(key, value)
            if len(self) > self.size_limit:
                excess = len(self) - self.size_limit + self.size_limit // 10
                to_remove = list(self.keys())[:excess]
                for k in to_remove:
                    del self[k]
if sys.version_info >= (3, 9):
    GenericTypesCache = WeakValueDictionary[GenericTypesCacheKey, 'type[BaseModel]']
else:
    GenericTypesCache = WeakValueDictionary
if TYPE_CHECKING:

    class DeepChainMap(ChainMap[KT, VT]):
        ...
else:

    class DeepChainMap(ChainMap):
        """Variant of ChainMap that allows direct updates to inner scopes.

        Taken from https://docs.python.org/3/library/collections.html#collections.ChainMap,
        with some light modifications for this use case.
        """

        def __setitem__(self, key: KT, value: VT) -> None:
            for mapping in self.maps:
                mapping[key] = value

        def __delitem__(self, key: KT) -> None:
            hit = False
            for mapping in self.maps:
                if key in mapping:
                    del mapping[key]
                    hit = True
            if not hit:
                raise KeyError(key)
_GENERIC_TYPES_CACHE = GenericTypesCache()

class PydanticGenericMetadata(typing_extensions.TypedDict):
    origin: type[BaseModel] | None
    args: tuple[Any, ...]
    parameters: tuple[type[Any], ...]

def create_generic_submodel(model_name: str, origin: type[BaseModel], args: tuple[Any, ...], params: tuple[Any, ...]) -> type[BaseModel]:
    """Dynamically create a submodel of a provided (generic) BaseModel.

    This is used when producing concrete parametrizations of generic models. This function
    only *creates* the new subclass; the schema/validators/serialization must be updated to
    reflect a concrete parametrization elsewhere.

    Args:
        model_name: The name of the newly created model.
        origin: The base class for the new model to inherit from.
        args: A tuple of generic metadata arguments.
        params: A tuple of generic metadata parameters.

    Returns:
        The created submodel.
    """
    namespace = {
        '__module__': origin.__module__,
        '__qualname__': f'{origin.__qualname__}[{", ".join(str(arg) for arg in args)}]',
        '__pydantic_generic_metadata__': PydanticGenericMetadata(
            origin=origin,
            args=args,
            parameters=params,
        ),
    }
    return types.new_class(model_name, (origin,), {}, lambda ns: ns.update(namespace))

def _get_caller_frame_info(depth: int=2) -> tuple[str | None, bool]:
    """Used inside a function to check whether it was called globally.

    Args:
        depth: The depth to get the frame.

    Returns:
        A tuple contains `module_name` and `called_globally`.

    Raises:
        RuntimeError: If the function is not called inside a function.
    """
    try:
        frame = sys._getframe(depth)
    except ValueError as e:
        raise RuntimeError('This function must be called inside another function') from e
    
    module_name = frame.f_globals.get('__name__')
    called_globally = frame.f_locals is frame.f_globals
    return module_name, called_globally
DictValues: type[Any] = {}.values().__class__

def iter_contained_typevars(v: Any) -> Iterator[TypeVarType]:
    """Recursively iterate through all subtypes and type args of `v` and yield any typevars that are found.

    This is inspired as an alternative to directly accessing the `__parameters__` attribute of a GenericAlias,
    since __parameters__ of (nested) generic BaseModel subclasses won't show up in that list.
    """
    if isinstance(v, TypeVar):
        yield v
    elif is_model_class(v) and hasattr(v, '__pydantic_generic_metadata__'):
        yield from v.__pydantic_generic_metadata__['parameters']
    elif hasattr(v, '__origin__') and hasattr(v, '__args__'):
        for arg in v.__args__:
            yield from iter_contained_typevars(arg)
    elif isinstance(v, (list, tuple)):
        for item in v:
            yield from iter_contained_typevars(item)

def get_standard_typevars_map(cls: type[Any]) -> dict[TypeVarType, Any] | None:
    """Package a generic type's typevars and parametrization (if present) into a dictionary compatible with the
    `replace_types` function. Specifically, this works with standard typing generics and typing._GenericAlias.
    """
    if not hasattr(cls, '__parameters__') or not hasattr(cls, '__args__'):
        return None
    
    parameters = getattr(cls, '__parameters__', None)
    args = getattr(cls, '__args__', None)
    
    if parameters is None or args is None:
        return None
    
    return {p: a for p, a in zip(parameters, args) if isinstance(p, TypeVar)}

def get_model_typevars_map(cls: type[BaseModel]) -> dict[TypeVarType, Any] | None:
    """Package a generic BaseModel's typevars and concrete parametrization (if present) into a dictionary compatible
    with the `replace_types` function.

    Since BaseModel.__class_getitem__ does not produce a typing._GenericAlias, and the BaseModel generic info is
    stored in the __pydantic_generic_metadata__ attribute, we need special handling here.
    """
    if not hasattr(cls, '__pydantic_generic_metadata__'):
        return None
    
    metadata = cls.__pydantic_generic_metadata__
    parameters = metadata.get('parameters')
    args = metadata.get('args')
    
    if parameters is None or args is None:
        return None
    
    return {p: a for p, a in zip(parameters, args) if isinstance(p, TypeVar)}

def replace_types(type_: Any, type_map: Mapping[Any, Any] | None) -> Any:
    """Return type with all occurrences of `type_map` keys recursively replaced with their values.

    Args:
        type_: The class or generic alias.
        type_map: Mapping from `TypeVar` instance to concrete types.

    Returns:
        A new type representing the basic structure of `type_` with all
        `typevar_map` keys recursively replaced.

    Example:
        ```py
        from typing import List, Tuple, Union

        from pydantic._internal._generics import replace_types

        replace_types(Tuple[str, Union[List[str], float]], {str: int})
        #> Tuple[int, Union[List[int], float]]
        ```
    """
    if type_map is None:
        return type_

    if isinstance(type_, TypeVar):
        return type_map.get(type_, type_)

    if hasattr(type_, '__origin__') and hasattr(type_, '__args__'):
        args = tuple(replace_types(arg, type_map) for arg in type_.__args__)
        if args == type_.__args__:
            return type_
        return type_.__origin__[args]

    if isinstance(type_, types.GenericAlias):
        args = tuple(replace_types(arg, type_map) for arg in type_.__args__)
        if args == type_.__args__:
            return type_
        return types.GenericAlias(type_.__origin__, args)

    return type_

def has_instance_in_type(type_: Any, isinstance_target: Any) -> bool:
    """Checks if the type, or any of its arbitrary nested args, satisfy
    `isinstance(<type>, isinstance_target)`.
    """
    if isinstance(type_, isinstance_target):
        return True
    
    if hasattr(type_, '__args__'):
        return any(has_instance_in_type(arg, isinstance_target) for arg in type_.__args__)
    
    return False

def check_parameters_count(cls: type[BaseModel], parameters: tuple[Any, ...]) -> None:
    """Check the generic model parameters count is equal.

    Args:
        cls: The generic model.
        parameters: A tuple of passed parameters to the generic model.

    Raises:
        TypeError: If the passed parameters count is not equal to generic model parameters count.
    """
    expected_params_len = len(cls.__pydantic_generic_metadata__['parameters'])
    if len(parameters) != expected_params_len:
        raise TypeError(f'Too {"many" if len(parameters) > expected_params_len else "few"} parameters for {cls.__name__}; actual {len(parameters)}, expected {expected_params_len}')
_generic_recursion_cache: ContextVar[set[str] | None] = ContextVar('_generic_recursion_cache', default=None)

@contextmanager
def generic_recursion_self_type(origin: type[BaseModel], args: tuple[Any, ...]) -> Iterator[PydanticRecursiveRef | None]:
    """This contextmanager should be placed around the recursive calls used to build a generic type,
    and accept as arguments the generic origin type and the type arguments being passed to it.

    If the same origin and arguments are observed twice, it implies that a self-reference placeholder
    can be used while building the core schema, and will produce a schema_ref that will be valid in the
    final parent schema.
    """
    cache = _generic_recursion_cache.get()
    if cache is None:
        cache = set()
        token = _generic_recursion_cache.set(cache)
    else:
        token = None

    key = f'{origin.__name__}[{", ".join(str(arg) for arg in args)}]'
    if key in cache:
        yield PydanticRecursiveRef(key)
    else:
        cache.add(key)
        try:
            yield None
        finally:
            cache.remove(key)

    if token is not None:
        _generic_recursion_cache.reset(token)

def get_cached_generic_type_early(parent: type[BaseModel], typevar_values: Any) -> type[BaseModel] | None:
    """The use of a two-stage cache lookup approach was necessary to have the highest performance possible for
    repeated calls to `__class_getitem__` on generic types (which may happen in tighter loops during runtime),
    while still ensuring that certain alternative parametrizations ultimately resolve to the same type.

    As a concrete example, this approach was necessary to make Model[List[T]][int] equal to Model[List[int]].
    The approach could be modified to not use two different cache keys at different points, but the
    _early_cache_key is optimized to be as quick to compute as possible (for repeated-access speed), and the
    _late_cache_key is optimized to be as "correct" as possible, so that two types that will ultimately be the
    same after resolving the type arguments will always produce cache hits.

    If we wanted to move to only using a single cache key per type, we would either need to always use the
    slower/more computationally intensive logic associated with _late_cache_key, or would need to accept
    that Model[List[T]][int] is a different type than Model[List[T]][int]. Because we rely on subclass relationships
    during validation, I think it is worthwhile to ensure that types that are functionally equivalent are actually
    equal.
    """
    cache_key = _early_cache_key(parent, typevar_values)
    return _GENERIC_TYPES_CACHE.get(cache_key)

def get_cached_generic_type_late(parent: type[BaseModel], typevar_values: Any, origin: type[BaseModel], args: tuple[Any, ...]) -> type[BaseModel] | None:
    """See the docstring of `get_cached_generic_type_early` for more information about the two-stage cache lookup."""
    cache_key = _late_cache_key(origin, args, typevar_values)
    cached_type = _GENERIC_TYPES_CACHE.get(cache_key)
    if cached_type is not None:
        early_key = _early_cache_key(parent, typevar_values)
        _GENERIC_TYPES_CACHE[early_key] = cached_type
    return cached_type

def set_cached_generic_type(parent: type[BaseModel], typevar_values: tuple[Any, ...], type_: type[BaseModel], origin: type[BaseModel] | None=None, args: tuple[Any, ...] | None=None) -> None:
    """See the docstring of `get_cached_generic_type_early` for more information about why items are cached with
    two different keys.
    """
    early_key = _early_cache_key(parent, typevar_values)
    _GENERIC_TYPES_CACHE[early_key] = type_
    
    if origin is not None and args is not None:
        late_key = _late_cache_key(origin, args, typevar_values)
        _GENERIC_TYPES_CACHE[late_key] = type_

def _union_orderings_key(typevar_values: Any) -> Any:
    """This is intended to help differentiate between Union types with the same arguments in different order.

    Thanks to caching internal to the `typing` module, it is not possible to distinguish between
    List[Union[int, float]] and List[Union[float, int]] (and similarly for other "parent" origins besides List)
    because `typing` considers Union[int, float] to be equal to Union[float, int].

    However, you _can_ distinguish between (top-level) Union[int, float] vs. Union[float, int].
    Because we parse items as the first Union type that is successful, we get slightly more consistent behavior
    if we make an effort to distinguish the ordering of items in a union. It would be best if we could _always_
    get the exact-correct order of items in the union, but that would require a change to the `typing` module itself.
    (See https://github.com/python/cpython/issues/86483 for reference.)
    """
    if sys.version_info >= (3, 10) and isinstance(typevar_values, _UnionGenericAlias):
        return tuple(get_type_ref(arg) for arg in typevar_values.__args__)
    elif hasattr(typevar_values, '__origin__') and typevar_values.__origin__ is typing.Union:
        return tuple(get_type_ref(arg) for arg in typevar_values.__args__)
    else:
        return typevar_values

def _early_cache_key(cls: type[BaseModel], typevar_values: Any) -> GenericTypesCacheKey:
    """This is intended for minimal computational overhead during lookups of cached types.

    Note that this is overly simplistic, and it's possible that two different cls/typevar_values
    inputs would ultimately result in the same type being created in BaseModel.__class_getitem__.
    To handle this, we have a fallback _late_cache_key that is checked later if the _early_cache_key
    lookup fails, and should result in a cache hit _precisely_ when the inputs to __class_getitem__
    would result in the same type.
    """
    return (cls, id(cls), _union_orderings_key(typevar_values))

def _late_cache_key(origin: type[BaseModel], args: tuple[Any, ...], typevar_values: Any) -> GenericTypesCacheKey:
    """This is intended for use later in the process of creating a new type, when we have more information
    about the exact args that will be passed. If it turns out that a different set of inputs to
    __class_getitem__ resulted in the same inputs to the generic type creation process, we can still
    return the cached type, and update the cache with the _early_cache_key as well.
    """
    return (origin, id(origin), tuple(get_type_ref(arg) for arg in args), _union_orderings_key(typevar_values))
