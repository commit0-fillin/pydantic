import keyword
import warnings
import weakref
from collections import OrderedDict, defaultdict, deque
from copy import deepcopy
from itertools import islice, zip_longest
from types import BuiltinFunctionType, CodeType, FunctionType, GeneratorType, LambdaType, ModuleType
from typing import TYPE_CHECKING, AbstractSet, Any, Callable, Collection, Dict, Generator, Iterable, Iterator, List, Mapping, NoReturn, Optional, Set, Tuple, Type, TypeVar, Union
from typing_extensions import Annotated
from pydantic.v1.errors import ConfigError
from pydantic.v1.typing import NoneType, WithArgsTypes, all_literal_values, display_as_type, get_args, get_origin, is_literal_type, is_union
from pydantic.v1.version import version_info
if TYPE_CHECKING:
    from inspect import Signature
    from pathlib import Path
    from pydantic.v1.config import BaseConfig
    from pydantic.v1.dataclasses import Dataclass
    from pydantic.v1.fields import ModelField
    from pydantic.v1.main import BaseModel
    from pydantic.v1.typing import AbstractSetIntStr, DictIntStrAny, IntStr, MappingIntStrAny, ReprArgs
    RichReprResult = Iterable[Union[Any, Tuple[Any], Tuple[str, Any], Tuple[str, Any, Any]]]
__all__ = ('import_string', 'sequence_like', 'validate_field_name', 'lenient_isinstance', 'lenient_issubclass', 'in_ipython', 'is_valid_identifier', 'deep_update', 'update_not_none', 'almost_equal_floats', 'get_model', 'to_camel', 'is_valid_field', 'smart_deepcopy', 'PyObjectStr', 'Representation', 'GetterDict', 'ValueItems', 'version_info', 'ClassAttribute', 'path_type', 'ROOT_KEY', 'get_unique_discriminator_alias', 'get_discriminator_alias_and_values', 'DUNDER_ATTRIBUTES')
ROOT_KEY = '__root__'
IMMUTABLE_NON_COLLECTIONS_TYPES: Set[Type[Any]] = {int, float, complex, str, bool, bytes, type, NoneType, FunctionType, BuiltinFunctionType, LambdaType, weakref.ref, CodeType, ModuleType, NotImplemented.__class__, Ellipsis.__class__}
BUILTIN_COLLECTIONS: Set[Type[Any]] = {list, set, tuple, frozenset, dict, OrderedDict, defaultdict, deque}

def import_string(dotted_path: str) -> Any:
    """
    Stolen approximately from django. Import a dotted module path and return the attribute/class designated by the
    last name in the path. Raise ImportError if the import fails.
    """
    try:
        module_path, class_name = dotted_path.rsplit('.', 1)
    except ValueError as err:
        raise ImportError("%s doesn't look like a module path" % dotted_path) from err

    module = __import__(module_path, fromlist=[class_name])
    try:
        return getattr(module, class_name)
    except AttributeError as err:
        raise ImportError('Module "%s" does not define a "%s" attribute/class' % (module_path, class_name)) from err

def truncate(v: Union[str], *, max_len: int=80) -> str:
    """
    Truncate a value and add a unicode ellipsis (three dots) to the end if it was too long
    """
    s = str(v)
    if len(s) <= max_len:
        return s
    return s[:max_len - 1].rstrip() + '…'

def validate_field_name(bases: List[Type['BaseModel']], field_name: str) -> None:
    """
    Ensure that the field's name does not shadow an existing attribute of the model.
    """
    for base in bases:
        if getattr(base, field_name, None):
            raise NameError(
                f'Field name "{field_name}" shadows an attribute in parent "{base.__name__}"'
            )

def in_ipython() -> bool:
    """
    Check whether we're in an ipython environment, including jupyter notebooks.
    """
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False

def is_valid_identifier(identifier: str) -> bool:
    """
    Checks that a string is a valid identifier and not a Python keyword.
    :param identifier: The identifier to test.
    :return: True if the identifier is valid.
    """
    return identifier.isidentifier() and not keyword.iskeyword(identifier)
KeyType = TypeVar('KeyType')

def almost_equal_floats(value_1: float, value_2: float, *, delta: float=1e-08) -> bool:
    """
    Return True if two floats are almost equal
    """
    return abs(value_1 - value_2) < delta

def generate_model_signature(init: Callable[..., None], fields: Dict[str, 'ModelField'], config: Type['BaseConfig']) -> 'Signature':
    """
    Generate signature for model based on its fields
    """
    from inspect import Parameter, Signature

    parameters = []
    for name, field in fields.items():
        default = Parameter.empty if field.required else field.default
        parameters.append(Parameter(name, Parameter.KEYWORD_ONLY, default=default, annotation=field.annotation))
    
    if config.extra == 'allow':
        parameters.append(Parameter('**extra', Parameter.VAR_KEYWORD))

    return Signature(parameters)
T = TypeVar('T')

def unique_list(input_list: Union[List[T], Tuple[T, ...]], *, name_factory: Callable[[T], str]=str) -> List[T]:
    """
    Make a list unique while maintaining order.
    We update the list if another one with the same name is set
    (e.g. root validator overridden in subclass)
    """
    result = []
    seen = set()
    for item in input_list:
        name = name_factory(item)
        if name not in seen:
            seen.add(name)
            result.append(item)
        else:
            # Update existing item if a new one with the same name is found
            index = next(i for i, x in enumerate(result) if name_factory(x) == name)
            result[index] = item
    return result

class PyObjectStr(str):
    """
    String class where repr doesn't include quotes. Useful with Representation when you want to return a string
    representation of something that valid (or pseudo-valid) python.
    """

    def __repr__(self) -> str:
        return str(self)

class Representation:
    """
    Mixin to provide __str__, __repr__, and __pretty__ methods. See #884 for more details.

    __pretty__ is used by [devtools](https://python-devtools.helpmanual.io/) to provide human readable representations
    of objects.
    """
    __slots__: Tuple[str, ...] = tuple()

    def __repr_args__(self) -> 'ReprArgs':
        """
        Returns the attributes to show in __str__, __repr__, and __pretty__ this is generally overridden.

        Can either return:
        * name - value pairs, e.g.: `[('foo_name', 'foo'), ('bar_name', ['b', 'a', 'r'])]`
        * or, just values, e.g.: `[(None, 'foo'), (None, ['b', 'a', 'r'])]`
        """
        attrs = ((s, getattr(self, s)) for s in self.__slots__)
        return [(a, v) for a, v in attrs if v is not None]

    def __repr_name__(self) -> str:
        """
        Name of the instance's class, used in __repr__.
        """
        return self.__class__.__name__

    def __repr_str__(self, join_str: str) -> str:
        return join_str.join((repr(v) if a is None else f'{a}={v!r}' for a, v in self.__repr_args__()))

    def __pretty__(self, fmt: Callable[[Any], Any], **kwargs: Any) -> Generator[Any, None, None]:
        """
        Used by devtools (https://python-devtools.helpmanual.io/) to provide a human readable representations of objects
        """
        yield (self.__repr_name__() + '(')
        yield 1
        for name, value in self.__repr_args__():
            if name is not None:
                yield (name + '=')
            yield fmt(value)
            yield ','
            yield 0
        yield (-1)
        yield ')'

    def __str__(self) -> str:
        return self.__repr_str__(' ')

    def __repr__(self) -> str:
        return f'{self.__repr_name__()}({self.__repr_str__(', ')})'

    def __rich_repr__(self) -> 'RichReprResult':
        """Get fields for Rich library"""
        for name, field_repr in self.__repr_args__():
            if name is None:
                yield field_repr
            else:
                yield (name, field_repr)

class GetterDict(Representation):
    """
    Hack to make object's smell just enough like dicts for validate_model.

    We can't inherit from Mapping[str, Any] because it upsets cython so we have to implement all methods ourselves.
    """
    __slots__ = ('_obj',)

    def __init__(self, obj: Any):
        self._obj = obj

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self._obj, key)
        except AttributeError as e:
            raise KeyError(key) from e

    def extra_keys(self) -> Set[Any]:
        """
        We don't want to get any other attributes of obj if the model didn't explicitly ask for them
        """
        return set()

    def keys(self) -> List[Any]:
        """
        Keys of the pseudo dictionary, uses a list not set so order information can be maintained like python
        dictionaries.
        """
        return [key for key in dir(self._obj) if not key.startswith('_')]

    def __iter__(self) -> Iterator[str]:
        for name in dir(self._obj):
            if not name.startswith('_'):
                yield name

    def __len__(self) -> int:
        return sum((1 for _ in self))

    def __contains__(self, item: Any) -> bool:
        return item in self.keys()

    def __eq__(self, other: Any) -> bool:
        return dict(self) == dict(other.items())

    def __repr_args__(self) -> 'ReprArgs':
        return [(None, dict(self))]

    def __repr_name__(self) -> str:
        return f'GetterDict[{display_as_type(self._obj)}]'

class ValueItems(Representation):
    """
    Class for more convenient calculation of excluded or included fields on values.
    """
    __slots__ = ('_items', '_type')

    def __init__(self, value: Any, items: Union['AbstractSetIntStr', 'MappingIntStrAny']) -> None:
        items = self._coerce_items(items)
        if isinstance(value, (list, tuple)):
            items = self._normalize_indexes(items, len(value))
        self._items: 'MappingIntStrAny' = items

    def is_excluded(self, item: Any) -> bool:
        """
        Check if item is fully excluded.

        :param item: key or index of a value
        """
        return item in self._items and self._items[item] is False

    def is_included(self, item: Any) -> bool:
        """
        Check if value is contained in self._items

        :param item: key or index of value
        """
        return item in self._items and self._items[item] is not False

    def for_element(self, e: 'IntStr') -> Optional[Union['AbstractSetIntStr', 'MappingIntStrAny']]:
        """
        :param e: key or index of element on value
        :return: raw values for element if self._items is dict and contain needed element
        """
        if isinstance(self._items, Mapping):
            return self._items.get(e)
        return None

    def _normalize_indexes(self, items: 'MappingIntStrAny', v_length: int) -> 'DictIntStrAny':
        """
        :param items: dict or set of indexes which will be normalized
        :param v_length: length of sequence indexes of which will be

        >>> self._normalize_indexes({0: True, -2: True, -1: True}, 4)
        {0: True, 2: True, 3: True}
        >>> self._normalize_indexes({'__all__': True}, 4)
        {0: True, 1: True, 2: True, 3: True}
        """
        if '__all__' in items:
            return {i: True for i in range(v_length)}
        return {(i if i >= 0 else v_length + i): v for i, v in items.items() if -v_length <= i < v_length}

    @classmethod
    def merge(cls, base: Any, override: Any, intersect: bool=False) -> Any:
        """
        Merge a ``base`` item with an ``override`` item.

        Both ``base`` and ``override`` are converted to dictionaries if possible.
        Sets are converted to dictionaries with the sets entries as keys and
        Ellipsis as values.

        Each key-value pair existing in ``base`` is merged with ``override``,
        while the rest of the key-value pairs are updated recursively with this function.

        Merging takes place based on the "union" of keys if ``intersect`` is
        set to ``False`` (default) and on the intersection of keys if
        ``intersect`` is set to ``True``.
        """
        if isinstance(base, set):
            base = {item: ... for item in base}
        if isinstance(override, set):
            override = {item: ... for item in override}
        
        if not isinstance(base, dict) or not isinstance(override, dict):
            return override if override is not None else base

        merged = {}
        keys = base.keys() & override.keys() if intersect else base.keys() | override.keys()
        
        for key in keys:
            if key in base and key in override:
                merged[key] = cls.merge(base[key], override[key], intersect)
            elif key in base:
                merged[key] = base[key]
            else:
                merged[key] = override[key]
        
        return merged

    def __repr_args__(self) -> 'ReprArgs':
        return [(None, self._items)]

class ClassAttribute:
    """
    Hide class attribute from its instances
    """
    __slots__ = ('name', 'value')

    def __init__(self, name: str, value: Any) -> None:
        self.name = name
        self.value = value

    def __get__(self, instance: Any, owner: Type[Any]) -> None:
        if instance is None:
            return self.value
        raise AttributeError(f'{self.name!r} attribute of {owner.__name__!r} is class-only')
path_types = {'is_dir': 'directory', 'is_file': 'file', 'is_mount': 'mount point', 'is_symlink': 'symlink', 'is_block_device': 'block device', 'is_char_device': 'char device', 'is_fifo': 'FIFO', 'is_socket': 'socket'}

def path_type(p: 'Path') -> str:
    """
    Find out what sort of thing a path is.
    """
    for method, description in path_types.items():
        if getattr(p, method)():
            return description
    return 'unknown'
Obj = TypeVar('Obj')

def smart_deepcopy(obj: Obj) -> Obj:
    """
    Return type as is for immutable built-in types
    Use obj.copy() for built-in empty collections
    Use copy.deepcopy() for non-empty collections and unknown objects
    """
    from copy import deepcopy
    
    if type(obj) in IMMUTABLE_NON_COLLECTIONS_TYPES:
        return obj
    if isinstance(obj, BUILTIN_COLLECTIONS):
        return obj.copy() if not obj else deepcopy(obj)
    return deepcopy(obj)
DUNDER_ATTRIBUTES = {'__annotations__', '__classcell__', '__doc__', '__module__', '__orig_bases__', '__orig_class__', '__qualname__'}
_EMPTY = object()

def all_identical(left: Iterable[Any], right: Iterable[Any]) -> bool:
    """
    Check that the items of `left` are the same objects as those in `right`.

    >>> a, b = object(), object()
    >>> all_identical([a, b, a], [a, b, a])
    True
    >>> all_identical([a, b, [a]], [a, b, [a]])  # new list object, while "equal" is not "identical"
    False
    """
    return all(l is r for l, r in zip_longest(left, right, fillvalue=_EMPTY))

def assert_never(obj: NoReturn, msg: str) -> NoReturn:
    """
    Helper to make sure that we have covered all possible types.

    This is mostly useful for ``mypy``, docs:
    https://mypy.readthedocs.io/en/latest/literal_types.html#exhaustive-checks
    """
    raise AssertionError(f"{msg}: {obj}")

def get_unique_discriminator_alias(all_aliases: Collection[str], discriminator_key: str) -> str:
    """Validate that all aliases are the same and if that's the case return the alias"""
    unique_aliases = set(all_aliases)
    if len(unique_aliases) == 0:
        return discriminator_key
    if len(unique_aliases) > 1:
        raise ValueError(f"Multiple aliases found for discriminator {discriminator_key}: {unique_aliases}")
    return next(iter(unique_aliases))

def get_discriminator_alias_and_values(tp: Any, discriminator_key: str) -> Tuple[str, Tuple[str, ...]]:
    """
    Get alias and all valid values in the `Literal` type of the discriminator field
    `tp` can be a `BaseModel` class or directly an `Annotated` `Union` of many.
    """
    from pydantic.v1.fields import ModelField
    from pydantic.v1.main import BaseModel

    if isinstance(tp, type) and issubclass(tp, BaseModel):
        field = tp.__fields__[discriminator_key]
    elif get_origin(tp) is Annotated:
        args = get_args(tp)
        field = ModelField.infer(name=discriminator_key, value=args[0], annotation=args[0])
    else:
        raise ValueError(f"Invalid type {tp} for discriminator {discriminator_key}")

    if not is_literal_type(field.type_):
        raise ValueError(f"Discriminator {discriminator_key} must be a Literal type")

    alias = field.alias or discriminator_key
    values = tuple(str(v) for v in all_literal_values(field.type_))
    return alias, values
