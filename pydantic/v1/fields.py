import copy
import re
from collections import Counter as CollectionCounter, defaultdict, deque
from collections.abc import Callable, Hashable as CollectionsHashable, Iterable as CollectionsIterable
from typing import TYPE_CHECKING, Any, Counter, DefaultDict, Deque, Dict, ForwardRef, FrozenSet, Generator, Iterable, Iterator, List, Mapping, Optional, Pattern, Sequence, Set, Tuple, Type, TypeVar, Union
from typing_extensions import Annotated, Final
from pydantic.v1 import errors as errors_
from pydantic.v1.class_validators import Validator, make_generic_validator, prep_validators
from pydantic.v1.error_wrappers import ErrorWrapper
from pydantic.v1.errors import ConfigError, InvalidDiscriminator, MissingDiscriminator, NoneIsNotAllowedError
from pydantic.v1.types import Json, JsonWrapper
from pydantic.v1.typing import NoArgAnyCallable, convert_generics, display_as_type, get_args, get_origin, is_finalvar, is_literal_type, is_new_type, is_none_type, is_typeddict, is_typeddict_special, is_union, new_type_supertype
from pydantic.v1.utils import PyObjectStr, Representation, ValueItems, get_discriminator_alias_and_values, get_unique_discriminator_alias, lenient_isinstance, lenient_issubclass, sequence_like, smart_deepcopy
from pydantic.v1.validators import constant_validator, dict_validator, find_validators, validate_json
Required: Any = Ellipsis
T = TypeVar('T')

class UndefinedType:

    def __repr__(self) -> str:
        return 'PydanticUndefined'

    def __copy__(self: T) -> T:
        return self

    def __reduce__(self) -> str:
        return 'Undefined'

    def __deepcopy__(self: T, _: Any) -> T:
        return self
Undefined = UndefinedType()
if TYPE_CHECKING:
    from pydantic.v1.class_validators import ValidatorsList
    from pydantic.v1.config import BaseConfig
    from pydantic.v1.error_wrappers import ErrorList
    from pydantic.v1.types import ModelOrDc
    from pydantic.v1.typing import AbstractSetIntStr, MappingIntStrAny, ReprArgs
    ValidateReturn = Tuple[Optional[Any], Optional[ErrorList]]
    LocStr = Union[Tuple[Union[int, str], ...], str]
    BoolUndefined = Union[bool, UndefinedType]

class FieldInfo(Representation):
    """
    Captures extra information about a field.
    """
    __slots__ = ('default', 'default_factory', 'alias', 'alias_priority', 'title', 'description', 'exclude', 'include', 'const', 'gt', 'ge', 'lt', 'le', 'multiple_of', 'allow_inf_nan', 'max_digits', 'decimal_places', 'min_items', 'max_items', 'unique_items', 'min_length', 'max_length', 'allow_mutation', 'repr', 'regex', 'discriminator', 'extra')
    __field_constraints__ = {'min_length': None, 'max_length': None, 'regex': None, 'gt': None, 'lt': None, 'ge': None, 'le': None, 'multiple_of': None, 'allow_inf_nan': None, 'max_digits': None, 'decimal_places': None, 'min_items': None, 'max_items': None, 'unique_items': None, 'allow_mutation': True}

    def __init__(self, default: Any=Undefined, **kwargs: Any) -> None:
        self.default = default
        self.default_factory = kwargs.pop('default_factory', None)
        self.alias = kwargs.pop('alias', None)
        self.alias_priority = kwargs.pop('alias_priority', 2 if self.alias is not None else None)
        self.title = kwargs.pop('title', None)
        self.description = kwargs.pop('description', None)
        self.exclude = kwargs.pop('exclude', None)
        self.include = kwargs.pop('include', None)
        self.const = kwargs.pop('const', None)
        self.gt = kwargs.pop('gt', None)
        self.ge = kwargs.pop('ge', None)
        self.lt = kwargs.pop('lt', None)
        self.le = kwargs.pop('le', None)
        self.multiple_of = kwargs.pop('multiple_of', None)
        self.allow_inf_nan = kwargs.pop('allow_inf_nan', None)
        self.max_digits = kwargs.pop('max_digits', None)
        self.decimal_places = kwargs.pop('decimal_places', None)
        self.min_items = kwargs.pop('min_items', None)
        self.max_items = kwargs.pop('max_items', None)
        self.unique_items = kwargs.pop('unique_items', None)
        self.min_length = kwargs.pop('min_length', None)
        self.max_length = kwargs.pop('max_length', None)
        self.allow_mutation = kwargs.pop('allow_mutation', True)
        self.regex = kwargs.pop('regex', None)
        self.discriminator = kwargs.pop('discriminator', None)
        self.repr = kwargs.pop('repr', True)
        self.extra = kwargs

    def __repr_args__(self) -> 'ReprArgs':
        field_defaults_to_hide: Dict[str, Any] = {'repr': True, **self.__field_constraints__}
        attrs = ((s, getattr(self, s)) for s in self.__slots__)
        return [(a, v) for a, v in attrs if v != field_defaults_to_hide.get(a, None)]

    def get_constraints(self) -> Set[str]:
        """
        Gets the constraints set on the field by comparing the constraint value with its default value

        :return: the constraints set on field_info
        """
        return {
            attr
            for attr, default in self.__field_constraints__.items()
            if getattr(self, attr) != default
        }

    def update_from_config(self, from_config: Dict[str, Any]) -> None:
        """
        Update this FieldInfo based on a dict from get_field_info, only fields which have not been set are updated.
        """
        for key, value in from_config.items():
            if getattr(self, key) is None:
                setattr(self, key, value)

def Field(default: Any=Undefined, *, default_factory: Optional[NoArgAnyCallable]=None, alias: Optional[str]=None, title: Optional[str]=None, description: Optional[str]=None, exclude: Optional[Union['AbstractSetIntStr', 'MappingIntStrAny', Any]]=None, include: Optional[Union['AbstractSetIntStr', 'MappingIntStrAny', Any]]=None, const: Optional[bool]=None, gt: Optional[float]=None, ge: Optional[float]=None, lt: Optional[float]=None, le: Optional[float]=None, multiple_of: Optional[float]=None, allow_inf_nan: Optional[bool]=None, max_digits: Optional[int]=None, decimal_places: Optional[int]=None, min_items: Optional[int]=None, max_items: Optional[int]=None, unique_items: Optional[bool]=None, min_length: Optional[int]=None, max_length: Optional[int]=None, allow_mutation: bool=True, regex: Optional[str]=None, discriminator: Optional[str]=None, repr: bool=True, **extra: Any) -> Any:
    """
    Used to provide extra information about a field, either for the model schema or complex validation. Some arguments
    apply only to number fields (``int``, ``float``, ``Decimal``) and some apply only to ``str``.

    :param default: since this is replacing the field’s default, its first argument is used
      to set the default, use ellipsis (``...``) to indicate the field is required
    :param default_factory: callable that will be called when a default value is needed for this field
      If both `default` and `default_factory` are set, an error is raised.
    :param alias: the public name of the field
    :param title: can be any string, used in the schema
    :param description: can be any string, used in the schema
    :param exclude: exclude this field while dumping.
      Takes same values as the ``include`` and ``exclude`` arguments on the ``.dict`` method.
    :param include: include this field while dumping.
      Takes same values as the ``include`` and ``exclude`` arguments on the ``.dict`` method.
    :param const: this field is required and *must* take it's default value
    :param gt: only applies to numbers, requires the field to be "greater than". The schema
      will have an ``exclusiveMinimum`` validation keyword
    :param ge: only applies to numbers, requires the field to be "greater than or equal to". The
      schema will have a ``minimum`` validation keyword
    :param lt: only applies to numbers, requires the field to be "less than". The schema
      will have an ``exclusiveMaximum`` validation keyword
    :param le: only applies to numbers, requires the field to be "less than or equal to". The
      schema will have a ``maximum`` validation keyword
    :param multiple_of: only applies to numbers, requires the field to be "a multiple of". The
      schema will have a ``multipleOf`` validation keyword
    :param allow_inf_nan: only applies to numbers, allows the field to be NaN or infinity (+inf or -inf),
        which is a valid Python float. Default True, set to False for compatibility with JSON.
    :param max_digits: only applies to Decimals, requires the field to have a maximum number
      of digits within the decimal. It does not include a zero before the decimal point or trailing decimal zeroes.
    :param decimal_places: only applies to Decimals, requires the field to have at most a number of decimal places
      allowed. It does not include trailing decimal zeroes.
    :param min_items: only applies to lists, requires the field to have a minimum number of
      elements. The schema will have a ``minItems`` validation keyword
    :param max_items: only applies to lists, requires the field to have a maximum number of
      elements. The schema will have a ``maxItems`` validation keyword
    :param unique_items: only applies to lists, requires the field not to have duplicated
      elements. The schema will have a ``uniqueItems`` validation keyword
    :param min_length: only applies to strings, requires the field to have a minimum length. The
      schema will have a ``minLength`` validation keyword
    :param max_length: only applies to strings, requires the field to have a maximum length. The
      schema will have a ``maxLength`` validation keyword
    :param allow_mutation: a boolean which defaults to True. When False, the field raises a TypeError if the field is
      assigned on an instance.  The BaseModel Config must set validate_assignment to True
    :param regex: only applies to strings, requires the field match against a regular expression
      pattern string. The schema will have a ``pattern`` validation keyword
    :param discriminator: only useful with a (discriminated a.k.a. tagged) `Union` of sub models with a common field.
      The `discriminator` is the name of this common field to shorten validation and improve generated schema
    :param repr: show this field in the representation
    :param **extra: any additional keyword arguments will be added as is to the schema
    """
    pass
SHAPE_SINGLETON = 1
SHAPE_LIST = 2
SHAPE_SET = 3
SHAPE_MAPPING = 4
SHAPE_TUPLE = 5
SHAPE_TUPLE_ELLIPSIS = 6
SHAPE_SEQUENCE = 7
SHAPE_FROZENSET = 8
SHAPE_ITERABLE = 9
SHAPE_GENERIC = 10
SHAPE_DEQUE = 11
SHAPE_DICT = 12
SHAPE_DEFAULTDICT = 13
SHAPE_COUNTER = 14
SHAPE_NAME_LOOKUP = {SHAPE_LIST: 'List[{}]', SHAPE_SET: 'Set[{}]', SHAPE_TUPLE_ELLIPSIS: 'Tuple[{}, ...]', SHAPE_SEQUENCE: 'Sequence[{}]', SHAPE_FROZENSET: 'FrozenSet[{}]', SHAPE_ITERABLE: 'Iterable[{}]', SHAPE_DEQUE: 'Deque[{}]', SHAPE_DICT: 'Dict[{}]', SHAPE_DEFAULTDICT: 'DefaultDict[{}]', SHAPE_COUNTER: 'Counter[{}]'}
MAPPING_LIKE_SHAPES: Set[int] = {SHAPE_DEFAULTDICT, SHAPE_DICT, SHAPE_MAPPING, SHAPE_COUNTER}

class ModelField(Representation):
    __slots__ = ('type_', 'outer_type_', 'annotation', 'sub_fields', 'sub_fields_mapping', 'key_field', 'validators', 'pre_validators', 'post_validators', 'default', 'default_factory', 'required', 'final', 'model_config', 'name', 'alias', 'has_alias', 'field_info', 'discriminator_key', 'discriminator_alias', 'validate_always', 'allow_none', 'shape', 'class_validators', 'parse_json')

    def __init__(self, *, name: str, type_: Type[Any], class_validators: Optional[Dict[str, Validator]], model_config: Type['BaseConfig'], default: Any=None, default_factory: Optional[NoArgAnyCallable]=None, required: 'BoolUndefined'=Undefined, final: bool=False, alias: Optional[str]=None, field_info: Optional[FieldInfo]=None) -> None:
        self.name: str = name
        self.has_alias: bool = alias is not None
        self.alias: str = alias if alias is not None else name
        self.annotation = type_
        self.type_: Any = convert_generics(type_)
        self.outer_type_: Any = type_
        self.class_validators = class_validators or {}
        self.default: Any = default
        self.default_factory: Optional[NoArgAnyCallable] = default_factory
        self.required: 'BoolUndefined' = required
        self.final: bool = final
        self.model_config = model_config
        self.field_info: FieldInfo = field_info or FieldInfo(default)
        self.discriminator_key: Optional[str] = self.field_info.discriminator
        self.discriminator_alias: Optional[str] = self.discriminator_key
        self.allow_none: bool = False
        self.validate_always: bool = False
        self.sub_fields: Optional[List[ModelField]] = None
        self.sub_fields_mapping: Optional[Dict[str, 'ModelField']] = None
        self.key_field: Optional[ModelField] = None
        self.validators: 'ValidatorsList' = []
        self.pre_validators: Optional['ValidatorsList'] = None
        self.post_validators: Optional['ValidatorsList'] = None
        self.parse_json: bool = False
        self.shape: int = SHAPE_SINGLETON
        self.model_config.prepare_field(self)
        self.prepare()

    @staticmethod
    def _get_field_info(field_name: str, annotation: Any, value: Any, config: Type['BaseConfig']) -> Tuple[FieldInfo, Any]:
        """
        Get a FieldInfo from a root typing.Annotated annotation, value, or config default.

        The FieldInfo may be set in typing.Annotated or the value, but not both. If neither contain
        a FieldInfo, a new one will be created using the config.

        :param field_name: name of the field for use in error messages
        :param annotation: a type hint such as `str` or `Annotated[str, Field(..., min_length=5)]`
        :param value: the field's assigned value
        :param config: the model's config object
        :return: the FieldInfo contained in the `annotation`, the value, or a new one from the config.
        """
        field_info = None
        if get_origin(annotation) is Annotated:
            annotation_args = get_args(annotation)
            for arg in annotation_args[1:]:
                if isinstance(arg, FieldInfo):
                    if field_info is not None:
                        raise ValueError(f'Found multiple FieldInfo annotations for {field_name}')
                    field_info = arg

        if field_info is None and isinstance(value, FieldInfo):
            field_info = value
            value = value.default

        if field_info is None:
            field_info = FieldInfo(default=value)

        return field_info, value

    def prepare(self) -> None:
        """
        Prepare the field but inspecting self.default, self.type_ etc.

        Note: this method is **not** idempotent (because _type_analysis is not idempotent),
        e.g. calling it it multiple times may modify the field and configure it incorrectly.
        """
        pass

    def _set_default_and_type(self) -> None:
        """
        Set the default value, infer the type if needed and check if `None` value is valid.
        """
        pass

    def prepare_discriminated_union_sub_fields(self) -> None:
        """
        Prepare the mapping <discriminator key> -> <ModelField> and update `sub_fields`
        Note that this process can be aborted if a `ForwardRef` is encountered
        """
        pass

    def populate_validators(self) -> None:
        """
        Prepare self.pre_validators, self.validators, and self.post_validators based on self.type_'s  __get_validators__
        and class validators. This method should be idempotent, e.g. it should be safe to call multiple times
        without mis-configuring the field.
        """
        pass

    def _validate_sequence_like(self, v: Any, values: Dict[str, Any], loc: 'LocStr', cls: Optional['ModelOrDc']) -> 'ValidateReturn':
        """
        Validate sequence-like containers: lists, tuples, sets and generators
        Note that large if-else blocks are necessary to enable Cython
        optimization, which is why we disable the complexity check above.
        """
        pass

    def _validate_iterable(self, v: Any, values: Dict[str, Any], loc: 'LocStr', cls: Optional['ModelOrDc']) -> 'ValidateReturn':
        """
        Validate Iterables.

        This intentionally doesn't validate values to allow infinite generators.
        """
        pass

    def _get_mapping_value(self, original: T, converted: Dict[Any, Any]) -> Union[T, Dict[Any, Any]]:
        """
        When type is `Mapping[KT, KV]` (or another unsupported mapping), we try to avoid
        coercing to `dict` unwillingly.
        """
        pass

    def is_complex(self) -> bool:
        """
        Whether the field is "complex" eg. env variables should be parsed as JSON.
        """
        pass

    def __repr_args__(self) -> 'ReprArgs':
        args = [('name', self.name), ('type', self._type_display()), ('required', self.required)]
        if not self.required:
            if self.default_factory is not None:
                args.append(('default_factory', f'<function {self.default_factory.__name__}>'))
            else:
                args.append(('default', self.default))
        if self.alt_alias:
            args.append(('alias', self.alias))
        return args

class ModelPrivateAttr(Representation):
    __slots__ = ('default', 'default_factory')

    def __init__(self, default: Any=Undefined, *, default_factory: Optional[NoArgAnyCallable]=None) -> None:
        self.default = default
        self.default_factory = default_factory

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and (self.default, self.default_factory) == (other.default, other.default_factory)

def PrivateAttr(default: Any=Undefined, *, default_factory: Optional[NoArgAnyCallable]=None) -> Any:
    """
    Indicates that attribute is only used internally and never mixed with regular fields.

    Types or values of private attrs are not checked by pydantic and it's up to you to keep them relevant.

    Private attrs are stored in model __slots__.

    :param default: the attribute’s default value
    :param default_factory: callable that will be called when a default value is needed for this attribute
      If both `default` and `default_factory` are set, an error is raised.
    """
    pass

class DeferredType:
    """
    Used to postpone field preparation, while creating recursive generic models.
    """
