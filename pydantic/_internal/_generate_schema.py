"""Convert python types to pydantic-core schema."""
from __future__ import annotations as _annotations
import collections.abc
import dataclasses
import inspect
import re
import sys
import typing
import warnings
from contextlib import ExitStack, contextmanager
from copy import copy, deepcopy
from enum import Enum
from functools import partial
from inspect import Parameter, _ParameterKind, signature
from itertools import chain
from operator import attrgetter
from types import FunctionType, LambdaType, MethodType
from typing import TYPE_CHECKING, Any, Callable, Dict, Final, ForwardRef, Iterable, Iterator, Mapping, Type, TypeVar, Union, cast, overload
from warnings import warn
from pydantic_core import CoreSchema, PydanticUndefined, core_schema, to_jsonable_python
from typing_extensions import Annotated, Literal, TypeAliasType, TypedDict, get_args, get_origin, is_typeddict
from ..aliases import AliasGenerator
from ..annotated_handlers import GetCoreSchemaHandler, GetJsonSchemaHandler
from ..config import ConfigDict, JsonDict, JsonEncoder
from ..errors import PydanticSchemaGenerationError, PydanticUndefinedAnnotation, PydanticUserError
from ..json_schema import JsonSchemaValue
from ..version import version_short
from ..warnings import PydanticDeprecatedSince20
from . import _core_utils, _decorators, _discriminated_union, _known_annotated_metadata, _typing_extra
from ._config import ConfigWrapper, ConfigWrapperStack
from ._core_metadata import CoreMetadataHandler, build_metadata_dict
from ._core_utils import CoreSchemaOrField, collect_invalid_schemas, define_expected_missing_refs, get_ref, get_type_ref, is_function_with_inner_schema, is_list_like_schema_with_items_schema, simplify_schema_references, validate_core_schema
from ._decorators import Decorator, DecoratorInfos, FieldSerializerDecoratorInfo, FieldValidatorDecoratorInfo, ModelSerializerDecoratorInfo, ModelValidatorDecoratorInfo, RootValidatorDecoratorInfo, ValidatorDecoratorInfo, get_attribute_from_bases, inspect_field_serializer, inspect_model_serializer, inspect_validator
from ._docs_extraction import extract_docstrings_from_cls
from ._fields import collect_dataclass_fields, get_type_hints_infer_globalns
from ._forward_ref import PydanticRecursiveRef
from ._generics import get_standard_typevars_map, has_instance_in_type, recursively_defined_type_refs, replace_types
from ._mock_val_ser import MockCoreSchema
from ._schema_generation_shared import CallbackGetCoreSchemaHandler
from ._typing_extra import is_finalvar, is_self_type
from ._utils import lenient_issubclass
if TYPE_CHECKING:
    from ..fields import ComputedFieldInfo, FieldInfo
    from ..main import BaseModel
    from ..types import Discriminator
    from ..validators import FieldValidatorModes
    from ._dataclasses import StandardDataclass
    from ._schema_generation_shared import GetJsonSchemaFunction
_SUPPORTS_TYPEDDICT = sys.version_info >= (3, 12)
_AnnotatedType = type(Annotated[int, 123])
FieldDecoratorInfo = Union[ValidatorDecoratorInfo, FieldValidatorDecoratorInfo, FieldSerializerDecoratorInfo]
FieldDecoratorInfoType = TypeVar('FieldDecoratorInfoType', bound=FieldDecoratorInfo)
AnyFieldDecorator = Union[Decorator[ValidatorDecoratorInfo], Decorator[FieldValidatorDecoratorInfo], Decorator[FieldSerializerDecoratorInfo]]
ModifyCoreSchemaWrapHandler = GetCoreSchemaHandler
GetCoreSchemaFunction = Callable[[Any, ModifyCoreSchemaWrapHandler], core_schema.CoreSchema]
TUPLE_TYPES: list[type] = [tuple, typing.Tuple]
LIST_TYPES: list[type] = [list, typing.List, collections.abc.MutableSequence]
SET_TYPES: list[type] = [set, typing.Set, collections.abc.MutableSet]
FROZEN_SET_TYPES: list[type] = [frozenset, typing.FrozenSet, collections.abc.Set]
DICT_TYPES: list[type] = [dict, typing.Dict, collections.abc.MutableMapping, collections.abc.Mapping]

def check_validator_fields_against_field_name(info: FieldDecoratorInfo, field: str) -> bool:
    """Check if field name is in validator fields.

    Args:
        info: The field info.
        field: The field name to check.

    Returns:
        `True` if field name is in validator fields, `False` otherwise.
    """
    if info.fields == '*':
        return True
    if isinstance(info.fields, str):
        return info.fields == field
    return field in info.fields

def check_decorator_fields_exist(decorators: Iterable[AnyFieldDecorator], fields: Iterable[str]) -> None:
    """Check if the defined fields in decorators exist in `fields` param.

    It ignores the check for a decorator if the decorator has `*` as field or `check_fields=False`.

    Args:
        decorators: An iterable of decorators.
        fields: An iterable of fields name.

    Raises:
        PydanticUserError: If one of the field names does not exist in `fields` param.
    """
    fields_set = set(fields)
    for decorator in decorators:
        if decorator.check_fields and decorator.fields != '*':
            for field in (decorator.fields if isinstance(decorator.fields, Iterable) else [decorator.fields]):
                if field not in fields_set:
                    raise PydanticUserError(
                        f"Field '{field}' does not exist on this model.",
                        code='validator-field-error',
                    )

def modify_model_json_schema(schema_or_field: CoreSchemaOrField, handler: GetJsonSchemaHandler, *, cls: Any, title: str | None=None) -> JsonSchemaValue:
    """Add title and description for model-like classes' JSON schema.

    Args:
        schema_or_field: The schema data to generate a JSON schema from.
        handler: The `GetCoreSchemaHandler` instance.
        cls: The model-like class.
        title: The title to set for the model's schema, defaults to the model's name

    Returns:
        JsonSchemaValue: The updated JSON schema.
    """
    json_schema = handler(schema_or_field)
    
    if title is None:
        title = cls.__name__
    
    json_schema['title'] = title
    
    if cls.__doc__:
        json_schema['description'] = inspect.cleandoc(cls.__doc__)
    
    return json_schema
JsonEncoders = Dict[Type[Any], JsonEncoder]

def _add_custom_serialization_from_json_encoders(json_encoders: JsonEncoders | None, tp: Any, schema: CoreSchema) -> CoreSchema:
    """Iterate over the json_encoders and add the first matching encoder to the schema.

    Args:
        json_encoders: A dictionary of types and their encoder functions.
        tp: The type to check for a matching encoder.
        schema: The schema to add the encoder to.
    """
    if json_encoders is None:
        return schema

    for encoder_type, encoder in json_encoders.items():
        if isinstance(tp, encoder_type):
            return core_schema.json_or_python_schema(
                json_schema=core_schema.with_info_plain_validator_function(encoder),
                python_schema=schema,
            )
    
    return schema
TypesNamespace = Union[Dict[str, Any], None]

class TypesNamespaceStack:
    """A stack of types namespaces."""

    def __init__(self, types_namespace: TypesNamespace):
        self._types_namespace_stack: list[TypesNamespace] = [types_namespace]

def _get_first_non_null(a: Any, b: Any) -> Any:
    """Return the first argument if it is not None, otherwise return the second argument.

    Use case: serialization_alias (argument a) and alias (argument b) are both defined, and serialization_alias is ''.
    This function will return serialization_alias, which is the first argument, even though it is an empty string.
    """
    return a if a is not None else b

class GenerateSchema:
    """Generate core schema for a Pydantic model, dataclass and types like `str`, `datetime`, ... ."""
    __slots__ = ('_config_wrapper_stack', '_types_namespace_stack', '_typevars_map', 'field_name_stack', 'model_type_stack', 'defs')

    def __init__(self, config_wrapper: ConfigWrapper, types_namespace: dict[str, Any] | None, typevars_map: dict[Any, Any] | None=None) -> None:
        self._config_wrapper_stack = ConfigWrapperStack(config_wrapper)
        self._types_namespace_stack = TypesNamespaceStack(types_namespace)
        self._typevars_map = typevars_map
        self.field_name_stack = _FieldNameStack()
        self.model_type_stack = _ModelTypeStack()
        self.defs = _Definitions()

    def str_schema(self) -> CoreSchema:
        """Generate a CoreSchema for `str`"""
        return core_schema.str_schema()

    class CollectedInvalid(Exception):
        pass

    def generate_schema(self, obj: Any, from_dunder_get_core_schema: bool=True) -> core_schema.CoreSchema:
        """Generate core schema.

        Args:
            obj: The object to generate core schema for.
            from_dunder_get_core_schema: Whether to generate schema from either the
                `__get_pydantic_core_schema__` function or `__pydantic_core_schema__` property.

        Returns:
            The generated core schema.

        Raises:
            PydanticUndefinedAnnotation:
                If it is not possible to evaluate forward reference.
            PydanticSchemaGenerationError:
                If it is not possible to generate pydantic-core schema.
            TypeError:
                - If `alias_generator` returns a disallowed type (must be str, AliasPath or AliasChoices).
                - If V1 style validator with `each_item=True` applied on a wrong field.
            PydanticUserError:
                - If `typing.TypedDict` is used instead of `typing_extensions.TypedDict` on Python < 3.12.
                - If `__modify_schema__` method is used instead of `__get_pydantic_json_schema__`.
        """
        if from_dunder_get_core_schema:
            schema = self._generate_schema_from_property(obj, obj)
            if schema is not None:
                return schema

        try:
            return self.match_type(obj)
        except PydanticUndefinedAnnotation:
            raise
        except Exception as e:
            raise PydanticSchemaGenerationError(
                f"Unable to generate pydantic-core schema for {obj!r}"
            ) from e

    def _model_schema(self, cls: type[BaseModel]) -> core_schema.CoreSchema:
        """Generate schema for a Pydantic model."""
        pass

    @staticmethod
    def _get_model_title_from_config(model: type[BaseModel | StandardDataclass], config_wrapper: ConfigWrapper | None=None) -> str | None:
        """Get the title of a model if `model_title_generator` or `title` are set in the config, else return None"""
        pass

    def _unpack_refs_defs(self, schema: CoreSchema) -> CoreSchema:
        """Unpack all 'definitions' schemas into `GenerateSchema.defs.definitions`
        and return the inner schema.
        """
        pass

    def _generate_schema_from_property(self, obj: Any, source: Any) -> core_schema.CoreSchema | None:
        """Try to generate schema from either the `__get_pydantic_core_schema__` function or
        `__pydantic_core_schema__` property.

        Note: `__get_pydantic_core_schema__` takes priority so it can
        decide whether to use a `__pydantic_core_schema__` attribute, or generate a fresh schema.
        """
        pass

    def match_type(self, obj: Any) -> core_schema.CoreSchema:
        """Main mapping of types to schemas.

        The general structure is a series of if statements starting with the simple cases
        (non-generic primitive types) and then handling generics and other more complex cases.

        Each case either generates a schema directly, calls into a public user-overridable method
        (like `GenerateSchema.tuple_variable_schema`) or calls into a private method that handles some
        boilerplate before calling into the user-facing method (e.g. `GenerateSchema._tuple_schema`).

        The idea is that we'll evolve this into adding more and more user facing methods over time
        as they get requested and we figure out what the right API for them is.
        """
        pass

    def _generate_td_field_schema(self, name: str, field_info: FieldInfo, decorators: DecoratorInfos, *, required: bool=True) -> core_schema.TypedDictField:
        """Prepare a TypedDictField to represent a model or typeddict field."""
        pass

    def _generate_md_field_schema(self, name: str, field_info: FieldInfo, decorators: DecoratorInfos) -> core_schema.ModelField:
        """Prepare a ModelField to represent a model field."""
        pass

    def _generate_dc_field_schema(self, name: str, field_info: FieldInfo, decorators: DecoratorInfos) -> core_schema.DataclassField:
        """Prepare a DataclassField to represent the parameter/field, of a dataclass."""
        pass

    @staticmethod
    def _apply_alias_generator_to_field_info(alias_generator: Callable[[str], str] | AliasGenerator, field_info: FieldInfo, field_name: str) -> None:
        """Apply an alias_generator to aliases on a FieldInfo instance if appropriate.

        Args:
            alias_generator: A callable that takes a string and returns a string, or an AliasGenerator instance.
            field_info: The FieldInfo instance to which the alias_generator is (maybe) applied.
            field_name: The name of the field from which to generate the alias.
        """
        pass

    @staticmethod
    def _apply_alias_generator_to_computed_field_info(alias_generator: Callable[[str], str] | AliasGenerator, computed_field_info: ComputedFieldInfo, computed_field_name: str):
        """Apply an alias_generator to alias on a ComputedFieldInfo instance if appropriate.

        Args:
            alias_generator: A callable that takes a string and returns a string, or an AliasGenerator instance.
            computed_field_info: The ComputedFieldInfo instance to which the alias_generator is (maybe) applied.
            computed_field_name: The name of the computed field from which to generate the alias.
        """
        pass

    @staticmethod
    def _apply_field_title_generator_to_field_info(config_wrapper: ConfigWrapper, field_info: FieldInfo | ComputedFieldInfo, field_name: str) -> None:
        """Apply a field_title_generator on a FieldInfo or ComputedFieldInfo instance if appropriate
        Args:
            config_wrapper: The config of the model
            field_info: The FieldInfo or ComputedField instance to which the title_generator is (maybe) applied.
            field_name: The name of the field from which to generate the title.
        """
        pass

    def _union_schema(self, union_type: Any) -> core_schema.CoreSchema:
        """Generate schema for a Union."""
        pass

    def _literal_schema(self, literal_type: Any) -> CoreSchema:
        """Generate schema for a Literal."""
        pass

    def _typed_dict_schema(self, typed_dict_cls: Any, origin: Any) -> core_schema.CoreSchema:
        """Generate schema for a TypedDict.

        It is not possible to track required/optional keys in TypedDict without __required_keys__
        since TypedDict.__new__ erases the base classes (it replaces them with just `dict`)
        and thus we can track usage of total=True/False
        __required_keys__ was added in Python 3.9
        (https://github.com/miss-islington/cpython/blob/1e9939657dd1f8eb9f596f77c1084d2d351172fc/Doc/library/typing.rst?plain=1#L1546-L1548)
        however it is buggy
        (https://github.com/python/typing_extensions/blob/ac52ac5f2cb0e00e7988bae1e2a1b8257ac88d6d/src/typing_extensions.py#L657-L666).

        On 3.11 but < 3.12 TypedDict does not preserve inheritance information.

        Hence to avoid creating validators that do not do what users expect we only
        support typing.TypedDict on Python >= 3.12 or typing_extension.TypedDict on all versions
        """
        pass

    def _namedtuple_schema(self, namedtuple_cls: Any, origin: Any) -> core_schema.CoreSchema:
        """Generate schema for a NamedTuple."""
        pass

    def _generate_parameter_schema(self, name: str, annotation: type[Any], default: Any=Parameter.empty, mode: Literal['positional_only', 'positional_or_keyword', 'keyword_only'] | None=None) -> core_schema.ArgumentsParameter:
        """Prepare a ArgumentsParameter to represent a field in a namedtuple or function signature."""
        pass

    def _tuple_schema(self, tuple_type: Any) -> core_schema.CoreSchema:
        """Generate schema for a Tuple, e.g. `tuple[int, str]` or `tuple[int, ...]`."""
        pass

    def _union_is_subclass_schema(self, union_type: Any) -> core_schema.CoreSchema:
        """Generate schema for `Type[Union[X, ...]]`."""
        pass

    def _subclass_schema(self, type_: Any) -> core_schema.CoreSchema:
        """Generate schema for a Type, e.g. `Type[int]`."""
        pass

    def _sequence_schema(self, sequence_type: Any) -> core_schema.CoreSchema:
        """Generate schema for a Sequence, e.g. `Sequence[int]`."""
        pass

    def _iterable_schema(self, type_: Any) -> core_schema.GeneratorSchema:
        """Generate a schema for an `Iterable`."""
        pass

    def _dataclass_schema(self, dataclass: type[StandardDataclass], origin: type[StandardDataclass] | None) -> core_schema.CoreSchema:
        """Generate schema for a dataclass."""
        pass

    def _callable_schema(self, function: Callable[..., Any]) -> core_schema.CallSchema:
        """Generate schema for a Callable.

        TODO support functional validators once we support them in Config
        """
        pass

    def _annotated_schema(self, annotated_type: Any) -> core_schema.CoreSchema:
        """Generate schema for an Annotated type, e.g. `Annotated[int, Field(...)]` or `Annotated[int, Gt(0)]`."""
        pass

    def _apply_annotations(self, source_type: Any, annotations: list[Any], transform_inner_schema: Callable[[CoreSchema], CoreSchema]=lambda x: x) -> CoreSchema:
        """Apply arguments from `Annotated` or from `FieldInfo` to a schema.

        This gets called by `GenerateSchema._annotated_schema` but differs from it in that it does
        not expect `source_type` to be an `Annotated` object, it expects it to be  the first argument of that
        (in other words, `GenerateSchema._annotated_schema` just unpacks `Annotated`, this process it).
        """
        pass

    def _apply_field_serializers(self, schema: core_schema.CoreSchema, serializers: list[Decorator[FieldSerializerDecoratorInfo]], computed_field: bool=False) -> core_schema.CoreSchema:
        """Apply field serializers to a schema."""
        pass

    def _apply_model_serializers(self, schema: core_schema.CoreSchema, serializers: Iterable[Decorator[ModelSerializerDecoratorInfo]]) -> core_schema.CoreSchema:
        """Apply model serializers to a schema."""
        pass
_VALIDATOR_F_MATCH: Mapping[tuple[FieldValidatorModes, Literal['no-info', 'with-info']], Callable[[Callable[..., Any], core_schema.CoreSchema, str | None], core_schema.CoreSchema]] = {('before', 'no-info'): lambda f, schema, _: core_schema.no_info_before_validator_function(f, schema), ('after', 'no-info'): lambda f, schema, _: core_schema.no_info_after_validator_function(f, schema), ('plain', 'no-info'): lambda f, _1, _2: core_schema.no_info_plain_validator_function(f), ('wrap', 'no-info'): lambda f, schema, _: core_schema.no_info_wrap_validator_function(f, schema), ('before', 'with-info'): lambda f, schema, field_name: core_schema.with_info_before_validator_function(f, schema, field_name=field_name), ('after', 'with-info'): lambda f, schema, field_name: core_schema.with_info_after_validator_function(f, schema, field_name=field_name), ('plain', 'with-info'): lambda f, _, field_name: core_schema.with_info_plain_validator_function(f, field_name=field_name), ('wrap', 'with-info'): lambda f, schema, field_name: core_schema.with_info_wrap_validator_function(f, schema, field_name=field_name)}

def apply_validators(schema: core_schema.CoreSchema, validators: Iterable[Decorator[RootValidatorDecoratorInfo]] | Iterable[Decorator[ValidatorDecoratorInfo]] | Iterable[Decorator[FieldValidatorDecoratorInfo]], field_name: str | None) -> core_schema.CoreSchema:
    """Apply validators to a schema.

    Args:
        schema: The schema to apply validators on.
        validators: An iterable of validators.
        field_name: The name of the field if validators are being applied to a model field.

    Returns:
        The updated schema.
    """
    pass

def _validators_require_validate_default(validators: Iterable[Decorator[ValidatorDecoratorInfo]]) -> bool:
    """In v1, if any of the validators for a field had `always=True`, the default value would be validated.

    This serves as an auxiliary function for re-implementing that logic, by looping over a provided
    collection of (v1-style) ValidatorDecoratorInfo's and checking if any of them have `always=True`.

    We should be able to drop this function and the associated logic calling it once we drop support
    for v1-style validator decorators. (Or we can extend it and keep it if we add something equivalent
    to the v1-validator `always` kwarg to `field_validator`.)
    """
    pass

def apply_model_validators(schema: core_schema.CoreSchema, validators: Iterable[Decorator[ModelValidatorDecoratorInfo]], mode: Literal['inner', 'outer', 'all']) -> core_schema.CoreSchema:
    """Apply model validators to a schema.

    If mode == 'inner', only "before" validators are applied
    If mode == 'outer', validators other than "before" are applied
    If mode == 'all', all validators are applied

    Args:
        schema: The schema to apply validators on.
        validators: An iterable of validators.
        mode: The validator mode.

    Returns:
        The updated schema.
    """
    pass

def wrap_default(field_info: FieldInfo, schema: core_schema.CoreSchema) -> core_schema.CoreSchema:
    """Wrap schema with default schema if default value or `default_factory` are available.

    Args:
        field_info: The field info object.
        schema: The schema to apply default on.

    Returns:
        Updated schema by default value or `default_factory`.
    """
    pass

def _extract_get_pydantic_json_schema(tp: Any, schema: CoreSchema) -> GetJsonSchemaFunction | None:
    """Extract `__get_pydantic_json_schema__` from a type, handling the deprecated `__modify_schema__`."""
    pass

class _CommonField(TypedDict):
    schema: core_schema.CoreSchema
    validation_alias: str | list[str | int] | list[list[str | int]] | None
    serialization_alias: str | None
    serialization_exclude: bool | None
    frozen: bool | None
    metadata: dict[str, Any]

class _Definitions:
    """Keeps track of references and definitions."""

    def __init__(self) -> None:
        self.seen: set[str] = set()
        self.definitions: dict[str, core_schema.CoreSchema] = {}

    @contextmanager
    def get_schema_or_ref(self, tp: Any) -> Iterator[tuple[str, None] | tuple[str, CoreSchema]]:
        """Get a definition for `tp` if one exists.

        If a definition exists, a tuple of `(ref_string, CoreSchema)` is returned.
        If no definition exists yet, a tuple of `(ref_string, None)` is returned.

        Note that the returned `CoreSchema` will always be a `DefinitionReferenceSchema`,
        not the actual definition itself.

        This should be called for any type that can be identified by reference.
        This includes any recursive types.

        At present the following types can be named/recursive:

        - BaseModel
        - Dataclasses
        - TypedDict
        - TypeAliasType
        """
        pass

class _FieldNameStack:
    __slots__ = ('_stack',)

    def __init__(self) -> None:
        self._stack: list[str] = []

class _ModelTypeStack:
    __slots__ = ('_stack',)

    def __init__(self) -> None:
        self._stack: list[type] = []
