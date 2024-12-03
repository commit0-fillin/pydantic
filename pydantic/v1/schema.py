import re
import warnings
from collections import defaultdict
from dataclasses import is_dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, ForwardRef, FrozenSet, Generic, Iterable, List, Optional, Pattern, Sequence, Set, Tuple, Type, TypeVar, Union, cast
from uuid import UUID
from typing_extensions import Annotated, Literal
from pydantic.v1.fields import MAPPING_LIKE_SHAPES, SHAPE_DEQUE, SHAPE_FROZENSET, SHAPE_GENERIC, SHAPE_ITERABLE, SHAPE_LIST, SHAPE_SEQUENCE, SHAPE_SET, SHAPE_SINGLETON, SHAPE_TUPLE, SHAPE_TUPLE_ELLIPSIS, FieldInfo, ModelField
from pydantic.v1.json import pydantic_encoder
from pydantic.v1.networks import AnyUrl, EmailStr
from pydantic.v1.types import ConstrainedDecimal, ConstrainedFloat, ConstrainedFrozenSet, ConstrainedInt, ConstrainedList, ConstrainedSet, ConstrainedStr, SecretBytes, SecretStr, StrictBytes, StrictStr, conbytes, condecimal, confloat, confrozenset, conint, conlist, conset, constr
from pydantic.v1.typing import all_literal_values, get_args, get_origin, get_sub_types, is_callable_type, is_literal_type, is_namedtuple, is_none_type, is_union
from pydantic.v1.utils import ROOT_KEY, get_model, lenient_issubclass
if TYPE_CHECKING:
    from pydantic.v1.dataclasses import Dataclass
    from pydantic.v1.main import BaseModel
default_prefix = '#/definitions/'
default_ref_template = '#/definitions/{model}'
TypeModelOrEnum = Union[Type['BaseModel'], Type[Enum]]
TypeModelSet = Set[TypeModelOrEnum]

def schema(models: Sequence[Union[Type['BaseModel'], Type['Dataclass']]], *, by_alias: bool=True, title: Optional[str]=None, description: Optional[str]=None, ref_prefix: Optional[str]=None, ref_template: str=default_ref_template) -> Dict[str, Any]:
    """
    Process a list of models and generate a single JSON Schema with all of them defined in the ``definitions``
    top-level JSON key, including their sub-models.

    :param models: a list of models to include in the generated JSON Schema
    :param by_alias: generate the schemas using the aliases defined, if any
    :param title: title for the generated schema that includes the definitions
    :param description: description for the generated schema
    :param ref_prefix: the JSON Pointer prefix for schema references with ``$ref``, if None, will be set to the
      default of ``#/definitions/``. Update it if you want the schemas to reference the definitions somewhere
      else, e.g. for OpenAPI use ``#/components/schemas/``. The resulting generated schemas will still be at the
      top-level key ``definitions``, so you can extract them from there. But all the references will have the set
      prefix.
    :param ref_template: Use a ``string.format()`` template for ``$ref`` instead of a prefix. This can be useful
      for references that cannot be represented by ``ref_prefix`` such as a definition stored in another file. For
      a sibling json file in a ``/schemas`` directory use ``"/schemas/${model}.json#"``.
    :return: dict with the JSON Schema with a ``definitions`` top-level key including the schema definitions for
      the models and sub-models passed in ``models``.
    """
    clean_models = [model for model in models if model is not None]
    flat_models = get_flat_models_from_models(clean_models)
    model_name_map = get_model_name_map(flat_models)
    definitions = {}
    output_schema: Dict[str, Any] = {}
    for model in clean_models:
        m_schema, m_definitions, m_nested_models = model_process_schema(
            model, by_alias=by_alias, model_name_map=model_name_map, ref_prefix=ref_prefix, ref_template=ref_template
        )
        definitions.update(m_definitions)
        model_name = model_name_map[model]
        definitions[model_name] = m_schema

    if title:
        output_schema['title'] = title
    if description:
        output_schema['description'] = description

    output_schema['definitions'] = definitions
    return output_schema

def model_schema(model: Union[Type['BaseModel'], Type['Dataclass']], by_alias: bool=True, ref_prefix: Optional[str]=None, ref_template: str=default_ref_template) -> Dict[str, Any]:
    """
    Generate a JSON Schema for one model. With all the sub-models defined in the ``definitions`` top-level
    JSON key.

    :param model: a Pydantic model (a class that inherits from BaseModel)
    :param by_alias: generate the schemas using the aliases defined, if any
    :param ref_prefix: the JSON Pointer prefix for schema references with ``$ref``, if None, will be set to the
      default of ``#/definitions/``. Update it if you want the schemas to reference the definitions somewhere
      else, e.g. for OpenAPI use ``#/components/schemas/``. The resulting generated schemas will still be at the
      top-level key ``definitions``, so you can extract them from there. But all the references will have the set
      prefix.
    :param ref_template: Use a ``string.format()`` template for ``$ref`` instead of a prefix. This can be useful for
      references that cannot be represented by ``ref_prefix`` such as a definition stored in another file. For a
      sibling json file in a ``/schemas`` directory use ``"/schemas/${model}.json#"``.
    :return: dict with the JSON Schema for the passed ``model``
    """
    flat_models = get_flat_models_from_model(model)
    model_name_map = get_model_name_map(flat_models)
    model_name = model_name_map[model]
    m_schema, definitions, nested_models = model_process_schema(
        model, by_alias=by_alias, model_name_map=model_name_map, ref_prefix=ref_prefix, ref_template=ref_template
    )
    definitions[model_name] = m_schema
    return {'definitions': definitions, **m_schema}

def field_schema(field: ModelField, *, by_alias: bool=True, model_name_map: Dict[TypeModelOrEnum, str], ref_prefix: Optional[str]=None, ref_template: str=default_ref_template, known_models: Optional[TypeModelSet]=None) -> Tuple[Dict[str, Any], Dict[str, Any], Set[str]]:
    """
    Process a Pydantic field and return a tuple with a JSON Schema for it as the first item.
    Also return a dictionary of definitions with models as keys and their schemas as values. If the passed field
    is a model and has sub-models, and those sub-models don't have overrides (as ``title``, ``default``, etc), they
    will be included in the definitions and referenced in the schema instead of included recursively.

    :param field: a Pydantic ``ModelField``
    :param by_alias: use the defined alias (if any) in the returned schema
    :param model_name_map: used to generate the JSON Schema references to other models included in the definitions
    :param ref_prefix: the JSON Pointer prefix to use for references to other schemas, if None, the default of
      #/definitions/ will be used
    :param ref_template: Use a ``string.format()`` template for ``$ref`` instead of a prefix. This can be useful for
      references that cannot be represented by ``ref_prefix`` such as a definition stored in another file. For a
      sibling json file in a ``/schemas`` directory use ``"/schemas/${model}.json#"``.
    :param known_models: used to solve circular references
    :return: tuple of the schema for this field and additional definitions
    """
    schema_overrides = False

    field_info = field.field_info
    if field_info.extra:
        schema = dict(field_info.extra)
        if 'type' in schema:
            schema_overrides = True
    else:
        schema = {}

    if field.field_info is not None and field.field_info.title is not None:
        schema['title'] = field.field_info.title

    validation_schema = get_field_schema_validations(field)
    if validation_schema:
        schema.update(validation_schema)

    f_schema, f_definitions, f_nested_models = field_type_schema(
        field,
        by_alias=by_alias,
        model_name_map=model_name_map,
        schema_overrides=schema_overrides,
        ref_prefix=ref_prefix,
        ref_template=ref_template,
        known_models=known_models or set(),
    )
    # Merge the field schema into the main schema
    schema.update(f_schema)

    # Remove None values from the schema
    schema = {k: v for k, v in schema.items() if v is not None}

    return schema, f_definitions, f_nested_models
numeric_types = (int, float, Decimal)
_str_types_attrs: Tuple[Tuple[str, Union[type, Tuple[type, ...]], str], ...] = (('max_length', numeric_types, 'maxLength'), ('min_length', numeric_types, 'minLength'), ('regex', str, 'pattern'))
_numeric_types_attrs: Tuple[Tuple[str, Union[type, Tuple[type, ...]], str], ...] = (('gt', numeric_types, 'exclusiveMinimum'), ('lt', numeric_types, 'exclusiveMaximum'), ('ge', numeric_types, 'minimum'), ('le', numeric_types, 'maximum'), ('multiple_of', numeric_types, 'multipleOf'))

def get_field_schema_validations(field: ModelField) -> Dict[str, Any]:
    """
    Get the JSON Schema validation keywords for a ``field`` with an annotation of
    a Pydantic ``FieldInfo`` with validation arguments.
    """
    validators = {}
    field_info = field.field_info
    if isinstance(field_info, FieldInfo):
        if field_info.const:
            validators['const'] = field.default
        if field_info.gt is not None:
            validators['exclusiveMinimum'] = field_info.gt
        if field_info.ge is not None:
            validators['minimum'] = field_info.ge
        if field_info.lt is not None:
            validators['exclusiveMaximum'] = field_info.lt
        if field_info.le is not None:
            validators['maximum'] = field_info.le
        if field_info.multiple_of is not None:
            validators['multipleOf'] = field_info.multiple_of
        if field_info.min_items is not None:
            validators['minItems'] = field_info.min_items
        if field_info.max_items is not None:
            validators['maxItems'] = field_info.max_items
        if field_info.min_length is not None:
            validators['minLength'] = field_info.min_length
        if field_info.max_length is not None:
            validators['maxLength'] = field_info.max_length
        if field_info.regex:
            validators['pattern'] = field_info.regex
        if field_info.unique_items:
            validators['uniqueItems'] = True
    return validators

def get_model_name_map(unique_models: TypeModelSet) -> Dict[TypeModelOrEnum, str]:
    """
    Process a set of models and generate unique names for them to be used as keys in the JSON Schema
    definitions. By default the names are the same as the class name. But if two models in different Python
    modules have the same name (e.g. "users.Model" and "items.Model"), the generated names will be
    based on the Python module path for those conflicting models to prevent name collisions.

    :param unique_models: a Python set of models
    :return: dict mapping models to names
    """
    name_model_map = {}
    conflicting_names: Set[str] = set()
    for model in unique_models:
        model_name = normalize_name(model.__name__)
        if model_name in name_model_map:
            conflicting_names.add(model_name)
        name_model_map[model_name] = model

    for model_name in conflicting_names:
        competing_models = [
            model for model in unique_models if normalize_name(model.__name__) == model_name
        ]
        for model in competing_models:
            model_name = f'{model.__module__}.{model.__name__}'
            module_parts = model_name.rsplit('.', 1)[0].split('.')
            model_name = '.'.join([normalize_name(part) for part in module_parts])

    return {model: name for name, model in name_model_map.items()}

def get_flat_models_from_model(model: Type['BaseModel'], known_models: Optional[TypeModelSet]=None) -> TypeModelSet:
    """
    Take a single ``model`` and generate a set with itself and all the sub-models in the tree. I.e. if you pass
    model ``Foo`` (subclass of Pydantic ``BaseModel``) as ``model``, and it has a field of type ``Bar`` (also
    subclass of ``BaseModel``) and that model ``Bar`` has a field of type ``Baz`` (also subclass of ``BaseModel``),
    the return value will be ``set([Foo, Bar, Baz])``.

    :param model: a Pydantic ``BaseModel`` subclass
    :param known_models: used to solve circular references
    :return: a set with the initial model and all its sub-models
    """
    pass

def get_flat_models_from_field(field: ModelField, known_models: TypeModelSet) -> TypeModelSet:
    """
    Take a single Pydantic ``ModelField`` (from a model) that could have been declared as a subclass of BaseModel
    (so, it could be a submodel), and generate a set with its model and all the sub-models in the tree.
    I.e. if you pass a field that was declared to be of type ``Foo`` (subclass of BaseModel) as ``field``, and that
    model ``Foo`` has a field of type ``Bar`` (also subclass of ``BaseModel``) and that model ``Bar`` has a field of
    type ``Baz`` (also subclass of ``BaseModel``), the return value will be ``set([Foo, Bar, Baz])``.

    :param field: a Pydantic ``ModelField``
    :param known_models: used to solve circular references
    :return: a set with the model used in the declaration for this field, if any, and all its sub-models
    """
    pass

def get_flat_models_from_fields(fields: Sequence[ModelField], known_models: TypeModelSet) -> TypeModelSet:
    """
    Take a list of Pydantic  ``ModelField``s (from a model) that could have been declared as subclasses of ``BaseModel``
    (so, any of them could be a submodel), and generate a set with their models and all the sub-models in the tree.
    I.e. if you pass a the fields of a model ``Foo`` (subclass of ``BaseModel``) as ``fields``, and on of them has a
    field of type ``Bar`` (also subclass of ``BaseModel``) and that model ``Bar`` has a field of type ``Baz`` (also
    subclass of ``BaseModel``), the return value will be ``set([Foo, Bar, Baz])``.

    :param fields: a list of Pydantic ``ModelField``s
    :param known_models: used to solve circular references
    :return: a set with any model declared in the fields, and all their sub-models
    """
    pass

def get_flat_models_from_models(models: Sequence[Type['BaseModel']]) -> TypeModelSet:
    """
    Take a list of ``models`` and generate a set with them and all their sub-models in their trees. I.e. if you pass
    a list of two models, ``Foo`` and ``Bar``, both subclasses of Pydantic ``BaseModel`` as models, and ``Bar`` has
    a field of type ``Baz`` (also subclass of ``BaseModel``), the return value will be ``set([Foo, Bar, Baz])``.
    """
    pass

def field_type_schema(field: ModelField, *, by_alias: bool, model_name_map: Dict[TypeModelOrEnum, str], ref_template: str, schema_overrides: bool=False, ref_prefix: Optional[str]=None, known_models: TypeModelSet) -> Tuple[Dict[str, Any], Dict[str, Any], Set[str]]:
    """
    Used by ``field_schema()``, you probably should be using that function.

    Take a single ``field`` and generate the schema for its type only, not including additional
    information as title, etc. Also return additional schema definitions, from sub-models.
    """
    pass

def model_process_schema(model: TypeModelOrEnum, *, by_alias: bool=True, model_name_map: Dict[TypeModelOrEnum, str], ref_prefix: Optional[str]=None, ref_template: str=default_ref_template, known_models: Optional[TypeModelSet]=None, field: Optional[ModelField]=None) -> Tuple[Dict[str, Any], Dict[str, Any], Set[str]]:
    """
    Used by ``model_schema()``, you probably should be using that function.

    Take a single ``model`` and generate its schema. Also return additional schema definitions, from sub-models. The
    sub-models of the returned schema will be referenced, but their definitions will not be included in the schema. All
    the definitions are returned as the second value.
    """
    pass

def model_type_schema(model: Type['BaseModel'], *, by_alias: bool, model_name_map: Dict[TypeModelOrEnum, str], ref_template: str, ref_prefix: Optional[str]=None, known_models: TypeModelSet) -> Tuple[Dict[str, Any], Dict[str, Any], Set[str]]:
    """
    You probably should be using ``model_schema()``, this function is indirectly used by that function.

    Take a single ``model`` and generate the schema for its type only, not including additional
    information as title, etc. Also return additional schema definitions, from sub-models.
    """
    pass

def enum_process_schema(enum: Type[Enum], *, field: Optional[ModelField]=None) -> Dict[str, Any]:
    """
    Take a single `enum` and generate its schema.

    This is similar to the `model_process_schema` function, but applies to ``Enum`` objects.
    """
    pass

def field_singleton_sub_fields_schema(field: ModelField, *, by_alias: bool, model_name_map: Dict[TypeModelOrEnum, str], ref_template: str, schema_overrides: bool=False, ref_prefix: Optional[str]=None, known_models: TypeModelSet) -> Tuple[Dict[str, Any], Dict[str, Any], Set[str]]:
    """
    This function is indirectly used by ``field_schema()``, you probably should be using that function.

    Take a list of Pydantic ``ModelField`` from the declaration of a type with parameters, and generate their
    schema. I.e., fields used as "type parameters", like ``str`` and ``int`` in ``Tuple[str, int]``.
    """
    pass
field_class_to_schema: Tuple[Tuple[Any, Dict[str, Any]], ...] = ((Path, {'type': 'string', 'format': 'path'}), (datetime, {'type': 'string', 'format': 'date-time'}), (date, {'type': 'string', 'format': 'date'}), (time, {'type': 'string', 'format': 'time'}), (timedelta, {'type': 'number', 'format': 'time-delta'}), (IPv4Network, {'type': 'string', 'format': 'ipv4network'}), (IPv6Network, {'type': 'string', 'format': 'ipv6network'}), (IPv4Interface, {'type': 'string', 'format': 'ipv4interface'}), (IPv6Interface, {'type': 'string', 'format': 'ipv6interface'}), (IPv4Address, {'type': 'string', 'format': 'ipv4'}), (IPv6Address, {'type': 'string', 'format': 'ipv6'}), (Pattern, {'type': 'string', 'format': 'regex'}), (str, {'type': 'string'}), (bytes, {'type': 'string', 'format': 'binary'}), (bool, {'type': 'boolean'}), (int, {'type': 'integer'}), (float, {'type': 'number'}), (Decimal, {'type': 'number'}), (UUID, {'type': 'string', 'format': 'uuid'}), (dict, {'type': 'object'}), (list, {'type': 'array', 'items': {}}), (tuple, {'type': 'array', 'items': {}}), (set, {'type': 'array', 'items': {}, 'uniqueItems': True}), (frozenset, {'type': 'array', 'items': {}, 'uniqueItems': True}))
json_scheme = {'type': 'string', 'format': 'json-string'}

def add_field_type_to_schema(field_type: Any, schema_: Dict[str, Any]) -> None:
    """
    Update the given `schema` with the type-specific metadata for the given `field_type`.

    This function looks through `field_class_to_schema` for a class that matches the given `field_type`,
    and then modifies the given `schema` with the information from that type.
    """
    pass

def field_singleton_schema(field: ModelField, *, by_alias: bool, model_name_map: Dict[TypeModelOrEnum, str], ref_template: str, schema_overrides: bool=False, ref_prefix: Optional[str]=None, known_models: TypeModelSet) -> Tuple[Dict[str, Any], Dict[str, Any], Set[str]]:
    """
    This function is indirectly used by ``field_schema()``, you should probably be using that function.

    Take a single Pydantic ``ModelField``, and return its schema and any additional definitions from sub-models.
    """
    pass

def multitypes_literal_field_for_schema(values: Tuple[Any, ...], field: ModelField) -> ModelField:
    """
    To support `Literal` with values of different types, we split it into multiple `Literal` with same type
    e.g. `Literal['qwe', 'asd', 1, 2]` becomes `Union[Literal['qwe', 'asd'], Literal[1, 2]]`
    """
    pass
_map_types_constraint: Dict[Any, Callable[..., type]] = {int: conint, float: confloat, Decimal: condecimal}

def get_annotation_from_field_info(annotation: Any, field_info: FieldInfo, field_name: str, validate_assignment: bool=False) -> Type[Any]:
    """
    Get an annotation with validation implemented for numbers and strings based on the field_info.
    :param annotation: an annotation from a field specification, as ``str``, ``ConstrainedStr``
    :param field_info: an instance of FieldInfo, possibly with declarations for validations and JSON Schema
    :param field_name: name of the field for use in error messages
    :param validate_assignment: default False, flag for BaseModel Config value of validate_assignment
    :return: the same ``annotation`` if unmodified or a new annotation with validation in place
    """
    pass

def get_annotation_with_constraints(annotation: Any, field_info: FieldInfo) -> Tuple[Type[Any], Set[str]]:
    """
    Get an annotation with used constraints implemented for numbers and strings based on the field_info.

    :param annotation: an annotation from a field specification, as ``str``, ``ConstrainedStr``
    :param field_info: an instance of FieldInfo, possibly with declarations for validations and JSON Schema
    :return: the same ``annotation`` if unmodified or a new annotation along with the used constraints.
    """
    pass

def normalize_name(name: str) -> str:
    """
    Normalizes the given name. This can be applied to either a model *or* enum.
    """
    pass

class SkipField(Exception):
    """
    Utility exception used to exclude fields from schema.
    """

    def __init__(self, message: str) -> None:
        self.message = message
