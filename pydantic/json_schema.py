"""
Usage docs: https://docs.pydantic.dev/2.5/concepts/json_schema/

The `json_schema` module contains classes and functions to allow the way [JSON Schema](https://json-schema.org/)
is generated to be customized.

In general you shouldn't need to use this module directly; instead, you can use
[`BaseModel.model_json_schema`][pydantic.BaseModel.model_json_schema] and
[`TypeAdapter.json_schema`][pydantic.TypeAdapter.json_schema].
"""
from __future__ import annotations as _annotations
import dataclasses
import inspect
import math
import re
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import is_dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Counter, Dict, Hashable, Iterable, NewType, Pattern, Sequence, Tuple, TypeVar, Union, cast
import pydantic_core
from pydantic_core import CoreSchema, PydanticOmit, core_schema, to_jsonable_python
from pydantic_core.core_schema import ComputedField
from typing_extensions import Annotated, Literal, TypeAlias, assert_never, deprecated, final
from pydantic.warnings import PydanticDeprecatedSince26
from ._internal import _config, _core_metadata, _core_utils, _decorators, _internal_dataclass, _mock_val_ser, _schema_generation_shared, _typing_extra
from .annotated_handlers import GetJsonSchemaHandler
from .config import JsonDict, JsonSchemaExtraCallable, JsonValue
from .errors import PydanticInvalidForJsonSchema, PydanticSchemaGenerationError, PydanticUserError
if TYPE_CHECKING:
    from . import ConfigDict
    from ._internal._core_utils import CoreSchemaField, CoreSchemaOrField
    from ._internal._dataclasses import PydanticDataclass
    from ._internal._schema_generation_shared import GetJsonSchemaFunction
    from .main import BaseModel
CoreSchemaOrFieldType = Literal[core_schema.CoreSchemaType, core_schema.CoreSchemaFieldType]
'\nA type alias for defined schema types that represents a union of\n`core_schema.CoreSchemaType` and\n`core_schema.CoreSchemaFieldType`.\n'
JsonSchemaValue = Dict[str, Any]
'\nA type alias for a JSON schema value. This is a dictionary of string keys to arbitrary JSON values.\n'
JsonSchemaMode = Literal['validation', 'serialization']
"\nA type alias that represents the mode of a JSON schema; either 'validation' or 'serialization'.\n\nFor some types, the inputs to validation differ from the outputs of serialization. For example,\ncomputed fields will only be present when serializing, and should not be provided when\nvalidating. This flag provides a way to indicate whether you want the JSON schema required\nfor validation inputs, or that will be matched by serialization outputs.\n"
_MODE_TITLE_MAPPING: dict[JsonSchemaMode, str] = {'validation': 'Input', 'serialization': 'Output'}

@deprecated('`update_json_schema` is deprecated, use a simple `my_dict.update(update_dict)` call instead.', category=None)
def update_json_schema(schema: JsonSchemaValue, updates: dict[str, Any]) -> JsonSchemaValue:
    """Update a JSON schema in-place by providing a dictionary of updates.

    This function sets the provided key-value pairs in the schema and returns the updated schema.

    Args:
        schema: The JSON schema to update.
        updates: A dictionary of key-value pairs to set in the schema.

    Returns:
        The updated JSON schema.
    """
    schema.update(updates)
    return schema
JsonSchemaWarningKind = Literal['skipped-choice', 'non-serializable-default']
'\nA type alias representing the kinds of warnings that can be emitted during JSON schema generation.\n\nSee [`GenerateJsonSchema.render_warning_message`][pydantic.json_schema.GenerateJsonSchema.render_warning_message]\nfor more details.\n'

class PydanticJsonSchemaWarning(UserWarning):
    """This class is used to emit warnings produced during JSON schema generation.
    See the [`GenerateJsonSchema.emit_warning`][pydantic.json_schema.GenerateJsonSchema.emit_warning] and
    [`GenerateJsonSchema.render_warning_message`][pydantic.json_schema.GenerateJsonSchema.render_warning_message]
    methods for more details; these can be overridden to control warning behavior.
    """
DEFAULT_REF_TEMPLATE = '#/$defs/{model}'
'The default format string used to generate reference names.'
CoreRef = NewType('CoreRef', str)
DefsRef = NewType('DefsRef', str)
JsonRef = NewType('JsonRef', str)
CoreModeRef = Tuple[CoreRef, JsonSchemaMode]
JsonSchemaKeyT = TypeVar('JsonSchemaKeyT', bound=Hashable)

@dataclasses.dataclass(**_internal_dataclass.slots_true)
class _DefinitionsRemapping:
    defs_remapping: dict[DefsRef, DefsRef]
    json_remapping: dict[JsonRef, JsonRef]

    @staticmethod
    def from_prioritized_choices(prioritized_choices: dict[DefsRef, list[DefsRef]], defs_to_json: dict[DefsRef, JsonRef], definitions: dict[DefsRef, JsonSchemaValue]) -> _DefinitionsRemapping:
        """
        This function should produce a remapping that replaces complex DefsRef with the simpler ones from the
        prioritized_choices such that applying the name remapping would result in an equivalent JSON schema.
        """
        defs_remapping = {}
        json_remapping = {}

        for complex_ref, simple_refs in prioritized_choices.items():
            for simple_ref in simple_refs:
                if definitions[complex_ref] == definitions[simple_ref]:
                    defs_remapping[complex_ref] = simple_ref
                    json_remapping[defs_to_json[complex_ref]] = defs_to_json[simple_ref]
                    break

        return _DefinitionsRemapping(defs_remapping, json_remapping)

    def remap_json_schema(self, schema: Any) -> Any:
        """
        Recursively update the JSON schema replacing all $refs
        """
        if isinstance(schema, dict):
            if '$ref' in schema:
                ref = schema['$ref']
                if ref in self.json_remapping:
                    schema['$ref'] = self.json_remapping[ref]
            return {k: self.remap_json_schema(v) for k, v in schema.items()}
        elif isinstance(schema, list):
            return [self.remap_json_schema(item) for item in schema]
        else:
            return schema

class GenerateJsonSchema:
    """Usage docs: https://docs.pydantic.dev/2.8/concepts/json_schema/#customizing-the-json-schema-generation-process

    A class for generating JSON schemas.

    This class generates JSON schemas based on configured parameters. The default schema dialect
    is [https://json-schema.org/draft/2020-12/schema](https://json-schema.org/draft/2020-12/schema).
    The class uses `by_alias` to configure how fields with
    multiple names are handled and `ref_template` to format reference names.

    Attributes:
        schema_dialect: The JSON schema dialect used to generate the schema. See
            [Declaring a Dialect](https://json-schema.org/understanding-json-schema/reference/schema.html#id4)
            in the JSON Schema documentation for more information about dialects.
        ignored_warning_kinds: Warnings to ignore when generating the schema. `self.render_warning_message` will
            do nothing if its argument `kind` is in `ignored_warning_kinds`;
            this value can be modified on subclasses to easily control which warnings are emitted.
        by_alias: Whether to use field aliases when generating the schema.
        ref_template: The format string used when generating reference names.
        core_to_json_refs: A mapping of core refs to JSON refs.
        core_to_defs_refs: A mapping of core refs to definition refs.
        defs_to_core_refs: A mapping of definition refs to core refs.
        json_to_defs_refs: A mapping of JSON refs to definition refs.
        definitions: Definitions in the schema.

    Args:
        by_alias: Whether to use field aliases in the generated schemas.
        ref_template: The format string to use when generating reference names.

    Raises:
        JsonSchemaError: If the instance of the class is inadvertently re-used after generating a schema.
    """
    schema_dialect = 'https://json-schema.org/draft/2020-12/schema'
    ignored_warning_kinds: set[JsonSchemaWarningKind] = {'skipped-choice'}

    def __init__(self, by_alias: bool=True, ref_template: str=DEFAULT_REF_TEMPLATE):
        self.by_alias = by_alias
        self.ref_template = ref_template
        self.core_to_json_refs: dict[CoreModeRef, JsonRef] = {}
        self.core_to_defs_refs: dict[CoreModeRef, DefsRef] = {}
        self.defs_to_core_refs: dict[DefsRef, CoreModeRef] = {}
        self.json_to_defs_refs: dict[JsonRef, DefsRef] = {}
        self.definitions: dict[DefsRef, JsonSchemaValue] = {}
        self._config_wrapper_stack = _config.ConfigWrapperStack(_config.ConfigWrapper({}))
        self._mode: JsonSchemaMode = 'validation'
        self._prioritized_defsref_choices: dict[DefsRef, list[DefsRef]] = {}
        self._collision_counter: dict[str, int] = defaultdict(int)
        self._collision_index: dict[str, int] = {}
        self._schema_type_to_method = self.build_schema_type_to_method()
        self._core_defs_invalid_for_json_schema: dict[DefsRef, PydanticInvalidForJsonSchema] = {}
        self._used = False

    def build_schema_type_to_method(self) -> dict[CoreSchemaOrFieldType, Callable[[CoreSchemaOrField], JsonSchemaValue]]:
        """Builds a dictionary mapping fields to methods for generating JSON schemas.

        Returns:
            A dictionary containing the mapping of `CoreSchemaOrFieldType` to a handler method.

        Raises:
            TypeError: If no method has been defined for generating a JSON schema for a given pydantic core schema type.
        """
        schema_type_to_method = {}
        for name in dir(self):
            if name.endswith('_schema'):
                method = getattr(self, name)
                schema_type = name[:-7]  # Remove '_schema' suffix
                if schema_type:
                    schema_type_to_method[schema_type] = method
        
        def default_handler(schema: CoreSchemaOrField) -> JsonSchemaValue:
            raise TypeError(f"No method defined for generating JSON schema for {schema['type']}")
        
        return defaultdict(lambda: default_handler, schema_type_to_method)

    def generate_definitions(self, inputs: Sequence[tuple[JsonSchemaKeyT, JsonSchemaMode, core_schema.CoreSchema]]) -> tuple[dict[tuple[JsonSchemaKeyT, JsonSchemaMode], JsonSchemaValue], dict[DefsRef, JsonSchemaValue]]:
        """Generates JSON schema definitions from a list of core schemas, pairing the generated definitions with a
        mapping that links the input keys to the definition references.

        Args:
            inputs: A sequence of tuples, where:

                - The first element is a JSON schema key type.
                - The second element is the JSON mode: either 'validation' or 'serialization'.
                - The third element is a core schema.

        Returns:
            A tuple where:

                - The first element is a dictionary whose keys are tuples of JSON schema key type and JSON mode, and
                    whose values are the JSON schema corresponding to that pair of inputs. (These schemas may have
                    JsonRef references to definitions that are defined in the second returned element.)
                - The second element is a dictionary whose keys are definition references for the JSON schemas
                    from the first returned element, and whose values are the actual JSON schema definitions.

        Raises:
            PydanticUserError: Raised if the JSON schema generator has already been used to generate a JSON schema.
        """
        if self._used:
            raise PydanticUserError('GenerateJsonSchema instance already used', code='json-schema-already-used')
        self._used = True

        schema_definitions: dict[tuple[JsonSchemaKeyT, JsonSchemaMode], JsonSchemaValue] = {}
        for key, mode, core_schema in inputs:
            self._mode = mode
            schema = self.generate(core_schema)
            schema_definitions[(key, mode)] = schema

        return schema_definitions, self.definitions

    def generate(self, schema: CoreSchema, mode: JsonSchemaMode='validation') -> JsonSchemaValue:
        """Generates a JSON schema for a specified schema in a specified mode.

        Args:
            schema: A Pydantic model.
            mode: The mode in which to generate the schema. Defaults to 'validation'.

        Returns:
            A JSON schema representing the specified schema.

        Raises:
            PydanticUserError: If the JSON schema generator has already been used to generate a JSON schema.
        """
        if self._used:
            raise PydanticUserError('GenerateJsonSchema instance already used', code='json-schema-already-used')
        self._used = True
        self._mode = mode
        return self.generate_inner(schema)

    def generate_inner(self, schema: CoreSchemaOrField) -> JsonSchemaValue:
        """Generates a JSON schema for a given core schema.

        Args:
            schema: The given core schema.

        Returns:
            The generated JSON schema.
        """
        schema_type = schema.get('type', 'any')
        handler = self._schema_type_to_method[schema_type]
        json_schema = handler(schema)
        
        if 'title' in schema:
            json_schema['title'] = schema['title']
        if 'description' in schema:
            json_schema['description'] = schema['description']
        
        return json_schema

    def any_schema(self, schema: core_schema.AnySchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches any value.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        return {}  # An empty schema matches any value

    def none_schema(self, schema: core_schema.NoneSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches `None`.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        return {'type': 'null'}

    def bool_schema(self, schema: core_schema.BoolSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a bool value.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        return {'type': 'boolean'}

    def int_schema(self, schema: core_schema.IntSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches an int value.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        json_schema: JsonSchemaValue = {'type': 'integer'}
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.numeric)
        return json_schema

    def float_schema(self, schema: core_schema.FloatSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a float value.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        json_schema: JsonSchemaValue = {'type': 'number'}
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.numeric)
        return json_schema

    def decimal_schema(self, schema: core_schema.DecimalSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a decimal value.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        json_schema: JsonSchemaValue = {'type': 'string', 'format': 'decimal'}
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.numeric)
        return json_schema

    def str_schema(self, schema: core_schema.StringSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a string value.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        json_schema: JsonSchemaValue = {'type': 'string'}
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.string)
        return json_schema

    def bytes_schema(self, schema: core_schema.BytesSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a bytes value.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        json_schema: JsonSchemaValue = {'type': 'string', 'format': 'binary'}
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.bytes)
        return json_schema

    def date_schema(self, schema: core_schema.DateSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a date value.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        json_schema: JsonSchemaValue = {'type': 'string', 'format': 'date'}
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.date)
        return json_schema

    def time_schema(self, schema: core_schema.TimeSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a time value.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        json_schema: JsonSchemaValue = {'type': 'string', 'format': 'time'}
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.numeric)
        return json_schema

    def datetime_schema(self, schema: core_schema.DatetimeSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a datetime value.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        json_schema: JsonSchemaValue = {'type': 'string', 'format': 'date-time'}
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.numeric)
        return json_schema

    def timedelta_schema(self, schema: core_schema.TimedeltaSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a timedelta value.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        json_schema: JsonSchemaValue = {'type': 'string', 'format': 'duration'}
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.numeric)
        return json_schema

    def literal_schema(self, schema: core_schema.LiteralSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a literal value.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        expected = schema.get('expected')
        if len(expected) == 1:
            return {'const': expected[0]}
        else:
            return {'enum': list(expected)}

    def enum_schema(self, schema: core_schema.EnumSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches an Enum value.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        enum_values = [member.value for member in schema['members']]
        return {'enum': enum_values}

    def is_instance_schema(self, schema: core_schema.IsInstanceSchema) -> JsonSchemaValue:
        """Handles JSON schema generation for a core schema that checks if a value is an instance of a class.

        Unless overridden in a subclass, this raises an error.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        raise NotImplementedError(f"JSON schema generation not implemented for IsInstanceSchema: {schema}")

    def is_subclass_schema(self, schema: core_schema.IsSubclassSchema) -> JsonSchemaValue:
        """Handles JSON schema generation for a core schema that checks if a value is a subclass of a class.

        For backwards compatibility with v1, this does not raise an error, but can be overridden to change this.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        return {}  # Return an empty schema for backwards compatibility

    def callable_schema(self, schema: core_schema.CallableSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a callable value.

        Unless overridden in a subclass, this raises an error.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        raise NotImplementedError(f"JSON schema generation not implemented for CallableSchema: {schema}")

    def list_schema(self, schema: core_schema.ListSchema) -> JsonSchemaValue:
        """Returns a schema that matches a list schema.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        json_schema: JsonSchemaValue = {
            'type': 'array',
            'items': self.generate_inner(schema['items_schema'])
        }
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.array)
        return json_schema

    @deprecated('`tuple_positional_schema` is deprecated. Use `tuple_schema` instead.', category=None)
    @final
    def tuple_positional_schema(self, schema: core_schema.TupleSchema) -> JsonSchemaValue:
        """Replaced by `tuple_schema`."""
        return self.tuple_schema(schema)

    @deprecated('`tuple_variable_schema` is deprecated. Use `tuple_schema` instead.', category=None)
    @final
    def tuple_variable_schema(self, schema: core_schema.TupleSchema) -> JsonSchemaValue:
        """Replaced by `tuple_schema`."""
        return self.tuple_schema(schema)

    def tuple_schema(self, schema: core_schema.TupleSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a tuple schema e.g. `Tuple[int,
        str, bool]` or `Tuple[int, ...]`.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        json_schema: JsonSchemaValue = {'type': 'array'}
        
        if 'items_schema' in schema:
            # Variable length tuple
            json_schema['items'] = self.generate_inner(schema['items_schema'])
        else:
            # Fixed length tuple
            json_schema['items'] = [self.generate_inner(item) for item in schema['items_schemas']]
            json_schema['minItems'] = json_schema['maxItems'] = len(schema['items_schemas'])
        
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.array)
        return json_schema

    def set_schema(self, schema: core_schema.SetSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a set schema.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        json_schema: JsonSchemaValue = {
            'type': 'array',
            'uniqueItems': True,
            'items': self.generate_inner(schema['items_schema'])
        }
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.array)
        return json_schema

    def frozenset_schema(self, schema: core_schema.FrozenSetSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a frozenset schema.

        Args: 
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        json_schema: JsonSchemaValue = {
            'type': 'array',
            'uniqueItems': True,
            'items': self.generate_inner(schema['items_schema'])
        }
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.array)
        return json_schema

    def generator_schema(self, schema: core_schema.GeneratorSchema) -> JsonSchemaValue:
        """Returns a JSON schema that represents the provided GeneratorSchema.

        Args:
            schema: The schema.

        Returns:
            The generated JSON schema.
        """
        json_schema: JsonSchemaValue = {
            'type': 'array',
            'items': self.generate_inner(schema['items_schema'])
        }
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.array)
        return json_schema

    def dict_schema(self, schema: core_schema.DictSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a dict schema.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        json_schema: JsonSchemaValue = {
            'type': 'object',
            'additionalProperties': self.generate_inner(schema['values_schema'])
        }
        
        if 'keys_schema' in schema:
            json_schema['propertyNames'] = self.generate_inner(schema['keys_schema'])
        
        self.update_with_validations(json_schema, schema, self.ValidationsMapping.object)
        return json_schema

    def function_before_schema(self, schema: core_schema.BeforeValidatorFunctionSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a function-before schema.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        return self.generate_inner(schema['schema'])

    def function_after_schema(self, schema: core_schema.AfterValidatorFunctionSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a function-after schema.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        return self.generate_inner(schema['schema'])

    def function_plain_schema(self, schema: core_schema.PlainValidatorFunctionSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a function-plain schema.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        return {}  # Plain validator functions don't provide enough information for a meaningful JSON schema

    def function_wrap_schema(self, schema: core_schema.WrapValidatorFunctionSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a function-wrap schema.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        return self.generate_inner(schema['schema'])

    def default_schema(self, schema: core_schema.WithDefaultSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a schema with a default value.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        json_schema = self.generate_inner(schema['schema'])
        if 'default' in schema:
            json_schema['default'] = self.encode_default(schema['default'])
        return json_schema

    def nullable_schema(self, schema: core_schema.NullableSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a schema that allows null values.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        inner_schema = self.generate_inner(schema['schema'])
        if inner_schema.get('type') == 'null':
            return inner_schema
        elif 'type' in inner_schema:
            inner_schema['type'] = [inner_schema['type'], 'null']
        else:
            inner_schema['anyOf'] = [inner_schema, {'type': 'null'}]
        return inner_schema

    def union_schema(self, schema: core_schema.UnionSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a schema that allows values matching any of the given schemas.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        any_of = [self.generate_inner(s) for s in schema['choices']]
        return {'anyOf': any_of}

    def tagged_union_schema(self, schema: core_schema.TaggedUnionSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a schema that allows values matching any of the given schemas, where
        the schemas are tagged with a discriminator field that indicates which schema should be used to validate
        the value.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        one_of = []
        for choice in schema['choices']:
            choice_schema = self.generate_inner(choice['schema'])
            if isinstance(choice['schema'], core_schema.ModelSchema):
                choice_schema['properties'] = choice_schema.get('properties', {})
                choice_schema['properties'][schema['discriminator']] = {'const': choice['tag']}
            one_of.append(choice_schema)
        
        json_schema = {'oneOf': one_of}
        
        discriminator = self._extract_discriminator(schema, one_of)
        if discriminator:
            json_schema['discriminator'] = {'propertyName': discriminator}
        
        return json_schema

    def _extract_discriminator(self, schema: core_schema.TaggedUnionSchema, one_of_choices: list[JsonDict]) -> str | None:
        """Extract a compatible OpenAPI discriminator from the schema and one_of choices that end up in the final
        schema."""
        discriminator = schema['discriminator']
        for choice in one_of_choices:
            if 'properties' not in choice or discriminator not in choice['properties']:
                return None
        return discriminator

    def chain_schema(self, schema: core_schema.ChainSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a core_schema.ChainSchema.

        When generating a schema for validation, we return the validation JSON schema for the first step in the chain.
        For serialization, we return the serialization JSON schema for the last step in the chain.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        if self._mode == 'validation':
            return self.generate_inner(schema['steps'][0])
        else:  # serialization
            return self.generate_inner(schema['steps'][-1])

    def lax_or_strict_schema(self, schema: core_schema.LaxOrStrictSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a schema that allows values matching either the lax schema or the
        strict schema.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        if schema['strict']:
            return self.generate_inner(schema['strict_schema'])
        else:
            return self.generate_inner(schema['lax_schema'])

    def json_or_python_schema(self, schema: core_schema.JsonOrPythonSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a schema that allows values matching either the JSON schema or the
        Python schema.

        The JSON schema is used instead of the Python schema. If you want to use the Python schema, you should override
        this method.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        return self.generate_inner(schema['json_schema'])

    def typed_dict_schema(self, schema: core_schema.TypedDictSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a schema that defines a typed dict.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        json_schema: JsonSchemaValue = {'type': 'object', 'properties': {}}
        required = []

        for field_name, field_schema in schema['fields'].items():
            field_json_schema = self.generate_inner(field_schema)
            json_schema['properties'][field_name] = field_json_schema

            if self.field_is_required(field_schema, schema.get('total', False)):
                required.append(field_name)

        if required:
            json_schema['required'] = required

        return json_schema

    def typed_dict_field_schema(self, schema: core_schema.TypedDictField) -> JsonSchemaValue:
        """Generates a JSON schema that matches a schema that defines a typed dict field.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        return self.generate_inner(schema['schema'])

    def dataclass_field_schema(self, schema: core_schema.DataclassField) -> JsonSchemaValue:
        """Generates a JSON schema that matches a schema that defines a dataclass field.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        json_schema = self.generate_inner(schema['schema'])
        if 'default' in schema:
            json_schema['default'] = self.encode_default(schema['default'])
        return json_schema

    def model_field_schema(self, schema: core_schema.ModelField) -> JsonSchemaValue:
        """Generates a JSON schema that matches a schema that defines a model field.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        json_schema = self.generate_inner(schema['schema'])
        if 'default' in schema:
            json_schema['default'] = self.encode_default(schema['default'])
        return json_schema

    def computed_field_schema(self, schema: core_schema.ComputedField) -> JsonSchemaValue:
        """Generates a JSON schema that matches a schema that defines a computed field.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        if self._mode == 'serialization':
            return self.generate_inner(schema['return_schema'])
        else:
            return {}  # Computed fields are not present during validation

    def model_schema(self, schema: core_schema.ModelSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a schema that defines a model.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        json_schema = self.model_fields_schema(schema)
        
        if schema.get('title') is not None:
            json_schema['title'] = schema['title']
        
        if schema.get('description') is not None:
            json_schema['description'] = schema['description']
        
        return json_schema

    def resolve_schema_to_update(self, json_schema: JsonSchemaValue) -> JsonSchemaValue:
        """Resolve a JsonSchemaValue to the non-ref schema if it is a $ref schema.

        Args:
            json_schema: The schema to resolve.

        Returns:
            The resolved schema.
        """
        if '$ref' in json_schema:
            ref = json_schema['$ref']
            if ref.startswith('#/$defs/'):
                return self.definitions[DefsRef(ref[len('#/$defs/'):])].copy()
        return json_schema

    def model_fields_schema(self, schema: core_schema.ModelFieldsSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a schema that defines a model's fields.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        json_schema: JsonSchemaValue = {'type': 'object', 'properties': {}}
        required = []

        for field_name, field in schema['fields'].items():
            if self.field_is_present(field):
                field_schema = self.generate_inner(field['schema'])
                json_schema['properties'][field_name] = field_schema

                if self.field_is_required(field, schema.get('extra', {}).get('total', True)):
                    required.append(field_name)

        if required:
            json_schema['required'] = required

        return json_schema

    def field_is_present(self, field: CoreSchemaField) -> bool:
        """Whether the field should be included in the generated JSON schema.

        Args:
            field: The schema for the field itself.

        Returns:
            `True` if the field should be included in the generated JSON schema, `False` otherwise.
        """
        if self._mode == 'serialization':
            return not field.get('serialization_exclude', False)
        else:
            return not field.get('validation_exclude', False)

    def field_is_required(self, field: core_schema.ModelField | core_schema.DataclassField | core_schema.TypedDictField, total: bool) -> bool:
        """Whether the field should be marked as required in the generated JSON schema.
        (Note that this is irrelevant if the field is not present in the JSON schema.).

        Args:
            field: The schema for the field itself.
            total: Only applies to `TypedDictField`s.
                Indicates if the `TypedDict` this field belongs to is total, in which case any fields that don't
                explicitly specify `required=False` are required.

        Returns:
            `True` if the field should be marked as required in the generated JSON schema, `False` otherwise.
        """
        if isinstance(field, core_schema.TypedDictField):
            return total if field.get('required') is None else field['required']
        else:
            return not field.get('default') and 'default_factory' not in field

    def dataclass_args_schema(self, schema: core_schema.DataclassArgsSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a schema that defines a dataclass's constructor arguments.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        json_schema: JsonSchemaValue = {'type': 'object', 'properties': {}}
        required = []

        for field in schema['fields']:
            field_schema = self.generate_inner(field['schema'])
            json_schema['properties'][field['name']] = field_schema

            if self.field_is_required(field, True):
                required.append(field['name'])

        if required:
            json_schema['required'] = required

        return json_schema

    def dataclass_schema(self, schema: core_schema.DataclassSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a schema that defines a dataclass.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        json_schema = self.dataclass_args_schema(schema)
        
        if schema.get('title') is not None:
            json_schema['title'] = schema['title']
        
        if schema.get('description') is not None:
            json_schema['description'] = schema['description']
        
        return json_schema

    def arguments_schema(self, schema: core_schema.ArgumentsSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a schema that defines a function's arguments.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        json_schema: JsonSchemaValue = {'type': 'object'}
        
        positional_schema = self.p_arguments_schema(schema['arguments'], schema.get('var_args_schema'))
        if positional_schema:
            json_schema['properties'] = positional_schema['properties']
            if 'required' in positional_schema:
                json_schema['required'] = positional_schema['required']
        
        kwargs_schema = self.kw_arguments_schema(schema['arguments'], schema.get('var_kwargs_schema'))
        if kwargs_schema:
            json_schema['properties'] = {**json_schema.get('properties', {}), **kwargs_schema['properties']}
            if 'required' in kwargs_schema:
                json_schema['required'] = json_schema.get('required', []) + kwargs_schema['required']
        
        return json_schema

    def kw_arguments_schema(self, arguments: list[core_schema.ArgumentsParameter], var_kwargs_schema: CoreSchema | None) -> JsonSchemaValue:
        """Generates a JSON schema that matches a schema that defines a function's keyword arguments.

        Args:
            arguments: The core schema.

        Returns:
            The generated JSON schema.
        """
        json_schema: JsonSchemaValue = {'type': 'object', 'properties': {}}
        required = []

        for arg in arguments:
            if arg['kind'] in {'keyword_only', 'positional_or_keyword'}:
                arg_name = self.get_argument_name(arg)
                json_schema['properties'][arg_name] = self.generate_inner(arg['schema'])
                if arg['default'] is None:
                    required.append(arg_name)

        if var_kwargs_schema:
            json_schema['additionalProperties'] = self.generate_inner(var_kwargs_schema)
        else:
            json_schema['additionalProperties'] = False

        if required:
            json_schema['required'] = required

        return json_schema

    def p_arguments_schema(self, arguments: list[core_schema.ArgumentsParameter], var_args_schema: CoreSchema | None) -> JsonSchemaValue:
        """Generates a JSON schema that matches a schema that defines a function's positional arguments.

        Args:
            arguments: The core schema.

        Returns:
            The generated JSON schema.
        """
        json_schema: JsonSchemaValue = {'type': 'object', 'properties': {}}
        required = []

        for arg in arguments:
            if arg['kind'] in {'positional_only', 'positional_or_keyword'}:
                arg_name = self.get_argument_name(arg)
                json_schema['properties'][arg_name] = self.generate_inner(arg['schema'])
                if arg['default'] is None:
                    required.append(arg_name)

        if var_args_schema:
            json_schema['additionalItems'] = self.generate_inner(var_args_schema)

        if required:
            json_schema['required'] = required

        return json_schema

    def get_argument_name(self, argument: core_schema.ArgumentsParameter) -> str:
        """Retrieves the name of an argument.

        Args:
            argument: The core schema.

        Returns:
            The name of the argument.
        """
        return argument['name']

    def call_schema(self, schema: core_schema.CallSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a schema that defines a function call.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        return self.generate_inner(schema['arguments_schema'])

    def custom_error_schema(self, schema: core_schema.CustomErrorSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a schema that defines a custom error.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        return self.generate_inner(schema['schema'])

    def json_schema(self, schema: core_schema.JsonSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a schema that defines a JSON object.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        return schema['schema']

    def url_schema(self, schema: core_schema.UrlSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a schema that defines a URL.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        return {'type': 'string', 'format': 'uri'}

    def multi_host_url_schema(self, schema: core_schema.MultiHostUrlSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a schema that defines a URL that can be used with multiple hosts.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        return {'type': 'string', 'format': 'uri'}

    def uuid_schema(self, schema: core_schema.UuidSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a UUID.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        return {'type': 'string', 'format': 'uuid'}

    def definitions_schema(self, schema: core_schema.DefinitionsSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a schema that defines a JSON object with definitions.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        for definition in schema['definitions']:
            self.generate_inner(definition)
        return self.generate_inner(schema['schema'])

    def definition_ref_schema(self, schema: core_schema.DefinitionReferenceSchema) -> JsonSchemaValue:
        """Generates a JSON schema that matches a schema that references a definition.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        ref, json_schema = self.get_cache_defs_ref_schema(schema['schema_ref'])
        return {'$ref': f'#/$defs/{ref}'}

    def ser_schema(self, schema: core_schema.SerSchema | core_schema.IncExSeqSerSchema | core_schema.IncExDictSerSchema) -> JsonSchemaValue | None:
        """Generates a JSON schema that matches a schema that defines a serialized object.

        Args:
            schema: The core schema.

        Returns:
            The generated JSON schema.
        """
        if self._mode == 'serialization':
            return self.generate_inner(schema['schema'])
        else:
            return None

    def get_title_from_name(self, name: str) -> str:
        """Retrieves a title from a name.

        Args:
            name: The name to retrieve a title from.

        Returns:
            The title.
        """
        return name.replace('_', ' ').title()

    def field_title_should_be_set(self, schema: CoreSchemaOrField) -> bool:
        """Returns true if a field with the given schema should have a title set based on the field name.

        Intuitively, we want this to return true for schemas that wouldn't otherwise provide their own title
        (e.g., int, float, str), and false for those that would (e.g., BaseModel subclasses).

        Args:
            schema: The schema to check.

        Returns:
            `True` if the field should have a title set, `False` otherwise.
        """
        return schema.get('type') in {'int', 'float', 'str', 'bool', 'bytes', 'date', 'time', 'datetime', 'timedelta'}

    def normalize_name(self, name: str) -> str:
        """Normalizes a name to be used as a key in a dictionary.

        Args:
            name: The name to normalize.

        Returns:
            The normalized name.
        """
        return re.sub(r'[^a-zA-Z0-9_]+', '_', name)

    def get_defs_ref(self, core_mode_ref: CoreModeRef) -> DefsRef:
        """Override this method to change the way that definitions keys are generated from a core reference.

        Args:
            core_mode_ref: The core reference.

        Returns:
            The definitions key.
        """
        core_ref, mode = core_mode_ref
        name = core_ref.split('/')[-1]
        normalized_name = self.normalize_name(name)
        return DefsRef(f'{normalized_name}__{mode}')

    def get_cache_defs_ref_schema(self, core_ref: CoreRef) -> tuple[DefsRef, JsonSchemaValue]:
        """This method wraps the get_defs_ref method with some cache-lookup/population logic,
        and returns both the produced defs_ref and the JSON schema that will refer to the right definition.

        Args:
            core_ref: The core reference to get the definitions reference for.

        Returns:
            A tuple of the definitions reference and the JSON schema that will refer to it.
        """
        core_mode_ref = (core_ref, self._mode)
        if core_mode_ref in self.core_to_defs_refs:
            defs_ref = self.core_to_defs_refs[core_mode_ref]
        else:
            defs_ref = self.get_defs_ref(core_mode_ref)
            self.core_to_defs_refs[core_mode_ref] = defs_ref
            self.defs_to_core_refs[defs_ref] = core_mode_ref

        json_ref = JsonRef(self.ref_template.format(model=defs_ref))
        self.core_to_json_refs[core_mode_ref] = json_ref
        self.json_to_defs_refs[json_ref] = defs_ref

        return defs_ref, {'$ref': json_ref}

    def handle_ref_overrides(self, json_schema: JsonSchemaValue) -> JsonSchemaValue:
        """It is not valid for a schema with a top-level $ref to have sibling keys.

        During our own schema generation, we treat sibling keys as overrides to the referenced schema,
        but this is not how the official JSON schema spec works.

        Because of this, we first remove any sibling keys that are redundant with the referenced schema, then if
        any remain, we transform the schema from a top-level '$ref' to use allOf to move the $ref out of the top level.
        (See bottom of https://swagger.io/docs/specification/using-ref/ for a reference about this behavior)
        """
        if '$ref' not in json_schema:
            return json_schema

        ref = json_schema['$ref']
        if not ref.startswith('#/$defs/'):
            return json_schema

        referenced_schema = self.definitions[DefsRef(ref[len('#/$defs/'):])]
        overrides = {k: v for k, v in json_schema.items() if k != '$ref'}
        for key, value in list(overrides.items()):
            if key in referenced_schema and referenced_schema[key] == value:
                del overrides[key]

        if not overrides:
            return {'$ref': ref}
        elif len(overrides) == len(json_schema) - 1:
            return json_schema
        else:
            return {'allOf': [{'$ref': ref}, overrides]}

    def encode_default(self, dft: Any) -> Any:
        """Encode a default value to a JSON-serializable value.

        This is used to encode default values for fields in the generated JSON schema.

        Args:
            dft: The default value to encode.

        Returns:
            The encoded default value.
        """
        try:
            return pydantic_core.to_jsonable_python(dft)
        except pydantic_core.PydanticSerializationError as e:
            self.emit_warning('non-serializable-default', f'Default value {dft} is not JSON serializable: {e}')
            return None

    def update_with_validations(self, json_schema: JsonSchemaValue, core_schema: CoreSchema, mapping: dict[str, str]) -> None:
        """Update the json_schema with the corresponding validations specified in the core_schema,
        using the provided mapping to translate keys in core_schema to the appropriate keys for a JSON schema.

        Args:
            json_schema: The JSON schema to update.
            core_schema: The core schema to get the validations from.
            mapping: A mapping from core_schema attribute names to the corresponding JSON schema attribute names.
        """
        for core_key, json_key in mapping.items():
            if core_key in core_schema:
                json_schema[json_key] = core_schema[core_key]

    class ValidationsMapping:
        """This class just contains mappings from core_schema attribute names to the corresponding
        JSON schema attribute names. While I suspect it is unlikely to be necessary, you can in
        principle override this class in a subclass of GenerateJsonSchema (by inheriting from
        GenerateJsonSchema.ValidationsMapping) to change these mappings.
        """
        numeric = {'multiple_of': 'multipleOf', 'le': 'maximum', 'ge': 'minimum', 'lt': 'exclusiveMaximum', 'gt': 'exclusiveMinimum'}
        bytes = {'min_length': 'minLength', 'max_length': 'maxLength'}
        string = {'min_length': 'minLength', 'max_length': 'maxLength', 'pattern': 'pattern'}
        array = {'min_length': 'minItems', 'max_length': 'maxItems'}
        object = {'min_length': 'minProperties', 'max_length': 'maxProperties'}
        date = {'le': 'maximum', 'ge': 'minimum', 'lt': 'exclusiveMaximum', 'gt': 'exclusiveMinimum'}

    def get_json_ref_counts(self, json_schema: JsonSchemaValue) -> dict[JsonRef, int]:
        """Get all values corresponding to the key '$ref' anywhere in the json_schema."""
        ref_counts: dict[JsonRef, int] = {}

        def count_refs(schema: Any) -> None:
            if isinstance(schema, dict):
                if '$ref' in schema:
                    ref = JsonRef(schema['$ref'])
                    ref_counts[ref] = ref_counts.get(ref, 0) + 1
                for value in schema.values():
                    count_refs(value)
            elif isinstance(schema, list):
                for item in schema:
                    count_refs(item)

        count_refs(json_schema)
        return ref_counts

    def emit_warning(self, kind: JsonSchemaWarningKind, detail: str) -> None:
        """This method simply emits PydanticJsonSchemaWarnings based on handling in the `warning_message` method."""
        message = self.render_warning_message(kind, detail)
        if message is not None:
            warnings.warn(PydanticJsonSchemaWarning(message), stacklevel=2)

    def render_warning_message(self, kind: JsonSchemaWarningKind, detail: str) -> str | None:
        """This method is responsible for ignoring warnings as desired, and for formatting the warning messages.

        You can override the value of `ignored_warning_kinds` in a subclass of GenerateJsonSchema
        to modify what warnings are generated. If you want more control, you can override this method;
        just return None in situations where you don't want warnings to be emitted.

        Args:
            kind: The kind of warning to render. It can be one of the following:

                - 'skipped-choice': A choice field was skipped because it had no valid choices.
                - 'non-serializable-default': A default value was skipped because it was not JSON-serializable.
            detail: A string with additional details about the warning.

        Returns:
            The formatted warning message, or `None` if no warning should be emitted.
        """
        if kind in self.ignored_warning_kinds:
            return None
        return f'{kind}: {detail}'

def model_json_schema(cls: type[BaseModel] | type[PydanticDataclass], by_alias: bool=True, ref_template: str=DEFAULT_REF_TEMPLATE, schema_generator: type[GenerateJsonSchema]=GenerateJsonSchema, mode: JsonSchemaMode='validation') -> dict[str, Any]:
    """Utility function to generate a JSON Schema for a model.

    Args:
        cls: The model class to generate a JSON Schema for.
        by_alias: If `True` (the default), fields will be serialized according to their alias.
            If `False`, fields will be serialized according to their attribute name.
        ref_template: The template to use for generating JSON Schema references.
        schema_generator: The class to use for generating the JSON Schema.
        mode: The mode to use for generating the JSON Schema. It can be one of the following:

            - 'validation': Generate a JSON Schema for validating data.
            - 'serialization': Generate a JSON Schema for serializing data.

    Returns:
        The generated JSON Schema.
    """
    generator = schema_generator(by_alias=by_alias, ref_template=ref_template)
    schema = generator.generate(cls.__pydantic_core_schema__, mode=mode)
    return generator.handle_ref_overrides(schema)

def models_json_schema(models: Sequence[tuple[type[BaseModel] | type[PydanticDataclass], JsonSchemaMode]], *, by_alias: bool=True, title: str | None=None, description: str | None=None, ref_template: str=DEFAULT_REF_TEMPLATE, schema_generator: type[GenerateJsonSchema]=GenerateJsonSchema) -> tuple[dict[tuple[type[BaseModel] | type[PydanticDataclass], JsonSchemaMode], JsonSchemaValue], JsonSchemaValue]:
    """Utility function to generate a JSON Schema for multiple models.

    Args:
        models: A sequence of tuples of the form (model, mode).
        by_alias: Whether field aliases should be used as keys in the generated JSON Schema.
        title: The title of the generated JSON Schema.
        description: The description of the generated JSON Schema.
        ref_template: The reference template to use for generating JSON Schema references.
        schema_generator: The schema generator to use for generating the JSON Schema.

    Returns:
        A tuple where:
            - The first element is a dictionary whose keys are tuples of JSON schema key type and JSON mode, and
                whose values are the JSON schema corresponding to that pair of inputs. (These schemas may have
                JsonRef references to definitions that are defined in the second returned element.)
            - The second element is a JSON schema containing all definitions referenced in the first returned
                    element, along with the optional title and description keys.
    """
    generator = schema_generator(by_alias=by_alias, ref_template=ref_template)
    model_schemas, definitions = generator.generate_definitions(
        [(model, mode, model.__pydantic_core_schema__) for model, mode in models]
    )

    json_schema: JsonSchemaValue = {'$defs': definitions}
    if title:
        json_schema['title'] = title
    if description:
        json_schema['description'] = description

    return model_schemas, json_schema
_HashableJsonValue: TypeAlias = Union[int, float, str, bool, None, Tuple['_HashableJsonValue', ...], Tuple[Tuple[str, '_HashableJsonValue'], ...]]

@dataclasses.dataclass(**_internal_dataclass.slots_true)
class WithJsonSchema:
    """Usage docs: https://docs.pydantic.dev/2.8/concepts/json_schema/#withjsonschema-annotation

    Add this as an annotation on a field to override the (base) JSON schema that would be generated for that field.
    This provides a way to set a JSON schema for types that would otherwise raise errors when producing a JSON schema,
    such as Callable, or types that have an is-instance core schema, without needing to go so far as creating a
    custom subclass of pydantic.json_schema.GenerateJsonSchema.
    Note that any _modifications_ to the schema that would normally be made (such as setting the title for model fields)
    will still be performed.

    If `mode` is set this will only apply to that schema generation mode, allowing you
    to set different json schemas for validation and serialization.
    """
    json_schema: JsonSchemaValue | None
    mode: Literal['validation', 'serialization'] | None = None

    def __get_pydantic_json_schema__(self, core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        mode = self.mode or handler.mode
        if mode != handler.mode:
            return handler(core_schema)
        if self.json_schema is None:
            raise PydanticOmit
        else:
            return self.json_schema

    def __hash__(self) -> int:
        return hash(type(self.mode))

@dataclasses.dataclass(**_internal_dataclass.slots_true)
class Examples:
    """Add examples to a JSON schema.

    Examples should be a map of example names (strings)
    to example values (any valid JSON).

    If `mode` is set this will only apply to that schema generation mode,
    allowing you to add different examples for validation and serialization.
    """
    examples: dict[str, Any]
    mode: Literal['validation', 'serialization'] | None = None

    def __get_pydantic_json_schema__(self, core_schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        mode = self.mode or handler.mode
        json_schema = handler(core_schema)
        if mode != handler.mode:
            return json_schema
        examples = json_schema.get('examples', {})
        examples.update(to_jsonable_python(self.examples))
        json_schema['examples'] = examples
        return json_schema

    def __hash__(self) -> int:
        return hash(type(self.mode))

def _get_all_json_refs(item: Any) -> set[JsonRef]:
    """Get all the definitions references from a JSON schema."""
    refs: set[JsonRef] = set()
    if isinstance(item, dict):
        for key, value in item.items():
            if key == '$ref' and isinstance(value, str):
                refs.add(JsonRef(value))
            else:
                refs.update(_get_all_json_refs(value))
    elif isinstance(item, list):
        for value in item:
            refs.update(_get_all_json_refs(value))
    return refs
AnyType = TypeVar('AnyType')
if TYPE_CHECKING:
    SkipJsonSchema = Annotated[AnyType, ...]
else:

    @dataclasses.dataclass(**_internal_dataclass.slots_true)
    class SkipJsonSchema:
        """Usage docs: https://docs.pydantic.dev/2.8/concepts/json_schema/#skipjsonschema-annotation

        Add this as an annotation on a field to skip generating a JSON schema for that field.

        Example:
            ```py
            from typing import Union

            from pydantic import BaseModel
            from pydantic.json_schema import SkipJsonSchema

            from pprint import pprint


            class Model(BaseModel):
                a: Union[int, None] = None  # (1)!
                b: Union[int, SkipJsonSchema[None]] = None  # (2)!
                c: SkipJsonSchema[Union[int, None]] = None  # (3)!


            pprint(Model.model_json_schema())
            '''
            {
                'properties': {
                    'a': {
                        'anyOf': [
                            {'type': 'integer'},
                            {'type': 'null'}
                        ],
                        'default': None,
                        'title': 'A'
                    },
                    'b': {
                        'default': None,
                        'title': 'B',
                        'type': 'integer'
                    }
                },
                'title': 'Model',
                'type': 'object'
            }
            '''
            ```

            1. The integer and null types are both included in the schema for `a`.
            2. The integer type is the only type included in the schema for `b`.
            3. The entirety of the `c` field is omitted from the schema.
        """

        def __class_getitem__(cls, item: AnyType) -> AnyType:
            return Annotated[item, cls()]

        def __get_pydantic_json_schema__(self, core_schema: CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
            raise PydanticOmit

        def __hash__(self) -> int:
            return hash(type(self))
