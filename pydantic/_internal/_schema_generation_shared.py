"""Types and utility functions used by various other internal tools."""
from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable
from pydantic_core import core_schema
from typing_extensions import Literal
from ..annotated_handlers import GetCoreSchemaHandler, GetJsonSchemaHandler
if TYPE_CHECKING:
    from ..json_schema import GenerateJsonSchema, JsonSchemaValue
    from ._core_utils import CoreSchemaOrField
    from ._generate_schema import GenerateSchema
    GetJsonSchemaFunction = Callable[[CoreSchemaOrField, GetJsonSchemaHandler], JsonSchemaValue]
    HandlerOverride = Callable[[CoreSchemaOrField], JsonSchemaValue]

class GenerateJsonSchemaHandler(GetJsonSchemaHandler):
    """JsonSchemaHandler implementation that doesn't do ref unwrapping by default.

    This is used for any Annotated metadata so that we don't end up with conflicting
    modifications to the definition schema.

    Used internally by Pydantic, please do not rely on this implementation.
    See `GetJsonSchemaHandler` for the handler API.
    """

    def __init__(self, generate_json_schema: GenerateJsonSchema, handler_override: HandlerOverride | None) -> None:
        self.generate_json_schema = generate_json_schema
        self.handler = handler_override or generate_json_schema.generate_inner
        self.mode = generate_json_schema.mode

    def __call__(self, core_schema: CoreSchemaOrField, /) -> JsonSchemaValue:
        return self.handler(core_schema)

    def resolve_ref_schema(self, maybe_ref_json_schema: JsonSchemaValue) -> JsonSchemaValue:
        """Resolves `$ref` in the json schema.

        This returns the input json schema if there is no `$ref` in json schema.

        Args:
            maybe_ref_json_schema: The input json schema that may contains `$ref`.

        Returns:
            Resolved json schema.

        Raises:
            LookupError: If it can't find the definition for `$ref`.
        """
        if isinstance(maybe_ref_json_schema, dict) and '$ref' in maybe_ref_json_schema:
            ref = maybe_ref_json_schema['$ref']
            if not ref.startswith('#/definitions/'):
                raise ValueError(f"Invalid $ref format: {ref}")
            definition_key = ref[len('#/definitions/'):]
            if definition_key not in self.generate_json_schema.definitions:
                raise LookupError(f"Definition '{definition_key}' not found")
            return self.generate_json_schema.definitions[definition_key]
        return maybe_ref_json_schema

class CallbackGetCoreSchemaHandler(GetCoreSchemaHandler):
    """Wrapper to use an arbitrary function as a `GetCoreSchemaHandler`.

    Used internally by Pydantic, please do not rely on this implementation.
    See `GetCoreSchemaHandler` for the handler API.
    """

    def __init__(self, handler: Callable[[Any], core_schema.CoreSchema], generate_schema: GenerateSchema, ref_mode: Literal['to-def', 'unpack']='to-def') -> None:
        self._handler = handler
        self._generate_schema = generate_schema
        self._ref_mode = ref_mode

    def __call__(self, source_type: Any, /) -> core_schema.CoreSchema:
        schema = self._handler(source_type)
        ref = schema.get('ref')
        if self._ref_mode == 'to-def':
            if ref is not None:
                self._generate_schema.defs.definitions[ref] = schema
                return core_schema.definition_reference_schema(ref)
            return schema
        else:
            return self.resolve_ref_schema(schema)

    def resolve_ref_schema(self, maybe_ref_schema: core_schema.CoreSchema) -> core_schema.CoreSchema:
        """Resolves reference in the core schema.

        Args:
            maybe_ref_schema: The input core schema that may contains reference.

        Returns:
            Resolved core schema.

        Raises:
            LookupError: If it can't find the definition for reference.
        """
        if isinstance(maybe_ref_schema, dict) and 'type' in maybe_ref_schema and maybe_ref_schema['type'] == 'definition-ref':
            ref = maybe_ref_schema.get('schema_ref')
            if ref is None:
                raise ValueError("Invalid definition-ref schema: missing 'schema_ref'")
            if ref not in self._generate_schema.defs.definitions:
                raise LookupError(f"Definition '{ref}' not found")
            return self._generate_schema.defs.definitions[ref]
        return maybe_ref_schema
