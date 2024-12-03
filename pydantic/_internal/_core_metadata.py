from __future__ import annotations as _annotations
import typing
from typing import Any
import typing_extensions
if typing.TYPE_CHECKING:
    from ._schema_generation_shared import CoreSchemaOrField as CoreSchemaOrField
    from ._schema_generation_shared import GetJsonSchemaFunction

class CoreMetadata(typing_extensions.TypedDict, total=False):
    """A `TypedDict` for holding the metadata dict of the schema.

    Attributes:
        pydantic_js_functions: List of JSON schema functions.
        pydantic_js_prefer_positional_arguments: Whether JSON schema generator will
            prefer positional over keyword arguments for an 'arguments' schema.
    """
    pydantic_js_functions: list[GetJsonSchemaFunction]
    pydantic_js_annotation_functions: list[GetJsonSchemaFunction]
    pydantic_js_prefer_positional_arguments: bool | None
    pydantic_typed_dict_cls: type[Any] | None

class CoreMetadataHandler:
    """Because the metadata field in pydantic_core is of type `Any`, we can't assume much about its contents.

    This class is used to interact with the metadata field on a CoreSchema object in a consistent
    way throughout pydantic.
    """
    __slots__ = ('_schema',)

    def __init__(self, schema: CoreSchemaOrField):
        self._schema = schema
        metadata = schema.get('metadata')
        if metadata is None:
            schema['metadata'] = CoreMetadata()
        elif not isinstance(metadata, dict):
            raise TypeError(f'CoreSchema metadata should be a dict; got {metadata!r}.')

    @property
    def metadata(self) -> CoreMetadata:
        """Retrieves the metadata dict from the schema, initializing it to a dict if it is None
        and raises an error if it is not a dict.
        """
        metadata = self._schema.get('metadata')
        if metadata is None:
            metadata = CoreMetadata()
            self._schema['metadata'] = metadata
        elif not isinstance(metadata, dict):
            raise TypeError(f'CoreSchema metadata should be a dict; got {metadata!r}.')
        return metadata

def build_metadata_dict(*, js_functions: list[GetJsonSchemaFunction] | None=None, js_annotation_functions: list[GetJsonSchemaFunction] | None=None, js_prefer_positional_arguments: bool | None=None, typed_dict_cls: type[Any] | None=None, initial_metadata: Any | None=None) -> Any:
    """Builds a dict to use as the metadata field of a CoreSchema object in a manner that is consistent
    with the CoreMetadataHandler class.
    """
    metadata = CoreMetadata() if initial_metadata is None else dict(initial_metadata)
    
    if js_functions is not None:
        metadata['pydantic_js_functions'] = js_functions
    
    if js_annotation_functions is not None:
        metadata['pydantic_js_annotation_functions'] = js_annotation_functions
    
    if js_prefer_positional_arguments is not None:
        metadata['pydantic_js_prefer_positional_arguments'] = js_prefer_positional_arguments
    
    if typed_dict_cls is not None:
        metadata['pydantic_typed_dict_cls'] = typed_dict_cls
    
    return metadata
