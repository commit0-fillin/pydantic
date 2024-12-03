from __future__ import annotations
from collections import defaultdict
from copy import copy
from functools import lru_cache, partial
from typing import TYPE_CHECKING, Any, Callable, Iterable
from pydantic_core import CoreSchema, PydanticCustomError, to_jsonable_python
from pydantic_core import core_schema as cs
from ._fields import PydanticMetadata
if TYPE_CHECKING:
    from ..annotated_handlers import GetJsonSchemaHandler
STRICT = {'strict'}
FAIL_FAST = {'fail_fast'}
LENGTH_CONSTRAINTS = {'min_length', 'max_length'}
INEQUALITY = {'le', 'ge', 'lt', 'gt'}
NUMERIC_CONSTRAINTS = {'multiple_of', *INEQUALITY}
ALLOW_INF_NAN = {'allow_inf_nan'}
STR_CONSTRAINTS = {*LENGTH_CONSTRAINTS, *STRICT, 'strip_whitespace', 'to_lower', 'to_upper', 'pattern', 'coerce_numbers_to_str'}
BYTES_CONSTRAINTS = {*LENGTH_CONSTRAINTS, *STRICT}
LIST_CONSTRAINTS = {*LENGTH_CONSTRAINTS, *STRICT, *FAIL_FAST}
TUPLE_CONSTRAINTS = {*LENGTH_CONSTRAINTS, *STRICT, *FAIL_FAST}
SET_CONSTRAINTS = {*LENGTH_CONSTRAINTS, *STRICT, *FAIL_FAST}
DICT_CONSTRAINTS = {*LENGTH_CONSTRAINTS, *STRICT}
GENERATOR_CONSTRAINTS = {*LENGTH_CONSTRAINTS, *STRICT}
SEQUENCE_CONSTRAINTS = {*LENGTH_CONSTRAINTS, *FAIL_FAST}
FLOAT_CONSTRAINTS = {*NUMERIC_CONSTRAINTS, *ALLOW_INF_NAN, *STRICT}
DECIMAL_CONSTRAINTS = {'max_digits', 'decimal_places', *FLOAT_CONSTRAINTS}
INT_CONSTRAINTS = {*NUMERIC_CONSTRAINTS, *ALLOW_INF_NAN, *STRICT}
BOOL_CONSTRAINTS = STRICT
UUID_CONSTRAINTS = STRICT
DATE_TIME_CONSTRAINTS = {*NUMERIC_CONSTRAINTS, *STRICT}
TIMEDELTA_CONSTRAINTS = {*NUMERIC_CONSTRAINTS, *STRICT}
TIME_CONSTRAINTS = {*NUMERIC_CONSTRAINTS, *STRICT}
LAX_OR_STRICT_CONSTRAINTS = STRICT
ENUM_CONSTRAINTS = STRICT
UNION_CONSTRAINTS = {'union_mode'}
URL_CONSTRAINTS = {'max_length', 'allowed_schemes', 'host_required', 'default_host', 'default_port', 'default_path'}
TEXT_SCHEMA_TYPES = ('str', 'bytes', 'url', 'multi-host-url')
SEQUENCE_SCHEMA_TYPES = ('list', 'tuple', 'set', 'frozenset', 'generator', *TEXT_SCHEMA_TYPES)
NUMERIC_SCHEMA_TYPES = ('float', 'int', 'date', 'time', 'timedelta', 'datetime')
CONSTRAINTS_TO_ALLOWED_SCHEMAS: dict[str, set[str]] = defaultdict(set)
constraint_schema_pairings: list[tuple[set[str], tuple[str, ...]]] = [(STR_CONSTRAINTS, TEXT_SCHEMA_TYPES), (BYTES_CONSTRAINTS, ('bytes',)), (LIST_CONSTRAINTS, ('list',)), (TUPLE_CONSTRAINTS, ('tuple',)), (SET_CONSTRAINTS, ('set', 'frozenset')), (DICT_CONSTRAINTS, ('dict',)), (GENERATOR_CONSTRAINTS, ('generator',)), (FLOAT_CONSTRAINTS, ('float',)), (INT_CONSTRAINTS, ('int',)), (DATE_TIME_CONSTRAINTS, ('date', 'time', 'datetime')), (TIMEDELTA_CONSTRAINTS, ('timedelta',)), (TIME_CONSTRAINTS, ('time',)), (STRICT, (*TEXT_SCHEMA_TYPES, *SEQUENCE_SCHEMA_TYPES, *NUMERIC_SCHEMA_TYPES, 'typed-dict', 'model')), (UNION_CONSTRAINTS, ('union',)), (URL_CONSTRAINTS, ('url', 'multi-host-url')), (BOOL_CONSTRAINTS, ('bool',)), (UUID_CONSTRAINTS, ('uuid',)), (LAX_OR_STRICT_CONSTRAINTS, ('lax-or-strict',)), (ENUM_CONSTRAINTS, ('enum',)), (DECIMAL_CONSTRAINTS, ('decimal',))]
for constraints, schemas in constraint_schema_pairings:
    for c in constraints:
        CONSTRAINTS_TO_ALLOWED_SCHEMAS[c].update(schemas)

def expand_grouped_metadata(annotations: Iterable[Any]) -> Iterable[Any]:
    """Expand the annotations.

    Args:
        annotations: An iterable of annotations.

    Returns:
        An iterable of expanded annotations.

    Example:
        ```py
        from annotated_types import Ge, Len

        from pydantic._internal._known_annotated_metadata import expand_grouped_metadata

        print(list(expand_grouped_metadata([Ge(4), Len(5)])))
        #> [Ge(ge=4), MinLen(min_length=5)]
        ```
    """
    from annotated_types import BaseMetadata, GroupedMetadata

    for annotation in annotations:
        if isinstance(annotation, GroupedMetadata):
            yield from expand_grouped_metadata(annotation.metadata)
        elif isinstance(annotation, BaseMetadata):
            yield annotation
        else:
            yield annotation

@lru_cache
def _get_at_to_constraint_map() -> dict[type, str]:
    """Return a mapping of annotated types to constraints.

    Normally, we would define a mapping like this in the module scope, but we can't do that
    because we don't permit module level imports of `annotated_types`, in an attempt to speed up
    the import time of `pydantic`. We still only want to have this dictionary defined in one place,
    so we use this function to cache the result.
    """
    import annotated_types as at
    return {
        at.Gt: 'gt',
        at.Ge: 'ge',
        at.Lt: 'lt',
        at.Le: 'le',
        at.MultipleOf: 'multiple_of',
        at.MinLen: 'min_length',
        at.MaxLen: 'max_length',
        at.Predicate: 'predicate',
    }

def apply_known_metadata(annotation: Any, schema: CoreSchema) -> CoreSchema | None:
    """Apply `annotation` to `schema` if it is an annotation we know about (Gt, Le, etc.).
    Otherwise return `None`.

    This does not handle all known annotations. If / when it does, it can always
    return a CoreSchema and return the unmodified schema if the annotation should be ignored.

    Assumes that GroupedMetadata has already been expanded via `expand_grouped_metadata`.

    Args:
        annotation: The annotation.
        schema: The schema.

    Returns:
        An updated schema with annotation if it is an annotation we know about, `None` otherwise.

    Raises:
        PydanticCustomError: If `Predicate` fails.
    """
    from annotated_types import BaseMetadata, Predicate

    at_to_constraint = _get_at_to_constraint_map()
    constraint = at_to_constraint.get(type(annotation))

    if constraint is None:
        return None

    if isinstance(annotation, Predicate):
        def predicate_validator(v: Any) -> Any:
            if not annotation.func(v):
                raise PydanticCustomError('predicate_failed', 'Predicate failed')
            return v
        return core_schema.no_info_wrap_validator_function(
            predicate_validator,
            schema,
            schema_type=schema['type']
        )

    new_schema = schema.copy()
    new_schema[constraint] = getattr(annotation, constraint)
    return new_schema

def collect_known_metadata(annotations: Iterable[Any]) -> tuple[dict[str, Any], list[Any]]:
    """Split `annotations` into known metadata and unknown annotations.

    Args:
        annotations: An iterable of annotations.

    Returns:
        A tuple contains a dict of known metadata and a list of unknown annotations.

    Example:
        ```py
        from annotated_types import Gt, Len

        from pydantic._internal._known_annotated_metadata import collect_known_metadata

        print(collect_known_metadata([Gt(1), Len(42), ...]))
        #> ({'gt': 1, 'min_length': 42}, [Ellipsis])
        ```
    """
    known_metadata = {}
    unknown_annotations = []
    at_to_constraint = _get_at_to_constraint_map()

    for annotation in expand_grouped_metadata(annotations):
        constraint = at_to_constraint.get(type(annotation))
        if constraint is not None:
            known_metadata[constraint] = getattr(annotation, constraint)
        else:
            unknown_annotations.append(annotation)

    return known_metadata, unknown_annotations

def check_metadata(metadata: dict[str, Any], allowed: Iterable[str], source_type: Any) -> None:
    """A small utility function to validate that the given metadata can be applied to the target.
    More than saving lines of code, this gives us a consistent error message for all of our internal implementations.

    Args:
        metadata: A dict of metadata.
        allowed: An iterable of allowed metadata.
        source_type: The source type.

    Raises:
        TypeError: If there is metadatas that can't be applied on source type.
    """
    allowed_set = set(allowed)
    invalid = set(metadata) - allowed_set
    if invalid:
        raise TypeError(
            f"The following constraints cannot be applied to {source_type}: {', '.join(invalid)}. "
            f"Allowed constraints are: {', '.join(allowed_set)}"
        )
