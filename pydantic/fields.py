"""Defining fields on models."""
from __future__ import annotations as _annotations
import dataclasses
import inspect
import sys
import typing
from copy import copy
from dataclasses import Field as DataclassField
from functools import cached_property
from typing import Any, ClassVar
from warnings import warn
import annotated_types
import typing_extensions
from pydantic_core import PydanticUndefined
from typing_extensions import Literal, TypeAlias, Unpack, deprecated
from . import types
from ._internal import _decorators, _fields, _generics, _internal_dataclass, _repr, _typing_extra, _utils
from .aliases import AliasChoices, AliasPath
from .config import JsonDict
from .errors import PydanticUserError
from .warnings import PydanticDeprecatedSince20
if typing.TYPE_CHECKING:
    from ._internal._repr import ReprArgs
else:
    DeprecationWarning = PydanticDeprecatedSince20
__all__ = ('Field', 'PrivateAttr', 'computed_field')
_Unset: Any = PydanticUndefined
if sys.version_info >= (3, 13):
    import warnings
    Deprecated: TypeAlias = warnings.deprecated | deprecated
else:
    Deprecated: TypeAlias = deprecated

class _FromFieldInfoInputs(typing_extensions.TypedDict, total=False):
    """This class exists solely to add type checking for the `**kwargs` in `FieldInfo.from_field`."""
    annotation: type[Any] | None
    default_factory: typing.Callable[[], Any] | None
    alias: str | None
    alias_priority: int | None
    validation_alias: str | AliasPath | AliasChoices | None
    serialization_alias: str | None
    title: str | None
    field_title_generator: typing_extensions.Callable[[str, FieldInfo], str] | None
    description: str | None
    examples: list[Any] | None
    exclude: bool | None
    gt: annotated_types.SupportsGt | None
    ge: annotated_types.SupportsGe | None
    lt: annotated_types.SupportsLt | None
    le: annotated_types.SupportsLe | None
    multiple_of: float | None
    strict: bool | None
    min_length: int | None
    max_length: int | None
    pattern: str | typing.Pattern[str] | None
    allow_inf_nan: bool | None
    max_digits: int | None
    decimal_places: int | None
    union_mode: Literal['smart', 'left_to_right'] | None
    discriminator: str | types.Discriminator | None
    deprecated: Deprecated | str | bool | None
    json_schema_extra: JsonDict | typing.Callable[[JsonDict], None] | None
    frozen: bool | None
    validate_default: bool | None
    repr: bool
    init: bool | None
    init_var: bool | None
    kw_only: bool | None
    coerce_numbers_to_str: bool | None
    fail_fast: bool | None

class _FieldInfoInputs(_FromFieldInfoInputs, total=False):
    """This class exists solely to add type checking for the `**kwargs` in `FieldInfo.__init__`."""
    default: Any

class FieldInfo(_repr.Representation):
    """This class holds information about a field.

    `FieldInfo` is used for any field definition regardless of whether the [`Field()`][pydantic.fields.Field]
    function is explicitly used.

    !!! warning
        You generally shouldn't be creating `FieldInfo` directly, you'll only need to use it when accessing
        [`BaseModel`][pydantic.main.BaseModel] `.model_fields` internals.

    Attributes:
        annotation: The type annotation of the field.
        default: The default value of the field.
        default_factory: The factory function used to construct the default for the field.
        alias: The alias name of the field.
        alias_priority: The priority of the field's alias.
        validation_alias: The validation alias of the field.
        serialization_alias: The serialization alias of the field.
        title: The title of the field.
        field_title_generator: A callable that takes a field name and returns title for it.
        description: The description of the field.
        examples: List of examples of the field.
        exclude: Whether to exclude the field from the model serialization.
        discriminator: Field name or Discriminator for discriminating the type in a tagged union.
        deprecated: A deprecation message, an instance of `warnings.deprecated` or the `typing_extensions.deprecated` backport,
            or a boolean. If `True`, a default deprecation message will be emitted when accessing the field.
        json_schema_extra: A dict or callable to provide extra JSON schema properties.
        frozen: Whether the field is frozen.
        validate_default: Whether to validate the default value of the field.
        repr: Whether to include the field in representation of the model.
        init: Whether the field should be included in the constructor of the dataclass.
        init_var: Whether the field should _only_ be included in the constructor of the dataclass, and not stored.
        kw_only: Whether the field should be a keyword-only argument in the constructor of the dataclass.
        metadata: List of metadata constraints.
    """
    annotation: type[Any] | None
    default: Any
    default_factory: typing.Callable[[], Any] | None
    alias: str | None
    alias_priority: int | None
    validation_alias: str | AliasPath | AliasChoices | None
    serialization_alias: str | None
    title: str | None
    field_title_generator: typing.Callable[[str, FieldInfo], str] | None
    description: str | None
    examples: list[Any] | None
    exclude: bool | None
    discriminator: str | types.Discriminator | None
    deprecated: Deprecated | str | bool | None
    json_schema_extra: JsonDict | typing.Callable[[JsonDict], None] | None
    frozen: bool | None
    validate_default: bool | None
    repr: bool
    init: bool | None
    init_var: bool | None
    kw_only: bool | None
    metadata: list[Any]
    __slots__ = ('annotation', 'default', 'default_factory', 'alias', 'alias_priority', 'validation_alias', 'serialization_alias', 'title', 'field_title_generator', 'description', 'examples', 'exclude', 'discriminator', 'deprecated', 'json_schema_extra', 'frozen', 'validate_default', 'repr', 'init', 'init_var', 'kw_only', 'metadata', '_attributes_set')
    metadata_lookup: ClassVar[dict[str, typing.Callable[[Any], Any] | None]] = {'strict': types.Strict, 'gt': annotated_types.Gt, 'ge': annotated_types.Ge, 'lt': annotated_types.Lt, 'le': annotated_types.Le, 'multiple_of': annotated_types.MultipleOf, 'min_length': annotated_types.MinLen, 'max_length': annotated_types.MaxLen, 'pattern': None, 'allow_inf_nan': None, 'max_digits': None, 'decimal_places': None, 'union_mode': None, 'coerce_numbers_to_str': None, 'fail_fast': types.FailFast}

    def __init__(self, **kwargs: Unpack[_FieldInfoInputs]) -> None:
        """This class should generally not be initialized directly; instead, use the `pydantic.fields.Field` function
        or one of the constructor classmethods.

        See the signature of `pydantic.fields.Field` for more details about the expected arguments.
        """
        self._attributes_set = {k: v for k, v in kwargs.items() if v is not _Unset}
        kwargs = {k: _DefaultValues.get(k) if v is _Unset else v for k, v in kwargs.items()}
        self.annotation, annotation_metadata = self._extract_metadata(kwargs.get('annotation'))
        default = kwargs.pop('default', PydanticUndefined)
        if default is Ellipsis:
            self.default = PydanticUndefined
        else:
            self.default = default
        self.default_factory = kwargs.pop('default_factory', None)
        if self.default is not PydanticUndefined and self.default_factory is not None:
            raise TypeError('cannot specify both default and default_factory')
        self.alias = kwargs.pop('alias', None)
        self.validation_alias = kwargs.pop('validation_alias', None)
        self.serialization_alias = kwargs.pop('serialization_alias', None)
        alias_is_set = any((alias is not None for alias in (self.alias, self.validation_alias, self.serialization_alias)))
        self.alias_priority = kwargs.pop('alias_priority', None) or 2 if alias_is_set else None
        self.title = kwargs.pop('title', None)
        self.field_title_generator = kwargs.pop('field_title_generator', None)
        self.description = kwargs.pop('description', None)
        self.examples = kwargs.pop('examples', None)
        self.exclude = kwargs.pop('exclude', None)
        self.discriminator = kwargs.pop('discriminator', None)
        self.deprecated = kwargs.pop('deprecated', getattr(self, 'deprecated', None))
        self.repr = kwargs.pop('repr', True)
        self.json_schema_extra = kwargs.pop('json_schema_extra', None)
        self.validate_default = kwargs.pop('validate_default', None)
        self.frozen = kwargs.pop('frozen', None)
        self.init = kwargs.pop('init', None)
        self.init_var = kwargs.pop('init_var', None)
        self.kw_only = kwargs.pop('kw_only', None)
        self.metadata = self._collect_metadata(kwargs) + annotation_metadata

    @staticmethod
    def from_field(default: Any=PydanticUndefined, **kwargs: Unpack[_FromFieldInfoInputs]) -> FieldInfo:
        """Create a new `FieldInfo` object with the `Field` function.

        Args:
            default: The default value for the field. Defaults to Undefined.
            **kwargs: Additional arguments dictionary.

        Raises:
            TypeError: If 'annotation' is passed as a keyword argument.

        Returns:
            A new FieldInfo object with the given parameters.

        Example:
            This is how you can create a field with default value like this:

            ```python
            import pydantic

            class MyModel(pydantic.BaseModel):
                foo: int = pydantic.Field(4)
            ```
        """
        if 'annotation' in kwargs:
            raise TypeError("'annotation' is not permitted as a Field keyword argument")
        return FieldInfo(default=default, **kwargs)

    @staticmethod
    def from_annotation(annotation: type[Any]) -> FieldInfo:
        """Creates a `FieldInfo` instance from a bare annotation.

        This function is used internally to create a `FieldInfo` from a bare annotation like this:

        ```python
        import pydantic

        class MyModel(pydantic.BaseModel):
            foo: int  # <-- like this
        ```

        We also account for the case where the annotation can be an instance of `Annotated` and where
        one of the (not first) arguments in `Annotated` is an instance of `FieldInfo`, e.g.:

        ```python
        import annotated_types
        from typing_extensions import Annotated

        import pydantic

        class MyModel(pydantic.BaseModel):
            foo: Annotated[int, annotated_types.Gt(42)]
            bar: Annotated[int, pydantic.Field(gt=42)]
        ```

        Args:
            annotation: An annotation object.

        Returns:
            An instance of the field metadata.
        """
        if _typing_extra.is_annotated(annotation):
            first_arg, *extra_args = _typing_extra.get_args(annotation)
            for arg in reversed(extra_args):
                if isinstance(arg, FieldInfo):
                    return arg
            metadata = [arg for arg in extra_args if not isinstance(arg, FieldInfo)]
            return FieldInfo(annotation=first_arg, metadata=metadata)
        return FieldInfo(annotation=annotation)

    @staticmethod
    def from_annotated_attribute(annotation: type[Any], default: Any) -> FieldInfo:
        """Create `FieldInfo` from an annotation with a default value.

        This is used in cases like the following:

        ```python
        import annotated_types
        from typing_extensions import Annotated

        import pydantic

        class MyModel(pydantic.BaseModel):
            foo: int = 4  # <-- like this
            bar: Annotated[int, annotated_types.Gt(4)] = 4  # <-- or this
            spam: Annotated[int, pydantic.Field(gt=4)] = 4  # <-- or this
        ```

        Args:
            annotation: The type annotation of the field.
            default: The default value of the field.

        Returns:
            A field object with the passed values.
        """
        if _typing_extra.is_annotated(annotation):
            first_arg, *extra_args = _typing_extra.get_args(annotation)
            for arg in reversed(extra_args):
                if isinstance(arg, FieldInfo):
                    return FieldInfo.merge_field_infos(arg, FieldInfo(default=default))
            metadata = [arg for arg in extra_args if not isinstance(arg, FieldInfo)]
            return FieldInfo(annotation=first_arg, default=default, metadata=metadata)
        return FieldInfo(annotation=annotation, default=default)

    @staticmethod
    def merge_field_infos(*field_infos: FieldInfo, **overrides: Any) -> FieldInfo:
        """Merge `FieldInfo` instances keeping only explicitly set attributes.

        Later `FieldInfo` instances override earlier ones.

        Returns:
            FieldInfo: A merged FieldInfo instance.
        """
        merged_dict = {}
        for field_info in field_infos:
            merged_dict.update(field_info._attributes_set)
        merged_dict.update(overrides)
        return FieldInfo(**merged_dict)

    @staticmethod
    def _from_dataclass_field(dc_field: DataclassField[Any]) -> FieldInfo:
        """Return a new `FieldInfo` instance from a `dataclasses.Field` instance.

        Args:
            dc_field: The `dataclasses.Field` instance to convert.

        Returns:
            The corresponding `FieldInfo` instance.

        Raises:
            TypeError: If any of the `FieldInfo` kwargs does not match the `dataclass.Field` kwargs.
        """
        field_info_kwargs = {}
        for key, value in dc_field.__dict__.items():
            if key in _FIELD_ARG_NAMES:
                field_info_kwargs[key] = value
            elif key == 'metadata':
                field_info_kwargs['json_schema_extra'] = value
        
        if dc_field.default is not dataclasses.MISSING:
            field_info_kwargs['default'] = dc_field.default
        elif dc_field.default_factory is not dataclasses.MISSING:
            field_info_kwargs['default_factory'] = dc_field.default_factory
        
        try:
            return FieldInfo(**field_info_kwargs)
        except TypeError as e:
            raise TypeError(f"Error converting dataclass Field to FieldInfo: {e}")

    @staticmethod
    def _extract_metadata(annotation: type[Any] | None) -> tuple[type[Any] | None, list[Any]]:
        """Tries to extract metadata/constraints from an annotation if it uses `Annotated`.

        Args:
            annotation: The type hint annotation for which metadata has to be extracted.

        Returns:
            A tuple containing the extracted metadata type and the list of extra arguments.
        """
        if annotation is None or not _typing_extra.is_annotated(annotation):
            return annotation, []
        
        first_arg, *extra_args = _typing_extra.get_args(annotation)
        metadata = [arg for arg in extra_args if not isinstance(arg, FieldInfo)]
        return first_arg, metadata

    @staticmethod
    def _collect_metadata(kwargs: dict[str, Any]) -> list[Any]:
        """Collect annotations from kwargs.

        Args:
            kwargs: Keyword arguments passed to the function.

        Returns:
            A list of metadata objects - a combination of `annotated_types.BaseMetadata` and
                `PydanticMetadata`.
        """
        metadata = []
        for key, value in kwargs.items():
            if key in FieldInfo.metadata_lookup and value is not None:
                if FieldInfo.metadata_lookup[key] is None:
                    metadata.append(value)
                else:
                    metadata.append(FieldInfo.metadata_lookup[key](value))
        return metadata

    @property
    def deprecation_message(self) -> str | None:
        """The deprecation message to be emitted, or `None` if not set."""
        if isinstance(self.deprecated, str):
            return self.deprecated
        elif isinstance(self.deprecated, (Deprecated, bool)):
            return "This field is deprecated" if self.deprecated else None
        return None

    def get_default(self, *, call_default_factory: bool=False) -> Any:
        """Get the default value.

        We expose an option for whether to call the default_factory (if present), as calling it may
        result in side effects that we want to avoid. However, there are times when it really should
        be called (namely, when instantiating a model via `model_construct`).

        Args:
            call_default_factory: Whether to call the default_factory or not. Defaults to `False`.

        Returns:
            The default value, calling the default factory if requested or `None` if not set.
        """
        if self.default_factory is not None and call_default_factory:
            return self.default_factory()
        return self.default

    def is_required(self) -> bool:
        """Check if the field is required (i.e., does not have a default value or factory).

        Returns:
            `True` if the field is required, `False` otherwise.
        """
        return self.default is PydanticUndefined and self.default_factory is None

    def rebuild_annotation(self) -> Any:
        """Attempts to rebuild the original annotation for use in function signatures.

        If metadata is present, it adds it to the original annotation using
        `Annotated`. Otherwise, it returns the original annotation as-is.

        Note that because the metadata has been flattened, the original annotation
        may not be reconstructed exactly as originally provided, e.g. if the original
        type had unrecognized annotations, or was annotated with a call to `pydantic.Field`.

        Returns:
            The rebuilt annotation.
        """
        if not self.metadata:
            return self.annotation
        return _typing_extra.Annotated[self.annotation, *self.metadata]

    def apply_typevars_map(self, typevars_map: dict[Any, Any] | None, types_namespace: dict[str, Any] | None) -> None:
        """Apply a `typevars_map` to the annotation.

        This method is used when analyzing parametrized generic types to replace typevars with their concrete types.

        This method applies the `typevars_map` to the annotation in place.

        Args:
            typevars_map: A dictionary mapping type variables to their concrete types.
            types_namespace (dict | None): A dictionary containing related types to the annotated type.

        See Also:
            pydantic._internal._generics.replace_types is used for replacing the typevars with
                their concrete types.
        """
        if typevars_map:
            self.annotation = _generics.replace_types(self.annotation, typevars_map, types_namespace)

    def __repr_args__(self) -> ReprArgs:
        yield ('annotation', _repr.PlainRepr(_repr.display_as_type(self.annotation)))
        yield ('required', self.is_required())
        for s in self.__slots__:
            if s == '_attributes_set':
                continue
            if s == 'annotation':
                continue
            elif s == 'metadata' and (not self.metadata):
                continue
            elif s == 'repr' and self.repr is True:
                continue
            if s == 'frozen' and self.frozen is False:
                continue
            if s == 'validation_alias' and self.validation_alias == self.alias:
                continue
            if s == 'serialization_alias' and self.serialization_alias == self.alias:
                continue
            if s == 'default' and self.default is not PydanticUndefined:
                yield ('default', self.default)
            elif s == 'default_factory' and self.default_factory is not None:
                yield ('default_factory', _repr.PlainRepr(_repr.display_as_type(self.default_factory)))
            else:
                value = getattr(self, s)
                if value is not None and value is not PydanticUndefined:
                    yield (s, value)

class _EmptyKwargs(typing_extensions.TypedDict):
    """This class exists solely to ensure that type checking warns about passing `**extra` in `Field`."""
_DefaultValues = dict(default=..., default_factory=None, alias=None, alias_priority=None, validation_alias=None, serialization_alias=None, title=None, description=None, examples=None, exclude=None, discriminator=None, json_schema_extra=None, frozen=None, validate_default=None, repr=True, init=None, init_var=None, kw_only=None, pattern=None, strict=None, gt=None, ge=None, lt=None, le=None, multiple_of=None, allow_inf_nan=None, max_digits=None, decimal_places=None, min_length=None, max_length=None, coerce_numbers_to_str=None)

def Field(default: Any=PydanticUndefined, *, default_factory: typing.Callable[[], Any] | None=_Unset, alias: str | None=_Unset, alias_priority: int | None=_Unset, validation_alias: str | AliasPath | AliasChoices | None=_Unset, serialization_alias: str | None=_Unset, title: str | None=_Unset, field_title_generator: typing_extensions.Callable[[str, FieldInfo], str] | None=_Unset, description: str | None=_Unset, examples: list[Any] | None=_Unset, exclude: bool | None=_Unset, discriminator: str | types.Discriminator | None=_Unset, deprecated: Deprecated | str | bool | None=_Unset, json_schema_extra: JsonDict | typing.Callable[[JsonDict], None] | None=_Unset, frozen: bool | None=_Unset, validate_default: bool | None=_Unset, repr: bool=_Unset, init: bool | None=_Unset, init_var: bool | None=_Unset, kw_only: bool | None=_Unset, pattern: str | typing.Pattern[str] | None=_Unset, strict: bool | None=_Unset, coerce_numbers_to_str: bool | None=_Unset, gt: annotated_types.SupportsGt | None=_Unset, ge: annotated_types.SupportsGe | None=_Unset, lt: annotated_types.SupportsLt | None=_Unset, le: annotated_types.SupportsLe | None=_Unset, multiple_of: float | None=_Unset, allow_inf_nan: bool | None=_Unset, max_digits: int | None=_Unset, decimal_places: int | None=_Unset, min_length: int | None=_Unset, max_length: int | None=_Unset, union_mode: Literal['smart', 'left_to_right']=_Unset, fail_fast: bool | None=_Unset, **extra: Unpack[_EmptyKwargs]) -> Any:
    """Usage docs: https://docs.pydantic.dev/2.8/concepts/fields

    Create a field for objects that can be configured.

    Used to provide extra information about a field, either for the model schema or complex validation. Some arguments
    apply only to number fields (`int`, `float`, `Decimal`) and some apply only to `str`.

    Note:
        - Any `_Unset` objects will be replaced by the corresponding value defined in the `_DefaultValues` dictionary. If a key for the `_Unset` object is not found in the `_DefaultValues` dictionary, it will default to `None`

    Args:
        default: Default value if the field is not set.
        default_factory: A callable to generate the default value, such as :func:`~datetime.utcnow`.
        alias: The name to use for the attribute when validating or serializing by alias.
            This is often used for things like converting between snake and camel case.
        alias_priority: Priority of the alias. This affects whether an alias generator is used.
        validation_alias: Like `alias`, but only affects validation, not serialization.
        serialization_alias: Like `alias`, but only affects serialization, not validation.
        title: Human-readable title.
        field_title_generator: A callable that takes a field name and returns title for it.
        description: Human-readable description.
        examples: Example values for this field.
        exclude: Whether to exclude the field from the model serialization.
        discriminator: Field name or Discriminator for discriminating the type in a tagged union.
        deprecated: A deprecation message, an instance of `warnings.deprecated` or the `typing_extensions.deprecated` backport,
            or a boolean. If `True`, a default deprecation message will be emitted when accessing the field.
        json_schema_extra: A dict or callable to provide extra JSON schema properties.
        frozen: Whether the field is frozen. If true, attempts to change the value on an instance will raise an error.
        validate_default: If `True`, apply validation to the default value every time you create an instance.
            Otherwise, for performance reasons,the default value of the field is trusted and not validated.
        repr: A boolean indicating whether to include the field in the `__repr__` output.
        init: Whether the field should be included in the constructor of the dataclass.
            (Only applies to dataclasses.)
        init_var: Whether the field should _only_ be included in the constructor of the dataclass.
            (Only applies to dataclasses.)
        kw_only: Whether the field should be a keyword-only argument in the constructor of the dataclass.
            (Only applies to dataclasses.)
        coerce_numbers_to_str: Whether to enable coercion of any `Number` type to `str` (not applicable in `strict` mode).
        strict: If `True`, strict validation is applied to the field.
            See [Strict Mode](../concepts/strict_mode.md) for details.
        gt: Greater than. If set, value must be greater than this. Only applicable to numbers.
        ge: Greater than or equal. If set, value must be greater than or equal to this. Only applicable to numbers.
        lt: Less than. If set, value must be less than this. Only applicable to numbers.
        le: Less than or equal. If set, value must be less than or equal to this. Only applicable to numbers.
        multiple_of: Value must be a multiple of this. Only applicable to numbers.
        min_length: Minimum length for iterables.
        max_length: Maximum length for iterables.
        pattern: Pattern for strings (a regular expression).
        allow_inf_nan: Allow `inf`, `-inf`, `nan`. Only applicable to numbers.
        max_digits: Maximum number of allow digits for strings.
        decimal_places: Maximum number of decimal places allowed for numbers.
        union_mode: The strategy to apply when validating a union. Can be `smart` (the default), or `left_to_right`.
            See [Union Mode](../concepts/unions.md#union-modes) for details.
        fail_fast: If `True`, validation will stop on the first error. If `False`, all validation errors will be collected.
            This option can be applied only to iterable types (list, tuple, set, and frozenset).
        extra: (Deprecated) Extra fields that will be included in the JSON schema.

            !!! warning Deprecated
                The `extra` kwargs is deprecated. Use `json_schema_extra` instead.

    Returns:
        A new [`FieldInfo`][pydantic.fields.FieldInfo]. The return annotation is `Any` so `Field` can be used on
            type-annotated fields without causing a type error.
    """
    field_info_kwargs = {
        'default': default,
        'default_factory': default_factory,
        'alias': alias,
        'alias_priority': alias_priority,
        'validation_alias': validation_alias,
        'serialization_alias': serialization_alias,
        'title': title,
        'field_title_generator': field_title_generator,
        'description': description,
        'examples': examples,
        'exclude': exclude,
        'discriminator': discriminator,
        'deprecated': deprecated,
        'json_schema_extra': json_schema_extra,
        'frozen': frozen,
        'validate_default': validate_default,
        'repr': repr,
        'init': init,
        'init_var': init_var,
        'kw_only': kw_only,
        'pattern': pattern,
        'strict': strict,
        'coerce_numbers_to_str': coerce_numbers_to_str,
        'gt': gt,
        'ge': ge,
        'lt': lt,
        'le': le,
        'multiple_of': multiple_of,
        'allow_inf_nan': allow_inf_nan,
        'max_digits': max_digits,
        'decimal_places': decimal_places,
        'min_length': min_length,
        'max_length': max_length,
        'union_mode': union_mode,
        'fail_fast': fail_fast,
    }

    if extra:
        warn('The `extra` Field argument is deprecated, use `json_schema_extra` instead', DeprecationWarning, stacklevel=2)
        if json_schema_extra is _Unset:
            field_info_kwargs['json_schema_extra'] = extra
        elif isinstance(json_schema_extra, dict):
            field_info_kwargs['json_schema_extra'] = {**extra, **json_schema_extra}
        else:
            field_info_kwargs['json_schema_extra'] = lambda schema: (schema.update(extra), json_schema_extra(schema))

    return FieldInfo(**{k: v for k, v in field_info_kwargs.items() if v is not _Unset})
_FIELD_ARG_NAMES = set(inspect.signature(Field).parameters)
_FIELD_ARG_NAMES.remove('extra')

class ModelPrivateAttr(_repr.Representation):
    """A descriptor for private attributes in class models.

    !!! warning
        You generally shouldn't be creating `ModelPrivateAttr` instances directly, instead use
        `pydantic.fields.PrivateAttr`. (This is similar to `FieldInfo` vs. `Field`.)

    Attributes:
        default: The default value of the attribute if not provided.
        default_factory: A callable function that generates the default value of the
            attribute if not provided.
    """
    __slots__ = ('default', 'default_factory')

    def __init__(self, default: Any=PydanticUndefined, *, default_factory: typing.Callable[[], Any] | None=None) -> None:
        self.default = default
        self.default_factory = default_factory
    if not typing.TYPE_CHECKING:

        def __getattr__(self, item: str) -> Any:
            """This function improves compatibility with custom descriptors by ensuring delegation happens
            as expected when the default value of a private attribute is a descriptor.
            """
            if item in {'__get__', '__set__', '__delete__'}:
                if hasattr(self.default, item):
                    return getattr(self.default, item)
            raise AttributeError(f'{type(self).__name__!r} object has no attribute {item!r}')

    def __set_name__(self, cls: type[Any], name: str) -> None:
        """Preserve `__set_name__` protocol defined in https://peps.python.org/pep-0487."""
        if self.default is PydanticUndefined:
            return
        if not hasattr(self.default, '__set_name__'):
            return
        set_name = self.default.__set_name__
        if callable(set_name):
            set_name(cls, name)

    def get_default(self) -> Any:
        """Retrieve the default value of the object.

        If `self.default_factory` is `None`, the method will return a deep copy of the `self.default` object.

        If `self.default_factory` is not `None`, it will call `self.default_factory` and return the value returned.

        Returns:
            The default value of the object.
        """
        if self.default_factory is not None:
            return self.default_factory()
        return copy(self.default)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and (self.default, self.default_factory) == (other.default, other.default_factory)

def PrivateAttr(default: Any=PydanticUndefined, *, default_factory: typing.Callable[[], Any] | None=None, init: Literal[False]=False) -> Any:
    """Usage docs: https://docs.pydantic.dev/2.8/concepts/models/#private-model-attributes

    Indicates that an attribute is intended for private use and not handled during normal validation/serialization.

    Private attributes are not validated by Pydantic, so it's up to you to ensure they are used in a type-safe manner.

    Private attributes are stored in `__private_attributes__` on the model.

    Args:
        default: The attribute's default value. Defaults to Undefined.
        default_factory: Callable that will be
            called when a default value is needed for this attribute.
            If both `default` and `default_factory` are set, an error will be raised.
        init: Whether the attribute should be included in the constructor of the dataclass. Always `False`.

    Returns:
        An instance of [`ModelPrivateAttr`][pydantic.fields.ModelPrivateAttr] class.

    Raises:
        ValueError: If both `default` and `default_factory` are set.
    """
    if default is not PydanticUndefined and default_factory is not None:
        raise ValueError('cannot specify both default and default_factory')
    if init is not False:
        raise ValueError('init must be False for private attributes')
    return ModelPrivateAttr(default, default_factory=default_factory)

@dataclasses.dataclass(**_internal_dataclass.slots_true)
class ComputedFieldInfo:
    """A container for data from `@computed_field` so that we can access it while building the pydantic-core schema.

    Attributes:
        decorator_repr: A class variable representing the decorator string, '@computed_field'.
        wrapped_property: The wrapped computed field property.
        return_type: The type of the computed field property's return value.
        alias: The alias of the property to be used during serialization.
        alias_priority: The priority of the alias. This affects whether an alias generator is used.
        title: Title of the computed field to include in the serialization JSON schema.
        field_title_generator: A callable that takes a field name and returns title for it.
        description: Description of the computed field to include in the serialization JSON schema.
        deprecated: A deprecation message, an instance of `warnings.deprecated` or the `typing_extensions.deprecated` backport,
            or a boolean. If `True`, a default deprecation message will be emitted when accessing the field.
        examples: Example values of the computed field to include in the serialization JSON schema.
        json_schema_extra: A dict or callable to provide extra JSON schema properties.
        repr: A boolean indicating whether to include the field in the __repr__ output.
    """
    decorator_repr: ClassVar[str] = '@computed_field'
    wrapped_property: property
    return_type: Any
    alias: str | None
    alias_priority: int | None
    title: str | None
    field_title_generator: typing.Callable[[str, ComputedFieldInfo], str] | None
    description: str | None
    deprecated: Deprecated | str | bool | None
    examples: list[Any] | None
    json_schema_extra: JsonDict | typing.Callable[[JsonDict], None] | None
    repr: bool

    @property
    def deprecation_message(self) -> str | None:
        """The deprecation message to be emitted, or `None` if not set."""
        if isinstance(self.deprecated, str):
            return self.deprecated
        elif isinstance(self.deprecated, (Deprecated, bool)):
            return "This field is deprecated" if self.deprecated else None
        return None

def _wrapped_property_is_private(property_: cached_property | property) -> bool:
    """Returns true if provided property is private, False otherwise."""
    return property_.fget.__name__.startswith('_')
PropertyT = typing.TypeVar('PropertyT')

def computed_field(func: PropertyT | None=None, /, *, alias: str | None=None, alias_priority: int | None=None, title: str | None=None, field_title_generator: typing.Callable[[str, ComputedFieldInfo], str] | None=None, description: str | None=None, deprecated: Deprecated | str | bool | None=None, examples: list[Any] | None=None, json_schema_extra: JsonDict | typing.Callable[[JsonDict], None] | None=None, repr: bool | None=None, return_type: Any=PydanticUndefined) -> PropertyT | typing.Callable[[PropertyT], PropertyT]:
    """Usage docs: https://docs.pydantic.dev/2.8/concepts/fields#the-computed_field-decorator

    Decorator to include `property` and `cached_property` when serializing models or dataclasses.

    This is useful for fields that are computed from other fields, or for fields that are expensive to compute and should be cached.

    ```py
    from pydantic import BaseModel, computed_field

    class Rectangle(BaseModel):
        width: int
        length: int

        @computed_field
        @property
        def area(self) -> int:
            return self.width * self.length

    print(Rectangle(width=3, length=2).model_dump())
    #> {'width': 3, 'length': 2, 'area': 6}
    ```

    If applied to functions not yet decorated with `@property` or `@cached_property`, the function is
    automatically wrapped with `property`. Although this is more concise, you will lose IntelliSense in your IDE,
    and confuse static type checkers, thus explicit use of `@property` is recommended.

    !!! warning "Mypy Warning"
        Even with the `@property` or `@cached_property` applied to your function before `@computed_field`,
        mypy may throw a `Decorated property not supported` error.
        See [mypy issue #1362](https://github.com/python/mypy/issues/1362), for more information.
        To avoid this error message, add `# type: ignore[misc]` to the `@computed_field` line.

        [pyright](https://github.com/microsoft/pyright) supports `@computed_field` without error.

    ```py
    import random

    from pydantic import BaseModel, computed_field

    class Square(BaseModel):
        width: float

        @computed_field
        def area(self) -> float:  # converted to a `property` by `computed_field`
            return round(self.width**2, 2)

        @area.setter
        def area(self, new_area: float) -> None:
            self.width = new_area**0.5

        @computed_field(alias='the magic number', repr=False)
        def random_number(self) -> int:
            return random.randint(0, 1_000)

    square = Square(width=1.3)

    # `random_number` does not appear in representation
    print(repr(square))
    #> Square(width=1.3, area=1.69)

    print(square.random_number)
    #> 3

    square.area = 4

    print(square.model_dump_json(by_alias=True))
    #> {"width":2.0,"area":4.0,"the magic number":3}
    ```

    !!! warning "Overriding with `computed_field`"
        You can't override a field from a parent class with a `computed_field` in the child class.
        `mypy` complains about this behavior if allowed, and `dataclasses` doesn't allow this pattern either.
        See the example below:

    ```py
    from pydantic import BaseModel, computed_field

    class Parent(BaseModel):
        a: str

    try:

        class Child(Parent):
            @computed_field
            @property
            def a(self) -> str:
                return 'new a'

    except ValueError as e:
        print(repr(e))
        #> ValueError("you can't override a field with a computed field")
    ```

    Private properties decorated with `@computed_field` have `repr=False` by default.

    ```py
    from functools import cached_property

    from pydantic import BaseModel, computed_field

    class Model(BaseModel):
        foo: int

        @computed_field
        @cached_property
        def _private_cached_property(self) -> int:
            return -self.foo

        @computed_field
        @property
        def _private_property(self) -> int:
            return -self.foo

    m = Model(foo=1)
    print(repr(m))
    #> M(foo=1)
    ```

    Args:
        func: the function to wrap.
        alias: alias to use when serializing this computed field, only used when `by_alias=True`
        alias_priority: priority of the alias. This affects whether an alias generator is used
        title: Title to use when including this computed field in JSON Schema
        field_title_generator: A callable that takes a field name and returns title for it.
        description: Description to use when including this computed field in JSON Schema, defaults to the function's
            docstring
        deprecated: A deprecation message (or an instance of `warnings.deprecated` or the `typing_extensions.deprecated` backport).
            to be emitted when accessing the field. Or a boolean. This will automatically be set if the property is decorated with the
            `deprecated` decorator.
        examples: Example values to use when including this computed field in JSON Schema
        json_schema_extra: A dict or callable to provide extra JSON schema properties.
        repr: whether to include this computed field in model repr.
            Default is `False` for private properties and `True` for public properties.
        return_type: optional return for serialization logic to expect when serializing to JSON, if included
            this must be correct, otherwise a `TypeError` is raised.
            If you don't include a return type Any is used, which does runtime introspection to handle arbitrary
            objects.

    Returns:
        A proxy wrapper for the property.
    """
    def wrapper(f: PropertyT) -> PropertyT:
        if not isinstance(f, (property, cached_property)):
            f = property(f)
        
        is_private = _wrapped_property_is_private(f)
        repr_value = repr if repr is not None else (not is_private)
        
        if description is None:
            desc = f.__doc__
        else:
            desc = description
        
        computed_field_info = ComputedFieldInfo(
            wrapped_property=f,
            return_type=return_type,
            alias=alias,
            alias_priority=alias_priority,
            title=title,
            field_title_generator=field_title_generator,
            description=desc,
            deprecated=deprecated,
            examples=examples,
            json_schema_extra=json_schema_extra,
            repr=repr_value
        )
        
        setattr(f, '__computed_field__', computed_field_info)
        return f

    if func is None:
        return wrapper
    else:
        return wrapper(func)
