"""Private logic for creating models."""
from __future__ import annotations as _annotations
import builtins
import operator
import typing
import warnings
import weakref
from abc import ABCMeta
from functools import partial
from types import FunctionType
from typing import Any, Callable, Generic, NoReturn
import typing_extensions
from pydantic_core import PydanticUndefined, SchemaSerializer
from typing_extensions import dataclass_transform, deprecated
from ..errors import PydanticUndefinedAnnotation, PydanticUserError
from ..plugin._schema_validator import create_schema_validator
from ..warnings import GenericBeforeBaseModelWarning, PydanticDeprecatedSince20
from ._config import ConfigWrapper
from ._decorators import DecoratorInfos, PydanticDescriptorProxy, get_attribute_from_bases, unwrap_wrapped_function
from ._fields import collect_model_fields, is_valid_field_name, is_valid_privateattr_name
from ._generate_schema import GenerateSchema
from ._generics import PydanticGenericMetadata, get_model_typevars_map
from ._mock_val_ser import set_model_mocks
from ._schema_generation_shared import CallbackGetCoreSchemaHandler
from ._signature import generate_pydantic_signature
from ._typing_extra import get_cls_types_namespace, is_annotated, is_classvar, parent_frame_namespace
from ._utils import ClassAttribute, SafeGetItemProxy
from ._validate_call import ValidateCallWrapper
if typing.TYPE_CHECKING:
    from ..fields import Field as PydanticModelField
    from ..fields import FieldInfo, ModelPrivateAttr
    from ..fields import PrivateAttr as PydanticModelPrivateAttr
    from ..main import BaseModel
else:
    DeprecationWarning = PydanticDeprecatedSince20
    PydanticModelField = object()
    PydanticModelPrivateAttr = object()
object_setattr = object.__setattr__

class _ModelNamespaceDict(dict):
    """A dictionary subclass that intercepts attribute setting on model classes and
    warns about overriding of decorators.
    """

    def __setitem__(self, k: str, v: object) -> None:
        existing: Any = self.get(k, None)
        if existing and v is not existing and isinstance(existing, PydanticDescriptorProxy):
            warnings.warn(f'`{k}` overrides an existing Pydantic `{existing.decorator_info.decorator_repr}` decorator')
        return super().__setitem__(k, v)

@dataclass_transform(kw_only_default=True, field_specifiers=(PydanticModelField, PydanticModelPrivateAttr))
class ModelMetaclass(ABCMeta):

    def __new__(mcs, cls_name: str, bases: tuple[type[Any], ...], namespace: dict[str, Any], __pydantic_generic_metadata__: PydanticGenericMetadata | None=None, __pydantic_reset_parent_namespace__: bool=True, _create_model_module: str | None=None, **kwargs: Any) -> type:
        """Metaclass for creating Pydantic models.

        Args:
            cls_name: The name of the class to be created.
            bases: The base classes of the class to be created.
            namespace: The attribute dictionary of the class to be created.
            __pydantic_generic_metadata__: Metadata for generic models.
            __pydantic_reset_parent_namespace__: Reset parent namespace.
            _create_model_module: The module of the class to be created, if created by `create_model`.
            **kwargs: Catch-all for any other keyword arguments.

        Returns:
            The new class created by the metaclass.
        """
        if bases:
            base_field_names, class_vars, base_private_attributes = mcs._collect_bases_data(bases)
            config_wrapper = ConfigWrapper.for_model(bases, namespace, kwargs)
            namespace['model_config'] = config_wrapper.config_dict
            private_attributes = inspect_namespace(namespace, config_wrapper.ignored_types, class_vars, base_field_names)
            if private_attributes or base_private_attributes:
                original_model_post_init = get_model_post_init(namespace, bases)
                if original_model_post_init is not None:

                    def wrapped_model_post_init(self: BaseModel, context: Any, /) -> None:
                        """We need to both initialize private attributes and call the user-defined model_post_init
                        method.
                        """
                        init_private_attributes(self, context)
                        original_model_post_init(self, context)
                    namespace['model_post_init'] = wrapped_model_post_init
                else:
                    namespace['model_post_init'] = init_private_attributes
            namespace['__class_vars__'] = class_vars
            namespace['__private_attributes__'] = {**base_private_attributes, **private_attributes}
            cls: type[BaseModel] = super().__new__(mcs, cls_name, bases, namespace, **kwargs)
            from ..main import BaseModel
            mro = cls.__mro__
            if Generic in mro and mro.index(Generic) < mro.index(BaseModel):
                warnings.warn(GenericBeforeBaseModelWarning('Classes should inherit from `BaseModel` before generic classes (e.g. `typing.Generic[T]`) for pydantic generics to work properly.'), stacklevel=2)
            cls.__pydantic_custom_init__ = not getattr(cls.__init__, '__pydantic_base_init__', False)
            cls.__pydantic_post_init__ = None if cls.model_post_init is BaseModel.model_post_init else 'model_post_init'
            cls.__pydantic_decorators__ = DecoratorInfos.build(cls)
            if __pydantic_generic_metadata__:
                cls.__pydantic_generic_metadata__ = __pydantic_generic_metadata__
            else:
                parent_parameters = getattr(cls, '__pydantic_generic_metadata__', {}).get('parameters', ())
                parameters = getattr(cls, '__parameters__', None) or parent_parameters
                if parameters and parent_parameters and (not all((x in parameters for x in parent_parameters))):
                    from ..root_model import RootModelRootType
                    missing_parameters = tuple((x for x in parameters if x not in parent_parameters))
                    if RootModelRootType in parent_parameters and RootModelRootType not in parameters:
                        parameters_str = ', '.join([x.__name__ for x in missing_parameters])
                        error_message = f'{cls.__name__} is a subclass of `RootModel`, but does not include the generic type identifier(s) {parameters_str} in its parameters. You should parametrize RootModel directly, e.g., `class {cls.__name__}(RootModel[{parameters_str}]): ...`.'
                    else:
                        combined_parameters = parent_parameters + missing_parameters
                        parameters_str = ', '.join([str(x) for x in combined_parameters])
                        generic_type_label = f'typing.Generic[{parameters_str}]'
                        error_message = f'All parameters must be present on typing.Generic; you should inherit from {generic_type_label}.'
                        if Generic not in bases:
                            bases_str = ', '.join([x.__name__ for x in bases] + [generic_type_label])
                            error_message += f' Note: `typing.Generic` must go last: `class {cls.__name__}({bases_str}): ...`)'
                    raise TypeError(error_message)
                cls.__pydantic_generic_metadata__ = {'origin': None, 'args': (), 'parameters': parameters}
            cls.__pydantic_complete__ = False
            for name, obj in private_attributes.items():
                obj.__set_name__(cls, name)
            if __pydantic_reset_parent_namespace__:
                cls.__pydantic_parent_namespace__ = build_lenient_weakvaluedict(parent_frame_namespace())
            parent_namespace = getattr(cls, '__pydantic_parent_namespace__', None)
            if isinstance(parent_namespace, dict):
                parent_namespace = unpack_lenient_weakvaluedict(parent_namespace)
            types_namespace = get_cls_types_namespace(cls, parent_namespace)
            set_model_fields(cls, bases, config_wrapper, types_namespace)
            if config_wrapper.frozen and '__hash__' not in namespace:
                set_default_hash_func(cls, bases)
            complete_model_class(cls, cls_name, config_wrapper, raise_errors=False, types_namespace=types_namespace, create_model_module=_create_model_module)
            cls.model_computed_fields = {k: v.info for k, v in cls.__pydantic_decorators__.computed_fields.items()}
            set_deprecated_descriptors(cls)
            super(cls, cls).__pydantic_init_subclass__(**kwargs)
            return cls
        else:
            return super().__new__(mcs, cls_name, bases, namespace, **kwargs)
    if not typing.TYPE_CHECKING:

        def __getattr__(self, item: str) -> Any:
            """This is necessary to keep attribute access working for class attribute access."""
            private_attributes = self.__dict__.get('__private_attributes__')
            if private_attributes and item in private_attributes:
                return private_attributes[item]
            raise AttributeError(item)

    @classmethod
    def __prepare__(cls, *args: Any, **kwargs: Any) -> dict[str, object]:
        return _ModelNamespaceDict()

    def __instancecheck__(self, instance: Any) -> bool:
        """Avoid calling ABC _abc_subclasscheck unless we're pretty sure.

        See #3829 and python/cpython#92810
        """
        return hasattr(instance, '__pydantic_validator__') and super().__instancecheck__(instance)

    @property
    @deprecated('The `__fields__` attribute is deprecated, use `model_fields` instead.', category=None)
    def __fields__(self) -> dict[str, FieldInfo]:
        warnings.warn('The `__fields__` attribute is deprecated, use `model_fields` instead.', PydanticDeprecatedSince20)
        return self.model_fields

    def __dir__(self) -> list[str]:
        attributes = list(super().__dir__())
        if '__fields__' in attributes:
            attributes.remove('__fields__')
        return attributes

def init_private_attributes(self: BaseModel, context: Any, /) -> None:
    """This function is meant to behave like a BaseModel method to initialise private attributes.

    It takes context as an argument since that's what pydantic-core passes when calling it.

    Args:
        self: The BaseModel instance.
        context: The context.
    """
    for name, private_attr in self.__private_attributes__.items():
        default = private_attr.get_default()
        if default is not PydanticUndefined:
            setattr(self, name, default)

def get_model_post_init(namespace: dict[str, Any], bases: tuple[type[Any], ...]) -> Callable[..., Any] | None:
    """Get the `model_post_init` method from the namespace or the class bases, or `None` if not defined."""
    if 'model_post_init' in namespace:
        return namespace['model_post_init']
    for base in bases:
        if hasattr(base, 'model_post_init'):
            return getattr(base, 'model_post_init')
    return None

def inspect_namespace(namespace: dict[str, Any], ignored_types: tuple[type[Any], ...], base_class_vars: set[str], base_class_fields: set[str]) -> dict[str, ModelPrivateAttr]:
    """Iterate over the namespace and:
    * gather private attributes
    * check for items which look like fields but are not (e.g. have no annotation) and warn.

    Args:
        namespace: The attribute dictionary of the class to be created.
        ignored_types: A tuple of ignore types.
        base_class_vars: A set of base class class variables.
        base_class_fields: A set of base class fields.

    Returns:
        A dict contains private attributes info.

    Raises:
        TypeError: If there is a `__root__` field in model.
        NameError: If private attribute name is invalid.
        PydanticUserError:
            - If a field does not have a type annotation.
            - If a field on base class was overridden by a non-annotated attribute.
    """
    private_attributes: dict[str, ModelPrivateAttr] = {}
    annotations = namespace.get('__annotations__', {})

    if '__root__' in namespace or '__root__' in annotations:
        raise TypeError("To define root models, use `pydantic.RootModel` rather than a field called '__root__'")

    for name, value in namespace.items():
        if name in base_class_vars:
            continue

        if isinstance(value, ModelPrivateAttr):
            if name.startswith('__') and not name.endswith('__'):
                raise NameError(f'Private attributes must not use dunder names; use a single underscore prefix instead: {name!r}')
            if not is_valid_privateattr_name(name):
                raise NameError(f'Private attribute names must start with an underscore: {name!r}')
            private_attributes[name] = value
        elif not is_valid_field_name(name):
            continue
        elif name in annotations:
            if name not in base_class_fields and isinstance(value, ignored_types):
                continue
        elif isinstance(value, ignored_types):
            continue
        elif name in base_class_fields:
            raise PydanticUserError(
                f'Field {name!r} defined on a base class was overridden by a non-annotated attribute. '
                f'All fields must be annotated. '
                f'To avoid this error, use `{name}: {type(value).__name__} = ...`',
                code='model-field-overridden',
            )
        else:
            warnings.warn(f'Field {name!r} has no type annotation', UserWarning, stacklevel=2)

    return private_attributes

def set_model_fields(cls: type[BaseModel], bases: tuple[type[Any], ...], config_wrapper: ConfigWrapper, types_namespace: dict[str, Any]) -> None:
    """Collect and set `cls.model_fields` and `cls.__class_vars__`.

    Args:
        cls: BaseModel or dataclass.
        bases: Parents of the class, generally `cls.__bases__`.
        config_wrapper: The config wrapper instance.
        types_namespace: Optional extra namespace to look for types in.
    """
    cls.model_fields, cls.__class_vars__ = collect_model_fields(
        cls,
        bases,
        config_wrapper,
        types_namespace,
    )

def complete_model_class(cls: type[BaseModel], cls_name: str, config_wrapper: ConfigWrapper, *, raise_errors: bool=True, types_namespace: dict[str, Any] | None, create_model_module: str | None=None) -> bool:
    """Finish building a model class.

    This logic must be called after class has been created since validation functions must be bound
    and `get_type_hints` requires a class object.

    Args:
        cls: BaseModel or dataclass.
        cls_name: The model or dataclass name.
        config_wrapper: The config wrapper instance.
        raise_errors: Whether to raise errors.
        types_namespace: Optional extra namespace to look for types in.
        create_model_module: The module of the class to be created, if created by `create_model`.

    Returns:
        `True` if the model is successfully completed, else `False`.

    Raises:
        PydanticUndefinedAnnotation: If `PydanticUndefinedAnnotation` occurs in`__get_pydantic_core_schema__`
            and `raise_errors=True`.
    """
    try:
        cls.__pydantic_core_schema__ = cls.__get_pydantic_core_schema__(
            cls,
            CallbackGetCoreSchemaHandler(
                partial(
                    GenerateSchema,
                    config_wrapper,
                    types_namespace,
                ),
            ),
        )
        cls.__pydantic_validator__ = create_schema_validator(
            cls.__pydantic_core_schema__,
            cls,
            config_wrapper.config_dict,
            cls_name=cls_name,
        )
        cls.__pydantic_serializer__ = SchemaSerializer(cls.__pydantic_core_schema__)
        cls.__pydantic_complete__ = True
        return True
    except PydanticUndefinedAnnotation as e:
        if raise_errors:
            raise
        cls.__pydantic_complete__ = False
        return False

def set_deprecated_descriptors(cls: type[BaseModel]) -> None:
    """Set data descriptors on the class for deprecated fields."""
    for field_name, field in cls.model_fields.items():
        if field.deprecated:
            msg = f'Field {field_name!r} is deprecated'
            if isinstance(field.deprecated, str):
                msg += f': {field.deprecated}'
            setattr(cls, field_name, _DeprecatedFieldDescriptor(msg))

class _DeprecatedFieldDescriptor:
    """Data descriptor used to emit a runtime deprecation warning before accessing a deprecated field.

    Attributes:
        msg: The deprecation message to be emitted.
        wrapped_property: The property instance if the deprecated field is a computed field, or `None`.
        field_name: The name of the field being deprecated.
    """
    field_name: str

    def __init__(self, msg: str, wrapped_property: property | None=None) -> None:
        self.msg = msg
        self.wrapped_property = wrapped_property

    def __set_name__(self, cls: type[BaseModel], name: str) -> None:
        self.field_name = name

    def __get__(self, obj: BaseModel | None, obj_type: type[BaseModel] | None=None) -> Any:
        if obj is None:
            raise AttributeError(self.field_name)
        warnings.warn(self.msg, builtins.DeprecationWarning, stacklevel=2)
        if self.wrapped_property is not None:
            return self.wrapped_property.__get__(obj, obj_type)
        return obj.__dict__[self.field_name]

    def __set__(self, obj: Any, value: Any) -> NoReturn:
        raise AttributeError(self.field_name)

class _PydanticWeakRef:
    """Wrapper for `weakref.ref` that enables `pickle` serialization.

    Cloudpickle fails to serialize `weakref.ref` objects due to an arcane error related
    to abstract base classes (`abc.ABC`). This class works around the issue by wrapping
    `weakref.ref` instead of subclassing it.

    See https://github.com/pydantic/pydantic/issues/6763 for context.

    Semantics:
        - If not pickled, behaves the same as a `weakref.ref`.
        - If pickled along with the referenced object, the same `weakref.ref` behavior
          will be maintained between them after unpickling.
        - If pickled without the referenced object, after unpickling the underlying
          reference will be cleared (`__call__` will always return `None`).
    """

    def __init__(self, obj: Any):
        if obj is None:
            self._wr = None
        else:
            self._wr = weakref.ref(obj)

    def __call__(self) -> Any:
        if self._wr is None:
            return None
        else:
            return self._wr()

    def __reduce__(self) -> tuple[Callable, tuple[weakref.ReferenceType | None]]:
        return (_PydanticWeakRef, (self(),))

def build_lenient_weakvaluedict(d: dict[str, Any] | None) -> dict[str, Any] | None:
    """Takes an input dictionary, and produces a new value that (invertibly) replaces the values with weakrefs.

    We can't just use a WeakValueDictionary because many types (including int, str, etc.) can't be stored as values
    in a WeakValueDictionary.

    The `unpack_lenient_weakvaluedict` function can be used to reverse this operation.
    """
    if d is None:
        return None
    return {k: _PydanticWeakRef(v) for k, v in d.items()}

def unpack_lenient_weakvaluedict(d: dict[str, Any] | None) -> dict[str, Any] | None:
    """Inverts the transform performed by `build_lenient_weakvaluedict`."""
    if d is None:
        return None
    return {k: v() for k, v in d.items()}
