"""Logic related to validators applied to models etc. via the `@field_validator` and `@model_validator` decorators."""
from __future__ import annotations as _annotations
from collections import deque
from dataclasses import dataclass, field
from functools import cached_property, partial, partialmethod
from inspect import Parameter, Signature, isdatadescriptor, ismethoddescriptor, signature
from itertools import islice
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Generic, Iterable, TypeVar, Union
from pydantic_core import PydanticUndefined, core_schema
from typing_extensions import Literal, TypeAlias, is_typeddict
from ..errors import PydanticUserError
from ._core_utils import get_type_ref
from ._internal_dataclass import slots_true
from ._typing_extra import get_function_type_hints
if TYPE_CHECKING:
    from ..fields import ComputedFieldInfo
    from ..functional_validators import FieldValidatorModes

@dataclass(**slots_true)
class ValidatorDecoratorInfo:
    """A container for data from `@validator` so that we can access it
    while building the pydantic-core schema.

    Attributes:
        decorator_repr: A class variable representing the decorator string, '@validator'.
        fields: A tuple of field names the validator should be called on.
        mode: The proposed validator mode.
        each_item: For complex objects (sets, lists etc.) whether to validate individual
            elements rather than the whole object.
        always: Whether this method and other validators should be called even if the value is missing.
        check_fields: Whether to check that the fields actually exist on the model.
    """
    decorator_repr: ClassVar[str] = '@validator'
    fields: tuple[str, ...]
    mode: Literal['before', 'after']
    each_item: bool
    always: bool
    check_fields: bool | None

@dataclass(**slots_true)
class FieldValidatorDecoratorInfo:
    """A container for data from `@field_validator` so that we can access it
    while building the pydantic-core schema.

    Attributes:
        decorator_repr: A class variable representing the decorator string, '@field_validator'.
        fields: A tuple of field names the validator should be called on.
        mode: The proposed validator mode.
        check_fields: Whether to check that the fields actually exist on the model.
    """
    decorator_repr: ClassVar[str] = '@field_validator'
    fields: tuple[str, ...]
    mode: FieldValidatorModes
    check_fields: bool | None

@dataclass(**slots_true)
class RootValidatorDecoratorInfo:
    """A container for data from `@root_validator` so that we can access it
    while building the pydantic-core schema.

    Attributes:
        decorator_repr: A class variable representing the decorator string, '@root_validator'.
        mode: The proposed validator mode.
    """
    decorator_repr: ClassVar[str] = '@root_validator'
    mode: Literal['before', 'after']

@dataclass(**slots_true)
class FieldSerializerDecoratorInfo:
    """A container for data from `@field_serializer` so that we can access it
    while building the pydantic-core schema.

    Attributes:
        decorator_repr: A class variable representing the decorator string, '@field_serializer'.
        fields: A tuple of field names the serializer should be called on.
        mode: The proposed serializer mode.
        return_type: The type of the serializer's return value.
        when_used: The serialization condition. Accepts a string with values `'always'`, `'unless-none'`, `'json'`,
            and `'json-unless-none'`.
        check_fields: Whether to check that the fields actually exist on the model.
    """
    decorator_repr: ClassVar[str] = '@field_serializer'
    fields: tuple[str, ...]
    mode: Literal['plain', 'wrap']
    return_type: Any
    when_used: core_schema.WhenUsed
    check_fields: bool | None

@dataclass(**slots_true)
class ModelSerializerDecoratorInfo:
    """A container for data from `@model_serializer` so that we can access it
    while building the pydantic-core schema.

    Attributes:
        decorator_repr: A class variable representing the decorator string, '@model_serializer'.
        mode: The proposed serializer mode.
        return_type: The type of the serializer's return value.
        when_used: The serialization condition. Accepts a string with values `'always'`, `'unless-none'`, `'json'`,
            and `'json-unless-none'`.
    """
    decorator_repr: ClassVar[str] = '@model_serializer'
    mode: Literal['plain', 'wrap']
    return_type: Any
    when_used: core_schema.WhenUsed

@dataclass(**slots_true)
class ModelValidatorDecoratorInfo:
    """A container for data from `@model_validator` so that we can access it
    while building the pydantic-core schema.

    Attributes:
        decorator_repr: A class variable representing the decorator string, '@model_serializer'.
        mode: The proposed serializer mode.
    """
    decorator_repr: ClassVar[str] = '@model_validator'
    mode: Literal['wrap', 'before', 'after']
DecoratorInfo: TypeAlias = 'Union[\n    ValidatorDecoratorInfo,\n    FieldValidatorDecoratorInfo,\n    RootValidatorDecoratorInfo,\n    FieldSerializerDecoratorInfo,\n    ModelSerializerDecoratorInfo,\n    ModelValidatorDecoratorInfo,\n    ComputedFieldInfo,\n]'
ReturnType = TypeVar('ReturnType')
DecoratedType: TypeAlias = 'Union[classmethod[Any, Any, ReturnType], staticmethod[Any, ReturnType], Callable[..., ReturnType], property]'

@dataclass
class PydanticDescriptorProxy(Generic[ReturnType]):
    """Wrap a classmethod, staticmethod, property or unbound function
    and act as a descriptor that allows us to detect decorated items
    from the class' attributes.

    This class' __get__ returns the wrapped item's __get__ result,
    which makes it transparent for classmethods and staticmethods.

    Attributes:
        wrapped: The decorator that has to be wrapped.
        decorator_info: The decorator info.
        shim: A wrapper function to wrap V1 style function.
    """
    wrapped: DecoratedType[ReturnType]
    decorator_info: DecoratorInfo
    shim: Callable[[Callable[..., Any]], Callable[..., Any]] | None = None

    def __post_init__(self):
        for attr in ('setter', 'deleter'):
            if hasattr(self.wrapped, attr):
                f = partial(self._call_wrapped_attr, name=attr)
                setattr(self, attr, f)

    def __get__(self, obj: object | None, obj_type: type[object] | None=None) -> PydanticDescriptorProxy[ReturnType]:
        try:
            return self.wrapped.__get__(obj, obj_type)
        except AttributeError:
            return self.wrapped

    def __set_name__(self, instance: Any, name: str) -> None:
        if hasattr(self.wrapped, '__set_name__'):
            self.wrapped.__set_name__(instance, name)

    def __getattr__(self, __name: str) -> Any:
        """Forward checks for __isabstractmethod__ and such."""
        return getattr(self.wrapped, __name)
DecoratorInfoType = TypeVar('DecoratorInfoType', bound=DecoratorInfo)

@dataclass(**slots_true)
class Decorator(Generic[DecoratorInfoType]):
    """A generic container class to join together the decorator metadata
    (metadata from decorator itself, which we have when the
    decorator is called but not when we are building the core-schema)
    and the bound function (which we have after the class itself is created).

    Attributes:
        cls_ref: The class ref.
        cls_var_name: The decorated function name.
        func: The decorated function.
        shim: A wrapper function to wrap V1 style function.
        info: The decorator info.
    """
    cls_ref: str
    cls_var_name: str
    func: Callable[..., Any]
    shim: Callable[[Any], Any] | None
    info: DecoratorInfoType

    @staticmethod
    def build(cls_: Any, *, cls_var_name: str, shim: Callable[[Any], Any] | None, info: DecoratorInfoType) -> Decorator[DecoratorInfoType]:
        """Build a new decorator.

        Args:
            cls_: The class.
            cls_var_name: The decorated function name.
            shim: A wrapper function to wrap V1 style function.
            info: The decorator info.

        Returns:
            The new decorator instance.
        """
        cls_ref = get_type_ref(cls_)
        func = get_attribute_from_base_dicts(cls_, cls_var_name)
        return Decorator(cls_ref=cls_ref, cls_var_name=cls_var_name, func=func, shim=shim, info=info)

    def bind_to_cls(self, cls: Any) -> Decorator[DecoratorInfoType]:
        """Bind the decorator to a class.

        Args:
            cls: the class.

        Returns:
            The new decorator instance.
        """
        cls_ref = get_type_ref(cls)
        func = get_attribute_from_base_dicts(cls, self.cls_var_name)
        return Decorator(cls_ref=cls_ref, cls_var_name=self.cls_var_name, func=func, shim=self.shim, info=self.info)

def get_bases(tp: type[Any]) -> tuple[type[Any], ...]:
    """Get the base classes of a class or typeddict.

    Args:
        tp: The type or class to get the bases.

    Returns:
        The base classes.
    """
    if is_typeddict(tp):
        return tp.__orig_bases__
    return tp.__bases__

def mro(tp: type[Any]) -> tuple[type[Any], ...]:
    """Calculate the Method Resolution Order of bases using the C3 algorithm.

    See https://www.python.org/download/releases/2.3/mro/
    """
    if hasattr(tp, '__mro__'):
        return tp.__mro__
    else:
        # Simulate MRO for TypedDict which doesn't have a real MRO
        result = [tp]
        for base in get_bases(tp):
            for cls in mro(base):
                if cls not in result:
                    result.append(cls)
        return tuple(result)
_sentinel = object()

def get_attribute_from_bases(tp: type[Any] | tuple[type[Any], ...], name: str) -> Any:
    """Get the attribute from the next class in the MRO that has it,
    aiming to simulate calling the method on the actual class.

    The reason for iterating over the mro instead of just getting
    the attribute (which would do that for us) is to support TypedDict,
    which lacks a real __mro__, but can have a virtual one constructed
    from its bases (as done here).

    Args:
        tp: The type or class to search for the attribute. If a tuple, this is treated as a set of base classes.
        name: The name of the attribute to retrieve.

    Returns:
        Any: The attribute value, if found.

    Raises:
        AttributeError: If the attribute is not found in any class in the MRO.
    """
    if isinstance(tp, tuple):
        classes = tp
    else:
        classes = mro(tp)

    for cls in classes:
        if name in cls.__dict__:
            return cls.__dict__[name]
    raise AttributeError(f"{tp} has no attribute '{name}'")

def get_attribute_from_base_dicts(tp: type[Any], name: str) -> Any:
    """Get an attribute out of the `__dict__` following the MRO.
    This prevents the call to `__get__` on the descriptor, and allows
    us to get the original function for classmethod properties.

    Args:
        tp: The type or class to search for the attribute.
        name: The name of the attribute to retrieve.

    Returns:
        Any: The attribute value, if found.

    Raises:
        KeyError: If the attribute is not found in any class's `__dict__` in the MRO.
    """
    for cls in mro(tp):
        try:
            return cls.__dict__[name]
        except KeyError:
            pass
    raise KeyError(f"{tp} has no attribute '{name}' in its __dict__ or its bases")

@dataclass(**slots_true)
class DecoratorInfos:
    """Mapping of name in the class namespace to decorator info.

    note that the name in the class namespace is the function or attribute name
    not the field name!
    """
    validators: dict[str, Decorator[ValidatorDecoratorInfo]] = field(default_factory=dict)
    field_validators: dict[str, Decorator[FieldValidatorDecoratorInfo]] = field(default_factory=dict)
    root_validators: dict[str, Decorator[RootValidatorDecoratorInfo]] = field(default_factory=dict)
    field_serializers: dict[str, Decorator[FieldSerializerDecoratorInfo]] = field(default_factory=dict)
    model_serializers: dict[str, Decorator[ModelSerializerDecoratorInfo]] = field(default_factory=dict)
    model_validators: dict[str, Decorator[ModelValidatorDecoratorInfo]] = field(default_factory=dict)
    computed_fields: dict[str, Decorator[ComputedFieldInfo]] = field(default_factory=dict)

    @staticmethod
    def build(model_dc: type[Any]) -> DecoratorInfos:
        """We want to collect all DecFunc instances that exist as
        attributes in the namespace of the class (a BaseModel or dataclass)
        that called us
        But we want to collect these in the order of the bases
        So instead of getting them all from the leaf class (the class that called us),
        we traverse the bases from root (the oldest ancestor class) to leaf
        and collect all of the instances as we go, taking care to replace
        any duplicate ones with the last one we see to mimic how function overriding
        works with inheritance.
        If we do replace any functions we put the replacement into the position
        the replaced function was in; that is, we maintain the order.
        """
        decorator_infos = DecoratorInfos()
        for base in reversed(mro(model_dc)):
            for name, value in base.__dict__.items():
                if isinstance(value, PydanticDescriptorProxy):
                    info = value.decorator_info
                    if isinstance(info, ValidatorDecoratorInfo):
                        decorator_infos.validators[name] = Decorator.build(base, cls_var_name=name, shim=value.shim, info=info)
                    elif isinstance(info, FieldValidatorDecoratorInfo):
                        decorator_infos.field_validators[name] = Decorator.build(base, cls_var_name=name, shim=value.shim, info=info)
                    elif isinstance(info, RootValidatorDecoratorInfo):
                        decorator_infos.root_validators[name] = Decorator.build(base, cls_var_name=name, shim=value.shim, info=info)
                    elif isinstance(info, FieldSerializerDecoratorInfo):
                        decorator_infos.field_serializers[name] = Decorator.build(base, cls_var_name=name, shim=value.shim, info=info)
                    elif isinstance(info, ModelSerializerDecoratorInfo):
                        decorator_infos.model_serializers[name] = Decorator.build(base, cls_var_name=name, shim=value.shim, info=info)
                    elif isinstance(info, ModelValidatorDecoratorInfo):
                        decorator_infos.model_validators[name] = Decorator.build(base, cls_var_name=name, shim=value.shim, info=info)
                    elif isinstance(info, ComputedFieldInfo):
                        decorator_infos.computed_fields[name] = Decorator.build(base, cls_var_name=name, shim=value.shim, info=info)
        return decorator_infos

def inspect_validator(validator: Callable[..., Any], mode: FieldValidatorModes) -> bool:
    """Look at a field or model validator function and determine whether it takes an info argument.

    An error is raised if the function has an invalid signature.

    Args:
        validator: The validator function to inspect.
        mode: The proposed validator mode.

    Returns:
        Whether the validator takes an info argument.
    """
    sig = signature(validator)
    params = list(sig.parameters.values())
    
    if mode == 'before':
        if len(params) == 1:
            return False
        elif len(params) == 2 and params[1].name == 'info':
            return True
        else:
            raise PydanticUserError(f"'before' validators should have one or two arguments, got {len(params)}")
    elif mode == 'after':
        if len(params) == 2:
            return False
        elif len(params) == 3 and params[2].name == 'info':
            return True
        else:
            raise PydanticUserError(f"'after' validators should have two or three arguments, got {len(params)}")
    else:
        raise ValueError(f"Invalid mode: {mode}")

def inspect_field_serializer(serializer: Callable[..., Any], mode: Literal['plain', 'wrap'], computed_field: bool=False) -> tuple[bool, bool]:
    """Look at a field serializer function and determine if it is a field serializer,
    and whether it takes an info argument.

    An error is raised if the function has an invalid signature.

    Args:
        serializer: The serializer function to inspect.
        mode: The serializer mode, either 'plain' or 'wrap'.
        computed_field: When serializer is applied on computed_field. It doesn't require
            info signature.

    Returns:
        Tuple of (is_field_serializer, info_arg).
    """
    sig = signature(serializer)
    params = list(sig.parameters.values())

    if mode == 'plain':
        if len(params) == 1:
            return True, False
        elif len(params) == 2 and params[1].name == 'info':
            return True, True
        elif computed_field and len(params) == 0:
            return True, False
        else:
            raise PydanticUserError(f"'plain' serializers should have one or two arguments, got {len(params)}")
    elif mode == 'wrap':
        if len(params) == 2:
            return True, False
        elif len(params) == 3 and params[2].name == 'info':
            return True, True
        else:
            raise PydanticUserError(f"'wrap' serializers should have two or three arguments, got {len(params)}")
    else:
        raise ValueError(f"Invalid mode: {mode}")

def inspect_annotated_serializer(serializer: Callable[..., Any], mode: Literal['plain', 'wrap']) -> bool:
    """Look at a serializer function used via `Annotated` and determine whether it takes an info argument.

    An error is raised if the function has an invalid signature.

    Args:
        serializer: The serializer function to check.
        mode: The serializer mode, either 'plain' or 'wrap'.

    Returns:
        info_arg
    """
    sig = signature(serializer)
    params = list(sig.parameters.values())

    if mode == 'plain':
        if len(params) == 1:
            return False
        elif len(params) == 2 and params[1].name == 'info':
            return True
        else:
            raise PydanticUserError(f"'plain' serializers should have one or two arguments, got {len(params)}")
    elif mode == 'wrap':
        if len(params) == 2:
            return False
        elif len(params) == 3 and params[2].name == 'info':
            return True
        else:
            raise PydanticUserError(f"'wrap' serializers should have two or three arguments, got {len(params)}")
    else:
        raise ValueError(f"Invalid mode: {mode}")

def inspect_model_serializer(serializer: Callable[..., Any], mode: Literal['plain', 'wrap']) -> bool:
    """Look at a model serializer function and determine whether it takes an info argument.

    An error is raised if the function has an invalid signature.

    Args:
        serializer: The serializer function to check.
        mode: The serializer mode, either 'plain' or 'wrap'.

    Returns:
        `info_arg` - whether the function expects an info argument.
    """
    sig = signature(serializer)
    params = list(sig.parameters.values())

    if mode == 'plain':
        if len(params) == 1:
            return False
        elif len(params) == 2 and params[1].name == 'info':
            return True
        else:
            raise PydanticUserError(f"'plain' model serializers should have one or two arguments, got {len(params)}")
    elif mode == 'wrap':
        if len(params) == 2:
            return False
        elif len(params) == 3 and params[2].name == 'info':
            return True
        else:
            raise PydanticUserError(f"'wrap' model serializers should have two or three arguments, got {len(params)}")
    else:
        raise ValueError(f"Invalid mode: {mode}")
AnyDecoratorCallable: TypeAlias = 'Union[classmethod[Any, Any, Any], staticmethod[Any, Any], partialmethod[Any], Callable[..., Any]]'

def is_instance_method_from_sig(function: AnyDecoratorCallable) -> bool:
    """Whether the function is an instance method.

    It will consider a function as instance method if the first parameter of
    function is `self`.

    Args:
        function: The function to check.

    Returns:
        `True` if the function is an instance method, `False` otherwise.
    """
    sig = signature(unwrap_wrapped_function(function))
    params = list(sig.parameters.values())
    return params and params[0].name == 'self'

def ensure_classmethod_based_on_signature(function: AnyDecoratorCallable) -> Any:
    """Apply the `@classmethod` decorator on the function.

    Args:
        function: The function to apply the decorator on.

    Return:
        The `@classmethod` decorator applied function.
    """
    if isinstance(function, (classmethod, staticmethod)):
        return function
    elif is_instance_method_from_sig(function):
        return function
    else:
        return classmethod(function)

def unwrap_wrapped_function(func: Any, *, unwrap_partial: bool=True, unwrap_class_static_method: bool=True) -> Any:
    """Recursively unwraps a wrapped function until the underlying function is reached.
    This handles property, functools.partial, functools.partialmethod, staticmethod, and classmethod.

    Args:
        func: The function to unwrap.
        unwrap_partial: If True (default), unwrap partial and partialmethod decorators.
        unwrap_class_static_method: If True (default), also unwrap classmethod and staticmethod
            decorators. If False, only unwrap partial and partialmethod decorators.

    Returns:
        The underlying function of the wrapped function.
    """
    while True:
        if isinstance(func, property):
            func = func.fget
        elif unwrap_partial and isinstance(func, (partial, partialmethod)):
            func = func.func
        elif unwrap_class_static_method and isinstance(func, (classmethod, staticmethod)):
            func = func.__func__
        elif hasattr(func, '__wrapped__'):
            func = func.__wrapped__
        else:
            return func

def get_function_return_type(func: Any, explicit_return_type: Any, types_namespace: dict[str, Any] | None=None) -> Any:
    """Get the function return type.

    It gets the return type from the type annotation if `explicit_return_type` is `None`.
    Otherwise, it returns `explicit_return_type`.

    Args:
        func: The function to get its return type.
        explicit_return_type: The explicit return type.
        types_namespace: The types namespace, defaults to `None`.

    Returns:
        The function return type.
    """
    if explicit_return_type is not None:
        return explicit_return_type

    func = unwrap_wrapped_function(func)
    type_hints = get_function_type_hints(func, types_namespace=types_namespace)
    return type_hints.get('return', Any)

def count_positional_required_params(sig: Signature) -> int:
    """Get the number of positional (required) arguments of a signature.

    This function should only be used to inspect signatures of validation and serialization functions.
    The first argument (the value being serialized or validated) is counted as a required argument
    even if a default value exists.

    Returns:
        The number of positional arguments of a signature.
    """
    count = 0
    for i, param in enumerate(sig.parameters.values()):
        if i == 0 or (param.default is Parameter.empty and param.kind in {Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD}):
            count += 1
        else:
            break
    return count

def ensure_property(f: Any) -> Any:
    """Ensure that a function is a `property` or `cached_property`, or is a valid descriptor.

    Args:
        f: The function to check.

    Returns:
        The function, or a `property` or `cached_property` instance wrapping the function.
    """
    if isinstance(f, (property, cached_property)) or isdatadescriptor(f) or ismethoddescriptor(f):
        return f
    else:
        return property(f)
