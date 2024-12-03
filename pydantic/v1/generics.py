import sys
import types
import typing
from typing import TYPE_CHECKING, Any, ClassVar, Dict, ForwardRef, Generic, Iterator, List, Mapping, Optional, Tuple, Type, TypeVar, Union, cast
from weakref import WeakKeyDictionary, WeakValueDictionary
from typing_extensions import Annotated, Literal as ExtLiteral
from pydantic.v1.class_validators import gather_all_validators
from pydantic.v1.fields import DeferredType
from pydantic.v1.main import BaseModel, create_model
from pydantic.v1.types import JsonWrapper
from pydantic.v1.typing import display_as_type, get_all_type_hints, get_args, get_origin, typing_base
from pydantic.v1.utils import all_identical, lenient_issubclass
if sys.version_info >= (3, 10):
    from typing import _UnionGenericAlias
if sys.version_info >= (3, 8):
    from typing import Literal
GenericModelT = TypeVar('GenericModelT', bound='GenericModel')
TypeVarType = Any
CacheKey = Tuple[Type[Any], Any, Tuple[Any, ...]]
Parametrization = Mapping[TypeVarType, Type[Any]]
if sys.version_info >= (3, 9):
    GenericTypesCache = WeakValueDictionary[CacheKey, Type[BaseModel]]
    AssignedParameters = WeakKeyDictionary[Type[BaseModel], Parametrization]
else:
    GenericTypesCache = WeakValueDictionary
    AssignedParameters = WeakKeyDictionary
_generic_types_cache = GenericTypesCache()
_assigned_parameters = AssignedParameters()

class GenericModel(BaseModel):
    __slots__ = ()
    __concrete__: ClassVar[bool] = False
    if TYPE_CHECKING:
        __parameters__: ClassVar[Tuple[TypeVarType, ...]]

    def __class_getitem__(cls: Type[GenericModelT], params: Union[Type[Any], Tuple[Type[Any], ...]]) -> Type[Any]:
        """Instantiates a new class from a generic class `cls` and type variables `params`.

        :param params: Tuple of types the class . Given a generic class
            `Model` with 2 type variables and a concrete model `Model[str, int]`,
            the value `(str, int)` would be passed to `params`.
        :return: New model class inheriting from `cls` with instantiated
            types described by `params`. If no parameters are given, `cls` is
            returned as is.

        """

        def _cache_key(_params: Any) -> CacheKey:
            args = get_args(_params)
            if len(args) == 2 and isinstance(args[0], list):
                args = (tuple(args[0]), args[1])
            return (cls, _params, args)
        cached = _generic_types_cache.get(_cache_key(params))
        if cached is not None:
            return cached
        if cls.__concrete__ and Generic not in cls.__bases__:
            raise TypeError('Cannot parameterize a concrete instantiation of a generic model')
        if not isinstance(params, tuple):
            params = (params,)
        if cls is GenericModel and any((isinstance(param, TypeVar) for param in params)):
            raise TypeError('Type parameters should be placed on typing.Generic, not GenericModel')
        if not hasattr(cls, '__parameters__'):
            raise TypeError(f'Type {cls.__name__} must inherit from typing.Generic before being parameterized')
        check_parameters_count(cls, params)
        typevars_map: Dict[TypeVarType, Type[Any]] = dict(zip(cls.__parameters__, params))
        if all_identical(typevars_map.keys(), typevars_map.values()) and typevars_map:
            return cls
        model_name = cls.__concrete_name__(params)
        validators = gather_all_validators(cls)
        type_hints = get_all_type_hints(cls).items()
        instance_type_hints = {k: v for k, v in type_hints if get_origin(v) is not ClassVar}
        fields = {k: (DeferredType(), cls.__fields__[k].field_info) for k in instance_type_hints if k in cls.__fields__}
        model_module, called_globally = get_caller_frame_info()
        created_model = cast(Type[GenericModel], create_model(model_name, __module__=model_module or cls.__module__, __base__=(cls,) + tuple(cls.__parameterized_bases__(typevars_map)), __config__=None, __validators__=validators, __cls_kwargs__=None, **fields))
        _assigned_parameters[created_model] = typevars_map
        if called_globally:
            object_by_reference = None
            reference_name = model_name
            reference_module_globals = sys.modules[created_model.__module__].__dict__
            while object_by_reference is not created_model:
                object_by_reference = reference_module_globals.setdefault(reference_name, created_model)
                reference_name += '_'
        created_model.Config = cls.Config
        new_params = tuple({param: None for param in iter_contained_typevars(typevars_map.values())})
        created_model.__concrete__ = not new_params
        if new_params:
            created_model.__parameters__ = new_params
        _generic_types_cache[_cache_key(params)] = created_model
        if len(params) == 1:
            _generic_types_cache[_cache_key(params[0])] = created_model
        _prepare_model_fields(created_model, fields, instance_type_hints, typevars_map)
        return created_model

    @classmethod
    def __concrete_name__(cls: Type[Any], params: Tuple[Type[Any], ...]) -> str:
        """Compute class name for child classes.

        :param params: Tuple of types the class . Given a generic class
            `Model` with 2 type variables and a concrete model `Model[str, int]`,
            the value `(str, int)` would be passed to `params`.
        :return: String representing a the new class where `params` are
            passed to `cls` as type variables.

        This method can be overridden to achieve a custom naming scheme for GenericModels.
        """
        param_names = [display_as_type(param) for param in params]
        params_component = ', '.join(param_names)
        return f'{cls.__name__}[{params_component}]'

    @classmethod
    def __parameterized_bases__(cls, typevars_map: Parametrization) -> Iterator[Type[Any]]:
        """
        Returns unbound bases of cls parameterised to given type variables

        :param typevars_map: Dictionary of type applications for binding subclasses.
            Given a generic class `Model` with 2 type variables [S, T]
            and a concrete model `Model[str, int]`,
            the value `{S: str, T: int}` would be passed to `typevars_map`.
        :return: an iterator of generic sub classes, parameterised by `typevars_map`
            and other assigned parameters of `cls`

        e.g.:
        ```
        class A(GenericModel, Generic[T]):
            ...

        class B(A[V], Generic[V]):
            ...

        assert A[int] in B.__parameterized_bases__({V: int})
        ```
        """

        def build_base_model(base_model: Type[GenericModel], mapped_types: Parametrization) -> Iterator[Type[GenericModel]]:
            base_parameters = tuple((mapped_types[param] for param in base_model.__parameters__))
            parameterized_base = base_model.__class_getitem__(base_parameters)
            if parameterized_base is base_model or parameterized_base is cls:
                return
            yield parameterized_base
        for base_model in cls.__bases__:
            if not issubclass(base_model, GenericModel):
                continue
            elif not getattr(base_model, '__parameters__', None):
                continue
            elif cls in _assigned_parameters:
                if base_model in _assigned_parameters:
                    continue
                else:
                    mapped_types: Parametrization = {key: typevars_map.get(value, value) for key, value in _assigned_parameters[cls].items()}
                    yield from build_base_model(base_model, mapped_types)
            else:
                yield from build_base_model(base_model, typevars_map)

def replace_types(type_: Any, type_map: Mapping[Any, Any]) -> Any:
    """Return type with all occurrences of `type_map` keys recursively replaced with their values.

    :param type_: Any type, class or generic alias
    :param type_map: Mapping from `TypeVar` instance to concrete types.
    :return: New type representing the basic structure of `type_` with all
        `typevar_map` keys recursively replaced.

    >>> replace_types(Tuple[str, Union[List[str], float]], {str: int})
    Tuple[int, Union[List[int], float]]

    """
    if isinstance(type_, TypeVar):
        return type_map.get(type_, type_)
    
    origin = get_origin(type_)
    if origin is None:
        return type_
    
    args = get_args(type_)
    if not args:
        return type_
    
    new_args = tuple(replace_types(arg, type_map) for arg in args)
    if new_args == args:
        return type_
    
    return origin[new_args]
DictValues: Type[Any] = {}.values().__class__

def iter_contained_typevars(v: Any) -> Iterator[TypeVarType]:
    """Recursively iterate through all subtypes and type args of `v` and yield any typevars that are found."""
    if isinstance(v, TypeVar):
        yield v
    elif isinstance(v, (GenericAlias, _GenericAlias)):
        for arg in get_args(v):
            yield from iter_contained_typevars(arg)
    elif hasattr(v, '__parameters__'):
        for param in v.__parameters__:
            yield from iter_contained_typevars(param)

def get_caller_frame_info() -> Tuple[Optional[str], bool]:
    """
    Used inside a function to check whether it was called globally

    Will only work against non-compiled code, therefore used only in pydantic.generics

    :returns Tuple[module_name, called_globally]
    """
    try:
        frame = sys._getframe(2)
    except ValueError:
        return None, False
    
    module_name = frame.f_globals.get('__name__')
    if module_name == '__main__':
        return None, True
    
    caller_module = sys.modules.get(module_name)
    return (
        module_name,
        caller_module is not None and frame.f_globals is caller_module.__dict__
    )

def _prepare_model_fields(created_model: Type[GenericModel], fields: Mapping[str, Any], instance_type_hints: Mapping[str, type], typevars_map: Mapping[Any, type]) -> None:
    """
    Replace DeferredType fields with concrete type hints and prepare them.
    """
    for name, field in fields.items():
        if not isinstance(field, ModelField):
            continue
        
        field.type_ = replace_types(field.type_, typevars_map)
        field.prepare()
        
        if name in instance_type_hints:
            field.outer_type_ = replace_types(instance_type_hints[name], typevars_map)
        
        if field.sub_fields:
            _prepare_model_fields(created_model, {i: f for i, f in enumerate(field.sub_fields)}, {}, typevars_map)
    
    created_model.__fields__ = fields
