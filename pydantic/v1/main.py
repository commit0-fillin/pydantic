import warnings
from abc import ABCMeta
from copy import deepcopy
from enum import Enum
from functools import partial
from pathlib import Path
from types import FunctionType, prepare_class, resolve_bases
from typing import TYPE_CHECKING, AbstractSet, Any, Callable, ClassVar, Dict, List, Mapping, Optional, Tuple, Type, TypeVar, Union, cast, no_type_check, overload
from typing_extensions import dataclass_transform
from pydantic.v1.class_validators import ValidatorGroup, extract_root_validators, extract_validators, inherit_validators
from pydantic.v1.config import BaseConfig, Extra, inherit_config, prepare_config
from pydantic.v1.error_wrappers import ErrorWrapper, ValidationError
from pydantic.v1.errors import ConfigError, DictError, ExtraError, MissingError
from pydantic.v1.fields import MAPPING_LIKE_SHAPES, Field, ModelField, ModelPrivateAttr, PrivateAttr, Undefined, is_finalvar_with_default_val
from pydantic.v1.json import custom_pydantic_encoder, pydantic_encoder
from pydantic.v1.parse import Protocol, load_file, load_str_bytes
from pydantic.v1.schema import default_ref_template, model_schema
from pydantic.v1.types import PyObject, StrBytes
from pydantic.v1.typing import AnyCallable, get_args, get_origin, is_classvar, is_namedtuple, is_union, resolve_annotations, update_model_forward_refs
from pydantic.v1.utils import DUNDER_ATTRIBUTES, ROOT_KEY, ClassAttribute, GetterDict, Representation, ValueItems, generate_model_signature, is_valid_field, is_valid_private_name, lenient_issubclass, sequence_like, smart_deepcopy, unique_list, validate_field_name
if TYPE_CHECKING:
    from inspect import Signature
    from pydantic.v1.class_validators import ValidatorListDict
    from pydantic.v1.types import ModelOrDc
    from pydantic.v1.typing import AbstractSetIntStr, AnyClassMethod, CallableGenerator, DictAny, DictStrAny, MappingIntStrAny, ReprArgs, SetStr, TupleGenerator
    Model = TypeVar('Model', bound='BaseModel')
__all__ = ('BaseModel', 'create_model', 'validate_model')
_T = TypeVar('_T')
ANNOTATED_FIELD_UNTOUCHED_TYPES: Tuple[Any, ...] = (property, type, classmethod, staticmethod)
UNTOUCHED_TYPES: Tuple[Any, ...] = (FunctionType,) + ANNOTATED_FIELD_UNTOUCHED_TYPES
_is_base_model_class_defined = False

@dataclass_transform(kw_only_default=True, field_specifiers=(Field,))
class ModelMetaclass(ABCMeta):

    @no_type_check
    def __new__(mcs, name, bases, namespace, **kwargs):
        fields: Dict[str, ModelField] = {}
        config = BaseConfig
        validators: 'ValidatorListDict' = {}
        pre_root_validators, post_root_validators = ([], [])
        private_attributes: Dict[str, ModelPrivateAttr] = {}
        base_private_attributes: Dict[str, ModelPrivateAttr] = {}
        slots: SetStr = namespace.get('__slots__', ())
        slots = {slots} if isinstance(slots, str) else set(slots)
        class_vars: SetStr = set()
        hash_func: Optional[Callable[[Any], int]] = None
        for base in reversed(bases):
            if _is_base_model_class_defined and issubclass(base, BaseModel) and (base != BaseModel):
                fields.update(smart_deepcopy(base.__fields__))
                config = inherit_config(base.__config__, config)
                validators = inherit_validators(base.__validators__, validators)
                pre_root_validators += base.__pre_root_validators__
                post_root_validators += base.__post_root_validators__
                base_private_attributes.update(base.__private_attributes__)
                class_vars.update(base.__class_vars__)
                hash_func = base.__hash__
        resolve_forward_refs = kwargs.pop('__resolve_forward_refs__', True)
        allowed_config_kwargs: SetStr = {key for key in dir(config) if not (key.startswith('__') and key.endswith('__'))}
        config_kwargs = {key: kwargs.pop(key) for key in kwargs.keys() & allowed_config_kwargs}
        config_from_namespace = namespace.get('Config')
        if config_kwargs and config_from_namespace:
            raise TypeError('Specifying config in two places is ambiguous, use either Config attribute or class kwargs')
        config = inherit_config(config_from_namespace, config, **config_kwargs)
        validators = inherit_validators(extract_validators(namespace), validators)
        vg = ValidatorGroup(validators)
        for f in fields.values():
            f.set_config(config)
            extra_validators = vg.get_validators(f.name)
            if extra_validators:
                f.class_validators.update(extra_validators)
                f.populate_validators()
        prepare_config(config, name)
        untouched_types = ANNOTATED_FIELD_UNTOUCHED_TYPES

        def is_untouched(v: Any) -> bool:
            return isinstance(v, untouched_types) or v.__class__.__name__ == 'cython_function_or_method'
        if (namespace.get('__module__'), namespace.get('__qualname__')) != ('pydantic.main', 'BaseModel'):
            annotations = resolve_annotations(namespace.get('__annotations__', {}), namespace.get('__module__', None))
            for ann_name, ann_type in annotations.items():
                if is_classvar(ann_type):
                    class_vars.add(ann_name)
                elif is_finalvar_with_default_val(ann_type, namespace.get(ann_name, Undefined)):
                    class_vars.add(ann_name)
                elif is_valid_field(ann_name):
                    validate_field_name(bases, ann_name)
                    value = namespace.get(ann_name, Undefined)
                    allowed_types = get_args(ann_type) if is_union(get_origin(ann_type)) else (ann_type,)
                    if is_untouched(value) and ann_type != PyObject and (not any((lenient_issubclass(get_origin(allowed_type), Type) for allowed_type in allowed_types))):
                        continue
                    fields[ann_name] = ModelField.infer(name=ann_name, value=value, annotation=ann_type, class_validators=vg.get_validators(ann_name), config=config)
                elif ann_name not in namespace and config.underscore_attrs_are_private:
                    private_attributes[ann_name] = PrivateAttr()
            untouched_types = UNTOUCHED_TYPES + config.keep_untouched
            for var_name, value in namespace.items():
                can_be_changed = var_name not in class_vars and (not is_untouched(value))
                if isinstance(value, ModelPrivateAttr):
                    if not is_valid_private_name(var_name):
                        raise NameError(f'Private attributes "{var_name}" must not be a valid field name; Use sunder or dunder names, e. g. "_{var_name}" or "__{var_name}__"')
                    private_attributes[var_name] = value
                elif config.underscore_attrs_are_private and is_valid_private_name(var_name) and can_be_changed:
                    private_attributes[var_name] = PrivateAttr(default=value)
                elif is_valid_field(var_name) and var_name not in annotations and can_be_changed:
                    validate_field_name(bases, var_name)
                    inferred = ModelField.infer(name=var_name, value=value, annotation=annotations.get(var_name, Undefined), class_validators=vg.get_validators(var_name), config=config)
                    if var_name in fields:
                        if lenient_issubclass(inferred.type_, fields[var_name].type_):
                            inferred.type_ = fields[var_name].type_
                        else:
                            raise TypeError(f'The type of {name}.{var_name} differs from the new default value; if you wish to change the type of this field, please use a type annotation')
                    fields[var_name] = inferred
        _custom_root_type = ROOT_KEY in fields
        if _custom_root_type:
            validate_custom_root_type(fields)
        vg.check_for_unused()
        if config.json_encoders:
            json_encoder = partial(custom_pydantic_encoder, config.json_encoders)
        else:
            json_encoder = pydantic_encoder
        pre_rv_new, post_rv_new = extract_root_validators(namespace)
        if hash_func is None:
            hash_func = generate_hash_function(config.frozen)
        exclude_from_namespace = fields | private_attributes.keys() | {'__slots__'}
        new_namespace = {'__config__': config, '__fields__': fields, '__exclude_fields__': {name: field.field_info.exclude for name, field in fields.items() if field.field_info.exclude is not None} or None, '__include_fields__': {name: field.field_info.include for name, field in fields.items() if field.field_info.include is not None} or None, '__validators__': vg.validators, '__pre_root_validators__': unique_list(pre_root_validators + pre_rv_new, name_factory=lambda v: v.__name__), '__post_root_validators__': unique_list(post_root_validators + post_rv_new, name_factory=lambda skip_on_failure_and_v: skip_on_failure_and_v[1].__name__), '__schema_cache__': {}, '__json_encoder__': staticmethod(json_encoder), '__custom_root_type__': _custom_root_type, '__private_attributes__': {**base_private_attributes, **private_attributes}, '__slots__': slots | private_attributes.keys(), '__hash__': hash_func, '__class_vars__': class_vars, **{n: v for n, v in namespace.items() if n not in exclude_from_namespace}}
        cls = super().__new__(mcs, name, bases, new_namespace, **kwargs)
        cls.__signature__ = ClassAttribute('__signature__', generate_model_signature(cls.__init__, fields, config))
        if resolve_forward_refs:
            cls.__try_update_forward_refs__()
        for name, obj in namespace.items():
            if name not in new_namespace:
                set_name = getattr(obj, '__set_name__', None)
                if callable(set_name):
                    set_name(cls, name)
        return cls

    def __instancecheck__(self, instance: Any) -> bool:
        """
        Avoid calling ABC _abc_subclasscheck unless we're pretty sure.

        See #3829 and python/cpython#92810
        """
        return hasattr(instance, '__fields__') and super().__instancecheck__(instance)
object_setattr = object.__setattr__

class BaseModel(Representation, metaclass=ModelMetaclass):
    if TYPE_CHECKING:
        __fields__: ClassVar[Dict[str, ModelField]] = {}
        __include_fields__: ClassVar[Optional[Mapping[str, Any]]] = None
        __exclude_fields__: ClassVar[Optional[Mapping[str, Any]]] = None
        __validators__: ClassVar[Dict[str, AnyCallable]] = {}
        __pre_root_validators__: ClassVar[List[AnyCallable]]
        __post_root_validators__: ClassVar[List[Tuple[bool, AnyCallable]]]
        __config__: ClassVar[Type[BaseConfig]] = BaseConfig
        __json_encoder__: ClassVar[Callable[[Any], Any]] = lambda x: x
        __schema_cache__: ClassVar['DictAny'] = {}
        __custom_root_type__: ClassVar[bool] = False
        __signature__: ClassVar['Signature']
        __private_attributes__: ClassVar[Dict[str, ModelPrivateAttr]]
        __class_vars__: ClassVar[SetStr]
        __fields_set__: ClassVar[SetStr] = set()
    Config = BaseConfig
    __slots__ = ('__dict__', '__fields_set__')
    __doc__ = ''

    def __init__(__pydantic_self__, **data: Any) -> None:
        """
        Create a new model by parsing and validating input data from keyword arguments.

        Raises ValidationError if the input data cannot be parsed to form a valid model.
        """
        values, fields_set, validation_error = validate_model(__pydantic_self__.__class__, data)
        if validation_error:
            raise validation_error
        try:
            object_setattr(__pydantic_self__, '__dict__', values)
        except TypeError as e:
            raise TypeError('Model values must be a dict; you may not have returned a dictionary from a root validator') from e
        object_setattr(__pydantic_self__, '__fields_set__', fields_set)
        __pydantic_self__._init_private_attributes()

    @no_type_check
    def __setattr__(self, name, value):
        if name in self.__private_attributes__ or name in DUNDER_ATTRIBUTES:
            return object_setattr(self, name, value)
        if self.__config__.extra is not Extra.allow and name not in self.__fields__:
            raise ValueError(f'"{self.__class__.__name__}" object has no field "{name}"')
        elif not self.__config__.allow_mutation or self.__config__.frozen:
            raise TypeError(f'"{self.__class__.__name__}" is immutable and does not support item assignment')
        elif name in self.__fields__ and self.__fields__[name].final:
            raise TypeError(f'"{self.__class__.__name__}" object "{name}" field is final and does not support reassignment')
        elif self.__config__.validate_assignment:
            new_values = {**self.__dict__, name: value}
            for validator in self.__pre_root_validators__:
                try:
                    new_values = validator(self.__class__, new_values)
                except (ValueError, TypeError, AssertionError) as exc:
                    raise ValidationError([ErrorWrapper(exc, loc=ROOT_KEY)], self.__class__)
            known_field = self.__fields__.get(name, None)
            if known_field:
                if not known_field.field_info.allow_mutation:
                    raise TypeError(f'"{known_field.name}" has allow_mutation set to False and cannot be assigned')
                dict_without_original_value = {k: v for k, v in self.__dict__.items() if k != name}
                value, error_ = known_field.validate(value, dict_without_original_value, loc=name, cls=self.__class__)
                if error_:
                    raise ValidationError([error_], self.__class__)
                else:
                    new_values[name] = value
            errors = []
            for skip_on_failure, validator in self.__post_root_validators__:
                if skip_on_failure and errors:
                    continue
                try:
                    new_values = validator(self.__class__, new_values)
                except (ValueError, TypeError, AssertionError) as exc:
                    errors.append(ErrorWrapper(exc, loc=ROOT_KEY))
            if errors:
                raise ValidationError(errors, self.__class__)
            object_setattr(self, '__dict__', new_values)
        else:
            self.__dict__[name] = value
        self.__fields_set__.add(name)

    def __getstate__(self) -> 'DictAny':
        private_attrs = ((k, getattr(self, k, Undefined)) for k in self.__private_attributes__)
        return {'__dict__': self.__dict__, '__fields_set__': self.__fields_set__, '__private_attribute_values__': {k: v for k, v in private_attrs if v is not Undefined}}

    def __setstate__(self, state: 'DictAny') -> None:
        object_setattr(self, '__dict__', state['__dict__'])
        object_setattr(self, '__fields_set__', state['__fields_set__'])
        for name, value in state.get('__private_attribute_values__', {}).items():
            object_setattr(self, name, value)

    def dict(self, *, include: Optional[Union['AbstractSetIntStr', 'MappingIntStrAny']]=None, exclude: Optional[Union['AbstractSetIntStr', 'MappingIntStrAny']]=None, by_alias: bool=False, skip_defaults: Optional[bool]=None, exclude_unset: bool=False, exclude_defaults: bool=False, exclude_none: bool=False) -> 'DictStrAny':
        """
        Generate a dictionary representation of the model, optionally specifying which fields to include or exclude.

        """
        return dict_func(
            self,
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            skip_defaults=skip_defaults,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
        )

    def json(self, *, include: Optional[Union['AbstractSetIntStr', 'MappingIntStrAny']]=None, exclude: Optional[Union['AbstractSetIntStr', 'MappingIntStrAny']]=None, by_alias: bool=False, skip_defaults: Optional[bool]=None, exclude_unset: bool=False, exclude_defaults: bool=False, exclude_none: bool=False, encoder: Optional[Callable[[Any], Any]]=None, models_as_dict: bool=True, **dumps_kwargs: Any) -> str:
        """
        Generate a JSON representation of the model, `include` and `exclude` arguments as per `dict()`.

        `encoder` is an optional function to supply as `default` to json.dumps(), other arguments as per `json.dumps()`.
        """
        return json_func(
            self,
            include=include,
            exclude=exclude,
            by_alias=by_alias,
            skip_defaults=skip_defaults,
            exclude_unset=exclude_unset,
            exclude_defaults=exclude_defaults,
            exclude_none=exclude_none,
            encoder=encoder,
            models_as_dict=models_as_dict,
            **dumps_kwargs
        )

    @classmethod
    def construct(cls: Type['Model'], _fields_set: Optional['SetStr']=None, **values: Any) -> 'Model':
        """
        Creates a new model setting __dict__ and __fields_set__ from trusted or pre-validated data.
        Default values are respected, but no other validation is performed.
        Behaves as if `Config.extra = 'allow'` was set since it adds all passed values
        """
        m = cls.__new__(cls)
        fields_values = {}
        for name, field in cls.__fields__.items():
            if name in values:
                fields_values[name] = values[name]
            elif field.default is not Undefined:
                fields_values[name] = field.default
            elif field.default_factory is not None:
                fields_values[name] = field.default_factory()
            else:
                fields_values[name] = None
        
        fields_values.update({k: v for k, v in values.items() if k not in cls.__fields__})
        object_setattr(m, '__dict__', fields_values)
        if _fields_set is None:
            _fields_set = set(values.keys())
        object_setattr(m, '__fields_set__', _fields_set)
        m._init_private_attributes()
        return m

    def copy(self: 'Model', *, include: Optional[Union['AbstractSetIntStr', 'MappingIntStrAny']]=None, exclude: Optional[Union['AbstractSetIntStr', 'MappingIntStrAny']]=None, update: Optional['DictStrAny']=None, deep: bool=False) -> 'Model':
        """
        Duplicate a model, optionally choose which fields to include, exclude and change.

        :param include: fields to include in new model
        :param exclude: fields to exclude from new model, as with values this takes precedence over include
        :param update: values to change/add in the new model. Note: the data is not validated before creating
            the new model: you should trust this data
        :param deep: set to `True` to make a deep copy of the model
        :return: new model instance
        """
        values = dict_func(
            self,
            include=include,
            exclude=exclude,
            exclude_unset=False,
            exclude_defaults=False,
        )

        if update:
            values.update(update)

        if deep:
            # Use deepcopy for nested structures
            values = deepcopy(values)

        # Use the construct method to create a new instance
        return self.__class__.construct(_fields_set=self.__fields_set__.copy(), **values)

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield cls.validate

    @classmethod
    def __try_update_forward_refs__(cls, **localns: Any) -> None:
        """
        Same as update_forward_refs but will not raise exception
        when forward references are not defined.
        """
        update_model_forward_refs(cls, cls.__fields__.values(), cls.__config__.json_encoders, localns, (NameError,))

    @classmethod
    def update_forward_refs(cls, **localns: Any) -> None:
        """
        Try to update ForwardRefs on fields based on this Model, globalns and localns.
        """
        globalns = sys.modules[cls.__module__].__dict__.copy()
        globalns.setdefault(cls.__name__, cls)
        for field in cls.__fields__.values():
            update_field_forward_refs(field, globalns=globalns, localns=localns)

    def __iter__(self) -> 'TupleGenerator':
        """
        so `dict(model)` works
        """
        yield from self.__dict__.items()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, BaseModel):
            return self.dict() == other.dict()
        else:
            return self.dict() == other

    def __repr_args__(self) -> 'ReprArgs':
        return [(k, v) for k, v in self.__dict__.items() if k not in DUNDER_ATTRIBUTES and (k not in self.__fields__ or self.__fields__[k].field_info.repr)]
_is_base_model_class_defined = True

def create_model(__model_name: str, *, __config__: Optional[Type[BaseConfig]]=None, __base__: Union[None, Type['Model'], Tuple[Type['Model'], ...]]=None, __module__: str=__name__, __validators__: Dict[str, 'AnyClassMethod']=None, __cls_kwargs__: Dict[str, Any]=None, __slots__: Optional[Tuple[str, ...]]=None, **field_definitions: Any) -> Type['Model']:
    """
    Dynamically create a model.
    :param __model_name: name of the created model
    :param __config__: config class to use for the new model
    :param __base__: base class for the new model to inherit from
    :param __module__: module of the created model
    :param __validators__: a dict of method names and @validator class methods
    :param __cls_kwargs__: a dict for class creation
    :param __slots__: Deprecated, `__slots__` should not be passed to `create_model`
    :param field_definitions: fields of the model (or extra fields if a base is supplied)
        in the format `<name>=(<type>, <default default>)` or `<name>=<default value>, e.g.
        `foobar=(str, ...)` or `foobar=123`, or, for complex use-cases, in the format
        `<name>=<Field>` or `<name>=(<type>, <FieldInfo>)`, e.g.
        `foo=Field(datetime, default_factory=datetime.utcnow, alias='bar')` or
        `foo=(str, FieldInfo(title='Foo'))`
    """
    if __slots__ is not None:
        warnings.warn('__slots__ should not be passed to create_model', DeprecationWarning)

    fields = {}
    annotations = {}

    for f_name, f_def in field_definitions.items():
        if isinstance(f_def, tuple):
            try:
                f_annotation, f_value = f_def
            except ValueError:
                raise ValueError(f'field {f_name} cannot have more than two components')
        else:
            f_annotation, f_value = None, f_def

        if f_annotation:
            annotations[f_name] = f_annotation
        fields[f_name] = f_value

    namespace = {
        '__annotations__': annotations,
        '__module__': __module__,
    }
    if __base__ is not None:
        namespace['__base__'] = __base__
    if __config__ is not None:
        namespace['Config'] = __config__
    if __validators__:
        namespace.update(__validators__)
    namespace.update(fields)

    if __cls_kwargs__ is None:
        __cls_kwargs__ = {}

    return ModelMetaclass(__model_name, (__base__,) if __base__ else (BaseModel,), namespace, **__cls_kwargs__)
_missing = object()

def validate_model(model: Type[BaseModel], input_data: 'DictStrAny', cls: 'ModelOrDc'=None) -> Tuple['DictStrAny', 'SetStr', Optional[ValidationError]]:
    """
    validate data against a model.
    """
    values = {}
    fields_set = set()
    errors = []
    
    for name, field in model.__fields__.items():
        value = input_data.get(field.alias, _missing)
        if value is _missing:
            if field.required:
                errors.append(ErrorWrapper(MissingError(), loc=field.alias))
            else:
                value = field.get_default()
        
        if value is not _missing:
            try:
                value, error = field.validate(value, values, loc=field.alias, cls=cls or model)
                if error:
                    errors.append(error)
                else:
                    values[name] = value
                    fields_set.add(name)
            except ValidationError as e:
                errors.extend(e.raw_errors)

    if model.__config__.extra == Extra.allow:
        for name, value in input_data.items():
            if name not in model.__fields__:
                values[name] = value
                fields_set.add(name)
    elif model.__config__.extra == Extra.ignore:
        pass
    else:  # Extra.forbid
        for name in input_data:
            if name not in model.__fields__:
                errors.append(ErrorWrapper(ExtraError(), loc=name))

    if errors:
        return {}, set(), ValidationError(errors, model)
    else:
        return values, fields_set, None
