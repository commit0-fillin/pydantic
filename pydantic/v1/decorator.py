from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, TypeVar, Union, overload
from pydantic.v1 import validator
from pydantic.v1.config import Extra
from pydantic.v1.errors import ConfigError
from pydantic.v1.main import BaseModel, create_model
from pydantic.v1.typing import get_all_type_hints
from pydantic.v1.utils import to_camel
__all__ = ('validate_arguments',)
if TYPE_CHECKING:
    from pydantic.v1.typing import AnyCallable
    AnyCallableT = TypeVar('AnyCallableT', bound=AnyCallable)
    ConfigType = Union[None, Type[Any], Dict[str, Any]]

def validate_arguments(func: Optional['AnyCallableT'] = None, *, config: 'ConfigType' = None) -> Any:
    """
    Decorator to validate the arguments passed to a function.
    """
    def decorator(f: 'AnyCallableT') -> 'AnyCallableT':
        validated_func = ValidatedFunction(f, config)
        @wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return validated_func.call(*args, **kwargs)
        wrapper.__validated_function__ = validated_func  # type: ignore
        return cast('AnyCallableT', wrapper)

    if func:
        return decorator(func)
    return decorator
ALT_V_ARGS = 'v__args'
ALT_V_KWARGS = 'v__kwargs'
V_POSITIONAL_ONLY_NAME = 'v__positional_only'
V_DUPLICATE_KWARGS = 'v__duplicate_kwargs'

class ValidatedFunction:
    def __init__(self, function: 'AnyCallableT', config: 'ConfigType'):
        from inspect import Parameter, signature
        parameters: Mapping[str, Parameter] = signature(function).parameters
        if parameters.keys() & {ALT_V_ARGS, ALT_V_KWARGS, V_POSITIONAL_ONLY_NAME, V_DUPLICATE_KWARGS}:
            raise ConfigError(f'"{ALT_V_ARGS}", "{ALT_V_KWARGS}", "{V_POSITIONAL_ONLY_NAME}" and "{V_DUPLICATE_KWARGS}" are not permitted as argument names when using the "{validate_arguments.__name__}" decorator')
        self.raw_function = function
        self.arg_mapping: Dict[int, str] = {}
        self.positional_only_args = set()
        self.v_args_name = 'args'
        self.v_kwargs_name = 'kwargs'
        type_hints = get_all_type_hints(function)
        takes_args = False
        takes_kwargs = False
        fields: Dict[str, Tuple[Any, Any]] = {}
        for i, (name, p) in enumerate(parameters.items()):
            if p.annotation is p.empty:
                annotation = Any
            else:
                annotation = type_hints[name]
            default = ... if p.default is p.empty else p.default
            if p.kind == Parameter.POSITIONAL_ONLY:
                self.arg_mapping[i] = name
                fields[name] = (annotation, default)
                fields[V_POSITIONAL_ONLY_NAME] = (List[str], None)
                self.positional_only_args.add(name)
            elif p.kind == Parameter.POSITIONAL_OR_KEYWORD:
                self.arg_mapping[i] = name
                fields[name] = (annotation, default)
                fields[V_DUPLICATE_KWARGS] = (List[str], None)
            elif p.kind == Parameter.KEYWORD_ONLY:
                fields[name] = (annotation, default)
            elif p.kind == Parameter.VAR_POSITIONAL:
                self.v_args_name = name
                fields[name] = (Tuple[annotation, ...], None)
                takes_args = True
            else:
                assert p.kind == Parameter.VAR_KEYWORD, p.kind
                self.v_kwargs_name = name
                fields[name] = (Dict[str, annotation], None)
                takes_kwargs = True
        if not takes_args and self.v_args_name in fields:
            self.v_args_name = ALT_V_ARGS
        if not takes_kwargs and self.v_kwargs_name in fields:
            self.v_kwargs_name = ALT_V_KWARGS
        if not takes_args:
            fields[self.v_args_name] = (List[Any], None)
        if not takes_kwargs:
            fields[self.v_kwargs_name] = (Dict[Any, Any], None)
        self.create_model(fields, takes_args, takes_kwargs, config)

    def create_model(self, fields: Dict[str, Tuple[Any, Any]], takes_args: bool, takes_kwargs: bool, config: 'ConfigType'):
        validators = {field_name: make_generic_validator(field_name) for field_name in fields}
        
        model_name = f'{self.raw_function.__name__}Model'
        model_module = self.raw_function.__module__
        
        self.model = create_model(
            model_name,
            __module__=model_module,
            __config__=prepare_config(config),
            __validators__=validators,
            **fields
        )

    def call(self, *args: Any, **kwargs: Any) -> Any:
        values = self.build_values(args, kwargs)
        m = self.model(**values)
        return self.execute(m)

    def build_values(self, args: Tuple[Any, ...], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        values: Dict[str, Any] = {}
        v_args = []
        for i, arg in enumerate(args):
            name = self.arg_mapping.get(i)
            if name:
                values[name] = arg
            else:
                v_args.append(arg)
        values[self.v_args_name] = tuple(v_args)
        
        duplicate_kwargs = []
        for k, v in kwargs.items():
            if k in values:
                duplicate_kwargs.append(k)
            else:
                values[k] = v
        values[self.v_kwargs_name] = kwargs
        values[V_DUPLICATE_KWARGS] = duplicate_kwargs
        return values

    def execute(self, m: BaseModel) -> Any:
        d = dict(m)
        var_kwargs = d.pop(self.v_kwargs_name, {})
        var_args = d.pop(self.v_args_name, ())
        d.pop(V_POSITIONAL_ONLY_NAME, None)
        d.pop(V_DUPLICATE_KWARGS, None)
        return self.raw_function(*var_args, **{**d, **var_kwargs})

def make_generic_validator(field_name: str) -> classmethod:
    @classmethod
    def generic_validator(cls, v: Any, values: Dict[str, Any], **kwargs: Any) -> Any:
        if field_name in {V_POSITIONAL_ONLY_NAME, V_DUPLICATE_KWARGS}:
            return v
        return v
    return generic_validator
