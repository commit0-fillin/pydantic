import warnings
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Optional, Tuple, Type, TypeVar, Union, overload
from typing_extensions import deprecated
from .._internal import _config, _typing_extra
from ..alias_generators import to_pascal
from ..errors import PydanticUserError
from ..functional_validators import field_validator
from ..main import BaseModel, create_model
from ..warnings import PydanticDeprecatedSince20
if not TYPE_CHECKING:
    DeprecationWarning = PydanticDeprecatedSince20
__all__ = ('validate_arguments',)
if TYPE_CHECKING:
    AnyCallable = Callable[..., Any]
    AnyCallableT = TypeVar('AnyCallableT', bound=AnyCallable)
    ConfigType = Union[None, Type[Any], Dict[str, Any]]

@deprecated('The `validate_arguments` method is deprecated; use `validate_call` instead.', category=None)
def validate_arguments(func: Optional['AnyCallableT']=None, *, config: 'ConfigType'=None) -> Any:
    """Decorator to validate the arguments passed to a function."""
    def decorator(f: 'AnyCallableT') -> 'AnyCallableT':
        validated_func = ValidatedFunction(f, config)
        @wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return validated_func.call(*args, **kwargs)
        return cast('AnyCallableT', wrapper)

    if func:
        return decorator(func)
    return decorator
ALT_V_ARGS = 'v__args'
ALT_V_KWARGS = 'v__kwargs'
V_POSITIONAL_ONLY_NAME = 'v__positional_only'
V_DUPLICATE_KWARGS = 'v__duplicate_kwargs'

class ValidatedFunction:

    def __init__(self, function: 'AnyCallable', config: 'ConfigType'):
        from inspect import Parameter, signature
        parameters: Mapping[str, Parameter] = signature(function).parameters
        if parameters.keys() & {ALT_V_ARGS, ALT_V_KWARGS, V_POSITIONAL_ONLY_NAME, V_DUPLICATE_KWARGS}:
            raise PydanticUserError(f'"{ALT_V_ARGS}", "{ALT_V_KWARGS}", "{V_POSITIONAL_ONLY_NAME}" and "{V_DUPLICATE_KWARGS}" are not permitted as argument names when using the "{validate_arguments.__name__}" decorator', code=None)
        self.raw_function = function
        self.arg_mapping: Dict[int, str] = {}
        self.positional_only_args: set[str] = set()
        self.v_args_name = 'args'
        self.v_kwargs_name = 'kwargs'
        type_hints = _typing_extra.get_type_hints(function, include_extras=True)
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
