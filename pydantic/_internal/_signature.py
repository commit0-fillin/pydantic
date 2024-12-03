from __future__ import annotations
import dataclasses
from inspect import Parameter, Signature, signature
from typing import TYPE_CHECKING, Any, Callable
from pydantic_core import PydanticUndefined
from ._config import ConfigWrapper
from ._utils import is_valid_identifier
if TYPE_CHECKING:
    from ..fields import FieldInfo

def _field_name_for_signature(field_name: str, field_info: FieldInfo) -> str:
    """Extract the correct name to use for the field when generating a signature.

    Assuming the field has a valid alias, this will return the alias. Otherwise, it will return the field name.
    First priority is given to the validation_alias, then the alias, then the field name.

    Args:
        field_name: The name of the field
        field_info: The corresponding FieldInfo object.

    Returns:
        The correct name to use when generating a signature.
    """
    if isinstance(field_info.validation_alias, str):
        return field_info.validation_alias
    elif field_info.alias:
        return field_info.alias
    return field_name

def _process_param_defaults(param: Parameter) -> Parameter:
    """Modify the signature for a parameter in a dataclass where the default value is a FieldInfo instance.

    Args:
        param (Parameter): The parameter

    Returns:
        Parameter: The custom processed parameter
    """
    if isinstance(param.default, FieldInfo):
        default = param.default.default
        if default is PydanticUndefined:
            default = Parameter.empty
        return param.replace(default=default)
    return param

def _generate_signature_parameters(init: Callable[..., None], fields: dict[str, FieldInfo], config_wrapper: ConfigWrapper) -> dict[str, Parameter]:
    """Generate a mapping of parameter names to Parameter objects for a pydantic BaseModel or dataclass."""
    signature = signature(init)
    parameters = {}

    for name, param in signature.parameters.items():
        if name == 'self':
            continue

        if name in fields:
            field_info = fields[name]
            param_name = _field_name_for_signature(name, field_info)
            default = field_info.get_default()
            if default is PydanticUndefined:
                default = Parameter.empty
            annotation = field_info.annotation if field_info.annotation is not None else param.annotation
            parameters[param_name] = Parameter(
                param_name,
                kind=param.kind,
                default=default,
                annotation=annotation
            )
        else:
            parameters[name] = param

    return parameters

def generate_pydantic_signature(init: Callable[..., None], fields: dict[str, FieldInfo], config_wrapper: ConfigWrapper, is_dataclass: bool=False) -> Signature:
    """Generate signature for a pydantic BaseModel or dataclass.

    Args:
        init: The class init.
        fields: The model fields.
        config_wrapper: The config wrapper instance.
        is_dataclass: Whether the model is a dataclass.

    Returns:
        The dataclass/BaseModel subclass signature.
    """
    parameters = _generate_signature_parameters(init, fields, config_wrapper)
    
    if is_dataclass:
        parameters = {name: _process_param_defaults(param) for name, param in parameters.items()}
    
    return Signature(list(parameters.values()))
