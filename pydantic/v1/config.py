import json
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, ForwardRef, Optional, Tuple, Type, Union
from typing_extensions import Literal, Protocol
from pydantic.v1.typing import AnyArgTCallable, AnyCallable
from pydantic.v1.utils import GetterDict
from pydantic.v1.version import compiled
if TYPE_CHECKING:
    from typing import overload
    from pydantic.v1.fields import ModelField
    from pydantic.v1.main import BaseModel
    ConfigType = Type['BaseConfig']

    class SchemaExtraCallable(Protocol):

        @overload
        def __call__(self, schema: Dict[str, Any]) -> None:
            """
            Update the schema dictionary with extra information.
            """
            pass

        @overload
        def __call__(self, schema: Dict[str, Any], model_class: Type[BaseModel]) -> None:
            """
            Update the schema dictionary with extra information, including the model class.
            """
            pass
else:
    SchemaExtraCallable = Callable[..., None]
__all__ = ('BaseConfig', 'ConfigDict', 'get_config', 'Extra', 'inherit_config', 'prepare_config')

class Extra(str, Enum):
    allow = 'allow'
    ignore = 'ignore'
    forbid = 'forbid'
if not compiled:
    from typing_extensions import TypedDict

    class ConfigDict(TypedDict, total=False):
        title: Optional[str]
        anystr_lower: bool
        anystr_strip_whitespace: bool
        min_anystr_length: int
        max_anystr_length: Optional[int]
        validate_all: bool
        extra: Extra
        allow_mutation: bool
        frozen: bool
        allow_population_by_field_name: bool
        use_enum_values: bool
        fields: Dict[str, Union[str, Dict[str, str]]]
        validate_assignment: bool
        error_msg_templates: Dict[str, str]
        arbitrary_types_allowed: bool
        orm_mode: bool
        getter_dict: Type[GetterDict]
        alias_generator: Optional[Callable[[str], str]]
        keep_untouched: Tuple[type, ...]
        schema_extra: Union[Dict[str, object], 'SchemaExtraCallable']
        json_loads: Callable[[str], object]
        json_dumps: AnyArgTCallable[str]
        json_encoders: Dict[Type[object], AnyCallable]
        underscore_attrs_are_private: bool
        allow_inf_nan: bool
        copy_on_model_validation: Literal['none', 'deep', 'shallow']
        post_init_call: Literal['before_validation', 'after_validation']
else:
    ConfigDict = dict

class BaseConfig:
    title: Optional[str] = None
    anystr_lower: bool = False
    anystr_upper: bool = False
    anystr_strip_whitespace: bool = False
    min_anystr_length: int = 0
    max_anystr_length: Optional[int] = None
    validate_all: bool = False
    extra: Extra = Extra.ignore
    allow_mutation: bool = True
    frozen: bool = False
    allow_population_by_field_name: bool = False
    use_enum_values: bool = False
    fields: Dict[str, Union[str, Dict[str, str]]] = {}
    validate_assignment: bool = False
    error_msg_templates: Dict[str, str] = {}
    arbitrary_types_allowed: bool = False
    orm_mode: bool = False
    getter_dict: Type[GetterDict] = GetterDict
    alias_generator: Optional[Callable[[str], str]] = None
    keep_untouched: Tuple[type, ...] = ()
    schema_extra: Union[Dict[str, Any], 'SchemaExtraCallable'] = {}
    json_loads: Callable[[str], Any] = json.loads
    json_dumps: Callable[..., str] = json.dumps
    json_encoders: Dict[Union[Type[Any], str, ForwardRef], AnyCallable] = {}
    underscore_attrs_are_private: bool = False
    allow_inf_nan: bool = True
    copy_on_model_validation: Literal['none', 'deep', 'shallow'] = 'shallow'
    smart_union: bool = False
    post_init_call: Literal['before_validation', 'after_validation'] = 'before_validation'

    @classmethod
    def get_field_info(cls, name: str) -> Dict[str, Any]:
        """
        Get properties of FieldInfo from the `fields` property of the config class.
        """
        pass

    @classmethod
    def prepare_field(cls, field: 'ModelField') -> None:
        """
        Optional hook to check or modify fields during model creation.
        """
        pass
