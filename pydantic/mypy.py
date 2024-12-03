"""This module includes classes and functions designed specifically for use with the mypy plugin."""
from __future__ import annotations
import sys
from configparser import ConfigParser
from typing import Any, Callable, Iterator
from mypy.errorcodes import ErrorCode
from mypy.expandtype import expand_type, expand_type_by_instance
from mypy.nodes import ARG_NAMED, ARG_NAMED_OPT, ARG_OPT, ARG_POS, ARG_STAR2, INVARIANT, MDEF, Argument, AssignmentStmt, Block, CallExpr, ClassDef, Context, Decorator, DictExpr, EllipsisExpr, Expression, FuncDef, IfStmt, JsonDict, MemberExpr, NameExpr, PassStmt, PlaceholderNode, RefExpr, Statement, StrExpr, SymbolTableNode, TempNode, TypeAlias, TypeInfo, Var
from mypy.options import Options
from mypy.plugin import CheckerPluginInterface, ClassDefContext, FunctionContext, MethodContext, Plugin, ReportConfigContext, SemanticAnalyzerPluginInterface
from mypy.plugins import dataclasses
from mypy.plugins.common import deserialize_and_fixup_type
from mypy.semanal import set_callable_name
from mypy.server.trigger import make_wildcard_trigger
from mypy.state import state
from mypy.typeops import map_type_from_supertype
from mypy.types import AnyType, CallableType, Instance, NoneType, Overloaded, Type, TypeOfAny, TypeType, TypeVarType, UnionType, get_proper_type
from mypy.typevars import fill_typevars
from mypy.util import get_unique_redefinition_name
from mypy.version import __version__ as mypy_version
from pydantic._internal import _fields
from pydantic.version import parse_mypy_version
try:
    from mypy.types import TypeVarDef
except ImportError:
    from mypy.types import TypeVarType as TypeVarDef
CONFIGFILE_KEY = 'pydantic-mypy'
METADATA_KEY = 'pydantic-mypy-metadata'
BASEMODEL_FULLNAME = 'pydantic.main.BaseModel'
BASESETTINGS_FULLNAME = 'pydantic_settings.main.BaseSettings'
ROOT_MODEL_FULLNAME = 'pydantic.root_model.RootModel'
MODEL_METACLASS_FULLNAME = 'pydantic._internal._model_construction.ModelMetaclass'
FIELD_FULLNAME = 'pydantic.fields.Field'
DATACLASS_FULLNAME = 'pydantic.dataclasses.dataclass'
MODEL_VALIDATOR_FULLNAME = 'pydantic.functional_validators.model_validator'
DECORATOR_FULLNAMES = {'pydantic.functional_validators.field_validator', 'pydantic.functional_validators.model_validator', 'pydantic.functional_serializers.serializer', 'pydantic.functional_serializers.model_serializer', 'pydantic.deprecated.class_validators.validator', 'pydantic.deprecated.class_validators.root_validator'}
MYPY_VERSION_TUPLE = parse_mypy_version(mypy_version)
BUILTINS_NAME = 'builtins' if MYPY_VERSION_TUPLE >= (0, 930) else '__builtins__'
__version__ = 2

def plugin(version: str) -> type[Plugin]:
    """`version` is the mypy version string.

    We might want to use this to print a warning if the mypy version being used is
    newer, or especially older, than we expect (or need).

    Args:
        version: The mypy version string.

    Return:
        The Pydantic mypy plugin type.
    """
    parsed_version = parse_mypy_version(version)
    if parsed_version < (0, 930):
        warnings.warn(f"Pydantic mypy plugin requires mypy version >= 0.930, found {version}")
    return PydanticPlugin

class PydanticPlugin(Plugin):
    """The Pydantic mypy plugin."""

    def __init__(self, options: Options) -> None:
        self.plugin_config = PydanticPluginConfig(options)
        self._plugin_data = self.plugin_config.to_data()
        super().__init__(options)

    def get_base_class_hook(self, fullname: str) -> Callable[[ClassDefContext], bool] | None:
        """Update Pydantic model class."""
        if fullname in (BASEMODEL_FULLNAME, BASESETTINGS_FULLNAME):
            return self._pydantic_model_class_maker_callback
        return None

    def get_metaclass_hook(self, fullname: str) -> Callable[[ClassDefContext], None] | None:
        """Update Pydantic `ModelMetaclass` definition."""
        if fullname == MODEL_METACLASS_FULLNAME:
            return self._pydantic_model_metaclass_marker_callback
        return None

    def get_function_hook(self, fullname: str) -> Callable[[FunctionContext], Type] | None:
        """Adjust the return type of the `Field` function."""
        if fullname == FIELD_FULLNAME:
            return self._pydantic_field_callback
        return None

    def get_method_hook(self, fullname: str) -> Callable[[MethodContext], Type] | None:
        """Adjust return type of `from_orm` method call."""
        if fullname.endswith('.from_attributes'):
            return from_attributes_callback
        return None

    def get_class_decorator_hook(self, fullname: str) -> Callable[[ClassDefContext], None] | None:
        """Mark pydantic.dataclasses as dataclass.

        Mypy version 1.1.1 added support for `@dataclass_transform` decorator.
        """
        if fullname == DATACLASS_FULLNAME:
            return dataclasses.dataclass_class_maker_callback
        return None

    def report_config_data(self, ctx: ReportConfigContext) -> dict[str, Any]:
        """Return all plugin config data.

        Used by mypy to determine if cache needs to be discarded.
        """
        pass

    def _pydantic_model_metaclass_marker_callback(self, ctx: ClassDefContext) -> None:
        """Reset dataclass_transform_spec attribute of ModelMetaclass.

        Let the plugin handle it. This behavior can be disabled
        if 'debug_dataclass_transform' is set to True', for testing purposes.
        """
        pass

    def _pydantic_field_callback(self, ctx: FunctionContext) -> Type:
        """Extract the type of the `default` argument from the Field function, and use it as the return type.

        In particular:
        * Check whether the default and default_factory argument is specified.
        * Output an error if both are specified.
        * Retrieve the type of the argument which is specified, and use it as return type for the function.
        """
        pass

class PydanticPluginConfig:
    """A Pydantic mypy plugin config holder.

    Attributes:
        init_forbid_extra: Whether to add a `**kwargs` at the end of the generated `__init__` signature.
        init_typed: Whether to annotate fields in the generated `__init__`.
        warn_required_dynamic_aliases: Whether to raise required dynamic aliases error.
        debug_dataclass_transform: Whether to not reset `dataclass_transform_spec` attribute
            of `ModelMetaclass` for testing purposes.
    """
    __slots__ = ('init_forbid_extra', 'init_typed', 'warn_required_dynamic_aliases', 'debug_dataclass_transform')
    init_forbid_extra: bool
    init_typed: bool
    warn_required_dynamic_aliases: bool
    debug_dataclass_transform: bool

    def __init__(self, options: Options) -> None:
        if options.config_file is None:
            return
        toml_config = parse_toml(options.config_file)
        if toml_config is not None:
            config = toml_config.get('tool', {}).get('pydantic-mypy', {})
            for key in self.__slots__:
                setting = config.get(key, False)
                if not isinstance(setting, bool):
                    raise ValueError(f'Configuration value must be a boolean for key: {key}')
                setattr(self, key, setting)
        else:
            plugin_config = ConfigParser()
            plugin_config.read(options.config_file)
            for key in self.__slots__:
                setting = plugin_config.getboolean(CONFIGFILE_KEY, key, fallback=False)
                setattr(self, key, setting)

    def to_data(self) -> dict[str, Any]:
        """Returns a dict of config names to their values."""
        pass

def from_attributes_callback(ctx: MethodContext) -> Type:
    """Raise an error if from_attributes is not enabled."""
    pass

class PydanticModelField:
    """Based on mypy.plugins.dataclasses.DataclassAttribute."""

    def __init__(self, name: str, alias: str | None, has_dynamic_alias: bool, has_default: bool, line: int, column: int, type: Type | None, info: TypeInfo):
        self.name = name
        self.alias = alias
        self.has_dynamic_alias = has_dynamic_alias
        self.has_default = has_default
        self.line = line
        self.column = column
        self.type = type
        self.info = info

    def to_argument(self, current_info: TypeInfo, typed: bool, force_optional: bool, use_alias: bool, api: SemanticAnalyzerPluginInterface, force_typevars_invariant: bool) -> Argument:
        """Based on mypy.plugins.dataclasses.DataclassAttribute.to_argument."""
        pass

    def expand_type(self, current_info: TypeInfo, api: SemanticAnalyzerPluginInterface, force_typevars_invariant: bool=False) -> Type | None:
        """Based on mypy.plugins.dataclasses.DataclassAttribute.expand_type."""
        pass

    def to_var(self, current_info: TypeInfo, api: SemanticAnalyzerPluginInterface, use_alias: bool, force_typevars_invariant: bool=False) -> Var:
        """Based on mypy.plugins.dataclasses.DataclassAttribute.to_var."""
        pass

    def serialize(self) -> JsonDict:
        """Based on mypy.plugins.dataclasses.DataclassAttribute.serialize."""
        pass

    @classmethod
    def deserialize(cls, info: TypeInfo, data: JsonDict, api: SemanticAnalyzerPluginInterface) -> PydanticModelField:
        """Based on mypy.plugins.dataclasses.DataclassAttribute.deserialize."""
        pass

    def expand_typevar_from_subtype(self, sub_type: TypeInfo, api: SemanticAnalyzerPluginInterface) -> None:
        """Expands type vars in the context of a subtype when an attribute is inherited
        from a generic super type.
        """
        pass

class PydanticModelClassVar:
    """Based on mypy.plugins.dataclasses.DataclassAttribute.

    ClassVars are ignored by subclasses.

    Attributes:
        name: the ClassVar name
    """

    def __init__(self, name):
        self.name = name

    @classmethod
    def deserialize(cls, data: JsonDict) -> PydanticModelClassVar:
        """Based on mypy.plugins.dataclasses.DataclassAttribute.deserialize."""
        pass

    def serialize(self) -> JsonDict:
        """Based on mypy.plugins.dataclasses.DataclassAttribute.serialize."""
        pass

class PydanticModelTransformer:
    """Transform the BaseModel subclass according to the plugin settings.

    Attributes:
        tracked_config_fields: A set of field configs that the plugin has to track their value.
    """
    tracked_config_fields: set[str] = {'extra', 'frozen', 'from_attributes', 'populate_by_name', 'alias_generator'}

    def __init__(self, cls: ClassDef, reason: Expression | Statement, api: SemanticAnalyzerPluginInterface, plugin_config: PydanticPluginConfig) -> None:
        self._cls = cls
        self._reason = reason
        self._api = api
        self.plugin_config = plugin_config

    def transform(self) -> bool:
        """Configures the BaseModel subclass according to the plugin settings.

        In particular:

        * determines the model config and fields,
        * adds a fields-aware signature for the initializer and construct methods
        * freezes the class if frozen = True
        * stores the fields, config, and if the class is settings in the mypy metadata for access by subclasses
        """
        pass

    def adjust_decorator_signatures(self) -> None:
        """When we decorate a function `f` with `pydantic.validator(...)`, `pydantic.field_validator`
        or `pydantic.serializer(...)`, mypy sees `f` as a regular method taking a `self` instance,
        even though pydantic internally wraps `f` with `classmethod` if necessary.

        Teach mypy this by marking any function whose outermost decorator is a `validator()`,
        `field_validator()` or `serializer()` call as a `classmethod`.
        """
        pass

    def collect_config(self) -> ModelConfigData:
        """Collects the values of the config attributes that are used by the plugin, accounting for parent classes."""
        pass

    def collect_fields_and_class_vars(self, model_config: ModelConfigData, is_root_model: bool) -> tuple[list[PydanticModelField] | None, list[PydanticModelClassVar] | None]:
        """Collects the fields for the model, accounting for parent classes."""
        pass

    def collect_field_or_class_var_from_stmt(self, stmt: AssignmentStmt, model_config: ModelConfigData, class_vars: dict[str, PydanticModelClassVar]) -> PydanticModelField | PydanticModelClassVar | None:
        """Get pydantic model field from statement.

        Args:
            stmt: The statement.
            model_config: Configuration settings for the model.
            class_vars: ClassVars already known to be defined on the model.

        Returns:
            A pydantic model field if it could find the field in statement. Otherwise, `None`.
        """
        pass

    def _infer_dataclass_attr_init_type(self, sym: SymbolTableNode, name: str, context: Context) -> Type | None:
        """Infer __init__ argument type for an attribute.

        In particular, possibly use the signature of __set__.
        """
        pass

    def add_initializer(self, fields: list[PydanticModelField], config: ModelConfigData, is_settings: bool, is_root_model: bool) -> None:
        """Adds a fields-aware `__init__` method to the class.

        The added `__init__` will be annotated with types vs. all `Any` depending on the plugin settings.
        """
        pass

    def add_model_construct_method(self, fields: list[PydanticModelField], config: ModelConfigData, is_settings: bool) -> None:
        """Adds a fully typed `model_construct` classmethod to the class.

        Similar to the fields-aware __init__ method, but always uses the field names (not aliases),
        and does not treat settings fields as optional.
        """
        pass

    def set_frozen(self, fields: list[PydanticModelField], api: SemanticAnalyzerPluginInterface, frozen: bool) -> None:
        """Marks all fields as properties so that attempts to set them trigger mypy errors.

        This is the same approach used by the attrs and dataclasses plugins.
        """
        pass

    def get_config_update(self, name: str, arg: Expression, lax_extra: bool=False) -> ModelConfigData | None:
        """Determines the config update due to a single kwarg in the ConfigDict definition.

        Warns if a tracked config attribute is set to a value the plugin doesn't know how to interpret (e.g., an int)
        """
        pass

    @staticmethod
    def get_has_default(stmt: AssignmentStmt) -> bool:
        """Returns a boolean indicating whether the field defined in `stmt` is a required field."""
        pass

    @staticmethod
    def get_alias_info(stmt: AssignmentStmt) -> tuple[str | None, bool]:
        """Returns a pair (alias, has_dynamic_alias), extracted from the declaration of the field defined in `stmt`.

        `has_dynamic_alias` is True if and only if an alias is provided, but not as a string literal.
        If `has_dynamic_alias` is True, `alias` will be None.
        """
        pass

    def get_field_arguments(self, fields: list[PydanticModelField], typed: bool, use_alias: bool, requires_dynamic_aliases: bool, is_settings: bool, force_typevars_invariant: bool=False) -> list[Argument]:
        """Helper function used during the construction of the `__init__` and `model_construct` method signatures.

        Returns a list of mypy Argument instances for use in the generated signatures.
        """
        pass

    def should_init_forbid_extra(self, fields: list[PydanticModelField], config: ModelConfigData) -> bool:
        """Indicates whether the generated `__init__` should get a `**kwargs` at the end of its signature.

        We disallow arbitrary kwargs if the extra config setting is "forbid", or if the plugin config says to,
        *unless* a required dynamic alias is present (since then we can't determine a valid signature).
        """
        pass

    @staticmethod
    def is_dynamic_alias_present(fields: list[PydanticModelField], has_alias_generator: bool) -> bool:
        """Returns whether any fields on the model have a "dynamic alias", i.e., an alias that cannot be
        determined during static analysis.
        """
        pass

class ModelConfigData:
    """Pydantic mypy plugin model config class."""

    def __init__(self, forbid_extra: bool | None=None, frozen: bool | None=None, from_attributes: bool | None=None, populate_by_name: bool | None=None, has_alias_generator: bool | None=None):
        self.forbid_extra = forbid_extra
        self.frozen = frozen
        self.from_attributes = from_attributes
        self.populate_by_name = populate_by_name
        self.has_alias_generator = has_alias_generator

    def get_values_dict(self) -> dict[str, Any]:
        """Returns a dict of Pydantic model config names to their values.

        It includes the config if config value is not `None`.
        """
        pass

    def update(self, config: ModelConfigData | None) -> None:
        """Update Pydantic model config values."""
        pass

    def setdefault(self, key: str, value: Any) -> None:
        """Set default value for Pydantic model config if config value is `None`."""
        pass
ERROR_ORM = ErrorCode('pydantic-orm', 'Invalid from_attributes call', 'Pydantic')
ERROR_CONFIG = ErrorCode('pydantic-config', 'Invalid config value', 'Pydantic')
ERROR_ALIAS = ErrorCode('pydantic-alias', 'Dynamic alias disallowed', 'Pydantic')
ERROR_UNEXPECTED = ErrorCode('pydantic-unexpected', 'Unexpected behavior', 'Pydantic')
ERROR_UNTYPED = ErrorCode('pydantic-field', 'Untyped field disallowed', 'Pydantic')
ERROR_FIELD_DEFAULTS = ErrorCode('pydantic-field', 'Invalid Field defaults', 'Pydantic')
ERROR_EXTRA_FIELD_ROOT_MODEL = ErrorCode('pydantic-field', 'Extra field on RootModel subclass', 'Pydantic')

def error_from_attributes(model_name: str, api: CheckerPluginInterface, context: Context) -> None:
    """Emits an error when the model does not have `from_attributes=True`."""
    pass

def error_invalid_config_value(name: str, api: SemanticAnalyzerPluginInterface, context: Context) -> None:
    """Emits an error when the config value is invalid."""
    pass

def error_required_dynamic_aliases(api: SemanticAnalyzerPluginInterface, context: Context) -> None:
    """Emits required dynamic aliases error.

    This will be called when `warn_required_dynamic_aliases=True`.
    """
    pass

def error_unexpected_behavior(detail: str, api: CheckerPluginInterface | SemanticAnalyzerPluginInterface, context: Context) -> None:
    """Emits unexpected behavior error."""
    pass

def error_untyped_fields(api: SemanticAnalyzerPluginInterface, context: Context) -> None:
    """Emits an error when there is an untyped field in the model."""
    pass

def error_extra_fields_on_root_model(api: CheckerPluginInterface, context: Context) -> None:
    """Emits an error when there is more than just a root field defined for a subclass of RootModel."""
    pass

def error_default_and_default_factory_specified(api: CheckerPluginInterface, context: Context) -> None:
    """Emits an error when `Field` has both `default` and `default_factory` together."""
    pass

def add_method(api: SemanticAnalyzerPluginInterface | CheckerPluginInterface, cls: ClassDef, name: str, args: list[Argument], return_type: Type, self_type: Type | None=None, tvar_def: TypeVarDef | None=None, is_classmethod: bool=False) -> None:
    """Very closely related to `mypy.plugins.common.add_method_to_class`, with a few pydantic-specific changes."""
    pass

def parse_toml(config_file: str) -> dict[str, Any] | None:
    """Returns a dict of config keys to values.

    It reads configs from toml file and returns `None` if the file is not a toml file.
    """
    pass
