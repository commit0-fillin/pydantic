import sys
from configparser import ConfigParser
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type as TypingType, Union
from mypy.errorcodes import ErrorCode
from mypy.nodes import ARG_NAMED, ARG_NAMED_OPT, ARG_OPT, ARG_POS, ARG_STAR2, MDEF, Argument, AssignmentStmt, Block, CallExpr, ClassDef, Context, Decorator, EllipsisExpr, FuncBase, FuncDef, JsonDict, MemberExpr, NameExpr, PassStmt, PlaceholderNode, RefExpr, StrExpr, SymbolNode, SymbolTableNode, TempNode, TypeInfo, TypeVarExpr, Var
from mypy.options import Options
from mypy.plugin import CheckerPluginInterface, ClassDefContext, FunctionContext, MethodContext, Plugin, ReportConfigContext, SemanticAnalyzerPluginInterface
from mypy.plugins import dataclasses
from mypy.semanal import set_callable_name
from mypy.server.trigger import make_wildcard_trigger
from mypy.types import AnyType, CallableType, Instance, NoneType, Overloaded, ProperType, Type, TypeOfAny, TypeType, TypeVarId, TypeVarType, UnionType, get_proper_type
from mypy.typevars import fill_typevars
from mypy.util import get_unique_redefinition_name
from mypy.version import __version__ as mypy_version
from pydantic.v1.utils import is_valid_field
try:
    from mypy.types import TypeVarDef
except ImportError:
    from mypy.types import TypeVarType as TypeVarDef
CONFIGFILE_KEY = 'pydantic-mypy'
METADATA_KEY = 'pydantic-mypy-metadata'
_NAMESPACE = __name__[:-5]
BASEMODEL_FULLNAME = f'{_NAMESPACE}.main.BaseModel'
BASESETTINGS_FULLNAME = f'{_NAMESPACE}.env_settings.BaseSettings'
MODEL_METACLASS_FULLNAME = f'{_NAMESPACE}.main.ModelMetaclass'
FIELD_FULLNAME = f'{_NAMESPACE}.fields.Field'
DATACLASS_FULLNAME = f'{_NAMESPACE}.dataclasses.dataclass'
MYPY_VERSION_TUPLE = parse_mypy_version(mypy_version)
BUILTINS_NAME = 'builtins' if MYPY_VERSION_TUPLE >= (0, 930) else '__builtins__'
__version__ = 2

def plugin(version: str) -> 'TypingType[Plugin]':
    """
    `version` is the mypy version string

    We might want to use this to print a warning if the mypy version being used is
    newer, or especially older, than we expect (or need).
    """
    return PydanticPlugin

class PydanticPlugin(Plugin):

    def __init__(self, options: Options) -> None:
        self.plugin_config = PydanticPluginConfig(options)
        self._plugin_data = self.plugin_config.to_data()
        super().__init__(options)

    def get_class_decorator_hook(self, fullname: str) -> Optional[Callable[[ClassDefContext], None]]:
        """Mark pydantic.dataclasses as dataclass.

        Mypy version 1.1.1 added support for `@dataclass_transform` decorator.
        """
        if fullname == DATACLASS_FULLNAME:
            return dataclasses.dataclass_class_maker_callback
        return None

    def report_config_data(self, ctx: ReportConfigContext) -> Dict[str, Any]:
        """Return all plugin config data.

        Used by mypy to determine if cache needs to be discarded.
        """
        return self._plugin_data

    def _pydantic_model_metaclass_marker_callback(self, ctx: ClassDefContext) -> None:
        """Reset dataclass_transform_spec attribute of ModelMetaclass.

        Let the plugin handle it. This behavior can be disabled
        if 'debug_dataclass_transform' is set to True', for testing purposes.
        """
        if not self.plugin_config.debug_dataclass_transform:
            ctx.cls.info.metadata.setdefault('dataclass_transform_spec', {})

    def _pydantic_field_callback(self, ctx: FunctionContext) -> 'Type':
        """
        Extract the type of the `default` argument from the Field function, and use it as the return type.

        In particular:
        * Check whether the default and default_factory argument is specified.
        * Output an error if both are specified.
        * Retrieve the type of the argument which is specified, and use it as return type for the function.
        """
        default_arg = ctx.arg_names.index('default') if 'default' in ctx.arg_names else None
        default_factory_arg = ctx.arg_names.index('default_factory') if 'default_factory' in ctx.arg_names else None

        if default_arg is not None and default_factory_arg is not None:
            ctx.api.fail('Field cannot specify both default and default_factory', ctx.context)
            return AnyType(TypeOfAny.from_error)

        if default_arg is not None and ctx.args[default_arg] is not None:
            default_type = ctx.arg_types[default_arg]
            return default_type
        elif default_factory_arg is not None and ctx.args[default_factory_arg] is not None:
            default_factory_type = ctx.arg_types[default_factory_arg]
            if isinstance(default_factory_type, CallableType):
                return default_factory_type.ret_type
            else:
                ctx.api.fail('default_factory must be a callable', ctx.context)
                return AnyType(TypeOfAny.from_error)

        return AnyType(TypeOfAny.implementation_artifact)

class PydanticPluginConfig:
    __slots__ = ('init_forbid_extra', 'init_typed', 'warn_required_dynamic_aliases', 'warn_untyped_fields', 'debug_dataclass_transform')
    init_forbid_extra: bool
    init_typed: bool
    warn_required_dynamic_aliases: bool
    warn_untyped_fields: bool
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

def from_orm_callback(ctx: MethodContext) -> Type:
    """
    Raise an error if orm_mode is not enabled
    """
    model_type = ctx.type
    if isinstance(model_type, Instance):
        model_class = model_type.type
        config = model_class.metadata.get('pydantic_config', {})
        if not config.get('orm_mode'):
            ctx.api.fail('You must have the config attribute orm_mode=True to use from_orm', ctx.context)
    return ctx.default_return_type

class PydanticModelTransformer:
    tracked_config_fields: Set[str] = {'extra', 'allow_mutation', 'frozen', 'orm_mode', 'allow_population_by_field_name', 'alias_generator'}

    def __init__(self, ctx: ClassDefContext, plugin_config: PydanticPluginConfig) -> None:
        self._ctx = ctx
        self.plugin_config = plugin_config

    def transform(self) -> None:
        """
        Configures the BaseModel subclass according to the plugin settings.

        In particular:
        * determines the model config and fields,
        * adds a fields-aware signature for the initializer and construct methods
        * freezes the class if allow_mutation = False or frozen = True
        * stores the fields, config, and if the class is settings in the mypy metadata for access by subclasses
        """
        pass

    def adjust_validator_signatures(self) -> None:
        """When we decorate a function `f` with `pydantic.validator(...), mypy sees
        `f` as a regular method taking a `self` instance, even though pydantic
        internally wraps `f` with `classmethod` if necessary.

        Teach mypy this by marking any function whose outermost decorator is a
        `validator()` call as a classmethod.
        """
        pass

    def collect_config(self) -> 'ModelConfigData':
        """
        Collects the values of the config attributes that are used by the plugin, accounting for parent classes.
        """
        pass

    def collect_fields(self, model_config: 'ModelConfigData') -> List['PydanticModelField']:
        """
        Collects the fields for the model, accounting for parent classes
        """
        pass

    def add_initializer(self, fields: List['PydanticModelField'], config: 'ModelConfigData', is_settings: bool) -> None:
        """
        Adds a fields-aware `__init__` method to the class.

        The added `__init__` will be annotated with types vs. all `Any` depending on the plugin settings.
        """
        pass

    def add_construct_method(self, fields: List['PydanticModelField']) -> None:
        """
        Adds a fully typed `construct` classmethod to the class.

        Similar to the fields-aware __init__ method, but always uses the field names (not aliases),
        and does not treat settings fields as optional.
        """
        pass

    def set_frozen(self, fields: List['PydanticModelField'], frozen: bool) -> None:
        """
        Marks all fields as properties so that attempts to set them trigger mypy errors.

        This is the same approach used by the attrs and dataclasses plugins.
        """
        pass

    def get_config_update(self, substmt: AssignmentStmt) -> Optional['ModelConfigData']:
        """
        Determines the config update due to a single statement in the Config class definition.

        Warns if a tracked config attribute is set to a value the plugin doesn't know how to interpret (e.g., an int)
        """
        pass

    @staticmethod
    def get_is_required(cls: ClassDef, stmt: AssignmentStmt, lhs: NameExpr) -> bool:
        """
        Returns a boolean indicating whether the field defined in `stmt` is a required field.
        """
        pass

    @staticmethod
    def type_has_implicit_default(type_: Optional[ProperType]) -> bool:
        """
        Returns True if the passed type will be given an implicit default value.

        In pydantic v1, this is the case for Optional types and Any (with default value None).
        """
        pass

    @staticmethod
    def get_alias_info(stmt: AssignmentStmt) -> Tuple[Optional[str], bool]:
        """
        Returns a pair (alias, has_dynamic_alias), extracted from the declaration of the field defined in `stmt`.

        `has_dynamic_alias` is True if and only if an alias is provided, but not as a string literal.
        If `has_dynamic_alias` is True, `alias` will be None.
        """
        pass

    def get_field_arguments(self, fields: List['PydanticModelField'], typed: bool, force_all_optional: bool, use_alias: bool) -> List[Argument]:
        """
        Helper function used during the construction of the `__init__` and `construct` method signatures.

        Returns a list of mypy Argument instances for use in the generated signatures.
        """
        pass

    def should_init_forbid_extra(self, fields: List['PydanticModelField'], config: 'ModelConfigData') -> bool:
        """
        Indicates whether the generated `__init__` should get a `**kwargs` at the end of its signature

        We disallow arbitrary kwargs if the extra config setting is "forbid", or if the plugin config says to,
        *unless* a required dynamic alias is present (since then we can't determine a valid signature).
        """
        pass

    @staticmethod
    def is_dynamic_alias_present(fields: List['PydanticModelField'], has_alias_generator: bool) -> bool:
        """
        Returns whether any fields on the model have a "dynamic alias", i.e., an alias that cannot be
        determined during static analysis.
        """
        pass

class PydanticModelField:

    def __init__(self, name: str, is_required: bool, alias: Optional[str], has_dynamic_alias: bool, line: int, column: int):
        self.name = name
        self.is_required = is_required
        self.alias = alias
        self.has_dynamic_alias = has_dynamic_alias
        self.line = line
        self.column = column

class ModelConfigData:

    def __init__(self, forbid_extra: Optional[bool]=None, allow_mutation: Optional[bool]=None, frozen: Optional[bool]=None, orm_mode: Optional[bool]=None, allow_population_by_field_name: Optional[bool]=None, has_alias_generator: Optional[bool]=None):
        self.forbid_extra = forbid_extra
        self.allow_mutation = allow_mutation
        self.frozen = frozen
        self.orm_mode = orm_mode
        self.allow_population_by_field_name = allow_population_by_field_name
        self.has_alias_generator = has_alias_generator
ERROR_ORM = ErrorCode('pydantic-orm', 'Invalid from_orm call', 'Pydantic')
ERROR_CONFIG = ErrorCode('pydantic-config', 'Invalid config value', 'Pydantic')
ERROR_ALIAS = ErrorCode('pydantic-alias', 'Dynamic alias disallowed', 'Pydantic')
ERROR_UNEXPECTED = ErrorCode('pydantic-unexpected', 'Unexpected behavior', 'Pydantic')
ERROR_UNTYPED = ErrorCode('pydantic-field', 'Untyped field disallowed', 'Pydantic')
ERROR_FIELD_DEFAULTS = ErrorCode('pydantic-field', 'Invalid Field defaults', 'Pydantic')

def add_method(ctx: ClassDefContext, name: str, args: List[Argument], return_type: Type, self_type: Optional[Type]=None, tvar_def: Optional[TypeVarDef]=None, is_classmethod: bool=False, is_new: bool=False) -> None:
    """
    Adds a new method to a class.

    This can be dropped if/when https://github.com/python/mypy/issues/7301 is merged
    """
    pass

def get_fullname(x: Union[FuncBase, SymbolNode]) -> str:
    """
    Used for compatibility with mypy 0.740; can be dropped once support for 0.740 is dropped.
    """
    pass

def get_name(x: Union[FuncBase, SymbolNode]) -> str:
    """
    Used for compatibility with mypy 0.740; can be dropped once support for 0.740 is dropped.
    """
    pass
