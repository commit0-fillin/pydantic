import os
import warnings
from pathlib import Path
from typing import AbstractSet, Any, Callable, ClassVar, Dict, List, Mapping, Optional, Tuple, Type, Union
from pydantic.v1.config import BaseConfig, Extra
from pydantic.v1.fields import ModelField
from pydantic.v1.main import BaseModel
from pydantic.v1.types import JsonWrapper
from pydantic.v1.typing import StrPath, display_as_type, get_origin, is_union
from pydantic.v1.utils import deep_update, lenient_issubclass, path_type, sequence_like
env_file_sentinel = str(object())
SettingsSourceCallable = Callable[['BaseSettings'], Dict[str, Any]]
DotenvType = Union[StrPath, List[StrPath], Tuple[StrPath, ...]]

class SettingsError(ValueError):
    pass

class BaseSettings(BaseModel):
    """
    Base class for settings, allowing values to be overridden by environment variables.

    This is useful in production for secrets you do not wish to save in code, it plays nicely with docker(-compose),
    Heroku and any 12 factor app design.
    """

    def __init__(__pydantic_self__, _env_file: Optional[DotenvType]=env_file_sentinel, _env_file_encoding: Optional[str]=None, _env_nested_delimiter: Optional[str]=None, _secrets_dir: Optional[StrPath]=None, **values: Any) -> None:
        super().__init__(**__pydantic_self__._build_values(values, _env_file=_env_file, _env_file_encoding=_env_file_encoding, _env_nested_delimiter=_env_nested_delimiter, _secrets_dir=_secrets_dir))

    class Config(BaseConfig):
        env_prefix: str = ''
        env_file: Optional[DotenvType] = None
        env_file_encoding: Optional[str] = None
        env_nested_delimiter: Optional[str] = None
        secrets_dir: Optional[StrPath] = None
        validate_all: bool = True
        extra: Extra = Extra.forbid
        arbitrary_types_allowed: bool = True
        case_sensitive: bool = False
    __config__: ClassVar[Type[Config]]

class InitSettingsSource:
    __slots__ = ('init_kwargs',)

    def __init__(self, init_kwargs: Dict[str, Any]):
        self.init_kwargs = init_kwargs

    def __call__(self, settings: BaseSettings) -> Dict[str, Any]:
        return self.init_kwargs

    def __repr__(self) -> str:
        return f'InitSettingsSource(init_kwargs={self.init_kwargs!r})'

class EnvSettingsSource:
    __slots__ = ('env_file', 'env_file_encoding', 'env_nested_delimiter', 'env_prefix_len')

    def __init__(self, env_file: Optional[DotenvType], env_file_encoding: Optional[str], env_nested_delimiter: Optional[str]=None, env_prefix_len: int=0):
        self.env_file: Optional[DotenvType] = env_file
        self.env_file_encoding: Optional[str] = env_file_encoding
        self.env_nested_delimiter: Optional[str] = env_nested_delimiter
        self.env_prefix_len: int = env_prefix_len

    def __call__(self, settings: BaseSettings) -> Dict[str, Any]:
        """
        Build environment variables suitable for passing to the Model.
        """
        d: Dict[str, Any] = {}
        if settings.__config__.case_sensitive:
            env_vars: Mapping[str, Optional[str]] = os.environ
        else:
            env_vars = {k.lower(): v for k, v in os.environ.items()}
        dotenv_vars = self._read_env_files(settings.__config__.case_sensitive)
        if dotenv_vars:
            env_vars = {**dotenv_vars, **env_vars}
        for field in settings.__fields__.values():
            env_val: Optional[str] = None
            for env_name in field.field_info.extra['env_names']:
                env_val = env_vars.get(env_name)
                if env_val is not None:
                    break
            is_complex, allow_parse_failure = self.field_is_complex(field)
            if is_complex:
                if env_val is None:
                    env_val_built = self.explode_env_vars(field, env_vars)
                    if env_val_built:
                        d[field.alias] = env_val_built
                else:
                    try:
                        env_val = settings.__config__.parse_env_var(field.name, env_val)
                    except ValueError as e:
                        if not allow_parse_failure:
                            raise SettingsError(f'error parsing env var "{env_name}"') from e
                    if isinstance(env_val, dict):
                        d[field.alias] = deep_update(env_val, self.explode_env_vars(field, env_vars))
                    else:
                        d[field.alias] = env_val
            elif env_val is not None:
                d[field.alias] = env_val
        return d

    def field_is_complex(self, field: ModelField) -> Tuple[bool, bool]:
        """
        Find out if a field is complex, and if so whether JSON errors should be ignored
        """
        if field.is_complex():
            if isinstance(field.type_, type) and issubclass(field.type_, BaseSettings):
                return True, True
            return True, False
        return False, False

    def explode_env_vars(self, field: ModelField, env_vars: Mapping[str, Optional[str]]) -> Dict[str, Any]:
        """
        Process env_vars and extract the values of keys containing env_nested_delimiter into nested dictionaries.

        This is applied to a single field, hence filtering by env_var prefix.
        """
        prefixes = [f'{name}_' for name in field.field_info.extra['env_names']]
        result: Dict[str, Any] = {}
        for env_name, env_val in env_vars.items():
            if any(env_name.startswith(prefix) for prefix in prefixes):
                if self.env_nested_delimiter:
                    env_name = env_name[len(next(prefix for prefix in prefixes if env_name.startswith(prefix))):]
                    keys = env_name.split(self.env_nested_delimiter)
                    tmp = result
                    for key in keys[:-1]:
                        if key not in tmp:
                            tmp[key] = {}
                        tmp = tmp[key]
                    tmp[keys[-1]] = env_val
                else:
                    result[env_name] = env_val
        return result

    def __repr__(self) -> str:
        return f'EnvSettingsSource(env_file={self.env_file!r}, env_file_encoding={self.env_file_encoding!r}, env_nested_delimiter={self.env_nested_delimiter!r})'

class SecretsSettingsSource:
    __slots__ = ('secrets_dir',)

    def __init__(self, secrets_dir: Optional[StrPath]):
        self.secrets_dir: Optional[StrPath] = secrets_dir

    def __call__(self, settings: BaseSettings) -> Dict[str, Any]:
        """
        Build fields from "secrets" files.
        """
        secrets: Dict[str, Optional[str]] = {}
        if self.secrets_dir is None:
            return secrets
        secrets_path = Path(self.secrets_dir).expanduser()
        if not secrets_path.exists():
            warnings.warn(f'directory "{secrets_path}" does not exist')
            return secrets
        if not secrets_path.is_dir():
            raise SettingsError(f'secrets_dir must reference a directory, not a {path_type(secrets_path)}')
        for field in settings.__fields__.values():
            for env_name in field.field_info.extra['env_names']:
                path = find_case_path(secrets_path, env_name, settings.__config__.case_sensitive)
                if not path:
                    continue
                if path.is_file():
                    secret_value = path.read_text().strip()
                    if field.is_complex():
                        try:
                            secret_value = settings.__config__.parse_env_var(field.name, secret_value)
                        except ValueError as e:
                            raise SettingsError(f'error parsing env var "{env_name}"') from e
                    secrets[field.alias] = secret_value
                else:
                    warnings.warn(f'attempted to load secret file "{path}" but found a {path_type(path)} instead.', stacklevel=4)
        return secrets

    def __repr__(self) -> str:
        return f'SecretsSettingsSource(secrets_dir={self.secrets_dir!r})'

def find_case_path(dir_path: Path, file_name: str, case_sensitive: bool) -> Optional[Path]:
    """
    Find a file within path's directory matching filename, optionally ignoring case.
    """
    if case_sensitive:
        path = dir_path / file_name
        return path if path.exists() else None
    else:
        for path in dir_path.iterdir():
            if path.name.lower() == file_name.lower():
                return path
    return None
