import re
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network, _BaseAddress, _BaseNetwork
from typing import TYPE_CHECKING, Any, Collection, Dict, Generator, List, Match, Optional, Pattern, Set, Tuple, Type, Union, cast, no_type_check
from pydantic.v1 import errors
from pydantic.v1.utils import Representation, update_not_none
from pydantic.v1.validators import constr_length_validator, str_validator
if TYPE_CHECKING:
    import email_validator
    from typing_extensions import TypedDict
    from pydantic.v1.config import BaseConfig
    from pydantic.v1.fields import ModelField
    from pydantic.v1.typing import AnyCallable
    CallableGenerator = Generator[AnyCallable, None, None]

    class Parts(TypedDict, total=False):
        scheme: str
        user: Optional[str]
        password: Optional[str]
        ipv4: Optional[str]
        ipv6: Optional[str]
        domain: Optional[str]
        port: Optional[str]
        path: Optional[str]
        query: Optional[str]
        fragment: Optional[str]

    class HostParts(TypedDict, total=False):
        host: str
        tld: Optional[str]
        host_type: Optional[str]
        port: Optional[str]
        rebuild: bool
else:
    email_validator = None

    class Parts(dict):
        pass
NetworkType = Union[str, bytes, int, Tuple[Union[str, bytes, int], Union[str, int]]]
__all__ = ['AnyUrl', 'AnyHttpUrl', 'FileUrl', 'HttpUrl', 'stricturl', 'EmailStr', 'NameEmail', 'IPvAnyAddress', 'IPvAnyInterface', 'IPvAnyNetwork', 'PostgresDsn', 'CockroachDsn', 'AmqpDsn', 'RedisDsn', 'MongoDsn', 'KafkaDsn', 'validate_email']
_url_regex_cache = None
_multi_host_url_regex_cache = None
_ascii_domain_regex_cache = None
_int_domain_regex_cache = None
_host_regex_cache = None
_host_regex = '(?:(?P<ipv4>(?:\\d{1,3}\\.){3}\\d{1,3})(?=$|[/:#?])|(?P<ipv6>\\[[A-F0-9]*:[A-F0-9:]+\\])(?=$|[/:#?])|(?P<domain>[^\\s/:?#]+))?(?::(?P<port>\\d+))?'
_scheme_regex = '(?:(?P<scheme>[a-z][a-z0-9+\\-.]+)://)?'
_user_info_regex = '(?:(?P<user>[^\\s:/]*)(?::(?P<password>[^\\s/]*))?@)?'
_path_regex = '(?P<path>/[^\\s?#]*)?'
_query_regex = '(?:\\?(?P<query>[^\\s#]*))?'
_fragment_regex = '(?:#(?P<fragment>[^\\s#]*))?'

def multi_host_url_regex() -> Pattern[str]:
    """
    Compiled multi host url regex.

    Additionally to `url_regex` it allows to match multiple hosts.
    E.g. host1.db.net,host2.db.net
    """
    global _multi_host_url_regex_cache
    if _multi_host_url_regex_cache is None:
        pattern = (
            rf'{_scheme_regex}'
            rf'{_user_info_regex}'
            r'(?P<hosts>([^/]+))'
            rf'({_path_regex})?'
            rf'{_query_regex}'
            rf'{_fragment_regex}'
        )
        _multi_host_url_regex_cache = re.compile(pattern, re.IGNORECASE)
    return _multi_host_url_regex_cache

class AnyUrl(str):
    strip_whitespace = True
    min_length = 1
    max_length = 2 ** 16
    allowed_schemes: Optional[Collection[str]] = None
    tld_required: bool = False
    user_required: bool = False
    host_required: bool = True
    hidden_parts: Set[str] = set()
    __slots__ = ('scheme', 'user', 'password', 'host', 'tld', 'host_type', 'port', 'path', 'query', 'fragment')

    @no_type_check
    def __new__(cls, url: Optional[str], **kwargs) -> object:
        return str.__new__(cls, cls.build(**kwargs) if url is None else url)

    def __init__(self, url: str, *, scheme: str, user: Optional[str]=None, password: Optional[str]=None, host: Optional[str]=None, tld: Optional[str]=None, host_type: str='domain', port: Optional[str]=None, path: Optional[str]=None, query: Optional[str]=None, fragment: Optional[str]=None) -> None:
        str.__init__(url)
        self.scheme = scheme
        self.user = user
        self.password = password
        self.host = host
        self.tld = tld
        self.host_type = host_type
        self.port = port
        self.path = path
        self.query = query
        self.fragment = fragment

        # Additional validation
        if self.host_required and not self.host:
            raise ValueError("Host is required for this URL type")
        if self.user_required and not self.user:
            raise ValueError("User is required for this URL type")
        if self.tld_required and not self.tld:
            raise ValueError("Top-level domain is required for this URL type")

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(field_schema, minLength=cls.min_length, maxLength=cls.max_length, format='uri')

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield cls.validate

    @classmethod
    def _build_url(cls, m: Match[str], url: str, parts: 'Parts') -> 'AnyUrl':
        """
        Validate hosts and build the AnyUrl object. Split from `validate` so this method
        can be altered in `MultiHostDsn`.
        """
        parts = cls.validate_parts(parts)

        host, tld, host_type, port = cls._validate_host(parts['domain'], parts['port'])
        return cls(
            url,
            scheme=parts['scheme'],
            user=parts['user'],
            password=parts['password'],
            host=host,
            tld=tld,
            host_type=host_type,
            port=port,
            path=parts['path'],
            query=parts['query'],
            fragment=parts['fragment'],
        )

    @classmethod
    def validate_parts(cls, parts: 'Parts', validate_port: bool=True) -> 'Parts':
        """
        A method used to validate parts of a URL.
        Could be overridden to set default values for parts if missing
        """
        if cls.allowed_schemes and parts['scheme'] not in cls.allowed_schemes:
            raise ValueError(f"Scheme {parts['scheme']} not allowed, allowed schemes: {', '.join(cls.allowed_schemes)}")

        if validate_port and parts['port']:
            port = int(parts['port'])
            if port > 65_535:
                raise ValueError('Port cannot exceed 65535')

        if cls.user_required and not parts['user']:
            raise ValueError('User is required in the URL')

        return parts

    def __repr__(self) -> str:
        extra = ', '.join((f'{n}={getattr(self, n)!r}' for n in self.__slots__ if getattr(self, n) is not None))
        return f'{self.__class__.__name__}({super().__repr__()}, {extra})'

class AnyHttpUrl(AnyUrl):
    allowed_schemes = {'http', 'https'}
    __slots__ = ()

class HttpUrl(AnyHttpUrl):
    tld_required = True
    max_length = 2083
    hidden_parts = {'port'}

class FileUrl(AnyUrl):
    allowed_schemes = {'file'}
    host_required = False
    __slots__ = ()

class MultiHostDsn(AnyUrl):
    __slots__ = AnyUrl.__slots__ + ('hosts',)

    def __init__(self, *args: Any, hosts: Optional[List['HostParts']]=None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.hosts = hosts

class PostgresDsn(MultiHostDsn):
    allowed_schemes = {'postgres', 'postgresql', 'postgresql+asyncpg', 'postgresql+pg8000', 'postgresql+psycopg', 'postgresql+psycopg2', 'postgresql+psycopg2cffi', 'postgresql+py-postgresql', 'postgresql+pygresql'}
    user_required = True
    __slots__ = ()

class CockroachDsn(AnyUrl):
    allowed_schemes = {'cockroachdb', 'cockroachdb+psycopg2', 'cockroachdb+asyncpg'}
    user_required = True

class AmqpDsn(AnyUrl):
    allowed_schemes = {'amqp', 'amqps'}
    host_required = False

class RedisDsn(AnyUrl):
    __slots__ = ()
    allowed_schemes = {'redis', 'rediss'}
    host_required = False

class MongoDsn(AnyUrl):
    allowed_schemes = {'mongodb'}

class KafkaDsn(AnyUrl):
    allowed_schemes = {'kafka'}

class EmailStr(str):

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        field_schema.update(type='string', format='email')

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        import_email_validator()
        yield str_validator
        yield cls.validate

class NameEmail(Representation):
    __slots__ = ('name', 'email')

    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, NameEmail) and (self.name, self.email) == (other.name, other.email)

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        field_schema.update(type='string', format='name-email')

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        import_email_validator()
        yield cls.validate

    def __str__(self) -> str:
        return f'{self.name} <{self.email}>'

class IPvAnyAddress(_BaseAddress):
    __slots__ = ()

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        field_schema.update(type='string', format='ipvanyaddress')

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield cls.validate

class IPvAnyInterface(_BaseAddress):
    __slots__ = ()

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        field_schema.update(type='string', format='ipvanyinterface')

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield cls.validate

class IPvAnyNetwork(_BaseNetwork):

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        field_schema.update(type='string', format='ipvanynetwork')

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield cls.validate
pretty_email_regex = re.compile('([\\w ]*?) *<(.*)> *')
MAX_EMAIL_LENGTH = 2048
'Maximum length for an email.\nA somewhat arbitrary but very generous number compared to what is allowed by most implementations.\n'

def validate_email(value: Union[str]) -> Tuple[str, str]:
    """
    Email address validation using https://pypi.org/project/email-validator/
    Notes:
    * raw ip address (literal) domain parts are not allowed.
    * "John Doe <local_part@domain.com>" style "pretty" email addresses are processed
    * spaces are striped from the beginning and end of addresses but no error is raised
    """
    if email_validator is None:
        raise ImportError('email-validator is not installed, run `pip install email-validator`')

    value = value.strip()

    if len(value) > MAX_EMAIL_LENGTH:
        raise ValueError(f'Email address too long, maximum allowed is {MAX_EMAIL_LENGTH} characters')

    try:
        parsed = email_validator.validate_email(value, check_deliverability=False)
        name = parsed.local_part
        email = parsed.email
        return name, email
    except email_validator.EmailNotValidError as e:
        raise ValueError(str(e))
