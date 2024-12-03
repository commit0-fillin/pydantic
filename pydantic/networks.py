"""The networks module contains types for common network-related fields."""
from __future__ import annotations as _annotations
import dataclasses as _dataclasses
import re
from importlib.metadata import version
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from typing import TYPE_CHECKING, Any
from pydantic_core import MultiHostUrl, PydanticCustomError, Url, core_schema
from typing_extensions import Annotated, Self, TypeAlias
from ._internal import _fields, _repr, _schema_generation_shared
from ._migration import getattr_migration
from .annotated_handlers import GetCoreSchemaHandler
from .json_schema import JsonSchemaValue
if TYPE_CHECKING:
    import email_validator
    NetworkType: TypeAlias = 'str | bytes | int | tuple[str | bytes | int, str | int]'
else:
    email_validator = None
__all__ = ['AnyUrl', 'AnyHttpUrl', 'FileUrl', 'FtpUrl', 'HttpUrl', 'WebsocketUrl', 'AnyWebsocketUrl', 'UrlConstraints', 'EmailStr', 'NameEmail', 'IPvAnyAddress', 'IPvAnyInterface', 'IPvAnyNetwork', 'PostgresDsn', 'CockroachDsn', 'AmqpDsn', 'RedisDsn', 'MongoDsn', 'KafkaDsn', 'NatsDsn', 'validate_email', 'MySQLDsn', 'MariaDBDsn', 'ClickHouseDsn']

@_dataclasses.dataclass
class UrlConstraints(_fields.PydanticMetadata):
    """Url constraints.

    Attributes:
        max_length: The maximum length of the url. Defaults to `None`.
        allowed_schemes: The allowed schemes. Defaults to `None`.
        host_required: Whether the host is required. Defaults to `None`.
        default_host: The default host. Defaults to `None`.
        default_port: The default port. Defaults to `None`.
        default_path: The default path. Defaults to `None`.
    """
    max_length: int | None = None
    allowed_schemes: list[str] | None = None
    host_required: bool | None = None
    default_host: str | None = None
    default_port: int | None = None
    default_path: str | None = None

    def __hash__(self) -> int:
        return hash((self.max_length, tuple(self.allowed_schemes) if self.allowed_schemes is not None else None, self.host_required, self.default_host, self.default_port, self.default_path))
AnyUrl = Url
'Base type for all URLs.\n\n* Any scheme allowed\n* Top-level domain (TLD) not required\n* Host required\n\nAssuming an input URL of `http://samuel:pass@example.com:8000/the/path/?query=here#fragment=is;this=bit`,\nthe types export the following properties:\n\n- `scheme`: the URL scheme (`http`), always set.\n- `host`: the URL host (`example.com`), always set.\n- `username`: optional username if included (`samuel`).\n- `password`: optional password if included (`pass`).\n- `port`: optional port (`8000`).\n- `path`: optional path (`/the/path/`).\n- `query`: optional URL query (for example, `GET` arguments or "search string", such as `query=here`).\n- `fragment`: optional fragment (`fragment=is;this=bit`).\n'
AnyHttpUrl = Annotated[Url, UrlConstraints(allowed_schemes=['http', 'https'])]
'A type that will accept any http or https URL.\n\n* TLD not required\n* Host required\n'
HttpUrl = Annotated[Url, UrlConstraints(max_length=2083, allowed_schemes=['http', 'https'])]
'A type that will accept any http or https URL.\n\n* TLD not required\n* Host required\n* Max length 2083\n\n```py\nfrom pydantic import BaseModel, HttpUrl, ValidationError\n\nclass MyModel(BaseModel):\n    url: HttpUrl\n\nm = MyModel(url=\'http://www.example.com\')  # (1)!\nprint(m.url)\n#> http://www.example.com/\n\ntry:\n    MyModel(url=\'ftp://invalid.url\')\nexcept ValidationError as e:\n    print(e)\n    \'\'\'\n    1 validation error for MyModel\n    url\n      URL scheme should be \'http\' or \'https\' [type=url_scheme, input_value=\'ftp://invalid.url\', input_type=str]\n    \'\'\'\n\ntry:\n    MyModel(url=\'not a url\')\nexcept ValidationError as e:\n    print(e)\n    \'\'\'\n    1 validation error for MyModel\n    url\n      Input should be a valid URL, relative URL without a base [type=url_parsing, input_value=\'not a url\', input_type=str]\n    \'\'\'\n```\n\n1. Note: mypy would prefer `m = MyModel(url=HttpUrl(\'http://www.example.com\'))`, but Pydantic will convert the string to an HttpUrl instance anyway.\n\n"International domains" (e.g. a URL where the host or TLD includes non-ascii characters) will be encoded via\n[punycode](https://en.wikipedia.org/wiki/Punycode) (see\n[this article](https://www.xudongz.com/blog/2017/idn-phishing/) for a good description of why this is important):\n\n```py\nfrom pydantic import BaseModel, HttpUrl\n\nclass MyModel(BaseModel):\n    url: HttpUrl\n\nm1 = MyModel(url=\'http://puny£code.com\')\nprint(m1.url)\n#> http://xn--punycode-eja.com/\nm2 = MyModel(url=\'https://www.аррӏе.com/\')\nprint(m2.url)\n#> https://www.xn--80ak6aa92e.com/\nm3 = MyModel(url=\'https://www.example.珠宝/\')\nprint(m3.url)\n#> https://www.example.xn--pbt977c/\n```\n\n\n!!! warning "Underscores in Hostnames"\n    In Pydantic, underscores are allowed in all parts of a domain except the TLD.\n    Technically this might be wrong - in theory the hostname cannot have underscores, but subdomains can.\n\n    To explain this; consider the following two cases:\n\n    - `exam_ple.co.uk`: the hostname is `exam_ple`, which should not be allowed since it contains an underscore.\n    - `foo_bar.example.com` the hostname is `example`, which should be allowed since the underscore is in the subdomain.\n\n    Without having an exhaustive list of TLDs, it would be impossible to differentiate between these two. Therefore\n    underscores are allowed, but you can always do further validation in a validator if desired.\n\n    Also, Chrome, Firefox, and Safari all currently accept `http://exam_ple.com` as a URL, so we\'re in good\n    (or at least big) company.\n'
AnyWebsocketUrl = Annotated[Url, UrlConstraints(allowed_schemes=['ws', 'wss'])]
'A type that will accept any ws or wss URL.\n\n* TLD not required\n* Host required\n'
WebsocketUrl = Annotated[Url, UrlConstraints(max_length=2083, allowed_schemes=['ws', 'wss'])]
'A type that will accept any ws or wss URL.\n\n* TLD not required\n* Host required\n* Max length 2083\n'
FileUrl = Annotated[Url, UrlConstraints(allowed_schemes=['file'])]
'A type that will accept any file URL.\n\n* Host not required\n'
FtpUrl = Annotated[Url, UrlConstraints(allowed_schemes=['ftp'])]
'A type that will accept ftp URL.\n\n* TLD not required\n* Host required\n'
PostgresDsn = Annotated[MultiHostUrl, UrlConstraints(host_required=True, allowed_schemes=['postgres', 'postgresql', 'postgresql+asyncpg', 'postgresql+pg8000', 'postgresql+psycopg', 'postgresql+psycopg2', 'postgresql+psycopg2cffi', 'postgresql+py-postgresql', 'postgresql+pygresql'])]
"A type that will accept any Postgres DSN.\n\n* User info required\n* TLD not required\n* Host required\n* Supports multiple hosts\n\nIf further validation is required, these properties can be used by validators to enforce specific behaviour:\n\n```py\nfrom pydantic import (\n    BaseModel,\n    HttpUrl,\n    PostgresDsn,\n    ValidationError,\n    field_validator,\n)\n\nclass MyModel(BaseModel):\n    url: HttpUrl\n\nm = MyModel(url='http://www.example.com')\n\n# the repr() method for a url will display all properties of the url\nprint(repr(m.url))\n#> Url('http://www.example.com/')\nprint(m.url.scheme)\n#> http\nprint(m.url.host)\n#> www.example.com\nprint(m.url.port)\n#> 80\n\nclass MyDatabaseModel(BaseModel):\n    db: PostgresDsn\n\n    @field_validator('db')\n    def check_db_name(cls, v):\n        assert v.path and len(v.path) > 1, 'database must be provided'\n        return v\n\nm = MyDatabaseModel(db='postgres://user:pass@localhost:5432/foobar')\nprint(m.db)\n#> postgres://user:pass@localhost:5432/foobar\n\ntry:\n    MyDatabaseModel(db='postgres://user:pass@localhost:5432')\nexcept ValidationError as e:\n    print(e)\n    '''\n    1 validation error for MyDatabaseModel\n    db\n      Assertion failed, database must be provided\n    assert (None)\n     +  where None = MultiHostUrl('postgres://user:pass@localhost:5432').path [type=assertion_error, input_value='postgres://user:pass@localhost:5432', input_type=str]\n    '''\n```\n"
CockroachDsn = Annotated[Url, UrlConstraints(host_required=True, allowed_schemes=['cockroachdb', 'cockroachdb+psycopg2', 'cockroachdb+asyncpg'])]
'A type that will accept any Cockroach DSN.\n\n* User info required\n* TLD not required\n* Host required\n'
AmqpDsn = Annotated[Url, UrlConstraints(allowed_schemes=['amqp', 'amqps'])]
'A type that will accept any AMQP DSN.\n\n* User info required\n* TLD not required\n* Host required\n'
RedisDsn = Annotated[Url, UrlConstraints(allowed_schemes=['redis', 'rediss'], default_host='localhost', default_port=6379, default_path='/0')]
'A type that will accept any Redis DSN.\n\n* User info required\n* TLD not required\n* Host required (e.g., `rediss://:pass@localhost`)\n'
MongoDsn = Annotated[MultiHostUrl, UrlConstraints(allowed_schemes=['mongodb', 'mongodb+srv'], default_port=27017)]
'A type that will accept any MongoDB DSN.\n\n* User info not required\n* Database name not required\n* Port not required\n* User info may be passed without user part (e.g., `mongodb://mongodb0.example.com:27017`).\n'
KafkaDsn = Annotated[Url, UrlConstraints(allowed_schemes=['kafka'], default_host='localhost', default_port=9092)]
'A type that will accept any Kafka DSN.\n\n* User info required\n* TLD not required\n* Host required\n'
NatsDsn = Annotated[MultiHostUrl, UrlConstraints(allowed_schemes=['nats', 'tls', 'ws'], default_host='localhost', default_port=4222)]
'A type that will accept any NATS DSN.\n\nNATS is a connective technology built for the ever increasingly hyper-connected world.\nIt is a single technology that enables applications to securely communicate across\nany combination of cloud vendors, on-premise, edge, web and mobile, and devices.\nMore: https://nats.io\n'
MySQLDsn = Annotated[Url, UrlConstraints(allowed_schemes=['mysql', 'mysql+mysqlconnector', 'mysql+aiomysql', 'mysql+asyncmy', 'mysql+mysqldb', 'mysql+pymysql', 'mysql+cymysql', 'mysql+pyodbc'], default_port=3306)]
'A type that will accept any MySQL DSN.\n\n* User info required\n* TLD not required\n* Host required\n'
MariaDBDsn = Annotated[Url, UrlConstraints(allowed_schemes=['mariadb', 'mariadb+mariadbconnector', 'mariadb+pymysql'], default_port=3306)]
'A type that will accept any MariaDB DSN.\n\n* User info required\n* TLD not required\n* Host required\n'
ClickHouseDsn = Annotated[Url, UrlConstraints(allowed_schemes=['clickhouse+native', 'clickhouse+asynch'], default_host='localhost', default_port=9000)]
'A type that will accept any ClickHouse DSN.\n\n* User info required\n* TLD not required\n* Host required\n'
if TYPE_CHECKING:
    EmailStr = Annotated[str, ...]
else:

    class EmailStr:
        """
        Info:
            To use this type, you need to install the optional
            [`email-validator`](https://github.com/JoshData/python-email-validator) package:

            ```bash
            pip install email-validator
            ```

        Validate email addresses.

        ```py
        from pydantic import BaseModel, EmailStr

        class Model(BaseModel):
            email: EmailStr

        print(Model(email='contact@mail.com'))
        #> email='contact@mail.com'
        ```
        """

        @classmethod
        def __get_pydantic_core_schema__(cls, _source: type[Any], _handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
            import_email_validator()
            return core_schema.no_info_after_validator_function(cls._validate, core_schema.str_schema())

        @classmethod
        def __get_pydantic_json_schema__(cls, core_schema: core_schema.CoreSchema, handler: _schema_generation_shared.GetJsonSchemaHandler) -> JsonSchemaValue:
            field_schema = handler(core_schema)
            field_schema.update(type='string', format='email')
            return field_schema

class NameEmail(_repr.Representation):
    """
    Info:
        To use this type, you need to install the optional
        [`email-validator`](https://github.com/JoshData/python-email-validator) package:

        ```bash
        pip install email-validator
        ```

    Validate a name and email address combination, as specified by
    [RFC 5322](https://datatracker.ietf.org/doc/html/rfc5322#section-3.4).

    The `NameEmail` has two properties: `name` and `email`.
    In case the `name` is not provided, it's inferred from the email address.

    ```py
    from pydantic import BaseModel, NameEmail

    class User(BaseModel):
        email: NameEmail

    user = User(email='Fred Bloggs <fred.bloggs@example.com>')
    print(user.email)
    #> Fred Bloggs <fred.bloggs@example.com>
    print(user.email.name)
    #> Fred Bloggs

    user = User(email='fred.bloggs@example.com')
    print(user.email)
    #> fred.bloggs <fred.bloggs@example.com>
    print(user.email.name)
    #> fred.bloggs
    ```
    """
    __slots__ = ('name', 'email')

    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, NameEmail) and (self.name, self.email) == (other.name, other.email)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: core_schema.CoreSchema, handler: _schema_generation_shared.GetJsonSchemaHandler) -> JsonSchemaValue:
        field_schema = handler(core_schema)
        field_schema.update(type='string', format='name-email')
        return field_schema

    @classmethod
    def __get_pydantic_core_schema__(cls, _source: type[Any], _handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        import_email_validator()
        return core_schema.no_info_after_validator_function(cls._validate, core_schema.json_or_python_schema(json_schema=core_schema.str_schema(), python_schema=core_schema.union_schema([core_schema.is_instance_schema(cls), core_schema.str_schema()], custom_error_type='name_email_type', custom_error_message='Input is not a valid NameEmail'), serialization=core_schema.to_string_ser_schema()))

    def __str__(self) -> str:
        if '@' in self.name:
            return f'"{self.name}" <{self.email}>'
        return f'{self.name} <{self.email}>'

class IPvAnyAddress:
    """Validate an IPv4 or IPv6 address.

    ```py
    from pydantic import BaseModel
    from pydantic.networks import IPvAnyAddress

    class IpModel(BaseModel):
        ip: IPvAnyAddress

    print(IpModel(ip='127.0.0.1'))
    #> ip=IPv4Address('127.0.0.1')

    try:
        IpModel(ip='http://www.example.com')
    except ValueError as e:
        print(e.errors())
        '''
        [
            {
                'type': 'ip_any_address',
                'loc': ('ip',),
                'msg': 'value is not a valid IPv4 or IPv6 address',
                'input': 'http://www.example.com',
            }
        ]
        '''
    ```
    """
    __slots__ = ()

    def __new__(cls, value: Any) -> IPv4Address | IPv6Address:
        """Validate an IPv4 or IPv6 address."""
        try:
            return IPv4Address(value)
        except ValueError:
            try:
                return IPv6Address(value)
            except ValueError:
                raise PydanticCustomError('ip_any_address', 'value is not a valid IPv4 or IPv6 address')

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: core_schema.CoreSchema, handler: _schema_generation_shared.GetJsonSchemaHandler) -> JsonSchemaValue:
        field_schema = {}
        field_schema.update(type='string', format='ipvanyaddress')
        return field_schema

    @classmethod
    def __get_pydantic_core_schema__(cls, _source: type[Any], _handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(cls._validate, serialization=core_schema.to_string_ser_schema())

class IPvAnyInterface:
    """Validate an IPv4 or IPv6 interface."""
    __slots__ = ()

    def __new__(cls, value: NetworkType) -> IPv4Interface | IPv6Interface:
        """Validate an IPv4 or IPv6 interface."""
        try:
            return IPv4Interface(value)
        except ValueError:
            try:
                return IPv6Interface(value)
            except ValueError:
                raise PydanticCustomError('ip_any_interface', 'value is not a valid IPv4 or IPv6 interface')

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema: core_schema.CoreSchema, handler: _schema_generation_shared.GetJsonSchemaHandler) -> JsonSchemaValue:
        field_schema = {}
        field_schema.update(type='string', format='ipvanyinterface')
        return field_schema

    @classmethod
    def __get_pydantic_core_schema__(cls, _source: type[Any], _handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_plain_validator_function(cls._validate, serialization=core_schema.to_string_ser_schema())
IPvAnyNetworkType: TypeAlias = 'IPv4Network | IPv6Network'
if TYPE_CHECKING:
    IPvAnyNetwork = IPvAnyNetworkType
else:

    class IPvAnyNetwork:
        """Validate an IPv4 or IPv6 network."""
        __slots__ = ()

        def __new__(cls, value: NetworkType) -> IPvAnyNetworkType:
            """Validate an IPv4 or IPv6 network."""
            try:
                return IPv4Network(value)
            except ValueError:
                try:
                    return IPv6Network(value)
                except ValueError:
                    raise PydanticCustomError('ip_any_network', 'value is not a valid IPv4 or IPv6 network')

        @classmethod
        def __get_pydantic_json_schema__(cls, core_schema: core_schema.CoreSchema, handler: _schema_generation_shared.GetJsonSchemaHandler) -> JsonSchemaValue:
            field_schema = {}
            field_schema.update(type='string', format='ipvanynetwork')
            return field_schema

        @classmethod
        def __get_pydantic_core_schema__(cls, _source: type[Any], _handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
            return core_schema.no_info_plain_validator_function(cls._validate, serialization=core_schema.to_string_ser_schema())
pretty_email_regex = _build_pretty_email_regex()
MAX_EMAIL_LENGTH = 2048
'Maximum length for an email.\nA somewhat arbitrary but very generous number compared to what is allowed by most implementations.\n'

def validate_email(value: str) -> tuple[str, str]:
    """Email address validation using [email-validator](https://pypi.org/project/email-validator/).

    Note:
        Note that:

        * Raw IP address (literal) domain parts are not allowed.
        * `"John Doe <local_part@domain.com>"` style "pretty" email addresses are processed.
        * Spaces are striped from the beginning and end of addresses, but no error is raised.
    """
    import_email_validator()
    from email_validator import EmailNotValidError, validate_email as _validate_email

    value = value.strip()

    try:
        email_info = _validate_email(value, check_deliverability=False)
        return email_info.local_part, email_info.domain
    except EmailNotValidError as e:
        raise PydanticCustomError('value_error.email', 'value is not a valid email address') from e
__getattr__ = getattr_migration(__name__)
