import abc
import math
import re
import warnings
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from types import new_class
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Dict, FrozenSet, List, Optional, Pattern, Set, Tuple, Type, TypeVar, Union, cast, overload
from uuid import UUID
from weakref import WeakSet
from pydantic.v1 import errors
from pydantic.v1.datetime_parse import parse_date
from pydantic.v1.utils import import_string, update_not_none
from pydantic.v1.validators import bytes_validator, constr_length_validator, constr_lower, constr_strip_whitespace, constr_upper, decimal_validator, float_finite_validator, float_validator, frozenset_validator, int_validator, list_validator, number_multiple_validator, number_size_validator, path_exists_validator, path_validator, set_validator, str_validator, strict_bytes_validator, strict_float_validator, strict_int_validator, strict_str_validator
__all__ = ['NoneStr', 'NoneBytes', 'StrBytes', 'NoneStrBytes', 'StrictStr', 'ConstrainedBytes', 'conbytes', 'ConstrainedList', 'conlist', 'ConstrainedSet', 'conset', 'ConstrainedFrozenSet', 'confrozenset', 'ConstrainedStr', 'constr', 'PyObject', 'ConstrainedInt', 'conint', 'PositiveInt', 'NegativeInt', 'NonNegativeInt', 'NonPositiveInt', 'ConstrainedFloat', 'confloat', 'PositiveFloat', 'NegativeFloat', 'NonNegativeFloat', 'NonPositiveFloat', 'FiniteFloat', 'ConstrainedDecimal', 'condecimal', 'UUID1', 'UUID3', 'UUID4', 'UUID5', 'FilePath', 'DirectoryPath', 'Json', 'JsonWrapper', 'SecretField', 'SecretStr', 'SecretBytes', 'StrictBool', 'StrictBytes', 'StrictInt', 'StrictFloat', 'PaymentCardNumber', 'ByteSize', 'PastDate', 'FutureDate', 'ConstrainedDate', 'condate']
NoneStr = Optional[str]
NoneBytes = Optional[bytes]
StrBytes = Union[str, bytes]
NoneStrBytes = Optional[StrBytes]
OptionalInt = Optional[int]
OptionalIntFloat = Union[OptionalInt, float]
OptionalIntFloatDecimal = Union[OptionalIntFloat, Decimal]
OptionalDate = Optional[date]
StrIntFloat = Union[str, int, float]
if TYPE_CHECKING:
    from typing_extensions import Annotated
    from pydantic.v1.dataclasses import Dataclass
    from pydantic.v1.main import BaseModel
    from pydantic.v1.typing import CallableGenerator
    ModelOrDc = Type[Union[BaseModel, Dataclass]]
T = TypeVar('T')
_DEFINED_TYPES: 'WeakSet[type]' = WeakSet()

class ConstrainedNumberMeta(type):

    def __new__(cls, name: str, bases: Any, dct: Dict[str, Any]) -> 'ConstrainedInt':
        new_cls = cast('ConstrainedInt', type.__new__(cls, name, bases, dct))
        if new_cls.gt is not None and new_cls.ge is not None:
            raise errors.ConfigError('bounds gt and ge cannot be specified at the same time')
        if new_cls.lt is not None and new_cls.le is not None:
            raise errors.ConfigError('bounds lt and le cannot be specified at the same time')
        return _registered(new_cls)
if TYPE_CHECKING:
    StrictBool = bool
else:

    class StrictBool(int):
        """
        StrictBool to allow for bools which are not type-coerced.
        """

        @classmethod
        def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
            field_schema.update(type='boolean')

        @classmethod
        def __get_validators__(cls) -> 'CallableGenerator':
            yield cls.validate

        @classmethod
        def validate(cls, value: Any) -> bool:
            """
            Ensure that we only allow bools.
            """
            if not isinstance(value, bool):
                raise ValueError('Value must be a boolean')
            return value

class ConstrainedInt(int, metaclass=ConstrainedNumberMeta):
    strict: bool = False
    gt: OptionalInt = None
    ge: OptionalInt = None
    lt: OptionalInt = None
    le: OptionalInt = None
    multiple_of: OptionalInt = None

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(field_schema, exclusiveMinimum=cls.gt, exclusiveMaximum=cls.lt, minimum=cls.ge, maximum=cls.le, multipleOf=cls.multiple_of)

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield (strict_int_validator if cls.strict else int_validator)
        yield number_size_validator
        yield number_multiple_validator
if TYPE_CHECKING:
    PositiveInt = int
    NegativeInt = int
    NonPositiveInt = int
    NonNegativeInt = int
    StrictInt = int
else:

    class PositiveInt(ConstrainedInt):
        gt = 0

    class NegativeInt(ConstrainedInt):
        lt = 0

    class NonPositiveInt(ConstrainedInt):
        le = 0

    class NonNegativeInt(ConstrainedInt):
        ge = 0

    class StrictInt(ConstrainedInt):
        strict = True

class ConstrainedFloat(float, metaclass=ConstrainedNumberMeta):
    strict: bool = False
    gt: OptionalIntFloat = None
    ge: OptionalIntFloat = None
    lt: OptionalIntFloat = None
    le: OptionalIntFloat = None
    multiple_of: OptionalIntFloat = None
    allow_inf_nan: Optional[bool] = None

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(field_schema, exclusiveMinimum=cls.gt, exclusiveMaximum=cls.lt, minimum=cls.ge, maximum=cls.le, multipleOf=cls.multiple_of)
        if field_schema.get('exclusiveMinimum') == -math.inf:
            del field_schema['exclusiveMinimum']
        if field_schema.get('minimum') == -math.inf:
            del field_schema['minimum']
        if field_schema.get('exclusiveMaximum') == math.inf:
            del field_schema['exclusiveMaximum']
        if field_schema.get('maximum') == math.inf:
            del field_schema['maximum']

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield (strict_float_validator if cls.strict else float_validator)
        yield number_size_validator
        yield number_multiple_validator
        yield float_finite_validator
if TYPE_CHECKING:
    PositiveFloat = float
    NegativeFloat = float
    NonPositiveFloat = float
    NonNegativeFloat = float
    StrictFloat = float
    FiniteFloat = float
else:

    class PositiveFloat(ConstrainedFloat):
        gt = 0

    class NegativeFloat(ConstrainedFloat):
        lt = 0

    class NonPositiveFloat(ConstrainedFloat):
        le = 0

    class NonNegativeFloat(ConstrainedFloat):
        ge = 0

    class StrictFloat(ConstrainedFloat):
        strict = True

    class FiniteFloat(ConstrainedFloat):
        allow_inf_nan = False

class ConstrainedBytes(bytes):
    strip_whitespace = False
    to_upper = False
    to_lower = False
    min_length: OptionalInt = None
    max_length: OptionalInt = None
    strict: bool = False

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(field_schema, minLength=cls.min_length, maxLength=cls.max_length)

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield (strict_bytes_validator if cls.strict else bytes_validator)
        yield constr_strip_whitespace
        yield constr_upper
        yield constr_lower
        yield constr_length_validator
if TYPE_CHECKING:
    StrictBytes = bytes
else:

    class StrictBytes(ConstrainedBytes):
        strict = True

class ConstrainedStr(str):
    strip_whitespace = False
    to_upper = False
    to_lower = False
    min_length: OptionalInt = None
    max_length: OptionalInt = None
    curtail_length: OptionalInt = None
    regex: Optional[Union[str, Pattern[str]]] = None
    strict = False

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(field_schema, minLength=cls.min_length, maxLength=cls.max_length, pattern=cls.regex and cls._get_pattern(cls.regex))

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield (strict_str_validator if cls.strict else str_validator)
        yield constr_strip_whitespace
        yield constr_upper
        yield constr_lower
        yield constr_length_validator
        yield cls.validate
if TYPE_CHECKING:
    StrictStr = str
else:

    class StrictStr(ConstrainedStr):
        strict = True

class ConstrainedSet(set):
    __origin__ = set
    __args__: Set[Type[T]]
    min_items: Optional[int] = None
    max_items: Optional[int] = None
    item_type: Type[T]

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield cls.set_length_validator

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(field_schema, minItems=cls.min_items, maxItems=cls.max_items)

class ConstrainedFrozenSet(frozenset):
    __origin__ = frozenset
    __args__: FrozenSet[Type[T]]
    min_items: Optional[int] = None
    max_items: Optional[int] = None
    item_type: Type[T]

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield cls.frozenset_length_validator

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(field_schema, minItems=cls.min_items, maxItems=cls.max_items)

class ConstrainedList(list):
    __origin__ = list
    __args__: Tuple[Type[T], ...]
    min_items: Optional[int] = None
    max_items: Optional[int] = None
    unique_items: Optional[bool] = None
    item_type: Type[T]

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield cls.list_length_validator
        if cls.unique_items:
            yield cls.unique_items_validator

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(field_schema, minItems=cls.min_items, maxItems=cls.max_items, uniqueItems=cls.unique_items)
if TYPE_CHECKING:
    PyObject = Callable[..., Any]
else:

    class PyObject:
        validate_always = True

        @classmethod
        def __get_validators__(cls) -> 'CallableGenerator':
            yield cls.validate

class ConstrainedDecimal(Decimal, metaclass=ConstrainedNumberMeta):
    gt: OptionalIntFloatDecimal = None
    ge: OptionalIntFloatDecimal = None
    lt: OptionalIntFloatDecimal = None
    le: OptionalIntFloatDecimal = None
    max_digits: OptionalInt = None
    decimal_places: OptionalInt = None
    multiple_of: OptionalIntFloatDecimal = None

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(field_schema, exclusiveMinimum=cls.gt, exclusiveMaximum=cls.lt, minimum=cls.ge, maximum=cls.le, multipleOf=cls.multiple_of)

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield decimal_validator
        yield number_size_validator
        yield number_multiple_validator
        yield cls.validate
if TYPE_CHECKING:
    UUID1 = UUID
    UUID3 = UUID
    UUID4 = UUID
    UUID5 = UUID
else:

    class UUID1(UUID):
        _required_version = 1

        @classmethod
        def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
            field_schema.update(type='string', format=f'uuid{cls._required_version}')

    class UUID3(UUID1):
        _required_version = 3

    class UUID4(UUID1):
        _required_version = 4

    class UUID5(UUID1):
        _required_version = 5
if TYPE_CHECKING:
    FilePath = Path
    DirectoryPath = Path
else:

    class FilePath(Path):

        @classmethod
        def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
            field_schema.update(format='file-path')

        @classmethod
        def __get_validators__(cls) -> 'CallableGenerator':
            yield path_validator
            yield path_exists_validator
            yield cls.validate

    class DirectoryPath(Path):

        @classmethod
        def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
            field_schema.update(format='directory-path')

        @classmethod
        def __get_validators__(cls) -> 'CallableGenerator':
            yield path_validator
            yield path_exists_validator
            yield cls.validate

class JsonWrapper:
    pass

class JsonMeta(type):

    def __getitem__(self, t: Type[Any]) -> Type[JsonWrapper]:
        if t is Any:
            return Json
        return _registered(type('JsonWrapperValue', (JsonWrapper,), {'inner_type': t}))
if TYPE_CHECKING:
    Json = Annotated[T, ...]
else:

    class Json(metaclass=JsonMeta):

        @classmethod
        def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
            field_schema.update(type='string', format='json-string')

class SecretField(abc.ABC):
    """
    Note: this should be implemented as a generic like `SecretField(ABC, Generic[T])`,
          the `__init__()` should be part of the abstract class and the
          `get_secret_value()` method should use the generic `T` type.

          However Cython doesn't support very well generics at the moment and
          the generated code fails to be imported (see
          https://github.com/cython/cython/issues/2753).
    """

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__) and self.get_secret_value() == other.get_secret_value()

    def __str__(self) -> str:
        return '**********' if self.get_secret_value() else ''

    def __hash__(self) -> int:
        return hash(self.get_secret_value())

class SecretStr(SecretField):
    min_length: OptionalInt = None
    max_length: OptionalInt = None

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(field_schema, type='string', writeOnly=True, format='password', minLength=cls.min_length, maxLength=cls.max_length)

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield cls.validate
        yield constr_length_validator

    def __init__(self, value: str):
        self._secret_value = value

    def __repr__(self) -> str:
        return f"SecretStr('{self}')"

    def __len__(self) -> int:
        return len(self._secret_value)

class SecretBytes(SecretField):
    min_length: OptionalInt = None
    max_length: OptionalInt = None

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(field_schema, type='string', writeOnly=True, format='password', minLength=cls.min_length, maxLength=cls.max_length)

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield cls.validate
        yield constr_length_validator

    def __init__(self, value: bytes):
        self._secret_value = value

    def __repr__(self) -> str:
        return f"SecretBytes(b'{self}')"

    def __len__(self) -> int:
        return len(self._secret_value)

class PaymentCardBrand(str, Enum):
    amex = 'American Express'
    mastercard = 'Mastercard'
    visa = 'Visa'
    other = 'other'

    def __str__(self) -> str:
        return self.value

class PaymentCardNumber(str):
    """
    Based on: https://en.wikipedia.org/wiki/Payment_card_number
    """
    strip_whitespace: ClassVar[bool] = True
    min_length: ClassVar[int] = 12
    max_length: ClassVar[int] = 19
    bin: str
    last4: str
    brand: PaymentCardBrand

    def __init__(self, card_number: str):
        self.bin = card_number[:6]
        self.last4 = card_number[-4:]
        self.brand = self._get_brand(card_number)

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield str_validator
        yield constr_strip_whitespace
        yield constr_length_validator
        yield cls.validate_digits
        yield cls.validate_luhn_check_digit
        yield cls
        yield cls.validate_length_for_brand

    @classmethod
    def validate_digits(cls, card_number: str) -> str:
        if not card_number.isdigit():
            raise ValueError('Card number must contain only digits')
        return card_number

    @classmethod
    def validate_luhn_check_digit(cls, card_number: str) -> str:
        """
        Based on: https://en.wikipedia.org/wiki/Luhn_algorithm
        """
        digits = [int(d) for d in card_number]
        checksum = 0
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        checksum += sum(odd_digits)
        for digit in even_digits:
            checksum += sum(divmod(digit * 2, 10))
        if checksum % 10 != 0:
            raise ValueError('Invalid card number (failed Luhn check)')
        return card_number

    @classmethod
    def validate_length_for_brand(cls, card_number: 'PaymentCardNumber') -> 'PaymentCardNumber':
        """
        Validate length based on BIN for major brands:
        https://en.wikipedia.org/wiki/Payment_card_number#Issuer_identification_number_(IIN)
        """
        if card_number.brand == PaymentCardBrand.amex and len(card_number) != 15:
            raise ValueError('American Express card numbers must be 15 digits long')
        elif card_number.brand in (PaymentCardBrand.mastercard, PaymentCardBrand.visa) and len(card_number) not in (16, 19):
            raise ValueError(f'{card_number.brand} card numbers must be 16 or 19 digits long')
        return card_number

    @classmethod
    def _get_brand(cls, card_number: str) -> PaymentCardBrand:
        if card_number.startswith(('34', '37')):
            return PaymentCardBrand.amex
        elif card_number.startswith('4'):
            return PaymentCardBrand.visa
        elif card_number.startswith(('51', '52', '53', '54', '55')):
            return PaymentCardBrand.mastercard
        else:
            return PaymentCardBrand.other
BYTE_SIZES = {'b': 1, 'kb': 10 ** 3, 'mb': 10 ** 6, 'gb': 10 ** 9, 'tb': 10 ** 12, 'pb': 10 ** 15, 'eb': 10 ** 18, 'kib': 2 ** 10, 'mib': 2 ** 20, 'gib': 2 ** 30, 'tib': 2 ** 40, 'pib': 2 ** 50, 'eib': 2 ** 60}
BYTE_SIZES.update({k.lower()[0]: v for k, v in BYTE_SIZES.items() if 'i' not in k})
byte_string_re = re.compile('^\\s*(\\d*\\.?\\d+)\\s*(\\w+)?', re.IGNORECASE)

class ByteSize(int):

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield cls.validate
if TYPE_CHECKING:
    PastDate = date
    FutureDate = date
else:

    class PastDate(date):

        @classmethod
        def __get_validators__(cls) -> 'CallableGenerator':
            yield parse_date
            yield cls.validate

    class FutureDate(date):

        @classmethod
        def __get_validators__(cls) -> 'CallableGenerator':
            yield parse_date
            yield cls.validate

class ConstrainedDate(date, metaclass=ConstrainedNumberMeta):
    gt: OptionalDate = None
    ge: OptionalDate = None
    lt: OptionalDate = None
    le: OptionalDate = None

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        update_not_none(field_schema, exclusiveMinimum=cls.gt, exclusiveMaximum=cls.lt, minimum=cls.ge, maximum=cls.le)

    @classmethod
    def __get_validators__(cls) -> 'CallableGenerator':
        yield parse_date
        yield number_size_validator
