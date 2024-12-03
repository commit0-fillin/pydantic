from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable, Generic, Iterator, Mapping, TypeVar, Union
from pydantic_core import CoreSchema, SchemaSerializer, SchemaValidator
from typing_extensions import Literal
from ..errors import PydanticErrorCodes, PydanticUserError
from ..plugin._schema_validator import PluggableSchemaValidator
if TYPE_CHECKING:
    from ..dataclasses import PydanticDataclass
    from ..main import BaseModel
ValSer = TypeVar('ValSer', bound=Union[SchemaValidator, PluggableSchemaValidator, SchemaSerializer])
T = TypeVar('T')

class MockCoreSchema(Mapping[str, Any]):
    """Mocker for `pydantic_core.CoreSchema` which optionally attempts to
    rebuild the thing it's mocking when one of its methods is accessed and raises an error if that fails.
    """
    __slots__ = ('_error_message', '_code', '_attempt_rebuild', '_built_memo')

    def __init__(self, error_message: str, *, code: PydanticErrorCodes, attempt_rebuild: Callable[[], CoreSchema | None] | None=None) -> None:
        self._error_message = error_message
        self._code: PydanticErrorCodes = code
        self._attempt_rebuild = attempt_rebuild
        self._built_memo: CoreSchema | None = None

    def __getitem__(self, key: str) -> Any:
        return self._get_built().__getitem__(key)

    def __len__(self) -> int:
        return self._get_built().__len__()

    def __iter__(self) -> Iterator[str]:
        return self._get_built().__iter__()

class MockValSer(Generic[ValSer]):
    """Mocker for `pydantic_core.SchemaValidator` or `pydantic_core.SchemaSerializer` which optionally attempts to
    rebuild the thing it's mocking when one of its methods is accessed and raises an error if that fails.
    """
    __slots__ = ('_error_message', '_code', '_val_or_ser', '_attempt_rebuild')

    def __init__(self, error_message: str, *, code: PydanticErrorCodes, val_or_ser: Literal['validator', 'serializer'], attempt_rebuild: Callable[[], ValSer | None] | None=None) -> None:
        self._error_message = error_message
        self._val_or_ser = SchemaValidator if val_or_ser == 'validator' else SchemaSerializer
        self._code: PydanticErrorCodes = code
        self._attempt_rebuild = attempt_rebuild

    def __getattr__(self, item: str) -> None:
        __tracebackhide__ = True
        if self._attempt_rebuild:
            val_ser = self._attempt_rebuild()
            if val_ser is not None:
                return getattr(val_ser, item)
        getattr(self._val_or_ser, item)
        raise PydanticUserError(self._error_message, code=self._code)

def set_model_mocks(cls: type[BaseModel], cls_name: str, undefined_name: str='all referenced types') -> None:
    """Set `__pydantic_validator__` and `__pydantic_serializer__` to `MockValSer`s on a model.

    Args:
        cls: The model class to set the mocks on
        cls_name: Name of the model class, used in error messages
        undefined_name: Name of the undefined thing, used in error messages
    """
    error_message = f"'{cls_name}' is not fully defined: {undefined_name} are not fully defined"
    cls.__pydantic_validator__ = MockValSer(error_message, code='class-not-fully-defined', val_or_ser='validator')
    cls.__pydantic_serializer__ = MockValSer(error_message, code='class-not-fully-defined', val_or_ser='serializer')

def set_dataclass_mocks(cls: type[PydanticDataclass], cls_name: str, undefined_name: str='all referenced types') -> None:
    """Set `__pydantic_validator__` and `__pydantic_serializer__` to `MockValSer`s on a dataclass.

    Args:
        cls: The model class to set the mocks on
        cls_name: Name of the model class, used in error messages
        undefined_name: Name of the undefined thing, used in error messages
    """
    error_message = f"'{cls_name}' is not fully defined: {undefined_name} are not fully defined"
    cls.__pydantic_validator__ = MockValSer(error_message, code='class-not-fully-defined', val_or_ser='validator')
    cls.__pydantic_serializer__ = MockValSer(error_message, code='class-not-fully-defined', val_or_ser='serializer')
