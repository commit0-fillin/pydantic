"""Support for alias configurations."""
from __future__ import annotations
import dataclasses
from typing import Any, Callable, Literal
from pydantic_core import PydanticUndefined
from ._internal import _internal_dataclass
__all__ = ('AliasGenerator', 'AliasPath', 'AliasChoices')

@dataclasses.dataclass(**_internal_dataclass.slots_true)
class AliasPath:
    """Usage docs: https://docs.pydantic.dev/2.8/concepts/alias#aliaspath-and-aliaschoices

    A data class used by `validation_alias` as a convenience to create aliases.

    Attributes:
        path: A list of string or integer aliases.
    """
    path: list[int | str]

    def __init__(self, first_arg: str, *args: str | int) -> None:
        self.path = [first_arg] + list(args)

    def convert_to_aliases(self) -> list[str | int]:
        """Converts arguments to a list of string or integer aliases.

        Returns:
            The list of aliases.
        """
        return self.path

    def search_dict_for_path(self, d: dict) -> Any:
        """Searches a dictionary for the path specified by the alias.

        Returns:
            The value at the specified path, or `PydanticUndefined` if the path is not found.
        """
        current = d
        for key in self.path:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return PydanticUndefined
        return current

@dataclasses.dataclass(**_internal_dataclass.slots_true)
class AliasChoices:
    """Usage docs: https://docs.pydantic.dev/2.8/concepts/alias#aliaspath-and-aliaschoices

    A data class used by `validation_alias` as a convenience to create aliases.

    Attributes:
        choices: A list containing a string or `AliasPath`.
    """
    choices: list[str | AliasPath]

    def __init__(self, first_choice: str | AliasPath, *choices: str | AliasPath) -> None:
        self.choices = [first_choice] + list(choices)

    def convert_to_aliases(self) -> list[list[str | int]]:
        """Converts arguments to a list of lists containing string or integer aliases.

        Returns:
            The list of aliases.
        """
        result = []
        for choice in self.choices:
            if isinstance(choice, str):
                result.append([choice])
            elif isinstance(choice, AliasPath):
                result.append(choice.convert_to_aliases())
        return result

@dataclasses.dataclass(**_internal_dataclass.slots_true)
class AliasGenerator:
    """Usage docs: https://docs.pydantic.dev/2.8/concepts/alias#using-an-aliasgenerator

    A data class used by `alias_generator` as a convenience to create various aliases.

    Attributes:
        alias: A callable that takes a field name and returns an alias for it.
        validation_alias: A callable that takes a field name and returns a validation alias for it.
        serialization_alias: A callable that takes a field name and returns a serialization alias for it.
    """
    alias: Callable[[str], str] | None = None
    validation_alias: Callable[[str], str | AliasPath | AliasChoices] | None = None
    serialization_alias: Callable[[str], str] | None = None

    def _generate_alias(self, alias_kind: Literal['alias', 'validation_alias', 'serialization_alias'], allowed_types: tuple[type[str] | type[AliasPath] | type[AliasChoices], ...], field_name: str) -> str | AliasPath | AliasChoices | None:
        """Generate an alias of the specified kind. Returns None if the alias generator is None.

        Raises:
            TypeError: If the alias generator produces an invalid type.
        """
        generator = getattr(self, alias_kind)
        if generator is None:
            return None
        
        alias = generator(field_name)
        if not isinstance(alias, allowed_types):
            raise TypeError(f"{alias_kind} must be one of {allowed_types}, not {type(alias)}")
        
        return alias

    def generate_aliases(self, field_name: str) -> tuple[str | None, str | AliasPath | AliasChoices | None, str | None]:
        """Generate `alias`, `validation_alias`, and `serialization_alias` for a field.

        Returns:
            A tuple of three aliases - validation, alias, and serialization.
        """
        alias = self._generate_alias('alias', (str,), field_name)
        validation_alias = self._generate_alias('validation_alias', (str, AliasPath, AliasChoices), field_name)
        serialization_alias = self._generate_alias('serialization_alias', (str,), field_name)
        
        return alias, validation_alias, serialization_alias
