"""Alias generators for converting between different capitalization conventions."""
import re
__all__ = ('to_pascal', 'to_camel', 'to_snake')

def to_pascal(snake: str) -> str:
    """Convert a snake_case string to PascalCase.

    Args:
        snake: The string to convert.

    Returns:
        The PascalCase string.
    """
    return ''.join(word.capitalize() for word in snake.split('_'))

def to_camel(snake: str) -> str:
    """Convert a snake_case string to camelCase.

    Args:
        snake: The string to convert.

    Returns:
        The converted camelCase string.
    """
    pascal = to_pascal(snake)
    return pascal[0].lower() + pascal[1:]

def to_snake(camel: str) -> str:
    """Convert a PascalCase, camelCase, or kebab-case string to snake_case.

    Args:
        camel: The string to convert.

    Returns:
        The converted string in snake_case.
    """
    # Replace hyphens with underscores
    s = camel.replace('-', '_')
    # Insert an underscore before any uppercase letter
    # that is preceded by a lowercase letter or number
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', s)
    # Convert to lowercase
    return s.lower()
