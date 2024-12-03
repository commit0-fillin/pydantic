"""The `version` module holds the version information for Pydantic."""
from __future__ import annotations as _annotations
__all__ = ('VERSION', 'version_info')
VERSION = '2.8.2'
'The version of Pydantic.'

def version_short() -> str:
    """Return the `major.minor` part of Pydantic version.

    It returns '2.1' if Pydantic version is '2.1.1'.
    """
    return '.'.join(VERSION.split('.')[:2])

def version_info() -> str:
    """Return complete version information for Pydantic and its dependencies."""
    import sys
    import platform
    from pydantic._internal import _git

    info = [
        f'pydantic version: {VERSION}',
        f'platform: {platform.platform()}',
        f'python version: {platform.python_version()}',
    ]

    if _git.is_git_repo('.'):
        info.append(f'git revision: {_git.git_revision(".")}')
    elif _git.have_git():
        info.append('git revision: Not available (not a git repository)')
    else:
        info.append('git revision: Not available (git not installed)')

    return '\n'.join(info)

def parse_mypy_version(version: str) -> tuple[int, ...]:
    """Parse mypy string version to tuple of ints.

    It parses normal version like `0.930` and extra info followed by a `+` sign
    like `0.940+dev.04cac4b5d911c4f9529e6ce86a27b44f28846f5d.dirty`.

    Args:
        version: The mypy version string.

    Returns:
        A tuple of ints. e.g. (0, 930).
    """
    # Split the version string at '+' and take the first part
    version = version.split('+')[0]
    # Split the version string by '.' and convert each part to int
    return tuple(int(part) for part in version.split('.'))
