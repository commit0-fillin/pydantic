"""Git utilities, adopted from mypy's git utilities (https://github.com/python/mypy/blob/master/mypy/git.py)."""
from __future__ import annotations
import os
import subprocess

def is_git_repo(dir: str) -> bool:
    """Is the given directory version-controlled with git?"""
    return os.path.isdir(os.path.join(dir, '.git'))

def have_git() -> bool:
    """Can we run the git executable?"""
    try:
        subprocess.check_output(['git', '--version'], stderr=subprocess.DEVNULL)
        return True
    except (subprocess.CalledProcessError, OSError):
        return False

def git_revision(dir: str) -> str:
    """Get the SHA-1 of the HEAD of a git repository."""
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=dir, stderr=subprocess.DEVNULL).decode('ascii').strip()
    except (subprocess.CalledProcessError, OSError):
        return 'unknown'
