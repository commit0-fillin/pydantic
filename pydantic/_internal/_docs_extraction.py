"""Utilities related to attribute docstring extraction."""
from __future__ import annotations
import ast
import inspect
import textwrap
from typing import Any

class DocstringVisitor(ast.NodeVisitor):

    def __init__(self) -> None:
        super().__init__()
        self.target: str | None = None
        self.attrs: dict[str, str] = {}
        self.previous_node_type: type[ast.AST] | None = None

def extract_docstrings_from_cls(cls: type[Any], use_inspect: bool=False) -> dict[str, str]:
    """Map model attributes and their corresponding docstring.

    Args:
        cls: The class of the Pydantic model to inspect.
        use_inspect: Whether to skip usage of frames to find the object and use
            the `inspect` module instead.

    Returns:
        A mapping containing attribute names and their corresponding docstring.
    """
    if use_inspect:
        source = inspect.getsource(cls)
    else:
        # Get the source code from the frame
        frame = inspect.currentframe()
        try:
            while frame:
                if frame.f_code.co_name == '<module>':
                    source = ''.join(frame.f_locals.get('__source__', ''))
                    break
                frame = frame.f_back
            else:
                raise ValueError("Could not find source code")
        finally:
            del frame

    # Parse the source code
    tree = ast.parse(textwrap.dedent(source))

    # Use the DocstringVisitor to extract docstrings
    visitor = DocstringVisitor()
    visitor.visit(tree)

    return visitor.attrs
