'''
- modified from tinygrad/docs/env_vars.md
'''

from __future__ import annotations
import os, contextlib
from typing import Dict, List, ClassVar, Any, TypeVar, Generic, Optional, overload


T = TypeVar('T')

@overload
def getenv(key: str, default: T, cast_to: None = None) -> T: ...

@overload  
def getenv(key: str, default: Any, cast_to: type[T]) -> T: ...

@overload
def getenv(key: str, default: Any = 0, cast_to: Optional[type] = None) -> Any: ...

def getenv(key: str, default: Any = 0, cast_to: Optional[type] = None) -> Any:
    val = os.getenv(key)
    if val is None: return default
    
    # Use cast_to if provided, otherwise use type of default
    target_type = cast_to if cast_to is not None else type(default)
    try:
        return target_type(val)
    except (ValueError, TypeError):
        return val



class Context(contextlib.ContextDecorator):
    stack: ClassVar[List[dict[str, int]]] = [{}]

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __enter__(self):
        Context.stack[-1] = {
            k: o.value
            for k, o in ContextVar._cache.items()
        }  # Store current state.
        for k, v in self.kwargs.items():
            ContextVar._cache[k].value = v  # Update to new temporary state.
        Context.stack.append(
            self.kwargs
        )  # Store the temporary state so we know what to undo later.

    def __exit__(self, *args):
        for k in Context.stack.pop():
            ContextVar._cache[k].value = Context.stack[-1].get(
                k, ContextVar._cache[k].value)


T = TypeVar('T')

class ContextVar(Generic[T]):
    _cache: ClassVar[Dict[str, ContextVar[Any]]] = {}
    value: T

    def __new__(cls, key: str, default_value: T, cast_to: type[T] = None) -> ContextVar[T]:
        if key in ContextVar._cache: return ContextVar._cache[key]
        instance = ContextVar._cache[key] = super().__new__(cls)
        instance.value = getenv(key, default_value, cast_to)
        return instance

    def __bool__(self) -> bool:
        return bool(self.value)

    def __ge__(self, x) -> bool:
        return self.value >= x

    def __gt__(self, x) -> bool:
        return self.value > x

    def __lt__(self, x) -> bool:
        return self.value < x


DEV = ContextVar("DEV", 0)
DEBUG = ContextVar("DEBUG", 0)
TRACING = ContextVar("TRACING", 0)
VERBOSE = ContextVar("VERBOSE", 0)
STRICT = ContextVar("STRICT", 0)