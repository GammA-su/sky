"""Simple registry helper."""

from __future__ import annotations

from typing import Any, Callable


class Registry:
    def __init__(self) -> None:
        self._items: dict[str, Any] = {}

    def register(self, name: str) -> Callable[[Any], Any]:
        def decorator(obj: Any) -> Any:
            self._items[name] = obj
            return obj
        return decorator

    def get(self, name: str) -> Any:
        if name not in self._items:
            raise KeyError(f"Unknown registry key: {name}")
        return self._items[name]

    def items(self) -> dict[str, Any]:
        return dict(self._items)
