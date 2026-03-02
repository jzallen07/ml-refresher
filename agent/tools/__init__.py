from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Awaitable


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    execute: Callable[..., Awaitable[dict]]

    @property
    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_schemas(self, names: list[str]) -> list[dict]:
        return [self._tools[n].schema for n in names if n in self._tools]

    @property
    def all_names(self) -> list[str]:
        return list(self._tools.keys())
