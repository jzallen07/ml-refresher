from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Awaitable

from anthropic import AsyncAnthropic

from agent.tools import ToolRegistry


@dataclass
class AgentResponse:
    text: str
    tools_called: list[str]
    stop_reason: str


class AgentLoop:
    def __init__(
        self,
        client: AsyncAnthropic,
        model: str,
        registry: ToolRegistry,
        on_text: Callable[[str], None] | None = None,
        on_tool_start: Callable[[str, dict], None] | None = None,
        on_tool_end: Callable[[str, dict], None] | None = None,
    ):
        self._client = client
        self._model = model
        self._registry = registry
        self._on_text = on_text
        self._on_tool_start = on_tool_start
        self._on_tool_end = on_tool_end

        self._system_prompt: str = ""
        self._messages: list[dict] = []
        self._available_tools: list[str] = []

    def set_system_prompt(self, prompt: str):
        self._system_prompt = prompt

    def set_available_tools(self, names: list[str]):
        self._available_tools = names

    def add_user_message(self, text: str):
        self._messages.append({"role": "user", "content": text})

    async def step(self, tool_choice: dict | None = None) -> AgentResponse:
        tools = self._registry.get_schemas(self._available_tools)

        kwargs: dict = {
            "model": self._model,
            "max_tokens": 4096,
            "system": self._system_prompt,
            "messages": self._messages,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        # Stream the response
        collected_text = ""
        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                collected_text += text
                if self._on_text:
                    self._on_text(text)

        message = await stream.get_final_message()

        # Build assistant message content blocks
        assistant_content = []
        for block in message.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        self._messages.append({"role": "assistant", "content": assistant_content})

        # Execute tool calls
        tools_called = []
        tool_results = []
        for block in message.content:
            if block.type != "tool_use":
                continue

            tool_name = block.name
            tool_input = block.input
            tools_called.append(tool_name)

            if self._on_tool_start:
                self._on_tool_start(tool_name, tool_input)

            tool = self._registry.get(tool_name)
            if tool:
                try:
                    result = await tool.execute(tool_input)
                except Exception as e:
                    result = {"error": str(e)}
            else:
                result = {"error": f"Unknown tool: {tool_name}"}

            if self._on_tool_end:
                self._on_tool_end(tool_name, result)

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })

        if tool_results:
            self._messages.append({"role": "user", "content": tool_results})

        return AgentResponse(
            text=collected_text,
            tools_called=tools_called,
            stop_reason=message.stop_reason,
        )
