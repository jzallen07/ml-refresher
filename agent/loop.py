from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Callable

from openrouter import OpenRouter
from openrouter.errors import (
    TooManyRequestsResponseError,
    RequestTimeoutResponseError,
    ServiceUnavailableResponseError,
    ProviderOverloadedResponseError,
)
from openrouter.utils.eventstreaming import EventStreamAsync

from agent.tools import ToolRegistry

_RETRYABLE_ERRORS = (
    TooManyRequestsResponseError,
    RequestTimeoutResponseError,
    ServiceUnavailableResponseError,
    ProviderOverloadedResponseError,
)
_MAX_RETRIES = 3


@dataclass
class AgentResponse:
    text: str
    tools_called: list[str]
    stop_reason: str
    tool_results_data: dict[str, dict] = field(default_factory=dict)


class AgentLoop:
    def __init__(
        self,
        client: OpenRouter,
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

    def _build_messages(self) -> list[dict]:
        msgs: list[dict] = []
        if self._system_prompt:
            msgs.append({"role": "system", "content": self._system_prompt})
        msgs.extend(self._messages)
        return msgs

    async def step(self, tool_choice: dict | None = None) -> AgentResponse:
        tools = self._registry.get_schemas(self._available_tools)

        kwargs: dict = {
            "model": self._model,
            "max_tokens": 4096,
            "messages": self._build_messages(),
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice

        # Stream the response with retry on transient errors
        collected_text = ""
        # Accumulate tool calls from streaming deltas
        tool_call_accum: dict[int, dict] = {}
        finish_reason = "stop"

        for attempt in range(_MAX_RETRIES):
            try:
                response = await self._client.chat.send_async(**kwargs)

                # Handle streaming response
                if isinstance(response, EventStreamAsync):
                    async with response:
                        async for chunk in response:
                            if not chunk.choices:
                                continue
                            choice = chunk.choices[0]

                            if choice.finish_reason is not None:
                                finish_reason = str(choice.finish_reason)

                            delta = choice.delta
                            if delta.content:
                                collected_text += delta.content
                                if self._on_text:
                                    self._on_text(delta.content)

                            if delta.tool_calls:
                                for tc in delta.tool_calls:
                                    idx = int(tc.index)
                                    if idx not in tool_call_accum:
                                        tool_call_accum[idx] = {
                                            "id": getattr(tc, "id", None) or "",
                                            "name": "",
                                            "arguments": "",
                                        }
                                    entry = tool_call_accum[idx]
                                    if tc.id:
                                        entry["id"] = tc.id
                                    if tc.function:
                                        if tc.function.name:
                                            entry["name"] += tc.function.name
                                        if tc.function.arguments:
                                            entry["arguments"] += tc.function.arguments
                else:
                    # Non-streaming response
                    choice = response.choices[0]
                    finish_reason = str(choice.finish_reason) if choice.finish_reason else "stop"
                    if choice.message.content:
                        collected_text = choice.message.content
                        if self._on_text:
                            self._on_text(collected_text)
                    if choice.message.tool_calls:
                        for i, tc in enumerate(choice.message.tool_calls):
                            tool_call_accum[i] = {
                                "id": tc.id,
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                break
            except _RETRYABLE_ERRORS:
                if attempt == _MAX_RETRIES - 1:
                    raise
                collected_text = ""
                tool_call_accum = {}
                if self._on_text:
                    self._on_text("\n\n*[Retrying...]*\n\n")
                await asyncio.sleep(2 ** attempt)

        # Build assistant message for history
        assistant_msg: dict = {"role": "assistant"}
        if collected_text:
            assistant_msg["content"] = collected_text
        if tool_call_accum:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    },
                }
                for tc in tool_call_accum.values()
            ]
        self._messages.append(assistant_msg)

        # Execute tool calls
        tools_called = []
        tool_results_data: dict[str, dict] = {}
        for tc_data in tool_call_accum.values():
            tool_name = tc_data["name"]
            tool_call_id = tc_data["id"]
            tools_called.append(tool_name)

            try:
                tool_input = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
            except json.JSONDecodeError:
                tool_input = {}

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

            tool_results_data[tool_name] = result

            if self._on_tool_end:
                self._on_tool_end(tool_name, result)

            self._messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(result),
            })

        return AgentResponse(
            text=collected_text,
            tools_called=tools_called,
            stop_reason=finish_reason,
            tool_results_data=tool_results_data,
        )
