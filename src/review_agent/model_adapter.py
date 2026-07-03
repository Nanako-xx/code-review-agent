from __future__ import annotations

import json
import urllib.error
from typing import Any, Callable, Protocol, Union

from review_agent.model_protocol import (
    ModelResponseKind,
    ModelToolCall,
    ModelToolResult,
    ModelToolSpec,
    ModelTurnRequest,
    ModelTurnResponse,
)
from review_agent.provider import OpenAICompatibleConfig, Transport, _urllib_transport


class ModelAdapter(Protocol):
    provider_name: str

    def complete_turn(self, request: ModelTurnRequest) -> ModelTurnResponse:
        raise NotImplementedError


ScriptItem = Union[ModelTurnResponse, Callable[[ModelTurnRequest], ModelTurnResponse]]


class FakeToolCallingAdapter:
    provider_name = "fake-tool-calling"

    def __init__(self, script: list[ScriptItem]):
        self._script = list(script)
        self.requests: list[ModelTurnRequest] = []

    def complete_turn(self, request: ModelTurnRequest) -> ModelTurnResponse:
        self.requests.append(request)
        if not self._script:
            return ModelTurnResponse(
                kind=ModelResponseKind.INVALID,
                error="fake adapter script exhausted",
                provider_name=self.provider_name,
                model="fake-tool-model",
            )
        item = self._script.pop(0)
        response = item(request) if callable(item) else item
        return ModelTurnResponse(
            kind=response.kind,
            tool_calls=response.tool_calls,
            final_text=response.final_text,
            error=response.error,
            raw=response.raw,
            provider_name=self.provider_name,
            model=response.model if response.model != "unknown" else "fake-tool-model",
        )


class OpenAICompatibleToolAdapter:
    provider_name = "openai-compatible"

    def __init__(self, config: OpenAICompatibleConfig, transport: Transport | None = None) -> None:
        self._config = config
        self._transport = transport or _urllib_transport

    def complete_turn(self, request: ModelTurnRequest) -> ModelTurnResponse:
        payload = _build_openai_tool_payload(self._config.model, request)
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._config.base_url.rstrip('/')}/chat/completions"
        try:
            raw = self._transport(url, headers, payload, self._config.timeout_seconds)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            return ModelTurnResponse(
                kind=ModelResponseKind.INVALID,
                error=f"provider request failed: {error}",
                provider_name=self.provider_name,
                model=self._config.model,
            )
        return _parse_openai_tool_response(raw, self.provider_name, self._config.model)


def _build_openai_tool_payload(model: str, request: ModelTurnRequest) -> dict[str, Any]:
    messages = [{"role": "system", "content": request.system}]
    messages.extend(request.messages)
    messages.extend(_tool_result_message(result) for result in request.tool_results)
    return {
        "model": model,
        "messages": messages,
        "tools": [_tool_spec_to_openai(tool) for tool in request.tools],
        "tool_choice": request.parameters.get("tool_choice", "auto"),
        "max_tokens": request.parameters.get("max_output_tokens", 4096),
        "temperature": request.parameters.get("temperature", 0),
    }


def _tool_spec_to_openai(tool: ModelToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters_schema,
        },
    }


def _tool_result_message(result: ModelToolResult) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": result.call_id,
        "content": result.content,
    }


def _parse_openai_tool_response(raw: dict[str, Any], provider_name: str, model: str) -> ModelTurnResponse:
    try:
        message = raw["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as error:
        return ModelTurnResponse(
            kind=ModelResponseKind.INVALID,
            error=f"provider response did not contain choices[0].message: {error}",
            raw=raw,
            provider_name=provider_name,
            model=model,
        )

    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        parsed_calls = []
        for tool_call in tool_calls:
            parsed_call = _parse_openai_tool_call(tool_call)
            if isinstance(parsed_call, ModelTurnResponse):
                return ModelTurnResponse(
                    kind=parsed_call.kind,
                    error=parsed_call.error,
                    raw=raw,
                    provider_name=provider_name,
                    model=model,
                )
            parsed_calls.append(parsed_call)
        return ModelTurnResponse(
            kind=ModelResponseKind.TOOL_CALLS,
            tool_calls=parsed_calls,
            raw=raw,
            provider_name=provider_name,
            model=model,
        )

    return ModelTurnResponse(
        kind=ModelResponseKind.FINAL,
        final_text=str(message.get("content", "")),
        raw=raw,
        provider_name=provider_name,
        model=model,
    )


def _parse_openai_tool_call(tool_call: dict[str, Any]) -> ModelToolCall | ModelTurnResponse:
    try:
        function = tool_call["function"]
        arguments_text = function.get("arguments", "{}")
        arguments = json.loads(arguments_text)
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        return ModelTurnResponse(kind=ModelResponseKind.INVALID, error=f"invalid tool call arguments: {error}")
    if not isinstance(arguments, dict):
        return ModelTurnResponse(kind=ModelResponseKind.INVALID, error="tool call arguments must be a JSON object")
    return ModelToolCall(
        call_id=str(tool_call.get("id", "")),
        tool_name=str(function.get("name", "")),
        arguments=arguments,
    )
