from __future__ import annotations

from dataclasses import dataclass
import json
import urllib.error
import urllib.request
from typing import Any, Callable, Protocol, Union

from review_agent.model_protocol import (
    ModelResponseKind,
    ModelToolCall,
    ModelToolResult,
    ModelToolSpec,
    ModelTurnRequest,
    ModelTurnResponse,
)


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


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: int = 60


Transport = Callable[[str, dict[str, str], dict[str, Any], int], dict[str, Any]]


def _urllib_transport(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_seconds: int,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url=url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


class OpenAICompatibleToolAdapter:
    provider_name = "openai-compatible"

    def __init__(self, config: OpenAICompatibleConfig, transport: Transport | None = None) -> None:
        self._config = config
        self._transport = transport or _urllib_transport
        self._assistant_tool_call_messages: list[dict[str, Any]] = []

    def complete_turn(self, request: ModelTurnRequest) -> ModelTurnResponse:
        payload = _build_openai_tool_payload(
            self._config.model,
            request,
            self._assistant_tool_call_messages,
        )
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._config.base_url.rstrip('/')}/chat/completions"
        try:
            raw = self._transport(url, headers, payload, self._config.timeout_seconds)
        except json.JSONDecodeError as error:
            return ModelTurnResponse(
                kind=ModelResponseKind.INVALID,
                error=f"provider response was not valid JSON: {error}",
                provider_name=self.provider_name,
                model=self._config.model,
            )
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            return ModelTurnResponse(
                kind=ModelResponseKind.INVALID,
                error=f"provider request failed: {error}",
                provider_name=self.provider_name,
                model=self._config.model,
            )
        response = _parse_openai_tool_response(raw, self.provider_name, self._config.model)
        if response.kind is ModelResponseKind.TOOL_CALLS:
            self._assistant_tool_call_messages.append(_assistant_tool_call_message(response.tool_calls))
        return response


def _build_openai_tool_payload(
    model: str,
    request: ModelTurnRequest,
    assistant_tool_call_messages: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    messages = [{"role": "system", "content": request.system}]
    messages.extend(request.messages)
    if request.tool_results:
        messages.extend(
            _assistant_and_tool_result_messages(
                request.tool_results,
                assistant_tool_call_messages or [],
            )
        )
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


def _assistant_tool_call_message(tool_calls: list[ModelToolCall]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call.call_id,
                "type": "function",
                "function": {
                    "name": call.tool_name,
                    "arguments": json.dumps(call.arguments),
                },
            }
            for call in tool_calls
        ],
    }


def _assistant_and_tool_result_messages(
    tool_results: list[ModelToolResult],
    assistant_tool_call_messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results_by_call_id = {result.call_id: result for result in tool_results}
    matched_call_ids: set[str] = set()
    messages: list[dict[str, Any]] = []

    for assistant_message in assistant_tool_call_messages:
        tool_calls = assistant_message.get("tool_calls", [])
        matching_call_ids = [tool_call.get("id") for tool_call in tool_calls if tool_call.get("id") in results_by_call_id]
        if not matching_call_ids:
            continue

        messages.append(assistant_message)
        for call_id in matching_call_ids:
            if call_id is None:
                continue
            messages.append(_tool_result_message(results_by_call_id[call_id]))
            matched_call_ids.add(call_id)

    messages.extend(
        _tool_result_message(result)
        for result in tool_results
        if result.call_id not in matched_call_ids
    )
    return messages


def _parse_openai_tool_response(raw: dict[str, Any], provider_name: str, model: str) -> ModelTurnResponse:
    if not isinstance(raw, dict):
        return ModelTurnResponse(
            kind=ModelResponseKind.INVALID,
            error="provider response must be an object",
            raw={"raw": raw},
            provider_name=provider_name,
            model=model,
        )

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

    if not isinstance(message, dict):
        return ModelTurnResponse(
            kind=ModelResponseKind.INVALID,
            error="provider response message must be an object",
            raw=raw,
            provider_name=provider_name,
            model=model,
        )

    tool_calls = message.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        return ModelTurnResponse(
            kind=ModelResponseKind.INVALID,
            error="provider response tool_calls must be a list",
            raw=raw,
            provider_name=provider_name,
            model=model,
        )
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


def _parse_openai_tool_call(tool_call: object) -> ModelToolCall | ModelTurnResponse:
    if not isinstance(tool_call, dict):
        return ModelTurnResponse(kind=ModelResponseKind.INVALID, error="tool call must be an object")

    function = tool_call.get("function")
    if not isinstance(function, dict):
        return ModelTurnResponse(kind=ModelResponseKind.INVALID, error="tool call function must be an object")

    arguments_text = function.get("arguments", "{}")
    if not isinstance(arguments_text, str):
        return ModelTurnResponse(
            kind=ModelResponseKind.INVALID,
            error="tool call function arguments must be a JSON string",
        )

    try:
        arguments = json.loads(arguments_text)
    except json.JSONDecodeError as error:
        return ModelTurnResponse(kind=ModelResponseKind.INVALID, error=f"invalid tool call arguments: {error}")
    if not isinstance(arguments, dict):
        return ModelTurnResponse(kind=ModelResponseKind.INVALID, error="tool call arguments must be a JSON object")
    return ModelToolCall(
        call_id=str(tool_call.get("id", "")),
        tool_name=str(function.get("name", "")),
        arguments=arguments,
    )
