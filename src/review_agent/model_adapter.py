from __future__ import annotations

from dataclasses import dataclass
import json
import math
import queue
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, NoReturn, Protocol, Union, cast

from review_agent.model_protocol import (
    ModelResponseKind,
    ModelToolCall,
    ModelToolResult,
    ModelToolSpec,
    ModelTurnRequest,
    ModelTurnResponse,
)
from review_agent.tool_result_protocol import (
    parse_tool_result_envelope,
    serialize_tool_result_envelope,
)


DEFAULT_MAX_RESPONSE_BYTES = 16 * 1024 * 1024
MAX_ALLOWED_RESPONSE_BYTES = 256 * 1024 * 1024
DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS = 180
PROVIDER_RESPONSE_TOO_LARGE_ERROR = (
    "provider response exceeded configured max_response_bytes"
)
MAX_HTTP_DEADLINE_WORKERS = 32
MAX_HTTP_CLOSE_WORKERS = 32
_TOOL_RESULT_METADATA_MISMATCH_DIAGNOSTIC = (
    "tool result metadata does not match transcript"
)
_HTTP_DEADLINE_SLOTS = threading.BoundedSemaphore(MAX_HTTP_DEADLINE_WORKERS)
_HTTP_CLOSE_SLOTS = threading.BoundedSemaphore(MAX_HTTP_CLOSE_WORKERS)


def provider_transport_projection() -> dict[str, Any]:
    return {
        "openai_compatible": {
            "request_timeout_seconds": (
                DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS
            ),
            "max_response_bytes": DEFAULT_MAX_RESPONSE_BYTES,
        },
    }


def _tool_result_metadata_mismatch() -> NoReturn:
    raise ValueError(_TOOL_RESULT_METADATA_MISMATCH_DIAGNOSTIC) from None


def _validate_max_response_bytes(value: object, context: str) -> int:
    if (
        type(value) is not int
        or value < 1
        or value > MAX_ALLOWED_RESPONSE_BYTES
    ):
        raise ValueError(
            f"{context} must be a positive integer no greater than "
            f"{MAX_ALLOWED_RESPONSE_BYTES}"
        )
    return value


@dataclass(frozen=True)
class ModelAdapterCapabilities:
    supports_tool_choice_none: bool
    enforces_request_timeout: bool
    max_response_bytes: int | None

    def __post_init__(self) -> None:
        if type(self.supports_tool_choice_none) is not bool:
            raise ValueError("supports_tool_choice_none must be a boolean")
        if type(self.enforces_request_timeout) is not bool:
            raise ValueError("enforces_request_timeout must be a boolean")
        if self.max_response_bytes is not None:
            _validate_max_response_bytes(
                self.max_response_bytes,
                "capabilities.max_response_bytes",
            )

    @property
    def tool_choice_none(self) -> bool:
        """Whether the adapter can honor an explicit ``tool_choice=none``."""

        return self.supports_tool_choice_none

    @property
    def request_timeout(self) -> bool:
        """Whether the transport enforces the supplied request timeout."""

        return self.enforces_request_timeout

    @property
    def response_byte_limit(self) -> int | None:
        """The enforced response-byte ceiling, or ``None`` when unproven."""

        return self.max_response_bytes

    def to_dict(self) -> dict[str, bool | int | None]:
        """Return the canonical, persistence-safe capability attestation."""

        return {
            "tool_choice_none": self.tool_choice_none,
            "request_timeout": self.request_timeout,
            "response_byte_limit": self.response_byte_limit,
        }


class ModelAdapter(Protocol):
    provider_name: str

    @property
    def capabilities(self) -> ModelAdapterCapabilities:
        raise NotImplementedError

    def complete_turn(self, request: ModelTurnRequest) -> ModelTurnResponse:
        raise NotImplementedError


ScriptItem = Union[ModelTurnResponse, Callable[[ModelTurnRequest], ModelTurnResponse]]


class FakeToolCallingAdapter:
    provider_name = "fake-tool-calling"
    capabilities = ModelAdapterCapabilities(
        supports_tool_choice_none=True,
        enforces_request_timeout=False,
        max_response_bytes=None,
    )

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
    timeout_seconds: int = DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES

    def __post_init__(self) -> None:
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be a positive finite number")
        _validate_max_response_bytes(
            self.max_response_bytes,
            "max_response_bytes",
        )


Transport = Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]]


class _ProviderResponseTooLargeError(OSError):
    pass


def _urllib_transport(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    timeout_seconds: float,
    *,
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
) -> dict[str, Any]:
    max_response_bytes = _validate_max_response_bytes(
        max_response_bytes,
        "max_response_bytes",
    )
    if (
        isinstance(timeout_seconds, bool)
        or not isinstance(timeout_seconds, (int, float))
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("timeout_seconds must be a positive finite number")
    timeout_seconds = float(timeout_seconds)
    deadline = time.monotonic() + timeout_seconds
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url=url, data=data, headers=headers, method="POST")
    outcome: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=1)
    state_lock = threading.Lock()
    response_holder: list[object] = []
    timed_out = False
    response_close_claimed = False
    completed_at: float | None = None

    def close_response_once(response: object) -> None:
        nonlocal response_close_claimed
        with state_lock:
            if response_close_claimed:
                return
            response_close_claimed = True

        try:
            close = getattr(response, "close", None)
            if callable(close):
                close()
                return
            exit_context = getattr(response, "__exit__", None)
            if callable(exit_context):
                exit_context(None, None, None)
        except Exception:
            # Cleanup must not replace the transport result or extend the caller's
            # wall-clock wait. A timed-out response is closed on a daemon thread.
            return

    def close_response_async(response: object) -> None:
        if not _HTTP_CLOSE_SLOTS.acquire(blocking=False):
            return

        def close_with_slot_release() -> None:
            try:
                close_response_once(response)
            finally:
                _HTTP_CLOSE_SLOTS.release()

        close_worker: threading.Thread | None = None
        try:
            close_worker = threading.Thread(
                target=close_with_slot_release,
                name="model-adapter-http-timeout-close",
                daemon=True,
            )
            close_worker.start()
        except Exception:
            if close_worker is None or close_worker.ident is None:
                _HTTP_CLOSE_SLOTS.release()
            # The worker remains a daemon and will still attempt cleanup in its
            # finally block if a separate cleanup thread cannot be started.
            return

    def request_and_read() -> None:
        nonlocal completed_at
        response: object | None = None
        result: tuple[str, object] | None = None
        try:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "provider request exceeded its wall-clock timeout"
                )
            response = urllib.request.urlopen(request, timeout=timeout_seconds)
            with state_lock:
                response_holder.append(response)
                cancelled = timed_out
            if not cancelled:
                response_bytes = response.read(max_response_bytes + 1)
                if len(response_bytes) > max_response_bytes:
                    raise _ProviderResponseTooLargeError(
                        PROVIDER_RESPONSE_TOO_LARGE_ERROR
                    )
                result = (
                    "ok",
                    json.loads(response_bytes.decode("utf-8")),
                )
        except Exception as error:
            result = ("error", error)
        finally:
            try:
                if response is not None:
                    close_response_once(response)
                if result is not None:
                    outcome.put(result)
                with state_lock:
                    completed_at = time.monotonic()
            finally:
                _HTTP_DEADLINE_SLOTS.release()

    remaining = deadline - time.monotonic()
    if remaining <= 0 or not _HTTP_DEADLINE_SLOTS.acquire(timeout=remaining):
        raise TimeoutError("provider request exceeded its wall-clock timeout")
    if time.monotonic() >= deadline:
        _HTTP_DEADLINE_SLOTS.release()
        raise TimeoutError("provider request exceeded its wall-clock timeout")
    worker: threading.Thread | None = None
    try:
        worker = threading.Thread(
            target=request_and_read,
            name="model-adapter-http-deadline",
            daemon=True,
        )
        if time.monotonic() >= deadline:
            raise TimeoutError("provider request exceeded its wall-clock timeout")
        worker.start()
    except BaseException:
        if worker is None or worker.ident is None:
            _HTTP_DEADLINE_SLOTS.release()
        raise
    worker.join(max(0.0, deadline - time.monotonic()))
    with state_lock:
        completed_before_deadline = (
            completed_at is not None and completed_at <= deadline
        )
        if not completed_before_deadline:
            timed_out = True
            response_to_close = response_holder[0] if response_holder else None
        else:
            response_to_close = None
    if not completed_before_deadline:
        if response_to_close is not None:
            close_response_async(response_to_close)
        raise TimeoutError("provider request exceeded its wall-clock timeout")
    try:
        status, value = outcome.get_nowait()
    except queue.Empty as exc:
        raise OSError("provider request worker exited without a result") from exc
    if status == "error":
        if isinstance(value, Exception):
            raise value
        raise OSError("provider request worker returned an invalid error")
    return cast(dict[str, Any], value)


class OpenAICompatibleToolAdapter:
    provider_name = "openai-compatible"

    def __init__(
        self,
        config: OpenAICompatibleConfig,
        transport: Transport | None = None,
        *,
        capabilities: ModelAdapterCapabilities | None = None,
    ) -> None:
        if not isinstance(config, OpenAICompatibleConfig):
            raise TypeError("config must be an OpenAICompatibleConfig")
        if capabilities is not None and not isinstance(
            capabilities,
            ModelAdapterCapabilities,
        ):
            raise TypeError(
                "capabilities must be a ModelAdapterCapabilities or None"
            )
        if transport is None:
            if capabilities is not None:
                raise ValueError(
                    "capabilities may only be supplied with a custom transport"
                )
            resolved_capabilities = ModelAdapterCapabilities(
                supports_tool_choice_none=True,
                enforces_request_timeout=True,
                max_response_bytes=config.max_response_bytes,
            )
        else:
            resolved_capabilities = capabilities or ModelAdapterCapabilities(
                supports_tool_choice_none=True,
                enforces_request_timeout=False,
                max_response_bytes=None,
            )
            if (
                resolved_capabilities.max_response_bytes is not None
                and resolved_capabilities.max_response_bytes
                > config.max_response_bytes
            ):
                raise ValueError(
                    "custom transport max_response_bytes may not exceed "
                    "the configured max_response_bytes"
                )
        self._config = config
        self._transport = transport
        self._capabilities = resolved_capabilities

    @property
    def capabilities(self) -> ModelAdapterCapabilities:
        return self._capabilities

    def complete_turn(self, request: ModelTurnRequest) -> ModelTurnResponse:
        payload = _build_openai_tool_payload(self._config.model, request)
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._config.base_url.rstrip('/')}/chat/completions"
        timeout_seconds = _transport_timeout_seconds(request, self._config.timeout_seconds)
        try:
            if self._transport is None:
                raw = _urllib_transport(
                    url,
                    headers,
                    payload,
                    timeout_seconds,
                    max_response_bytes=self._config.max_response_bytes,
                )
            else:
                raw = self._transport(url, headers, payload, timeout_seconds)
        except _ProviderResponseTooLargeError:
            return ModelTurnResponse(
                kind=ModelResponseKind.INVALID,
                error=PROVIDER_RESPONSE_TOO_LARGE_ERROR,
                provider_name=self.provider_name,
                model=self._config.model,
            )
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
        return _parse_openai_tool_response(raw, self.provider_name, self._config.model)


def _transport_timeout_seconds(
    request: ModelTurnRequest,
    configured_timeout_seconds: int,
) -> float:
    requested = request.parameters.get("timeout_seconds")
    if (
        isinstance(requested, bool)
        or not isinstance(requested, (int, float))
        or not math.isfinite(requested)
        or requested <= 0
    ):
        return float(configured_timeout_seconds)
    return float(min(configured_timeout_seconds, requested))


def _build_openai_tool_payload(
    model: str,
    request: ModelTurnRequest,
) -> dict[str, Any]:
    runtime_messages = [dict(message) for message in request.messages]
    _pair_tool_results_with_assistant_calls(
        runtime_messages,
        request.tool_results,
    )
    messages = [{"role": "system", "content": request.system}, *runtime_messages]
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": request.parameters.get("max_output_tokens", 4096),
        "temperature": request.parameters.get("temperature", 0),
    }
    response_format = request.parameters.get("response_format")
    if response_format is not None:
        if response_format != "json_object":
            raise ValueError("unsupported response_format")
        if request.tools:
            raise ValueError("json_object response_format requires a no-tool request")
        payload["response_format"] = {"type": "json_object"}
        return payload

    payload["tools"] = [_tool_spec_to_openai(tool) for tool in request.tools]
    payload["tool_choice"] = request.parameters.get("tool_choice", "auto")
    return payload


def _tool_spec_to_openai(tool: ModelToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters_schema,
        },
    }


def model_tool_result_to_message(result: ModelToolResult) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": result.call_id,
        "content": serialize_tool_result_envelope(result),
    }


def _pair_tool_results_with_assistant_calls(
    messages: list[dict[str, Any]],
    tool_results: list[ModelToolResult],
) -> None:
    has_message_tool_results = any(
        message.get("role") == "tool" for message in messages
    )

    assistant_batches: list[tuple[int, list[str]]] = []
    assistant_call_ids: set[str] = set()
    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if tool_calls is None:
            continue
        if not isinstance(tool_calls, list):
            raise ValueError("assistant tool_calls must be a list")
        batch_call_ids: list[str] = []
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                raise ValueError("assistant tool call must be an object")
            call_id = tool_call.get("id")
            if not isinstance(call_id, str) or not call_id.strip():
                raise ValueError("assistant tool call id must be non-empty")
            if call_id in assistant_call_ids:
                raise ValueError(
                    f"duplicate assistant tool call id {call_id!r}"
                )
            assistant_call_ids.add(call_id)
            batch_call_ids.append(call_id)
        if batch_call_ids:
            assistant_batches.append((message_index, batch_call_ids))

    if has_message_tool_results:
        message_tool_results = _validate_complete_tool_transcript(
            messages,
            assistant_batches,
            assistant_call_ids,
        )
        if tool_results:
            _validate_tool_result_metadata(
                message_tool_results,
                tool_results,
            )
        return

    _insert_legacy_tool_results(
        messages,
        assistant_batches,
        assistant_call_ids,
        tool_results,
    )


def _validate_complete_tool_transcript(
    messages: list[dict[str, Any]],
    assistant_batches: list[tuple[int, list[str]]],
    assistant_call_ids: set[str],
) -> dict[str, ModelToolResult]:
    tool_message_ids: set[str] = set()
    tool_message_indices: dict[int, str] = {}
    tool_message_results: dict[str, ModelToolResult] = {}
    for message_index, message in enumerate(messages):
        if message.get("role") != "tool":
            continue
        call_id = message.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("tool result call id must be non-empty")
        if call_id in tool_message_ids:
            raise ValueError(f"duplicate tool result call id {call_id!r}")
        if call_id not in assistant_call_ids:
            raise ValueError(
                f"tool result {call_id!r} has no matching assistant tool call"
            )
        tool_message_ids.add(call_id)
        tool_message_indices[message_index] = call_id
        tool_message_results[call_id] = parse_tool_result_envelope(
            call_id,
            message.get("content"),
        )

    for call_id in assistant_call_ids:
        if call_id not in tool_message_ids:
            raise ValueError(
                f"missing tool result for assistant call {call_id!r}"
            )

    expected_tool_indices: set[int] = set()
    for assistant_index, batch_call_ids in assistant_batches:
        for offset, call_id in enumerate(batch_call_ids, start=1):
            tool_index = assistant_index + offset
            if (
                tool_index >= len(messages)
                or messages[tool_index].get("role") != "tool"
                or messages[tool_index].get("tool_call_id") != call_id
            ):
                raise ValueError(
                    f"tool result for assistant call {call_id!r} must be adjacent"
                )
            expected_tool_indices.add(tool_index)

    unexpected_indices = set(tool_message_indices) - expected_tool_indices
    if unexpected_indices:
        call_id = tool_message_indices[min(unexpected_indices)]
        raise ValueError(
            f"tool result for assistant call {call_id!r} must be adjacent"
        )

    return tool_message_results


def _validate_tool_result_metadata(
    message_tool_results: dict[str, ModelToolResult],
    tool_results: list[ModelToolResult],
) -> None:
    metadata_by_call_id: dict[str, ModelToolResult] = {}
    for result in tool_results:
        if not isinstance(result, ModelToolResult):
            _tool_result_metadata_mismatch()
        call_id = result.call_id
        if not isinstance(call_id, str) or not call_id.strip():
            _tool_result_metadata_mismatch()
        if call_id in metadata_by_call_id:
            _tool_result_metadata_mismatch()
        metadata_by_call_id[call_id] = result

    if set(metadata_by_call_id) != set(message_tool_results):
        _tool_result_metadata_mismatch()

    for call_id, result in metadata_by_call_id.items():
        try:
            metadata_canonical = serialize_tool_result_envelope(result)
            transcript_canonical = serialize_tool_result_envelope(
                message_tool_results[call_id]
            )
        except Exception:
            _tool_result_metadata_mismatch()
        if metadata_canonical != transcript_canonical:
            _tool_result_metadata_mismatch()


def _insert_legacy_tool_results(
    messages: list[dict[str, Any]],
    assistant_batches: list[tuple[int, list[str]]],
    assistant_call_ids: set[str],
    tool_results: list[ModelToolResult],
) -> None:
    results_by_call_id: dict[str, ModelToolResult] = {}
    for result in tool_results:
        call_id = result.call_id
        if not isinstance(call_id, str) or not call_id.strip():
            raise ValueError("tool result call id must be non-empty")
        if call_id in results_by_call_id:
            raise ValueError(f"duplicate tool result call id {call_id!r}")
        if call_id not in assistant_call_ids:
            raise ValueError(
                f"tool result {call_id!r} has no matching assistant tool call"
            )
        results_by_call_id[call_id] = result

    for call_id in assistant_call_ids:
        if call_id not in results_by_call_id:
            raise ValueError(
                f"missing tool result for assistant call {call_id!r}"
            )

    for assistant_index, batch_call_ids in reversed(assistant_batches):
        messages[assistant_index + 1:assistant_index + 1] = [
            model_tool_result_to_message(results_by_call_id[call_id])
            for call_id in batch_call_ids
        ]


def model_response_to_assistant_message(
    response: ModelTurnResponse,
) -> dict[str, Any]:
    source_message: dict[str, Any] = {}
    try:
        candidate = response.raw["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        candidate = None
    if isinstance(candidate, dict):
        source_message = candidate

    fallback_content = (
        response.final_text
        if response.kind is ModelResponseKind.FINAL
        else None
    )
    content = source_message.get("content", fallback_content)
    message = {
        "role": "assistant",
        "content": (
            content
            if isinstance(content, (str, list))
            else fallback_content
        ),
    }
    if response.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.call_id,
                "type": "function",
                "function": {
                    "name": call.tool_name,
                    "arguments": json.dumps(call.arguments),
                },
            }
            for call in response.tool_calls
        ]
    reasoning_content = source_message.get("reasoning_content")
    if isinstance(reasoning_content, str):
        message["reasoning_content"] = reasoning_content
    return message


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
