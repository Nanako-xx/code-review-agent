import json
import threading
import time
from dataclasses import FrozenInstanceError

import pytest

import review_agent.model_adapter as model_adapter_module
from review_agent.model_adapter import (
    MAX_ALLOWED_RESPONSE_BYTES,
    MAX_HTTP_DEADLINE_WORKERS,
    PROVIDER_RESPONSE_TOO_LARGE_ERROR,
    FakeToolCallingAdapter,
    ModelAdapterCapabilities,
    OpenAICompatibleConfig,
    OpenAICompatibleToolAdapter,
    _urllib_transport,
    model_response_to_assistant_message,
    model_tool_result_to_message,
    review_tool_projection_to_message,
)
from review_agent.model_protocol import (
    ModelResponse,
    ModelResponseKind,
    ModelToolCall,
    ModelToolResult,
    ModelToolSpec,
    ModelTurnRequest,
    ModelTurnResponse,
)
from review_agent.tool_result_protocol import serialize_tool_result_envelope
from review_agent.tool_result_protocol import (
    ReviewToolResult,
    ToolResultProjectionV2,
)


class _BoundedHttpResponse:
    def __init__(self, body: bytes):
        self.body = body
        self.read_sizes = []
        self.close_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def read(self, size=-1):
        self.read_sizes.append(size)
        return self.body if size is None or size < 0 else self.body[:size]

    def close(self):
        self.close_calls += 1


class _BlockingReadHttpResponse(_BoundedHttpResponse):
    def __init__(self, body: bytes):
        super().__init__(body)
        self.read_started = threading.Event()
        self.read_finished = threading.Event()
        self.closed = threading.Event()

    def read(self, size=-1):
        self.read_started.set()
        self.closed.wait(timeout=1.0)
        try:
            return super().read(size)
        finally:
            self.read_finished.set()

    def close(self):
        super().close()
        self.closed.set()


class _ExplodingUtf8Str(str):
    def encode(self, encoding="utf-8", errors="strict"):
        raise RuntimeError("sensitive malicious metadata error")


def make_request(tool_results=None):
    return ModelTurnRequest(
        system="system",
        tools=[],
        messages=[{"role": "user", "content": "Review change"}],
        tool_results=tool_results or [],
        parameters={"trace_id": "review-1-reviewer-0"},
    )


def test_model_response_lives_in_model_protocol():
    response = ModelResponse(
        content='{"status": "completed"}',
        provider_name="adapter",
        model="review-model",
        raw={"trace_id": "trace-1"},
    )

    assert response.provider_name == "adapter"
    assert response.raw["trace_id"] == "trace-1"


def test_openai_compatible_config_lives_in_model_adapter():
    assert OpenAICompatibleConfig.__module__ == "review_agent.model_adapter"


def test_model_adapter_capabilities_are_frozen():
    capabilities = ModelAdapterCapabilities(
        supports_tool_choice_none=True,
        enforces_request_timeout=False,
        max_response_bytes=None,
    )

    with pytest.raises(FrozenInstanceError):
        capabilities.enforces_request_timeout = True


def test_model_adapter_capabilities_expose_canonical_audit_fields():
    capabilities = ModelAdapterCapabilities(
        supports_tool_choice_none=True,
        enforces_request_timeout=True,
        max_response_bytes=2048,
    )

    assert capabilities.tool_choice_none is True
    assert capabilities.request_timeout is True
    assert capabilities.response_byte_limit == 2048
    assert capabilities.to_dict() == {
        "tool_choice_none": True,
        "request_timeout": True,
        "response_byte_limit": 2048,
    }


@pytest.mark.parametrize(
    "values",
    [
        {"supports_tool_choice_none": 1},
        {"enforces_request_timeout": "yes"},
        {"max_response_bytes": True},
        {"max_response_bytes": 0},
        {"max_response_bytes": MAX_ALLOWED_RESPONSE_BYTES + 1},
    ],
)
def test_model_adapter_capabilities_require_strict_values(values):
    parameters = {
        "supports_tool_choice_none": True,
        "enforces_request_timeout": True,
        "max_response_bytes": 1024,
        **values,
    }

    with pytest.raises(ValueError):
        ModelAdapterCapabilities(**parameters)


@pytest.mark.parametrize("timeout", [0, -1, float("nan")])
def test_openai_compatible_config_requires_finite_positive_timeout(timeout):
    with pytest.raises(ValueError, match="timeout_seconds"):
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
            timeout_seconds=timeout,
        )


@pytest.mark.parametrize(
    "max_response_bytes",
    [True, 0, -1, 1.5, "1024", MAX_ALLOWED_RESPONSE_BYTES + 1],
)
def test_openai_compatible_config_requires_bounded_positive_integer_response_limit(
    max_response_bytes,
):
    with pytest.raises(ValueError, match="max_response_bytes"):
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
            max_response_bytes=max_response_bytes,
        )


def test_fake_adapter_returns_scripted_tool_call():
    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.TOOL_CALLS,
                tool_calls=[ModelToolCall("call-1", "compare_base_head", {"path": "app.py"})],
            )
        ]
    )

    response = adapter.complete_turn(make_request())

    assert response.kind is ModelResponseKind.TOOL_CALLS
    assert response.tool_calls[0].tool_name == "compare_base_head"
    assert response.provider_name == "fake-tool-calling"


def test_fake_adapter_reports_only_capabilities_it_can_enforce():
    adapter = FakeToolCallingAdapter([])

    assert adapter.capabilities.supports_tool_choice_none is True
    assert adapter.capabilities.enforces_request_timeout is False
    assert adapter.capabilities.max_response_bytes is None


def test_fake_adapter_can_compute_final_response_from_request():
    def final_response(request):
        observation_id = request.tool_results[0].observation_ids[0]
        return ModelTurnResponse(
            kind=ModelResponseKind.FINAL,
            final_text=json.dumps(
                {
                    "contract_assessments": [
                        {
                            "contract": "regression_safety",
                            "status": "covered",
                            "summary": "Checked diff observation.",
                            "evidence_refs": [observation_id],
                        }
                    ],
                    "confirmed_findings": [],
                    "rejected_hypotheses": [],
                    "uncertainties": [],
                    "observation_refs": [observation_id],
                    "investigation_summary": "Used tool observation.",
                    "status": "completed",
                }
            ),
        )

    adapter = FakeToolCallingAdapter(script=[final_response])
    response = adapter.complete_turn(
        make_request(
            tool_results=[
                type(
                    "Result",
                    (),
                    {
                        "observation_ids": ["O-abc"],
                    },
                )()
            ]
        )
    )

    assert response.kind is ModelResponseKind.FINAL
    assert "O-abc" in response.final_text


def test_openai_compatible_adapter_converts_tool_call_response():
    captured = {}

    def transport(url, headers, payload, timeout_seconds):
        captured["payload"] = payload
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "read_range",
                                    "arguments": (
                                        '{"path": "app.py", "revision": "head", '
                                        '"line_start": 1, "line_end": 10}'
                                    ),
                                },
                            }
                        ]
                    }
                }
            ]
        }

    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
        ),
        transport=transport,
    )
    request = ModelTurnRequest(
        system="system",
        tools=[
            ModelToolSpec(
                name="read_range",
                description="Read range",
                parameters_schema={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            )
        ],
        messages=[{"role": "user", "content": "Review"}],
        tool_results=[],
        parameters={"max_output_tokens": 1000, "temperature": 0},
    )

    response = adapter.complete_turn(request)

    assert captured["payload"]["tools"][0]["function"]["name"] == "read_range"
    assert response.kind is ModelResponseKind.TOOL_CALLS
    assert response.tool_calls[0].tool_name == "read_range"
    assert response.tool_calls[0].arguments["path"] == "app.py"


def test_openai_compatible_adapter_converts_final_text_response():
    def transport(url, headers, payload, timeout_seconds):
        return {"choices": [{"message": {"content": '{"status": "partial"}'}}]}

    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
        ),
        transport=transport,
    )

    response = adapter.complete_turn(make_request())

    assert response.kind is ModelResponseKind.FINAL
    assert response.final_text == '{"status": "partial"}'
    assert response.provider_name == "openai-compatible"


def test_openai_compatible_adapter_maps_explicit_json_finalization_without_tools():
    captured = {}

    def transport(url, headers, payload, timeout_seconds):
        captured["payload"] = payload
        return {"choices": [{"message": {"content": '{"status": "partial"}'}}]}

    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
        ),
        transport=transport,
    )
    request = ModelTurnRequest(
        system="Return JSON.",
        tools=[],
        messages=[{"role": "user", "content": "Finalize the review as JSON."}],
        tool_results=[],
        parameters={
            "max_output_tokens": 1000,
            "temperature": 0,
            "tool_choice": "none",
            "response_format": "json_object",
        },
    )

    adapter.complete_turn(request)

    assert "tools" not in captured["payload"]
    assert "tool_choice" not in captured["payload"]
    assert captured["payload"]["response_format"] == {"type": "json_object"}


def test_openai_compatible_adapter_does_not_enable_json_mode_for_tool_turns():
    captured = {}

    def transport(url, headers, payload, timeout_seconds):
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "done"}}]}

    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
        ),
        transport=transport,
    )
    request = ModelTurnRequest(
        system="system",
        tools=[
            ModelToolSpec(
                name="read_range",
                description="Read range",
                parameters_schema={"type": "object", "properties": {}},
            )
        ],
        messages=[{"role": "user", "content": "Review"}],
        tool_results=[],
        parameters={
            "tool_choice": "auto",
            "response_schema": "reviewer_assignment_result_v2",
        },
    )

    adapter.complete_turn(request)

    assert captured["payload"]["tools"]
    assert captured["payload"]["tool_choice"] == "auto"
    assert "response_format" not in captured["payload"]


def test_default_transport_capabilities_bind_configured_response_limit():
    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
            max_response_bytes=4096,
        )
    )

    assert adapter.capabilities == ModelAdapterCapabilities(
        supports_tool_choice_none=True,
        enforces_request_timeout=True,
        max_response_bytes=4096,
    )


def test_custom_transport_defaults_to_unproven_transport_capabilities():
    def transport(url, headers, payload, timeout_seconds):
        return {"choices": [{"message": {"content": "ok"}}]}

    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
        ),
        transport=transport,
    )

    assert adapter.capabilities == ModelAdapterCapabilities(
        supports_tool_choice_none=True,
        enforces_request_timeout=False,
        max_response_bytes=None,
    )


def test_custom_transport_can_explicitly_prove_strict_capabilities():
    def transport(url, headers, payload, timeout_seconds):
        return {"choices": [{"message": {"content": "ok"}}]}

    capabilities = ModelAdapterCapabilities(
        supports_tool_choice_none=True,
        enforces_request_timeout=True,
        max_response_bytes=2048,
    )
    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
            max_response_bytes=4096,
        ),
        transport=transport,
        capabilities=capabilities,
    )

    assert adapter.capabilities == capabilities


def test_custom_transport_rejects_invalid_or_overstated_capabilities():
    def transport(url, headers, payload, timeout_seconds):
        return {"choices": [{"message": {"content": "ok"}}]}

    config = OpenAICompatibleConfig(
        base_url="https://example.test/v1",
        api_key="secret",
        model="review-model",
        max_response_bytes=1024,
    )

    with pytest.raises(TypeError, match="capabilities"):
        OpenAICompatibleToolAdapter(
            config,
            transport=transport,
            capabilities={"enforces_request_timeout": True},
        )
    with pytest.raises(ValueError, match="may not exceed"):
        OpenAICompatibleToolAdapter(
            config,
            transport=transport,
            capabilities=ModelAdapterCapabilities(
                supports_tool_choice_none=True,
                enforces_request_timeout=True,
                max_response_bytes=2048,
            ),
        )


def test_default_transport_accepts_response_at_exact_byte_limit(monkeypatch):
    body = json.dumps(
        {"choices": [{"message": {"content": "exact"}}]},
        separators=(",", ":"),
    ).encode("utf-8")
    http_response = _BoundedHttpResponse(body)
    monkeypatch.setattr(
        "review_agent.model_adapter.urllib.request.urlopen",
        lambda request, timeout: http_response,
    )
    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="credential-must-not-leak",
            model="review-model",
            max_response_bytes=len(body),
        )
    )

    response = adapter.complete_turn(make_request())

    assert response.kind is ModelResponseKind.FINAL
    assert response.final_text == "exact"
    assert http_response.read_sizes == [len(body) + 1]


def test_default_transport_passes_request_timeout_to_http_layer(monkeypatch):
    body = b'{"choices":[{"message":{"content":"ok"}}]}'
    http_response = _BoundedHttpResponse(body)
    captured = {}

    def urlopen(request, timeout):
        captured["timeout"] = timeout
        return http_response

    monkeypatch.setattr(
        "review_agent.model_adapter.urllib.request.urlopen",
        urlopen,
    )
    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
            timeout_seconds=60,
            max_response_bytes=len(body),
        )
    )
    request = make_request()
    request.parameters["timeout_seconds"] = 2.5

    response = adapter.complete_turn(request)

    assert response.kind is ModelResponseKind.FINAL
    assert captured["timeout"] == 2.5
    assert http_response.read_sizes == [len(body) + 1]


def test_default_transport_wall_clock_deadline_covers_slow_urlopen(monkeypatch):
    urlopen_started = threading.Event()
    release_urlopen = threading.Event()
    urlopen_finished = threading.Event()

    def slow_urlopen(request, timeout):
        urlopen_started.set()
        try:
            release_urlopen.wait(timeout=1.0)
            raise OSError("released slow urlopen")
        finally:
            urlopen_finished.set()

    monkeypatch.setattr(
        "review_agent.model_adapter.urllib.request.urlopen",
        slow_urlopen,
    )

    started_at = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="wall-clock timeout"):
            _urllib_transport(
                "https://example.test/v1/chat/completions",
                {},
                {"model": "review-model"},
                0.05,
                max_response_bytes=1024,
            )
    finally:
        release_urlopen.set()
    elapsed = time.monotonic() - started_at

    assert urlopen_started.is_set()
    assert elapsed < 0.5
    assert urlopen_finished.wait(timeout=0.5)


def test_default_transport_rechecks_deadline_after_worker_slot_admission(
    monkeypatch,
):
    class LateAdmissionSlots:
        acquire_calls = 0
        release_calls = 0

        def acquire(self, *, timeout):
            self.acquire_calls += 1
            time.sleep(timeout + 0.02)
            return True

        def release(self):
            self.release_calls += 1

    slots = LateAdmissionSlots()
    monkeypatch.setattr(model_adapter_module, "_HTTP_DEADLINE_SLOTS", slots)
    monkeypatch.setattr(
        model_adapter_module.threading,
        "Thread",
        lambda *args, **kwargs: pytest.fail(
            "deadline-expired admission must not create a worker"
        ),
    )

    with pytest.raises(TimeoutError, match="wall-clock timeout"):
        _urllib_transport(
            "https://example.test/v1/chat/completions",
            {},
            {"model": "review-model"},
            0.01,
            max_response_bytes=1024,
        )

    assert slots.acquire_calls == 1
    assert slots.release_calls == 1


def test_default_transport_releases_close_slot_when_cleanup_thread_start_fails(
    monkeypatch,
):
    class RecordingCloseSlots:
        acquire_calls = 0
        release_calls = 0

        def acquire(self, *, blocking):
            assert blocking is False
            self.acquire_calls += 1
            return True

        def release(self):
            self.release_calls += 1

    body = b'{"choices":[{"message":{"content":"late"}}]}'
    http_response = _BlockingReadHttpResponse(body)
    close_slots = RecordingCloseSlots()
    real_thread = threading.Thread

    def thread_factory(*args, **kwargs):
        if kwargs.get("name") == "model-adapter-http-timeout-close":
            raise OSError("cleanup thread unavailable")
        return real_thread(*args, **kwargs)

    monkeypatch.setattr(model_adapter_module, "_HTTP_CLOSE_SLOTS", close_slots)
    monkeypatch.setattr(model_adapter_module.threading, "Thread", thread_factory)
    monkeypatch.setattr(
        model_adapter_module.urllib.request,
        "urlopen",
        lambda request, timeout: http_response,
    )

    try:
        with pytest.raises(TimeoutError, match="wall-clock timeout"):
            _urllib_transport(
                "https://example.test/v1/chat/completions",
                {},
                {"model": "review-model"},
                0.02,
                max_response_bytes=len(body),
            )
        assert close_slots.acquire_calls == 1
        assert close_slots.release_calls == 1
    finally:
        http_response.closed.set()
    assert http_response.read_finished.wait(timeout=0.5)
    cleanup_deadline = time.monotonic() + 0.5
    while time.monotonic() < cleanup_deadline and any(
        thread.name == "model-adapter-http-deadline"
        for thread in threading.enumerate()
    ):
        time.sleep(0.01)
    assert not any(
        thread.name == "model-adapter-http-deadline"
        for thread in threading.enumerate()
    )


def test_default_transport_bounds_permanently_blocked_deadline_workers(monkeypatch):
    release_workers = threading.Event()
    start_callers = threading.Event()
    all_started = threading.Event()
    state_lock = threading.Lock()
    started = 0
    finished = 0
    caller_failures = []

    def permanently_blocked_urlopen(request, timeout):
        nonlocal started, finished
        with state_lock:
            started += 1
            if started == MAX_HTTP_DEADLINE_WORKERS:
                all_started.set()
        try:
            release_workers.wait(timeout=5.0)
            raise OSError("released blocked urlopen")
        finally:
            with state_lock:
                finished += 1

    monkeypatch.setattr(
        "review_agent.model_adapter.urllib.request.urlopen",
        permanently_blocked_urlopen,
    )

    def run_timed_request():
        start_callers.wait(timeout=1.0)
        try:
            _urllib_transport(
                "https://example.test/v1/chat/completions",
                {},
                {"model": "review-model"},
                0.5,
                max_response_bytes=1024,
            )
        except TimeoutError:
            return
        except BaseException as error:
            with state_lock:
                caller_failures.append(error)
        else:
            with state_lock:
                caller_failures.append(AssertionError("request unexpectedly succeeded"))

    callers = [
        threading.Thread(target=run_timed_request, name=f"timeout-caller-{index}")
        for index in range(MAX_HTTP_DEADLINE_WORKERS)
    ]

    try:
        for caller in callers:
            caller.start()
        start_callers.set()
        assert all_started.wait(timeout=2.0)

        with pytest.raises(TimeoutError, match="wall-clock timeout"):
            _urllib_transport(
                "https://example.test/v1/chat/completions",
                {},
                {"model": "review-model"},
                0.02,
                max_response_bytes=1024,
            )
        with state_lock:
            assert started == MAX_HTTP_DEADLINE_WORKERS
        for caller in callers:
            caller.join(timeout=1.0)
        assert not any(caller.is_alive() for caller in callers)
        with state_lock:
            assert caller_failures == []
    finally:
        start_callers.set()
        release_workers.set()
        for caller in callers:
            caller.join(timeout=1.0)

    cleanup_deadline = time.monotonic() + 2.0
    while time.monotonic() < cleanup_deadline:
        with state_lock:
            counts_match = finished == started
        active_deadline_workers = any(
            thread.name == "model-adapter-http-deadline"
            for thread in threading.enumerate()
        )
        if counts_match and not active_deadline_workers:
            break
        time.sleep(0.01)
    with state_lock:
        assert finished == started
    assert not any(
        thread.name == "model-adapter-http-deadline"
        for thread in threading.enumerate()
    )


def test_default_transport_wall_clock_deadline_covers_slow_read_and_closes_response(
    monkeypatch,
):
    body = b'{"choices":[{"message":{"content":"late"}}]}'
    http_response = _BlockingReadHttpResponse(body)
    monkeypatch.setattr(
        "review_agent.model_adapter.urllib.request.urlopen",
        lambda request, timeout: http_response,
    )

    started_at = time.monotonic()
    with pytest.raises(TimeoutError, match="wall-clock timeout"):
        _urllib_transport(
            "https://example.test/v1/chat/completions",
            {},
            {"model": "review-model"},
            0.05,
            max_response_bytes=len(body),
        )
    elapsed = time.monotonic() - started_at

    assert http_response.read_started.is_set()
    assert elapsed < 0.5
    assert http_response.closed.wait(timeout=0.5)
    assert http_response.read_finished.wait(timeout=0.5)
    assert http_response.close_calls == 1


def test_default_transport_rejects_limit_plus_one_without_leaking_body_or_credentials(
    monkeypatch,
):
    body_secret = "response-body-must-not-leak"
    credential = "credential-must-not-leak"
    body = json.dumps(
        {"choices": [{"message": {"content": body_secret}}]},
        separators=(",", ":"),
    ).encode("utf-8")
    configured_limit = len(body) - 1
    http_response = _BoundedHttpResponse(body)
    monkeypatch.setattr(
        "review_agent.model_adapter.urllib.request.urlopen",
        lambda request, timeout: http_response,
    )
    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key=credential,
            model="review-model",
            max_response_bytes=configured_limit,
        )
    )

    response = adapter.complete_turn(make_request())

    assert response.kind is ModelResponseKind.INVALID
    assert response.error == PROVIDER_RESPONSE_TOO_LARGE_ERROR
    assert response.raw == {}
    assert body_secret not in response.error
    assert credential not in response.error
    assert http_response.read_sizes == [configured_limit + 1]


def test_openai_adapter_places_tool_history_before_json_finalization_messages():
    captured_payloads = []
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": "I will inspect the requested range.",
                        "reasoning_content": "The file range is needed before concluding.",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "read_range",
                                    "arguments": '{"path": "app.py"}',
                                },
                            }
                        ]
                    }
                }
            ]
        },
        {"choices": [{"message": {"content": '{"status": "completed"}'}}]},
    ]

    def transport(url, headers, payload, timeout_seconds):
        captured_payloads.append(payload)
        return responses.pop(0)

    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
        ),
        transport=transport,
    )
    first_request = ModelTurnRequest(
        system="system",
        tools=[
            ModelToolSpec(
                name="read_range",
                description="Read range",
                parameters_schema={"type": "object"},
            )
        ],
        messages=[{"role": "user", "content": "Review"}],
        tool_results=[],
        parameters={"max_output_tokens": 1000, "temperature": 0},
    )

    first_response = adapter.complete_turn(first_request)

    assert first_response.kind is ModelResponseKind.TOOL_CALLS
    assert first_response.tool_calls[0].call_id == "call-1"

    tool_result = ModelToolResult(
        call_id="call-1",
        tool_name="read_range",
        content="app.py contents",
        observation_ids=["O-read"],
    )
    second_response = adapter.complete_turn(
        ModelTurnRequest(
            system="system",
            tools=[],
            messages=[
                *first_request.messages,
                model_response_to_assistant_message(first_response),
                model_tool_result_to_message(tool_result),
                {"role": "assistant", "content": "prose final"},
                {"role": "user", "content": "Return corrected JSON."},
            ],
            tool_results=[tool_result],
            parameters={
                **first_request.parameters,
                "tool_choice": "none",
                "response_format": "json_object",
            },
        )
    )

    second_messages = captured_payloads[1]["messages"]
    assert [message["role"] for message in second_messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
    ]
    assistant_message = second_messages[2]
    assert assistant_message["content"] == "I will inspect the requested range."
    assert (
        assistant_message["reasoning_content"]
        == "The file range is needed before concluding."
    )
    assert assistant_message["tool_calls"][0]["id"] == "call-1"
    assert assistant_message["tool_calls"][0]["type"] == "function"
    assert assistant_message["tool_calls"][0]["function"]["name"] == "read_range"
    assert json.loads(assistant_message["tool_calls"][0]["function"]["arguments"]) == {"path": "app.py"}
    tool_message = second_messages[3]
    assert tool_message == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": serialize_tool_result_envelope(tool_result),
    }
    assert json.loads(tool_message["content"]) == {
        "schema_version": "review_agent_tool_result_v1",
        "tool_name": "read_range",
        "observation_ids": ["O-read"],
        "is_error": False,
        "content": "app.py contents",
    }
    assert second_messages[4] == {"role": "assistant", "content": "prose final"}
    assert second_messages[5] == {
        "role": "user",
        "content": "Return corrected JSON.",
    }
    assert second_response.kind is ModelResponseKind.FINAL
    assert second_response.final_text == '{"status": "completed"}'


def test_openai_adapter_does_not_reuse_tool_history_across_conversations():
    captured_payloads = []
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": "session A tool call",
                        "reasoning_content": "session A reasoning",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "read_range",
                                    "arguments": '{"path": "a.py"}',
                                },
                            }
                        ],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": "session B tool call",
                        "reasoning_content": "session B reasoning",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "read_range",
                                    "arguments": '{"path": "b.py"}',
                                },
                            }
                        ],
                    }
                }
            ]
        },
        {"choices": [{"message": {"content": "done"}}]},
    ]

    def transport(url, headers, payload, timeout_seconds):
        captured_payloads.append(payload)
        return responses.pop(0)

    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
        ),
        transport=transport,
    )
    tool = ModelToolSpec(
        name="read_range",
        description="Read range",
        parameters_schema={"type": "object"},
    )
    adapter.complete_turn(
        ModelTurnRequest(
            system="system",
            tools=[tool],
            messages=[{"role": "user", "content": "Review session A"}],
            tool_results=[],
            parameters={"trace_id": "session-a"},
        )
    )
    session_b_response = adapter.complete_turn(
        ModelTurnRequest(
            system="system",
            tools=[tool],
            messages=[{"role": "user", "content": "Review session B"}],
            tool_results=[],
            parameters={"trace_id": "session-b"},
        )
    )

    tool_result = ModelToolResult(
        call_id="call-1",
        tool_name="read_range",
        content="b.py contents",
    )
    adapter.complete_turn(
        ModelTurnRequest(
            system="system",
            tools=[tool],
            messages=[
                {"role": "user", "content": "Review session B"},
                model_response_to_assistant_message(session_b_response),
                model_tool_result_to_message(tool_result),
            ],
            tool_results=[tool_result],
            parameters={"trace_id": "session-b"},
        )
    )

    messages = captured_payloads[-1]["messages"]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert messages[2]["content"] == "session B tool call"
    assert messages[2]["reasoning_content"] == "session B reasoning"
    assert json.loads(
        messages[2]["tool_calls"][0]["function"]["arguments"]
    ) == {"path": "b.py"}
    assert messages[3] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": serialize_tool_result_envelope(tool_result),
    }


def test_openai_adapter_rejects_orphan_tool_result_before_transport():
    transport_called = False

    def transport(url, headers, payload, timeout_seconds):
        nonlocal transport_called
        transport_called = True
        return {"choices": [{"message": {"content": "must not be called"}}]}

    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
        ),
        transport=transport,
    )

    with pytest.raises(ValueError, match="no matching assistant tool call"):
        adapter.complete_turn(
            ModelTurnRequest(
                system="system",
                tools=[],
                messages=[{"role": "user", "content": "Review"}],
                tool_results=[
                    ModelToolResult(
                        call_id="orphan-call",
                        tool_name="read_range",
                        content="orphan result",
                    )
                ],
                parameters={},
            )
        )

    assert transport_called is False


def test_openai_adapter_rejects_orphan_tool_message_before_transport():
    transport_called = False

    def transport(url, headers, payload, timeout_seconds):
        nonlocal transport_called
        transport_called = True
        return {"choices": [{"message": {"content": "must not be called"}}]}

    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
        ),
        transport=transport,
    )
    orphan = ModelToolResult(
        call_id="orphan-message",
        tool_name="read_range",
        content="orphan result",
    )

    with pytest.raises(ValueError, match="no matching assistant tool call"):
        adapter.complete_turn(
            ModelTurnRequest(
                system="system",
                tools=[],
                messages=[
                    {"role": "user", "content": "Review"},
                    model_tool_result_to_message(orphan),
                ],
                tool_results=[],
                parameters={},
            )
        )

    assert transport_called is False


def test_openai_adapter_accepts_exact_tool_result_metadata_without_duplicate():
    captured_payloads = []

    def transport(url, headers, payload, timeout_seconds):
        captured_payloads.append(payload)
        return {"choices": [{"message": {"content": "done"}}]}

    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
        ),
        transport=transport,
    )
    result = ModelToolResult(
        call_id="call-1",
        tool_name="read_range",
        content="app.py contents",
        observation_ids=["O-read"],
        is_error=True,
    )
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "read_range",
                    "arguments": '{"path": "app.py"}',
                },
            }
        ],
    }

    adapter.complete_turn(
        ModelTurnRequest(
            system="system",
            tools=[],
            messages=[
                {"role": "user", "content": "Review"},
                assistant,
                model_tool_result_to_message(result),
            ],
            tool_results=[result],
            parameters={},
        )
    )

    messages = captured_payloads[0]["messages"]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert sum(message["role"] == "tool" for message in messages) == 1
    assert messages[-1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": serialize_tool_result_envelope(result),
    }


def test_openai_adapter_rejects_mismatched_tool_result_metadata_before_transport():
    transport_called = False

    def transport(url, headers, payload, timeout_seconds):
        nonlocal transport_called
        transport_called = True
        return {"choices": [{"message": {"content": "must not be called"}}]}

    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
        ),
        transport=transport,
    )
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "read_range",
                    "arguments": '{"path": "app.py"}',
                },
            }
        ],
    }
    old_result = ModelToolResult(
        call_id="call-1",
        tool_name="read_range",
        content="old-sensitive-content",
    )
    new_result = ModelToolResult(
        call_id="call-1",
        tool_name="read_range",
        content="new-sensitive-content",
    )

    with pytest.raises(ValueError) as error:
        adapter.complete_turn(
            ModelTurnRequest(
                system="system",
                tools=[],
                messages=[
                    {"role": "user", "content": "Review"},
                    assistant,
                    model_tool_result_to_message(old_result),
                ],
                tool_results=[new_result],
                parameters={},
            )
        )

    assert transport_called is False
    assert str(error.value) == "tool result metadata does not match transcript"


@pytest.mark.parametrize(
    "metadata_override",
    [
        pytest.param(
            {"observation_ids": ["O-metadata-secret"]},
            id="observation-ids",
        ),
        pytest.param(
            {"tool_name": "metadata-secret-tool"},
            id="tool-name",
        ),
        pytest.param({"is_error": True}, id="is-error"),
    ],
)
def test_openai_adapter_rejects_full_metadata_mismatch_when_raw_content_matches(
    metadata_override,
):
    transport_called = False

    def transport(url, headers, payload, timeout_seconds):
        nonlocal transport_called
        transport_called = True
        return {"choices": [{"message": {"content": "must not be called"}}]}

    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
        ),
        transport=transport,
    )
    call_id = "call-sensitive-metadata"
    transcript_result = ModelToolResult(
        call_id=call_id,
        tool_name="read_range",
        content="same-sensitive-content",
        observation_ids=["O-transcript-secret"],
        is_error=False,
    )
    metadata_values = {
        "call_id": call_id,
        "tool_name": transcript_result.tool_name,
        "content": transcript_result.content,
        "observation_ids": list(transcript_result.observation_ids),
        "is_error": transcript_result.is_error,
        **metadata_override,
    }
    metadata_result = ModelToolResult(**metadata_values)
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "read_range", "arguments": "{}"},
            }
        ],
    }

    with pytest.raises(ValueError) as error:
        adapter.complete_turn(
            ModelTurnRequest(
                system="system",
                tools=[],
                messages=[
                    {"role": "user", "content": "Review"},
                    assistant,
                    model_tool_result_to_message(transcript_result),
                ],
                tool_results=[metadata_result],
                parameters={},
            )
        )

    assert transport_called is False
    assert str(error.value) == "tool result metadata does not match transcript"


@pytest.mark.parametrize(
    "metadata_override",
    [
        pytest.param({"is_error": 1}, id="integer-is-error"),
        pytest.param(
            {"observation_ids": ("O-read",)},
            id="non-list-observation-ids",
        ),
        pytest.param(
            {"observation_ids": ["O-read", "O-read"]},
            id="duplicate-observation-ids",
        ),
        pytest.param({"content": 7}, id="non-string-content"),
        pytest.param({"tool_name": " \n"}, id="blank-tool-name"),
        pytest.param(
            {"content": _ExplodingUtf8Str("same-sensitive-content")},
            id="malicious-content-exception",
        ),
    ],
)
def test_openai_adapter_rejects_noncanonical_typed_metadata_before_transport(
    metadata_override,
):
    transport_called = False

    def transport(url, headers, payload, timeout_seconds):
        nonlocal transport_called
        transport_called = True
        return {"choices": [{"message": {"content": "must not be called"}}]}

    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
        ),
        transport=transport,
    )
    call_id = "call-sensitive-strict-metadata"
    transcript_result = ModelToolResult(
        call_id=call_id,
        tool_name="read_range",
        content="same-sensitive-content",
        observation_ids=["O-read"],
        is_error=True,
    )
    metadata_values = {
        "call_id": call_id,
        "tool_name": transcript_result.tool_name,
        "content": transcript_result.content,
        "observation_ids": list(transcript_result.observation_ids),
        "is_error": transcript_result.is_error,
        **metadata_override,
    }
    metadata_result = ModelToolResult(**metadata_values)
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "read_range", "arguments": "{}"},
            }
        ],
    }

    with pytest.raises(ValueError) as error:
        adapter.complete_turn(
            ModelTurnRequest(
                system="system",
                tools=[],
                messages=[
                    {"role": "user", "content": "Review"},
                    assistant,
                    model_tool_result_to_message(transcript_result),
                ],
                tool_results=[metadata_result],
                parameters={},
            )
        )

    assert transport_called is False
    assert str(error.value) == "tool result metadata does not match transcript"
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__ is True


@pytest.mark.parametrize(
    "encoding",
    [
        pytest.param("plain", id="plain-content"),
        pytest.param("noncanonical", id="noncanonical-json"),
        pytest.param("extra-field", id="extra-field"),
    ],
)
def test_openai_adapter_rejects_invalid_adjacent_tool_result_envelope_before_transport(
    encoding,
):
    transport_called = False

    def transport(url, headers, payload, timeout_seconds):
        nonlocal transport_called
        transport_called = True
        return {"choices": [{"message": {"content": "must not be called"}}]}

    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
        ),
        transport=transport,
    )
    call_id = "call-sensitive-envelope"
    result = ModelToolResult(
        call_id=call_id,
        tool_name="read_range",
        content="sensitive-envelope-content",
        observation_ids=["O-sensitive-envelope"],
    )
    canonical_content = serialize_tool_result_envelope(result)
    if encoding == "plain":
        message_content = result.content
    else:
        envelope = json.loads(canonical_content)
        if encoding == "extra-field":
            envelope["extra_field"] = "sensitive-extra-field"
            message_content = json.dumps(
                envelope,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        else:
            message_content = json.dumps(
                envelope,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "read_range", "arguments": "{}"},
            }
        ],
    }

    with pytest.raises(ValueError) as error:
        adapter.complete_turn(
            ModelTurnRequest(
                system="system",
                tools=[],
                messages=[
                    {"role": "user", "content": "Review"},
                    assistant,
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": message_content,
                    },
                ],
                tool_results=[],
                parameters={},
            )
        )

    assert transport_called is False
    assert str(error.value) == "invalid tool result envelope"


def test_openai_adapter_rejects_duplicate_call_id_across_assistant_turns():
    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
        ),
        transport=lambda url, headers, payload, timeout_seconds: {
            "choices": [{"message": {"content": "must not be called"}}]
        },
    )
    first_assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "duplicate-call",
                "type": "function",
                "function": {"name": "read_range", "arguments": "{}"},
            }
        ],
    }
    second_assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "duplicate-call",
                "type": "function",
                "function": {"name": "read_range", "arguments": "{}"},
            }
        ],
    }

    with pytest.raises(ValueError, match="duplicate assistant tool call id"):
        adapter.complete_turn(
            ModelTurnRequest(
                system="system",
                tools=[],
                messages=[
                    {"role": "user", "content": "Review"},
                    first_assistant,
                    {
                        "role": "tool",
                        "tool_call_id": "duplicate-call",
                        "content": "first result",
                    },
                    {"role": "user", "content": "Continue"},
                    second_assistant,
                    {
                        "role": "tool",
                        "tool_call_id": "duplicate-call",
                        "content": "second result",
                    },
                ],
                tool_results=[],
                parameters={},
            )
        )


def test_openai_adapter_rejects_partial_assistant_tool_call_batch():
    transport_called = False

    def transport(url, headers, payload, timeout_seconds):
        nonlocal transport_called
        transport_called = True
        return {"choices": [{"message": {"content": "must not be called"}}]}

    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
        ),
        transport=transport,
    )
    assistant = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-a",
                "type": "function",
                "function": {"name": "read_range", "arguments": "{}"},
            },
            {
                "id": "call-b",
                "type": "function",
                "function": {"name": "read_range", "arguments": "{}"},
            },
        ],
    }

    with pytest.raises(ValueError, match="missing tool result for assistant call 'call-b'"):
        adapter.complete_turn(
            ModelTurnRequest(
                system="system",
                tools=[],
                messages=[
                    {"role": "user", "content": "Review"},
                    assistant,
                ],
                tool_results=[
                    ModelToolResult(
                        call_id="call-a",
                        tool_name="read_range",
                        content="result a",
                    )
                ],
                parameters={},
            )
        )

    assert transport_called is False


def test_openai_adapter_pairs_separate_results_across_two_ordered_tool_turns():
    captured_payloads = []
    responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": "first tool turn",
                        "reasoning_content": "first reasoning",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "read_range",
                                    "arguments": '{"path": "a.py"}',
                                },
                            }
                        ],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": "second tool turn",
                        "reasoning_content": "second reasoning",
                        "tool_calls": [
                            {
                                "id": "call-2",
                                "type": "function",
                                "function": {
                                    "name": "read_range",
                                    "arguments": '{"path": "b.py"}',
                                },
                            }
                        ],
                    }
                }
            ]
        },
        {"choices": [{"message": {"content": "done"}}]},
    ]

    def transport(url, headers, payload, timeout_seconds):
        captured_payloads.append(payload)
        return responses.pop(0)

    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
        ),
        transport=transport,
    )
    tool = ModelToolSpec(
        name="read_range",
        description="Read range",
        parameters_schema={"type": "object"},
    )
    initial_user = {"role": "user", "content": "Review"}
    first_response = adapter.complete_turn(
        ModelTurnRequest(
            system="system",
            tools=[tool],
            messages=[initial_user],
            tool_results=[],
            parameters={},
        )
    )
    first_result = ModelToolResult(
        call_id="call-1",
        tool_name="read_range",
        content="a.py contents",
    )
    first_checkpoint = {"role": "user", "content": "Runtime checkpoint one"}
    second_response = adapter.complete_turn(
        ModelTurnRequest(
            system="system",
            tools=[tool],
            messages=[
                initial_user,
                model_response_to_assistant_message(first_response),
                first_checkpoint,
            ],
            tool_results=[first_result],
            parameters={},
        )
    )
    assert [message["role"] for message in captured_payloads[1]["messages"]] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
    ]
    assert captured_payloads[1]["messages"][3] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": serialize_tool_result_envelope(first_result),
    }

    second_result = ModelToolResult(
        call_id="call-2",
        tool_name="read_range",
        content="b.py contents",
    )
    second_checkpoint = {"role": "user", "content": "Runtime checkpoint two"}
    adapter.complete_turn(
        ModelTurnRequest(
            system="system",
            tools=[tool],
            messages=[
                initial_user,
                model_response_to_assistant_message(first_response),
                model_tool_result_to_message(first_result),
                first_checkpoint,
                model_response_to_assistant_message(second_response),
                model_tool_result_to_message(second_result),
                second_checkpoint,
            ],
            tool_results=[first_result, second_result],
            parameters={},
        )
    )

    messages = captured_payloads[2]["messages"]
    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "user",
        "assistant",
        "tool",
        "user",
    ]
    assert [
        message["tool_call_id"]
        for message in messages
        if message["role"] == "tool"
    ] == ["call-1", "call-2"]
    assert [
        message["content"]
        for message in messages
        if message["role"] == "tool"
    ] == [
        serialize_tool_result_envelope(first_result),
        serialize_tool_result_envelope(second_result),
    ]
    assert messages[5]["reasoning_content"] == "second reasoning"
    assert json.loads(
        messages[5]["tool_calls"][0]["function"]["arguments"]
    ) == {"path": "b.py"}


def test_openai_compatible_adapter_returns_invalid_when_transport_raises_json_decode_error():
    def transport(url, headers, payload, timeout_seconds):
        raise json.JSONDecodeError("bad json", "{", 0)

    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
        ),
        transport=transport,
    )

    response = adapter.complete_turn(make_request())

    assert response.kind is ModelResponseKind.INVALID
    assert "provider response was not valid JSON" in response.error


def test_openai_compatible_adapter_returns_invalid_when_message_is_non_object():
    def transport(url, headers, payload, timeout_seconds):
        return {"choices": [{"message": "not an object"}]}

    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
        ),
        transport=transport,
    )

    response = adapter.complete_turn(make_request())

    assert response.kind is ModelResponseKind.INVALID
    assert "provider response message must be an object" in response.error


def test_openai_compatible_adapter_returns_invalid_when_tool_call_function_is_non_object():
    def transport(url, headers, payload, timeout_seconds):
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": "not an object",
                            }
                        ]
                    }
                }
            ]
        }

    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
        ),
        transport=transport,
    )

    response = adapter.complete_turn(make_request())

    assert response.kind is ModelResponseKind.INVALID
    assert "tool call function must be an object" in response.error


def test_openai_compatible_adapter_caps_transport_timeout_to_runtime_budget():
    captured = {}

    def transport(url, headers, payload, timeout_seconds):
        captured["timeout_seconds"] = timeout_seconds
        return {"choices": [{"message": {"content": '{"status": "partial"}'}}]}

    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
            timeout_seconds=60,
        ),
        transport=transport,
    )
    request = make_request()
    request.parameters["timeout_seconds"] = 2.5

    adapter.complete_turn(request)

    assert captured["timeout_seconds"] == 2.5


def test_v2_tool_projection_serializes_directly_to_tool_message() -> None:
    raw = ReviewToolResult.success(
        tool_call_id="call-v2",
        session_id="session-v2",
        snapshot_id="S-" + "a" * 64,
        tool_name="read_range",
        arguments={"path": "src/api.py"},
        content="line 1",
        reacquirable=True,
    )

    message = review_tool_projection_to_message(
        ToolResultProjectionV2.inline(raw)
    )
    payload = json.loads(message["content"])

    assert message["role"] == "tool"
    assert message["tool_call_id"] == "call-v2"
    assert payload["schema_version"] == "review_tool_result_v2"
    assert payload["content"] == "line 1"
    assert "observation_ids" not in payload


def test_openai_adapter_accepts_canonical_v2_tool_transcript_without_legacy_metadata() -> None:
    captured = {}

    def transport(url, headers, payload, timeout_seconds):
        captured.update(payload)
        return {"choices": [{"message": {"content": '{"findings":[]}'}}]}

    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
        ),
        transport=transport,
    )
    raw = ReviewToolResult.success(
        tool_call_id="call-v2-transcript",
        session_id="session-v2",
        snapshot_id="S-" + "b" * 64,
        tool_name="read_range",
        arguments={"path": "src/api.py"},
        content="line 1",
        reacquirable=True,
    )
    tool_message = review_tool_projection_to_message(
        ToolResultProjectionV2.inline(raw)
    )
    request = ModelTurnRequest(
        system="system",
        tools=[],
        messages=[
            {"role": "user", "content": "Review"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-v2-transcript",
                        "type": "function",
                        "function": {
                            "name": "read_range",
                            "arguments": '{"path":"src/api.py"}',
                        },
                    }
                ],
            },
            tool_message,
        ],
        tool_results=[],
        parameters={"tool_choice": "none"},
    )

    response = adapter.complete_turn(request)

    assert response.kind is ModelResponseKind.FINAL
    assert captured["messages"][-1] == tool_message


def test_openai_adapter_omits_reviewer_output_limit_when_runtime_does_not_set_one() -> None:
    captured = {}

    def transport(url, headers, payload, timeout_seconds):
        captured.update(payload)
        return {"choices": [{"message": {"content": "{}"}}]}

    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
        ),
        transport=transport,
    )

    adapter.complete_turn(make_request())

    assert "max_tokens" not in captured


def test_adapter_uses_model_capability_only_when_provider_requires_output_limit() -> None:
    captured = {}

    def transport(url, headers, payload, timeout_seconds):
        captured.update(payload)
        return {"choices": [{"message": {"content": "{}"}}]}

    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
        ),
        transport=transport,
    )
    request = make_request()
    request.parameters.update(
        {
            "requires_max_output_tokens": True,
            "model_max_output_tokens": 131_072,
        }
    )

    adapter.complete_turn(request)

    assert captured["max_tokens"] == 131_072
