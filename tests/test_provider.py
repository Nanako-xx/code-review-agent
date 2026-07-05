import urllib.error

import pytest

from review_agent.models import ModelInvocationEnvelope
from review_agent.model_adapter import OpenAICompatibleConfig
from review_agent.model_protocol import ModelResponse
from review_agent.provider import (
    FakeProvider,
    ModelProviderError,
    OpenAICompatibleProvider,
    ProviderConfigError,
    build_provider_from_config,
)


def make_envelope() -> ModelInvocationEnvelope:
    return ModelInvocationEnvelope(
        system="system rules",
        tools=[],
        messages=[{"role": "user", "content": "review this"}],
        parameters={
            "model": "test-model",
            "max_output_tokens": 256,
            "temperature": 0,
            "trace_id": "trace-1",
        },
    )


def test_fake_provider_returns_configured_text():
    provider = FakeProvider('{"status":"completed"}')

    response = provider.complete(make_envelope())

    assert response.content == '{"status":"completed"}'
    assert response.provider_name == "fake"
    assert response.model == "fake-reviewer"


def test_build_provider_rejects_missing_api_key(monkeypatch):
    monkeypatch.delenv("REVIEW_AGENT_API_KEY", raising=False)

    with pytest.raises(ProviderConfigError, match="REVIEW_AGENT_API_KEY"):
        build_provider_from_config(
            provider_name="openai-compatible",
            model="review-model",
            base_url="https://example.test/v1",
            api_key_env="REVIEW_AGENT_API_KEY",
        )


def test_openai_compatible_provider_builds_expected_payload():
    captured = {}

    def fake_transport(url, headers, payload, timeout_seconds):
        captured["url"] = url
        captured["headers"] = headers
        captured["payload"] = payload
        captured["timeout_seconds"] = timeout_seconds
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"status":"completed"}',
                    }
                }
            ],
            "usage": {"total_tokens": 12},
        }

    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret-key",
            model="review-model",
            timeout_seconds=7,
        ),
        transport=fake_transport,
    )

    response = provider.complete(make_envelope())

    assert captured["url"] == "https://example.test/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    assert captured["payload"]["model"] == "review-model"
    assert captured["payload"]["messages"][0] == {"role": "system", "content": "system rules"}
    assert captured["payload"]["messages"][1] == {"role": "user", "content": "review this"}
    assert captured["payload"]["max_tokens"] == 256
    assert captured["payload"]["temperature"] == 0
    assert captured["timeout_seconds"] == 7
    assert response.content == '{"status":"completed"}'
    assert response.raw["usage"]["total_tokens"] == 12


def test_openai_compatible_provider_wraps_transport_errors():
    def failing_transport(url, headers, payload, timeout_seconds):
        raise urllib.error.URLError("connection refused")

    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret-key",
            model="review-model",
        ),
        transport=failing_transport,
    )

    with pytest.raises(ModelProviderError, match="provider request failed"):
        provider.complete(make_envelope())
