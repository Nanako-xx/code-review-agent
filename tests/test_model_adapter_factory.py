import json

import pytest

from review_agent.model_adapter import FakeToolCallingAdapter, OpenAICompatibleToolAdapter
from review_agent.model_adapter_factory import (
    AdapterConfigError,
    ModelAdapterConfig,
    build_model_adapter_factory_from_config,
)
from review_agent.model_protocol import ModelResponseKind, ModelTurnRequest


def test_factory_returns_none_for_provider_none():
    factory = build_model_adapter_factory_from_config(
        ModelAdapterConfig(
            provider_name="none",
            model=None,
            base_url=None,
            api_key_env="REVIEW_AGENT_API_KEY",
        )
    )

    assert factory is None


def test_factory_creates_fresh_fake_adapters():
    factory = build_model_adapter_factory_from_config(
        ModelAdapterConfig(
            provider_name="fake",
            model=None,
            base_url=None,
            api_key_env="REVIEW_AGENT_API_KEY",
        )
    )

    first = factory.create()
    second = factory.create()

    assert isinstance(first, FakeToolCallingAdapter)
    assert isinstance(second, FakeToolCallingAdapter)
    assert first is not second


def test_factory_fake_adapter_returns_fake_final_response_metadata():
    factory = build_model_adapter_factory_from_config(
        ModelAdapterConfig(
            provider_name="fake",
            model=None,
            base_url=None,
            api_key_env="REVIEW_AGENT_API_KEY",
        )
    )

    response = factory.create().complete_turn(
        ModelTurnRequest(
            system="system",
            tools=[],
            messages=[{"role": "user", "content": "Review change"}],
            tool_results=[],
            parameters={},
        )
    )

    assert response.kind is ModelResponseKind.FINAL
    assert response.provider_name == "fake"
    assert response.model == "fake-reviewer"
    assert response.raw == {"fake": True}
    assert json.loads(response.final_text)["status"] == "partial"


def test_factory_creates_openai_compatible_adapter(monkeypatch):
    monkeypatch.setenv("REVIEW_AGENT_API_KEY", "secret-key")

    factory = build_model_adapter_factory_from_config(
        ModelAdapterConfig(
            provider_name="openai-compatible",
            model="review-model",
            base_url="https://example.test/v1",
            api_key_env="REVIEW_AGENT_API_KEY",
        )
    )

    adapter = factory.create()

    assert isinstance(adapter, OpenAICompatibleToolAdapter)
    assert adapter.provider_name == "openai-compatible"


def test_factory_rejects_missing_openai_api_key(monkeypatch):
    monkeypatch.delenv("REVIEW_AGENT_API_KEY", raising=False)

    with pytest.raises(AdapterConfigError, match="REVIEW_AGENT_API_KEY"):
        build_model_adapter_factory_from_config(
            ModelAdapterConfig(
                provider_name="openai-compatible",
                model="review-model",
                base_url="https://example.test/v1",
                api_key_env="REVIEW_AGENT_API_KEY",
            )
        )


def test_factory_rejects_missing_openai_model(monkeypatch):
    monkeypatch.setenv("REVIEW_AGENT_API_KEY", "secret-key")

    with pytest.raises(AdapterConfigError, match="--reviewer-model"):
        build_model_adapter_factory_from_config(
            ModelAdapterConfig(
                provider_name="openai-compatible",
                model=None,
                base_url="https://example.test/v1",
                api_key_env="REVIEW_AGENT_API_KEY",
            )
        )


def test_factory_rejects_missing_openai_base_url(monkeypatch):
    monkeypatch.setenv("REVIEW_AGENT_API_KEY", "secret-key")

    with pytest.raises(AdapterConfigError, match="--reviewer-base-url"):
        build_model_adapter_factory_from_config(
            ModelAdapterConfig(
                provider_name="openai-compatible",
                model="review-model",
                base_url=None,
                api_key_env="REVIEW_AGENT_API_KEY",
            )
        )
