from __future__ import annotations

import json
import os
from typing import Any, Protocol
import urllib.error

from review_agent.model_adapter import OpenAICompatibleConfig, Transport, _urllib_transport
from review_agent.model_protocol import ModelResponse
from review_agent.models import ModelInvocationEnvelope


class ModelProviderError(RuntimeError):
    pass


class ProviderConfigError(ValueError):
    pass


class ModelProvider(Protocol):
    def complete(self, envelope: ModelInvocationEnvelope) -> ModelResponse:
        raise NotImplementedError


class FakeProvider:
    def __init__(self, content: str, model: str = "fake-reviewer") -> None:
        self._content = content
        self._model = model

    def complete(self, envelope: ModelInvocationEnvelope) -> ModelResponse:
        return ModelResponse(
            content=self._content,
            provider_name="fake",
            model=self._model,
            raw={"trace_id": envelope.parameters.get("trace_id")},
        )


class OpenAICompatibleProvider:
    def __init__(self, config: OpenAICompatibleConfig, transport: Transport | None = None) -> None:
        self._config = config
        self._transport = transport or _urllib_transport

    def complete(self, envelope: ModelInvocationEnvelope) -> ModelResponse:
        payload = _build_chat_payload(self._config.model, envelope)
        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self._config.base_url.rstrip('/')}/chat/completions"
        try:
            raw = self._transport(url, headers, payload, self._config.timeout_seconds)
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise ModelProviderError(f"provider request failed: {error}") from error

        content = _extract_chat_content(raw)
        return ModelResponse(
            content=content,
            provider_name="openai-compatible",
            model=self._config.model,
            raw=raw,
        )


def build_provider_from_config(
    provider_name: str | None,
    model: str | None,
    base_url: str | None,
    api_key_env: str,
) -> ModelProvider | None:
    if provider_name in (None, "none"):
        return None
    if provider_name == "fake":
        return FakeProvider(_fake_reviewer_result_json())
    if provider_name == "openai-compatible":
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ProviderConfigError(f"missing API key environment variable: {api_key_env}")
        if not model:
            raise ProviderConfigError("--reviewer-model is required for openai-compatible provider")
        if not base_url:
            raise ProviderConfigError("--reviewer-base-url is required for openai-compatible provider")
        return OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                base_url=base_url,
                api_key=api_key,
                model=model,
            )
        )
    raise ProviderConfigError(f"unsupported reviewer provider: {provider_name}")


def _build_chat_payload(model: str, envelope: ModelInvocationEnvelope) -> dict[str, Any]:
    messages = [{"role": "system", "content": envelope.system}]
    messages.extend(envelope.messages)
    return {
        "model": model,
        "messages": messages,
        "max_tokens": envelope.parameters.get("max_output_tokens", 4096),
        "temperature": envelope.parameters.get("temperature", 0),
    }


def _extract_chat_content(raw: dict[str, Any]) -> str:
    try:
        return str(raw["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as error:
        raise ModelProviderError("provider response did not contain choices[0].message.content") from error


def _fake_reviewer_result_json() -> str:
    return json.dumps(
        {
            "contract_assessments": [],
            "confirmed_findings": [],
            "rejected_hypotheses": [],
            "uncertainties": ["Fake provider does not perform semantic review."],
            "observation_refs": [],
            "investigation_summary": "Fake reviewer executed.",
            "status": "partial",
        }
    )
