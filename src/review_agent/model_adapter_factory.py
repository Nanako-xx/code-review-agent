from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Protocol

from review_agent.model_adapter import (
    FakeToolCallingAdapter,
    ModelAdapter,
    OpenAICompatibleConfig,
    OpenAICompatibleToolAdapter,
)
from review_agent.model_protocol import ModelResponseKind, ModelTurnResponse


class AdapterConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ModelAdapterConfig:
    provider_name: str | None
    model: str | None
    base_url: str | None
    api_key_env: str


class ModelAdapterFactory(Protocol):
    def create(self) -> ModelAdapter:
        raise NotImplementedError


@dataclass(frozen=True)
class FakeModelAdapterFactory:
    def create(self) -> ModelAdapter:
        return _fake_single_shot_adapter()


@dataclass(frozen=True)
class OpenAICompatibleModelAdapterFactory:
    config: OpenAICompatibleConfig

    def create(self) -> ModelAdapter:
        return OpenAICompatibleToolAdapter(self.config)


def build_model_adapter_factory_from_config(
    config: ModelAdapterConfig,
) -> ModelAdapterFactory | None:
    provider_name = config.provider_name or "none"
    if provider_name == "none":
        return None
    if provider_name == "fake":
        return FakeModelAdapterFactory()
    if provider_name == "openai-compatible":
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise AdapterConfigError(f"missing API key environment variable: {config.api_key_env}")
        if not config.model:
            raise AdapterConfigError("--reviewer-model is required for openai-compatible provider")
        if not config.base_url:
            raise AdapterConfigError("--reviewer-base-url is required for openai-compatible provider")
        return OpenAICompatibleModelAdapterFactory(
            OpenAICompatibleConfig(
                base_url=config.base_url,
                api_key=api_key,
                model=config.model,
            )
        )
    raise AdapterConfigError(f"unsupported reviewer provider: {provider_name}")


def _fake_single_shot_adapter() -> FakeToolCallingAdapter:
    return FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text=json.dumps(
                    {
                        "contract_assessments": [],
                        "confirmed_findings": [],
                        "rejected_hypotheses": [],
                        "uncertainties": ["Fake provider does not perform semantic review."],
                        "observation_refs": [],
                        "investigation_summary": "Fake reviewer executed.",
                        "status": "partial",
                    }
                ),
                provider_name="fake",
                model="fake-reviewer",
                raw={"fake": True},
            )
        ]
    )
