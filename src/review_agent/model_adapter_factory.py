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
from review_agent.model_protocol import ModelToolCall, ModelTurnRequest


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
        return _factory_fake_adapter()


@dataclass(frozen=True)
class OpenAICompatibleModelAdapterFactory:
    config: OpenAICompatibleConfig

    def create(self) -> ModelAdapter:
        return OpenAICompatibleToolAdapter(self.config)


class _FactoryFakeToolCallingAdapter(FakeToolCallingAdapter):
    provider_name = "fake"


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


def _factory_fake_adapter() -> FakeToolCallingAdapter:
    return _FactoryFakeToolCallingAdapter(
        script=[
            _fake_response_for_request,
            _fake_response_for_request,
        ]
    )


def _fake_response_for_request(request: ModelTurnRequest) -> ModelTurnResponse:
    if not request.tools or request.parameters.get("tool_choice") == "none":
        return _fake_single_shot_response()

    observation_id = _latest_observation_id(request)
    if observation_id:
        return _fake_completed_agent_loop_response(observation_id)

    changed_file = _first_changed_file(request)
    if changed_file:
        return ModelTurnResponse(
            kind=ModelResponseKind.TOOL_CALLS,
            tool_calls=[ModelToolCall("call-1", "compare_base_head", {"path": changed_file})],
            provider_name="fake",
            model="fake-reviewer",
        )

    return _fake_completed_agent_loop_response("")


def _fake_single_shot_response() -> ModelTurnResponse:
    return ModelTurnResponse(
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


def _fake_completed_agent_loop_response(observation_id: str) -> ModelTurnResponse:
    evidence_refs = [observation_id] if observation_id else []
    contract_assessments = (
        [
            {
                "contract": "regression_safety",
                "status": "covered",
                "summary": "Fake agent loop used a tool observation.",
                "evidence_refs": evidence_refs,
            }
        ]
        if evidence_refs
        else []
    )
    return ModelTurnResponse(
        kind=ModelResponseKind.FINAL,
        final_text=json.dumps(
            {
                "contract_assessments": contract_assessments,
                "confirmed_findings": [],
                "rejected_hypotheses": [],
                "uncertainties": [],
                "observation_refs": evidence_refs,
                "investigation_summary": "Fake agent loop reviewer executed.",
                "status": "completed",
            }
        ),
        provider_name="fake",
        model="fake-reviewer",
        raw={"fake": True},
    )


def _latest_observation_id(request: ModelTurnRequest) -> str:
    for result in reversed(request.tool_results):
        if result.observation_ids:
            return result.observation_ids[-1]
    return ""


def _first_changed_file(request: ModelTurnRequest) -> str:
    prefix = "Changed Files:"
    for message in request.messages:
        content = message.get("content", "")
        if not isinstance(content, str):
            continue
        for line in content.splitlines():
            if not line.startswith(prefix):
                continue
            changed_files = line.removeprefix(prefix).strip()
            for changed_file in changed_files.split(","):
                changed_file = changed_file.strip()
                if changed_file:
                    return changed_file
    return ""
