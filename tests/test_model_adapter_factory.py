from __future__ import annotations

import json

import pytest

from review_agent.model_adapter import FakeToolCallingAdapter, OpenAICompatibleToolAdapter
from review_agent.model_adapter_factory import (
    AdapterConfigError,
    ModelAdapterConfig,
    build_model_adapter_factory_from_config,
)
from review_agent.model_protocol import ModelResponseKind, ModelToolResult, ModelToolSpec, ModelTurnRequest


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


def test_factory_fake_agent_loop_first_turn_requests_compare_for_first_changed_file():
    factory = build_model_adapter_factory_from_config(
        ModelAdapterConfig(
            provider_name="fake",
            model=None,
            base_url=None,
            api_key_env="REVIEW_AGENT_API_KEY",
        )
    )

    adapter = factory.create()
    response = adapter.complete_turn(_agent_loop_request("app.py, README.md"))

    assert isinstance(adapter, FakeToolCallingAdapter)
    assert response.kind is ModelResponseKind.TOOL_CALLS
    assert response.provider_name == "fake"
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].tool_name == "compare_base_head"
    assert response.tool_calls[0].arguments == {"path": "app.py"}


def test_factory_fake_agent_loop_no_changed_files_completes_without_evidence_refs():
    factory = build_model_adapter_factory_from_config(
        ModelAdapterConfig(
            provider_name="fake",
            model=None,
            base_url=None,
            api_key_env="REVIEW_AGENT_API_KEY",
        )
    )

    response = factory.create().complete_turn(_agent_loop_request(""))
    payload = json.loads(response.final_text)

    assert response.kind is ModelResponseKind.FINAL
    assert response.provider_name == "fake"
    assert payload["status"] == "completed"
    assert payload["observation_refs"] == []
    assert all(not assessment["evidence_refs"] for assessment in payload["contract_assessments"])


def test_factory_fake_agent_loop_after_tool_result_completes_citing_latest_observation():
    factory = build_model_adapter_factory_from_config(
        ModelAdapterConfig(
            provider_name="fake",
            model=None,
            base_url=None,
            api_key_env="REVIEW_AGENT_API_KEY",
        )
    )

    response = factory.create().complete_turn(
        _agent_loop_request(
            "app.py",
            tool_results=[
                ModelToolResult(
                    call_id="call-1",
                    tool_name="compare_base_head",
                    content="diff",
                    observation_ids=["obs-123"],
                )
            ],
        )
    )
    payload = json.loads(response.final_text)

    assert response.kind is ModelResponseKind.FINAL
    assert response.provider_name == "fake"
    assert payload["status"] == "completed"
    assert payload["observation_refs"] == ["obs-123"]
    assert payload["contract_assessments"][0]["evidence_refs"] == ["obs-123"]


def test_factory_fake_adapter_dispatches_intent_inference_schema_without_changing_reviewer_behavior():
    factory = build_model_adapter_factory_from_config(
        ModelAdapterConfig(
            provider_name="fake",
            model=None,
            base_url=None,
            api_key_env="REVIEW_AGENT_API_KEY",
        )
    )
    adapter = factory.create()

    intent_response = adapter.complete_turn(
        ModelTurnRequest(
            system="You are the Intent Analyst.",
            tools=[ModelToolSpec(name="read_commit_messages", description="Read commits")],
            messages=[{"role": "user", "content": "Infer intent"}],
            tool_results=[],
            parameters={
                "tool_choice": "auto",
                "response_schema": "intent_inference_result_v1",
            },
        )
    )
    reviewer_response = adapter.complete_turn(_agent_loop_request(""))
    intent_payload = json.loads(intent_response.final_text)
    reviewer_payload = json.loads(reviewer_response.final_text)

    assert intent_response.kind is ModelResponseKind.FINAL
    assert intent_response.model == "fake-intent-analyst"
    assert intent_payload["summary"] == "Fake intent inference executed."
    assert intent_payload["candidates"][0]["origin"] == "llm_inference"
    assert set(intent_payload) == {"candidates", "uncertainties", "summary"}
    assert reviewer_response.kind is ModelResponseKind.FINAL
    assert reviewer_response.model == "fake-reviewer"
    assert reviewer_payload["status"] == "completed"


@pytest.mark.parametrize(
    ("response_schema", "expected_model", "expected_field"),
    [
        ("risk_proposal_v1", "fake-risk-assessor", "dimensions"),
        ("portfolio_proposal_v1", "fake-portfolio-planner", "candidates"),
    ],
)
def test_factory_fake_adapter_dispatches_planning_stage_schemas(
    response_schema: str,
    expected_model: str,
    expected_field: str,
) -> None:
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
            system="planning stage",
            tools=[],
            messages=[{"role": "user", "content": "{}"}],
            tool_results=[],
            parameters={
                "tool_choice": "none",
                "response_schema": response_schema,
            },
        )
    )
    payload = json.loads(response.final_text)

    assert response.kind is ModelResponseKind.FINAL
    assert response.model == expected_model
    assert expected_field in payload
    assert response.raw == {"fake": True, "response_schema": response_schema}


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


@pytest.mark.parametrize(
    ("missing_field", "expected_option"),
    [
        ("model", "--risk-assessor-model"),
        ("base_url", "--risk-assessor-base-url"),
    ],
)
def test_factory_uses_custom_stage_label_in_configuration_errors(
    monkeypatch,
    missing_field: str,
    expected_option: str,
) -> None:
    monkeypatch.setenv("RISK_API_KEY", "secret-key")
    values = {
        "provider_name": "openai-compatible",
        "model": "risk-model",
        "base_url": "https://example.test/v1",
        "api_key_env": "RISK_API_KEY",
    }
    values[missing_field] = None

    with pytest.raises(AdapterConfigError, match=expected_option):
        build_model_adapter_factory_from_config(
            ModelAdapterConfig(**values),
            stage_label="risk_assessor",
        )


def test_factory_preserves_reviewer_as_default_error_label(monkeypatch) -> None:
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


def test_factory_accepts_stage_label_from_adapter_config(monkeypatch) -> None:
    monkeypatch.setenv("PLANNER_API_KEY", "secret-key")

    with pytest.raises(AdapterConfigError, match="--portfolio-planner-base-url"):
        build_model_adapter_factory_from_config(
            ModelAdapterConfig(
                provider_name="openai-compatible",
                model="planner-model",
                base_url=None,
                api_key_env="PLANNER_API_KEY",
                stage_label="portfolio_planner",
            )
        )


def test_factory_labels_stage_specific_missing_api_key(monkeypatch) -> None:
    monkeypatch.delenv("RISK_API_KEY", raising=False)

    with pytest.raises(AdapterConfigError, match="risk-assessor.*RISK_API_KEY"):
        build_model_adapter_factory_from_config(
            ModelAdapterConfig(
                provider_name="openai-compatible",
                model="risk-model",
                base_url="https://example.test/v1",
                api_key_env="RISK_API_KEY",
                stage_label="risk_assessor",
            )
        )


def _agent_loop_request(changed_files: str, tool_results: list[ModelToolResult] | None = None) -> ModelTurnRequest:
    return ModelTurnRequest(
        system="system",
        tools=[ModelToolSpec(name="compare_base_head", description="Compare base and head")],
        messages=[
            {
                "role": "user",
                "content": "\n".join(
                    [
                        "Assignment",
                        "Initial Context",
                        f"Changed Files: {changed_files}",
                        "Diff Ranges: ",
                    ]
                ),
            }
        ],
        tool_results=tool_results or [],
        parameters={"tool_choice": "auto"},
    )
