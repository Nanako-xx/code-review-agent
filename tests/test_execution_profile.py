from __future__ import annotations

import json

import pytest

from review_agent.command import review_execution_profile_from_arguments
from review_agent.execution_profile import (
    AGENT_EXECUTION_PROFILE_SCHEMA_VERSION,
    AgentExecutionProfile,
)
from review_agent.model_adapter_factory import ModelAdapterConfig
from review_agent.review_policy import DeveloperReviewPolicy


def _config(
    *,
    provider: str = "fake",
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str = "REVIEW_AGENT_API_KEY",
) -> ModelAdapterConfig:
    return ModelAdapterConfig(
        provider_name=provider,
        model=model,
        base_url=base_url,
        api_key_env=api_key_env,
    )


def test_profile_v2_binds_the_complete_product_protocol() -> None:
    profile = AgentExecutionProfile.from_product_configuration(
        reviewer=_config(),
        risk=None,
    ).to_dict()

    assert profile["schema_version"] == AGENT_EXECUTION_PROFILE_SCHEMA_VERSION
    assert set(profile) == {
        "schema_version",
        "configuration",
        "product_protocol",
        "reviewer_execution",
        "tool_result_policy",
        "context_window_policy",
        "provider_transport",
        "minimum_outer_timeout_seconds",
        "capabilities",
    }
    protocol = profile["product_protocol"]
    assert protocol["session"] == {
        "schema_version": 6,
        "phases": [
            "preflight",
            "intent",
            "planning",
            "reviewers",
            "aggregation",
        ],
    }
    assert protocol["intent"]["fields"] == [
        "goal",
        "source",
        "uncertainties",
    ]
    assert protocol["risk"]["fields"] == ["level"]
    assert protocol["reviewer_output"]["finding_fields"] == [
        "claim",
        "severity",
        "path",
        "line",
        "suggestion",
    ]
    assert protocol["aggregation"]["review_result_fields"] == [
        "pr_id",
        "snapshot_id",
        "status",
        "risk_level",
        "findings",
        "uncertainties",
    ]
    assert protocol["aggregation"]["model_calls"] == 0


def test_profile_v2_binds_fixed_slots_and_runtime_limits() -> None:
    payload = AgentExecutionProfile.from_product_configuration(
        reviewer=_config(),
        risk=None,
    ).to_dict()

    slots = payload["product_protocol"]["review_planning"]["slots"]
    assert {level: len(values) for level, values in slots.items()} == {
        "low": 1,
        "medium": 2,
        "high": 3,
        "critical": 4,
    }
    assert payload["reviewer_execution"]["runtime_limits"] == {
        "max_elapsed_seconds": 1800.0,
        "max_provider_attempts": 3,
        "tool_timeout_seconds": 300.0,
    }
    assert payload["tool_result_policy"] == {
        "artifact_threshold_chars": 50_000,
        "turn_budget_chars": 200_000,
        "preview_chars": 2_000,
        "max_artifact_page_chars": 50_000,
    }


def test_profile_v2_binds_one_million_context_compaction_policy() -> None:
    context = AgentExecutionProfile.from_product_configuration(
        reviewer=_config(),
        risk=None,
    ).to_dict()["context_window_policy"]

    assert context["context_window_tokens"] == 1_000_000
    assert context["prompt_cache_idle_eviction_seconds"] == 3_600
    assert context["recent_reacquirable_tool_results_to_keep"] == 5
    assert context["soft_compaction_trigger_tokens"] == 700_000
    assert context["compaction_summary_max_tokens"] == 50_000
    assert len(context["compaction_system_prompt_sha256"]) == 64
    assert len(context["compaction_user_prompt_sha256"]) == 64


def test_profile_v2_contains_no_legacy_post_review_or_character_budgets() -> None:
    payload = AgentExecutionProfile.from_product_configuration(
        reviewer=_config(),
        risk=None,
    ).to_dict()
    encoded = json.dumps(payload, sort_keys=True).casefold()

    for removed in (
        "completion_policy",
        "semantic_reconciler",
        "portfolio_planner",
        "memory_curator",
        "supplemental_policy",
        "review_brief",
        "max_turns",
        "max_tool_calls",
        "max_output_tokens",
        "max_total_tokens",
        "16000",
    ):
        assert removed not in encoded


def test_profile_digest_changes_for_provider_risk_and_policy_identity() -> None:
    baseline = AgentExecutionProfile.from_product_configuration(
        reviewer=_config(),
        risk=None,
    ).digest()
    changed_reviewer = AgentExecutionProfile.from_product_configuration(
        reviewer=_config(model="different"),
        risk=None,
    ).digest()
    changed_risk = AgentExecutionProfile.from_product_configuration(
        reviewer=_config(),
        risk=_config(provider="fake", model="risk-v2"),
    ).digest()
    changed_policy = AgentExecutionProfile.from_product_configuration(
        reviewer=_config(),
        risk=None,
        policy=DeveloperReviewPolicy(
            policy_id="custom-review-policy-v1",
            content="Report only reproducible defects.",
        ),
    ).digest()

    assert len({baseline, changed_reviewer, changed_risk, changed_policy}) == 4


def test_profile_argument_resolution_matches_product_configuration() -> None:
    arguments = (
        "--reviewer-provider=openai-compatible",
        "--reviewer-model=deepseek-v4-pro",
        "--reviewer-base-url=https://api.deepseek.com",
        "--reviewer-api-key-env=REVIEW_AGENT_API_KEY",
        "--risk-assessor-mode=model",
        "--risk-assessor-provider=inherit",
        "--risk-assessor-model=deepseek-v4-risk",
    )

    from_arguments = review_execution_profile_from_arguments(arguments)
    direct = AgentExecutionProfile.from_product_configuration(
        reviewer=_config(
            provider="openai-compatible",
            model="deepseek-v4-pro",
            base_url="https://api.deepseek.com",
        ),
        risk=_config(
            provider="openai-compatible",
            model="deepseek-v4-risk",
            base_url="https://api.deepseek.com",
        ),
    )

    assert from_arguments.to_dict() == direct.to_dict()
    assert from_arguments.digest() == direct.digest()


def test_profile_round_trip_is_strict_and_immutable() -> None:
    original = AgentExecutionProfile.from_product_configuration(
        reviewer=_config(),
        risk=None,
    )
    payload = original.to_dict()

    hydrated = AgentExecutionProfile.from_dict(payload)

    assert hydrated.digest() == original.digest()
    payload["configuration"]["reviewer"]["provider"] = "none"
    assert hydrated.to_dict()["configuration"]["reviewer"]["provider"] == "fake"
    with pytest.raises(ValueError, match="fields are not canonical"):
        AgentExecutionProfile.from_dict({**original.to_dict(), "legacy": True})


def test_profile_never_contains_api_key_values(monkeypatch) -> None:
    secret = "sk-profile-secret-that-must-not-appear"
    monkeypatch.setenv("REVIEW_AGENT_API_KEY", secret)

    encoded = json.dumps(
        AgentExecutionProfile.from_product_configuration(
            reviewer=_config(provider="openai-compatible"),
            risk=None,
        ).to_dict(),
        sort_keys=True,
    )

    assert secret not in encoded
    assert "REVIEW_AGENT_API_KEY" in encoded
