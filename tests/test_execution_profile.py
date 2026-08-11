from __future__ import annotations

from dataclasses import asdict, replace

import pytest

import review_agent.execution_profile as profile_module
from review_agent.command import review_execution_profile_from_arguments
from review_agent.context import ContextBudget
from review_agent.execution_profile import (
    AgentExecutionProfile,
    reviewer_execution_profile_v2,
)
from review_agent.memory_models import MemoryExecutionConfig, MemoryMode
from review_agent.models import ReviewProfile, RiskLevel
from review_agent.session import ReviewExecutionConfig, SupplementalPolicy
from review_agent.review_policy import DeveloperReviewPolicy


def execution_config(
    root: object,
    *,
    reviewer_mode: str = "single",
    reviewer_loop: str = "agent-loop",
) -> ReviewExecutionConfig:
    return ReviewExecutionConfig(
        reviewer_provider="openai-compatible",
        reviewer_model="deepseek-v4-pro",
        reviewer_base_url="https://api.deepseek.com",
        reviewer_api_key_env="REVIEW_AGENT_API_KEY",
        reviewer_mode=reviewer_mode,
        reviewer_loop=reviewer_loop,
        non_interactive=False,
        memory=MemoryExecutionConfig(
            mode=MemoryMode.OFF,
            root_path=str(root),
        ),
    )


def test_profile_is_product_projection_and_ignores_trial_memory_root(
    tmp_path,
) -> None:
    config = execution_config(tmp_path / "memory-a")
    first = AgentExecutionProfile.from_execution(config)
    same_config_different_memory_root = AgentExecutionProfile.from_execution(
        replace(
            config,
            memory=replace(config.memory, root_path=str(tmp_path / "memory-b")),
        )
    )

    assert first.digest() == same_config_different_memory_root.digest()
    single_shot = AgentExecutionProfile.from_execution(
        replace(config, reviewer_loop="single-shot")
    )
    assert first.digest() != single_shot.digest()
    assert first.payload["capabilities"] == {
        "shell": "unavailable",
        "network": "provider_only",
        "repository": "read_only",
        "run_safe_check": "unavailable",
    }
    assert first.payload["execution"]["memory"]["root_binding"] == (
        "trial_private"
    )
    assert "root_path" not in first.payload["execution"]["memory"]
    assert "search_code" in first.payload["reviewer_protocol"]["tool_names"]
    assert "bash" not in first.payload["reviewer_protocol"]["tool_names"]
    assert "read_commit_messages" in first.payload["intent_protocol"][
        "tool_names"
    ]
    assert first.payload["reviewer_protocol"]["context_budget"] == asdict(
        ContextBudget()
    )
    assert first.payload["reviewer_protocol"]["invocation_defaults"] == {
        "reasoning_effort": "medium",
        "temperature": 0,
        "tool_choice_policy": "auto_if_tools_else_none",
        "response_schema": "reviewer_assignment_result_v2",
    }
    assert first.payload["tool_gateway_limits"] == {
        "max_context_chars": 4_000,
        "timeout_seconds": 10,
        "max_commit_messages": 50,
        "max_commit_body_chars": 4_000,
    }
    assert first.payload["intent_protocol"]["runtime_limits"] == {
        "max_turns": 4,
        "max_tool_calls": 8,
        "max_output_tokens": 4_096,
    }
    assert first.payload["intent_protocol"]["invocation_defaults"] == {
        "reasoning_effort": "low",
        "temperature": 0,
        "tool_choice": "auto",
        "response_schema": "intent_inference_result_v1",
    }
    assert first.payload["provider_transport"]["openai_compatible"] == {
        "request_timeout_seconds": 180,
        "max_response_bytes": 16 * 1024 * 1024,
    }
    assert AgentExecutionProfile.from_dict(first.to_dict()).digest() == (
        first.digest()
    )
    with pytest.raises(TypeError):
        first.payload["execution"]["reviewer_loop"] = "single-shot"


def test_profile_timeout_accounts_for_sequential_and_parallel_portfolios(
    tmp_path,
) -> None:
    single = AgentExecutionProfile.from_execution(
        execution_config(tmp_path / "single", reviewer_mode="single")
    )
    multi = AgentExecutionProfile.from_execution(
        execution_config(tmp_path / "multi", reviewer_mode="multi")
    )
    profiles = {
        risk: ReviewProfile.for_risk(risk) for risk in RiskLevel
    }
    sequential_review = max(
        item.reviewer_count * item.max_elapsed_seconds
        for item in profiles.values()
    )
    parallel_review = max(
        item.max_elapsed_seconds for item in profiles.values()
    )
    fixed_work = 4 * 180 + 8 * 10 + SupplementalPolicy().max_elapsed_seconds + 300

    assert single.payload["minimum_outer_timeout_seconds"] == (
        fixed_work + sequential_review
    )
    assert multi.payload["minimum_outer_timeout_seconds"] == (
        fixed_work + parallel_review
    )
    assert single.payload["minimum_outer_timeout_seconds"] > multi.payload[
        "minimum_outer_timeout_seconds"
    ]


def test_product_argument_resolution_builds_the_same_profile(tmp_path) -> None:
    memory_root = (tmp_path / "argument-memory").resolve()
    from_arguments = review_execution_profile_from_arguments(
        (
            "--reviewer-provider=openai-compatible",
            "--reviewer-model=deepseek-v4-pro",
            "--reviewer-base-url=https://api.deepseek.com",
            "--reviewer-api-key-env=REVIEW_AGENT_API_KEY",
            "--reviewer-loop=agent-loop",
        ),
        memory_mode="off",
        memory_root=memory_root,
    )
    direct = AgentExecutionProfile.from_execution(
        execution_config(memory_root)
    )

    assert from_arguments.to_dict() == direct.to_dict()
    assert from_arguments.digest() == direct.digest()


def test_profile_digest_changes_with_risk_and_protocol_inputs(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = execution_config(tmp_path / "sensitivity")
    baseline = AgentExecutionProfile.from_execution(config).digest()

    original_for_risk = ReviewProfile.for_risk

    def changed_for_risk(cls, risk: RiskLevel) -> ReviewProfile:
        profile = original_for_risk(risk)
        if risk is RiskLevel.CRITICAL:
            return replace(
                profile,
                max_elapsed_seconds=profile.max_elapsed_seconds + 1,
            )
        return profile

    monkeypatch.setattr(
        ReviewProfile,
        "for_risk",
        classmethod(changed_for_risk),
    )
    assert AgentExecutionProfile.from_execution(config).digest() != baseline
    monkeypatch.undo()

    projections = (
        "reviewer_protocol_projection",
        "intent_inference_protocol_projection",
        "tool_gateway_limits_projection",
        "provider_transport_projection",
    )
    for name in projections:
        original = getattr(profile_module, name)

        def changed_projection(original=original):
            payload = original()
            payload["test_identity_change"] = True
            return payload

        monkeypatch.setattr(profile_module, name, changed_projection)
        assert AgentExecutionProfile.from_execution(config).digest() != baseline
        monkeypatch.undo()

    monkeypatch.setattr(
        profile_module,
        "RISK_MODEL_SYSTEM_PROMPT",
        profile_module.RISK_MODEL_SYSTEM_PROMPT + "\nidentity change",
    )
    assert AgentExecutionProfile.from_execution(config).digest() != baseline


def test_v2_reviewer_profile_binds_developer_policy_without_legacy_budgets() -> None:
    policy = DeveloperReviewPolicy(
        policy_id="product-review-policy-v1",
        content="Report concrete defects and preserve evidence-backed findings.",
        locked_topics=("finding-suppression",),
    )

    profile = reviewer_execution_profile_v2(policy)
    encoded = str(profile)

    assert profile["developer_policy_sha256"] == policy.digest()
    assert len(profile["reviewer_system_prompt_sha256"]) == 64
    assert profile["diff_fit_policy"]["target_initial_tokens"] >= 500_000
    assert profile["runtime_limits"] == {
        "max_elapsed_seconds": 1_800.0,
        "max_provider_attempts": 3,
        "tool_timeout_seconds": 300.0,
    }
    assert "max_message_chars" not in encoded
    assert "compacted_section_min_chars" not in encoded
    assert "memory_subbudget_ratio" not in encoded
    assert "max_output_tokens" not in encoded
