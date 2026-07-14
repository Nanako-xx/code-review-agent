from types import SimpleNamespace

import pytest

from review_agent.models import InitialContext, RiskLevel
from review_agent.supplemental import (
    BudgetAmount,
    BudgetExceededError,
    BudgetLedger,
    ReviewerBudgetCaps,
    SupplementalInvestigationRequest,
    compile_supplemental_plan,
    deduplicate_supplemental_requests,
    effective_policy_for_risk,
    limits_for_risk,
    stable_invocation_id,
    stable_request_id,
    stable_wave_id,
)


BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40


def _request(
    question: str = "Does the retry path duplicate jobs?",
    *,
    disagreement_id: str = "D-retry",
    evidence: tuple[str, ...] = ("inspect retry caller", "check idempotency guard"),
    perspective: str = "concurrency",
    candidates: tuple[str, ...] = ("F-2", "F-1"),
) -> SupplementalInvestigationRequest:
    return SupplementalInvestigationRequest(
        source_disagreement_id=disagreement_id,
        question=question,
        required_evidence=evidence,
        preferred_perspective=perspective,
        source_candidate_ids=candidates,
        reason_refs=("O-retry",),
    )


def test_stable_ids_ignore_request_list_order_and_provider_retry_order() -> None:
    first = _request()
    equivalent = _request(
        "  DOES the retry path   duplicate jobs? ",
        evidence=("check idempotency guard", "inspect retry caller"),
        perspective="Concurrency",
        candidates=("F-1", "F-2"),
    )

    assert first.request_id == equivalent.request_id
    assert first.request_id == stable_request_id(first)
    wave_id = stable_wave_id(
        review_id="review-1",
        base_sha=BASE_SHA,
        head_sha=HEAD_SHA,
        wave_index=1,
        trigger_digest="trigger-a",
        policy_version="supplemental_policy_v1",
    )
    assert wave_id == stable_wave_id(
        review_id="review-1",
        base_sha=BASE_SHA,
        head_sha=HEAD_SHA,
        wave_index=1,
        trigger_digest="trigger-a",
        policy_version="supplemental_policy_v1",
    )
    assert wave_id != stable_wave_id(
        review_id="review-1",
        base_sha=BASE_SHA,
        head_sha="c" * 40,
        wave_index=1,
        trigger_digest="trigger-a",
        policy_version="supplemental_policy_v1",
    )
    invocation = stable_invocation_id(
        task_or_batch_id="STASK-a",
        logical_turn=2,
        request_digest=first.request_id,
    )
    assert invocation == stable_invocation_id(
        task_or_batch_id="STASK-a",
        logical_turn=2,
        request_digest=first.request_id,
        provider_attempt=99,
    )


def test_revision_drift_recomputes_request_wave_task_and_invocation_ids() -> None:
    before_request = _request(candidates=("F-before-1", "F-before-2"))
    after_request = _request(candidates=("F-after-1", "F-after-2"))

    before = compile_supplemental_plan(
        review_id="review-drift",
        base_sha=BASE_SHA,
        head_sha=HEAD_SHA,
        risk_level=RiskLevel.HIGH,
        wave_index=1,
        trigger_digest="same-semantic-trigger",
        requests=[before_request],
    )
    after = compile_supplemental_plan(
        review_id="review-drift",
        base_sha=BASE_SHA,
        head_sha="c" * 40,
        risk_level=RiskLevel.HIGH,
        wave_index=1,
        trigger_digest="same-semantic-trigger",
        requests=[after_request],
    )

    before_task = before.tasks[0]
    after_task = after.tasks[0]
    before_invocation = stable_invocation_id(
        task_or_batch_id=before_task.task_id,
        logical_turn=1,
        request_digest=before_task.request_id,
    )
    after_invocation = stable_invocation_id(
        task_or_batch_id=after_task.task_id,
        logical_turn=1,
        request_digest=after_task.request_id,
    )

    assert before_request.request_id != after_request.request_id
    assert before.wave_id != after.wave_id
    assert before_task.assignment.assignment_id != after_task.assignment.assignment_id
    assert before_task.task_id != after_task.task_id
    assert before_invocation != after_invocation


@pytest.mark.parametrize(
    ("risk", "expected"),
    [
        (RiskLevel.LOW, (1, 1, 1, 1, 4, 8, 16_384, 16_384, 120.0)),
        (RiskLevel.MEDIUM, (1, 2, 2, 2, 6, 12, 32_768, 65_536, 240.0)),
        (RiskLevel.HIGH, (2, 3, 2, 2, 8, 16, 49_152, 147_456, 480.0)),
        (RiskLevel.CRITICAL, (2, 4, 2, 2, 10, 24, 65_536, 262_144, 600.0)),
    ],
)
def test_risk_defaults_compile_to_concrete_runtime_limits(
    risk: RiskLevel,
    expected: tuple[int, int, int, int, int, int, int, int, float],
) -> None:
    limits = limits_for_risk(risk)

    assert (
        limits.max_waves,
        limits.max_tasks,
        limits.max_tasks_per_wave,
        limits.max_concurrency,
        limits.max_turns_per_task,
        limits.max_tool_calls_per_task,
        limits.max_total_tokens_per_task,
        limits.max_total_tokens,
        limits.max_elapsed_seconds,
    ) == expected


def test_session_policy_can_only_lower_risk_limits() -> None:
    configured = SimpleNamespace(
        policy_version="project-policy-v2",
        max_waves=99,
        max_tasks=1,
        max_tasks_per_wave=99,
        max_concurrency=1,
        max_turns_per_task=3,
        max_tool_calls_per_task=99,
        max_total_tokens_per_task=10_000,
        max_total_tokens=999_999,
        max_elapsed_seconds=45,
    )

    limits = limits_for_risk(RiskLevel.HIGH, configured)

    assert limits.policy_version == "project-policy-v2"
    assert limits.max_waves == 2
    assert limits.max_tasks == 1
    assert limits.max_tasks_per_wave == 2
    assert limits.max_concurrency == 1
    assert limits.max_turns_per_task == 3
    assert limits.max_tool_calls_per_task == 16
    assert limits.max_total_tokens_per_task == 10_000
    assert limits.max_total_tokens == 147_456
    assert limits.max_elapsed_seconds == 45.0


def test_effective_policy_clamps_cross_field_capacity_and_preserves_risk() -> None:
    configured = SimpleNamespace(
        policy_version="supplemental_policy_v1",
        max_waves=1,
        max_tasks=4,
        max_tasks_per_wave=2,
        max_concurrency=2,
        max_turns_per_task=8,
        max_tool_calls_per_task=16,
        max_total_tokens_per_task=49_152,
        max_total_tokens=147_456,
        max_elapsed_seconds=480,
    )

    limits = limits_for_risk(RiskLevel.HIGH, configured)
    policy = effective_policy_for_risk(RiskLevel.HIGH, configured)

    assert limits.max_tasks == 2
    assert policy.risk_level == "high"
    assert policy.max_waves == 1
    assert policy.max_tasks == 2
    assert policy.max_tasks_per_wave == 2


def test_request_deduplication_is_stable_and_reports_all_dropped_ids() -> None:
    first = _request()
    duplicate = _request("does the retry path duplicate jobs?")
    second = _request(
        "Can cancellation lose the persisted result?",
        disagreement_id="D-cancel",
        candidates=("F-3",),
    )

    unique, dropped = deduplicate_supplemental_requests(
        [second, duplicate, first],
    )

    assert [item.request_id for item in unique] == sorted(
        {first.request_id, second.request_id}
    )
    assert dropped == (duplicate.request_id,)


def test_compiler_enforces_role_contract_permissions_tools_and_task_budget() -> None:
    request = _request()
    target_context = InitialContext(
        changed_files=["src/retry.py"],
        code_ranges=["src/retry.py:40-80"],
        observation_refs=["O-retry"],
        signal_refs=["F-1", "F-2"],
    )
    caps = ReviewerBudgetCaps(
        max_output_tokens=2_048,
        max_total_tokens=20_000,
        max_elapsed_seconds=30,
        max_provider_attempts=1,
    )

    plan = compile_supplemental_plan(
        review_id="review-1",
        base_sha=BASE_SHA,
        head_sha=HEAD_SHA,
        risk_level=RiskLevel.MEDIUM,
        wave_index=1,
        trigger_digest="trigger-a",
        requests=[request],
        initial_context_by_request={request.request_id: target_context},
        reviewer_budget_caps=caps,
        allowed_tools=("read_range", "search_code"),
    )

    assert plan.status == "planned"
    assert len(plan.tasks) == 1
    task = plan.tasks[0]
    assignment = task.assignment
    assert task.request_id == request.request_id
    assert task.bootstrap_policy == "targeted_only"
    assert task.allowed_tools == ("search_code", "read_range")
    assert assignment.planner_source == "semantic_reconciler"
    assert assignment.role_kind == "specialist"
    assert assignment.perspective_key == "supplemental:concurrency"
    assert assignment.assigned_contract == ["supplemental_investigation:D-retry"]
    assert assignment.repository_permission == "read_only"
    assert assignment.command_permission == "safe_checks_only"
    assert assignment.initial_context == target_context
    assert assignment.max_turns == 6
    assert assignment.max_tool_calls == 12
    assert assignment.max_output_tokens == 2_048
    assert assignment.max_total_tokens == 20_000
    assert assignment.max_elapsed_seconds == 30.0
    assert assignment.max_provider_attempts == 1
    assert task.budget_reservation == BudgetAmount(
        tasks=1,
        tool_calls=12,
        tokens=20_000,
        elapsed_seconds=30,
    )
    assert task.counts_toward_initial_coverage is False


def test_compiler_deduplicates_prior_requests_and_records_policy_truncation() -> None:
    prior = _request()
    duplicate = _request("does the retry path duplicate jobs?")
    accepted = _request(
        "Can cancellation lose the persisted result?",
        disagreement_id="D-cancel",
        candidates=("F-3",),
    )
    truncated = _request(
        "Can shutdown leak a worker?",
        disagreement_id="D-shutdown",
        candidates=("F-4",),
    )
    configured = SimpleNamespace(
        max_waves=1,
        max_tasks=2,
        max_tasks_per_wave=1,
        max_concurrency=1,
        max_turns_per_task=6,
        max_tool_calls_per_task=12,
        max_total_tokens_per_task=32_768,
        max_total_tokens=65_536,
        max_elapsed_seconds=240,
    )

    plan = compile_supplemental_plan(
        review_id="review-1",
        base_sha=BASE_SHA,
        head_sha=HEAD_SHA,
        risk_level=RiskLevel.MEDIUM,
        wave_index=1,
        trigger_digest="trigger-a",
        requests=[duplicate, truncated, accepted],
        configured_policy=configured,
        prior_request_ids=(prior.request_id,),
    )

    assert len(plan.tasks) == 1
    assert plan.tasks[0].request_id == min(
        accepted.request_id,
        truncated.request_id,
    )
    assert duplicate.request_id in plan.dropped_request_ids
    assert len(plan.dropped_request_ids) == 2
    assert any(action.startswith("deduplicated_request:") for action in plan.policy_actions)
    assert any(action.startswith("truncated_request:") for action in plan.policy_actions)


def test_wave_and_task_bounds_have_no_off_by_one() -> None:
    request = _request()
    with pytest.raises(ValueError, match="wave_index"):
        compile_supplemental_plan(
            review_id="review-1",
            base_sha=BASE_SHA,
            head_sha=HEAD_SHA,
            risk_level=RiskLevel.HIGH,
            wave_index=0,
            trigger_digest="trigger-a",
            requests=[request],
        )

    final_allowed = compile_supplemental_plan(
        review_id="review-1",
        base_sha=BASE_SHA,
        head_sha=HEAD_SHA,
        risk_level=RiskLevel.HIGH,
        wave_index=2,
        trigger_digest="trigger-a",
        requests=[request],
    )
    beyond_limit = compile_supplemental_plan(
        review_id="review-1",
        base_sha=BASE_SHA,
        head_sha=HEAD_SHA,
        risk_level=RiskLevel.HIGH,
        wave_index=3,
        trigger_digest="trigger-a",
        requests=[request],
    )
    exhausted_tasks = compile_supplemental_plan(
        review_id="review-1",
        base_sha=BASE_SHA,
        head_sha=HEAD_SHA,
        risk_level=RiskLevel.LOW,
        wave_index=1,
        trigger_digest="trigger-a",
        requests=[request],
        prior_task_count=1,
    )

    assert len(final_allowed.tasks) == 1
    assert beyond_limit.status == "max_waves"
    assert beyond_limit.tasks == ()
    assert exhausted_tasks.status == "policy_limited"
    assert exhausted_tasks.tasks == ()


def test_budget_ledger_reserves_before_submit_and_prevents_oversubscription() -> None:
    ledger = BudgetLedger(
        limits=BudgetAmount(tasks=2, tool_calls=10, tokens=100, elapsed_seconds=20)
    )
    first = BudgetAmount(tasks=1, tool_calls=4, tokens=60, elapsed_seconds=8)

    assert ledger.reserve("STASK-1", first) == first
    assert ledger.reserve("STASK-1", first) == first
    assert ledger.reserved == first
    with pytest.raises(BudgetExceededError, match="tokens"):
        ledger.reserve(
            "STASK-2",
            BudgetAmount(tasks=1, tool_calls=4, tokens=50, elapsed_seconds=8),
        )
    assert ledger.stop_reason == "budget_exhausted"


@pytest.mark.parametrize(
    "reservation",
    [
        BudgetAmount(tasks=1, tool_calls=11, tokens=1, elapsed_seconds=1),
        BudgetAmount(tasks=1, tool_calls=1, tokens=101, elapsed_seconds=1),
        BudgetAmount(tasks=1, tool_calls=1, tokens=1, elapsed_seconds=21),
    ],
)
def test_budget_ledger_exhausts_each_global_resource_dimension(
    reservation: BudgetAmount,
) -> None:
    ledger = BudgetLedger(
        limits=BudgetAmount(tasks=2, tool_calls=10, tokens=100, elapsed_seconds=20)
    )

    with pytest.raises(BudgetExceededError):
        ledger.reserve("STASK-over-limit", reservation)

    assert ledger.stop_reason == "budget_exhausted"


def test_budget_ledger_charges_actual_usage_when_provider_usage_is_available() -> None:
    ledger = BudgetLedger(
        limits=BudgetAmount(tasks=2, tool_calls=10, tokens=100, elapsed_seconds=20)
    )
    ledger.reserve(
        "STASK-1",
        BudgetAmount(tasks=1, tool_calls=4, tokens=60, elapsed_seconds=8),
    )

    charged = ledger.charge(
        "STASK-1",
        BudgetAmount(tasks=1, tool_calls=2, tokens=31, elapsed_seconds=3),
        usage_available=True,
    )

    assert charged == BudgetAmount(tasks=1, tool_calls=2, tokens=31, elapsed_seconds=3)
    assert ledger.reserved == BudgetAmount()
    assert ledger.charged == charged
    assert ledger.remaining == BudgetAmount(
        tasks=1,
        tool_calls=8,
        tokens=69,
        elapsed_seconds=17,
    )


def test_missing_provider_usage_is_charged_at_conservative_reservation() -> None:
    ledger = BudgetLedger(
        limits=BudgetAmount(tasks=1, tool_calls=4, tokens=60, elapsed_seconds=8)
    )
    reservation = BudgetAmount(tasks=1, tool_calls=4, tokens=60, elapsed_seconds=8)
    ledger.reserve("STASK-1", reservation)

    charged = ledger.charge(
        "STASK-1",
        BudgetAmount(tasks=1, tool_calls=1, tokens=0, elapsed_seconds=2),
        usage_available=False,
    )

    assert charged == reservation
    assert ledger.remaining == BudgetAmount()


def test_unknown_consumption_cannot_be_refunded_on_resume() -> None:
    ledger = BudgetLedger(
        limits=BudgetAmount(tasks=2, tool_calls=10, tokens=100, elapsed_seconds=20)
    )
    reservation = BudgetAmount(tasks=1, tool_calls=4, tokens=60, elapsed_seconds=8)
    ledger.reserve("STASK-1", reservation)

    ledger.mark_unknown("STASK-1", invocation_id="INV-returned-before-commit")

    assert ledger.reserved == BudgetAmount()
    assert ledger.charged == BudgetAmount(tasks=1, tool_calls=4)
    assert ledger.unknown_consumed.tokens == 60
    assert ledger.unknown_consumed.elapsed_seconds == 8.0
    assert ledger.unknown_consumed.invocation_ids == (
        "INV-returned-before-commit",
    )
    assert ledger.remaining == BudgetAmount(
        tasks=1,
        tool_calls=6,
        tokens=40,
        elapsed_seconds=12,
    )
    with pytest.raises(KeyError, match="STASK-1"):
        ledger.charge("STASK-1", BudgetAmount(tasks=1), usage_available=True)
