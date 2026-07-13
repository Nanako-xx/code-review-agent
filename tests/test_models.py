import pytest

from review_agent.models import (
    Assignment,
    ContractItemStatus,
    InitialContext,
    IntentPacket,
    IntentSource,
    IntentStatus,
    QualityGateResult,
    ReviewProfile,
    ReviewRequest,
    ReviewerRuntimeMetadata,
    ReviewerTerminationReason,
    RiskAssessment,
    RiskLevel,
)


def test_review_request_requires_base_and_head():
    request = ReviewRequest(
        repository_path="C:/repo",
        base_revision="main",
        head_revision="HEAD",
        user_intent="tighten auth checks",
        review_focus="backward compatibility",
    )

    assert request.base_revision == "main"
    assert request.head_revision == "HEAD"
    assert request.user_intent == "tighten auth checks"
    assert request.review_focus == "backward compatibility"


def test_intent_packet_tracks_source_status_and_uncertainties():
    packet = IntentPacket(
        goal="Add idempotency to payment callback",
        acceptance_criteria=["duplicate callbacks are safe"],
        scope=["payment callback"],
        constraints=["do not double charge"],
        sources={"goal": IntentSource.INFERRED},
        status=IntentStatus.PARTIAL,
        uncertainties=["whether duplicate callback should return 200 or 409"],
    )

    assert IntentSource.EXPLICIT.value == "explicit"
    assert IntentSource.INFERRED.value == "inferred"
    assert packet.sources["goal"] is IntentSource.INFERRED
    assert packet.status is IntentStatus.PARTIAL
    assert packet.uncertainties == ["whether duplicate callback should return 200 or 409"]


def test_quality_gate_uses_observation_ref_name():
    result = QualityGateResult(
        name="python_compile",
        status="passed",
        command=["python", "-m", "compileall"],
        summary="compiled 2 files",
        observation_ref="O-quality-python-compile",
    )

    assert result.observation_ref == "O-quality-python-compile"


def test_quality_gate_rejects_unknown_status_and_non_finite_duration():
    with pytest.raises(ValueError, match="status"):
        QualityGateResult(
            name="compile",
            status="running",
            command=["python", "-m", "compileall"],
            summary="not terminal",
        )

    with pytest.raises(ValueError, match="duration_seconds"):
        QualityGateResult(
            name="compile",
            status="passed",
            command=["python", "-m", "compileall"],
            summary="ok",
            duration_seconds=float("nan"),
        )


def test_risk_assessment_uses_signal_refs_and_uncertainties():
    assessment = RiskAssessment(
        level=RiskLevel.HIGH,
        dimensions={"impact": "sensitive path"},
        reasons=["sensitive path changed: auth.py"],
        signal_refs=["diff:auth.py", "quality_gate:python_compile"],
        uncertainties=["acceptance criteria are not explicitly declared"],
        suggested_focus=["caller compatibility"],
    )

    assert assessment.signal_refs == ["diff:auth.py", "quality_gate:python_compile"]
    assert assessment.uncertainties == ["acceptance criteria are not explicitly declared"]


def test_assignment_has_structured_initial_context():
    assignment = Assignment(
        role="Caller Compatibility Reviewer",
        mission="Inspect callers affected by changed public API",
        assignment_reason=["public API changed", "legacy callers exist"],
        assigned_contract=["regression_safety"],
        required_checks=["inspect direct callers or record why unavailable"],
        initial_context=InitialContext(
            changed_files=["src/api.py"],
            diff_ranges=["src/api.py:10-30"],
            code_ranges=["src/api.py:10-30"],
            quality_gate_summary={"python_compile": "passed"},
            observation_refs=["O-diff-api"],
        ),
        max_turns=8,
        max_tool_calls=20,
    )

    assert assignment.initial_context.changed_files == ["src/api.py"]
    assert assignment.initial_context.observation_refs == ["O-diff-api"]


def test_review_profile_maps_risk_to_depth():
    profile = ReviewProfile.for_risk(RiskLevel.HIGH)

    assert profile.reviewer_count == 3
    assert profile.max_turns_per_reviewer == 16
    assert "dynamic_specialist" in profile.reviewer_roles


def test_assignment_persists_runtime_compiled_identity() -> None:
    assignment = Assignment(
        role="Async Lifecycle Reviewer",
        mission="Inspect cancellation and retry lifecycle",
        assignment_reason=["risk_reason:0"],
        assigned_contract=["regression_safety", "unresolved_uncertainties"],
        required_checks=["inspect cancellation paths"],
        initial_context=InitialContext(),
        max_turns=16,
        max_tool_calls=40,
        assignment_id="assignment-2",
        role_kind="specialist",
        perspective_key="async_lifecycle",
        planner_source="model",
    )

    assert assignment.assignment_id == "assignment-2"
    assert assignment.role_kind == "specialist"
    assert assignment.repository_permission == "read_only"
    assert assignment.command_permission == "safe_checks_only"


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        ("max_turns", 0, "max_turns"),
        ("max_tool_calls", -1, "max_tool_calls"),
        ("role_kind", "judge", "role_kind"),
        ("repository_permission", "write", "repository_permission"),
        ("command_permission", "shell", "command_permission"),
    ],
)
def test_assignment_rejects_runtime_authority_escalation(
    field_name: str,
    invalid_value: object,
    message: str,
) -> None:
    values = {
        "role": "Core Reviewer",
        "mission": "Review the change",
        "assignment_reason": [],
        "assigned_contract": [],
        "required_checks": [],
        "initial_context": InitialContext(),
        "max_turns": 1,
        "max_tool_calls": 1,
        field_name: invalid_value,
    }

    with pytest.raises(ValueError, match=message):
        Assignment(**values)


def test_review_profiles_expand_every_runtime_budget_by_risk():
    profiles = [ReviewProfile.for_risk(level) for level in RiskLevel]

    assert [profile.max_total_tokens for profile in profiles] == sorted(
        profile.max_total_tokens for profile in profiles
    )
    assert [profile.max_elapsed_seconds for profile in profiles] == sorted(
        profile.max_elapsed_seconds for profile in profiles
    )
    for profile in profiles:
        assert profile.max_output_tokens > 0
        assert profile.max_total_tokens > 0
        assert profile.max_elapsed_seconds > 0
        assert profile.max_provider_attempts > 0


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        ("max_output_tokens", 0, "max_output_tokens"),
        ("max_total_tokens", -1, "max_total_tokens"),
        ("max_elapsed_seconds", 0.0, "max_elapsed_seconds"),
        ("max_elapsed_seconds", float("nan"), "max_elapsed_seconds"),
        ("max_provider_attempts", True, "max_provider_attempts"),
    ],
)
def test_assignment_requires_strictly_positive_runtime_budgets(
    field_name: str,
    invalid_value: object,
    message: str,
):
    values = {
        "role": "core",
        "mission": "review",
        "assignment_reason": [],
        "assigned_contract": [],
        "required_checks": [],
        "initial_context": InitialContext(),
        "max_turns": 1,
        "max_tool_calls": 1,
        field_name: invalid_value,
    }

    with pytest.raises(ValueError, match=message):
        Assignment(**values)


def test_reviewer_runtime_metadata_has_legacy_unknown_defaults():
    runtime = ReviewerRuntimeMetadata()

    assert runtime.provider_attempts == 0
    assert runtime.total_tokens == 0
    assert runtime.usage_available is False
    assert runtime.elapsed_seconds == 0.0
    assert runtime.termination_reason is ReviewerTerminationReason.LEGACY_UNKNOWN


def test_contract_status_values_are_stable():
    assert ContractItemStatus.COVERED.value == "covered"
    assert ContractItemStatus.NOT_APPLICABLE.value == "not_applicable"
