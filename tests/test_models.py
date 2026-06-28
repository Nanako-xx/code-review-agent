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


def test_contract_status_values_are_stable():
    assert ContractItemStatus.COVERED.value == "covered"
    assert ContractItemStatus.NOT_APPLICABLE.value == "not_applicable"
