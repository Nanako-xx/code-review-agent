from review_agent.models import (
    Assignment,
    ContractItemStatus,
    IntentPacket,
    IntentSource,
    IntentStatus,
    ReviewProfile,
    ReviewRequest,
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


def test_intent_packet_tracks_source_and_status():
    packet = IntentPacket(
        goal="Add idempotency to payment callback",
        acceptance_criteria=["duplicate callbacks are safe"],
        scope=["payment callback"],
        constraints=["do not double charge"],
        sources={"goal": IntentSource.DECLARED},
        status=IntentStatus.PARTIAL,
        unknowns=["whether duplicate callback should return 200 or 409"],
    )

    assert packet.sources["goal"] is IntentSource.DECLARED
    assert packet.status is IntentStatus.PARTIAL
    assert packet.unknowns == ["whether duplicate callback should return 200 or 409"]


def test_assignment_is_runtime_expanded_not_raw_risk_label():
    assignment = Assignment(
        role="Caller Compatibility Reviewer",
        mission="Inspect callers affected by changed public API",
        assignment_reason=["public API changed", "legacy callers exist"],
        assigned_contract=["regression_safety"],
        required_checks=["inspect direct callers or record why unavailable"],
        provided_evidence_refs=["ev_1"],
        code_ranges=["src/api.py:10-30"],
        max_turns=8,
        max_tool_calls=20,
    )

    assert assignment.assignment_reason == ["public API changed", "legacy callers exist"]
    assert assignment.required_checks == ["inspect direct callers or record why unavailable"]


def test_review_profile_maps_risk_to_depth():
    profile = ReviewProfile.for_risk(RiskLevel.HIGH)

    assert profile.reviewer_count == 3
    assert profile.max_turns_per_reviewer == 16
    assert "dynamic_specialist" in profile.reviewer_roles


def test_contract_status_values_are_stable():
    assert ContractItemStatus.COVERED.value == "covered"
    assert ContractItemStatus.NOT_APPLICABLE.value == "not_applicable"
