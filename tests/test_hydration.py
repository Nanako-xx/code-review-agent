from __future__ import annotations

from dataclasses import asdict
from enum import Enum
import json

import pytest

from review_agent.brief import ReviewBrief, build_review_brief, review_brief_to_dict
from review_agent.completion import CompletionResult, completion_to_dict
from review_agent.evidence import (
    CanonicalFinding,
    ContractCoverage,
    EvidenceReconciliation,
    RejectedFinding,
    reconciliation_to_dict,
)
from review_agent.final_risk import FinalRiskAssessment, final_risk_to_dict
from review_agent.hydration import (
    assignments_from_dict,
    completion_from_dict,
    final_risk_from_dict,
    intent_from_dict,
    quality_results_from_dict,
    reconciliation_from_dict,
    repository_intelligence_from_dict,
    review_brief_from_dict,
    review_request_from_dict,
    reviewer_execution_from_artifacts,
    reviewer_result_from_dict,
    risk_assessment_from_dict,
    risk_packet_from_dict,
)
from review_agent.model_protocol import ModelResponse
from review_agent.models import (
    Assignment,
    ContractAssessment,
    ContractItemStatus,
    InitialContext,
    IntentPacket,
    IntentSource,
    IntentStatus,
    ModelInvocationEnvelope,
    QualityGateResult,
    ReviewRequest,
    ReviewerFinding,
    ReviewerResult,
    ReviewerResultStatus,
    RiskAssessment,
    RiskAssessmentPacket,
    RiskLevel,
)
from review_agent.repository_intelligence import ChangedSymbol, RepositoryIntelligenceSnapshot
from review_agent.reviewer import reviewer_result_to_dict


def _json(value: object) -> object:
    return json.loads(
        json.dumps(value, default=lambda item: item.value if isinstance(item, Enum) else item)
    )


def _assignment() -> Assignment:
    return Assignment(
        role="core",
        mission="review",
        assignment_reason=["risk"],
        assigned_contract=["correctness"],
        required_checks=["diff"],
        initial_context=InitialContext(
            changed_files=["a.py"],
            diff_ranges=["a.py:1-2"],
            code_ranges=[],
            quality_gate_summary={"compile": "passed"},
            observation_refs=["O-1"],
        ),
        max_turns=3,
        max_tool_calls=4,
    )


def _reviewer_result() -> ReviewerResult:
    return ReviewerResult(
        contract_assessments=[
            ContractAssessment("correctness", ContractItemStatus.COVERED, "checked", ["O-1"])
        ],
        confirmed_findings=[
            ReviewerFinding("bug", "high", "high", ["O-1"], "fix")
        ],
        rejected_hypotheses=["not a race"],
        uncertainties=[],
        observation_refs=["O-1"],
        investigation_summary="done",
        status=ReviewerResultStatus.COMPLETED,
    )


def _review_brief() -> ReviewBrief:
    intent = IntentPacket(
        goal="ship",
        acceptance_criteria=["tests pass"],
        scope=["a.py"],
        constraints=["read only"],
        sources={"goal": IntentSource.EXPLICIT},
        status=IntentStatus.PARTIAL,
        uncertainties=["criteria incomplete"],
    )
    risk = RiskAssessment(
        RiskLevel.MEDIUM,
        {"impact": "bounded"},
        ["reason"],
        ["changed_file_count"],
        ["unknown"],
        ["tests"],
    )
    reconciliation = EvidenceReconciliation(
        canonical_findings=[
            CanonicalFinding("bug", "high", "high", ["O-1"], [0], ["core"], "fix")
        ],
        rejected_findings=[
            RejectedFinding(1, "adversarial", "claim", "unsupported", [], ["O-x"])
        ],
        remaining_disagreements=["severity"],
        contract_coverage=[
            ContractCoverage(0, "core", "correctness", "covered", "ok", ["O-1"], [])
        ],
        evidence_quality="verified",
    )
    final_risk = FinalRiskAssessment(
        "reassessed",
        RiskLevel.MEDIUM,
        RiskLevel.HIGH,
        ["finding"],
        ["finding"],
        [],
        [],
        ["O-1"],
    )
    return build_review_brief(
        review_id="review-1",
        base_revision="a" * 40,
        head_revision="b" * 40,
        intent_packet=intent,
        risk_assessment=risk,
        changed_files=["a.py"],
        quality_results=[QualityGateResult("compile", "passed", ["python", "-m", "compileall"], "ok")],
        observation_summaries={"O-1": "a.py changed"},
        repository_intelligence_summary="modified function f a.py:1-2",
        multi_reviewer_summary={
            "reviewer_count": 2,
            "status_counts": {"completed": 2},
            "roles": ["core", "adversarial"],
        },
        reconciliation_payload=reconciliation_to_dict(reconciliation),
        completion_summary=completion_to_dict(
            CompletionResult("completed", "needs_work", [], [], [])
        ),
        final_risk_assessment=final_risk_to_dict(final_risk),
        incremental_priority={
            "from_revision": "b" * 40,
            "to_revision": "c" * 40,
            "changed_files": ["a.py"],
            "diff_stat": "1 file changed",
            "diff_excerpt": ["+change"],
        },
    )


def test_foundation_artifacts_round_trip_to_typed_values() -> None:
    request = ReviewRequest("C:/repo", "main", "HEAD", user_intent="ship")
    intent = IntentPacket(
        goal="ship",
        acceptance_criteria=["tests pass"],
        scope=["a.py"],
        constraints=["read only"],
        sources={"goal": IntentSource.EXPLICIT},
        status=IntentStatus.PARTIAL,
        uncertainties=["criteria incomplete"],
    )
    packet = RiskAssessmentPacket(
        change_summary={"changed_files": ["a.py"]},
        deterministic_signals={"changed_file_count": 1},
        intent_status=IntentStatus.PARTIAL,
        intent_uncertainties=["criteria incomplete"],
        diff_excerpt=["+pass"],
    )
    risk = RiskAssessment(
        RiskLevel.MEDIUM,
        {"impact": "bounded"},
        ["reason"],
        ["changed_file_count"],
        ["unknown"],
        ["tests"],
    )
    quality = [QualityGateResult("compile", "passed", ["python", "-m", "compileall"], "ok")]

    assert review_request_from_dict(_json(asdict(request))) == request
    assert intent_from_dict(_json(asdict(intent))) == intent
    assert risk_packet_from_dict(_json(asdict(packet))) == packet
    assert risk_assessment_from_dict(_json(asdict(risk))) == risk
    assert assignments_from_dict(_json({"assignments": [asdict(_assignment())]})) == [_assignment()]
    assert quality_results_from_dict(_json({"results": [asdict(quality[0])]})) == quality


def test_repository_and_reviewer_artifacts_round_trip() -> None:
    repository = RepositoryIntelligenceSnapshot(
        base_revision="a" * 40,
        revision="b" * 40,
        changed_symbols=[ChangedSymbol("a.py", "f", "function", "modified", 1, 2)],
    )
    result = _reviewer_result()
    assignment = _assignment()
    envelope = ModelInvocationEnvelope("system", [], [{"role": "user", "content": "review"}], {"trace_id": "t"})
    response = ModelResponse("{}", "fake", "fake-reviewer", {"fake": True})

    assert repository_intelligence_from_dict(_json(asdict(repository))) == repository
    assert reviewer_result_from_dict(_json(reviewer_result_to_dict(result))) == result
    execution = reviewer_execution_from_artifacts(
        reviewer_index=0,
        trace_id="review-1-reviewer-0",
        assignment=assignment,
        envelope_payload=_json(asdict(envelope)),
        response_payload=_json(asdict(response)),
        result_payload=_json(reviewer_result_to_dict(result)),
    )
    assert execution.assignment == assignment
    assert execution.result == result
    assert execution.response == response


def test_downstream_artifacts_round_trip() -> None:
    reconciliation = EvidenceReconciliation(
        canonical_findings=[CanonicalFinding("bug", "high", "high", ["O-1"], [0], ["core"], "fix")],
        rejected_findings=[RejectedFinding(1, "adversarial", "claim", "unsupported", [], ["O-x"])],
        remaining_disagreements=["severity"],
        contract_coverage=[ContractCoverage(0, "core", "correctness", "covered", "ok", ["O-1"], [])],
        evidence_quality="verified",
    )
    completion = CompletionResult("completed", "approve", [], [], [])
    final_risk = FinalRiskAssessment(
        "reassessed",
        RiskLevel.MEDIUM,
        RiskLevel.HIGH,
        ["finding"],
        ["finding"],
        [],
        [],
        ["O-1"],
    )
    brief = _review_brief()

    assert reconciliation_from_dict(_json(reconciliation_to_dict(reconciliation))) == reconciliation
    assert completion_from_dict(_json(completion_to_dict(completion))) == completion
    assert final_risk_from_dict(_json(final_risk_to_dict(final_risk))) == final_risk
    assert review_brief_from_dict(_json(review_brief_to_dict(brief))) == brief


def test_review_brief_hydration_accepts_abbreviated_final_risk_payload() -> None:
    brief = build_review_brief(
        review_id="review-single",
        base_revision="a" * 40,
        head_revision="b" * 40,
        intent_packet=IntentPacket(goal="ship"),
        risk_assessment=RiskAssessment(
            RiskLevel.MEDIUM,
            {"impact": "bounded"},
            ["reason"],
            [],
            [],
            [],
        ),
        changed_files=["a.py"],
        quality_results=[],
        reviewer_result=_reviewer_result(),
    )
    payload = _json(review_brief_to_dict(brief))

    assert review_brief_from_dict(payload) == brief


def test_review_brief_hydration_rejects_malformed_nested_payloads() -> None:
    payload = _json(review_brief_to_dict(_review_brief()))
    assert isinstance(payload, dict)
    payload["initial_and_final_risk_assessment"] = {}
    with pytest.raises(ValueError, match="missing required"):
        review_brief_from_dict(payload)

    payload = _json(review_brief_to_dict(_review_brief()))
    assert isinstance(payload, dict)
    del payload["quality_gates"][0]["status"]
    with pytest.raises(ValueError, match="missing required"):
        review_brief_from_dict(payload)

    payload = _json(review_brief_to_dict(_review_brief()))
    assert isinstance(payload, dict)
    payload["verification_evidence"][0]["kind"] = "unsupported"
    with pytest.raises(ValueError, match="unsupported value"):
        review_brief_from_dict(payload)


@pytest.mark.parametrize(
    ("loader", "payload", "message"),
    [
        (intent_from_dict, {"goal": "missing everything"}, "missing required"),
        (
            risk_assessment_from_dict,
            {
                "level": "impossible",
                "dimensions": {},
                "reasons": [],
                "signal_refs": [],
                "uncertainties": [],
                "suggested_focus": [],
            },
            "unsupported value",
        ),
        (
            quality_results_from_dict,
            {"results": [{"name": "x", "status": "passed", "command": "not-list", "summary": "ok", "observation_ref": None}]},
            "must be a list",
        ),
    ],
)
def test_typed_loaders_reject_semantically_unsafe_payloads(loader, payload, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        loader(payload)
