from review_agent.final_risk import final_risk_to_dict, reassess_final_risk
from review_agent.models import (
    IntentPacket,
    IntentStatus,
    QualityGateResult,
    ReviewerFinding,
    ReviewerResult,
    RiskAssessment,
    RiskLevel,
)


def initial(level=RiskLevel.LOW) -> RiskAssessment:
    return RiskAssessment(
        level=level,
        dimensions={"impact": "local"},
        reasons=["initial local change"],
        signal_refs=[],
        uncertainties=[],
        suggested_focus=["intent alignment"],
    )


def intent() -> IntentPacket:
    return IntentPacket(goal="Review change", status=IntentStatus.SUFFICIENT)


def test_final_risk_escalates_for_verified_high_finding() -> None:
    result = reassess_final_risk(
        initial_risk=initial(RiskLevel.LOW),
        intent_packet=intent(),
        quality_results=[],
        reviewer_result=None,
        reconciliation_payload={
            "canonical_findings": [
                {
                    "claim": "Authorization bypass remains possible",
                    "severity": "high",
                    "confidence": "medium",
                    "evidence_refs": ["O-1"],
                }
            ],
            "rejected_findings": [],
            "remaining_disagreements": [],
        },
        completion_summary={"status": "completed", "recommendation": "approve"},
    )

    payload = final_risk_to_dict(result)

    assert payload["status"] == "reassessed"
    assert payload["initial_level"] == "low"
    assert payload["level"] == "high"
    assert "verified high finding: Authorization bypass remains possible" in payload["reasons"]


def test_final_risk_escalates_for_failed_quality_gate() -> None:
    result = reassess_final_risk(
        initial_risk=initial(RiskLevel.LOW),
        intent_packet=intent(),
        quality_results=[
            QualityGateResult(
                name="python_compile",
                status="failed",
                command=["python", "-m", "compileall"],
                summary="SyntaxError",
            )
        ],
        reviewer_result=None,
        reconciliation_payload={},
        completion_summary={},
    )

    assert result.level is RiskLevel.HIGH
    assert "quality gate failed after review: python_compile" in result.reasons


def test_final_risk_does_not_escalate_for_rejected_findings_only() -> None:
    result = reassess_final_risk(
        initial_risk=initial(RiskLevel.LOW),
        intent_packet=intent(),
        quality_results=[],
        reviewer_result=None,
        reconciliation_payload={
            "canonical_findings": [],
            "rejected_findings": [{"claim": "Unsupported critical issue", "reason": "unsupported_claim"}],
            "remaining_disagreements": [],
        },
        completion_summary={"status": "completed", "recommendation": "approve"},
    )

    assert result.level is RiskLevel.LOW
    assert "rejected unsupported findings were not used for escalation" in result.reasons


def test_final_risk_uses_single_reviewer_findings_when_no_reconciliation_exists() -> None:
    result = reassess_final_risk(
        initial_risk=initial(RiskLevel.LOW),
        intent_packet=intent(),
        quality_results=[],
        reviewer_result=ReviewerResult(
            confirmed_findings=[
                ReviewerFinding(
                    claim="Missing rollback path",
                    severity="medium",
                    confidence="medium",
                    evidence_refs=[],
                )
            ]
        ),
        reconciliation_payload=None,
        completion_summary=None,
    )

    assert result.level is RiskLevel.MEDIUM
    assert "single reviewer medium finding: Missing rollback path" in result.reasons


def test_final_risk_propagates_semantic_fallback_conflicts_and_stop_reason() -> None:
    result = reassess_final_risk(
        initial_risk=initial(RiskLevel.LOW),
        intent_packet=intent(),
        quality_results=[],
        reviewer_result=None,
        reconciliation_payload={
            "canonical_findings": [],
            "rejected_findings": [],
            "remaining_disagreements": [],
        },
        completion_summary={
            "status": "budget_exhausted",
            "recommendation": "manual_review",
        },
        semantic_reconciliation={
            "status": "partial",
            "model": {"status": "fallback"},
            "conflicts_resolved": [{"issue": "duplicate wording"}],
            "remaining_disagreements": [{"issue": "runtime behavior"}],
            "supplemental": {
                "status": "budget_exhausted",
                "stop_reason": "max_waves",
                "budget": {
                    "unknown_consumed": {"tasks": 1, "tokens": 100},
                },
            },
            "uncertainties": ["Provider response was malformed."],
        },
    )

    assert result.level is RiskLevel.MEDIUM
    assert "semantic reconciliation fell back to deterministic reconciliation" in result.reasons
    assert "semantic reconciliation resolved 1 conflict(s)" in result.reasons
    assert "supplemental investigation reached its maximum wave count" in result.reasons
    assert "semantic_reconciliation:fallback" in result.signal_refs
    assert "semantic_reconciliation:remaining_disagreements:1" in result.signal_refs
    assert "supplemental_stop:max_waves" in result.signal_refs
    assert "semantic reconciliation uncertainty: Provider response was malformed." in result.uncertainties
