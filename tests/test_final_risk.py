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
