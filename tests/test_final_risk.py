import pytest

from review_agent.final_risk import (
    FinalRiskEvidence,
    final_risk_memory_projection_from_risk,
    final_risk_to_dict,
    reassess_final_risk,
    reassess_final_risk_typed,
)
from review_agent.models import (
    CompiledRiskFloor,
    FinalRiskMemoryProjection,
    IntentPacket,
    IntentStatus,
    MemoryDiagnostic,
    MemoryDiagnosticCode,
    MemoryReference,
    MemoryRiskSignal,
    QualityGateResult,
    ReviewerFinding,
    ReviewerResult,
    RiskAssessment,
    RiskMemoryProjection,
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


def test_final_risk_explains_only_typed_applied_memory_and_residual_risk() -> None:
    memory_id = "MEM-" + "e" * 64
    reference = MemoryReference(
        memory_id=memory_id,
        kind="incident_lesson",
        source_refs=("memory-source:" + "f" * 64,),
    )
    projection = FinalRiskMemoryProjection(
        applied_memory=(reference,),
        risk_signals=(
            MemoryRiskSignal(
                signal_ref=f"memory:{memory_id}",
                summary="Retries have caused duplicate delivery incidents.",
                memory=reference,
            ),
        ),
        risk_floor=CompiledRiskFloor(
            minimum_level=RiskLevel.HIGH,
            memory_ids=(memory_id,),
        ),
        diagnostics=(
            MemoryDiagnostic(
                code=MemoryDiagnosticCode.STALE,
                message="one related record requires revalidation",
                memory_ids=(memory_id,),
            ),
        ),
        residual_risk=("Retry behavior still needs manual verification.",),
    )

    result = reassess_final_risk(
        initial_risk=initial(RiskLevel.LOW),
        intent_packet=intent(),
        quality_results=[],
        reviewer_result=None,
        reconciliation_payload={},
        completion_summary={"status": "completed", "recommendation": "approve"},
        memory_projection=projection,
    )
    payload = final_risk_to_dict(result)

    assert result.level is RiskLevel.HIGH
    assert payload["applied_memory"] == [reference.to_dict()]
    assert payload["residual_risk"] == [
        "Retry behavior still needs manual verification."
    ]
    assert payload["memory_diagnostics"][0]["code"] == "stale"
    encoded = str(payload)
    assert "pending_approval" not in encoded
    assert "raw_feedback" not in encoded

    with pytest.raises(ValueError, match="FinalRiskMemoryProjection"):
        reassess_final_risk(
            initial_risk=initial(),
            intent_packet=intent(),
            quality_results=[],
            reviewer_result=None,
            reconciliation_payload={},
            completion_summary={},
            memory_projection={"pending_candidates": ["untrusted"]},
        )


def test_legacy_final_risk_payload_has_no_memory_fields() -> None:
    result = reassess_final_risk(
        initial_risk=initial(),
        intent_packet=intent(),
        quality_results=[],
        reviewer_result=None,
        reconciliation_payload={},
        completion_summary={},
    )

    payload = final_risk_to_dict(result)
    assert "applied_memory" not in payload
    assert "memory_diagnostics" not in payload
    assert "residual_risk" not in payload


def test_authoritative_final_risk_entry_is_typed_only_and_legacy_adapter_is_strict() -> None:
    typed = reassess_final_risk_typed(
        initial_risk=initial(),
        intent_packet=intent(),
        quality_results=[],
        reviewer_result=None,
        evidence=FinalRiskEvidence(),
    )

    assert typed.level is RiskLevel.LOW
    with pytest.raises(ValueError, match="FinalRiskEvidence"):
        reassess_final_risk_typed(
            initial_risk=initial(),
            intent_packet=intent(),
            quality_results=[],
            reviewer_result=None,
            evidence={},
        )
    with pytest.raises(ValueError, match="unsupported field"):
        reassess_final_risk(
            initial_risk=initial(),
            intent_packet=intent(),
            quality_results=[],
            reviewer_result=None,
            reconciliation_payload={"pending_candidates": []},
            completion_summary={},
        )


def test_final_risk_projection_independently_blocks_floor_overflow() -> None:
    memory_id = "MEM-" + "9" * 64
    reference = MemoryReference(
        memory_id=memory_id,
        kind="review_rule",
        source_refs=("memory-source:" + "a" * 64,),
    )
    risk_projection = RiskMemoryProjection(
        risk_floor=CompiledRiskFloor(RiskLevel.HIGH, (memory_id,)),
        policy_sources=(reference,),
    )

    final_projection = final_risk_memory_projection_from_risk(
        risk_projection,
        max_hard_policy_items=0,
    )

    assert final_projection.risk_floor is risk_projection.risk_floor
    assert final_projection.applied_memory == (reference,)
    assert any(
        item.code is MemoryDiagnosticCode.HARD_POLICY_OVERFLOW
        and item.blocking
        for item in final_projection.diagnostics
    )
