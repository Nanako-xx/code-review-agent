from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from review_agent.models import (
    IntentPacket,
    IntentStatus,
    QualityGateResult,
    ReviewerFinding,
    ReviewerResult,
    RiskAssessment,
    RiskLevel,
)


RISK_ORDER = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}


@dataclass(frozen=True)
class FinalRiskAssessment:
    status: str
    initial_level: RiskLevel
    level: RiskLevel
    reasons: list[str]
    escalations: list[str]
    deescalations: list[str]
    uncertainties: list[str]
    signal_refs: list[str]


def reassess_final_risk(
    *,
    initial_risk: RiskAssessment,
    intent_packet: IntentPacket,
    quality_results: list[QualityGateResult],
    reviewer_result: ReviewerResult | None,
    reconciliation_payload: dict[str, Any] | None,
    completion_summary: dict[str, Any] | None,
) -> FinalRiskAssessment:
    level = initial_risk.level
    reasons = list(initial_risk.reasons)
    escalations: list[str] = []
    deescalations: list[str] = []
    uncertainties = list(initial_risk.uncertainties)
    signal_refs = list(initial_risk.signal_refs)
    reconciliation = reconciliation_payload or {}
    completion = completion_summary or {}

    if intent_packet.status == IntentStatus.INSUFFICIENT:
        level = _raise_to(level, RiskLevel.HIGH)
        message = "intent insufficient at final reassessment"
        escalations.append(message)
        reasons.append(message)
    elif intent_packet.status == IntentStatus.PARTIAL:
        level = _raise_to(level, RiskLevel.MEDIUM)
        uncertainties.append("Intent Packet partial at final reassessment")

    for result in quality_results:
        if result.status == "failed":
            level = _raise_to(level, RiskLevel.HIGH)
            message = f"quality gate failed after review: {result.name}"
            escalations.append(message)
            reasons.append(message)
            signal_refs.append(f"quality_gate:{result.name}")

    canonical_findings = list(reconciliation.get("canonical_findings", []))
    for item in canonical_findings:
        claim = _field(item, "claim")
        severity = _field(item, "severity").casefold()
        target = _risk_for_finding_severity(severity)
        if target is None:
            continue
        level = _raise_to(level, target)
        label = "critical" if target == RiskLevel.CRITICAL else target.value
        message = f"verified {label} finding: {claim}"
        escalations.append(message)
        reasons.append(message)

    if not canonical_findings and reviewer_result is not None:
        for finding in reviewer_result.confirmed_findings:
            target = _risk_for_finding_severity(finding.severity.casefold())
            if target is None:
                continue
            level = _raise_to(level, target)
            message = f"single reviewer {target.value} finding: {finding.claim}"
            escalations.append(message)
            reasons.append(message)

    if reconciliation.get("rejected_findings") and not canonical_findings:
        reasons.append("rejected unsupported findings were not used for escalation")

    if reconciliation.get("remaining_disagreements"):
        level = _raise_to(level, RiskLevel.MEDIUM)
        uncertainties.append("reviewer disagreements remain unresolved")

    blockers = [str(item) for item in completion.get("blockers", [])]
    if blockers:
        level = _raise_to(level, RiskLevel.HIGH)
        uncertainties.extend(blockers)
        reasons.append("completion blockers remain at final reassessment")

    if completion.get("status") == "completed_with_uncertainties":
        level = _raise_to(level, RiskLevel.MEDIUM)

    return FinalRiskAssessment(
        status="reassessed",
        initial_level=initial_risk.level,
        level=level,
        reasons=_dedupe(reasons),
        escalations=_dedupe(escalations),
        deescalations=deescalations,
        uncertainties=_dedupe(uncertainties),
        signal_refs=_dedupe(signal_refs),
    )


def final_risk_to_dict(result: FinalRiskAssessment) -> dict[str, Any]:
    return {
        "status": result.status,
        "initial_level": result.initial_level.value,
        "level": result.level.value,
        "reasons": result.reasons,
        "escalations": result.escalations,
        "deescalations": result.deescalations,
        "uncertainties": result.uncertainties,
        "signal_refs": result.signal_refs,
    }


def _risk_for_finding_severity(severity: str) -> RiskLevel | None:
    if severity in {"critical", "blocker"}:
        return RiskLevel.CRITICAL
    if severity == "high":
        return RiskLevel.HIGH
    if severity == "medium":
        return RiskLevel.MEDIUM
    return None


def _raise_to(current: RiskLevel, target: RiskLevel) -> RiskLevel:
    if RISK_ORDER[target] > RISK_ORDER[current]:
        return target
    return current


def _field(item: dict[str, Any] | ReviewerFinding, name: str) -> str:
    if isinstance(item, dict):
        return str(item.get(name, ""))
    return str(getattr(item, name, ""))


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
