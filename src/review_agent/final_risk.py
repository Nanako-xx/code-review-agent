from __future__ import annotations

from collections.abc import Mapping
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
    reconciliation_payload: Mapping[str, Any] | None,
    completion_summary: Mapping[str, Any] | None,
    semantic_reconciliation: Mapping[str, Any] | None = None,
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
        elif result.status in {"unavailable", "timed_out", "error"}:
            target = RiskLevel.HIGH if result.blocking else RiskLevel.MEDIUM
            level = _raise_to(level, target)
            message = (
                f"quality gate {result.status} after review: {result.name}"
            )
            escalations.append(message)
            reasons.append(message)
            uncertainties.append(result.reason or result.summary)
            signal_refs.append(f"quality_gate:{result.name}")
        elif result.status == "skipped":
            if result.blocking:
                level = _raise_to(level, RiskLevel.HIGH)
                message = f"blocking quality gate skipped: {result.name}"
                escalations.append(message)
                reasons.append(message)
                uncertainties.append(result.reason or result.summary)
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

    if semantic_reconciliation is not None:
        level = _apply_semantic_risk_signals(
            semantic_reconciliation,
            level=level,
            reasons=reasons,
            escalations=escalations,
            uncertainties=uncertainties,
            signal_refs=signal_refs,
        )

    blockers = [str(item) for item in completion.get("blockers", [])]
    if blockers:
        level = _raise_to(level, RiskLevel.HIGH)
        uncertainties.extend(blockers)
        reasons.append("completion blockers remain at final reassessment")

    completion_status = completion.get("status")
    if completion_status == "completed_with_uncertainties":
        level = _raise_to(level, RiskLevel.MEDIUM)
    elif completion_status == "budget_exhausted":
        level = _record_minimum_risk(
            level,
            RiskLevel.MEDIUM,
            "completion stopped after supplemental budget exhaustion",
            reasons,
            escalations,
        )
        uncertainties.append("supplemental investigation budget was exhausted")
        signal_refs.append("completion:budget_exhausted")

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


def _apply_semantic_risk_signals(
    semantic: Mapping[str, Any],
    *,
    level: RiskLevel,
    reasons: list[str],
    escalations: list[str],
    uncertainties: list[str],
    signal_refs: list[str],
) -> RiskLevel:
    status = _normalized_semantic_status(semantic.get("status"))
    model = _mapping(semantic.get("model"))
    model_status = _normalized_text(model.get("status"))
    supplemental = _mapping(semantic.get("supplemental"))
    budget = _mapping(supplemental.get("budget"))
    stop_reason = _first_normalized_text(
        supplemental.get("stop_reason"),
        semantic.get("stop_reason"),
        budget.get("stop_reason"),
    )

    if status:
        signal_refs.append(f"semantic_reconciliation:{status}")
    else:
        level = _record_minimum_risk(
            level,
            RiskLevel.MEDIUM,
            "semantic reconciliation status is missing",
            reasons,
            escalations,
        )
        uncertainties.append("semantic reconciliation status is missing")

    fallback = (
        status == "fallback"
        or model_status == "fallback"
        or stop_reason == "model_fallback"
        or semantic.get("fallback") is True
    )
    if fallback:
        level = _record_minimum_risk(
            level,
            RiskLevel.MEDIUM,
            "semantic reconciliation fell back to deterministic reconciliation",
            reasons,
            escalations,
        )
        uncertainties.append("semantic reconciliation fallback requires manual review")
        signal_refs.append("semantic_reconciliation:fallback")
    elif status == "partial":
        level = _record_minimum_risk(
            level,
            RiskLevel.MEDIUM,
            "semantic reconciliation completed only partially",
            reasons,
            escalations,
        )
        uncertainties.append("semantic reconciliation is partial")
    elif status not in {"accepted", "local_only", ""}:
        level = _record_minimum_risk(
            level,
            RiskLevel.MEDIUM,
            f"semantic reconciliation status is unrecognized: {status}",
            reasons,
            escalations,
        )
        uncertainties.append(f"semantic reconciliation status is unrecognized: {status}")

    resolved_count = _item_count(
        semantic.get("conflicts_resolved", semantic.get("resolved_conflicts"))
    )
    if resolved_count:
        reasons.append(
            f"semantic reconciliation resolved {resolved_count} conflict(s)"
        )
        signal_refs.append(
            f"semantic_reconciliation:resolved_conflicts:{resolved_count}"
        )

    disagreement_count = _item_count(semantic.get("remaining_disagreements"))
    if disagreement_count:
        level = _record_minimum_risk(
            level,
            RiskLevel.MEDIUM,
            "semantic disagreements remain unresolved",
            reasons,
            escalations,
        )
        uncertainties.append("reviewer disagreements remain unresolved")
        signal_refs.append(
            f"semantic_reconciliation:remaining_disagreements:{disagreement_count}"
        )

    supplemental_status = _normalized_text(supplemental.get("status"))
    if supplemental_status == "partial" or _item_count(supplemental.get("partial")):
        level = _record_minimum_risk(
            level,
            RiskLevel.MEDIUM,
            "supplemental investigation returned partial results",
            reasons,
            escalations,
        )
        uncertainties.append("supplemental investigation returned partial results")
    if (
        supplemental_status == "failed"
        or _item_count(supplemental.get("failed"))
        or stop_reason == "task_failure"
    ):
        level = _record_minimum_risk(
            level,
            RiskLevel.MEDIUM,
            "supplemental investigation failed",
            reasons,
            escalations,
        )
        uncertainties.append("supplemental investigation failed")
    if (
        supplemental_status == "unavailable"
        or _item_count(supplemental.get("unavailable"))
        or stop_reason in {"unavailable", "provider_unavailable"}
    ):
        level = _record_minimum_risk(
            level,
            RiskLevel.MEDIUM,
            "supplemental investigation was unavailable",
            reasons,
            escalations,
        )
        uncertainties.append("supplemental investigation was unavailable")

    if stop_reason:
        signal_refs.append(f"supplemental_stop:{stop_reason}")
    if stop_reason == "budget_exhausted":
        level = _record_minimum_risk(
            level,
            RiskLevel.MEDIUM,
            "supplemental investigation stopped after budget exhaustion",
            reasons,
            escalations,
        )
        uncertainties.append("supplemental investigation budget was exhausted")
    elif stop_reason == "max_waves":
        level = _record_minimum_risk(
            level,
            RiskLevel.MEDIUM,
            "supplemental investigation reached its maximum wave count",
            reasons,
            escalations,
        )
        uncertainties.append("supplemental investigation reached its maximum wave count")

    for item in _string_items(semantic.get("uncertainties")):
        uncertainties.append(f"semantic reconciliation uncertainty: {item}")
    return level


def _record_minimum_risk(
    current: RiskLevel,
    target: RiskLevel,
    message: str,
    reasons: list[str],
    escalations: list[str],
) -> RiskLevel:
    updated = _raise_to(current, target)
    reasons.append(message)
    if updated is not current:
        escalations.append(message)
    return updated


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _normalized_semantic_status(value: object) -> str:
    status = _normalized_text(value)
    if status in {"local", "disabled"}:
        return "local_only"
    return status


def _normalized_text(value: object) -> str:
    return value.strip().casefold() if isinstance(value, str) else ""


def _first_normalized_text(*values: object) -> str:
    for value in values:
        normalized = _normalized_text(value)
        if normalized:
            return normalized
    return ""


def _item_count(value: object) -> int:
    if value is None or value is False:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return max(0, int(value))
    if isinstance(value, str):
        return int(bool(value.strip()))
    try:
        return len(value)  # type: ignore[arg-type]
    except TypeError:
        return 1


def _string_items(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if str(item).strip()]


def _field(item: Mapping[str, Any] | ReviewerFinding, name: str) -> str:
    if isinstance(item, Mapping):
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
