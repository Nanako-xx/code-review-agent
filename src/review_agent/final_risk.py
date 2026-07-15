from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from review_agent.models import (
    FinalRiskMemoryProjection,
    IntentPacket,
    IntentStatus,
    MemoryDiagnostic,
    MemoryReference,
    QualityGateResult,
    ReviewerFinding,
    ReviewerResult,
    RiskAssessment,
    RiskMemoryProjection,
    RiskLevel,
    hard_policy_overflow_diagnostic,
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
    applied_memory: tuple[MemoryReference, ...] = ()
    memory_diagnostics: tuple[MemoryDiagnostic, ...] = ()
    residual_risk: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status != "reassessed":
            raise ValueError("final risk status must be reassessed")
        if not isinstance(self.initial_level, RiskLevel) or not isinstance(
            self.level, RiskLevel
        ):
            raise ValueError("final risk levels must be RiskLevel values")
        for name in (
            "reasons",
            "escalations",
            "deescalations",
            "uncertainties",
            "signal_refs",
        ):
            values = getattr(self, name)
            if not isinstance(values, list) or any(
                not isinstance(item, str) or not item.strip() for item in values
            ):
                raise ValueError(f"final risk {name} must contain non-empty strings")
            object.__setattr__(self, name, list(values))
        applied = tuple(self.applied_memory)
        diagnostics = tuple(self.memory_diagnostics)
        residual = tuple(self.residual_risk)
        if any(not isinstance(item, MemoryReference) for item in applied):
            raise ValueError("final risk applied_memory must contain MemoryReference values")
        if any(not isinstance(item, MemoryDiagnostic) for item in diagnostics):
            raise ValueError(
                "final risk memory_diagnostics must contain MemoryDiagnostic values"
            )
        if any(not isinstance(item, str) or not item.strip() for item in residual):
            raise ValueError("final risk residual_risk must contain non-empty strings")
        if len({item.memory_id for item in applied}) != len(applied):
            raise ValueError("final risk applied_memory must not repeat memory_id")
        object.__setattr__(self, "applied_memory", applied)
        object.__setattr__(self, "memory_diagnostics", diagnostics)
        object.__setattr__(self, "residual_risk", residual)


@dataclass(frozen=True)
class FinalRiskFindingEvidence:
    claim: str
    severity: str

    def __post_init__(self) -> None:
        if not isinstance(self.claim, str) or not self.claim.strip():
            raise ValueError("final risk finding claim must be a non-empty string")
        if not isinstance(self.severity, str) or not self.severity.strip():
            raise ValueError("final risk finding severity must be a non-empty string")


@dataclass(frozen=True)
class FinalRiskSemanticEvidence:
    status: str = ""
    model_status: str = ""
    resolved_conflict_count: int = 0
    remaining_disagreement_count: int = 0
    supplemental_status: str = ""
    supplemental_partial_count: int = 0
    supplemental_failed_count: int = 0
    supplemental_unavailable_count: int = 0
    stop_reason: str = ""
    uncertainties: tuple[str, ...] = ()
    fallback: bool = False

    def __post_init__(self) -> None:
        for name in ("status", "model_status", "supplemental_status", "stop_reason"):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ValueError(f"semantic evidence {name} must be a string")
        for name in (
            "resolved_conflict_count",
            "remaining_disagreement_count",
            "supplemental_partial_count",
            "supplemental_failed_count",
            "supplemental_unavailable_count",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"semantic evidence {name} must be non-negative")
        if type(self.fallback) is not bool:
            raise ValueError("semantic evidence fallback must be a boolean")
        uncertainties = tuple(self.uncertainties)
        if any(not isinstance(item, str) or not item.strip() for item in uncertainties):
            raise ValueError("semantic evidence uncertainties must contain non-empty strings")
        object.__setattr__(self, "uncertainties", uncertainties)


@dataclass(frozen=True)
class FinalRiskEvidence:
    canonical_findings: tuple[FinalRiskFindingEvidence, ...] = ()
    rejected_finding_count: int = 0
    remaining_disagreement_count: int = 0
    completion_status: str = ""
    completion_blockers: tuple[str, ...] = ()
    semantic: FinalRiskSemanticEvidence | None = None

    def __post_init__(self) -> None:
        findings = tuple(self.canonical_findings)
        if any(not isinstance(item, FinalRiskFindingEvidence) for item in findings):
            raise ValueError("canonical_findings must contain typed final-risk evidence")
        for name in ("rejected_finding_count", "remaining_disagreement_count"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"final risk evidence {name} must be non-negative")
        if not isinstance(self.completion_status, str):
            raise ValueError("completion_status must be a string")
        blockers = tuple(self.completion_blockers)
        if any(not isinstance(item, str) or not item.strip() for item in blockers):
            raise ValueError("completion_blockers must contain non-empty strings")
        if self.semantic is not None and not isinstance(
            self.semantic, FinalRiskSemanticEvidence
        ):
            raise ValueError("semantic must be FinalRiskSemanticEvidence or None")
        object.__setattr__(self, "canonical_findings", findings)
        object.__setattr__(self, "completion_blockers", blockers)


def reassess_final_risk(
    *,
    initial_risk: RiskAssessment,
    intent_packet: IntentPacket,
    quality_results: list[QualityGateResult],
    reviewer_result: ReviewerResult | None,
    reconciliation_payload: Mapping[str, Any] | None,
    completion_summary: Mapping[str, Any] | None,
    semantic_reconciliation: Mapping[str, Any] | None = None,
    memory_projection: FinalRiskMemoryProjection | None = None,
) -> FinalRiskAssessment:
    """Strict peripheral adapter for legacy Mapping-shaped pipeline inputs."""

    evidence = final_risk_evidence_from_legacy_mappings(
        reconciliation_payload=reconciliation_payload,
        completion_summary=completion_summary,
        semantic_reconciliation=semantic_reconciliation,
    )
    return reassess_final_risk_typed(
        initial_risk=initial_risk,
        intent_packet=intent_packet,
        quality_results=quality_results,
        reviewer_result=reviewer_result,
        evidence=evidence,
        memory_projection=memory_projection,
    )


def reassess_final_risk_typed(
    *,
    initial_risk: RiskAssessment,
    intent_packet: IntentPacket,
    quality_results: list[QualityGateResult],
    reviewer_result: ReviewerResult | None,
    evidence: FinalRiskEvidence,
    memory_projection: FinalRiskMemoryProjection | None = None,
) -> FinalRiskAssessment:
    """Authoritative Final Risk entry; every downstream artifact is typed."""

    if not isinstance(initial_risk, RiskAssessment):
        raise ValueError("initial_risk must be a RiskAssessment")
    if not isinstance(intent_packet, IntentPacket):
        raise ValueError("intent_packet must be an IntentPacket")
    if not isinstance(quality_results, list) or any(
        not isinstance(item, QualityGateResult) for item in quality_results
    ):
        raise ValueError("quality_results must contain QualityGateResult values")
    if reviewer_result is not None and not isinstance(reviewer_result, ReviewerResult):
        raise ValueError("reviewer_result must be a ReviewerResult or None")
    if not isinstance(evidence, FinalRiskEvidence):
        raise ValueError("evidence must be a FinalRiskEvidence")
    if memory_projection is not None and not isinstance(
        memory_projection,
        FinalRiskMemoryProjection,
    ):
        raise ValueError(
            "memory_projection must be a FinalRiskMemoryProjection or None"
        )
    level = initial_risk.level
    reasons = list(initial_risk.reasons)
    escalations: list[str] = []
    deescalations: list[str] = []
    uncertainties = list(initial_risk.uncertainties)
    signal_refs = list(initial_risk.signal_refs)

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

    canonical_findings = list(evidence.canonical_findings)
    for item in canonical_findings:
        claim = item.claim
        severity = item.severity.casefold()
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

    if evidence.rejected_finding_count and not canonical_findings:
        reasons.append("rejected unsupported findings were not used for escalation")

    if evidence.remaining_disagreement_count:
        level = _raise_to(level, RiskLevel.MEDIUM)
        uncertainties.append("reviewer disagreements remain unresolved")

    if evidence.semantic is not None:
        level = _apply_semantic_risk_signals(
            evidence.semantic,
            level=level,
            reasons=reasons,
            escalations=escalations,
            uncertainties=uncertainties,
            signal_refs=signal_refs,
        )

    blockers = list(evidence.completion_blockers)
    if blockers:
        level = _raise_to(level, RiskLevel.HIGH)
        uncertainties.extend(blockers)
        reasons.append("completion blockers remain at final reassessment")

    completion_status = evidence.completion_status
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

    if memory_projection is not None:
        for signal in memory_projection.risk_signals:
            reasons.append(
                f"approved memory applied ({signal.memory.memory_id}): {signal.summary}"
            )
            signal_refs.append(signal.signal_ref)
        if memory_projection.risk_floor is not None:
            target = memory_projection.risk_floor.minimum_level
            level = _record_minimum_risk(
                level,
                target,
                f"compiled Memory risk floor retained at {target.value}",
                reasons,
                escalations,
            )
            signal_refs.extend(
                f"memory_floor:{memory_id}"
                for memory_id in memory_projection.risk_floor.memory_ids
            )
        for diagnostic in memory_projection.diagnostics:
            message = f"memory {diagnostic.code.value}: {diagnostic.message}"
            uncertainties.append(message)
            signal_refs.append(f"memory_diagnostic:{diagnostic.code.value}")
            if diagnostic.blocking:
                level = _record_minimum_risk(
                    level,
                    RiskLevel.HIGH,
                    "blocking Memory diagnostic requires manual review",
                    reasons,
                    escalations,
                )
        uncertainties.extend(memory_projection.residual_risk)

    return FinalRiskAssessment(
        status="reassessed",
        initial_level=initial_risk.level,
        level=level,
        reasons=_dedupe(reasons),
        escalations=_dedupe(escalations),
        deescalations=deescalations,
        uncertainties=_dedupe(uncertainties),
        signal_refs=_dedupe(signal_refs),
        applied_memory=(
            () if memory_projection is None else memory_projection.applied_memory
        ),
        memory_diagnostics=(
            () if memory_projection is None else memory_projection.diagnostics
        ),
        residual_risk=(
            () if memory_projection is None else memory_projection.residual_risk
        ),
    )


def final_risk_evidence_from_legacy_mappings(
    *,
    reconciliation_payload: Mapping[str, Any] | None,
    completion_summary: Mapping[str, Any] | None,
    semantic_reconciliation: Mapping[str, Any] | None = None,
) -> FinalRiskEvidence:
    """Validate legacy JSON projections before they reach Final Risk authority."""

    reconciliation = _legacy_mapping(
        reconciliation_payload,
        "reconciliation_payload",
    )
    _legacy_fields(
        reconciliation,
        {
            "canonical_findings",
            "rejected_findings",
            "remaining_disagreements",
            "contract_coverage",
            "evidence_quality",
        },
        "reconciliation_payload",
    )
    findings: list[FinalRiskFindingEvidence] = []
    for index, raw in enumerate(
        _legacy_list(reconciliation.get("canonical_findings", []), "canonical_findings")
    ):
        context = f"reconciliation_payload.canonical_findings[{index}]"
        row = _legacy_mapping(raw, context, allow_none=False)
        _legacy_fields(
            row,
            {
                "finding_id",
                "claim",
                "severity",
                "confidence",
                "evidence_refs",
                "reviewer_indices",
                "roles",
                "suggested_action",
                "path",
                "line",
                "impact",
                "verification_performed",
            },
            context,
        )
        findings.append(
            FinalRiskFindingEvidence(
                claim=_legacy_non_empty_string(row.get("claim"), f"{context}.claim"),
                severity=_legacy_non_empty_string(
                    row.get("severity"), f"{context}.severity"
                ),
            )
        )
    rejected = _legacy_list(
        reconciliation.get("rejected_findings", []),
        "reconciliation_payload.rejected_findings",
    )
    disagreements = _legacy_list(
        reconciliation.get("remaining_disagreements", []),
        "reconciliation_payload.remaining_disagreements",
    )
    _legacy_string_list(
        reconciliation.get("remaining_disagreements", []),
        "reconciliation_payload.remaining_disagreements",
    )
    if "contract_coverage" in reconciliation:
        for index, raw in enumerate(
            _legacy_list(
                reconciliation["contract_coverage"],
                "reconciliation_payload.contract_coverage",
            )
        ):
            _legacy_mapping(
                raw,
                f"reconciliation_payload.contract_coverage[{index}]",
                allow_none=False,
            )
    if "evidence_quality" in reconciliation:
        _legacy_non_empty_string(
            reconciliation["evidence_quality"],
            "reconciliation_payload.evidence_quality",
        )

    completion = _legacy_mapping(completion_summary, "completion_summary")
    _legacy_fields(
        completion,
        {
            "status",
            "recommendation",
            "blockers",
            "uncertainties",
            "missing_perspectives",
            "memory_diagnostics",
        },
        "completion_summary",
    )
    completion_status = _legacy_optional_string(
        completion.get("status"),
        "completion_summary.status",
    )
    for field_name in ("recommendation",):
        if field_name in completion:
            _legacy_non_empty_string(
                completion[field_name],
                f"completion_summary.{field_name}",
            )
    for field_name in ("uncertainties", "missing_perspectives"):
        if field_name in completion:
            _legacy_string_list(
                completion[field_name],
                f"completion_summary.{field_name}",
            )
    blockers = tuple(
        _legacy_string_list(
            completion.get("blockers", []),
            "completion_summary.blockers",
        )
    )
    if "memory_diagnostics" in completion:
        for index, raw in enumerate(
            _legacy_list(
                completion["memory_diagnostics"],
                "completion_summary.memory_diagnostics",
            )
        ):
            row = _legacy_mapping(
                raw,
                f"completion_summary.memory_diagnostics[{index}]",
                allow_none=False,
            )
            _legacy_fields(
                row,
                {"code", "message", "memory_ids", "blocking"},
                f"completion_summary.memory_diagnostics[{index}]",
            )

    semantic = (
        _semantic_evidence_from_legacy(semantic_reconciliation)
        if semantic_reconciliation is not None
        else None
    )
    return FinalRiskEvidence(
        canonical_findings=tuple(findings),
        rejected_finding_count=len(rejected),
        remaining_disagreement_count=len(disagreements),
        completion_status=completion_status,
        completion_blockers=blockers,
        semantic=semantic,
    )


def _semantic_evidence_from_legacy(
    payload: Mapping[str, Any],
) -> FinalRiskSemanticEvidence:
    row = _legacy_mapping(payload, "semantic_reconciliation", allow_none=False)
    _legacy_fields(
        row,
        {
            "schema_version",
            "status",
            "canonical_findings",
            "rejected_findings",
            "conflicts_resolved",
            "resolved_conflicts",
            "remaining_disagreements",
            "contract_coverage",
            "evidence_quality",
            "supplemental",
            "policy_actions",
            "uncertainties",
            "model",
            "fallback",
            "stop_reason",
        },
        "semantic_reconciliation",
    )
    model = _legacy_mapping(row.get("model"), "semantic_reconciliation.model")
    _legacy_fields(
        model,
        {"status", "invocation_ids", "input_digests"},
        "semantic_reconciliation.model",
    )
    for field_name in (
        "canonical_findings",
        "rejected_findings",
        "conflicts_resolved",
        "resolved_conflicts",
        "remaining_disagreements",
        "contract_coverage",
        "policy_actions",
    ):
        if field_name in row and not isinstance(row[field_name], list):
            raise ValueError(
                f"semantic_reconciliation.{field_name} must be a list"
            )
    if "uncertainties" in row:
        _legacy_string_list(
            row["uncertainties"],
            "semantic_reconciliation.uncertainties",
        )
    for field_name in ("invocation_ids", "input_digests"):
        if field_name in model:
            _legacy_string_list(
                model[field_name],
                f"semantic_reconciliation.model.{field_name}",
            )
    supplemental = _legacy_mapping(
        row.get("supplemental"),
        "semantic_reconciliation.supplemental",
    )
    _legacy_fields(
        supplemental,
        {
            "status",
            "waves",
            "tasks",
            "completed",
            "partial",
            "failed",
            "unavailable",
            "budget",
            "stop_reason",
        },
        "semantic_reconciliation.supplemental",
    )
    for field_name in (
        "waves",
        "tasks",
        "completed",
        "partial",
        "failed",
        "unavailable",
    ):
        if field_name in supplemental and type(supplemental[field_name]) is not int:
            raise ValueError(
                f"semantic_reconciliation.supplemental.{field_name} must be an integer"
            )
    budget = _legacy_mapping(
        supplemental.get("budget"),
        "semantic_reconciliation.supplemental.budget",
    )
    stop_reason = _first_normalized_text(
        supplemental.get("stop_reason"),
        row.get("stop_reason"),
        budget.get("stop_reason"),
    )
    return FinalRiskSemanticEvidence(
        status=_legacy_optional_string(
            row.get("status"), "semantic_reconciliation.status"
        ),
        model_status=_legacy_optional_string(
            model.get("status"), "semantic_reconciliation.model.status"
        ),
        resolved_conflict_count=_legacy_count(
            row.get("conflicts_resolved", row.get("resolved_conflicts")),
            "semantic_reconciliation.conflicts_resolved",
        ),
        remaining_disagreement_count=_legacy_count(
            row.get("remaining_disagreements"),
            "semantic_reconciliation.remaining_disagreements",
        ),
        supplemental_status=_legacy_optional_string(
            supplemental.get("status"),
            "semantic_reconciliation.supplemental.status",
        ),
        supplemental_partial_count=_legacy_count(
            supplemental.get("partial"),
            "semantic_reconciliation.supplemental.partial",
        ),
        supplemental_failed_count=_legacy_count(
            supplemental.get("failed"),
            "semantic_reconciliation.supplemental.failed",
        ),
        supplemental_unavailable_count=_legacy_count(
            supplemental.get("unavailable"),
            "semantic_reconciliation.supplemental.unavailable",
        ),
        stop_reason=stop_reason,
        uncertainties=tuple(
            _legacy_string_list(
                row.get("uncertainties", []),
                "semantic_reconciliation.uncertainties",
            )
        ),
        fallback=(
            row.get("fallback", False)
            if type(row.get("fallback", False)) is bool
            else _raise_legacy_value_error(
                "semantic_reconciliation.fallback must be a boolean"
            )
        ),
    )


def final_risk_to_dict(result: FinalRiskAssessment) -> dict[str, Any]:
    payload = {
        "status": result.status,
        "initial_level": result.initial_level.value,
        "level": result.level.value,
        "reasons": result.reasons,
        "escalations": result.escalations,
        "deescalations": result.deescalations,
        "uncertainties": result.uncertainties,
        "signal_refs": result.signal_refs,
    }
    if result.applied_memory or result.memory_diagnostics or result.residual_risk:
        payload["applied_memory"] = [item.to_dict() for item in result.applied_memory]
        payload["memory_diagnostics"] = [
            item.to_dict() for item in result.memory_diagnostics
        ]
        payload["residual_risk"] = list(result.residual_risk)
    return payload


def final_risk_memory_projection_from_risk(
    projection: RiskMemoryProjection,
    *,
    residual_risk: tuple[str, ...] = (),
    max_hard_policy_items: int = 64,
    max_hard_policy_bytes: int = 32_768,
) -> FinalRiskMemoryProjection:
    if not isinstance(projection, RiskMemoryProjection):
        raise ValueError("projection must be a RiskMemoryProjection")
    references = {
        signal.memory.memory_id: signal.memory for signal in projection.signals
    }
    references.update(
        {reference.memory_id: reference for reference in projection.policy_sources}
    )
    diagnostics = list(projection.diagnostics)
    if projection.risk_floor is not None:
        overflow = hard_policy_overflow_diagnostic(
            "final_risk",
            (projection.risk_floor.to_dict(),),
            projection.risk_floor.memory_ids,
            max_items=max_hard_policy_items,
            max_bytes=max_hard_policy_bytes,
        )
        if overflow is not None:
            diagnostics.append(overflow)
    return FinalRiskMemoryProjection(
        applied_memory=tuple(references.values()),
        risk_signals=projection.signals,
        risk_floor=projection.risk_floor,
        diagnostics=tuple(diagnostics),
        residual_risk=residual_risk,
    )


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
    semantic: FinalRiskSemanticEvidence,
    *,
    level: RiskLevel,
    reasons: list[str],
    escalations: list[str],
    uncertainties: list[str],
    signal_refs: list[str],
) -> RiskLevel:
    if not isinstance(semantic, FinalRiskSemanticEvidence):
        raise ValueError("semantic must be FinalRiskSemanticEvidence")
    status = _normalized_semantic_status(semantic.status)
    model_status = _normalized_text(semantic.model_status)
    stop_reason = _normalized_text(semantic.stop_reason)

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
        or semantic.fallback
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

    resolved_count = semantic.resolved_conflict_count
    if resolved_count:
        reasons.append(
            f"semantic reconciliation resolved {resolved_count} conflict(s)"
        )
        signal_refs.append(
            f"semantic_reconciliation:resolved_conflicts:{resolved_count}"
        )

    disagreement_count = semantic.remaining_disagreement_count
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

    supplemental_status = _normalized_text(semantic.supplemental_status)
    if supplemental_status == "partial" or semantic.supplemental_partial_count:
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
        or semantic.supplemental_failed_count
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
        or semantic.supplemental_unavailable_count
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

    for item in semantic.uncertainties:
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


def _legacy_mapping(
    value: object,
    context: str,
    *,
    allow_none: bool = True,
) -> dict[str, Any]:
    if value is None and allow_none:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    result: dict[str, Any] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{context} keys must be strings")
        result[key] = item
    return result


def _legacy_fields(
    value: Mapping[str, Any],
    allowed: set[str],
    context: str,
) -> None:
    unexpected = set(value) - allowed
    if unexpected:
        raise ValueError(
            f"{context} contains unsupported field(s): "
            + ", ".join(sorted(unexpected))
        )


def _legacy_list(value: object, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{context} must be a list")
    return list(value)


def _legacy_non_empty_string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _legacy_optional_string(value: object, context: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string or null")
    return value


def _legacy_string_list(value: object, context: str) -> list[str]:
    items = _legacy_list(value, context)
    result: list[str] = []
    for index, item in enumerate(items):
        result.append(_legacy_non_empty_string(item, f"{context}[{index}]"))
    return result


def _legacy_count(value: object, context: str) -> int:
    if value is None:
        return 0
    if type(value) is int and value >= 0:
        return value
    if isinstance(value, (list, tuple)):
        return len(value)
    raise ValueError(f"{context} must be a non-negative integer or list")


def _raise_legacy_value_error(message: str) -> bool:
    raise ValueError(message)


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
