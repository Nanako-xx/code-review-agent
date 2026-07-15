from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from typing import Any

from review_agent.models import (
    ClarificationQuestion,
    ClarificationStatus,
    IntentClaim,
    IntentClaimState,
    IntentPacket,
    IntentSource,
    QualityGateResult,
    ReviewerResult,
    RiskAssessment,
)


@dataclass(frozen=True)
class BriefFinding:
    claim: str
    severity: str
    confidence: str
    evidence_refs: list[str]
    reviewer_indices: list[int] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    suggested_action: str | None = None
    path: str | None = None
    line: int | None = None
    impact: str = ""
    verification_performed: list[str] = field(default_factory=list)
    finding_id: str | None = None


@dataclass(frozen=True)
class RejectedHypothesis:
    claim: str
    reason: str
    evidence_refs: list[str] = field(default_factory=list)
    reviewer_index: int | None = None
    role: str | None = None


@dataclass(frozen=True)
class ReviewBrief:
    review_id: str
    base_revision: str
    head_revision: str
    change_intent: dict[str, Any]
    intent_assessment: dict[str, Any]
    initial_and_final_risk_assessment: dict[str, Any]
    quality_gates: list[dict[str, Any]]
    change_map_and_repository_impact: dict[str, Any]
    verified_findings: list[BriefFinding]
    rejected_hypotheses: list[RejectedHypothesis]
    uncertainties: list[str]
    reviewer_disagreements: list[str]
    review_contract_coverage: list[dict[str, Any]]
    verification_evidence: list[dict[str, Any]]
    human_review_checklist_and_reading_order: list[str]
    non_binding_recommendation: str
    orchestration: dict[str, Any] = field(default_factory=dict)
    semantic_reconciliation: dict[str, Any] = field(default_factory=dict)


def build_review_brief(
    *,
    review_id: str,
    base_revision: str,
    head_revision: str,
    intent_packet: IntentPacket,
    risk_assessment: RiskAssessment,
    changed_files: list[str],
    quality_results: list[QualityGateResult],
    observation_summaries: dict[str, str] | None = None,
    repository_intelligence_summary: str | None = None,
    reviewer_result: ReviewerResult | None = None,
    multi_reviewer_summary: dict[str, object] | None = None,
    reconciliation_payload: dict[str, Any] | None = None,
    completion_summary: dict[str, Any] | None = None,
    final_risk_assessment: dict[str, Any] | None = None,
    incremental_priority: dict[str, Any] | None = None,
    planning_summary: dict[str, Any] | None = None,
    semantic_reconciliation_payload: dict[str, Any] | None = None,
) -> ReviewBrief:
    observations = observation_summaries or {}
    reconciliation = reconciliation_payload or {}
    completion = completion_summary or {}
    verified_findings = _verified_findings(reconciliation)
    rejected_hypotheses = _rejected_hypotheses(reconciliation, reviewer_result)
    uncertainties = _uncertainties(intent_packet, risk_assessment, reviewer_result, completion)
    if planning_summary is not None:
        planning_uncertainties = planning_summary.get("uncertainties", [])
        if isinstance(planning_uncertainties, list):
            uncertainties = _dedupe(
                [
                    *uncertainties,
                    *(
                        str(item)
                        for item in planning_uncertainties
                        if str(item).strip()
                    ),
                ]
            )

    change_map: dict[str, Any] = {
        "changed_files": list(changed_files),
        "repository_intelligence_summary": repository_intelligence_summary or "",
        "observation_count": len(observations),
        "reviewer_summary": _reviewer_summary(multi_reviewer_summary, reviewer_result),
    }
    if incremental_priority is not None:
        change_map["incremental_priority"] = dict(incremental_priority)

    return ReviewBrief(
        review_id=review_id,
        base_revision=base_revision,
        head_revision=head_revision,
        change_intent={
            "goal": intent_packet.goal,
            "acceptance_criteria": list(intent_packet.acceptance_criteria),
            "scope": list(intent_packet.scope),
            "constraints": list(intent_packet.constraints),
            "sources": {key: value.value for key, value in intent_packet.sources.items()},
            "provenance": [
                _intent_claim_to_dict(claim) for claim in intent_packet.provenance
            ],
        },
        intent_assessment={
            "status": intent_packet.status.value,
            "uncertainties": list(intent_packet.uncertainties),
            "source_counts": _source_counts(intent_packet),
            "clarification_history": [
                _clarification_to_dict(question)
                for question in intent_packet.clarifications
            ],
            "unresolved_questions": [
                _clarification_to_dict(question)
                for question in intent_packet.clarifications
                if question.status
                in {ClarificationStatus.PENDING, ClarificationStatus.OPEN}
            ],
            "unconfirmed_inferred_claims": [
                _intent_claim_to_dict(claim)
                for claim in intent_packet.provenance
                if claim.source is IntentSource.INFERRED
                and claim.claim_state is IntentClaimState.ACTIVE
            ],
        },
        initial_and_final_risk_assessment={
            "initial": _risk_to_dict(risk_assessment),
            "final": final_risk_assessment
            or {
                "status": "not_reassessed",
                "level": risk_assessment.level.value,
                "reasons": ["Final risk reassessment has not run in the local M1 path."],
            },
        },
        quality_gates=[_quality_result_to_dict(result) for result in quality_results],
        change_map_and_repository_impact=change_map,
        verified_findings=verified_findings,
        rejected_hypotheses=rejected_hypotheses,
        uncertainties=uncertainties,
        reviewer_disagreements=[str(item) for item in reconciliation.get("remaining_disagreements", [])],
        review_contract_coverage=_contract_coverage(reconciliation, reviewer_result),
        verification_evidence=_verification_evidence(quality_results, observations),
        human_review_checklist_and_reading_order=_human_review_checklist(
            changed_files=changed_files,
            risk_assessment=risk_assessment,
            verified_findings=verified_findings,
            uncertainties=uncertainties,
        ),
        non_binding_recommendation=str(completion.get("recommendation", "manual_review")),
        orchestration=dict(planning_summary or {}),
        semantic_reconciliation=dict(semantic_reconciliation_payload or {}),
    )


def review_brief_to_dict(brief: ReviewBrief) -> dict[str, Any]:
    payload = _json_ready(asdict(brief))
    for finding in payload["verified_findings"]:
        if finding.get("finding_id") is None:
            finding.pop("finding_id", None)
    if not brief.semantic_reconciliation:
        payload.pop("semantic_reconciliation", None)
    return payload


def _verified_findings(reconciliation: dict[str, Any]) -> list[BriefFinding]:
    findings: list[BriefFinding] = []
    for item in reconciliation.get("canonical_findings", []):
        row = dict(item)
        findings.append(
            BriefFinding(
                claim=str(row.get("claim", "")),
                severity=str(row.get("severity", "")),
                confidence=str(row.get("confidence", "")),
                evidence_refs=[str(ref) for ref in row.get("evidence_refs", [])],
                reviewer_indices=[int(index) for index in row.get("reviewer_indices", [])],
                roles=[str(role) for role in row.get("roles", [])],
                suggested_action=str(row["suggested_action"]) if row.get("suggested_action") is not None else None,
                path=str(row["path"]) if row.get("path") is not None else None,
                line=int(row["line"]) if row.get("line") is not None else None,
                impact=str(row.get("impact", "")),
                verification_performed=[
                    str(item) for item in row.get("verification_performed", [])
                ],
                finding_id=(
                    str(row["finding_id"])
                    if row.get("finding_id") is not None
                    else None
                ),
            )
        )
    return findings


def _rejected_hypotheses(
    reconciliation: dict[str, Any],
    reviewer_result: ReviewerResult | None,
) -> list[RejectedHypothesis]:
    rejected: list[RejectedHypothesis] = []
    for item in reconciliation.get("rejected_findings", []):
        row = dict(item)
        rejected.append(
            RejectedHypothesis(
                claim=str(row.get("claim", "")),
                reason=str(row.get("reason", "unsupported_claim")),
                evidence_refs=[str(ref) for ref in row.get("evidence_refs", [])],
                reviewer_index=int(row["reviewer_index"]) if row.get("reviewer_index") is not None else None,
                role=str(row["role"]) if row.get("role") is not None else None,
            )
        )
    if reviewer_result is not None:
        rejected.extend(
            RejectedHypothesis(claim=str(item), reason="reviewer_rejected_hypothesis")
            for item in reviewer_result.rejected_hypotheses
        )
    return rejected


def _uncertainties(
    intent_packet: IntentPacket,
    risk_assessment: RiskAssessment,
    reviewer_result: ReviewerResult | None,
    completion: dict[str, Any],
) -> list[str]:
    items: list[str] = []
    items.extend(intent_packet.uncertainties)
    items.extend(risk_assessment.uncertainties)
    if reviewer_result is not None:
        items.extend(reviewer_result.uncertainties)
    items.extend(str(item) for item in completion.get("uncertainties", []))
    items.extend(str(item) for item in completion.get("blockers", []))
    items.extend(f"Missing perspective: {item}" for item in completion.get("missing_perspectives", []))
    return _dedupe(items)


def _contract_coverage(
    reconciliation: dict[str, Any],
    reviewer_result: ReviewerResult | None,
) -> list[dict[str, Any]]:
    if reconciliation.get("contract_coverage"):
        return [dict(item) for item in reconciliation["contract_coverage"]]
    if reviewer_result is None:
        return []
    return [
        {
            "contract": assessment.contract,
            "status": assessment.status.value,
            "summary": assessment.summary,
            "evidence_refs": list(assessment.evidence_refs),
        }
        for assessment in reviewer_result.contract_assessments
    ]


def _verification_evidence(
    quality_results: list[QualityGateResult],
    observations: dict[str, str],
) -> list[dict[str, Any]]:
    evidence = [
        {
            "kind": "quality_gate",
            "name": result.name,
            "status": result.status,
            "summary": result.summary,
            "command": list(result.command),
            "observation_ref": result.observation_ref,
            "category": result.category,
            "cost": result.cost,
            "source": result.source,
            "blocking": result.blocking,
            "reason": result.reason,
            "duration_seconds": result.duration_seconds,
        }
        for result in quality_results
    ]
    evidence.extend(
        {
            "kind": "observation",
            "id": observation_id,
            "summary": summary,
        }
        for observation_id, summary in observations.items()
    )
    return evidence


def _reviewer_summary(
    multi_reviewer_summary: dict[str, object] | None,
    reviewer_result: ReviewerResult | None,
) -> dict[str, object]:
    if multi_reviewer_summary:
        return dict(multi_reviewer_summary)
    if reviewer_result is None:
        return {}
    return {
        "reviewer_count": 1,
        "status_counts": {reviewer_result.status.value: 1},
        "single_reviewer_summary": reviewer_result.investigation_summary,
    }


def _human_review_checklist(
    *,
    changed_files: list[str],
    risk_assessment: RiskAssessment,
    verified_findings: list[BriefFinding],
    uncertainties: list[str],
) -> list[str]:
    checklist: list[str] = []
    checklist.extend(f"Read changed file: {path}" for path in changed_files)
    checklist.extend(f"Check review focus: {focus}" for focus in risk_assessment.suggested_focus)
    checklist.extend(f"Verify finding: {finding.claim}" for finding in verified_findings)
    checklist.extend(f"Resolve uncertainty: {uncertainty}" for uncertainty in uncertainties)
    return checklist or ["No prioritized human review items were generated."]


def _source_counts(intent_packet: IntentPacket) -> dict[str, int]:
    counts: dict[str, int] = {}
    for source in intent_packet.sources.values():
        counts[source.value] = counts.get(source.value, 0) + 1
    return counts


def _intent_claim_to_dict(claim: IntentClaim) -> dict[str, Any]:
    return {
        "claim_id": claim.claim_id,
        "field": claim.field.value,
        "value": claim.value,
        "source": claim.source.value,
        "origin": claim.origin.value,
        "confidence": claim.confidence.value,
        "source_refs": list(claim.source_refs),
        "evidence_refs": list(claim.evidence_refs),
        "claim_state": claim.claim_state.value,
        "conclusion_impact": claim.conclusion_impact.value,
    }


def _clarification_to_dict(question: ClarificationQuestion) -> dict[str, Any]:
    return {
        "question_id": question.question_id,
        "field": question.field.value,
        "question": question.question,
        "rationale": question.rationale,
        "proposed_values": list(question.proposed_values),
        "claim_ids": list(question.claim_ids),
        "status": question.status.value,
        "user_response": question.user_response,
        "continuation_basis": question.continuation_basis,
        "resolved_values": list(question.resolved_values),
        "decision_id": question.decision_id,
    }


def _risk_to_dict(risk_assessment: RiskAssessment) -> dict[str, Any]:
    return {
        "level": risk_assessment.level.value,
        "dimensions": dict(risk_assessment.dimensions),
        "reasons": list(risk_assessment.reasons),
        "signal_refs": list(risk_assessment.signal_refs),
        "uncertainties": list(risk_assessment.uncertainties),
        "suggested_focus": list(risk_assessment.suggested_focus),
    }


def _quality_result_to_dict(result: QualityGateResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "status": result.status,
        "command": list(result.command),
        "summary": result.summary,
        "observation_ref": result.observation_ref,
        "category": result.category,
        "cost": result.cost,
        "source": result.source,
        "blocking": result.blocking,
        "reason": result.reason,
        "exit_code": result.exit_code,
        "duration_seconds": result.duration_seconds,
        "output_truncated": result.output_truncated,
        "sandbox": result.sandbox,
    }


def _dedupe(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _json_ready(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value
