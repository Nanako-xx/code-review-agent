from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from review_agent.models import ContractAssessment, ReviewerFinding
from review_agent.orchestrator import ReviewerExecution


@dataclass(frozen=True)
class CanonicalFinding:
    claim: str
    severity: str
    confidence: str
    evidence_refs: list[str]
    reviewer_indices: list[int]
    roles: list[str]
    suggested_action: str | None = None


@dataclass(frozen=True)
class RejectedFinding:
    reviewer_index: int
    role: str
    claim: str
    reason: str
    evidence_refs: list[str]
    missing_evidence_refs: list[str]


@dataclass(frozen=True)
class ContractCoverage:
    reviewer_index: int
    role: str
    contract: str
    status: str
    summary: str
    evidence_refs: list[str]
    unsupported_evidence_refs: list[str]


@dataclass(frozen=True)
class EvidenceReconciliation:
    canonical_findings: list[CanonicalFinding]
    rejected_findings: list[RejectedFinding]
    remaining_disagreements: list[str]
    contract_coverage: list[ContractCoverage]
    evidence_quality: str


def reconcile_evidence(
    executions: list[ReviewerExecution],
    authorized_observation_ids: set[str],
) -> EvidenceReconciliation:
    canonical_by_key: dict[tuple[str, tuple[str, ...]], CanonicalFinding] = {}
    rejected: list[RejectedFinding] = []
    contract_coverage: list[ContractCoverage] = []

    for execution in executions:
        for finding in execution.result.confirmed_findings:
            missing_refs = _missing_refs(finding.evidence_refs, authorized_observation_ids)
            if not finding.evidence_refs or missing_refs:
                rejected.append(_rejected_finding(execution, finding, missing_refs))
                continue
            key = (_normalize_claim(finding.claim), tuple(sorted(finding.evidence_refs)))
            canonical_by_key[key] = _merge_canonical_finding(
                existing=canonical_by_key.get(key),
                execution=execution,
                finding=finding,
            )

        for assessment in execution.result.contract_assessments:
            contract_coverage.append(_contract_coverage(execution, assessment, authorized_observation_ids))

    return EvidenceReconciliation(
        canonical_findings=list(canonical_by_key.values()),
        rejected_findings=rejected,
        remaining_disagreements=[],
        contract_coverage=contract_coverage,
        evidence_quality="verified" if not rejected else "unsupported_claims",
    )


def reconciliation_to_dict(reconciliation: EvidenceReconciliation) -> dict[str, Any]:
    return asdict(reconciliation)


def _missing_refs(evidence_refs: list[str], authorized_observation_ids: set[str]) -> list[str]:
    return [ref for ref in evidence_refs if ref not in authorized_observation_ids]


def _rejected_finding(
    execution: ReviewerExecution,
    finding: ReviewerFinding,
    missing_refs: list[str],
) -> RejectedFinding:
    return RejectedFinding(
        reviewer_index=execution.reviewer_index,
        role=execution.assignment.role,
        claim=finding.claim,
        reason="unsupported_claim",
        evidence_refs=list(finding.evidence_refs),
        missing_evidence_refs=missing_refs or list(finding.evidence_refs),
    )


def _merge_canonical_finding(
    existing: CanonicalFinding | None,
    execution: ReviewerExecution,
    finding: ReviewerFinding,
) -> CanonicalFinding:
    if existing is None:
        return CanonicalFinding(
            claim=finding.claim.strip(),
            severity=finding.severity,
            confidence=finding.confidence,
            evidence_refs=list(finding.evidence_refs),
            reviewer_indices=[execution.reviewer_index],
            roles=[execution.assignment.role],
            suggested_action=finding.suggested_action,
        )
    return CanonicalFinding(
        claim=existing.claim,
        severity=existing.severity,
        confidence=existing.confidence,
        evidence_refs=existing.evidence_refs,
        reviewer_indices=[*existing.reviewer_indices, execution.reviewer_index],
        roles=[*existing.roles, execution.assignment.role],
        suggested_action=existing.suggested_action,
    )


def _contract_coverage(
    execution: ReviewerExecution,
    assessment: ContractAssessment,
    authorized_observation_ids: set[str],
) -> ContractCoverage:
    return ContractCoverage(
        reviewer_index=execution.reviewer_index,
        role=execution.assignment.role,
        contract=assessment.contract,
        status=assessment.status.value,
        summary=assessment.summary,
        evidence_refs=list(assessment.evidence_refs),
        unsupported_evidence_refs=_missing_refs(assessment.evidence_refs, authorized_observation_ids),
    )


def _normalize_claim(claim: str) -> str:
    return " ".join(claim.casefold().split())
