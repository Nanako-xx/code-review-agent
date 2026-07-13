from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from review_agent.evidence import EvidenceReconciliation
from review_agent.models import IntentPacket, IntentStatus, QualityGateResult, ReviewerResultStatus
from review_agent.orchestrator import ReviewerExecution


@dataclass(frozen=True)
class CompletionResult:
    status: str
    recommendation: str
    blockers: list[str]
    uncertainties: list[str]
    missing_perspectives: list[str]


def check_completion(
    intent: IntentPacket,
    quality_results: list[QualityGateResult],
    executions: list[ReviewerExecution],
    reconciliation: EvidenceReconciliation,
    *,
    require_final_risk: bool = False,
    final_risk_level: str | None = None,
) -> CompletionResult:
    blockers: list[str] = []
    uncertainties: list[str] = []
    missing_perspectives: list[str] = []

    if intent.status is IntentStatus.INSUFFICIENT:
        blockers.append("Intent Packet insufficient")
    elif intent.status is IntentStatus.PARTIAL:
        uncertainties.append("Intent Packet partial")
    uncertainties.extend(intent.uncertainties)

    if not any(_is_core_reviewer(execution.assignment.role) for execution in executions):
        blockers.append("Core Reviewer did not run")

    for result in quality_results:
        if result.status == "failed":
            uncertainties.append(f"Quality gate failed: {result.name}")

    contract_coverage = _contract_coverage(reconciliation)
    for execution in executions:
        role = execution.assignment.role
        if execution.result.status is ReviewerResultStatus.FAILED:
            if _is_core_reviewer(role):
                blockers.append(f"{role} failed")
            else:
                missing_perspectives.append(role)
        elif execution.result.status is ReviewerResultStatus.PARTIAL:
            uncertainties.append(f"{role} returned partial review")
        elif execution.result.status is ReviewerResultStatus.BLOCKED:
            if _is_core_reviewer(role):
                blockers.append(f"{role} was blocked")
            else:
                missing_perspectives.append(role)
                uncertainties.append(f"{role} was blocked")

        if execution.result.status is ReviewerResultStatus.COMPLETED:
            for contract in execution.assignment.assigned_contract:
                coverage_status = contract_coverage.get((execution.reviewer_index, contract))
                if coverage_status == "complete":
                    continue
                coverage_problem = "missing" if coverage_status is None else "incomplete"
                message = f"{role} {coverage_problem} contract coverage: {contract}"
                if _is_core_reviewer(role):
                    blockers.append(message)
                else:
                    uncertainties.append(message)

    if reconciliation.rejected_findings:
        uncertainties.append("unsupported findings rejected")

    if reconciliation.remaining_disagreements:
        uncertainties.append("reviewer disagreements remain unresolved")

    if require_final_risk and final_risk_level is None:
        blockers.append("Final risk reassessment not completed")

    if final_risk_level in {"high", "critical"}:
        uncertainties.append(f"Final risk is {final_risk_level}")

    if blockers:
        return CompletionResult(
            status="blocked",
            recommendation="manual_review",
            blockers=blockers,
            uncertainties=uncertainties,
            missing_perspectives=missing_perspectives,
        )

    recommendation = _recommendation(reconciliation, uncertainties, missing_perspectives)
    status = "completed_with_uncertainties" if uncertainties or missing_perspectives else "completed"
    return CompletionResult(
        status=status,
        recommendation=recommendation,
        blockers=[],
        uncertainties=uncertainties,
        missing_perspectives=missing_perspectives,
    )


def completion_to_dict(result: CompletionResult) -> dict[str, Any]:
    return asdict(result)


def _is_core_reviewer(role: str) -> bool:
    return "core" in role.casefold()


def _contract_coverage(reconciliation: EvidenceReconciliation) -> dict[tuple[int, str], str]:
    coverage: dict[tuple[int, str], str] = {}
    for row in reconciliation.contract_coverage:
        unsupported_refs = getattr(row, "unsupported_evidence_refs", [])
        status = str(getattr(row, "status", ""))
        key = (int(getattr(row, "reviewer_index")), str(getattr(row, "contract")))
        if unsupported_refs:
            coverage[key] = "incomplete"
            continue
        if status in {"covered", "not_applicable"}:
            coverage[key] = "complete"
            continue
        coverage[key] = "incomplete"
    return coverage


def _recommendation(
    reconciliation: EvidenceReconciliation,
    uncertainties: list[str],
    missing_perspectives: list[str],
) -> str:
    if uncertainties or missing_perspectives:
        return "manual_review"
    if any(_is_blocking_finding(finding) for finding in reconciliation.canonical_findings):
        return "needs_work"
    return "approve"


def _is_blocking_finding(finding: object) -> bool:
    severity = str(getattr(finding, "severity", "")).casefold()
    return severity in {"critical", "high", "blocker"}
