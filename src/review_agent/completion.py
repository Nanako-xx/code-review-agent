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
) -> CompletionResult:
    blockers: list[str] = []
    uncertainties: list[str] = []
    missing_perspectives: list[str] = []

    if intent.status is IntentStatus.INSUFFICIENT:
        blockers.append("Intent Packet insufficient")

    for result in quality_results:
        if result.status == "failed":
            uncertainties.append(f"Quality gate failed: {result.name}")

    for execution in executions:
        if execution.result.status is not ReviewerResultStatus.FAILED:
            continue
        role = execution.assignment.role
        if _is_core_reviewer(role):
            blockers.append(f"{role} failed")
        else:
            missing_perspectives.append(role)

    if reconciliation.rejected_findings:
        uncertainties.append("unsupported findings rejected")

    if reconciliation.remaining_disagreements:
        uncertainties.append("reviewer disagreements remain unresolved")

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
