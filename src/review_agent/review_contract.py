from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import PurePosixPath, PureWindowsPath

from review_agent.models import (
    Assignment,
    CompletionMemoryProjection,
    ContractAssessment,
    ContractItemStatus,
    PlannerMemoryProjection,
    ReviewerResult,
    ReviewerResultStatus,
    hard_policy_overflow_diagnostic,
)


REVIEW_CONTRACT_VALIDATION_VERSION = "review-contract-validation-v1"


@dataclass(frozen=True)
class ReviewerCompletionValidation:
    accepted: bool
    deficiencies: tuple[str, ...]


def validate_reviewer_completion(
    assignment: Assignment,
    result: ReviewerResult,
    authorized_observation_ids: set[str],
    *,
    memory_projection: CompletionMemoryProjection | None = None,
) -> ReviewerCompletionValidation:
    """Validate evidence authority and the requirements for a completed review."""
    deficiencies: list[str] = []
    seen_deficiencies: set[str] = set()

    def add_deficiency(message: str) -> None:
        if message in seen_deficiencies:
            return
        seen_deficiencies.add(message)
        deficiencies.append(message)

    if memory_projection is not None and not isinstance(
        memory_projection,
        CompletionMemoryProjection,
    ):
        raise ValueError(
            "memory_projection must be a CompletionMemoryProjection or None"
        )
    # Memory requirements are Portfolio-wide. Each Assignment is validated only
    # against its own expanded Contract; global assignment/check coverage is
    # enforced once by Completion across the complete execution set.

    for observation_ref in result.observation_refs:
        if observation_ref not in authorized_observation_ids:
            add_deficiency(f"unauthorized result observation ref: {observation_ref}")

    for finding_index, finding in enumerate(result.confirmed_findings):
        prefix = f"confirmed finding {finding_index}"
        if not isinstance(finding.claim, str) or not finding.claim.strip():
            add_deficiency(f"{prefix} has an empty claim")
        if finding.severity not in {"blocker", "high", "medium", "low"}:
            add_deficiency(f"{prefix} has invalid severity: {finding.severity}")
        if finding.confidence not in {"high", "medium", "low"}:
            add_deficiency(f"{prefix} has invalid confidence: {finding.confidence}")
        path_error = finding_path_error(finding.path)
        if path_error == "missing":
            add_deficiency(f"{prefix} has no path")
        elif path_error is not None:
            add_deficiency(f"{prefix} has unsafe path: {finding.path}")
        if type(finding.line) is not int or finding.line < 1:
            add_deficiency(f"{prefix} has no positive line")
        if not isinstance(finding.impact, str) or not finding.impact.strip():
            add_deficiency(f"{prefix} has no impact")
        if (
            not isinstance(finding.suggested_action, str)
            or not finding.suggested_action.strip()
        ):
            add_deficiency(f"{prefix} has no suggested action")
        if not finding.verification_performed or any(
            not isinstance(item, str) or not item.strip()
            for item in finding.verification_performed
        ):
            add_deficiency(f"{prefix} has no valid verification performed")
        if not finding.evidence_refs:
            add_deficiency(f"{prefix} has no evidence refs")
        for evidence_ref in finding.evidence_refs:
            if evidence_ref not in authorized_observation_ids:
                add_deficiency(f"unauthorized finding evidence ref: {evidence_ref}")

    for assessment in result.contract_assessments:
        for evidence_ref in assessment.evidence_refs:
            if evidence_ref not in authorized_observation_ids:
                add_deficiency(f"unauthorized contract assessment evidence ref: {evidence_ref}")

    if result.status is ReviewerResultStatus.COMPLETED:
        if not result.investigation_summary.strip():
            add_deficiency("completed result has an empty investigation summary")

        assessments_by_contract: dict[str, list[ContractAssessment]] = {}
        for assessment in result.contract_assessments:
            assessments_by_contract.setdefault(assessment.contract, []).append(assessment)

        complete_statuses = {ContractItemStatus.COVERED, ContractItemStatus.NOT_APPLICABLE}
        for contract in assignment.assigned_contract:
            assessments = assessments_by_contract.get(contract, [])
            if not assessments:
                add_deficiency(f"missing contract assessment: {contract}")
                continue
            if len(assessments) > 1:
                add_deficiency(f"duplicate contract assessment: {contract}")
                continue

            assessment = assessments[0]
            if assessment.status not in complete_statuses:
                add_deficiency(
                    f"incomplete contract assessment: {contract} ({assessment.status.value})"
                )

    return ReviewerCompletionValidation(
        accepted=not deficiencies,
        deficiencies=tuple(deficiencies),
    )


def completion_memory_projection_from_planner(
    projection: PlannerMemoryProjection,
    *,
    max_hard_policy_items: int = 64,
    max_hard_policy_bytes: int = 32_768,
) -> CompletionMemoryProjection:
    if not isinstance(projection, PlannerMemoryProjection):
        raise ValueError("projection must be a PlannerMemoryProjection")
    diagnostics = list(projection.diagnostics)
    policies = (*projection.required_contracts, *projection.required_checks)
    memory_ids = tuple(
        sorted(
            {
                memory_id
                for item in policies
                for memory_id in item.memory_ids
            }
        )
    )
    if policies:
        overflow = hard_policy_overflow_diagnostic(
            "completion",
            tuple(item.to_dict() for item in policies),
            memory_ids,
            max_items=max_hard_policy_items,
            max_bytes=max_hard_policy_bytes,
        )
        if overflow is not None:
            diagnostics.append(overflow)
    return CompletionMemoryProjection(
        required_contracts=projection.required_contracts,
        required_checks=projection.required_checks,
        diagnostics=tuple(diagnostics),
    )


def result_with_validation_deficiencies(
    result: ReviewerResult,
    deficiencies: tuple[str, ...],
) -> ReviewerResult:
    if not deficiencies:
        return result
    message = "Runtime rejected reviewer completion: " + "; ".join(deficiencies)
    status = result.status
    if status is ReviewerResultStatus.COMPLETED:
        status = ReviewerResultStatus.PARTIAL
    uncertainties = list(result.uncertainties)
    if message not in uncertainties:
        uncertainties.append(message)
    return replace(
        result,
        uncertainties=uncertainties,
        investigation_summary=result.investigation_summary.strip() or message,
        status=status,
    )


def finding_path_error(path: object) -> str | None:
    if not isinstance(path, str) or not path.strip():
        return "missing"
    if path != path.strip() or "\\" in path:
        return "unsafe"
    posix = PurePosixPath(path)
    windows = PureWindowsPath(path)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or any(part in {"", ".", ".."} for part in path.split("/"))
        or posix.parts[0] in {".git", ".env"}
        or posix.as_posix() != path
    ):
        return "unsafe"
    return None
