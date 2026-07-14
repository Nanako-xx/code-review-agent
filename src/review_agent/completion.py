from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from review_agent.evidence import EvidenceReconciliation
from review_agent.models import IntentPacket, IntentStatus, QualityGateResult, ReviewerResultStatus
from review_agent.orchestrator import ReviewerExecution
from review_agent.quality import QualityGatePlan


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
    quality_plan: QualityGatePlan | None = None,
    quality_observation_refs: set[str] | None = None,
    require_final_risk: bool = False,
    final_risk_level: str | None = None,
    semantic_reconciliation: Mapping[str, Any] | None = None,
) -> CompletionResult:
    """Determine the authoritative review outcome.

    ``executions`` is the initial Portfolio execution set. Supplemental
    executions resolve individual semantic disagreements and must not satisfy
    initial Core presence or assigned Contract coverage. The filtering below
    preserves that boundary defensively if a caller accidentally mixes them.
    """
    blockers: list[str] = []
    uncertainties: list[str] = []
    missing_perspectives: list[str] = []
    initial_executions = [
        execution for execution in executions if _is_initial_execution(execution)
    ]

    if intent.status is IntentStatus.INSUFFICIENT:
        blockers.append("Intent Packet insufficient")
    elif intent.status is IntentStatus.PARTIAL:
        uncertainties.append("Intent Packet partial")
    uncertainties.extend(intent.uncertainties)

    if not any(
        _is_core_reviewer(execution.assignment) for execution in initial_executions
    ):
        blockers.append("Core Reviewer did not run")

    if quality_plan is None:
        for result in quality_results:
            if result.status == "failed":
                uncertainties.append(f"Quality gate failed: {result.name}")
    else:
        _check_quality_gates(
            quality_plan,
            quality_results,
            blockers,
            uncertainties,
            quality_observation_refs,
        )

    contract_coverage = _contract_coverage(reconciliation)
    for execution in initial_executions:
        role = execution.assignment.role
        is_core = _is_core_reviewer(execution.assignment)
        if execution.result.status is ReviewerResultStatus.FAILED:
            if is_core:
                blockers.append(f"{role} failed")
            else:
                missing_perspectives.append(role)
        elif execution.result.status is ReviewerResultStatus.PARTIAL:
            uncertainties.append(f"{role} returned partial review")
        elif execution.result.status is ReviewerResultStatus.BLOCKED:
            if is_core:
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
                if is_core:
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

    budget_exhausted = False
    if semantic_reconciliation is not None:
        budget_exhausted = _apply_semantic_completion_signals(
            semantic_reconciliation,
            uncertainties,
        )

    if blockers:
        return CompletionResult(
            status="blocked",
            recommendation="manual_review",
            blockers=blockers,
            uncertainties=uncertainties,
            missing_perspectives=missing_perspectives,
        )

    if budget_exhausted:
        return CompletionResult(
            status="budget_exhausted",
            recommendation="manual_review",
            blockers=[],
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


def _is_core_reviewer(assignment: object) -> bool:
    role_kind = getattr(assignment, "role_kind", None)
    if role_kind is not None and role_kind != "legacy":
        return role_kind == "core"
    role = getattr(assignment, "role", "")
    return isinstance(role, str) and "core" in role.casefold()


def _is_initial_execution(execution: ReviewerExecution) -> bool:
    assignment = execution.assignment
    return getattr(assignment, "planner_source", None) != "semantic_reconciler"


def _apply_semantic_completion_signals(
    semantic: Mapping[str, Any],
    uncertainties: list[str],
) -> bool:
    status = _normalized_semantic_status(semantic.get("status"))
    supplemental = _mapping(semantic.get("supplemental"))
    budget = _mapping(supplemental.get("budget"))
    stop_reason = _normalized_text(
        supplemental.get(
            "stop_reason",
            semantic.get("stop_reason", budget.get("stop_reason")),
        )
    )
    model_status = _normalized_text(_mapping(semantic.get("model")).get("status"))

    if status == "fallback" or model_status == "fallback" or stop_reason == "model_fallback":
        _append_once(
            uncertainties,
            "Semantic reconciliation used deterministic fallback",
        )
    elif status == "partial":
        _append_once(uncertainties, "Semantic reconciliation is partial")
    elif status not in {"accepted", "local_only"}:
        detail = status or "missing"
        _append_once(
            uncertainties,
            f"Semantic reconciliation status is {detail}",
        )

    if _has_items(semantic.get("remaining_disagreements")):
        _append_once(uncertainties, "reviewer disagreements remain unresolved")

    supplemental_status = _normalized_text(supplemental.get("status"))
    if supplemental_status == "partial" or _has_items(supplemental.get("partial")):
        _append_once(
            uncertainties,
            "Supplemental investigation returned partial results",
        )
    if (
        supplemental_status == "failed"
        or _has_items(supplemental.get("failed"))
        or stop_reason == "task_failure"
    ):
        _append_once(uncertainties, "Supplemental investigation failed")
    if (
        supplemental_status == "unavailable"
        or _has_items(supplemental.get("unavailable"))
        or stop_reason in {"unavailable", "provider_unavailable"}
    ):
        _append_once(uncertainties, "Supplemental investigation unavailable")

    budget_exhausted = (
        supplemental_status == "budget_exhausted"
        or stop_reason in {"budget_exhausted", "max_waves"}
    )
    if budget_exhausted:
        _append_once(
            uncertainties,
            "Supplemental investigation stopped because budget was exhausted",
        )

    for item in _string_items(semantic.get("uncertainties")):
        _append_once(uncertainties, f"Semantic reconciliation uncertainty: {item}")
    return budget_exhausted


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _normalized_semantic_status(value: object) -> str:
    status = _normalized_text(value)
    if status in {"local", "disabled"}:
        return "local_only"
    return status


def _normalized_text(value: object) -> str:
    return value.strip().casefold() if isinstance(value, str) else ""


def _has_items(value: object) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    if isinstance(value, str):
        return bool(value.strip())
    try:
        return len(value) > 0  # type: ignore[arg-type]
    except TypeError:
        return True


def _string_items(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value if str(item).strip()]


def _append_once(items: list[str], item: str) -> None:
    if item not in items:
        items.append(item)


def _check_quality_gates(
    plan: QualityGatePlan,
    results: list[QualityGateResult],
    blockers: list[str],
    uncertainties: list[str],
    known_observation_refs: set[str] | None,
) -> None:
    for issue in plan.discovery_issues:
        uncertainties.append(f"Quality gate discovery issue: {issue}")

    by_name: dict[str, QualityGateResult] = {}
    duplicate_names: set[str] = set()
    for result in results:
        if result.name in by_name:
            duplicate_names.add(result.name)
            continue
        by_name[result.name] = result
    for name in sorted(duplicate_names):
        blockers.append(f"Quality gate result duplicated: {name}")

    planned_names = {gate.name for gate in plan.gates}
    for gate in plan.gates:
        result = by_name.get(gate.name)
        if result is None:
            blockers.append(f"Quality gate result missing: {gate.name}")
            continue
        if (
            result.category != gate.category
            or result.cost != gate.cost
            or result.source != gate.source
            or result.blocking != gate.blocking
            or result.command != gate.command
        ):
            blockers.append(f"Quality gate result metadata mismatch: {gate.name}")
            continue
        if result.observation_ref is None:
            blockers.append(f"Quality gate observation missing: {gate.name}")
            continue
        if (
            known_observation_refs is not None
            and result.observation_ref not in known_observation_refs
        ):
            blockers.append(f"Quality gate observation unknown: {gate.name}")
            continue
        if result.status == "passed":
            continue
        if result.status == "skipped" and not gate.blocking:
            continue
        detail = result.reason or result.summary
        message = f"Quality gate {result.status}: {gate.name} ({detail})"
        if gate.blocking:
            blockers.append(message)
        else:
            uncertainties.append(message)

    for name in sorted(set(by_name) - planned_names):
        uncertainties.append(f"Unplanned Quality gate result: {name}")


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
