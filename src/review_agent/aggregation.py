from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import re
from typing import Any, Iterable
import unicodedata

from review_agent.pr_workspace import (
    PRWorkspaceStore,
    SnapshotWorkspace,
)
from review_agent.review_protocol import (
    FinalFinding,
    FindingSeverity,
    ReviewPlan,
    ReviewerFinding,
    ReviewerRoleKind,
    ReviewResult,
    ReviewResultStatus,
)
from review_agent.reviewer_executor import ReviewerExecutionResultV2
from review_agent.reviewer_output import RejectedReviewerFinding
from review_agent.safe_io import (
    SafeIOError,
    canonical_json_bytes,
    strict_json_loads,
)


AGGREGATION_RECORD_SCHEMA = "aggregation_record_v1"
_SAFE_REVIEWER_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_SAFE_ERROR_CODE = re.compile(r"\A[a-z][a-z0-9_]{0,63}\Z")
_SNAPSHOT_ID = re.compile(r"\AS-[0-9a-f]{64}\Z")
_SEVERITY_RANK = {
    FindingSeverity.LOW: 1,
    FindingSeverity.MEDIUM: 2,
    FindingSeverity.HIGH: 3,
    FindingSeverity.BLOCKER: 4,
}
_ROLE_RANK = {
    ReviewerRoleKind.CORE: 0,
    ReviewerRoleKind.ADVERSARIAL: 1,
    ReviewerRoleKind.DYNAMIC: 2,
}
_REJECTION_TEXT = {
    "candidate_not_object": "the candidate was not a Finding object",
    "finding_fields_invalid": "the candidate did not use the Finding v2 fields",
    "claim_invalid": "the defect claim was invalid",
    "severity_invalid": "the severity was invalid",
    "path_invalid": "the repository path was invalid",
    "line_invalid": "the line anchor was invalid",
    "suggestion_invalid": "the correction suggestion was invalid",
    "suggestion_not_actionable": "the correction suggestion was not actionable",
    "diff_index_unavailable": "the current Diff index was unavailable",
    "path_outside_assignment": "the path was outside the Assignment",
    "path_not_in_diff": "the path did not resolve in the current Diff",
    "line_not_in_diff": "the line anchor did not resolve in the current Diff",
}


class AggregationError(ValueError):
    pass


class AggregationIntegrityError(AggregationError):
    pass


@dataclass(frozen=True)
class ReviewAggregationInput:
    reviewer_id: str
    execution: ReviewerExecutionResultV2

    def __post_init__(self) -> None:
        if type(self.reviewer_id) is not str or _SAFE_REVIEWER_ID.fullmatch(
            self.reviewer_id
        ) is None:
            raise AggregationError("reviewer_id is invalid")
        if not isinstance(self.execution, ReviewerExecutionResultV2):
            raise AggregationError(
                "execution must be ReviewerExecutionResultV2"
            )


@dataclass(frozen=True)
class AggregationBundle:
    review_result: ReviewResult
    aggregation_record: dict[str, Any]
    review_result_bytes: bytes
    aggregation_bytes: bytes
    reused: bool = False


@dataclass(frozen=True)
class _Candidate:
    reviewer_id: str
    assignment_id: str
    assignment_order: int
    role_kind: ReviewerRoleKind
    candidate_index: int
    finding: ReviewerFinding
    normalized_claim: str
    normalized_suggestion: str
    finding_id: str


def normalize_review_text(value: str) -> str:
    if type(value) is not str or not value.strip() or "\x00" in value:
        raise AggregationError("Review text is invalid")
    return " ".join(unicodedata.normalize("NFKC", value).strip().split())


def finding_fingerprint(snapshot_id: str, finding: ReviewerFinding) -> str:
    if type(snapshot_id) is not str or _SNAPSHOT_ID.fullmatch(snapshot_id) is None:
        raise AggregationError("snapshot_id is invalid")
    if type(finding) is not ReviewerFinding:
        raise AggregationError("finding must be ReviewerFinding")
    identity = {
        "snapshot_id": snapshot_id,
        "path": finding.path,
        "line": finding.line,
        "claim": normalize_review_text(finding.claim),
    }
    try:
        digest = hashlib.sha256(canonical_json_bytes(identity)).hexdigest()
    except SafeIOError as error:
        raise AggregationError("Finding identity is not canonical") from error
    return "F-" + digest


class DeterministicReviewAggregator:
    def aggregate(
        self,
        pr_id: str,
        plan: ReviewPlan,
        reviewer_inputs: Iterable[ReviewAggregationInput],
    ) -> AggregationBundle:
        if type(plan) is not ReviewPlan:
            raise AggregationError("plan must be ReviewPlan")
        inputs = tuple(reviewer_inputs)
        if any(type(item) is not ReviewAggregationInput for item in inputs):
            raise AggregationError(
                "reviewer_inputs must contain ReviewAggregationInput values"
            )
        by_assignment: dict[str, ReviewAggregationInput] = {}
        known_assignments = {
            assignment.assignment_id for assignment in plan.assignments
        }
        for item in inputs:
            assignment_id = item.execution.assignment_id
            if assignment_id not in known_assignments:
                raise AggregationIntegrityError(
                    "Reviewer result references an unknown Assignment"
                )
            if assignment_id in by_assignment:
                raise AggregationIntegrityError(
                    "Assignment has multiple Reviewer results"
                )
            by_assignment[assignment_id] = item
        reviewer_ids = [item.reviewer_id for item in inputs]
        if len(reviewer_ids) != len(set(reviewer_ids)):
            raise AggregationIntegrityError("reviewer_id values must be unique")

        reviewer_records: list[dict[str, Any]] = []
        candidates: list[_Candidate] = []
        uncertainty_groups: list[list[str]] = []
        valid_count = 0
        for assignment_order, assignment in enumerate(plan.assignments):
            item = by_assignment.get(assignment.assignment_id)
            source_uncertainties: list[str] = []
            if item is None:
                reviewer_id = assignment.assignment_id
                execution = None
                output_valid = False
                status = "missing"
                error_code = "missing_result"
            else:
                reviewer_id = item.reviewer_id
                execution = item.execution
                status = execution.status
                error_code = _safe_error_code(execution.error_code, status)
                output_valid = _valid_execution_output(execution)

            if output_valid and execution is not None:
                valid_count += 1
                assert execution.reviewer_output is not None
                for candidate_index, finding in enumerate(
                    execution.reviewer_output.findings
                ):
                    candidates.append(
                        _Candidate(
                            reviewer_id=reviewer_id,
                            assignment_id=assignment.assignment_id,
                            assignment_order=assignment_order,
                            role_kind=assignment.role_kind,
                            candidate_index=candidate_index,
                            finding=finding,
                            normalized_claim=normalize_review_text(finding.claim),
                            normalized_suggestion=normalize_review_text(
                                finding.suggestion
                            ),
                            finding_id=finding_fingerprint(
                                plan.snapshot_id, finding
                            ),
                        )
                    )
                source_uncertainties.extend(
                    normalize_review_text(value)
                    for value in execution.reviewer_output.uncertainties
                )
                source_uncertainties.extend(
                    _rejection_uncertainty(assignment.role, rejection)
                    for rejection in execution.rejected_findings
                )
            else:
                source_uncertainties.append(
                    _coverage_uncertainty(
                        assignment.role,
                        status,
                        error_code,
                    )
                )

            reviewer_records.append(
                {
                    "reviewer_id": reviewer_id,
                    "assignment_id": assignment.assignment_id,
                    "assignment_order": assignment_order,
                    "role": assignment.role,
                    "role_kind": assignment.role_kind.value,
                    "status": status,
                    "error_code": error_code,
                    "output_valid": output_valid,
                    "rejected_findings": (
                        [item.to_dict() for item in execution.rejected_findings]
                        if execution is not None
                        else []
                    ),
                }
            )
            uncertainty_groups.append(sorted(set(source_uncertainties)))

        result_status = _result_status(valid_count, len(plan.assignments))
        final_findings, merge_groups, candidate_records = _merge_candidates(candidates)
        uncertainties = _ordered_uncertainties(uncertainty_groups)
        review_result = ReviewResult(
            pr_id=pr_id,
            snapshot_id=plan.snapshot_id,
            status=result_status,
            risk_level=plan.risk_level,
            findings=final_findings,
            uncertainties=uncertainties,
        )
        review_result_bytes = review_result.to_json_bytes()
        aggregation_record = {
            "schema_version": AGGREGATION_RECORD_SCHEMA,
            "pr_id": pr_id,
            "snapshot_id": plan.snapshot_id,
            "risk_level": plan.risk_level.value,
            "status": result_status.value,
            "reviewers": reviewer_records,
            "candidates": candidate_records,
            "merge_groups": merge_groups,
            "uncertainties": list(uncertainties),
            "review_result_sha256": hashlib.sha256(
                review_result_bytes
            ).hexdigest(),
        }
        try:
            aggregation_bytes = canonical_json_bytes(aggregation_record)
        except SafeIOError as error:
            raise AggregationError(
                "Aggregation Record is not canonical JSON"
            ) from error
        return AggregationBundle(
            review_result=review_result,
            aggregation_record=aggregation_record,
            review_result_bytes=review_result_bytes,
            aggregation_bytes=aggregation_bytes,
        )

    def publish_or_reuse(
        self,
        workspace_store: PRWorkspaceStore,
        snapshot: SnapshotWorkspace,
        pr_id: str,
        plan: ReviewPlan,
        reviewer_inputs: Iterable[ReviewAggregationInput],
    ) -> AggregationBundle:
        if not isinstance(workspace_store, PRWorkspaceStore):
            raise AggregationError("workspace_store must be PRWorkspaceStore")
        workspace_store.verify_snapshot(snapshot)
        if snapshot.workspace.pr_id != pr_id or snapshot.snapshot_id != plan.snapshot_id:
            raise AggregationIntegrityError(
                "Review Result publication binding does not match"
            )
        state = workspace_store.review_result_bundle_state(snapshot)
        if state == "complete":
            loaded = self.load_published(workspace_store, snapshot)
            if loaded is None:
                raise AggregationIntegrityError(
                    "Complete Review Result bundle could not be loaded"
                )
            _validate_loaded_plan(loaded, plan)
            return loaded

        bundle = self.aggregate(pr_id, plan, reviewer_inputs)
        workspace_store.publish_review_result_bundle(
            snapshot,
            aggregation_bytes=bundle.aggregation_bytes,
            review_result_bytes=bundle.review_result_bytes,
        )
        loaded = self.load_published(workspace_store, snapshot)
        if loaded is None:
            raise AggregationIntegrityError("Review Result publication disappeared")
        _validate_loaded_plan(loaded, plan)
        return replace(loaded, reused=False)

    def load_published(
        self,
        workspace_store: PRWorkspaceStore,
        snapshot: SnapshotWorkspace,
    ) -> AggregationBundle | None:
        if not isinstance(workspace_store, PRWorkspaceStore):
            raise AggregationError("workspace_store must be PRWorkspaceStore")
        stored = workspace_store.load_review_result_bundle(snapshot)
        if stored is None:
            return None
        try:
            result_payload = strict_json_loads(stored.review_result_bytes)
            record = strict_json_loads(stored.aggregation_bytes)
        except SafeIOError as error:
            raise AggregationIntegrityError(
                "Published Review Result JSON is invalid"
            ) from error
        if type(result_payload) is not dict or type(record) is not dict:
            raise AggregationIntegrityError(
                "Published Review Result roots must be objects"
            )
        try:
            result = ReviewResult.from_dict(result_payload)
        except ValueError as error:
            raise AggregationIntegrityError(
                "Published ReviewResult protocol is invalid"
            ) from error
        if result.to_json_bytes() != stored.review_result_bytes:
            raise AggregationIntegrityError(
                "Published ReviewResult is not canonical"
            )
        if (
            result.pr_id != snapshot.workspace.pr_id
            or result.snapshot_id != snapshot.snapshot_id
        ):
            raise AggregationIntegrityError(
                "Published ReviewResult Snapshot binding changed"
            )
        _validate_aggregation_record(
            record,
            result=result,
            review_result_bytes=stored.review_result_bytes,
            aggregation_bytes=stored.aggregation_bytes,
        )
        return AggregationBundle(
            review_result=result,
            aggregation_record=record,
            review_result_bytes=stored.review_result_bytes,
            aggregation_bytes=stored.aggregation_bytes,
            reused=True,
        )


def _valid_execution_output(execution: ReviewerExecutionResultV2) -> bool:
    return (
        execution.status == "completed"
        and execution.error_code is None
        and execution.reviewer_output is not None
        and execution.output == execution.reviewer_output.to_json()
    )


def _safe_error_code(value: str | None, status: str) -> str | None:
    if status == "completed" and value is None:
        return None
    if type(value) is str and _SAFE_ERROR_CODE.fullmatch(value) is not None:
        return value
    return "runtime_failure"


def _coverage_uncertainty(role: str, status: str, error_code: str | None) -> str:
    descriptions = {
        "failed": "failed",
        "timeout": "timed out",
        "invalid_output": "returned invalid output",
        "cancelled": "was cancelled",
        "missing": "did not produce a terminal result",
        "completed": "did not produce a valid ReviewerOutput",
    }
    description = descriptions.get(status, "failed")
    code = error_code or "runtime_failure"
    return normalize_review_text(
        f"{role} coverage is incomplete: the Reviewer {description} ({code})."
    )


def _rejection_uncertainty(
    role: str,
    rejection: RejectedReviewerFinding,
) -> str:
    reason = _REJECTION_TEXT[rejection.reason]
    return normalize_review_text(
        f"{role} omitted finding candidate {rejection.candidate_index + 1}: {reason}."
    )


def _result_status(valid_count: int, planned_count: int) -> ReviewResultStatus:
    if valid_count == planned_count:
        return ReviewResultStatus.COMPLETED
    if valid_count > 0:
        return ReviewResultStatus.PARTIAL
    return ReviewResultStatus.FAILED


def _merge_candidates(
    candidates: list[_Candidate],
) -> tuple[
    tuple[FinalFinding, ...],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    grouped: dict[str, list[_Candidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.finding_id, []).append(candidate)

    selected_by_id: dict[str, _Candidate] = {}
    finals: list[FinalFinding] = []
    for finding_id, values in grouped.items():
        selected = min(values, key=_candidate_selection_key)
        selected_by_id[finding_id] = selected
        severity = max(
            (item.finding.severity for item in values),
            key=lambda item: _SEVERITY_RANK[item],
        )
        finals.append(
            FinalFinding(
                finding_id=finding_id,
                claim=selected.normalized_claim,
                severity=severity,
                path=selected.finding.path,
                line=selected.finding.line,
                suggestion=selected.normalized_suggestion,
            )
        )
    finals.sort(
        key=lambda item: (
            -_SEVERITY_RANK[item.severity],
            item.path,
            item.line,
            item.finding_id,
        )
    )

    merge_groups: list[dict[str, Any]] = []
    for final in finals:
        values = sorted(grouped[final.finding_id], key=_source_order_key)
        selected = selected_by_id[final.finding_id]
        merge_groups.append(
            {
                "finding_id": final.finding_id,
                "selected_reviewer_id": selected.reviewer_id,
                "selected_assignment_id": selected.assignment_id,
                "sources": [_candidate_source(item) for item in values],
            }
        )

    candidate_records = [
        {
            **_candidate_source(candidate),
            "finding_id": candidate.finding_id,
            "finding": candidate.finding.to_dict(),
            "normalized_claim": candidate.normalized_claim,
            "selected": selected_by_id[candidate.finding_id] == candidate,
        }
        for candidate in sorted(candidates, key=_source_order_key)
    ]
    return tuple(finals), merge_groups, candidate_records


def _candidate_selection_key(candidate: _Candidate) -> tuple[Any, ...]:
    return (
        -_SEVERITY_RANK[candidate.finding.severity],
        _ROLE_RANK[candidate.role_kind],
        candidate.assignment_order,
        candidate.reviewer_id,
        candidate.candidate_index,
    )


def _source_order_key(candidate: _Candidate) -> tuple[Any, ...]:
    return (
        candidate.assignment_order,
        candidate.reviewer_id,
        candidate.candidate_index,
    )


def _candidate_source(candidate: _Candidate) -> dict[str, Any]:
    return {
        "reviewer_id": candidate.reviewer_id,
        "assignment_id": candidate.assignment_id,
        "assignment_order": candidate.assignment_order,
        "role_kind": candidate.role_kind.value,
        "candidate_index": candidate.candidate_index,
    }


def _ordered_uncertainties(groups: list[list[str]]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for group in groups:
        for value in sorted(normalize_review_text(item) for item in group):
            if value in seen:
                continue
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)


def _validate_aggregation_record(
    record: dict[str, Any],
    *,
    result: ReviewResult,
    review_result_bytes: bytes,
    aggregation_bytes: bytes,
) -> None:
    expected = {
        "schema_version",
        "pr_id",
        "snapshot_id",
        "risk_level",
        "status",
        "reviewers",
        "candidates",
        "merge_groups",
        "uncertainties",
        "review_result_sha256",
    }
    if set(record) != expected or record["schema_version"] != AGGREGATION_RECORD_SCHEMA:
        raise AggregationIntegrityError("Aggregation Record schema is invalid")
    if (
        record["pr_id"] != result.pr_id
        or record["snapshot_id"] != result.snapshot_id
        or record["risk_level"] != result.risk_level.value
        or record["status"] != result.status.value
        or record["uncertainties"] != list(result.uncertainties)
    ):
        raise AggregationIntegrityError("Aggregation Record binding changed")
    if any(type(record[name]) is not list for name in ("reviewers", "candidates", "merge_groups")):
        raise AggregationIntegrityError("Aggregation Record collections are invalid")
    if record["review_result_sha256"] != hashlib.sha256(
        review_result_bytes
    ).hexdigest():
        raise AggregationIntegrityError("ReviewResult hash binding changed")
    try:
        canonical = canonical_json_bytes(record)
    except SafeIOError as error:
        raise AggregationIntegrityError(
            "Aggregation Record is not canonical JSON"
        ) from error
    if canonical != aggregation_bytes:
        raise AggregationIntegrityError("Aggregation Record is not canonical")


def _validate_loaded_plan(
    bundle: AggregationBundle,
    plan: ReviewPlan,
) -> None:
    result = bundle.review_result
    if (
        result.snapshot_id != plan.snapshot_id
        or result.risk_level is not plan.risk_level
    ):
        raise AggregationIntegrityError(
            "Published ReviewResult does not match the immutable ReviewPlan"
        )
    reviewers = bundle.aggregation_record.get("reviewers")
    if type(reviewers) is not list or [
        item.get("assignment_id") if type(item) is dict else None
        for item in reviewers
    ] != [assignment.assignment_id for assignment in plan.assignments]:
        raise AggregationIntegrityError(
            "Published Aggregation Record Assignment order changed"
        )


__all__ = [
    "AGGREGATION_RECORD_SCHEMA",
    "AggregationBundle",
    "AggregationError",
    "AggregationIntegrityError",
    "DeterministicReviewAggregator",
    "ReviewAggregationInput",
    "finding_fingerprint",
    "normalize_review_text",
]
