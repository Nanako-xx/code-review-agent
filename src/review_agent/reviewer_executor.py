from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Protocol

from review_agent.review_protocol import ReviewerOutput, WireProtocolError
from review_agent.reviewer_output import RejectedReviewerFinding

_ASSIGNMENT_ID = re.compile(r"\AASG-[0-9a-f]{64}\Z")


class ReviewerLoopV2(Protocol):
    def run(self) -> Any:
        ...


@dataclass(frozen=True)
class ReviewerExecutionResultV2:
    assignment_id: str
    status: str
    output: str | None
    reviewer_output: ReviewerOutput | None
    rejected_findings: tuple[RejectedReviewerFinding, ...]
    error_code: str | None
    active_elapsed_seconds: float

    def __post_init__(self) -> None:
        if type(self.assignment_id) is not str or _ASSIGNMENT_ID.fullmatch(
            self.assignment_id
        ) is None:
            raise ValueError("assignment_id is invalid")
        if self.status not in {
            "completed",
            "failed",
            "timeout",
            "invalid_output",
            "cancelled",
        }:
            raise ValueError("Reviewer execution status is invalid")
        if self.output is not None and type(self.output) is not str:
            raise ValueError("output must be text or null")
        if self.reviewer_output is not None and type(
            self.reviewer_output
        ) is not ReviewerOutput:
            raise ValueError("reviewer_output must be ReviewerOutput or null")
        if type(self.rejected_findings) is not tuple or any(
            type(item) is not RejectedReviewerFinding
            for item in self.rejected_findings
        ):
            raise ValueError(
                "rejected_findings must contain RejectedReviewerFinding values"
            )
        if self.error_code is not None and (
            type(self.error_code) is not str or not self.error_code
        ):
            raise ValueError("error_code must be non-empty text or null")
        if (
            isinstance(self.active_elapsed_seconds, bool)
            or not isinstance(self.active_elapsed_seconds, (int, float))
            or not math.isfinite(self.active_elapsed_seconds)
            or self.active_elapsed_seconds < 0
        ):
            raise ValueError("active_elapsed_seconds must be non-negative")


class ReviewerExecutorV2:
    """Convert one Reviewer failure into one isolated execution result."""

    def execute(
        self,
        assignment_id: str,
        loop: ReviewerLoopV2,
    ) -> ReviewerExecutionResultV2:
        if type(assignment_id) is not str or _ASSIGNMENT_ID.fullmatch(
            assignment_id
        ) is None:
            raise ValueError("assignment_id is invalid")
        if not hasattr(loop, "run"):
            raise ValueError("loop must implement run")
        try:
            run = loop.run()
        except Exception:
            return ReviewerExecutionResultV2(
                assignment_id=assignment_id,
                status="failed",
                output=None,
                reviewer_output=None,
                rejected_findings=(),
                error_code="reviewer_runtime_error",
                active_elapsed_seconds=0.0,
            )
        status = getattr(run, "status", None)
        if status not in {
            "completed",
            "failed",
            "timeout",
            "invalid_output",
            "cancelled",
        }:
            return ReviewerExecutionResultV2(
                assignment_id=assignment_id,
                status="failed",
                output=None,
                reviewer_output=None,
                rejected_findings=(),
                error_code="invalid_reviewer_run",
                active_elapsed_seconds=0.0,
            )
        runtime = getattr(run, "runtime", None)
        elapsed = getattr(runtime, "active_elapsed_seconds", 0.0)
        output = getattr(run, "final_text", None)
        reviewer_output = getattr(run, "reviewer_output", None)
        if status == "completed" and reviewer_output is None and type(output) is str:
            try:
                reviewer_output = ReviewerOutput.from_json(output)
            except WireProtocolError:
                return ReviewerExecutionResultV2(
                    assignment_id=assignment_id,
                    status="failed",
                    output=None,
                    reviewer_output=None,
                    rejected_findings=(),
                    error_code="invalid_reviewer_run",
                    active_elapsed_seconds=float(elapsed),
                )
        if reviewer_output is not None and type(reviewer_output) is not ReviewerOutput:
            return ReviewerExecutionResultV2(
                assignment_id=assignment_id,
                status="failed",
                output=None,
                reviewer_output=None,
                rejected_findings=(),
                error_code="invalid_reviewer_run",
                active_elapsed_seconds=float(elapsed),
            )
        if status == "completed" and (
            type(output) is not str or reviewer_output is None
        ):
            return ReviewerExecutionResultV2(
                assignment_id=assignment_id,
                status="failed",
                output=None,
                reviewer_output=None,
                rejected_findings=(),
                error_code="invalid_reviewer_run",
                active_elapsed_seconds=float(elapsed),
            )
        rejected_findings = getattr(run, "rejected_findings", ())
        if type(rejected_findings) is not tuple or any(
            type(item) is not RejectedReviewerFinding
            for item in rejected_findings
        ):
            return ReviewerExecutionResultV2(
                assignment_id=assignment_id,
                status="failed",
                output=None,
                reviewer_output=None,
                rejected_findings=(),
                error_code="invalid_reviewer_run",
                active_elapsed_seconds=float(elapsed),
            )
        return ReviewerExecutionResultV2(
            assignment_id=assignment_id,
            status=status,
            output=output,
            reviewer_output=reviewer_output,
            rejected_findings=rejected_findings,
            error_code=getattr(run, "error_code", None),
            active_elapsed_seconds=float(elapsed),
        )


def reviewer_execution_result_v2_to_dict(
    result: ReviewerExecutionResultV2,
) -> dict[str, Any]:
    if not isinstance(result, ReviewerExecutionResultV2):
        raise ValueError("result must be ReviewerExecutionResultV2")
    return {
        "assignment_id": result.assignment_id,
        "status": result.status,
        "output": result.output,
        "reviewer_output": (
            result.reviewer_output.to_dict()
            if result.reviewer_output is not None
            else None
        ),
        "rejected_findings": [
            item.to_dict() for item in result.rejected_findings
        ],
        "error_code": result.error_code,
        "active_elapsed_seconds": round(result.active_elapsed_seconds, 6),
    }


def reviewer_execution_result_v2_from_dict(
    payload: Any,
) -> ReviewerExecutionResultV2:
    expected = {
        "assignment_id",
        "status",
        "output",
        "reviewer_output",
        "rejected_findings",
        "error_code",
        "active_elapsed_seconds",
    }
    if type(payload) is not dict or set(payload) != expected:
        raise ValueError("Reviewer execution result v2 schema is invalid")
    reviewer_output_payload = payload["reviewer_output"]
    reviewer_output = (
        ReviewerOutput.from_dict(reviewer_output_payload)
        if reviewer_output_payload is not None
        else None
    )
    rejected = payload["rejected_findings"]
    if type(rejected) is not list:
        raise ValueError("Reviewer execution rejections must be an array")
    result = ReviewerExecutionResultV2(
        assignment_id=payload["assignment_id"],
        status=payload["status"],
        output=payload["output"],
        reviewer_output=reviewer_output,
        rejected_findings=tuple(
            RejectedReviewerFinding.from_dict(item) for item in rejected
        ),
        error_code=payload["error_code"],
        active_elapsed_seconds=payload["active_elapsed_seconds"],
    )
    if result.status == "completed" and (
        result.reviewer_output is None
        or result.output != result.reviewer_output.to_json()
        or result.error_code is not None
    ):
        raise ValueError("Completed Reviewer execution result is inconsistent")
    return result


__all__ = [
    "ReviewerExecutionResultV2",
    "ReviewerExecutorV2",
    "ReviewerLoopV2",
    "reviewer_execution_result_v2_from_dict",
    "reviewer_execution_result_v2_to_dict",
]
