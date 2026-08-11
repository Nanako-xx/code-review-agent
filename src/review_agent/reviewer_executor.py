from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Protocol


_ASSIGNMENT_ID = re.compile(r"\AASG-[0-9a-f]{64}\Z")


class ReviewerLoopV2(Protocol):
    def run(self) -> Any:
        ...


@dataclass(frozen=True)
class ReviewerExecutionResultV2:
    assignment_id: str
    status: str
    output: str | None
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
                error_code="invalid_reviewer_run",
                active_elapsed_seconds=0.0,
            )
        runtime = getattr(run, "runtime", None)
        elapsed = getattr(runtime, "active_elapsed_seconds", 0.0)
        return ReviewerExecutionResultV2(
            assignment_id=assignment_id,
            status=status,
            output=getattr(run, "final_text", None),
            error_code=getattr(run, "error_code", None),
            active_elapsed_seconds=float(elapsed),
        )


__all__ = [
    "ReviewerExecutionResultV2",
    "ReviewerExecutorV2",
    "ReviewerLoopV2",
]
