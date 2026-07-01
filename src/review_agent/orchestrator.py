from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from review_agent.context import build_reviewer_envelope
from review_agent.models import (
    Assignment,
    IntentPacket,
    ModelInvocationEnvelope,
    ReviewerResult,
    ReviewerResultStatus,
)
from review_agent.provider import ModelProvider, ModelResponse
from review_agent.reviewer import ReviewerRun, reviewer_result_to_dict, run_single_reviewer


@dataclass(frozen=True)
class ReviewerExecution:
    reviewer_index: int
    trace_id: str
    assignment: Assignment
    envelope: ModelInvocationEnvelope
    response: ModelResponse
    result: ReviewerResult


@dataclass(frozen=True)
class MultiReviewerRun:
    executions: list[ReviewerExecution]

    @property
    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for execution in self.executions:
            status = execution.result.status.value
            counts[status] = counts.get(status, 0) + 1
        return counts


def run_multi_reviewer(
    provider: ModelProvider,
    assignments: list[Assignment],
    intent: IntentPacket,
    diff_excerpt: list[str],
    observations: dict[str, str],
    trace_id_prefix: str,
) -> MultiReviewerRun:
    executions: list[ReviewerExecution] = []
    for index, assignment in enumerate(assignments):
        trace_id = f"{trace_id_prefix}-reviewer-{index}"
        try:
            run = run_single_reviewer(
                provider=provider,
                assignment=assignment,
                intent=intent,
                diff_excerpt=diff_excerpt,
                observations=observations,
                trace_id=trace_id,
            )
            executions.append(_execution_from_run(index, trace_id, assignment, run))
        except Exception as error:
            executions.append(
                _failed_execution(
                    index=index,
                    trace_id=trace_id,
                    assignment=assignment,
                    intent=intent,
                    diff_excerpt=diff_excerpt,
                    observations=observations,
                    error=error,
                )
            )
    return MultiReviewerRun(executions=executions)


def multi_reviewer_run_to_dict(run: MultiReviewerRun) -> dict[str, Any]:
    return {
        "reviewer_count": len(run.executions),
        "status_counts": run.status_counts,
        "executions": [
            {
                "reviewer_index": execution.reviewer_index,
                "trace_id": execution.trace_id,
                "role": execution.assignment.role,
                "result": reviewer_result_to_dict(execution.result),
                "provider_name": execution.response.provider_name,
                "model": execution.response.model,
            }
            for execution in run.executions
        ],
    }


def _execution_from_run(
    index: int,
    trace_id: str,
    assignment: Assignment,
    run: ReviewerRun,
) -> ReviewerExecution:
    return ReviewerExecution(
        reviewer_index=index,
        trace_id=trace_id,
        assignment=assignment,
        envelope=run.envelope,
        response=run.response,
        result=run.result,
    )


def _failed_execution(
    index: int,
    trace_id: str,
    assignment: Assignment,
    intent: IntentPacket,
    diff_excerpt: list[str],
    observations: dict[str, str],
    error: Exception,
) -> ReviewerExecution:
    error_type = type(error).__name__
    error_message = str(error)
    envelope = build_reviewer_envelope(
        assignment=assignment,
        intent=intent,
        code_snippets={"Diff Excerpt": "\n".join(diff_excerpt)},
        observations=observations,
        trace_id=trace_id,
    )
    return ReviewerExecution(
        reviewer_index=index,
        trace_id=trace_id,
        assignment=assignment,
        envelope=envelope,
        response=ModelResponse(
            content="",
            provider_name="review-agent",
            model="unavailable",
            raw={"error_type": error_type, "error": error_message},
        ),
        result=ReviewerResult(
            uncertainties=[f"{assignment.role} failed before completing review: {error_message}"],
            investigation_summary=f"{assignment.role} reviewer failed: {error_type}: {error_message}",
            status=ReviewerResultStatus.FAILED,
        ),
    )
