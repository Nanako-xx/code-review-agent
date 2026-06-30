from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from review_agent.models import Assignment, IntentPacket, ModelInvocationEnvelope, ReviewerResult
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
        run = run_single_reviewer(
            provider=provider,
            assignment=assignment,
            intent=intent,
            diff_excerpt=diff_excerpt,
            observations=observations,
            trace_id=trace_id,
        )
        executions.append(_execution_from_run(index, trace_id, assignment, run))
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
