from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from review_agent.context import build_reviewer_envelope
from review_agent.models import (
    Assignment,
    IntentPacket,
    ModelInvocationEnvelope,
    ReviewerResult,
    ReviewerResultStatus,
    ReviewerRuntimeMetadata,
    ReviewerTerminationReason,
)
from review_agent.model_adapter_factory import ModelAdapterFactory
from review_agent.model_adapter import ModelAdapter
from review_agent.model_protocol import ModelResponse
from review_agent.reviewer import ReviewerRun, reviewer_result_to_dict, run_single_reviewer
from review_agent.reviewer_runtime import (
    reviewer_runtime_to_dict,
)


@dataclass(frozen=True)
class ReviewerExecution:
    reviewer_index: int
    trace_id: str
    assignment: Assignment
    envelope: ModelInvocationEnvelope
    response: ModelResponse
    result: ReviewerResult
    runtime: ReviewerRuntimeMetadata = field(default_factory=ReviewerRuntimeMetadata)


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
    adapter_factory: ModelAdapterFactory,
    assignments: list[Assignment],
    intent: IntentPacket,
    diff_excerpt: list[str],
    observations: dict[str, str],
    trace_id_prefix: str,
    *,
    model: str = "configured-reviewer-model",
) -> MultiReviewerRun:
    prepared: list[tuple[int, Assignment, ModelAdapter | None, Exception | None]] = []
    for index, assignment in enumerate(assignments):
        try:
            prepared.append((index, assignment, adapter_factory.create(), None))
        except Exception as error:
            prepared.append((index, assignment, None, error))

    def execute(
        item: tuple[int, Assignment, ModelAdapter | None, Exception | None],
    ) -> ReviewerExecution:
        index, assignment, adapter, creation_error = item
        trace_id = f"{trace_id_prefix}-reviewer-{index}"
        started_at = perf_counter()
        try:
            if creation_error is not None:
                raise creation_error
            if adapter is None:
                raise RuntimeError("reviewer adapter creation returned no adapter")
            run = run_single_reviewer(
                adapter=adapter,
                assignment=assignment,
                intent=intent,
                diff_excerpt=diff_excerpt,
                observations=observations,
                trace_id=trace_id,
                model=model,
            )
            return _execution_from_run(index, trace_id, assignment, run)
        except Exception as error:
            return failed_reviewer_execution(
                index=index,
                trace_id=trace_id,
                assignment=assignment,
                intent=intent,
                diff_excerpt=diff_excerpt,
                observations=observations,
                error=error,
                model=model,
                elapsed_seconds=perf_counter() - started_at,
                retained_observation_refs=tuple(sorted(observations)),
            )

    if len(prepared) < 2:
        executions = [execute(item) for item in prepared]
    else:
        with ThreadPoolExecutor(
            max_workers=min(len(prepared), 32),
            thread_name_prefix="reviewer",
        ) as executor:
            # executor.map preserves input order while the work itself overlaps.
            executions = list(executor.map(execute, prepared))
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
                "runtime": reviewer_runtime_to_dict(execution.runtime),
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
        runtime=run.runtime,
    )


def failed_reviewer_execution(
    index: int,
    trace_id: str,
    assignment: Assignment,
    intent: IntentPacket,
    diff_excerpt: list[str],
    observations: dict[str, str],
    error: Exception,
    *,
    model: str,
    elapsed_seconds: float = 0.0,
    retained_observation_refs: tuple[str, ...] = (),
) -> ReviewerExecution:
    error_type = type(error).__name__
    error_message = str(error)
    envelope = build_reviewer_envelope(
        assignment=assignment,
        intent=intent,
        code_snippets={"Diff Excerpt": "\n".join(diff_excerpt)},
        observations=observations,
        trace_id=trace_id,
        model=model,
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
            observation_refs=list(retained_observation_refs),
            investigation_summary=f"{assignment.role} reviewer failed: {error_type}: {error_message}",
            status=ReviewerResultStatus.FAILED,
        ),
        runtime=ReviewerRuntimeMetadata(
            elapsed_seconds=max(0.0, elapsed_seconds),
            termination_reason=ReviewerTerminationReason.RUNTIME_FAILURE,
        ),
    )
