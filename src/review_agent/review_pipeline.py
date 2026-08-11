from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Iterable

from review_agent.aggregation import (
    DeterministicReviewAggregator,
    ReviewAggregationInput,
)
from review_agent.pr_workspace import (
    ArtifactDescriptor,
    PRWorkspaceStore,
    SessionWorkspace,
    SnapshotWorkspace,
)
from review_agent.review_protocol import ReviewPlan, ReviewerAssignment
from review_agent.review_renderer import render_review_result_markdown
from review_agent.reviewer_executor import (
    ReviewerExecutionResultV2,
    reviewer_execution_result_v2_from_dict,
    reviewer_execution_result_v2_to_dict,
)
from review_agent.run_state import RunPhase
from review_agent.safe_io import SafeIOError, canonical_json_bytes, strict_json_loads
from review_agent.session import (
    SESSION_V6_PHASES,
    PhaseStatus,
    SessionV6ArtifactRef,
    SessionV6Manifest,
)
from review_agent.session_store import SessionV6Store


REVIEWER_EXECUTION_RECORD_SCHEMA = "reviewer_execution_record_v1"


class ReviewPipelineV6Error(ValueError):
    pass


@dataclass(frozen=True)
class PipelineContextV6:
    workspace_store: PRWorkspaceStore
    snapshot: SnapshotWorkspace
    session: SessionWorkspace
    session_store: SessionV6Store

    def __post_init__(self) -> None:
        if not isinstance(self.workspace_store, PRWorkspaceStore):
            raise ReviewPipelineV6Error(
                "workspace_store must be PRWorkspaceStore"
            )
        self.workspace_store.verify_snapshot(self.snapshot)
        self.workspace_store.verify_session(self.session)
        if (
            self.session.snapshot != self.snapshot
            or self.session_store.workspace_store is not self.workspace_store
            or self.session_store.session != self.session
        ):
            raise ReviewPipelineV6Error("Pipeline Context binding does not match")


PhaseRunner = Callable[[PipelineContextV6], Iterable[SessionV6ArtifactRef]]
PlanLoader = Callable[[PipelineContextV6], ReviewPlan]
ReviewerRunner = Callable[
    [PipelineContextV6, ReviewerAssignment],
    ReviewAggregationInput,
]
ReviewerPersister = Callable[
    [PipelineContextV6, ReviewAggregationInput],
    SessionV6ArtifactRef,
]
ReviewerLoader = Callable[
    [PipelineContextV6, ReviewPlan],
    Iterable[ReviewAggregationInput],
]
AggregationRunner = Callable[
    [PipelineContextV6, ReviewPlan, tuple[ReviewAggregationInput, ...]],
    Iterable[SessionV6ArtifactRef],
]


@dataclass(frozen=True)
class ReviewPipelineServicesV6:
    preflight: PhaseRunner
    intent: PhaseRunner
    planning: PhaseRunner
    load_review_plan: PlanLoader
    assemble_and_run_reviewer: ReviewerRunner
    persist_reviewer_result: ReviewerPersister
    load_reviewer_results: ReviewerLoader
    aggregate_and_render: AggregationRunner

    def __post_init__(self) -> None:
        for name in (
            "preflight",
            "intent",
            "planning",
            "load_review_plan",
            "assemble_and_run_reviewer",
            "persist_reviewer_result",
            "load_reviewer_results",
            "aggregate_and_render",
        ):
            if not callable(getattr(self, name)):
                raise ReviewPipelineV6Error(f"{name} must be callable")


class ReviewPipelineV6:
    def __init__(
        self,
        context: PipelineContextV6,
        services: ReviewPipelineServicesV6,
        *,
        max_reviewer_workers: int | None = None,
    ) -> None:
        if not isinstance(context, PipelineContextV6):
            raise ReviewPipelineV6Error("context must be PipelineContextV6")
        if not isinstance(services, ReviewPipelineServicesV6):
            raise ReviewPipelineV6Error(
                "services must be ReviewPipelineServicesV6"
            )
        if max_reviewer_workers is not None and (
            type(max_reviewer_workers) is not int or max_reviewer_workers <= 0
        ):
            raise ReviewPipelineV6Error(
                "max_reviewer_workers must be positive or None"
            )
        self.context = context
        self.services = services
        self.max_reviewer_workers = max_reviewer_workers

    def run(self) -> SessionV6Manifest:
        store = self.context.session_store
        store.create_or_load()
        fresh_reviewer_inputs: tuple[ReviewAggregationInput, ...] | None = None
        for phase in SESSION_V6_PHASES:
            manifest = store.load()
            checkpoint = manifest.phases[phase.value]
            if checkpoint.status is PhaseStatus.COMPLETED:
                continue
            try:
                store.start_phase(phase)
                if phase is RunPhase.PREFLIGHT:
                    artifacts = tuple(self.services.preflight(self.context))
                elif phase is RunPhase.INTENT:
                    artifacts = tuple(self.services.intent(self.context))
                elif phase is RunPhase.PLANNING:
                    artifacts = tuple(self.services.planning(self.context))
                elif phase is RunPhase.REVIEWERS:
                    plan = self._load_bound_plan()
                    fresh_reviewer_inputs, artifacts = self._run_reviewers(plan)
                elif phase is RunPhase.AGGREGATION:
                    plan = self._load_bound_plan()
                    reviewer_inputs = (
                        fresh_reviewer_inputs
                        if fresh_reviewer_inputs is not None
                        else self._load_bound_reviewer_inputs(plan)
                    )
                    artifacts = tuple(
                        self.services.aggregate_and_render(
                            self.context,
                            plan,
                            reviewer_inputs,
                        )
                    )
                else:
                    raise ReviewPipelineV6Error(
                        "Unsupported Session v6 Phase"
                    )
                store.complete_phase(phase, artifacts)
            except Exception:
                current = store.load()
                if current.phases[phase.value].status is PhaseStatus.RUNNING:
                    store.fail_phase(phase, f"{phase.value}_failed")
                return store.load()
        return store.load()

    def _load_bound_plan(self) -> ReviewPlan:
        plan = self.services.load_review_plan(self.context)
        if type(plan) is not ReviewPlan:
            raise ReviewPipelineV6Error("Planning did not produce a ReviewPlan")
        if plan.snapshot_id != self.context.snapshot.snapshot_id:
            raise ReviewPipelineV6Error("ReviewPlan Snapshot binding changed")
        return plan

    def _run_reviewers(
        self,
        plan: ReviewPlan,
    ) -> tuple[
        tuple[ReviewAggregationInput, ...],
        tuple[SessionV6ArtifactRef, ...],
    ]:
        assignments = plan.assignments
        workers = self.max_reviewer_workers or len(assignments)
        workers = max(1, min(workers, len(assignments)))
        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="reviewer-v6",
        ) as executor:
            futures = [
                executor.submit(
                    self.services.assemble_and_run_reviewer,
                    self.context,
                    assignment,
                )
                for assignment in assignments
            ]
            ordered_inputs: list[ReviewAggregationInput] = []
            artifacts: list[SessionV6ArtifactRef] = []
            for assignment, future in zip(assignments, futures):
                try:
                    item = future.result()
                    if (
                        type(item) is not ReviewAggregationInput
                        or item.execution.assignment_id
                        != assignment.assignment_id
                    ):
                        raise ValueError("Reviewer result binding changed")
                except Exception:
                    item = _failed_reviewer_input(assignment)
                ordered_inputs.append(item)
                artifacts.append(
                    self.services.persist_reviewer_result(
                        self.context,
                        item,
                    )
                )
        return tuple(ordered_inputs), tuple(artifacts)

    def _load_bound_reviewer_inputs(
        self,
        plan: ReviewPlan,
    ) -> tuple[ReviewAggregationInput, ...]:
        values = tuple(self.services.load_reviewer_results(self.context, plan))
        if [item.execution.assignment_id for item in values] != [
            assignment.assignment_id for assignment in plan.assignments
        ]:
            raise ReviewPipelineV6Error(
                "Persisted Reviewer result order or binding changed"
            )
        return values


def publish_reviewer_result_v6(
    context: PipelineContextV6,
    item: ReviewAggregationInput,
) -> SessionV6ArtifactRef:
    payload = {
        "schema_version": REVIEWER_EXECUTION_RECORD_SCHEMA,
        "reviewer_id": item.reviewer_id,
        "execution": reviewer_execution_result_v2_to_dict(item.execution),
    }
    try:
        content = canonical_json_bytes(payload)
    except SafeIOError as error:
        raise ReviewPipelineV6Error(
            "Reviewer execution record is not canonical"
        ) from error
    relative_path = _reviewer_result_path(item.execution.assignment_id)
    descriptor = context.workspace_store.publish_create_only(
        context.snapshot,
        relative_path,
        content,
    )
    return _artifact_ref(
        f"reviewer.result:{item.execution.assignment_id}",
        descriptor,
    )


def load_reviewer_results_v6(
    context: PipelineContextV6,
    plan: ReviewPlan,
) -> tuple[ReviewAggregationInput, ...]:
    values: list[ReviewAggregationInput] = []
    for assignment in plan.assignments:
        relative_path = _reviewer_result_path(assignment.assignment_id)
        descriptor = context.workspace_store.find_snapshot_artifact(
            context.snapshot,
            relative_path,
        )
        content = context.workspace_store.read_verified_artifact(
            context.snapshot,
            descriptor.artifact_id,
        )
        try:
            payload = strict_json_loads(content)
        except SafeIOError as error:
            raise ReviewPipelineV6Error(
                "Reviewer execution record JSON is invalid"
            ) from error
        if type(payload) is not dict or set(payload) != {
            "schema_version",
            "reviewer_id",
            "execution",
        }:
            raise ReviewPipelineV6Error(
                "Reviewer execution record schema is invalid"
            )
        if payload["schema_version"] != REVIEWER_EXECUTION_RECORD_SCHEMA:
            raise ReviewPipelineV6Error(
                "Reviewer execution record version is unsupported"
            )
        execution = reviewer_execution_result_v2_from_dict(payload["execution"])
        if execution.assignment_id != assignment.assignment_id:
            raise ReviewPipelineV6Error(
                "Reviewer execution Assignment binding changed"
            )
        values.append(
            ReviewAggregationInput(
                reviewer_id=payload["reviewer_id"],
                execution=execution,
            )
        )
    return tuple(values)


def aggregate_and_render_v6(
    context: PipelineContextV6,
    plan: ReviewPlan,
    reviewer_inputs: tuple[ReviewAggregationInput, ...],
) -> tuple[SessionV6ArtifactRef, ...]:
    aggregator = DeterministicReviewAggregator()
    bundle = aggregator.publish_or_reuse(
        context.workspace_store,
        context.snapshot,
        context.session.workspace.pr_id,
        plan,
        reviewer_inputs,
    )
    markdown = render_review_result_markdown(bundle.review_result).encode("utf-8")
    markdown_descriptor = context.workspace_store.publish_create_only(
        context.snapshot,
        "Results/review.md",
        markdown,
    )
    aggregation_descriptor = context.workspace_store.find_snapshot_artifact(
        context.snapshot,
        "Results/aggregation.json",
    )
    result_descriptor = context.workspace_store.find_snapshot_artifact(
        context.snapshot,
        "Results/review-result.json",
    )
    return (
        _artifact_ref("aggregation.record", aggregation_descriptor),
        _artifact_ref("aggregation.review_result", result_descriptor),
        _artifact_ref("aggregation.review_markdown", markdown_descriptor),
    )


def _failed_reviewer_input(
    assignment: ReviewerAssignment,
) -> ReviewAggregationInput:
    return ReviewAggregationInput(
        reviewer_id=assignment.assignment_id,
        execution=ReviewerExecutionResultV2(
            assignment_id=assignment.assignment_id,
            status="failed",
            output=None,
            reviewer_output=None,
            rejected_findings=(),
            error_code="reviewer_runtime_error",
            active_elapsed_seconds=0.0,
        ),
    )


def _reviewer_result_path(assignment_id: str) -> str:
    if type(assignment_id) is not str or not assignment_id.startswith("ASG-"):
        raise ReviewPipelineV6Error("assignment_id is invalid")
    return f"Results/reviewers/r-{assignment_id[4:36]}.json"


def _artifact_ref(
    logical_name: str,
    descriptor: ArtifactDescriptor,
) -> SessionV6ArtifactRef:
    return SessionV6ArtifactRef(
        logical_name=logical_name,
        artifact_id=descriptor.artifact_id,
        relative_path=descriptor.relative_path,
        sha256=descriptor.sha256,
    )


__all__ = [
    "PipelineContextV6",
    "REVIEWER_EXECUTION_RECORD_SCHEMA",
    "ReviewPipelineServicesV6",
    "ReviewPipelineV6",
    "ReviewPipelineV6Error",
    "aggregate_and_render_v6",
    "load_reviewer_results_v6",
    "publish_reviewer_result_v6",
]
