from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping

from review_agent.agent_loop import (
    AgentLoopTrace,
    run_reviewer_agent_loop,
)
from review_agent.context import (
    normalize_reviewer_allowed_tools,
    reviewer_tool_scope,
)
from review_agent.model_adapter import ModelAdapter
from review_agent.models import Assignment, IntentPacket
from review_agent.observations import ObservationStore
from review_agent.orchestrator import ReviewerExecution, failed_reviewer_execution
from review_agent.reviewer import run_single_reviewer
from review_agent.supplemental import (
    BudgetAmount,
    SupplementalTaskSpec,
    is_supplemental_assignment,
    stable_invocation_id,
)
from review_agent.tool_gateway import ToolGateway


class ReviewerTaskOrigin(str, Enum):
    INITIAL = "initial"
    SUPPLEMENTAL = "supplemental"


class ReviewerBootstrapPolicy(str, Enum):
    COMPARE_CHANGED_FILES = "compare_changed_files"
    TARGETED_ONLY = "targeted_only"


@dataclass(frozen=True)
class ReviewerTask:
    task_id: str
    reviewer_index: int
    assignment: Assignment
    intent: IntentPacket
    trace_id: str
    origin: ReviewerTaskOrigin
    bootstrap_policy: ReviewerBootstrapPolicy
    allowed_tools: tuple[str, ...]
    changed_files: tuple[str, ...] = ()
    diff_excerpt: tuple[str, ...] = ()
    initial_observations: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _non_empty(self.task_id, "task_id")
        _non_empty(self.trace_id, "trace_id")
        if type(self.reviewer_index) is not int or self.reviewer_index < 0:
            raise ValueError("reviewer_index must be a non-negative integer")
        if not isinstance(self.assignment, Assignment):
            raise ValueError("assignment must be an Assignment")
        if not isinstance(self.intent, IntentPacket):
            raise ValueError("intent must be an IntentPacket")
        if not isinstance(self.origin, ReviewerTaskOrigin):
            raise ValueError("origin must be a ReviewerTaskOrigin")
        if not isinstance(self.bootstrap_policy, ReviewerBootstrapPolicy):
            raise ValueError(
                "bootstrap_policy must be a ReviewerBootstrapPolicy"
            )
        allowed_tools = normalize_reviewer_allowed_tools(self.allowed_tools)
        changed_files = _string_tuple(self.changed_files, "changed_files")
        diff_excerpt = _string_tuple(
            self.diff_excerpt,
            "diff_excerpt",
            allow_empty_items=False,
        )
        observations = dict(self.initial_observations)
        if any(
            not isinstance(key, str)
            or not key.strip()
            or not isinstance(value, str)
            for key, value in observations.items()
        ):
            raise ValueError(
                "initial_observations must map non-empty IDs to strings"
            )
        if self.origin is ReviewerTaskOrigin.SUPPLEMENTAL:
            if not is_supplemental_assignment(self.assignment):
                raise ValueError(
                    "supplemental tasks require an isolated supplemental Assignment"
                )
            if self.bootstrap_policy is not ReviewerBootstrapPolicy.TARGETED_ONLY:
                raise ValueError(
                    "supplemental tasks require targeted_only bootstrap"
                )
            if changed_files:
                raise ValueError(
                    "supplemental tasks cannot prefetch changed-file diffs"
                )
        else:
            if is_supplemental_assignment(self.assignment):
                raise ValueError(
                    "semantic_reconciler Assignment cannot be marked initial"
                )
            if (
                self.bootstrap_policy
                is ReviewerBootstrapPolicy.COMPARE_CHANGED_FILES
                and changed_files
                and "compare_base_head" not in allowed_tools
            ):
                raise ValueError(
                    "compare_changed_files bootstrap requires compare_base_head"
                )
        object.__setattr__(self, "allowed_tools", allowed_tools)
        object.__setattr__(self, "changed_files", tuple(sorted(set(changed_files))))
        object.__setattr__(self, "diff_excerpt", diff_excerpt)
        object.__setattr__(self, "initial_observations", observations)

    @classmethod
    def for_supplemental(
        cls,
        spec: SupplementalTaskSpec,
        *,
        reviewer_index: int,
        intent: IntentPacket,
        initial_observations: Mapping[str, str],
        trace_id: str | None = None,
        diff_excerpt: tuple[str, ...] = (),
    ) -> ReviewerTask:
        if not isinstance(spec, SupplementalTaskSpec):
            raise ValueError("spec must be a SupplementalTaskSpec")
        return cls(
            task_id=spec.task_id,
            reviewer_index=reviewer_index,
            assignment=spec.assignment,
            intent=intent,
            trace_id=(
                trace_id
                or stable_invocation_id(
                    task_or_batch_id=spec.task_id,
                    logical_turn=0,
                    request_digest=spec.request_id,
                )
            ),
            origin=ReviewerTaskOrigin.SUPPLEMENTAL,
            bootstrap_policy=ReviewerBootstrapPolicy.TARGETED_ONLY,
            allowed_tools=spec.allowed_tools,
            changed_files=(),
            diff_excerpt=diff_excerpt,
            initial_observations=initial_observations,
        )

    @classmethod
    def for_initial(
        cls,
        *,
        task_id: str,
        reviewer_index: int,
        assignment: Assignment,
        intent: IntentPacket,
        trace_id: str,
        changed_files: tuple[str, ...],
        initial_observations: Mapping[str, str],
        allowed_tools: tuple[str, ...],
        diff_excerpt: tuple[str, ...] = (),
    ) -> ReviewerTask:
        return cls(
            task_id=task_id,
            reviewer_index=reviewer_index,
            assignment=assignment,
            intent=intent,
            trace_id=trace_id,
            origin=ReviewerTaskOrigin.INITIAL,
            bootstrap_policy=ReviewerBootstrapPolicy.COMPARE_CHANGED_FILES,
            allowed_tools=allowed_tools,
            changed_files=changed_files,
            diff_excerpt=diff_excerpt,
            initial_observations=initial_observations,
        )

    @property
    def counts_toward_initial_coverage(self) -> bool:
        return self.origin is ReviewerTaskOrigin.INITIAL


@dataclass(frozen=True)
class ReviewerTaskRun:
    task: ReviewerTask
    execution: ReviewerExecution
    observation_summaries: Mapping[str, str]
    gateway_attempted_tool_calls: int
    gateway_denied_tool_calls: int
    elapsed_seconds: float
    loop_trace: AgentLoopTrace | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.task, ReviewerTask):
            raise ValueError("task must be a ReviewerTask")
        if not isinstance(self.execution, ReviewerExecution):
            raise ValueError("execution must be a ReviewerExecution")
        for name, value in {
            "gateway_attempted_tool_calls": self.gateway_attempted_tool_calls,
            "gateway_denied_tool_calls": self.gateway_denied_tool_calls,
        }.items():
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.gateway_denied_tool_calls > self.gateway_attempted_tool_calls:
            raise ValueError("denied tool calls cannot exceed attempted tool calls")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")
        object.__setattr__(self, "observation_summaries", dict(self.observation_summaries))

    @property
    def counts_toward_initial_coverage(self) -> bool:
        return self.task.counts_toward_initial_coverage

    @property
    def budget_consumption(self) -> BudgetAmount:
        return BudgetAmount(
            tasks=1,
            tool_calls=self.gateway_attempted_tool_calls,
            tokens=self.execution.runtime.total_tokens,
            elapsed_seconds=self.elapsed_seconds,
        )

    @property
    def usage_available(self) -> bool:
        return self.execution.runtime.usage_available


GatewayFactory = Callable[..., ToolGateway]


class ReviewerTaskExecutor:
    """Execute one initial or supplemental reviewer task; callers own commit order."""

    def __init__(
        self,
        *,
        repository_path: Path,
        base_revision: str,
        head_revision: str,
        reviewer_loop: str,
        model: str = "configured-reviewer-model",
        gateway_factory: GatewayFactory = ToolGateway,
    ) -> None:
        self.repository_path = Path(repository_path)
        self.base_revision = _non_empty(base_revision, "base_revision")
        self.head_revision = _non_empty(head_revision, "head_revision")
        if reviewer_loop not in {"single-shot", "agent-loop"}:
            raise ValueError("reviewer_loop must be single-shot or agent-loop")
        self.reviewer_loop = reviewer_loop
        self.model = _non_empty(model, "model")
        if not callable(gateway_factory):
            raise ValueError("gateway_factory must be callable")
        self.gateway_factory = gateway_factory

    def execute(
        self,
        task: ReviewerTask,
        *,
        adapter: ModelAdapter | None,
        observation_store: ObservationStore,
        creation_error: Exception | None = None,
    ) -> ReviewerTaskRun:
        if not isinstance(task, ReviewerTask):
            raise ValueError("task must be a ReviewerTask")
        if not isinstance(observation_store, ObservationStore):
            raise ValueError("observation_store must be an ObservationStore")
        if creation_error is not None and not isinstance(creation_error, Exception):
            raise ValueError("creation_error must be an Exception")
        started_at = perf_counter()
        gateway = self.gateway_factory(
            repository_path=self.repository_path,
            base_revision=self.base_revision,
            head_revision=self.head_revision,
            observation_store=observation_store,
            allowed_tools=task.allowed_tools,
        )
        reviewer_observations = dict(task.initial_observations)
        loop_trace: AgentLoopTrace | None = None
        try:
            if (
                task.bootstrap_policy
                is ReviewerBootstrapPolicy.COMPARE_CHANGED_FILES
            ):
                for changed_file in task.changed_files:
                    gateway.execute(
                        "compare_base_head",
                        {"path": changed_file},
                    )
            # targeted_only deliberately performs no automatic repository read.
            reviewer_observations.update(observation_store.summaries_by_id())
            if creation_error is not None:
                raise creation_error
            if adapter is None:
                raise RuntimeError("reviewer adapter creation returned no adapter")
            with reviewer_tool_scope(task.allowed_tools):
                if self.reviewer_loop == "agent-loop":
                    loop_run = run_reviewer_agent_loop(
                        adapter=adapter,
                        gateway=gateway,
                        assignment=task.assignment,
                        intent=task.intent,
                        diff_excerpt=list(task.diff_excerpt),
                        observations=reviewer_observations,
                        trace_id=task.trace_id,
                        model=self.model,
                    )
                    loop_trace = loop_run.trace
                    execution = ReviewerExecution(
                        reviewer_index=task.reviewer_index,
                        trace_id=task.trace_id,
                        assignment=task.assignment,
                        envelope=loop_run.envelope,
                        response=loop_run.response,
                        result=loop_run.result,
                        runtime=loop_run.runtime,
                    )
                else:
                    reviewer_run = run_single_reviewer(
                        adapter=adapter,
                        assignment=task.assignment,
                        intent=task.intent,
                        diff_excerpt=list(task.diff_excerpt),
                        observations=reviewer_observations,
                        trace_id=task.trace_id,
                        model=self.model,
                    )
                    execution = ReviewerExecution(
                        reviewer_index=task.reviewer_index,
                        trace_id=task.trace_id,
                        assignment=task.assignment,
                        envelope=reviewer_run.envelope,
                        response=reviewer_run.response,
                        result=reviewer_run.result,
                        runtime=reviewer_run.runtime,
                    )
        except Exception as error:
            reviewer_observations = dict(task.initial_observations)
            reviewer_observations.update(observation_store.summaries_by_id())
            with reviewer_tool_scope(task.allowed_tools):
                execution = failed_reviewer_execution(
                    index=task.reviewer_index,
                    trace_id=task.trace_id,
                    assignment=task.assignment,
                    intent=task.intent,
                    diff_excerpt=list(task.diff_excerpt),
                    observations=reviewer_observations,
                    error=error,
                    model=self.model,
                    elapsed_seconds=perf_counter() - started_at,
                    retained_observation_refs=tuple(
                        sorted(reviewer_observations)
                    ),
                )
        elapsed_seconds = max(0.0, perf_counter() - started_at)
        return ReviewerTaskRun(
            task=task,
            execution=execution,
            observation_summaries=observation_store.summaries_by_id(),
            gateway_attempted_tool_calls=gateway.attempted_tool_calls,
            gateway_denied_tool_calls=gateway.denied_tool_calls,
            elapsed_seconds=elapsed_seconds,
            loop_trace=loop_trace,
        )


def _non_empty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _string_tuple(
    values: Any,
    name: str,
    *,
    allow_empty_items: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an iterable of strings")
    try:
        rows = tuple(values)
    except TypeError as error:
        raise ValueError(f"{name} must be an iterable of strings") from error
    if any(not isinstance(value, str) for value in rows):
        raise ValueError(f"{name} must contain strings")
    if not allow_empty_items and any(not value.strip() for value in rows):
        raise ValueError(f"{name} must contain non-empty strings")
    return rows
