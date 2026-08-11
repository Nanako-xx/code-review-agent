from __future__ import annotations

import json
from pathlib import Path
import threading

from review_agent.aggregation import ReviewAggregationInput
from review_agent.pr_workspace import PRMetadata, PRWorkspaceStore
from review_agent.review_pipeline import (
    PipelineContextV6,
    ReviewPipelineServicesV6,
    ReviewPipelineV6,
    aggregate_and_render_v6,
    load_reviewer_results_v6,
    publish_reviewer_result_v6,
)
from review_agent.review_planning import compile_review_plan
from review_agent.review_protocol import (
    FindingSeverity,
    ReviewerFinding,
    ReviewerOutput,
    ReviewResultStatus,
    RiskLevel,
)
from review_agent.reviewer_executor import ReviewerExecutionResultV2
from review_agent.revision import RepositoryIdentity
from review_agent.run_state import RunPhase, RunStatus
from review_agent.session import PhaseStatus, SessionV6ArtifactRef
from review_agent.session_store import SessionV6Store


def _context(tmp_path: Path):
    repository = tmp_path / "repo"
    git_common = repository / ".git"
    git_common.mkdir(parents=True)
    identity = RepositoryIdentity(
        canonical_path=str(repository.resolve()),
        git_common_dir=str(git_common.resolve()),
        origin_url=None,
    )
    workspace_store = PRWorkspaceStore(tmp_path / "ra")
    workspace = workspace_store.create_or_load_workspace(
        workspace_store.resolve_pr(identity, "local", "pipeline-v6"),
        PRMetadata(title="Pipeline v6"),
    )
    snapshot = workspace_store.create_or_load_snapshot(
        workspace, "a" * 40, "b" * 40
    )
    session = workspace_store.create_session(workspace, snapshot)
    state_store = SessionV6Store(workspace_store, session)
    return PipelineContextV6(
        workspace_store=workspace_store,
        snapshot=snapshot,
        session=session,
        session_store=state_store,
    )


class _Services:
    def __init__(self, context: PipelineContextV6, *, fail_intent=False):
        self.context = context
        self.plan = compile_review_plan(
            snapshot_id=context.snapshot.snapshot_id,
            risk_level=RiskLevel.MEDIUM,
            allowed_files=("src/cache.py",),
            allowed_symbols=(),
            allowed_hunks=(),
        )
        self.events = []
        self.persist_order = []
        self.persisted = {}
        self.fail_intent = fail_intent
        self.barrier = threading.Barrier(len(self.plan.assignments))
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def callbacks(self) -> ReviewPipelineServicesV6:
        return ReviewPipelineServicesV6(
            preflight=self.preflight,
            intent=self.intent,
            planning=self.planning,
            load_review_plan=lambda _context: self.plan,
            assemble_and_run_reviewer=self.run_reviewer,
            persist_reviewer_result=self.persist,
            load_reviewer_results=load_reviewer_results_v6,
            aggregate_and_render=self.aggregate,
        )

    def preflight(self, _context):
        self.events.append("preflight")
        return (
            self._publish(
                "preflight.diff_patch",
                "DiffArtifact/diff.patch",
                b"diff",
            ),
            self._publish(
                "preflight.diff_index",
                "DiffArtifact/index.json",
                b"{}",
            ),
            self._publish(
                "preflight.quality_gate",
                "QualityGate/quality-gate.json",
                b"{}",
            ),
            self._publish(
                "preflight.changed_symbols",
                "ChangedSymbols/changed-symbols.json",
                b"[]",
            ),
        )

    def intent(self, _context):
        self.events.append("intent")
        if self.fail_intent:
            raise RuntimeError("private intent failure")
        return (
            self._publish(
                "intent.packet",
                "Intent/current.json",
                b'{"goal":null,"source":null,"uncertainties":[]}',
            ),
        )

    def planning(self, _context):
        self.events.append("planning")
        refs = [
            self._publish("planning.risk", "Risk/risk.json", b'{"level":"medium"}'),
            self._publish(
                "planning.review_plan",
                "ReviewPlan/plan.json",
                self.plan.to_json_bytes(),
            ),
        ]
        for index, assignment in enumerate(self.plan.assignments):
            refs.append(
                self._publish(
                    f"planning.assignment:{assignment.assignment_id}",
                    f"ReviewPlan/Assignments/assignment-{index}.json",
                    assignment.to_json_bytes(),
                )
            )
        return tuple(refs)

    def run_reviewer(self, _context, assignment):
        self.events.append(f"run:{assignment.assignment_id}")
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.barrier.wait(timeout=5)
        with self.lock:
            self.active -= 1
        if assignment is self.plan.assignments[0]:
            execution = ReviewerExecutionResultV2(
                assignment_id=assignment.assignment_id,
                status="failed",
                output=None,
                reviewer_output=None,
                rejected_findings=(),
                error_code="reviewer_runtime_error",
                active_elapsed_seconds=1.0,
            )
        else:
            output = ReviewerOutput(
                findings=(
                    ReviewerFinding(
                        claim=(
                            "When the cache is empty, dereferencing it raises "
                            "and returns 500."
                        ),
                        severity=FindingSeverity.HIGH,
                        path="src/cache.py",
                        line=10,
                        suggestion=(
                            "Guard the missing value and add a cold-cache test."
                        ),
                    ),
                ),
                uncertainties=(),
            )
            execution = ReviewerExecutionResultV2(
                assignment_id=assignment.assignment_id,
                status="completed",
                output=output.to_json(),
                reviewer_output=output,
                rejected_findings=(),
                error_code=None,
                active_elapsed_seconds=2.0,
            )
        return ReviewAggregationInput(
            reviewer_id=f"reviewer-{assignment.assignment_id[4:12]}",
            execution=execution,
        )

    def persist(self, context, item):
        self.persist_order.append(item.execution.assignment_id)
        self.persisted[item.execution.assignment_id] = item
        return publish_reviewer_result_v6(context, item)

    def aggregate(self, context, plan, inputs):
        self.events.append("aggregation")
        assert list(self.persisted) == [
            assignment.assignment_id for assignment in plan.assignments
        ]
        assert [item.execution.status for item in inputs] == ["failed", "completed"]
        return aggregate_and_render_v6(context, plan, inputs)

    def _publish(self, logical_name, relative_path, content):
        descriptor = self.context.workspace_store.publish_create_only(
            self.context.snapshot,
            relative_path,
            content,
        )
        return SessionV6ArtifactRef(
            logical_name=logical_name,
            artifact_id=descriptor.artifact_id,
            relative_path=descriptor.relative_path,
            sha256=descriptor.sha256,
        )


def test_pipeline_runs_five_phases_and_isolates_parallel_reviewer_failure(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    services = _Services(context)
    pipeline = ReviewPipelineV6(context, services.callbacks())

    manifest = pipeline.run()

    assert manifest.status is RunStatus.COMPLETED
    assert manifest.current_phase is RunPhase.COMPLETED
    assert tuple(manifest.phases) == (
        "preflight",
        "intent",
        "planning",
        "reviewers",
        "aggregation",
    )
    assert services.max_active == 2
    assert services.persist_order == [
        assignment.assignment_id for assignment in services.plan.assignments
    ]
    assert [
        item.execution.status
        for item in load_reviewer_results_v6(context, services.plan)
    ] == ["failed", "completed"]
    assert services.events[-1] == "aggregation"
    result_payload = json.loads(
        (context.snapshot.path / "Results" / "review-result.json").read_text("utf-8")
    )
    assert result_payload["status"] == ReviewResultStatus.PARTIAL.value
    assert len(result_payload["findings"]) == 1
    assert (context.snapshot.path / "Results" / "review.md").is_file()


def test_completed_pipeline_reuses_all_phases_without_callbacks(tmp_path: Path) -> None:
    context = _context(tmp_path)
    services = _Services(context)
    pipeline = ReviewPipelineV6(context, services.callbacks())
    first = pipeline.run()
    events = list(services.events)

    second = pipeline.run()

    assert first.status is second.status is RunStatus.COMPLETED
    assert services.events == events


def test_running_phase_is_restarted_by_dispatcher(tmp_path: Path) -> None:
    context = _context(tmp_path)
    services = _Services(context)
    context.session_store.create()
    context.session_store.start_phase(RunPhase.PREFLIGHT)

    manifest = ReviewPipelineV6(context, services.callbacks()).run()

    assert manifest.status is RunStatus.COMPLETED
    assert manifest.phases["preflight"].attempt == 2


def test_control_layer_failure_stops_successors_with_sanitized_error(tmp_path: Path) -> None:
    context = _context(tmp_path)
    services = _Services(context, fail_intent=True)

    manifest = ReviewPipelineV6(context, services.callbacks()).run()

    assert manifest.status is RunStatus.FAILED
    assert manifest.current_phase is RunPhase.INTENT
    assert manifest.phases["intent"].error_code == "intent_failed"
    assert manifest.phases["planning"].status is PhaseStatus.PENDING
    assert "planning" not in services.events
    assert "private intent failure" not in str(manifest.to_dict())


def test_aggregation_retry_reuses_completed_reviewers_and_loads_results(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    services = _Services(context)
    real_aggregate = services.aggregate
    attempts = 0

    def fail_once(*args):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("private aggregation failure")
        return real_aggregate(*args)

    services.aggregate = fail_once
    pipeline = ReviewPipelineV6(context, services.callbacks())
    failed = pipeline.run()
    reviewer_events = [item for item in services.events if item.startswith("run:")]

    resumed = pipeline.run()

    assert failed.status is RunStatus.FAILED
    assert failed.current_phase is RunPhase.AGGREGATION
    assert resumed.status is RunStatus.COMPLETED
    assert [item for item in services.events if item.startswith("run:")] == reviewer_events
    assert resumed.phases["aggregation"].attempt == 2
