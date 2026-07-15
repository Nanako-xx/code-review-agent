import json
from dataclasses import replace

from conftest import run_git
from review_agent.model_adapter import FakeToolCallingAdapter
from review_agent.context import remote_visible_memory_snapshot
from review_agent.model_protocol import (
    ModelResponseKind,
    ModelToolCall,
    ModelTurnResponse,
)
from review_agent.models import IntentPacket, IntentStatus, RiskLevel
from review_agent.memory_models import MemoryScope
from review_agent.memory_retrieval import SnapshotMemoryQueryService
from review_agent.observations import ObservationStore
from review_agent.reviewer_task_executor import (
    ReviewerTask,
    ReviewerTaskExecutor,
    ReviewerTaskOrigin,
)
from review_agent.supplemental import (
    SupplementalInvestigationRequest,
    compile_supplemental_plan,
)
from tests.test_context import _memory_snapshot


def _partial_response(summary: str = "Targeted investigation completed.") -> ModelTurnResponse:
    return ModelTurnResponse(
        kind=ModelResponseKind.FINAL,
        final_text=json.dumps(
            {
                "contract_assessments": [],
                "confirmed_findings": [],
                "rejected_hypotheses": [],
                "uncertainties": ["The disagreement remains unresolved."],
                "observation_refs": [],
                "investigation_summary": summary,
                "status": "partial",
            }
        ),
        raw={"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
    )


def _supplemental_spec(*, allowed_tools=("read_range",)):
    request = SupplementalInvestigationRequest(
        source_disagreement_id="D-retry",
        question="Does the retry path duplicate jobs?",
        required_evidence=("inspect retry caller",),
        preferred_perspective="concurrency",
        source_candidate_ids=("F-1", "F-2"),
        reason_refs=(),
    )
    plan = compile_supplemental_plan(
        review_id="review-1",
        base_sha="a" * 40,
        head_sha="b" * 40,
        risk_level=RiskLevel.LOW,
        wave_index=1,
        trigger_digest="trigger-a",
        requests=[request],
        allowed_tools=allowed_tools,
    )
    return plan.tasks[0]


def test_supplemental_executor_filters_envelope_denies_illegal_call_and_skips_diff_bootstrap(
    git_repo,
    tmp_path,
):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "app.py").write_text(
        "def add(a, b):\n    return a - b\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "change app")
    head = run_git(git_repo, "rev-parse", "HEAD")
    store = ObservationStore(tmp_path / "supplemental-attempt")
    adapter = FakeToolCallingAdapter(
        [
            ModelTurnResponse(
                kind=ModelResponseKind.TOOL_CALLS,
                tool_calls=[
                    ModelToolCall(
                        "call-denied",
                        "compare_base_head",
                        {"path": "app.py"},
                    )
                ],
            ),
            _partial_response(),
        ]
    )
    spec = _supplemental_spec(allowed_tools=("read_range",))
    task = ReviewerTask.for_supplemental(
        spec,
        reviewer_index=0,
        intent=IntentPacket(goal="Review retry behavior", status=IntentStatus.PARTIAL),
        initial_observations={},
    )
    executor = ReviewerTaskExecutor(
        repository_path=git_repo,
        base_revision=base,
        head_revision=head,
        reviewer_loop="agent-loop",
        model="fake-reviewer",
    )

    run = executor.execute(task, adapter=adapter, observation_store=store)

    assert run.task.origin is ReviewerTaskOrigin.SUPPLEMENTAL
    assert run.counts_toward_initial_coverage is False
    assert [tool.name for tool in adapter.requests[0].tools] == ["read_range"]
    assert [tool["name"] for tool in run.execution.envelope.tools] == ["read_range"]
    assert run.execution.runtime.tool_calls == 1
    assert run.gateway_attempted_tool_calls == 1
    assert run.gateway_denied_tool_calls == 1
    assert store.list_observations() == []
    assert run.budget_consumption.tasks == 1
    assert run.budget_consumption.tool_calls == 1
    assert run.budget_consumption.tokens == 15


def test_initial_executor_can_use_explicit_changed_file_bootstrap(
    git_repo,
    tmp_path,
):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "app.py").write_text(
        "def add(a, b):\n    return a - b\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "change app")
    head = run_git(git_repo, "rev-parse", "HEAD")
    spec = _supplemental_spec(allowed_tools=("compare_base_head",))
    # The executor is shared: an initial task can opt into the historical
    # changed-file compare bootstrap, while supplemental construction cannot.
    task = ReviewerTask.for_initial(
        task_id="reviewer-0",
        reviewer_index=0,
        assignment=replace(
            spec.assignment,
            assigned_contract=["regression_safety"],
            assignment_id="assignment-initial",
            perspective_key="initial-specialist",
            planner_source="local",
        ),
        intent=IntentPacket(goal="Review app.py", status=IntentStatus.PARTIAL),
        trace_id="review-1-reviewer-0",
        changed_files=("app.py",),
        initial_observations={},
        allowed_tools=("compare_base_head",),
    )
    adapter = FakeToolCallingAdapter([_partial_response("Initial review completed.")])
    store = ObservationStore(tmp_path / "initial-attempt")
    executor = ReviewerTaskExecutor(
        repository_path=git_repo,
        base_revision=base,
        head_revision=head,
        reviewer_loop="agent-loop",
    )

    run = executor.execute(task, adapter=adapter, observation_store=store)

    assert run.task.origin is ReviewerTaskOrigin.INITIAL
    assert run.counts_toward_initial_coverage is True
    assert run.gateway_attempted_tool_calls == 1
    assert run.gateway_denied_tool_calls == 0
    assert len(store.list_observations()) == 1
    assert "Observation Summary" in adapter.requests[0].messages[0]["content"]


def test_reviewer_executor_passes_fixed_snapshot_query_service_and_context(
    git_repo,
    tmp_path,
):
    head = run_git(git_repo, "rev-parse", "HEAD")
    snapshot = _memory_snapshot(head=head)
    service = SnapshotMemoryQueryService(
        snapshot,
        assignment_id="assignment-memory",
        assignment_scope=MemoryScope(paths=("app.py",)),
    )
    spec = _supplemental_spec(allowed_tools=("query_project_memory",))
    task = ReviewerTask.for_supplemental(
        spec,
        reviewer_index=0,
        intent=IntentPacket(goal="Review app.py", status=IntentStatus.PARTIAL),
        initial_observations={},
        memory_snapshot=snapshot,
        memory_query_service=service,
    )
    task = replace(
        task,
        assignment=replace(task.assignment, assignment_id="assignment-memory"),
    )
    adapter = FakeToolCallingAdapter([_partial_response("Memory context completed.")])
    store = ObservationStore(tmp_path / "memory-executor")
    executor = ReviewerTaskExecutor(
        repository_path=git_repo,
        base_revision=head,
        head_revision="HEAD",
        reviewer_loop="agent-loop",
    )

    run = executor.execute(task, adapter=adapter, observation_store=store)

    projection = remote_visible_memory_snapshot(snapshot)
    assert run.execution.envelope.parameters["context"]["snapshot_id"] == projection.snapshot_id
    assert "Approved Project Memory" in adapter.requests[0].messages[0]["content"]
    assert "query_project_memory" in [tool.name for tool in adapter.requests[0].tools]
    memory_tool = next(
        tool for tool in adapter.requests[0].tools
        if tool.name == "query_project_memory"
    )
    assert memory_tool.parameters_schema["properties"]["assignment_id"]["const"] == (
        "assignment-memory"
    )


def test_executor_safely_rebuilds_preconstructed_query_service_assignment_scope(
    git_repo,
):
    head = run_git(git_repo, "rev-parse", "HEAD")
    snapshot = _memory_snapshot(head=head)
    stale_service = SnapshotMemoryQueryService(
        snapshot,
        assignment_id="assignment-memory",
        assignment_scope=MemoryScope(paths=("private.py",)),
    )
    spec = _supplemental_spec(allowed_tools=("query_project_memory",))
    assignment = replace(
        spec.assignment,
        assignment_id="assignment-memory",
        initial_context=replace(
            spec.assignment.initial_context,
            changed_files=["app.py"],
        ),
    )
    task = ReviewerTask.for_supplemental(
        spec,
        reviewer_index=0,
        intent=IntentPacket(goal="Review app.py", status=IntentStatus.PARTIAL),
        initial_observations={},
        memory_snapshot=snapshot,
        memory_query_service=stale_service,
    )
    task = replace(task, assignment=assignment)
    executor = ReviewerTaskExecutor(
        repository_path=git_repo,
        base_revision=head,
        head_revision="HEAD",
        reviewer_loop="agent-loop",
    )

    context, rebuilt = executor._memory_context_for_task(task)

    expected_scope = MemoryScope(
        paths=("app.py",),
        contracts=tuple(assignment.assigned_contract),
    )
    assert rebuilt is not stale_service
    assert rebuilt._assignment_id == "assignment-memory"
    assert rebuilt._assignment_scope == expected_scope
    assert rebuilt._snapshot.snapshot_id == snapshot.snapshot_id
    assert context.query_service is rebuilt
    assert stale_service.call_count == 0
