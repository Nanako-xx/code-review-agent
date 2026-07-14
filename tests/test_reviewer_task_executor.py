import json
from dataclasses import replace

from conftest import run_git
from review_agent.model_adapter import FakeToolCallingAdapter
from review_agent.model_protocol import (
    ModelResponseKind,
    ModelToolCall,
    ModelTurnResponse,
)
from review_agent.models import IntentPacket, IntentStatus, RiskLevel
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
