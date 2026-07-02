import json

from review_agent.agent_loop import agent_loop_run_to_dict, run_reviewer_agent_loop
from review_agent.model_adapter import FakeToolCallingAdapter
from review_agent.model_protocol import ModelResponseKind, ModelToolCall, ModelTurnResponse
from review_agent.models import IntentPacket, IntentSource, IntentStatus
from review_agent.observations import ObservationStore
from review_agent.tool_gateway import ToolGateway
from tests.conftest import run_git
from tests.test_orchestrator import make_assignment


def make_intent():
    return IntentPacket(
        goal="Review risky change",
        sources={"goal": IntentSource.EXPLICIT},
        status=IntentStatus.SUFFICIENT,
    )


def final_response(request):
    observation_id = request.tool_results[-1].observation_ids[0]
    return ModelTurnResponse(
        kind=ModelResponseKind.FINAL,
        final_text=json.dumps(
            {
                "contract_assessments": [
                    {
                        "contract": "regression_safety",
                        "status": "covered",
                        "summary": "Compared base and head.",
                        "evidence_refs": [observation_id],
                    }
                ],
                "confirmed_findings": [],
                "rejected_hypotheses": [],
                "uncertainties": [],
                "observation_refs": [observation_id],
                "investigation_summary": "Reviewed with a tool observation.",
                "status": "completed",
            }
        ),
    )


def test_agent_loop_executes_tool_call_and_returns_final_result(git_repo):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "app.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "change app")
    head = run_git(git_repo, "rev-parse", "HEAD")
    run_dir = git_repo / ".review-agent" / "runs" / "review-1"
    observation_store = ObservationStore(run_dir)
    gateway = ToolGateway(git_repo, base, head, observation_store)
    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.TOOL_CALLS,
                tool_calls=[ModelToolCall("call-1", "compare_base_head", {"path": "app.py"})],
            ),
            final_response,
        ]
    )

    run = run_reviewer_agent_loop(
        adapter=adapter,
        gateway=gateway,
        assignment=make_assignment("Core Reviewer"),
        intent=make_intent(),
        diff_excerpt=["diff excerpt"],
        observations={},
        trace_id="review-1-reviewer-0",
    )

    assert run.result.status.value == "completed"
    assert run.result.investigation_summary == "Reviewed with a tool observation."
    assert run.trace.tool_call_count == 1
    assert run.trace.turns[0].tool_calls[0].tool_name == "compare_base_head"
    assert run.trace.turns[0].tool_results[0].observation_ids
    assert adapter.requests[1].tool_results[0].content
    assert list(observation_store.summaries_by_id())


def test_agent_loop_returns_partial_when_tool_budget_is_exhausted(git_repo):
    base = run_git(git_repo, "rev-parse", "HEAD")
    head = base
    observation_store = ObservationStore(git_repo / ".review-agent" / "runs" / "review-budget")
    gateway = ToolGateway(git_repo, base, head, observation_store)
    assignment = make_assignment("Core Reviewer")
    assignment = type(assignment)(
        role=assignment.role,
        mission=assignment.mission,
        assignment_reason=assignment.assignment_reason,
        assigned_contract=assignment.assigned_contract,
        required_checks=assignment.required_checks,
        initial_context=assignment.initial_context,
        max_turns=assignment.max_turns,
        max_tool_calls=0,
    )
    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.TOOL_CALLS,
                tool_calls=[ModelToolCall("call-1", "compare_base_head", {"path": "app.py"})],
            )
        ]
    )

    run = run_reviewer_agent_loop(
        adapter=adapter,
        gateway=gateway,
        assignment=assignment,
        intent=make_intent(),
        diff_excerpt=[],
        observations={},
        trace_id="review-budget-reviewer-0",
    )

    assert run.result.status.value == "partial"
    assert "tool budget exhausted" in run.result.uncertainties
    assert run.trace.final_status == "partial"
