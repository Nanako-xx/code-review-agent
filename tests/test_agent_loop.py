import json
from dataclasses import replace

from review_agent.agent_loop import agent_loop_run_to_dict, run_reviewer_agent_loop
from review_agent.model_adapter import FakeToolCallingAdapter
from review_agent.model_protocol import ModelResponseKind, ModelToolCall, ModelTurnResponse
from review_agent.models import (
    IntentPacket,
    IntentSource,
    IntentStatus,
    ReviewerTerminationReason,
)
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


def final_response_after_tool_error(request):
    assert request.tool_results[-1].is_error
    return ModelTurnResponse(
        kind=ModelResponseKind.FINAL,
        final_text=json.dumps(
            {
                "contract_assessments": [],
                "confirmed_findings": [],
                "rejected_hypotheses": [],
                "uncertainties": ["tool failed"],
                "observation_refs": [],
                "investigation_summary": "Handled tool error.",
                "status": "partial",
            }
        ),
        raw={"turn": "final-after-tool-error"},
        model="final-model",
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
    assignment = replace(make_assignment("Core Reviewer"), max_tool_calls=0)
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


def test_agent_loop_rejects_incomplete_completion_and_accepts_correction(
    git_repo,
):
    base = run_git(git_repo, "rev-parse", "HEAD")
    observation_store = ObservationStore(
        git_repo / ".review-agent" / "runs" / "review-contract-retry"
    )
    gateway = ToolGateway(git_repo, base, base, observation_store)
    incomplete = ModelTurnResponse(
        kind=ModelResponseKind.FINAL,
        final_text=json.dumps(
            {
                "contract_assessments": [],
                "confirmed_findings": [],
                "rejected_hypotheses": [],
                "uncertainties": [],
                "observation_refs": [],
                "investigation_summary": "Requested completion too early.",
                "status": "completed",
            }
        ),
    )
    corrected = ModelTurnResponse(
        kind=ModelResponseKind.FINAL,
        final_text=json.dumps(
            {
                "contract_assessments": [
                    {
                        "contract": "regression_safety",
                        "status": "not_applicable",
                        "summary": "No changed behavior exists in this range.",
                        "evidence_refs": [],
                    }
                ],
                "confirmed_findings": [],
                "rejected_hypotheses": [],
                "uncertainties": [],
                "observation_refs": [],
                "investigation_summary": "Closed the assigned contract.",
                "status": "completed",
            }
        ),
    )
    adapter = FakeToolCallingAdapter(script=[incomplete, corrected])

    run = run_reviewer_agent_loop(
        adapter=adapter,
        gateway=gateway,
        assignment=make_assignment("Core Reviewer"),
        intent=make_intent(),
        diff_excerpt=[],
        observations={},
        trace_id="review-contract-retry-reviewer-0",
    )

    assert run.result.status.value == "completed"
    assert len(adapter.requests) == 2
    assert "Runtime rejected completion" in adapter.requests[1].messages[-1]["content"]
    assert "missing contract assessment" in run.trace.turns[0].error


def test_agent_loop_downgrades_invalid_completion_when_budget_ends(git_repo):
    base = run_git(git_repo, "rev-parse", "HEAD")
    observation_store = ObservationStore(
        git_repo / ".review-agent" / "runs" / "review-contract-budget"
    )
    gateway = ToolGateway(git_repo, base, base, observation_store)
    assignment = replace(make_assignment("Core Reviewer"), max_turns=1)
    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text=json.dumps(
                    {
                        "contract_assessments": [],
                        "confirmed_findings": [],
                        "rejected_hypotheses": [],
                        "uncertainties": [],
                        "observation_refs": [],
                        "investigation_summary": "Requested completion too early.",
                        "status": "completed",
                    }
                ),
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
        trace_id="review-contract-budget-reviewer-0",
    )

    assert run.result.status.value == "partial"
    assert any(
        "Runtime rejected reviewer completion" in item
        for item in run.result.uncertainties
    )
    assert run.result.uncertainties[-1] == "turn budget exhausted"
    assert run.runtime.termination_reason is ReviewerTerminationReason.TURN_BUDGET_EXHAUSTED
    assert run.trace.final_status == "partial"


def test_agent_loop_converts_gateway_argument_error_to_error_tool_result(git_repo):
    base = run_git(git_repo, "rev-parse", "HEAD")
    observation_store = ObservationStore(git_repo / ".review-agent" / "runs" / "review-tool-error")
    gateway = ToolGateway(git_repo, base, base, observation_store)
    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.TOOL_CALLS,
                tool_calls=[ModelToolCall("call-1", "compare_base_head", {})],
            ),
            final_response_after_tool_error,
        ]
    )

    run = run_reviewer_agent_loop(
        adapter=adapter,
        gateway=gateway,
        assignment=make_assignment("Core Reviewer"),
        intent=make_intent(),
        diff_excerpt=[],
        observations={},
        trace_id="review-tool-error-reviewer-0",
    )

    error_result = run.trace.turns[0].tool_results[0]
    assert run.result.status.value == "partial"
    assert error_result.is_error is True
    assert error_result.call_id == "call-1"
    assert error_result.tool_name == "compare_base_head"
    assert "KeyError" in error_result.content
    assert "'path'" in error_result.content
    assert adapter.requests[1].tool_results[0].is_error is True


def test_agent_loop_returns_failed_result_when_final_response_cannot_be_parsed(git_repo):
    base = run_git(git_repo, "rev-parse", "HEAD")
    observation_store = ObservationStore(git_repo / ".review-agent" / "runs" / "review-parse-error")
    gateway = ToolGateway(git_repo, base, base, observation_store)
    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text="not json",
                raw={"turn": "bad-final"},
                model="bad-final-model",
            )
        ]
    )

    run = run_reviewer_agent_loop(
        adapter=adapter,
        gateway=gateway,
        assignment=make_assignment("Core Reviewer"),
        intent=make_intent(),
        diff_excerpt=[],
        observations={},
        trace_id="review-parse-error-reviewer-0",
    )

    assert run.result.status.value == "failed"
    assert "final response parse failed" in run.result.uncertainties[0]
    assert run.trace.final_status == "failed"
    assert run.trace.turns[0].error == run.result.uncertainties[0]


def test_agent_loop_run_to_dict_serializes_trace_response_and_result(git_repo):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "app.py").write_text("def add(a, b):\n    return a * b\n", encoding="utf-8")
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "change app for serialization")
    head = run_git(git_repo, "rev-parse", "HEAD")
    observation_store = ObservationStore(git_repo / ".review-agent" / "runs" / "review-serialize")
    gateway = ToolGateway(git_repo, base, head, observation_store)
    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.TOOL_CALLS,
                tool_calls=[ModelToolCall("call-serialize", "compare_base_head", {"path": "app.py"})],
            ),
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text=json.dumps(
                    {
                        "contract_assessments": [
                            {
                                "contract": "regression_safety",
                                "status": "covered",
                                "summary": "Compared the changed implementation.",
                                "evidence_refs": [],
                            }
                        ],
                        "confirmed_findings": [],
                        "rejected_hypotheses": [],
                        "uncertainties": [],
                        "observation_refs": [],
                        "investigation_summary": "Serialized run.",
                        "status": "completed",
                    }
                ),
                raw={"turn": "serialize-final"},
                model="serialize-model",
            ),
        ]
    )

    run = run_reviewer_agent_loop(
        adapter=adapter,
        gateway=gateway,
        assignment=make_assignment("Core Reviewer"),
        intent=make_intent(),
        diff_excerpt=[],
        observations={},
        trace_id="review-serialize-reviewer-0",
    )

    payload = agent_loop_run_to_dict(run)

    assert payload["response"]["provider_name"] == "fake-tool-calling"
    assert payload["response"]["model"] == "serialize-model"
    assert payload["response"]["raw"] == {"turn": "serialize-final"}
    assert payload["result"]["status"] == "completed"
    assert payload["trace"]["trace_id"] == "review-serialize-reviewer-0"
    assert payload["trace"]["final_status"] == "completed"
    assert payload["trace"]["turns"][0]["tool_calls"][0]["tool_name"] == "compare_base_head"
    assert payload["trace"]["turns"][0]["tool_calls"][0]["call_id"] == "call-serialize"
    assert payload["trace"]["turns"][0]["tool_results"][0]["tool_name"] == "compare_base_head"
    assert payload["trace"]["turns"][0]["tool_results"][0]["observation_ids"]
    assert payload["trace"]["provider_attempt_count"] == 2
    assert payload["trace"]["turns"][0]["provider_attempts"][0]["provider_attempt"] == 1
    assert payload["runtime"]["termination_reason"] == "completed"


def test_agent_loop_retries_provider_exception_without_consuming_an_extra_turn(
    git_repo,
):
    base = run_git(git_repo, "rev-parse", "HEAD")
    observation_store = ObservationStore(
        git_repo / ".review-agent" / "runs" / "review-provider-retry"
    )
    gateway = ToolGateway(git_repo, base, base, observation_store)

    def raise_once(_request):
        raise TimeoutError("temporary outage")

    adapter = FakeToolCallingAdapter(
        script=[
            raise_once,
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text=json.dumps(
                    {
                        "contract_assessments": [],
                        "confirmed_findings": [],
                        "rejected_hypotheses": [],
                        "uncertainties": ["bounded retry test"],
                        "observation_refs": [],
                        "investigation_summary": "Recovered after provider retry.",
                        "status": "partial",
                    }
                ),
            ),
        ]
    )

    run = run_reviewer_agent_loop(
        adapter=adapter,
        gateway=gateway,
        assignment=make_assignment("Core Reviewer"),
        intent=make_intent(),
        diff_excerpt=[],
        observations={},
        trace_id="review-provider-retry-reviewer-0",
    )

    assert run.result.status.value == "partial"
    assert run.runtime.provider_attempts == 2
    assert run.runtime.model_turns == 1
    assert len(run.trace.turns) == 1
    assert [
        attempt.response_kind for attempt in run.trace.turns[0].provider_attempts
    ] == ["exception", "final"]


def test_agent_loop_token_budget_retains_tool_observation(git_repo):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "app.py").write_text(
        "def add(a, b):\n    return a - b\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "change for token budget")
    head = run_git(git_repo, "rev-parse", "HEAD")
    observation_store = ObservationStore(
        git_repo / ".review-agent" / "runs" / "review-token-budget"
    )
    gateway = ToolGateway(git_repo, base, head, observation_store)
    assignment = replace(
        make_assignment("Core Reviewer"),
        max_total_tokens=10,
    )
    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.TOOL_CALLS,
                tool_calls=[
                    ModelToolCall(
                        "call-token",
                        "compare_base_head",
                        {"path": "app.py"},
                    )
                ],
                raw={
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                    }
                },
            ),
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text="{}",
                raw={
                    "usage": {
                        "prompt_tokens": 4,
                        "completion_tokens": 2,
                        "total_tokens": 6,
                    }
                },
            ),
        ]
    )

    run = run_reviewer_agent_loop(
        adapter=adapter,
        gateway=gateway,
        assignment=assignment,
        intent=make_intent(),
        diff_excerpt=[],
        observations={},
        trace_id="review-token-budget-reviewer-0",
    )

    observation_ids = list(observation_store.summaries_by_id())
    assert run.result.status.value == "partial"
    assert run.result.observation_refs == observation_ids
    assert run.runtime.total_tokens == 11
    assert run.runtime.termination_reason is ReviewerTerminationReason.TOKEN_BUDGET_EXHAUSTED
