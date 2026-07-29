import json
from dataclasses import replace

from review_agent.agent_loop import agent_loop_run_to_dict, run_reviewer_agent_loop
from review_agent.model_adapter import (
    FakeToolCallingAdapter,
    OpenAICompatibleConfig,
    OpenAICompatibleToolAdapter,
)
from review_agent.model_protocol import ModelResponseKind, ModelToolCall, ModelTurnResponse
from review_agent.memory_models import MemoryScope
from review_agent.memory_retrieval import SnapshotMemoryQueryService
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
from tests.test_context import _memory_snapshot


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


def test_agent_loop_executes_snapshot_memory_tool_turn(git_repo):
    base = run_git(git_repo, "rev-parse", "HEAD")
    snapshot = _memory_snapshot(head=base)
    assignment = replace(
        make_assignment("Core Reviewer"),
        assignment_id="assignment-memory",
        initial_context=replace(
            make_assignment("Core Reviewer").initial_context,
            changed_files=["app.py"],
        ),
    )
    observation_store = ObservationStore(
        git_repo / ".review-agent" / "runs" / "review-memory-tool"
    )
    service = SnapshotMemoryQueryService(
        snapshot,
        assignment_id="assignment-memory",
        assignment_scope=MemoryScope(
            paths=("app.py",),
            contracts=("regression_safety",),
        ),
    )
    gateway = ToolGateway(
        git_repo,
        base,
        base,
        observation_store,
        allowed_tools=("query_project_memory",),
        memory_query_service=service,
    )
    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.TOOL_CALLS,
                tool_calls=[
                    ModelToolCall(
                        "memory-call",
                        "query_project_memory",
                        {
                            "assignment_id": "assignment-memory",
                            "path": "app.py",
                            "query": "approved rule",
                        },
                    )
                ],
            ),
            final_response,
        ]
    )

    run = run_reviewer_agent_loop(
        adapter=adapter,
        gateway=gateway,
        assignment=assignment,
        intent=make_intent(),
        diff_excerpt=[],
        observations={},
        trace_id="review-memory-tool-reviewer-0",
    )

    assert run.trace.turns[0].tool_calls[0].tool_name == "query_project_memory"
    assert run.trace.turns[0].tool_results[0].observation_ids
    assert gateway.memory_snapshot.snapshot_id in adapter.requests[1].tool_results[0].content
    assert run.result.observation_refs == list(observation_store.summaries_by_id())
    metadata = run.envelope.parameters["context"]
    query_bytes = len(
        adapter.requests[1].tool_results[0].content.encode("utf-8")
    )
    assert gateway.memory_context_used_bytes == (
        metadata["memory_ledger_initial_bytes"] + query_bytes
    )
    assert gateway.memory_context_used_bytes <= metadata["memory_ledger_limit_bytes"]


def test_agent_loop_hard_policy_budget_failure_blocks_without_model_recovery(
    git_repo,
):
    head = run_git(git_repo, "rev-parse", "HEAD")
    snapshot = _memory_snapshot(head=head, hard_policy=True)
    base_assignment = make_assignment("Core Reviewer")
    assignment = replace(
        base_assignment,
        assignment_id="assignment-memory",
        initial_context=replace(
            base_assignment.initial_context,
            changed_files=["app.py"],
        ),
    )
    service = SnapshotMemoryQueryService(
        snapshot,
        assignment_id="assignment-memory",
        assignment_scope=MemoryScope(
            paths=("app.py",),
            contracts=("regression_safety",),
        ),
    )
    store = ObservationStore(
        git_repo / ".review-agent" / "runs" / "review-memory-hard-block"
    )
    gateway = ToolGateway(
        git_repo,
        head,
        "HEAD",
        store,
        max_context_chars=200,
        allowed_tools=("query_project_memory",),
        memory_query_service=service,
    )

    def must_not_recover(_request):
        raise AssertionError("model must not receive a recoverable tool error")

    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.TOOL_CALLS,
                tool_calls=[
                    ModelToolCall(
                        "memory-hard-call",
                        "query_project_memory",
                        {
                            "assignment_id": "assignment-memory",
                            "path": "app.py",
                        },
                    )
                ],
            ),
            must_not_recover,
        ]
    )

    run = run_reviewer_agent_loop(
        adapter=adapter,
        gateway=gateway,
        assignment=assignment,
        intent=make_intent(),
        diff_excerpt=[],
        observations={},
        trace_id="review-memory-hard-block-reviewer-0",
    )

    assert len(adapter.requests) == 1
    assert run.result.status.value == "blocked"
    assert run.runtime.termination_reason is ReviewerTerminationReason.REVIEWER_BLOCKED
    assert "hard-policy" in run.trace.turns[0].error
    assert run.trace.turns[0].tool_results == []
    assert store.list_observations() == []


def test_agent_loop_memory_queries_share_one_cumulative_ten_percent_ledger(
    git_repo,
):
    head = run_git(git_repo, "rev-parse", "HEAD")
    snapshot = _memory_snapshot(head=head)
    base_assignment = make_assignment("Core Reviewer")
    assignment = replace(
        base_assignment,
        assignment_id="assignment-memory",
        initial_context=replace(
            base_assignment.initial_context,
            changed_files=["app.py"],
        ),
        max_turns=12,
        max_tool_calls=12,
    )
    service = SnapshotMemoryQueryService(
        snapshot,
        assignment_id="assignment-memory",
        assignment_scope=MemoryScope(
            paths=("app.py",),
            contracts=("regression_safety",),
        ),
    )
    gateway = ToolGateway(
        git_repo,
        head,
        "HEAD",
        ObservationStore(
            git_repo / ".review-agent" / "runs" / "review-memory-ledger"
        ),
        allowed_tools=("query_project_memory",),
        memory_query_service=service,
    )

    def continue_until_ledger_error(request):
        if request.tool_results[-1].is_error:
            return final_response_after_tool_error(request)
        next_index = len(request.tool_results) + 1
        return ModelTurnResponse(
            kind=ModelResponseKind.TOOL_CALLS,
            tool_calls=[
                ModelToolCall(
                    f"memory-ledger-{next_index}",
                    "query_project_memory",
                    {
                        "assignment_id": "assignment-memory",
                        "path": "app.py",
                    },
                )
            ],
        )

    first_call = ModelTurnResponse(
        kind=ModelResponseKind.TOOL_CALLS,
        tool_calls=[
            ModelToolCall(
                "memory-ledger-1",
                "query_project_memory",
                {
                    "assignment_id": "assignment-memory",
                    "path": "app.py",
                },
            )
        ],
    )
    adapter = FakeToolCallingAdapter(
        script=[first_call, *([continue_until_ledger_error] * 11)]
    )

    run = run_reviewer_agent_loop(
        adapter=adapter,
        gateway=gateway,
        assignment=assignment,
        intent=make_intent(),
        diff_excerpt=[],
        observations={},
        trace_id="review-memory-ledger-reviewer-0",
    )

    all_results = [
        result
        for turn in run.trace.turns
        for result in turn.tool_results
    ]
    successful = [result for result in all_results if not result.is_error]
    errors = [result for result in all_results if result.is_error]
    assert len(successful) >= 2
    assert errors
    assert "remaining Context budget" in errors[-1].content
    metadata = run.envelope.parameters["context"]
    assert gateway.memory_context_used_bytes == (
        metadata["memory_ledger_initial_bytes"]
        + sum(len(result.content.encode("utf-8")) for result in successful)
    )
    assert gateway.memory_context_used_bytes <= metadata["memory_ledger_limit_bytes"]
    assert run.result.status.value == "partial"


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


def test_agent_loop_sends_ordered_transcript_after_rejected_final_and_second_tool(
    git_repo,
):
    base = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "app.py").write_text(
        "def add(a, b):\n    return a - b\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "change app for ordered transcript")
    head = run_git(git_repo, "rev-parse", "HEAD")
    observation_store = ObservationStore(
        git_repo / ".review-agent" / "runs" / "review-ordered-transcript"
    )
    gateway = ToolGateway(git_repo, base, head, observation_store)
    rejected_final = json.dumps(
        {
            "contract_assessments": [],
            "confirmed_findings": [],
            "rejected_hypotheses": [],
            "uncertainties": [],
            "observation_refs": [],
            "investigation_summary": "Structured, but missing the assigned contract.",
            "status": "completed",
        }
    )
    accepted_final = json.dumps(
        {
            "contract_assessments": [
                {
                    "contract": "regression_safety",
                    "status": "covered",
                    "summary": "Compared the changed implementation twice.",
                    "evidence_refs": [],
                }
            ],
            "confirmed_findings": [],
            "rejected_hypotheses": [],
            "uncertainties": [],
            "observation_refs": [],
            "investigation_summary": "Completed after the Runtime correction.",
            "status": "completed",
        }
    )
    provider_responses = [
        {
            "choices": [
                {
                    "message": {
                        "content": "Inspecting the first range.",
                        "reasoning_content": "The first comparison is required.",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "compare_base_head",
                                    "arguments": '{"path": "app.py"}',
                                },
                            }
                        ],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": rejected_final,
                        "reasoning_content": "I initially considered the review complete.",
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "message": {
                        "content": "Inspecting the second range.",
                        "reasoning_content": "The Runtime rejection requires more evidence.",
                        "tool_calls": [
                            {
                                "id": "call-2",
                                "type": "function",
                                "function": {
                                    "name": "compare_base_head",
                                    "arguments": '{"path": "app.py"}',
                                },
                            }
                        ],
                    }
                }
            ]
        },
        {"choices": [{"message": {"content": accepted_final}}]},
    ]
    captured_payloads = []

    def transport(url, headers, payload, timeout_seconds):
        captured_payloads.append(payload)
        return provider_responses.pop(0)

    adapter = OpenAICompatibleToolAdapter(
        OpenAICompatibleConfig(
            base_url="https://example.test/v1",
            api_key="secret",
            model="review-model",
        ),
        transport=transport,
    )

    run = run_reviewer_agent_loop(
        adapter=adapter,
        gateway=gateway,
        assignment=make_assignment("Core Reviewer"),
        intent=make_intent(),
        diff_excerpt=["diff excerpt"],
        observations={},
        trace_id="review-ordered-transcript-reviewer-0",
    )

    assert run.result.status.value == "completed"
    assert len(captured_payloads) == 4
    provider_messages = captured_payloads[-1]["messages"]
    assert provider_messages[0]["role"] == "system"
    transcript = provider_messages[1:]
    assert [message["role"] for message in transcript] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
        "assistant",
        "tool",
    ]
    assert len(transcript) == 7
    assert transcript[1]["content"] == "Inspecting the first range."
    assert transcript[1]["reasoning_content"] == "The first comparison is required."
    assert transcript[1]["tool_calls"][0]["id"] == "call-1"
    assert json.loads(
        transcript[1]["tool_calls"][0]["function"]["arguments"]
    ) == {"path": "app.py"}
    assert transcript[2]["tool_call_id"] == "call-1"
    assert transcript[3] == {
        "role": "assistant",
        "content": rejected_final,
        "reasoning_content": "I initially considered the review complete.",
    }
    assert "Runtime rejected completion" in transcript[4]["content"]
    assert transcript[5]["content"] == "Inspecting the second range."
    assert (
        transcript[5]["reasoning_content"]
        == "The Runtime rejection requires more evidence."
    )
    assert transcript[5]["tool_calls"][0]["id"] == "call-2"
    assert json.loads(
        transcript[5]["tool_calls"][0]["function"]["arguments"]
    ) == {"path": "app.py"}
    assert transcript[6]["tool_call_id"] == "call-2"


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
    assert run.trace.turns[0].error == (
        "final response JSON finalization returned invalid"
    )
    assert run.trace.turns[0].error in run.result.uncertainties


def test_agent_loop_performs_one_no_tool_json_finalization_after_parse_failure(
    git_repo,
):
    base = run_git(git_repo, "rev-parse", "HEAD")
    observation_store = ObservationStore(
        git_repo / ".review-agent" / "runs" / "review-json-finalization"
    )
    gateway = ToolGateway(git_repo, base, base, observation_store)

    def corrected_json(request):
        assert request.tools == []
        assert request.parameters["tool_choice"] == "none"
        assert request.parameters["response_format"] == "json_object"
        assert request.messages[-2] == {
            "role": "assistant",
            "content": "## Structured Findings\n- prose, not JSON",
        }
        assert "exactly one JSON object" in request.messages[-1]["content"]
        return ModelTurnResponse(
            kind=ModelResponseKind.FINAL,
            final_text=json.dumps(
                {
                    "contract_assessments": [],
                    "confirmed_findings": [],
                    "rejected_hypotheses": [],
                    "uncertainties": ["No repository observation was available."],
                    "observation_refs": [],
                    "investigation_summary": "Finalized the previously completed analysis.",
                    "status": "partial",
                }
            ),
        )

    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text="## Structured Findings\n- prose, not JSON",
            ),
            corrected_json,
        ]
    )

    run = run_reviewer_agent_loop(
        adapter=adapter,
        gateway=gateway,
        assignment=make_assignment("Core Reviewer"),
        intent=make_intent(),
        diff_excerpt=[],
        observations={},
        trace_id="review-json-finalization-reviewer-0",
    )

    assert run.result.status.value == "partial"
    assert run.result.investigation_summary.startswith("Finalized")
    assert len(adapter.requests) == 2
    assert run.runtime.provider_attempts == 2
    assert run.runtime.model_turns == 1


def test_agent_loop_fails_after_single_malformed_json_finalization_response(
    git_repo,
):
    base = run_git(git_repo, "rev-parse", "HEAD")
    observation_store = ObservationStore(
        git_repo / ".review-agent" / "runs" / "review-json-finalization-malformed"
    )
    gateway = ToolGateway(git_repo, base, base, observation_store)
    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text="first final is not JSON",
            ),
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text="finalization response is still not JSON",
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
        trace_id="review-json-finalization-malformed-reviewer-0",
    )

    diagnostic = "final response JSON finalization parse failed"
    assert len(adapter.requests) == 2
    assert run.runtime.provider_attempts == 2
    assert run.runtime.model_turns == 1
    assert run.runtime.termination_reason is ReviewerTerminationReason.RUNTIME_FAILURE
    assert run.result.status.value == "failed"
    assert any(diagnostic in item for item in run.result.uncertainties)
    assert diagnostic in (run.trace.turns[0].error or "")
    assert run.trace.final_status == "failed"
    assert run.response.content == "finalization response is still not JSON"


def test_agent_loop_does_not_finalize_json_again_after_rejected_repair(git_repo):
    base = run_git(git_repo, "rev-parse", "HEAD")
    observation_store = ObservationStore(
        git_repo / ".review-agent" / "runs" / "review-json-finalization-once"
    )
    gateway = ToolGateway(git_repo, base, base, observation_store)
    repaired_json = json.dumps(
        {
            "contract_assessments": [],
            "confirmed_findings": [],
            "rejected_hypotheses": [],
            "uncertainties": [],
            "observation_refs": [],
            "investigation_summary": "Repaired but incomplete.",
            "status": "completed",
        }
    )
    extra_repair = ModelTurnResponse(
        kind=ModelResponseKind.FINAL,
        final_text=repaired_json,
    )
    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text="first Markdown final",
            ),
            extra_repair,
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text="second Markdown final",
            ),
            extra_repair,
        ]
    )

    run = run_reviewer_agent_loop(
        adapter=adapter,
        gateway=gateway,
        assignment=make_assignment("Core Reviewer"),
        intent=make_intent(),
        diff_excerpt=[],
        observations={},
        trace_id="review-json-finalization-once-reviewer-0",
    )

    assert len(adapter.requests) == 3
    assert run.result.status.value == "failed"
    assert adapter.requests[2].messages[-2] == {
        "role": "assistant",
        "content": repaired_json,
    }
    assert "Runtime rejected completion" in (
        adapter.requests[2].messages[-1]["content"]
    )
    assert "final response JSON finalization already attempted" in run.result.uncertainties
    assert "final response JSON finalization already attempted" in (
        run.trace.turns[-1].error or ""
    )


def test_agent_loop_checks_time_budget_when_json_finalization_raises(
    git_repo,
    monkeypatch,
):
    clock = [0.0]
    monkeypatch.setattr(
        "review_agent.reviewer_runtime.time.monotonic",
        lambda: clock[0],
    )
    base = run_git(git_repo, "rev-parse", "HEAD")
    observation_store = ObservationStore(
        git_repo / ".review-agent" / "runs" / "review-json-finalization-time"
    )
    gateway = ToolGateway(git_repo, base, base, observation_store)
    assignment = replace(
        make_assignment("Core Reviewer"),
        max_elapsed_seconds=1.0,
        max_provider_attempts=2,
    )

    def finalization_times_out(_request):
        clock[0] = 2.0
        raise TimeoutError("finalization timed out")

    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text="not JSON",
            ),
            finalization_times_out,
        ]
    )

    run = run_reviewer_agent_loop(
        adapter=adapter,
        gateway=gateway,
        assignment=assignment,
        intent=make_intent(),
        diff_excerpt=[],
        observations={},
        trace_id="review-json-finalization-time-reviewer-0",
    )

    assert run.result.status.value == "partial"
    assert run.runtime.provider_attempts == 2
    assert (
        run.runtime.termination_reason
        is ReviewerTerminationReason.TIME_BUDGET_EXHAUSTED
    )
    assert any(
        "final response JSON finalization raised TimeoutError" in item
        for item in run.result.uncertainties
    )
    assert "time budget exhausted" in run.result.uncertainties


def test_agent_loop_does_not_exceed_provider_attempt_budget_for_json_finalization(
    git_repo,
):
    base = run_git(git_repo, "rev-parse", "HEAD")
    observation_store = ObservationStore(
        git_repo / ".review-agent" / "runs" / "review-json-finalization-budget"
    )
    gateway = ToolGateway(git_repo, base, base, observation_store)
    assignment = replace(
        make_assignment("Core Reviewer"),
        max_provider_attempts=1,
    )
    adapter = FakeToolCallingAdapter(
        script=[
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text="not JSON",
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
        trace_id="review-json-finalization-budget-reviewer-0",
    )

    assert run.result.status.value == "failed"
    assert len(adapter.requests) == 1
    assert run.runtime.provider_attempts == 1
    assert any(
        "JSON finalization skipped: provider attempt budget exhausted" in item
        for item in run.result.uncertainties
    )


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
