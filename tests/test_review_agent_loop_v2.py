from __future__ import annotations

from pathlib import Path

import pytest

from review_agent.execution_journal import ExecutionJournal, ToolCallIdentity
from review_agent.model_protocol import (
    ModelResponseKind,
    ModelToolCall,
    ModelTurnResponse,
)
from review_agent.pr_workspace import PRMetadata, PRWorkspaceStore
from review_agent.review_agent_loop import ReviewAgentLoopV2
from review_agent.review_context import ReviewerInvocationV2
from review_agent.review_planning import compile_review_plan
from review_agent.review_protocol import RiskLevel
from review_agent.review_tool_gateway import (
    ReviewToolGateway,
    ToolBackendResult,
)
from review_agent.reviewer_runtime import ReviewerRuntimeLimitsV2
from review_agent.revision import RepositoryIdentity
from review_agent.tool_artifacts import ToolResultArtifactStore, ToolResultProjector
from review_agent.tool_result_protocol import ReviewToolResult, ToolResultProjectionV2


BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40


class _Adapter:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def complete_turn(self, request):
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _Backend:
    def __init__(self):
        self.calls = []

    def execute(self, tool_name, arguments, timeout_seconds):
        self.calls.append((tool_name, dict(arguments), timeout_seconds))
        return ToolBackendResult(
            content=f"result:{arguments.get('path', 'none')}",
            reacquirable=True,
        )


def _runtime(tmp_path: Path, adapter: _Adapter):
    repository = tmp_path / "repo"
    git_common = repository / ".git"
    git_common.mkdir(parents=True)
    identity = RepositoryIdentity(
        canonical_path=str(repository.resolve()),
        git_common_dir=str(git_common.resolve()),
        origin_url=None,
    )
    store = PRWorkspaceStore(tmp_path / "ra")
    workspace = store.create_or_load_workspace(
        store.resolve_pr(identity, "local", "loop-task"),
        PRMetadata(title="Loop task"),
    )
    snapshot = store.create_or_load_snapshot(workspace, BASE_SHA, HEAD_SHA)
    session = store.create_session(workspace, snapshot)
    assignment = compile_review_plan(
        snapshot_id=snapshot.snapshot_id,
        risk_level=RiskLevel.LOW,
        allowed_files=("src/api.py",),
        allowed_symbols=(),
        allowed_hunks=(),
    ).assignments[0]
    journal = ExecutionJournal(store, session, assignment)
    artifact_store = ToolResultArtifactStore(store, snapshot)
    backend = _Backend()
    gateway = ReviewToolGateway(
        snapshot_id=snapshot.snapshot_id,
        session_id=session.session_id,
        allowed_tools=("read_range",),
        backend=backend,
        artifact_store=artifact_store,
    )
    invocation = ReviewerInvocationV2(
        system="system",
        tools=(),
        messages=({"role": "user", "content": "Review the Assignment."},),
        parameters={"model": "fake", "temperature": 0},
    )
    loop = ReviewAgentLoopV2(
        adapter=adapter,
        gateway=gateway,
        projector=ToolResultProjector(artifact_store),
        journal=journal,
        assignment=assignment,
        invocation=invocation,
    )
    return loop, journal, backend, assignment


def _tool_response(call_count: int = 1) -> ModelTurnResponse:
    return ModelTurnResponse(
        kind=ModelResponseKind.TOOL_CALLS,
        tool_calls=[
            ModelToolCall(
                call_id=f"call-{index}",
                tool_name="read_range",
                arguments={"path": f"src/file_{index}.py"},
            )
            for index in range(call_count)
        ],
        raw={"choices": [{"message": {"content": "Inspect files."}}]},
        provider_name="fake",
        model="fake",
    )


def _final() -> ModelTurnResponse:
    return ModelTurnResponse(
        kind=ModelResponseKind.FINAL,
        final_text='{"findings":[],"uncertainties":[]}',
        raw={"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
        provider_name="fake",
        model="fake",
    )


def test_v2_loop_has_no_turn_tool_or_token_stop_condition(tmp_path: Path) -> None:
    adapter = _Adapter((_tool_response(30), _final()))
    loop, journal, backend, _assignment = _runtime(tmp_path, adapter)

    run = loop.run()

    assert run.status == "completed", run.error_code
    assert len(backend.calls) == 30
    assert run.runtime.tool_calls == 30
    assert run.runtime.total_tokens == 15
    assert "budget_exhausted" not in str(run)
    assert journal.replay().pending_turn is None
    for request in adapter.requests:
        assert "max_turns" not in request.parameters
        assert "max_tool_calls" not in request.parameters
        assert "max_total_tokens" not in request.parameters
        assert "max_output_tokens" not in request.parameters


def test_provider_transport_retries_three_times_without_executing_tools_twice(
    tmp_path: Path,
) -> None:
    adapter = _Adapter(
        (
            ConnectionError("first"),
            TimeoutError("second"),
            _tool_response(1),
            _final(),
        )
    )
    loop, _journal, backend, _assignment = _runtime(tmp_path, adapter)

    run = loop.run()

    assert run.status == "completed"
    assert len(adapter.requests) == 4
    assert len(backend.calls) == 1
    assert run.runtime.provider_attempts == 4


@pytest.mark.parametrize(
    ("crash_after", "expected_backend_calls"),
    [
        ("model_response", 1),
        ("tool_started", 1),
        ("tool_completed", 0),
        ("turn_committed", 0),
    ],
)
def test_loop_resumes_each_crash_window_without_reexecuting_completed_call(
    tmp_path: Path,
    crash_after: str,
    expected_backend_calls: int,
) -> None:
    adapter = _Adapter((_final(),))
    loop, journal, backend, _assignment = _runtime(tmp_path, adapter)
    call = _tool_response(1).tool_calls[0]
    assistant = {
        "role": "assistant",
        "content": "Inspect files.",
        "tool_calls": [
            {
                "id": call.call_id,
                "type": "function",
                "function": {
                    "name": call.tool_name,
                    "arguments": '{"path":"src/file_0.py"}',
                },
            }
        ],
    }
    identity = ToolCallIdentity.from_call(
        session_id=journal.session.session_id,
        assignment_id=journal.assignment.assignment_id,
        snapshot_id=journal.session.snapshot.snapshot_id,
        call=call,
    )
    raw = ReviewToolResult.success(
        tool_call_id=call.call_id,
        session_id=journal.session.session_id,
        snapshot_id=journal.session.snapshot.snapshot_id,
        tool_name=call.tool_name,
        arguments=call.arguments,
        content="persisted-result",
        reacquirable=True,
    )
    projection = ToolResultProjectionV2.inline(raw)
    journal.record_model_response(
        turn_index=0,
        assistant_message=assistant,
        tool_calls=(call,),
        active_elapsed_seconds=1.0,
    )
    if crash_after != "model_response":
        journal.record_tool_started(
            identity,
            arguments=call.arguments,
            active_elapsed_seconds=2.0,
        )
    if crash_after in {"tool_completed", "turn_committed"}:
        journal.record_tool_completed(
            identity,
            projection,
            active_elapsed_seconds=3.0,
        )
    if crash_after == "turn_committed":
        journal.record_turn_committed(
            turn_index=0,
            assistant_message=assistant,
            projections=(projection,),
            active_elapsed_seconds=4.0,
        )

    run = loop.run()

    assert run.status == "completed", run.error_code
    assert len(backend.calls) == expected_backend_calls
    assert journal.replay().pending_turn is None


def test_invalid_model_output_is_not_a_transport_retry(tmp_path: Path) -> None:
    adapter = _Adapter(
        (
            ModelTurnResponse(
                kind=ModelResponseKind.INVALID,
                error="invalid protocol",
            ),
        )
    )
    loop, _journal, backend, _assignment = _runtime(tmp_path, adapter)

    run = loop.run()

    assert run.status == "invalid_output"
    assert len(adapter.requests) == 1
    assert backend.calls == []


def test_provider_transport_stops_after_three_attempts(tmp_path: Path) -> None:
    adapter = _Adapter(
        (
            ConnectionError("one"),
            ConnectionError("two"),
            TimeoutError("three"),
            _final(),
        )
    )
    loop, _journal, backend, _assignment = _runtime(tmp_path, adapter)

    run = loop.run()

    assert run.status == "failed"
    assert run.error_code == "provider_transport_failed"
    assert len(adapter.requests) == 3
    assert backend.calls == []


def test_tool_timeout_is_retryable_tool_error_and_does_not_end_reviewer(
    tmp_path: Path,
) -> None:
    class TimeoutBackend:
        def execute(self, tool_name, arguments, timeout_seconds):
            raise TimeoutError("tool timed out")

    adapter = _Adapter((_tool_response(1), _final()))
    loop, journal, _backend, _assignment = _runtime(tmp_path, adapter)
    loop.gateway.backend = TimeoutBackend()

    run = loop.run()

    assert run.status == "completed"
    completed = journal.replay().completed_calls["call-0"].projection
    assert completed.error is not None
    assert completed.error.code == "tool_timeout"
    assert completed.error.retryable is True


def test_completed_resume_reuses_final_result_without_provider_call(
    tmp_path: Path,
) -> None:
    adapter = _Adapter((_final(),))
    loop, _journal, _backend, _assignment = _runtime(tmp_path, adapter)
    first = loop.run()

    resumed = loop.run()

    assert first.status == resumed.status == "completed"
    assert resumed.final_text == first.final_text
    assert len(adapter.requests) == 1


def test_offline_time_does_not_increase_persisted_active_elapsed(
    tmp_path: Path,
) -> None:
    adapter = _Adapter(())
    loop, journal, _backend, _assignment = _runtime(tmp_path, adapter)
    journal.record_final_result(
        final_text='{"findings":[],"uncertainties":[]}',
        active_elapsed_seconds=125.0,
    )
    loop.clock = lambda: 9_999_999.0

    resumed = loop.run()

    assert resumed.status == "completed"
    assert resumed.runtime.active_elapsed_seconds == 125.0
    assert adapter.requests == []


def test_repeated_completed_call_id_fails_as_no_progress_without_reexecution(
    tmp_path: Path,
) -> None:
    adapter = _Adapter((_tool_response(1), _tool_response(1)))
    loop, _journal, backend, _assignment = _runtime(tmp_path, adapter)

    run = loop.run()

    assert run.status == "failed"
    assert run.error_code == "repeated_tool_call_no_progress"
    assert len(backend.calls) == 1


def test_v2_runtime_limits_are_only_1800_3_and_300() -> None:
    limits = ReviewerRuntimeLimitsV2()

    assert limits.max_elapsed_seconds == 1_800.0
    assert limits.max_provider_attempts == 3
    assert limits.tool_timeout_seconds == 300.0
    for legacy in (
        "max_turns",
        "max_tool_calls",
        "max_total_tokens",
        "max_output_tokens",
    ):
        assert not hasattr(limits, legacy)
