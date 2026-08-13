from __future__ import annotations

from pathlib import Path
import hashlib
import json

import pytest

from review_agent.context_window import (
    COMPACTION_SUMMARY_TAG,
    COMPACTION_SYSTEM_PROMPT,
    ContextWindowPolicy,
)
from review_agent.execution_journal import ExecutionJournal, ToolCallIdentity
from review_agent.diff_artifact import DiffArtifactIndex, DiffFileIndex, DiffHunkIndex
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
from review_agent.reviewer_output import ReviewerOutputParser
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


def _runtime(
    tmp_path: Path,
    adapter: _Adapter,
    *,
    context_window_policy=None,
    token_estimator=None,
    with_output_index: bool = False,
):
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
        context_window_policy=context_window_policy,
        token_estimator=token_estimator,
        output_parser=(
            ReviewerOutputParser(
                diff_index=_output_index(snapshot.snapshot_id),
                assignment=assignment,
            )
            if with_output_index
            else None
        ),
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


def _output_index(snapshot_id: str) -> DiffArtifactIndex:
    patch = b"diff --git a/src/api.py b/src/api.py\n@@ -1,3 +1,3 @@\n"
    return DiffArtifactIndex(
        snapshot_id=snapshot_id,
        base_sha=BASE_SHA,
        head_sha=HEAD_SHA,
        patch_artifact_id="A-" + "d" * 64,
        diff_sha256=hashlib.sha256(patch).hexdigest(),
        diff_size_bytes=len(patch),
        files=(
            DiffFileIndex(
                file_index=0,
                path="src/api.py",
                previous_path=None,
                status="modify",
                additions=1,
                deletions=1,
                binary=False,
                submodule=False,
                byte_start=0,
                byte_end=len(patch),
                hunks=(
                    DiffHunkIndex(
                        hunk_index=0,
                        old_start=1,
                        old_count=3,
                        new_start=1,
                        new_count=3,
                        byte_start=40,
                        byte_end=len(patch),
                    ),
                ),
            ),
        ),
    )


def _final() -> ModelTurnResponse:
    return ModelTurnResponse(
        kind=ModelResponseKind.FINAL,
        final_text='{"findings":[],"uncertainties":[]}',
        raw={"usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}},
        provider_name="fake",
        model="fake",
    )


class _CompactionEstimator:
    def estimate_request(self, request) -> int:
        if request.system == COMPACTION_SYSTEM_PROMPT:
            return 100
        if any(
            COMPACTION_SUMMARY_TAG in str(message.get("content", ""))
            for message in request.messages
        ):
            return 100
        if any(message.get("role") == "assistant" for message in request.messages):
            return 700_000
        return 100

    def estimate_text(self, text: str) -> int:
        return len(text.encode("utf-8"))


def _context_policy() -> ContextWindowPolicy:
    return ContextWindowPolicy(
        output_reserve_tokens=100_000,
        safety_reserve_tokens=50_000,
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


def test_loop_compacts_full_dynamic_history_before_next_reviewer_call(
    tmp_path: Path,
) -> None:
    summary_response = ModelTurnResponse(
        kind=ModelResponseKind.FINAL,
        final_text=(
            "Completed one investigation. Retain the read result. "
            "No candidate finding yet. No uncertainty. Next: finish review."
        ),
        raw={
            "usage": {
                "input_tokens": 20,
                "output_tokens": 10,
                "total_tokens": 30,
            }
        },
        provider_name="fake",
        model="fake",
    )
    adapter = _Adapter((_tool_response(1), summary_response, _final()))
    loop, journal, backend, _assignment = _runtime(
        tmp_path,
        adapter,
        context_window_policy=_context_policy(),
        token_estimator=_CompactionEstimator(),
    )

    run = loop.run()

    assert run.status == "completed", run.error_code
    assert len(backend.calls) == 1
    assert len(adapter.requests) == 3
    assert adapter.requests[1].system == COMPACTION_SYSTEM_PROMPT
    assert adapter.requests[1].tools == []
    assert adapter.requests[1].parameters["max_output_tokens"] == 50_000
    final_dynamic = adapter.requests[2].messages[1:]
    assert len(final_dynamic) == 1
    assert final_dynamic[0]["role"] == "user"
    assert COMPACTION_SUMMARY_TAG in final_dynamic[0]["content"]
    assert not any(
        message.get("role") in {"assistant", "tool"}
        for message in final_dynamic
    )
    assert journal.replay().context_compaction is not None
    assert run.runtime.provider_attempts == 3
    assert run.runtime.total_tokens == 45
    manifest = json.loads(
        (journal.session.path / "context-manifest.json").read_text("utf-8")
    )
    assert manifest["last_api_request_at"] is not None


def test_compaction_provider_failure_rolls_back_without_reexecuting_tool(
    tmp_path: Path,
) -> None:
    adapter = _Adapter(
        (
            _tool_response(1),
            ConnectionError("one"),
            TimeoutError("two"),
            ConnectionError("three"),
        )
    )
    loop, journal, backend, _assignment = _runtime(
        tmp_path,
        adapter,
        context_window_policy=_context_policy(),
        token_estimator=_CompactionEstimator(),
    )

    run = loop.run()

    assert run.status == "failed"
    assert run.error_code == "context_compaction_failed"
    assert len(backend.calls) == 1
    assert journal.replay().context_compaction is None
    assert journal.replay().pending_turn is None
    assert run.runtime.provider_attempts == 4


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


def test_invalid_final_output_gets_one_strict_json_finalization(
    tmp_path: Path,
) -> None:
    adapter = _Adapter(
        (
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text=(
                    '{"findings":[],"uncertainties":[],"status":"completed"}'
                ),
                raw={
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"findings":[],"uncertainties":[],'
                                    '"status":"completed"}'
                                )
                            }
                        }
                    ]
                },
            ),
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text='{"findings":[],"uncertainties":[]}',
                raw={
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 5,
                        "total_tokens": 25,
                    }
                },
            ),
        )
    )
    loop, journal, _backend, _assignment = _runtime(tmp_path, adapter)

    run = loop.run()

    assert run.status == "completed"
    assert run.error_code is None
    assert run.final_text == '{"findings":[],"uncertainties":[]}'
    assert journal.replay().final_text == run.final_text
    assert len(adapter.requests) == 2
    finalization = adapter.requests[1]
    assert finalization.tools == []
    assert finalization.parameters["tool_choice"] == "none"
    assert finalization.parameters["response_format"] == "json_object"
    assert finalization.messages[-2]["role"] == "assistant"
    assert '"status":"completed"' in finalization.messages[-2]["content"]
    assert finalization.messages[-1]["role"] == "user"
    assert "top_level_fields_invalid" in finalization.messages[-1]["content"]
    assert run.runtime.provider_attempts == 2
    assert run.runtime.model_turns == 2


def test_json_finalization_remains_fail_closed_when_correction_is_invalid(
    tmp_path: Path,
) -> None:
    adapter = _Adapter(
        (
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text="review prose",
            ),
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text='{"findings":[]}',
            ),
            _final(),
        )
    )
    loop, journal, _backend, _assignment = _runtime(tmp_path, adapter)

    run = loop.run()

    assert run.status == "invalid_output"
    assert run.error_code == "invalid_reviewer_output"
    assert journal.replay().final_text is None
    assert len(adapter.requests) == 2
    assert run.runtime.provider_attempts == 2
    assert run.runtime.model_turns == 2


def test_json_finalization_transport_retries_share_provider_attempt_budget(
    tmp_path: Path,
) -> None:
    adapter = _Adapter(
        (
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text="review prose",
            ),
            ConnectionError("first finalization transport failure"),
            TimeoutError("second finalization transport failure"),
            _final(),
        )
    )
    loop, journal, _backend, _assignment = _runtime(tmp_path, adapter)

    run = loop.run()

    assert run.status == "failed"
    assert run.error_code == "json_finalization_transport_failed"
    assert journal.replay().final_text is None
    assert len(adapter.requests) == 3
    assert run.runtime.provider_attempts == 3
    assert run.runtime.model_turns == 1


def test_loop_keeps_good_finding_and_persists_bad_candidate_rejection(
    tmp_path: Path,
) -> None:
    payload = {
        "findings": [
            {
                "claim": (
                    "When the value is absent, dereferencing it raises and the "
                    "request returns 500."
                ),
                "severity": "high",
                "path": "src/api.py",
                "line": 1,
                "suggestion": (
                    "Handle the absent value before dereferencing it and add a test."
                ),
            },
            {
                "claim": "This candidate points outside the changed line range.",
                "severity": "low",
                "path": "src/api.py",
                "line": 99,
                "suggestion": "Add a guard and a regression test for this path.",
            },
        ],
        "uncertainties": [],
    }
    adapter = _Adapter(
        (
            ModelTurnResponse(
                kind=ModelResponseKind.FINAL,
                final_text=json.dumps(payload),
            ),
        )
    )
    loop, journal, _backend, _assignment = _runtime(
        tmp_path,
        adapter,
        with_output_index=True,
    )

    first = loop.run()
    resumed = loop.run()

    assert first.status == resumed.status == "completed"
    assert first.reviewer_output is not None
    assert len(first.reviewer_output.findings) == 1
    assert first.final_text == first.reviewer_output.to_json()
    assert len(first.rejected_findings) == 1
    assert first.rejected_findings[0].reason == "line_not_in_diff"
    assert resumed.rejected_findings == first.rejected_findings
    assert journal.replay().final_rejections == (
        {"candidate_index": 1, "reason": "line_not_in_diff"},
    )
    assert len(adapter.requests) == 1


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
