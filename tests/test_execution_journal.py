from __future__ import annotations

from pathlib import Path

import pytest

from review_agent.execution_journal import (
    ExecutionJournal,
    JournalIntegrityError,
    ToolCallIdentity,
)
from review_agent.model_protocol import ModelToolCall
from review_agent.pr_workspace import PRMetadata, PRWorkspaceStore
from review_agent.review_planning import compile_review_plan
from review_agent.review_protocol import RiskLevel
from review_agent.revision import RepositoryIdentity
from review_agent.tool_result_protocol import (
    ReviewToolResult,
    ToolResultProjectionV2,
)


BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40
SESSION_ID = "SESSION-" + "c" * 64


def _journal(tmp_path: Path):
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
        store.resolve_pr(identity, "local", "journal-task"),
        PRMetadata(title="Journal task"),
    )
    snapshot = store.create_or_load_snapshot(workspace, BASE_SHA, HEAD_SHA)
    session = store.create_session(
        workspace,
        snapshot,
        session_id=SESSION_ID,
    )
    assignment = compile_review_plan(
        snapshot_id=snapshot.snapshot_id,
        risk_level=RiskLevel.LOW,
        allowed_files=("src/api.py",),
        allowed_symbols=(),
        allowed_hunks=(),
    ).assignments[0]
    return (
        ExecutionJournal(
            store,
            session,
            assignment,
            utc_now=lambda: "2026-08-11T00:00:00Z",
        ),
        assignment,
        snapshot,
    )


def test_private_reviewer_runtime_preserves_windows_staging_path_budget(
    tmp_path: Path,
) -> None:
    relative_runtime = (
        Path("pr")
        / ("p-" + "0" * 32)
        / "Sessions"
        / ("u-" + "0" * 32)
        / "Reviewers"
        / ("r-" + "0" * 8)
    )
    staging_name = ".stage-" + "0" * 32 + ".tmp"
    desired_staging_length = 254
    fixed_length = len(str(tmp_path / relative_runtime / staging_name))
    padding = desired_staging_length - fixed_length - 1
    if padding < 1:
        pytest.skip("pytest temporary root leaves no legacy MAX_PATH budget")

    repository = tmp_path / "repo-long-root"
    git_common = repository / ".git"
    git_common.mkdir(parents=True)
    identity = RepositoryIdentity(
        canonical_path=str(repository.resolve()),
        git_common_dir=str(git_common.resolve()),
        origin_url=None,
    )
    store = PRWorkspaceStore(tmp_path / ("w" * padding))
    workspace = store.create_or_load_workspace(
        store.resolve_pr(identity, "local", "long-reviewer-runtime"),
        PRMetadata(title="Long Reviewer Runtime"),
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
    hypothetical_staging = journal.runtime_path / staging_name

    assert journal.runtime_path.name == "r-" + assignment.assignment_id[4:12]
    assert len(str(hypothetical_staging)) <= 259
    assert journal.path.is_file()
    assert (journal.runtime_path / "reviewer.json").is_file()


def _call() -> ModelToolCall:
    return ModelToolCall(
        call_id="call-1",
        tool_name="read_range",
        arguments={"path": "src/api.py", "line_start": 1, "line_end": 10},
    )


def _assistant_message() -> dict:
    return {
        "role": "assistant",
        "content": "I will inspect the changed range.",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": "read_range",
                    "arguments": (
                        '{"line_end":10,"line_start":1,"path":"src/api.py"}'
                    ),
                },
            }
        ],
    }


def _identity(journal: ExecutionJournal) -> ToolCallIdentity:
    return ToolCallIdentity.from_call(
        session_id=journal.session.session_id,
        assignment_id=journal.assignment.assignment_id,
        snapshot_id=journal.session.snapshot.snapshot_id,
        call=_call(),
    )


def _projection(journal: ExecutionJournal) -> ToolResultProjectionV2:
    return ToolResultProjectionV2.inline(
        ReviewToolResult.success(
            tool_call_id="call-1",
            session_id=journal.session.session_id,
            snapshot_id=journal.session.snapshot.snapshot_id,
            tool_name="read_range",
            arguments=_call().arguments,
            content="src/api.py:1: value",
            reacquirable=True,
        )
    )


@pytest.mark.parametrize(
    "crash_after",
    ["model_response", "tool_started", "tool_completed", "turn_committed"],
)
def test_replay_handles_every_tool_turn_crash_window(
    tmp_path: Path,
    crash_after: str,
) -> None:
    journal, _assignment, _snapshot = _journal(tmp_path)
    identity = _identity(journal)
    projection = _projection(journal)
    journal.record_model_response(
        turn_index=0,
        assistant_message=_assistant_message(),
        tool_calls=(_call(),),
        active_elapsed_seconds=1.0,
    )
    if crash_after != "model_response":
        journal.record_tool_started(
            identity,
            arguments=_call().arguments,
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
            assistant_message=_assistant_message(),
            projections=(projection,),
            active_elapsed_seconds=4.0,
        )

    replay = journal.replay()

    assert replay.active_elapsed_seconds == {
        "model_response": 1.0,
        "tool_started": 2.0,
        "tool_completed": 3.0,
        "turn_committed": 4.0,
    }[crash_after]
    if crash_after == "turn_committed":
        assert replay.pending_turn is None
        assert [message["role"] for message in replay.committed_messages] == [
            "assistant",
            "tool",
        ]
        assert replay.committed_messages[1]["tool_call_id"] == "call-1"
    else:
        assert replay.pending_turn is not None
        assert replay.committed_messages == ()
    assert ("call-1" in replay.completed_calls) is (
        crash_after in {"tool_completed", "turn_committed"}
    )
    assert ("call-1" in replay.started_without_terminal) is (
        crash_after == "tool_started"
    )


def test_same_call_id_with_different_arguments_fails_closed(tmp_path: Path) -> None:
    journal, _assignment, _snapshot = _journal(tmp_path)
    identity = _identity(journal)
    journal.record_tool_started(
        identity,
        arguments=_call().arguments,
        active_elapsed_seconds=1.0,
    )
    changed = ModelToolCall(
        call_id="call-1",
        tool_name="read_range",
        arguments={"path": "src/other.py", "line_start": 1, "line_end": 10},
    )

    with pytest.raises(JournalIntegrityError, match="call_id|identity"):
        journal.record_tool_started(
            ToolCallIdentity.from_call(
                session_id=journal.session.session_id,
                assignment_id=journal.assignment.assignment_id,
                snapshot_id=journal.session.snapshot.snapshot_id,
                call=changed,
            ),
            arguments=changed.arguments,
            active_elapsed_seconds=2.0,
        )


def test_completed_tool_call_is_idempotent_and_not_appended_twice(tmp_path: Path) -> None:
    journal, _assignment, _snapshot = _journal(tmp_path)
    identity = _identity(journal)
    projection = _projection(journal)
    journal.record_tool_started(
        identity,
        arguments=_call().arguments,
        active_elapsed_seconds=1.0,
    )
    journal.record_tool_completed(identity, projection, active_elapsed_seconds=2.0)
    event_count = len(journal.read_events())

    reused = journal.record_tool_completed(
        identity,
        projection,
        active_elapsed_seconds=3.0,
    )

    assert reused == projection
    assert len(journal.read_events()) == event_count


def test_non_empty_private_journal_reopens_for_resume(tmp_path: Path) -> None:
    journal, assignment, _snapshot = _journal(tmp_path)
    journal.record_model_response(
        turn_index=0,
        assistant_message=_assistant_message(),
        tool_calls=(_call(),),
        active_elapsed_seconds=1.0,
    )

    resumed = ExecutionJournal(
        journal.workspace_store,
        journal.session,
        assignment,
    )

    assert resumed.runtime_path == journal.runtime_path
    assert resumed.path == journal.path
    assert resumed.read_events() == journal.read_events()


def test_journal_hash_chain_detects_tampering(tmp_path: Path) -> None:
    journal, _assignment, _snapshot = _journal(tmp_path)
    journal.record_model_response(
        turn_index=0,
        assistant_message=_assistant_message(),
        tool_calls=(_call(),),
        active_elapsed_seconds=1.0,
    )
    path = journal.path
    text = path.read_text("utf-8").replace("read_range", "search_code", 1)
    path.write_text(text, "utf-8")

    with pytest.raises(JournalIntegrityError, match="hash"):
        journal.read_events()
