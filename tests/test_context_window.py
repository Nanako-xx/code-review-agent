from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pytest

from review_agent.context_window import (
    COMPACTION_SUMMARY_TAG,
    ContextCompactionError,
    ContextWindowIntegrityError,
    ContextWindowManager,
    ContextWindowPolicy,
    ProviderPreferredTokenEstimator,
    Utf8ByteTokenEstimator,
    canonical_context_eviction_marker,
    estimate_complete_request,
    validate_context_eviction_marker,
    CompactionSummaryResult,
)
from review_agent.execution_journal import ExecutionJournal, ToolCallIdentity
from review_agent.model_protocol import (
    ModelToolCall,
    ModelToolSpec,
    ModelTurnRequest,
)
from review_agent.pr_workspace import PRMetadata, PRWorkspaceStore
from review_agent.review_context import (
    ReviewerInvocationV2,
    canonical_pinned_context_bytes_v2,
)
from review_agent.review_planning import compile_review_plan
from review_agent.review_protocol import RiskLevel
from review_agent.revision import RepositoryIdentity
from review_agent.tool_result_protocol import ReviewToolResult, ToolResultProjectionV2


BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40


class _Clock:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


class _ThresholdEstimator:
    def __init__(self, *, compacted_tokens: int = 100) -> None:
        self.compacted_tokens = compacted_tokens

    def estimate_request(self, request: ModelTurnRequest) -> int:
        if any(
            COMPACTION_SUMMARY_TAG in str(message.get("content", ""))
            for message in request.messages
        ):
            return self.compacted_tokens
        if any(message.get("role") == "assistant" for message in request.messages):
            return 700_000
        return 100

    def estimate_text(self, text: str) -> int:
        return len(text.encode("utf-8"))


def _runtime(tmp_path: Path):
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
        store.resolve_pr(identity, "local", "context-window-task"),
        PRMetadata(title="Context window task"),
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
    invocation = ReviewerInvocationV2(
        system="immutable-system",
        tools=(
            {
                "name": "read_range",
                "description": "Read a Snapshot range.",
                "parameters": {"type": "object", "properties": {}},
            },
        ),
        messages=(
            {
                "role": "user",
                "content": "Immutable Intent, Assignment, Preflight and Diff.",
            },
        ),
        parameters={"model": "fake", "temperature": 0, "tool_choice": "auto"},
    )
    return store, session, journal, invocation


def test_parallel_reviewers_have_private_journal_and_compaction_state(
    tmp_path: Path,
) -> None:
    store, session, first, invocation = _runtime(tmp_path)
    assignments = compile_review_plan(
        snapshot_id=session.snapshot.snapshot_id,
        risk_level=RiskLevel.MEDIUM,
        allowed_files=("src/api.py",),
        allowed_symbols=(),
        allowed_hunks=(),
    ).assignments
    first = ExecutionJournal(store, session, assignments[0])
    second = ExecutionJournal(store, session, assignments[1])
    first_manager = ContextWindowManager(
        journal=first,
        invocation=invocation,
        adapter=object(),
        estimator=_ThresholdEstimator(),
        policy=_policy(),
        utc_now=_Clock(datetime(2026, 8, 11, tzinfo=timezone.utc)),
    )
    second_manager = ContextWindowManager(
        journal=second,
        invocation=invocation,
        adapter=object(),
        estimator=_ThresholdEstimator(),
        policy=_policy(),
        utc_now=_Clock(datetime(2026, 8, 11, tzinfo=timezone.utc)),
    )

    _commit_turn(first, 0)
    first_manager.prepare_request(
        parameters=dict(invocation.parameters),
        summarizer=lambda _work: CompactionSummaryResult(
            summary="First Reviewer private progress.",
            active_elapsed_seconds=5.0,
        ),
        active_elapsed_seconds=4.0,
    )

    assert first.runtime_path != second.runtime_path
    assert first.path != second.path
    assert first.replay().context_compaction is not None
    assert second.replay().context_compaction is None
    assert second_manager.active_messages() == invocation.messages
    first_manifest = json.loads(
        (first.runtime_path / "context-manifest.json").read_text("utf-8")
    )
    second_manifest = json.loads(
        (second.runtime_path / "context-manifest.json").read_text("utf-8")
    )
    assert first_manifest["assignment_id"] == assignments[0].assignment_id
    assert second_manifest["assignment_id"] == assignments[1].assignment_id
    assert first_manifest["compaction_generation"] == 1
    assert second_manifest["compaction_generation"] == 0


def _commit_turn(
    journal: ExecutionJournal,
    turn_index: int,
    *,
    reacquirable: bool = True,
) -> None:
    call = ModelToolCall(
        call_id=f"call-{turn_index}",
        tool_name="read_range",
        arguments={"path": f"src/file_{turn_index}.py", "line_start": 1},
    )
    assistant = {
        "role": "assistant",
        "content": f"Inspect turn {turn_index}.",
        "tool_calls": [
            {
                "id": call.call_id,
                "type": "function",
                "function": {
                    "name": call.tool_name,
                    "arguments": json.dumps(
                        call.arguments,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                },
            },
        ],
    }
    identity = ToolCallIdentity.from_call(
        session_id=journal.session.session_id,
        assignment_id=journal.assignment.assignment_id,
        snapshot_id=journal.session.snapshot.snapshot_id,
        call=call,
    )
    result = ReviewToolResult.success(
        tool_call_id=call.call_id,
        session_id=journal.session.session_id,
        snapshot_id=journal.session.snapshot.snapshot_id,
        tool_name=call.tool_name,
        arguments=call.arguments,
        content=f"complete-result-{turn_index}",
        reacquirable=reacquirable,
    )
    elapsed = float(turn_index * 4)
    journal.record_model_response(
        turn_index=turn_index,
        assistant_message=assistant,
        tool_calls=(call,),
        active_elapsed_seconds=elapsed,
    )
    journal.record_tool_started(
        identity,
        arguments=call.arguments,
        active_elapsed_seconds=elapsed + 1,
    )
    projection = ToolResultProjectionV2.inline(result)
    journal.record_tool_completed(
        identity,
        projection,
        active_elapsed_seconds=elapsed + 2,
    )
    journal.record_turn_committed(
        turn_index=turn_index,
        assistant_message=assistant,
        projections=(projection,),
        active_elapsed_seconds=elapsed + 3,
    )


def _policy(**overrides: int) -> ContextWindowPolicy:
    values = {
        "context_window_tokens": 1_000_000,
        "soft_compaction_trigger_tokens": 700_000,
        "compaction_summary_max_tokens": 50_000,
        "prompt_cache_idle_eviction_seconds": 3_600,
        "recent_reacquirable_tool_results_to_keep": 5,
        "output_reserve_tokens": 100_000,
        "safety_reserve_tokens": 50_000,
    }
    values.update(overrides)
    return ContextWindowPolicy(**values)


def test_fallback_estimator_is_utf8_byte_upper_bound_and_counts_reserves() -> None:
    request = ModelTurnRequest(
        system="系统",
        tools=[ModelToolSpec("read", "读取", {"type": "object"})],
        messages=[{"role": "user", "content": "é"}],
        tool_results=[],
        parameters={"temperature": 0},
    )
    estimator = Utf8ByteTokenEstimator()

    estimate = estimate_complete_request(
        request,
        estimator=estimator,
        policy=_policy(output_reserve_tokens=123, safety_reserve_tokens=456),
    )

    assert estimate.input_tokens == estimator.estimate_request(request)
    assert estimate.total_tokens == estimate.input_tokens + 123 + 456
    assert estimate.input_tokens >= len("系统é".encode("utf-8"))


def test_provider_estimator_is_preferred_and_invalid_value_falls_back() -> None:
    request = ModelTurnRequest("s", [], [{"role": "user", "content": "m"}], [], {})

    class Adapter:
        def __init__(self, value) -> None:
            self.value = value

        def estimate_request_tokens(self, candidate) -> int:
            assert candidate is request
            return self.value

        def estimate_text_tokens(self, text: str) -> int:
            return 7

    assert ProviderPreferredTokenEstimator(Adapter(321)).estimate_request(request) == 321
    assert ProviderPreferredTokenEstimator(Adapter(321)).estimate_text("summary") == 7
    assert (
        ProviderPreferredTokenEstimator(Adapter(None)).estimate_request(request)
        == Utf8ByteTokenEstimator().estimate_request(request)
    )


def test_prepare_uses_fixed_pipeline_order_and_preserves_pinned_bytes(
    tmp_path: Path,
) -> None:
    _store, _session, journal, invocation = _runtime(tmp_path)
    before = canonical_pinned_context_bytes_v2(invocation)
    manager = ContextWindowManager(
        journal=journal,
        invocation=invocation,
        adapter=object(),
        policy=_policy(),
        utc_now=_Clock(datetime(2026, 8, 11, tzinfo=timezone.utc)),
    )

    prepared = manager.prepare_request(
        parameters=dict(invocation.parameters),
        active_elapsed_seconds=0.0,
    )

    assert prepared.pipeline_trace == (
        "assemble",
        "layer_1",
        "layer_2",
        "estimate",
        "layer_3",
        "re_estimate",
        "hard_check",
    )
    assert prepared.compacted is False
    assert canonical_pinned_context_bytes_v2(invocation) == before
    assert prepared.request.system == invocation.system
    assert tuple(prepared.request.messages[:1]) == invocation.messages


def test_idle_eviction_boundary_keeps_latest_five_and_preserves_tool_pairs(
    tmp_path: Path,
) -> None:
    _store, _session, journal, invocation = _runtime(tmp_path)
    for turn in range(7):
        _commit_turn(journal, turn)
    baseline = datetime(2026, 8, 11, tzinfo=timezone.utc)
    clock = _Clock(baseline)
    manager = ContextWindowManager(
        journal=journal,
        invocation=invocation,
        adapter=object(),
        policy=_policy(),
        utc_now=clock,
    )
    manager.mark_api_request()

    clock.value = baseline + timedelta(seconds=3_599)
    not_evicted = manager.prepare_request(
        parameters=dict(invocation.parameters),
        active_elapsed_seconds=30.0,
    )
    assert not [
        message
        for message in not_evicted.request.messages
        if message.get("role") == "tool"
        and _is_eviction_marker(message["content"])
    ]

    clock.value = baseline + timedelta(seconds=3_600)
    evicted = manager.prepare_request(
        parameters=dict(invocation.parameters),
        active_elapsed_seconds=30.0,
    )
    tool_messages = [
        message for message in evicted.request.messages if message.get("role") == "tool"
    ]
    markers = [
        validate_context_eviction_marker(message["content"])
        for message in tool_messages[:2]
    ]
    assert [marker["tool_call_id"] for marker in markers] == ["call-0", "call-1"]
    assert all(marker["reason"] == "prompt_cache_idle_60m" for marker in markers)
    assert all(marker["reacquirable"] is True for marker in markers)
    assert all(marker["arguments_hash"].startswith("sha256:") for marker in markers)
    assert all("complete-result" in message["content"] for message in tool_messages[2:])
    _assert_adjacent_tool_pairs(evicted.request.messages)


def test_full_compaction_commits_summary_and_resume_uses_only_summary(
    tmp_path: Path,
) -> None:
    _store, session, journal, invocation = _runtime(tmp_path)
    _commit_turn(journal, 0)
    _commit_turn(journal, 1, reacquirable=False)
    pinned = canonical_pinned_context_bytes_v2(invocation)
    manager = ContextWindowManager(
        journal=journal,
        invocation=invocation,
        adapter=object(),
        estimator=_ThresholdEstimator(),
        policy=_policy(),
        utc_now=_Clock(datetime(2026, 8, 11, tzinfo=timezone.utc)),
    )
    work_seen = []

    def summarize(work):
        work_seen.append(work)
        return CompactionSummaryResult(
            summary=(
                "Completed investigations: two files. Key facts retained. "
                "Candidate findings: none. Uncertainties: none. Next: finish review."
            ),
            active_elapsed_seconds=20.0,
        )

    prepared = manager.prepare_request(
        parameters=dict(invocation.parameters),
        summarizer=summarize,
        active_elapsed_seconds=10.0,
    )

    assert prepared.compacted is True
    assert len(work_seen) == 1
    assert work_seen[0].through_turn == 1
    assert any(message.get("role") == "assistant" for message in work_seen[0].messages)
    dynamic = prepared.request.messages[len(invocation.messages) :]
    assert len(dynamic) == 1
    assert dynamic[0]["role"] == "user"
    assert COMPACTION_SUMMARY_TAG in dynamic[0]["content"]
    assert 'trust="untrusted-data"' in dynamic[0]["content"]
    assert not any(message.get("role") in {"assistant", "tool"} for message in dynamic)
    assert canonical_pinned_context_bytes_v2(invocation) == pinned

    replay = journal.replay()
    assert replay.context_compaction is not None
    assert replay.context_compaction.generation == 1
    assert replay.context_compaction.through_turn == 1
    summary_path = journal.runtime_path / replay.context_compaction.summary_path
    assert summary_path.read_text("utf-8").startswith("Completed investigations")
    manifest = json.loads(
        (journal.runtime_path / "context-manifest.json").read_text("utf-8")
    )
    assert manifest["compaction_generation"] == 1
    assert manifest["compacted_through_turn"] == 1
    assert manifest["compaction_summary_hash"] == replay.context_compaction.summary_hash

    resumed = ContextWindowManager(
        journal=journal,
        invocation=invocation,
        adapter=object(),
        estimator=_ThresholdEstimator(),
        policy=_policy(),
        utc_now=_Clock(datetime(2026, 8, 11, tzinfo=timezone.utc)),
    ).active_messages()
    assert resumed == tuple(prepared.request.messages)


def test_orphan_compaction_started_is_ignored_and_retry_uses_new_generation(
    tmp_path: Path,
) -> None:
    _store, _session, journal, invocation = _runtime(tmp_path)
    _commit_turn(journal, 0)
    manager = ContextWindowManager(
        journal=journal,
        invocation=invocation,
        adapter=object(),
        estimator=_ThresholdEstimator(),
        policy=_policy(),
        utc_now=_Clock(datetime(2026, 8, 11, tzinfo=timezone.utc)),
    )
    before = manager.active_messages()

    with pytest.raises(ContextCompactionError, match="summary generation failed"):
        manager.prepare_request(
            parameters=dict(invocation.parameters),
            summarizer=lambda _work: (_ for _ in ()).throw(RuntimeError("boom")),
            active_elapsed_seconds=5.0,
        )

    assert journal.replay().context_compaction is None
    assert manager.active_messages() == before
    assert [
        event.event_type for event in journal.read_events()
    ].count("context_compaction_started") == 1

    orphan_summary = b"orphan summary that must never become active"
    (journal.runtime_path / "context-compaction-00000001.txt").write_bytes(
        orphan_summary
    )
    manifest_path = journal.runtime_path / "context-manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest.update(
        {
            "compaction_generation": 1,
            "compacted_through_turn": 0,
            "compaction_trigger": "soft_threshold",
            "compaction_summary_hash": hashlib.sha256(orphan_summary).hexdigest(),
        }
    )
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    resumed_after_manifest_crash = ContextWindowManager(
        journal=journal,
        invocation=invocation,
        adapter=object(),
        estimator=_ThresholdEstimator(),
        policy=_policy(),
        utc_now=_Clock(datetime(2026, 8, 11, tzinfo=timezone.utc)),
    )
    assert resumed_after_manifest_crash.active_messages() == before

    prepared = resumed_after_manifest_crash.prepare_request(
        parameters=dict(invocation.parameters),
        summarizer=lambda _work: CompactionSummaryResult(
            summary="Investigated. Facts retained. No findings. No uncertainty. Done.",
            active_elapsed_seconds=6.0,
        ),
        active_elapsed_seconds=5.0,
    )
    assert prepared.compacted is True
    assert journal.replay().context_compaction.generation == 2


def test_compaction_that_remains_at_700k_rolls_back_without_publication(
    tmp_path: Path,
) -> None:
    _store, session, journal, invocation = _runtime(tmp_path)
    _commit_turn(journal, 0)
    manager = ContextWindowManager(
        journal=journal,
        invocation=invocation,
        adapter=object(),
        estimator=_ThresholdEstimator(compacted_tokens=700_000),
        policy=_policy(),
        utc_now=_Clock(datetime(2026, 8, 11, tzinfo=timezone.utc)),
    )
    before = manager.active_messages()

    with pytest.raises(ContextCompactionError, match="below the soft threshold"):
        manager.prepare_request(
            parameters=dict(invocation.parameters),
            summarizer=lambda _work: CompactionSummaryResult(
                summary="Still too large.",
                active_elapsed_seconds=8.0,
            ),
            active_elapsed_seconds=7.0,
        )

    assert manager.active_messages() == before
    assert journal.replay().context_compaction is None
    assert not list(journal.runtime_path.glob("context-compaction-*.txt"))


def test_committed_compaction_summary_is_hash_verified_on_resume(
    tmp_path: Path,
) -> None:
    _store, session, journal, invocation = _runtime(tmp_path)
    _commit_turn(journal, 0)
    manager = ContextWindowManager(
        journal=journal,
        invocation=invocation,
        adapter=object(),
        estimator=_ThresholdEstimator(),
        policy=_policy(),
        utc_now=_Clock(datetime(2026, 8, 11, tzinfo=timezone.utc)),
    )
    manager.prepare_request(
        parameters=dict(invocation.parameters),
        summarizer=lambda _work: CompactionSummaryResult(
            summary="Investigated. Facts retained. No finding. Next: finish.",
            active_elapsed_seconds=5.0,
        ),
        active_elapsed_seconds=4.0,
    )
    record = journal.replay().context_compaction
    assert record is not None
    (journal.runtime_path / record.summary_path).write_text(
        "tampered", encoding="utf-8"
    )

    with pytest.raises(
        ContextWindowIntegrityError,
        match="Committed Compaction Summary is unavailable",
    ):
        ContextWindowManager(
            journal=journal,
            invocation=invocation,
            adapter=object(),
            estimator=_ThresholdEstimator(),
            policy=_policy(),
            utc_now=_Clock(datetime(2026, 8, 11, tzinfo=timezone.utc)),
        )


def test_eviction_marker_is_canonical_and_bound_to_outer_call() -> None:
    marker = canonical_context_eviction_marker(
        tool_call_id="call-1",
        tool_name="read_range",
        canonical_arguments_hash="a" * 64,
    )

    assert validate_context_eviction_marker(marker, expected_call_id="call-1") == {
        "arguments_hash": "sha256:" + "a" * 64,
        "reacquirable": True,
        "reason": "prompt_cache_idle_60m",
        "status": "context_evicted",
        "tool_call_id": "call-1",
        "tool_name": "read_range",
    }
    with pytest.raises(ValueError, match="call ID"):
        validate_context_eviction_marker(marker, expected_call_id="call-other")


def _is_eviction_marker(content: str) -> bool:
    try:
        validate_context_eviction_marker(content)
    except ValueError:
        return False
    return True


def _assert_adjacent_tool_pairs(messages) -> None:
    for index, message in enumerate(messages):
        calls = message.get("tool_calls") if message.get("role") == "assistant" else None
        if not calls:
            continue
        expected = [call["id"] for call in calls]
        actual = [
            messages[index + offset].get("tool_call_id")
            for offset in range(1, len(expected) + 1)
        ]
        assert actual == expected
