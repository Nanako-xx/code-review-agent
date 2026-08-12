from __future__ import annotations

import json
from pathlib import Path

import pytest

from review_agent.pr_workspace import PRMetadata, PRWorkspaceStore
from review_agent.revision import RepositoryIdentity
from review_agent.tool_artifacts import (
    MAX_ARTIFACT_PAGE_CHARS,
    ToolResultArtifactStore,
    ToolResultLimits,
    ToolResultProjector,
)
from review_agent.tool_result_protocol import (
    ReviewToolResult,
    serialized_tool_content_chars,
    serialize_tool_result_projection_v2,
    ToolResultProjectionV2,
)


BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40
SESSION_ID = "session-tool-artifacts"


def _workspace(tmp_path: Path):
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
        workspace_store.resolve_pr(identity, "local", "tool-task"),
        PRMetadata(title="Tool task"),
    )
    snapshot = workspace_store.create_or_load_snapshot(
        workspace, BASE_SHA, HEAD_SHA
    )
    return workspace_store, snapshot


def _result(
    snapshot_id: str,
    call_id: str,
    content: str,
    *,
    reacquirable: bool,
) -> ReviewToolResult:
    return ReviewToolResult.success(
        tool_call_id=call_id,
        session_id=SESSION_ID,
        snapshot_id=snapshot_id,
        tool_name="read_range",
        arguments={"path": "src/api.py", "line_start": 1, "line_end": 20},
        content=content,
        reacquirable=reacquirable,
    )


@pytest.mark.parametrize(
    ("size", "externalized"),
    [
        (49_999, False),
        (50_000, False),
        (50_001, True),
    ],
)
def test_non_reacquirable_single_result_has_exact_50k_boundary(
    tmp_path: Path,
    size: int,
    externalized: bool,
) -> None:
    workspace_store, snapshot = _workspace(tmp_path)
    artifact_store = ToolResultArtifactStore(workspace_store, snapshot)
    projector = ToolResultProjector(
        artifact_store,
        limits=ToolResultLimits(turn_budget_chars=1_000_000),
    )

    batch = projector.project_turn(
        (_result(snapshot.snapshot_id, "call-1", "x" * size, reacquirable=False),)
    )
    projection = batch.projections[0]

    assert projection.original_size == size
    assert (projection.artifact_id is not None) is externalized
    assert (projection.content is None) is externalized
    if externalized:
        assert serialized_tool_content_chars(projection.preview or "") <= 2_000
        page = artifact_store.read_artifact(projection.artifact_id, max_chars=50_000)
        assert page.content == "x" * 50_000
        assert page.has_more is True


def test_unicode_and_json_escaping_use_serialized_character_size(tmp_path: Path) -> None:
    workspace_store, snapshot = _workspace(tmp_path)
    content = '"' * 25_001
    assert len(content) < 50_000
    assert serialized_tool_content_chars(content) > 50_000

    projection = ToolResultProjector(
        ToolResultArtifactStore(workspace_store, snapshot),
        limits=ToolResultLimits(turn_budget_chars=1_000_000),
    ).project_turn(
        (_result(snapshot.snapshot_id, "call-json", content, reacquirable=False),)
    ).projections[0]

    assert projection.artifact_id is not None
    assert serialized_tool_content_chars(projection.preview or "") <= 2_000


def test_reacquirable_large_result_stays_inline_until_turn_budget_requires_eviction(
    tmp_path: Path,
) -> None:
    workspace_store, snapshot = _workspace(tmp_path)
    store = ToolResultArtifactStore(workspace_store, snapshot)
    generous = ToolResultProjector(
        store,
        limits=ToolResultLimits(turn_budget_chars=1_000_000),
    ).project_turn(
        (_result(snapshot.snapshot_id, "call-large", "r" * 80_000, reacquirable=True),)
    ).projections[0]

    assert generous.content == "r" * 80_000
    assert generous.artifact_id is None

    constrained = ToolResultProjector(store).project_turn(
        (
            _result(snapshot.snapshot_id, "call-old", "a" * 120_000, reacquirable=True),
            _result(snapshot.snapshot_id, "call-new", "b" * 120_000, reacquirable=True),
        )
    )

    assert constrained.total_rendered_chars <= 200_000
    assert constrained.projections[0].status == "evicted"
    assert constrained.projections[0].artifact_id is None
    assert constrained.projections[0].reacquire_arguments == {
        "line_end": 20,
        "line_start": 1,
        "path": "src/api.py",
    }
    assert constrained.projections[1].status == "inline"


@pytest.mark.parametrize(
    ("rendered_size", "expected_status"),
    [
        (199_999, "inline"),
        (200_000, "inline"),
        (200_001, "evicted"),
    ],
)
def test_turn_budget_uses_final_serialized_200k_boundary(
    tmp_path: Path,
    rendered_size: int,
    expected_status: str,
) -> None:
    workspace_store, snapshot = _workspace(tmp_path)
    low = 0
    high = rendered_size
    raw = None
    while low <= high:
        middle = (low + high) // 2
        candidate = _result(
            snapshot.snapshot_id,
            "call-boundary",
            "x" * middle,
            reacquirable=True,
        )
        candidate_size = len(
            serialize_tool_result_projection_v2(
                ToolResultProjectionV2.inline(candidate)
            )
        )
        if candidate_size == rendered_size:
            raw = candidate
            break
        if candidate_size < rendered_size:
            low = middle + 1
        else:
            high = middle - 1
    assert raw is not None
    assert len(
        serialize_tool_result_projection_v2(ToolResultProjectionV2.inline(raw))
    ) == rendered_size

    batch = ToolResultProjector(
        ToolResultArtifactStore(workspace_store, snapshot)
    ).project_turn((raw,))

    assert batch.projections[0].status == expected_status
    assert batch.total_rendered_chars <= 200_000


def test_non_reacquirable_small_results_over_200k_form_one_aggregate_artifact(
    tmp_path: Path,
) -> None:
    workspace_store, snapshot = _workspace(tmp_path)
    store = ToolResultArtifactStore(workspace_store, snapshot)
    results = tuple(
        _result(
            snapshot.snapshot_id,
            f"call-{index}",
            f"sentinel-{index}:" + (str(index) * 45_000),
            reacquirable=False,
        )
        for index in range(5)
    )

    batch = ToolResultProjector(store).project_turn(results)

    artifact_ids = {projection.artifact_id for projection in batch.projections}
    assert None not in artifact_ids
    assert len(artifact_ids) == 1
    assert all(projection.status == "aggregate_artifact" for projection in batch.projections)
    assert batch.total_rendered_chars <= 200_000
    aggregate_id = next(iter(artifact_ids))
    page = store.read_artifact(aggregate_id, max_chars=50_000)
    aggregate_text = page.content
    cursor = page.next_cursor
    while cursor is not None:
        page = store.read_artifact(
            aggregate_id,
            cursor=cursor,
            max_chars=50_000,
        )
        aggregate_text += page.content
        cursor = page.next_cursor
    payload = json.loads(aggregate_text)
    assert [entry["tool_call_id"] for entry in payload["entries"]] == [
        f"call-{index}" for index in range(5)
    ]
    assert all(f"sentinel-{index}:" in aggregate_text for index in range(5))


def test_read_artifact_pages_by_character_cursor_at_50k_maximum(tmp_path: Path) -> None:
    workspace_store, snapshot = _workspace(tmp_path)
    store = ToolResultArtifactStore(workspace_store, snapshot)
    artifact = store.publish_text("雪" * 120_001, logical_kind="test")

    first = store.read_artifact(artifact.artifact_id, max_chars=MAX_ARTIFACT_PAGE_CHARS)
    second = store.read_artifact(
        artifact.artifact_id,
        cursor=first.next_cursor,
        max_chars=MAX_ARTIFACT_PAGE_CHARS,
    )
    third = store.read_artifact(
        artifact.artifact_id,
        cursor=second.next_cursor,
        max_chars=MAX_ARTIFACT_PAGE_CHARS,
    )

    assert len(first.content) == len(second.content) == 50_000
    assert len(third.content) == 20_001
    assert first.next_cursor == 50_000
    assert second.next_cursor == 100_000
    assert third.next_cursor is None and third.has_more is False


def test_artifact_write_failure_is_explicit_and_retains_raw_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_store, snapshot = _workspace(tmp_path)
    store = ToolResultArtifactStore(workspace_store, snapshot)
    raw = _result(
        snapshot.snapshot_id,
        "call-failed-write",
        "must-survive:" + ("x" * 60_000),
        reacquirable=False,
    )

    def fail_publish(*args, **kwargs):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(store, "publish_text", fail_publish)
    batch = ToolResultProjector(store).project_turn((raw,))
    projection = batch.projections[0]

    assert projection.is_error is True
    assert projection.error is not None
    assert projection.error.code == "artifact_write_failed"
    assert projection.error.retryable is True
    assert batch.retained_unexternalized == (raw,)
    assert raw.content.startswith("must-survive:")
    assert "must-survive" not in serialize_tool_result_projection_v2(projection)


def test_projection_writes_resume_index_with_call_level_reacquirability(
    tmp_path: Path,
) -> None:
    workspace_store, snapshot = _workspace(tmp_path)
    store = ToolResultArtifactStore(workspace_store, snapshot)
    ToolResultProjector(store).project_turn(
        (
            _result(snapshot.snapshot_id, "call-a", "small", reacquirable=True),
            _result(snapshot.snapshot_id, "call-b", "large" * 12_000, reacquirable=False),
        )
    )

    records = store.read_index()

    assert [record["tool_call_id"] for record in records] == ["call-a", "call-b"]
    assert records[0]["reacquirable"] is True
    assert records[1]["reacquirable"] is False
    assert records[1]["artifact_id"].startswith("A-")
    assert records[0]["canonical_arguments_hash"] == records[1]["canonical_arguments_hash"]


def test_preflight_sink_uses_same_snapshot_artifact_store(tmp_path: Path) -> None:
    workspace_store, snapshot = _workspace(tmp_path)
    store = ToolResultArtifactStore(workspace_store, snapshot)
    sink = store.preflight_sink()

    artifact_id = sink.publish(
        snapshot_id=snapshot.snapshot_id,
        logical_name="quality/python_compile/stdout.log",
        content=b"quality-output",
        content_type="text/plain",
    )

    page = store.read_artifact(artifact_id, max_chars=50_000)
    assert page.content == "quality-output"
