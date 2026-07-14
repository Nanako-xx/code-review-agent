from __future__ import annotations

import re
from pathlib import Path

import pytest

from conftest import run_git
from review_agent.git_repo import ChangeSummary
from review_agent.incremental import (
    IncrementalPriorityMap,
    build_incremental_priority_map,
    build_incremental_priority_map_from_summary,
    classify_revision_change,
    deterministic_child_review_id,
    incremental_priority_from_dict,
    incremental_priority_to_dict,
)
from review_agent.revision import RepositoryIdentity, ResolvedRevisions
from review_agent.session import (
    ReviewExecutionConfig,
    RevisionChangeKind,
    initial_session_manifest,
)


def _parent_manifest():
    return initial_session_manifest(
        review_id="review-parent",
        repository=RepositoryIdentity("C:/repo", "C:/repo/.git", None),
        revisions=ResolvedRevisions("main", "HEAD", "a" * 40, "b" * 40),
        execution=ReviewExecutionConfig(
            reviewer_provider="fake",
            reviewer_model=None,
            reviewer_base_url=None,
            reviewer_api_key_env="REVIEW_AGENT_API_KEY",
            reviewer_mode="single",
            reviewer_loop="single-shot",
            non_interactive=True,
        ),
        now="2026-07-12T00:00:00Z",
    )


@pytest.mark.parametrize(
    ("base_sha", "head_sha", "expected"),
    [
        ("a" * 40, "b" * 40, RevisionChangeKind.INITIAL),
        ("a" * 40, "c" * 40, RevisionChangeKind.HEAD_MOVED),
        ("c" * 40, "b" * 40, RevisionChangeKind.BASE_MOVED),
        ("c" * 40, "d" * 40, RevisionChangeKind.BASE_AND_HEAD_MOVED),
    ],
)
def test_classify_revision_change_covers_all_drift_kinds(
    base_sha: str,
    head_sha: str,
    expected: RevisionChangeKind,
) -> None:
    current = ResolvedRevisions("main", "HEAD", base_sha, head_sha)

    assert classify_revision_change(_parent_manifest(), current) is expected


def test_deterministic_child_review_id_is_stable_and_revision_sensitive() -> None:
    repository = RepositoryIdentity("C:/repo", "C:/repo/.git", None)
    revisions = ResolvedRevisions("main", "HEAD", "a" * 40, "c" * 40)

    first = deterministic_child_review_id(
        repository=repository,
        parent_review_id="review-parent",
        revisions=revisions,
    )
    repeated = deterministic_child_review_id(
        repository=repository,
        parent_review_id="review-parent",
        revisions=revisions,
    )
    different = deterministic_child_review_id(
        repository=repository,
        parent_review_id="review-parent",
        revisions=ResolvedRevisions("main", "HEAD", "a" * 40, "d" * 40),
    )

    assert first == repeated
    assert first != different
    assert re.fullmatch(r"review-[0-9a-f]{12}", first)


def test_incremental_priority_map_round_trips_strictly() -> None:
    priority = IncrementalPriorityMap(
        from_revision="b" * 40,
        to_revision="c" * 40,
        changed_files=["later.py"],
        diff_stat="1 file changed",
        diff_excerpt=["+value = 1"],
    )

    payload = incremental_priority_to_dict(priority)

    assert incremental_priority_from_dict(payload) == priority
    with pytest.raises(ValueError, match="missing"):
        incremental_priority_from_dict({"from_revision": "b" * 40})
    with pytest.raises(ValueError, match="unsupported fields"):
        incremental_priority_from_dict({**payload, "parent_observations": []})
    with pytest.raises(ValueError, match="list of strings"):
        incremental_priority_from_dict({**payload, "changed_files": [1]})
    with pytest.raises(ValueError, match="full Git object ID"):
        incremental_priority_from_dict({**payload, "from_revision": "main"})


def test_incremental_priority_builder_rejects_mixed_revision_summary() -> None:
    summary = ChangeSummary(
        repository_path="C:/repo",
        base_revision="a" * 40,
        head_revision="b" * 40,
        changed_files=["a.py"],
        diff_stat="1 file changed",
        diff_excerpt=["+change"],
    )

    with pytest.raises(ValueError, match="must match"):
        build_incremental_priority_map_from_summary(
            summary,
            from_revision="c" * 40,
            to_revision="b" * 40,
        )


def test_build_incremental_priority_map_reads_only_requested_range(
    git_repo: Path,
) -> None:
    from_revision = run_git(git_repo, "rev-parse", "HEAD")
    (git_repo / "later.py").write_text("value = 1\n", encoding="utf-8")
    run_git(git_repo, "add", "later.py")
    run_git(git_repo, "commit", "-m", "add later change")
    to_revision = run_git(git_repo, "rev-parse", "HEAD")

    priority = build_incremental_priority_map(
        git_repo,
        from_revision=from_revision,
        to_revision=to_revision,
    )

    assert priority.from_revision == from_revision
    assert priority.to_revision == to_revision
    assert priority.changed_files == ["later.py"]
    assert any("value = 1" in line for line in priority.diff_excerpt)
