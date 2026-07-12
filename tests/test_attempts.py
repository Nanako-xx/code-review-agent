from __future__ import annotations

from pathlib import Path

import pytest

from review_agent.attempts import AttemptWorkspace
from review_agent.run_state import RunPhase


def test_attempt_workspace_uses_phase_and_reviewer_isolation(tmp_path: Path) -> None:
    workspace = AttemptWorkspace(tmp_path, RunPhase.REVIEWERS, 2, reviewer_index=1)

    assert workspace.prepare() == (
        tmp_path / "attempts" / "reviewers" / "2" / "reviewer-1"
    )


def test_attempt_workspace_writes_then_atomically_promotes_file(tmp_path: Path) -> None:
    workspace = AttemptWorkspace(tmp_path, RunPhase.PREFLIGHT, 1)
    workspace.write_json("artifacts/intent.json", {"goal": "safe resume"})

    promoted = workspace.promote_file("artifacts/intent.json", "intent.json")

    assert promoted == tmp_path / "intent.json"
    assert promoted.read_text(encoding="utf-8").endswith("}")
    assert not list(tmp_path.glob(".promote-*"))


def test_failed_attempt_never_overwrites_authoritative_artifact(tmp_path: Path) -> None:
    authoritative = tmp_path / "intent.json"
    authoritative.write_text("old", encoding="utf-8")
    workspace = AttemptWorkspace(tmp_path, RunPhase.PREFLIGHT, 2)
    workspace.write_text("intent.json", "incomplete")

    assert authoritative.read_text(encoding="utf-8") == "old"


@pytest.mark.parametrize(
    "relative_path",
    ["../escape.json", "nested/../../escape.json", "/absolute.json", "C:/escape.json", "a\\b.json"],
)
def test_attempt_workspace_rejects_unsafe_paths(
    tmp_path: Path,
    relative_path: str,
) -> None:
    workspace = AttemptWorkspace(tmp_path, RunPhase.PREFLIGHT, 1)

    with pytest.raises(ValueError, match="path|inside|managed"):
        workspace.write_text(relative_path, "unsafe")


def test_attempt_workspace_rejects_runtime_managed_destinations(tmp_path: Path) -> None:
    workspace = AttemptWorkspace(tmp_path, RunPhase.PREFLIGHT, 1)
    workspace.write_text("result.json", "{}")

    with pytest.raises(ValueError, match="runtime-managed"):
        workspace.promote_file("result.json", "session.json")


def test_reviewer_index_is_only_valid_for_reviewer_phase(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="reviewers phase"):
        AttemptWorkspace(tmp_path, RunPhase.PREFLIGHT, 1, reviewer_index=0)


def test_attempt_workspace_rejects_symlink_source(tmp_path: Path) -> None:
    workspace = AttemptWorkspace(tmp_path, RunPhase.PREFLIGHT, 1)
    workspace.prepare()
    target = tmp_path / "outside.json"
    target.write_text("{}", encoding="utf-8")
    link = workspace.path / "link.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    with pytest.raises(ValueError, match="regular file|outside"):
        workspace.promote_file("link.json", "result.json")
