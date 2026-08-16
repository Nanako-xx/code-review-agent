from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess

import pytest

import review_agent.diff_artifact as diff_artifact_module
from conftest import run_git
from review_agent.diff_artifact import (
    DiffArtifactIntegrityError,
    DiffArtifactStore,
    parse_diff_patch,
    validate_diff_index,
)
from review_agent.git_repo import collect_complete_diff_bytes
from review_agent.pr_workspace import PRMetadata, PRWorkspaceStore, SnapshotWorkspace
from review_agent.revision import RepositoryIdentity


def _init_repository(path: Path, *, object_format: str = "sha1") -> Path:
    path.mkdir()
    command = ["git", "init"]
    if object_format == "sha256":
        command.append("--object-format=sha256")
    result = subprocess.run(
        command,
        cwd=path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        if object_format == "sha256":
            pytest.skip("installed Git does not support SHA-256 repositories")
        raise AssertionError(result.stderr)
    run_git(path, "config", "user.email", "review-agent@example.test")
    run_git(path, "config", "user.name", "Review Agent")
    run_git(path, "config", "core.filemode", "true")
    return path


def _complex_repository(tmp_path: Path) -> tuple[Path, str, str]:
    repo = _init_repository(tmp_path / "repo")
    (repo / "modify.txt").write_text("one\ntwo\nthree\n", encoding="utf-8")
    (repo / "delete.txt").write_text("remove me\n", encoding="utf-8")
    (repo / "rename-old.txt").write_text("stable rename payload\n", encoding="utf-8")
    (repo / "copy-source.txt").write_text("stable copy payload\n", encoding="utf-8")
    (repo / "mode.sh").write_text("#!/bin/sh\necho mode\n", encoding="utf-8")
    (repo / "no-newline.txt").write_bytes(b"old-without-newline")
    (repo / ".gitattributes").write_text("crlf.txt -text\n", encoding="utf-8")
    (repo / "crlf.txt").write_bytes(b"first\r\nsecond\r\n")
    unicode_dir = repo / "目录"
    unicode_dir.mkdir()
    (unicode_dir / "文件.txt").write_text("旧值\n", encoding="utf-8")
    (repo / "binary.bin").write_bytes(b"\x00\x01old-binary\xff")
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-m", "base")
    base = run_git(repo, "rev-parse", "HEAD")

    (repo / "modify.txt").write_text(
        "one\ntwo changed\nthree\nfour\n", encoding="utf-8"
    )
    (repo / "add.txt").write_text("new file\n", encoding="utf-8")
    (repo / "delete.txt").unlink()
    (repo / "rename-old.txt").rename(repo / "rename-new.txt")
    (repo / "copy-target.txt").write_bytes((repo / "copy-source.txt").read_bytes())
    (repo / "no-newline.txt").write_bytes(b"new-without-newline")
    (repo / "crlf.txt").write_bytes(b"first\r\nsecond changed\r\n")
    (unicode_dir / "文件.txt").write_text("新值\n", encoding="utf-8")
    (repo / "binary.bin").write_bytes(b"\x00\x01new-binary\xfe\xff")
    run_git(repo, "add", "-A")
    run_git(repo, "update-index", "--chmod=+x", "mode.sh")
    run_git(repo, "commit", "-m", "complex head")
    head = run_git(repo, "rev-parse", "HEAD")
    return repo, base, head


def _snapshot_store(
    tmp_path: Path,
    repo: Path,
    base: str,
    head: str,
) -> tuple[PRWorkspaceStore, SnapshotWorkspace]:
    workspace_store = PRWorkspaceStore(tmp_path / "ra")
    identity = RepositoryIdentity(
        canonical_path=str(repo.resolve()),
        git_common_dir=str((repo / ".git").resolve()),
        origin_url=None,
    )
    workspace = workspace_store.create_or_load_workspace(
        workspace_store.resolve_pr(identity, "local", "diff-test"),
        PRMetadata(base_ref=base, head_ref=head),
    )
    snapshot = workspace_store.create_or_load_snapshot(workspace, base, head)
    return workspace_store, snapshot


def test_complete_diff_is_persisted_byte_for_byte_and_fully_indexed(
    tmp_path: Path,
) -> None:
    repo, base, head = _complex_repository(tmp_path)
    workspace_store, snapshot = _snapshot_store(tmp_path, repo, base, head)
    store = DiffArtifactStore(workspace_store)

    artifact = store.materialize(repo, snapshot)
    expected_patch = collect_complete_diff_bytes(repo, base, head)

    assert artifact.patch.path.read_bytes() == expected_patch
    assert artifact.index.diff_size_bytes == len(expected_patch)
    assert artifact.index.diff_sha256 == artifact.patch.sha256
    assert artifact.index.patch_artifact_id == artifact.patch.artifact_id
    assert artifact.index.base_sha == base
    assert artifact.index.head_sha == head
    assert not list((snapshot.path / "DiffArtifact").glob(".stage-*.tmp"))

    files = {item.path: item for item in artifact.index.files}
    assert files["modify.txt"].status == "modify"
    assert files["modify.txt"].additions > 0
    assert files["modify.txt"].deletions > 0
    assert files["add.txt"].status == "add"
    assert files["delete.txt"].status == "delete"
    assert files["rename-new.txt"].status == "rename"
    assert files["rename-new.txt"].previous_path == "rename-old.txt"
    assert files["copy-target.txt"].status == "copy"
    assert files["copy-target.txt"].previous_path == "copy-source.txt"
    assert files["binary.bin"].binary is True
    assert files["mode.sh"].status == "modify"
    assert files["目录/文件.txt"].path == "目录/文件.txt"
    assert b"\\ No newline at end of file" in expected_patch
    assert b"+second changed\r\n" in expected_patch

    for file_entry in artifact.index.files:
        file_slice = store.read_file(artifact, file_entry.file_index)
        assert file_slice == expected_patch[file_entry.byte_start : file_entry.byte_end]
        assert file_slice.startswith(b"diff --git ")
        for hunk in file_entry.hunks:
            hunk_slice = store.read_hunk(
                artifact,
                file_entry.file_index,
                hunk.hunk_index,
            )
            assert hunk_slice == expected_patch[hunk.byte_start : hunk.byte_end]
            assert hunk_slice.startswith(b"@@ ")


def test_materialization_runs_one_authoritative_diff_and_reuses_it_for_indexing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, base, head = _complex_repository(tmp_path)
    workspace_store, snapshot = _snapshot_store(tmp_path, repo, base, head)
    calls: list[tuple[Path, str, str]] = []
    real_collect = diff_artifact_module.collect_complete_diff_bytes

    def recording_collect(path: Path, base_sha: str, head_sha: str) -> bytes:
        calls.append((path, base_sha, head_sha))
        return real_collect(path, base_sha, head_sha)

    monkeypatch.setattr(
        diff_artifact_module,
        "collect_complete_diff_bytes",
        recording_collect,
    )

    DiffArtifactStore(workspace_store).materialize(repo, snapshot)

    assert calls == [(repo, base, head)]


def test_materialization_rejects_a_repository_outside_the_snapshot_binding(
    tmp_path: Path,
) -> None:
    repo, base, head = _complex_repository(tmp_path)
    workspace_store, snapshot = _snapshot_store(tmp_path, repo, base, head)
    other = _init_repository(tmp_path / "other")

    with pytest.raises(DiffArtifactIntegrityError, match="repository identity"):
        DiffArtifactStore(workspace_store).materialize(other, snapshot)


def test_diff_index_offsets_and_content_hash_fail_closed_when_tampered(
    tmp_path: Path,
) -> None:
    repo, base, head = _complex_repository(tmp_path)
    workspace_store, snapshot = _snapshot_store(tmp_path, repo, base, head)
    store = DiffArtifactStore(workspace_store)
    artifact = store.materialize(repo, snapshot)
    patch = artifact.patch.path.read_bytes()

    first = artifact.index.files[0]
    bad_file = replace(first, byte_end=first.byte_end + 1)
    bad_index = replace(
        artifact.index,
        files=(bad_file, *artifact.index.files[1:]),
    )
    with pytest.raises(DiffArtifactIntegrityError, match="index|offset|replay"):
        validate_diff_index(bad_index, patch)

    artifact.patch.path.write_bytes(patch + b"tampered")
    with pytest.raises(DiffArtifactIntegrityError, match="hash|size bound"):
        store.read_file(artifact, 0)
    artifact.patch.path.write_bytes(patch)

    index_payload = json.loads(artifact.index_artifact.path.read_text("utf-8"))
    index_payload["files"][0]["byte_end"] += 1
    artifact.index_artifact.path.write_text(
        json.dumps(index_payload, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(DiffArtifactIntegrityError, match="hash|size bound"):
        store.load(snapshot)


def test_diff_pages_are_bounded_reconstructable_and_explicit_about_more_data(
    tmp_path: Path,
) -> None:
    repo = _init_repository(tmp_path / "repo")
    (repo / "large.txt").write_text("base\n", encoding="utf-8")
    run_git(repo, "add", "large.txt")
    run_git(repo, "commit", "-m", "base")
    base = run_git(repo, "rev-parse", "HEAD")
    (repo / "large.txt").write_text(
        "".join(f"line-{index:06d}-{'x' * 40}\n" for index in range(4000)),
        encoding="utf-8",
    )
    run_git(repo, "add", "large.txt")
    run_git(repo, "commit", "-m", "large")
    head = run_git(repo, "rev-parse", "HEAD")
    workspace_store, snapshot = _snapshot_store(tmp_path, repo, base, head)
    store = DiffArtifactStore(workspace_store)
    artifact = store.materialize(repo, snapshot)

    cursor = 0
    chunks: list[bytes] = []
    saw_more = False
    while True:
        page = store.read_page(artifact, cursor=cursor, max_bytes=50_000)
        assert len(page.data) <= 50_000
        assert page.cursor == cursor
        chunks.append(page.data)
        saw_more = saw_more or page.has_more
        if not page.has_more:
            assert page.next_cursor is None
            break
        assert page.next_cursor is not None
        cursor = page.next_cursor

    assert saw_more is True
    assert b"".join(chunks) == artifact.patch.path.read_bytes()


def test_sha256_git_repository_produces_a_bound_diff_artifact(
    tmp_path: Path,
) -> None:
    repo = _init_repository(tmp_path / "sha256-repo", object_format="sha256")
    (repo / "value.txt").write_text("base\n", encoding="utf-8")
    run_git(repo, "add", "value.txt")
    run_git(repo, "commit", "-m", "base")
    base = run_git(repo, "rev-parse", "HEAD")
    (repo / "value.txt").write_text("head\n", encoding="utf-8")
    run_git(repo, "add", "value.txt")
    run_git(repo, "commit", "-m", "head")
    head = run_git(repo, "rev-parse", "HEAD")
    workspace_store, snapshot = _snapshot_store(tmp_path, repo, base, head)

    artifact = DiffArtifactStore(workspace_store).materialize(repo, snapshot)

    assert len(base) == len(head) == 64
    assert artifact.index.base_sha == base
    assert artifact.index.head_sha == head
    assert len(artifact.index.files) == 1


def test_submodule_patch_is_marked_without_a_second_git_query() -> None:
    old_commit = "1" * 40
    new_commit = "2" * 40
    patch = (
        "diff --git a/vendor b/vendor\n"
        f"index {old_commit}..{new_commit} 160000\n"
        "--- a/vendor\n"
        "+++ b/vendor\n"
        "@@ -1 +1 @@\n"
        f"-Subproject commit {old_commit}\n"
        f"+Subproject commit {new_commit}\n"
    ).encode("ascii")

    index = parse_diff_patch(
        patch,
        snapshot_id="S-" + "a" * 64,
        base_sha="b" * 40,
        head_sha="c" * 40,
        patch_artifact_id="A-" + "d" * 64,
    )

    assert len(index.files) == 1
    assert index.files[0].submodule is True
    assert index.files[0].additions == index.files[0].deletions == 1


def test_unquoted_git_header_with_spaces_is_parsed_without_splitting_path() -> None:
    path = "src/Gui/Stylesheets/FreeCAD Dark.qss"
    patch = (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    ).encode("utf-8")

    index = parse_diff_patch(
        patch,
        snapshot_id="S-" + "a" * 64,
        base_sha="b" * 40,
        head_sha="c" * 40,
        patch_artifact_id="A-" + "d" * 64,
    )

    assert len(index.files) == 1
    assert index.files[0].path == path
    assert index.files[0].previous_path is None
