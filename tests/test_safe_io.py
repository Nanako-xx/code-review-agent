from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import review_agent.safe_io as safe_io
from review_agent.safe_io import (
    SafeIOError,
    assert_regular_file,
    atomic_replace_bytes,
    canonical_json_bytes,
    canonical_relative_path,
    cleanup_staging_files,
    ensure_secure_directory,
    metadata_is_reparse_point,
    publish_create_only_bytes,
    read_strict_json,
    read_verified_bytes,
    resolve_managed_path,
)


def test_canonical_json_is_stable_utf8_and_strict_json_rejects_duplicates(
    tmp_path: Path,
) -> None:
    payload = {"z": "缓存", "a": [1, True, None]}
    expected = '{"a":[1,true,null],"z":"缓存"}'.encode("utf-8")
    path = tmp_path / "payload.json"

    assert canonical_json_bytes(payload) == expected
    path.write_bytes(expected)
    assert read_strict_json(path) == {"a": [1, True, None], "z": "缓存"}

    path.write_text('{"a":1,"a":2}', encoding="utf-8")
    with pytest.raises(SafeIOError, match="duplicate JSON key"):
        read_strict_json(path)


@pytest.mark.parametrize(
    "relative_path",
    [
        "",
        " file.txt",
        "file.txt ",
        ".",
        "..",
        "../file.txt",
        "a/../file.txt",
        "a/./file.txt",
        "/absolute/file.txt",
        "C:/absolute/file.txt",
        "a\\file.txt",
        "a//file.txt",
        "a/file.txt/",
        "a/file.txt:stream",
        "a/file.txt.",
        "a/CON.txt",
        "a/*.txt",
        "a/\x00.txt",
    ],
)
def test_canonical_relative_path_rejects_traversal_ads_and_windows_aliases(
    relative_path: str,
) -> None:
    with pytest.raises(SafeIOError, match="path"):
        canonical_relative_path(relative_path)


def test_managed_paths_reject_case_aliases_and_symbolic_link_components(
    tmp_path: Path,
) -> None:
    root = ensure_secure_directory(tmp_path / "root")
    (root / "ExactCase").mkdir()

    assert resolve_managed_path(root, "ExactCase/result.json") == (
        root / "ExactCase" / "result.json"
    )
    with pytest.raises(SafeIOError, match="case alias"):
        resolve_managed_path(root, "exactcase/result.json")

    outside = tmp_path / "outside"
    outside.mkdir()
    link = root / "linked"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symbolic links are unavailable for this Windows identity")

    with pytest.raises(SafeIOError, match="link|reparse"):
        resolve_managed_path(root, "linked/escape.json")


def test_secure_root_rejects_extended_namespace_and_reparse_metadata(
    tmp_path: Path,
) -> None:
    with pytest.raises(SafeIOError, match="extended-length"):
        ensure_secure_directory(Path(r"\\?\C:\ra"))

    assert metadata_is_reparse_point(
        SimpleNamespace(st_file_attributes=0x400)
    )
    assert not metadata_is_reparse_point(
        SimpleNamespace(st_file_attributes=0)
    )


def test_regular_file_check_rejects_windows_reparse_points_deterministically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "junction-target"
    candidate.write_bytes(b"data")
    metadata = SimpleNamespace(st_mode=0o100644, st_file_attributes=0x400)
    monkeypatch.setattr(safe_io.os, "lstat", lambda _path: metadata)

    with pytest.raises(SafeIOError, match="reparse point"):
        assert_regular_file(candidate)


def test_atomic_replace_never_exposes_the_staged_content_before_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "state.json"
    atomic_replace_bytes(destination, b"old")
    observed_before_replace: list[bytes] = []
    staged_paths: list[Path] = []
    real_replace = safe_io.os.replace

    def recording_replace(source: object, target: object) -> None:
        staged_paths.append(Path(source))
        observed_before_replace.append(destination.read_bytes())
        assert Path(source).read_bytes() == b"new"
        real_replace(source, target)

    monkeypatch.setattr(safe_io.os, "replace", recording_replace)
    atomic_replace_bytes(destination, b"new")

    assert observed_before_replace == [b"old"]
    assert destination.read_bytes() == b"new"
    assert all(not path.exists() for path in staged_paths)


def test_create_only_publication_cannot_overwrite_existing_content(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.json"
    publish_create_only_bytes(destination, b"first")

    with pytest.raises(SafeIOError, match="already exists"):
        publish_create_only_bytes(destination, b"second")

    assert destination.read_bytes() == b"first"


def test_regular_file_and_hash_verification_fail_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    content = b"authoritative artifact"
    artifact.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()

    assert assert_regular_file(artifact) == artifact
    assert read_verified_bytes(artifact, digest) == content

    artifact.write_bytes(b"tampered")
    with pytest.raises(SafeIOError, match="hash"):
        read_verified_bytes(artifact, digest)

    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(SafeIOError, match="regular file"):
        assert_regular_file(directory)


def test_interrupted_staging_cleanup_removes_only_owned_regular_staging_files(
    tmp_path: Path,
) -> None:
    stage = tmp_path / (".stage-" + "a" * 32 + ".tmp")
    unrelated = tmp_path / ".stage-user-file.tmp"
    stage.write_bytes(b"partial")
    unrelated.write_bytes(b"keep")

    removed = cleanup_staging_files(tmp_path)

    assert removed == (stage,)
    assert not stage.exists()
    assert unrelated.read_bytes() == b"keep"


def test_interrupted_staging_cleanup_refuses_a_link_disguised_as_staging(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"keep")
    stage = tmp_path / (".stage-" + "b" * 32 + ".tmp")
    try:
        stage.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable for this Windows identity")

    with pytest.raises(SafeIOError, match="staging.*regular"):
        cleanup_staging_files(tmp_path)

    assert stage.is_symlink()
    assert target.read_bytes() == b"keep"
