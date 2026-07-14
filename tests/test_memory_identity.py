from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import pytest

from conftest import run_git
from review_agent.memory_identity import (
    MemoryIdentityError,
    MemoryRootResolver,
    MemoryRootSource,
    build_relink_descriptor,
    build_repository_memory_namespace,
    repository_key,
    repository_namespace_path,
    resolve_memory_root,
)
from review_agent.revision import (
    RepositoryIdentity,
    RevisionResolver,
    normalize_repository_identity_path,
    normalize_repository_origin,
)


def _resolve_root(
    path: Path,
    *,
    env: Mapping[str, str] | None = None,
) -> Path:
    return resolve_memory_root(path, env=env or {})


def test_memory_root_cli_override_wins_over_environment(tmp_path: Path) -> None:
    cli_root = tmp_path / "cli-memory"
    env_root = tmp_path / "env-memory"

    resolution = MemoryRootResolver().resolve(
        cli_root,
        env={"REVIEW_AGENT_MEMORY_ROOT": str(env_root)},
    )

    assert resolution.source is MemoryRootSource.CLI_OVERRIDE
    assert resolution.path == str(cli_root.resolve())
    assert cli_root.is_dir()
    assert not env_root.exists()


def test_memory_root_environment_wins_over_platform_default(tmp_path: Path) -> None:
    env_root = tmp_path / "explicit-env-memory"
    local_app_data = tmp_path / "local-app-data"

    resolution = MemoryRootResolver().resolve(
        env={
            "REVIEW_AGENT_MEMORY_ROOT": str(env_root),
            "LOCALAPPDATA": str(local_app_data),
        },
        platform_name="win32",
        home=tmp_path / "home",
    )

    assert resolution.source is MemoryRootSource.ENVIRONMENT
    assert resolution.path == str(env_root.resolve())
    assert not local_app_data.exists()


@pytest.mark.parametrize(
    ("platform_name", "env_name", "suffix"),
    [
        ("win32", "LOCALAPPDATA", ("code-review-agent", "memory")),
        ("linux", "XDG_STATE_HOME", ("code-review-agent", "memory")),
    ],
)
def test_memory_root_uses_configured_platform_state_directory(
    tmp_path: Path,
    platform_name: str,
    env_name: str,
    suffix: tuple[str, ...],
) -> None:
    platform_state = tmp_path / f"{platform_name}-state"

    resolution = MemoryRootResolver().resolve(
        env={env_name: str(platform_state)},
        platform_name=platform_name,
        home=tmp_path / "home",
    )

    assert resolution.source is MemoryRootSource.PLATFORM_DEFAULT
    assert Path(resolution.path) == platform_state.joinpath(*suffix).resolve()


def test_memory_root_uses_linux_home_fallback_for_relative_xdg_state(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"

    resolution = MemoryRootResolver().resolve(
        env={"XDG_STATE_HOME": "relative-state"},
        platform_name="linux",
        home=home,
    )

    assert Path(resolution.path) == (
        home / ".local" / "state" / "code-review-agent" / "memory"
    ).resolve()


def test_memory_root_uses_macos_application_support(tmp_path: Path) -> None:
    home = tmp_path / "home"

    resolution = MemoryRootResolver().resolve(
        env={},
        platform_name="darwin",
        home=home,
    )

    assert Path(resolution.path) == (
        home
        / "Library"
        / "Application Support"
        / "code-review-agent"
        / "memory"
    ).resolve()


def test_memory_root_windows_falls_back_to_home_when_local_app_data_is_missing(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"

    resolution = MemoryRootResolver().resolve(
        env={},
        platform_name="win32",
        home=home,
    )

    assert Path(resolution.path) == (
        home / "AppData" / "Local" / "code-review-agent" / "memory"
    ).resolve()


@pytest.mark.parametrize("value", ["relative/path", "", "   "])
def test_explicit_memory_root_must_be_a_non_empty_absolute_path(
    value: str,
) -> None:
    with pytest.raises(MemoryIdentityError, match="absolute path"):
        resolve_memory_root(value, env={})


def test_environment_memory_root_must_be_absolute() -> None:
    with pytest.raises(MemoryIdentityError, match="absolute path"):
        resolve_memory_root(None, env={"REVIEW_AGENT_MEMORY_ROOT": "relative"})


def test_memory_root_is_canonical_and_rejects_an_existing_file(
    tmp_path: Path,
) -> None:
    root = tmp_path / "parent" / "memory"
    resolved = _resolve_root(root)

    assert resolved.is_absolute()
    assert resolved == root.resolve()

    file_root = tmp_path / "not-a-directory"
    file_root.write_text("not a memory root", encoding="utf-8")
    with pytest.raises(MemoryIdentityError, match="not a directory"):
        _resolve_root(file_root)


def test_memory_root_reports_unsafe_parent_creation_without_traceback_details(
    tmp_path: Path,
) -> None:
    blocking_file = tmp_path / "blocking-parent"
    blocking_file.write_text("file", encoding="utf-8")

    with pytest.raises(MemoryIdentityError, match="cannot safely create"):
        _resolve_root(blocking_file / "memory")


def test_repository_key_uses_normalized_common_dir_and_sanitized_origin(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    run_git(
        git_repo,
        "remote",
        "add",
        "origin",
        "https://user:token@Example.Test/acme/review-target.git?secret=yes",
    )
    identity = RevisionResolver().repository_identity(git_repo)

    actual = repository_key(identity)
    expected_material = (
        normalize_repository_identity_path(identity.git_common_dir)
        + "\0"
        + (normalize_repository_origin(identity.origin_url) or "")
    )

    assert actual == hashlib.sha256(expected_material.encode("utf-8")).hexdigest()
    assert len(actual) == 64
    assert actual == actual.casefold()


def test_worktrees_with_the_same_git_common_dir_share_namespace(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "linked-worktree"
    run_git(
        git_repo,
        "worktree",
        "add",
        "-b",
        "memory-identity-worktree",
        str(worktree),
        "HEAD",
    )
    root = _resolve_root(tmp_path / "memory")

    primary = build_repository_memory_namespace(
        RevisionResolver().repository_identity(git_repo),
        root,
    )
    linked = build_repository_memory_namespace(
        RevisionResolver().repository_identity(worktree),
        root,
    )

    assert primary.repository_key == linked.repository_key
    assert primary.namespace_path == linked.namespace_path
    assert primary.metadata.canonical_path != linked.metadata.canonical_path
    assert primary.metadata.git_common_dir == linked.metadata.git_common_dir


def test_independent_clones_with_the_same_origin_remain_isolated(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    clone = tmp_path / "independent-clone"
    run_git(tmp_path, "clone", str(git_repo), str(clone))
    origin = "https://example.test/acme/review-target.git"
    run_git(git_repo, "remote", "add", "origin", origin)
    run_git(clone, "remote", "set-url", "origin", origin)
    root = _resolve_root(tmp_path / "memory")

    original = build_repository_memory_namespace(
        RevisionResolver().repository_identity(git_repo),
        root,
    )
    cloned = build_repository_memory_namespace(
        RevisionResolver().repository_identity(clone),
        root,
    )

    assert original.metadata.origin_url == cloned.metadata.origin_url == origin
    assert original.metadata.git_common_dir != cloned.metadata.git_common_dir
    assert original.repository_key != cloned.repository_key
    assert original.namespace_path != cloned.namespace_path


def test_repository_without_origin_has_a_stable_key(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    identity = RevisionResolver().repository_identity(git_repo)
    root = _resolve_root(tmp_path / "memory")

    first = build_repository_memory_namespace(identity, root)
    second = build_repository_memory_namespace(identity, root)

    assert first == second
    assert first.metadata.origin_url is None


def test_repository_metadata_defensively_redacts_raw_origin_credentials(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    raw_identity = RevisionResolver().repository_identity(git_repo)
    identity = RepositoryIdentity(
        canonical_path=raw_identity.canonical_path,
        git_common_dir=raw_identity.git_common_dir,
        origin_url=(
            "https://alice:top-secret@example.test/acme/repo.git"
            "?access_token=also-secret#credential"
        ),
    )

    namespace = build_repository_memory_namespace(
        identity,
        _resolve_root(tmp_path / "memory"),
    )
    serialized = json.dumps(namespace.metadata.to_payload(), sort_keys=True)
    rendered = repr(namespace)

    assert namespace.metadata.origin_url == "https://example.test/acme/repo.git"
    for secret in ("alice", "top-secret", "also-secret", "credential"):
        assert secret not in serialized
        assert secret not in rendered


@pytest.mark.parametrize(
    "invalid_key",
    ["../escape", "A" * 64, "0" * 63, "0" * 65, "g" * 64],
)
def test_repository_namespace_accepts_only_fixed_lowercase_sha256_keys(
    tmp_path: Path,
    invalid_key: str,
) -> None:
    root = _resolve_root(tmp_path / "memory")

    with pytest.raises(MemoryIdentityError, match="repository key"):
        repository_namespace_path(root, invalid_key)


def test_repository_namespace_path_is_canonical_absolute_and_contained(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    root = _resolve_root(tmp_path / "memory")
    namespace = build_repository_memory_namespace(
        RevisionResolver().repository_identity(git_repo),
        root,
    )
    namespace_path = Path(namespace.namespace_path)

    assert namespace_path.is_absolute()
    assert namespace_path == (
        root / "repositories" / namespace.repository_key
    ).resolve()
    assert namespace_path.parent.parent == root


def test_repository_namespace_rejects_symlink_escape(tmp_path: Path) -> None:
    root = _resolve_root(tmp_path / "memory")
    outside = tmp_path / "outside"
    outside.mkdir()
    repositories = root / "repositories"
    try:
        repositories.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")

    with pytest.raises(MemoryIdentityError, match="symbolic link"):
        repository_namespace_path(root, "0" * 64)


def test_relink_descriptor_requires_explicit_old_and_new_identities(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    clone = tmp_path / "replacement-clone"
    run_git(tmp_path, "clone", str(git_repo), str(clone))
    origin = "https://example.test/acme/review-target.git"
    run_git(git_repo, "remote", "add", "origin", origin)
    run_git(clone, "remote", "set-url", "origin", origin)
    root = _resolve_root(tmp_path / "memory")
    old = build_repository_memory_namespace(
        RevisionResolver().repository_identity(git_repo),
        root,
    )
    new = build_repository_memory_namespace(
        RevisionResolver().repository_identity(clone),
        root,
    )

    descriptor = build_relink_descriptor(old.metadata, new.metadata)
    payload = descriptor.to_payload()

    assert descriptor.operation == "explicit_relink"
    assert payload["old_identity"]["repository_key"] == old.repository_key
    assert payload["new_identity"]["repository_key"] == new.repository_key
    assert old.metadata.origin_url == new.metadata.origin_url
    assert old.repository_key != new.repository_key


def test_relink_descriptor_rejects_same_repository_namespace(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    namespace = build_repository_memory_namespace(
        RevisionResolver().repository_identity(git_repo),
        _resolve_root(tmp_path / "memory"),
    )

    with pytest.raises(MemoryIdentityError, match="different repository keys"):
        build_relink_descriptor(namespace.metadata, namespace.metadata)
