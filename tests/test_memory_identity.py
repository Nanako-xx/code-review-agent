from __future__ import annotations

from dataclasses import FrozenInstanceError
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
    RepositoryIdentityCore,
    RepositoryIdentityDescriptor,
    VerifiedRepositoryIdentity,
    build_path_budget_report,
    build_relink_descriptor,
    build_repository_identity_core,
    build_repository_identity_descriptor,
    build_repository_memory_namespace,
    hydrate_repository_identity_descriptor,
    materialize_repository_memory_namespace,
    plan_repository_memory_namespace,
    repository_identity_core_hash,
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


def test_descriptor_constructor_and_strict_hydration_reject_forged_key(
    git_repo: Path,
) -> None:
    identity = RevisionResolver().repository_identity(git_repo)
    descriptor = build_repository_identity_descriptor(identity)
    payload = descriptor.to_payload()
    payload["repository_key"] = "0" * 64

    with pytest.raises(MemoryIdentityError, match="does not match identity core"):
        RepositoryIdentityDescriptor(
            repository_key="0" * 64,
            canonical_path=identity.canonical_path,
            git_common_dir=identity.git_common_dir,
            origin_url=identity.origin_url,
        )
    with pytest.raises(MemoryIdentityError, match="does not match identity core"):
        hydrate_repository_identity_descriptor(payload)
    with pytest.raises(MemoryIdentityError, match="does not match identity core"):
        VerifiedRepositoryIdentity.from_payload(payload)


def test_strict_descriptor_hydration_rejects_forged_core(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    descriptor = build_repository_identity_descriptor(
        RevisionResolver().repository_identity(git_repo)
    )
    payload = descriptor.to_payload()
    payload["git_common_dir"] = str((tmp_path / "forged.git").resolve())

    with pytest.raises(MemoryIdentityError, match="does not match identity core"):
        RepositoryIdentityDescriptor.from_payload(payload)


def test_identity_core_hash_is_canonical_and_location_independent(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    worktree = tmp_path / "identity-core-linked-worktree"
    run_git(
        git_repo,
        "worktree",
        "add",
        "-b",
        "memory-identity-core-worktree",
        str(worktree),
        "HEAD",
    )
    resolver = RevisionResolver()
    primary = build_repository_identity_descriptor(
        resolver.repository_identity(git_repo)
    )
    linked = build_repository_identity_descriptor(
        resolver.repository_identity(worktree)
    )

    assert isinstance(primary.core, RepositoryIdentityCore)
    assert primary.core == linked.core
    assert primary == linked
    assert primary.canonical_path != linked.canonical_path
    assert primary.repository_key == linked.repository_key
    assert repository_identity_core_hash(primary.core) == primary.repository_key
    assert build_repository_identity_core(
        resolver.repository_identity(worktree)
    ) == primary.core


def test_verified_identity_hydration_preserves_canonical_core(
    git_repo: Path,
) -> None:
    descriptor = build_repository_identity_descriptor(
        RevisionResolver().repository_identity(git_repo)
    )

    verified = VerifiedRepositoryIdentity.from_payload(descriptor.to_payload())

    assert verified.descriptor == descriptor
    assert verified.core == descriptor.core
    assert verified.core_hash == descriptor.repository_key
    assert verified.repository_key == descriptor.repository_key


@pytest.mark.parametrize("relation", ["equal", "ancestor", "descendant"])
def test_namespace_planner_rejects_worktree_overlap_in_every_direction(
    git_repo: Path,
    relation: str,
) -> None:
    identity = RevisionResolver().repository_identity(git_repo)
    if relation == "equal":
        root = git_repo
    elif relation == "ancestor":
        root = git_repo.parent
    else:
        root = git_repo / "memory-child"

    with pytest.raises(MemoryIdentityError, match="overlaps protected repository paths"):
        plan_repository_memory_namespace(identity, root)


def test_namespace_planner_rejects_git_common_git_dir_and_review_agent_paths(
    git_repo: Path,
) -> None:
    resolver = RevisionResolver()
    identity = resolver.repository_identity(git_repo)
    layout = resolver.repository_layout(git_repo)
    protected_roots = {
        Path(identity.git_common_dir),
        *(Path(path) for path in layout.git_dirs),
        git_repo / ".git",
        git_repo / ".review-agent",
    }

    for protected in protected_roots:
        for root in (protected, protected / "memory-child"):
            with pytest.raises(
                MemoryIdentityError,
                match="overlaps protected repository paths",
            ):
                plan_repository_memory_namespace(identity, root)


def test_namespace_planner_protects_every_linked_worktree_and_real_git_dir(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    linked = tmp_path / "planner-linked-worktree"
    run_git(
        git_repo,
        "worktree",
        "add",
        "-b",
        "memory-planner-linked-worktree",
        str(linked),
        "HEAD",
    )
    identity = RevisionResolver().repository_identity(git_repo)
    layout = RevisionResolver().repository_layout(git_repo)
    linked_git_dirs = [
        Path(path)
        for path in layout.git_dirs
        if Path(path) != Path(identity.git_common_dir)
    ]

    assert linked_git_dirs
    for root in (
        linked,
        linked / ".git",
        linked / ".review-agent",
        linked_git_dirs[0],
    ):
        with pytest.raises(MemoryIdentityError, match="overlaps protected"):
            plan_repository_memory_namespace(identity, root)


def test_namespace_planner_rejects_symlink_alias_without_creating_storage(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    alias = tmp_path / "repository-alias"
    try:
        alias.symlink_to(git_repo, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")

    requested = alias / "memory"
    with pytest.raises(MemoryIdentityError, match="symbolic link or reparse point"):
        plan_repository_memory_namespace(
            RevisionResolver().repository_identity(git_repo),
            requested,
        )

    assert not (git_repo / "memory").exists()


def test_namespace_planner_rejects_unrelated_symlink_escape(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "memory-alias"
    try:
        alias.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks unavailable: {error}")

    with pytest.raises(MemoryIdentityError, match="symbolic link or reparse point"):
        plan_repository_memory_namespace(
            RevisionResolver().repository_identity(git_repo),
            alias / "memory",
        )

    assert list(outside.iterdir()) == []


def _utf16_units(value: str) -> int:
    return len(value.encode("utf-16-le")) // 2


def _windows_path_with_units(target_units: int) -> Path:
    prefix = "C:\\"
    assert target_units >= _utf16_units(prefix)
    return Path(prefix + ("a" * (target_units - _utf16_units(prefix))))


def test_windows_path_budget_accepts_exact_file_boundary_and_rejects_plus_one(
) -> None:
    baseline_namespace = Path("C:\\m")
    baseline = build_path_budget_report(
        baseline_namespace,
        platform_name="win32",
    )
    namespace_units = _utf16_units(str(baseline_namespace))
    file_suffix_units = baseline.maximum_file_units - namespace_units
    exact_namespace = _windows_path_with_units(259 - file_suffix_units)

    exact = build_path_budget_report(exact_namespace, platform_name="win32")
    over = build_path_budget_report(
        _windows_path_with_units(_utf16_units(str(exact_namespace)) + 1),
        platform_name="win32",
    )

    assert exact.enforced is True
    assert exact.directory_limit == 247
    assert exact.file_limit == 259
    assert exact.maximum_file_units == 259
    assert exact.within_budget is True
    assert over.maximum_file_units == 260
    assert over.within_budget is False


def test_windows_path_budget_counts_non_bmp_characters_as_two_utf16_units() -> None:
    ascii_namespace = Path("C:\\" + ("a" * 100))
    emoji_namespace = Path("C:\\" + ("a" * 99) + "\U0001f600")

    ascii_report = build_path_budget_report(
        ascii_namespace,
        platform_name="win32",
    )
    emoji_report = build_path_budget_report(
        emoji_namespace,
        platform_name="win32",
    )

    assert len(str(ascii_namespace)) == len(str(emoji_namespace))
    assert emoji_report.maximum_file_units == ascii_report.maximum_file_units + 1
    assert emoji_report.namespace_units == ascii_report.namespace_units + 1


def test_non_windows_path_budget_is_auditable_but_not_enforced() -> None:
    namespace = Path.cwd().anchor + ("long" * 100)

    report = build_path_budget_report(Path(namespace), platform_name="linux")

    assert report.enforced is False
    assert report.entries
    assert report.windows_compatible is False
    assert report.within_budget is True
    assert all(entry.utf16_code_units > 0 for entry in report.entries)


def test_planning_path_budget_failure_creates_no_root_and_hides_full_path(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    first_missing = tmp_path / ("x" * 80)
    root = first_missing / ("y" * 80) / ("z" * 80)
    identity = RevisionResolver().repository_identity(git_repo)

    with pytest.raises(MemoryIdentityError, match="bounded Windows path policy") as captured:
        plan_repository_memory_namespace(identity, root, platform_name="win32")

    assert str(root) not in str(captured.value)
    assert not first_missing.exists()


def test_planner_is_immutable_and_materialization_is_the_only_creation_step(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    root = tmp_path / "planned-memory"
    identity = RevisionResolver().repository_identity(git_repo)

    plan = plan_repository_memory_namespace(identity, root)

    assert not root.exists()
    assert plan.path_budget.entries
    assert plan.locator.identity.repository_key == plan.repository_key
    with pytest.raises(FrozenInstanceError):
        plan.memory_root = "changed"  # type: ignore[misc]

    namespace = materialize_repository_memory_namespace(plan)

    assert namespace == plan.namespace
    assert Path(namespace.memory_root).is_dir()
    assert Path(namespace.namespace_path).is_dir()
    assert not (Path(namespace.namespace_path) / "memory.sqlite3").exists()


def test_legacy_direct_namespace_builder_remains_unbound_and_non_materializing(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    root = tmp_path / "direct-memory"

    namespace = build_repository_memory_namespace(
        RevisionResolver().repository_identity(git_repo),
        root,
    )

    assert Path(namespace.memory_root) == root.resolve()
    assert not root.exists()


def test_planner_errors_and_plan_repr_never_expose_origin_credentials(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    raw_origin = (
        "https://alice:top-secret@example.test/acme/repo.git"
        "?access_token=also-secret#credential"
    )
    run_git(git_repo, "remote", "add", "origin", raw_origin)
    identity = RevisionResolver().repository_identity(git_repo)
    plan = plan_repository_memory_namespace(identity, tmp_path / "safe-memory")

    with pytest.raises(MemoryIdentityError) as captured:
        plan_repository_memory_namespace(identity, git_repo)

    rendered = repr(plan) + str(captured.value)
    for secret in ("alice", "top-secret", "also-secret", "credential"):
        assert secret not in rendered
