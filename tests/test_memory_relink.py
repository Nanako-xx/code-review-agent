from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
from typing import Dict

import pytest

from conftest import run_git
from review_agent.memory_identity import (
    RepositoryIdentityDescriptor,
    build_repository_identity_descriptor,
    repository_namespace_path,
)
from review_agent.memory_models import canonical_sha256, stable_request_id
from review_agent.memory_relink import (
    PreparedRepositoryRelink,
    RepositoryAuthorityResolution,
    RepositoryRelinkConflictError,
    RepositoryRelinkError,
    RepositoryRelinkIntegrityError,
    RepositoryRelinkRegistry,
    RepositoryRelinkRequestReceipt,
    RepositoryRelinkResult,
    RepositoryRelinkValidationError,
    apply_relink,
    get_repository_relink_receipt,
    prepare_relink,
    resolve_repository_authority,
)
from review_agent.memory_store import MemoryStore
from review_agent.revision import RepositoryIdentity, RepositoryLayout, RevisionResolver


_SHARED_ORIGIN = "https://example.test/acme/review-target.git"


class _MutableRevisionResolver:
    """Small deterministic live-locator double; authority descriptors stay offline."""

    def __init__(self, parent: Path) -> None:
        self._parent = parent
        self._identities: Dict[str, RepositoryIdentity] = {}
        self.queries: list[str] = []

    def add(
        self,
        name: str,
        *,
        origin: str | None = _SHARED_ORIGIN,
    ) -> RepositoryIdentityDescriptor:
        worktree = self._parent / name
        git_common_dir = worktree / ".git"
        git_common_dir.mkdir(parents=True)
        identity = RepositoryIdentity(
            canonical_path=str(worktree.resolve()),
            git_common_dir=str(git_common_dir.resolve()),
            origin_url=origin,
        )
        self._identities[str(worktree.resolve())] = identity
        return build_repository_identity_descriptor(identity)

    def repository_identity(self, repository: Path) -> RepositoryIdentity:
        key = str(Path(repository).resolve())
        self.queries.append(key)
        try:
            return self._identities[key]
        except KeyError:
            raise ValueError("repository locator is unavailable") from None

    def add_worktree(
        self,
        name: str,
        shared_identity: RepositoryIdentityDescriptor,
    ) -> RepositoryIdentityDescriptor:
        worktree = self._parent / name
        worktree.mkdir(parents=True)
        identity = RepositoryIdentity(
            canonical_path=str(worktree.resolve()),
            git_common_dir=shared_identity.git_common_dir,
            origin_url=shared_identity.origin_url,
        )
        self._identities[str(worktree.resolve())] = identity
        return build_repository_identity_descriptor(identity)

    def repository_layout(self, repository: Path) -> RepositoryLayout:
        identity = self.repository_identity(repository)
        return RepositoryLayout(
            git_common_dir=identity.git_common_dir,
            worktree_paths=(identity.canonical_path,),
            git_dirs=(identity.git_common_dir,),
        )

    def set_origin(
        self,
        identity: RepositoryIdentityDescriptor,
        origin: str | None,
    ) -> None:
        key = str(Path(identity.canonical_path).resolve())
        current = self._identities[key]
        self._identities[key] = RepositoryIdentity(
            canonical_path=current.canonical_path,
            git_common_dir=current.git_common_dir,
            origin_url=origin,
        )

    def forget(self, identity: RepositoryIdentityDescriptor) -> None:
        self._identities.pop(str(Path(identity.canonical_path).resolve()), None)


@dataclass(frozen=True)
class _RelinkCase:
    root: Path
    resolver: _MutableRevisionResolver
    authority: RepositoryIdentityDescriptor
    locator: RepositoryIdentityDescriptor
    authority_store: MemoryStore


@pytest.fixture
def relink_case(tmp_path: Path) -> _RelinkCase:
    resolver = _MutableRevisionResolver(tmp_path / "repositories")
    authority = resolver.add("authority")
    locator = resolver.add("replacement-clone")
    root = tmp_path / "memory-root"
    authority_store = MemoryStore(
        repository_namespace_path(root, authority.repository_key)
    )
    authority_store.register_repository(authority)
    return _RelinkCase(
        root=root,
        resolver=resolver,
        authority=authority,
        locator=locator,
        authority_store=authority_store,
    )


def _token(label: str) -> str:
    return canonical_sha256({"authority_state": label})


def _request(label: str) -> str:
    return stable_request_id("repository-relink-test", label)


def _namespace_has_no_store_state(root: Path, repository_key: str) -> bool:
    namespace = repository_namespace_path(root, repository_key)
    if not namespace.exists():
        return True
    return {entry.name for entry in namespace.iterdir()} <= {".memory-store.lock"}


def _wait_for_file(path: Path, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 5.0
    while not path.is_file():
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(
                "store worker exited before readiness: %s%s" % (stdout, stderr)
            )
        if time.monotonic() >= deadline:
            raise AssertionError("store worker did not become ready")
        time.sleep(0.01)


def _start_store_worker(
    *,
    mode: str,
    namespace: Path,
    descriptor: RepositoryIdentityDescriptor,
    ready: Path,
    go: Path,
    done: Path,
) -> subprocess.Popen[str]:
    source_root = Path(__file__).resolve().parents[1] / "src"
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        value
        for value in (str(source_root), environment.get("PYTHONPATH", ""))
        if value
    )
    worker = """
import json
import sys
import time
from pathlib import Path

from review_agent.memory_identity import RepositoryIdentityDescriptor
from review_agent.memory_store import MemoryStore

mode, namespace, descriptor_json, ready, go, done = sys.argv[1:]
descriptor = RepositoryIdentityDescriptor.from_payload(json.loads(descriptor_json))
store = None
if mode == "authority":
    store = MemoryStore(namespace, busy_timeout_ms=5_000)
Path(ready).write_text("ready", encoding="ascii")
while not Path(go).is_file():
    time.sleep(0.01)
if store is None:
    store = MemoryStore(namespace, busy_timeout_ms=5_000)
store.register_repository(descriptor)
Path(done).write_text("done", encoding="ascii")
"""
    return subprocess.Popen(
        [
            sys.executable,
            "-c",
            worker,
            mode,
            str(namespace),
            json.dumps(descriptor.to_payload(), sort_keys=True),
            str(ready),
            str(go),
            str(done),
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _prepare(
    registry: RepositoryRelinkRegistry,
    authority: RepositoryIdentityDescriptor,
    locator: RepositoryIdentityDescriptor,
    *,
    label: str,
    actor: str = "amy",
    reason: str = "restore the explicitly selected local authority",
) -> PreparedRepositoryRelink:
    return registry.prepare_relink(
        authority,
        locator,
        from_repository_key=authority.repository_key,
        actor=actor,
        reason=reason,
        request_id=_request(label),
    )


def _apply(
    registry: RepositoryRelinkRegistry,
    prepared: PreparedRepositoryRelink,
) -> RepositoryRelinkResult:
    return registry.apply_relink(prepared)


def test_prepare_is_write_free_and_requires_the_exact_from_key(
    relink_case: _RelinkCase,
) -> None:
    registry = RepositoryRelinkRegistry(
        relink_case.root,
        revision_resolver=relink_case.resolver,
    )
    wrong_key = (
        "0" * 64
        if relink_case.authority.repository_key != "0" * 64
        else "1" * 64
    )
    before_token = relink_case.authority_store.repository_authority_state_token(
        relink_case.authority.repository_key
    )
    before_entries = tuple(
        sorted(
            path.relative_to(relink_case.root).as_posix()
            for path in relink_case.root.rglob("*")
        )
    )

    with pytest.raises(RepositoryRelinkConflictError, match="from-key"):
        registry.prepare_relink(
            relink_case.authority,
            relink_case.locator,
            from_repository_key=wrong_key,
            actor="amy",
            reason="explicit selection",
            request_id=_request("wrong-from-key"),
        )

    assert relink_case.resolver.queries == []
    assert not registry.database_path.exists()

    prepared = _prepare(
        registry,
        relink_case.authority,
        relink_case.locator,
        label="dry-run",
    )

    assert prepared.from_repository_key == relink_case.authority.repository_key
    assert prepared.authority_repository_key == relink_case.authority.repository_key
    assert prepared.locator_repository_key == relink_case.locator.repository_key
    assert prepared.registry_generation == 0
    assert PreparedRepositoryRelink.from_payload(prepared.to_payload()) == prepared
    assert not registry.database_path.exists()
    assert before_entries == tuple(
        sorted(
            path.relative_to(relink_case.root).as_posix()
            for path in relink_case.root.rglob("*")
        )
    )
    assert before_token == relink_case.authority_store.repository_authority_state_token(
        relink_case.authority.repository_key
    )


def test_apply_binds_live_locator_to_old_namespace_without_copying_or_rekeying(
    relink_case: _RelinkCase,
) -> None:
    old_namespace = repository_namespace_path(
        relink_case.root,
        relink_case.authority.repository_key,
    )
    old_namespace.mkdir(parents=True, exist_ok=True)
    old_sentinel = old_namespace / "authority-sentinel.bin"
    old_sentinel.write_bytes(b"old-authority-database-sentinel")
    target_namespace = repository_namespace_path(
        relink_case.root,
        relink_case.locator.repository_key,
    )
    registry = RepositoryRelinkRegistry(
        relink_case.root,
        revision_resolver=relink_case.resolver,
    )
    relink_case.resolver.forget(relink_case.authority)
    prepared = _prepare(
        registry,
        relink_case.authority,
        relink_case.locator,
        label="apply",
    )
    result = registry.apply_relink(prepared)

    assert result.applied is True
    assert result.outcome == "applied"
    assert result.resolution.locator_repository_key == relink_case.locator.repository_key
    assert (
        result.resolution.authority_repository_key
        == relink_case.authority.repository_key
    )
    assert old_sentinel.read_bytes() == b"old-authority-database-sentinel"
    assert _namespace_has_no_store_state(
        relink_case.root,
        relink_case.locator.repository_key,
    )
    assert (target_namespace / ".memory-store.lock").is_file()
    assert not (target_namespace / "memory.sqlite3").exists()
    assert registry.database_path.is_file()

    resolved = registry.resolve_authority(relink_case.locator)
    assert resolved.is_bound is True
    assert resolved.authority_identity.to_payload() == relink_case.authority.to_payload()
    event = registry.verify_event_chain()[0]
    assert event.actor == "amy"
    assert event.reason == "restore the explicitly selected local authority"
    assert event.old_authority_state_token == prepared.old_authority_state_token
    receipt = registry.get_request_receipt(prepared.request_id)
    assert receipt is not None
    assert receipt.prepared.to_payload() == prepared.to_payload()
    assert receipt.result.to_payload() == result.to_payload()


def test_same_origin_never_selects_an_authority_without_explicit_relink(
    relink_case: _RelinkCase,
) -> None:
    assert relink_case.authority.origin_url == relink_case.locator.origin_url
    assert relink_case.authority.repository_key != relink_case.locator.repository_key
    registry = RepositoryRelinkRegistry(
        relink_case.root,
        revision_resolver=relink_case.resolver,
    )

    direct = registry.resolve_authority(relink_case.locator)

    assert direct.is_bound is False
    assert direct.authority_repository_key == relink_case.locator.repository_key
    assert not registry.database_path.exists()
    assert not repository_namespace_path(
        relink_case.root,
        relink_case.locator.repository_key,
    ).exists()

    prepared = _prepare(
        registry,
        relink_case.authority,
        relink_case.locator,
        label="explicit-only",
    )
    _apply(registry, prepared)

    bound = registry.resolve_authority(relink_case.locator)
    assert bound.is_bound is True
    assert bound.authority_repository_key == relink_case.authority.repository_key


def test_registry_is_root_scoped_and_prepared_plan_cannot_cross_roots(
    relink_case: _RelinkCase,
    tmp_path: Path,
) -> None:
    first = RepositoryRelinkRegistry(
        relink_case.root,
        revision_resolver=relink_case.resolver,
    )
    second_root = tmp_path / "other-memory-root"
    second = RepositoryRelinkRegistry(
        second_root,
        revision_resolver=relink_case.resolver,
    )
    prepared = _prepare(
        first,
        relink_case.authority,
        relink_case.locator,
        label="root-scope",
    )
    _apply(first, prepared)

    assert first.resolve_authority(relink_case.locator).is_bound is True
    assert second.resolve_authority(relink_case.locator).is_bound is False
    with pytest.raises(RepositoryRelinkConflictError, match="different Memory root"):
        _apply(second, prepared)
    assert not second_root.exists()


def test_registry_database_cannot_be_reused_under_a_different_root(
    relink_case: _RelinkCase,
    tmp_path: Path,
) -> None:
    source = RepositoryRelinkRegistry(
        relink_case.root,
        revision_resolver=relink_case.resolver,
    )
    _apply(
        source,
        _prepare(
            source,
            relink_case.authority,
            relink_case.locator,
            label="copied-registry",
        ),
    )
    destination_root = tmp_path / "copied-registry-root"
    destination_root.mkdir()
    shutil.copyfile(
        source.database_path,
        destination_root / source.database_path.name,
    )
    copied = RepositoryRelinkRegistry(
        destination_root,
        revision_resolver=relink_case.resolver,
    )

    with pytest.raises(RepositoryRelinkIntegrityError, match="state is invalid"):
        copied.generation()


def test_apply_revalidates_the_live_locator_and_rejects_identity_drift(
    relink_case: _RelinkCase,
) -> None:
    registry = RepositoryRelinkRegistry(
        relink_case.root,
        revision_resolver=relink_case.resolver,
    )
    prepared = _prepare(
        registry,
        relink_case.authority,
        relink_case.locator,
        label="locator-drift",
    )
    relink_case.resolver.set_origin(
        relink_case.locator,
        "https://example.test/acme/different.git",
    )

    with pytest.raises(RepositoryRelinkConflictError, match="live repository locator"):
        _apply(registry, prepared)

    assert registry.generation() == 0
    assert not registry.database_path.exists()
    assert relink_case.authority_store.database_path.is_file()
    assert _namespace_has_no_store_state(
        relink_case.root,
        relink_case.locator.repository_key,
    )


def test_apply_rejects_external_cas_values_and_verifiers(
    relink_case: _RelinkCase,
) -> None:
    registry = RepositoryRelinkRegistry(
        relink_case.root,
        revision_resolver=relink_case.resolver,
    )
    prepared = _prepare(
        registry,
        relink_case.authority,
        relink_case.locator,
        label="cas-argument-contract",
    )

    with pytest.raises(TypeError, match="new_namespace_empty"):
        registry.apply_relink(prepared, new_namespace_empty=True)
    with pytest.raises(TypeError, match="current_old_authority_state_token"):
        registry.apply_relink(
            prepared,
            current_old_authority_state_token=prepared.old_authority_state_token,
        )
    with pytest.raises(TypeError, match="old_authority_state_verifier"):
        registry.apply_relink(
            prepared,
            old_authority_state_verifier=lambda _identity: (
                prepared.old_authority_state_token
            ),
        )
    with pytest.raises(TypeError, match="new_namespace_empty_verifier"):
        registry.apply_relink(
            prepared,
            new_namespace_empty_verifier=lambda _identity: True,
        )

    assert not registry.database_path.exists()


def test_apply_rejects_authority_descriptor_drift_under_namespace_locks(
    relink_case: _RelinkCase,
    tmp_path: Path,
) -> None:
    registry = RepositoryRelinkRegistry(
        relink_case.root,
        revision_resolver=relink_case.resolver,
    )
    prepared = _prepare(
        registry,
        relink_case.authority,
        relink_case.locator,
        label="verifier-conflict",
    )

    moved = tmp_path / "moved-authority"
    moved.mkdir()
    relink_case.authority_store.register_repository(
        replace(
            relink_case.authority,
            canonical_path=str(moved.resolve()),
        )
    )

    with pytest.raises(
        RepositoryRelinkConflictError,
        match="authority descriptor changed",
    ):
        registry.apply_relink(prepared)

    assert registry.generation() == 0
    assert not registry.database_path.exists()


@pytest.mark.parametrize("changed_precondition", ["old_state", "target_namespace"])
def test_apply_rejects_changed_external_cas_preconditions_without_registry_write(
    relink_case: _RelinkCase,
    changed_precondition: str,
) -> None:
    registry = RepositoryRelinkRegistry(
        relink_case.root,
        revision_resolver=relink_case.resolver,
    )
    prepared = _prepare(
        registry,
        relink_case.authority,
        relink_case.locator,
        label=f"cas-{changed_precondition}",
    )
    if changed_precondition == "old_state":
        prepared = replace(
            prepared,
            old_authority_state_token=_token("generation-8"),
        )
    else:
        locator_namespace = repository_namespace_path(
            relink_case.root,
            relink_case.locator.repository_key,
        )
        locator_namespace.mkdir(parents=True, exist_ok=True)
        (locator_namespace / "state-created-after-prepare").write_text(
            "state",
            encoding="ascii",
        )

    with pytest.raises(RepositoryRelinkConflictError):
        registry.apply_relink(prepared)

    assert registry.generation() == 0
    assert not registry.database_path.exists()
    assert relink_case.authority_store.database_path.is_file()
    assert _namespace_has_no_store_state(
        relink_case.root,
        relink_case.locator.repository_key,
    ) is (changed_precondition == "old_state")


def test_apply_blocks_authority_writer_and_locator_creator_across_processes(
    relink_case: _RelinkCase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority_namespace = repository_namespace_path(
        relink_case.root,
        relink_case.authority.repository_key,
    )
    locator_namespace = repository_namespace_path(
        relink_case.root,
        relink_case.locator.repository_key,
    )
    authority_store = MemoryStore(authority_namespace, busy_timeout_ms=5_000)
    authority_store.register_repository(relink_case.authority)
    registry_entered = threading.Event()
    release_registry = threading.Event()

    def pause_transaction(stage: str) -> None:
        if stage == "after_binding_insert":
            registry_entered.set()
            if not release_registry.wait(timeout=5):
                raise RuntimeError("test did not release relink transaction")

    registry = RepositoryRelinkRegistry(
        relink_case.root,
        revision_resolver=relink_case.resolver,
        transaction_hook=pause_transaction,
    )
    prepared = _prepare(
        registry,
        relink_case.authority,
        relink_case.locator,
        label="cross-process-namespace-locks",
    )
    live_reverify_entered = threading.Event()
    release_live_reverify = threading.Event()
    original_repository_identity = relink_case.resolver.repository_identity
    paused_live_reverify = False

    def pause_live_reverify(repository: Path) -> RepositoryIdentity:
        nonlocal paused_live_reverify
        if (
            not paused_live_reverify
            and threading.current_thread().name == "repository-relink"
            and Path(repository).resolve()
            == Path(relink_case.locator.canonical_path).resolve()
        ):
            paused_live_reverify = True
            live_reverify_entered.set()
            if not release_live_reverify.wait(timeout=5):
                raise RuntimeError("test did not release live reverify")
        return original_repository_identity(repository)

    monkeypatch.setattr(
        relink_case.resolver,
        "repository_identity",
        pause_live_reverify,
    )
    go = tmp_path / "writers-go"
    worker_specs = (
        (
            "authority",
            authority_namespace,
            relink_case.authority,
            tmp_path / "authority-ready",
            tmp_path / "authority-done",
        ),
        (
            "locator",
            locator_namespace,
            relink_case.locator,
            tmp_path / "locator-ready",
            tmp_path / "locator-done",
        ),
    )
    workers = [
        _start_store_worker(
            mode=mode,
            namespace=namespace,
            descriptor=descriptor,
            ready=ready,
            go=go,
            done=done,
        )
        for mode, namespace, descriptor, ready, done in worker_specs
    ]
    apply_results: list[RepositoryRelinkResult] = []
    apply_errors: list[Exception] = []

    def apply() -> None:
        try:
            apply_results.append(registry.apply_relink(prepared))
        except Exception as error:  # pragma: no cover - asserted below
            apply_errors.append(error)

    apply_thread = threading.Thread(target=apply, name="repository-relink")
    try:
        for worker, (_, _, _, ready, _) in zip(workers, worker_specs):
            _wait_for_file(ready, worker)
        apply_thread.start()
        assert live_reverify_entered.wait(timeout=3)

        go.write_text("go", encoding="ascii")
        time.sleep(0.25)

        assert all(worker.poll() is None for worker in workers)
        assert all(not done.exists() for *_, done in worker_specs)
        assert not (locator_namespace / "memory.sqlite3").exists()

        release_live_reverify.set()
        assert registry_entered.wait(timeout=3)
        assert all(worker.poll() is None for worker in workers)
        assert all(not done.exists() for *_, done in worker_specs)
    finally:
        release_live_reverify.set()
        release_registry.set()
        go.write_text("go", encoding="ascii")
        if apply_thread.ident is not None:
            apply_thread.join(timeout=5)
        for worker in workers:
            try:
                worker.communicate(timeout=6)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.communicate()

    assert not apply_thread.is_alive()
    assert apply_errors == []
    assert len(apply_results) == 1 and apply_results[0].applied
    assert all(worker.returncode == 0 for worker in workers)
    assert all(done.is_file() for *_, done in worker_specs)
    assert (locator_namespace / "memory.sqlite3").is_file()


def test_disjoint_relinks_do_not_deadlock_and_one_has_a_generation_conflict(
    relink_case: _RelinkCase,
) -> None:
    first_registry = RepositoryRelinkRegistry(
        relink_case.root,
        revision_resolver=relink_case.resolver,
        sqlite_timeout_seconds=2,
    )
    second_registry = RepositoryRelinkRegistry(
        relink_case.root,
        revision_resolver=relink_case.resolver,
        sqlite_timeout_seconds=2,
    )
    second_authority = relink_case.resolver.add("second-authority")
    second_locator = relink_case.resolver.add("second-locator")
    second_authority_store = MemoryStore(
        repository_namespace_path(
            relink_case.root,
            second_authority.repository_key,
        )
    )
    second_authority_store.register_repository(second_authority)
    first = _prepare(
        first_registry,
        relink_case.authority,
        relink_case.locator,
        label="disjoint-lock-first",
    )
    second = _prepare(
        second_registry,
        second_authority,
        second_locator,
        label="disjoint-lock-second",
    )
    start = threading.Event()
    results: list[RepositoryRelinkResult] = []
    errors: list[Exception] = []

    def apply(registry: RepositoryRelinkRegistry, plan: PreparedRepositoryRelink) -> None:
        start.wait(timeout=2)
        try:
            results.append(_apply(registry, plan))
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)

    threads = (
        threading.Thread(target=apply, args=(first_registry, first)),
        threading.Thread(target=apply, args=(second_registry, second)),
    )
    for thread in threads:
        thread.start()
    start.set()
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert len(results) == 1 and results[0].applied
    assert len(errors) == 1
    assert isinstance(errors[0], RepositoryRelinkConflictError)
    assert "generation changed" in str(errors[0])
    assert first_registry.generation() == 1


def test_apply_exception_releases_both_namespace_locks(
    relink_case: _RelinkCase,
) -> None:
    def fail_transaction(stage: str) -> None:
        if stage == "after_binding_insert":
            raise RuntimeError("injected transaction failure")

    registry = RepositoryRelinkRegistry(
        relink_case.root,
        revision_resolver=relink_case.resolver,
        transaction_hook=fail_transaction,
    )
    prepared = _prepare(
        registry,
        relink_case.authority,
        relink_case.locator,
        label="namespace-lock-release",
    )

    with pytest.raises(RepositoryRelinkError, match="transaction failed"):
        _apply(registry, prepared)

    authority_namespace = repository_namespace_path(
        relink_case.root,
        relink_case.authority.repository_key,
    )
    locator_namespace = repository_namespace_path(
        relink_case.root,
        relink_case.locator.repository_key,
    )
    acquired = threading.Event()
    lock_errors: list[Exception] = []

    def acquire_in_another_thread() -> None:
        try:
            with MemoryStore.lock_namespaces(
                locator_namespace,
                authority_namespace,
                busy_timeout_ms=500,
            ):
                acquired.set()
        except Exception as error:  # pragma: no cover - asserted below
            lock_errors.append(error)

    contender = threading.Thread(target=acquire_in_another_thread)
    contender.start()
    contender.join(timeout=2)

    assert not contender.is_alive()
    assert acquired.is_set()
    assert lock_errors == []
    assert registry.generation() == 0
    recovered = RepositoryRelinkRegistry(
        relink_case.root,
        revision_resolver=relink_case.resolver,
    )
    assert _apply(recovered, prepared).applied


def test_registry_generation_is_a_compare_and_swap_precondition(
    relink_case: _RelinkCase,
) -> None:
    second_locator = relink_case.resolver.add("second-replacement-clone")
    registry = RepositoryRelinkRegistry(
        relink_case.root,
        revision_resolver=relink_case.resolver,
    )
    first = _prepare(
        registry,
        relink_case.authority,
        relink_case.locator,
        label="generation-first",
    )
    stale = _prepare(
        registry,
        relink_case.authority,
        second_locator,
        label="generation-stale",
    )
    _apply(registry, first)

    with pytest.raises(RepositoryRelinkConflictError, match="generation changed"):
        _apply(registry, stale)

    assert registry.generation() == 1
    assert len(registry.verify_event_chain()) == 1
    assert registry.resolve_authority(second_locator).is_bound is False


def test_reprepared_exact_request_replays_after_registry_generation_advances(
    relink_case: _RelinkCase,
) -> None:
    registry = RepositoryRelinkRegistry(
        relink_case.root,
        revision_resolver=relink_case.resolver,
    )
    original = _prepare(
        registry,
        relink_case.authority,
        relink_case.locator,
        label="idempotent",
    )
    committed = _apply(registry, original)
    reprepared = _prepare(
        registry,
        relink_case.authority,
        relink_case.locator,
        label="idempotent",
    )

    assert reprepared.registry_generation == 1
    assert reprepared.semantic_hash == original.semantic_hash
    assert reprepared.prepared_hash != original.prepared_hash
    replayed = registry.apply_relink(reprepared)

    assert replayed.to_payload() == committed.to_payload()
    assert registry.generation() == 1
    assert len(registry.verify_event_chain()) == 1


def test_request_id_reuse_with_different_semantics_is_rejected(
    relink_case: _RelinkCase,
) -> None:
    registry = RepositoryRelinkRegistry(
        relink_case.root,
        revision_resolver=relink_case.resolver,
    )
    original = _prepare(
        registry,
        relink_case.authority,
        relink_case.locator,
        label="semantic-conflict",
    )
    _apply(registry, original)
    conflicting = _prepare(
        registry,
        relink_case.authority,
        relink_case.locator,
        label="semantic-conflict",
        reason="a different audited reason",
    )

    with pytest.raises(RepositoryRelinkConflictError, match="different semantics"):
        registry.apply_relink(conflicting)


def test_one_locator_cannot_bind_to_multiple_authorities(
    relink_case: _RelinkCase,
) -> None:
    other_authority = relink_case.resolver.add("other-authority")
    registry = RepositoryRelinkRegistry(
        relink_case.root,
        revision_resolver=relink_case.resolver,
    )
    _apply(
        registry,
        _prepare(
            registry,
            relink_case.authority,
            relink_case.locator,
            label="first-authority",
        ),
    )

    with pytest.raises(RepositoryRelinkConflictError, match="different authority"):
        _prepare(
            registry,
            other_authority,
            relink_case.locator,
            label="second-authority",
        )


def test_authority_chains_and_cycles_are_rejected(relink_case: _RelinkCase) -> None:
    third = relink_case.resolver.add("third-clone")
    registry = RepositoryRelinkRegistry(
        relink_case.root,
        revision_resolver=relink_case.resolver,
    )
    _apply(
        registry,
        _prepare(
            registry,
            relink_case.authority,
            relink_case.locator,
            label="chain-root",
        ),
    )

    with pytest.raises(RepositoryRelinkConflictError, match="unbound root"):
        _prepare(
            registry,
            relink_case.locator,
            third,
            label="authority-chain",
        )
    with pytest.raises(RepositoryRelinkConflictError, match="unbound root"):
        _prepare(
            registry,
            relink_case.locator,
            relink_case.authority,
            label="authority-cycle",
        )


def test_multiple_locators_may_share_one_unbound_root_authority(
    relink_case: _RelinkCase,
) -> None:
    second_locator = relink_case.resolver.add("second-locator")
    registry = RepositoryRelinkRegistry(
        relink_case.root,
        revision_resolver=relink_case.resolver,
    )
    _apply(
        registry,
        _prepare(
            registry,
            relink_case.authority,
            relink_case.locator,
            label="shared-authority-one",
        ),
    )
    _apply(
        registry,
        _prepare(
            registry,
            relink_case.authority,
            second_locator,
            label="shared-authority-two",
        ),
    )

    assert registry.generation() == 2
    assert len(registry.verify_event_chain()) == 2
    assert (
        registry.resolve_authority(relink_case.locator).authority_repository_key
        == relink_case.authority.repository_key
    )
    assert (
        registry.resolve_authority(second_locator).authority_repository_key
        == relink_case.authority.repository_key
    )


def test_already_bound_request_is_auditable_without_a_second_binding_event(
    relink_case: _RelinkCase,
) -> None:
    registry = RepositoryRelinkRegistry(
        relink_case.root,
        revision_resolver=relink_case.resolver,
    )
    first = _prepare(
        registry,
        relink_case.authority,
        relink_case.locator,
        label="already-bound-first",
    )
    _apply(registry, first)
    second = _prepare(
        registry,
        relink_case.authority,
        relink_case.locator,
        label="already-bound-second",
        actor="bob",
        reason="confirm the existing explicit binding",
    )

    result = _apply(registry, second)

    assert result.applied is False
    assert result.outcome == "already_bound"
    assert registry.generation() == 1
    assert len(registry.verify_event_chain()) == 1
    receipt = registry.get_request_receipt(second.request_id)
    assert receipt is not None
    assert receipt.prepared.actor == "bob"
    assert receipt.prepared.reason == "confirm the existing explicit binding"
    assert receipt.result.outcome == "already_bound"


def test_linked_worktree_locator_reuses_the_same_direct_binding(
    relink_case: _RelinkCase,
) -> None:
    registry = RepositoryRelinkRegistry(
        relink_case.root,
        revision_resolver=relink_case.resolver,
    )
    _apply(
        registry,
        _prepare(
            registry,
            relink_case.authority,
            relink_case.locator,
            label="worktree-primary",
        ),
    )
    linked = relink_case.resolver.add_worktree(
        "linked-worktree",
        relink_case.locator,
    )
    assert linked.repository_key == relink_case.locator.repository_key
    assert linked.canonical_path != relink_case.locator.canonical_path
    prepared = _prepare(
        registry,
        relink_case.authority,
        linked,
        label="worktree-linked",
        actor="bob",
        reason="use the existing binding from a linked worktree",
    )

    result = _apply(registry, prepared)

    assert result.outcome == "already_bound"
    assert result.resolution.locator_identity.to_payload() == linked.to_payload()
    assert registry.resolve_authority(linked).authority_repository_key == (
        relink_case.authority.repository_key
    )
    receipt = registry.get_request_receipt(prepared.request_id)
    assert receipt is not None
    assert receipt.prepared.locator_identity.to_payload() == linked.to_payload()
    assert len(registry.verify_event_chain()) == 1


@pytest.mark.parametrize(
    "failure_stage",
    [
        "after_binding_insert",
        "after_event_insert",
        "after_generation_update",
        "after_receipt_insert",
    ],
)
def test_apply_rolls_back_every_registry_projection_on_failure(
    relink_case: _RelinkCase,
    failure_stage: str,
) -> None:
    def fail_transaction(stage: str) -> None:
        if stage == failure_stage:
            raise RuntimeError("injected transaction failure")

    registry = RepositoryRelinkRegistry(
        relink_case.root,
        revision_resolver=relink_case.resolver,
        transaction_hook=fail_transaction,
    )
    prepared = _prepare(
        registry,
        relink_case.authority,
        relink_case.locator,
        label=f"rollback-{failure_stage}",
    )

    with pytest.raises(RepositoryRelinkError, match="transaction failed"):
        _apply(registry, prepared)

    assert registry.generation() == 0
    assert registry.verify_event_chain() == ()
    assert registry.get_request_receipt(prepared.request_id) is None
    assert registry.resolve_authority(relink_case.locator).is_bound is False

    recovered = RepositoryRelinkRegistry(
        relink_case.root,
        revision_resolver=relink_case.resolver,
    )
    assert _apply(recovered, prepared).applied is True


def test_registry_integrity_verification_rejects_tampered_audit_data(
    relink_case: _RelinkCase,
) -> None:
    registry = RepositoryRelinkRegistry(
        relink_case.root,
        revision_resolver=relink_case.resolver,
    )
    prepared = _prepare(
        registry,
        relink_case.authority,
        relink_case.locator,
        label="tamper",
    )
    _apply(registry, prepared)
    with sqlite3.connect(registry.database_path) as connection:
        connection.execute("UPDATE events SET event_json = '{}' WHERE sequence = 1")

    with pytest.raises(RepositoryRelinkIntegrityError):
        registry.verify_event_chain()


def test_registry_verified_reads_use_one_sqlite_snapshot(
    relink_case: _RelinkCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = RepositoryRelinkRegistry(
        relink_case.root,
        revision_resolver=relink_case.resolver,
    )
    _apply(
        registry,
        _prepare(
            registry,
            relink_case.authority,
            relink_case.locator,
            label="read-snapshot",
        ),
    )
    original_verify = registry._verify_connection
    transaction_states: list[bool] = []

    def observe_transaction(connection):
        transaction_states.append(connection.in_transaction)
        return original_verify(connection)

    monkeypatch.setattr(registry, "_verify_connection", observe_transaction)

    assert registry.generation() == 1
    assert transaction_states == [True]


def test_public_models_round_trip_and_receipt_binds_the_prepared_audit_request(
    relink_case: _RelinkCase,
) -> None:
    registry = RepositoryRelinkRegistry(
        relink_case.root,
        revision_resolver=relink_case.resolver,
    )
    prepared = _prepare(
        registry,
        relink_case.authority,
        relink_case.locator,
        label="model-round-trip",
    )
    result = _apply(registry, prepared)
    receipt = registry.get_request_receipt(prepared.request_id)
    assert receipt is not None

    assert PreparedRepositoryRelink.from_payload(prepared.to_payload()) == prepared
    assert (
        RepositoryAuthorityResolution.from_payload(
            result.resolution.to_payload()
        ).to_payload()
        == result.resolution.to_payload()
    )
    assert RepositoryRelinkResult.from_payload(result.to_payload()) == result
    assert RepositoryRelinkRequestReceipt.from_payload(receipt.to_payload()) == receipt

    tampered = receipt.to_payload()
    tampered["prepared"]["actor"] = "mallory"
    with pytest.raises(RepositoryRelinkError):
        RepositoryRelinkRequestReceipt.from_payload(tampered)


def test_module_level_public_api_preserves_exact_from_key_and_audit(
    relink_case: _RelinkCase,
) -> None:
    request_id = _request("module-api")
    prepared = prepare_relink(
        relink_case.root,
        relink_case.authority,
        relink_case.locator,
        from_repository_key=relink_case.authority.repository_key,
        actor="amy",
        reason="exercise the public service functions",
        request_id=request_id,
        revision_resolver=relink_case.resolver,
    )
    result = apply_relink(
        relink_case.root,
        prepared,
        revision_resolver=relink_case.resolver,
    )
    resolution = resolve_repository_authority(
        relink_case.root,
        relink_case.locator,
        revision_resolver=relink_case.resolver,
    )
    receipt = get_repository_relink_receipt(relink_case.root, request_id)

    assert result.applied is True
    assert resolution.authority_repository_key == relink_case.authority.repository_key
    assert receipt is not None
    assert receipt.prepared.from_repository_key == relink_case.authority.repository_key
    assert receipt.prepared.actor == "amy"


def test_real_git_clone_works_with_the_default_revision_resolver(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    clone = tmp_path / "real-clone"
    run_git(tmp_path, "clone", str(git_repo), str(clone))
    run_git(git_repo, "remote", "add", "origin", _SHARED_ORIGIN)
    run_git(clone, "remote", "set-url", "origin", _SHARED_ORIGIN)
    resolver = RevisionResolver()
    authority = build_repository_identity_descriptor(
        resolver.repository_identity(git_repo)
    )
    locator = build_repository_identity_descriptor(resolver.repository_identity(clone))
    memory_root = tmp_path / "real-memory-root"
    authority_store = MemoryStore(
        repository_namespace_path(memory_root, authority.repository_key)
    )
    authority_store.register_repository(authority)
    registry = RepositoryRelinkRegistry(memory_root)
    prepared = _prepare(
        registry,
        authority,
        locator,
        label="real-git",
    )

    result = _apply(registry, prepared)

    assert result.applied is True
    assert registry.resolve_authority(locator).authority_repository_key == (
        authority.repository_key
    )
