from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import pytest

from conftest import run_git
import review_agent_eval.repository as repository_module
from review_agent_eval.artifacts import TrialManifest
from review_agent_eval.cases import (
    CaseSplit,
    PublicSuitePreparationBindingV2,
    REPOSITORY_MATERIALIZER_PROTOCOL,
    SuiteCase,
    SuiteKind,
    SuiteSource,
    WireContractV2,
)
from review_agent_eval.config import (
    derive_case_path_id,
    derive_trial_id,
    derive_trial_seed,
)
from review_agent_eval.models import (
    EVAL_CASE_SCHEMA_VERSION,
    EVAL_SUBMISSION_SCHEMA_VERSION,
    EvalInput,
    Repository,
    RepositoryReviewTarget,
    RepositorySource,
    ReviewRequest,
    ReviewTargetKind,
    TrialStatus,
    TruthCompleteness,
    stable_id,
)
from review_agent_eval.repository import (
    FixtureRepositoryBuilder,
    PreparedRepository,
    PreparedRepositoryManifest,
    RepositoryAcquisitionBinding,
    RepositoryIntegrityError,
    RepositoryLimitError,
    RepositoryPolicyError,
    RepositoryPreparer,
    RepositoryPreparationError,
    RepositorySecurityError,
    WorkspaceManifest,
    WorkspaceRetentionPolicy,
)


def _git(repo: Path, *args: str, input_bytes: bytes | None = None) -> str:
    command = ["git"]
    if (repo / "HEAD").is_file() and (repo / "objects").is_dir():
        command.extend(["--git-dir", str(repo)])
        cwd = repo.parent
    else:
        command.extend(["-C", str(repo)])
        cwd = repo.parent
    result = subprocess.run(
        [*command, *args],
        cwd=cwd,
        input=input_bytes,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.decode("utf-8").strip()


def _write_fixture(root: Path) -> Path:
    fixture = root / "repositories" / "demo"
    base = fixture / "base"
    head = fixture / "head"
    base.mkdir(parents=True)
    head.mkdir(parents=True)
    (base / "app.py").write_bytes(b"def allowed(user):\n    return False\n")
    (head / "app.py").write_bytes(
        b"def allowed(user):\n    return user.is_admin\n"
    )
    (head / "README.md").write_bytes(b"# Demo\n")
    return fixture


def _author_fixture(
    suite_root: Path,
    tmp_path: Path,
    *,
    relative_path: str = "repositories/demo",
) -> tuple[Repository, object]:
    fixture = _write_fixture(suite_root)
    built = FixtureRepositoryBuilder().build(fixture, tmp_path / "authored.git")
    return built.to_repository(relative_path), built


def _init_local_repository(path: Path) -> tuple[str, str]:
    path.mkdir(parents=True)
    run_git(path, "init")
    run_git(path, "config", "user.email", "eval@example.test")
    run_git(path, "config", "user.name", "Eval Test")
    (path / "app.py").write_text("value = 1\n", encoding="utf-8")
    run_git(path, "add", "app.py")
    run_git(path, "commit", "-m", "base")
    base = run_git(path, "rev-parse", "HEAD")
    (path / "app.py").write_text("value = 2\n", encoding="utf-8")
    run_git(path, "commit", "-am", "head")
    head = run_git(path, "rev-parse", "HEAD")
    return base, head


def _init_bare_repository_with_tree_names(
    path: Path,
    names: tuple[bytes, ...],
) -> tuple[str, str]:
    path.mkdir(parents=True)
    _git(path, "init", "--bare")
    _git(path, "config", "user.email", "eval@example.test")
    _git(path, "config", "user.name", "Eval Test")
    empty_tree = _git(path, "mktree", "-z", input_bytes=b"")
    base = _git(path, "commit-tree", empty_tree, input_bytes=b"base\n")
    blob = _git(path, "hash-object", "-w", "--stdin", input_bytes=b"payload\n")
    tree_input = b"".join(
        b"100644 blob " + blob.encode("ascii") + b"\t" + name + b"\0"
        for name in sorted(names)
    )
    tree = _git(path, "mktree", "-z", input_bytes=tree_input)
    head = _git(path, "commit-tree", tree, "-p", base, input_bytes=b"head\n")
    _git(path, "update-ref", "refs/heads/main", head)
    return base, head


def _expanding_tree_objects(
    *, fanout: int = 47, levels: int = 3
) -> tuple[str, str, dict[str, object]]:
    """Build a tiny tree-object DAG with a large logical expansion."""

    objects: dict[str, object] = {}

    def add(object_type: str, raw: bytes) -> str:
        oid = repository_module._object_hash("sha1", object_type, raw)
        objects[oid] = repository_module._GitObject(oid, object_type, raw)
        return oid

    empty_tree = add("tree", b"")
    child = empty_tree
    for _level in range(levels):
        raw = b"".join(
            b"40000 d%03d\0" % index + bytes.fromhex(child)
            for index in range(fanout)
        )
        child = add("tree", raw)
    return empty_tree, child, objects


def _commit_object(
    objects: dict[str, object], tree: str, *, parent: str | None = None
) -> str:
    headers = ["tree %s" % tree]
    if parent is not None:
        headers.append("parent %s" % parent)
    raw = (("\n".join(headers)) + "\n\nfixture\n").encode("ascii")
    oid = repository_module._object_hash("sha1", "commit", raw)
    objects[oid] = repository_module._GitObject(oid, "commit", raw)
    return oid


def _local_descriptor(path: str, base: str, head: str) -> Repository:
    return Repository(
        source=RepositorySource.GIT,
        path=path,
        url=None,
        base_revision=base,
        head_revision=head,
    )


def _trial_id(index: int = 1) -> str:
    return derive_trial_id(stable_id("run", "repository-tests"), "case-a", index)


def _git_executable() -> Path:
    executable = shutil.which("git")
    assert executable is not None
    return Path(executable).absolute()


def _trial_binding(
    descriptor: Repository,
    index: int,
) -> tuple[TrialManifest, SuiteCase, EvalInput]:
    eval_input = EvalInput(
        schema_version=EvalInput.SCHEMA_VERSION,
        task_id="case-a",
        review_target=RepositoryReviewTarget(
            kind=ReviewTargetKind.REPOSITORY,
            repository=descriptor,
            review_request=ReviewRequest(
                title="Review the change",
                description=None,
                user_intent=None,
                review_focus=None,
                linked_requirements=(),
                project_rules=(),
                existing_ci_evidence=(),
            ),
        ),
    )
    suite_case = SuiteCase(
        task_id=eval_input.task_id,
        case_version=1,
        path="cases/case-a.json",
        split=CaseSplit.REGRESSION,
        protocol_id="core-code-review-v1",
        dimensions=(),
        raw_file_size_bytes=1,
        raw_file_sha256="1" * 64,
        canonical_case_digest="c" * 64,
        eval_input_digest=eval_input.digest(),
        truth_completeness=TruthCompleteness.CLOSED_WORLD,
    )
    run_id = stable_id("run", "repository-tests")
    wire_contract = WireContractV2(
        case_schema_version=EVAL_CASE_SCHEMA_VERSION,
        input_schema_version=EvalInput.SCHEMA_VERSION,
        submission_schema_version=EVAL_SUBMISSION_SCHEMA_VERSION,
        review_target_kind=ReviewTargetKind.REPOSITORY,
        materializer_protocol=REPOSITORY_MATERIALIZER_PROTOCOL,
    )
    trial_manifest = TrialManifest(
        schema_version=TrialManifest.SCHEMA_VERSION,
        run_id=run_id,
        task_id=eval_input.task_id,
        case_path_id=derive_case_path_id(eval_input.task_id),
        canonical_case_digest=suite_case.canonical_case_digest,
        eval_input_digest=eval_input.digest(),
        wire_contract=wire_contract,
        target_kind=ReviewTargetKind.REPOSITORY,
        materializer_protocol=REPOSITORY_MATERIALIZER_PROTOCOL,
        suite_preparation_binding_digest=None,
        adapter_capabilities_digest="d" * 64,
        trial_id=derive_trial_id(run_id, eval_input.task_id, index),
        trial_index=index,
        seed=derive_trial_seed(run_id, eval_input.task_id, index),
        agent_config_digest="a" * 64,
        initial_evaluator_execution_digest="b" * 64,
    )
    return trial_manifest, suite_case, eval_input


def _trial_workspace(
    preparer: RepositoryPreparer,
    prepared: object,
    descriptor: Repository,
    *,
    index: int = 1,
    attempt: int = 1,
):
    trial_manifest, suite_case, eval_input = _trial_binding(descriptor, index)
    return preparer.trial_workspace(
        prepared,
        trial_manifest=trial_manifest,
        suite_case=suite_case,
        eval_input=eval_input,
        attempt=attempt,
    )


def _preparer(
    tmp_path: Path,
    suite_root: Path,
    **kwargs: object,
) -> RepositoryPreparer:
    return RepositoryPreparer(
        suite_root=suite_root,
        data_root=tmp_path / ".eval-data",
        workspace_root=tmp_path / ".eval-workspaces",
        git_executable=_git_executable(),
        **kwargs,
    )


def _public_preparation_binding() -> PublicSuitePreparationBindingV2:
    return PublicSuitePreparationBindingV2(
        schema_version="public_suite_preparation_binding_v2",
        source_catalog_digest="1" * 64,
        acquisition_receipt_digest="2" * 64,
        source_manifest_digest="3" * 64,
        filter_manifest_digest="4" * 64,
        preparation_packet_digest="5" * 64,
        repository_catalog_digest="6" * 64,
        frozen_bundle_trust_digest=None,
    )


def _remote_repository_and_binding(
    *,
    host: str = "example.test",
) -> tuple[Repository, RepositoryAcquisitionBinding]:
    remote = Repository(
        source=RepositorySource.GIT,
        path=None,
        url=f"https://{host}/project.git",
        base_revision="a" * 40,
        head_revision="b" * 40,
    )
    suite_source = SuiteSource(
        kind=SuiteKind.PUBLIC,
        source_id="public-benchmark",
        source_version="2026-07-16",
        source_uri="https://example.test/benchmark",
        license="MIT",
        content_hash="d" * 64,
        preparation_binding=_public_preparation_binding(),
    )
    return remote, RepositoryAcquisitionBinding(
        repository=remote,
        expected_source_digest="c" * 64,
        suite_source=suite_source,
        allowed_host=host,
        allowed_port=443,
    )


def test_fixture_builder_creates_reproducible_commits_and_tree_digests(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path / "suite")
    first = FixtureRepositoryBuilder().build(fixture, tmp_path / "first.git")

    for path in sorted(fixture.rglob("*")):
        if path.is_file():
            os.utime(path, (1_900_000_000, 1_900_000_000))
    second = FixtureRepositoryBuilder().build(fixture, tmp_path / "second.git")

    assert first.base_revision == second.base_revision
    assert first.head_revision == second.head_revision
    assert first.base_tree == second.base_tree
    assert first.head_tree == second.head_tree
    assert first.base_source_digest == second.base_source_digest
    assert first.head_source_digest == second.head_source_digest
    assert _git(first.repository_path, "rev-parse", f"{first.base_revision}^{{tree}}") == first.base_tree
    assert _git(first.repository_path, "rev-parse", f"{first.head_revision}^{{tree}}") == first.head_tree
    assert _git(first.repository_path, "rev-parse", f"{first.head_revision}^") == first.base_revision


def test_head_tree_dag_expansion_is_bounded_by_logical_entries() -> None:
    empty_tree, expanding_tree, objects = _expanding_tree_objects()
    base = _commit_object(objects, empty_tree)
    head = _commit_object(objects, expanding_tree, parent=base)

    with pytest.raises(RepositoryLimitError, match="logical entry"):
        repository_module._closure_from_objects(
            objects,
            object_format="sha1",
            base_revision=base,
            head_revision=head,
        )


def test_base_tree_dag_expansion_is_bounded_before_replay_allocation() -> None:
    empty_tree, expanding_tree, objects = _expanding_tree_objects()
    base = _commit_object(objects, expanding_tree)
    head = _commit_object(objects, empty_tree, parent=base)
    closure = repository_module._closure_from_objects(
        objects,
        object_format="sha1",
        base_revision=base,
        head_revision=head,
    )

    with pytest.raises(RepositoryLimitError, match="logical entry"):
        repository_module._tree_file_index(closure, closure.base_tree)


def test_fixture_builder_rejects_links_special_nodes_and_vcs_metadata(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path / "suite")
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    linked = fixture / "head" / "linked.txt"
    try:
        linked.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    with pytest.raises(RepositorySecurityError, match="link|reparse"):
        FixtureRepositoryBuilder().build(fixture, tmp_path / "unsafe.git")

    linked.unlink()
    metadata = fixture / "head" / ".git"
    metadata.mkdir()
    with pytest.raises(RepositorySecurityError, match=r"VCS|metadata|\.git"):
        FixtureRepositoryBuilder().build(fixture, tmp_path / "metadata.git")


def test_fixture_prepare_rebuilds_exact_descriptor_and_keeps_truth_out(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    descriptor, authored = _author_fixture(suite, tmp_path)
    (suite / "truth.json").write_text('{"answer": "admin bypass"}', encoding="utf-8")
    (suite / "suite_manifest.json").write_text("{}", encoding="utf-8")

    with _preparer(tmp_path, suite) as preparer:
        prepared = preparer.prepare(descriptor)
        assert prepared.manifest.base_revision == authored.base_revision
        assert prepared.manifest.head_revision == authored.head_revision
        assert prepared.manifest.base_tree == authored.base_tree
        assert prepared.manifest.head_tree == authored.head_tree

        lease = _trial_workspace(preparer, prepared, descriptor)
        with lease as workspace:
            assert (workspace.path / "app.py").read_text(encoding="utf-8").endswith(
                "return user.is_admin\n"
            )
            assert not (workspace.path / "truth.json").exists()
            assert not (workspace.path / "suite_manifest.json").exists()
            assert not (workspace.path / "workspace_manifest.json").exists()
        assert not lease.path.exists()
        assert lease.cleanup_diagnostic is None


def test_fixture_prepare_rejects_revision_drift(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    descriptor, _built = _author_fixture(suite, tmp_path)
    drifted = Repository(
        source=descriptor.source,
        path=descriptor.path,
        url=None,
        base_revision="a" * 40,
        head_revision=descriptor.head_revision,
    )
    with _preparer(tmp_path, suite) as preparer:
        with pytest.raises(RepositoryIntegrityError, match="revision|commit"):
            preparer.prepare(drifted)


def test_prepare_accepts_only_the_canonical_repository_descriptor(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    with _preparer(tmp_path, suite) as preparer:
        with pytest.raises(TypeError, match="Repository"):
            preparer.prepare(  # type: ignore[arg-type]
                {
                    "source": "git",
                    "path": "imports/repo",
                    "url": None,
                    "base_revision": "a" * 40,
                    "head_revision": "b" * 40,
                }
            )


def test_local_git_materialization_ignores_dirty_and_untracked_source_state(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    source = tmp_path / ".eval-data" / "imports" / "repo"
    base, head = _init_local_repository(source)
    (source / "app.py").write_text("dirty = True\n", encoding="utf-8")
    (source / "truth.json").write_text("not for the agent", encoding="utf-8")
    product_state = source / ".review-agent"
    product_state.mkdir()
    (product_state / "memory.sqlite3").write_bytes(b"state")

    descriptor = _local_descriptor("imports/repo", base, head)
    with _preparer(tmp_path, suite) as preparer:
        prepared = preparer.prepare(descriptor)
        source.rename(tmp_path / "source-removed-after-prepare")
        with _trial_workspace(preparer, prepared, descriptor) as workspace:
            assert (workspace.path / "app.py").read_text(encoding="utf-8") == "value = 2\n"
            assert not (workspace.path / "truth.json").exists()
            assert not (workspace.path / ".review-agent").exists()
            assert _git(workspace.path, "status", "--porcelain=v1") == ""


def test_inherited_git_environment_cannot_redirect_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    source = tmp_path / ".eval-data" / "imports" / "repo"
    base, head = _init_local_repository(source)
    attacker = tmp_path / "attacker.git"
    attacker.mkdir()
    monkeypatch.setenv("GIT_DIR", str(attacker))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "attacker-worktree"))
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "core.hooksPath")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", str(tmp_path / "attacker-hooks"))
    monkeypatch.setenv("GIT_SSH_COMMAND", "attacker-ssh")
    monkeypatch.setenv("GIT_ASKPASS", "attacker-askpass")

    descriptor = _local_descriptor("imports/repo", base, head)
    with _preparer(tmp_path, suite) as preparer:
        prepared = preparer.prepare(descriptor)
        with _trial_workspace(preparer, prepared, descriptor) as workspace:
            for key in (
                "GIT_DIR",
                "GIT_WORK_TREE",
                "GIT_CONFIG_COUNT",
                "GIT_CONFIG_KEY_0",
                "GIT_CONFIG_VALUE_0",
                "GIT_SSH_COMMAND",
                "GIT_ASKPASS",
            ):
                monkeypatch.delenv(key, raising=False)
            assert _git(workspace.path, "rev-parse", "HEAD") == head


def test_prepare_git_processes_inherit_operation_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    source = tmp_path / ".eval-data" / "imports" / "repo"
    base, head = _init_local_repository(source)
    descriptor = _local_descriptor("imports/repo", base, head)
    launches: list[dict[str, object]] = []

    with _preparer(tmp_path, suite) as preparer:
        original_popen = repository_module.subprocess.Popen

        def capture_popen(*args: object, **kwargs: object):
            launches.append(dict(kwargs))
            return original_popen(*args, **kwargs)

        monkeypatch.setattr(
            repository_module.subprocess,
            "Popen",
            capture_popen,
        )
        prepared = preparer.prepare(descriptor)
        assert prepared.repository == descriptor

    assert launches
    if os.name == "nt":
        for launch in launches:
            assert int(launch["creationflags"]) & 0x00000004
            startupinfo = launch.get("startupinfo")
            assert startupinfo is not None
            handle_list = startupinfo.lpAttributeList["handle_list"]
            assert len(handle_list) == 1
    else:
        for launch in launches:
            pass_fds = launch.get("pass_fds")
            assert isinstance(pass_fds, tuple)
            assert len(pass_fds) == 1


def test_prepare_rejects_missing_and_non_commit_objects(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    source = tmp_path / ".eval-data" / "imports" / "repo"
    base, head = _init_local_repository(source)
    blob = _git(source, "hash-object", "-w", "--stdin", input_bytes=b"blob")
    run_git(source, "tag", "-a", "annotated", "-m", "annotated", base)
    annotated_tag = run_git(source, "rev-parse", "refs/tags/annotated")

    with _preparer(tmp_path, suite) as preparer:
        with pytest.raises(RepositoryIntegrityError, match="commit"):
            preparer.prepare(_local_descriptor("imports/repo", blob, head))
        with pytest.raises(RepositoryIntegrityError, match="commit|tag"):
            preparer.prepare(_local_descriptor("imports/repo", annotated_tag, head))
        with pytest.raises(RepositoryIntegrityError, match="commit|revision"):
            preparer.prepare(_local_descriptor("imports/repo", "f" * 40, head))


def test_sha256_git_repository_is_supported_when_git_supports_it(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    source = tmp_path / ".eval-data" / "imports" / "sha256-repo"
    source.mkdir(parents=True)
    result = subprocess.run(
        ["git", "init", "--object-format=sha256"],
        cwd=source,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        pytest.skip("installed Git does not support SHA-256 repositories")
    run_git(source, "config", "user.email", "eval@example.test")
    run_git(source, "config", "user.name", "Eval Test")
    (source / "app.py").write_text("value = 1\n", encoding="utf-8")
    run_git(source, "add", "app.py")
    run_git(source, "commit", "-m", "base")
    base = run_git(source, "rev-parse", "HEAD")
    (source / "app.py").write_text("value = 2\n", encoding="utf-8")
    run_git(source, "commit", "-am", "head")
    head = run_git(source, "rev-parse", "HEAD")
    assert len(base) == len(head) == 64

    with _preparer(tmp_path, suite) as preparer:
        descriptor = _local_descriptor("imports/sha256-repo", base, head)
        prepared = preparer.prepare(descriptor)
        with _trial_workspace(preparer, prepared, descriptor) as workspace:
            assert _git(workspace.path, "rev-parse", "HEAD") == head
            assert _git(workspace.path, "rev-parse", "--show-object-format") == "sha256"


@pytest.mark.parametrize("hazard", ["include", "alternates", "replace", "shallow"])
def test_local_source_git_authority_hazards_are_rejected(
    tmp_path: Path,
    hazard: str,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    source = tmp_path / ".eval-data" / "imports" / "repo"
    base, head = _init_local_repository(source)
    if hazard == "include":
        included = tmp_path / "attacker.gitconfig"
        included.write_text("[core]\n\thooksPath = attacker-hooks\n", encoding="utf-8")
        with (source / ".git" / "config").open("a", encoding="utf-8") as stream:
            stream.write(f"\n[include]\n\tpath = {included.as_posix()}\n")
    elif hazard == "alternates":
        outside_objects = tmp_path / "outside-objects"
        outside_objects.mkdir()
        (source / ".git" / "objects" / "info" / "alternates").write_text(
            str(outside_objects), encoding="utf-8"
        )
    elif hazard == "replace":
        run_git(source, "replace", head, base)
    else:
        (source / ".git" / "shallow").write_text(head + "\n", encoding="ascii")

    with _preparer(tmp_path, suite) as preparer:
        with pytest.raises(
            RepositorySecurityError,
            match="include|alternate|replace|shallow|source",
        ):
            preparer.prepare(_local_descriptor("imports/repo", base, head))


def test_local_source_rejects_gitdir_indirection(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    source = tmp_path / ".eval-data" / "imports" / "repo"
    base, head = _init_local_repository(source)
    actual_git = source.parent / "actual.git"
    (source / ".git").rename(actual_git)
    (source / ".git").write_text("gitdir: ../actual.git\n", encoding="utf-8")

    with _preparer(tmp_path, suite) as preparer:
        with pytest.raises(RepositorySecurityError, match=r"gitdir|indirection|\.git"):
            preparer.prepare(_local_descriptor("imports/repo", base, head))


def test_trials_are_independent_writable_detached_checkouts_without_remote_or_alternates(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    source = tmp_path / ".eval-data" / "imports" / "repo"
    base, head = _init_local_repository(source)
    descriptor = _local_descriptor("imports/repo", base, head)

    with _preparer(tmp_path, suite) as preparer:
        prepared = preparer.prepare(descriptor)
        first = _trial_workspace(preparer, prepared, descriptor, index=1)
        with first as workspace:
            (workspace.path / "agent.tmp").write_text("first", encoding="utf-8")
            assert _git(workspace.path, "rev-parse", "HEAD") == head
            assert _git(workspace.path, "cat-file", "-t", base) == "commit"
            assert _git(workspace.path, "remote") == ""
            assert not (workspace.path / ".git" / "objects" / "info" / "alternates").exists()
            assert _git(workspace.path, "rev-parse", "--abbrev-ref", "HEAD") == "HEAD"
            assert Path(_git(workspace.path, "rev-parse", "--git-common-dir")).name == ".git"
            config_text = (workspace.path / ".git" / "config").read_text(
                encoding="utf-8"
            )
            assert str(prepared.cache_path) not in config_text

            cache_objects = prepared.cache_path / "objects"
            trial_objects = workspace.path / ".git" / "objects"
            cache_files = {
                item.relative_to(cache_objects)
                for item in cache_objects.rglob("*")
                if item.is_file()
            }
            trial_files = {
                item.relative_to(trial_objects)
                for item in trial_objects.rglob("*")
                if item.is_file()
            }
            shared_names = cache_files & trial_files
            assert shared_names
            for relative in shared_names:
                assert not os.path.samefile(
                    cache_objects / relative,
                    trial_objects / relative,
                )

        with _trial_workspace(preparer, prepared, descriptor, index=2) as second:
            assert not (second.path / "agent.tmp").exists()
            assert (second.path / "app.py").write_text("agent edit\n", encoding="utf-8") > 0


def test_gitlinks_and_lfs_are_rejected_by_fixed_v1_policy(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    source = tmp_path / ".eval-data" / "imports" / "repo"
    base, _head = _init_local_repository(source)
    run_git(source, "update-index", "--add", "--cacheinfo", f"160000,{base},vendor")
    run_git(source, "commit", "-m", "gitlink")
    gitlink_head = run_git(source, "rev-parse", "HEAD")

    with _preparer(tmp_path, suite) as preparer:
        with pytest.raises(RepositoryPolicyError, match="submodule|gitlink|nested"):
            preparer.prepare(_local_descriptor("imports/repo", base, gitlink_head))

    run_git(source, "reset", "--hard", base)
    pointer = (
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:" + "a" * 64 + "\n"
        "size 123\n"
    )
    attributes_blob = _git(
        source,
        "hash-object",
        "-w",
        "--stdin",
        input_bytes=b"*.bin filter=lfs diff=lfs merge=lfs -text\n",
    )
    pointer_blob = _git(
        source,
        "hash-object",
        "-w",
        "--stdin",
        input_bytes=pointer.encode("utf-8"),
    )
    run_git(
        source,
        "update-index",
        "--add",
        "--cacheinfo",
        f"100644,{attributes_blob},.gitattributes",
    )
    run_git(
        source,
        "update-index",
        "--add",
        "--cacheinfo",
        f"100644,{pointer_blob},model.bin",
    )
    run_git(source, "commit", "-m", "lfs pointer")
    lfs_head = run_git(source, "rev-parse", "HEAD")

    with _preparer(tmp_path, suite) as preparer:
        with pytest.raises(RepositoryPolicyError, match="LFS|lfs"):
            preparer.prepare(_local_descriptor("imports/repo", base, lfs_head))


def test_symlink_git_entries_are_rejected_before_checkout(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    source = tmp_path / ".eval-data" / "imports" / "repo"
    base, _head = _init_local_repository(source)
    link_blob = _git(
        source,
        "hash-object",
        "-w",
        "--stdin",
        input_bytes=b"../../outside",
    )
    run_git(
        source,
        "update-index",
        "--add",
        "--cacheinfo",
        f"120000,{link_blob},outside-link",
    )
    run_git(source, "commit", "-m", "symlink")
    head = run_git(source, "rev-parse", "HEAD")

    with _preparer(tmp_path, suite) as preparer:
        with pytest.raises(RepositoryPolicyError, match="symlink"):
            preparer.prepare(_local_descriptor("imports/repo", base, head))


@pytest.mark.parametrize(
    "names",
    [
        (b"CON",),
        (b"COM1 .txt",),
        (b"COM2 .json",),
        (b"NUL .json",),
        (b"AUX .x",),
        (b"LPT9 .data",),
        (b"CONIN$",),
        (b"conout$",),
        (b"ConIn$.txt",),
        (b"CONOUT$.log",),
        (b"CONIN$ .txt",),
        (b"cOnOuT$ .log",),
        (b"trailing.",),
        (b"trailing ",),
        (b"stream:ads",),
        (b"dir\\file.py",),
        (b"A.py", b"a.py"),
        ("caf\u00e9.py".encode("utf-8"), "cafe\u0301.py".encode("utf-8")),
        (b"invalid-\xff.py",),
    ],
)
def test_nonportable_or_colliding_git_tree_paths_fail_closed(
    tmp_path: Path,
    names: tuple[bytes, ...],
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    source = tmp_path / ".eval-data" / "imports" / "malicious.git"
    base, head = _init_bare_repository_with_tree_names(source, names)
    with _preparer(tmp_path, suite) as preparer:
        with pytest.raises(RepositoryPolicyError, match="path|portable|collision|UTF"):
            preparer.prepare(_local_descriptor("imports/malicious.git", base, head))


def test_fixture_builder_accepts_windows_reserved_near_miss_and_unicode_names(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "suite" / "repositories" / "near-miss"
    (fixture / "base").mkdir(parents=True)
    (fixture / "head").mkdir()
    for side in ("base", "head"):
        (fixture / side / "COM10 .txt").write_bytes(b"payload")
        (fixture / side / "CONINX$.txt").write_bytes(b"payload")
        (fixture / side / "CONOUTER$.log").write_bytes(b"payload")
        (fixture / side / "普通话.txt").write_bytes(b"payload")

    built = FixtureRepositoryBuilder().build(fixture, tmp_path / "near-miss.git")

    assert built.head_revision


def test_remote_requires_prepare_authorization_and_secret_free_url(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    remote = Repository(
        source=RepositorySource.GIT,
        path=None,
        url="https://example.test/project.git",
        base_revision="a" * 40,
        head_revision="b" * 40,
    )
    with _preparer(tmp_path, suite) as preparer:
        with pytest.raises(RepositoryPreparationError, match="remote|network|authorized"):
            preparer.prepare(remote)

    query_secret = Repository(
        source=RepositorySource.GIT,
        path=None,
        url="https://example.test/project.git?token=secret",
        base_revision="a" * 40,
        head_revision="b" * 40,
    )
    with _preparer(tmp_path, suite, allow_remote=True) as preparer:
        with pytest.raises(RepositorySecurityError, match="query|fragment|credential"):
            preparer.prepare(query_secret)

    for unsupported_url in (
        "http://example.test/project.git",
        "ssh://example.test/project.git",
        "git://example.test/project.git",
    ):
        unsupported = Repository(
            source=RepositorySource.GIT,
            path=None,
            url=unsupported_url,
            base_revision="a" * 40,
            head_revision="b" * 40,
        )
        with _preparer(tmp_path, suite, allow_remote=True) as preparer:
            with pytest.raises(RepositorySecurityError, match="HTTPS|scheme|transport"):
                preparer.prepare(unsupported)

    with _preparer(tmp_path, suite, allow_remote=True) as preparer:
        with pytest.raises(
            RepositoryPreparationError,
            match="attestation|acquisition|binding",
        ) as error:
            preparer.prepare(remote)
        assert remote.url not in str(error.value)


def test_remote_acquisition_binding_is_strict_and_binds_provenance() -> None:
    remote = Repository(
        source=RepositorySource.GIT,
        path=None,
        url="https://example.test/project.git",
        base_revision="a" * 40,
        head_revision="b" * 40,
    )
    suite_source = SuiteSource(
        kind=SuiteKind.PUBLIC,
        source_id="public-benchmark",
        source_version="2026-07-16",
        source_uri="https://example.test/benchmark",
        license="MIT",
        content_hash="d" * 64,
        preparation_binding=_public_preparation_binding(),
    )
    binding = RepositoryAcquisitionBinding(
        repository=remote,
        expected_source_digest="c" * 64,
        suite_source=suite_source,
        allowed_host="example.test",
        allowed_port=443,
    )
    assert RepositoryAcquisitionBinding.from_json(binding.to_json()) == binding

    payload = binding.to_dict()
    assert payload["repository"] == remote.to_dict()
    assert "url" not in payload
    assert "repository_descriptor_digest" not in payload
    assert "allowed_base_revision" not in payload
    assert "allowed_head_revision" not in payload
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="exact|unknown|fields"):
        RepositoryAcquisitionBinding.from_dict(payload)

    with pytest.raises(ValueError, match="remote|Repository|canonical"):
        RepositoryAcquisitionBinding(
            repository=_local_descriptor(
                "imports/not-a-remote",
                remote.base_revision,
                remote.head_revision,
            ),
            expected_source_digest="c" * 64,
            suite_source=suite_source,
            allowed_host="example.test",
            allowed_port=443,
        )


def test_workspace_manifest_is_strict_replayable_and_binds_repository_provenance(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    source = tmp_path / ".eval-data" / "imports" / "repo"
    base, head = _init_local_repository(source)
    descriptor = _local_descriptor("imports/repo", base, head)

    with _preparer(tmp_path, suite) as preparer:
        prepared = preparer.prepare(descriptor)
        with _trial_workspace(preparer, prepared, descriptor) as workspace:
            payload = workspace.manifest.to_dict()
            restored = WorkspaceManifest.from_json(workspace.manifest.to_json())
            assert restored == workspace.manifest
            prepared_payload = payload["prepared_repository"]
            trial_payload = payload["trial_manifest"]
            assert prepared_payload["source_digest"] == prepared.manifest.source_digest
            assert payload["repository"] == descriptor.to_dict()
            assert payload["git_version"] == prepared.git_version
            assert (
                payload["git_executable_sha256"]
                == prepared.git_executable_sha256
            )
            assert "repository_descriptor_digest" not in payload
            assert "git_version" not in prepared_payload
            assert "git_executable_sha256" not in prepared_payload
            assert trial_payload["run_id"] == stable_id("run", "repository-tests")
            assert trial_payload["trial_id"] == _trial_id()
            assert payload["attempt"] == 1
            assert payload["suite_case"]["protocol_id"] == "core-code-review-v1"
            assert trial_payload["eval_input_digest"] == _trial_binding(
                descriptor, 1
            )[2].digest()
            assert prepared_payload["base_revision"] == base
            assert prepared_payload["head_revision"] == head
            policy = prepared_payload["isolation_policy"]
            assert policy["materialization"] == "verified_tree_manual_materialization"
            assert policy["git_object_storage"] == "independent_loose_objects_no_hardlinks"
            assert policy["git_remotes"] == "absent"
            assert policy["lfs"] == "rejected"
            assert policy["trial_network"] == "adapter_required_os_egress_not_proven"
            serialized = json.dumps(payload, sort_keys=True)
            assert "imports/repo" in serialized
            assert str(source) not in serialized
            assert "url" not in payload

            tampered = json.loads(workspace.manifest.to_json())
            tampered["prepared_repository"]["source_digest"] = "e" * 64
            with pytest.raises(ValueError, match="identity|prepared|binding"):
                WorkspaceManifest.from_dict(tampered)

            payload["unexpected"] = True
            with pytest.raises(ValueError, match="exact|unknown|fields"):
                WorkspaceManifest.from_dict(payload)


def test_default_cleanup_is_safe_and_cleanup_failure_is_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import review_agent_eval.repository as repository_module

    suite = tmp_path / "suite"
    descriptor, _built = _author_fixture(suite, tmp_path)
    preparer = _preparer(tmp_path, suite)
    prepared = preparer.prepare(descriptor)
    lease = _trial_workspace(preparer, prepared, descriptor)

    def fail_cleanup(_root: Path, _target: Path) -> None:
        raise OSError("simulated cleanup failure")

    monkeypatch.setattr(repository_module, "_remove_tree_safely", fail_cleanup)
    with lease as workspace:
        assert workspace.path.exists()

    assert lease.path.exists()
    assert lease.cleanup_diagnostic is not None
    assert lease.cleanup_diagnostic.code == "cleanup_failed"
    assert "simulated cleanup failure" in lease.cleanup_diagnostic.message


def test_cleanup_never_follows_agent_created_directory_link(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    descriptor, _built = _author_fixture(suite, tmp_path)
    outside = tmp_path / "outside-cleanup-target"
    outside.mkdir()
    marker = outside / "must-survive.txt"
    marker.write_text("keep", encoding="utf-8")

    with _preparer(tmp_path, suite) as preparer:
        prepared = preparer.prepare(descriptor)
        lease = _trial_workspace(preparer, prepared, descriptor)
        with lease as workspace:
            try:
                (workspace.path / "agent-link").symlink_to(
                    outside,
                    target_is_directory=True,
                )
            except OSError:
                pytest.skip("directory symlink creation is unavailable")
        assert not lease.path.exists()
        assert marker.read_text(encoding="utf-8") == "keep"


def test_cleanup_failure_never_masks_trial_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import review_agent_eval.repository as repository_module

    suite = tmp_path / "suite"
    descriptor, _built = _author_fixture(suite, tmp_path)
    preparer = _preparer(tmp_path, suite)
    prepared = preparer.prepare(descriptor)
    lease = _trial_workspace(preparer, prepared, descriptor)

    def fail_cleanup(_root: Path, _target: Path) -> None:
        raise OSError("cleanup did not win")

    monkeypatch.setattr(repository_module, "_remove_tree_safely", fail_cleanup)
    with pytest.raises(RuntimeError, match="trial failed"):
        with lease:
            raise RuntimeError("trial failed")
    assert lease.cleanup_diagnostic is not None
    assert "cleanup did not win" in lease.cleanup_diagnostic.message


def test_failed_workspace_retention_is_explicit_and_bounded(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    descriptor, _built = _author_fixture(suite, tmp_path)
    with _preparer(
        tmp_path,
        suite,
        retention_policy=WorkspaceRetentionPolicy.RETAIN_ON_FAILURE,
        max_retained_workspaces=2,
        max_retained_bytes=10 * 1024 * 1024,
        retention_ttl_seconds=3600,
    ) as preparer:
        prepared = preparer.prepare(descriptor)
        leases = []
        for index in range(1, 4):
            lease = _trial_workspace(
                preparer, prepared, descriptor, index=index
            )
            with lease as workspace:
                workspace.record_terminal_status(TrialStatus.FAILED)
            leases.append(lease)
        assert not leases[0].retained
        assert leases[1].retained
        assert leases[2].retained
        retained = [entry for entry in preparer.retained_root.iterdir() if entry.is_dir()]
        assert len(retained) == 2


def test_repository_roots_and_source_paths_reject_link_escape(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_workspace = tmp_path / ".eval-workspaces"
    try:
        linked_workspace.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")

    with pytest.raises(RepositorySecurityError, match="link|reparse"):
        RepositoryPreparer(
            suite_root=suite,
            data_root=tmp_path / ".eval-data",
            workspace_root=linked_workspace,
            git_executable=_git_executable(),
        )

    linked_workspace.unlink()
    data_root = tmp_path / ".eval-data"
    data_root.mkdir()
    source_root = tmp_path / "source-root"
    source = source_root / "repo"
    base, head = _init_local_repository(source)
    imports = data_root / "imports"
    try:
        imports.symlink_to(source_root, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation is unavailable")
    with _preparer(tmp_path, suite) as preparer:
        with pytest.raises(RepositorySecurityError, match="link|reparse"):
            preparer.prepare(_local_descriptor("imports/repo", base, head))


def test_agent_workspace_root_cannot_overlap_suite_or_data_roots(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    data = tmp_path / ".eval-data"
    with pytest.raises(RepositorySecurityError, match="overlap|separate|isolat"):
        RepositoryPreparer(
            suite_root=suite,
            data_root=data,
            workspace_root=suite / ".eval-workspaces",
            git_executable=_git_executable(),
        )
    data.mkdir(exist_ok=True)
    with pytest.raises(RepositorySecurityError, match="overlap|separate|isolat"):
        RepositoryPreparer(
            suite_root=suite,
            data_root=data,
            workspace_root=data / ".eval-workspaces",
            git_executable=_git_executable(),
        )

    with pytest.raises(ValueError, match=r"\.eval-data"):
        RepositoryPreparer(
            suite_root=suite,
            data_root=tmp_path / "ordinary-data",
            workspace_root=tmp_path / ".eval-workspaces",
            git_executable=_git_executable(),
        )
    with pytest.raises(ValueError, match=r"\.eval-workspaces"):
        RepositoryPreparer(
            suite_root=suite,
            data_root=data,
            workspace_root=tmp_path / "ordinary-workspaces",
            git_executable=_git_executable(),
        )


def test_prepare_cache_is_content_addressed_and_reused(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    descriptor, _built = _author_fixture(suite, tmp_path)
    with _preparer(tmp_path, suite) as preparer:
        first = preparer.prepare(descriptor)
        second = preparer.prepare(descriptor)
        assert second.cache_path == first.cache_path
        assert second.manifest.source_digest == first.manifest.source_digest
        assert first.cache_path.is_dir()
        assert first.cache_path.parent.name == first.cache_id


def test_cache_manifest_tamper_and_forged_runtime_handle_fail_closed(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    descriptor, _built = _author_fixture(suite, tmp_path)
    with _preparer(tmp_path, suite) as preparer:
        prepared = preparer.prepare(descriptor)
        forged = PreparedRepository(
            manifest=prepared.manifest,
            cache_path=tmp_path / "outside-cache.git",
            repository=prepared.repository,
            acquisition_binding_digest=prepared.acquisition_binding_digest,
            git_version=prepared.git_version,
            git_executable_sha256=prepared.git_executable_sha256,
        )
        with pytest.raises(RepositorySecurityError, match="cache|outside|handle"):
            _trial_workspace(preparer, forged, descriptor)

        manifest_path = prepared.cache_path.parent / "manifest.json"
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["source_digest"] = "0" * 64
        manifest_path.write_text(
            json.dumps(payload, sort_keys=True),
            encoding="utf-8",
        )
        with pytest.raises(RepositoryIntegrityError, match="manifest|modified|drift"):
            preparer.prepare(descriptor)


def test_prepared_repository_rejects_forged_repository_revisions(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    descriptor, _built = _author_fixture(suite, tmp_path)
    with _preparer(tmp_path, suite) as preparer:
        prepared = preparer.prepare(descriptor)
        different_revisions = Repository(
            source=descriptor.source,
            path=descriptor.path,
            url=descriptor.url,
            base_revision="0" * 40,
            head_revision="1" * 40,
        )
        with pytest.raises(
            RepositoryIntegrityError,
            match=r"(?i)(Repository|revision|content)",
        ):
            PreparedRepository(
                manifest=prepared.manifest,
                cache_path=prepared.cache_path,
                repository=different_revisions,
                acquisition_binding_digest=prepared.acquisition_binding_digest,
                git_version=prepared.git_version,
                git_executable_sha256=prepared.git_executable_sha256,
            )


def test_forged_prepared_repository_with_same_revisions_new_locator_cannot_trial(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    descriptor, _built = _author_fixture(suite, tmp_path)
    with _preparer(tmp_path, suite) as preparer:
        prepared = preparer.prepare(descriptor)
        different_locator = _local_descriptor(
            "imports/different-locator",
            descriptor.base_revision,
            descriptor.head_revision,
        )
        forged = PreparedRepository(
            manifest=prepared.manifest,
            cache_path=prepared.cache_path,
            repository=different_locator,
            acquisition_binding_digest=prepared.acquisition_binding_digest,
            git_version=prepared.git_version,
            git_executable_sha256=prepared.git_executable_sha256,
        )

        with pytest.raises(
            RepositoryIntegrityError,
            match=r"(?i)(request|index|binding|provenance)",
        ):
            _trial_workspace(preparer, forged, different_locator)


def test_prepared_repository_budget_policy_is_immutable(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    descriptor, _built = _author_fixture(suite, tmp_path)
    with _preparer(tmp_path, suite) as preparer:
        manifest = preparer.prepare(descriptor).manifest
        original_budget = dict(manifest.budget_policy)

        with pytest.raises(TypeError):
            manifest.budget_policy["actual_objects"] = 0  # type: ignore[index]

        assert dict(manifest.budget_policy) == original_budget
        assert (
            manifest.budget_policy["max_logical_tree_entries"]
            == repository_module.MAX_LOGICAL_TREE_ENTRIES
        )
        assert "actual_cache_bytes" not in manifest.budget_policy
        detached = manifest.to_dict()["budget_policy"]
        detached["actual_objects"] = 0
        assert dict(manifest.budget_policy) == original_budget


def test_prepared_repository_id_binds_every_serialized_content_field(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    descriptor, _built = _author_fixture(suite, tmp_path)
    with _preparer(tmp_path, suite) as preparer:
        manifest = preparer.prepare(descriptor).manifest

    payload = manifest.to_dict()
    mutations: list[tuple[tuple[str, ...], object]] = [
        (("schema_version",), payload["schema_version"] + ".tampered"),
        (
            ("logical_source_version",),
            payload["logical_source_version"] + ".tampered",
        ),
        (("source_digest",), "0" * 64),
        (("base_source_digest",), "1" * 64),
        (("head_source_digest",), "2" * 64),
        (("object_format",), "sha256"),
        (("base_revision",), "0" * 40),
        (("head_revision",), "1" * 40),
        (("base_tree",), "2" * 40),
        (("head_tree",), "3" * 40),
    ]
    for policy_name in ("isolation_policy", "path_policy", "budget_policy"):
        policy = payload[policy_name]
        assert isinstance(policy, dict)
        for field, value in policy.items():
            replacement = value + 1 if isinstance(value, int) else f"{value}.tampered"
            mutations.append(((policy_name, field), replacement))

    for path, replacement in mutations:
        tampered = json.loads(manifest.to_json())
        target = tampered
        for component in path[:-1]:
            child = target[component]
            assert isinstance(child, dict)
            target = child
        target[path[-1]] = replacement
        try:
            PreparedRepositoryManifest.from_dict(tampered)
        except ValueError:
            pass
        else:
            pytest.fail(
                "manifest accepted a stale prepared_repository_id after %s changed"
                % ".".join(path)
            )


def test_git_provenance_does_not_change_pure_content_manifest(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    descriptor, _built = _author_fixture(suite, tmp_path)
    with _preparer(tmp_path, suite) as first_preparer:
        first = first_preparer.prepare(descriptor)

    alternate_version = first.git_version + "-alternate"
    alternate_executable_digest = "0" * 64
    assert alternate_executable_digest != first.git_executable_sha256
    with _preparer(tmp_path, suite) as second_preparer:
        second_preparer._runner.version = alternate_version
        second_preparer._runner.executable_sha256 = alternate_executable_digest
        second = second_preparer.prepare(descriptor)

        assert second.git_version == alternate_version
        assert second.git_executable_sha256 == alternate_executable_digest
        assert second.manifest == first.manifest
        assert second.manifest.prepared_repository_id == first.manifest.prepared_repository_id
        assert second.cache_path == first.cache_path
        assert len(tuple(second_preparer.index_root.glob("*.json"))) == 2

    manifest_payload = first.manifest.to_dict()
    assert "git_version" not in manifest_payload
    assert "git_executable_sha256" not in manifest_payload
    assert "actual_cache_bytes" not in manifest_payload["budget_policy"]


def test_source_digest_is_git_content_identity_not_descriptor_identity(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    data = tmp_path / ".eval-data"
    first_source = data / "imports" / "first"
    base, head = _init_local_repository(first_source)
    second_source = data / "imports" / "second"
    run_git(data, "clone", "--no-local", str(first_source), str(second_source))
    run_git(second_source, "remote", "remove", "origin")

    first_descriptor = _local_descriptor("imports/first", base, head)
    second_descriptor = _local_descriptor("imports/second", base, head)
    assert first_descriptor.digest() != second_descriptor.digest()

    with _preparer(tmp_path, suite) as preparer:
        first = preparer.prepare(first_descriptor)
        second = preparer.prepare(second_descriptor)
        assert first.manifest.source_digest == second.manifest.source_digest
        assert first.cache_path == second.cache_path
        assert first.repository == first_descriptor
        assert second.repository == second_descriptor
        assert first.repository != second.repository


def test_concurrent_descriptors_with_identical_git_content_share_one_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    data = tmp_path / ".eval-data"
    first_source = data / "imports" / "first"
    base, head = _init_local_repository(first_source)
    second_source = data / "imports" / "second"
    run_git(data, "clone", "--no-local", str(first_source), str(second_source))
    run_git(second_source, "remote", "remove", "origin")

    first_descriptor = _local_descriptor("imports/first", base, head)
    second_descriptor = _local_descriptor("imports/second", base, head)
    assert first_descriptor.digest() != second_descriptor.digest()

    publish_barrier = threading.Barrier(2)
    original_publish = RepositoryPreparer._publish_cache

    def synchronized_publish(
        preparer: RepositoryPreparer,
        *args: object,
        **kwargs: object,
    ):
        publish_barrier.wait(timeout=10)
        return original_publish(preparer, *args, **kwargs)

    monkeypatch.setattr(
        RepositoryPreparer,
        "_publish_cache",
        synchronized_publish,
    )

    with _preparer(tmp_path, suite) as first_preparer:
        with _preparer(tmp_path, suite) as second_preparer:
            with ThreadPoolExecutor(max_workers=2) as executor:
                first_future = executor.submit(
                    first_preparer.prepare,
                    first_descriptor,
                )
                second_future = executor.submit(
                    second_preparer.prepare,
                    second_descriptor,
                )
                first = first_future.result(timeout=30)
                second = second_future.result(timeout=30)

            assert first.cache_path == second.cache_path
            assert first.cache_id == second.cache_id
            assert first.manifest == second.manifest
            assert first.repository == first_descriptor
            assert second.repository == second_descriptor
            assert first.repository != second.repository
            cache_entries = tuple(
                path for path in first_preparer.cache_root.iterdir() if path.is_dir()
            )
            assert cache_entries == (first.cache_path.parent,)
            assert len(tuple(first_preparer.index_root.glob("*.json"))) == 2
            assert not tuple(first_preparer.staging_root.glob("operation-*"))

            manifest_path = first.cache_path.parent / "manifest.json"
            manifest_bytes = manifest_path.read_bytes()
            assert first_preparer.prepare(first_descriptor).cache_path == first.cache_path
            assert (
                second_preparer.prepare(second_descriptor).cache_path
                == first.cache_path
            )
            assert manifest_path.read_bytes() == manifest_bytes


def test_initialization_reclaims_only_unlocked_orphan_operation_staging(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    data = tmp_path / ".eval-data"
    staging = data / ".staging"
    locks = data / ".locks"
    staging.mkdir(parents=True)
    locks.mkdir()

    orphan_request_id = stable_id("repository-request", "orphan")
    held_request_id = stable_id("repository-request", "active")
    orphan_lock_key = repository_module._request_lock_key(orphan_request_id)
    held_lock_key = repository_module._request_lock_key(held_request_id)
    orphan = staging / f"operation-{orphan_lock_key}"
    held = staging / f"operation-{held_lock_key}"
    orphan.mkdir()
    held.mkdir()
    (orphan / "partial-object").write_bytes(b"orphan")
    (held / "partial-object").write_bytes(b"active")
    outside_staging = data / "imports" / "do-not-touch"
    outside_staging.parent.mkdir()
    outside_staging.write_bytes(b"preserve me")

    held_lock_path = locks / f"request-{held_lock_key}.lock"
    with repository_module._ProcessLock(held_lock_path, 1.0):
        with _preparer(
            tmp_path,
            suite,
            lock_timeout_seconds=0.05,
        ):
            assert not orphan.exists()
            assert held.is_dir()
            assert (held / "partial-object").read_bytes() == b"active"
            assert outside_staging.read_bytes() == b"preserve me"

    assert held.is_dir()
    assert outside_staging.read_bytes() == b"preserve me"


def test_operation_lease_prevents_recovery_from_deleting_live_staging(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    data = tmp_path / ".eval-data"
    staging = data / ".staging"
    locks = data / ".locks"
    reservations = data / ".reservations"
    staging.mkdir(parents=True)
    locks.mkdir()
    reservations.mkdir()
    request_id = stable_id("repository-request", "child-operation")
    request_lock_key = repository_module._request_lock_key(request_id)
    operation = staging / f"operation-{request_lock_key}"
    operation.mkdir()
    (operation / "live-write").write_bytes(b"in progress")
    reservation = reservations / f"{request_lock_key}.json"
    reservation.write_text(
        json.dumps(
            repository_module._repository_reservation_payload(request_id),
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    with repository_module._OperationLease(
        locks, request_lock_key, 1.0
    ):
        with _preparer(
            tmp_path,
            suite,
            lock_timeout_seconds=0.05,
        ):
            assert operation.is_dir()
            assert reservation.is_file()
            assert (operation / "live-write").read_bytes() == b"in progress"

    with _preparer(tmp_path, suite, lock_timeout_seconds=0.05):
        assert not operation.exists()
        assert not reservation.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows inherited sentinel regression")
def test_new_prepare_cannot_reuse_child_held_windows_sentinel(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    data = tmp_path / ".eval-data"
    source = data / "imports" / "repo"
    base, head = _init_local_repository(source)
    descriptor = _local_descriptor("imports/repo", base, head)
    with _preparer(tmp_path, suite) as initial:
        request_id = repository_module._request_id(
            descriptor.digest(),
            None,
            initial._runner.version,
            initial._runner.executable_sha256,
        )

    request_lock_key = repository_module._request_lock_key(request_id)
    operation = data / ".staging" / f"operation-{request_lock_key}"
    operation.mkdir()
    (operation / "child-write").write_bytes(b"still live")
    reservation = data / ".reservations" / f"{request_lock_key}.json"
    reservation.write_text(
        json.dumps(
            repository_module._repository_reservation_payload(request_id),
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    sentinel_path = data / ".locks" / f"operation-{request_lock_key}.lease"
    sentinel_handle = repository_module._windows_open_operation_sentinel(
        sentinel_path
    )
    try:
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.lpAttributeList = {"handle_list": [sentinel_handle]}
        with subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import sys; "
                    "sys.stdout.buffer.write(b'ready\\n'); "
                    "sys.stdout.buffer.flush(); "
                    "sys.stdin.buffer.read(1)"
                ),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            startupinfo=startupinfo,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ) as child:
            assert child.stdout is not None
            assert child.stdout.readline() == b"ready\n"
            repository_module._windows_close_handle(sentinel_handle)
            sentinel_handle = 0
            assert repository_module._windows_operation_sentinel_is_held(
                sentinel_path
            )

            with _preparer(
                tmp_path,
                suite,
                lock_timeout_seconds=0.05,
            ) as recovered:
                with pytest.raises(
                    RepositoryPreparationError,
                    match=r"(?i)(operation|lease|child|held)",
                ):
                    recovered.prepare(descriptor)
                assert reservation.is_file()
                assert (operation / "child-write").read_bytes() == b"still live"

                assert child.stdin is not None
                child.stdin.write(b"x")
                child.stdin.close()
                assert child.wait(timeout=5) == 0
                assert not repository_module._windows_operation_sentinel_is_held(
                    sentinel_path
                )
                prepared = recovered.prepare(descriptor)
                assert prepared.repository == descriptor
                assert not reservation.exists()
                assert not operation.exists()
    finally:
        if sentinel_handle:
            repository_module._windows_close_handle(sentinel_handle)


@pytest.mark.skipif(os.name != "nt", reason="Windows DACL regression")
def test_windows_control_tree_replaces_unrelated_explicit_aces(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    data = tmp_path / ".eval-data"
    lock_root = data / ".locks"
    lock_root.mkdir(parents=True)
    poisoned = lock_root / "poisoned.lock"
    poisoned.write_bytes(b"lock")
    icacls = shutil.which("icacls.exe") or shutil.which("icacls")
    assert icacls is not None
    result = subprocess.run(
        [
            icacls,
            str(poisoned),
            "/grant",
            "*S-1-5-32-546:F",
            "/Q",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
    )
    if result.returncode != 0:
        pytest.skip("could not inject a Windows explicit ACE")

    with _preparer(tmp_path, suite):
        assert poisoned.is_file()
        repository_module._replace_and_verify_windows_control_acl(
            poisoned,
            "poisoned lock verification",
            repository_module._windows_current_user_sid(),
        )


@pytest.mark.parametrize(
    "quota_name",
    ["MAX_DATA_ROOT_BYTES", "MAX_DATA_ROOT_NODES"],
    ids=["bytes", "nodes"],
)
def test_data_root_global_quota_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quota_name: str,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    data = tmp_path / ".eval-data"
    data.mkdir()
    if quota_name == "MAX_DATA_ROOT_BYTES":
        preserved = {data / "oversized.bin": b"ab"}
    else:
        preserved = {
            data / "first-node": b"",
            data / "second-node": b"",
        }
    for path, content in preserved.items():
        path.write_bytes(content)

    monkeypatch.setattr(repository_module, quota_name, 1)
    with pytest.raises(
        RepositoryLimitError,
        match=r"(?i)(\.eval-data|data root|quota|budget|byte|node)",
    ):
        _preparer(tmp_path, suite)

    for path, content in preserved.items():
        assert path.read_bytes() == content


def test_concurrent_prepare_capacity_is_reserved_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    data = tmp_path / ".eval-data"
    first_source = data / "imports" / "first"
    base, head = _init_local_repository(first_source)
    second_source = data / "imports" / "second"
    run_git(data, "clone", "--no-local", str(first_source), str(second_source))
    run_git(second_source, "remote", "remove", "origin")
    first_descriptor = _local_descriptor("imports/first", base, head)
    second_descriptor = _local_descriptor("imports/second", base, head)

    first_entered = threading.Event()
    release_first = threading.Event()
    original_acquire = RepositoryPreparer._acquire_closure

    def blocking_acquire(
        preparer: RepositoryPreparer,
        descriptor: Repository,
        *args: object,
        **kwargs: object,
    ):
        if descriptor == first_descriptor:
            first_entered.set()
            assert release_first.wait(timeout=20)
        return original_acquire(preparer, descriptor, *args, **kwargs)

    monkeypatch.setattr(
        RepositoryPreparer,
        "_acquire_closure",
        blocking_acquire,
    )
    with _preparer(tmp_path, suite) as first_preparer:
        with _preparer(tmp_path, suite) as second_preparer:
            monkeypatch.setattr(
                repository_module,
                "MAX_DATA_ROOT_BYTES",
                2 * repository_module.MAX_PREPARE_RESERVATION_BYTES - 1,
            )
            with ThreadPoolExecutor(max_workers=1) as executor:
                first_future = executor.submit(
                    first_preparer.prepare, first_descriptor
                )
                assert first_entered.wait(timeout=20)
                try:
                    with pytest.raises(
                        RepositoryLimitError,
                        match=r"(?i)(quota|capacity|byte)",
                    ):
                        second_preparer.prepare(second_descriptor)
                finally:
                    release_first.set()
                first = first_future.result(timeout=30)

            assert first.repository == first_descriptor
            assert not tuple(first_preparer.reservation_root.glob("*.json"))


def test_startup_reclaims_unindexed_cache_and_stale_reservation(
    tmp_path: Path,
) -> None:
    suite = tmp_path / "suite"
    descriptor, _built = _author_fixture(suite, tmp_path)
    with _preparer(tmp_path, suite) as preparer:
        prepared = preparer.prepare(descriptor)
        request_id = repository_module._request_id(
            descriptor.digest(),
            prepared.acquisition_binding_digest,
            prepared.git_version,
            prepared.git_executable_sha256,
        )
        index_path = preparer.index_root / f"{request_id}.json"
        cache_entry = prepared.cache_path.parent
        assert index_path.is_file()
        assert cache_entry.is_dir()

    index_path.unlink()
    request_lock_key = repository_module._request_lock_key(request_id)
    reservation_path = (
        tmp_path
        / ".eval-data"
        / ".reservations"
        / f"{request_lock_key}.json"
    )
    reservation_path.write_text(
        json.dumps(
            repository_module._repository_reservation_payload(request_id),
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    operation = (
        tmp_path
        / ".eval-data"
        / ".staging"
        / f"operation-{request_lock_key}"
    )
    operation.mkdir()
    (operation / "partial").write_bytes(b"partial")

    with _preparer(tmp_path, suite) as recovered:
        assert not reservation_path.exists()
        assert not operation.exists()
        assert not cache_entry.exists()
        assert not tuple(recovered.cache_root.iterdir())


@pytest.mark.parametrize(
    "address",
    ["127.0.0.1", "10.0.0.1", "169.254.10.20", "0.0.0.0"],
    ids=["loopback", "private", "link-local", "unspecified"],
)
def test_remote_prepare_rejects_non_global_dns_before_git_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    address: str,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    remote, binding = _remote_repository_and_binding()
    fetch_calls: list[object] = []

    def fake_getaddrinfo(*_args: object, **_kwargs: object):
        return [
            (
                repository_module.socket.AF_INET,
                repository_module.socket.SOCK_STREAM,
                repository_module.socket.IPPROTO_TCP,
                "",
                (address, 443),
            )
        ]

    def unexpected_fetch(*args: object, **_kwargs: object) -> None:
        fetch_calls.append(args)
        raise AssertionError("Git fetch started before DNS policy rejection")

    monkeypatch.setattr(
        repository_module.socket,
        "getaddrinfo",
        fake_getaddrinfo,
    )
    monkeypatch.setattr(
        repository_module,
        "_fetch_quarantine",
        unexpected_fetch,
    )

    with _preparer(
        tmp_path,
        suite,
        allow_remote=True,
        acquisition_bindings=(binding,),
    ) as preparer:
        with pytest.raises(
            RepositorySecurityError,
            match=r"(?i)(DNS|public|global|address)",
        ):
            preparer.prepare(remote)
    assert fetch_calls == []


def test_remote_prepare_requires_git_endpoint_pinning_capability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    remote, binding = _remote_repository_and_binding()
    dns_calls: list[object] = []

    with _preparer(
        tmp_path,
        suite,
        allow_remote=True,
        acquisition_bindings=(binding,),
    ) as preparer:
        preparer._runner._curlopt_resolve_supported = False

        def unexpected_dns(*args: object, **_kwargs: object):
            dns_calls.append(args)
            raise AssertionError("DNS ran without a verified Git pinning capability")

        monkeypatch.setattr(
            repository_module.socket,
            "getaddrinfo",
            unexpected_dns,
        )
        with pytest.raises(
            RepositorySecurityError,
            match=r"(?i)(Git|support|pinned|resolution)",
        ):
            preparer.prepare(remote)
    assert dns_calls == []


def test_remote_prepare_pins_public_dns_answers_with_curlopt_resolve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    remote, binding = _remote_repository_and_binding()
    ipv4 = "1.1.1.1"
    ipv6 = "2606:4700:4700::1111"
    dns_queries: list[tuple[object, object]] = []

    def fake_getaddrinfo(
        host: object,
        port: object,
        **_kwargs: object,
    ):
        dns_queries.append((host, port))
        return [
            (
                repository_module.socket.AF_INET6,
                repository_module.socket.SOCK_STREAM,
                repository_module.socket.IPPROTO_TCP,
                "",
                (ipv6, 443, 0, 0),
            ),
            (
                repository_module.socket.AF_INET,
                repository_module.socket.SOCK_STREAM,
                repository_module.socket.IPPROTO_TCP,
                "",
                (ipv4, 443),
            ),
            (
                repository_module.socket.AF_INET,
                repository_module.socket.SOCK_STREAM,
                repository_module.socket.IPPROTO_TCP,
                "",
                (ipv4, 443),
            ),
        ]

    monkeypatch.setattr(
        repository_module.socket,
        "getaddrinfo",
        fake_getaddrinfo,
    )

    class FetchBoundaryReached(RuntimeError):
        pass

    captured: dict[str, object] = {}

    with _preparer(
        tmp_path,
        suite,
        allow_remote=True,
        acquisition_bindings=(binding,),
    ) as preparer:

        def capture_fetch_command(
            args: object,
            **kwargs: object,
        ) -> None:
            captured["args"] = list(args)  # type: ignore[arg-type]
            captured["kwargs"] = kwargs
            raise FetchBoundaryReached

        monkeypatch.setattr(preparer._runner, "run", capture_fetch_command)
        with pytest.raises(FetchBoundaryReached):
            preparer.prepare(remote)
        assert not tuple(preparer.staging_root.glob("operation-*"))
        assert not tuple(preparer.reservation_root.glob("*.json"))

    args = captured["args"]
    kwargs = captured["kwargs"]
    assert isinstance(args, list)
    assert isinstance(kwargs, dict)
    resolve_values = {
        args[index + 1]
        for index in range(len(args) - 1)
        if args[index] == "-c"
        and args[index + 1].startswith("http.curloptResolve=")
    }
    assert resolve_values == {
        "http.curloptResolve=example.test:443:1.1.1.1",
        "http.curloptResolve=example.test:443:[2606:4700:4700::1111]",
    }
    assert "fetch" in args
    assert remote.url in args
    assert kwargs["allow_https"] is True
    assert kwargs["allow_file"] is False
    assert dns_queries == [("example.test", 443)]
