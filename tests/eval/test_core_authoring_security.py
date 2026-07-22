from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Callable, Iterator

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
AUTHORING_SCRIPT = REPOSITORY_ROOT / "eval" / "authoring" / "build_core_suites.py"


UNSAFE_OUTPUT_PATHS = (
    "",
    ".",
    "./case.json",
    "cases//case.json",
    "cases/./case.json",
    "cases/core/",
    "cases/core/../case.json",
    "../outside.json",
    "cases/core/../../../outside.json",
    "/absolute.json",
    "C:/absolute.json",
    "C:drive-relative.json",
    r"C:\absolute.json",
    r"cases\core\case.json",
    r"..\outside.json",
    "cases/core\\../../outside.json",
)


class _OutsideMutationAttempt(BaseException):
    pass


@pytest.fixture(scope="module")
def authoring_module() -> Iterator[ModuleType]:
    module_name = "_review_agent_core_authoring_security"
    spec = importlib.util.spec_from_file_location(module_name, AUTHORING_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)


def _normalized_absolute(path: os.PathLike[str] | str) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _inside(root: Path, candidate: os.PathLike[str] | str) -> bool:
    root_text = _normalized_absolute(root.resolve(strict=True))
    candidate_path = Path(candidate)
    try:
        candidate_path = candidate_path.resolve(strict=False)
    except OSError:
        candidate_path = Path(_normalized_absolute(candidate_path))
    candidate_text = _normalized_absolute(candidate_path)
    try:
        common = os.path.commonpath((root_text, candidate_text))
    except ValueError:
        return False
    return os.path.normcase(common) == root_text


def _install_outside_mutation_guard(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    eval_root: Path,
) -> None:
    """Fail before a vulnerable implementation can mutate through an escape."""

    real_mkdir = Path.mkdir
    real_write_bytes = Path.write_bytes
    real_unlink = module.os.unlink
    real_replace = module.os.replace
    real_named_temporary_file = module.tempfile.NamedTemporaryFile

    def require_inside(path: os.PathLike[str] | str, operation: str) -> None:
        if not _inside(eval_root, path):
            raise _OutsideMutationAttempt(
                "%s attempted outside eval_root via %s" % (operation, path)
            )

    def guarded_mkdir(path: Path, *args: object, **kwargs: object) -> None:
        require_inside(path, "mkdir")
        real_mkdir(path, *args, **kwargs)

    def guarded_write_bytes(path: Path, data: bytes) -> int:
        require_inside(path, "write_bytes")
        return real_write_bytes(path, data)

    def guarded_unlink(
        path: os.PathLike[str] | str, *args: object, **kwargs: object
    ) -> None:
        require_inside(path, "unlink")
        real_unlink(path, *args, **kwargs)

    def guarded_replace(
        source: os.PathLike[str] | str,
        destination: os.PathLike[str] | str,
        *args: object,
        **kwargs: object,
    ) -> None:
        require_inside(source, "replace source")
        require_inside(destination, "replace destination")
        real_replace(source, destination, *args, **kwargs)

    def guarded_named_temporary_file(*args: object, **kwargs: object):
        directory = kwargs.get("dir")
        if directory is not None:
            require_inside(directory, "temporary-file creation")
        return real_named_temporary_file(*args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", guarded_mkdir)
    monkeypatch.setattr(Path, "write_bytes", guarded_write_bytes)
    monkeypatch.setattr(module.os, "unlink", guarded_unlink)
    monkeypatch.setattr(module.os, "replace", guarded_replace)
    monkeypatch.setattr(
        module.tempfile,
        "NamedTemporaryFile",
        guarded_named_temporary_file,
    )


def _assert_rejected(
    expected_exception: type[Exception], action: Callable[[], object]
) -> None:
    try:
        with pytest.raises(expected_exception):
            action()
    except _OutsideMutationAttempt as exc:
        pytest.fail(str(exc))


def _remove_directory_link(path: Path) -> None:
    if not os.path.lexists(path):
        return
    if path.is_symlink() or os.name != "nt":
        path.unlink()
        return
    # A Windows junction is a directory reparse point but Path.is_symlink()
    # may report False. rmdir removes the junction itself without traversing it.
    os.rmdir(path)


@contextmanager
def _directory_link_or_skip(link: Path, target: Path) -> Iterator[None]:
    error: BaseException | None = None
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        error = exc
        if os.name == "nt":
            try:
                completed = subprocess.run(
                    ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
            except OSError as junction_error:
                error = junction_error
            else:
                if completed.returncode != 0 or not os.path.lexists(link):
                    error = OSError(
                        "mklink /J failed: %s"
                        % (completed.stderr.strip() or completed.stdout.strip())
                    )
    if not os.path.lexists(link):
        pytest.skip("directory symlink/junction creation is unavailable: %s" % error)
    try:
        yield
    finally:
        _remove_directory_link(link)


@contextmanager
def _file_symlink_or_skip(link: Path, target: Path) -> Iterator[None]:
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError) as exc:
        pytest.skip("file symlink creation is unavailable: %s" % exc)
    try:
        yield
    finally:
        if os.path.lexists(link):
            link.unlink()


def _invoke(
    module: ModuleType,
    operation: str,
    eval_root: Path,
    relative: str,
    expected: bytes,
) -> object:
    if operation == "scan":
        return module._existing_generated_files(eval_root)
    if operation == "check":
        return module.check_outputs(
            eval_root,
            _build_plan(module, writable_outputs={relative: expected}),
        )
    if operation == "write":
        return module.write_outputs(
            eval_root,
            _build_plan(
                module,
                writable_outputs={relative: b"attacker replacement"},
            ),
        )
    raise AssertionError("unknown operation: %s" % operation)


def _build_plan(
    module: ModuleType,
    *,
    writable_outputs: dict[str, bytes] | None = None,
    check_only_fixtures: dict[str, bytes] | None = None,
) -> object:
    return module.CoreBuildPlan(
        writable_outputs={} if writable_outputs is None else writable_outputs,
        check_only_fixtures=(
            {} if check_only_fixtures is None else check_only_fixtures
        ),
    )


def _write_test_file(eval_root: Path, relative: str, data: bytes) -> Path:
    target = eval_root.joinpath(*relative.split("/"))
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return target


FIXTURE_BASE_PATH = (
    "cases/core/core-py-test/repository/base/src/input.py"
)
FIXTURE_HEAD_PATH = (
    "cases/core/core-py-test/repository/head/src/input.py"
)
DERIVED_CASE_PATH = "cases/core/core-py-test/case.json"


def test_authoring_module_exposes_only_the_core_v2_projection(
    authoring_module: ModuleType,
) -> None:
    assert authoring_module.CORE_SOURCE_VERSION == "core-2026-07-21-v3"
    assert authoring_module.CASE_VERSION == 3
    assert authoring_module.REPOSITORY_WIRE_CONTRACT == {
        "case_schema_version": "eval_case_v2",
        "input_schema_version": "eval_input_v2",
        "submission_schema_version": "eval_submission_v2",
        "review_target_kind": "repository",
        "materializer_protocol": "repository-materializer-v2",
    }
    assert authoring_module.HUMAN_REVIEW_STATUS == (
        "requires_independent_re_review"
    )


def test_write_rejects_drifted_fixture_before_any_derived_output_mutation(
    authoring_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    drifted = b"locally modified immutable fixture"
    fixture = _write_test_file(eval_root, FIXTURE_BASE_PATH, drifted)
    original_derived = b"existing derived output"
    derived = _write_test_file(eval_root, DERIVED_CASE_PATH, original_derived)
    plan = _build_plan(
        authoring_module,
        writable_outputs={DERIVED_CASE_PATH: b"replacement derived output"},
        check_only_fixtures={FIXTURE_BASE_PATH: b"expected fixture"},
    )
    writes: list[str] = []
    monkeypatch.setattr(
        authoring_module,
        "_write_bytes_safely",
        lambda _root, relative, _data: writes.append(relative),
    )

    with pytest.raises(RuntimeError, match="check-only fixture validation failed"):
        authoring_module.write_outputs(eval_root, plan)

    assert writes == []
    assert fixture.read_bytes() == drifted
    assert derived.read_bytes() == original_derived


def test_write_rejects_missing_fixture_without_creating_or_mutating_anything(
    authoring_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    original_derived = b"existing derived output"
    derived = _write_test_file(eval_root, DERIVED_CASE_PATH, original_derived)
    missing = eval_root.joinpath(*FIXTURE_BASE_PATH.split("/"))
    plan = _build_plan(
        authoring_module,
        writable_outputs={DERIVED_CASE_PATH: b"replacement derived output"},
        check_only_fixtures={FIXTURE_BASE_PATH: b"expected fixture"},
    )
    writes: list[str] = []
    monkeypatch.setattr(
        authoring_module,
        "_write_bytes_safely",
        lambda _root, relative, _data: writes.append(relative),
    )

    with pytest.raises(RuntimeError, match="check-only fixture validation failed"):
        authoring_module.write_outputs(eval_root, plan)

    assert writes == []
    assert not os.path.lexists(missing)
    assert derived.read_bytes() == original_derived


def test_write_rejects_symlinked_fixture_without_replacing_it(
    authoring_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    fixture = eval_root.joinpath(*FIXTURE_BASE_PATH.split("/"))
    fixture.parent.mkdir(parents=True)
    outside = tmp_path / "outside-fixture.py"
    sentinel = b"outside fixture target"
    outside.write_bytes(sentinel)
    plan = _build_plan(
        authoring_module,
        writable_outputs={DERIVED_CASE_PATH: b"derived"},
        check_only_fixtures={FIXTURE_BASE_PATH: sentinel},
    )
    writes: list[str] = []
    monkeypatch.setattr(
        authoring_module,
        "_write_bytes_safely",
        lambda _root, relative, _data: writes.append(relative),
    )

    with _file_symlink_or_skip(fixture, outside):
        with pytest.raises(RuntimeError, match="check-only fixture"):
            authoring_module.write_outputs(eval_root, plan)
        assert writes == []
        assert fixture.is_symlink()
        assert outside.read_bytes() == sentinel


def test_write_rejects_reparse_marked_fixture_without_replacing_it(
    authoring_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    sentinel = b"reparse-marked fixture"
    fixture = _write_test_file(eval_root, FIXTURE_BASE_PATH, sentinel)
    plan = _build_plan(
        authoring_module,
        writable_outputs={DERIVED_CASE_PATH: b"derived"},
        check_only_fixtures={FIXTURE_BASE_PATH: sentinel},
    )
    writes: list[str] = []
    monkeypatch.setattr(
        authoring_module,
        "_write_bytes_safely",
        lambda _root, relative, _data: writes.append(relative),
    )
    _mark_as_reparse_point(monkeypatch, authoring_module, fixture)

    with pytest.raises(RuntimeError, match="check-only fixture"):
        authoring_module.write_outputs(eval_root, plan)

    assert writes == []
    assert fixture.read_bytes() == sentinel


def test_write_rejects_non_regular_fixture_without_replacing_it(
    authoring_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    fixture = eval_root.joinpath(*FIXTURE_BASE_PATH.split("/"))
    fixture.mkdir(parents=True)
    plan = _build_plan(
        authoring_module,
        writable_outputs={DERIVED_CASE_PATH: b"derived"},
        check_only_fixtures={FIXTURE_BASE_PATH: b"expected fixture"},
    )
    writes: list[str] = []
    monkeypatch.setattr(
        authoring_module,
        "_write_bytes_safely",
        lambda _root, relative, _data: writes.append(relative),
    )

    with pytest.raises(RuntimeError, match="check-only fixture"):
        authoring_module.write_outputs(eval_root, plan)

    assert writes == []
    assert fixture.is_dir()


def test_write_never_targets_check_only_repository_fixture_paths(
    authoring_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    fixture_bytes = b"immutable fixture"
    _write_test_file(eval_root, FIXTURE_BASE_PATH, fixture_bytes)
    plan = _build_plan(
        authoring_module,
        writable_outputs={DERIVED_CASE_PATH: b"derived"},
        check_only_fixtures={FIXTURE_BASE_PATH: fixture_bytes},
    )
    writes: list[str] = []
    monkeypatch.setattr(
        authoring_module,
        "_write_bytes_safely",
        lambda _root, relative, _data: writes.append(relative),
    )

    authoring_module.write_outputs(eval_root, plan)

    assert writes == [DERIVED_CASE_PATH]
    assert not any("/repository/base/" in item for item in writes)
    assert not any("/repository/head/" in item for item in writes)


def test_check_reports_drifted_and_missing_check_only_fixtures(
    authoring_module: ModuleType,
    tmp_path: Path,
) -> None:
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    _write_test_file(eval_root, FIXTURE_BASE_PATH, b"drifted fixture")
    _write_test_file(eval_root, DERIVED_CASE_PATH, b"derived")
    plan = _build_plan(
        authoring_module,
        writable_outputs={DERIVED_CASE_PATH: b"derived"},
        check_only_fixtures={
            FIXTURE_BASE_PATH: b"expected base fixture",
            FIXTURE_HEAD_PATH: b"expected head fixture",
        },
    )

    assert authoring_module.check_outputs(eval_root, plan) == [
        "drifted check-only fixture: " + FIXTURE_BASE_PATH,
        "missing check-only fixture: " + FIXTURE_HEAD_PATH,
    ]


def test_inventory_accepts_known_fixtures_and_rejects_unexpected_derived_files(
    authoring_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    fixture_bytes = b"immutable fixture"
    _write_test_file(eval_root, FIXTURE_BASE_PATH, fixture_bytes)
    _write_test_file(eval_root, DERIVED_CASE_PATH, b"derived")
    unexpected = "cases/core/unexpected-derived.json"
    _write_test_file(eval_root, unexpected, b"stale")
    plan = _build_plan(
        authoring_module,
        writable_outputs={DERIVED_CASE_PATH: b"derived"},
        check_only_fixtures={FIXTURE_BASE_PATH: fixture_bytes},
    )

    assert authoring_module.check_outputs(eval_root, plan) == [
        "unexpected: " + unexpected
    ]
    writes: list[str] = []
    monkeypatch.setattr(
        authoring_module,
        "_write_bytes_safely",
        lambda _root, relative, _data: writes.append(relative),
    )
    with pytest.raises(RuntimeError, match="unexpected generated files"):
        authoring_module.write_outputs(eval_root, plan)
    assert writes == []


@pytest.mark.parametrize("relative", UNSAFE_OUTPUT_PATHS)
def test_check_outputs_rejects_noncanonical_or_escaping_output_paths(
    authoring_module: ModuleType,
    tmp_path: Path,
    relative: str,
) -> None:
    eval_root = tmp_path / "eval"
    eval_root.mkdir()

    with pytest.raises(ValueError):
        authoring_module.check_outputs(
            eval_root,
            _build_plan(
                authoring_module,
                writable_outputs={relative: b"expected"},
            ),
        )


@pytest.mark.parametrize("relative", UNSAFE_OUTPUT_PATHS)
def test_write_outputs_rejects_noncanonical_or_escaping_output_paths_without_mutation(
    authoring_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
) -> None:
    eval_root = tmp_path / "eval"
    eval_root.mkdir()
    _install_outside_mutation_guard(monkeypatch, authoring_module, eval_root)

    _assert_rejected(
        ValueError,
        lambda: authoring_module.write_outputs(
            eval_root,
            _build_plan(
                authoring_module,
                writable_outputs={relative: b"payload"},
            ),
        ),
    )


@pytest.mark.parametrize("operation", ("scan", "check", "write"))
def test_authoring_rejects_a_target_file_symlink_without_touching_its_target(
    authoring_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    eval_root = tmp_path / "eval"
    generated = eval_root / "cases" / "core"
    generated.mkdir(parents=True)
    outside = tmp_path / "outside-target.bin"
    sentinel = b"outside target must survive"
    outside.write_bytes(sentinel)
    relative = "cases/core/linked.bin"
    linked = generated / "linked.bin"

    with _file_symlink_or_skip(linked, outside):
        if operation == "write":
            _install_outside_mutation_guard(monkeypatch, authoring_module, eval_root)
        _assert_rejected(
            RuntimeError,
            lambda: _invoke(
                authoring_module,
                operation,
                eval_root,
                relative,
                sentinel,
            ),
        )
        assert outside.read_bytes() == sentinel
        assert linked.is_symlink()


@pytest.mark.parametrize("operation", ("scan", "check", "write"))
def test_authoring_rejects_a_linked_parent_directory_without_external_mutation(
    authoring_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    eval_root = tmp_path / "eval"
    generated = eval_root / "cases" / "core"
    generated.mkdir(parents=True)
    outside = tmp_path / "outside-directory"
    outside.mkdir()
    sentinel = b"outside directory must survive"
    marker = outside / "payload.bin"
    marker.write_bytes(sentinel)
    linked_parent = generated / "linked-parent"
    relative = "cases/core/linked-parent/payload.bin"

    with _directory_link_or_skip(linked_parent, outside):
        if operation == "write":
            _install_outside_mutation_guard(monkeypatch, authoring_module, eval_root)
        _assert_rejected(
            RuntimeError,
            lambda: _invoke(
                authoring_module,
                operation,
                eval_root,
                relative,
                sentinel,
            ),
        )
        assert marker.read_bytes() == sentinel
        assert sorted(path.name for path in outside.iterdir()) == ["payload.bin"]


def test_existing_generated_files_rejects_a_linked_eval_root(
    authoring_module: ModuleType,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside-eval"
    generated = outside / "cases" / "core"
    generated.mkdir(parents=True)
    marker = generated / "must-survive.bin"
    sentinel = b"do not traverse the linked eval root"
    marker.write_bytes(sentinel)
    linked_eval_root = tmp_path / "eval-link"

    with _directory_link_or_skip(linked_eval_root, outside):
        with pytest.raises(RuntimeError):
            authoring_module._existing_generated_files(linked_eval_root)
        assert marker.read_bytes() == sentinel


def _mark_as_reparse_point(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    target: Path,
) -> None:
    real_path_lstat = Path.lstat
    real_scandir = module.os.scandir
    target_key = _normalized_absolute(target)

    class ReparseMetadata:
        def __init__(self, wrapped: os.stat_result) -> None:
            self._wrapped = wrapped
            self.st_file_attributes = (
                getattr(wrapped, "st_file_attributes", 0) | 0x0400
            )

        def __getattr__(self, name: str) -> object:
            return getattr(self._wrapped, name)

    def marked_path_lstat(path: Path) -> os.stat_result:
        metadata = real_path_lstat(path)
        if _normalized_absolute(path) == target_key:
            return ReparseMetadata(metadata)  # type: ignore[return-value]
        return metadata

    class MarkedDirectoryEntry:
        def __init__(self, wrapped: os.DirEntry[str]) -> None:
            self._wrapped = wrapped

        def stat(self, *, follow_symlinks: bool = True) -> os.stat_result:
            metadata = self._wrapped.stat(follow_symlinks=follow_symlinks)
            if _normalized_absolute(self._wrapped.path) == target_key:
                return ReparseMetadata(metadata)  # type: ignore[return-value]
            return metadata

        def __getattr__(self, name: str) -> object:
            return getattr(self._wrapped, name)

    class MarkedScandir:
        def __init__(self, wrapped: os.ScandirIterator[str]) -> None:
            self._wrapped = wrapped

        def __enter__(self) -> "MarkedScandir":
            self._wrapped.__enter__()
            return self

        def __exit__(self, *args: object) -> object:
            return self._wrapped.__exit__(*args)

        def __iter__(self) -> Iterator[MarkedDirectoryEntry]:
            return (MarkedDirectoryEntry(entry) for entry in self._wrapped)

    def marked_scandir(
        path: os.PathLike[str] | str,
    ) -> MarkedScandir:
        return MarkedScandir(real_scandir(path))

    monkeypatch.setattr(Path, "lstat", marked_path_lstat)
    monkeypatch.setattr(module.os, "scandir", marked_scandir)


@pytest.mark.parametrize("operation", ("scan", "check", "write"))
def test_authoring_rejects_reparse_metadata_without_platform_link_privileges(
    authoring_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    eval_root = tmp_path / "eval"
    target = eval_root / "cases" / "core" / "marked.bin"
    target.parent.mkdir(parents=True)
    sentinel = b"reparse-marked target must not change"
    target.write_bytes(sentinel)
    _mark_as_reparse_point(monkeypatch, authoring_module, target)

    _assert_rejected(
        RuntimeError,
        lambda: _invoke(
            authoring_module,
            operation,
            eval_root,
            "cases/core/marked.bin",
            sentinel,
        ),
    )
    assert target.read_bytes() == sentinel
