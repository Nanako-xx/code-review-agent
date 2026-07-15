"""Fail-closed filesystem loading for canonical evaluation Case Suites."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Tuple

from .cases import (
    MAX_SUITE_MANIFEST_BYTES,
    AgentCaseView,
    CaseHandle,
    CaseSplit,
    RunCaseSnapshot,
    SuiteManifest,
    _portable_case_path,
    validate_case_for_manifest,
)
from .models import (
    MAX_EVAL_CASE_BYTES,
    EvalCase,
    EvalInput,
    SchemaError,
    _JsonModel,
    _digest,
    _identifier,
)


DEFAULT_SUITE_MANIFEST_PATH = "suite_manifest.json"
_READ_CHUNK_SIZE = 1024 * 1024
_REPARSE_POINT_FLAG = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x0400)


class DatasetError(SchemaError):
    """A Suite filesystem or manifest loading failure."""


class UnsafeCasePathError(DatasetError):
    """A Case path crossed a symlink, reparse point, or Suite boundary."""


class CaseIntegrityError(DatasetError):
    """Loaded bytes do not match their immutable manifest bindings."""


def _has_reparse_attribute(metadata: os.stat_result) -> bool:
    attributes = getattr(metadata, "st_file_attributes", 0)
    return bool(attributes & _REPARSE_POINT_FLAG)


def _is_link_or_reparse(metadata: os.stat_result) -> bool:
    return stat.S_ISLNK(metadata.st_mode) or _has_reparse_attribute(metadata)


def _coerce_suite_root(value: Any) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise DatasetError("suite root must be a filesystem path")
    lexical = Path(os.path.abspath(os.fspath(value)))
    try:
        metadata = os.lstat(str(lexical))
    except OSError as exc:
        raise DatasetError("suite root does not exist or is not accessible") from exc
    if _is_link_or_reparse(metadata):
        raise UnsafeCasePathError("suite root may not be a symlink or reparse point")
    if not stat.S_ISDIR(metadata.st_mode):
        raise DatasetError("suite root must be a directory")
    try:
        resolved = lexical.resolve(strict=True)
    except OSError as exc:
        raise DatasetError("suite root could not be resolved") from exc
    return resolved


def _assert_root_still_safe(root: Path) -> None:
    try:
        metadata = os.lstat(str(root))
    except OSError as exc:
        raise DatasetError("suite root no longer exists or is not accessible") from exc
    if _is_link_or_reparse(metadata):
        raise UnsafeCasePathError(
            "suite root became a symlink or reparse point"
        )
    if not stat.S_ISDIR(metadata.st_mode):
        raise DatasetError("suite root is no longer a directory")


def _inside_root(root: Path, target: Path) -> bool:
    root_text = os.path.normcase(str(root))
    target_text = os.path.normcase(str(target))
    try:
        common = os.path.commonpath((root_text, target_text))
    except ValueError:
        return False
    return os.path.normcase(common) == root_text


def _secure_regular_file(
    root: Path, relative_path: str, context: str
) -> Tuple[Path, os.stat_result]:
    _assert_root_still_safe(root)
    relative = _portable_case_path(relative_path, "%s path" % context)
    current = root
    components = relative.split("/")
    final_metadata: Optional[os.stat_result] = None
    for index, component in enumerate(components):
        current = current / component
        try:
            metadata = os.lstat(str(current))
        except FileNotFoundError as exc:
            raise DatasetError("%s does not exist: %s" % (context, relative)) from exc
        except OSError as exc:
            raise DatasetError("%s is not accessible: %s" % (context, relative)) from exc
        if _is_link_or_reparse(metadata):
            raise UnsafeCasePathError(
                "%s may not traverse a symlink or reparse point: %s"
                % (context, relative)
            )
        if index < len(components) - 1:
            if not stat.S_ISDIR(metadata.st_mode):
                raise DatasetError(
                    "%s has a non-directory path component: %s"
                    % (context, relative)
                )
        else:
            final_metadata = metadata

    assert final_metadata is not None
    if not stat.S_ISREG(final_metadata.st_mode):
        raise DatasetError("%s must be a regular file: %s" % (context, relative))
    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:
        raise DatasetError("%s could not be resolved: %s" % (context, relative)) from exc
    if not _inside_root(root, resolved):
        raise UnsafeCasePathError(
            "%s escapes the Suite root: %s" % (context, relative)
        )
    return current, final_metadata


def _open_relative_descriptor(
    root: Path,
    relative_path: str,
    path: Path,
) -> int:
    """Open without allowing an intermediate component swap to escape root."""

    binary = getattr(os, "O_BINARY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    if os.name != "nt" and os.open in getattr(os, "supports_dir_fd", set()):
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | nofollow
            | cloexec
        )
        directory_descriptor = os.open(str(root), directory_flags)
        try:
            components = relative_path.split("/")
            for component in components[:-1]:
                next_descriptor = os.open(
                    component,
                    directory_flags,
                    dir_fd=directory_descriptor,
                )
                os.close(directory_descriptor)
                directory_descriptor = next_descriptor
                metadata = os.fstat(directory_descriptor)
                if _is_link_or_reparse(metadata) or not stat.S_ISDIR(
                    metadata.st_mode
                ):
                    raise UnsafeCasePathError(
                        "artifact path contains an unsafe directory component"
                    )
            return os.open(
                components[-1],
                os.O_RDONLY | binary | nofollow | cloexec,
                dir_fd=directory_descriptor,
            )
        finally:
            os.close(directory_descriptor)
    return os.open(str(path), os.O_RDONLY | binary | nofollow | cloexec)


def _windows_descriptor_path(descriptor: int) -> Optional[Path]:
    if os.name != "nt":
        return None
    try:
        import ctypes
        import msvcrt
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_final_path = kernel32.GetFinalPathNameByHandleW
        get_final_path.argtypes = (
            wintypes.HANDLE,
            wintypes.LPWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
        )
        get_final_path.restype = wintypes.DWORD
        handle = wintypes.HANDLE(msvcrt.get_osfhandle(descriptor))
        required = get_final_path(handle, None, 0, 0)
        if not required:
            raise OSError(ctypes.get_last_error(), "GetFinalPathNameByHandleW failed")
        buffer = ctypes.create_unicode_buffer(required + 1)
        written = get_final_path(handle, buffer, len(buffer), 0)
        if not written or written >= len(buffer):
            raise OSError(ctypes.get_last_error(), "GetFinalPathNameByHandleW failed")
        value = buffer.value
    except (ImportError, OSError, ValueError) as exc:
        raise UnsafeCasePathError(
            "could not verify the opened Windows file identity"
        ) from exc
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return Path(value)


def _read_relative_regular_file(
    root: Path,
    relative_path: str,
    maximum_bytes: int,
    context: str,
) -> bytes:
    path, before = _secure_regular_file(root, relative_path, context)
    descriptor: Optional[int] = None
    try:
        descriptor = _open_relative_descriptor(root, relative_path, path)
        opened = os.fstat(descriptor)
        if _is_link_or_reparse(opened) or not stat.S_ISREG(opened.st_mode):
            raise UnsafeCasePathError(
                "%s changed into a link, reparse point, or special file" % context
            )
        if (
            before.st_ino
            and opened.st_ino
            and (before.st_dev, before.st_ino) != (opened.st_dev, opened.st_ino)
        ):
            raise UnsafeCasePathError("%s changed while it was being opened" % context)
        opened_path = _windows_descriptor_path(descriptor)
        if opened_path is not None and not _inside_root(root, opened_path):
            raise UnsafeCasePathError(
                "%s resolved outside the Suite root while open" % context
            )

        chunks = []
        size = 0
        while True:
            chunk = os.read(
                descriptor,
                min(_READ_CHUNK_SIZE, maximum_bytes + 1 - size),
            )
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > maximum_bytes:
                raise DatasetError(
                    "%s exceeds the raw byte limit of %d"
                    % (context, maximum_bytes)
                )
        after_open = os.fstat(descriptor)
        if (
            (opened.st_dev, opened.st_ino) != (after_open.st_dev, after_open.st_ino)
            or not stat.S_ISREG(after_open.st_mode)
        ):
            raise UnsafeCasePathError("%s changed while it was being read" % context)
        _path_after, after_path = _secure_regular_file(root, relative_path, context)
        if (
            opened.st_ino
            and after_path.st_ino
            and (opened.st_dev, opened.st_ino)
            != (after_path.st_dev, after_path.st_ino)
        ):
            raise UnsafeCasePathError(
                "%s path changed while its file was open" % context
            )
        return b"".join(chunks)
    except DatasetError:
        raise
    except OSError as exc:
        raise DatasetError("%s could not be read safely" % context) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def read_suite_manifest(path: Any) -> SuiteManifest:
    """Read a standalone manifest path with the same safe-file rules."""

    if not isinstance(path, (str, os.PathLike)):
        raise DatasetError("manifest path must be a filesystem path")
    lexical = Path(os.path.abspath(os.fspath(path)))
    root = _coerce_suite_root(lexical.parent)
    raw = _read_relative_regular_file(
        root,
        lexical.name,
        MAX_SUITE_MANIFEST_BYTES,
        "Suite manifest",
    )
    return SuiteManifest.from_json(raw)


def _verify_manifest_current(handle: CaseHandle) -> SuiteManifest:
    raw = _read_relative_regular_file(
        handle._suite_root,
        handle._manifest_path,
        MAX_SUITE_MANIFEST_BYTES,
        "Suite manifest",
    )
    try:
        current = SuiteManifest.from_json(raw)
    except SchemaError as exc:
        raise CaseIntegrityError(
            "Suite manifest changed into an invalid document: %s" % exc
        ) from exc
    if current.digest() != handle.manifest.digest() or current != handle.manifest:
        raise CaseIntegrityError(
            "Suite manifest changed after the CaseBank was loaded"
        )
    return current


def _load_case_from_handle(
    handle: CaseHandle, *, verify_manifest: bool = True
) -> EvalCase:
    """Package-private implementation used by immutable CaseHandle methods."""

    if not isinstance(handle, CaseHandle):
        raise DatasetError("Case loader requires a CaseHandle")
    if verify_manifest:
        _verify_manifest_current(handle)

    raw = _read_relative_regular_file(
        handle._suite_root,
        handle.entry.path,
        MAX_EVAL_CASE_BYTES,
        "Case %s" % handle.task_id,
    )
    if len(raw) != handle.entry.raw_file_size_bytes:
        raise CaseIntegrityError(
            "Case %s file size does not match the Suite manifest" % handle.task_id
        )
    file_hash = hashlib.sha256(raw).hexdigest()
    if file_hash != handle.entry.raw_file_sha256:
        raise CaseIntegrityError(
            "Case %s file hash does not match the Suite manifest" % handle.task_id
        )
    try:
        case = EvalCase.from_json(raw)
    except SchemaError as exc:
        raise CaseIntegrityError(
            "Case %s is not valid EvalCase v1: %s" % (handle.task_id, exc)
        ) from exc
    try:
        validate_case_for_manifest(case, handle.entry, handle.manifest)
    except SchemaError as exc:
        raise CaseIntegrityError(str(exc)) from exc
    return case


@dataclass(frozen=True)
class CaseBank(_JsonModel):
    """Immutable Suite inventory with revalidating evaluator/Agent projections."""

    _suite_root: Path
    _manifest_path: str
    manifest: SuiteManifest
    handles: Tuple[CaseHandle, ...]

    def __post_init__(self) -> None:
        if not isinstance(self._suite_root, Path) or not self._suite_root.is_absolute():
            raise DatasetError("CaseBank suite root must be an absolute Path")
        _portable_case_path(self._manifest_path, "CaseBank manifest path")
        if not isinstance(self.manifest, SuiteManifest):
            raise DatasetError("CaseBank.manifest must be a SuiteManifest")
        if type(self.handles) is not tuple:
            raise DatasetError("CaseBank.handles must be an immutable tuple")
        if len(self.handles) != len(self.manifest.cases):
            raise DatasetError("CaseBank handles do not cover the Suite manifest")
        for handle, manifest_case in zip(self.handles, self.manifest.cases):
            if not isinstance(handle, CaseHandle):
                raise DatasetError("CaseBank.handles must contain CaseHandle values")
            if (
                handle._suite_root != self._suite_root
                or handle._manifest_path != self._manifest_path
                or handle.manifest != self.manifest
                or handle.entry != manifest_case
            ):
                raise DatasetError("CaseBank contains an inconsistent CaseHandle")

    @classmethod
    def open(
        cls,
        suite_root: Any,
        manifest_path: str = DEFAULT_SUITE_MANIFEST_PATH,
        *,
        expected_manifest_digest: Optional[str] = None,
    ) -> "CaseBank":
        root = _coerce_suite_root(suite_root)
        relative_manifest = _portable_case_path(
            manifest_path, "CaseBank manifest path"
        )
        raw = _read_relative_regular_file(
            root,
            relative_manifest,
            MAX_SUITE_MANIFEST_BYTES,
            "Suite manifest",
        )
        manifest = SuiteManifest.from_json(raw)
        if expected_manifest_digest is not None:
            expected = _digest(
                expected_manifest_digest, "expected Suite manifest digest"
            )
            if manifest.digest() != expected:
                raise CaseIntegrityError(
                    "Suite manifest digest does not match the expected digest"
                )

        handles = tuple(
            CaseHandle(root, relative_manifest, manifest, entry)
            for entry in manifest.cases
        )
        bank = cls(root, relative_manifest, manifest, handles)
        # Opening a Suite is an audit, not a lazy promise: missing, malformed,
        # or tampered Cases fail now.  Individual use revalidates them again.
        bank.verify()
        return bank

    @property
    def root(self) -> Path:
        return self._suite_root

    @property
    def suite_id(self) -> str:
        return self.manifest.suite_id

    @property
    def suite_version(self) -> str:
        return self.manifest.suite_version

    @property
    def manifest_digest(self) -> str:
        return self.manifest.digest()

    def __len__(self) -> int:
        return len(self.handles)

    def __iter__(self) -> Iterator[CaseHandle]:
        return iter(self.handles)

    def handle(self, task_id: str) -> CaseHandle:
        wanted = _identifier(task_id, "task_id")
        try:
            index = self.manifest.case_index(wanted)
        except SchemaError as exc:
            raise DatasetError("CaseBank has no task_id %r" % wanted) from exc
        return self.handles[index]

    def handles_for_split(self, split: CaseSplit) -> Tuple[CaseHandle, ...]:
        if not isinstance(split, CaseSplit):
            raise DatasetError("split must be a CaseSplit")
        return tuple(handle for handle in self.handles if handle.split is split)

    def verify(self) -> None:
        if self.handles:
            _verify_manifest_current(self.handles[0])
        for handle in self.handles:
            _load_case_from_handle(handle, verify_manifest=False)
        if self.handles:
            _verify_manifest_current(self.handles[0])

    def evaluator_case(self, task_id: str) -> EvalCase:
        return _load_case_from_handle(self.handle(task_id))

    def runner_case(self, task_id: str) -> EvalCase:
        return _load_case_from_handle(self.handle(task_id))

    def agent_input(self, task_id: str) -> EvalInput:
        # This is the only payload an Agent adapter should receive.
        return _load_case_from_handle(self.handle(task_id)).eval_input()

    def agent_view(self, task_id: str) -> AgentCaseView:
        return AgentCaseView(input=self.agent_input(task_id))

    def snapshot(
        self,
        task_ids: Optional[Tuple[str, ...]] = None,
        *,
        split: Optional[CaseSplit] = None,
    ) -> RunCaseSnapshot:
        if split is not None and not isinstance(split, CaseSplit):
            raise DatasetError("split must be a CaseSplit or null")
        if task_ids is None:
            selected = self.handles
            if split is not None:
                selected = tuple(
                    handle for handle in selected if handle.split is split
                )
        else:
            if type(task_ids) not in (tuple, list):
                raise DatasetError("task_ids must be an immutable tuple or list")
            normalized = tuple(
                _identifier(item, "task_ids[%d]" % index)
                for index, item in enumerate(task_ids)
            )
            if len(set(normalized)) != len(normalized):
                raise DatasetError("task_ids contains a duplicate task ID")
            selected = tuple(self.handle(task_id) for task_id in normalized)
            if split is not None:
                for handle in selected:
                    if handle.split is not split:
                        raise DatasetError(
                            "task_id %r does not belong to split %s"
                            % (handle.task_id, split.value)
                        )
        if not selected:
            raise DatasetError("run Case snapshot selection may not be empty")

        _verify_manifest_current(selected[0])
        loaded = tuple(
            (
                handle.entry,
                _load_case_from_handle(handle, verify_manifest=False),
            )
            for handle in selected
        )
        _verify_manifest_current(selected[0])
        return RunCaseSnapshot.build(self.manifest, loaded)

    def to_dict(self) -> Dict[str, Any]:
        # The bank serializes only immutable inventory metadata, never root
        # paths or private EvalCase payloads.
        return {
            "manifest": self.manifest.to_dict(),
            "manifest_digest": self.manifest.digest(),
        }


__all__ = [
    "DEFAULT_SUITE_MANIFEST_PATH",
    "DatasetError",
    "UnsafeCasePathError",
    "CaseIntegrityError",
    "CaseBank",
    "read_suite_manifest",
]
