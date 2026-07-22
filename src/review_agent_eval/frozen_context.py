"""Exact-byte Frozen Context materialization and immutable replay."""

from __future__ import annotations

import hashlib
import os
import stat
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Optional, Tuple

from .adapters._public import PublicDatasetError
from .adapters.swe_prbench import (
    FrozenContextBinding,
    FrozenContextEnvelope,
    PreparedFrozenContextBundle,
    read_swe_prbench_frozen_bundle,
)
from .artifacts import AgentVisibleFileBinding, TrialMaterializationManifest
from .cases import PublicSuitePreparationBindingV2
from .materialization import (
    MaterializationError,
    MaterializationRequest,
    PreparedTargetMaterialization,
    _validate_materialization_request,
)
from .models import (
    FrozenContextReviewTarget,
    ReviewTargetKind,
    TrialStatus,
    canonical_sha256,
    stable_id,
)
from .repository import _remove_tree_safely


FROZEN_CONTEXT_TARGET_PATH = "target/context.txt"
FROZEN_CONTEXT_WORK_PATH = "work"
FROZEN_CONTEXT_SOURCE_BINDING_VERSION = "frozen-context-source-binding-v2"
FROZEN_CONTEXT_REPLAY_BINDING_VERSION = "frozen-context-replay-binding-v2"
FROZEN_BUNDLE_TRUST_BINDING_VERSION = "frozen-bundle-trust-binding-v2"

_REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)


def _is_link_or_reparse(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        int(getattr(info, "st_file_attributes", 0) or 0) & _REPARSE_POINT
    )


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    left_inode = int(getattr(left, "st_ino", 0) or 0)
    right_inode = int(getattr(right, "st_ino", 0) or 0)
    if not left_inode or not right_inode:
        return False
    return (
        int(getattr(left, "st_dev", 0) or 0),
        left_inode,
    ) == (
        int(getattr(right, "st_dev", 0) or 0),
        right_inode,
    )


def _secure_directory(path: Path, *, create: bool, context: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.parts[0])
    for component in absolute.parts[1:]:
        current = current / component
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            if not create:
                raise MaterializationError("%s does not exist" % context)
            try:
                os.mkdir(current, 0o700)
            except FileExistsError:
                pass
            except OSError as exc:
                raise MaterializationError(
                    "could not create %s" % context
                ) from exc
            info = os.lstat(current)
        except OSError as exc:
            raise MaterializationError("could not inspect %s" % context) from exc
        if _is_link_or_reparse(info) or not stat.S_ISDIR(info.st_mode):
            raise MaterializationError("%s is not a secure directory" % context)
    return absolute


def _portable_relative_path(value: str) -> Tuple[str, ...]:
    if type(value) is not str or not value or "\\" in value:
        raise MaterializationError("Frozen bundle path is not canonical")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise MaterializationError("Frozen bundle path contains traversal")
    return tuple(path.parts)


def _secure_read(
    root: Path,
    relative_path: str,
    *,
    expected_size: int,
    expected_sha256: str,
) -> bytes:
    root = _secure_directory(root, create=False, context="Frozen bundle root")
    current = root
    parts = _portable_relative_path(relative_path)
    for index, component in enumerate(parts):
        current = current / component
        try:
            info = os.lstat(current)
        except OSError as exc:
            raise MaterializationError("Frozen bundle record is unavailable") from exc
        if _is_link_or_reparse(info):
            raise MaterializationError("Frozen bundle record traverses a link")
        if index < len(parts) - 1 and not stat.S_ISDIR(info.st_mode):
            raise MaterializationError("Frozen bundle record parent is invalid")
    before = os.lstat(current)
    if not stat.S_ISREG(before.st_mode) or before.st_size != expected_size:
        raise MaterializationError("Frozen bundle record size drifted")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(current, flags)
    except OSError as exc:
        raise MaterializationError("could not open Frozen bundle record") from exc
    try:
        opened = os.fstat(descriptor)
        if _is_link_or_reparse(opened) or not stat.S_ISREG(opened.st_mode):
            raise MaterializationError("Frozen bundle record changed type")
        if getattr(before, "st_ino", 0) and not _same_identity(before, opened):
            raise MaterializationError("Frozen bundle record identity drifted")
        data = bytearray()
        while len(data) <= expected_size:
            chunk = os.read(descriptor, min(1024 * 1024, expected_size + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
        after = os.fstat(descriptor)
        if len(data) != expected_size or opened.st_size != after.st_size:
            raise MaterializationError("Frozen bundle record changed while read")
    finally:
        os.close(descriptor)
    raw = bytes(data)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise MaterializationError("Frozen bundle record hash drifted")
    return raw


def _relax_owned_workspace_permissions(path: Path) -> None:
    """Make fixed Frozen workspace nodes removable without walking links."""

    try:
        root_info = os.lstat(path)
    except OSError:
        return
    if _is_link_or_reparse(root_info) or not stat.S_ISDIR(root_info.st_mode):
        return
    for known in (
        path / FROZEN_CONTEXT_TARGET_PATH,
        path / "target",
        path / FROZEN_CONTEXT_WORK_PATH,
        path,
    ):
        try:
            known_info = os.lstat(known)
            if not _is_link_or_reparse(known_info):
                os.chmod(known, 0o700)
        except OSError:
            pass


def frozen_context_record_id(binding: FrozenContextBinding) -> str:
    if not isinstance(binding, FrozenContextBinding):
        raise TypeError("binding must be FrozenContextBinding")
    return binding.task_id


def frozen_bundle_trust_digest(
    bundle: PreparedFrozenContextBundle,
    preparation: PublicSuitePreparationBindingV2,
) -> str:
    """Close external acquisition lineage over one exact Frozen bundle."""

    if not isinstance(bundle, PreparedFrozenContextBundle):
        raise TypeError("bundle must be PreparedFrozenContextBundle")
    if not isinstance(preparation, PublicSuitePreparationBindingV2):
        raise TypeError(
            "preparation must be PublicSuitePreparationBindingV2"
        )
    if preparation.repository_catalog_digest is not None:
        raise MaterializationError(
            "Frozen trust cannot contain a Repository catalog binding"
        )
    return canonical_sha256(
        {
            "schema_version": FROZEN_BUNDLE_TRUST_BINDING_VERSION,
            "bundle_id": bundle.manifest.bundle_id,
            "bundle_manifest_digest": bundle.manifest.digest(),
            "source_catalog_digest": preparation.source_catalog_digest,
            "acquisition_receipt_digest": (
                preparation.acquisition_receipt_digest
            ),
            "source_manifest_digest": preparation.source_manifest_digest,
            "filter_manifest_digest": preparation.filter_manifest_digest,
            "preparation_packet_digest": (
                preparation.preparation_packet_digest
            ),
        }
    )


def frozen_context_source_binding_digest(
    bundle: PreparedFrozenContextBundle,
    binding: FrozenContextBinding,
) -> str:
    if not isinstance(bundle, PreparedFrozenContextBundle):
        raise TypeError("bundle must be PreparedFrozenContextBundle")
    if not isinstance(binding, FrozenContextBinding):
        raise TypeError("binding must be FrozenContextBinding")
    if binding not in bundle.manifest.records:
        raise MaterializationError("Frozen binding is not part of the bundle")
    return canonical_sha256(
        {
            "schema_version": FROZEN_CONTEXT_SOURCE_BINDING_VERSION,
            "bundle_id": bundle.manifest.bundle_id,
            "bundle_manifest_digest": bundle.manifest.digest(),
            "source_manifest_digest": bundle.manifest.source_manifest_digest,
            "filter_manifest_digest": bundle.manifest.filter_manifest_digest,
            "record": {
                "record_id": frozen_context_record_id(binding),
                "task_id": binding.task_id,
                "config_name": binding.config_name,
                "path": binding.path,
                "record_digest": binding.record_digest,
                "rendered_sha256": binding.rendered_sha256,
                "rendered_utf8_bytes": binding.rendered_utf8_bytes,
                "source_role": binding.source_role,
                "source_file_sha256": binding.source_file_sha256,
                "pr_record_sha256": binding.pr_record_sha256,
                "annotation_record_sha256": binding.annotation_record_sha256,
                "review_truth_status": binding.review_truth_status,
                "review_truth_reason": binding.review_truth_reason,
                "offending_record_sha256": binding.offending_record_sha256,
            },
        }
    )


@dataclass(frozen=True)
class FrozenContextReplay:
    bundle_id: str
    record_id: str
    context_ref: str
    context_format: str
    rendered_sha256: str
    rendered_utf8_bytes: int
    source_binding_digest: str
    replay_binding_digest: str
    _bundle_root: Path = field(repr=False, compare=False)
    _binding: FrozenContextBinding = field(repr=False, compare=False)

    def _record(self):
        raw = _secure_read(
            self._bundle_root,
            self._binding.path,
            expected_size=self._binding.size_bytes,
            expected_sha256=self._binding.sha256,
        )
        try:
            envelope = FrozenContextEnvelope.from_json(raw)
        except Exception as exc:
            raise MaterializationError("Frozen bundle record is malformed") from exc
        if envelope.bundle_id != self.bundle_id:
            raise MaterializationError("Frozen envelope bundle identity drifted")
        record = envelope.record
        if (
            record.task_id != self.record_id
            or record.digest() != self._binding.record_digest
            or record.rendered_sha256 != self.rendered_sha256
            or record.rendered_utf8_bytes != self.rendered_utf8_bytes
        ):
            raise MaterializationError("Frozen record binding drifted")
        return record

    def read_exact(self) -> bytes:
        data = self._record().rendered.encode("utf-8", "strict")
        if (
            len(data) != self.rendered_utf8_bytes
            or hashlib.sha256(data).hexdigest() != self.rendered_sha256
        ):
            raise MaterializationError("Frozen rendered bytes drifted")
        return data

    def read_lines(self, from_line: int, to_line: int) -> bytes:
        if type(from_line) is not int or type(to_line) is not int:
            raise TypeError("Frozen line range must use integers")
        if from_line < 1 or to_line < from_line:
            raise ValueError("Frozen line range is invalid")
        data = self.read_exact()
        starts = [0]
        for index, value in enumerate(data):
            if value == 0x0A and index + 1 < len(data):
                starts.append(index + 1)
        if from_line > len(starts) or to_line > len(starts):
            raise ValueError("Frozen line range exceeds rendered content")
        start = starts[from_line - 1]
        end = starts[to_line] if to_line < len(starts) else len(data)
        return data[start:end]


@dataclass
class _FrozenWorkspace:
    path: Path
    active_root: Path
    expected_content: bytes
    _identity: Tuple[int, int]
    _closed: bool = False

    @property
    def closed(self) -> bool:
        return self._closed

    def validate(self) -> None:
        if self._closed:
            raise MaterializationError("Frozen workspace lease is closed")
        active_root = _secure_directory(
            self.active_root,
            create=False,
            context="Frozen active workspace root",
        )
        workspace_path = Path(os.path.abspath(os.fspath(self.path)))
        if workspace_path.parent != active_root or workspace_path != self.path:
            raise MaterializationError(
                "Frozen workspace is not a direct child of the active root"
            )
        try:
            info = os.lstat(workspace_path)
        except OSError as exc:
            raise MaterializationError("Frozen workspace is unavailable") from exc
        identity = (
            int(getattr(info, "st_dev", 0) or 0),
            int(getattr(info, "st_ino", 0) or 0),
        )
        if (
            _is_link_or_reparse(info)
            or not stat.S_ISDIR(info.st_mode)
            or identity != self._identity
        ):
            raise MaterializationError("Frozen workspace identity drifted")
        target_root = workspace_path / "target"
        try:
            target_root_info = os.lstat(target_root)
        except OSError as exc:
            raise MaterializationError(
                "Frozen Agent-visible Target root is unavailable"
            ) from exc
        if _is_link_or_reparse(target_root_info) or not stat.S_ISDIR(
            target_root_info.st_mode
        ):
            raise MaterializationError("Frozen Target root drifted")
        try:
            content = _secure_read(
                workspace_path,
                FROZEN_CONTEXT_TARGET_PATH,
                expected_size=len(self.expected_content),
                expected_sha256=hashlib.sha256(self.expected_content).hexdigest(),
            )
        except MaterializationError as exc:
            raise MaterializationError(
                "Frozen Agent-visible Target drifted"
            ) from exc
        if content != self.expected_content:
            raise MaterializationError("Frozen Agent-visible Target drifted")
        target_entries = sorted(
            item.name for item in target_root.iterdir()
        )
        if target_entries != ["context.txt"]:
            raise MaterializationError("Frozen Target contains unexpected files")
        work = workspace_path / FROZEN_CONTEXT_WORK_PATH
        try:
            work_info = os.lstat(work)
        except OSError as exc:
            raise MaterializationError("Frozen work directory is unavailable") from exc
        if _is_link_or_reparse(work_info) or not stat.S_ISDIR(work_info.st_mode):
            raise MaterializationError("Frozen work directory drifted")

    def close(self) -> None:
        if self._closed:
            return
        try:
            info = os.lstat(self.path)
        except FileNotFoundError:
            self._closed = True
            return
        except OSError as exc:
            raise MaterializationError(
                "Frozen workspace could not be inspected for cleanup"
            ) from exc
        identity = (
            int(getattr(info, "st_dev", 0) or 0),
            int(getattr(info, "st_ino", 0) or 0),
        )
        if (
            _is_link_or_reparse(info)
            or not stat.S_ISDIR(info.st_mode)
            or identity != self._identity
        ):
            raise MaterializationError(
                "Frozen workspace identity drifted before cleanup"
            )
        _relax_owned_workspace_permissions(self.path)
        _remove_tree_safely(self.active_root, self.path)
        try:
            os.lstat(self.path)
        except FileNotFoundError:
            self._closed = True
            return
        except OSError as exc:
            raise MaterializationError(
                "Frozen workspace cleanup could not be verified"
            ) from exc
        raise MaterializationError("Frozen workspace still exists after cleanup")


def _publish_workspace(
    workspace_root: Path,
    workspace_id: str,
    content: bytes,
) -> _FrozenWorkspace:
    root = _secure_directory(
        workspace_root,
        create=True,
        context="Frozen workspace root",
    )
    active = _secure_directory(
        root / "active",
        create=True,
        context="Frozen active workspace root",
    )
    target = active / workspace_id
    if os.path.lexists(target):
        raise MaterializationError("Frozen Trial workspace already exists")
    staging = active / (".%s.%s.staging" % (workspace_id, uuid.uuid4().hex))
    try:
        os.mkdir(staging, 0o700)
        os.mkdir(staging / "target", 0o700)
        os.mkdir(staging / FROZEN_CONTEXT_WORK_PATH, 0o700)
        context_path = staging / FROZEN_CONTEXT_TARGET_PATH
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_BINARY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        descriptor = os.open(context_path, flags, 0o400)
        try:
            view = memoryview(content)
            written = 0
            while written < len(view):
                amount = os.write(descriptor, view[written:])
                if amount <= 0:
                    raise MaterializationError("Frozen Target write made no progress")
                written += amount
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(context_path, 0o400)
        os.chmod(staging / "target", 0o500)
        os.rename(staging, target)
    except BaseException:
        if os.path.lexists(staging):
            try:
                _relax_owned_workspace_permissions(staging)
                _remove_tree_safely(active, staging)
            except Exception:
                pass
        raise
    workspace: Optional[_FrozenWorkspace] = None
    try:
        info = os.lstat(target)
        workspace = _FrozenWorkspace(
            path=target,
            active_root=active,
            expected_content=content,
            _identity=(
                int(getattr(info, "st_dev", 0) or 0),
                int(getattr(info, "st_ino", 0) or 0),
            ),
        )
        workspace.validate()
        return workspace
    except BaseException:
        if workspace is not None:
            try:
                workspace.close()
            except Exception:
                pass
        elif os.path.lexists(target):
            try:
                _relax_owned_workspace_permissions(target)
                _remove_tree_safely(active, target)
            except Exception:
                pass
        raise


@dataclass
class _FrozenMaterializationLease:
    request: MaterializationRequest
    manifest: TrialMaterializationManifest
    replay: FrozenContextReplay
    workspace: _FrozenWorkspace

    @property
    def closed(self) -> bool:
        return self.workspace.closed

    @property
    def work_root(self) -> Path:
        return self.workspace.path

    def validate(self) -> None:
        _validate_materialization_request(
            self.request,
            ReviewTargetKind.FROZEN_CONTEXT,
        )
        target = self.request.eval_input.review_target
        if not isinstance(target, FrozenContextReviewTarget):
            raise MaterializationError("Frozen Target has the wrong tagged shape")
        if (
            target.bundle_id != self.replay.bundle_id
            or target.record_id != self.replay.record_id
            or target.context_format != self.replay.context_format
            or target.rendered_sha256 != self.replay.rendered_sha256
            or target.rendered_utf8_bytes != self.replay.rendered_utf8_bytes
            or target.source_binding_digest != self.replay.source_binding_digest
        ):
            raise MaterializationError("Frozen Target binding drifted")
        if self.manifest.replay_binding_digest != self.replay.replay_binding_digest:
            raise MaterializationError("Frozen replay binding drifted")
        content = self.replay.read_exact()
        if hashlib.sha256(content).hexdigest() != self.manifest.files[0].sha256:
            raise MaterializationError("Frozen materialization content drifted")
        self.workspace.validate()

    def close(self, status: TrialStatus) -> None:
        del status
        self.workspace.close()


class FrozenContextTargetMaterializer:
    """Materialize one verified local SWE-PRBench Frozen Context bundle."""

    def __init__(self, *, bundle_root: Path, workspace_root: Path) -> None:
        self.bundle_root = _secure_directory(
            Path(bundle_root),
            create=False,
            context="Frozen bundle root",
        )
        self.workspace_root = _secure_directory(
            Path(workspace_root),
            create=True,
            context="Frozen workspace root",
        )

    def materialize(
        self,
        request: MaterializationRequest,
    ) -> PreparedTargetMaterialization[FrozenContextReplay]:
        _validate_materialization_request(
            request,
            ReviewTargetKind.FROZEN_CONTEXT,
        )
        target = request.eval_input.review_target
        if not isinstance(target, FrozenContextReviewTarget):
            raise MaterializationError("Frozen Target has the wrong tagged shape")
        try:
            bundle = read_swe_prbench_frozen_bundle(
                self.bundle_root,
                expected_bundle_id=target.bundle_id,
            )
        except (PublicDatasetError, OSError) as exc:
            raise MaterializationError("Frozen bundle trust validation failed") from exc
        preparation = request.suite_preparation_binding
        if (
            preparation is None
            or preparation.repository_catalog_digest is not None
            or preparation.frozen_bundle_trust_digest is None
        ):
            raise MaterializationError(
                "Frozen Suite preparation trust is unavailable"
            )
        if (
            preparation.frozen_bundle_trust_digest
            != frozen_bundle_trust_digest(bundle, preparation)
            or preparation.source_manifest_digest
            != bundle.manifest.source_manifest_digest
            or preparation.filter_manifest_digest
            != bundle.manifest.filter_manifest_digest
        ):
            raise MaterializationError(
                "Frozen bundle does not match external Suite trust"
            )
        matches = [
            item
            for item in bundle.manifest.records
            if frozen_context_record_id(item) == target.record_id
        ]
        if len(matches) != 1:
            raise MaterializationError("Frozen record identity is not unique")
        binding = matches[0]
        if binding.task_id != request.eval_input.task_id:
            raise MaterializationError("Frozen record task does not match EvalInput")
        source_binding = frozen_context_source_binding_digest(bundle, binding)
        if source_binding != target.source_binding_digest:
            raise MaterializationError("Frozen source binding drifted")
        replay_digest = canonical_sha256(
            {
                "schema_version": FROZEN_CONTEXT_REPLAY_BINDING_VERSION,
                "bundle_id": target.bundle_id,
                "record_id": target.record_id,
                "context_format": target.context_format,
                "rendered_sha256": target.rendered_sha256,
                "rendered_utf8_bytes": target.rendered_utf8_bytes,
                "source_binding_digest": target.source_binding_digest,
                "bundle_manifest_digest": bundle.manifest.digest(),
                "record_digest": binding.record_digest,
                "suite_preparation_binding_digest": (
                    request.suite_preparation_binding_digest
                ),
                "preparation_packet_digest": (
                    preparation.preparation_packet_digest
                ),
                "frozen_bundle_trust_digest": (
                    preparation.frozen_bundle_trust_digest
                ),
            }
        )
        # ``record_id`` is already present in the Agent-visible EvalInput.
        # Materialization identity supplies the bundle/attempt namespace, so
        # using it as ``context_ref`` avoids a hidden identifier that a
        # subprocess Agent could never reproduce in Frozen Evidence.
        context_ref = target.record_id
        replay = FrozenContextReplay(
            bundle_id=target.bundle_id,
            record_id=target.record_id,
            context_ref=context_ref,
            context_format=target.context_format,
            rendered_sha256=target.rendered_sha256,
            rendered_utf8_bytes=target.rendered_utf8_bytes,
            source_binding_digest=target.source_binding_digest,
            replay_binding_digest=replay_digest,
            _bundle_root=bundle.root,
            _binding=binding,
        )
        content = replay.read_exact()
        workspace_id = stable_id(
            "frozen-workspace",
            request.trial_manifest.trial_id,
            request.attempt,
            request.eval_input.digest(),
            target.digest(),
        )
        workspace: Optional[_FrozenWorkspace] = None
        try:
            workspace = _publish_workspace(
                self.workspace_root,
                workspace_id,
                content,
            )
            files = (
                AgentVisibleFileBinding(
                    role="frozen_context",
                    relative_path=FROZEN_CONTEXT_TARGET_PATH,
                    size_bytes=len(content),
                    sha256=hashlib.sha256(content).hexdigest(),
                ),
            )
            manifest = TrialMaterializationManifest.create(
                run_id=request.trial_manifest.run_id,
                task_id=request.eval_input.task_id,
                trial_id=request.trial_manifest.trial_id,
                attempt=request.attempt,
                eval_input_digest=request.eval_input.digest(),
                review_target_digest=target.digest(),
                wire_contract=request.wire_contract,
                suite_preparation_binding_digest=(
                    request.suite_preparation_binding_digest
                ),
                prepared_source_id=target.bundle_id,
                adapter_capabilities_digest=request.adapter_capabilities.digest(),
                readable_relative_paths=(FROZEN_CONTEXT_TARGET_PATH,),
                files=files,
                replay_binding_digest=replay_digest,
            )
            lease = _FrozenMaterializationLease(
                request=request,
                manifest=manifest,
                replay=replay,
                workspace=workspace,
            )
            materialized = PreparedTargetMaterialization(
                request=request,
                manifest=manifest,
                replay=replay,
                _lease=lease,
            )
            materialized.validate()
            return materialized
        except BaseException:
            if workspace is not None:
                workspace.close()
            raise


def frozen_materialization_workspace(
    materialized: PreparedTargetMaterialization[FrozenContextReplay],
) -> Path:
    lease = materialized._lease
    path = getattr(lease, "work_root", None)
    if not isinstance(path, Path):
        raise TypeError("materialization does not contain a Frozen workspace")
    return path


__all__ = [
    "FROZEN_BUNDLE_TRUST_BINDING_VERSION",
    "FROZEN_CONTEXT_REPLAY_BINDING_VERSION",
    "FROZEN_CONTEXT_SOURCE_BINDING_VERSION",
    "FROZEN_CONTEXT_TARGET_PATH",
    "FROZEN_CONTEXT_WORK_PATH",
    "FrozenContextReplay",
    "FrozenContextTargetMaterializer",
    "frozen_bundle_trust_digest",
    "frozen_context_record_id",
    "frozen_context_source_binding_digest",
    "frozen_materialization_workspace",
]
