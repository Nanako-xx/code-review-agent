"""Target materialization behavior for the v2 evaluation protocol.

The immutable materialization DTOs live in :mod:`artifacts`.  This module
owns the runtime behavior that creates and revalidates those DTOs.  Keeping
the behavior separate lets the Repository implementation remain the single
owner of Git/cache safety while the Runner can later add a Frozen Context
materializer without creating a second Repository path.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Optional, Protocol, Tuple, TypeVar

from .artifacts import (
    AgentVisibleFileBinding,
    TrialManifest,
    TrialMaterializationManifest,
    TargetAccess,
)
from .cases import (
    PublicSuitePreparationBindingV2,
    SuiteCase,
    WireContractV2,
)
from .config import AdapterCapabilitiesV2
from .models import (
    canonical_sha256,
    EvalInput,
    RepositoryReviewTarget,
    ReviewTargetKind,
    SchemaError,
    TrialStatus,
)
from .repository import (
    PreparedRepository,
    PreparedRepositoryReplay,
    RepositoryMode,
    RepositoryPreparer,
    RepositoryPreparationError,
    TrialWorkspace,
    repository_from_eval_input,
)


class MaterializationError(RepositoryPreparationError):
    """A Target cannot be materialized or no longer matches its lease."""


ReplayT = TypeVar("ReplayT")


class _MaterializationLease(Protocol):
    @property
    def closed(self) -> bool:
        ...

    @property
    def work_root(self) -> Path:
        ...

    def validate(self) -> None:
        ...

    def close(self, status: TrialStatus) -> None:
        ...


@dataclass(frozen=True)
class MaterializationRequest:
    """Immutable inputs required to materialize one Trial attempt."""

    eval_input: EvalInput
    trial_manifest: TrialManifest
    suite_case: SuiteCase
    attempt: int
    wire_contract: WireContractV2
    suite_preparation_binding: Optional[PublicSuitePreparationBindingV2]
    suite_preparation_binding_digest: Optional[str]
    adapter_capabilities: AdapterCapabilitiesV2

    def __post_init__(self) -> None:
        if not isinstance(self.eval_input, EvalInput):
            raise TypeError("materialization request requires EvalInput")
        if not isinstance(self.trial_manifest, TrialManifest):
            raise TypeError("materialization request requires TrialManifest")
        if not isinstance(self.suite_case, SuiteCase):
            raise TypeError("materialization request requires SuiteCase")
        if type(self.attempt) is not int or self.attempt < 1:
            raise SchemaError("materialization request attempt must be positive")
        if not isinstance(self.wire_contract, WireContractV2):
            raise TypeError("materialization request requires WireContractV2")
        if self.suite_preparation_binding is not None and not isinstance(
            self.suite_preparation_binding,
            PublicSuitePreparationBindingV2,
        ):
            raise TypeError(
                "materialization request preparation binding is invalid"
            )
        if self.suite_preparation_binding_digest is not None:
            if (
                type(self.suite_preparation_binding_digest) is not str
                or len(self.suite_preparation_binding_digest) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in self.suite_preparation_binding_digest
                )
            ):
                raise SchemaError(
                    "materialization request preparation digest is invalid"
                )
        expected_preparation_digest = (
            None
            if self.suite_preparation_binding is None
            else self.suite_preparation_binding.digest()
        )
        if self.suite_preparation_binding_digest != expected_preparation_digest:
            raise SchemaError(
                "materialization request preparation binding digest drifted"
            )
        if not isinstance(self.adapter_capabilities, AdapterCapabilitiesV2):
            raise TypeError(
                "materialization request requires AdapterCapabilitiesV2"
            )


@dataclass(frozen=True)
class PreparedTargetMaterialization(Generic[ReplayT]):
    """Target-neutral runtime lease joining Agent access and replay."""

    request: MaterializationRequest
    manifest: TrialMaterializationManifest
    replay: ReplayT
    _lease: _MaterializationLease = field(repr=False, compare=False)

    @property
    def target_access(self) -> TargetAccess:
        return self.manifest.target_access

    @property
    def materialization_id(self) -> str:
        return self.manifest.materialization_id

    @property
    def closed(self) -> bool:
        return self._lease.closed

    @property
    def work_root(self) -> Path:
        return self._lease.work_root

    def validate(self) -> None:
        """Revalidate the immutable Target before handing it to an Agent."""

        self._lease.validate()

    def close(self, status: TrialStatus = TrialStatus.COMPLETED) -> None:
        if not isinstance(status, TrialStatus):
            raise TypeError("materialization close status must be TrialStatus")
        self._lease.close(status)

    def __enter__(self) -> "PreparedTargetMaterialization[ReplayT]":
        try:
            self.validate()
        except BaseException:
            try:
                self.close(TrialStatus.FAILED)
            except BaseException:
                pass
            raise
        return self

    def __exit__(self, exc_type: object, _value: object, _traceback: object) -> bool:
        if exc_type is None:
            self.close(TrialStatus.COMPLETED)
        else:
            try:
                self.close(TrialStatus.FAILED)
            except BaseException:
                pass
        return False


class TargetMaterializer(Protocol):
    def materialize(
        self, request: MaterializationRequest
    ) -> PreparedTargetMaterialization[Any]:
        ...


@dataclass
class _RepositoryMaterializationLease:
    request: MaterializationRequest
    manifest: TrialMaterializationManifest
    prepared_repository: PreparedRepository
    replay: PreparedRepositoryReplay
    workspace: TrialWorkspace
    preparer: RepositoryPreparer

    @property
    def closed(self) -> bool:
        return self.workspace.closed

    @property
    def work_root(self) -> Path:
        return self.workspace.path

    def validate(self) -> None:
        _validate_repository_request(self.request)
        try:
            self.workspace.validate()
        except RepositoryPreparationError as exc:
            raise MaterializationError(
                "materialization workspace lease is invalid"
            ) from exc
        if self.workspace.manifest.attempt != self.request.attempt:
            raise MaterializationError(
                "workspace attempt does not match materialization"
            )
        if self.workspace.manifest.trial_manifest.trial_id != (
            self.request.trial_manifest.trial_id
        ):
            raise MaterializationError("workspace Trial identity drifted")
        repository = repository_from_eval_input(self.request.eval_input)
        if self.prepared_repository.repository != repository:
            raise MaterializationError(
                "prepared Repository does not match EvalInput Target"
            )
        try:
            current_replay = self.preparer.open_replay(self.prepared_repository)
            current_files = _repository_file_bindings(
                self.workspace,
                current_replay,
                repository.head_revision,
            )
        except MaterializationError:
            raise
        except RepositoryPreparationError as exc:
            raise MaterializationError(
                "Repository replay validation failed"
            ) from exc
        if _replay_identity(current_replay) != _replay_identity(self.replay):
            raise MaterializationError("Repository replay identity drifted")
        if current_files != self.manifest.files:
            raise MaterializationError(
                "Agent-visible Repository Target drifted after materialization"
            )
        expected_replay_digest = repository_replay_binding_digest(
            self.prepared_repository,
            current_replay,
        )
        if expected_replay_digest != self.manifest.replay_binding_digest:
            raise MaterializationError("materialization replay binding drifted")

    def close(self, status: TrialStatus) -> None:
        if not self.workspace.closed:
            self.workspace.record_terminal_status(status)
            self.workspace.close()


def _validate_repository_request(request: MaterializationRequest) -> None:
    if request.eval_input.review_target.kind is not ReviewTargetKind.REPOSITORY:
        raise MaterializationError(
            "RepositoryTargetMaterializer received a non-Repository Target"
        )
    _validate_materialization_request(request, ReviewTargetKind.REPOSITORY)
    if not isinstance(request.eval_input.review_target, RepositoryReviewTarget):
        raise MaterializationError("Repository Target has the wrong tagged shape")


def _validate_materialization_request(
    request: MaterializationRequest,
    expected_kind: ReviewTargetKind,
) -> None:
    if not isinstance(request, MaterializationRequest):
        raise TypeError("materialization validation requires MaterializationRequest")
    if not isinstance(expected_kind, ReviewTargetKind):
        raise TypeError("expected materialization kind must be ReviewTargetKind")
    if request.eval_input.review_target.kind is not expected_kind:
        raise MaterializationError(
            "Target materializer received the wrong review Target kind"
        )
    if request.wire_contract.review_target_kind is not expected_kind:
        raise MaterializationError("wire contract Target kind drifted")
    if request.trial_manifest.wire_contract != request.wire_contract:
        raise MaterializationError("Trial wire contract does not match request")
    if request.trial_manifest.target_kind is not expected_kind:
        raise MaterializationError("Trial Target kind drifted")
    if request.trial_manifest.task_id != request.eval_input.task_id:
        raise MaterializationError("Trial task does not match EvalInput")
    if request.trial_manifest.eval_input_digest != request.eval_input.digest():
        raise MaterializationError("Trial input digest does not match EvalInput")
    if request.suite_case.task_id != request.eval_input.task_id:
        raise MaterializationError("Suite Case task does not match EvalInput")
    if request.suite_case.eval_input_digest != request.eval_input.digest():
        raise MaterializationError("Suite Case input digest does not match EvalInput")
    if (
        request.suite_case.canonical_case_digest
        != request.trial_manifest.canonical_case_digest
    ):
        raise MaterializationError("Suite Case digest does not match Trial")
    if request.trial_manifest.adapter_capabilities_digest != (
        request.adapter_capabilities.digest()
    ):
        raise MaterializationError("Adapter capability digest drifted")
    if expected_kind not in request.adapter_capabilities.target_kinds:
        raise MaterializationError("Adapter does not support review Target kind")
    if request.suite_preparation_binding_digest != (
        request.trial_manifest.suite_preparation_binding_digest
    ):
        raise MaterializationError("Suite preparation binding drifted")


def _replay_identity(replay: PreparedRepositoryReplay) -> Tuple[str, str, str, str]:
    return (
        replay.prepared_repository_id,
        replay.repository_descriptor_digest,
        replay.base_revision,
        replay.head_revision,
    )


def repository_replay_binding_digest(
    prepared: PreparedRepository,
    replay: PreparedRepositoryReplay,
) -> str:
    if replay.prepared_repository_id != prepared.manifest.prepared_repository_id:
        raise MaterializationError("replay is not bound to PreparedRepository")
    if replay.repository_descriptor_digest != prepared.repository.digest():
        raise MaterializationError("replay Repository descriptor drifted")
    return canonical_sha256(
        {
            "kind": "repository_replay_binding_v2",
            "identity": list(_replay_identity(replay)),
        }
    )


def _repository_file_bindings(
    workspace: TrialWorkspace,
    replay: PreparedRepositoryReplay,
    head_revision: str,
) -> Tuple[AgentVisibleFileBinding, ...]:
    paths = replay.paths(head_revision)
    if not paths:
        raise MaterializationError("Repository Target contains no readable files")
    bindings = []
    for path in paths:
        replay_bytes = replay.read_file(head_revision, path)
        if replay_bytes is None:
            raise MaterializationError("Repository replay file disappeared")
        try:
            workspace_bytes = workspace.read_file(path)
        except RepositoryPreparationError as exc:
            raise MaterializationError(
                "Agent-visible Repository file is unavailable"
            ) from exc
        if workspace_bytes != replay_bytes:
            raise MaterializationError(
                "Agent-visible Repository file differs from replay source"
            )
        bindings.append(
            AgentVisibleFileBinding(
                role="repository_file",
                relative_path=path,
                size_bytes=len(replay_bytes),
                sha256=hashlib.sha256(replay_bytes).hexdigest(),
            )
        )
    return tuple(bindings)


class RepositoryTargetMaterializer:
    """Materialize one Repository Target using a verified cache only."""

    def __init__(self, preparer: RepositoryPreparer) -> None:
        if not isinstance(preparer, RepositoryPreparer):
            raise TypeError("RepositoryTargetMaterializer requires RepositoryPreparer")
        if preparer.repository_mode is not RepositoryMode.CACHE_ONLY:
            raise MaterializationError(
                "RepositoryTargetMaterializer requires CACHE_ONLY RepositoryPreparer"
            )
        self._preparer = preparer

    def materialize(
        self, request: MaterializationRequest
    ) -> PreparedTargetMaterialization[PreparedRepositoryReplay]:
        if not isinstance(request, MaterializationRequest):
            raise TypeError("materialize requires MaterializationRequest")
        _validate_repository_request(request)
        repository = repository_from_eval_input(request.eval_input)
        prepared = self._preparer.require_cached(repository)
        replay = self._preparer.open_replay(prepared)
        workspace: Optional[TrialWorkspace] = None
        try:
            workspace = self._preparer.trial_workspace(
                prepared,
                trial_manifest=request.trial_manifest,
                suite_case=request.suite_case,
                eval_input=request.eval_input,
                attempt=request.attempt,
            )
            files = _repository_file_bindings(
                workspace,
                replay,
                repository.head_revision,
            )
            replay_digest = repository_replay_binding_digest(prepared, replay)
            manifest = TrialMaterializationManifest.create(
                run_id=request.trial_manifest.run_id,
                task_id=request.eval_input.task_id,
                trial_id=request.trial_manifest.trial_id,
                attempt=request.attempt,
                eval_input_digest=request.eval_input.digest(),
                review_target_digest=request.eval_input.review_target.digest(),
                wire_contract=request.wire_contract,
                suite_preparation_binding_digest=(
                    request.suite_preparation_binding_digest
                ),
                prepared_source_id=prepared.manifest.prepared_repository_id,
                adapter_capabilities_digest=request.adapter_capabilities.digest(),
                readable_relative_paths=tuple(item.relative_path for item in files),
                files=files,
                replay_binding_digest=replay_digest,
            )
            lease = _RepositoryMaterializationLease(
                request=request,
                manifest=manifest,
                prepared_repository=prepared,
                replay=replay,
                workspace=workspace,
                preparer=self._preparer,
            )
            result = PreparedTargetMaterialization(
                request=request,
                manifest=manifest,
                replay=replay,
                _lease=lease,
            )
            result.validate()
            return result
        except BaseException:
            if workspace is not None and not workspace.closed:
                try:
                    workspace.record_terminal_status(TrialStatus.FAILED)
                    workspace.close()
                except BaseException:
                    pass
            raise


__all__ = [
    "MaterializationError",
    "MaterializationRequest",
    "PreparedTargetMaterialization",
    "RepositoryTargetMaterializer",
    "TargetMaterializer",
    "repository_replay_binding_digest",
]
