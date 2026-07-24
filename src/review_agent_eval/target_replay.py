"""Verified, workspace-free replay resolution for evaluator stages."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, Union

from .artifacts import VerifiedTrialMaterialization
from .frozen_context import (
    FROZEN_CONTEXT_TARGET_PATH,
    FrozenContextReplay,
    open_frozen_context_replay,
)
from .models import (
    FrozenContextReviewTarget,
    RepositoryReviewTarget,
    ReviewTargetKind,
    canonical_sha256,
)
from .repository import (
    PreparedRepositoryReplay,
    RepositoryPreparer,
    repository_from_eval_input,
)


TargetReplay = Union[PreparedRepositoryReplay, FrozenContextReplay]


class TargetReplayIntegrityError(RuntimeError):
    """A resolver candidate is not bound to the committed Target replay."""


class TargetReplayResolver(Protocol):
    def resolve(self, source: VerifiedTrialMaterialization) -> TargetReplay:
        ...


def _repository_replay_binding_digest(replay: PreparedRepositoryReplay) -> str:
    return canonical_sha256(
        {
            "kind": "repository_replay_binding_v2",
            "identity": [
                replay.prepared_repository_id,
                replay.repository_descriptor_digest,
                replay.base_revision,
                replay.head_revision,
            ],
        }
    )


def validate_target_replay(
    source: VerifiedTrialMaterialization,
    replay: TargetReplay,
) -> None:
    """Independently bind a resolver result to the committed PREPARE manifest."""

    if type(source) is not VerifiedTrialMaterialization:
        raise TypeError("target replay requires VerifiedTrialMaterialization")
    target = source.eval_input.review_target
    manifest = source.manifest
    if target.kind is ReviewTargetKind.REPOSITORY:
        if (
            type(target) is not RepositoryReviewTarget
            or type(replay) is not PreparedRepositoryReplay
        ):
            raise TargetReplayIntegrityError(
                "Repository resolver returned the wrong replay type"
            )
        repository = repository_from_eval_input(source.eval_input)
        if (
            replay.prepared_repository_id != manifest.prepared_source_id
            or replay.repository_descriptor_digest != repository.digest()
            or replay.base_revision != repository.base_revision
            or replay.head_revision != repository.head_revision
            or _repository_replay_binding_digest(replay)
            != manifest.replay_binding_digest
        ):
            raise TargetReplayIntegrityError("Repository replay binding drifted")
        paths = replay.paths(repository.head_revision)
        files = manifest.files
        if (
            tuple(item.relative_path for item in files) != paths
            or manifest.target_access.readable_relative_paths != paths
            or any(item.role != "repository_file" for item in files)
        ):
            raise TargetReplayIntegrityError(
                "Repository replay path coverage drifted"
            )
        for item in files:
            data = replay.read_file(repository.head_revision, item.relative_path)
            if (
                data is None
                or len(data) != item.size_bytes
                or hashlib.sha256(data).hexdigest() != item.sha256
            ):
                raise TargetReplayIntegrityError(
                    "Repository replay file binding drifted"
                )
        return

    if (
        type(target) is not FrozenContextReviewTarget
        or type(replay) is not FrozenContextReplay
    ):
        raise TargetReplayIntegrityError(
            "Frozen resolver returned the wrong replay type"
        )
    if (
        manifest.prepared_source_id != target.bundle_id
        or replay.bundle_id != target.bundle_id
        or replay.record_id != target.record_id
        or replay.context_ref != target.record_id
        or replay.context_format != target.context_format
        or replay.rendered_sha256 != target.rendered_sha256
        or replay.rendered_utf8_bytes != target.rendered_utf8_bytes
        or replay.source_binding_digest != target.source_binding_digest
        or replay.replay_binding_digest != manifest.replay_binding_digest
    ):
        raise TargetReplayIntegrityError("Frozen replay binding drifted")
    data = replay.read_exact()
    if (
        len(manifest.files) != 1
        or manifest.files[0].role != "frozen_context"
        or manifest.files[0].relative_path != FROZEN_CONTEXT_TARGET_PATH
        or manifest.target_access.readable_relative_paths
        != (FROZEN_CONTEXT_TARGET_PATH,)
        or manifest.files[0].size_bytes != len(data)
        or manifest.files[0].sha256 != hashlib.sha256(data).hexdigest()
    ):
        raise TargetReplayIntegrityError("Frozen replay file binding drifted")


class RepositoryReplayResolver:
    def __init__(self, preparer: RepositoryPreparer) -> None:
        if not isinstance(preparer, RepositoryPreparer):
            raise TypeError("RepositoryReplayResolver requires RepositoryPreparer")
        self.preparer = preparer

    def resolve(
        self,
        source: VerifiedTrialMaterialization,
    ) -> PreparedRepositoryReplay:
        repository = repository_from_eval_input(source.eval_input)
        prepared = self.preparer.require_cached(repository)
        return self.preparer.open_replay(prepared)


class FrozenContextReplayResolver:
    def __init__(self, *, bundle_root: Path) -> None:
        self.bundle_root = Path(bundle_root)

    def resolve(self, source: VerifiedTrialMaterialization) -> FrozenContextReplay:
        return open_frozen_context_replay(
            bundle_root=self.bundle_root,
            eval_input=source.eval_input,
            suite_preparation_binding=source.suite_preparation_binding,
            suite_preparation_binding_digest=(
                source.manifest.suite_preparation_binding_digest
            ),
        )


__all__ = [
    "FrozenContextReplayResolver",
    "RepositoryReplayResolver",
    "TargetReplay",
    "TargetReplayIntegrityError",
    "TargetReplayResolver",
    "validate_target_replay",
]
