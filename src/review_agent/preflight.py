from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from review_agent.diff_artifact import DiffArtifactStore
from review_agent.local_quality import (
    LocalQualityPlan,
    LocalQualityRunner,
    PreflightArtifactSink,
    QualityGateResult,
    QualityGateStatus,
)
from review_agent.pr_workspace import PRWorkspaceStore, SnapshotWorkspace
from review_agent.repository_intelligence import (
    ChangedSymbolsV2,
    build_changed_symbols_v2,
    changed_symbols_v2_to_dict,
)
from review_agent.safe_io import canonical_json_bytes


class PreflightBlockedError(RuntimeError):
    """The immutable Snapshot or authoritative Diff could not be established."""


@dataclass(frozen=True)
class PreflightResult:
    snapshot_id: str
    diff_artifact_id: str
    diff_index_artifact_id: str
    quality_artifact_id: str
    changed_symbols_artifact_id: str
    quality: QualityGateResult
    changed_symbols: ChangedSymbolsV2

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "preflight_result_v1",
            "snapshot_id": self.snapshot_id,
            "diff_artifact_id": self.diff_artifact_id,
            "diff_index_artifact_id": self.diff_index_artifact_id,
            "quality_artifact_id": self.quality_artifact_id,
            "changed_symbols_artifact_id": self.changed_symbols_artifact_id,
            "quality_status": self.quality.status.value,
            "changed_symbol_count": len(self.changed_symbols.symbols),
        }


ChangedSymbolsBuilder = Callable[..., ChangedSymbolsV2]


class DeterministicPreflight:
    """Run the sole v6 sequence: Diff -> local quality -> ChangedSymbols."""

    def __init__(
        self,
        *,
        workspace_store: PRWorkspaceStore,
        diff_store: DiffArtifactStore,
        quality_runner: LocalQualityRunner,
        changed_symbols_builder: ChangedSymbolsBuilder = build_changed_symbols_v2,
    ) -> None:
        self._workspace_store = workspace_store
        self._diff_store = diff_store
        self._quality_runner = quality_runner
        self._changed_symbols_builder = changed_symbols_builder

    def run(
        self,
        repository: Path,
        snapshot: SnapshotWorkspace,
        quality_plan: LocalQualityPlan,
        *,
        sink: PreflightArtifactSink,
    ) -> PreflightResult:
        try:
            diff = self._diff_store.materialize(Path(repository), snapshot)
        except Exception as error:
            raise PreflightBlockedError(
                "DiffArtifact could not be established for the Snapshot"
            ) from error
        if diff.index.snapshot_id != snapshot.snapshot_id:
            raise PreflightBlockedError(
                "DiffArtifact Snapshot binding does not match"
            )

        try:
            quality = self._quality_runner.run(
                Path(repository),
                snapshot.snapshot_id,
                quality_plan,
                sink,
            )
        except Exception:
            quality = QualityGateResult(
                snapshot_id=snapshot.snapshot_id,
                status=QualityGateStatus.ERROR,
                commands=(),
                reason_code="quality_runtime_error",
            )
        if quality.snapshot_id != snapshot.snapshot_id:
            quality = QualityGateResult(
                snapshot_id=snapshot.snapshot_id,
                status=QualityGateStatus.ERROR,
                commands=(),
                reason_code="quality_snapshot_mismatch",
            )

        changed_files = [file.path for file in diff.index.files]
        try:
            changed_symbols = self._changed_symbols_builder(
                Path(repository),
                snapshot_id=snapshot.snapshot_id,
                base_sha=snapshot.base_sha,
                head_sha=snapshot.head_sha,
                changed_files=changed_files,
            )
        except Exception:
            changed_symbols = ChangedSymbolsV2.empty(
                snapshot_id=snapshot.snapshot_id,
                base_sha=snapshot.base_sha,
                head_sha=snapshot.head_sha,
                changed_files=changed_files,
                coverage_status="error",
                reason_code="analyzer_runtime_error",
            )
        if changed_symbols.snapshot_id != snapshot.snapshot_id:
            changed_symbols = ChangedSymbolsV2.empty(
                snapshot_id=snapshot.snapshot_id,
                base_sha=snapshot.base_sha,
                head_sha=snapshot.head_sha,
                changed_files=changed_files,
                coverage_status="error",
                reason_code="analyzer_snapshot_mismatch",
            )

        quality_artifact = self._workspace_store.publish_create_only(
            snapshot,
            "QualityGate/quality-gate.json",
            canonical_json_bytes(quality.to_dict()),
        )
        symbols_artifact = self._workspace_store.publish_create_only(
            snapshot,
            "ChangedSymbols/changed-symbols.json",
            canonical_json_bytes(changed_symbols_v2_to_dict(changed_symbols)),
        )
        result = PreflightResult(
            snapshot_id=snapshot.snapshot_id,
            diff_artifact_id=diff.patch.artifact_id,
            diff_index_artifact_id=diff.index_artifact.artifact_id,
            quality_artifact_id=quality_artifact.artifact_id,
            changed_symbols_artifact_id=symbols_artifact.artifact_id,
            quality=quality,
            changed_symbols=changed_symbols,
        )
        self._workspace_store.publish_create_only(
            snapshot,
            "preflight.json",
            canonical_json_bytes(result.to_dict()),
        )
        return result


__all__ = [
    "DeterministicPreflight",
    "PreflightArtifactSink",
    "PreflightBlockedError",
    "PreflightResult",
]
