from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import threading
from typing import Any, Iterable

from review_agent.pr_workspace import (
    ArtifactDescriptor,
    PRWorkspaceStore,
    SnapshotWorkspace,
)
from review_agent.safe_io import (
    SafeIOError,
    atomic_replace_bytes,
    assert_regular_file,
    canonical_json_bytes,
    resolve_managed_path,
    strict_json_loads,
)
from review_agent.tool_result_protocol import (
    ReviewToolResult,
    ToolErrorEnvelope,
    ToolResultProjectionV2,
    serialize_tool_result_projection_v2,
    serialized_tool_content_chars,
)


NON_REACQUIRABLE_ARTIFACT_THRESHOLD_CHARS = 50_000
TOOL_RESULTS_PER_TURN_BUDGET_CHARS = 200_000
TOOL_RESULT_PREVIEW_CHARS = 2_000
MAX_ARTIFACT_PAGE_CHARS = 50_000
TOOL_RESULT_AGGREGATE_SCHEMA = "tool_result_aggregate_v1"


class ToolArtifactError(ValueError):
    pass


class ToolResultBudgetError(ToolArtifactError):
    pass


@dataclass(frozen=True)
class ToolResultLimits:
    artifact_threshold_chars: int = NON_REACQUIRABLE_ARTIFACT_THRESHOLD_CHARS
    turn_budget_chars: int = TOOL_RESULTS_PER_TURN_BUDGET_CHARS
    preview_chars: int = TOOL_RESULT_PREVIEW_CHARS

    def __post_init__(self) -> None:
        for name, value in (
            ("artifact_threshold_chars", self.artifact_threshold_chars),
            ("turn_budget_chars", self.turn_budget_chars),
            ("preview_chars", self.preview_chars),
        ):
            if type(value) is not int or value <= 0:
                raise ToolArtifactError(f"{name} must be a positive integer")
        if self.preview_chars > self.artifact_threshold_chars:
            raise ToolArtifactError(
                "preview_chars must not exceed artifact_threshold_chars"
            )


@dataclass(frozen=True)
class ArtifactPage:
    artifact_id: str
    cursor: int
    next_cursor: int | None
    has_more: bool
    content: str
    total_chars: int


@dataclass(frozen=True)
class ToolResultBatch:
    projections: tuple[ToolResultProjectionV2, ...]
    total_rendered_chars: int
    retained_unexternalized: tuple[ReviewToolResult, ...] = ()

    def __post_init__(self) -> None:
        if type(self.projections) is not tuple or any(
            type(item) is not ToolResultProjectionV2 for item in self.projections
        ):
            raise ToolArtifactError("projections are invalid")
        actual = sum(
            len(serialize_tool_result_projection_v2(item))
            for item in self.projections
        )
        if actual != self.total_rendered_chars:
            raise ToolArtifactError("total_rendered_chars does not match projections")


class _PreflightArtifactSink:
    def __init__(self, owner: "ToolResultArtifactStore") -> None:
        self._owner = owner

    def publish(
        self,
        *,
        snapshot_id: str,
        logical_name: str,
        content: bytes,
        content_type: str,
    ) -> str:
        if snapshot_id != self._owner.snapshot.snapshot_id:
            raise ToolArtifactError("Preflight Artifact Snapshot binding does not match")
        if type(content) is not bytes:
            raise ToolArtifactError("Preflight Artifact content must be bytes")
        if type(logical_name) is not str or not logical_name.strip():
            raise ToolArtifactError("Preflight logical_name must be non-empty")
        if type(content_type) is not str or not content_type.strip():
            raise ToolArtifactError("Preflight content_type must be non-empty")
        digest = hashlib.sha256(content).hexdigest()
        suffix = Path(logical_name).suffix or ".bin"
        descriptor = self._owner.workspace_store.publish_create_only(
            self._owner.snapshot,
            f"QualityGate/artifacts/q-{digest[:32]}{suffix}",
            content,
        )
        return descriptor.artifact_id


class ToolResultArtifactStore:
    def __init__(
        self,
        workspace_store: PRWorkspaceStore,
        snapshot: SnapshotWorkspace,
    ) -> None:
        if not isinstance(workspace_store, PRWorkspaceStore):
            raise ToolArtifactError("workspace_store must be PRWorkspaceStore")
        workspace_store.verify_snapshot(snapshot)
        self.workspace_store = workspace_store
        self.snapshot = snapshot
        self._index_lock = threading.Lock()

    def publish_text(
        self,
        content: str,
        *,
        logical_kind: str,
    ) -> ArtifactDescriptor:
        if type(content) is not str:
            raise ToolArtifactError("Tool Artifact content must be text")
        try:
            encoded = content.encode("utf-8", "strict")
        except UnicodeError as error:
            raise ToolArtifactError("Tool Artifact content must be UTF-8") from error
        if type(logical_kind) is not str or re.fullmatch(
            r"[a-z0-9][a-z0-9_-]{0,63}", logical_kind
        ) is None:
            raise ToolArtifactError("logical_kind is invalid")
        digest = hashlib.sha256(encoded).hexdigest()
        return self.workspace_store.publish_create_only(
            self.snapshot,
            f"ToolResults/artifacts/t-{digest[:32]}.txt",
            encoded,
        )

    def publish_aggregate(
        self,
        results: Iterable[ReviewToolResult],
    ) -> ArtifactDescriptor:
        values = tuple(results)
        if not values or any(
            not isinstance(result, ReviewToolResult) or result.is_error
            for result in values
        ):
            raise ToolArtifactError(
                "aggregate requires successful ReviewToolResult values"
            )
        snapshot_ids = {result.snapshot_id for result in values}
        if snapshot_ids != {self.snapshot.snapshot_id}:
            raise ToolArtifactError("aggregate Snapshot binding does not match")
        payload = {
            "schema_version": TOOL_RESULT_AGGREGATE_SCHEMA,
            "entries": [
                {
                    "tool_call_id": result.tool_call_id,
                    "session_id": result.session_id,
                    "snapshot_id": result.snapshot_id,
                    "tool_name": result.tool_name,
                    "canonical_arguments_hash": result.canonical_arguments_hash,
                    "content": result.content,
                }
                for result in values
            ],
        }
        text = canonical_json_bytes(payload).decode("utf-8")
        return self.publish_text(text, logical_kind="aggregate")

    def read_artifact(
        self,
        artifact_id: str,
        *,
        cursor: int = 0,
        max_chars: int = MAX_ARTIFACT_PAGE_CHARS,
    ) -> ArtifactPage:
        if type(cursor) is not int or cursor < 0:
            raise ToolArtifactError("Artifact cursor must be non-negative")
        if (
            type(max_chars) is not int
            or not 1 <= max_chars <= MAX_ARTIFACT_PAGE_CHARS
        ):
            raise ToolArtifactError(
                f"max_chars must be between 1 and {MAX_ARTIFACT_PAGE_CHARS}"
            )
        try:
            content = self.workspace_store.read_verified_artifact(
                self.snapshot, artifact_id
            ).decode("utf-8", "replace")
        except Exception as error:
            raise ToolArtifactError("Artifact is unavailable") from error
        if cursor > len(content):
            raise ToolArtifactError("Artifact cursor is out of range")
        end = min(len(content), cursor + max_chars)
        has_more = end < len(content)
        return ArtifactPage(
            artifact_id=artifact_id,
            cursor=cursor,
            next_cursor=end if has_more else None,
            has_more=has_more,
            content=content[cursor:end],
            total_chars=len(content),
        )

    def preflight_sink(self) -> _PreflightArtifactSink:
        return _PreflightArtifactSink(self)

    def append_index(self, record: dict[str, Any]) -> None:
        if type(record) is not dict:
            raise ToolArtifactError("Tool Result index record must be an object")
        encoded = canonical_json_bytes(record) + b"\n"
        try:
            path = resolve_managed_path(
                self.snapshot.path / "ToolResults", "index.jsonl"
            )
        except SafeIOError as error:
            raise ToolArtifactError("Tool Result index path is unsafe") from error
        with self._index_lock:
            self.workspace_store.verify_snapshot(self.snapshot)
            try:
                existing = (
                    assert_regular_file(path).read_bytes()
                    if path.exists()
                    else b""
                )
                atomic_replace_bytes(path, existing + encoded)
            except (OSError, SafeIOError) as error:
                raise ToolArtifactError("Tool Result index is unavailable") from error

    def read_index(self) -> list[dict[str, Any]]:
        try:
            path = resolve_managed_path(
                self.snapshot.path / "ToolResults", "index.jsonl"
            )
        except SafeIOError as error:
            raise ToolArtifactError("Tool Result index path is unsafe") from error
        self.workspace_store.verify_snapshot(self.snapshot)
        try:
            raw = assert_regular_file(path).read_bytes() if path.exists() else b""
        except (OSError, SafeIOError) as error:
            raise ToolArtifactError("Tool Result index is unavailable") from error
        records: list[dict[str, Any]] = []
        for line in raw.splitlines():
            value = strict_json_loads(line)
            if type(value) is not dict:
                raise ToolArtifactError("Tool Result index record is invalid")
            records.append(value)
        return records


def _preview(content: str, maximum: int) -> str:
    if serialized_tool_content_chars(content) <= maximum:
        return content
    low = 0
    high = len(content)
    while low < high:
        middle = (low + high + 1) // 2
        if serialized_tool_content_chars(content[:middle]) <= maximum:
            low = middle
        else:
            high = middle - 1
    return content[:low]


def _rendered_size(values: Iterable[ToolResultProjectionV2]) -> int:
    return sum(len(serialize_tool_result_projection_v2(value)) for value in values)


class ToolResultProjector:
    def __init__(
        self,
        artifact_store: ToolResultArtifactStore,
        *,
        limits: ToolResultLimits | None = None,
    ) -> None:
        if not isinstance(artifact_store, ToolResultArtifactStore):
            raise ToolArtifactError(
                "artifact_store must be ToolResultArtifactStore"
            )
        self._artifacts = artifact_store
        self._limits = limits or ToolResultLimits()

    def project_turn(
        self,
        results: Iterable[ReviewToolResult],
    ) -> ToolResultBatch:
        raw_results = tuple(results)
        if any(type(result) is not ReviewToolResult for result in raw_results):
            raise ToolArtifactError(
                "results must contain only ReviewToolResult values"
            )
        if any(
            result.snapshot_id != self._artifacts.snapshot.snapshot_id
            for result in raw_results
        ):
            raise ToolArtifactError("Tool Result Snapshot binding does not match")
        call_ids = [result.tool_call_id for result in raw_results]
        if len(call_ids) != len(set(call_ids)):
            raise ToolArtifactError("Tool Result call IDs must be unique per turn")

        projections: list[ToolResultProjectionV2] = []
        retained: list[ReviewToolResult] = []
        for result in raw_results:
            if result.error is not None:
                projections.append(
                    ToolResultProjectionV2.from_error(
                        tool_call_id=result.tool_call_id,
                        tool_name=result.tool_name,
                        error=result.error,
                    )
                )
                continue
            original_size = serialized_tool_content_chars(result.content)
            if (
                not result.reacquirable
                and original_size > self._limits.artifact_threshold_chars
            ):
                try:
                    descriptor = self._artifacts.publish_text(
                        result.content,
                        logical_kind="tool-result",
                    )
                    projections.append(
                        ToolResultProjectionV2(
                            tool_call_id=result.tool_call_id,
                            tool_name=result.tool_name,
                            status="artifact",
                            original_size=original_size,
                            reacquirable=False,
                            preview=_preview(
                                result.content, self._limits.preview_chars
                            ),
                            artifact_id=descriptor.artifact_id,
                        )
                    )
                except Exception:
                    retained.append(result)
                    projections.append(
                        ToolResultProjectionV2.from_error(
                            tool_call_id=result.tool_call_id,
                            tool_name=result.tool_name,
                            original_size=original_size,
                            error=ToolErrorEnvelope(
                                code="artifact_write_failed",
                                retryable=True,
                                message=(
                                    "The complete Tool Result could not be "
                                    "persisted; Runtime retained it for retry"
                                ),
                            ),
                        )
                    )
            else:
                projections.append(ToolResultProjectionV2.inline(result))

        if _rendered_size(projections) > self._limits.turn_budget_chars:
            for index, result in enumerate(raw_results):
                if _rendered_size(projections) <= self._limits.turn_budget_chars:
                    break
                projection = projections[index]
                if result.reacquirable and projection.status == "inline":
                    projections[index] = ToolResultProjectionV2(
                        tool_call_id=result.tool_call_id,
                        tool_name=result.tool_name,
                        status="evicted",
                        original_size=projection.original_size,
                        reacquirable=True,
                        preview=_preview(
                            result.content, self._limits.preview_chars
                        ),
                        reacquire_arguments=result.arguments,
                    )

        if _rendered_size(projections) > self._limits.turn_budget_chars:
            aggregate_indices = [
                index
                for index, (result, projection) in enumerate(
                    zip(raw_results, projections)
                )
                if not result.reacquirable
                and not result.is_error
                and projection.status == "inline"
            ]
            if aggregate_indices:
                aggregate_results = tuple(
                    raw_results[index] for index in aggregate_indices
                )
                try:
                    descriptor = self._artifacts.publish_aggregate(
                        aggregate_results
                    )
                    for entry_index, result_index in enumerate(aggregate_indices):
                        result = raw_results[result_index]
                        projections[result_index] = ToolResultProjectionV2(
                            tool_call_id=result.tool_call_id,
                            tool_name=result.tool_name,
                            status="aggregate_artifact",
                            original_size=serialized_tool_content_chars(
                                result.content
                            ),
                            reacquirable=False,
                            preview=_preview(
                                result.content, self._limits.preview_chars
                            ),
                            artifact_id=descriptor.artifact_id,
                            aggregate_entry=entry_index,
                        )
                except Exception:
                    for result_index in aggregate_indices:
                        result = raw_results[result_index]
                        retained.append(result)
                        projections[result_index] = (
                            ToolResultProjectionV2.from_error(
                                tool_call_id=result.tool_call_id,
                                tool_name=result.tool_name,
                                original_size=serialized_tool_content_chars(
                                    result.content
                                ),
                                error=ToolErrorEnvelope(
                                    code="artifact_write_failed",
                                    retryable=True,
                                    message=(
                                        "The aggregate Tool Result could not be "
                                        "persisted; Runtime retained it for retry"
                                    ),
                                ),
                            )
                        )

        if _rendered_size(projections) > self._limits.turn_budget_chars:
            for index, projection in enumerate(projections):
                if _rendered_size(projections) <= self._limits.turn_budget_chars:
                    break
                if projection.preview:
                    projections[index] = replace(projection, preview="")

        total = _rendered_size(projections)
        if total > self._limits.turn_budget_chars:
            raise ToolResultBudgetError(
                "Tool Result references exceed the per-turn hard limit"
            )

        now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        for result, projection in zip(raw_results, projections):
            self._artifacts.append_index(
                {
                    "tool_call_id": result.tool_call_id,
                    "session_id": result.session_id,
                    "snapshot_id": result.snapshot_id,
                    "tool_name": result.tool_name,
                    "canonical_arguments_hash": result.canonical_arguments_hash,
                    "status": "failed" if projection.is_error else "completed",
                    "is_error": projection.is_error,
                    "error_code": (
                        projection.error.code
                        if projection.error is not None
                        else None
                    ),
                    "retryable": (
                        projection.error.retryable
                        if projection.error is not None
                        else None
                    ),
                    "exit_code": result.exit_code,
                    "created_at": now,
                    "content_hash": hashlib.sha256(
                        result.content.encode("utf-8")
                    ).hexdigest(),
                    "rendered_size": len(
                        serialize_tool_result_projection_v2(projection)
                    ),
                    "reacquirable": result.reacquirable,
                    "artifact_id": projection.artifact_id,
                    "context_evicted_at": (
                        now if projection.status == "evicted" else None
                    ),
                }
            )
        retained_by_call_id = {
            result.tool_call_id: result for result in retained
        }
        return ToolResultBatch(
            projections=tuple(projections),
            total_rendered_chars=total,
            retained_unexternalized=tuple(retained_by_call_id.values()),
        )


__all__ = [
    "ArtifactPage",
    "MAX_ARTIFACT_PAGE_CHARS",
    "NON_REACQUIRABLE_ARTIFACT_THRESHOLD_CHARS",
    "TOOL_RESULTS_PER_TURN_BUDGET_CHARS",
    "TOOL_RESULT_PREVIEW_CHARS",
    "ToolArtifactError",
    "ToolResultArtifactStore",
    "ToolResultBatch",
    "ToolResultBudgetError",
    "ToolResultLimits",
    "ToolResultProjector",
]
