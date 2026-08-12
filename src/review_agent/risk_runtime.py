from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping

from review_agent.diff_artifact import DiffArtifactIndex
from review_agent.pr_workspace import (
    PRWorkspaceError,
    PRWorkspaceStore,
    SnapshotWorkspace,
)
from review_agent.review_protocol import (
    IntentPacket,
    IntentSource,
    RiskDecision,
    RiskLevel,
)
from review_agent.safe_io import canonical_json_bytes


RISK_RECORD_SCHEMA = "risk_decision_v2"
_SNAPSHOT_ID = re.compile(r"\AS-[0-9a-f]{64}\Z")
_RISK_ORDER = {
    RiskLevel.LOW: 0,
    RiskLevel.MEDIUM: 1,
    RiskLevel.HIGH: 2,
    RiskLevel.CRITICAL: 3,
}


class RiskRuntimeError(ValueError):
    pass


class RiskRuntimeIntegrityError(RiskRuntimeError):
    pass


def _require_level(value: Any, field_name: str) -> RiskLevel:
    if not isinstance(value, RiskLevel):
        raise RiskRuntimeError(f"{field_name} must be a RiskLevel")
    return value


def _max_level(levels: Iterable[RiskLevel]) -> RiskLevel:
    values = tuple(levels)
    if not values:
        raise RiskRuntimeError("at least one risk level is required")
    for index, level in enumerate(values):
        _require_level(level, f"risk_levels[{index}]")
    return max(values, key=_RISK_ORDER.__getitem__)


def deterministic_risk_floor(
    changed_file_count: int,
    intent: IntentPacket,
    *,
    additional_floors: Iterable[RiskLevel] = (),
) -> RiskLevel:
    """Return the monotonic Runtime floor for one immutable Snapshot."""

    if type(changed_file_count) is not int or changed_file_count < 0:
        raise RiskRuntimeError("changed_file_count must be a non-negative integer")
    if type(intent) is not IntentPacket:
        raise RiskRuntimeError("intent must be an IntentPacket")

    floors = [RiskLevel.LOW]
    if changed_file_count > 50:
        floors.append(RiskLevel.MEDIUM)
    if intent.source is IntentSource.INFERRED:
        floors.append(RiskLevel.MEDIUM)
    elif intent.source is None:
        floors.append(RiskLevel.HIGH)
    elif intent.source is not IntentSource.EXPLICIT:
        raise RiskRuntimeError("Intent source is unsupported")
    floors.extend(additional_floors)
    return _max_level(floors)


@dataclass(frozen=True)
class RiskRecord:
    snapshot_id: str
    deterministic_floor: RiskLevel
    model_level: RiskLevel | None
    final_level: RiskLevel

    def __post_init__(self) -> None:
        if type(self.snapshot_id) is not str or _SNAPSHOT_ID.fullmatch(
            self.snapshot_id
        ) is None:
            raise RiskRuntimeError("snapshot_id is invalid")
        _require_level(self.deterministic_floor, "deterministic_floor")
        if self.model_level is not None:
            _require_level(self.model_level, "model_level")
        _require_level(self.final_level, "final_level")
        expected = _max_level(
            (
                self.deterministic_floor,
                *(() if self.model_level is None else (self.model_level,)),
            )
        )
        if self.final_level is not expected:
            raise RiskRuntimeError(
                "final_level must equal the maximum deterministic/model level"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": RISK_RECORD_SCHEMA,
            "snapshot_id": self.snapshot_id,
            "deterministic_floor": self.deterministic_floor.value,
            "model_level": (
                self.model_level.value if self.model_level is not None else None
            ),
            "final_level": self.final_level.value,
        }

    def to_decision(self) -> RiskDecision:
        return RiskDecision(self.final_level)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RiskRecord":
        expected = {
            "schema_version",
            "snapshot_id",
            "deterministic_floor",
            "model_level",
            "final_level",
        }
        if type(payload) is not dict or set(payload) != expected:
            raise RiskRuntimeIntegrityError("Risk record has an invalid exact schema")
        if payload["schema_version"] != RISK_RECORD_SCHEMA:
            raise RiskRuntimeIntegrityError("Risk record schema is unsupported")
        try:
            model_level = payload["model_level"]
            return cls(
                snapshot_id=payload["snapshot_id"],
                deterministic_floor=RiskLevel(payload["deterministic_floor"]),
                model_level=(
                    None if model_level is None else RiskLevel(model_level)
                ),
                final_level=RiskLevel(payload["final_level"]),
            )
        except (TypeError, ValueError) as error:
            raise RiskRuntimeIntegrityError("Risk record is invalid") from error


def compile_risk_record(
    *,
    snapshot_id: str,
    changed_file_count: int,
    intent: IntentPacket,
    model_decision: RiskDecision | None,
    additional_floors: Iterable[RiskLevel] = (),
) -> RiskRecord:
    if model_decision is not None and type(model_decision) is not RiskDecision:
        raise RiskRuntimeError("model_decision must be a RiskDecision or null")
    floor = deterministic_risk_floor(
        changed_file_count,
        intent,
        additional_floors=additional_floors,
    )
    model_level = model_decision.level if model_decision is not None else None
    final_level = _max_level(
        (floor, *(() if model_level is None else (model_level,)))
    )
    return RiskRecord(
        snapshot_id=snapshot_id,
        deterministic_floor=floor,
        model_level=model_level,
        final_level=final_level,
    )


class RiskRuntime:
    def __init__(self, workspace_store: PRWorkspaceStore) -> None:
        if not isinstance(workspace_store, PRWorkspaceStore):
            raise RiskRuntimeError("Risk Runtime requires a PRWorkspaceStore")
        self._store = workspace_store

    def finalize(
        self,
        snapshot: SnapshotWorkspace,
        diff_index: DiffArtifactIndex,
        intent: IntentPacket,
        *,
        model_decision: RiskDecision | None = None,
        additional_floors: Iterable[RiskLevel] = (),
    ) -> RiskRecord:
        self._store.verify_snapshot(snapshot)
        if not isinstance(diff_index, DiffArtifactIndex):
            raise RiskRuntimeError("diff_index must be a DiffArtifactIndex")
        if diff_index.snapshot_id != snapshot.snapshot_id:
            raise RiskRuntimeIntegrityError(
                "DiffArtifact Snapshot binding does not match Risk Snapshot"
            )
        record = compile_risk_record(
            snapshot_id=snapshot.snapshot_id,
            changed_file_count=len(diff_index.files),
            intent=intent,
            model_decision=model_decision,
            additional_floors=additional_floors,
        )
        self._store.publish_create_only(
            snapshot,
            "Risk/risk.json",
            canonical_json_bytes(record.to_dict()),
        )
        return record

    def load(self, snapshot: SnapshotWorkspace) -> RiskRecord:
        self._store.verify_snapshot(snapshot)
        try:
            descriptor = self._store.find_snapshot_artifact(
                snapshot, "Risk/risk.json"
            )
            payload = self._store.read_verified_json(
                snapshot, descriptor.artifact_id
            )
        except PRWorkspaceError as error:
            raise RiskRuntimeIntegrityError("Risk record is unavailable") from error
        record = RiskRecord.from_dict(payload)
        if record.snapshot_id != snapshot.snapshot_id:
            raise RiskRuntimeIntegrityError(
                "Risk record Snapshot binding does not match"
            )
        return record


__all__ = [
    "RISK_RECORD_SCHEMA",
    "RiskRecord",
    "RiskRuntime",
    "RiskRuntimeError",
    "RiskRuntimeIntegrityError",
    "compile_risk_record",
    "deterministic_risk_floor",
]
