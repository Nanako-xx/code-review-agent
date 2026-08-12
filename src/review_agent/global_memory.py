from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import re
from typing import Iterable

from review_agent.memory_models import (
    DurableMemoryRecord,
    MemoryKind,
    RecordStatus,
    Sensitivity,
)
from review_agent.safe_io import canonical_json_bytes


GLOBAL_MEMORY_SCHEMA = "global_memory_snapshot_v1"
_MEMORY_ID = re.compile(r"\AMEM-[0-9a-f]{64}\Z")
_SNAPSHOT_ID = re.compile(r"\AGM-[0-9a-f]{64}\Z")


class GlobalMemoryCategory(str, Enum):
    USER_RULE = "user_rule"
    APPROVED_EXPERIENCE = "approved_experience"


_CATEGORY_BY_KIND = {
    MemoryKind.REVIEW_RULE: GlobalMemoryCategory.USER_RULE,
    MemoryKind.INCIDENT_LESSON: GlobalMemoryCategory.APPROVED_EXPERIENCE,
}


class GlobalMemoryError(ValueError):
    pass


def _text(value: object, field_name: str) -> str:
    if type(value) is not str or not value.strip() or value != value.strip():
        raise GlobalMemoryError(f"{field_name} must be canonical non-empty text")
    if "\x00" in value:
        raise GlobalMemoryError(f"{field_name} contains an unsafe control character")
    return value


def global_memory_topic(record: DurableMemoryRecord) -> str:
    if not isinstance(record, DurableMemoryRecord):
        raise GlobalMemoryError("record must be a DurableMemoryRecord")
    scopes: list[str] = []
    for name, values in (
        ("paths", record.scope.paths),
        ("symbols", record.scope.symbols),
        ("contracts", record.scope.contracts),
        ("languages", record.scope.languages),
    ):
        if values:
            scopes.append(f"{name}=" + ",".join(values))
    suffix = ";".join(scopes) if scopes else "global"
    return f"{record.kind.value}|{suffix}"


@dataclass(frozen=True)
class GlobalMemoryEntry:
    memory_id: str
    category: GlobalMemoryCategory
    topic: str
    content: str

    def __post_init__(self) -> None:
        if type(self.memory_id) is not str or _MEMORY_ID.fullmatch(
            self.memory_id
        ) is None:
            raise GlobalMemoryError("memory_id is invalid")
        if not isinstance(self.category, GlobalMemoryCategory):
            raise GlobalMemoryError("category is invalid")
        _text(self.topic, "topic")
        _text(self.content, "content")

    def to_dict(self) -> dict[str, str]:
        return {
            "memory_id": self.memory_id,
            "category": self.category.value,
            "topic": self.topic,
            "content": self.content,
        }


@dataclass(frozen=True)
class GlobalMemorySnapshot:
    snapshot_id: str
    entries: tuple[GlobalMemoryEntry, ...]

    def __post_init__(self) -> None:
        if type(self.snapshot_id) is not str or _SNAPSHOT_ID.fullmatch(
            self.snapshot_id
        ) is None:
            raise GlobalMemoryError("snapshot_id is invalid")
        if type(self.entries) is not tuple or any(
            type(entry) is not GlobalMemoryEntry for entry in self.entries
        ):
            raise GlobalMemoryError(
                "entries must be a tuple of GlobalMemoryEntry values"
            )
        memory_ids = [entry.memory_id for entry in self.entries]
        if memory_ids != sorted(memory_ids) or len(memory_ids) != len(set(memory_ids)):
            raise GlobalMemoryError(
                "entries must have unique memory IDs in canonical order"
            )
        expected = _snapshot_id(self.entries)
        if expected != self.snapshot_id:
            raise GlobalMemoryError("snapshot_id does not match entries")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": GLOBAL_MEMORY_SCHEMA,
            "snapshot_id": self.snapshot_id,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def to_json(self) -> str:
        return canonical_json_bytes(self.to_dict()).decode("utf-8")


def _snapshot_id(entries: tuple[GlobalMemoryEntry, ...]) -> str:
    digest = hashlib.sha256(
        canonical_json_bytes([entry.to_dict() for entry in entries])
    ).hexdigest()
    return "GM-" + digest


class GlobalMemoryFacade:
    """Freeze approved durable records before any Reviewer is constructed."""

    def freeze(
        self,
        records: Iterable[DurableMemoryRecord],
    ) -> GlobalMemorySnapshot:
        selected: dict[str, GlobalMemoryEntry] = {}
        for index, record in enumerate(records):
            if not isinstance(record, DurableMemoryRecord):
                raise GlobalMemoryError(
                    f"records[{index}] must be a DurableMemoryRecord"
                )
            try:
                verified = DurableMemoryRecord.from_dict(record.to_dict())
            except ValueError as error:
                raise GlobalMemoryError(
                    f"records[{index}] failed durable-memory integrity validation"
                ) from error
            category = _CATEGORY_BY_KIND.get(verified.kind)
            if (
                category is None
                or verified.status is not RecordStatus.ACTIVE
                or verified.sensitivity is not Sensitivity.NORMAL
            ):
                continue
            entry = GlobalMemoryEntry(
                memory_id=verified.memory_id,
                category=category,
                topic=global_memory_topic(verified),
                content=verified.statement,
            )
            previous = selected.get(entry.memory_id)
            if previous is not None and previous != entry:
                raise GlobalMemoryError("conflicting durable-memory identity")
            selected[entry.memory_id] = entry
        entries = tuple(selected[key] for key in sorted(selected))
        return GlobalMemorySnapshot(
            snapshot_id=_snapshot_id(entries),
            entries=entries,
        )


__all__ = [
    "GLOBAL_MEMORY_SCHEMA",
    "GlobalMemoryCategory",
    "GlobalMemoryEntry",
    "GlobalMemoryError",
    "GlobalMemoryFacade",
    "GlobalMemorySnapshot",
    "global_memory_topic",
]
