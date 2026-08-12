from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from review_agent.global_memory import (
    GlobalMemoryCategory,
    GlobalMemoryFacade,
    global_memory_topic,
)
from review_agent.memory_models import (
    DurableMemoryRecord,
    GitCommitSourceRef,
    MemoryConfidence,
    MemoryKind,
    MemoryScope,
    RecordStatus,
    Sensitivity,
    ValidityPolicy,
    stable_event_id,
    stable_id,
)


HEAD_SHA = "b" * 40


def _record(
    label: str,
    *,
    kind: MemoryKind,
    statement: str,
    status: RecordStatus = RecordStatus.ACTIVE,
    sensitivity: Sensitivity = Sensitivity.NORMAL,
) -> DurableMemoryRecord:
    candidate_id = stable_id("MC", "global-memory", label)
    return DurableMemoryRecord(
        candidate_id=candidate_id,
        repository_key="a" * 64,
        kind=kind,
        statement=statement,
        scope=MemoryScope(paths=("src/api.py",)),
        source_refs=(GitCommitSourceRef(HEAD_SHA),),
        source_bundle_hash="c" * 64,
        valid_from_sha=HEAD_SHA,
        validity_policies=(ValidityPolicy.MANUAL_UNTIL_REVOKED,),
        confidence=MemoryConfidence.HIGH,
        sensitivity=sensitivity,
        policy_effect=None,
        approved_by="amy",
        approval_event_id=stable_event_id("approve", candidate_id),
        status=status,
        created_at="2026-08-11T00:00:00Z",
    )


def test_facade_freezes_only_active_user_rules_and_approved_experience() -> None:
    user_rule = _record(
        "rule",
        kind=MemoryKind.REVIEW_RULE,
        statement="Public API changes require compatibility checks.",
    )
    experience = _record(
        "lesson",
        kind=MemoryKind.INCIDENT_LESSON,
        statement="Retry cleanup previously leaked temporary files.",
    )
    unrelated = _record(
        "module",
        kind=MemoryKind.HIGH_RISK_MODULE,
        statement="This module has a broad blast radius.",
    )
    revoked = _record(
        "revoked",
        kind=MemoryKind.REVIEW_RULE,
        statement="Obsolete rule.",
        status=RecordStatus.REVOKED,
    )
    blocked = _record(
        "blocked",
        kind=MemoryKind.REVIEW_RULE,
        statement="Must never leave the local machine.",
        sensitivity=Sensitivity.BLOCKED,
    )
    source = [user_rule, experience, unrelated, revoked, blocked]

    snapshot = GlobalMemoryFacade().freeze(source)
    source.clear()

    assert [entry.memory_id for entry in snapshot.entries] == sorted(
        [user_rule.memory_id, experience.memory_id]
    )
    assert {entry.category for entry in snapshot.entries} == {
        GlobalMemoryCategory.USER_RULE,
        GlobalMemoryCategory.APPROVED_EXPERIENCE,
    }
    assert "Obsolete rule" not in snapshot.to_json()
    assert "broad blast radius" not in snapshot.to_json()
    assert "local machine" not in snapshot.to_json()
    assert len(snapshot.snapshot_id) == len("GM-") + 64


def test_global_memory_snapshot_contains_no_pr_or_runtime_artifacts() -> None:
    record = _record(
        "minimal",
        kind=MemoryKind.REVIEW_RULE,
        statement="Check backward compatibility.",
    )

    payload = GlobalMemoryFacade().freeze((record,)).to_dict()

    assert set(payload) == {"schema_version", "snapshot_id", "entries"}
    assert set(payload["entries"][0]) == {
        "memory_id",
        "category",
        "topic",
        "content",
    }
    forbidden = {
        "diff",
        "intent",
        "assignment",
        "tool_result",
        "session_id",
        "pr_id",
    }
    assert forbidden.isdisjoint(payload)
    assert forbidden.isdisjoint(payload["entries"][0])


def test_global_memory_topic_is_deterministic_and_snapshot_is_immutable() -> None:
    record = _record(
        "topic",
        kind=MemoryKind.REVIEW_RULE,
        statement="Check backward compatibility.",
    )
    snapshot = GlobalMemoryFacade().freeze((record,))

    assert snapshot.entries[0].topic == global_memory_topic(record)
    assert GlobalMemoryFacade().freeze((record,)) == snapshot
    with pytest.raises(FrozenInstanceError):
        snapshot.snapshot_id = "GM-" + "0" * 64  # type: ignore[misc]
