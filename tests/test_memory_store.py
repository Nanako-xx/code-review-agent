from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
import os
import sqlite3
import stat
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional, Tuple

import pytest

import review_agent.memory_store as memory_store_module

from conftest import run_git
from review_agent.memory_identity import (
    materialize_repository_memory_namespace,
    plan_repository_memory_namespace,
)
from review_agent.memory_models import (
    CandidateAuthorityReceipt,
    CandidateStatus,
    DurableMemoryRecord,
    ExpiryCondition,
    ExpiryConditionKind,
    FeedbackDecision,
    FeedbackReasonCode,
    FeedbackRecord,
    FeedbackStatus,
    FindingSeverity,
    FindingSnapshot,
    HumanDeclarationSourceRef,
    MemoryCandidate,
    MemoryConfidence,
    MemoryKind,
    MemoryScope,
    Producer,
    ProducerType,
    RecordStatus,
    RepositoryKnowledgeCapability,
    RepositoryKnowledgeEntry,
    RepositoryKnowledgeKey,
    RepositoryRangeSourceRef,
    Sensitivity,
    SourceBundleDescriptor,
    ValidityPolicy,
    canonical_json,
    canonical_sha256,
    stable_event_id,
    stable_request_id,
)
from review_agent.memory_store import (
    BlobGCResult,
    EXPORT_SCHEMA_NAME,
    STORE_SCHEMA_NAME,
    STORE_SCHEMA_VERSION,
    MemoryStore,
    MemoryStoreAuditSchema,
    MemoryStoreBusyError,
    MemoryStoreConflictError,
    MemoryStoreCorruptionError,
    MemoryStoreMigrationError,
    MemoryStoreReadOnlyError,
    MemoryStoreSchemaError,
    MemoryStoreUnavailableError,
    MemoryStoreValidationError,
)
from review_agent.revision import RevisionResolver


SHA_A = "a" * 40
HASH_1 = "1" * 64
HASH_2 = "2" * 64
CREATED_AT = "2026-07-14T12:00:00Z"
REPOSITORY_KEY = "4" * 64
AUTHORITY_RESOLUTION = "7" * 64


def _source() -> RepositoryRangeSourceRef:
    return RepositoryRangeSourceRef(
        revision=SHA_A,
        path="payments/money.py",
        line_start=10,
        line_end=18,
        content_hash=HASH_1,
    )


def _human_source(actor: str = "amy") -> HumanDeclarationSourceRef:
    return HumanDeclarationSourceRef(
        request_id=stable_request_id("human-source", actor),
        actor=actor,
        declaration_hash=HASH_2,
        created_at=CREATED_AT,
        review_id="review-001",
    )


def _candidate(
    *,
    statement: str = "Amounts must use Decimal.",
    status: CandidateStatus = CandidateStatus.PROPOSED,
) -> MemoryCandidate:
    return MemoryCandidate(
        repository_key=REPOSITORY_KEY,
        kind=MemoryKind.BUSINESS_INVARIANT,
        statement=statement,
        scope=MemoryScope(
            paths=("payments/**",),
            contracts=("numeric_correctness",),
            languages=("python",),
        ),
        source_refs=(_source(), _human_source()),
        valid_from_sha=SHA_A,
        validity_policies=(ValidityPolicy.SOURCE_CONTENT_HASH,),
        confidence=MemoryConfidence.HIGH,
        sensitivity=Sensitivity.NORMAL,
        policy_effect=None,
        producer=Producer(
            producer_type=ProducerType.MODEL,
            name="memory-curator",
            version="1.0.0",
        ),
        origin_review_id="review-001",
        status=status,
        created_at=CREATED_AT,
    )


def _authority_receipt(
    candidate: MemoryCandidate,
    *,
    authority_resolution_hash: str = AUTHORITY_RESOLUTION,
    created_at: str = CREATED_AT,
) -> CandidateAuthorityReceipt:
    return CandidateAuthorityReceipt(
        candidate_id=candidate.candidate_id,
        authority_repository_key=candidate.repository_key,
        locator_repository_key=candidate.repository_key,
        origin=candidate.producer.producer_type,
        review_id=candidate.origin_review_id,
        proposal_head_sha=candidate.valid_from_sha,
        authorized_source_refs=candidate.source_refs,
        human_declarations=(),
        initial_validation_report_hash="8" * 64,
        authority_resolution_hash=authority_resolution_hash,
        binding_id=None,
        created_at=created_at,
    )


def _bundle(candidate: MemoryCandidate, content: bytes) -> SourceBundleDescriptor:
    return SourceBundleDescriptor(
        repository_key=candidate.repository_key,
        candidate_id=candidate.candidate_id,
        source_refs=candidate.source_refs,
        blob_hash=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        media_type="application/vnd.review-agent.source-bundle+json",
        created_at=CREATED_AT,
    )


def _record(
    candidate: MemoryCandidate,
    bundle: SourceBundleDescriptor,
    request_id: str,
) -> DurableMemoryRecord:
    return DurableMemoryRecord(
        candidate_id=candidate.candidate_id,
        repository_key=candidate.repository_key,
        kind=candidate.kind,
        statement=candidate.statement,
        scope=candidate.scope,
        source_refs=candidate.source_refs,
        source_bundle_hash=bundle.bundle_hash,
        valid_from_sha=candidate.valid_from_sha,
        validity_policies=candidate.validity_policies,
        confidence=candidate.confidence,
        sensitivity=candidate.sensitivity,
        policy_effect=candidate.policy_effect,
        approved_by="amy",
        approval_event_id=stable_event_id("approve", candidate.candidate_id, request_id),
        status=RecordStatus.ACTIVE,
        created_at=CREATED_AT,
    )


def _record_v2(
    candidate: MemoryCandidate,
    bundle: SourceBundleDescriptor,
    request_id: str,
    *,
    expiry_conditions: Optional[Tuple[ExpiryCondition, ...]] = None,
) -> DurableMemoryRecord:
    conditions = expiry_conditions or (
        ExpiryCondition(
            condition_kind=ExpiryConditionKind.AT_TIME,
            value="2027-01-01T00:00:00Z",
        ),
    )
    return replace(
        _record(candidate, bundle, request_id),
        schema_version=2,
        expiry_conditions=conditions,
    )


def _feedback() -> FeedbackRecord:
    finding = FindingSnapshot(
        finding_id="F-" + "5" * 32,
        claim="Rounding can lose cents.",
        path="payments/money.py",
        line=42,
        contracts=("numeric_correctness",),
        original_severity=FindingSeverity.HIGH,
        evidence_refs=("O-" + "6" * 32,),
    )
    return FeedbackRecord(
        repository_key=REPOSITORY_KEY,
        review_id="review-001",
        finding_id=finding.finding_id,
        head_sha=SHA_A,
        finding_snapshot=finding,
        decision=FeedbackDecision.ACCEPTED,
        original_severity=FindingSeverity.HIGH,
        final_severity=FindingSeverity.HIGH,
        reason_code=FeedbackReasonCode.OTHER,
        reason="Confirmed by maintainer.",
        actor="amy",
        source_refs=(_human_source(),),
        status=FeedbackStatus.RECORDED,
        created_at=CREATED_AT,
    )


def _knowledge(content: bytes) -> RepositoryKnowledgeEntry:
    key = RepositoryKnowledgeKey(
        repository_key=REPOSITORY_KEY,
        revision_binding="head@" + SHA_A,
        capability=RepositoryKnowledgeCapability.SYMBOL_INDEX,
        analyzer_name="python-ast",
        analyzer_version="3.12-v1",
        configuration_digest=HASH_1,
        input_digest=HASH_2,
    )
    return RepositoryKnowledgeEntry(
        key=key,
        blob_hash=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        content_type="application/json",
        artifact_schema="symbol_index_v1",
        created_at=CREATED_AT,
        pinned_by_review_ids=("review-001",),
    )


def _downgrade_store_to_frozen_v1(store: MemoryStore) -> None:
    store.checkpoint("TRUNCATE")
    connection = sqlite3.connect(str(store.database_path), isolation_level=None)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("BEGIN IMMEDIATE")
        for trigger in (
            "candidate_authority_receipts_no_update",
            "candidate_authority_receipts_no_delete",
            "outbox_receipts_no_update",
            "outbox_receipts_no_delete",
        ):
            connection.execute("DROP TRIGGER IF EXISTS " + trigger)
        connection.execute("DROP TABLE candidate_authority_receipts")
        connection.execute("ALTER TABLE outbox_receipts RENAME TO outbox_receipts_v2")
        # Frozen hand-written v1 shape: this compatibility fixture must not be
        # regenerated from the current serializer/schema builder.
        connection.execute(
            """
            CREATE TABLE outbox_receipts (
                request_id TEXT PRIMARY KEY,
                repository_key TEXT NOT NULL
                    REFERENCES repositories(repository_key) ON DELETE CASCADE,
                operation TEXT NOT NULL,
                request_hash TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                event_id TEXT REFERENCES events(event_id) ON DELETE RESTRICT,
                result_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                CHECK (length(request_hash) = 64)
            )
            """
        )
        connection.execute(
            """
            INSERT INTO outbox_receipts(
                request_id, repository_key, operation, request_hash,
                subject_id, event_id, result_json, created_at
            )
            SELECT request_id, repository_key, operation, request_hash,
                   subject_id, event_id, result_json, created_at
            FROM outbox_receipts_v2
            """
        )
        connection.execute("DROP TABLE outbox_receipts_v2")
        connection.executemany(
            "UPDATE metadata SET value = ? WHERE key = ?",
            (
                ("memory_store_schema_v1", "schema_name"),
                ("1", "schema_version"),
                (
                    memory_store_module._V1_SCHEMA_DEFINITION_FINGERPRINT,
                    "schema_definition_hash",
                ),
            ),
        )
        connection.execute("DELETE FROM schema_migrations")
        connection.execute(
            """
            INSERT INTO schema_migrations(
                schema_version, schema_name, definition_hash, applied_at
            ) VALUES (1, 'memory_store_schema_v1', ?, ?)
            """,
            (memory_store_module._V1_SCHEMA_DEFINITION_FINGERPRINT, CREATED_AT),
        )
        connection.execute("PRAGMA user_version = 1")
        connection.commit()
        assert memory_store_module._schema_object_digest(connection) == (
            memory_store_module._V1_SCHEMA_OBJECT_DIGEST
        )
    finally:
        connection.close()


def _table_snapshot(database_path: Path, table: str) -> tuple[tuple[object, ...], ...]:
    connection = sqlite3.connect(str(database_path))
    try:
        return tuple(tuple(row) for row in connection.execute("SELECT * FROM " + table))
    finally:
        connection.close()


def test_open_creates_final_schema_once_and_enables_required_pragmas(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "namespace", busy_timeout_ms=275)
    first_metadata = store.metadata()
    reopened = MemoryStore(tmp_path / "namespace", busy_timeout_ms=275)

    assert first_metadata == reopened.metadata()
    assert first_metadata["schema_name"] == STORE_SCHEMA_NAME
    assert first_metadata["schema_version"] == str(STORE_SCHEMA_VERSION)
    with store._maintenance_connection() as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].casefold() == "wal"
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 275
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert {
        "metadata",
        "repositories",
        "generations",
        "blobs",
        "knowledge_entries",
        "candidates",
        "records",
        "events",
        "feedback",
        "source_bundles",
        "outbox_receipts",
    }.issubset(tables)

    with store._maintenance_connection() as connection:
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO repositories(
                    repository_key, created_at, last_accessed_at
                ) VALUES ('not-a-canonical-key', ?, ?)
                """,
                (CREATED_AT, CREATED_AT),
            )

    with store.open_connection() as connection:
        assert connection.execute("PRAGMA query_only").fetchone()[0] == 1
    with pytest.raises(MemoryStoreReadOnlyError, match="raw writable"):
        with store.open_connection(read_only=False):
            pass


def test_linked_worktrees_share_registered_identity_and_store_state(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    linked_worktree = tmp_path / "linked-worktree"
    run_git(
        git_repo,
        "worktree",
        "add",
        "-b",
        "memory-store-linked-worktree",
        str(linked_worktree),
        "HEAD",
    )
    resolver = RevisionResolver()
    memory_root = tmp_path / "memory-root"
    primary_namespace = materialize_repository_memory_namespace(
        plan_repository_memory_namespace(
            resolver.repository_identity(git_repo),
            memory_root,
            revision_resolver=resolver,
        ),
        revision_resolver=resolver,
    )
    linked_namespace = materialize_repository_memory_namespace(
        plan_repository_memory_namespace(
            resolver.repository_identity(linked_worktree),
            memory_root,
            revision_resolver=resolver,
        ),
        revision_resolver=resolver,
    )

    assert primary_namespace.namespace_path == linked_namespace.namespace_path
    assert primary_namespace.metadata.core == linked_namespace.metadata.core
    assert (
        primary_namespace.metadata.canonical_path
        != linked_namespace.metadata.canonical_path
    )

    primary_store = MemoryStore(primary_namespace)
    linked_store = MemoryStore(linked_namespace)
    candidate = replace(
        _candidate(),
        repository_key=primary_namespace.repository_key,
    )
    result = linked_store.put_candidate(
        candidate,
        request_id=stable_request_id("linked-worktree", candidate.candidate_id),
    )

    assert result.applied
    assert primary_store.get_candidate(candidate.candidate_id) == candidate
    assert (
        primary_store.get_generations(primary_namespace.repository_key)
        == linked_store.get_generations(linked_namespace.repository_key)
    )
    assert primary_store.verify_event_chain(primary_namespace.repository_key) == 1
    assert linked_store.verify_event_chain(linked_namespace.repository_key) == 1

    primary_read_only = MemoryStore(primary_namespace, read_only=True)
    linked_read_only = MemoryStore(linked_namespace, read_only=True)
    assert primary_read_only.get_candidate(candidate.candidate_id) == candidate
    assert linked_read_only.get_candidate(candidate.candidate_id) == candidate


def test_store_revalidates_a_namespace_after_a_symlink_swap(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    resolver = RevisionResolver()
    memory_root = tmp_path / "stale-namespace-root"
    plan = plan_repository_memory_namespace(
        resolver.repository_identity(git_repo),
        memory_root,
        revision_resolver=resolver,
    )
    namespace = plan.namespace
    memory_root.mkdir()
    outside = tmp_path / "stale-namespace-outside"
    outside.mkdir()
    repositories = memory_root / "repositories"
    created_junction = False
    try:
        repositories.symlink_to(
            outside,
            target_is_directory=True,
        )
    except OSError as error:
        if os.name != "nt":
            pytest.skip(f"directory symlinks unavailable: {error}")
        result = subprocess.run(
            [
                "cmd.exe",
                "/d",
                "/c",
                "mklink",
                "/J",
                str(repositories),
                str(outside),
            ],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        if result.returncode != 0:
            pytest.skip("Windows junction creation is unavailable")
        created_junction = True

    try:
        with pytest.raises(
            ValueError,
            match="symbolic link or reparse point",
        ):
            MemoryStore(namespace)

        assert list(outside.iterdir()) == []
    finally:
        if created_junction:
            repositories.rmdir()
        else:
            repositories.unlink()


def test_unknown_schema_fails_closed_without_rewriting_database(tmp_path: Path) -> None:
    namespace = tmp_path / "future"
    namespace.mkdir()
    database = namespace / "memory.sqlite3"
    connection = sqlite3.connect(str(database))
    connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    connection.executemany(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        (("schema_name", "memory_store_schema_v99"), ("schema_version", "99")),
    )
    connection.execute("PRAGMA user_version = 99")
    connection.commit()
    connection.close()
    before = database.read_bytes()

    with pytest.raises(MemoryStoreSchemaError) as raised:
        MemoryStore(namespace)

    assert raised.value.code.value == "unsupported_schema"
    assert database.read_bytes() == before


def test_live_schema_definition_cannot_hide_behind_valid_metadata(
    tmp_path: Path,
) -> None:
    namespace = tmp_path / "memory"
    store = MemoryStore(namespace)
    with store._maintenance_connection() as connection:
        connection.execute("DROP TRIGGER events_no_update")
        connection.execute(
            """
            CREATE TRIGGER events_no_update BEFORE UPDATE ON events
            BEGIN SELECT 1; END
            """
        )

    with pytest.raises(MemoryStoreCorruptionError, match="live schema"):
        MemoryStore(namespace)


def test_candidate_write_generation_event_and_request_replay_are_atomic(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    candidate = _candidate()
    request_id = stable_request_id("candidate", candidate.candidate_id)

    first = store.put_candidate(candidate, request_id=request_id)
    replay = store.put_candidate(candidate, request_id=request_id)

    assert first.applied and not first.replayed
    assert not replay.applied and replay.replayed
    assert store.get_candidate(candidate.candidate_id) == candidate
    assert store.get_generations(REPOSITORY_KEY).memory_generation == 1
    assert len(store.list_events(REPOSITORY_KEY)) == 1

    conflicting = _candidate(statement="Amounts must use exact integer cents.")
    with pytest.raises(MemoryStoreConflictError):
        store.put_candidate(conflicting, request_id=request_id)
    assert store.get_generations(REPOSITORY_KEY).memory_generation == 1


def test_transition_request_replay_ignores_runtime_timestamp_and_stale_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    candidate = _candidate()
    receipt = _authority_receipt(candidate)
    store.put_candidate(
        candidate,
        receipt,
        request_id=stable_request_id("candidate", candidate.candidate_id),
    )
    request_id = stable_request_id("validate", candidate.candidate_id)
    timestamps = iter(
        (
            "2026-07-14T12:00:01Z",
            "2026-07-14T12:00:02Z",
            "2026-07-14T12:00:03Z",
        )
    )
    monkeypatch.setattr(memory_store_module, "_utc_now", lambda: next(timestamps))

    first = store.transition_candidate(
        candidate.candidate_id,
        expected_status=CandidateStatus.PROPOSED,
        new_status=CandidateStatus.VALIDATED,
        action="candidate_validated",
        actor_type="runtime",
        actor_id="memory_sources",
        reason_code="sources_valid",
        request_id=request_id,
        expected_generation=1,
    )
    replay = store.transition_candidate(
        candidate.candidate_id,
        expected_status=CandidateStatus.PROPOSED,
        new_status=CandidateStatus.VALIDATED,
        action="candidate_validated",
        actor_type="runtime",
        actor_id="memory_sources",
        reason_code="sources_valid",
        request_id=request_id,
        expected_generation=999,
    )

    assert first.applied and replay.replayed and not replay.applied
    assert replay.event_id == first.event_id
    assert store.get_generations(REPOSITORY_KEY).memory_generation == 2
    with pytest.raises(MemoryStoreConflictError, match="reused"):
        store.transition_candidate(
            candidate.candidate_id,
            expected_status=CandidateStatus.PROPOSED,
            new_status=CandidateStatus.VALIDATED,
            action="candidate_validated",
            actor_type="runtime",
            actor_id="memory_sources",
            reason_code="different_semantics",
            request_id=request_id,
            expected_generation=2,
        )


def test_candidate_authority_receipts_are_atomic_multi_context_and_immutable(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    candidate = _candidate()
    first = _authority_receipt(candidate)
    first_request = stable_request_id("candidate-authority", first.receipt_id)

    written = store.put_candidate(
        candidate,
        first,
        request_id=first_request,
    )
    replay = store.put_candidate(
        candidate,
        first,
        request_id=first_request,
        expected_generation=0,
    )
    assert written.applied and replay.replayed
    assert store.get_generations(REPOSITORY_KEY).memory_generation == 1

    second = _authority_receipt(
        candidate,
        authority_resolution_hash="9" * 64,
    )
    added = store.put_candidate(
        candidate,
        second,
        request_id=stable_request_id("candidate-authority", second.receipt_id),
        expected_generation=1,
    )
    assert added.applied and added.event_id is None
    assert store.get_generations(REPOSITORY_KEY).memory_generation == 1
    assert store.list_candidate_authority_receipts(candidate.candidate_id) == (
        first,
        second,
    )
    with pytest.raises(MemoryStoreConflictError, match="multiple authority"):
        store.get_candidate_authority_receipt(candidate.candidate_id)
    assert store.select_candidate_authority_receipt(
        candidate.candidate_id,
        authority_resolution_hash=second.authority_resolution_hash,
    ) == second

    conflicting = replace(second, created_at="2026-07-14T12:00:01Z")
    with pytest.raises(MemoryStoreConflictError, match="different immutable"):
        store.put_candidate(
            candidate,
            conflicting,
            request_id=stable_request_id("authority-conflict", conflicting.receipt_id),
        )
    with pytest.raises(MemoryStoreValidationError, match="does not match"):
        store.put_candidate(
            candidate,
            _authority_receipt(_candidate(statement="Different candidate.")),
            request_id=stable_request_id("authority-mismatch", candidate.candidate_id),
        )

    with pytest.raises(MemoryStoreConflictError):
        with store._maintenance_connection() as connection:
            connection.execute(
                "UPDATE candidate_authority_receipts SET created_at = ? "
                "WHERE receipt_id = ?",
                ("2026-07-14T12:00:02Z", first.receipt_id),
            )
    with pytest.raises(MemoryStoreConflictError):
        with store._maintenance_connection() as connection:
            connection.execute(
                "DELETE FROM candidate_authority_receipts WHERE receipt_id = ?",
                (first.receipt_id,),
            )
    store.validate_integrity()


def test_candidate_authority_empty_set_guard_is_atomic(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    recoverable = _candidate(statement="Recover only from an empty receipt set.")
    store.put_candidate(
        recoverable,
        request_id=stable_request_id("legacy-candidate", recoverable.candidate_id),
    )
    recovered_receipt = _authority_receipt(recoverable)
    recovery_request = stable_request_id(
        "legacy-authority-recovery",
        recovered_receipt.receipt_id,
    )
    recovered = store.put_candidate(
        recoverable,
        recovered_receipt,
        request_id=recovery_request,
        require_no_authority_receipts=True,
    )
    replayed = store.put_candidate(
        recoverable,
        recovered_receipt,
        request_id=recovery_request,
        require_no_authority_receipts=True,
    )

    assert recovered.applied and not recovered.replayed
    assert replayed.replayed and not replayed.applied
    assert store.list_candidate_authority_receipts(recoverable.candidate_id) == (
        recovered_receipt,
    )

    competing = _candidate(statement="Reject recovery after a competing receipt.")
    store.put_candidate(
        competing,
        request_id=stable_request_id("legacy-candidate", competing.candidate_id),
    )
    competing_receipt = _authority_receipt(
        competing,
        authority_resolution_hash="8" * 64,
    )
    store.put_candidate(
        competing,
        competing_receipt,
        request_id=stable_request_id(
            "competing-authority",
            competing_receipt.receipt_id,
        ),
    )
    attempted = _authority_receipt(competing)

    with pytest.raises(MemoryStoreConflictError, match="empty receipt set"):
        store.put_candidate(
            competing,
            attempted,
            request_id=stable_request_id(
                "legacy-authority-recovery",
                attempted.receipt_id,
            ),
            require_no_authority_receipts=True,
        )
    assert store.list_candidate_authority_receipts(competing.candidate_id) == (
        competing_receipt,
    )


def test_candidate_and_authority_receipt_insert_roll_back_together(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    candidate = _candidate(statement="Receipt insertion must be atomic.")
    receipt = _authority_receipt(candidate)
    with store._maintenance_connection() as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_authority_receipt
            BEFORE INSERT ON candidate_authority_receipts
            BEGIN SELECT RAISE(ABORT, 'injected receipt failure'); END
            """
        )

    with pytest.raises(MemoryStoreConflictError):
        store.put_candidate(
            candidate,
            receipt,
            request_id=stable_request_id("atomic-authority", candidate.candidate_id),
        )
    assert store.find_candidate(candidate.candidate_id) is None
    assert store.list_candidate_authority_receipts(candidate.candidate_id) == ()
    assert store.get_generations(REPOSITORY_KEY).memory_generation == 0
    assert store.list_events(REPOSITORY_KEY) == ()


def test_failed_event_insert_rolls_back_projection_and_generation(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory")
    candidate = _candidate()
    with store._maintenance_connection() as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_test_events BEFORE INSERT ON events
            BEGIN SELECT RAISE(ABORT, 'injected failure'); END
            """
        )

    with pytest.raises(MemoryStoreConflictError):
        store.put_candidate(
            candidate,
            request_id=stable_request_id("rollback", candidate.candidate_id),
        )

    assert store.find_candidate(candidate.candidate_id) is None
    assert store.get_generations(REPOSITORY_KEY).memory_generation == 0


def test_find_request_receipt_operation_is_canonical_and_read_only(
    tmp_path: Path,
) -> None:
    namespace = tmp_path / "memory"
    writer = MemoryStore(namespace)
    candidate = _candidate()
    request_id = stable_request_id("receipt-operation", candidate.candidate_id)
    writer.put_candidate(candidate, request_id=request_id)
    writer.checkpoint("TRUNCATE")
    before = writer.database_path.read_bytes()

    read_only = MemoryStore(namespace, read_only=True)
    assert read_only.find_request_receipt_operation(request_id) == "put_candidate"
    assert (
        read_only.find_request_receipt_operation(
            stable_request_id("missing-receipt-operation", candidate.candidate_id)
        )
        is None
    )
    with pytest.raises(MemoryStoreValidationError, match="request_id"):
        read_only.find_request_receipt_operation("not-a-canonical-request-id")
    assert writer.database_path.read_bytes() == before

    with writer._maintenance_connection() as connection:
        connection.execute("DROP TRIGGER outbox_receipts_no_update")
        connection.execute(
            "UPDATE outbox_receipts SET operation = ? WHERE request_id = ?",
            (" put_candidate", request_id),
        )
        connection.execute(
            """
            CREATE TRIGGER outbox_receipts_no_update
            BEFORE UPDATE ON outbox_receipts
            BEGIN
                SELECT RAISE(ABORT, 'request receipts are immutable');
            END
            """
        )

    with pytest.raises(MemoryStoreCorruptionError, match="request receipt"):
        read_only.find_request_receipt_operation(request_id)


def test_approval_uses_generation_and_status_cas_without_duplicate_record(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    candidate = _candidate(status=CandidateStatus.VALIDATED)
    store.put_candidate(
        candidate,
        _authority_receipt(candidate),
        request_id=stable_request_id("candidate", candidate.candidate_id),
    )
    content = b'{"sources":["payments/money.py:10-18"]}'
    store.put_blob(
        content,
        media_type="application/vnd.review-agent.source-bundle+json",
    )
    bundle = _bundle(candidate, content)
    store.put_source_bundle(
        bundle,
        request_id=stable_request_id("bundle", bundle.bundle_hash),
        authority_resolution_hash=AUTHORITY_RESOLUTION,
    )
    request_id = stable_request_id("approve", candidate.candidate_id)
    record = _record(candidate, bundle, request_id)
    expected_generation = store.get_generations(REPOSITORY_KEY).memory_generation

    with pytest.raises(MemoryStoreValidationError, match="active"):
        store.approve_candidate(
            replace(record, status=RecordStatus.REVOKED),
            request_id=stable_request_id(
                "invalid-record-status", candidate.candidate_id
            ),
            expected_candidate_status=CandidateStatus.VALIDATED,
            expected_generation=expected_generation,
            authority_resolution_hash=AUTHORITY_RESOLUTION,
        )
    assert store.get_generations(REPOSITORY_KEY).memory_generation == expected_generation

    result = store.approve_candidate(
        record,
        request_id=request_id,
        expected_candidate_status=CandidateStatus.VALIDATED,
        expected_generation=expected_generation,
        authority_resolution_hash=AUTHORITY_RESOLUTION,
    )

    assert result.applied
    assert store.get_candidate(candidate.candidate_id).status is CandidateStatus.APPROVED
    assert store.get_record(record.memory_id) == record
    assert store.count_records(REPOSITORY_KEY) == 1

    with pytest.raises(MemoryStoreConflictError):
        store.approve_candidate(
            record,
            request_id=stable_request_id("second-approval", candidate.candidate_id),
            expected_candidate_status=CandidateStatus.VALIDATED,
            authority_resolution_hash=AUTHORITY_RESOLUTION,
        )
    assert store.count_records(REPOSITORY_KEY) == 1

    tampered_record = replace(record, statement="Amounts must use integer cents.")
    tampered_json = canonical_json(tampered_record.to_dict())
    with store._maintenance_connection() as connection:
        connection.execute("DROP TRIGGER records_body_immutable")
        connection.execute(
            "UPDATE records SET model_json = ?, body_hash = ? WHERE memory_id = ?",
            (
                tampered_json,
                hashlib.sha256(tampered_json.encode("utf-8")).hexdigest(),
                record.memory_id,
            ),
        )
    with pytest.raises(MemoryStoreCorruptionError):
        store.get_record(record.memory_id)


def test_atomic_lifecycle_approval_rolls_back_bundle_pin_and_authority(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    candidate = _candidate(status=CandidateStatus.PENDING_APPROVAL)
    store.put_candidate(
        candidate,
        _authority_receipt(candidate),
        request_id=stable_request_id("candidate", candidate.candidate_id),
    )
    content = b'{"schema":"memory_source_bundle_v1","sources":[]}'
    blob = store.put_blob(
        content,
        media_type="application/vnd.review-agent.source-bundle+json",
        created_at=CREATED_AT,
    )
    bundle = SourceBundleDescriptor(
        repository_key=candidate.repository_key,
        candidate_id=candidate.candidate_id,
        source_refs=candidate.source_refs,
        blob_hash=blob.blob_hash,
        size_bytes=blob.size_bytes,
        media_type=blob.media_type,
        created_at=CREATED_AT,
    )
    request_id = stable_request_id("atomic-approve", candidate.candidate_id)
    record = _record(candidate, bundle, request_id)
    generation = store.get_generations(REPOSITORY_KEY).memory_generation

    with store._maintenance_connection() as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_atomic_record BEFORE INSERT ON records
            BEGIN SELECT RAISE(ABORT, 'injected record failure'); END
            """
        )

    with pytest.raises(MemoryStoreConflictError):
        store.approve_candidate_with_source_bundle(
            record,
            bundle,
            request_id=request_id,
            expected_candidate_status=CandidateStatus.PENDING_APPROVAL,
            expected_generation=generation,
            authority_resolution_hash=AUTHORITY_RESOLUTION,
            actor_id="amy",
            reason="Approved after source revalidation.",
        )

    assert store.get_candidate(candidate.candidate_id).status is CandidateStatus.PENDING_APPROVAL
    assert store.find_source_bundle(bundle.bundle_hash) is None
    assert store.count_records(REPOSITORY_KEY) == 0
    assert store.get_generations(REPOSITORY_KEY).memory_generation == generation
    with store.open_connection(read_only=True) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM blob_pins WHERE pin_type = 'source_bundle'"
        ).fetchone()[0] == 0

    with store._maintenance_connection() as connection:
        connection.execute("DROP TRIGGER reject_atomic_record")
    result = store.approve_candidate_with_source_bundle(
        record,
        bundle,
        request_id=request_id,
        expected_candidate_status=CandidateStatus.PENDING_APPROVAL,
        expected_generation=generation,
        authority_resolution_hash=AUTHORITY_RESOLUTION,
        actor_id="amy",
        reason="Approved after source revalidation.",
    )

    assert result.applied
    assert store.get_candidate(candidate.candidate_id).status is CandidateStatus.APPROVED
    assert store.get_record(record.memory_id) == record
    assert store.get_source_bundle(bundle.bundle_hash) == bundle
    with store.open_connection(read_only=True) as connection:
        pin = connection.execute(
            """
            SELECT pin_type, pin_id FROM blob_pins
            WHERE blob_hash = ?
            """,
            (blob.blob_hash,),
        ).fetchone()
    assert tuple(pin) == ("source_bundle", bundle.bundle_hash)


def test_v2_record_expiry_is_approved_audited_and_projection_only(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    candidate = _candidate(status=CandidateStatus.PENDING_APPROVAL)
    store.put_candidate(
        candidate,
        _authority_receipt(candidate),
        request_id=stable_request_id("expiry-candidate", candidate.candidate_id),
    )
    content = b'{"schema":"memory_source_bundle_v1","sources":[]}'
    store.put_blob(
        content,
        media_type="application/vnd.review-agent.source-bundle+json",
        created_at=CREATED_AT,
    )
    bundle = _bundle(candidate, content)
    approval_request = stable_request_id("expiry-approval", candidate.candidate_id)
    record = _record_v2(
        candidate,
        bundle,
        approval_request,
        expiry_conditions=(
            ExpiryCondition(
                condition_kind=ExpiryConditionKind.AT_TIME,
                value="2027-01-01T00:00:00Z",
            ),
            ExpiryCondition(
                condition_kind=ExpiryConditionKind.AT_COMMIT,
                value="b" * 40,
            ),
        ),
    )

    assert "expiry_conditions" not in candidate.to_dict()
    assert record.memory_id == _record(candidate, bundle, approval_request).memory_id
    result = store.approve_candidate_with_source_bundle(
        record,
        bundle,
        request_id=approval_request,
        expected_candidate_status=CandidateStatus.PENDING_APPROVAL,
        authority_resolution_hash=AUTHORITY_RESOLUTION,
    )

    assert result.applied
    assert store.get_record(record.memory_id) == record
    assert store.list_records(REPOSITORY_KEY) == (record,)
    assert store.read_view(REPOSITORY_KEY).records == (record,)
    assert store.verify_event_chain(REPOSITORY_KEY) == 3
    assert store.validate_integrity().record_count == 1
    with store.open_connection() as connection:
        before = connection.execute(
            "SELECT model_json, body_hash FROM records WHERE memory_id = ?",
            (record.memory_id,),
        ).fetchone()

    transition_request = stable_request_id("expire-record", record.memory_id)
    store.transition_record(
        record.memory_id,
        expected_status=RecordStatus.ACTIVE,
        new_status=RecordStatus.EXPIRED,
        action="expire",
        actor_type="runtime",
        actor_id="memory_lifecycle",
        reason_code="expiry_condition_matched",
        request_id=transition_request,
        created_at="2027-01-01T00:00:00Z",
    )

    expired = replace(record, status=RecordStatus.EXPIRED)
    assert store.get_record(record.memory_id) == expired
    assert store.read_view(REPOSITORY_KEY).records == (expired,)
    with store.open_connection() as connection:
        after = connection.execute(
            "SELECT model_json, body_hash FROM records WHERE memory_id = ?",
            (record.memory_id,),
        ).fetchone()
    assert tuple(after) == tuple(before)
    assert json.loads(after["model_json"])["status"] == RecordStatus.ACTIVE.value
    assert json.loads(after["model_json"])["expiry_conditions"] == [
        condition.to_dict() for condition in record.expiry_conditions
    ]
    assert store.verify_event_chain(REPOSITORY_KEY) == 4
    assert store.validate_integrity().record_count == 1

    changed_expiry = _record_v2(
        candidate,
        bundle,
        approval_request,
        expiry_conditions=(
            ExpiryCondition(
                condition_kind=ExpiryConditionKind.AT_TIME,
                value="2028-01-01T00:00:00Z",
            ),
        ),
    )
    assert changed_expiry.memory_id == record.memory_id
    with pytest.raises(MemoryStoreConflictError):
        store.approve_candidate_with_source_bundle(
            changed_expiry,
            bundle,
            request_id=stable_request_id(
                "duplicate-expiry-approval",
                candidate.candidate_id,
            ),
            expected_candidate_status=CandidateStatus.PENDING_APPROVAL,
            authority_resolution_hash=AUTHORITY_RESOLUTION,
        )
    assert store.count_records(REPOSITORY_KEY) == 1


@pytest.mark.parametrize("tamper", ["expiry_condition", "body_hash"])
def test_v2_record_expiry_body_tampering_fails_closed(
    tmp_path: Path,
    tamper: str,
) -> None:
    store = MemoryStore(tmp_path / tamper)
    candidate = _candidate(status=CandidateStatus.PENDING_APPROVAL)
    store.put_candidate(
        candidate,
        _authority_receipt(candidate),
        request_id=stable_request_id("tamper-candidate", candidate.candidate_id),
    )
    content = b'{"schema":"memory_source_bundle_v1","sources":[]}'
    store.put_blob(
        content,
        media_type="application/vnd.review-agent.source-bundle+json",
        created_at=CREATED_AT,
    )
    bundle = _bundle(candidate, content)
    approval_request = stable_request_id("tamper-approval", candidate.candidate_id)
    record = _record_v2(candidate, bundle, approval_request)
    store.approve_candidate_with_source_bundle(
        record,
        bundle,
        request_id=approval_request,
        expected_candidate_status=CandidateStatus.PENDING_APPROVAL,
        authority_resolution_hash=AUTHORITY_RESOLUTION,
    )

    with store._maintenance_connection() as connection:
        connection.execute("DROP TRIGGER records_body_immutable")
        if tamper == "body_hash":
            connection.execute(
                "UPDATE records SET body_hash = ? WHERE memory_id = ?",
                (HASH_1, record.memory_id),
            )
        else:
            payload = record.to_dict()
            payload["expiry_conditions"] = [
                ExpiryCondition(
                    condition_kind=ExpiryConditionKind.AT_TIME,
                    value="2028-01-01T00:00:00Z",
                ).to_dict()
            ]
            connection.execute(
                "UPDATE records SET model_json = ? WHERE memory_id = ?",
                (canonical_json(payload), record.memory_id),
            )
        connection.execute(
            """
            CREATE TRIGGER records_body_immutable
            BEFORE UPDATE OF memory_id, candidate_id, repository_key,
                             source_bundle_hash, model_json, body_hash, created_at
            ON records
            BEGIN
                SELECT RAISE(ABORT, 'record bodies are immutable');
            END
            """
        )

    with pytest.raises(MemoryStoreCorruptionError, match="canonical memory row"):
        store.get_record(record.memory_id)
    with pytest.raises(MemoryStoreCorruptionError, match="canonical memory row"):
        store.validate_integrity(validate_blob_files=False)


@pytest.mark.parametrize(
    "tamper_sql",
    [
        "UPDATE events SET action = 'tampered' WHERE sequence = 1",
        "DELETE FROM events WHERE sequence = 1",
        "UPDATE events SET sequence = sequence + 100 WHERE sequence = 1",
    ],
)
def test_event_hash_chain_detects_tamper_delete_and_reorder(
    tmp_path: Path,
    tamper_sql: str,
) -> None:
    store = MemoryStore(tmp_path / hashlib.sha256(tamper_sql.encode()).hexdigest())
    candidate = _candidate()
    store.put_candidate(
        candidate,
        request_id=stable_request_id("event-chain", tamper_sql),
    )
    assert store.verify_event_chain(REPOSITORY_KEY) == 1
    with store._maintenance_connection() as connection:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute("DROP TRIGGER events_no_update")
        connection.execute("DROP TRIGGER events_no_delete")
        connection.execute(tamper_sql)

    with pytest.raises(MemoryStoreCorruptionError):
        store.verify_event_chain(REPOSITORY_KEY)


def test_validated_read_view_rejects_a_tampered_event_chain(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory")
    candidate = _candidate()
    store.put_candidate(
        candidate,
        request_id=stable_request_id("read-view-chain", candidate.candidate_id),
    )
    with store._maintenance_connection() as connection:
        connection.execute("DROP TRIGGER events_no_update")
        connection.execute("UPDATE events SET action = 'tampered' WHERE sequence = 1")

    with pytest.raises(MemoryStoreCorruptionError):
        store.read_view(REPOSITORY_KEY)


@pytest.mark.parametrize(
    "tamper_sql",
    [
        "UPDATE candidates SET current_status = 'validated'",
        "UPDATE candidates SET generation = generation + 1",
    ],
)
def test_validated_reads_reject_projection_only_tampering(
    tmp_path: Path,
    tamper_sql: str,
) -> None:
    store = MemoryStore(tmp_path / hashlib.sha256(tamper_sql.encode()).hexdigest())
    candidate = _candidate()
    store.put_candidate(
        candidate,
        request_id=stable_request_id("projection-tamper", tamper_sql),
    )
    with store._maintenance_connection() as connection:
        connection.execute(tamper_sql)

    # The append-only event bytes were not touched; the projection/event join
    # is what must detect this authority split.
    assert store.verify_event_chain(REPOSITORY_KEY) == 1
    with pytest.raises(MemoryStoreCorruptionError, match="projection"):
        store.read_view(REPOSITORY_KEY)
    with pytest.raises(MemoryStoreCorruptionError, match="projection"):
        store.validate_integrity()


def test_blob_hash_size_validation_atomic_promotion_and_fail_closed_read(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    content = b"canonical repository knowledge"
    digest = hashlib.sha256(content).hexdigest()

    with pytest.raises(MemoryStoreValidationError):
        store.put_blob(content, media_type="application/octet-stream", expected_hash=HASH_1)
    assert not store.blob_path(digest).exists()

    blob = store.put_blob(
        content,
        media_type="application/octet-stream",
        expected_hash=digest,
        expected_size=len(content),
    )
    assert blob.blob_hash == digest
    assert Path(blob.path).read_bytes() == content
    assert store.read_blob(digest) == content

    Path(blob.path).write_bytes(b"tampered")
    with pytest.raises(MemoryStoreCorruptionError):
        store.read_blob(digest)


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_blob_repair_is_serialized_store_owned_and_preserves_metadata(
    tmp_path: Path,
    damage: str,
) -> None:
    namespace = tmp_path / "memory"
    store = MemoryStore(namespace)
    content = b"canonical repository knowledge"
    blob = store.put_blob(
        content,
        media_type="application/octet-stream",
        created_at=CREATED_AT,
    )
    path = Path(blob.path)
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"tampered")

    repaired = store.repair_blob(
        content,
        media_type=blob.media_type,
        expected_hash=blob.blob_hash,
        expected_size=blob.size_bytes,
    )

    assert repaired == blob
    assert store.read_blob(blob.blob_hash) == content
    with pytest.raises(MemoryStoreCorruptionError, match="metadata"):
        store.repair_blob(content, media_type="application/json")
    assert store.read_blob(blob.blob_hash) == content

    with pytest.raises(MemoryStoreReadOnlyError):
        MemoryStore(namespace, read_only=True).repair_blob(
            content,
            media_type=blob.media_type,
        )


def test_blob_atomic_staging_uses_bounded_random_components(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    content = b"bounded temporary blob name"
    digest = hashlib.sha256(content).hexdigest()
    opened_staging_names: list[str] = []
    real_open = memory_store_module.os.open

    def tracking_open(*args, **kwargs):
        path = Path(args[0])
        if path.parent.name == ".tmp":
            opened_staging_names.append(path.name)
        return real_open(*args, **kwargs)

    monkeypatch.setattr(memory_store_module.os, "open", tracking_open)
    blob = store.put_blob(content, media_type="application/octet-stream")
    Path(blob.path).write_bytes(b"corrupt")
    store.repair_blob(
        content,
        media_type="application/octet-stream",
        expected_hash=digest,
        expected_size=len(content),
    )

    assert len(opened_staging_names) == 2
    assert all(digest not in name for name in opened_staging_names)
    assert all(len(name) <= 40 for name in opened_staging_names)


def test_read_only_store_never_promotes_a_blob_before_rejecting_write(
    tmp_path: Path,
) -> None:
    namespace = tmp_path / "memory"
    MemoryStore(namespace)
    store = MemoryStore(namespace, read_only=True)
    content = b"must not be written"
    digest = hashlib.sha256(content).hexdigest()

    with pytest.raises(MemoryStoreReadOnlyError):
        store.put_blob(content, media_type="application/octet-stream")

    assert not store.blob_path(digest).exists()


def test_read_only_blob_read_does_not_create_a_blob_lock_file(
    tmp_path: Path,
) -> None:
    namespace = tmp_path / "memory"
    writer = MemoryStore(namespace)
    content = b"read-only blob access must not mutate its namespace"
    blob = writer.put_blob(content, media_type="application/octet-stream")
    lock_path = namespace / "blobs" / ".blob-store.lock"
    lock_path.unlink()
    read_only = MemoryStore(namespace, read_only=True)
    before = {
        path.relative_to(namespace).as_posix(): path.stat().st_mtime_ns
        for path in namespace.rglob("*")
        if path.is_file()
    }

    assert read_only.read_blob(blob.blob_hash) == content

    after = {
        path.relative_to(namespace).as_posix(): path.stat().st_mtime_ns
        for path in namespace.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert not lock_path.exists()


def test_genuinely_read_only_store_reads_without_coordination_file_or_mutation(
    tmp_path: Path,
) -> None:
    namespace = tmp_path / "memory"
    writer = MemoryStore(namespace)
    candidate = _candidate(statement="A read-only filesystem remains auditable.")
    writer.put_candidate(
        candidate,
        request_id=stable_request_id("read-only-root", candidate.candidate_id),
    )
    writer.checkpoint("TRUNCATE")
    lock_path = namespace / ".memory-store.lock"
    lock_path.unlink()
    database_mode = writer.database_path.stat().st_mode
    namespace_mode = namespace.stat().st_mode
    writer.database_path.chmod(stat.S_IREAD)
    namespace.chmod(stat.S_IREAD | stat.S_IEXEC)
    before = {
        path.relative_to(namespace).as_posix(): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in namespace.rglob("*")
        if path.is_file()
    }
    try:
        read_only = MemoryStore(namespace, read_only=True)
        assert read_only.get_candidate(candidate.candidate_id) == candidate
        assert read_only.audit_status().read_only is True
        after = {
            path.relative_to(namespace).as_posix(): (
                path.stat().st_size,
                path.stat().st_mtime_ns,
            )
            for path in namespace.rglob("*")
            if path.is_file()
        }
        assert before == after
        assert not lock_path.exists()
    finally:
        namespace.chmod(namespace_mode)
        writer.database_path.chmod(database_mode)


def test_public_namespace_locks_are_ordered_reentrant_and_database_free(
    tmp_path: Path,
) -> None:
    later = tmp_path / "z-namespace"
    earlier = tmp_path / "a-namespace"

    with MemoryStore.lock_namespaces(later, earlier, later) as locked:
        assert locked == (earlier.resolve(), later.resolve())
        with MemoryStore.lock_namespaces(earlier, later) as nested:
            assert nested == locked
        assert all(
            (namespace / ".memory-store.lock").is_file()
            for namespace in locked
        )
        assert all(
            not (namespace / "memory.sqlite3").exists()
            for namespace in locked
        )


def test_public_namespace_lock_is_the_reentrant_authority_mutation_lock(
    tmp_path: Path,
) -> None:
    namespace = tmp_path / "memory"
    store = MemoryStore(namespace, busy_timeout_ms=80)
    reentrant = _candidate(statement="The owning thread can mutate reentrantly.")
    blocked = _candidate(statement="A different thread must wait for authority.")
    writer_started = threading.Event()
    writer_errors: list[Exception] = []

    def write_from_another_thread() -> None:
        writer_started.set()
        try:
            store.put_candidate(
                blocked,
                request_id=stable_request_id(
                    "blocked-by-public-namespace-lock",
                    blocked.candidate_id,
                ),
            )
        except Exception as error:  # pragma: no cover - asserted below
            writer_errors.append(error)

    with MemoryStore.lock_namespaces(namespace):
        store.put_candidate(
            reentrant,
            request_id=stable_request_id(
                "reentrant-public-namespace-lock",
                reentrant.candidate_id,
            ),
        )
        writer = threading.Thread(target=write_from_another_thread)
        writer.start()
        assert writer_started.wait(timeout=1)
        writer.join(timeout=2)

        assert not writer.is_alive()
        assert len(writer_errors) == 1
        assert isinstance(writer_errors[0], MemoryStoreBusyError)
        assert store.find_candidate(blocked.candidate_id) is None

    assert store.put_candidate(
        blocked,
        request_id=stable_request_id(
            "blocked-by-public-namespace-lock",
            blocked.candidate_id,
        ),
    ).applied


def test_namespace_empty_check_allows_only_the_coordination_lock(
    tmp_path: Path,
) -> None:
    namespace = tmp_path / "memory"

    assert MemoryStore.namespace_has_no_store_state(namespace)
    assert not namespace.exists()
    with MemoryStore.lock_namespaces(namespace):
        assert MemoryStore.namespace_has_no_store_state(namespace)
    assert MemoryStore.namespace_has_no_store_state(namespace)

    (namespace / "unexpected-state").write_text("state", encoding="ascii")
    assert not MemoryStore.namespace_has_no_store_state(namespace)


def test_repository_authority_state_token_binds_descriptor_and_event_head(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    resolver = RevisionResolver()
    namespace = materialize_repository_memory_namespace(
        plan_repository_memory_namespace(
            resolver.repository_identity(git_repo),
            tmp_path / "memory-root",
            revision_resolver=resolver,
        ),
        revision_resolver=resolver,
    )
    store = MemoryStore(namespace)

    snapshot = store.repository_authority_snapshot(namespace.repository_key)
    assert snapshot.repository_identity == namespace.metadata
    assert snapshot.generations == store.get_generations(namespace.repository_key)
    assert snapshot.event_count == 0
    assert snapshot.event_head_sequence is None
    assert snapshot.event_head_hash == "0" * 64
    assert snapshot.candidate_authority_receipt_count == 0
    assert store.get_repository_descriptor(namespace.repository_key) == namespace.metadata
    initial = snapshot.state_token
    assert initial == store.repository_authority_state_token(
        namespace.repository_key
    )

    # Blob housekeeping does not change logical repository authority.
    store.put_blob(b"unreferenced", media_type="application/octet-stream")
    assert store.repository_authority_state_token(namespace.repository_key) == initial

    candidate = replace(_candidate(), repository_key=namespace.repository_key)
    store.put_candidate(
        candidate,
        request_id=stable_request_id(
            "authority-state-token",
            candidate.candidate_id,
        ),
    )
    changed = store.repository_authority_state_token(namespace.repository_key)
    assert changed != initial
    after_candidate = store.repository_authority_snapshot(
        namespace.repository_key
    )
    assert after_candidate.state_token == changed
    assert after_candidate.candidate_authority_receipt_count == 0

    first_receipt = _authority_receipt(
        candidate,
        authority_resolution_hash="7" * 64,
    )
    store.put_candidate(
        candidate,
        first_receipt,
        request_id=stable_request_id(
            "authority-state-token-receipt-1",
            candidate.candidate_id,
        ),
    )
    after_first_receipt = store.repository_authority_snapshot(
        namespace.repository_key
    )
    assert after_first_receipt.generations == after_candidate.generations
    assert after_first_receipt.event_count == after_candidate.event_count
    assert after_first_receipt.candidate_authority_receipt_count == 1
    assert after_first_receipt.state_token != after_candidate.state_token

    second_receipt = _authority_receipt(
        candidate,
        authority_resolution_hash="9" * 64,
    )
    store.put_candidate(
        candidate,
        second_receipt,
        request_id=stable_request_id(
            "authority-state-token-receipt-2",
            candidate.candidate_id,
        ),
    )
    after_second_receipt = store.repository_authority_snapshot(
        namespace.repository_key
    )
    assert after_second_receipt.generations == after_candidate.generations
    assert after_second_receipt.event_count == after_candidate.event_count
    assert after_second_receipt.candidate_authority_receipt_count == 2
    assert (
        after_second_receipt.candidate_authority_receipt_set_hash
        != after_first_receipt.candidate_authority_receipt_set_hash
    )
    assert after_second_receipt.state_token != after_first_receipt.state_token
    assert MemoryStore(namespace, read_only=True).repository_authority_state_token(
        namespace.repository_key
    ) == after_second_receipt.state_token


def test_blob_promotion_before_database_failure_is_collected_as_orphan(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    content = b"promoted before a simulated database crash"
    digest = hashlib.sha256(content).hexdigest()
    with store._maintenance_connection() as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_test_blob BEFORE INSERT ON blobs
            BEGIN SELECT RAISE(ABORT, 'injected failure'); END
            """
        )

    with pytest.raises(MemoryStoreConflictError):
        store.put_blob(content, media_type="application/octet-stream")
    assert store.blob_path(digest).is_file()

    with store._maintenance_connection() as connection:
        connection.execute("DROP TRIGGER reject_test_blob")
    result = store.apply_blob_gc(store.gc_blobs(dry_run=True))
    assert str(store.blob_path(digest)) in result.deleted_orphan_paths
    assert not store.blob_path(digest).exists()


def test_blob_promotion_and_gc_are_serialized_across_store_instances(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = tmp_path / "memory"
    writer = MemoryStore(namespace, busy_timeout_ms=1_000)
    collector = MemoryStore(namespace, busy_timeout_ms=40)
    entered = threading.Event()
    release = threading.Event()
    errors = []
    original = writer._put_blob_locked

    def paused_put(*args, **kwargs):
        entered.set()
        if not release.wait(timeout=2):
            raise RuntimeError("test did not release blob writer")
        return original(*args, **kwargs)

    monkeypatch.setattr(writer, "_put_blob_locked", paused_put)

    def write_blob() -> None:
        try:
            writer.put_blob(b"serialized", media_type="application/octet-stream")
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)

    thread = threading.Thread(target=write_blob)
    thread.start()
    assert entered.wait(timeout=2)
    try:
        assert collector.gc_blobs(dry_run=True).candidate_hashes == ()
    finally:
        release.set()
        thread.join(timeout=2)

    assert not thread.is_alive()
    assert errors == []


def test_blob_gc_preserves_session_and_source_bundle_pins(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory")
    orphan = store.put_blob(b"orphan", media_type="application/octet-stream")
    store.pin_blob(orphan.blob_hash, pin_type="session", pin_id="review-001")
    assert store.apply_blob_gc(store.gc_blobs(dry_run=True)).deleted_hashes == ()
    store.unpin_blob(orphan.blob_hash, pin_type="session", pin_id="review-001")
    assert store.apply_blob_gc(store.gc_blobs(dry_run=True)).deleted_hashes == (
        orphan.blob_hash,
    )

    candidate = _candidate()
    store.put_candidate(
        candidate,
        _authority_receipt(candidate),
        request_id=stable_request_id("candidate", candidate.candidate_id),
    )
    content = b"source bundle"
    store.put_blob(
        content,
        media_type="application/vnd.review-agent.source-bundle+json",
    )
    bundle = _bundle(candidate, content)
    store.put_source_bundle(
        bundle,
        request_id=stable_request_id("bundle", bundle.bundle_hash),
        authority_resolution_hash=AUTHORITY_RESOLUTION,
    )

    result = store.apply_blob_gc(store.gc_blobs(dry_run=True))
    assert bundle.blob_hash not in result.deleted_hashes
    assert store.read_blob(bundle.blob_hash) == content


def test_blob_gc_apply_is_preview_bounded_and_rechecks_live_pins(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    previewed = store.put_blob(
        b"previewed orphan",
        media_type="application/octet-stream",
    )
    preview = store.gc_blobs(dry_run=True)
    assert preview.candidate_hashes == (previewed.blob_hash,)

    created_after_preview = store.put_blob(
        b"new orphan after preview",
        media_type="application/octet-stream",
    )
    store.pin_blob(
        previewed.blob_hash,
        pin_type="session",
        pin_id="review-raced-with-gc",
    )
    result = store.apply_blob_gc(preview)

    assert result.deleted_hashes == ()
    assert store.read_blob(previewed.blob_hash) == b"previewed orphan"
    assert store.read_blob(created_after_preview.blob_hash) == (
        b"new orphan after preview"
    )
    store.unpin_blob(
        previewed.blob_hash,
        pin_type="session",
        pin_id="review-raced-with-gc",
    )
    refreshed = store.gc_blobs(dry_run=True)
    assert set(refreshed.candidate_hashes) == {
        previewed.blob_hash,
        created_after_preview.blob_hash,
    }


def test_blob_gc_rejects_forged_and_cross_store_previews(tmp_path: Path) -> None:
    first = MemoryStore(tmp_path / "first")
    second = MemoryStore(tmp_path / "second")
    content = b"same unreferenced content"
    first_blob = first.put_blob(content, media_type="application/octet-stream")
    second_blob = second.put_blob(content, media_type="application/octet-stream")
    assert first_blob.blob_hash == second_blob.blob_hash

    first_preview = first.gc_blobs(dry_run=True)
    with pytest.raises(MemoryStoreValidationError, match="not issued"):
        second.apply_blob_gc(first_preview)
    assert second.read_blob(second_blob.blob_hash) == content

    forged = BlobGCResult(
        candidate_hashes=(second_blob.blob_hash,),
        deleted_hashes=(),
        orphan_paths=(),
        deleted_orphan_paths=(),
        reclaimed_bytes=len(content),
        dry_run=True,
        cutoff=first_preview.cutoff,
        preview_token="0" * 64,
    )
    with pytest.raises(MemoryStoreValidationError, match="not issued"):
        second.apply_blob_gc(forged)
    assert second.read_blob(second_blob.blob_hash) == content


def test_blob_gc_dry_run_is_filesystem_read_only(tmp_path: Path) -> None:
    namespace = tmp_path / "memory"
    writer = MemoryStore(namespace)
    orphan = writer.put_blob(
        b"read-only GC candidate",
        media_type="application/octet-stream",
    )

    def file_snapshot() -> dict[str, tuple[int, int]]:
        return {
            path.relative_to(namespace).as_posix(): (
                path.stat().st_size,
                path.stat().st_mtime_ns,
            )
            for path in namespace.rglob("*")
            if path.is_file()
        }

    before = file_snapshot()
    preview = MemoryStore(namespace, read_only=True).gc_blobs(dry_run=True)
    after = file_snapshot()

    assert preview.candidate_hashes == (orphan.blob_hash,)
    assert before == after


def test_read_only_store_reads_committed_frames_from_a_live_wal(
    tmp_path: Path,
) -> None:
    namespace = tmp_path / "memory"
    writer = MemoryStore(namespace)
    snapshot_reader = sqlite3.connect(
        str(writer.database_path),
        isolation_level=None,
    )
    try:
        snapshot_reader.execute("BEGIN")
        snapshot_reader.execute("SELECT COUNT(*) FROM candidates").fetchone()
        candidate = _candidate(statement="Committed WAL frames remain visible.")
        writer.put_candidate(
            candidate,
            request_id=stable_request_id("live-wal-reader", candidate.candidate_id),
        )
        wal_path = Path(str(writer.database_path) + "-wal")
        assert wal_path.is_file() and wal_path.stat().st_size > 0

        read_only = MemoryStore(namespace, read_only=True)

        assert read_only.get_candidate(candidate.candidate_id) == candidate
    finally:
        snapshot_reader.rollback()
        snapshot_reader.close()


def test_read_only_reader_uses_sqlite_snapshot_without_blocking_writer(
    tmp_path: Path,
) -> None:
    namespace = tmp_path / "memory"
    writer = MemoryStore(namespace, busy_timeout_ms=2_000)
    writer.checkpoint("TRUNCATE")
    read_only = MemoryStore(namespace, read_only=True, busy_timeout_ms=2_000)
    candidate = _candidate(statement="A WAL writer can commit beside a reader.")
    reader_started = threading.Event()
    release_reader = threading.Event()
    writer_finished = threading.Event()
    reader_counts: list[int] = []
    reader_errors: list[Exception] = []
    writer_errors: list[Exception] = []

    def read() -> None:
        try:
            with read_only.open_connection() as connection:
                connection.execute("BEGIN")
                reader_counts.append(
                    int(connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
                )
                reader_started.set()
                if not release_reader.wait(timeout=3):
                    raise RuntimeError("test did not release WAL reader")
                reader_counts.append(
                    int(connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
                )
                connection.rollback()
        except Exception as error:  # pragma: no cover - asserted below
            reader_errors.append(error)

    def write() -> None:
        try:
            writer.put_candidate(
                candidate,
                request_id=stable_request_id("first-wal", candidate.candidate_id),
            )
            writer_finished.set()
        except Exception as error:  # pragma: no cover - asserted below
            writer_errors.append(error)

    reader_thread = threading.Thread(target=read, name="snapshot-reader")
    writer_thread = threading.Thread(target=write, name="first-wal-writer")
    reader_thread.start()
    assert reader_started.wait(timeout=2)
    writer_thread.start()
    try:
        assert writer_finished.wait(timeout=2)
    finally:
        release_reader.set()
        reader_thread.join(timeout=3)
        writer_thread.join(timeout=3)

    assert not reader_thread.is_alive() and not writer_thread.is_alive()
    assert reader_errors == [] and writer_errors == []
    assert reader_counts == [0, 0]
    assert read_only.get_candidate(candidate.candidate_id) == candidate


def test_read_only_readers_are_not_serialized_by_namespace_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = tmp_path / "memory"
    MemoryStore(namespace).checkpoint("TRUNCATE")
    read_only = MemoryStore(namespace, read_only=True, busy_timeout_ms=2_000)
    rendezvous = threading.Barrier(2, timeout=2)
    real_connect = read_only._connect
    results: list[int] = []
    errors: list[Exception] = []

    def concurrent_connect(*, read_only: bool):
        if read_only:
            rendezvous.wait()
        return real_connect(read_only=read_only)

    monkeypatch.setattr(read_only, "_connect", concurrent_connect)

    def read() -> None:
        try:
            results.append(
                read_only.get_generations(REPOSITORY_KEY).memory_generation
            )
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)

    readers = [threading.Thread(target=read) for _ in range(2)]
    for reader in readers:
        reader.start()
    for reader in readers:
        reader.join(timeout=3)

    assert all(not reader.is_alive() for reader in readers)
    assert errors == []
    assert results == [0, 0]


def test_feedback_and_knowledge_update_only_their_generations(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory")
    content = b'{"symbols":[]}'
    store.put_blob(content, media_type="application/json")
    knowledge = _knowledge(content)
    store.put_knowledge_entry(
        knowledge,
        request_id=stable_request_id("knowledge", knowledge.entry_id),
    )
    feedback = _feedback()
    store.put_feedback(
        feedback,
        request_id=stable_request_id("feedback", feedback.feedback_id),
    )

    generations = store.get_generations(REPOSITORY_KEY)
    assert generations.memory_generation == 0
    assert generations.knowledge_generation == 1
    assert generations.feedback_generation == 1
    assert store.get_knowledge_entry(knowledge.entry_id) == knowledge
    assert store.get_feedback(feedback.feedback_id) == feedback

    Path(store.blob_path(knowledge.blob_hash)).unlink()
    with pytest.raises(MemoryStoreCorruptionError):
        store.get_knowledge_entry(knowledge.entry_id)


def test_project_memory_read_ignores_an_unrelated_missing_knowledge_blob(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    candidate = _candidate(status=CandidateStatus.PENDING_APPROVAL)
    authority = _authority_receipt(candidate)
    store.put_candidate(
        candidate,
        authority,
        request_id=stable_request_id("memory-view-candidate", candidate.candidate_id),
    )
    source_content = b'{"schema":"memory_source_bundle_v1","sources":[]}'
    store.put_blob(
        source_content,
        media_type="application/vnd.review-agent.source-bundle+json",
        created_at=CREATED_AT,
    )
    bundle = _bundle(candidate, source_content)
    approval_request = stable_request_id(
        "memory-view-approval",
        candidate.candidate_id,
    )
    record = _record(candidate, bundle, approval_request)
    store.approve_candidate_with_source_bundle(
        record,
        bundle,
        request_id=approval_request,
        expected_candidate_status=CandidateStatus.PENDING_APPROVAL,
        authority_resolution_hash=authority.authority_resolution_hash,
    )
    feedback = _feedback()
    store.put_feedback(
        feedback,
        request_id=stable_request_id("memory-view-feedback", feedback.feedback_id),
    )
    old_cache_content = b'{"symbols":["old"]}'
    store.put_blob(
        old_cache_content,
        media_type="application/json",
        created_at=CREATED_AT,
    )
    old_cache = _knowledge(old_cache_content)
    store.put_knowledge_entry(
        old_cache,
        request_id=stable_request_id("memory-view-cache", old_cache.entry_id),
    )
    store.blob_path(old_cache.blob_hash).unlink()

    view = store.read_view(REPOSITORY_KEY)

    assert view.records == (record,)
    assert view.feedback == (feedback,)
    assert view.knowledge_entries == (old_cache,)
    with pytest.raises(MemoryStoreCorruptionError):
        store.get_knowledge_entry(old_cache.entry_id)


def test_busy_authority_writer_has_stable_error_and_reader_remains_available(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory", busy_timeout_ms=40)
    writer = sqlite3.connect(str(store.database_path), timeout=0, isolation_level=None)
    writer.execute("BEGIN IMMEDIATE")
    try:
        assert store.get_generations(REPOSITORY_KEY).memory_generation == 0
        candidate = _candidate()
        with pytest.raises(MemoryStoreBusyError) as raised:
            store.put_candidate(
                candidate,
                request_id=stable_request_id("busy", candidate.candidate_id),
            )
        assert "INSERT" not in str(raised.value)
    finally:
        writer.rollback()
        writer.close()


def test_outbox_replay_audit_is_canonical_idempotent_and_projection_free(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    candidate = _candidate(statement="Replay audit has an owning repository.")
    store.put_candidate(
        candidate,
        request_id=stable_request_id("replay-audit-candidate", candidate.candidate_id),
    )
    replay_request = stable_request_id("explicit-outbox-replay", "review-001")

    first = store.record_outbox_replay_audit(
        REPOSITORY_KEY,
        review_id="review-001",
        outbox_hash=HASH_1,
        actor="  amy  ",
        reason="  Maintainer\r\n requested   a deterministic replay.  ",
        request_id=replay_request,
        created_at=CREATED_AT,
    )
    replay = store.record_outbox_replay_audit(
        REPOSITORY_KEY,
        review_id="review-001",
        outbox_hash=HASH_1,
        actor="amy",
        reason="Maintainer requested a deterministic replay.",
        request_id=replay_request,
        created_at=CREATED_AT,
    )

    assert first.applied and not first.replayed
    assert not replay.applied and replay.replayed
    assert replay.event_id == first.event_id
    assert first.actor == "amy"
    assert first.reason == "Maintainer requested a deterministic replay."
    events = store.list_events(
        REPOSITORY_KEY,
        subject_type="outbox_replay",
        subject_id="review-001",
    )
    assert len(events) == 1
    assert events[0].reason == first.reason
    assert store.read_view(REPOSITORY_KEY).generations.memory_generation == 2
    integrity = store.validate_integrity(validate_blob_files=False)
    assert integrity.event_count == 2

    manifest = store.build_export_manifest(redact=False, created_at=CREATED_AT)
    target = MemoryStore(tmp_path / "imported")
    preview = target.import_manifest(manifest, dry_run=True)
    assert preview.event_count == 2
    assert preview.outbox_receipt_count == 2
    applied = target.import_manifest(manifest, dry_run=False)
    assert applied.applied
    imported_replay = target.record_outbox_replay_audit(
        REPOSITORY_KEY,
        review_id="review-001",
        outbox_hash=HASH_1,
        actor="amy",
        reason=first.reason,
        request_id=replay_request,
        created_at=CREATED_AT,
    )
    assert imported_replay.replayed
    assert imported_replay.event_id == first.event_id
    assert target.validate_integrity(validate_blob_files=False).event_count == 2

    with pytest.raises(MemoryStoreConflictError, match="request ID"):
        store.record_outbox_replay_audit(
            REPOSITORY_KEY,
            review_id="review-001",
            outbox_hash=HASH_2,
            actor="amy",
            reason=first.reason,
            request_id=replay_request,
            created_at=CREATED_AT,
        )


def test_mixed_v1_v2_records_round_trip_export_state_token_and_backup(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    resolver = RevisionResolver()
    namespace = materialize_repository_memory_namespace(
        plan_repository_memory_namespace(
            resolver.repository_identity(git_repo),
            tmp_path / "memory-root",
            revision_resolver=resolver,
        ),
        revision_resolver=resolver,
    )
    source = MemoryStore(namespace)

    def approve(
        candidate: MemoryCandidate,
        content: bytes,
        *,
        v2: bool,
    ) -> DurableMemoryRecord:
        source.put_candidate(
            candidate,
            _authority_receipt(candidate),
            request_id=stable_request_id(
                "mixed-schema-candidate",
                candidate.candidate_id,
            ),
        )
        source.put_blob(
            content,
            media_type="application/vnd.review-agent.source-bundle+json",
            created_at=CREATED_AT,
        )
        bundle = _bundle(candidate, content)
        request_id = stable_request_id(
            "mixed-schema-approval",
            candidate.candidate_id,
        )
        record = (
            _record_v2(candidate, bundle, request_id)
            if v2
            else _record(candidate, bundle, request_id)
        )
        source.approve_candidate_with_source_bundle(
            record,
            bundle,
            request_id=request_id,
            expected_candidate_status=CandidateStatus.PENDING_APPROVAL,
            authority_resolution_hash=AUTHORITY_RESOLUTION,
        )
        return record

    v1_candidate = replace(
        _candidate(
            statement="Schema v1 records remain readable beside expiring records.",
            status=CandidateStatus.PENDING_APPROVAL,
        ),
        repository_key=namespace.repository_key,
    )
    v2_candidate = replace(
        _candidate(
            statement="Schema v2 records retain immutable expiry predicates.",
            status=CandidateStatus.PENDING_APPROVAL,
        ),
        repository_key=namespace.repository_key,
    )
    v1_record = approve(
        v1_candidate,
        b'{"schema":"memory_source_bundle_v1","record":1}',
        v2=False,
    )
    v2_record = approve(
        v2_candidate,
        b'{"schema":"memory_source_bundle_v1","record":2}',
        v2=True,
    )
    source.transition_record(
        v2_record.memory_id,
        expected_status=RecordStatus.ACTIVE,
        new_status=RecordStatus.EXPIRED,
        action="expire",
        actor_type="runtime",
        actor_id="memory_lifecycle",
        reason_code="expiry_condition_matched",
        request_id=stable_request_id("mixed-schema-expiry", v2_record.memory_id),
        created_at="2027-01-01T00:00:00Z",
    )
    expired_v2 = replace(v2_record, status=RecordStatus.EXPIRED)
    expected_records = tuple(sorted((v1_record, expired_v2), key=lambda item: item.memory_id))
    source_token = source.repository_authority_state_token(
        namespace.repository_key
    )

    with source.open_connection() as connection:
        stored_hashes = {
            str(row["memory_id"]): str(row["body_hash"])
            for row in connection.execute(
                "SELECT memory_id, body_hash FROM records ORDER BY memory_id"
            )
        }
    redacted = source.build_export_manifest(redact=True, created_at=CREATED_AT)
    assert all(envelope["model"] is None for envelope in redacted["records"])
    assert {
        envelope["id"]: envelope["body_hash"] for envelope in redacted["records"]
    } == stored_hashes
    assert {
        envelope["id"]: envelope["current_status"]
        for envelope in redacted["records"]
    }[v2_record.memory_id] == RecordStatus.EXPIRED.value
    redacted_plan = MemoryStore(tmp_path / "redacted-target").import_manifest(
        redacted,
        dry_run=True,
    )
    assert redacted_plan.record_count == 2
    assert redacted_plan.redacted and not redacted_plan.restorable

    portable_directory = tmp_path / "portable-export"
    portable = source.export_to_directory(
        portable_directory,
        redact=False,
        include_blobs=True,
        created_at=CREATED_AT,
    )
    portable_records = {
        envelope["id"]: envelope for envelope in portable["records"]
    }
    assert portable_records[v1_record.memory_id]["model"]["schema_version"] == 1
    assert "expiry_conditions" not in portable_records[v1_record.memory_id]["model"]
    assert portable_records[v2_record.memory_id]["model"]["schema_version"] == 2
    assert portable_records[v2_record.memory_id]["model"]["expiry_conditions"] == [
        condition.to_dict() for condition in v2_record.expiry_conditions
    ]

    def rehash_manifest(manifest: dict[str, object]) -> None:
        body = {
            key: value for key, value in manifest.items() if key != "manifest_hash"
        }
        manifest["manifest_hash"] = hashlib.sha256(
            canonical_json(body).encode("utf-8")
        ).hexdigest()

    for tamper in ("expiry_condition", "body_hash"):
        tampered = json.loads(canonical_json(portable))
        envelope = next(
            item
            for item in tampered["records"]
            if item["id"] == v2_record.memory_id
        )
        if tamper == "expiry_condition":
            envelope["model"]["expiry_conditions"] = [
                ExpiryCondition(
                    condition_kind=ExpiryConditionKind.AT_TIME,
                    value="2028-01-01T00:00:00Z",
                ).to_dict()
            ]
        else:
            envelope["body_hash"] = HASH_1
        rehash_manifest(tampered)
        with pytest.raises(MemoryStoreValidationError, match="model hash"):
            MemoryStore(tmp_path / ("tampered-" + tamper)).import_manifest(
                tampered,
                dry_run=True,
            )

    imported = MemoryStore(tmp_path / "imported")
    plan = imported.import_manifest(
        portable_directory / "manifest.json",
        dry_run=False,
        blob_source_root=portable_directory,
    )
    assert plan.applied and plan.record_count == 2
    assert imported.read_view(namespace.repository_key).records == expected_records
    assert imported.validate_integrity().record_count == 2
    assert imported.verify_event_chain(namespace.repository_key) == 7
    assert (
        imported.repository_authority_state_token(namespace.repository_key)
        == source_token
    )

    backup_path = source.backup_to(tmp_path / "backups" / "memory.sqlite3")
    backup = MemoryStore(backup_path, read_only=True)
    assert backup.validate_integrity(validate_blob_files=False).record_count == 2
    with backup.open_connection() as connection:
        backup_records = tuple(
            memory_store_module._record_from_row(row)
            for row in connection.execute("SELECT * FROM records ORDER BY memory_id")
        )
    assert backup_records == expected_records
    assert (
        backup.repository_authority_state_token(namespace.repository_key)
        == source_token
    )


def test_export_is_canonical_hashed_and_import_dry_run_never_writes(
    tmp_path: Path,
) -> None:
    source = MemoryStore(tmp_path / "source")
    candidate = _candidate()
    source.put_candidate(
        candidate,
        request_id=stable_request_id("candidate", candidate.candidate_id),
    )
    export_path = tmp_path / "memory-export.json"
    manifest = source.export_manifest(
        export_path,
        redact=True,
        created_at=CREATED_AT,
    )

    assert manifest["schema_name"] == EXPORT_SCHEMA_NAME
    assert candidate.statement not in export_path.read_text(encoding="utf-8")
    assert json.loads(export_path.read_text(encoding="utf-8")) == manifest
    assert export_path.read_text(encoding="utf-8") == canonical_json(manifest)
    body = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    assert manifest["manifest_hash"] == hashlib.sha256(
        canonical_json(body).encode("utf-8")
    ).hexdigest()

    target = MemoryStore(tmp_path / "target")
    plan = target.import_manifest(manifest, dry_run=True)
    assert plan.candidate_count == 1
    assert target.find_candidate(candidate.candidate_id) is None
    assert target.get_generations(REPOSITORY_KEY).memory_generation == 0

    tampered = dict(manifest)
    tampered["manifest_hash"] = HASH_1
    with pytest.raises(MemoryStoreValidationError):
        target.import_manifest(tampered, dry_run=True)


def test_prepared_import_is_detached_from_a_replaced_manifest_path(
    tmp_path: Path,
) -> None:
    first_source = MemoryStore(tmp_path / "first-source")
    first = _candidate(statement="The prepared manifest is authoritative.")
    first_source.put_candidate(
        first,
        request_id=stable_request_id("first", first.candidate_id),
    )
    manifest_path = tmp_path / "portable.json"
    first_manifest = first_source.export_manifest(
        manifest_path,
        redact=False,
        created_at=CREATED_AT,
    )

    target = MemoryStore(tmp_path / "target")
    prepared = target.prepare_import_manifest(manifest_path)

    second_source = MemoryStore(tmp_path / "second-source")
    second = _candidate(statement="A later path replacement must not be imported.")
    second_source.put_candidate(
        second,
        request_id=stable_request_id("second", second.candidate_id),
    )
    second_manifest = second_source.export_manifest(
        manifest_path,
        redact=False,
        created_at="2026-07-14T12:00:01Z",
    )
    assert first_manifest["manifest_hash"] != second_manifest["manifest_hash"]

    plan = target.apply_prepared_import(prepared)

    assert plan.applied
    assert target.get_candidate(first.candidate_id) == first
    assert target.find_candidate(second.candidate_id) is None


def test_import_rejects_projection_and_receipt_relationship_tampering_before_write(
    tmp_path: Path,
) -> None:
    source = MemoryStore(tmp_path / "source")
    candidate = _candidate()
    source.put_candidate(
        candidate,
        request_id=stable_request_id("candidate", candidate.candidate_id),
    )
    manifest = source.build_export_manifest(redact=False, created_at=CREATED_AT)

    def rehash(payload: dict[str, object]) -> dict[str, object]:
        body = {key: value for key, value in payload.items() if key != "manifest_hash"}
        payload["manifest_hash"] = hashlib.sha256(
            canonical_json(body).encode("utf-8")
        ).hexdigest()
        return payload

    projection_tampered = json.loads(canonical_json(manifest))
    projection_tampered["candidates"][0]["current_status"] = (
        CandidateStatus.VALIDATED.value
    )
    rehash(projection_tampered)
    target = MemoryStore(tmp_path / "target")
    with pytest.raises(MemoryStoreValidationError, match="projection"):
        target.import_manifest(projection_tampered, dry_run=True)
    with pytest.raises(MemoryStoreValidationError, match="projection"):
        target.import_manifest(projection_tampered, dry_run=False)
    assert target.find_candidate(candidate.candidate_id) is None
    assert target.get_generations(REPOSITORY_KEY).memory_generation == 0

    receipt_tampered = json.loads(canonical_json(manifest))
    receipt_tampered["outbox_receipts"][0]["request_id"] = stable_request_id(
        "different-import-receipt",
        candidate.candidate_id,
    )
    rehash(receipt_tampered)
    with pytest.raises(MemoryStoreValidationError, match="receipt event relationship"):
        target.import_manifest(receipt_tampered, dry_run=True)
    assert target.find_candidate(candidate.candidate_id) is None


def test_redacted_import_binds_initial_record_projection_to_approval_receipt(
    tmp_path: Path,
) -> None:
    source = MemoryStore(tmp_path / "source")
    candidate = _candidate(status=CandidateStatus.VALIDATED)
    source.put_candidate(
        candidate,
        _authority_receipt(candidate),
        request_id=stable_request_id("candidate", candidate.candidate_id),
    )
    content = b'{"sources":["payments/money.py:10-18"]}'
    source.put_blob(
        content,
        media_type="application/vnd.review-agent.source-bundle+json",
        created_at=CREATED_AT,
    )
    bundle = _bundle(candidate, content)
    source.put_source_bundle(
        bundle,
        request_id=stable_request_id("bundle", bundle.bundle_hash),
        authority_resolution_hash=AUTHORITY_RESOLUTION,
    )
    approval_request = stable_request_id("approve", candidate.candidate_id)
    source.approve_candidate(
        _record(candidate, bundle, approval_request),
        request_id=approval_request,
        expected_candidate_status=CandidateStatus.VALIDATED,
        authority_resolution_hash=AUTHORITY_RESOLUTION,
    )
    manifest = source.build_export_manifest(redact=True, created_at=CREATED_AT)
    tampered = json.loads(canonical_json(manifest))
    tampered["records"][0]["current_status"] = RecordStatus.REVOKED.value
    body = {key: value for key, value in tampered.items() if key != "manifest_hash"}
    tampered["manifest_hash"] = hashlib.sha256(
        canonical_json(body).encode("utf-8")
    ).hexdigest()

    with pytest.raises(MemoryStoreValidationError, match="candidate approval"):
        MemoryStore(tmp_path / "target").import_manifest(tampered, dry_run=True)


def test_import_transaction_failure_restores_empty_target_and_new_blob_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = MemoryStore(tmp_path / "source")
    content = b'{"symbols":["payments.money.calculate_total"]}'
    source.put_blob(content, media_type="application/json", created_at=CREATED_AT)
    knowledge = _knowledge(content)
    source.put_knowledge_entry(
        knowledge,
        request_id=stable_request_id("knowledge", knowledge.entry_id),
    )
    export_directory = tmp_path / "portable-export"
    source.export_to_directory(
        export_directory,
        redact=False,
        include_blobs=True,
        created_at=CREATED_AT,
    )
    target = MemoryStore(tmp_path / "target")

    def reject_projection(
        connection: sqlite3.Connection,
        repository_key: str,
    ) -> None:
        raise MemoryStoreCorruptionError("injected import projection failure")

    monkeypatch.setattr(target, "_verify_projection_connection", reject_projection)
    with pytest.raises(MemoryStoreCorruptionError, match="injected"):
        target.import_manifest(
            export_directory / "manifest.json",
            dry_run=False,
            blob_source_root=export_directory,
        )

    with target.open_connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM repositories").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM blobs").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM knowledge_entries").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 0
    assert not target.blob_path(knowledge.blob_hash).exists()


def test_import_blob_registration_failure_removes_newly_promoted_file(
    tmp_path: Path,
) -> None:
    source = MemoryStore(tmp_path / "source")
    content = b'{"symbols":["payments.money.calculate_total"]}'
    source.put_blob(content, media_type="application/json", created_at=CREATED_AT)
    knowledge = _knowledge(content)
    source.put_knowledge_entry(
        knowledge,
        request_id=stable_request_id("knowledge", knowledge.entry_id),
    )
    export_directory = tmp_path / "portable-export"
    source.export_to_directory(
        export_directory,
        redact=False,
        include_blobs=True,
        created_at=CREATED_AT,
    )
    target = MemoryStore(tmp_path / "target")
    with target._maintenance_connection() as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_import_blob BEFORE INSERT ON blobs
            BEGIN SELECT RAISE(ABORT, 'injected blob registration failure'); END
            """
        )

    with pytest.raises(MemoryStoreConflictError):
        target.import_manifest(
            export_directory / "manifest.json",
            dry_run=False,
            blob_source_root=export_directory,
        )

    with target.open_connection() as connection:
        assert connection.execute("SELECT COUNT(*) FROM blobs").fetchone()[0] == 0
    assert not target.blob_path(knowledge.blob_hash).exists()


def test_manifest_file_read_is_bounded_on_one_open_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = MemoryStore(tmp_path / "target")
    requested_sizes: list[int] = []

    class BoundedStream(io.BytesIO):
        def read(self, size: int = -1) -> bytes:
            requested_sizes.append(size)
            return super().read(size)

    monkeypatch.setattr(memory_store_module, "MAX_IMPORT_MANIFEST_BYTES", 8)
    monkeypatch.setattr(
        Path,
        "open",
        lambda self, mode="r", *args, **kwargs: BoundedStream(b"123456789"),
    )

    with pytest.raises(MemoryStoreValidationError, match="too large"):
        target.prepare_import_manifest(tmp_path / "swappable-manifest.json")

    assert requested_sizes == [9]


def test_redacted_registered_identity_is_validated_without_fabricated_paths(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    run_git(
        git_repo,
        "remote",
        "add",
        "origin",
        "https://user:token@example.test/org/repo.git?secret=yes#fragment",
    )
    resolver = RevisionResolver()
    namespace = materialize_repository_memory_namespace(
        plan_repository_memory_namespace(
            resolver.repository_identity(git_repo),
            tmp_path / "memory-root",
            revision_resolver=resolver,
        ),
        revision_resolver=resolver,
    )
    source = MemoryStore(namespace)
    manifest = source.build_export_manifest(redact=True, created_at=CREATED_AT)

    target = MemoryStore(tmp_path / "target")
    plan = target.import_manifest(manifest, dry_run=True)

    assert plan.repository_keys == (namespace.repository_key,)
    assert plan.redacted is True
    assert plan.restorable is False
    assert manifest["repositories"][0]["origin_url"] == (
        "https://example.test/org/repo.git"
    )

    def tamper_repository(**updates: object) -> dict[str, object]:
        tampered = dict(manifest)
        tampered["repositories"] = [
            {**manifest["repositories"][0], **updates}
        ]
        body = {
            key: value for key, value in tampered.items() if key != "manifest_hash"
        }
        tampered["manifest_hash"] = hashlib.sha256(
            canonical_json(body).encode("utf-8")
        ).hexdigest()
        return tampered

    invalid_updates = (
        {"identity_schema": "repository_identity_v0"},
        {"origin_url": "https://user:token@example.test/org/repo.git"},
        {"origin_url": "https://example.test/org/repo.git?secret=yes#fragment"},
        {"origin_url": " https://example.test/org/repo.git"},
        {"origin_url": 7},
    )
    for updates in invalid_updates:
        with pytest.raises(MemoryStoreValidationError):
            target.import_manifest(tamper_repository(**updates), dry_run=True)

    portable = source.build_export_manifest(redact=False, created_at=CREATED_AT)
    portable_plan = target.import_manifest(portable, dry_run=True)
    assert portable_plan.repository_keys == (namespace.repository_key,)
    assert portable_plan.redacted is False
    assert portable_plan.restorable is True

    for field in ("canonical_path", "git_common_dir"):
        tampered = dict(portable)
        tampered["repositories"] = [dict(portable["repositories"][0])]
        tampered["repositories"][0][field] = None
        body = {
            key: value for key, value in tampered.items() if key != "manifest_hash"
        }
        tampered["manifest_hash"] = hashlib.sha256(
            canonical_json(body).encode("utf-8")
        ).hexdigest()
        with pytest.raises(MemoryStoreValidationError):
            target.import_manifest(tampered, dry_run=True)


def test_nonredacted_import_restores_events_blobs_and_request_receipts(
    tmp_path: Path,
) -> None:
    source = MemoryStore(tmp_path / "source")
    candidate = _candidate()
    candidate_request = stable_request_id("candidate", candidate.candidate_id)
    source.put_candidate(candidate, request_id=candidate_request)
    content = b'{"symbols":["payments.money.calculate_total"]}'
    source.put_blob(content, media_type="application/json")
    knowledge = _knowledge(content)
    source.put_knowledge_entry(
        knowledge,
        request_id=stable_request_id("knowledge", knowledge.entry_id),
    )
    export_directory = tmp_path / "portable-export"
    source.export_to_directory(
        export_directory,
        redact=False,
        include_blobs=True,
        created_at=CREATED_AT,
    )

    target = MemoryStore(tmp_path / "target")
    plan = target.import_manifest(
        export_directory / "manifest.json",
        dry_run=False,
        blob_source_root=export_directory,
    )

    assert plan.applied and plan.outbox_receipt_count == 2
    assert target.get_candidate(candidate.candidate_id) == candidate
    assert target.get_knowledge_entry(knowledge.entry_id) == knowledge
    assert target.verify_event_chain(REPOSITORY_KEY) == 2
    replay = target.put_candidate(candidate, request_id=candidate_request)
    assert replay.replayed and not replay.applied


def test_export_import_round_trips_multiple_candidate_authority_contexts(
    tmp_path: Path,
) -> None:
    source = MemoryStore(tmp_path / "source")
    candidate = _candidate()
    first = _authority_receipt(candidate)
    second = _authority_receipt(
        candidate,
        authority_resolution_hash="9" * 64,
        created_at="2026-07-14T12:00:01Z",
    )
    source.put_candidate(
        candidate,
        first,
        request_id=stable_request_id("candidate-authority", first.receipt_id),
    )
    source.put_candidate(
        candidate,
        second,
        request_id=stable_request_id("candidate-authority", second.receipt_id),
        expected_generation=1,
    )

    redacted = source.build_export_manifest(redact=True, created_at=CREATED_AT)
    redacted_plan = MemoryStore(tmp_path / "redacted-target").import_manifest(
        redacted,
        dry_run=True,
    )
    assert redacted_plan.authority_receipt_count == 2
    assert not redacted_plan.restorable
    assert all(
        envelope["model"] is None
        for envelope in redacted["candidate_authority_receipts"]
    )

    export_directory = tmp_path / "portable-export"
    source.export_to_directory(
        export_directory,
        redact=False,
        created_at=CREATED_AT,
    )
    target = MemoryStore(tmp_path / "target")
    plan = target.import_manifest(
        export_directory / "manifest.json",
        dry_run=False,
    )

    assert plan.applied and plan.authority_receipt_count == 2
    assert target.list_candidate_authority_receipts(candidate.candidate_id) == (
        first,
        second,
    )
    assert target.validate_integrity().candidate_count == 1


def test_legacy_v1_audit_is_metadata_only_byte_preserving_and_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = tmp_path / "memory"
    writer = MemoryStore(namespace)
    candidate = _candidate()
    writer.put_candidate(
        candidate,
        request_id=stable_request_id("legacy-audit", candidate.candidate_id),
    )
    _downgrade_store_to_frozen_v1(writer)

    def snapshot() -> dict[str, tuple[bytes, int]]:
        return {
            path.relative_to(namespace).as_posix(): (
                path.read_bytes(),
                path.stat().st_mtime_ns,
            )
            for path in namespace.rglob("*")
            if path.is_file()
        }

    before = snapshot()

    def reject_current_model_hydration(*args, **kwargs):
        raise AssertionError("legacy audit hydrated a current authority model")

    for helper in (
        "_candidate_from_row",
        "_record_from_row",
        "_feedback_from_row",
        "_knowledge_from_row",
    ):
        monkeypatch.setattr(
            memory_store_module,
            helper,
            reject_current_model_hydration,
        )

    legacy = MemoryStore(namespace, read_only=True)
    status = legacy.audit_status()

    assert status.audit_schema is MemoryStoreAuditSchema.LEGACY_V1
    assert status.schema_name == "memory_store_schema_v1"
    assert status.schema_version == 1
    assert status.read_only
    assert status.migration_required
    assert status.repository_count == 1
    assert status.candidate_count == 1
    assert status.event_count == 1
    assert status.request_receipt_count == 1
    assert status.event_chain_verified
    with pytest.raises(MemoryStoreSchemaError, match="audit_status"):
        legacy.get_candidate(candidate.candidate_id)
    with pytest.raises(MemoryStoreReadOnlyError):
        legacy.put_candidate(
            candidate,
            request_id=stable_request_id(
                "legacy-audit-mutation",
                candidate.candidate_id,
            ),
        )
    with pytest.raises(MemoryStoreReadOnlyError):
        legacy.put_blob(b"forbidden", media_type="application/octet-stream")

    assert snapshot() == before
    assert not Path(str(writer.database_path) + "-wal").exists()
    assert not Path(str(writer.database_path) + "-shm").exists()


def test_v1_to_v2_migration_preserves_authority_data_and_event_bytes(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    candidate = _candidate(status=CandidateStatus.PENDING_APPROVAL)
    receipt = _authority_receipt(candidate)
    store.put_candidate(
        candidate,
        receipt,
        request_id=stable_request_id("migration-candidate", candidate.candidate_id),
    )
    source_content = b'{"schema":"memory_source_bundle_v1","sources":[]}'
    store.put_blob(
        source_content,
        media_type="application/vnd.review-agent.source-bundle+json",
        created_at=CREATED_AT,
    )
    bundle = _bundle(candidate, source_content)
    approval_request = stable_request_id("migration-approve", candidate.candidate_id)
    record = _record(candidate, bundle, approval_request)
    store.approve_candidate_with_source_bundle(
        record,
        bundle,
        request_id=approval_request,
        expected_candidate_status=CandidateStatus.PENDING_APPROVAL,
        expected_generation=1,
        authority_resolution_hash=receipt.authority_resolution_hash,
    )
    feedback = _feedback()
    store.put_feedback(
        feedback,
        request_id=stable_request_id("migration-feedback", feedback.feedback_id),
    )
    knowledge_content = b'{"symbols":[]}'
    store.put_blob(knowledge_content, media_type="application/json", created_at=CREATED_AT)
    knowledge = _knowledge(knowledge_content)
    store.put_knowledge_entry(
        knowledge,
        request_id=stable_request_id("migration-knowledge", knowledge.entry_id),
    )
    pinned = store.put_blob(
        b"manual pin",
        media_type="application/octet-stream",
        created_at=CREATED_AT,
    )
    store.pin_blob(
        pinned.blob_hash,
        pin_type="manual",
        pin_id="migration-pin",
        created_at=CREATED_AT,
    )
    legacy_unapproved = _candidate(
        statement="Legacy candidates require renewed authority before approval.",
        status=CandidateStatus.PENDING_APPROVAL,
    )
    store.put_candidate(
        legacy_unapproved,
        request_id=stable_request_id(
            "migration-legacy-candidate",
            legacy_unapproved.candidate_id,
        ),
    )
    _downgrade_store_to_frozen_v1(store)

    preserved_tables = (
        "repositories",
        "generations",
        "blobs",
        "blob_pins",
        "candidates",
        "source_bundles",
        "records",
        "feedback",
        "knowledge_entries",
        "events",
        "event_chain_heads",
        "sqlite_sequence",
    )
    before = {
        table: _table_snapshot(store.database_path, table)
        for table in preserved_tables
    }
    receipts_before = _table_snapshot(store.database_path, "outbox_receipts")
    v1_migration = _table_snapshot(store.database_path, "schema_migrations")

    migrated = MemoryStore(store.database_path)

    assert migrated.metadata()["schema_name"] == "memory_store_schema_v2"
    assert migrated.metadata()["schema_version"] == "2"
    assert all(
        _table_snapshot(store.database_path, table) == rows
        for table, rows in before.items()
    )
    with migrated.open_connection() as connection:
        receipt_rows = connection.execute(
            """
            SELECT request_id, repository_key, operation, request_hash,
                   subject_id, event_id, result_json, created_at,
                   request_hash_version
            FROM outbox_receipts
            """
        ).fetchall()
        assert tuple(tuple(row[:8]) for row in receipt_rows) == receipts_before
        assert {row[8] for row in receipt_rows} == {1}
        assert connection.execute(
            "SELECT COUNT(*) FROM candidate_authority_receipts"
        ).fetchone()[0] == 0
        assert tuple(
            tuple(row)
            for row in connection.execute(
                "SELECT * FROM schema_migrations WHERE schema_version = 1"
            )
        ) == v1_migration
    assert migrated.get_record(record.memory_id) == record
    assert migrated.get_candidate(candidate.candidate_id).status is CandidateStatus.APPROVED
    assert migrated.get_candidate(legacy_unapproved.candidate_id) == legacy_unapproved
    assert migrated.list_candidate_authority_receipts(
        legacy_unapproved.candidate_id
    ) == ()
    legacy_content = b'{"schema":"memory_source_bundle_v1","legacy":true}'
    migrated.put_blob(
        legacy_content,
        media_type="application/vnd.review-agent.source-bundle+json",
        created_at=CREATED_AT,
    )
    legacy_bundle = _bundle(legacy_unapproved, legacy_content)
    legacy_request = stable_request_id(
        "migration-legacy-approval",
        legacy_unapproved.candidate_id,
    )
    legacy_record = _record(legacy_unapproved, legacy_bundle, legacy_request)
    with pytest.raises(MemoryStoreValidationError, match="exact current-context"):
        migrated.approve_candidate_with_source_bundle(
            legacy_record,
            legacy_bundle,
            request_id=legacy_request,
            expected_candidate_status=CandidateStatus.PENDING_APPROVAL,
        )
    with pytest.raises(MemoryStoreConflictError, match="not stored"):
        migrated.approve_candidate_with_source_bundle(
            legacy_record,
            legacy_bundle,
            request_id=legacy_request,
            expected_candidate_status=CandidateStatus.PENDING_APPROVAL,
            authority_resolution_hash="f" * 64,
        )
    migrated.validate_integrity()


def test_v1_migration_failure_leaves_original_database_intact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    candidate = _candidate()
    store.put_candidate(
        candidate,
        request_id=stable_request_id("migration-failure", candidate.candidate_id),
    )
    _downgrade_store_to_frozen_v1(store)
    before_bytes = store.database_path.read_bytes()
    before_events = _table_snapshot(store.database_path, "events")

    def fail_migration(connection: sqlite3.Connection) -> None:
        connection.execute("CREATE TABLE must_not_escape(value TEXT)")
        raise RuntimeError("injected migration failure")

    monkeypatch.setattr(
        MemoryStore,
        "_migrate_v1_connection",
        staticmethod(fail_migration),
    )
    with pytest.raises(MemoryStoreMigrationError):
        MemoryStore(store.database_path)

    assert store.database_path.read_bytes() == before_bytes
    assert _table_snapshot(store.database_path, "events") == before_events
    connection = sqlite3.connect(str(store.database_path))
    connection.row_factory = sqlite3.Row
    try:
        assert memory_store_module._is_v1_schema_connection(connection)
        assert "must_not_escape" not in {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    finally:
        connection.close()


def test_v1_migration_rejects_semantically_corrupt_staging_before_replace(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    candidate = _candidate()
    store.put_candidate(
        candidate,
        request_id=stable_request_id("corrupt-v1", candidate.candidate_id),
    )
    _downgrade_store_to_frozen_v1(store)
    connection = sqlite3.connect(str(store.database_path), isolation_level=None)
    try:
        connection.execute(
            "UPDATE generations SET memory_generation = memory_generation + 1 "
            "WHERE repository_key = ?",
            (REPOSITORY_KEY,),
        )
    finally:
        connection.close()
    before = store.database_path.read_bytes()

    with pytest.raises(MemoryStoreMigrationError):
        MemoryStore(store.database_path)

    assert store.database_path.read_bytes() == before
    connection = sqlite3.connect(str(store.database_path))
    connection.row_factory = sqlite3.Row
    try:
        assert memory_store_module._is_v1_schema_connection(connection)
        assert connection.execute(
            "SELECT memory_generation FROM generations WHERE repository_key = ?",
            (REPOSITORY_KEY,),
        ).fetchone()[0] == 2
    finally:
        connection.close()


def test_concurrent_v1_migration_openers_converge_on_the_same_v2_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryStore(tmp_path / "memory", busy_timeout_ms=2_000)
    candidate = _candidate()
    store.put_candidate(
        candidate,
        request_id=stable_request_id("concurrent-migration", candidate.candidate_id),
    )
    _downgrade_store_to_frozen_v1(store)
    entered = threading.Event()
    release = threading.Event()
    original = MemoryStore._migrate_v1_connection

    def paused_migration(connection: sqlite3.Connection) -> None:
        original(connection)
        entered.set()
        if not release.wait(timeout=2):
            raise RuntimeError("test did not release migration")

    monkeypatch.setattr(
        MemoryStore,
        "_migrate_v1_connection",
        staticmethod(paused_migration),
    )
    opened: list[MemoryStore] = []
    errors: list[Exception] = []

    def open_store() -> None:
        try:
            opened.append(MemoryStore(store.database_path, busy_timeout_ms=2_000))
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)

    first = threading.Thread(target=open_store)
    second = threading.Thread(target=open_store)
    first.start()
    assert entered.wait(timeout=2)
    second.start()
    try:
        assert second.is_alive()
    finally:
        release.set()
        first.join(timeout=3)
        second.join(timeout=3)

    assert not first.is_alive() and not second.is_alive()
    assert errors == []
    assert len(opened) == 2
    assert all(item.metadata()["schema_version"] == "2" for item in opened)
    assert opened[0].get_candidate(candidate.candidate_id) == candidate


def test_authority_writer_waits_for_migration_and_is_not_lost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryStore(tmp_path / "memory", busy_timeout_ms=2_000)
    initial = _candidate()
    store.put_candidate(
        initial,
        request_id=stable_request_id("initial", initial.candidate_id),
    )
    _downgrade_store_to_frozen_v1(store)
    entered = threading.Event()
    release = threading.Event()
    original = MemoryStore._migrate_v1_connection

    def paused_migration(connection: sqlite3.Connection) -> None:
        original(connection)
        entered.set()
        if not release.wait(timeout=2):
            raise RuntimeError("test did not release migration")

    monkeypatch.setattr(
        MemoryStore,
        "_migrate_v1_connection",
        staticmethod(paused_migration),
    )
    migrated: list[MemoryStore] = []
    migration_errors: list[Exception] = []
    writer_errors: list[Exception] = []
    late = _candidate(statement="A writer committed after migration began.")

    def migrate() -> None:
        try:
            migrated.append(MemoryStore(store.database_path, busy_timeout_ms=2_000))
        except Exception as error:  # pragma: no cover - asserted below
            migration_errors.append(error)

    def write() -> None:
        try:
            store.put_candidate(
                late,
                request_id=stable_request_id("late", late.candidate_id),
            )
        except Exception as error:  # pragma: no cover - asserted below
            writer_errors.append(error)

    migration_thread = threading.Thread(target=migrate)
    writer_thread = threading.Thread(target=write)
    migration_thread.start()
    assert entered.wait(timeout=2)
    writer_thread.start()
    try:
        assert writer_thread.is_alive()
    finally:
        release.set()
        migration_thread.join(timeout=3)
        writer_thread.join(timeout=3)

    assert not migration_thread.is_alive() and not writer_thread.is_alive()
    assert migration_errors == [] and writer_errors == []
    assert migrated[0].get_candidate(initial.candidate_id) == initial
    assert migrated[0].get_candidate(late.candidate_id) == late


def test_failed_v1_migration_does_not_checkpoint_an_existing_wal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    candidate = _candidate()
    store.put_candidate(
        candidate,
        request_id=stable_request_id("live-wal", candidate.candidate_id),
    )
    _downgrade_store_to_frozen_v1(store)

    reader = sqlite3.connect(str(store.database_path), isolation_level=None)
    try:
        reader.execute("BEGIN")
        reader.execute("SELECT COUNT(*) FROM candidates").fetchone()
        writer = sqlite3.connect(str(store.database_path), isolation_level=None)
        try:
            writer.execute(
                "UPDATE repositories SET last_accessed_at = ? WHERE repository_key = ?",
                ("2026-07-14T12:00:01Z", REPOSITORY_KEY),
            )
        finally:
            writer.close()
        wal_path = Path(str(store.database_path) + "-wal")
        assert wal_path.is_file() and wal_path.stat().st_size > 0
        before_database = store.database_path.read_bytes()
        before_wal = wal_path.read_bytes()

        def fail_migration(connection: sqlite3.Connection) -> None:
            connection.execute("CREATE TABLE must_not_escape(value TEXT)")
            raise RuntimeError("injected migration failure")

        monkeypatch.setattr(
            MemoryStore,
            "_migrate_v1_connection",
            staticmethod(fail_migration),
        )
        with pytest.raises(MemoryStoreMigrationError):
            MemoryStore(store.database_path)

        assert store.database_path.read_bytes() == before_database
        assert wal_path.read_bytes() == before_wal
    finally:
        reader.rollback()
        reader.close()


def test_failed_v1_migration_preflight_does_not_checkpoint_crash_wal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    candidate = _candidate()
    store.put_candidate(
        candidate,
        request_id=stable_request_id("crash-wal", candidate.candidate_id),
    )
    _downgrade_store_to_frozen_v1(store)
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os, sqlite3, sys; "
                "connection = sqlite3.connect(sys.argv[1], isolation_level=None); "
                "connection.execute('PRAGMA wal_autocheckpoint = 0'); "
                "connection.execute(\"UPDATE repositories SET last_accessed_at = "
                "'2026-07-14T12:00:02Z'\"); "
                "os._exit(0)"
            ),
            str(store.database_path),
        ],
        check=True,
    )
    wal_path = Path(str(store.database_path) + "-wal")
    assert wal_path.is_file() and wal_path.stat().st_size > 0
    before_database = store.database_path.read_bytes()
    before_wal = wal_path.read_bytes()

    def fail_before_migration(self: MemoryStore) -> None:
        raise MemoryStoreMigrationError("injected migration failure")

    monkeypatch.setattr(MemoryStore, "_migrate_v1_store", fail_before_migration)
    with pytest.raises(MemoryStoreMigrationError, match="injected"):
        MemoryStore(store.database_path)

    assert store.database_path.read_bytes() == before_database
    assert wal_path.read_bytes() == before_wal


def test_migrated_v1_request_receipt_keeps_legacy_replay_semantics(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    candidate = _candidate()
    request_id = stable_request_id("legacy-v1-replay", candidate.candidate_id)
    store.put_candidate(
        candidate,
        request_id=request_id,
        expected_generation=0,
    )
    legacy_hash = canonical_sha256(
        {
            "operation": "put_candidate",
            "candidate": candidate.to_dict(),
            "expected_generation": 0,
            "actor_type": candidate.producer.producer_type.value,
            "actor_id": candidate.producer.name,
            "reason_code": "candidate_submitted",
            "reason": None,
        }
    )
    with store._maintenance_connection() as connection:
        connection.execute("DROP TRIGGER outbox_receipts_no_update")
        connection.execute(
            "UPDATE outbox_receipts SET request_hash = ? WHERE request_id = ?",
            (legacy_hash, request_id),
        )
    _downgrade_store_to_frozen_v1(store)
    migrated = MemoryStore(store.database_path)

    advanced = _candidate(statement="Advance the migrated memory generation.")
    migrated.put_candidate(
        advanced,
        request_id=stable_request_id("legacy-v1-advance", advanced.candidate_id),
    )
    replay = migrated.put_candidate(
        candidate,
        request_id=request_id,
        expected_generation=0,
    )
    assert replay.replayed and not replay.applied
    with pytest.raises(MemoryStoreConflictError, match="reused"):
        migrated.put_candidate(
            candidate,
            request_id=request_id,
            expected_generation=1,
        )
    with pytest.raises(MemoryStoreValidationError, match="expected generation"):
        migrated.put_candidate(
            candidate,
            request_id=request_id,
            expected_generation=True,
        )
    with migrated.open_connection() as connection:
        assert connection.execute(
            "SELECT request_hash_version FROM outbox_receipts WHERE request_id = ?",
            (request_id,),
        ).fetchone()[0] == 1


def test_staged_migration_failure_preserves_original_and_success_replaces_it(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    candidate = _candidate()
    store.put_candidate(
        candidate,
        request_id=stable_request_id("candidate", candidate.candidate_id),
    )

    def fail_migration(connection: sqlite3.Connection) -> None:
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES ('migration_marker', 'bad')"
        )
        raise RuntimeError("boom")

    with pytest.raises(MemoryStoreMigrationError):
        store.replace_with_staged_copy(fail_migration)
    assert store.get_candidate(candidate.candidate_id) == candidate
    assert "migration_marker" not in store.metadata()

    def successful_migration(connection: sqlite3.Connection) -> None:
        connection.execute(
            "INSERT INTO metadata(key, value) VALUES ('migration_marker', 'ok')"
        )

    store.replace_with_staged_copy(successful_migration)
    assert store.metadata()["migration_marker"] == "ok"
    assert store.get_candidate(candidate.candidate_id) == candidate


def test_staged_migration_callback_can_read_blob_while_blob_writer_is_paused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryStore(tmp_path / "memory", busy_timeout_ms=1_000)
    content = b"blob read by a staged migration callback"
    existing = store.put_blob(content, media_type="application/octet-stream")
    writer_entered = threading.Event()
    release_writer = threading.Event()
    callback_read = threading.Event()
    writer_errors: list[Exception] = []
    migration_errors: list[Exception] = []
    original_put_blob_locked = store._put_blob_locked

    def paused_put_blob(*args, **kwargs):
        writer_entered.set()
        if not release_writer.wait(timeout=3):
            raise RuntimeError("test did not release blob writer")
        return original_put_blob_locked(*args, **kwargs)

    monkeypatch.setattr(store, "_put_blob_locked", paused_put_blob)

    def write_blob() -> None:
        try:
            store.put_blob(b"concurrent blob", media_type="application/octet-stream")
        except Exception as error:  # pragma: no cover - asserted below
            writer_errors.append(error)

    def migration(connection: sqlite3.Connection) -> None:
        assert store.read_blob(existing.blob_hash) == content
        callback_read.set()

    def replace_store() -> None:
        try:
            store.replace_with_staged_copy(migration)
        except Exception as error:  # pragma: no cover - asserted below
            migration_errors.append(error)

    writer_thread = threading.Thread(target=write_blob)
    migration_thread = threading.Thread(target=replace_store)
    writer_thread.start()
    assert writer_entered.wait(timeout=2)
    migration_thread.start()
    try:
        assert callback_read.wait(timeout=2)
        migration_thread.join(timeout=3)
        assert not migration_thread.is_alive()
    finally:
        release_writer.set()
        writer_thread.join(timeout=3)
        migration_thread.join(timeout=3)

    assert writer_errors == [] and migration_errors == []
    assert store.read_blob(existing.blob_hash) == content


def test_database_backup_is_atomic_and_auditable_without_copying_blob_tree(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    candidate = _candidate()
    store.put_candidate(
        candidate,
        request_id=stable_request_id("candidate", candidate.candidate_id),
    )
    store.put_blob(b"separate blob tree", media_type="application/octet-stream")

    backup_path = store.backup_to(tmp_path / "backups" / "memory.sqlite3")
    backup = MemoryStore(backup_path, read_only=True)

    assert backup.get_candidate(candidate.candidate_id) == candidate
    assert backup.validate_integrity(validate_blob_files=False).candidate_count == 1
