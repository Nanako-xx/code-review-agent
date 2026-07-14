from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import sqlite3
import threading
from pathlib import Path

import pytest

from review_agent.memory_models import (
    CandidateStatus,
    DurableMemoryRecord,
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
    stable_event_id,
    stable_request_id,
)
from review_agent.memory_store import (
    EXPORT_SCHEMA_NAME,
    STORE_SCHEMA_NAME,
    STORE_SCHEMA_VERSION,
    MemoryStore,
    MemoryStoreBusyError,
    MemoryStoreConflictError,
    MemoryStoreCorruptionError,
    MemoryStoreMigrationError,
    MemoryStoreReadOnlyError,
    MemoryStoreSchemaError,
    MemoryStoreValidationError,
)


SHA_A = "a" * 40
HASH_1 = "1" * 64
HASH_2 = "2" * 64
CREATED_AT = "2026-07-14T12:00:00Z"
REPOSITORY_KEY = "4" * 64


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


def test_approval_uses_generation_and_status_cas_without_duplicate_record(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "memory")
    candidate = _candidate(status=CandidateStatus.VALIDATED)
    store.put_candidate(
        candidate,
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
        )
    assert store.get_generations(REPOSITORY_KEY).memory_generation == expected_generation

    result = store.approve_candidate(
        record,
        request_id=request_id,
        expected_candidate_status=CandidateStatus.VALIDATED,
        expected_generation=expected_generation,
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
    result = store.gc_blobs(dry_run=False)
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
        with pytest.raises(MemoryStoreBusyError):
            collector.gc_blobs(dry_run=True)
    finally:
        release.set()
        thread.join(timeout=2)

    assert not thread.is_alive()
    assert errors == []


def test_blob_gc_preserves_session_and_source_bundle_pins(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "memory")
    orphan = store.put_blob(b"orphan", media_type="application/octet-stream")
    store.pin_blob(orphan.blob_hash, pin_type="session", pin_id="review-001")
    assert store.gc_blobs(dry_run=False).deleted_hashes == ()
    store.unpin_blob(orphan.blob_hash, pin_type="session", pin_id="review-001")
    assert store.gc_blobs(dry_run=False).deleted_hashes == (orphan.blob_hash,)

    candidate = _candidate()
    store.put_candidate(
        candidate,
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
    )

    result = store.gc_blobs(dry_run=False)
    assert bundle.blob_hash not in result.deleted_hashes
    assert store.read_blob(bundle.blob_hash) == content


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
