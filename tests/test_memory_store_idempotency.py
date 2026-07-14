from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Callable, Literal

import pytest

import review_agent.memory_store as memory_store_module
from review_agent.memory_models import (
    CandidateAuthorityReceipt,
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
    stable_event_id,
    stable_request_id,
)
from review_agent.memory_store import (
    MemoryStore,
    MemoryStoreConflictError,
    WriteResult,
)


SHA = "a" * 40
CONTENT_HASH = "1" * 64
DECLARATION_HASH = "2" * 64
REPOSITORY_KEY = "4" * 64
AUTHORITY_RESOLUTION = "7" * 64
CREATED_AT = "2026-07-14T12:00:00Z"
SOURCE_BUNDLE_MEDIA_TYPE = "application/vnd.review-agent.source-bundle+json"

GenerationKind = Literal["memory", "feedback", "knowledge"]


@dataclass(frozen=True)
class ReplayCase:
    generation_kind: GenerationKind
    initial_expected_generation: int
    execute: Callable[[int], WriteResult]
    execute_conflict: Callable[[int], WriteResult]
    target_projection: Callable[[], object]


CaseFactory = Callable[[MemoryStore], ReplayCase]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source() -> RepositoryRangeSourceRef:
    return RepositoryRangeSourceRef(
        revision=SHA,
        path="payments/money.py",
        line_start=10,
        line_end=18,
        content_hash=CONTENT_HASH,
    )


def _human_source(actor: str = "amy") -> HumanDeclarationSourceRef:
    return HumanDeclarationSourceRef(
        request_id=stable_request_id("human-source", actor),
        actor=actor,
        declaration_hash=DECLARATION_HASH,
        created_at=CREATED_AT,
        review_id="review-001",
    )


def _candidate(
    marker: str,
    *,
    status: CandidateStatus = CandidateStatus.PROPOSED,
) -> MemoryCandidate:
    return MemoryCandidate(
        repository_key=REPOSITORY_KEY,
        kind=MemoryKind.BUSINESS_INVARIANT,
        statement=f"Amounts must use Decimal ({marker}).",
        scope=MemoryScope(
            paths=("payments/**",),
            contracts=("numeric_correctness",),
            languages=("python",),
        ),
        source_refs=(_source(), _human_source()),
        valid_from_sha=SHA,
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


def _authority_receipt(candidate: MemoryCandidate) -> CandidateAuthorityReceipt:
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
        authority_resolution_hash=AUTHORITY_RESOLUTION,
        binding_id=None,
        created_at=CREATED_AT,
    )


def _put_candidate_prerequisite(
    store: MemoryStore,
    marker: str,
    *,
    status: CandidateStatus = CandidateStatus.PROPOSED,
) -> tuple[MemoryCandidate, CandidateAuthorityReceipt]:
    candidate = _candidate(marker, status=status)
    receipt = _authority_receipt(candidate)
    store.put_candidate(
        candidate,
        receipt,
        request_id=stable_request_id("setup-candidate", candidate.candidate_id),
    )
    return candidate, receipt


def _source_bundle(
    store: MemoryStore,
    candidate: MemoryCandidate,
    marker: str,
) -> SourceBundleDescriptor:
    content = (f'{{"case":"{marker}","sources":[]}}').encode("utf-8")
    blob = store.put_blob(
        content,
        media_type=SOURCE_BUNDLE_MEDIA_TYPE,
        created_at=CREATED_AT,
    )
    return SourceBundleDescriptor(
        repository_key=candidate.repository_key,
        candidate_id=candidate.candidate_id,
        source_refs=candidate.source_refs,
        blob_hash=blob.blob_hash,
        size_bytes=blob.size_bytes,
        media_type=blob.media_type,
        created_at=CREATED_AT,
    )


def _record(
    candidate: MemoryCandidate,
    bundle: SourceBundleDescriptor,
    approval_request_id: str,
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
        approval_event_id=stable_event_id(
            "approve",
            candidate.candidate_id,
            approval_request_id,
        ),
        status=RecordStatus.ACTIVE,
        created_at=CREATED_AT,
    )


def _feedback(marker: str, *, reason: str = "Confirmed by maintainer.") -> FeedbackRecord:
    marker_digest = _digest(marker)
    finding = FindingSnapshot(
        finding_id="F-" + marker_digest[:32],
        claim=f"Rounding can lose cents ({marker}).",
        path="payments/money.py",
        line=42,
        contracts=("numeric_correctness",),
        original_severity=FindingSeverity.HIGH,
        evidence_refs=("O-" + _digest(marker + "-evidence")[:32],),
    )
    return FeedbackRecord(
        repository_key=REPOSITORY_KEY,
        review_id="review-" + marker_digest[:12],
        finding_id=finding.finding_id,
        head_sha=SHA,
        finding_snapshot=finding,
        decision=FeedbackDecision.ACCEPTED,
        original_severity=FindingSeverity.HIGH,
        final_severity=FindingSeverity.HIGH,
        reason_code=FeedbackReasonCode.OTHER,
        reason=reason,
        actor="amy",
        source_refs=(_human_source(),),
        status=FeedbackStatus.RECORDED,
        created_at=CREATED_AT,
    )


def _knowledge(
    store: MemoryStore,
    marker: str,
    *,
    pinned_by_review_ids: tuple[str, ...] = (),
) -> RepositoryKnowledgeEntry:
    content = (f'{{"symbols":[],"case":"{marker}"}}').encode("utf-8")
    blob = store.put_blob(
        content,
        media_type="application/json",
        created_at=CREATED_AT,
    )
    key = RepositoryKnowledgeKey(
        repository_key=REPOSITORY_KEY,
        revision_binding="head@" + SHA,
        capability=RepositoryKnowledgeCapability.SYMBOL_INDEX,
        analyzer_name="python-ast",
        analyzer_version="3.12-v1",
        configuration_digest=CONTENT_HASH,
        input_digest=_digest(marker),
    )
    return RepositoryKnowledgeEntry(
        key=key,
        blob_hash=blob.blob_hash,
        size_bytes=blob.size_bytes,
        content_type=blob.media_type,
        artifact_schema="symbol_index_v1",
        created_at=CREATED_AT,
        pinned_by_review_ids=pinned_by_review_ids,
    )


def _generation(store: MemoryStore, kind: GenerationKind) -> int:
    generations = store.get_generations(REPOSITORY_KEY)
    return getattr(generations, f"{kind}_generation")


def _advance_generation(store: MemoryStore, kind: GenerationKind, marker: str) -> None:
    if kind == "memory":
        candidate = _candidate("advance-" + marker)
        store.put_candidate(
            candidate,
            request_id=stable_request_id("advance-memory", candidate.candidate_id),
        )
        return
    if kind == "feedback":
        feedback = _feedback("advance-" + marker)
        store.put_feedback(
            feedback,
            request_id=stable_request_id("advance-feedback", feedback.feedback_id),
        )
        return
    entry = _knowledge(store, "advance-" + marker)
    store.put_knowledge_entry(
        entry,
        request_id=stable_request_id("advance-knowledge", entry.entry_id),
    )


def _public_state(store: MemoryStore, target_projection: Callable[[], object]) -> object:
    candidates = store.list_candidates(REPOSITORY_KEY)
    return (
        store.get_generations(REPOSITORY_KEY),
        store.list_events(REPOSITORY_KEY),
        candidates,
        tuple(
            (
                candidate.candidate_id,
                store.list_candidate_authority_receipts(candidate.candidate_id),
            )
            for candidate in candidates
        ),
        store.list_records(REPOSITORY_KEY),
        store.list_feedback(REPOSITORY_KEY),
        store.list_knowledge_entries(REPOSITORY_KEY),
        target_projection(),
    )


def _put_candidate_case(store: MemoryStore) -> ReplayCase:
    candidate = _candidate("put-candidate")
    receipt = _authority_receipt(candidate)
    request_id = stable_request_id("matrix-put-candidate", candidate.candidate_id)

    def execute(expected_generation: int) -> WriteResult:
        return store.put_candidate(
            candidate,
            receipt,
            request_id=request_id,
            expected_generation=expected_generation,
        )

    def execute_conflict(expected_generation: int) -> WriteResult:
        changed_receipt = replace(
            receipt,
            initial_validation_report_hash="9" * 64,
        )
        return store.put_candidate(
            candidate,
            changed_receipt,
            request_id=request_id,
            expected_generation=expected_generation,
        )

    return ReplayCase(
        generation_kind="memory",
        initial_expected_generation=_generation(store, "memory"),
        execute=execute,
        execute_conflict=execute_conflict,
        target_projection=lambda: (
            store.get_candidate(candidate.candidate_id),
            store.list_candidate_authority_receipts(candidate.candidate_id),
        ),
    )


def _transition_candidate_case(store: MemoryStore) -> ReplayCase:
    candidate, _ = _put_candidate_prerequisite(store, "transition-candidate")
    request_id = stable_request_id(
        "matrix-transition-candidate",
        candidate.candidate_id,
    )

    def execute_with_reason(expected_generation: int, reason: str) -> WriteResult:
        return store.transition_candidate(
            candidate.candidate_id,
            expected_status=CandidateStatus.PROPOSED,
            new_status=CandidateStatus.VALIDATED,
            action="candidate_validated",
            actor_type="runtime",
            actor_id="memory_sources",
            reason_code="sources_valid",
            reason=reason,
            request_id=request_id,
            expected_generation=expected_generation,
            authority_resolution_hash=AUTHORITY_RESOLUTION,
        )

    return ReplayCase(
        generation_kind="memory",
        initial_expected_generation=_generation(store, "memory"),
        execute=lambda expected: execute_with_reason(expected, "Sources validated."),
        execute_conflict=lambda expected: execute_with_reason(
            expected,
            "Semantically different validation reason.",
        ),
        target_projection=lambda: store.get_candidate(candidate.candidate_id),
    )


def _put_source_bundle_case(store: MemoryStore) -> ReplayCase:
    candidate, _ = _put_candidate_prerequisite(store, "put-source-bundle")
    bundle = _source_bundle(store, candidate, "put-source-bundle")
    request_id = stable_request_id("matrix-put-source-bundle", bundle.bundle_hash)

    def execute_with_authority(
        expected_generation: int,
        authority_resolution_hash: str,
    ) -> WriteResult:
        return store.put_source_bundle(
            bundle,
            request_id=request_id,
            expected_generation=expected_generation,
            authority_resolution_hash=authority_resolution_hash,
        )

    return ReplayCase(
        generation_kind="memory",
        initial_expected_generation=_generation(store, "memory"),
        execute=lambda expected: execute_with_authority(
            expected,
            AUTHORITY_RESOLUTION,
        ),
        execute_conflict=lambda expected: execute_with_authority(expected, "9" * 64),
        target_projection=lambda: store.get_source_bundle(bundle.bundle_hash),
    )


def _approve_with_source_bundle_case(store: MemoryStore) -> ReplayCase:
    candidate, _ = _put_candidate_prerequisite(
        store,
        "approve-with-source-bundle",
        status=CandidateStatus.PENDING_APPROVAL,
    )
    bundle = _source_bundle(store, candidate, "approve-with-source-bundle")
    request_id = stable_request_id(
        "matrix-approve-with-source-bundle",
        candidate.candidate_id,
    )
    record = _record(candidate, bundle, request_id)

    def execute_with_reason(expected_generation: int, reason: str) -> WriteResult:
        return store.approve_candidate_with_source_bundle(
            record,
            bundle,
            request_id=request_id,
            expected_candidate_status=CandidateStatus.PENDING_APPROVAL,
            expected_generation=expected_generation,
            authority_resolution_hash=AUTHORITY_RESOLUTION,
            actor_id="amy",
            reason=reason,
        )

    return ReplayCase(
        generation_kind="memory",
        initial_expected_generation=_generation(store, "memory"),
        execute=lambda expected: execute_with_reason(expected, "Approved with evidence."),
        execute_conflict=lambda expected: execute_with_reason(
            expected,
            "Different approval rationale.",
        ),
        target_projection=lambda: (
            store.get_candidate(candidate.candidate_id),
            store.get_source_bundle(bundle.bundle_hash),
            store.get_record(record.memory_id),
        ),
    )


def _approve_candidate_case(store: MemoryStore) -> ReplayCase:
    candidate, _ = _put_candidate_prerequisite(
        store,
        "approve-candidate",
        status=CandidateStatus.VALIDATED,
    )
    bundle = _source_bundle(store, candidate, "approve-candidate")
    store.put_source_bundle(
        bundle,
        request_id=stable_request_id("setup-source-bundle", bundle.bundle_hash),
        authority_resolution_hash=AUTHORITY_RESOLUTION,
    )
    request_id = stable_request_id("matrix-approve-candidate", candidate.candidate_id)
    record = _record(candidate, bundle, request_id)

    def execute_with_reason(expected_generation: int, reason: str) -> WriteResult:
        return store.approve_candidate(
            record,
            request_id=request_id,
            expected_candidate_status=CandidateStatus.VALIDATED,
            expected_generation=expected_generation,
            authority_resolution_hash=AUTHORITY_RESOLUTION,
            actor_id="amy",
            reason=reason,
        )

    return ReplayCase(
        generation_kind="memory",
        initial_expected_generation=_generation(store, "memory"),
        execute=lambda expected: execute_with_reason(expected, "Approved."),
        execute_conflict=lambda expected: execute_with_reason(
            expected,
            "Different approval rationale.",
        ),
        target_projection=lambda: (
            store.get_candidate(candidate.candidate_id),
            store.get_record(record.memory_id),
        ),
    )


def _transition_record_case(store: MemoryStore) -> ReplayCase:
    candidate, _ = _put_candidate_prerequisite(
        store,
        "transition-record",
        status=CandidateStatus.VALIDATED,
    )
    bundle = _source_bundle(store, candidate, "transition-record")
    store.put_source_bundle(
        bundle,
        request_id=stable_request_id("setup-source-bundle", bundle.bundle_hash),
        authority_resolution_hash=AUTHORITY_RESOLUTION,
    )
    approval_request_id = stable_request_id(
        "setup-approval",
        candidate.candidate_id,
    )
    record = _record(candidate, bundle, approval_request_id)
    store.approve_candidate(
        record,
        request_id=approval_request_id,
        expected_candidate_status=CandidateStatus.VALIDATED,
        authority_resolution_hash=AUTHORITY_RESOLUTION,
    )
    request_id = stable_request_id("matrix-transition-record", record.memory_id)

    def execute_with_reason(expected_generation: int, reason: str) -> WriteResult:
        return store.transition_record(
            record.memory_id,
            expected_status=RecordStatus.ACTIVE,
            new_status=RecordStatus.REVALIDATION_REQUIRED,
            action="require_revalidation",
            actor_type="runtime",
            actor_id="memory_lifecycle",
            reason_code="source_changed",
            reason=reason,
            request_id=request_id,
            expected_generation=expected_generation,
        )

    return ReplayCase(
        generation_kind="memory",
        initial_expected_generation=_generation(store, "memory"),
        execute=lambda expected: execute_with_reason(expected, "Source changed."),
        execute_conflict=lambda expected: execute_with_reason(
            expected,
            "Different lifecycle rationale.",
        ),
        target_projection=lambda: store.get_record(record.memory_id),
    )


def _put_feedback_case(store: MemoryStore) -> ReplayCase:
    feedback = _feedback("put-feedback")
    request_id = stable_request_id("matrix-put-feedback", feedback.feedback_id)

    def execute(expected_generation: int) -> WriteResult:
        return store.put_feedback(
            feedback,
            request_id=request_id,
            expected_generation=expected_generation,
        )

    def execute_conflict(expected_generation: int) -> WriteResult:
        return store.put_feedback(
            replace(feedback, reason="Different maintainer decision context."),
            request_id=request_id,
            expected_generation=expected_generation,
        )

    return ReplayCase(
        generation_kind="feedback",
        initial_expected_generation=_generation(store, "feedback"),
        execute=execute,
        execute_conflict=execute_conflict,
        target_projection=lambda: store.get_feedback(feedback.feedback_id),
    )


def _transition_feedback_case(store: MemoryStore) -> ReplayCase:
    feedback = _feedback("transition-feedback")
    store.put_feedback(
        feedback,
        request_id=stable_request_id("setup-feedback", feedback.feedback_id),
    )
    request_id = stable_request_id(
        "matrix-transition-feedback",
        feedback.feedback_id,
    )

    def execute_with_reason(expected_generation: int, reason: str) -> WriteResult:
        return store.transition_feedback(
            feedback.feedback_id,
            expected_status=FeedbackStatus.RECORDED,
            new_status=FeedbackStatus.REVOKED,
            action="feedback_revoked",
            actor_id="amy",
            reason_code="maintainer_correction",
            reason=reason,
            request_id=request_id,
            expected_generation=expected_generation,
        )

    return ReplayCase(
        generation_kind="feedback",
        initial_expected_generation=_generation(store, "feedback"),
        execute=lambda expected: execute_with_reason(expected, "Feedback withdrawn."),
        execute_conflict=lambda expected: execute_with_reason(
            expected,
            "Different withdrawal rationale.",
        ),
        target_projection=lambda: store.get_feedback(feedback.feedback_id),
    )


def _put_knowledge_entry_case(store: MemoryStore) -> ReplayCase:
    entry = _knowledge(store, "put-knowledge-entry")
    request_id = stable_request_id("matrix-put-knowledge", entry.entry_id)

    def execute(expected_generation: int) -> WriteResult:
        return store.put_knowledge_entry(
            entry,
            request_id=request_id,
            expected_generation=expected_generation,
        )

    def execute_conflict(expected_generation: int) -> WriteResult:
        return store.put_knowledge_entry(
            replace(entry, artifact_schema="symbol_index_v2"),
            request_id=request_id,
            expected_generation=expected_generation,
        )

    return ReplayCase(
        generation_kind="knowledge",
        initial_expected_generation=_generation(store, "knowledge"),
        execute=execute,
        execute_conflict=execute_conflict,
        target_projection=lambda: store.get_knowledge_entry(entry.entry_id),
    )


def _delete_knowledge_entry_case(store: MemoryStore) -> ReplayCase:
    entry = _knowledge(store, "delete-knowledge-entry")
    alternate_entry = _knowledge(store, "delete-knowledge-entry-alternate")
    for prefix, item in (
        ("setup-delete-target", entry),
        ("setup-delete-alternate", alternate_entry),
    ):
        store.put_knowledge_entry(
            item,
            request_id=stable_request_id(prefix, item.entry_id),
        )
    request_id = stable_request_id("matrix-delete-knowledge", entry.entry_id)

    def delete(subject_id: str, expected_generation: int) -> WriteResult:
        return store.delete_knowledge_entry(
            subject_id,
            request_id=request_id,
            expected_generation=expected_generation,
        )

    return ReplayCase(
        generation_kind="knowledge",
        initial_expected_generation=_generation(store, "knowledge"),
        execute=lambda expected: delete(entry.entry_id, expected),
        execute_conflict=lambda expected: delete(alternate_entry.entry_id, expected),
        target_projection=lambda: (
            store.find_knowledge_entry(entry.entry_id),
            store.get_knowledge_entry(alternate_entry.entry_id),
        ),
    )


CASES: tuple[tuple[str, CaseFactory], ...] = (
    ("put_candidate_with_authority_receipt", _put_candidate_case),
    ("transition_candidate_auto_timestamp", _transition_candidate_case),
    ("put_source_bundle", _put_source_bundle_case),
    ("approve_candidate_with_source_bundle", _approve_with_source_bundle_case),
    ("approve_candidate", _approve_candidate_case),
    ("transition_record", _transition_record_case),
    ("put_feedback", _put_feedback_case),
    ("transition_feedback", _transition_feedback_case),
    ("put_knowledge_entry", _put_knowledge_entry_case),
    ("delete_knowledge_entry", _delete_knowledge_entry_case),
)


@pytest.mark.parametrize(
    ("case_name", "case_factory"),
    CASES,
    ids=[name for name, _ in CASES],
)
def test_v2_request_id_semantic_replay_ignores_stale_cas_without_duplication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    case_name: str,
    case_factory: CaseFactory,
) -> None:
    store = MemoryStore(tmp_path / case_name)
    case = case_factory(store)
    runtime_timestamps = iter(
        (
            "2026-07-14T12:00:01Z",
            "2026-07-14T12:00:02Z",
            "2026-07-14T12:00:03Z",
        )
    )
    monkeypatch.setattr(
        memory_store_module,
        "_utc_now",
        lambda: next(runtime_timestamps),
    )

    first = case.execute(case.initial_expected_generation)

    assert first.applied
    assert not first.replayed
    generation_after_first = _generation(store, case.generation_kind)
    assert generation_after_first > case.initial_expected_generation

    _advance_generation(store, case.generation_kind, case_name)
    current_generation = _generation(store, case.generation_kind)
    assert current_generation > generation_after_first

    state_before_replay = _public_state(store, case.target_projection)
    replay = case.execute(generation_after_first)

    assert generation_after_first != case.initial_expected_generation
    assert generation_after_first < current_generation
    assert replay.replayed
    assert not replay.applied
    assert replay.operation == first.operation
    assert replay.subject_id == first.subject_id
    assert replay.event_id == first.event_id
    assert replay.generations == first.generations
    assert _public_state(store, case.target_projection) == state_before_replay

    with pytest.raises(MemoryStoreConflictError, match="reused"):
        case.execute_conflict(generation_after_first)

    assert _public_state(store, case.target_projection) == state_before_replay
    store.validate_integrity()
