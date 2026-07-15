from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import threading
from typing import Optional, Tuple

import pytest
import review_agent.memory_lifecycle as memory_lifecycle_module

from conftest import run_git
from review_agent.memory_identity import repository_key
from review_agent.memory_lifecycle import (
    ApprovalResult,
    CandidateLifecycleResult,
    CandidateDedupeKind,
    ExpiryEvaluationStatus,
    MemoryLifecycle,
    MemoryLifecycleError,
    MemoryLifecycleErrorCode,
    RecordAuditStatus,
    TargetHeadApplicabilityEvaluator,
    build_canonical_source_bundle,
    evaluate_expiry_conditions,
)
from review_agent.memory_models import (
    Applicability,
    CandidateStatus,
    DurableMemoryRecord,
    ExpiryCondition,
    ExpiryConditionKind,
    GitCommitSourceRef,
    MemoryCandidate,
    MemoryConfidence,
    MemoryKind,
    MemoryScope,
    Producer,
    ProducerType,
    RecordStatus,
    RepositoryRangeSourceRef,
    Sensitivity,
    SourceRef,
    ValidityPolicy,
    stable_event_id,
    stable_request_id,
)
from review_agent.memory_sources import (
    SourceValidationCode,
    SourceValidator,
    TrustedCandidateProvenance,
    candidate_authority_resolution_hash,
    repository_range_hash,
)
from review_agent.memory_store import (
    MemoryStore,
    MemoryStoreConflictError,
)
from review_agent.revision import RevisionResolver


NOW = "2026-07-14T08:00:00Z"
LATER = "2026-07-14T08:01:00Z"
REVIEW_ID = "review-memory-lifecycle"


def _head(repo: Path) -> str:
    return run_git(repo, "rev-parse", "HEAD")


def _repository_key(repo: Path) -> str:
    resolver = RevisionResolver()
    return repository_key(resolver.repository_identity(repo))


def _range_source(
    repo: Path,
    revision: str,
    *,
    path: str = "app.py",
    line_start: int = 1,
    line_end: int = 2,
) -> RepositoryRangeSourceRef:
    return RepositoryRangeSourceRef(
        revision=revision,
        path=path,
        line_start=line_start,
        line_end=line_end,
        content_hash=repository_range_hash(
            repo,
            revision,
            path,
            line_start,
            line_end,
        ),
    )


def _candidate(
    repo: Path,
    revision: str,
    *,
    source_refs: Optional[Tuple[SourceRef, ...]] = None,
    statement: str = "Arithmetic changes must preserve exact addition semantics.",
    status: CandidateStatus = CandidateStatus.PROPOSED,
    sensitivity: Sensitivity = Sensitivity.NORMAL,
    producer_type: ProducerType = ProducerType.MODEL,
    review_id: str = REVIEW_ID,
    validity_policies: Tuple[ValidityPolicy, ...] = (
        ValidityPolicy.SOURCE_CONTENT_HASH,
    ),
    scope: Optional[MemoryScope] = None,
    created_at: str = NOW,
) -> MemoryCandidate:
    refs = source_refs or (_range_source(repo, revision),)
    return MemoryCandidate(
        repository_key=_repository_key(repo),
        kind=MemoryKind.REVIEW_RULE,
        statement=statement,
        scope=scope or MemoryScope(paths=("app.py",)),
        source_refs=refs,
        valid_from_sha=revision,
        validity_policies=validity_policies,
        confidence=MemoryConfidence.HIGH,
        sensitivity=sensitivity,
        policy_effect=None,
        producer=Producer(producer_type, "memory-curator", "1.0"),
        origin_review_id=review_id,
        status=status,
        created_at=created_at,
    )


def _provenance(
    candidate: MemoryCandidate,
    target_head: str,
    *,
    origin: ProducerType = ProducerType.MODEL,
    allow_sources: bool = True,
) -> TrustedCandidateProvenance:
    repository_key_value = candidate.repository_key
    return TrustedCandidateProvenance(
        origin=origin,
        review_id=candidate.origin_review_id,
        target_head_sha=target_head,
        locator_repository_key=repository_key_value,
        authority_repository_key=repository_key_value,
        authority_resolution_hash=candidate_authority_resolution_hash(
            repository_key_value,
            repository_key_value,
        ),
        allowed_source_refs=(candidate.source_refs if allow_sources else ()),
    )


def _lifecycle(repo: Path, tmp_path: Path) -> tuple[MemoryLifecycle, MemoryStore]:
    store = MemoryStore(tmp_path / "memory")
    return MemoryLifecycle(store, SourceValidator(repo)), store


def _submit(
    lifecycle: MemoryLifecycle,
    candidate: MemoryCandidate,
    *,
    provenance: TrustedCandidateProvenance,
    label: str,
):
    return lifecycle.submit_candidate(
        candidate,
        runtime_provenance=provenance,
        request_id=stable_request_id("lifecycle-submit", label),
    )


def _approve(
    lifecycle: MemoryLifecycle,
    candidate: MemoryCandidate,
    *,
    provenance: TrustedCandidateProvenance,
    label: str,
    expected_generation: Optional[int] = None,
    expiry_conditions: Tuple[ExpiryCondition, ...] = (),
):
    return lifecycle.approve_candidate(
        candidate.candidate_id,
        runtime_provenance=provenance,
        actor="amy",
        reason="Maintainer verified this durable project rule.",
        request_id=stable_request_id("lifecycle-approve", label),
        created_at=LATER,
        expected_generation=expected_generation,
        expiry_conditions=expiry_conditions,
    )


def _detached_record(
    candidate: MemoryCandidate,
    *,
    status: RecordStatus = RecordStatus.ACTIVE,
    expiry_conditions: Tuple[ExpiryCondition, ...] = (),
) -> DurableMemoryRecord:
    return DurableMemoryRecord(
        candidate_id=candidate.candidate_id,
        repository_key=candidate.repository_key,
        kind=candidate.kind,
        statement=candidate.statement,
        scope=candidate.scope,
        source_refs=candidate.source_refs,
        source_bundle_hash="a" * 64,
        valid_from_sha=candidate.valid_from_sha,
        validity_policies=candidate.validity_policies,
        confidence=candidate.confidence,
        sensitivity=candidate.sensitivity,
        policy_effect=candidate.policy_effect,
        approved_by="amy",
        approval_event_id=stable_event_id("detached-record", candidate.candidate_id),
        status=status,
        created_at=LATER,
        expiry_conditions=expiry_conditions,
    )


def test_submission_uses_trusted_runtime_origin_and_reaches_pending(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    lifecycle, store = _lifecycle(git_repo, tmp_path)
    sha = _head(git_repo)

    # Persisted producer metadata says MODEL, but trusted Runtime origin is LOCAL.
    # Candidate metadata must not be allowed to choose validation authority.
    candidate = _candidate(git_repo, sha, producer_type=ProducerType.MODEL)
    trusted_local = _provenance(
        candidate,
        sha,
        origin=ProducerType.LOCAL,
        allow_sources=False,
    )
    result = _submit(
        lifecycle,
        candidate,
        provenance=trusted_local,
        label="trusted-local",
    )

    assert result.status is CandidateStatus.PENDING_APPROVAL
    assert result.validation.valid
    assert result.persisted
    assert result.dedupe.kind is CandidateDedupeKind.UNIQUE
    assert store.get_candidate(candidate.candidate_id).status is CandidateStatus.PENDING_APPROVAL
    receipt = store.select_candidate_authority_receipt(
        candidate.candidate_id,
        authority_resolution_hash=trusted_local.authority_resolution_hash,
    )
    assert receipt.origin is ProducerType.LOCAL
    assert receipt.proposal_head_sha == sha
    assert [event.action for event in store.list_events(candidate.repository_key)] == [
        "candidate_submitted",
        "validate",
        "request_approval",
    ]

    model_authority_candidate = _candidate(
        git_repo,
        sha,
        statement="A second rule must use Runtime-allowlisted model sources.",
        producer_type=ProducerType.LOCAL,
    )
    denied = _submit(
        lifecycle,
        model_authority_candidate,
        provenance=_provenance(
            model_authority_candidate,
            sha,
            origin=ProducerType.MODEL,
            allow_sources=False,
        ),
        label="trusted-model-denied",
    )

    assert denied.status is CandidateStatus.REJECTED
    assert SourceValidationCode.SOURCE_NOT_ALLOWLISTED in {
        issue.code for issue in denied.validation.issues
    }
    assert store.get_candidate(model_authority_candidate.candidate_id).status is CandidateStatus.REJECTED


def test_approval_restores_the_stored_runtime_authority_context(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    lifecycle, store = _lifecycle(git_repo, tmp_path)
    sha = _head(git_repo)
    candidate = _candidate(git_repo, sha, producer_type=ProducerType.MODEL)
    trusted_local = _provenance(
        candidate,
        sha,
        origin=ProducerType.LOCAL,
        allow_sources=False,
    )
    _submit(
        lifecycle,
        candidate,
        provenance=trusted_local,
        label="authority-restore",
    )

    with pytest.raises(MemoryLifecycleError, match="authority"):
        lifecycle.approve_candidate(
            candidate.candidate_id,
            runtime_provenance=_provenance(
                candidate,
                sha,
                origin=ProducerType.MODEL,
                allow_sources=True,
            ),
            actor="amy",
            reason="Maintainer verified this durable project rule.",
            request_id=stable_request_id("approve", "wrong-runtime-origin"),
            created_at=LATER,
        )

    assert store.get_candidate(candidate.candidate_id).status is (
        CandidateStatus.PENDING_APPROVAL
    )
    approved = _approve(
        lifecycle,
        candidate,
        provenance=trusted_local,
        label="authority-restore",
    )
    assert approved.record.status is RecordStatus.ACTIVE


def test_invalid_is_rejected_but_blocked_content_is_never_persisted(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    lifecycle, store = _lifecycle(git_repo, tmp_path)
    sha = _head(git_repo)
    invalid_ref = replace(_range_source(git_repo, sha), content_hash="0" * 64)
    invalid = _candidate(git_repo, sha, source_refs=(invalid_ref,))

    rejected = _submit(
        lifecycle,
        invalid,
        provenance=_provenance(invalid, sha),
        label="invalid",
    )

    assert rejected.status is CandidateStatus.REJECTED
    assert rejected.persisted
    assert SourceValidationCode.HASH_MISMATCH in {
        issue.code for issue in rejected.validation.issues
    }
    assert store.get_candidate(invalid.candidate_id).status is CandidateStatus.REJECTED

    blocked = _candidate(
        git_repo,
        sha,
        statement="Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
        sensitivity=Sensitivity.NORMAL,
    )
    blocked_result = _submit(
        lifecycle,
        blocked,
        provenance=_provenance(blocked, sha),
        label="blocked",
    )

    assert blocked_result.status is CandidateStatus.REJECTED
    assert not blocked_result.persisted
    assert not blocked_result.validation.retain_content
    assert store.find_candidate(blocked.candidate_id) is None
    assert blocked.statement not in blocked_result.validation.to_json()


def test_exact_content_and_rejected_duplicates_are_deterministic(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    lifecycle, store = _lifecycle(git_repo, tmp_path)
    first_sha = _head(git_repo)
    original = _candidate(git_repo, first_sha)
    provenance = _provenance(original, first_sha)
    first = _submit(lifecycle, original, provenance=provenance, label="original")
    replay = _submit(lifecycle, original, provenance=provenance, label="original")

    assert first.status is CandidateStatus.PENDING_APPROVAL
    assert replay.status is CandidateStatus.PENDING_APPROVAL
    assert replay.dedupe.kind is CandidateDedupeKind.EXACT_REPLAY
    event_count_before_approval = len(store.list_events(original.repository_key))
    _approve(lifecycle, original, provenance=provenance, label="original")

    run_git(git_repo, "commit", "--allow-empty", "-m", "descendant")
    descendant = _head(git_repo)
    duplicate = _candidate(
        git_repo,
        descendant,
        source_refs=original.source_refs,
        created_at=LATER,
    )
    duplicate_result = _submit(
        lifecycle,
        duplicate,
        provenance=_provenance(duplicate, descendant),
        label="active-duplicate",
    )

    assert duplicate.content_fingerprint == original.content_fingerprint
    assert duplicate.candidate_id != original.candidate_id
    assert duplicate_result.status is CandidateStatus.REJECTED
    assert duplicate_result.dedupe.kind is CandidateDedupeKind.ACTIVE_DUPLICATE

    enhanced = _candidate(
        git_repo,
        descendant,
        source_refs=original.source_refs + (GitCommitSourceRef(descendant),),
        created_at=LATER,
    )
    enhanced_result = _submit(
        lifecycle,
        enhanced,
        provenance=_provenance(enhanced, descendant),
        label="enhanced",
    )

    assert enhanced_result.status is CandidateStatus.PENDING_APPROVAL
    assert enhanced_result.dedupe.kind is CandidateDedupeKind.ENHANCED_PROVENANCE

    lifecycle.reject_candidate(
        enhanced.candidate_id,
        actor="amy",
        reason_code="insufficient_evidence",
        reason="The additional commit reference does not establish the rule.",
        request_id=stable_request_id("reject", "enhanced"),
        created_at=LATER,
    )
    unchanged_reproposal = replace(
        enhanced,
        origin_review_id="review-memory-reproposal",
        created_at="2026-07-14T08:02:00Z",
    )
    suppressed = _submit(
        lifecycle,
        unchanged_reproposal,
        provenance=_provenance(unchanged_reproposal, descendant),
        label="unchanged-rejected",
    )

    assert suppressed.status is CandidateStatus.REJECTED
    assert suppressed.dedupe.kind is CandidateDedupeKind.REJECTED_UNCHANGED
    assert len(store.list_events(original.repository_key)) > event_count_before_approval


def test_concurrent_same_fingerprint_submissions_have_one_pending_authority(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle, store = _lifecycle(git_repo, tmp_path)
    sha = _head(git_repo)
    first = _candidate(git_repo, sha)
    second = replace(first, confidence=MemoryConfidence.MEDIUM)
    assert first.candidate_id != second.candidate_id
    assert first.content_fingerprint == second.content_fingerprint

    # Rendezvous before either authority transaction starts. The transaction
    # itself is intentionally serialized, so a barrier inside it would deadlock.
    transaction_gate = threading.Barrier(2, timeout=5)
    original_transaction = store.candidate_submission_transaction

    @contextmanager
    def synchronized_transaction():
        transaction_gate.wait()
        with original_transaction():
            yield

    monkeypatch.setattr(
        store,
        "candidate_submission_transaction",
        synchronized_transaction,
    )
    candidates = {"first": first, "second": second}
    results: dict[str, CandidateLifecycleResult] = {}
    errors: list[Exception] = []

    def submit(label: str) -> None:
        candidate = candidates[label]
        try:
            results[label] = lifecycle.submit_candidate(
                candidate,
                runtime_provenance=_provenance(candidate, sha),
                request_id=stable_request_id("concurrent-submit", label),
            )
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)

    submitters = [
        threading.Thread(target=submit, args=(label,), name="submit-" + label)
        for label in candidates
    ]
    for submitter in submitters:
        submitter.start()
    for submitter in submitters:
        submitter.join(timeout=10)

    assert all(not submitter.is_alive() for submitter in submitters)
    assert errors == []
    assert set(results) == set(candidates)
    pending = [
        result
        for result in results.values()
        if result.status is CandidateStatus.PENDING_APPROVAL
    ]
    rejected = [
        result
        for result in results.values()
        if result.status is CandidateStatus.REJECTED
    ]
    assert len(pending) == len(rejected) == 1
    winner, duplicate = pending[0], rejected[0]
    assert winner.dedupe.kind is CandidateDedupeKind.UNIQUE
    assert duplicate.dedupe.kind is CandidateDedupeKind.PENDING_DUPLICATE
    assert duplicate.dedupe.related_candidate_id == winner.candidate_id

    stored = {
        candidate.candidate_id: candidate
        for candidate in store.list_candidates(first.repository_key)
    }
    assert stored[winner.candidate_id].status is CandidateStatus.PENDING_APPROVAL
    assert stored[duplicate.candidate_id].status is CandidateStatus.REJECTED
    assert len(store.list_candidate_authority_receipts(winner.candidate_id)) == 1
    assert len(store.list_candidate_authority_receipts(duplicate.candidate_id)) == 1

    events = store.list_events(first.repository_key)
    winner_events = [
        event for event in events if event.subject_id == winner.candidate_id
    ]
    duplicate_events = [
        event for event in events if event.subject_id == duplicate.candidate_id
    ]
    assert [event.action for event in winner_events] == [
        "candidate_submitted",
        "validate",
        "request_approval",
    ]
    assert [event.action for event in duplicate_events] == [
        "candidate_submitted",
        "duplicate",
    ]
    assert [event.subject_id for event in events] == [
        winner.candidate_id,
        winner.candidate_id,
        winner.candidate_id,
        duplicate.candidate_id,
        duplicate.candidate_id,
    ]
    assert store.verify_event_chain(first.repository_key) == len(events) == 5
    assert store.validate_integrity().candidate_count == 2


def test_approval_revalidates_and_atomically_materializes_pinned_bundle(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    lifecycle, store = _lifecycle(git_repo, tmp_path)
    sha = _head(git_repo)
    candidate = _candidate(
        git_repo,
        sha,
        sensitivity=Sensitivity.LOCAL_ONLY,
    )
    provenance = _provenance(candidate, sha)
    submitted = _submit(lifecycle, candidate, provenance=provenance, label="approve")
    expected_generation = submitted.write_results[-1].generations.memory_generation

    approval = _approve(
        lifecycle,
        candidate,
        provenance=provenance,
        label="approve",
        expected_generation=expected_generation,
    )

    assert approval.record.status is RecordStatus.ACTIVE
    assert approval.record.approved_by == "amy"
    assert approval.bundle.candidate_id == candidate.candidate_id
    assert approval.bundle.source_refs == candidate.source_refs
    assert approval.validation.valid
    assert not approval.validation.remote_sendable
    assert store.get_candidate(candidate.candidate_id).status is CandidateStatus.APPROVED
    assert store.get_record(approval.record.memory_id) == approval.record
    assert store.get_source_bundle(approval.bundle.bundle_hash) == approval.bundle
    assert store.read_blob(approval.bundle.blob_hash) == approval.bundle_payload

    payload = json.loads(approval.bundle_payload)
    assert payload["schema"] == "memory_source_bundle_v1"
    assert payload["candidate_id"] == candidate.candidate_id
    assert payload["validation_report_hash"] == approval.validation.report_hash
    assert "statement" not in payload
    assert candidate.statement not in approval.bundle_payload.decode("utf-8")
    assert approval.bundle_payload == build_canonical_source_bundle(
        candidate,
        approval.validation,
    )

    with store.open_connection(read_only=True) as connection:
        pin = connection.execute(
            """
            SELECT pin_type, pin_id FROM blob_pins
            WHERE blob_hash = ?
            """,
            (approval.bundle.blob_hash,),
        ).fetchone()
    assert tuple(pin) == ("source_bundle", approval.bundle.bundle_hash)

    replay = _approve(
        lifecycle,
        candidate,
        provenance=provenance,
        label="approve",
    )
    assert replay.record == approval.record
    assert replay.write_result.replayed
    assert store.count_records(candidate.repository_key) == 1


def test_concurrent_approvals_converge_on_the_winning_final_record(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle, store = _lifecycle(git_repo, tmp_path)
    sha = _head(git_repo)
    candidate = _candidate(
        git_repo,
        sha,
        statement="Concurrent approval has one durable authority result.",
    )
    provenance = _provenance(candidate, sha)
    expiry = ExpiryCondition(
        ExpiryConditionKind.AT_TIME,
        "2027-01-01T00:00:00Z",
    )
    _submit(
        lifecycle,
        candidate,
        provenance=provenance,
        label="concurrent-approval-submit",
    )

    # Both callers finish validation and blob materialization before either
    # enters the Store's serialized approval transaction.
    approval_gate = threading.Barrier(2, timeout=5)
    original_approve = store.approve_candidate_with_source_bundle

    def synchronized_approve(*args, **kwargs):
        approval_gate.wait()
        return original_approve(*args, **kwargs)

    monkeypatch.setattr(
        store,
        "approve_candidate_with_source_bundle",
        synchronized_approve,
    )
    request_ids = {
        label: stable_request_id("concurrent-approval", label)
        for label in ("first", "second")
    }
    results: dict[str, ApprovalResult] = {}
    errors: list[Exception] = []

    def approve(label: str) -> None:
        try:
            results[label] = lifecycle.approve_candidate(
                candidate.candidate_id,
                runtime_provenance=provenance,
                actor="amy",
                reason="Maintainer verified this durable project rule.",
                request_id=request_ids[label],
                created_at=LATER,
                expiry_conditions=(expiry,),
            )
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)

    approvers = [
        threading.Thread(target=approve, args=(label,), name="approve-" + label)
        for label in request_ids
    ]
    for approver in approvers:
        approver.start()
    for approver in approvers:
        approver.join(timeout=10)

    assert all(not approver.is_alive() for approver in approvers)
    assert errors == []
    assert set(results) == set(request_ids)
    applied = [
        (label, result)
        for label, result in results.items()
        if result.write_result.applied
    ]
    converged = [
        (label, result)
        for label, result in results.items()
        if result.converged
    ]
    assert len(applied) == len(converged) == 1
    winner_label, winner = applied[0]
    loser_label, loser = converged[0]
    assert winner_label != loser_label
    assert not loser.write_result.applied
    assert not loser.write_result.replayed
    assert loser.record == winner.record == store.get_record(winner.record.memory_id)
    assert loser.bundle == winner.bundle
    assert loser.bundle_payload == winner.bundle_payload
    assert loser.write_result.operation == "transition_candidate"
    assert loser.write_result.subject_id == candidate.candidate_id
    assert loser.write_result.event_id is None
    assert winner.write_result.event_id == winner.record.approval_event_id

    with store.open_connection(read_only=True) as connection:
        receipt_rows = connection.execute(
            """
            SELECT request_id, operation, subject_id, event_id, result_json
            FROM outbox_receipts
            WHERE request_id IN (?, ?)
            ORDER BY request_id
            """,
            (request_ids[winner_label], request_ids[loser_label]),
        ).fetchall()
        receipt_count_before_replay = connection.execute(
            "SELECT COUNT(*) FROM outbox_receipts"
        ).fetchone()[0]
    receipts = {row["request_id"]: row for row in receipt_rows}
    assert set(receipts) == set(request_ids.values())
    assert receipts[request_ids[winner_label]]["operation"] == (
        "approve_candidate_with_source_bundle"
    )
    assert receipts[request_ids[winner_label]]["event_id"] == (
        winner.record.approval_event_id
    )
    loser_receipt = receipts[request_ids[loser_label]]
    assert loser_receipt["operation"] == "transition_candidate"
    assert loser_receipt["subject_id"] == candidate.candidate_id
    assert loser_receipt["event_id"] is None
    assert json.loads(loser_receipt["result_json"])["applied"] is False

    approval_events = [
        event
        for event in store.list_events(
            candidate.repository_key,
            subject_type="candidate",
            subject_id=candidate.candidate_id,
        )
        if event.action == "approve"
    ]
    assert len(approval_events) == 1
    assert approval_events[0].request_id == request_ids[winner_label]
    assert approval_events[0].event_id == winner.record.approval_event_id
    assert store.count_records(candidate.repository_key) == 1
    assert store.get_candidate(candidate.candidate_id).status is CandidateStatus.APPROVED
    assert store.verify_event_chain(candidate.repository_key) == 5
    assert store.validate_integrity().record_count == 1

    events_before_replay = store.list_events(candidate.repository_key)
    generations_before_replay = store.get_generations(candidate.repository_key)
    replay = lifecycle.approve_candidate(
        candidate.candidate_id,
        runtime_provenance=provenance,
        actor="amy",
        reason="Maintainer verified this durable project rule.",
        request_id=request_ids[loser_label],
        created_at=LATER,
        expiry_conditions=(expiry,),
    )

    assert replay.converged
    assert replay.record == loser.record
    assert replay.bundle == loser.bundle
    assert replay.bundle_payload == loser.bundle_payload
    assert replay.write_result.operation == loser.write_result.operation
    assert replay.write_result.subject_id == loser.write_result.subject_id
    assert replay.write_result.event_id is None
    assert not replay.write_result.applied
    assert replay.write_result.replayed
    assert replace(replay.write_result, replayed=False) == loser.write_result
    assert store.list_events(candidate.repository_key) == events_before_replay
    assert store.get_generations(candidate.repository_key) == generations_before_replay
    assert store.count_records(candidate.repository_key) == 1
    with store.open_connection(read_only=True) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM outbox_receipts"
        ).fetchone()[0] == receipt_count_before_replay

    with pytest.raises(MemoryStoreConflictError, match="request ID"):
        lifecycle.approve_candidate(
            candidate.candidate_id,
            runtime_provenance=provenance,
            actor="amy",
            reason="A conflicting decision cannot reuse the loser request ID.",
            request_id=request_ids[loser_label],
            created_at=LATER,
            expiry_conditions=(expiry,),
        )
    with pytest.raises(MemoryStoreConflictError, match="expiry conditions"):
        lifecycle.approve_candidate(
            candidate.candidate_id,
            runtime_provenance=provenance,
            actor="amy",
            reason="Maintainer verified this durable project rule.",
            request_id=request_ids[loser_label],
            created_at=LATER,
            expiry_conditions=(
                ExpiryCondition(
                    ExpiryConditionKind.AT_TIME,
                    "2027-02-01T00:00:00Z",
                ),
            ),
        )
    assert store.list_events(candidate.repository_key) == events_before_replay
    assert store.get_generations(candidate.repository_key) == generations_before_replay
    assert store.count_records(candidate.repository_key) == 1
    with store.open_connection(read_only=True) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM outbox_receipts"
        ).fetchone()[0] == receipt_count_before_replay


def test_human_reject_and_revoke_require_reason_and_are_request_idempotent(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    lifecycle, store = _lifecycle(git_repo, tmp_path)
    sha = _head(git_repo)
    rejected_candidate = _candidate(
        git_repo,
        sha,
        statement="A candidate selected for explicit human rejection.",
    )
    rejected_provenance = _provenance(rejected_candidate, sha)
    _submit(
        lifecycle,
        rejected_candidate,
        provenance=rejected_provenance,
        label="human-reject",
    )

    with pytest.raises(MemoryLifecycleError, match="actor"):
        lifecycle.reject_candidate(
            rejected_candidate.candidate_id,
            actor="",
            reason_code="wrong_scope",
            reason="Not a durable project rule.",
            request_id=stable_request_id("reject", "missing-actor"),
        )
    with pytest.raises(MemoryLifecycleError, match="reason"):
        lifecycle.reject_candidate(
            rejected_candidate.candidate_id,
            actor="amy",
            reason_code="wrong_scope",
            reason="",
            request_id=stable_request_id("reject", "missing-reason"),
        )

    rejection_request = stable_request_id("reject", "human")
    rejected = lifecycle.reject_candidate(
        rejected_candidate.candidate_id,
        actor="amy",
        reason_code="wrong_scope",
        reason="This proposal is specific to one review.",
        request_id=rejection_request,
        created_at=LATER,
    )
    replayed_rejection = lifecycle.reject_candidate(
        rejected_candidate.candidate_id,
        actor="amy",
        reason_code="wrong_scope",
        reason="This proposal is specific to one review.",
        request_id=rejection_request,
        created_at=LATER,
    )
    assert rejected.applied
    assert replayed_rejection.replayed

    active_candidate = _candidate(
        git_repo,
        sha,
        statement="A durable rule selected for explicit revocation.",
    )
    active_provenance = _provenance(active_candidate, sha)
    _submit(
        lifecycle,
        active_candidate,
        provenance=active_provenance,
        label="revoke",
    )
    active = _approve(
        lifecycle,
        active_candidate,
        provenance=active_provenance,
        label="revoke",
    ).record
    revoke_request = stable_request_id("revoke", active.memory_id)
    generation = store.get_generations(active.repository_key).memory_generation
    revoked = lifecycle.revoke_record(
        active.memory_id,
        actor="amy",
        reason="The project explicitly retired this rule.",
        request_id=revoke_request,
        expected_generation=generation,
        created_at=LATER,
    )
    replayed_revoke = lifecycle.revoke_record(
        active.memory_id,
        actor="amy",
        reason="The project explicitly retired this rule.",
        request_id=revoke_request,
        expected_generation=generation,
        created_at=LATER,
    )

    assert revoked.applied
    assert replayed_revoke.replayed
    assert store.get_record(active.memory_id).status is RecordStatus.REVOKED
    with pytest.raises(MemoryStoreConflictError, match="request ID"):
        lifecycle.revoke_record(
            active.memory_id,
            actor="amy",
            reason="A different reason cannot reuse the request ID.",
            request_id=revoke_request,
            created_at=LATER,
        )


def test_revalidate_creates_new_immutable_authority_and_supersedes_old(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    lifecycle, store = _lifecycle(git_repo, tmp_path)
    original_sha = _head(git_repo)
    original = _candidate(git_repo, original_sha)
    original_provenance = _provenance(original, original_sha)
    _submit(lifecycle, original, provenance=original_provenance, label="old")
    old_record = _approve(
        lifecycle,
        original,
        provenance=original_provenance,
        label="old",
    ).record
    lifecycle.mark_revalidation_required(
        old_record.memory_id,
        actor="amy",
        reason="The current project line changed the source-bearing scope.",
        request_id=stable_request_id("mark-stale", old_record.memory_id),
        created_at=LATER,
    )

    (git_repo / "app.py").write_text(
        "def add(a, b):\n    return int(a) + int(b)\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "change arithmetic implementation")
    new_sha = _head(git_repo)
    replacement = _candidate(
        git_repo,
        new_sha,
        statement="Arithmetic changes must preserve explicit integer conversion.",
        created_at="2026-07-14T08:02:00Z",
    )
    replacement_provenance = _provenance(replacement, new_sha)
    replacement_expiry = ExpiryCondition(
        ExpiryConditionKind.AT_TIME,
        "2027-03-01T00:00:00Z",
    )
    revalidation_request = stable_request_id("revalidate", old_record.memory_id)

    result = lifecycle.revalidate_record(
        old_record.memory_id,
        replacement,
        runtime_provenance=replacement_provenance,
        actor="amy",
        reason="Maintainer approved refreshed evidence and wording.",
        request_id=revalidation_request,
        created_at="2026-07-14T08:03:00Z",
        expiry_conditions=(replacement_expiry,),
    )

    assert result.record.memory_id != old_record.memory_id
    assert result.record.candidate_id == replacement.candidate_id
    assert result.record.status is RecordStatus.ACTIVE
    assert result.record.expiry_conditions == (replacement_expiry,)
    assert store.get_candidate(replacement.candidate_id).status is CandidateStatus.APPROVED
    assert store.get_record(old_record.memory_id).status is RecordStatus.SUPERSEDED
    assert store.get_record(result.record.memory_id).status is RecordStatus.ACTIVE
    assert store.get_record(old_record.memory_id).statement == old_record.statement
    assert store.get_candidate(original.candidate_id).statement == original.statement

    events = store.list_events(original.repository_key)
    supersede_events = [event for event in events if event.action == "supersede"]
    assert len(supersede_events) == 1
    assert supersede_events[0].subject_id == old_record.memory_id
    assert supersede_events[0].actor_type == "human"
    assert supersede_events[0].actor_id == "amy"

    generations = store.get_generations(original.repository_key)
    replay = lifecycle.revalidate_record(
        old_record.memory_id,
        replacement,
        runtime_provenance=replacement_provenance,
        actor="amy",
        reason="Maintainer approved refreshed evidence and wording.",
        request_id=revalidation_request,
        created_at="2026-07-14T08:03:00Z",
        expiry_conditions=(replacement_expiry,),
    )
    assert replay.record == result.record
    assert replay.write_result.replayed
    assert store.get_generations(original.repository_key) == generations
    with pytest.raises(MemoryStoreConflictError, match="expiry conditions"):
        lifecycle.revalidate_record(
            old_record.memory_id,
            replacement,
            runtime_provenance=replacement_provenance,
            actor="amy",
            reason="Maintainer approved refreshed evidence and wording.",
            request_id=revalidation_request,
            created_at="2026-07-14T08:03:00Z",
            expiry_conditions=(
                ExpiryCondition(
                    ExpiryConditionKind.AT_TIME,
                    "2027-04-01T00:00:00Z",
                ),
            ),
        )


@pytest.mark.parametrize("mutation", ["missing", "tampered"])
def test_missing_or_tampered_bundle_makes_record_unauditable(
    git_repo: Path,
    tmp_path: Path,
    mutation: str,
) -> None:
    lifecycle, store = _lifecycle(git_repo, tmp_path)
    sha = _head(git_repo)
    candidate = _candidate(git_repo, sha)
    provenance = _provenance(candidate, sha)
    _submit(lifecycle, candidate, provenance=provenance, label=mutation)
    approval = _approve(
        lifecycle,
        candidate,
        provenance=provenance,
        label=mutation,
    )

    assert lifecycle.audit_record(approval.record.memory_id).status is RecordAuditStatus.AUDITABLE
    blob_path = store.blob_path(approval.bundle.blob_hash)
    if mutation == "missing":
        blob_path.unlink()
    else:
        blob_path.write_bytes(b'{"tampered":true}')

    audit = lifecycle.audit_record(approval.record.memory_id)

    assert audit.status is RecordAuditStatus.UNAUDITABLE
    assert not audit.auditable
    assert audit.memory_id == approval.record.memory_id
    assert audit.bundle is None
    assert audit.payload is None


def test_bundle_is_audit_only_and_never_proves_target_head_applicability(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    lifecycle, _ = _lifecycle(git_repo, tmp_path)
    original_sha = _head(git_repo)
    candidate = _candidate(git_repo, original_sha)
    provenance = _provenance(candidate, original_sha)
    _submit(lifecycle, candidate, provenance=provenance, label="audit-only")
    record = _approve(
        lifecycle,
        candidate,
        provenance=provenance,
        label="audit-only",
    ).record
    assert lifecycle.audit_record(record.memory_id).auditable

    (git_repo / "app.py").write_text(
        "def add(a, b):\n    return a - b\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "change source after approval")
    target_head = _head(git_repo)

    decision = TargetHeadApplicabilityEvaluator(
        git_repo,
        SourceValidator(git_repo),
    ).evaluate(record, target_head=target_head)

    assert decision.applicability is Applicability.SOURCE_CHANGED
    assert decision.requires_revalidation
    assert lifecycle.audit_record(record.memory_id).auditable


def test_target_head_applicability_covers_ancestry_missing_scope_and_manual_policy(
    git_repo: Path,
) -> None:
    main_branch = run_git(git_repo, "branch", "--show-current")
    root_sha = _head(git_repo)
    run_git(git_repo, "commit", "--allow-empty", "-m", "valid-from")
    valid_from = _head(git_repo)
    source = _range_source(git_repo, valid_from)
    candidate = _candidate(
        git_repo,
        valid_from,
        source_refs=(source,),
        scope=MemoryScope(paths=("app.py",)),
    )
    record = _detached_record(candidate)
    evaluator = TargetHeadApplicabilityEvaluator(git_repo, SourceValidator(git_repo))

    assert evaluator.evaluate(record, target_head=valid_from).applicability is Applicability.SELECTED
    assert evaluator.evaluate(record, target_head=root_sha).applicability is Applicability.NOT_YET_VALID

    run_git(git_repo, "checkout", "-b", "diverged", root_sha)
    run_git(git_repo, "commit", "--allow-empty", "-m", "diverged commit")
    diverged_sha = _head(git_repo)
    assert evaluator.evaluate(record, target_head=diverged_sha).applicability is Applicability.LINEAGE_MISMATCH

    run_git(git_repo, "checkout", main_branch)
    (git_repo / "docs.txt").write_text("unrelated\n", encoding="utf-8")
    run_git(git_repo, "add", "docs.txt")
    run_git(git_repo, "commit", "-m", "unrelated descendant")
    unrelated_sha = _head(git_repo)
    assert evaluator.evaluate(record, target_head=unrelated_sha).applicability is Applicability.SELECTED
    assert evaluator.evaluate(
        record,
        target_head=unrelated_sha,
        changed_paths=("docs.txt",),
    ).applicability is Applicability.OUT_OF_SCOPE

    (git_repo / "app.py").unlink()
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "remove source")
    missing_sha = _head(git_repo)
    missing = evaluator.evaluate(record, target_head=missing_sha)
    assert missing.applicability is Applicability.SOURCE_MISSING
    assert missing.requires_revalidation

    manual_candidate = _candidate(
        git_repo,
        valid_from,
        source_refs=(source,),
        validity_policies=(ValidityPolicy.MANUAL_UNTIL_REVOKED,),
    )
    manual_record = _detached_record(manual_candidate)
    assert evaluator.evaluate(manual_record, target_head=missing_sha).applicability is Applicability.SELECTED
    assert evaluator.evaluate(
        replace(manual_record, status=RecordStatus.REVOKED),
        target_head=missing_sha,
    ).applicability is Applicability.REVOKED
    assert evaluator.evaluate(
        replace(manual_record, status=RecordStatus.SUPERSEDED),
        target_head=missing_sha,
    ).applicability is Applicability.SUPERSEDED


def test_scope_change_trigger_requires_revalidation_only_for_matching_diff(
    git_repo: Path,
) -> None:
    valid_from = _head(git_repo)
    candidate = _candidate(
        git_repo,
        valid_from,
        validity_policies=(ValidityPolicy.SCOPE_CHANGE_TRIGGER,),
        scope=MemoryScope(paths=("app.py",)),
    )
    record = _detached_record(candidate)
    evaluator = TargetHeadApplicabilityEvaluator(git_repo, SourceValidator(git_repo))

    (git_repo / "notes.txt").write_text("notes\n", encoding="utf-8")
    run_git(git_repo, "add", "notes.txt")
    run_git(git_repo, "commit", "-m", "outside scope")
    outside_sha = _head(git_repo)
    assert evaluator.evaluate(record, target_head=outside_sha).applicability is Applicability.SELECTED

    (git_repo / "app.py").write_text(
        "def add(a, b):\n    return a + b + 0\n",
        encoding="utf-8",
    )
    run_git(git_repo, "add", "app.py")
    run_git(git_repo, "commit", "-m", "inside scope")
    inside_sha = _head(git_repo)
    decision = evaluator.evaluate(record, target_head=inside_sha)

    assert decision.applicability is Applicability.SOURCE_CHANGED
    assert decision.reason_code == "scope_changed"
    assert decision.requires_revalidation


def test_scope_change_trigger_preserves_dot_prefixed_repository_paths(
    git_repo: Path,
) -> None:
    valid_from = _head(git_repo)
    record = _detached_record(
        _candidate(
            git_repo,
            valid_from,
            validity_policies=(ValidityPolicy.SCOPE_CHANGE_TRIGGER,),
            scope=MemoryScope(paths=(".github/**",)),
        )
    )
    workflow = git_repo / ".github" / "workflows" / "ci.yml"
    workflow.parent.mkdir(parents=True)
    workflow.write_text("name: ci\n", encoding="utf-8")
    run_git(git_repo, "add", ".github/workflows/ci.yml")
    run_git(git_repo, "commit", "-m", "add workflow")

    decision = TargetHeadApplicabilityEvaluator(
        git_repo,
        SourceValidator(git_repo),
    ).evaluate(record, target_head=_head(git_repo))

    assert decision.applicability is Applicability.SOURCE_CHANGED
    assert decision.reason_code == "scope_changed"


def test_scope_change_inspection_failure_is_a_bounded_fail_closed_decision(
    git_repo: Path,
    monkeypatch,
) -> None:
    valid_from = _head(git_repo)
    run_git(git_repo, "commit", "--allow-empty", "-m", "descendant")
    record = _detached_record(
        _candidate(
            git_repo,
            valid_from,
            validity_policies=(ValidityPolicy.SCOPE_CHANGE_TRIGGER,),
        )
    )

    def unavailable(*args, **kwargs):
        raise MemoryLifecycleError(
            "unable to inspect scope changes",
            MemoryLifecycleErrorCode.SOURCE_VALIDATION_FAILED,
        )

    monkeypatch.setattr(memory_lifecycle_module, "_git_changed_paths", unavailable)
    decision = TargetHeadApplicabilityEvaluator(
        git_repo,
        SourceValidator(git_repo),
    ).evaluate(record, target_head=_head(git_repo))

    assert decision.applicability is Applicability.SOURCE_MISSING
    assert decision.reason_code == "scope_change_unavailable"
    assert decision.requires_revalidation


@pytest.mark.parametrize(
    ("evaluated_at", "expected"),
    (
        (
            datetime(2026, 7, 14, 7, 59, 59, tzinfo=timezone.utc),
            ExpiryEvaluationStatus.NOT_DUE,
        ),
        (
            datetime(2026, 7, 14, 16, 0, 0, tzinfo=timezone(timedelta(hours=8))),
            ExpiryEvaluationStatus.DUE,
        ),
        (
            datetime(2026, 7, 14, 8, 0, 1, tzinfo=timezone.utc),
            ExpiryEvaluationStatus.DUE,
        ),
    ),
)
def test_time_expiry_before_boundary_and_after_are_timezone_aware(
    git_repo: Path,
    evaluated_at: datetime,
    expected: ExpiryEvaluationStatus,
) -> None:
    condition = ExpiryCondition(
        ExpiryConditionKind.AT_TIME,
        "2026-07-14T08:00:00Z",
    )

    result = evaluate_expiry_conditions(
        (condition,),
        repository=git_repo,
        target_head=_head(git_repo),
        evaluated_at=evaluated_at,
    )

    assert result.status is expected
    assert result.evaluated_at.endswith("Z")
    if expected is ExpiryEvaluationStatus.DUE:
        assert result.due_condition == condition
        assert result.reason_code == "expiry_time_reached"

    with pytest.raises(MemoryLifecycleError) as error:
        evaluate_expiry_conditions(
            (condition,),
            repository=git_repo,
            target_head=_head(git_repo),
            evaluated_at=datetime(2026, 7, 14, 8, 0, 0),
        )
    assert error.value.code is MemoryLifecycleErrorCode.INVALID_INPUT


def test_commit_expiry_lineage_missing_and_or_semantics(git_repo: Path) -> None:
    main_branch = run_git(git_repo, "branch", "--show-current")
    root = _head(git_repo)
    run_git(git_repo, "commit", "--allow-empty", "-m", "expiry boundary")
    boundary = _head(git_repo)
    run_git(git_repo, "commit", "--allow-empty", "-m", "after expiry boundary")
    descendant = _head(git_repo)
    run_git(git_repo, "checkout", "-b", "expiry-diverged", root)
    run_git(git_repo, "commit", "--allow-empty", "-m", "diverged expiry target")
    diverged = _head(git_repo)
    run_git(git_repo, "checkout", main_branch)

    condition = ExpiryCondition(ExpiryConditionKind.AT_COMMIT, boundary)
    evaluated_at = datetime(2026, 7, 14, 8, 0, tzinfo=timezone.utc)
    expectations = {
        root: ExpiryEvaluationStatus.NOT_DUE,
        boundary: ExpiryEvaluationStatus.DUE,
        descendant: ExpiryEvaluationStatus.DUE,
        diverged: ExpiryEvaluationStatus.UNRESOLVED,
        "f" * 40: ExpiryEvaluationStatus.UNRESOLVED,
    }
    for target, expected in expectations.items():
        result = evaluate_expiry_conditions(
            (condition,),
            repository=git_repo,
            target_head=target,
            evaluated_at=evaluated_at,
        )
        assert result.status is expected
        if expected is ExpiryEvaluationStatus.DUE:
            assert result.reason_code == "expiry_commit_reached"

    missing_commit = ExpiryCondition(ExpiryConditionKind.AT_COMMIT, "e" * 40)
    time_due = ExpiryCondition(
        ExpiryConditionKind.AT_TIME,
        "2026-07-14T08:00:00Z",
    )
    due_wins = evaluate_expiry_conditions(
        (time_due, missing_commit),
        repository=git_repo,
        target_head=descendant,
        evaluated_at=evaluated_at,
    )
    assert due_wins.status is ExpiryEvaluationStatus.DUE
    assert due_wins.due_condition == time_due
    assert due_wins.unresolved_conditions == (missing_commit,)

    future_time = ExpiryCondition(
        ExpiryConditionKind.AT_TIME,
        "2027-07-14T08:00:00Z",
    )
    unresolved_wins = evaluate_expiry_conditions(
        (future_time, missing_commit),
        repository=git_repo,
        target_head=descendant,
        evaluated_at=evaluated_at,
    )
    assert unresolved_wins.status is ExpiryEvaluationStatus.UNRESOLVED


def test_approval_rejects_invalid_commit_boundaries_and_replay_changes(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    lifecycle, store = _lifecycle(git_repo, tmp_path)
    main_branch = run_git(git_repo, "branch", "--show-current")
    before_valid_from = _head(git_repo)
    run_git(git_repo, "commit", "--allow-empty", "-m", "valid from")
    valid_from = _head(git_repo)
    run_git(git_repo, "checkout", "-b", "expiry-boundary-diverged", before_valid_from)
    run_git(git_repo, "commit", "--allow-empty", "-m", "diverged boundary")
    diverged = _head(git_repo)
    run_git(git_repo, "checkout", main_branch)
    run_git(git_repo, "commit", "--allow-empty", "-m", "valid expiry boundary")
    valid_boundary = _head(git_repo)

    invalid_boundaries = {
        "before": before_valid_from,
        "diverged": diverged,
        "missing": "d" * 40,
    }
    for label, boundary in invalid_boundaries.items():
        candidate = _candidate(
            git_repo,
            valid_from,
            statement="Invalid %s expiry boundary must fail closed." % label,
        )
        provenance = _provenance(candidate, valid_boundary)
        _submit(
            lifecycle,
            candidate,
            provenance=provenance,
            label="invalid-expiry-" + label,
        )
        with pytest.raises(MemoryLifecycleError) as error:
            lifecycle.approve_candidate(
                candidate.candidate_id,
                runtime_provenance=provenance,
                actor="amy",
                reason="Maintainer supplied an invalid expiry boundary.",
                request_id=stable_request_id("invalid-expiry", label),
                created_at=LATER,
                expiry_conditions=(
                    ExpiryCondition(ExpiryConditionKind.AT_COMMIT, boundary),
                ),
            )
        assert error.value.code is MemoryLifecycleErrorCode.INVALID_INPUT
        assert store.get_candidate(candidate.candidate_id).status is (
            CandidateStatus.PENDING_APPROVAL
        )

    candidate = _candidate(
        git_repo,
        valid_from,
        statement="Valid expiry boundaries persist on immutable authority.",
    )
    provenance = _provenance(candidate, valid_boundary)
    _submit(lifecycle, candidate, provenance=provenance, label="valid-expiry")
    conditions = (
        ExpiryCondition(ExpiryConditionKind.AT_TIME, "2026-01-01T00:00:00Z"),
        ExpiryCondition(ExpiryConditionKind.AT_COMMIT, valid_boundary),
    )
    request_id = stable_request_id("valid-expiry", candidate.candidate_id)
    approval = lifecycle.approve_candidate(
        candidate.candidate_id,
        runtime_provenance=provenance,
        actor="amy",
        reason="Maintainer approved canonical automatic expiry.",
        request_id=request_id,
        created_at=LATER,
        expiry_conditions=conditions,
    )
    generation = store.get_generations(candidate.repository_key)
    events = store.list_events(candidate.repository_key)
    replay = lifecycle.approve_candidate(
        candidate.candidate_id,
        runtime_provenance=provenance,
        actor="amy",
        reason="Maintainer approved canonical automatic expiry.",
        request_id=request_id,
        created_at=LATER,
        expiry_conditions=tuple(reversed(conditions)),
    )

    assert approval.record.expiry_conditions == tuple(reversed(conditions))
    assert replay.record == approval.record
    assert replay.write_result.replayed
    assert store.get_generations(candidate.repository_key) == generation
    assert store.list_events(candidate.repository_key) == events
    with pytest.raises(MemoryStoreConflictError, match="expiry conditions"):
        lifecycle.approve_candidate(
            candidate.candidate_id,
            runtime_provenance=provenance,
            actor="amy",
            reason="Maintainer approved canonical automatic expiry.",
            request_id=request_id,
            created_at=LATER,
            expiry_conditions=(
                ExpiryCondition(
                    ExpiryConditionKind.AT_TIME,
                    "2026-02-01T00:00:00Z",
                ),
                ExpiryCondition(ExpiryConditionKind.AT_COMMIT, valid_boundary),
            ),
        )


def test_applicability_evaluates_expiry_after_status_gate_before_sources(
    git_repo: Path,
) -> None:
    sha = _head(git_repo)
    time_condition = ExpiryCondition(
        ExpiryConditionKind.AT_TIME,
        "2026-07-14T08:00:00Z",
    )
    active = _detached_record(
        _candidate(git_repo, sha),
        expiry_conditions=(time_condition,),
    )
    evaluator = TargetHeadApplicabilityEvaluator(git_repo, SourceValidator(git_repo))
    evaluated_at = datetime(2026, 7, 14, 8, 0, tzinfo=timezone.utc)

    due = evaluator.evaluate(
        active,
        target_head=sha,
        evaluated_at="2026-07-14T08:00:00Z",
    )
    assert due.applicability is Applicability.EXPIRED
    assert due.reason_code == "expiry_time_reached"
    assert not due.requires_revalidation

    persisted = evaluator.evaluate(
        replace(active, status=RecordStatus.EXPIRED),
        target_head="f" * 40,
        evaluated_at=evaluated_at,
    )
    assert persisted.applicability is Applicability.EXPIRED
    assert persisted.reason_code == "record_expired"

    unresolved = evaluator.evaluate(
        _detached_record(
            _candidate(
                git_repo,
                sha,
                statement="Unresolved expiry must fail closed before source checks.",
            ),
            expiry_conditions=(
                ExpiryCondition(ExpiryConditionKind.AT_COMMIT, "e" * 40),
            ),
        ),
        target_head=sha,
        evaluated_at=evaluated_at,
    )
    assert unresolved.applicability is Applicability.SOURCE_MISSING
    assert unresolved.reason_code == "expiry_condition_unresolved"
    assert unresolved.requires_revalidation


def test_runtime_expiry_sweep_events_generation_replay_and_unresolved(
    git_repo: Path,
    tmp_path: Path,
) -> None:
    lifecycle, store = _lifecycle(git_repo, tmp_path)
    main_branch = run_git(git_repo, "branch", "--show-current")
    valid_from = _head(git_repo)
    run_git(git_repo, "checkout", "-b", "expiry-unresolved", valid_from)
    run_git(git_repo, "commit", "--allow-empty", "-m", "side expiry boundary")
    unresolved_boundary = _head(git_repo)
    run_git(git_repo, "checkout", main_branch)
    run_git(git_repo, "commit", "--allow-empty", "-m", "runtime expiry target")
    target = _head(git_repo)

    configured = {
        "time": ExpiryCondition(
            ExpiryConditionKind.AT_TIME,
            "2026-07-14T08:00:00Z",
        ),
        "commit": ExpiryCondition(ExpiryConditionKind.AT_COMMIT, valid_from),
        "unresolved": ExpiryCondition(
            ExpiryConditionKind.AT_COMMIT,
            unresolved_boundary,
        ),
        "future": ExpiryCondition(
            ExpiryConditionKind.AT_TIME,
            "2027-07-14T08:00:00Z",
        ),
    }
    records: dict[str, DurableMemoryRecord] = {}
    for label, condition in configured.items():
        candidate = _candidate(
            git_repo,
            valid_from,
            statement="Runtime expiry sweep record %s." % label,
        )
        provenance = _provenance(candidate, target)
        _submit(
            lifecycle,
            candidate,
            provenance=provenance,
            label="expiry-sweep-" + label,
        )
        records[label] = _approve(
            lifecycle,
            candidate,
            provenance=provenance,
            label="expiry-sweep-" + label,
            expiry_conditions=(condition,),
        ).record

    evaluated_at = datetime(2026, 7, 14, 8, 5, tzinfo=timezone.utc)
    generation_before = store.get_generations(
        records["time"].repository_key
    ).memory_generation
    sweep = lifecycle.expire_due_records(
        records["time"].repository_key,
        target_head=target,
        evaluated_at=evaluated_at,
        max_records=10,
    )

    assert set(sweep.scanned_ids) == {
        record.memory_id for record in records.values()
    }
    assert set(sweep.expired_ids) == {
        records["time"].memory_id,
        records["commit"].memory_id,
    }
    assert sweep.unresolved_ids == (records["unresolved"].memory_id,)
    assert len(sweep.write_results) == 2
    assert all(write.applied for write in sweep.write_results)
    assert not sweep.truncated
    assert store.get_generations(
        records["time"].repository_key
    ).memory_generation == generation_before + 2
    assert store.get_record(records["future"].memory_id).status is RecordStatus.ACTIVE
    assert store.get_record(records["unresolved"].memory_id).status is RecordStatus.ACTIVE

    expiry_events = [
        event
        for event in store.list_events(records["time"].repository_key)
        if event.action == "expire"
    ]
    assert len(expiry_events) == 2
    assert {event.reason_code for event in expiry_events} == {
        "expiry_time_reached",
        "expiry_commit_reached",
    }
    for event in expiry_events:
        assert event.actor_type == "runtime"
        assert event.actor_id == "memory_expiry"
        reason = json.loads(event.reason or "")
        assert set(reason) == {
            "condition_fingerprint",
            "evaluated_at",
            "target",
        }
        assert reason["evaluated_at"] == "2026-07-14T08:05:00Z"
        assert reason["target"] == target
        assert len(reason["condition_fingerprint"]) == 64
        assert len(event.reason or "") < 512

    events_before_replay = store.list_events(records["time"].repository_key)
    generations_before_replay = store.get_generations(
        records["time"].repository_key
    )
    replay = lifecycle.expire_record_if_due(
        records["time"].memory_id,
        target_head=target,
        evaluated_at=evaluated_at,
    )
    assert replay.expired
    assert replay.converged
    assert replay.write_result is not None
    assert replay.write_result.replayed
    assert store.list_events(records["time"].repository_key) == events_before_replay
    assert store.get_generations(
        records["time"].repository_key
    ) == generations_before_replay


def test_expiry_cas_conflict_never_overwrites_concurrent_revocation(
    git_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle, store = _lifecycle(git_repo, tmp_path)
    sha = _head(git_repo)
    candidate = _candidate(
        git_repo,
        sha,
        statement="Concurrent revocation must win over runtime expiry.",
    )
    provenance = _provenance(candidate, sha)
    _submit(lifecycle, candidate, provenance=provenance, label="expiry-race")
    record = _approve(
        lifecycle,
        candidate,
        provenance=provenance,
        label="expiry-race",
        expiry_conditions=(
            ExpiryCondition(
                ExpiryConditionKind.AT_TIME,
                "2026-07-14T08:00:00Z",
            ),
        ),
    ).record
    original_transition = store.transition_record

    def revoke_then_conflict(memory_id: str, **kwargs):
        if kwargs.get("action") == "expire":
            original_transition(
                memory_id,
                expected_status=RecordStatus.ACTIVE,
                new_status=RecordStatus.REVOKED,
                action="revoke",
                actor_type="human",
                actor_id="amy",
                reason_code="revoked",
                reason="Concurrent maintainer revocation won the CAS.",
                request_id=stable_request_id("expiry-race", "revoke", memory_id),
                created_at="2026-07-14T08:04:00Z",
            )
            raise MemoryStoreConflictError()
        return original_transition(memory_id, **kwargs)

    monkeypatch.setattr(store, "transition_record", revoke_then_conflict)
    result = lifecycle.expire_record_if_due(
        record.memory_id,
        target_head=sha,
        evaluated_at=datetime(2026, 7, 14, 8, 5, tzinfo=timezone.utc),
    )

    assert not result.expired
    assert result.write_result is None
    assert result.current_status is RecordStatus.REVOKED
    assert store.get_record(record.memory_id).status is RecordStatus.REVOKED
    record_events = store.list_events(
        record.repository_key,
        subject_type="record",
        subject_id=record.memory_id,
    )
    assert [event.action for event in record_events] == ["revoke"]
