from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Optional, Tuple

import pytest
import review_agent.memory_lifecycle as memory_lifecycle_module

from conftest import run_git
from review_agent.memory_identity import repository_key
from review_agent.memory_lifecycle import (
    CandidateDedupeKind,
    MemoryLifecycle,
    MemoryLifecycleError,
    MemoryLifecycleErrorCode,
    RecordAuditStatus,
    TargetHeadApplicabilityEvaluator,
    build_canonical_source_bundle,
)
from review_agent.memory_models import (
    Applicability,
    CandidateStatus,
    DurableMemoryRecord,
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
):
    return lifecycle.approve_candidate(
        candidate.candidate_id,
        runtime_provenance=provenance,
        actor="amy",
        reason="Maintainer verified this durable project rule.",
        request_id=stable_request_id("lifecycle-approve", label),
        created_at=LATER,
        expected_generation=expected_generation,
    )


def _detached_record(
    candidate: MemoryCandidate,
    *,
    status: RecordStatus = RecordStatus.ACTIVE,
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
    result = _submit(
        lifecycle,
        candidate,
        provenance=_provenance(
            candidate,
            sha,
            origin=ProducerType.LOCAL,
            allow_sources=False,
        ),
        label="trusted-local",
    )

    assert result.status is CandidateStatus.PENDING_APPROVAL
    assert result.validation.valid
    assert result.persisted
    assert result.dedupe.kind is CandidateDedupeKind.UNIQUE
    assert store.get_candidate(candidate.candidate_id).status is CandidateStatus.PENDING_APPROVAL
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

    result = lifecycle.revalidate_record(
        old_record.memory_id,
        replacement,
        runtime_provenance=replacement_provenance,
        actor="amy",
        reason="Maintainer approved refreshed evidence and wording.",
        request_id=stable_request_id("revalidate", old_record.memory_id),
        created_at="2026-07-14T08:03:00Z",
    )

    assert result.record.memory_id != old_record.memory_id
    assert result.record.candidate_id == replacement.candidate_id
    assert result.record.status is RecordStatus.ACTIVE
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
