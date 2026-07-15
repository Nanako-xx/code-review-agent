"""Human-owned lifecycle and target-revision validity for durable Memory.

This module is the authority boundary above :mod:`memory_sources` and
:mod:`memory_store`.  Runtime provenance is supplied out of band, candidates are
never made authoritative by their persisted producer metadata, and only explicit
human decisions can approve, reject, revoke, or replace durable records.

SourceBundle blobs are deterministic, content-free audit envelopes.  They retain
the exact typed references and validation digests considered at approval without
copying a Session, source file, declaration, statement, credential, or secret.
They are deliberately never consulted by target-HEAD applicability checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
import fnmatch
import hashlib
import hmac
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from review_agent.memory_models import (
    Applicability,
    CandidateAuthorityReceipt,
    CandidateStatus,
    DurableMemoryRecord,
    ExpiryCondition,
    ExpiryConditionKind,
    GitCommitSourceRef,
    MemoryCandidate,
    RecordStatus,
    RepositoryRangeSourceRef,
    RepositorySymbolSourceRef,
    Sensitivity,
    SourceBundleDescriptor,
    SourceRef,
    ValidityPolicy,
    canonical_json,
    canonical_sha256,
    stable_event_id,
    stable_request_id,
)
from review_agent.memory_sources import (
    SourceValidationCode,
    SourceValidationError,
    SourceValidationReport,
    SourceValidator,
    TrustedCandidateProvenance,
    scan_sensitive_text,
)
from review_agent.memory_store import (
    MemoryEvent,
    MemoryStore,
    MemoryStoreConflictError,
    MemoryStoreCorruptionError,
    MemoryStoreError,
    MemoryStoreNotFoundError,
    WriteResult,
)
from review_agent.revision import RevisionResolver, sanitized_git_environment


SOURCE_BUNDLE_SCHEMA = "memory_source_bundle_v1"
SOURCE_BUNDLE_SCHEMA_VERSION = 1
SOURCE_BUNDLE_MEDIA_TYPE = "application/vnd.review-agent.source-bundle+json"

_DIGEST_PATTERN = re.compile(r"^[0-9a-f]{64}$")
MAX_EXPIRY_SWEEP_RECORDS = 512


class MemoryLifecycleErrorCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    INVALID_TRANSITION = "invalid_transition"
    HUMAN_ACTOR_REQUIRED = "human_actor_required"
    HUMAN_REASON_REQUIRED = "human_reason_required"
    SOURCE_VALIDATION_FAILED = "source_validation_failed"
    DUPLICATE_SUPPRESSED = "duplicate_suppressed"
    INVALID_REPLACEMENT = "invalid_replacement"


class MemoryLifecycleError(RuntimeError):
    def __init__(self, message: str, code: MemoryLifecycleErrorCode) -> None:
        self.code = code
        super().__init__(message)


class CandidateDedupeKind(str, Enum):
    UNIQUE = "unique"
    EXACT_REPLAY = "exact_replay"
    ACTIVE_DUPLICATE = "active_duplicate"
    PENDING_DUPLICATE = "pending_duplicate"
    REJECTED_UNCHANGED = "rejected_unchanged"
    ENHANCED_PROVENANCE = "enhanced_provenance"


@dataclass(frozen=True)
class CandidateDedupeDecision:
    kind: CandidateDedupeKind
    related_candidate_id: Optional[str] = None
    related_memory_id: Optional[str] = None

    @property
    def suppressed(self) -> bool:
        return self.kind in {
            CandidateDedupeKind.ACTIVE_DUPLICATE,
            CandidateDedupeKind.PENDING_DUPLICATE,
            CandidateDedupeKind.REJECTED_UNCHANGED,
        }

    @property
    def provenance_enhanced(self) -> bool:
        return self.kind is CandidateDedupeKind.ENHANCED_PROVENANCE


@dataclass(frozen=True)
class CandidateLifecycleResult:
    candidate_id: str
    status: CandidateStatus
    validation: SourceValidationReport = field(repr=False)
    dedupe: CandidateDedupeDecision
    persisted: bool
    write_results: Tuple[WriteResult, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ApprovalResult:
    record: DurableMemoryRecord
    bundle: SourceBundleDescriptor
    bundle_payload: bytes = field(repr=False)
    validation: Optional[SourceValidationReport] = field(repr=False)
    write_result: WriteResult
    converged: bool = False


class ExpiryEvaluationStatus(str, Enum):
    NOT_CONFIGURED = "not_configured"
    NOT_DUE = "not_due"
    DUE = "due"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class ExpiryEvaluationResult:
    status: ExpiryEvaluationStatus
    evaluated_at: str
    target_head: str
    target_commit: Optional[str] = None
    due_condition: Optional[ExpiryCondition] = None
    unresolved_conditions: Tuple[ExpiryCondition, ...] = ()

    @property
    def due(self) -> bool:
        return self.status is ExpiryEvaluationStatus.DUE

    @property
    def unresolved(self) -> bool:
        return self.status is ExpiryEvaluationStatus.UNRESOLVED

    @property
    def reason_code(self) -> Optional[str]:
        if self.due_condition is None:
            return None
        if self.due_condition.condition_kind is ExpiryConditionKind.AT_TIME:
            return "expiry_time_reached"
        return "expiry_commit_reached"


@dataclass(frozen=True)
class RecordExpiryResult:
    memory_id: str
    evaluation: ExpiryEvaluationResult
    current_status: RecordStatus
    expired: bool
    write_result: Optional[WriteResult] = None
    converged: bool = False


@dataclass(frozen=True)
class ExpirySweepResult:
    repository_key: str
    target_head: str
    evaluated_at: str
    max_records: int
    scanned_ids: Tuple[str, ...]
    expired_ids: Tuple[str, ...]
    unresolved_ids: Tuple[str, ...]
    write_results: Tuple[WriteResult, ...]
    truncated: bool


class RecordAuditStatus(str, Enum):
    AUDITABLE = "auditable"
    UNAUDITABLE = "unauditable"
    NOT_FOUND = "not_found"


@dataclass(frozen=True)
class RecordAuditResult:
    memory_id: str
    status: RecordAuditStatus
    bundle: Optional[SourceBundleDescriptor] = None
    payload: Optional[Mapping[str, Any]] = field(default=None, repr=False)

    @property
    def auditable(self) -> bool:
        return self.status is RecordAuditStatus.AUDITABLE


@dataclass(frozen=True)
class ApplicabilityDecision:
    memory_id: str
    target_head: str
    applicability: Applicability
    reason_code: str
    requires_revalidation: bool = False
    source_validation: Optional[SourceValidationReport] = field(
        default=None,
        repr=False,
    )


def evaluate_expiry_conditions(
    conditions: Sequence[ExpiryCondition],
    *,
    repository: Path,
    target_head: str,
    evaluated_at: datetime | str,
    revision_resolver: Optional[RevisionResolver] = None,
) -> ExpiryEvaluationResult:
    """Evaluate canonical expiry predicates without mutating lifecycle state.

    Conditions are disjunctive.  A due condition wins over any condition whose
    Git lineage cannot be resolved; unresolved wins only when no condition is
    due.  The result is therefore stable for the same repository object graph,
    target commit, and timezone-aware evaluation instant.
    """

    canonical_conditions = _canonical_requested_expiry_conditions(conditions)
    target = _bounded_target_head(target_head)
    evaluated_utc = _aware_utc_datetime(evaluated_at)
    evaluated_text = _canonical_utc_datetime(evaluated_utc)
    if not canonical_conditions:
        return ExpiryEvaluationResult(
            status=ExpiryEvaluationStatus.NOT_CONFIGURED,
            evaluated_at=evaluated_text,
            target_head=target,
        )

    resolver = revision_resolver or RevisionResolver()
    target_commit: Optional[str]
    try:
        target_commit = resolver.resolve_commit(Path(repository), target).casefold()
    except (OSError, RuntimeError, TypeError, ValueError):
        target_commit = None

    due_conditions: List[ExpiryCondition] = []
    unresolved_conditions: List[ExpiryCondition] = []
    for condition in canonical_conditions:
        if condition.condition_kind is ExpiryConditionKind.AT_TIME:
            boundary = datetime.fromisoformat(
                condition.value[:-1] + "+00:00"
            )
            if evaluated_utc >= boundary:
                due_conditions.append(condition)
            continue

        if target_commit is None:
            unresolved_conditions.append(condition)
            continue
        try:
            boundary_commit = resolver.resolve_commit(
                Path(repository),
                condition.value,
            ).casefold()
            if boundary_commit == target_commit or resolver.is_ancestor(
                Path(repository),
                boundary_commit,
                target_commit,
            ):
                due_conditions.append(condition)
            elif not resolver.is_ancestor(
                Path(repository),
                target_commit,
                boundary_commit,
            ):
                unresolved_conditions.append(condition)
        except (OSError, RuntimeError, TypeError, ValueError):
            unresolved_conditions.append(condition)

    if due_conditions:
        return ExpiryEvaluationResult(
            status=ExpiryEvaluationStatus.DUE,
            evaluated_at=evaluated_text,
            target_head=target,
            target_commit=target_commit,
            due_condition=due_conditions[0],
            unresolved_conditions=tuple(unresolved_conditions),
        )
    if unresolved_conditions:
        return ExpiryEvaluationResult(
            status=ExpiryEvaluationStatus.UNRESOLVED,
            evaluated_at=evaluated_text,
            target_head=target,
            target_commit=target_commit,
            unresolved_conditions=tuple(unresolved_conditions),
        )
    return ExpiryEvaluationResult(
        status=ExpiryEvaluationStatus.NOT_DUE,
        evaluated_at=evaluated_text,
        target_head=target,
        target_commit=target_commit,
    )


def build_canonical_source_bundle(
    candidate: MemoryCandidate,
    validation: SourceValidationReport,
) -> bytes:
    """Build the deterministic, non-secret audit payload for one approval.

    Validation reports intentionally contain no evidence body.  This bundle
    persists only exact typed descriptors, verified hashes/bindings, sizes, and
    the report hash.  The approved statement remains in the immutable Record and
    is not duplicated into the blob.
    """

    if type(candidate) is not MemoryCandidate:
        raise MemoryLifecycleError(
            "candidate must be a canonical MemoryCandidate",
            MemoryLifecycleErrorCode.INVALID_INPUT,
        )
    if type(validation) is not SourceValidationReport or not validation.valid:
        raise MemoryLifecycleError(
            "source bundle requires a valid source report",
            MemoryLifecycleErrorCode.SOURCE_VALIDATION_FAILED,
        )
    if validation.subject_id != candidate.candidate_id:
        raise MemoryLifecycleError(
            "source report does not belong to the candidate",
            MemoryLifecycleErrorCode.SOURCE_VALIDATION_FAILED,
        )
    if len(validation.source_results) != len(candidate.source_refs):
        raise MemoryLifecycleError(
            "source report does not cover every candidate source",
            MemoryLifecycleErrorCode.SOURCE_VALIDATION_FAILED,
        )

    sources: List[Dict[str, Any]] = []
    by_index = {item.source_index: item for item in validation.source_results}
    if set(by_index) != set(range(len(candidate.source_refs))):
        raise MemoryLifecycleError(
            "source report indices are not canonical",
            MemoryLifecycleErrorCode.SOURCE_VALIDATION_FAILED,
        )
    for index, source_ref in enumerate(candidate.source_refs):
        result = by_index[index]
        expected_ref_hash = canonical_sha256(source_ref.to_dict())
        if (
            not result.valid
            or result.source_type is not source_ref.source_type
            or result.source_ref_hash is None
            or not hmac.compare_digest(result.source_ref_hash, expected_ref_hash)
        ):
            raise MemoryLifecycleError(
                "source report does not match the candidate source",
                MemoryLifecycleErrorCode.SOURCE_VALIDATION_FAILED,
            )
        sources.append(
            {
                "source_index": index,
                "source_ref": source_ref.to_dict(),
                "source_ref_hash": result.source_ref_hash,
                "verified_content_hash": result.verified_content_hash,
                "revision_binding": result.revision_binding,
                "content_size_bytes": result.content_size_bytes,
            }
        )

    payload = {
        "schema": SOURCE_BUNDLE_SCHEMA,
        "schema_version": SOURCE_BUNDLE_SCHEMA_VERSION,
        "repository_key": candidate.repository_key,
        "candidate_id": candidate.candidate_id,
        "validation_report_hash": validation.report_hash,
        "sensitivity": validation.sensitivity.to_dict(),
        "sources": sources,
    }
    encoded = canonical_json(payload).encode("utf-8")
    scan = scan_sensitive_text(
        encoded.decode("utf-8"),
        schema=SOURCE_BUNDLE_SCHEMA,
        field_name="source_bundle",
    )
    if not scan.safe:
        raise MemoryLifecycleError(
            "source bundle failed the sensitive-content boundary",
            MemoryLifecycleErrorCode.SOURCE_VALIDATION_FAILED,
        )
    return encoded


class MemoryLifecycle:
    """Coordinate validation, human decisions, and atomic Memory authority."""

    def __init__(self, store: MemoryStore, source_validator: SourceValidator) -> None:
        if not isinstance(store, MemoryStore):
            raise MemoryLifecycleError(
                "store must be a MemoryStore",
                MemoryLifecycleErrorCode.INVALID_INPUT,
            )
        if not isinstance(source_validator, SourceValidator):
            raise MemoryLifecycleError(
                "source_validator must be a SourceValidator",
                MemoryLifecycleErrorCode.INVALID_INPUT,
            )
        self.store = store
        self.source_validator = source_validator

    def submit_candidate(
        self,
        candidate: MemoryCandidate,
        *,
        runtime_provenance: TrustedCandidateProvenance,
        request_id: str,
        require_no_authority_receipts: bool = False,
    ) -> CandidateLifecycleResult:
        if type(candidate) is not MemoryCandidate:
            raise MemoryLifecycleError(
                "candidate must be a canonical MemoryCandidate",
                MemoryLifecycleErrorCode.INVALID_INPUT,
            )
        if candidate.status is not CandidateStatus.PROPOSED:
            raise MemoryLifecycleError(
                "new candidates must start in proposed status",
                MemoryLifecycleErrorCode.INVALID_TRANSITION,
            )
        if type(runtime_provenance) is not TrustedCandidateProvenance:
            raise MemoryLifecycleError(
                "trusted Runtime provenance is required",
                MemoryLifecycleErrorCode.INVALID_INPUT,
            )
        if type(require_no_authority_receipts) is not bool:
            raise MemoryLifecycleError(
                "require_no_authority_receipts must be a bool",
                MemoryLifecycleErrorCode.INVALID_INPUT,
            )

        validation = self.source_validator.validate_candidate(
            candidate,
            runtime_provenance=runtime_provenance,
        )
        authority_receipt = (
            self._build_candidate_authority_receipt(
                candidate,
                runtime_provenance=runtime_provenance,
                validation=validation,
            )
            if validation.valid
            else None
        )
        # Validation is intentionally outside the write transaction.  Once its
        # immutable report and authority receipt are ready, every Store read and
        # write participating in dedupe and status advancement shares one
        # BEGIN IMMEDIATE transaction.  A concurrent same-content proposal can
        # therefore observe either the complete winner or no winner, never the
        # old read-before-write gap.
        with self.store.candidate_submission_transaction():
            existing = self.store.find_candidate(candidate.candidate_id)
            if require_no_authority_receipts and existing is None:
                raise MemoryLifecycleError(
                    "authority recovery requires an existing migrated candidate",
                    MemoryLifecycleErrorCode.INVALID_INPUT,
                )
            if not validation.valid:
                return self._reject_invalid_candidate(
                    candidate,
                    validation,
                    request_id=request_id,
                    existing=existing,
                    runtime_provenance=runtime_provenance,
                )

            if authority_receipt is None:  # pragma: no cover - guarded above
                raise MemoryLifecycleError(
                    "valid candidate authority receipt is missing",
                    MemoryLifecycleErrorCode.SOURCE_VALIDATION_FAILED,
                )

            dedupe = self._dedupe_candidate(candidate, existing=existing)
            if existing is not None:
                writes: Tuple[WriteResult, ...] = ()
                if replace(existing, status=CandidateStatus.PROPOSED) == candidate:
                    writes = (
                        self.store.put_candidate(
                            candidate,
                            authority_receipt,
                            request_id=stable_request_id(
                                "memory_lifecycle",
                                "candidate_authority",
                                authority_receipt.receipt_id,
                            ),
                            actor_type=runtime_provenance.origin.value,
                            actor_id="memory_lifecycle",
                            reason_code="candidate_authority_revalidated",
                            require_no_authority_receipts=(
                                require_no_authority_receipts
                            ),
                        ),
                    )
                return CandidateLifecycleResult(
                    candidate_id=existing.candidate_id,
                    status=existing.status,
                    validation=validation,
                    dedupe=dedupe,
                    persisted=True,
                    write_results=writes,
                )

            if dedupe.suppressed:
                writes = self._persist_then_reject(
                    candidate,
                    request_id=request_id,
                    runtime_provenance=runtime_provenance,
                    authority_receipt=authority_receipt,
                    action="duplicate",
                    reason_code=dedupe.kind.value,
                    reason=(
                        "Candidate content was deterministically suppressed "
                        "as a duplicate."
                    ),
                )
                return CandidateLifecycleResult(
                    candidate_id=candidate.candidate_id,
                    status=CandidateStatus.REJECTED,
                    validation=validation,
                    dedupe=dedupe,
                    persisted=True,
                    write_results=writes,
                )

            writes_list: List[WriteResult] = []
            writes_list.append(
                self.store.put_candidate(
                    candidate,
                    authority_receipt,
                    request_id=_child_request_id(
                        request_id,
                        candidate.candidate_id,
                        "put",
                    ),
                    actor_type=runtime_provenance.origin.value,
                    actor_id="memory_lifecycle",
                    reason_code="candidate_submitted",
                )
            )
            current = self.store.get_candidate(candidate.candidate_id)
            if current.status is CandidateStatus.PROPOSED:
                writes_list.append(
                    self.store.transition_candidate(
                        candidate.candidate_id,
                        expected_status=CandidateStatus.PROPOSED,
                        new_status=CandidateStatus.VALIDATED,
                        action="validate",
                        actor_type="runtime",
                        actor_id="memory_lifecycle",
                        reason_code="source_validation_passed",
                        request_id=_child_request_id(
                            request_id,
                            candidate.candidate_id,
                            "validate",
                        ),
                        created_at=candidate.created_at,
                    )
                )
                current = self.store.get_candidate(candidate.candidate_id)
            if current.status is CandidateStatus.VALIDATED:
                writes_list.append(
                    self.store.transition_candidate(
                        candidate.candidate_id,
                        expected_status=CandidateStatus.VALIDATED,
                        new_status=CandidateStatus.PENDING_APPROVAL,
                        action="request_approval",
                        actor_type="runtime",
                        actor_id="memory_lifecycle",
                        reason_code="human_approval_required",
                        request_id=_child_request_id(
                            request_id,
                            candidate.candidate_id,
                            "pending",
                        ),
                        created_at=candidate.created_at,
                    )
                )
                current = self.store.get_candidate(candidate.candidate_id)
            if current.status is not CandidateStatus.PENDING_APPROVAL:
                raise MemoryLifecycleError(
                    "candidate did not reach pending approval",
                    MemoryLifecycleErrorCode.INVALID_TRANSITION,
                )
            return CandidateLifecycleResult(
                candidate_id=candidate.candidate_id,
                status=current.status,
                validation=validation,
                dedupe=dedupe,
                persisted=True,
                write_results=tuple(writes_list),
            )

    propose_candidate = submit_candidate
    propose = submit_candidate

    def approve_candidate(
        self,
        candidate_id: str,
        *,
        runtime_provenance: TrustedCandidateProvenance,
        actor: str,
        reason: str,
        request_id: str,
        reason_code: str = "approved",
        created_at: Optional[str] = None,
        expected_generation: Optional[int] = None,
        expiry_conditions: Sequence[ExpiryCondition] = (),
    ) -> ApprovalResult:
        checked_actor, checked_reason = _human_decision(actor, reason)
        canonical_expiry = _canonical_requested_expiry_conditions(
            expiry_conditions
        )
        candidate = self.store.get_candidate(candidate_id)
        replay = self._replay_approval(
            candidate,
            request_id=request_id,
            actor=checked_actor,
            reason=checked_reason,
            reason_code=reason_code,
            expiry_conditions=canonical_expiry,
        )
        if replay is not None:
            return replay
        if candidate.status is CandidateStatus.APPROVED:
            if type(runtime_provenance) is not TrustedCandidateProvenance:
                raise MemoryLifecycleError(
                    "trusted Runtime provenance is required",
                    MemoryLifecycleErrorCode.INVALID_INPUT,
                )
            converged = self._converged_approval(
                candidate.candidate_id,
                authority_resolution_hash=(
                    runtime_provenance.authority_resolution_hash
                ),
                actor=checked_actor,
                reason=checked_reason,
                reason_code=reason_code,
                request_id=request_id,
                created_at=created_at,
                expected_generation=expected_generation,
                expiry_conditions=canonical_expiry,
                require_existing_receipt=True,
            )
            if converged is not None:
                return converged
        if candidate.status is not CandidateStatus.PENDING_APPROVAL:
            raise MemoryLifecycleError(
                "only a pending candidate can be approved",
                MemoryLifecycleErrorCode.INVALID_TRANSITION,
            )
        return self._approve_pending_candidate(
            candidate,
            runtime_provenance=runtime_provenance,
            actor=checked_actor,
            reason=checked_reason,
            reason_code=reason_code,
            request_id=request_id,
            created_at=created_at,
            expected_generation=expected_generation,
            expiry_conditions=canonical_expiry,
        )

    approve = approve_candidate

    def reject_candidate(
        self,
        candidate_id: str,
        *,
        actor: str,
        reason_code: str,
        reason: str,
        request_id: str,
        created_at: Optional[str] = None,
        expected_generation: Optional[int] = None,
    ) -> WriteResult:
        checked_actor, checked_reason = _human_decision(actor, reason)
        checked_reason_code = _required_nonempty(reason_code, "reason_code")
        candidate = self.store.get_candidate(candidate_id)
        expected = self._candidate_replay_previous_status(
            candidate,
            request_id=request_id,
            action="reject",
        )
        if expected is None:
            if candidate.status not in {
                CandidateStatus.VALIDATED,
                CandidateStatus.PENDING_APPROVAL,
            }:
                raise MemoryLifecycleError(
                    "only validated or pending candidates can be rejected",
                    MemoryLifecycleErrorCode.INVALID_TRANSITION,
                )
            expected = candidate.status
        return self.store.transition_candidate(
            candidate.candidate_id,
            expected_status=expected,
            new_status=CandidateStatus.REJECTED,
            action="reject",
            actor_type="human",
            actor_id=checked_actor,
            reason_code=checked_reason_code,
            reason=checked_reason,
            request_id=request_id,
            created_at=created_at,
            expected_generation=expected_generation,
        )

    reject = reject_candidate

    def revoke_record(
        self,
        memory_id: str,
        *,
        actor: str,
        reason: str,
        request_id: str,
        created_at: Optional[str] = None,
        expected_generation: Optional[int] = None,
    ) -> WriteResult:
        checked_actor, checked_reason = _human_decision(actor, reason)
        record = self.store.get_record(memory_id)
        expected = self._record_replay_previous_status(
            record,
            request_id=request_id,
            action="revoke",
        )
        if expected is None:
            if record.status is not RecordStatus.ACTIVE:
                raise MemoryLifecycleError(
                    "only an active record can be revoked",
                    MemoryLifecycleErrorCode.INVALID_TRANSITION,
                )
            expected = RecordStatus.ACTIVE
        return self.store.transition_record(
            record.memory_id,
            expected_status=expected,
            new_status=RecordStatus.REVOKED,
            action="revoke",
            actor_type="human",
            actor_id=checked_actor,
            reason_code="revoked",
            reason=checked_reason,
            request_id=request_id,
            created_at=created_at,
            expected_generation=expected_generation,
        )

    revoke = revoke_record

    def mark_revalidation_required(
        self,
        memory_id: str,
        *,
        actor: str,
        reason: str,
        request_id: str,
        created_at: Optional[str] = None,
        expected_generation: Optional[int] = None,
    ) -> WriteResult:
        checked_actor, checked_reason = _human_decision(actor, reason)
        record = self.store.get_record(memory_id)
        expected = self._record_replay_previous_status(
            record,
            request_id=request_id,
            action="require_revalidation",
        )
        if expected is None:
            if record.status is not RecordStatus.ACTIVE:
                raise MemoryLifecycleError(
                    "only an active record can require revalidation",
                    MemoryLifecycleErrorCode.INVALID_TRANSITION,
                )
            expected = RecordStatus.ACTIVE
        return self.store.transition_record(
            record.memory_id,
            expected_status=expected,
            new_status=RecordStatus.REVALIDATION_REQUIRED,
            action="require_revalidation",
            actor_type="human",
            actor_id=checked_actor,
            reason_code="source_revalidation_required",
            reason=checked_reason,
            request_id=request_id,
            created_at=created_at,
            expected_generation=expected_generation,
        )

    def revalidate_record(
        self,
        memory_id: str,
        replacement: MemoryCandidate,
        *,
        runtime_provenance: TrustedCandidateProvenance,
        actor: str,
        reason: str,
        request_id: str,
        created_at: Optional[str] = None,
        expected_generation: Optional[int] = None,
        expiry_conditions: Sequence[ExpiryCondition] = (),
    ) -> ApprovalResult:
        checked_actor, checked_reason = _human_decision(actor, reason)
        canonical_expiry = _canonical_requested_expiry_conditions(
            expiry_conditions
        )
        if type(replacement) is not MemoryCandidate:
            raise MemoryLifecycleError(
                "replacement must be a canonical MemoryCandidate",
                MemoryLifecycleErrorCode.INVALID_REPLACEMENT,
            )

        existing_replacement = self.store.find_candidate(replacement.candidate_id)
        if existing_replacement is not None:
            replay = self._replay_approval(
                existing_replacement,
                request_id=request_id,
                actor=checked_actor,
                reason=checked_reason,
                reason_code="revalidated",
                expiry_conditions=canonical_expiry,
            )
            if replay is not None:
                predecessor = self.store.get_record(memory_id)
                if predecessor.status is not RecordStatus.SUPERSEDED:
                    raise MemoryStoreCorruptionError(
                        "revalidation replay did not supersede its predecessor"
                    )
                return replay
            if existing_replacement.status is CandidateStatus.APPROVED:
                if type(runtime_provenance) is not TrustedCandidateProvenance:
                    raise MemoryLifecycleError(
                        "trusted Runtime provenance is required",
                        MemoryLifecycleErrorCode.INVALID_INPUT,
                    )
                converged = self._converged_approval(
                    existing_replacement.candidate_id,
                    authority_resolution_hash=(
                        runtime_provenance.authority_resolution_hash
                    ),
                    actor=checked_actor,
                    reason=checked_reason,
                    reason_code="revalidated",
                    request_id=request_id,
                    created_at=created_at,
                    expected_generation=expected_generation,
                    supersede_memory_id=memory_id,
                    expiry_conditions=canonical_expiry,
                    require_existing_receipt=True,
                )
                if converged is not None:
                    predecessor = self.store.get_record(memory_id)
                    if predecessor.status is not RecordStatus.SUPERSEDED:
                        raise MemoryStoreCorruptionError(
                            "revalidation convergence did not supersede its predecessor"
                        )
                    return converged

        predecessor = self.store.get_record(memory_id)
        if predecessor.status not in {
            RecordStatus.ACTIVE,
            RecordStatus.REVALIDATION_REQUIRED,
        }:
            raise MemoryLifecycleError(
                "only active or revalidation-required records can be revalidated",
                MemoryLifecycleErrorCode.INVALID_TRANSITION,
            )
        if replacement.repository_key != predecessor.repository_key:
            raise MemoryLifecycleError(
                "replacement belongs to another repository",
                MemoryLifecycleErrorCode.INVALID_REPLACEMENT,
            )
        if replacement.candidate_id == predecessor.candidate_id:
            raise MemoryLifecycleError(
                "revalidation must create a new immutable candidate",
                MemoryLifecycleErrorCode.INVALID_REPLACEMENT,
            )
        self._validate_approval_expiry_conditions(
            replacement,
            canonical_expiry,
        )
        if expected_generation is not None:
            current_generation = self.store.get_generations(
                predecessor.repository_key
            ).memory_generation
            if current_generation != expected_generation:
                raise MemoryStoreConflictError()

        submitted = self.submit_candidate(
            replacement,
            runtime_provenance=runtime_provenance,
            request_id=_child_request_id(
                request_id,
                replacement.candidate_id,
                "replacement",
            ),
        )
        if submitted.status is not CandidateStatus.PENDING_APPROVAL:
            raise MemoryLifecycleError(
                "replacement candidate was not eligible for approval",
                MemoryLifecycleErrorCode.DUPLICATE_SUPPRESSED,
            )
        replacement_projection = self.store.get_candidate(
            replacement.candidate_id
        )
        approval_generation = self.store.get_generations(
            predecessor.repository_key
        ).memory_generation
        return self._approve_pending_candidate(
            replacement_projection,
            runtime_provenance=runtime_provenance,
            actor=checked_actor,
            reason=checked_reason,
            reason_code="revalidated",
            request_id=request_id,
            created_at=created_at,
            expected_generation=approval_generation,
            supersede_memory_id=predecessor.memory_id,
            expected_supersede_status=predecessor.status,
            expiry_conditions=canonical_expiry,
        )

    revalidate = revalidate_record

    def expire_due_records(
        self,
        repository_key: str,
        *,
        target_head: str,
        evaluated_at: datetime | str,
        max_records: int = MAX_EXPIRY_SWEEP_RECORDS,
    ) -> ExpirySweepResult:
        """Evaluate and CAS-expire a bounded prefix of ACTIVE records."""

        if (
            type(max_records) is not int
            or max_records < 1
            or max_records > MAX_EXPIRY_SWEEP_RECORDS
        ):
            raise MemoryLifecycleError(
                "max_records must be between 1 and %d"
                % MAX_EXPIRY_SWEEP_RECORDS,
                MemoryLifecycleErrorCode.INVALID_INPUT,
            )
        target = _bounded_target_head(target_head)
        evaluated_utc = _aware_utc_datetime(evaluated_at)
        evaluated_text = _canonical_utc_datetime(evaluated_utc)
        active_records = self.store.list_records(
            repository_key,
            status=RecordStatus.ACTIVE,
        )
        selected = active_records[:max_records]
        expired_ids: List[str] = []
        unresolved_ids: List[str] = []
        writes: List[WriteResult] = []
        for record in selected:
            result = self.expire_record_if_due(
                record.memory_id,
                target_head=target,
                evaluated_at=evaluated_utc,
            )
            if result.expired:
                expired_ids.append(record.memory_id)
            if result.evaluation.unresolved:
                unresolved_ids.append(record.memory_id)
            if result.write_result is not None:
                writes.append(result.write_result)
        return ExpirySweepResult(
            repository_key=repository_key,
            target_head=target,
            evaluated_at=evaluated_text,
            max_records=max_records,
            scanned_ids=tuple(record.memory_id for record in selected),
            expired_ids=tuple(expired_ids),
            unresolved_ids=tuple(unresolved_ids),
            write_results=tuple(writes),
            truncated=len(active_records) > max_records,
        )

    def expire_record_if_due(
        self,
        memory_id: str,
        *,
        target_head: str,
        evaluated_at: datetime | str,
    ) -> RecordExpiryResult:
        """Evaluate one record and perform only an ACTIVE-to-EXPIRED CAS."""

        record = self.store.get_record(memory_id)
        evaluation = evaluate_expiry_conditions(
            record.expiry_conditions,
            repository=self.source_validator.repository,
            target_head=target_head,
            evaluated_at=evaluated_at,
            revision_resolver=self.source_validator.revision_resolver,
        )
        if not evaluation.due or record.status not in {
            RecordStatus.ACTIVE,
            RecordStatus.EXPIRED,
        }:
            return RecordExpiryResult(
                memory_id=record.memory_id,
                evaluation=evaluation,
                current_status=record.status,
                expired=False,
            )

        condition = evaluation.due_condition
        reason_code = evaluation.reason_code
        if condition is None or reason_code is None:  # pragma: no cover
            raise MemoryStoreCorruptionError(
                "due expiry evaluation is missing its triggering condition"
            )
        target_label = _expiry_target_label(evaluation)
        reason = _expiry_event_reason(
            evaluated_at=evaluation.evaluated_at,
            target=target_label,
            condition_fingerprint=condition.condition_fingerprint,
        )
        request_id = stable_request_id(
            "memory_expiry",
            "expire",
            record.memory_id,
            evaluation.evaluated_at,
            target_label,
            condition.condition_fingerprint,
        )
        try:
            write = self.store.transition_record(
                record.memory_id,
                expected_status=RecordStatus.ACTIVE,
                new_status=RecordStatus.EXPIRED,
                action="expire",
                actor_type="runtime",
                actor_id="memory_expiry",
                reason_code=reason_code,
                reason=reason,
                request_id=request_id,
                created_at=evaluation.evaluated_at,
            )
            return RecordExpiryResult(
                memory_id=record.memory_id,
                evaluation=evaluation,
                current_status=RecordStatus.EXPIRED,
                expired=True,
                write_result=write,
                converged=not write.applied,
            )
        except MemoryStoreConflictError:
            current = self.store.get_record(record.memory_id)
            if current.status is not RecordStatus.EXPIRED:
                return RecordExpiryResult(
                    memory_id=record.memory_id,
                    evaluation=evaluation,
                    current_status=current.status,
                    expired=False,
                )
            converged_write = self._converged_expiry_write(
                current,
                request_id=request_id,
                reason_code=reason_code,
                reason=reason,
            )
            if converged_write is None:
                return RecordExpiryResult(
                    memory_id=record.memory_id,
                    evaluation=evaluation,
                    current_status=current.status,
                    expired=False,
                )
            return RecordExpiryResult(
                memory_id=record.memory_id,
                evaluation=evaluation,
                current_status=current.status,
                expired=True,
                write_result=converged_write,
                converged=True,
            )

    def _converged_expiry_write(
        self,
        record: DurableMemoryRecord,
        *,
        request_id: str,
        reason_code: str,
        reason: str,
    ) -> Optional[WriteResult]:
        matching = [
            event
            for event in self.store.list_events(
                record.repository_key,
                subject_type="record",
                subject_id=record.memory_id,
            )
            if (
                event.action == "expire"
                and event.actor_type == "runtime"
                and event.actor_id == "memory_expiry"
                and event.reason_code == reason_code
                and event.reason == reason
                and event.previous_status == RecordStatus.ACTIVE.value
                and event.new_status == RecordStatus.EXPIRED.value
            )
        ]
        if len(matching) != 1:
            return None
        event = matching[0]
        return WriteResult(
            operation="transition_record",
            subject_id=record.memory_id,
            event_id=event.event_id,
            generations=self.store.get_generations(record.repository_key),
            applied=False,
            replayed=event.request_id == request_id,
        )

    def audit_record(self, memory_id: str) -> RecordAuditResult:
        try:
            record = self.store.get_record(memory_id)
            bundle = self.store.get_source_bundle(record.source_bundle_hash)
            if (
                bundle.repository_key != record.repository_key
                or bundle.candidate_id != record.candidate_id
                or bundle.source_refs != record.source_refs
            ):
                raise MemoryStoreCorruptionError(
                    "source bundle does not match the audited record"
                )
            raw = self.store.read_blob(bundle.blob_hash)
            payload = _validate_source_bundle_payload(raw, record, bundle)
            return RecordAuditResult(
                memory_id=record.memory_id,
                status=RecordAuditStatus.AUDITABLE,
                bundle=bundle,
                payload=payload,
            )
        except MemoryStoreNotFoundError:
            return RecordAuditResult(
                memory_id=memory_id,
                status=RecordAuditStatus.NOT_FOUND,
            )
        except (MemoryStoreError, UnicodeError, ValueError, json.JSONDecodeError):
            return RecordAuditResult(
                memory_id=memory_id,
                status=RecordAuditStatus.UNAUDITABLE,
            )

    def _approve_pending_candidate(
        self,
        candidate: MemoryCandidate,
        *,
        runtime_provenance: TrustedCandidateProvenance,
        actor: str,
        reason: str,
        reason_code: str,
        request_id: str,
        created_at: Optional[str],
        expected_generation: Optional[int],
        expiry_conditions: Tuple[ExpiryCondition, ...],
        supersede_memory_id: Optional[str] = None,
        expected_supersede_status: Optional[RecordStatus] = None,
    ) -> ApprovalResult:
        if type(runtime_provenance) is not TrustedCandidateProvenance:
            raise MemoryLifecycleError(
                "trusted Runtime provenance is required",
                MemoryLifecycleErrorCode.INVALID_INPUT,
            )
        self._validate_approval_expiry_conditions(candidate, expiry_conditions)
        proposal = replace(candidate, status=CandidateStatus.PROPOSED)
        try:
            receipt = self.store.select_candidate_authority_receipt(
                candidate.candidate_id,
                authority_resolution_hash=(
                    runtime_provenance.authority_resolution_hash
                ),
            )
            current_target_head = self._resolve_runtime_target_head(
                runtime_provenance
            )
            restoration = self.source_validator.restore_candidate_authority(
                receipt,
                proposal,
                current_provenance=runtime_provenance,
                current_target_head_sha=current_target_head,
            )
        except MemoryStoreNotFoundError:
            raise MemoryLifecycleError(
                "candidate authority receipt is unavailable for this Runtime context",
                MemoryLifecycleErrorCode.SOURCE_VALIDATION_FAILED,
            ) from None
        except SourceValidationError as error:
            if error.report is not None and not error.report.valid:
                self._reject_failed_approval_validation(
                    candidate,
                    request_id=request_id,
                    created_at=created_at,
                    expected_generation=expected_generation,
                )
            raise MemoryLifecycleError(
                "candidate authority failed approval-time restoration",
                MemoryLifecycleErrorCode.SOURCE_VALIDATION_FAILED,
            ) from None
        validation = self.source_validator.validate_candidate(
            proposal,
            runtime_provenance=restoration.provenance,
        )
        if not validation.valid:
            self._reject_failed_approval_validation(
                candidate,
                request_id=request_id,
                created_at=created_at,
                expected_generation=expected_generation,
            )
            raise MemoryLifecycleError(
                "candidate sources failed approval-time validation",
                MemoryLifecycleErrorCode.SOURCE_VALIDATION_FAILED,
            )

        payload = build_canonical_source_bundle(
            proposal,
            validation,
        )
        timestamp = _timestamp_or_now(created_at)
        blob = self.store.put_blob(
            payload,
            media_type=SOURCE_BUNDLE_MEDIA_TYPE,
            expected_hash=hashlib.sha256(payload).hexdigest(),
            expected_size=len(payload),
            created_at=timestamp,
        )
        bundle = SourceBundleDescriptor(
            repository_key=candidate.repository_key,
            candidate_id=candidate.candidate_id,
            source_refs=candidate.source_refs,
            blob_hash=blob.blob_hash,
            size_bytes=blob.size_bytes,
            media_type=blob.media_type,
            created_at=timestamp,
        )
        approval_event_id = stable_event_id(
            "memory_lifecycle",
            "approve",
            candidate.candidate_id,
            request_id,
        )
        record = DurableMemoryRecord(
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
            approved_by=actor,
            approval_event_id=approval_event_id,
            status=RecordStatus.ACTIVE,
            created_at=timestamp,
            expiry_conditions=expiry_conditions,
        )
        try:
            write = self.store.approve_candidate_with_source_bundle(
                record,
                bundle,
                request_id=request_id,
                expected_candidate_status=CandidateStatus.PENDING_APPROVAL,
                expected_generation=expected_generation,
                authority_resolution_hash=(
                    restoration.provenance.authority_resolution_hash
                ),
                actor_type="human",
                actor_id=actor,
                reason_code=reason_code,
                reason=reason,
                supersede_memory_id=supersede_memory_id,
                expected_supersede_status=expected_supersede_status,
            )
        except MemoryStoreConflictError:
            converged = self._converged_approval(
                candidate.candidate_id,
                authority_resolution_hash=(
                    restoration.provenance.authority_resolution_hash
                ),
                actor=actor,
                reason=reason,
                reason_code=reason_code,
                request_id=request_id,
                created_at=created_at,
                expected_generation=expected_generation,
                supersede_memory_id=supersede_memory_id,
                expiry_conditions=expiry_conditions,
            )
            if converged is None:
                raise
            return converged
        return ApprovalResult(
            record=record,
            bundle=bundle,
            bundle_payload=payload,
            validation=validation,
            write_result=write,
        )

    def _converged_approval(
        self,
        candidate_id: str,
        *,
        authority_resolution_hash: str,
        actor: str,
        reason: str,
        reason_code: str,
        request_id: str,
        created_at: Optional[str],
        expected_generation: Optional[int],
        supersede_memory_id: Optional[str] = None,
        expiry_conditions: Tuple[ExpiryCondition, ...] = (),
        require_existing_receipt: bool = False,
    ) -> Optional[ApprovalResult]:
        """Persist or replay one same-decision concurrent approval receipt."""

        current = self.store.find_candidate(candidate_id)
        if current is None or current.status is not CandidateStatus.APPROVED:
            return None
        receipt_operation = self._request_receipt_operation(request_id)
        if require_existing_receipt:
            if receipt_operation is None:
                return None
            if receipt_operation != "transition_candidate":
                raise MemoryStoreConflictError(
                    "request ID was reused for different content"
                )
        elif receipt_operation is not None:
            # This path may create only the receipt for the fresh request that
            # just lost the approval CAS.  Existing receipts retain their Store
            # request-reuse semantics.
            return None
        memory_id = _memory_id(candidate_id)
        try:
            record = self.store.get_record(memory_id)
            if (
                record.candidate_id != candidate_id
                or record.status is not RecordStatus.ACTIVE
            ):
                return None
            bundle = self.store.get_source_bundle(record.source_bundle_hash)
            payload = self.store.read_blob(bundle.blob_hash)
            _validate_source_bundle_payload(payload, record, bundle)
        except (MemoryStoreError, UnicodeError, ValueError, json.JSONDecodeError):
            return None
        if record.expiry_conditions != expiry_conditions:
            raise MemoryStoreConflictError(
                "approval expiry conditions differ from the winning record"
            )

        approval_event = self._winning_approval_event(record)
        supersede_event = self._approval_supersede_event(record, approval_event)
        if require_existing_receipt:
            write = self._write_converged_approval_receipt(
                record,
                authority_resolution_hash=authority_resolution_hash,
                actor=actor,
                reason=reason,
                reason_code=reason_code,
                request_id=request_id,
                created_at=created_at,
                expected_generation=expected_generation,
                supersede_memory_id=supersede_memory_id,
                expiry_conditions=expiry_conditions,
            )
            if not write.replayed:
                raise MemoryStoreCorruptionError(
                    "converged approval receipt was not replayed"
                )
        else:
            if (
                approval_event.actor_type != "human"
                or approval_event.actor_id != actor
                or approval_event.reason_code != reason_code
                or approval_event.reason != reason
                or (created_at is not None and approval_event.created_at != created_at)
                or record.approved_by != actor
                or (
                    None if supersede_event is None else supersede_event.subject_id
                )
                != supersede_memory_id
            ):
                # A concurrent request with a different human decision or a
                # different predecessor is a real conflict, not convergence.
                return None
            write = self._write_converged_approval_receipt(
                record,
                authority_resolution_hash=authority_resolution_hash,
                actor=actor,
                reason=reason,
                reason_code=reason_code,
                request_id=request_id,
                created_at=created_at,
                expected_generation=expected_generation,
                supersede_memory_id=supersede_memory_id,
                expiry_conditions=expiry_conditions,
            )
            if write.applied or write.event_id is not None:
                raise MemoryStoreCorruptionError(
                    "converged approval receipt changed authoritative state"
                )
        return ApprovalResult(
            record=record,
            bundle=bundle,
            bundle_payload=payload,
            validation=None,
            write_result=write,
            converged=True,
        )

    def _request_receipt_operation(self, request_id: str) -> Optional[str]:
        """Read only enough receipt metadata to distinguish replay from create."""

        return self.store.find_request_receipt_operation(request_id)

    def _winning_approval_event(
        self,
        record: DurableMemoryRecord,
    ) -> MemoryEvent:
        matching = [
            event
            for event in self.store.list_events(
                record.repository_key,
                subject_type="candidate",
                subject_id=record.candidate_id,
            )
            if event.event_id == record.approval_event_id
        ]
        if len(matching) != 1:
            raise MemoryStoreCorruptionError(
                "approved record does not have one winning approval event"
            )
        event = matching[0]
        if (
            event.action != "approve"
            or event.previous_status != CandidateStatus.PENDING_APPROVAL.value
            or event.new_status != CandidateStatus.APPROVED.value
        ):
            raise MemoryStoreCorruptionError(
                "approved record winning event is invalid"
            )
        return event

    def _approval_supersede_event(
        self,
        record: DurableMemoryRecord,
        approval_event: MemoryEvent,
    ) -> Optional[MemoryEvent]:
        matching = []
        for event in self.store.list_events(
            record.repository_key,
            subject_type="record",
        ):
            if event.action != "supersede":
                continue
            expected_request_id = stable_request_id(
                "approve_candidate_with_source_bundle",
                "supersede",
                approval_event.request_id,
                event.subject_id,
                record.memory_id,
            )
            if event.request_id == expected_request_id:
                matching.append(event)
        if len(matching) > 1:
            raise MemoryStoreCorruptionError(
                "approval has multiple predecessor supersession events"
            )
        if not matching:
            return None
        event = matching[0]
        if (
            event.actor_type != approval_event.actor_type
            or event.actor_id != approval_event.actor_id
            or event.reason != approval_event.reason
            or event.new_status != RecordStatus.SUPERSEDED.value
        ):
            raise MemoryStoreCorruptionError(
                "approval predecessor supersession event is invalid"
            )
        return event

    def _write_converged_approval_receipt(
        self,
        record: DurableMemoryRecord,
        *,
        authority_resolution_hash: str,
        actor: str,
        reason: str,
        reason_code: str,
        request_id: str,
        created_at: Optional[str],
        expected_generation: Optional[int],
        supersede_memory_id: Optional[str],
        expiry_conditions: Tuple[ExpiryCondition, ...],
    ) -> WriteResult:
        decision_hash = canonical_sha256(
            {
                "operation": "approve_candidate_with_source_bundle",
                "candidate_id": record.candidate_id,
                "winning_memory_id": record.memory_id,
                "winning_approval_event_id": record.approval_event_id,
                "winning_source_bundle_hash": record.source_bundle_hash,
                "authority_resolution_hash": authority_resolution_hash,
                "actor_type": "human",
                "actor_id": actor,
                "reason_code": reason_code,
                "reason": reason,
                "created_at": created_at,
                "expected_generation": expected_generation,
                "supersede_memory_id": supersede_memory_id,
                "expiry_conditions": [
                    condition.to_dict() for condition in expiry_conditions
                ],
            }
        )
        return self.store.transition_candidate(
            record.candidate_id,
            expected_status=CandidateStatus.APPROVED,
            new_status=CandidateStatus.APPROVED,
            action="approve_converged",
            actor_type="human",
            actor_id=actor,
            reason_code="approval_converged:" + decision_hash,
            reason=reason,
            request_id=request_id,
            created_at=created_at,
            authority_resolution_hash=authority_resolution_hash,
        )

    def _build_candidate_authority_receipt(
        self,
        candidate: MemoryCandidate,
        *,
        runtime_provenance: TrustedCandidateProvenance,
        validation: SourceValidationReport,
    ) -> CandidateAuthorityReceipt:
        try:
            return self.source_validator.build_candidate_authority_receipt(
                candidate,
                runtime_provenance,
                validation,
                current_target_head_sha=self._resolve_runtime_target_head(
                    runtime_provenance
                ),
                created_at=candidate.created_at,
            )
        except SourceValidationError:
            raise MemoryLifecycleError(
                "candidate authority could not be issued from the live Runtime context",
                MemoryLifecycleErrorCode.SOURCE_VALIDATION_FAILED,
            ) from None

    def _resolve_runtime_target_head(
        self,
        runtime_provenance: TrustedCandidateProvenance,
    ) -> str:
        try:
            return self.source_validator.revision_resolver.resolve_commit(
                self.source_validator.repository,
                runtime_provenance.target_head_sha,
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            raise MemoryLifecycleError(
                "Runtime target revision cannot be resolved in the live repository",
                MemoryLifecycleErrorCode.SOURCE_VALIDATION_FAILED,
            ) from None

    def _validate_approval_expiry_conditions(
        self,
        candidate: MemoryCandidate,
        expiry_conditions: Tuple[ExpiryCondition, ...],
    ) -> None:
        resolver = self.source_validator.revision_resolver
        repository = self.source_validator.repository
        for condition in expiry_conditions:
            if condition.condition_kind is not ExpiryConditionKind.AT_COMMIT:
                continue
            try:
                valid_from = resolver.resolve_commit(
                    repository,
                    candidate.valid_from_sha,
                ).casefold()
                boundary = resolver.resolve_commit(
                    repository,
                    condition.value,
                ).casefold()
                valid_boundary = boundary == valid_from or resolver.is_ancestor(
                    repository,
                    valid_from,
                    boundary,
                )
            except (OSError, RuntimeError, TypeError, ValueError):
                valid_boundary = False
            if not valid_boundary:
                raise MemoryLifecycleError(
                    "at_commit expiry boundary must exist at or after "
                    "valid_from_sha on the same lineage",
                    MemoryLifecycleErrorCode.INVALID_INPUT,
                )

    def _reject_failed_approval_validation(
        self,
        candidate: MemoryCandidate,
        *,
        request_id: str,
        created_at: Optional[str],
        expected_generation: Optional[int],
    ) -> None:
        self.store.transition_candidate(
            candidate.candidate_id,
            expected_status=CandidateStatus.PENDING_APPROVAL,
            new_status=CandidateStatus.REJECTED,
            action="reject",
            actor_type="runtime",
            actor_id="memory_lifecycle",
            reason_code="approval_source_validation_failed",
            reason=None,
            request_id=_child_request_id(
                request_id,
                candidate.candidate_id,
                "approval-validation-failed",
            ),
            created_at=created_at,
            expected_generation=expected_generation,
        )

    def _replay_approval(
        self,
        candidate: MemoryCandidate,
        *,
        request_id: str,
        actor: str,
        reason: str,
        reason_code: str,
        expiry_conditions: Tuple[ExpiryCondition, ...],
    ) -> Optional[ApprovalResult]:
        matching = [
            event
            for event in self.store.list_events(
                candidate.repository_key,
                subject_type="candidate",
                subject_id=candidate.candidate_id,
            )
            if event.request_id == request_id
        ]
        if not matching:
            return None
        event = matching[-1]
        if (
            event.action != "approve"
            or event.actor_type != "human"
            or event.actor_id != actor
            or event.reason_code != reason_code
            or event.reason != reason
        ):
            raise MemoryStoreConflictError(
                "request ID was reused for different content"
            )
        memory_id = _memory_id(candidate.candidate_id)
        record = self.store.get_record(memory_id)
        if record.expiry_conditions != expiry_conditions:
            raise MemoryStoreConflictError(
                "request ID was reused with different expiry conditions"
            )
        bundle = self.store.get_source_bundle(record.source_bundle_hash)
        payload = self.store.read_blob(bundle.blob_hash)
        _validate_source_bundle_payload(payload, record, bundle)
        write = WriteResult(
            operation="approve_candidate_with_source_bundle",
            subject_id=record.memory_id,
            event_id=event.event_id,
            generations=self.store.get_generations(record.repository_key),
            applied=False,
            replayed=True,
        )
        return ApprovalResult(
            record=record,
            bundle=bundle,
            bundle_payload=payload,
            validation=None,
            write_result=write,
        )

    def _dedupe_candidate(
        self,
        candidate: MemoryCandidate,
        *,
        existing: Optional[MemoryCandidate],
    ) -> CandidateDedupeDecision:
        if existing is not None:
            kind = (
                CandidateDedupeKind.REJECTED_UNCHANGED
                if existing.status is CandidateStatus.REJECTED
                else CandidateDedupeKind.EXACT_REPLAY
            )
            return CandidateDedupeDecision(
                kind=kind,
                related_candidate_id=existing.candidate_id,
            )

        candidates = self.store.list_candidates(candidate.repository_key)
        related = [
            item
            for item in candidates
            if item.content_fingerprint == candidate.content_fingerprint
        ]
        if not related:
            return CandidateDedupeDecision(CandidateDedupeKind.UNIQUE)

        records_by_candidate: Dict[str, DurableMemoryRecord] = {}
        for record in self.store.list_records(candidate.repository_key):
            records_by_candidate[record.candidate_id] = record

        candidate_sources = _source_ref_keys(candidate.source_refs)
        enhanced: Optional[MemoryCandidate] = None
        for prior in related:
            prior_sources = _source_ref_keys(prior.source_refs)
            is_strict_enhancement = candidate_sources > prior_sources
            if is_strict_enhancement:
                enhanced = prior
                continue
            if prior.status is CandidateStatus.APPROVED:
                record = records_by_candidate.get(prior.candidate_id)
                if record is not None and record.status in {
                    RecordStatus.ACTIVE,
                    RecordStatus.REVALIDATION_REQUIRED,
                }:
                    return CandidateDedupeDecision(
                        CandidateDedupeKind.ACTIVE_DUPLICATE,
                        related_candidate_id=prior.candidate_id,
                        related_memory_id=record.memory_id,
                    )
            if prior.status in {
                CandidateStatus.PROPOSED,
                CandidateStatus.VALIDATED,
                CandidateStatus.PENDING_APPROVAL,
            }:
                return CandidateDedupeDecision(
                    CandidateDedupeKind.PENDING_DUPLICATE,
                    related_candidate_id=prior.candidate_id,
                )
            if (
                prior.status is CandidateStatus.REJECTED
                and candidate_sources == prior_sources
                and candidate.producer.schema_version
                == prior.producer.schema_version
            ):
                return CandidateDedupeDecision(
                    CandidateDedupeKind.REJECTED_UNCHANGED,
                    related_candidate_id=prior.candidate_id,
                )
        if enhanced is not None:
            return CandidateDedupeDecision(
                CandidateDedupeKind.ENHANCED_PROVENANCE,
                related_candidate_id=enhanced.candidate_id,
            )
        return CandidateDedupeDecision(CandidateDedupeKind.UNIQUE)

    def _reject_invalid_candidate(
        self,
        candidate: MemoryCandidate,
        validation: SourceValidationReport,
        *,
        request_id: str,
        existing: Optional[MemoryCandidate],
        runtime_provenance: TrustedCandidateProvenance,
    ) -> CandidateLifecycleResult:
        dedupe = (
            self._dedupe_candidate(candidate, existing=existing)
            if existing is not None
            else CandidateDedupeDecision(CandidateDedupeKind.UNIQUE)
        )
        if not validation.retain_content or validation.sensitivity.effective is Sensitivity.BLOCKED:
            return CandidateLifecycleResult(
                candidate_id=candidate.candidate_id,
                status=CandidateStatus.REJECTED,
                validation=validation,
                dedupe=dedupe,
                persisted=False,
            )
        if existing is not None and existing.status is CandidateStatus.REJECTED:
            return CandidateLifecycleResult(
                candidate_id=existing.candidate_id,
                status=existing.status,
                validation=validation,
                dedupe=CandidateDedupeDecision(
                    CandidateDedupeKind.REJECTED_UNCHANGED,
                    related_candidate_id=existing.candidate_id,
                ),
                persisted=True,
            )
        writes = self._persist_then_reject(
            candidate,
            request_id=request_id,
            runtime_provenance=runtime_provenance,
            action="reject",
            reason_code="validation_failed",
            reason=None,
        )
        return CandidateLifecycleResult(
            candidate_id=candidate.candidate_id,
            status=CandidateStatus.REJECTED,
            validation=validation,
            dedupe=dedupe,
            persisted=True,
            write_results=writes,
        )

    def _persist_then_reject(
        self,
        candidate: MemoryCandidate,
        *,
        request_id: str,
        runtime_provenance: TrustedCandidateProvenance,
        authority_receipt: Optional[CandidateAuthorityReceipt] = None,
        action: str,
        reason_code: str,
        reason: Optional[str],
    ) -> Tuple[WriteResult, ...]:
        writes: List[WriteResult] = []
        existing = self.store.find_candidate(candidate.candidate_id)
        if existing is None:
            writes.append(
                self.store.put_candidate(
                    candidate,
                    authority_receipt,
                    request_id=_child_request_id(
                        request_id,
                        candidate.candidate_id,
                        "put",
                    ),
                    actor_type=runtime_provenance.origin.value,
                    actor_id="memory_lifecycle",
                    reason_code="candidate_submitted",
                )
            )
            existing = self.store.get_candidate(candidate.candidate_id)
        if existing.status is CandidateStatus.REJECTED:
            return tuple(writes)
        if existing.status not in {
            CandidateStatus.PROPOSED,
            CandidateStatus.VALIDATED,
            CandidateStatus.PENDING_APPROVAL,
        }:
            raise MemoryLifecycleError(
                "authoritative candidates cannot be automatically rejected",
                MemoryLifecycleErrorCode.INVALID_TRANSITION,
            )
        writes.append(
            self.store.transition_candidate(
                candidate.candidate_id,
                expected_status=existing.status,
                new_status=CandidateStatus.REJECTED,
                action=action,
                actor_type="runtime",
                actor_id="memory_lifecycle",
                reason_code=reason_code,
                reason=reason,
                request_id=_child_request_id(
                    request_id,
                    candidate.candidate_id,
                    "reject",
                ),
                created_at=candidate.created_at,
            )
        )
        return tuple(writes)

    def _candidate_replay_previous_status(
        self,
        candidate: MemoryCandidate,
        *,
        request_id: str,
        action: str,
    ) -> Optional[CandidateStatus]:
        event = _request_event(
            self.store.list_events(
                candidate.repository_key,
                subject_type="candidate",
                subject_id=candidate.candidate_id,
            ),
            request_id,
        )
        if event is None:
            return None
        if event.action != action or event.previous_status is None:
            raise MemoryStoreConflictError(
                "request ID was reused for different content"
            )
        try:
            return CandidateStatus(event.previous_status)
        except ValueError:
            raise MemoryStoreCorruptionError(
                "candidate lifecycle event has an invalid previous status"
            ) from None

    def _record_replay_previous_status(
        self,
        record: DurableMemoryRecord,
        *,
        request_id: str,
        action: str,
    ) -> Optional[RecordStatus]:
        event = _request_event(
            self.store.list_events(
                record.repository_key,
                subject_type="record",
                subject_id=record.memory_id,
            ),
            request_id,
        )
        if event is None:
            return None
        if event.action != action or event.previous_status is None:
            raise MemoryStoreConflictError(
                "request ID was reused for different content"
            )
        try:
            return RecordStatus(event.previous_status)
        except ValueError:
            raise MemoryStoreCorruptionError(
                "record lifecycle event has an invalid previous status"
            ) from None


class TargetHeadApplicabilityEvaluator:
    """Evaluate one immutable Record for one target without mutating lifecycle."""

    def __init__(
        self,
        repository: Path,
        source_validator: SourceValidator,
        *,
        revision_resolver: Optional[RevisionResolver] = None,
    ) -> None:
        self.repository = Path(repository)
        if not isinstance(source_validator, SourceValidator):
            raise MemoryLifecycleError(
                "source_validator must be a SourceValidator",
                MemoryLifecycleErrorCode.INVALID_INPUT,
            )
        self.source_validator = source_validator
        self.revision_resolver = revision_resolver or RevisionResolver()

    def evaluate(
        self,
        record: DurableMemoryRecord,
        *,
        target_head: str,
        evaluated_at: Optional[datetime | str] = None,
        changed_paths: Optional[Sequence[str]] = None,
        changed_symbols: Optional[Sequence[str]] = None,
        changed_contracts: Optional[Sequence[str]] = None,
        changed_languages: Optional[Sequence[str]] = None,
    ) -> ApplicabilityDecision:
        if type(record) is not DurableMemoryRecord:
            raise MemoryLifecycleError(
                "record must be a canonical DurableMemoryRecord",
                MemoryLifecycleErrorCode.INVALID_INPUT,
            )

        status_decision = self._status_decision(record, str(target_head))
        if status_decision is not None:
            return status_decision

        if record.expiry_conditions:
            if evaluated_at is None:
                return self._decision(
                    record,
                    str(target_head),
                    Applicability.SOURCE_MISSING,
                    "expiry_condition_unresolved",
                    requires_revalidation=True,
                )
            expiry = evaluate_expiry_conditions(
                record.expiry_conditions,
                repository=self.repository,
                target_head=target_head,
                evaluated_at=evaluated_at,
                revision_resolver=self.revision_resolver,
            )
            if expiry.due:
                reason_code = expiry.reason_code
                if reason_code is None:  # pragma: no cover
                    raise MemoryStoreCorruptionError(
                        "due expiry evaluation is missing its reason"
                    )
                return self._decision(
                    record,
                    expiry.target_commit or expiry.target_head,
                    Applicability.EXPIRED,
                    reason_code,
                )
            if expiry.unresolved:
                return self._decision(
                    record,
                    expiry.target_commit or expiry.target_head,
                    Applicability.SOURCE_MISSING,
                    "expiry_condition_unresolved",
                    requires_revalidation=True,
                )
        try:
            target = self.revision_resolver.resolve_commit(
                self.repository,
                target_head,
            ).casefold()
        except (OSError, RuntimeError, TypeError, ValueError):
            return self._decision(
                record,
                str(target_head),
                Applicability.SOURCE_MISSING,
                "target_head_missing",
                requires_revalidation=True,
            )

        try:
            if record.valid_from_sha == target:
                relation = "at_valid_from"
            elif self.revision_resolver.is_ancestor(
                self.repository,
                record.valid_from_sha,
                target,
            ):
                relation = "descendant"
            elif self.revision_resolver.is_ancestor(
                self.repository,
                target,
                record.valid_from_sha,
            ):
                return self._decision(
                    record,
                    target,
                    Applicability.NOT_YET_VALID,
                    "target_precedes_valid_from",
                )
            else:
                return self._decision(
                    record,
                    target,
                    Applicability.LINEAGE_MISMATCH,
                    "diverged_lineage",
                )
        except (OSError, RuntimeError, TypeError, ValueError):
            return self._decision(
                record,
                target,
                Applicability.SOURCE_MISSING,
                "valid_from_missing",
                requires_revalidation=True,
            )

        policies = set(record.validity_policies)
        if ValidityPolicy.MANUAL_UNTIL_REVOKED not in policies:
            if policies.intersection(
                {
                    ValidityPolicy.SOURCE_CONTENT_HASH,
                    ValidityPolicy.SYMBOL_SIGNATURE,
                }
            ):
                source_decision = self._source_decision(record, target)
                if source_decision is not None:
                    return source_decision
            if (
                relation == "descendant"
                and ValidityPolicy.SCOPE_CHANGE_TRIGGER in policies
            ):
                try:
                    scope_changed = self._scope_changed_since(
                        record,
                        target,
                        changed_symbols=changed_symbols,
                        changed_contracts=changed_contracts,
                        changed_languages=changed_languages,
                    )
                except MemoryLifecycleError:
                    return self._decision(
                        record,
                        target,
                        Applicability.SOURCE_MISSING,
                        "scope_change_unavailable",
                        requires_revalidation=True,
                    )
            else:
                scope_changed = False
            if scope_changed:
                return self._decision(
                    record,
                    target,
                    Applicability.SOURCE_CHANGED,
                    "scope_changed",
                    requires_revalidation=True,
                )

        if _scope_context_provided(
            changed_paths,
            changed_symbols,
            changed_contracts,
            changed_languages,
        ) and not _scope_matches(
            record,
            changed_paths=changed_paths,
            changed_symbols=changed_symbols,
            changed_contracts=changed_contracts,
            changed_languages=changed_languages,
        ):
            return self._decision(
                record,
                target,
                Applicability.OUT_OF_SCOPE,
                "target_scope_does_not_match",
            )
        return self._decision(
            record,
            target,
            Applicability.SELECTED,
            "applicable",
        )

    def _status_decision(
        self,
        record: DurableMemoryRecord,
        target: str,
    ) -> Optional[ApplicabilityDecision]:
        if record.status is RecordStatus.REVOKED:
            return self._decision(
                record,
                target,
                Applicability.REVOKED,
                "record_revoked",
            )
        if record.status is RecordStatus.SUPERSEDED:
            return self._decision(
                record,
                target,
                Applicability.SUPERSEDED,
                "record_superseded",
            )
        if record.status is RecordStatus.EXPIRED:
            return self._decision(
                record,
                target,
                Applicability.EXPIRED,
                "record_expired",
            )
        if record.status is RecordStatus.REVALIDATION_REQUIRED:
            return self._decision(
                record,
                target,
                Applicability.SOURCE_CHANGED,
                "record_revalidation_required",
                requires_revalidation=True,
            )
        if record.status is not RecordStatus.ACTIVE:
            return self._decision(
                record,
                target,
                Applicability.SOURCE_CHANGED,
                "record_status_not_authoritative",
                requires_revalidation=True,
            )
        return None

    def _source_decision(
        self,
        record: DurableMemoryRecord,
        target: str,
    ) -> Optional[ApplicabilityDecision]:
        target_refs: List[SourceRef] = []
        for source_ref in record.source_refs:
            if type(source_ref) is RepositoryRangeSourceRef:
                target_refs.append(replace(source_ref, revision=target))
            elif type(source_ref) is RepositorySymbolSourceRef:
                target_refs.append(replace(source_ref, revision=target))
            else:
                target_refs.append(source_ref)
        report = self.source_validator.validate_sources(
            tuple(target_refs),
            sensitivity=record.sensitivity,
            statement=record.statement,
        )
        if report.valid:
            return None
        codes = {issue.code for issue in report.issues}
        missing_codes = {
            SourceValidationCode.REPOSITORY_UNAVAILABLE,
            SourceValidationCode.REVISION_NOT_FOUND,
            SourceValidationCode.SOURCE_NOT_FOUND,
            SourceValidationCode.SOURCE_NOT_REGULAR,
            SourceValidationCode.SYMBOL_NOT_FOUND,
            SourceValidationCode.SESSION_NOT_FOUND,
            SourceValidationCode.DESCRIPTOR_NOT_FOUND,
            SourceValidationCode.OBSERVATION_NOT_FOUND,
            SourceValidationCode.HUMAN_DECLARATION_UNAUTHORIZED,
        }
        applicability = (
            Applicability.SOURCE_MISSING
            if codes.intersection(missing_codes)
            else Applicability.SOURCE_CHANGED
        )
        return self._decision(
            record,
            target,
            applicability,
            (
                "source_missing"
                if applicability is Applicability.SOURCE_MISSING
                else "source_changed"
            ),
            requires_revalidation=True,
            source_validation=report,
        )

    def _scope_changed_since(
        self,
        record: DurableMemoryRecord,
        target: str,
        *,
        changed_symbols: Optional[Sequence[str]],
        changed_contracts: Optional[Sequence[str]],
        changed_languages: Optional[Sequence[str]],
    ) -> bool:
        paths = _git_changed_paths(
            self.repository,
            record.valid_from_sha,
            target,
        )
        if record.scope.is_empty:
            return bool(paths)
        if record.scope.paths and any(
            _path_matches(path, record.scope.paths) for path in paths
        ):
            return True
        if changed_symbols is not None and set(record.scope.symbols).intersection(
            changed_symbols
        ):
            return True
        if changed_contracts is not None and set(record.scope.contracts).intersection(
            item.casefold() for item in changed_contracts
        ):
            return True
        if changed_languages is not None and set(record.scope.languages).intersection(
            item.casefold() for item in changed_languages
        ):
            return True
        has_non_path_scope = bool(
            record.scope.symbols
            or record.scope.contracts
            or record.scope.languages
        )
        lacks_non_path_authority = (
            changed_symbols is None
            and changed_contracts is None
            and changed_languages is None
        )
        return bool(paths) and not record.scope.paths and has_non_path_scope and lacks_non_path_authority

    @staticmethod
    def _decision(
        record: DurableMemoryRecord,
        target: str,
        applicability: Applicability,
        reason_code: str,
        *,
        requires_revalidation: bool = False,
        source_validation: Optional[SourceValidationReport] = None,
    ) -> ApplicabilityDecision:
        return ApplicabilityDecision(
            memory_id=record.memory_id,
            target_head=target,
            applicability=applicability,
            reason_code=reason_code,
            requires_revalidation=requires_revalidation,
            source_validation=source_validation,
        )


ApplicabilityEvaluator = TargetHeadApplicabilityEvaluator


def _validate_source_bundle_payload(
    raw: bytes,
    record: DurableMemoryRecord,
    bundle: SourceBundleDescriptor,
) -> Mapping[str, Any]:
    if bundle.media_type != SOURCE_BUNDLE_MEDIA_TYPE:
        raise ValueError("source bundle media type is unsupported")
    text = raw.decode("utf-8")
    payload = json.loads(text)
    if not isinstance(payload, Mapping) or canonical_json(payload) != text:
        raise ValueError("source bundle is not canonical JSON")
    expected_fields = {
        "schema",
        "schema_version",
        "repository_key",
        "candidate_id",
        "validation_report_hash",
        "sensitivity",
        "sources",
    }
    if set(payload) != expected_fields:
        raise ValueError("source bundle fields are invalid")
    if (
        payload["schema"] != SOURCE_BUNDLE_SCHEMA
        or payload["schema_version"] != SOURCE_BUNDLE_SCHEMA_VERSION
        or payload["repository_key"] != record.repository_key
        or payload["candidate_id"] != record.candidate_id
        or not isinstance(payload["validation_report_hash"], str)
        or _DIGEST_PATTERN.fullmatch(payload["validation_report_hash"]) is None
        or not isinstance(payload["sensitivity"], Mapping)
        or not isinstance(payload["sources"], list)
    ):
        raise ValueError("source bundle authority binding is invalid")
    if len(payload["sources"]) != len(bundle.source_refs):
        raise ValueError("source bundle source count is invalid")
    for index, (item, source_ref) in enumerate(
        zip(payload["sources"], bundle.source_refs)
    ):
        if not isinstance(item, Mapping) or set(item) != {
            "source_index",
            "source_ref",
            "source_ref_hash",
            "verified_content_hash",
            "revision_binding",
            "content_size_bytes",
        }:
            raise ValueError("source bundle source entry is invalid")
        if (
            item["source_index"] != index
            or item["source_ref"] != source_ref.to_dict()
            or item["source_ref_hash"] != canonical_sha256(source_ref.to_dict())
            or type(item["content_size_bytes"]) is not int
            or item["content_size_bytes"] < 0
        ):
            raise ValueError("source bundle source binding is invalid")
    if hashlib.sha256(raw).hexdigest() != bundle.blob_hash:
        raise ValueError("source bundle blob digest is invalid")
    scan = scan_sensitive_text(
        text,
        schema=SOURCE_BUNDLE_SCHEMA,
        field_name="source_bundle",
    )
    if not scan.safe:
        raise ValueError("source bundle contains sensitive material")
    return payload


def _canonical_requested_expiry_conditions(
    values: Sequence[ExpiryCondition],
) -> Tuple[ExpiryCondition, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise MemoryLifecycleError(
            "expiry_conditions must be a list or tuple of ExpiryCondition values",
            MemoryLifecycleErrorCode.INVALID_INPUT,
        )
    if len(values) > len(ExpiryConditionKind):
        raise MemoryLifecycleError(
            "expiry_conditions must contain at most two conditions",
            MemoryLifecycleErrorCode.INVALID_INPUT,
        )
    by_kind: Dict[ExpiryConditionKind, ExpiryCondition] = {}
    for condition in values:
        if type(condition) is not ExpiryCondition:
            raise MemoryLifecycleError(
                "expiry_conditions must contain canonical ExpiryCondition values",
                MemoryLifecycleErrorCode.INVALID_INPUT,
            )
        if condition.condition_kind in by_kind:
            raise MemoryLifecycleError(
                "expiry_conditions must contain at most one condition of each kind",
                MemoryLifecycleErrorCode.INVALID_INPUT,
            )
        by_kind[condition.condition_kind] = condition
    return tuple(
        by_kind[kind]
        for kind in sorted(by_kind, key=lambda item: item.value)
    )


def _aware_utc_datetime(value: datetime | str) -> datetime:
    if isinstance(value, str):
        if not value or len(value) > 64 or "\0" in value:
            raise MemoryLifecycleError(
                "evaluated_at must carry an explicit timezone",
                MemoryLifecycleErrorCode.INVALID_INPUT,
            )
        try:
            value = datetime.fromisoformat(
                value[:-1] + "+00:00" if value.endswith("Z") else value
            )
        except ValueError:
            raise MemoryLifecycleError(
                "evaluated_at must carry an explicit timezone",
                MemoryLifecycleErrorCode.INVALID_INPUT,
            ) from None
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise MemoryLifecycleError(
            "evaluated_at must carry an explicit timezone",
            MemoryLifecycleErrorCode.INVALID_INPUT,
        )
    try:
        offset = value.utcoffset()
        if offset is None:
            raise ValueError
        return value.astimezone(timezone.utc)
    except (OverflowError, TypeError, ValueError):
        raise MemoryLifecycleError(
            "evaluated_at must carry an explicit timezone",
            MemoryLifecycleErrorCode.INVALID_INPUT,
        ) from None


def _canonical_utc_datetime(value: datetime) -> str:
    timespec = "microseconds" if value.microsecond else "seconds"
    return value.isoformat(timespec=timespec).replace("+00:00", "Z")


def _bounded_target_head(value: str) -> str:
    if not isinstance(value, str):
        raise MemoryLifecycleError(
            "target_head must be bounded text",
            MemoryLifecycleErrorCode.INVALID_INPUT,
        )
    normalized = value.strip()
    if not normalized or len(normalized) > 512 or "\0" in normalized:
        raise MemoryLifecycleError(
            "target_head must be bounded text",
            MemoryLifecycleErrorCode.INVALID_INPUT,
        )
    return normalized


def _expiry_target_label(evaluation: ExpiryEvaluationResult) -> str:
    if evaluation.target_commit is not None:
        return evaluation.target_commit
    return "unresolved:" + canonical_sha256(
        {"target_head": evaluation.target_head}
    )


def _expiry_event_reason(
    *,
    evaluated_at: str,
    target: str,
    condition_fingerprint: str,
) -> str:
    return canonical_json(
        {
            "condition_fingerprint": condition_fingerprint,
            "evaluated_at": evaluated_at,
            "target": target,
        }
    )


def _human_decision(actor: str, reason: str) -> Tuple[str, str]:
    try:
        checked_actor = _required_nonempty(actor, "actor")
        checked_reason = _required_nonempty(reason, "reason")
    except MemoryLifecycleError:
        raise
    if not checked_actor:
        raise MemoryLifecycleError(
            "human actor is required",
            MemoryLifecycleErrorCode.HUMAN_ACTOR_REQUIRED,
        )
    if not checked_reason:
        raise MemoryLifecycleError(
            "human reason is required",
            MemoryLifecycleErrorCode.HUMAN_REASON_REQUIRED,
        )
    return checked_actor, checked_reason


def _required_nonempty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        code = (
            MemoryLifecycleErrorCode.HUMAN_ACTOR_REQUIRED
            if field_name == "actor"
            else MemoryLifecycleErrorCode.HUMAN_REASON_REQUIRED
        )
        raise MemoryLifecycleError(
            "%s is required" % field_name,
            code,
        )
    return " ".join(value.split())


def _timestamp_or_now(value: Optional[str]) -> str:
    if value is not None:
        return value
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00",
        "Z",
    )


def _child_request_id(request_id: str, subject_id: str, phase: str) -> str:
    return stable_request_id("memory_lifecycle", request_id, subject_id, phase)


def _memory_id(candidate_id: str) -> str:
    return "MEM-" + hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()


def _source_ref_keys(source_refs: Iterable[SourceRef]) -> frozenset:
    return frozenset(canonical_json(item.to_dict()) for item in source_refs)


def _request_event(
    events: Sequence[MemoryEvent],
    request_id: str,
) -> Optional[MemoryEvent]:
    matches = [event for event in events if event.request_id == request_id]
    if len(matches) > 1:
        raise MemoryStoreCorruptionError(
            "request ID appears more than once in the event chain"
        )
    return None if not matches else matches[0]


def _scope_context_provided(*values: Optional[Sequence[str]]) -> bool:
    return any(value is not None for value in values)


def _scope_matches(
    record: DurableMemoryRecord,
    *,
    changed_paths: Optional[Sequence[str]],
    changed_symbols: Optional[Sequence[str]],
    changed_contracts: Optional[Sequence[str]],
    changed_languages: Optional[Sequence[str]],
) -> bool:
    if record.scope.is_empty:
        return True
    if changed_paths is not None and record.scope.paths:
        if any(_path_matches(path, record.scope.paths) for path in changed_paths):
            return True
    if changed_symbols is not None and record.scope.symbols:
        if set(record.scope.symbols).intersection(changed_symbols):
            return True
    if changed_contracts is not None and record.scope.contracts:
        if set(record.scope.contracts).intersection(
            item.casefold() for item in changed_contracts
        ):
            return True
    if changed_languages is not None and record.scope.languages:
        if set(record.scope.languages).intersection(
            item.casefold() for item in changed_languages
        ):
            return True
    return False


def _path_matches(path: str, patterns: Sequence[str]) -> bool:
    normalized = str(path).replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.startswith("/") or ".." in normalized.split("/"):
        return False
    return any(fnmatch.fnmatchcase(normalized, pattern) for pattern in patterns)


def _git_changed_paths(repository: Path, start: str, end: str) -> Tuple[str, ...]:
    try:
        result = subprocess.run(
            [
                "git",
                "--no-replace-objects",
                "diff",
                "--name-only",
                "--no-renames",
                "-z",
                start,
                end,
                "--",
            ],
            cwd=repository,
            env=sanitized_git_environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise MemoryLifecycleError(
            "unable to inspect scope changes",
            MemoryLifecycleErrorCode.SOURCE_VALIDATION_FAILED,
        ) from None
    if result.returncode != 0:
        raise MemoryLifecycleError(
            "unable to inspect scope changes",
            MemoryLifecycleErrorCode.SOURCE_VALIDATION_FAILED,
        )
    try:
        decoded = result.stdout.decode("utf-8")
    except UnicodeError:
        raise MemoryLifecycleError(
            "scope change paths are not valid UTF-8",
            MemoryLifecycleErrorCode.SOURCE_VALIDATION_FAILED,
        ) from None
    return tuple(sorted(item for item in decoded.split("\0") if item))


__all__ = [
    "ApplicabilityDecision",
    "ApplicabilityEvaluator",
    "ApprovalResult",
    "CandidateDedupeDecision",
    "CandidateDedupeKind",
    "CandidateLifecycleResult",
    "ExpiryEvaluationResult",
    "ExpiryEvaluationStatus",
    "ExpirySweepResult",
    "MAX_EXPIRY_SWEEP_RECORDS",
    "MemoryLifecycle",
    "MemoryLifecycleError",
    "MemoryLifecycleErrorCode",
    "RecordAuditResult",
    "RecordAuditStatus",
    "RecordExpiryResult",
    "SOURCE_BUNDLE_MEDIA_TYPE",
    "SOURCE_BUNDLE_SCHEMA",
    "TargetHeadApplicabilityEvaluator",
    "build_canonical_source_bundle",
    "evaluate_expiry_conditions",
]
