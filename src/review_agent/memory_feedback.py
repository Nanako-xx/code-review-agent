"""Human review feedback import and non-suppressive calibration.

Feedback is an audited human decision about one review; it is not project
policy.  This module therefore has two deliberately separate boundaries:

* :class:`FeedbackImportService` re-hydrates Session and Observation authority,
  materializes a minimal immutable :class:`~review_agent.memory_models.FindingSnapshot`,
  validates every typed source, and delegates the only write to
  :class:`~review_agent.memory_store.MemoryStore`.
* :func:`aggregate_feedback` reads a validated Store snapshot and emits only
  bounded taxonomy, counts, and provenance.  Raw claims, reasons, actors, and
  Feedback records are never part of reviewer or reconciler projections.

The aggregation policy is monotonic: its closed action enum can only raise a
check/perspective priority or demand more evidence.  There is intentionally no
suppression, risk-lowering, severity-lowering, permission, or Durable Memory
conversion action.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import hmac
import json
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from review_agent.artifacts import artifact_schema
from review_agent.evidence import CanonicalFinding, EvidenceReconciliation
from review_agent.memory_identity import repository_key as canonical_repository_key
from review_agent.memory_models import (
    FEEDBACK_AGGREGATION_POLICY_VERSION,
    FeedbackCalibrationSignal,
    FeedbackCalibrationSignalKind,
    FeedbackCalibrationSummary,
    FeedbackDecision,
    FeedbackReasonCode,
    FeedbackRecord,
    FeedbackStatus,
    FindingSeverity,
    FindingSnapshot,
    HumanDeclarationAuthority,
    MemoryScope,
    ObservationSourceRef,
    RepositoryRangeSourceRef,
    RepositorySymbolSourceRef,
    Sensitivity,
    SessionArtifactSourceRef,
    SourceRef,
    canonical_sha256,
    validate_stable_id,
)
from review_agent.memory_sources import (
    SourceValidationReport,
    SourceValidator,
    scan_sensitive_text,
)
from review_agent.memory_policy import RuntimePolicyRegistry
from review_agent.memory_store import (
    MemoryStore,
    MemoryStoreConflictError,
    WriteResult,
)
from review_agent.observations import Observation, ObservationStore
from review_agent.run_state import RunPhase, RunStatus
from review_agent.session import (
    SESSION_SCHEMA_VERSION,
    ArtifactDescriptor,
    PhaseStatus,
    SessionManifest,
)
from review_agent.session_store import SessionStore


FEEDBACK_AGGREGATION_MIN_RECORDS = 5
FEEDBACK_AGGREGATION_MIN_REVIEWS = 3
MAX_AGGREGATED_FEEDBACK_RECORDS = 10_000
MAX_AGGREGATION_SIGNALS = 128
MAX_FEEDBACK_ARTIFACT_BYTES = 8 * 1024 * 1024
AUTOMATIC_DURABLE_MEMORY_CONVERSION_ALLOWED = False

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT_ID_PATTERN = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_FINDING_ID_PATTERN = re.compile(r"^F-[0-9a-f]{32}(?:[0-9a-f]{32})?$")
_OBSERVATION_ID_PATTERN = re.compile(r"^O-[0-9a-f]{12,64}$")
_REVIEW_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_UTC_TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)
_MAX_REASON_LENGTH = 2_048
_MAX_IDENTIFIER_LENGTH = 512


class FeedbackErrorCode(str, Enum):
    INVALID_INPUT = "invalid_input"
    SESSION_NOT_FOUND = "session_not_found"
    SESSION_UNTRUSTED = "session_untrusted"
    SESSION_NOT_COMPLETED = "session_not_completed"
    REPOSITORY_MISMATCH = "repository_mismatch"
    HEAD_MISMATCH = "head_mismatch"
    FINDING_NOT_FOUND = "finding_not_found"
    FINDING_NOT_CANONICAL = "finding_not_canonical"
    FINDING_HASH_MISMATCH = "finding_hash_mismatch"
    EVIDENCE_MISMATCH = "evidence_mismatch"
    HUMAN_DECLARATION_REQUIRED = "human_declaration_required"
    VERIFIABLE_SOURCE_REQUIRED = "verifiable_source_required"
    SOURCE_VALIDATION_FAILED = "source_validation_failed"
    CONFLICTING_DECISION = "conflicting_decision"
    AGGREGATION_LIMIT_EXCEEDED = "aggregation_limit_exceeded"
    DURABLE_MEMORY_CONVERSION_PROHIBITED = "durable_memory_conversion_prohibited"


class FeedbackError(RuntimeError):
    """Fail-closed feedback error with a stable machine-readable code."""

    def __init__(self, message: str, code: FeedbackErrorCode) -> None:
        self.code = code
        super().__init__(message)


# Compatibility spelling for callers that distinguish validation failures.
FeedbackValidationError = FeedbackError


class CalibrationAction(str, Enum):
    """The complete allowlist of effects feedback aggregation may request."""

    RAISE_CHECK_PRIORITY = "raise_check_priority"
    RAISE_PERSPECTIVE_PRIORITY = "raise_perspective_priority"
    DEMAND_MORE_EVIDENCE = "demand_more_evidence"


class FeedbackAggregationDisposition(str, Enum):
    CALIBRATION = "calibration"
    EVAL_ONLY = "eval_only"


@dataclass(frozen=True)
class MissedFindingInput:
    """Human-submitted minimal Finding fields for a genuinely missed issue."""

    finding_id: str
    claim: str
    path: str
    line: int
    contracts: Tuple[str, ...]
    original_severity: FindingSeverity
    evidence_refs: Tuple[str, ...]

    def __post_init__(self) -> None:
        try:
            snapshot = self.materialize()
        except (TypeError, ValueError) as error:
            raise ValueError("missed finding input is invalid") from error
        object.__setattr__(self, "finding_id", snapshot.finding_id)
        object.__setattr__(self, "claim", snapshot.claim)
        object.__setattr__(self, "path", snapshot.path)
        object.__setattr__(self, "line", snapshot.line)
        object.__setattr__(self, "contracts", snapshot.contracts)
        object.__setattr__(self, "original_severity", snapshot.original_severity)
        object.__setattr__(self, "evidence_refs", snapshot.evidence_refs)

    def materialize(self) -> FindingSnapshot:
        """Return a fresh immutable snapshot rather than retaining caller state."""

        return FindingSnapshot(
            finding_id=self.finding_id,
            claim=self.claim,
            path=self.path,
            line=self.line,
            contracts=tuple(self.contracts),
            original_severity=self.original_severity,
            evidence_refs=tuple(self.evidence_refs),
        )


@dataclass(frozen=True)
class FeedbackImportRequest:
    request_id: str
    repository_key: str
    review_id: str
    finding_id: str
    head_sha: str
    finding_hash: str
    evidence_refs: Tuple[str, ...]
    decision: FeedbackDecision
    final_severity: FindingSeverity
    reason_code: FeedbackReasonCode
    reason: str
    actor: str
    created_at: str
    missed_finding: Optional[MissedFindingInput] = None
    human_declaration: Optional[HumanDeclarationAuthority] = field(
        default=None,
        repr=False,
    )
    source_refs: Tuple[SourceRef, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        try:
            request_id = validate_stable_id(
                self.request_id,
                "REQ",
                "feedback request_id",
            )
        except (TypeError, ValueError) as error:
            raise ValueError("feedback request_id must be a canonical REQ ID") from error
        repository_key = _digest(self.repository_key, "repository_key")
        review_id = _review_id(self.review_id)
        finding_id = _finding_id(self.finding_id)
        head_sha = _git_object_id(self.head_sha, "head_sha")
        finding_hash = _digest(self.finding_hash, "finding_hash")
        evidence_refs = _observation_ids(self.evidence_refs, "evidence_refs")
        if not isinstance(self.decision, FeedbackDecision):
            raise ValueError("decision must be a FeedbackDecision")
        if not isinstance(self.final_severity, FindingSeverity):
            raise ValueError("final_severity must be a FindingSeverity")
        if not isinstance(self.reason_code, FeedbackReasonCode):
            raise ValueError("reason_code must be a FeedbackReasonCode")
        reason = _text(self.reason, "reason", _MAX_REASON_LENGTH)
        actor = _identifier(self.actor, "actor")
        created_at = _timestamp(self.created_at, "created_at")

        if self.missed_finding is not None and type(self.missed_finding) is not MissedFindingInput:
            raise ValueError("missed_finding must be a MissedFindingInput or null")
        declaration = self.human_declaration
        if declaration is not None:
            if type(declaration) is not HumanDeclarationAuthority:
                raise ValueError(
                    "human_declaration must be a HumanDeclarationAuthority or null"
                )
            try:
                hydrated_declaration = HumanDeclarationAuthority.from_dict(
                    declaration.to_dict()
                )
            except (TypeError, ValueError) as error:
                raise ValueError("human_declaration is not canonical") from error
            if hydrated_declaration != declaration:
                raise ValueError("human_declaration is not canonical")

        source_refs = _canonical_source_refs(self.source_refs)
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "repository_key", repository_key)
        object.__setattr__(self, "review_id", review_id)
        object.__setattr__(self, "finding_id", finding_id)
        object.__setattr__(self, "head_sha", head_sha)
        object.__setattr__(self, "finding_hash", finding_hash)
        object.__setattr__(self, "evidence_refs", evidence_refs)
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "actor", actor)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "source_refs", source_refs)


# Product-facing spelling used by CLI/integration callers.
ReviewFeedbackRequest = FeedbackImportRequest


@dataclass(frozen=True)
class FeedbackImportResult:
    record: FeedbackRecord
    write_result: WriteResult
    validation: Optional[SourceValidationReport] = field(default=None, repr=False)


@dataclass(frozen=True)
class PreparedFeedbackImport:
    """Write-free, hash-bound result of complete Feedback validation."""

    request_id: str
    record: FeedbackRecord = field(repr=False)
    validation: SourceValidationReport = field(repr=False)
    preparation_hash: str = field(init=False)

    def __post_init__(self) -> None:
        try:
            request_id = validate_stable_id(
                self.request_id,
                "REQ",
                "prepared feedback request_id",
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "prepared feedback request_id must be a canonical REQ ID"
            ) from error
        if type(self.record) is not FeedbackRecord:
            raise ValueError("prepared feedback record must be a FeedbackRecord")
        if self.record.status is not FeedbackStatus.RECORDED:
            raise ValueError("prepared feedback record must have recorded status")
        if type(self.validation) is not SourceValidationReport:
            raise ValueError(
                "prepared feedback validation must be a SourceValidationReport"
            )
        if not self.validation.valid:
            raise ValueError("prepared feedback validation must be successful")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(
            self,
            "preparation_hash",
            canonical_sha256(self._identity_payload()),
        )

    def _identity_payload(self) -> Dict[str, Any]:
        return {
            "schema_version": "prepared_feedback_import_v1",
            "request_id": self.request_id,
            "record": self.record.to_dict(),
            "validation_hash": self.validation.report_hash,
        }

    def require_intact(self) -> None:
        expected = canonical_sha256(self._identity_payload())
        if not hmac.compare_digest(expected, self.preparation_hash):
            raise FeedbackError(
                "prepared feedback changed after validation",
                FeedbackErrorCode.SESSION_UNTRUSTED,
            )


@dataclass(frozen=True)
class FeedbackAggregateGroup:
    decision: FeedbackDecision
    reason_code: FeedbackReasonCode
    scope: MemoryScope
    action: CalibrationAction
    signal_kind: FeedbackCalibrationSignalKind
    disposition: FeedbackAggregationDisposition
    feedback_ids: Tuple[str, ...]
    review_ids: Tuple[str, ...]
    first_created_at: str
    last_created_at: str
    sample_count: int
    review_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.decision, FeedbackDecision):
            raise ValueError("aggregate decision is invalid")
        if not isinstance(self.reason_code, FeedbackReasonCode):
            raise ValueError("aggregate reason_code is invalid")
        if not isinstance(self.scope, MemoryScope):
            raise ValueError("aggregate scope is invalid")
        if not isinstance(self.action, CalibrationAction):
            raise ValueError("aggregate action is invalid")
        if not isinstance(self.signal_kind, FeedbackCalibrationSignalKind):
            raise ValueError("aggregate signal_kind is invalid")
        if not isinstance(self.disposition, FeedbackAggregationDisposition):
            raise ValueError("aggregate disposition is invalid")
        feedback_ids = tuple(sorted(self.feedback_ids))
        review_ids = tuple(sorted(self.review_ids))
        if len(feedback_ids) != len(set(feedback_ids)) or not feedback_ids:
            raise ValueError("aggregate feedback IDs must be non-empty and unique")
        if len(review_ids) != len(set(review_ids)) or not review_ids:
            raise ValueError("aggregate review IDs must be non-empty and unique")
        for feedback_id in feedback_ids:
            validate_stable_id(feedback_id, "FB", "aggregate feedback_id")
        for review_id in review_ids:
            _review_id(review_id)
        first = _timestamp(self.first_created_at, "first_created_at")
        last = _timestamp(self.last_created_at, "last_created_at")
        if _timestamp_key(last) < _timestamp_key(first):
            raise ValueError("aggregate time range is reversed")
        if self.sample_count != len(feedback_ids):
            raise ValueError("aggregate sample_count must equal feedback ID count")
        if self.review_count != len(review_ids):
            raise ValueError("aggregate review_count must equal review ID count")
        object.__setattr__(self, "feedback_ids", feedback_ids)
        object.__setattr__(self, "review_ids", review_ids)
        object.__setattr__(self, "first_created_at", first)
        object.__setattr__(self, "last_created_at", last)

    def to_dict(self) -> Dict[str, Any]:
        """Safe projection: taxonomy, typed scope, counts, and provenance only."""

        return {
            "decision": self.decision.value,
            "reason_code": self.reason_code.value,
            "scope": self.scope.to_dict(),
            "action": self.action.value,
            "signal_kind": self.signal_kind.value,
            "disposition": self.disposition.value,
            "sample_count": self.sample_count,
            "review_count": self.review_count,
            "feedback_ids": list(self.feedback_ids),
            "review_ids": list(self.review_ids),
            "time_range": {
                "first_created_at": self.first_created_at,
                "last_created_at": self.last_created_at,
            },
        }


@dataclass(frozen=True)
class FeedbackAggregationResult:
    repository_key: str
    feedback_generation: int
    summary: FeedbackCalibrationSummary
    groups: Tuple[FeedbackAggregateGroup, ...]
    source_feedback_ids: Tuple[str, ...]
    source_review_ids: Tuple[str, ...]
    first_created_at: Optional[str]
    last_created_at: Optional[str]
    sample_count: int
    review_count: int

    @property
    def eligible(self) -> bool:
        return self.summary.eligible

    @property
    def eval_only(self) -> bool:
        return not self.summary.eligible

    @property
    def calibration_summary(self) -> FeedbackCalibrationSummary:
        return self.summary

    @property
    def context_summary(self) -> Optional[FeedbackCalibrationSummary]:
        """Return live calibration only; eval-only samples cannot enter Context."""

        return self.summary if self.summary.eligible else None


@dataclass(frozen=True)
class FeedbackStageProjection:
    """Content-free projection suitable for model Context or scheduling."""

    repository_key: str
    feedback_generation: int
    policy_version: str
    groups: Tuple[FeedbackAggregateGroup, ...]
    source_feedback_ids: Tuple[str, ...]
    source_review_ids: Tuple[str, ...]
    first_created_at: Optional[str]
    last_created_at: Optional[str]
    sample_count: int
    review_count: int

    def to_dict(self) -> Dict[str, Any]:
        decision_counts: Counter = Counter(group.decision for group in self.groups)
        return {
            "schema_version": "feedback_stage_projection_v1",
            "repository_key": self.repository_key,
            "feedback_generation": self.feedback_generation,
            "policy_version": self.policy_version,
            "sample_count": self.sample_count,
            "review_count": self.review_count,
            "feedback_ids": list(self.source_feedback_ids),
            "review_ids": list(self.source_review_ids),
            "time_range": (
                None
                if self.first_created_at is None
                else {
                    "first_created_at": self.first_created_at,
                    "last_created_at": self.last_created_at,
                }
            ),
            "decision_taxonomy": [
                {"decision": decision.value, "group_count": count}
                for decision, count in sorted(
                    decision_counts.items(),
                    key=lambda item: item[0].value,
                )
            ],
            "groups": [group.to_dict() for group in self.groups],
        }


class FeedbackImportService:
    """Validate and persist one human feedback decision through MemoryStore."""

    def __init__(
        self,
        store: Optional[MemoryStore],
        source_validator: SourceValidator,
        contract_registry: Optional[RuntimePolicyRegistry] = None,
    ) -> None:
        if store is not None and not isinstance(store, MemoryStore):
            raise ValueError("store must be a MemoryStore or null")
        if not isinstance(source_validator, SourceValidator):
            raise ValueError("source_validator must be a SourceValidator")
        if contract_registry is not None and type(
            contract_registry
        ) is not RuntimePolicyRegistry:
            raise ValueError(
                "contract_registry must be a RuntimePolicyRegistry or null"
            )
        self.store = store
        self.source_validator = source_validator
        # Compatibility input for direct service callers.  Contract authority
        # is always rehydrated from the bound Session's Runtime-owned packet.
        self.contract_registry = contract_registry
        self._materializer = FindingSnapshotMaterializer(
            source_validator.sessions_root
        )

    def record_feedback(self, request: FeedbackImportRequest) -> FeedbackImportResult:
        prepared = self.prepare_feedback(request)
        return self.record_prepared(prepared)

    def prepare_feedback(
        self,
        request: FeedbackImportRequest,
    ) -> PreparedFeedbackImport:
        """Fully validate Feedback without opening or mutating a Memory Store."""

        if type(request) is not FeedbackImportRequest:
            raise FeedbackError(
                "feedback request must be a canonical FeedbackImportRequest",
                FeedbackErrorCode.INVALID_INPUT,
            )

        if request.decision is FeedbackDecision.MISSED:
            snapshot, source_refs = self._materialize_missed(request)
        else:
            snapshot, source_refs = self._materialize_session_finding(request)

        _validate_decision_severity(request, snapshot)
        _require_safe_feedback_text(snapshot, request.reason, request.actor)
        validation = self.source_validator.validate_sources(
            source_refs,
            sensitivity=Sensitivity.NORMAL,
            statement=request.reason,
        )
        if not validation.valid:
            raise FeedbackError(
                "feedback typed source validation failed",
                FeedbackErrorCode.SOURCE_VALIDATION_FAILED,
            )

        try:
            record = FeedbackRecord(
                repository_key=request.repository_key,
                review_id=request.review_id,
                finding_id=request.finding_id,
                head_sha=request.head_sha,
                finding_snapshot=snapshot,
                decision=request.decision,
                original_severity=snapshot.original_severity,
                final_severity=request.final_severity,
                reason_code=request.reason_code,
                reason=request.reason,
                actor=request.actor,
                source_refs=source_refs,
                status=FeedbackStatus.RECORDED,
                created_at=request.created_at,
            )
            return PreparedFeedbackImport(
                request_id=request.request_id,
                record=record,
                validation=validation,
            )
        except (TypeError, ValueError) as error:
            raise FeedbackError(
                "feedback record failed canonical validation",
                FeedbackErrorCode.INVALID_INPUT,
            ) from error

    def record_prepared(
        self,
        prepared: PreparedFeedbackImport,
    ) -> FeedbackImportResult:
        """Persist exactly one previously validated Feedback preparation."""

        if type(prepared) is not PreparedFeedbackImport:
            raise FeedbackError(
                "feedback write requires a prepared import",
                FeedbackErrorCode.INVALID_INPUT,
            )
        prepared.require_intact()
        if self.store is None:
            raise FeedbackError(
                "feedback write requires an available Memory Store",
                FeedbackErrorCode.INVALID_INPUT,
            )

        # The public namespace lock serializes the service-level uniqueness
        # check with Store.put_feedback's authoritative transaction.  No SQL or
        # mutable projection is accessed outside MemoryStore.
        with MemoryStore.lock_namespaces(
            self.store.namespace_path,
            busy_timeout_ms=self.store.busy_timeout_ms,
        ):
            existing = self._existing_decision(prepared.record)
            if existing is not None:
                write_result = self.store.put_feedback(
                    existing,
                    request_id=prepared.request_id,
                )
                return FeedbackImportResult(
                    record=existing,
                    write_result=write_result,
                    validation=prepared.validation,
                )

            try:
                write_result = self.store.put_feedback(
                    prepared.record,
                    request_id=prepared.request_id,
                )
            except MemoryStoreConflictError as error:
                raise FeedbackError(
                    "feedback request conflicts with existing authority",
                    FeedbackErrorCode.CONFLICTING_DECISION,
                ) from error
            except (TypeError, ValueError) as error:
                raise FeedbackError(
                    "feedback record failed canonical validation",
                    FeedbackErrorCode.INVALID_INPUT,
                ) from error
            return FeedbackImportResult(
                record=prepared.record,
                write_result=write_result,
                validation=prepared.validation,
            )

    # Friendly spellings for command/runtime orchestration.
    record = record_feedback
    import_feedback = record_feedback

    def _existing_decision(
        self,
        prepared_record: FeedbackRecord,
    ) -> Optional[FeedbackRecord]:
        assert self.store is not None
        view = self.store.read_view(prepared_record.repository_key)
        matches = tuple(
            record
            for record in view.feedback
            if record.review_id == prepared_record.review_id
            and record.finding_id == prepared_record.finding_id
            and record.head_sha == prepared_record.head_sha
        )
        if not matches:
            return None
        exact = tuple(record for record in matches if record == prepared_record)
        if len(matches) == 1 and len(exact) == 1:
            return exact[0]
        raise FeedbackError(
            "a feedback decision already exists for this review Finding",
            FeedbackErrorCode.CONFLICTING_DECISION,
        )

    def _materialize_session_finding(
        self,
        request: FeedbackImportRequest,
    ) -> Tuple[FindingSnapshot, Tuple[SourceRef, ...]]:
        if (
            request.missed_finding is not None
            or request.human_declaration is not None
            or request.source_refs
        ):
            raise FeedbackError(
                "Session-bound feedback cannot replace canonical source fields",
                FeedbackErrorCode.INVALID_INPUT,
            )
        return self._materializer.materialize_canonical(request)

    def _materialize_missed(
        self,
        request: FeedbackImportRequest,
    ) -> Tuple[FindingSnapshot, Tuple[SourceRef, ...]]:
        manifest, session_store = self._materializer.load_bound_session(request)
        if request.missed_finding is None:
            raise FeedbackError(
                "missed feedback requires a human-submitted Finding",
                FeedbackErrorCode.INVALID_INPUT,
            )
        declaration = request.human_declaration
        if declaration is None:
            raise FeedbackError(
                "missed feedback requires an explicit human declaration",
                FeedbackErrorCode.HUMAN_DECLARATION_REQUIRED,
            )
        if (
            declaration.source_ref.request_id != request.request_id
            or declaration.source_ref.actor != request.actor
            or declaration.source_ref.review_id != request.review_id
        ):
            raise FeedbackError(
                "human declaration does not bind the feedback request",
                FeedbackErrorCode.HUMAN_DECLARATION_REQUIRED,
            )

        snapshot = request.missed_finding.materialize()
        _require_request_snapshot_match(request, snapshot)
        contract_registry = self._materializer.runtime_contract_registry(
            manifest,
            session_store,
        )
        if not set(snapshot.contracts) <= set(contract_registry.contract_ids):
            raise FeedbackError(
                "missed Finding references an unregistered contract",
                FeedbackErrorCode.INVALID_INPUT,
            )
        if self._materializer.has_canonical_finding(
            manifest,
            session_store,
            request,
        ):
            raise FeedbackError(
                "missed Finding ID already exists in final reconciliation",
                FeedbackErrorCode.FINDING_NOT_CANONICAL,
            )
        refs = request.source_refs
        if declaration.source_ref not in refs:
            raise FeedbackError(
                "missed feedback sources omit the human declaration",
                FeedbackErrorCode.HUMAN_DECLARATION_REQUIRED,
            )
        verifiable = tuple(
            source_ref
            for source_ref in refs
            if type(source_ref)
            in {
                RepositoryRangeSourceRef,
                RepositorySymbolSourceRef,
                ObservationSourceRef,
            }
        )
        if not verifiable:
            raise FeedbackError(
                "missed feedback requires a repository or Observation source",
                FeedbackErrorCode.VERIFIABLE_SOURCE_REQUIRED,
            )
        for source_ref in verifiable:
            if isinstance(source_ref, (RepositoryRangeSourceRef, RepositorySymbolSourceRef)):
                if source_ref.revision != request.head_sha:
                    raise FeedbackError(
                        "missed repository evidence must bind the review HEAD",
                        FeedbackErrorCode.HEAD_MISMATCH,
                    )
            elif (
                source_ref.review_id != request.review_id
                or source_ref.observation_id not in request.evidence_refs
            ):
                raise FeedbackError(
                    "missed Observation evidence does not match the Finding",
                    FeedbackErrorCode.EVIDENCE_MISMATCH,
                )

        # Every snapshot evidence ref must still resolve through the Session,
        # even when an exact repository range is the primary missed source.
        observation_catalog = self._materializer.observation_catalog(
            manifest,
            session_store,
            request.evidence_refs,
        )
        if not _missed_source_covers_snapshot(
            snapshot,
            verifiable,
            observation_catalog,
        ):
            raise FeedbackError(
                "missed Finding location is not covered by a verified source",
                FeedbackErrorCode.VERIFIABLE_SOURCE_REQUIRED,
            )
        observation_refs = self._materializer.observation_refs_from_catalog(
            manifest,
            observation_catalog,
        )
        additional_observations = tuple(
            source_ref
            for source_ref in observation_refs
            if source_ref not in refs
        )
        return snapshot, _canonical_source_refs((*refs, *additional_observations))


# Product-facing spelling.
ReviewFeedbackService = FeedbackImportService
FeedbackValidationService = FeedbackImportService


class FeedbackAggregator:
    def __init__(self, store: MemoryStore) -> None:
        if not isinstance(store, MemoryStore):
            raise ValueError("store must be a MemoryStore")
        self.store = store

    def aggregate(
        self,
        repository_key: str,
        *,
        created_at: Optional[str] = None,
    ) -> FeedbackAggregationResult:
        key = _digest(repository_key, "repository_key")
        view = self.store.read_view(key)
        return feedback_aggregation_v1(
            view.feedback,
            repository_key=key,
            feedback_generation=view.generations.feedback_generation,
            created_at=created_at,
        )


def aggregate_feedback(
    store: MemoryStore,
    repository_key: str,
    *,
    created_at: Optional[str] = None,
) -> FeedbackAggregationResult:
    return FeedbackAggregator(store).aggregate(
        repository_key,
        created_at=created_at,
    )


def feedback_aggregation_v1(
    records: Sequence[FeedbackRecord],
    *,
    repository_key: str,
    feedback_generation: int,
    created_at: Optional[str] = None,
) -> FeedbackAggregationResult:
    """Deterministically aggregate comparable, currently recorded decisions."""

    key = _digest(repository_key, "repository_key")
    if type(feedback_generation) is not int or feedback_generation < 0:
        raise FeedbackError(
            "feedback generation is invalid",
            FeedbackErrorCode.INVALID_INPUT,
        )
    if isinstance(records, (str, bytes)) or not isinstance(records, (list, tuple)):
        raise FeedbackError(
            "feedback records must be a bounded sequence",
            FeedbackErrorCode.INVALID_INPUT,
        )
    active: List[FeedbackRecord] = []
    for record in records:
        if type(record) is not FeedbackRecord:
            raise FeedbackError(
                "feedback aggregation requires canonical FeedbackRecord values",
                FeedbackErrorCode.INVALID_INPUT,
            )
        if record.repository_key != key:
            raise FeedbackError(
                "feedback aggregation crossed repository authority",
                FeedbackErrorCode.REPOSITORY_MISMATCH,
            )
        if record.status is FeedbackStatus.RECORDED:
            active.append(record)
    if len(active) > MAX_AGGREGATED_FEEDBACK_RECORDS:
        raise FeedbackError(
            "feedback aggregation exceeds its bounded record count",
            FeedbackErrorCode.AGGREGATION_LIMIT_EXCEEDED,
        )
    active.sort(key=lambda item: item.feedback_id)

    grouped: Dict[Tuple[str, ...], List[FeedbackRecord]] = defaultdict(list)
    for record in active:
        snapshot = record.finding_snapshot
        group_key = (
            record.decision.value,
            record.reason_code.value,
            snapshot.path,
            *snapshot.contracts,
        )
        grouped[group_key].append(record)

    groups: List[FeedbackAggregateGroup] = []
    for group_key in sorted(grouped):
        values = grouped[group_key]
        decision = values[0].decision
        reason_code = values[0].reason_code
        paths = {item.finding_snapshot.path for item in values}
        contracts = {item.finding_snapshot.contracts for item in values}
        if len(paths) != 1 or len(contracts) != 1:
            # Defensive assertion: the grouping key must remain complete if the
            # model evolves.  Silent coalescing would make feedback unsafe.
            raise FeedbackError(
                "feedback comparability key is incomplete",
                FeedbackErrorCode.INVALID_INPUT,
            )
        feedback_ids = tuple(sorted(item.feedback_id for item in values))
        review_ids = tuple(sorted({item.review_id for item in values}))
        timestamps = sorted(
            (item.created_at for item in values),
            key=_timestamp_key,
        )
        disposition = (
            FeedbackAggregationDisposition.CALIBRATION
            if len(feedback_ids) >= FEEDBACK_AGGREGATION_MIN_RECORDS
            and len(review_ids) >= FEEDBACK_AGGREGATION_MIN_REVIEWS
            else FeedbackAggregationDisposition.EVAL_ONLY
        )
        action, signal_kind = _safe_effect(decision)
        groups.append(
            FeedbackAggregateGroup(
                decision=decision,
                reason_code=reason_code,
                scope=MemoryScope(
                    paths=(next(iter(paths)),),
                    contracts=next(iter(contracts)),
                ),
                action=action,
                signal_kind=signal_kind,
                disposition=disposition,
                feedback_ids=feedback_ids,
                review_ids=review_ids,
                first_created_at=timestamps[0],
                last_created_at=timestamps[-1],
                sample_count=len(feedback_ids),
                review_count=len(review_ids),
            )
        )

    eligible_groups = tuple(
        group
        for group in groups
        if group.disposition is FeedbackAggregationDisposition.CALIBRATION
    )
    if len(eligible_groups) > MAX_AGGREGATION_SIGNALS:
        raise FeedbackError(
            "feedback aggregation exceeds its bounded signal count",
            FeedbackErrorCode.AGGREGATION_LIMIT_EXCEEDED,
        )

    all_feedback_ids = tuple(sorted(item.feedback_id for item in active))
    all_review_ids = tuple(sorted({item.review_id for item in active}))
    summary_records = (
        tuple(
            item
            for item in active
            if item.feedback_id
            in {feedback_id for group in eligible_groups for feedback_id in group.feedback_ids}
        )
        if eligible_groups
        else tuple(active)
    )
    summary_feedback_ids = tuple(sorted(item.feedback_id for item in summary_records))
    summary_review_ids = tuple(sorted({item.review_id for item in summary_records}))
    counts = Counter(item.decision for item in summary_records)
    signals = tuple(
        FeedbackCalibrationSignal(
            signal_kind=group.signal_kind,
            scope=group.scope,
            message=_safe_signal_message(group),
            sample_count=group.sample_count,
            review_count=group.review_count,
            feedback_ids=group.feedback_ids,
        )
        for group in eligible_groups
    )
    timestamps = sorted(
        (item.created_at for item in active),
        key=_timestamp_key,
    )
    aggregation_created_at = (
        _timestamp(created_at, "created_at")
        if created_at is not None
        else (timestamps[-1] if timestamps else "1970-01-01T00:00:00Z")
    )
    summary = FeedbackCalibrationSummary(
        repository_key=key,
        feedback_generation=feedback_generation,
        policy_version=FEEDBACK_AGGREGATION_POLICY_VERSION,
        eligible=bool(eligible_groups),
        source_feedback_ids=summary_feedback_ids,
        source_review_ids=summary_review_ids,
        decision_counts=tuple(
            (decision, counts[decision])
            for decision in sorted(counts, key=lambda item: item.value)
        ),
        signals=signals,
        created_at=aggregation_created_at,
    )
    return FeedbackAggregationResult(
        repository_key=key,
        feedback_generation=feedback_generation,
        summary=summary,
        groups=tuple(groups),
        source_feedback_ids=all_feedback_ids,
        source_review_ids=all_review_ids,
        first_created_at=(timestamps[0] if timestamps else None),
        last_created_at=(timestamps[-1] if timestamps else None),
        sample_count=len(all_feedback_ids),
        review_count=len(all_review_ids),
    )


def project_feedback_for_reviewer(
    result: FeedbackAggregationResult,
) -> Optional[FeedbackStageProjection]:
    return _live_projection(result)


def project_feedback_for_reconciler(
    result: FeedbackAggregationResult,
) -> Optional[FeedbackStageProjection]:
    return _live_projection(result)


def project_feedback_for_scheduling(
    result: FeedbackAggregationResult,
) -> Optional[FeedbackStageProjection]:
    return _live_projection(result)


def project_feedback_for_context(
    result: FeedbackAggregationResult,
) -> Optional[FeedbackStageProjection]:
    return _live_projection(result)


def project_feedback_for_eval(
    result: FeedbackAggregationResult,
) -> FeedbackStageProjection:
    _require_aggregation_result(result)
    return _projection(result, result.groups)


def feedback_to_durable_memory(*_args: Any, **_kwargs: Any) -> None:
    """Explicitly reject the prohibited Feedback -> Durable Memory path."""

    raise FeedbackError(
        "Feedback cannot automatically create Durable Project Memory",
        FeedbackErrorCode.DURABLE_MEMORY_CONVERSION_PROHIBITED,
    )


automatic_feedback_to_memory = feedback_to_durable_memory


class FindingSnapshotMaterializer:
    """Materialize snapshots only from completed, hash-valid Session authority."""

    def __init__(self, sessions_root: Path) -> None:
        self.sessions_root = Path(sessions_root)

    def load_bound_session(
        self,
        request: FeedbackImportRequest,
    ) -> Tuple[SessionManifest, SessionStore]:
        run_dir = _safe_session_run_dir(self.sessions_root, request.review_id)
        session_store = SessionStore(run_dir)
        try:
            manifest = session_store.load()
        except FileNotFoundError as error:
            raise FeedbackError(
                "feedback Session was not found",
                FeedbackErrorCode.SESSION_NOT_FOUND,
            ) from error
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as error:
            raise FeedbackError(
                "feedback Session failed trusted hydration",
                FeedbackErrorCode.SESSION_UNTRUSTED,
            ) from error
        if manifest.review_id != request.review_id:
            raise FeedbackError(
                "feedback review_id does not match Session authority",
                FeedbackErrorCode.SESSION_UNTRUSTED,
            )
        if manifest.status is not RunStatus.COMPLETED:
            raise FeedbackError(
                "feedback requires a completed Session",
                FeedbackErrorCode.SESSION_NOT_COMPLETED,
            )
        try:
            session_repository_key = canonical_repository_key(manifest.repository)
        except (OSError, ValueError) as error:
            raise FeedbackError(
                "Session repository identity is invalid",
                FeedbackErrorCode.SESSION_UNTRUSTED,
            ) from error
        if not hmac.compare_digest(session_repository_key, request.repository_key):
            raise FeedbackError(
                "feedback repository does not match Session authority",
                FeedbackErrorCode.REPOSITORY_MISMATCH,
            )
        if manifest.revisions.resolved_head_sha.casefold() != request.head_sha:
            raise FeedbackError(
                "feedback HEAD does not match Session authority",
                FeedbackErrorCode.HEAD_MISMATCH,
            )
        return manifest, session_store

    def materialize_canonical(
        self,
        request: FeedbackImportRequest,
    ) -> Tuple[FindingSnapshot, Tuple[SourceRef, ...]]:
        manifest, session_store = self.load_bound_session(request)
        reconciliation, authority_refs = self._load_final_reconciliation(
            manifest,
            session_store,
        )
        matches = [
            finding
            for finding in reconciliation.canonical_findings
            if finding.finding_id == request.finding_id
        ]
        if not matches:
            raise FeedbackError(
                "canonical Finding ID is absent from the Session",
                FeedbackErrorCode.FINDING_NOT_FOUND,
            )
        if len(matches) != 1:
            raise FeedbackError(
                "canonical Finding ID is duplicated in final reconciliation",
                FeedbackErrorCode.SESSION_UNTRUSTED,
            )
        finding = matches[0]
        if (
            finding.path is None
            or finding.line is None
            or not finding.evidence_refs
        ):
            raise FeedbackError(
                "Session Finding is not canonical supported output",
                FeedbackErrorCode.FINDING_NOT_CANONICAL,
            )
        contracts, contracts_source = self._canonical_contracts(
            finding,
            manifest,
            session_store,
        )
        if contracts:
            contract_registry = self.runtime_contract_registry(
                manifest,
                session_store,
            )
            if not set(contracts) <= set(contract_registry.contract_ids):
                raise FeedbackError(
                    "Finding assignments exceed the Session Runtime contract registry",
                    FeedbackErrorCode.SESSION_UNTRUSTED,
                )
        try:
            snapshot = FindingSnapshot(
                finding_id=finding.finding_id or "",
                claim=finding.claim,
                path=finding.path,
                line=finding.line,
                contracts=contracts,
                original_severity=FindingSeverity(finding.severity),
                evidence_refs=tuple(finding.evidence_refs),
            )
        except (TypeError, ValueError) as error:
            raise FeedbackError(
                "Session Finding cannot form a canonical snapshot",
                FeedbackErrorCode.FINDING_NOT_CANONICAL,
            ) from error
        _require_request_snapshot_match(request, snapshot)
        observation_refs = self.observation_refs(
            manifest,
            session_store,
            request.evidence_refs,
        )
        source_refs: Tuple[SourceRef, ...] = (
            *authority_refs,
            *observation_refs,
        )
        if contracts_source is not None:
            source_refs = (*source_refs, contracts_source)
        return snapshot, _canonical_source_refs(source_refs)

    @staticmethod
    def runtime_contract_registry(
        manifest: SessionManifest,
        session_store: SessionStore,
    ) -> RuntimePolicyRegistry:
        """Hydrate the contract catalog from the Runtime-owned Session packet."""

        descriptor = _required_artifact(
            manifest,
            session_store,
            "portfolio_packet",
            "portfolio_packet_v1",
            expected_phase=RunPhase.PLANNING,
        )
        _require_exact_revision_binding(
            manifest,
            descriptor,
            "portfolio packet",
        )
        payload = _read_json_artifact(
            session_store,
            descriptor,
            MAX_FEEDBACK_ARTIFACT_BYTES,
        )
        required_fields = {
            "risk",
            "change_map",
            "changed_symbols",
            "intent",
            "allowed_role_kinds",
            "reviewer_count_bounds",
            "contract_allowlist",
            "check_allowlist",
            "command_template_allowlist",
            "perspective_allowlist",
            "ref_allowlist",
            "ref_catalog",
            "budget_policy",
        }
        if frozenset(payload) not in {
            frozenset(required_fields),
            frozenset(required_fields | {"memory_policy"}),
        }:
            raise FeedbackError(
                "portfolio packet has an invalid Runtime authority schema",
                FeedbackErrorCode.SESSION_UNTRUSTED,
            )
        contracts = payload.get("contract_allowlist")
        if not isinstance(contracts, list):
            raise FeedbackError(
                "portfolio packet contract registry is invalid",
                FeedbackErrorCode.SESSION_UNTRUSTED,
            )
        try:
            registry = RuntimePolicyRegistry(contract_ids=tuple(contracts))
        except (TypeError, ValueError) as error:
            raise FeedbackError(
                "portfolio packet contract registry failed strict hydration",
                FeedbackErrorCode.SESSION_UNTRUSTED,
            ) from error
        if len(contracts) != len(registry.contract_ids) or set(
            contracts
        ) != set(registry.contract_ids):
            raise FeedbackError(
                "portfolio packet contract registry is not canonical",
                FeedbackErrorCode.SESSION_UNTRUSTED,
            )
        return registry

    def has_canonical_finding(
        self,
        manifest: SessionManifest,
        session_store: SessionStore,
        request: FeedbackImportRequest,
    ) -> bool:
        reconciliation, _authority_refs = self._load_final_reconciliation(
            manifest,
            session_store,
        )
        return any(
            finding.finding_id == request.finding_id
            for finding in reconciliation.canonical_findings
        )

    @staticmethod
    def _load_final_reconciliation(
        manifest: SessionManifest,
        session_store: SessionStore,
    ) -> Tuple[EvidenceReconciliation, Tuple[SessionArtifactSourceRef, ...]]:
        from review_agent.hydration import (
            reconciliation_from_dict,
            semantic_reconciliation_from_dict,
        )

        descriptor = _required_artifact(
            manifest,
            session_store,
            "reconciliation",
            "evidence_reconciliation_v1",
            expected_phase=RunPhase.RECONCILIATION,
        )
        _require_exact_revision_binding(manifest, descriptor, "final reconciliation")
        payload = _read_json_artifact(
            session_store,
            descriptor,
            MAX_FEEDBACK_ARTIFACT_BYTES,
        )
        try:
            reconciliation = reconciliation_from_dict(payload)
        except (TypeError, ValueError) as error:
            raise FeedbackError(
                "final reconciliation failed strict hydration",
                FeedbackErrorCode.SESSION_UNTRUSTED,
            ) from error
        authority_refs: Tuple[SessionArtifactSourceRef, ...] = (
            _artifact_source_ref(manifest, descriptor),
        )
        if manifest.schema_version >= SESSION_SCHEMA_VERSION:
            semantic_descriptor = _required_artifact(
                manifest,
                session_store,
                "semantic_reconciliation",
                "semantic_reconciliation_v1",
                expected_phase=RunPhase.RECONCILIATION,
            )
            _require_exact_revision_binding(
                manifest,
                semantic_descriptor,
                "semantic reconciliation",
            )
            semantic_payload = _read_json_artifact(
                session_store,
                semantic_descriptor,
                MAX_FEEDBACK_ARTIFACT_BYTES,
            )
            try:
                semantic = semantic_reconciliation_from_dict(semantic_payload)
            except (TypeError, ValueError) as error:
                raise FeedbackError(
                    "semantic reconciliation failed strict hydration",
                    FeedbackErrorCode.SESSION_UNTRUSTED,
                ) from error
            if (
                tuple(reconciliation.canonical_findings)
                != semantic.canonical_findings
                or reconciliation.evidence_quality != semantic.evidence_quality
                or tuple(reconciliation.contract_coverage)
                != semantic.contract_coverage
            ):
                raise FeedbackError(
                    "final reconciliation is not the semantic authority projection",
                    FeedbackErrorCode.SESSION_UNTRUSTED,
                )
            authority_refs = (
                *authority_refs,
                _artifact_source_ref(manifest, semantic_descriptor),
            )
        return reconciliation, authority_refs

    def observation_catalog(
        self,
        manifest: SessionManifest,
        session_store: SessionStore,
        evidence_refs: Sequence[str],
    ) -> Dict[str, Observation]:
        expected = set(evidence_refs)
        catalog: Dict[str, Observation] = {}
        revisions = _session_observation_revisions(manifest)
        descriptors = sorted(
            (
                descriptor
                for descriptor in manifest.artifacts.values()
                if descriptor.schema == "observation_log_jsonl_v1"
            ),
            key=lambda item: item.name,
        )
        for descriptor in descriptors:
            try:
                if artifact_schema(descriptor.name) != "observation_log_jsonl_v1":
                    continue
                _require_completed_artifact(manifest, descriptor)
                if not session_store.validate_artifact(descriptor):
                    continue
                root = session_store.run_dir.joinpath(
                    *PurePosixPath(descriptor.path).parent.parts
                )
                observations = ObservationStore.load(
                    root,
                    revisions,
                    max_log_bytes=MAX_FEEDBACK_ARTIFACT_BYTES,
                    max_raw_artifact_bytes=MAX_FEEDBACK_ARTIFACT_BYTES,
                    max_total_raw_bytes=MAX_FEEDBACK_ARTIFACT_BYTES,
                )
            except (OSError, UnicodeError, ValueError):
                continue
            for observation in observations.list_observations():
                if observation.observation_id not in expected:
                    continue
                existing = catalog.get(observation.observation_id)
                if existing is not None and existing != observation:
                    raise FeedbackError(
                        "Observation ID has conflicting Session authority",
                        FeedbackErrorCode.SESSION_UNTRUSTED,
                    )
                catalog[observation.observation_id] = observation
        if set(catalog) != expected:
            raise FeedbackError(
                "Finding evidence refs are not verifiable Session Observations",
                FeedbackErrorCode.EVIDENCE_MISMATCH,
            )
        return {key: catalog[key] for key in sorted(catalog)}

    def observation_refs(
        self,
        manifest: SessionManifest,
        session_store: SessionStore,
        evidence_refs: Sequence[str],
    ) -> Tuple[ObservationSourceRef, ...]:
        return self.observation_refs_from_catalog(
            manifest,
            self.observation_catalog(manifest, session_store, evidence_refs),
        )

    @staticmethod
    def observation_refs_from_catalog(
        manifest: SessionManifest,
        catalog: Mapping[str, Observation],
    ) -> Tuple[ObservationSourceRef, ...]:
        return tuple(
            ObservationSourceRef(
                review_id=manifest.review_id,
                observation_id=observation_id,
                revision_binding=catalog[observation_id].revision,
                content_hash=catalog[observation_id].content_hash,
            )
            for observation_id in sorted(catalog)
        )

    @staticmethod
    def _canonical_contracts(
        finding: CanonicalFinding,
        manifest: SessionManifest,
        session_store: SessionStore,
    ) -> Tuple[Tuple[str, ...], Optional[SessionArtifactSourceRef]]:
        from review_agent.hydration import assignments_from_dict

        reviewer_indices = tuple(sorted(set(finding.reviewer_indices)))
        if not reviewer_indices:
            return (), None
        descriptor = _required_artifact(
            manifest,
            session_store,
            "assignments",
            "reviewer_assignments_v1",
            expected_phase=RunPhase.PLANNING,
        )
        payload = _read_json_artifact(
            session_store,
            descriptor,
            MAX_FEEDBACK_ARTIFACT_BYTES,
        )
        try:
            assignments = assignments_from_dict(payload)
        except (TypeError, ValueError) as error:
            raise FeedbackError(
                "reviewer assignments failed strict hydration",
                FeedbackErrorCode.SESSION_UNTRUSTED,
            ) from error
        if any(
            not 0 <= reviewer_index < len(assignments)
            for reviewer_index in reviewer_indices
        ):
            raise FeedbackError(
                "Finding reviewer index has no Session assignment",
                FeedbackErrorCode.SESSION_UNTRUSTED,
            )
        expected_binding = (
            manifest.revisions.resolved_base_sha
            + ".."
            + manifest.revisions.resolved_head_sha
        )
        if (
            descriptor.revision_binding is None
            or descriptor.revision_binding.casefold() != expected_binding.casefold()
        ):
            raise FeedbackError(
                "reviewer assignments have an invalid revision binding",
                FeedbackErrorCode.SESSION_UNTRUSTED,
            )
        return (
            tuple(
                sorted(
                    {
                        contract
                        for reviewer_index in reviewer_indices
                        for contract in assignments[
                            reviewer_index
                        ].assigned_contract
                    }
                )
            ),
            SessionArtifactSourceRef(
                review_id=manifest.review_id,
                artifact_name=descriptor.name,
                artifact_schema=descriptor.schema,
                revision_binding=descriptor.revision_binding,
                artifact_hash=descriptor.sha256,
            ),
        )


def _record_matches_request(
    record: FeedbackRecord,
    request: FeedbackImportRequest,
) -> bool:
    if (
        record.repository_key != request.repository_key
        or record.review_id != request.review_id
        or record.finding_id != request.finding_id
        or record.head_sha != request.head_sha
        or record.finding_snapshot.finding_hash != request.finding_hash
        or record.finding_snapshot.evidence_refs != request.evidence_refs
        or record.decision is not request.decision
        or record.final_severity is not request.final_severity
        or record.reason_code is not request.reason_code
        or record.reason != request.reason
        or record.actor != request.actor
        or record.created_at != request.created_at
    ):
        return False
    if request.missed_finding is not None:
        try:
            if request.missed_finding.materialize() != record.finding_snapshot:
                return False
        except (TypeError, ValueError):
            return False
    if request.source_refs and not set(request.source_refs).issubset(set(record.source_refs)):
        return False
    return True


def _validate_decision_severity(
    request: FeedbackImportRequest,
    snapshot: FindingSnapshot,
) -> None:
    changed = request.final_severity is not snapshot.original_severity
    if request.decision is FeedbackDecision.SEVERITY_CHANGED:
        if not changed:
            raise FeedbackError(
                "severity_changed requires a different final severity",
                FeedbackErrorCode.INVALID_INPUT,
            )
    elif changed:
        raise FeedbackError(
            "only severity_changed may change final severity",
            FeedbackErrorCode.INVALID_INPUT,
        )


def _require_request_snapshot_match(
    request: FeedbackImportRequest,
    snapshot: FindingSnapshot,
) -> None:
    if snapshot.finding_id != request.finding_id:
        raise FeedbackError(
            "feedback finding_id does not match materialized Finding",
            FeedbackErrorCode.FINDING_NOT_FOUND,
        )
    if not hmac.compare_digest(snapshot.finding_hash, request.finding_hash):
        raise FeedbackError(
            "feedback Finding hash does not match Session authority",
            FeedbackErrorCode.FINDING_HASH_MISMATCH,
        )
    if snapshot.evidence_refs != request.evidence_refs:
        raise FeedbackError(
            "feedback evidence refs do not match Session authority",
            FeedbackErrorCode.EVIDENCE_MISMATCH,
        )


def _require_safe_feedback_text(
    snapshot: FindingSnapshot,
    reason: str,
    actor: str,
) -> None:
    for field_name, value in (
        ("feedback.finding_snapshot.claim", snapshot.claim),
        ("feedback.reason", reason),
        ("feedback.actor", actor),
    ):
        if not scan_sensitive_text(value, field_name=field_name).safe:
            raise FeedbackError(
                "feedback contains sensitive text and cannot be persisted",
                FeedbackErrorCode.SOURCE_VALIDATION_FAILED,
            )


def _safe_effect(
    decision: FeedbackDecision,
) -> Tuple[CalibrationAction, FeedbackCalibrationSignalKind]:
    if decision is FeedbackDecision.MISSED:
        return (
            CalibrationAction.RAISE_PERSPECTIVE_PRIORITY,
            FeedbackCalibrationSignalKind.INCREASE_CHECK_PRIORITY,
        )
    if decision is FeedbackDecision.ACCEPTED:
        return (
            CalibrationAction.RAISE_CHECK_PRIORITY,
            FeedbackCalibrationSignalKind.INCREASE_CHECK_PRIORITY,
        )
    if decision is FeedbackDecision.SEVERITY_CHANGED:
        return (
            CalibrationAction.DEMAND_MORE_EVIDENCE,
            FeedbackCalibrationSignalKind.SEVERITY_UNCERTAINTY,
        )
    if decision is FeedbackDecision.REJECTED:
        return (
            CalibrationAction.DEMAND_MORE_EVIDENCE,
            FeedbackCalibrationSignalKind.EVIDENCE_GAP_WARNING,
        )
    raise FeedbackError(
        "unsupported feedback decision taxonomy",
        FeedbackErrorCode.INVALID_INPUT,
    )


def _safe_signal_message(group: FeedbackAggregateGroup) -> str:
    # This value is generated only from closed enums.  It never includes a
    # Finding claim, human reason, actor, source excerpt, or model text.
    return (
        "feedback_taxonomy_v1:decision=%s;reason=%s;action=%s"
        % (
            group.decision.value,
            group.reason_code.value,
            group.action.value,
        )
    )


def _live_projection(
    result: FeedbackAggregationResult,
) -> Optional[FeedbackStageProjection]:
    _require_aggregation_result(result)
    groups = tuple(
        group
        for group in result.groups
        if group.disposition is FeedbackAggregationDisposition.CALIBRATION
    )
    if not groups:
        return None
    return _projection(result, groups)


def _projection(
    result: FeedbackAggregationResult,
    groups: Sequence[FeedbackAggregateGroup],
) -> FeedbackStageProjection:
    feedback_ids = tuple(
        sorted({feedback_id for group in groups for feedback_id in group.feedback_ids})
    )
    review_ids = tuple(
        sorted({review_id for group in groups for review_id in group.review_ids})
    )
    first_values = sorted(
        (group.first_created_at for group in groups),
        key=_timestamp_key,
    )
    last_values = sorted(
        (group.last_created_at for group in groups),
        key=_timestamp_key,
    )
    return FeedbackStageProjection(
        repository_key=result.repository_key,
        feedback_generation=result.feedback_generation,
        policy_version=FEEDBACK_AGGREGATION_POLICY_VERSION,
        groups=tuple(groups),
        source_feedback_ids=feedback_ids,
        source_review_ids=review_ids,
        first_created_at=(first_values[0] if first_values else None),
        last_created_at=(last_values[-1] if last_values else None),
        sample_count=len(feedback_ids),
        review_count=len(review_ids),
    )


def _missed_source_covers_snapshot(
    snapshot: FindingSnapshot,
    sources: Sequence[SourceRef],
    observations: Mapping[str, Observation],
) -> bool:
    for source_ref in sources:
        if type(source_ref) is RepositoryRangeSourceRef:
            if (
                source_ref.path == snapshot.path
                and source_ref.line_start <= snapshot.line <= source_ref.line_end
            ):
                return True
        elif type(source_ref) is RepositorySymbolSourceRef:
            if source_ref.path == snapshot.path:
                return True
        elif type(source_ref) is ObservationSourceRef:
            observation = observations.get(source_ref.observation_id)
            if (
                observation is not None
                and observation.path == snapshot.path
                and observation.line_start is not None
                and observation.line_end is not None
                and observation.line_start <= snapshot.line <= observation.line_end
            ):
                return True
    return False


def _require_exact_revision_binding(
    manifest: SessionManifest,
    descriptor: ArtifactDescriptor,
    label: str,
) -> None:
    expected = (
        manifest.revisions.resolved_base_sha
        + ".."
        + manifest.revisions.resolved_head_sha
    )
    if (
        descriptor.revision_binding is None
        or descriptor.revision_binding.casefold() != expected.casefold()
    ):
        raise FeedbackError(
            label + " revision binding is invalid",
            FeedbackErrorCode.SESSION_UNTRUSTED,
        )


def _artifact_source_ref(
    manifest: SessionManifest,
    descriptor: ArtifactDescriptor,
) -> SessionArtifactSourceRef:
    if descriptor.revision_binding is None:
        raise FeedbackError(
            "Session artifact lacks a revision binding",
            FeedbackErrorCode.SESSION_UNTRUSTED,
        )
    return SessionArtifactSourceRef(
        review_id=manifest.review_id,
        artifact_name=descriptor.name,
        artifact_schema=descriptor.schema,
        revision_binding=descriptor.revision_binding,
        artifact_hash=descriptor.sha256,
    )


def _require_aggregation_result(result: FeedbackAggregationResult) -> None:
    if type(result) is not FeedbackAggregationResult:
        raise FeedbackError(
            "projection requires a FeedbackAggregationResult",
            FeedbackErrorCode.INVALID_INPUT,
        )


def _required_artifact(
    manifest: SessionManifest,
    session_store: SessionStore,
    name: str,
    schema: str,
    *,
    expected_phase: Optional[RunPhase] = None,
) -> ArtifactDescriptor:
    descriptor = manifest.artifacts.get(name)
    if descriptor is None or descriptor.schema != schema:
        raise FeedbackError(
            "Session is missing required canonical artifact: " + name,
            FeedbackErrorCode.SESSION_UNTRUSTED,
        )
    if expected_phase is not None and descriptor.phase is not expected_phase:
        raise FeedbackError(
            "Session canonical artifact is registered in the wrong phase: " + name,
            FeedbackErrorCode.SESSION_UNTRUSTED,
        )
    try:
        if artifact_schema(name) != schema:
            raise ValueError("artifact registry mismatch")
        _require_completed_artifact(manifest, descriptor)
    except (TypeError, ValueError) as error:
        raise FeedbackError(
            "required Session artifact is outside completed authority",
            FeedbackErrorCode.SESSION_UNTRUSTED,
        ) from error
    if not session_store.validate_artifact(descriptor):
        raise FeedbackError(
            "required Session artifact failed hash validation",
            FeedbackErrorCode.SESSION_UNTRUSTED,
        )
    return descriptor


def _require_completed_artifact(
    manifest: SessionManifest,
    descriptor: ArtifactDescriptor,
) -> None:
    checkpoint = manifest.phases.get(descriptor.phase.value)
    if (
        checkpoint is None
        or checkpoint.status is not PhaseStatus.COMPLETED
        or descriptor.name not in checkpoint.artifacts
    ):
        raise ValueError("artifact is not in completed Session authority")


def _read_json_artifact(
    session_store: SessionStore,
    descriptor: ArtifactDescriptor,
    max_bytes: int,
) -> Mapping[str, Any]:
    path = session_store.run_dir.joinpath(*PurePosixPath(descriptor.path).parts)
    try:
        root = session_store.run_dir.resolve(strict=True)
        relative = path.relative_to(root)
        current = root
        for index, part in enumerate(relative.parts):
            current = current / part
            metadata = current.lstat()
            if current.is_symlink():
                raise ValueError("artifact path contains a symlink")
            if index < len(relative.parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
                raise ValueError("artifact parent is not a directory")
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
        metadata = resolved.lstat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > max_bytes:
            raise ValueError("artifact is not a bounded regular file")
        content = resolved.read_bytes()
        if len(content) > max_bytes:
            raise ValueError("artifact grew beyond its bound")
    except (OSError, ValueError) as error:
        raise FeedbackError(
            "Session artifact cannot be read safely",
            FeedbackErrorCode.SESSION_UNTRUSTED,
        ) from error
    digest = hashlib.sha256(content).hexdigest()
    if not hmac.compare_digest(digest, descriptor.sha256):
        raise FeedbackError(
            "Session artifact content hash changed",
            FeedbackErrorCode.SESSION_UNTRUSTED,
        )
    try:
        text = content.decode("utf-8")
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_json_keys)
    except (UnicodeError, ValueError, json.JSONDecodeError) as error:
        raise FeedbackError(
            "Session artifact is not strict UTF-8 JSON",
            FeedbackErrorCode.SESSION_UNTRUSTED,
        ) from error
    if not isinstance(payload, Mapping):
        raise FeedbackError(
            "Session artifact must contain one JSON object",
            FeedbackErrorCode.SESSION_UNTRUSTED,
        )
    if not session_store.validate_artifact(descriptor):
        raise FeedbackError(
            "Session artifact changed during validation",
            FeedbackErrorCode.SESSION_UNTRUSTED,
        )
    return payload


def _safe_session_run_dir(sessions_root: Path, review_id: str) -> Path:
    checked_review = _review_id(review_id)
    try:
        root = Path(sessions_root).resolve(strict=True)
        candidate = root / checked_review
        metadata = candidate.lstat()
        if candidate.is_symlink() or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("Session run directory is unsafe")
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
        if not resolved.is_dir():
            raise ValueError("Session run directory is unsafe")
        return resolved
    except FileNotFoundError as error:
        raise FeedbackError(
            "feedback Session was not found",
            FeedbackErrorCode.SESSION_NOT_FOUND,
        ) from error
    except (OSError, ValueError) as error:
        raise FeedbackError(
            "feedback Session path is untrusted",
            FeedbackErrorCode.SESSION_UNTRUSTED,
        ) from error


def _session_observation_revisions(manifest: SessionManifest) -> set[str]:
    base = manifest.revisions.resolved_base_sha.casefold()
    head = manifest.revisions.resolved_head_sha.casefold()
    return {
        base,
        head,
        "base@" + base,
        "head@" + head,
        base + ".." + head,
    }


def _canonical_source_refs(values: Iterable[SourceRef]) -> Tuple[SourceRef, ...]:
    try:
        source_refs = tuple(values)
    except TypeError as error:
        raise ValueError("source_refs must be typed SourceRef values") from error
    by_json: Dict[str, SourceRef] = {}
    for source_ref in source_refs:
        if not isinstance(source_ref, SourceRef):
            raise ValueError("source_refs must contain typed SourceRef values")
        try:
            hydrated = SourceRef.from_dict(source_ref.to_dict())
        except (TypeError, ValueError) as error:
            raise ValueError("source_refs contain a non-canonical value") from error
        if type(hydrated) is not type(source_ref) or hydrated != source_ref:
            raise ValueError("source_refs contain a non-canonical value")
        key = source_ref.to_json()
        if key in by_json:
            raise ValueError("source_refs must not contain duplicates")
        by_json[key] = source_ref
    return tuple(by_json[key] for key in sorted(by_json))


def _observation_ids(values: Any, context: str) -> Tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise ValueError(context + " must be a list or tuple")
    result: List[str] = []
    for value in values:
        if not isinstance(value, str) or _OBSERVATION_ID_PATTERN.fullmatch(value) is None:
            raise ValueError(context + " contains an invalid Observation ID")
        result.append(value)
    if not result or len(result) != len(set(result)):
        raise ValueError(context + " must be non-empty and unique")
    return tuple(sorted(result))


def _digest(value: Any, context: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(context + " must be a lowercase SHA-256 digest")
    return value


def _git_object_id(value: Any, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(context + " must be a full Git object ID")
    normalized = value.casefold()
    if value != normalized or _GIT_OBJECT_ID_PATTERN.fullmatch(normalized) is None:
        raise ValueError(context + " must be a lowercase full Git object ID")
    return normalized


def _finding_id(value: Any) -> str:
    if not isinstance(value, str) or _FINDING_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("finding_id must be a canonical Finding ID")
    return value


def _review_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or _REVIEW_ID_PATTERN.fullmatch(value) is None
        or value in {".", ".."}
    ):
        raise ValueError("review_id must be a safe canonical Session ID")
    return value


def _identifier(value: Any, context: str) -> str:
    text = _text(value, context, _MAX_IDENTIFIER_LENGTH)
    if any(character.isspace() for character in text):
        raise ValueError(context + " must not contain whitespace")
    return text


def _text(value: Any, context: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(context + " must be a non-empty canonical string")
    if "\x00" in value or len(value) > maximum:
        raise ValueError(context + " is invalid or exceeds its bound")
    value.encode("utf-8")
    return value


def _timestamp(value: Any, context: str) -> str:
    if not isinstance(value, str) or _UTC_TIMESTAMP_PATTERN.fullmatch(value) is None:
        raise ValueError(context + " must be a canonical UTC timestamp")
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(context + " must be a valid UTC timestamp") from error
    return value


def _timestamp_key(value: str) -> datetime:
    checked = _timestamp(value, "timestamp")
    return datetime.fromisoformat(checked[:-1] + "+00:00").astimezone(timezone.utc)


def _reject_duplicate_json_keys(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key: " + key)
        result[key] = value
    return result


def _exact_object(
    value: Any,
    fields: set,
    context: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(context + " must be an object")
    if set(value) != fields:
        raise ValueError(context + " fields are invalid")
    return value


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(context + " must be a string-keyed object")
    return value


def _list(value: Any, context: str) -> List[Any]:
    if not isinstance(value, list):
        raise ValueError(context + " must be a list")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(context + " must be a non-empty string")
    return value


def _string_list(value: Any, context: str) -> Tuple[str, ...]:
    items = _list(value, context)
    return tuple(_string(item, context + " item") for item in items)


__all__ = [
    "AUTOMATIC_DURABLE_MEMORY_CONVERSION_ALLOWED",
    "CalibrationAction",
    "FeedbackAggregateGroup",
    "FeedbackAggregationDisposition",
    "FeedbackAggregationResult",
    "FeedbackAggregator",
    "FeedbackError",
    "FeedbackErrorCode",
    "FeedbackImportRequest",
    "FeedbackImportResult",
    "FeedbackImportService",
    "FeedbackStageProjection",
    "FeedbackValidationError",
    "FeedbackValidationService",
    "FindingSnapshotMaterializer",
    "MissedFindingInput",
    "PreparedFeedbackImport",
    "ReviewFeedbackRequest",
    "ReviewFeedbackService",
    "aggregate_feedback",
    "automatic_feedback_to_memory",
    "feedback_aggregation_v1",
    "feedback_to_durable_memory",
    "project_feedback_for_eval",
    "project_feedback_for_context",
    "project_feedback_for_reconciler",
    "project_feedback_for_reviewer",
    "project_feedback_for_scheduling",
]
