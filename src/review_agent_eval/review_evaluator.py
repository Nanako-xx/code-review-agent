"""Strict Review Finding reconciliation for the core code-review Eval harness.

This module is deliberately a domain layer.  It does not know about the
product Runtime, Session, Memory, or Reviewer orchestration.  The only model
boundary it consumes is the typed ``JudgeExecutionResult`` produced by the
blind Judge protocol.

The evaluator is staged and fail-closed:

* every known-invalid pair is represented before an expected assignment is
  considered;
* only ``equivalent`` semantic decisions create assignment edges;
* location and Evidence integrity are audit signals, never issue-match
  weights;
* a persisted result is accepted only after it is hydrated and regenerated
  from the real Submission, truth, replay, evaluator, and Judge results.

The few policy choices that are not encoded by lower layers are versioned in
the constants below.  In particular, exact claim equality is the only
deterministic semantic shortcut, and assignment weights are an internal
ordering device rather than a user-facing score.
"""

from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass, fields as dataclass_fields
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from .assignment import (
    ASSIGNMENT_POLICY_VERSION,
    AssignedPair,
    WeightedAssignmentEdge,
    maximum_weight_bipartite_assignment,
)
from .config import EvaluatorExecutionConfig, JudgeKind
from .evidence_checker import (
    EVIDENCE_INTEGRITY_POLICY_VERSION,
    CommandOutputAttestation,
    EvidenceDiagnostic,
    EvidenceIntegrityChecker,
    EvidenceIntegrityResult,
    EvidenceItemIntegrityResult,
    EvidenceReasonCode,
)
from .frozen_context import FrozenContextReplay
from .judge import (
    DEFAULT_JUDGE_RUBRICS,
    GLOBAL_JUDGE_SYSTEM_PROMPT,
    JUDGE_CONTEXT_BUILDER_VERSION,
    JUDGE_PARSER_VERSION,
    JUDGE_SYSTEM_PROMPT_VERSION,
    BlindJudgeInput,
    EvidenceSupportJudgeDecision,
    FindingEquivalenceJudgeDecision,
    FindingMatchRelation,
    JudgeContextKind,
    JudgeContextSource,
    JudgeContextTrust,
    JudgeExecutionResult,
    JudgeExecutionSource,
    JudgeFailure,
    JudgeFailureCode,
    JudgeReferenceBinding,
    JudgeRubricCatalog,
    JudgeRunStatus,
    JudgeTask,
    JudgeUngradedReason,
    NovelFactuality,
    NovelFactualityJudgeDecision,
    ActionabilityAssessment,
    SeverityAssessment,
    build_evidence_support_judge_input,
    build_finding_equivalence_judge_input,
    build_novel_factuality_judge_input,
    JudgeInputArtifact,
)
from .match_location import (
    LocationCandidate,
    LocationMatchPolicy,
    LocationMatchReason,
    LocationMatchResult,
    LocationMatcher,
    SidePathCatalog,
    TruthLocationTarget,
)
from .models import (
    DiffSide,
    EvidenceAnchor,
    EvidenceIntegrity,
    EvidenceSupport,
    EvalInput,
    EvalSubmission,
    EvaluatorContextSource,
    EvaluatorContextSourceKind,
    EvaluatorContextTask,
    ExpectedFinding,
    FindingSeverity,
    KnownInvalidFinding,
    NovelFindingPolicy,
    ReviewEvaluatorContext,
    ReviewTargetKind,
    ReviewTruth,
    SchemaError,
    SubmissionEvidence,
    SubmissionFinding,
    SubmissionReview,
    SubmissionStatus,
    TruthCompleteness,
    TruthLocation,
    canonical_json,
    canonical_json_bytes,
    canonical_sha256,
    stable_id,
    _json_tree,
    _strict_json_loads,
)
from .repository import PreparedRepositoryReplay, repository_from_eval_input


REVIEW_EVALUATION_SCHEMA_VERSION = "eval_review_evaluation_v1"
REVIEW_EVALUATOR_REVISION = "review-evaluator-v1"
REVIEW_MATCH_POLICY_VERSION = "review-finding-match-v1"
REVIEW_LOCATION_POLICY_VERSION = "review-location-audit-v1"
SWE_TRUTH_DIFF_HUNK_SOURCE_KIND = "swe_truth_diff_hunk_v1"
SWE_TRUTH_DIFF_HUNK_SOURCE_ID_KIND = "swe-truth-diff-hunk-v1"

# These are intentionally independent from the generic model limits.  A
# Review artifact must reject an over-sized graph, not silently sample it.
MAX_REVIEW_EVALUATION_BYTES = 256 * 1024 * 1024
MAX_REVIEW_FINDINGS = 2048
MAX_REVIEW_TRUTH_FINDINGS = 2048
MAX_REVIEW_CANDIDATES = 131_072
MAX_REVIEW_LOCATION_AUDITS = 131_072
MAX_REVIEW_JUDGE_REQUESTS = 65_536
MAX_REVIEW_ASSIGNMENTS = 2048
MAX_REVIEW_EVIDENCE_RESULTS = 2048
MAX_REVIEW_RECORD_BYTES = 64 * 1024 * 1024
MAX_REVIEW_JUDGE_DECISION_BYTES = 16 * 1024 * 1024
MAX_REVIEW_JUDGE_RECEIPT_BYTES = 16 * 1024 * 1024
MAX_REVIEW_REASON_REF_BYTES = 8 * 1024 * 1024
MAX_REVIEW_CONTEXT_SOURCES = 256
MAX_REVIEW_CONTEXT_BYTES = 64 * 1024 * 1024

# Assignment weights only establish a deterministic total order.  Location,
# severity, actionability, and Evidence are intentionally absent.
EXACT_REVIEW_EDGE_WEIGHT = 2_000_000
SEMANTIC_REVIEW_EDGE_WEIGHT_BASE = 1_000_000
MAX_REVIEW_SCORE_PPM = 999_999


class ReviewEvaluationError(ValueError):
    """The Review input, Judge merge, or persisted artifact is invalid."""


class ReviewEvaluationStatus(str, Enum):
    GRADED = "graded"
    PENDING_JUDGE = "pending_judge"
    UNGRADED = "ungraded"


class ReviewEvaluationPhase(str, Enum):
    KNOWN_INVALID = "known_invalid"
    EXPECTED_ASSIGNMENT = "expected_assignment"
    NOVEL_FACTUALITY = "novel_factuality"
    EVIDENCE_SUPPORT = "evidence_support"
    COMPLETE = "complete"


class ReviewTruthKind(str, Enum):
    EXPECTED = "expected"
    KNOWN_INVALID = "known_invalid"


class FindingMatchKind(str, Enum):
    EXACT = "exact"
    SEMANTIC = "semantic"


class FindingDisposition(str, Enum):
    MATCHED = "matched"
    DUPLICATE = "duplicate"
    NOVEL_ALLOWED = "novel_allowed"
    NOVEL_DISALLOWED = "novel_disallowed"
    KNOWN_INVALID = "known_invalid"
    UNGRADED = "ungraded"


class FindingResolution(str, Enum):
    RESOLVED = "resolved"
    PENDING_JUDGE = "pending_judge"
    JUDGE_FAILED = "judge_failed"
    UNGRADED = "ungraded"


class EvidenceSupportResolution(str, Enum):
    RESOLVED = "resolved"
    PENDING_JUDGE = "pending_judge"
    JUDGE_FAILED = "judge_failed"
    UNGRADED = "ungraded"
    NOT_REQUESTED = "not_requested"


class ReviewReasonCode(str, Enum):
    SUBMISSION_REVIEW_MISSING = "submission_review_missing"
    SUBMISSION_NOT_COMPLETED = "submission_not_completed"
    TRUTH_CLAIM_CONFLICT = "truth_claim_conflict"
    DETERMINISTIC_EXACT = "deterministic_exact"
    SEMANTIC_EQUIVALENT = "semantic_equivalent"
    SEMANTIC_PARTIAL = "semantic_partial"
    SEMANTIC_DIFFERENT = "semantic_different"
    SEMANTIC_UNKNOWN = "semantic_unknown"
    KNOWN_INVALID_MATCH = "known_invalid_match"
    EXPECTED_MATCH = "expected_match"
    DUPLICATE_FINDING = "duplicate_finding"
    NO_EXPECTED_MATCH = "no_expected_match"
    NOVEL_VERIFY = "novel_verify"
    NOVEL_FORBID = "novel_disallowed"
    NOVEL_PLAUSIBLE = "novel_plausible"
    NOVEL_FABRICATED = "novel_fabricated"
    NOVEL_UNKNOWN = "novel_unknown"
    JUDGE_PENDING = "judge_pending"
    JUDGE_FAILED = "judge_failed"
    JUDGE_UNGRADED = "judge_ungraded"
    JUDGE_UNKNOWN = "judge_unknown"
    CANDIDATE_LIMIT_EXCEEDED = "candidate_limit_exceeded"
    LOCATION_LIMIT_EXCEEDED = "location_limit_exceeded"
    JUDGE_REQUEST_LIMIT_EXCEEDED = "judge_request_limit_exceeded"
    ARTIFACT_BYTE_LIMIT_EXCEEDED = "artifact_byte_limit_exceeded"
    EVIDENCE_VALID = "evidence_valid"
    EVIDENCE_INVALID = "evidence_invalid"
    EVIDENCE_MISSING = "evidence_missing"
    EVIDENCE_SUPPORT_REQUESTED = "evidence_support_requested"
    EVIDENCE_SUPPORT_SUPPORTED = "evidence_support_supported"
    EVIDENCE_SUPPORT_WEAK = "evidence_support_weak"
    EVIDENCE_SUPPORT_UNSUPPORTED = "evidence_support_unsupported"
    EVIDENCE_SUPPORT_UNKNOWN = "evidence_support_unknown"
    EVIDENCE_SUPPORT_NOT_REQUESTED = "evidence_support_not_requested"
    REQUIRED_TRUTH_MISSED = "required_truth_missed"
    OPTIONAL_TRUTH_MISSED = "optional_truth_missed"
    NOT_STRICT_PUBLISHABLE = "not_strict_publishable"


class ReviewLimitScope(str, Enum):
    CANDIDATES = "candidates"
    LOCATION_AUDITS = "location_audits"
    JUDGE_REQUESTS = "judge_requests"
    JUDGE_REQUEST_BYTES = "judge_request_bytes"
    JUDGE_REQUEST_TOKENS = "judge_request_tokens"
    JUDGE_RESPONSE_BYTES = "judge_response_bytes"
    JUDGE_RESPONSE_TOKENS = "judge_response_tokens"
    ARTIFACT_BYTES = "artifact_bytes"
    CONTEXT = "context"


def _error(message: str) -> ReviewEvaluationError:
    return ReviewEvaluationError(message)


def _id(value: Any, context: str, maximum: int = 512) -> str:
    if type(value) is not str or not value or len(value) > maximum:
        raise _error(f"{context} must be a bounded non-empty identifier")
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError as exc:
        raise _error(f"{context} must contain valid Unicode") from exc
    if value != value.strip() or any(
        character.isspace() or ord(character) < 32 or ord(character) == 127
        for character in value
    ):
        raise _error(f"{context} contains whitespace or controls")
    return value


def _text(value: Any, context: str, maximum: int = 2 * 1024 * 1024) -> str:
    if type(value) is not str or not value or len(value) > maximum:
        raise _error(f"{context} must be bounded non-empty text")
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError as exc:
        raise _error(f"{context} must contain valid Unicode") from exc
    if "\x00" in value:
        raise _error(f"{context} may not contain NUL")
    return value


def _digest(value: Any, context: str) -> str:
    if type(value) is not str or len(value) != 64:
        raise _error(f"{context} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise _error(f"{context} must be lowercase hexadecimal") from exc
    if value != value.lower():
        raise _error(f"{context} must be lowercase hexadecimal")
    return value


def _enum_value(enum_type: type[Enum], value: Any, context: str) -> Any:
    if type(value) is not str:
        raise _error(f"{context} must be an enum string")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise _error(f"{context} has an unknown enum value") from exc


def _strict_object(value: Any, fields: Sequence[str], context: str) -> Dict[str, Any]:
    if type(value) is not dict or set(value) != set(fields) or len(value) != len(fields):
        raise _error(f"{context} has unknown or missing fields")
    return value


def _array(value: Any, context: str, maximum: int) -> list[Any]:
    if type(value) is not list or len(value) > maximum:
        raise _error(f"{context} must be a bounded array")
    return value


def _bool(value: Any, context: str) -> bool:
    if type(value) is not bool:
        raise _error(f"{context} must be a boolean")
    return value


def _optional_enum(enum_type: type[Enum], value: Any, context: str) -> Any:
    return None if value is None else _enum_value(enum_type, value, context)


def _optional_id(value: Any, context: str) -> Optional[str]:
    return None if value is None else _id(value, context)


def _optional_digest(value: Any, context: str) -> Optional[str]:
    return None if value is None else _digest(value, context)


def _score(value: Any, context: str) -> int:
    if type(value) is not int or not 0 <= value <= MAX_REVIEW_SCORE_PPM:
        raise _error(f"{context} must be an integer from 0 through 999999")
    return value


def _optional_score(value: Any, context: str) -> Optional[int]:
    return None if value is None else _score(value, context)


def _optional_location_score(value: Any, context: str) -> Optional[int]:
    if value is None:
        return None
    if type(value) is not int or not 0 <= value <= 1_000_000:
        raise _error(f"{context} must be an integer from 0 through 1000000")
    return value


def _canonical_reason_codes(
    values: Iterable[ReviewReasonCode], context: str
) -> Tuple[ReviewReasonCode, ...]:
    result = tuple(values)
    if any(type(item) is not ReviewReasonCode for item in result):
        raise _error(f"{context} contains an invalid reason code")
    if len(result) != len(set(result)) or result != tuple(sorted(result, key=lambda x: x.value)):
        raise _error(f"{context} must be unique and canonically ordered")
    return result


def _canonical_tuple(values: Sequence[Any], key: Callable[[Any], Any], context: str) -> Tuple[Any, ...]:
    result = tuple(values)
    ordered = tuple(sorted(result, key=key))
    if result != ordered:
        raise _error(f"{context} is not canonically ordered")
    return result


def _array_bytes(values: Iterable[Any], context: str, maximum: int) -> None:
    payload = [item.to_dict() if hasattr(item, "to_dict") else item for item in values]
    if len(canonical_json_bytes(payload)) > maximum:
        raise _error(f"{context} exceeds its byte budget")


def _severity_assessment(
    generated: FindingSeverity, expected: FindingSeverity
) -> SeverityAssessment:
    order = {
        FindingSeverity.LOW: 0,
        FindingSeverity.MEDIUM: 1,
        FindingSeverity.HIGH: 2,
        FindingSeverity.CRITICAL: 3,
    }
    if order[generated] == order[expected]:
        return SeverityAssessment.CONSISTENT
    return (
        SeverityAssessment.OVERSTATED
        if order[generated] > order[expected]
        else SeverityAssessment.UNDERSTATED
    )


def _deterministic_actionability(finding: SubmissionFinding) -> ActionabilityAssessment:
    # This is only a deterministic protocol fallback for exact claims.  The
    # semantic Judge remains authoritative for non-exact claims.
    return (
        ActionabilityAssessment.ACTIONABLE
        if finding.suggested_action is not None and finding.suggested_action.strip()
        else ActionabilityAssessment.NOT_ACTIONABLE
    )


def _source_to_dict(source: JudgeContextSource) -> Dict[str, Any]:
    return {
        "source_id": source.source_id,
        "source_kind": source.source_kind,
        "kind": source.kind.value,
        "trust": source.trust.value,
        "content": source.content,
        "metadata": source.metadata,
        "source_digest": source.source_digest,
    }


def _source_identity(source: JudgeContextSource) -> Dict[str, Any]:
    return {
        "source_id": source.source_id,
        "source_kind": source.source_kind,
        "kind": source.kind.value,
        "trust": source.trust.value,
        "content": source.content,
        "metadata": source.metadata,
    }


def _validate_source_digest(source: JudgeContextSource, context: str) -> None:
    if source.source_digest != canonical_sha256(_source_identity(source)):
        raise _error(f"{context} source_digest is not canonical")


def _evidence_diag_to_dict(value: EvidenceDiagnostic) -> Dict[str, Any]:
    return value.to_dict()


def _evidence_item_to_dict(value: EvidenceItemIntegrityResult) -> Dict[str, Any]:
    return value.to_dict()


def _evidence_result_to_dict(value: EvidenceIntegrityResult) -> Dict[str, Any]:
    return value.to_dict()


def _context_source_from_dict(value: Any) -> JudgeContextSource:
    payload = _strict_object(
        value,
        (
            "source_id",
            "source_kind",
            "kind",
            "trust",
            "content",
            "metadata",
            "source_digest",
        ),
        "Review context source",
    )
    from .judge import JudgeContextTrust

    source = JudgeContextSource.create(
        source_id=_id(payload["source_id"], "context source.source_id"),
        source_kind=_id(payload["source_kind"], "context source.source_kind"),
        kind=_enum_value(JudgeContextKind, payload["kind"], "context source.kind"),
        trust=_enum_value(JudgeContextTrust, payload["trust"], "context source.trust"),
        content=_text(payload["content"], "context source.content"),
        metadata=payload["metadata"],
        source_digest=_digest(payload["source_digest"], "context source.source_digest"),
    )
    _validate_source_digest(source, "Review context")
    return source


def _canonical_context_sources(
    values: Sequence[JudgeContextSource], context: str
) -> Tuple[JudgeContextSource, ...]:
    result = tuple(values)
    if any(type(item) is not JudgeContextSource for item in result):
        raise _error(f"{context} contains a non-JudgeContextSource")
    for item in result:
        _validate_source_digest(item, context)
    if len({item.source_id for item in result}) != len(result):
        raise _error(f"{context} contains duplicate source identities")
    ordered = tuple(sorted(result, key=lambda item: item.source_id))
    if result != ordered:
        raise _error(f"{context} is not canonically ordered")
    return result


def _merge_context_sources(
    values: Iterable[JudgeContextSource], context: str
) -> Tuple[JudgeContextSource, ...]:
    by_id: Dict[str, JudgeContextSource] = {}
    for source in values:
        previous = by_id.get(source.source_id)
        if previous is not None and previous != source:
            raise _error(f"{context} has a conflicting source identity: {source.source_id}")
        by_id[source.source_id] = source
    return tuple(sorted(by_id.values(), key=lambda item: item.source_id))


def _swe_truth_diff_hunk_source(
    truth_id: str,
    source: EvaluatorContextSource,
) -> JudgeContextSource:
    """Project one truth-bound SWE hunk into untrusted Judge data."""

    _id(truth_id, "evaluator context truth_id")
    if type(source) is not EvaluatorContextSource:
        raise _error("evaluator context contains an invalid source")
    if source.kind is not EvaluatorContextSourceKind.DIFF_HUNK:
        raise _error("evaluator context source must be a diff hunk")
    source_id = stable_id(
        SWE_TRUTH_DIFF_HUNK_SOURCE_ID_KIND,
        truth_id,
        source.content_sha256,
        source.provenance.digest(),
    )
    return JudgeContextSource.create(
        source_id=source_id,
        source_kind=SWE_TRUTH_DIFF_HUNK_SOURCE_KIND,
        kind=JudgeContextKind.DIFF,
        trust=JudgeContextTrust.UNTRUSTED_REPOSITORY_DATA,
        content=source.content,
        metadata={
            "revision": None,
            "path": None,
            "side": None,
            "from_line": None,
            "to_line": None,
        },
    )


@dataclass(frozen=True)
class ReviewFindingContextEntry:
    finding_id: str
    sources: Tuple[JudgeContextSource, ...] = ()

    def __post_init__(self) -> None:
        _id(self.finding_id, "finding context entry.finding_id")
        object.__setattr__(
            self,
            "sources",
            _canonical_context_sources(self.sources, "finding context entry.sources"),
        )

    @classmethod
    def create(cls, finding_id: str, sources: Sequence[JudgeContextSource] = ()) -> "ReviewFindingContextEntry":
        return cls(finding_id, tuple(sorted(tuple(sources), key=lambda item: item.source_id)))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "sources": [_source_to_dict(item) for item in self.sources],
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ReviewFindingContextEntry":
        payload = _strict_object(value, ("finding_id", "sources"), "Finding context entry")
        raw = _array(payload["sources"], "finding context entry.sources", MAX_REVIEW_CONTEXT_SOURCES)
        return cls(
            finding_id=_id(payload["finding_id"], "finding context entry.finding_id"),
            sources=tuple(_context_source_from_dict(item) for item in raw),
        )


@dataclass(frozen=True)
class ReviewPairContextEntry:
    finding_id: str
    truth_kind: ReviewTruthKind
    truth_id: str
    sources: Tuple[JudgeContextSource, ...] = ()

    def __post_init__(self) -> None:
        _id(self.finding_id, "pair context entry.finding_id")
        if type(self.truth_kind) is not ReviewTruthKind:
            raise _error("pair context entry.truth_kind has an invalid type")
        _id(self.truth_id, "pair context entry.truth_id")
        object.__setattr__(
            self,
            "sources",
            _canonical_context_sources(self.sources, "pair context entry.sources"),
        )

    @classmethod
    def create(
        cls,
        finding_id: str,
        truth_kind: ReviewTruthKind,
        truth_id: str,
        sources: Sequence[JudgeContextSource] = (),
    ) -> "ReviewPairContextEntry":
        return cls(
            finding_id,
            truth_kind,
            truth_id,
            tuple(sorted(tuple(sources), key=lambda item: item.source_id)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "truth_kind": self.truth_kind.value,
            "truth_id": self.truth_id,
            "sources": [_source_to_dict(item) for item in self.sources],
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ReviewPairContextEntry":
        payload = _strict_object(
            value,
            ("finding_id", "truth_kind", "truth_id", "sources"),
            "Pair context entry",
        )
        raw = _array(payload["sources"], "pair context entry.sources", MAX_REVIEW_CONTEXT_SOURCES)
        return cls(
            finding_id=_id(payload["finding_id"], "pair context entry.finding_id"),
            truth_kind=_enum_value(ReviewTruthKind, payload["truth_kind"], "pair context entry.truth_kind"),
            truth_id=_id(payload["truth_id"], "pair context entry.truth_id"),
            sources=tuple(_context_source_from_dict(item) for item in raw),
        )


@dataclass(frozen=True)
class ReviewContextBundle:
    """Scoped context selector used by Review Judge requests.

    Sources are not broadcast implicitly.  ``resolve_finding`` returns global
    plus one Finding scope; ``resolve_pair`` additionally returns the exact
    Finding--truth scope.  Reusing an identical source identity across scopes
    is allowed and deduplicated.  Reusing an identity with different content,
    metadata, trust, or digest is rejected at construction time.
    """

    global_sources: Tuple[JudgeContextSource, ...] = ()
    finding_entries: Tuple[ReviewFindingContextEntry, ...] = ()
    pair_entries: Tuple[ReviewPairContextEntry, ...] = ()

    def __post_init__(self) -> None:
        global_sources = _canonical_context_sources(self.global_sources, "context global_sources")
        findings = tuple(self.finding_entries)
        pairs = tuple(self.pair_entries)
        if any(type(item) is not ReviewFindingContextEntry for item in findings):
            raise _error("context finding_entries contains an invalid item")
        if any(type(item) is not ReviewPairContextEntry for item in pairs):
            raise _error("context pair_entries contains an invalid item")
        if len({item.finding_id for item in findings}) != len(findings):
            raise _error("context finding_entries contains duplicate selectors")
        if len({(item.finding_id, item.truth_kind, item.truth_id) for item in pairs}) != len(pairs):
            raise _error("context pair_entries contains duplicate selectors")
        if findings != tuple(sorted(findings, key=lambda item: item.finding_id)):
            raise _error("context finding_entries are not canonically ordered")
        if pairs != tuple(sorted(pairs, key=lambda item: (item.finding_id, item.truth_kind.value, item.truth_id))):
            raise _error("context pair_entries are not canonically ordered")
        appearances = len(global_sources) + sum(len(item.sources) for item in findings) + sum(len(item.sources) for item in pairs)
        if appearances > MAX_REVIEW_CONTEXT_SOURCES:
            raise _error("context bundle exceeds its scoped source limit")
        all_sources = [*global_sources, *(source for item in findings for source in item.sources), *(source for item in pairs for source in item.sources)]
        _merge_context_sources(all_sources, "context bundle")
        if len(canonical_json_bytes(self.to_dict())) > MAX_REVIEW_CONTEXT_BYTES:
            raise _error("context bundle exceeds its byte budget")
        object.__setattr__(self, "global_sources", global_sources)
        object.__setattr__(self, "finding_entries", findings)
        object.__setattr__(self, "pair_entries", pairs)

    @classmethod
    def create(
        cls,
        global_sources: Sequence[JudgeContextSource] = (),
        finding_entries: Sequence[ReviewFindingContextEntry] = (),
        pair_entries: Sequence[ReviewPairContextEntry] = (),
    ) -> "ReviewContextBundle":
        return cls(
            tuple(sorted(tuple(global_sources), key=lambda item: item.source_id)),
            tuple(sorted(tuple(finding_entries), key=lambda item: item.finding_id)),
            tuple(sorted(tuple(pair_entries), key=lambda item: (item.finding_id, item.truth_kind.value, item.truth_id))),
        )

    def _entry_sources(self, finding_id: str) -> Tuple[JudgeContextSource, ...]:
        return next((item.sources for item in self.finding_entries if item.finding_id == finding_id), ())

    def resolve_finding(self, finding_id: str) -> Tuple[JudgeContextSource, ...]:
        _id(finding_id, "context resolve finding_id")
        return _merge_context_sources(
            (*self.global_sources, *self._entry_sources(finding_id)),
            "scoped Finding context",
        )

    def resolve_pair(
        self,
        finding_id: str,
        truth_kind: ReviewTruthKind,
        truth_id: str,
    ) -> Tuple[JudgeContextSource, ...]:
        _id(finding_id, "context resolve finding_id")
        if type(truth_kind) is not ReviewTruthKind:
            raise _error("context resolve truth_kind has an invalid type")
        _id(truth_id, "context resolve truth_id")
        pair_sources = ()
        for item in self.pair_entries:
            if item.finding_id == finding_id and item.truth_kind is truth_kind and item.truth_id == truth_id:
                pair_sources = item.sources
                break
        return _merge_context_sources(
            (*self.resolve_finding(finding_id), *pair_sources),
            "scoped pair context",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_sources": [_source_to_dict(item) for item in self.global_sources],
            "finding_entries": [item.to_dict() for item in self.finding_entries],
            "pair_entries": [item.to_dict() for item in self.pair_entries],
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ReviewContextBundle":
        payload = _strict_object(value, ("global_sources", "finding_entries", "pair_entries"), "Review context bundle")
        global_raw = _array(payload["global_sources"], "context global_sources", MAX_REVIEW_CONTEXT_SOURCES)
        finding_raw = _array(payload["finding_entries"], "context finding_entries", MAX_REVIEW_CONTEXT_SOURCES)
        pair_raw = _array(payload["pair_entries"], "context pair_entries", MAX_REVIEW_CONTEXT_SOURCES)
        return cls(
            global_sources=tuple(_context_source_from_dict(item) for item in global_raw),
            finding_entries=tuple(ReviewFindingContextEntry.from_dict(item) for item in finding_raw),
            pair_entries=tuple(ReviewPairContextEntry.from_dict(item) for item in pair_raw),
        )

    def digest(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class ReviewLimitFailure:
    scope: ReviewLimitScope
    observed: int
    limit: int
    reason_code: ReviewReasonCode

    def __post_init__(self) -> None:
        if type(self.scope) is not ReviewLimitScope:
            raise _error("limit failure.scope has an invalid type")
        if type(self.observed) is not int or self.observed < 0:
            raise _error("limit failure.observed must be non-negative")
        if type(self.limit) is not int or self.limit < 1:
            raise _error("limit failure.limit must be positive")
        if type(self.reason_code) is not ReviewReasonCode:
            raise _error("limit failure.reason_code has an invalid type")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scope": self.scope.value,
            "observed": self.observed,
            "limit": self.limit,
            "reason_code": self.reason_code.value,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ReviewLimitFailure":
        payload = _strict_object(value, ("scope", "observed", "limit", "reason_code"), "Review limit failure")
        return cls(
            scope=_enum_value(ReviewLimitScope, payload["scope"], "limit failure.scope"),
            observed=payload["observed"],
            limit=payload["limit"],
            reason_code=_enum_value(ReviewReasonCode, payload["reason_code"], "limit failure.reason_code"),
        )


@dataclass(frozen=True)
class LocationAuditRecord:
    finding_id: str
    truth_kind: ReviewTruthKind
    truth_id: str
    truth_location_index: int
    truth_location: TruthLocation
    match: LocationMatchResult

    def __post_init__(self) -> None:
        _id(self.finding_id, "location audit.finding_id")
        if type(self.truth_kind) is not ReviewTruthKind:
            raise _error("location audit.truth_kind has an invalid type")
        _id(self.truth_id, "location audit.truth_id")
        if type(self.truth_location_index) is not int or self.truth_location_index < 0:
            raise _error("location audit.truth_location_index is invalid")
        if type(self.truth_location) is not TruthLocation:
            raise _error("location audit.truth_location has an invalid type")
        if type(self.match) is not LocationMatchResult:
            raise _error("location audit.match has an invalid type")

    @property
    def score(self) -> int:
        return self.match.score

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "truth_kind": self.truth_kind.value,
            "truth_id": self.truth_id,
            "truth_location_index": self.truth_location_index,
            "truth_location": self.truth_location.to_dict(),
            "match": {
                "matched": self.match.matched,
                "score": self.match.score,
                "reasons": [item.value for item in self.match.reasons],
            },
        }

    @classmethod
    def from_dict(cls, value: Any) -> "LocationAuditRecord":
        payload = _strict_object(
            value,
            ("finding_id", "truth_kind", "truth_id", "truth_location_index", "truth_location", "match"),
            "location audit",
        )
        match_payload = _strict_object(payload["match"], ("matched", "score", "reasons"), "location audit.match")
        reasons = _array(match_payload["reasons"], "location audit.match.reasons", 64)
        return cls(
            finding_id=_id(payload["finding_id"], "location audit.finding_id"),
            truth_kind=_enum_value(ReviewTruthKind, payload["truth_kind"], "location audit.truth_kind"),
            truth_id=_id(payload["truth_id"], "location audit.truth_id"),
            truth_location_index=payload["truth_location_index"],
            truth_location=TruthLocation.from_dict(payload["truth_location"]),
            match=LocationMatchResult(
                matched=_bool(match_payload["matched"], "location audit.match.matched"),
                score=match_payload["score"],
                reasons=tuple(_enum_value(LocationMatchReason, item, "location audit.match.reason") for item in reasons),
            ),
        )


@dataclass(frozen=True)
class ReviewCandidateRecord:
    finding_id: str
    truth_kind: ReviewTruthKind
    truth_id: str
    match_kind: FindingMatchKind
    request_id: Optional[str]
    blind_request_id: Optional[str]
    relation: Optional[FindingMatchRelation]
    score_ppm: Optional[int]
    edge_weight: Optional[int]
    selected: bool
    resolution: FindingResolution
    severity_assessment: Optional[SeverityAssessment]
    actionability: Optional[ActionabilityAssessment]
    location_match_count: int
    best_location_score: Optional[int]
    best_location_matched: Optional[bool]
    reason_codes: Tuple[ReviewReasonCode, ...]

    def __post_init__(self) -> None:
        _id(self.finding_id, "candidate.finding_id")
        if type(self.truth_kind) is not ReviewTruthKind:
            raise _error("candidate.truth_kind has an invalid type")
        _id(self.truth_id, "candidate.truth_id")
        if type(self.match_kind) is not FindingMatchKind:
            raise _error("candidate.match_kind has an invalid type")
        if type(self.resolution) is not FindingResolution:
            raise _error("candidate.resolution has an invalid type")
        if self.severity_assessment is not None and type(self.severity_assessment) is not SeverityAssessment:
            raise _error("candidate.severity_assessment has an invalid type")
        if self.actionability is not None and type(self.actionability) is not ActionabilityAssessment:
            raise _error("candidate.actionability has an invalid type")
        _optional_id(self.request_id, "candidate.request_id")
        _optional_id(self.blind_request_id, "candidate.blind_request_id")
        if self.match_kind is FindingMatchKind.EXACT:
            if self.request_id is not None or self.blind_request_id is not None:
                raise _error("exact candidate cannot have a Judge request")
            if self.relation is not FindingMatchRelation.EQUIVALENT:
                raise _error("exact candidate must be equivalent")
            if self.score_ppm is not None or self.edge_weight != EXACT_REVIEW_EDGE_WEIGHT:
                raise _error("exact candidate score/weight is not canonical")
            if self.resolution is not FindingResolution.RESOLVED:
                raise _error("exact candidate must be resolved")
        else:
            if self.request_id is None or self.blind_request_id is None:
                raise _error("semantic candidate must bind a Judge request")
        if self.relation is not None and type(self.relation) is not FindingMatchRelation:
            raise _error("candidate.relation has an invalid type")
        _optional_score(self.score_ppm, "candidate.score_ppm")
        if self.edge_weight is not None:
            if type(self.edge_weight) is not int or self.edge_weight < 1:
                raise _error("candidate.edge_weight must be positive")
            if self.relation is not FindingMatchRelation.EQUIVALENT:
                raise _error("only equivalent candidates may have an edge weight")
        if self.match_kind is FindingMatchKind.SEMANTIC:
            if self.relation is None:
                if any(
                    item is not None
                    for item in (
                        self.score_ppm,
                        self.severity_assessment,
                        self.actionability,
                    )
                ) or self.edge_weight is not None:
                    raise _error("unresolved semantic candidate contains Judge decision fields")
                if self.resolution not in {
                    FindingResolution.PENDING_JUDGE,
                    FindingResolution.JUDGE_FAILED,
                    FindingResolution.UNGRADED,
                }:
                    raise _error("semantic candidate without a decision has invalid resolution")
            else:
                decision_fields = (
                    self.score_ppm,
                    self.actionability,
                )
                if self.truth_kind is ReviewTruthKind.KNOWN_INVALID:
                    decision_fields += (self.severity_assessment,)
                if any(item is None for item in decision_fields):
                    raise _error("resolved semantic candidate lacks Judge decision fields")
                if self.relation is FindingMatchRelation.UNKNOWN:
                    if self.resolution is not FindingResolution.UNGRADED or self.edge_weight is not None:
                        raise _error("unknown semantic candidate must remain ungraded")
                elif self.resolution is not FindingResolution.RESOLVED:
                    raise _error("terminal semantic relation must be resolved")
                if self.relation is FindingMatchRelation.EQUIVALENT:
                    if self.edge_weight != SEMANTIC_REVIEW_EDGE_WEIGHT_BASE + int(self.score_ppm):
                        raise _error("semantic equivalent edge weight is not canonical")
                elif self.edge_weight is not None:
                    raise _error("non-equivalent semantic candidate cannot have an edge")
        if type(self.selected) is not bool:
            raise _error("candidate.selected must be boolean")
        if self.selected and (
            self.relation is not FindingMatchRelation.EQUIVALENT
            or self.resolution is not FindingResolution.RESOLVED
            or self.edge_weight is None
        ):
            raise _error("selected candidate must be a resolved equivalent edge")
        if type(self.location_match_count) is not int or self.location_match_count < 0:
            raise _error("candidate.location_match_count is invalid")
        _optional_location_score(self.best_location_score, "candidate.best_location_score")
        if self.best_location_matched is not None and type(self.best_location_matched) is not bool:
            raise _error("candidate.best_location_matched must be bool or null")
        _canonical_reason_codes(self.reason_codes, "candidate.reason_codes")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "truth_kind": self.truth_kind.value,
            "truth_id": self.truth_id,
            "match_kind": self.match_kind.value,
            "request_id": self.request_id,
            "blind_request_id": self.blind_request_id,
            "relation": None if self.relation is None else self.relation.value,
            "score_ppm": self.score_ppm,
            "edge_weight": self.edge_weight,
            "selected": self.selected,
            "resolution": self.resolution.value,
            "severity_assessment": None if self.severity_assessment is None else self.severity_assessment.value,
            "actionability": None if self.actionability is None else self.actionability.value,
            "location_match_count": self.location_match_count,
            "best_location_score": self.best_location_score,
            "best_location_matched": self.best_location_matched,
            "reason_codes": [item.value for item in self.reason_codes],
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ReviewCandidateRecord":
        fields = (
            "finding_id", "truth_kind", "truth_id", "match_kind", "request_id", "blind_request_id",
            "relation", "score_ppm", "edge_weight", "selected", "resolution", "severity_assessment",
            "actionability", "location_match_count", "best_location_score", "best_location_matched", "reason_codes",
        )
        payload = _strict_object(value, fields, "Review candidate")
        reasons = _array(payload["reason_codes"], "candidate.reason_codes", 64)
        return cls(
            finding_id=_id(payload["finding_id"], "candidate.finding_id"),
            truth_kind=_enum_value(ReviewTruthKind, payload["truth_kind"], "candidate.truth_kind"),
            truth_id=_id(payload["truth_id"], "candidate.truth_id"),
            match_kind=_enum_value(FindingMatchKind, payload["match_kind"], "candidate.match_kind"),
            request_id=_optional_id(payload["request_id"], "candidate.request_id"),
            blind_request_id=_optional_id(payload["blind_request_id"], "candidate.blind_request_id"),
            relation=_optional_enum(FindingMatchRelation, payload["relation"], "candidate.relation"),
            score_ppm=_optional_score(payload["score_ppm"], "candidate.score_ppm"),
            edge_weight=payload["edge_weight"],
            selected=_bool(payload["selected"], "candidate.selected"),
            resolution=_enum_value(FindingResolution, payload["resolution"], "candidate.resolution"),
            severity_assessment=_optional_enum(SeverityAssessment, payload["severity_assessment"], "candidate.severity_assessment"),
            actionability=_optional_enum(ActionabilityAssessment, payload["actionability"], "candidate.actionability"),
            location_match_count=payload["location_match_count"],
            best_location_score=_optional_location_score(payload["best_location_score"], "candidate.best_location_score"),
            best_location_matched=payload["best_location_matched"],
            reason_codes=tuple(_enum_value(ReviewReasonCode, item, "candidate.reason_code") for item in reasons),
        )


@dataclass(frozen=True)
class ReviewAssignmentRecord:
    finding_id: str
    truth_id: str
    match_kind: FindingMatchKind
    weight: int
    request_id: Optional[str]

    def __post_init__(self) -> None:
        _id(self.finding_id, "assignment.finding_id")
        _id(self.truth_id, "assignment.truth_id")
        if type(self.match_kind) is not FindingMatchKind:
            raise _error("assignment.match_kind has an invalid type")
        if type(self.weight) is not int or self.weight < 1:
            raise _error("assignment.weight must be positive")
        if self.match_kind is FindingMatchKind.EXACT and self.weight != EXACT_REVIEW_EDGE_WEIGHT:
            raise _error("exact assignment weight is not canonical")
        if self.match_kind is FindingMatchKind.SEMANTIC and self.weight < SEMANTIC_REVIEW_EDGE_WEIGHT_BASE:
            raise _error("semantic assignment weight is not canonical")
        _optional_id(self.request_id, "assignment.request_id")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "truth_id": self.truth_id,
            "match_kind": self.match_kind.value,
            "weight": self.weight,
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ReviewAssignmentRecord":
        payload = _strict_object(value, ("finding_id", "truth_id", "match_kind", "weight", "request_id"), "Review assignment")
        return cls(
            finding_id=_id(payload["finding_id"], "assignment.finding_id"),
            truth_id=_id(payload["truth_id"], "assignment.truth_id"),
            match_kind=_enum_value(FindingMatchKind, payload["match_kind"], "assignment.match_kind"),
            weight=payload["weight"],
            request_id=_optional_id(payload["request_id"], "assignment.request_id"),
        )


def _strict_publishable(
    issue_resolution: FindingResolution,
    issue_judgement: Any,
    disposition: FindingDisposition,
    evidence_integrity: EvidenceIntegrity,
    support_resolution: EvidenceSupportResolution,
    evidence_support: EvidenceSupport,
) -> bool:
    from .models import IssueJudgement

    return (
        issue_resolution is FindingResolution.RESOLVED
        and issue_judgement is IssueJudgement.CONFIRMED
        and disposition is FindingDisposition.MATCHED
        and evidence_integrity is EvidenceIntegrity.VALID
        and support_resolution is EvidenceSupportResolution.RESOLVED
        and evidence_support is EvidenceSupport.SUPPORTED
    )


@dataclass(frozen=True)
class FindingOutcome:
    finding_id: str
    issue_resolution: FindingResolution
    issue_judgement: Any
    disposition: FindingDisposition
    matched_expected_truth_id: Optional[str]
    matched_known_invalid_truth_id: Optional[str]
    duplicate_truth_id: Optional[str]
    duplicate_of_finding_id: Optional[str]
    novel_request_id: Optional[str]
    severity_assessment: Optional[SeverityAssessment]
    actionability: Optional[ActionabilityAssessment]
    evidence_integrity: EvidenceIntegrity
    evidence_support_resolution: EvidenceSupportResolution
    evidence_support: EvidenceSupport
    evidence_support_request_id: Optional[str]
    strict_publishable: bool
    reason_codes: Tuple[ReviewReasonCode, ...]

    def __post_init__(self) -> None:
        from .models import IssueJudgement

        _id(self.finding_id, "finding outcome.finding_id")
        if type(self.issue_resolution) is not FindingResolution:
            raise _error("finding outcome.issue_resolution has an invalid type")
        if type(self.issue_judgement) is not IssueJudgement:
            raise _error("finding outcome.issue_judgement has an invalid type")
        if type(self.disposition) is not FindingDisposition:
            raise _error("finding outcome.disposition has an invalid type")
        for value, context in (
            (self.matched_expected_truth_id, "matched_expected_truth_id"),
            (self.matched_known_invalid_truth_id, "matched_known_invalid_truth_id"),
            (self.duplicate_truth_id, "duplicate_truth_id"),
            (self.duplicate_of_finding_id, "duplicate_of_finding_id"),
            (self.novel_request_id, "novel_request_id"),
            (self.evidence_support_request_id, "evidence_support_request_id"),
        ):
            _optional_id(value, f"finding outcome.{context}")
        if self.severity_assessment is not None and type(self.severity_assessment) is not SeverityAssessment:
            raise _error("finding outcome.severity_assessment has an invalid type")
        if self.actionability is not None and type(self.actionability) is not ActionabilityAssessment:
            raise _error("finding outcome.actionability has an invalid type")
        if type(self.evidence_integrity) is not EvidenceIntegrity:
            raise _error("finding outcome.evidence_integrity has an invalid type")
        if type(self.evidence_support_resolution) is not EvidenceSupportResolution:
            raise _error("finding outcome.evidence_support_resolution has an invalid type")
        if type(self.evidence_support) is not EvidenceSupport:
            raise _error("finding outcome.evidence_support has an invalid type")
        _canonical_reason_codes(self.reason_codes, "finding outcome.reason_codes")
        self._validate_disposition()
        expected_publishable = _strict_publishable(
            self.issue_resolution,
            self.issue_judgement,
            self.disposition,
            self.evidence_integrity,
            self.evidence_support_resolution,
            self.evidence_support,
        )
        if self.strict_publishable is not expected_publishable:
            raise _error("strict_publishable is not the canonical projection")

    def _validate_disposition(self) -> None:
        if self.disposition is FindingDisposition.MATCHED:
            if self.matched_expected_truth_id is None or any(
                value is not None
                for value in (self.matched_known_invalid_truth_id, self.duplicate_truth_id, self.duplicate_of_finding_id, self.novel_request_id)
            ):
                raise _error("matched outcome has invalid truth fields")
            if self.issue_resolution is not FindingResolution.RESOLVED or self.issue_judgement.value != "confirmed":
                raise _error("matched outcome must be resolved and confirmed")
        elif self.disposition is FindingDisposition.KNOWN_INVALID:
            if self.matched_known_invalid_truth_id is None or any(
                value is not None
                for value in (self.matched_expected_truth_id, self.duplicate_truth_id, self.duplicate_of_finding_id, self.novel_request_id)
            ):
                raise _error("known-invalid outcome has invalid truth fields")
            if self.issue_judgement.value != "fabricated":
                raise _error("known-invalid outcome must be fabricated")
            if self.issue_resolution is not FindingResolution.RESOLVED:
                raise _error("known-invalid outcome must be resolved")
        elif self.disposition is FindingDisposition.DUPLICATE:
            if self.duplicate_truth_id is None or self.duplicate_of_finding_id is None or any(
                value is not None
                for value in (self.matched_expected_truth_id, self.matched_known_invalid_truth_id, self.novel_request_id)
            ):
                raise _error("duplicate outcome has invalid truth fields")
            if self.issue_judgement.value != "plausible":
                raise _error("duplicate outcome must be plausible")
            if self.issue_resolution is not FindingResolution.RESOLVED:
                raise _error("duplicate outcome must be resolved")
        elif self.disposition is FindingDisposition.NOVEL_ALLOWED:
            if self.novel_request_id is None or any(
                value is not None
                for value in (self.matched_expected_truth_id, self.matched_known_invalid_truth_id, self.duplicate_truth_id, self.duplicate_of_finding_id)
            ):
                raise _error("novel outcome must have exactly one factuality request")
            if self.issue_resolution is not FindingResolution.RESOLVED or self.issue_judgement.value not in {"plausible", "fabricated"}:
                raise _error("resolved novel outcome has invalid judgement")
        elif self.disposition is FindingDisposition.NOVEL_DISALLOWED:
            if self.novel_request_id is not None or self.issue_judgement.value != "unknown" or self.issue_resolution is not FindingResolution.UNGRADED:
                raise _error("novel-disallowed outcome is not canonical")
        elif self.disposition is FindingDisposition.UNGRADED:
            if any(
                value is not None
                for value in (self.matched_expected_truth_id, self.matched_known_invalid_truth_id, self.duplicate_truth_id, self.duplicate_of_finding_id)
            ):
                raise _error("ungraded outcome cannot contain a truth assignment")
            if self.issue_judgement.value != "unknown":
                raise _error("ungraded outcome must be unknown")
            if self.issue_resolution is FindingResolution.RESOLVED:
                raise _error("ungraded outcome cannot be resolved")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "issue_resolution": self.issue_resolution.value,
            "issue_judgement": self.issue_judgement.value,
            "disposition": self.disposition.value,
            "matched_expected_truth_id": self.matched_expected_truth_id,
            "matched_known_invalid_truth_id": self.matched_known_invalid_truth_id,
            "duplicate_truth_id": self.duplicate_truth_id,
            "duplicate_of_finding_id": self.duplicate_of_finding_id,
            "novel_request_id": self.novel_request_id,
            "severity_assessment": None if self.severity_assessment is None else self.severity_assessment.value,
            "actionability": None if self.actionability is None else self.actionability.value,
            "evidence_integrity": self.evidence_integrity.value,
            "evidence_support_resolution": self.evidence_support_resolution.value,
            "evidence_support": self.evidence_support.value,
            "evidence_support_request_id": self.evidence_support_request_id,
            "strict_publishable": self.strict_publishable,
            "reason_codes": [item.value for item in self.reason_codes],
        }

    @classmethod
    def from_dict(cls, value: Any) -> "FindingOutcome":
        fields = (
            "finding_id", "issue_resolution", "issue_judgement", "disposition",
            "matched_expected_truth_id", "matched_known_invalid_truth_id", "duplicate_truth_id",
            "duplicate_of_finding_id", "novel_request_id", "severity_assessment", "actionability",
            "evidence_integrity", "evidence_support_resolution", "evidence_support",
            "evidence_support_request_id", "strict_publishable", "reason_codes",
        )
        payload = _strict_object(value, fields, "Finding outcome")
        reasons = _array(payload["reason_codes"], "finding outcome.reason_codes", 64)
        from .models import IssueJudgement

        return cls(
            finding_id=_id(payload["finding_id"], "finding outcome.finding_id"),
            issue_resolution=_enum_value(FindingResolution, payload["issue_resolution"], "finding outcome.issue_resolution"),
            issue_judgement=_enum_value(IssueJudgement, payload["issue_judgement"], "finding outcome.issue_judgement"),
            disposition=_enum_value(FindingDisposition, payload["disposition"], "finding outcome.disposition"),
            matched_expected_truth_id=_optional_id(payload["matched_expected_truth_id"], "matched_expected_truth_id"),
            matched_known_invalid_truth_id=_optional_id(payload["matched_known_invalid_truth_id"], "matched_known_invalid_truth_id"),
            duplicate_truth_id=_optional_id(payload["duplicate_truth_id"], "duplicate_truth_id"),
            duplicate_of_finding_id=_optional_id(payload["duplicate_of_finding_id"], "duplicate_of_finding_id"),
            novel_request_id=_optional_id(payload["novel_request_id"], "novel_request_id"),
            severity_assessment=_optional_enum(SeverityAssessment, payload["severity_assessment"], "finding outcome.severity_assessment"),
            actionability=_optional_enum(ActionabilityAssessment, payload["actionability"], "finding outcome.actionability"),
            evidence_integrity=_enum_value(EvidenceIntegrity, payload["evidence_integrity"], "finding outcome.evidence_integrity"),
            evidence_support_resolution=_enum_value(EvidenceSupportResolution, payload["evidence_support_resolution"], "finding outcome.evidence_support_resolution"),
            evidence_support=_enum_value(EvidenceSupport, payload["evidence_support"], "finding outcome.evidence_support"),
            evidence_support_request_id=_optional_id(payload["evidence_support_request_id"], "evidence_support_request_id"),
            strict_publishable=_bool(payload["strict_publishable"], "finding outcome.strict_publishable"),
            reason_codes=tuple(_enum_value(ReviewReasonCode, item, "finding outcome.reason_code") for item in reasons),
        )


@dataclass(frozen=True)
class ReviewJudgeRequestRecord:
    request_id: str
    phase: ReviewEvaluationPhase
    finding_id: str
    truth_kind: Optional[ReviewTruthKind]
    truth_id: Optional[str]
    request: BlindJudgeInput

    def __post_init__(self) -> None:
        _id(self.request_id, "Judge request.request_id")
        if self.phase is ReviewEvaluationPhase.COMPLETE:
            raise _error("Judge request cannot belong to complete phase")
        _id(self.finding_id, "Judge request.finding_id")
        if self.truth_kind is not None and type(self.truth_kind) is not ReviewTruthKind:
            raise _error("Judge request.truth_kind has an invalid type")
        _optional_id(self.truth_id, "Judge request.truth_id")
        if type(self.request) is not BlindJudgeInput:
            raise _error("Judge request.request must be BlindJudgeInput")
        if self.request.source_request_id != self.request_id:
            raise _error("Judge request source ID does not match record")
        expected_task = {
            ReviewEvaluationPhase.KNOWN_INVALID: JudgeTask.FINDING_EQUIVALENCE,
            ReviewEvaluationPhase.EXPECTED_ASSIGNMENT: JudgeTask.FINDING_EQUIVALENCE,
            ReviewEvaluationPhase.NOVEL_FACTUALITY: JudgeTask.NOVEL_FACTUALITY,
            ReviewEvaluationPhase.EVIDENCE_SUPPORT: JudgeTask.EVIDENCE_SUPPORT,
        }[self.phase]
        if self.request.task is not expected_task:
            raise _error("Judge request task does not match its phase")

    @property
    def task(self) -> JudgeTask:
        return self.request.task

    @property
    def request_digest(self) -> str:
        return self.request.digest()

    @property
    def blind_request_id(self) -> str:
        return self.request.request_id

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "phase": self.phase.value,
            "finding_id": self.finding_id,
            "truth_kind": None if self.truth_kind is None else self.truth_kind.value,
            "truth_id": self.truth_id,
            "request": self.request.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ReviewJudgeRequestRecord":
        payload = _strict_object(value, ("request_id", "phase", "finding_id", "truth_kind", "truth_id", "request"), "Review Judge request")
        return cls(
            request_id=_id(payload["request_id"], "Judge request.request_id"),
            phase=_enum_value(ReviewEvaluationPhase, payload["phase"], "Judge request.phase"),
            finding_id=_id(payload["finding_id"], "Judge request.finding_id"),
            truth_kind=_optional_enum(ReviewTruthKind, payload["truth_kind"], "Judge request.truth_kind"),
            truth_id=_optional_id(payload["truth_id"], "Judge request.truth_id"),
            request=BlindJudgeInput.from_dict(payload["request"]),
        )


def _receipt_fields(
    request_id: str,
    task: JudgeTask,
    request_digest: str,
    evaluator_execution_digest: str,
    judge_result_digest: str,
    blind_request_id: str,
) -> None:
    _id(request_id, "Judge receipt.request_id")
    if type(task) is not JudgeTask:
        raise _error("Judge receipt.task has an invalid type")
    _digest(request_digest, "Judge receipt.request_digest")
    _digest(evaluator_execution_digest, "Judge receipt.evaluator_execution_digest")
    _digest(judge_result_digest, "Judge receipt.judge_result_digest")
    _id(blind_request_id, "Judge receipt.blind_request_id")


@dataclass(frozen=True)
class ReviewJudgeDecisionReceipt:
    request_id: str
    task: JudgeTask
    request_digest: str
    evaluator_execution_digest: str
    judge_result_digest: str
    blind_request_id: str
    decision: Union[FindingEquivalenceJudgeDecision, NovelFactualityJudgeDecision, EvidenceSupportJudgeDecision]

    def __post_init__(self) -> None:
        _receipt_fields(self.request_id, self.task, self.request_digest, self.evaluator_execution_digest, self.judge_result_digest, self.blind_request_id)
        if type(self.decision) not in (FindingEquivalenceJudgeDecision, NovelFactualityJudgeDecision, EvidenceSupportJudgeDecision):
            raise _error("Judge decision receipt has an invalid decision type")
        expected = {
            JudgeTask.FINDING_EQUIVALENCE: FindingEquivalenceJudgeDecision,
            JudgeTask.NOVEL_FACTUALITY: NovelFactualityJudgeDecision,
            JudgeTask.EVIDENCE_SUPPORT: EvidenceSupportJudgeDecision,
        }.get(self.task)
        if expected is None:
            raise _error("Intent Judge results are not Review receipts")
        if type(self.decision) is not expected or self.decision.request_id != self.request_id:
            raise _error("Judge decision receipt task/request binding is invalid")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "task": self.task.value,
            "request_digest": self.request_digest,
            "evaluator_execution_digest": self.evaluator_execution_digest,
            "judge_result_digest": self.judge_result_digest,
            "blind_request_id": self.blind_request_id,
            "decision": self.decision.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ReviewJudgeDecisionReceipt":
        payload = _strict_object(value, ("request_id", "task", "request_digest", "evaluator_execution_digest", "judge_result_digest", "blind_request_id", "decision"), "Judge decision receipt")
        task = _enum_value(JudgeTask, payload["task"], "Judge receipt.task")
        decision_type = {
            JudgeTask.FINDING_EQUIVALENCE: FindingEquivalenceJudgeDecision,
            JudgeTask.NOVEL_FACTUALITY: NovelFactualityJudgeDecision,
            JudgeTask.EVIDENCE_SUPPORT: EvidenceSupportJudgeDecision,
        }.get(task)
        if decision_type is None:
            raise _error("Intent Judge results are not Review receipts")
        return cls(
            request_id=_id(payload["request_id"], "Judge receipt.request_id"),
            task=task,
            request_digest=_digest(payload["request_digest"], "Judge receipt.request_digest"),
            evaluator_execution_digest=_digest(payload["evaluator_execution_digest"], "Judge receipt.evaluator_execution_digest"),
            judge_result_digest=_digest(payload["judge_result_digest"], "Judge receipt.judge_result_digest"),
            blind_request_id=_id(payload["blind_request_id"], "Judge receipt.blind_request_id"),
            decision=decision_type.from_dict(payload["decision"]),
        )


@dataclass(frozen=True)
class ReviewJudgeFailureReceipt:
    request_id: str
    task: JudgeTask
    request_digest: str
    evaluator_execution_digest: str
    judge_result_digest: str
    blind_request_id: str
    failure: JudgeFailure

    def __post_init__(self) -> None:
        _receipt_fields(self.request_id, self.task, self.request_digest, self.evaluator_execution_digest, self.judge_result_digest, self.blind_request_id)
        if self.task is JudgeTask.INTENT_EQUIVALENCE or type(self.failure) is not JudgeFailure:
            raise _error("Judge failure receipt has an invalid task/failure")

    @property
    def failure_code(self) -> JudgeFailureCode:
        return self.failure.code

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "task": self.task.value,
            "request_digest": self.request_digest,
            "evaluator_execution_digest": self.evaluator_execution_digest,
            "judge_result_digest": self.judge_result_digest,
            "blind_request_id": self.blind_request_id,
            "failure": self.failure.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ReviewJudgeFailureReceipt":
        payload = _strict_object(value, ("request_id", "task", "request_digest", "evaluator_execution_digest", "judge_result_digest", "blind_request_id", "failure"), "Judge failure receipt")
        from .judge import JudgeFailure

        return cls(
            request_id=_id(payload["request_id"], "Judge receipt.request_id"),
            task=_enum_value(JudgeTask, payload["task"], "Judge receipt.task"),
            request_digest=_digest(payload["request_digest"], "Judge receipt.request_digest"),
            evaluator_execution_digest=_digest(payload["evaluator_execution_digest"], "Judge receipt.evaluator_execution_digest"),
            judge_result_digest=_digest(payload["judge_result_digest"], "Judge receipt.judge_result_digest"),
            blind_request_id=_id(payload["blind_request_id"], "Judge receipt.blind_request_id"),
            failure=JudgeFailure.from_dict(payload["failure"]),
        )


@dataclass(frozen=True)
class ReviewJudgeUngradedReceipt:
    request_id: str
    task: JudgeTask
    request_digest: str
    evaluator_execution_digest: str
    judge_result_digest: str
    blind_request_id: str
    ungraded_reason: JudgeUngradedReason

    def __post_init__(self) -> None:
        _receipt_fields(self.request_id, self.task, self.request_digest, self.evaluator_execution_digest, self.judge_result_digest, self.blind_request_id)
        if self.task is JudgeTask.INTENT_EQUIVALENCE or type(self.ungraded_reason) is not JudgeUngradedReason:
            raise _error("Judge ungraded receipt has an invalid task/reason")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "task": self.task.value,
            "request_digest": self.request_digest,
            "evaluator_execution_digest": self.evaluator_execution_digest,
            "judge_result_digest": self.judge_result_digest,
            "blind_request_id": self.blind_request_id,
            "ungraded_reason": self.ungraded_reason.value,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ReviewJudgeUngradedReceipt":
        payload = _strict_object(value, ("request_id", "task", "request_digest", "evaluator_execution_digest", "judge_result_digest", "blind_request_id", "ungraded_reason"), "Judge ungraded receipt")
        return cls(
            request_id=_id(payload["request_id"], "Judge receipt.request_id"),
            task=_enum_value(JudgeTask, payload["task"], "Judge receipt.task"),
            request_digest=_digest(payload["request_digest"], "Judge receipt.request_digest"),
            evaluator_execution_digest=_digest(payload["evaluator_execution_digest"], "Judge receipt.evaluator_execution_digest"),
            judge_result_digest=_digest(payload["judge_result_digest"], "Judge receipt.judge_result_digest"),
            blind_request_id=_id(payload["blind_request_id"], "Judge receipt.blind_request_id"),
            ungraded_reason=_enum_value(JudgeUngradedReason, payload["ungraded_reason"], "Judge receipt.ungraded_reason"),
        )


@dataclass(frozen=True)
class ReviewCoverage:
    judge_request_count: int
    judge_graded_count: int
    judge_failed_count: int
    judge_ungraded_count: int
    judge_pending_count: int
    semantic_unknown_count: int
    finding_count: int
    finding_resolved_count: int
    evidence_result_count: int

    def __post_init__(self) -> None:
        for name in (
            "judge_request_count",
            "judge_graded_count",
            "judge_failed_count",
            "judge_ungraded_count",
            "judge_pending_count",
            "semantic_unknown_count",
            "finding_count",
            "finding_resolved_count",
            "evidence_result_count",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise _error(f"coverage.{name} must be non-negative")
        if (
            self.judge_graded_count
            + self.judge_failed_count
            + self.judge_ungraded_count
            + self.judge_pending_count
            != self.judge_request_count
        ):
            raise _error("coverage Judge resolution counts do not add up")
        if self.finding_resolved_count > self.finding_count:
            raise _error("coverage resolved Findings exceed Finding count")
        if self.evidence_result_count != self.finding_count:
            raise _error("coverage Evidence count must equal Finding count")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "judge_request_count": self.judge_request_count,
            "judge_graded_count": self.judge_graded_count,
            "judge_failed_count": self.judge_failed_count,
            "judge_ungraded_count": self.judge_ungraded_count,
            "judge_pending_count": self.judge_pending_count,
            "semantic_unknown_count": self.semantic_unknown_count,
            "finding_count": self.finding_count,
            "finding_resolved_count": self.finding_resolved_count,
            "evidence_result_count": self.evidence_result_count,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ReviewCoverage":
        fields = (
            "judge_request_count", "judge_graded_count", "judge_failed_count",
            "judge_ungraded_count", "judge_pending_count", "semantic_unknown_count",
            "finding_count", "finding_resolved_count", "evidence_result_count",
        )
        payload = _strict_object(value, fields, "Review coverage")
        return cls(**{name: payload[name] for name in fields})


@dataclass(frozen=True)
class ReviewMetricInputs:
    scorable: bool
    generated_finding_count: int
    expected_truth_count: int
    required_expected_truth_count: int
    matched_finding_count: int
    matched_expected_truth_count: int
    matched_required_truth_count: int
    duplicate_finding_count: int
    known_invalid_finding_count: int
    plausible_novel_count: int
    fabricated_finding_count: int
    unknown_finding_count: int
    unmatched_expected_truth_count: int
    unmatched_required_truth_count: int
    evidence_valid_count: int
    evidence_invalid_count: int
    evidence_missing_count: int
    evidence_supported_count: int
    evidence_weak_count: int
    evidence_unsupported_count: int
    evidence_support_unknown_count: int
    strict_publishable_count: int

    def __post_init__(self) -> None:
        if type(self.scorable) is not bool:
            raise _error("metrics.scorable must be boolean")
        for name in self.__dataclass_fields__:
            if name == "scorable":
                continue
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise _error(f"metrics.{name} must be non-negative")
        if self.matched_expected_truth_count > self.expected_truth_count:
            raise _error("metrics matched truth count exceeds expected truth")
        if self.matched_required_truth_count > self.required_expected_truth_count:
            raise _error("metrics matched required truth count exceeds required truth")
        if (
            self.evidence_valid_count
            + self.evidence_invalid_count
            + self.evidence_missing_count
            != self.generated_finding_count
        ):
            raise _error("metrics Evidence integrity counts do not cover Findings")
        if self.strict_publishable_count > self.matched_finding_count:
            raise _error("metrics publishable count exceeds matched Findings")

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Any) -> "ReviewMetricInputs":
        fields = tuple(cls.__dataclass_fields__)
        payload = _strict_object(value, fields, "Review metric inputs")
        return cls(**{name: payload[name] for name in fields})


def _evidence_diagnostic_from_dict(value: Any) -> EvidenceDiagnostic:
    payload = _strict_object(
        value,
        ("reason_code", "evidence_id", "finding_id", "ref_index"),
        "Evidence diagnostic",
    )
    return EvidenceDiagnostic(
        reason_code=_enum_value(EvidenceReasonCode, payload["reason_code"], "Evidence diagnostic.reason_code"),
        evidence_id=_optional_id(payload["evidence_id"], "Evidence diagnostic.evidence_id"),
        finding_id=_optional_id(payload["finding_id"], "Evidence diagnostic.finding_id"),
        ref_index=payload["ref_index"],
    )


def _evidence_item_from_dict(value: Any) -> EvidenceItemIntegrityResult:
    payload = _strict_object(
        value,
        ("policy_version", "evidence_id", "kind", "integrity", "diagnostics"),
        "Evidence item integrity result",
    )
    diagnostics = _array(payload["diagnostics"], "Evidence item diagnostics", 64)
    from .models import EvidenceKind

    return EvidenceItemIntegrityResult(
        policy_version=payload["policy_version"],
        evidence_id=_id(payload["evidence_id"], "Evidence item.evidence_id"),
        kind=_enum_value(EvidenceKind, payload["kind"], "Evidence item.kind"),
        integrity=_enum_value(EvidenceIntegrity, payload["integrity"], "Evidence item.integrity"),
        diagnostics=tuple(_evidence_diagnostic_from_dict(item) for item in diagnostics),
    )


def _evidence_result_from_dict(value: Any) -> EvidenceIntegrityResult:
    payload = _strict_object(
        value,
        ("policy_version", "finding_id", "integrity", "referenced_evidence_ids", "item_results", "diagnostics"),
        "Evidence integrity result",
    )
    refs = _array(payload["referenced_evidence_ids"], "Evidence integrity refs", 256)
    items = _array(payload["item_results"], "Evidence integrity items", 256)
    diagnostics = _array(payload["diagnostics"], "Evidence integrity diagnostics", 16_897)
    return EvidenceIntegrityResult(
        policy_version=payload["policy_version"],
        finding_id=_id(payload["finding_id"], "Evidence result.finding_id"),
        integrity=_enum_value(EvidenceIntegrity, payload["integrity"], "Evidence result.integrity"),
        referenced_evidence_ids=tuple(_id(item, "Evidence result.ref") for item in refs),
        item_results=tuple(_evidence_item_from_dict(item) for item in items),
        diagnostics=tuple(_evidence_diagnostic_from_dict(item) for item in diagnostics),
    )


@dataclass(frozen=True, init=False)
class ReviewEvaluationResult:
    """The canonical, inspectable output for one immutable Review evaluation.

    A trusted result is sealed: callers cannot manufacture one by invoking the
    dataclass constructor or ``dataclasses.replace``.  The only public trusted
    paths are ``ReviewEvaluator.evaluate`` and source-bound ``from_dict`` /
    ``from_json`` hydration, which deterministically replays the real
    Submission, truth, evaluator context, and Judge execution results.
    """

    schema_version: str
    evaluator_revision: str
    evaluator_execution_digest: str
    submission_digest: str
    submission_review_digest: str
    submission_evidence_digest: str
    eval_input_digest: str
    review_truth_digest: str
    deterministic_context_digest: str
    review_policy_version: str
    assignment_policy_version: str
    location_policy_version: str
    evidence_integrity_policy_version: str
    truth_completeness: TruthCompleteness
    novel_finding_policy: NovelFindingPolicy
    status: ReviewEvaluationStatus
    phase: ReviewEvaluationPhase
    generated_findings: Tuple[SubmissionFinding, ...]
    expected_truth_findings: Tuple[ExpectedFinding, ...]
    known_invalid_truth_findings: Tuple[KnownInvalidFinding, ...]
    location_candidates: Tuple[LocationAuditRecord, ...]
    known_invalid_candidates: Tuple[ReviewCandidateRecord, ...]
    expected_candidates: Tuple[ReviewCandidateRecord, ...]
    assignments: Tuple[ReviewAssignmentRecord, ...]
    finding_outcomes: Tuple[FindingOutcome, ...]
    unmatched_expected_truth_ids: Tuple[str, ...]
    judge_requests: Tuple[ReviewJudgeRequestRecord, ...]
    judge_decisions: Tuple[ReviewJudgeDecisionReceipt, ...]
    judge_failures: Tuple[ReviewJudgeFailureReceipt, ...]
    judge_ungraded: Tuple[ReviewJudgeUngradedReceipt, ...]
    evidence_integrity_results: Tuple[EvidenceIntegrityResult, ...]
    coverage: ReviewCoverage
    metrics: ReviewMetricInputs
    reason_codes: Tuple[ReviewReasonCode, ...]
    limit_failure: Optional[ReviewLimitFailure] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError(
            "ReviewEvaluationResult must be created by ReviewEvaluator.evaluate "
            "or source-bound hydration"
        )

    def __post_init__(self) -> None:
        if self.schema_version != REVIEW_EVALUATION_SCHEMA_VERSION:
            raise _error("unsupported Review evaluation schema version")
        _id(self.evaluator_revision, "Review evaluation.evaluator_revision")
        for name in (
            "evaluator_execution_digest",
            "submission_digest",
            "submission_review_digest",
            "submission_evidence_digest",
            "eval_input_digest",
            "review_truth_digest",
            "deterministic_context_digest",
        ):
            _digest(getattr(self, name), f"Review evaluation.{name}")
        if self.review_policy_version != REVIEW_MATCH_POLICY_VERSION:
            raise _error("unsupported Review match policy version")
        if self.assignment_policy_version != ASSIGNMENT_POLICY_VERSION:
            raise _error("unsupported assignment policy version")
        _id(self.location_policy_version, "Review evaluation.location_policy_version")
        if self.evidence_integrity_policy_version != EVIDENCE_INTEGRITY_POLICY_VERSION:
            raise _error("unsupported Evidence integrity policy version")
        if type(self.truth_completeness) is not TruthCompleteness:
            raise _error("Review evaluation.truth_completeness has an invalid type")
        if type(self.novel_finding_policy) is not NovelFindingPolicy:
            raise _error("Review evaluation.novel_finding_policy has an invalid type")
        if self.novel_finding_policy is NovelFindingPolicy.FORBID and self.truth_completeness is not TruthCompleteness.CLOSED_WORLD:
            raise _error("forbid is only valid for closed-world truth")
        if type(self.status) is not ReviewEvaluationStatus or type(self.phase) is not ReviewEvaluationPhase:
            raise _error("Review evaluation status/phase has an invalid type")
        if self.status is ReviewEvaluationStatus.GRADED and self.phase is not ReviewEvaluationPhase.COMPLETE:
            raise _error("graded Review evaluation must be complete")
        self._validate_sequences()
        self._validate_cross_references()
        if self.limit_failure is not None:
            if type(self.limit_failure) is not ReviewLimitFailure or self.status is not ReviewEvaluationStatus.UNGRADED:
                raise _error("limit failure requires an ungraded Review evaluation")
        if len(canonical_json_bytes(self.to_dict())) > MAX_REVIEW_EVALUATION_BYTES:
            raise _error("Review evaluation exceeds its canonical byte budget")

    def _validate_sequences(self) -> None:
        sequence_specs = (
            ("generated_findings", SubmissionFinding, MAX_REVIEW_FINDINGS, lambda item: item.finding_id),
            ("expected_truth_findings", ExpectedFinding, MAX_REVIEW_TRUTH_FINDINGS, lambda item: item.truth_id),
            ("known_invalid_truth_findings", KnownInvalidFinding, MAX_REVIEW_TRUTH_FINDINGS, lambda item: item.truth_id),
            ("location_candidates", LocationAuditRecord, MAX_REVIEW_LOCATION_AUDITS, lambda item: (item.finding_id, item.truth_kind.value, item.truth_id, item.truth_location_index)),
            ("known_invalid_candidates", ReviewCandidateRecord, MAX_REVIEW_CANDIDATES, lambda item: (item.finding_id, item.truth_id)),
            ("expected_candidates", ReviewCandidateRecord, MAX_REVIEW_CANDIDATES, lambda item: (item.finding_id, item.truth_id)),
            ("assignments", ReviewAssignmentRecord, MAX_REVIEW_ASSIGNMENTS, lambda item: (item.finding_id, item.truth_id)),
            ("finding_outcomes", FindingOutcome, MAX_REVIEW_FINDINGS, lambda item: item.finding_id),
            ("judge_requests", ReviewJudgeRequestRecord, MAX_REVIEW_JUDGE_REQUESTS, lambda item: item.request_id),
            ("judge_decisions", ReviewJudgeDecisionReceipt, MAX_REVIEW_JUDGE_REQUESTS, lambda item: item.request_id),
            ("judge_failures", ReviewJudgeFailureReceipt, MAX_REVIEW_JUDGE_REQUESTS, lambda item: item.request_id),
            ("judge_ungraded", ReviewJudgeUngradedReceipt, MAX_REVIEW_JUDGE_REQUESTS, lambda item: item.request_id),
            ("evidence_integrity_results", EvidenceIntegrityResult, MAX_REVIEW_EVIDENCE_RESULTS, lambda item: item.finding_id),
        )
        for name, expected_type, maximum, key in sequence_specs:
            values = tuple(getattr(self, name))
            if len(values) > maximum or any(type(item) is not expected_type for item in values):
                raise _error(f"Review evaluation.{name} violates its item/type limit")
            _canonical_tuple(values, key, f"Review evaluation.{name}")
            object.__setattr__(self, name, values)
        unmatched = tuple(_id(item, "unmatched expected truth ID") for item in self.unmatched_expected_truth_ids)
        if unmatched != tuple(sorted(set(unmatched))):
            raise _error("unmatched_expected_truth_ids must be unique and sorted")
        object.__setattr__(self, "unmatched_expected_truth_ids", unmatched)
        object.__setattr__(self, "reason_codes", _canonical_reason_codes(self.reason_codes, "Review evaluation.reason_codes"))
        if type(self.coverage) is not ReviewCoverage or type(self.metrics) is not ReviewMetricInputs:
            raise _error("Review evaluation coverage/metrics has an invalid type")
        _array_bytes(self.location_candidates, "location candidates", MAX_REVIEW_RECORD_BYTES)
        _array_bytes(self.known_invalid_candidates, "known-invalid candidates", MAX_REVIEW_RECORD_BYTES)
        _array_bytes(self.expected_candidates, "expected candidates", MAX_REVIEW_RECORD_BYTES)
        _array_bytes(self.judge_requests, "Judge requests", MAX_REVIEW_RECORD_BYTES)
        _array_bytes(self.judge_decisions, "Judge decisions", MAX_REVIEW_JUDGE_DECISION_BYTES)
        _array_bytes(self.judge_failures, "Judge failures", MAX_REVIEW_JUDGE_RECEIPT_BYTES)
        _array_bytes(self.judge_ungraded, "Judge ungraded receipts", MAX_REVIEW_JUDGE_RECEIPT_BYTES)
        reason_refs = [
            ref
            for receipt in self.judge_decisions
            for ref in receipt.decision.reason_refs
        ]
        if len(canonical_json_bytes(reason_refs)) > MAX_REVIEW_REASON_REF_BYTES:
            raise _error("Judge decision reason refs exceed their byte budget")

    def _validate_cross_references(self) -> None:
        generated = {item.finding_id: item for item in self.generated_findings}
        expected = {item.truth_id: item for item in self.expected_truth_findings}
        invalid = {item.truth_id: item for item in self.known_invalid_truth_findings}
        if len(generated) != len(self.generated_findings):
            raise _error("Review evaluation contains duplicate Finding IDs")
        if len(expected) != len(self.expected_truth_findings) or len(invalid) != len(self.known_invalid_truth_findings):
            raise _error("Review evaluation contains duplicate truth IDs")
        if set(expected).intersection(invalid):
            raise _error("Review evaluation truth IDs overlap")
        request_map = {item.request_id: item for item in self.judge_requests}
        if len(request_map) != len(self.judge_requests):
            raise _error("Review evaluation contains duplicate Judge request IDs")
        locations_by_pair: Dict[
            Tuple[str, ReviewTruthKind, str], list[LocationAuditRecord]
        ] = {}
        for item in self.location_candidates:
            truth_map = expected if item.truth_kind is ReviewTruthKind.EXPECTED else invalid
            if item.finding_id not in generated or item.truth_id not in truth_map:
                raise _error("location candidate references an unknown Finding/truth")
            truth = truth_map[item.truth_id]
            if (
                item.truth_kind is ReviewTruthKind.EXPECTED
                and not truth.metric_authority.location_scorable
            ):
                raise _error(
                    "location candidate targets a location-unscorable expected truth"
                )
            if (
                item.truth_location_index >= len(truth.locations)
                or item.truth_location
                != truth.locations[item.truth_location_index]
            ):
                raise _error("location candidate differs from its truth target")
            locations_by_pair.setdefault(
                (item.finding_id, item.truth_kind, item.truth_id), []
            ).append(item)
        candidate_request_ids: set[str] = set()
        for items, truth_kind, truth_map in (
            (self.known_invalid_candidates, ReviewTruthKind.KNOWN_INVALID, invalid),
            (self.expected_candidates, ReviewTruthKind.EXPECTED, expected),
        ):
            pairs = set()
            for item in items:
                pair = (item.finding_id, item.truth_id)
                if pair in pairs:
                    raise _error("Review evaluation contains a duplicate candidate pair")
                pairs.add(pair)
                if item.truth_kind is not truth_kind or item.finding_id not in generated or item.truth_id not in truth_map:
                    raise _error("candidate references an unknown or wrong-kind Finding/truth")
                if truth_kind is ReviewTruthKind.EXPECTED and item.relation is not None:
                    severity_scorable = truth_map[
                        item.truth_id
                    ].metric_authority.severity_scorable
                    if severity_scorable != (item.severity_assessment is not None):
                        raise _error(
                            "expected candidate severity assessment differs from MetricAuthority"
                        )
                locations = locations_by_pair.get(
                    (item.finding_id, item.truth_kind, item.truth_id), []
                )
                best_location_score = max(
                    (location.match.score for location in locations),
                    default=None,
                )
                best_location_matched = (
                    None
                    if best_location_score is None
                    else any(
                        location.match.matched
                        and location.match.score == best_location_score
                        for location in locations
                    )
                )
                if (
                    item.location_match_count
                    != sum(location.match.matched for location in locations)
                    or item.best_location_score != best_location_score
                    or item.best_location_matched is not best_location_matched
                ):
                    raise _error("candidate location projection is inconsistent")
                if item.match_kind is FindingMatchKind.SEMANTIC:
                    request = request_map.get(item.request_id or "")
                    expected_phase = (
                        ReviewEvaluationPhase.KNOWN_INVALID
                        if truth_kind is ReviewTruthKind.KNOWN_INVALID
                        else ReviewEvaluationPhase.EXPECTED_ASSIGNMENT
                    )
                    if (
                        request is None
                        or request.phase is not expected_phase
                        or request.finding_id != item.finding_id
                        or request.truth_kind is not truth_kind
                        or request.truth_id != item.truth_id
                        or request.blind_request_id != item.blind_request_id
                    ):
                        raise _error("semantic candidate Judge request binding is invalid")
                    if request.request_id in candidate_request_ids:
                        raise _error("one Judge request resolves multiple candidate pairs")
                    candidate_request_ids.add(request.request_id)
        equivalence_request_ids = {
            item.request_id
            for item in self.judge_requests
            if item.phase
            in {
                ReviewEvaluationPhase.KNOWN_INVALID,
                ReviewEvaluationPhase.EXPECTED_ASSIGNMENT,
            }
        }
        if candidate_request_ids != equivalence_request_ids:
            raise _error("equivalence Judge requests and semantic candidates differ")
        for request in self.judge_requests:
            if request.finding_id not in generated:
                raise _error("Judge request references an unknown Finding")
            if request.phase in {
                ReviewEvaluationPhase.KNOWN_INVALID,
                ReviewEvaluationPhase.EXPECTED_ASSIGNMENT,
            }:
                expected_kind = (
                    ReviewTruthKind.KNOWN_INVALID
                    if request.phase is ReviewEvaluationPhase.KNOWN_INVALID
                    else ReviewTruthKind.EXPECTED
                )
                truth_map = invalid if expected_kind is ReviewTruthKind.KNOWN_INVALID else expected
                if (
                    request.truth_kind is not expected_kind
                    or request.truth_id not in truth_map
                ):
                    raise _error("equivalence Judge request truth binding is invalid")
            elif request.phase is ReviewEvaluationPhase.NOVEL_FACTUALITY:
                if request.truth_kind is not None or request.truth_id is not None:
                    raise _error("novel factuality request cannot bind a truth Finding")
            elif request.phase is ReviewEvaluationPhase.EVIDENCE_SUPPORT:
                if request.truth_kind is not None or (
                    request.truth_id is not None and request.truth_id not in expected
                ):
                    raise _error("Evidence support request truth binding is invalid")
        if self.limit_failure is None:
            expected_location_keys = {
                (finding_id, ReviewTruthKind.EXPECTED, truth_id, index)
                for finding_id in generated
                for truth_id, truth in expected.items()
                if truth.metric_authority.location_scorable
                for index, _location in enumerate(truth.locations)
            } | {
                (finding_id, ReviewTruthKind.KNOWN_INVALID, truth_id, index)
                for finding_id in generated
                for truth_id, truth in invalid.items()
                for index, _location in enumerate(truth.locations)
            }
            actual_location_keys = {
                (
                    item.finding_id,
                    item.truth_kind,
                    item.truth_id,
                    item.truth_location_index,
                )
                for item in self.location_candidates
            }
            if actual_location_keys != expected_location_keys:
                raise _error("location candidates do not cover the canonical graph")
            expected_known_pairs = {
                (finding_id, truth_id)
                for finding_id in generated
                for truth_id in invalid
            }
            actual_known_pairs = {
                (item.finding_id, item.truth_id)
                for item in self.known_invalid_candidates
            }
            if actual_known_pairs != expected_known_pairs:
                raise _error("known-invalid candidates do not cover the canonical graph")
        assignment_pairs = {(item.finding_id, item.truth_id) for item in self.assignments}
        if len(assignment_pairs) != len(self.assignments):
            raise _error("Review evaluation contains duplicate assignments")
        if len({item.finding_id for item in self.assignments}) != len(self.assignments) or len({item.truth_id for item in self.assignments}) != len(self.assignments):
            raise _error("Review assignments are not one-to-one")
        selected_candidates = {
            (item.finding_id, item.truth_id): item
            for item in self.expected_candidates
            if item.selected
        }
        if not assignment_pairs.issubset(selected_candidates):
            raise _error("Review assignment lacks a selected expected candidate")
        if set(selected_candidates) != assignment_pairs:
            raise _error("selected expected candidates and assignments differ")
        for assignment in self.assignments:
            candidate = selected_candidates[(assignment.finding_id, assignment.truth_id)]
            if (
                candidate.edge_weight != assignment.weight
                or candidate.match_kind is not assignment.match_kind
                or candidate.request_id != assignment.request_id
                or candidate.relation is not FindingMatchRelation.EQUIVALENT
                or candidate.resolution is not FindingResolution.RESOLVED
            ):
                raise _error("Review assignment differs from its selected candidate")
        expected_unmatched = set(expected) - {
            item.truth_id for item in self.assignments
        }
        if set(self.unmatched_expected_truth_ids) != expected_unmatched:
            raise _error("unmatched expected truth IDs differ from Assignment")

        outcomes = {item.finding_id: item for item in self.finding_outcomes}
        if set(outcomes) != set(generated):
            raise _error("Finding outcomes must cover generated Findings exactly")
        matched_outcome_pairs = {
            (item.finding_id, item.matched_expected_truth_id)
            for item in outcomes.values()
            if item.matched_expected_truth_id is not None
        }
        if matched_outcome_pairs != assignment_pairs:
            raise _error("matched Finding outcomes and Assignments differ")
        selected_known_candidates = {
            (item.finding_id, item.truth_id): item
            for item in self.known_invalid_candidates
            if item.selected
        }
        if len({item.finding_id for item in selected_known_candidates.values()}) != len(
            selected_known_candidates
        ):
            raise _error("a Finding selected more than one known-invalid candidate")
        known_outcome_pairs = {
            (item.finding_id, item.matched_known_invalid_truth_id)
            for item in outcomes.values()
            if item.matched_known_invalid_truth_id is not None
        }
        if set(selected_known_candidates) != known_outcome_pairs:
            raise _error("known-invalid outcomes and selected candidates differ")
        expected_candidate_map = {
            (item.finding_id, item.truth_id): item
            for item in self.expected_candidates
        }
        assignment_by_truth = {
            item.truth_id: item.finding_id for item in self.assignments
        }
        for outcome in outcomes.values():
            if outcome.disposition is not FindingDisposition.DUPLICATE:
                continue
            truth_id = outcome.duplicate_truth_id or ""
            candidate = expected_candidate_map.get((outcome.finding_id, truth_id))
            if (
                assignment_by_truth.get(truth_id)
                != outcome.duplicate_of_finding_id
                or candidate is None
                or candidate.selected
                or candidate.relation is not FindingMatchRelation.EQUIVALENT
                or candidate.resolution is not FindingResolution.RESOLVED
                or candidate.edge_weight is None
            ):
                raise _error("duplicate Finding outcome lacks its assigned equivalent edge")
        evidence = {item.finding_id: item for item in self.evidence_integrity_results}
        if set(evidence) != set(generated):
            raise _error("Evidence integrity results must cover Findings exactly")
        if self.limit_failure is None and self.phase is not ReviewEvaluationPhase.KNOWN_INVALID:
            eligible_finding_ids = {
                finding_id
                for finding_id, outcome in outcomes.items()
                if outcome.disposition is not FindingDisposition.KNOWN_INVALID
            }
            expected_expected_pairs = {
                (finding_id, truth_id)
                for finding_id in eligible_finding_ids
                for truth_id in expected
            }
            actual_expected_pairs = {
                (item.finding_id, item.truth_id)
                for item in self.expected_candidates
            }
            if actual_expected_pairs != expected_expected_pairs:
                raise _error("expected candidates do not cover the canonical graph")
        for outcome in outcomes.values():
            if (
                outcome.matched_expected_truth_id is not None
                and outcome.matched_expected_truth_id not in expected
            ):
                raise _error("Finding outcome references an unknown expected truth")
            if (
                outcome.matched_known_invalid_truth_id is not None
                and outcome.matched_known_invalid_truth_id not in invalid
            ):
                raise _error("Finding outcome references an unknown known-invalid truth")
            if (
                outcome.duplicate_truth_id is not None
                and outcome.duplicate_truth_id not in expected
            ):
                raise _error("Finding outcome references an unknown duplicate truth")
            if (
                outcome.duplicate_of_finding_id is not None
                and (
                    outcome.duplicate_of_finding_id not in generated
                    or outcome.duplicate_of_finding_id == outcome.finding_id
                )
            ):
                raise _error("Finding outcome references an invalid duplicate Finding")
            if outcome.matched_expected_truth_id is not None:
                severity_scorable = expected[
                    outcome.matched_expected_truth_id
                ].metric_authority.severity_scorable
                if severity_scorable != (outcome.severity_assessment is not None):
                    raise _error(
                        "matched outcome severity assessment differs from MetricAuthority"
                    )
        for outcome in outcomes.values():
            if outcome.evidence_integrity is not evidence[outcome.finding_id].integrity:
                raise _error("Finding outcome Evidence integrity disagrees with audit")

        for outcome in outcomes.values():
            if outcome.novel_request_id is not None:
                request = request_map.get(outcome.novel_request_id)
                if (
                    request is None
                    or request.phase is not ReviewEvaluationPhase.NOVEL_FACTUALITY
                    or request.finding_id != outcome.finding_id
                ):
                    raise _error("Finding outcome novel request binding is invalid")
            if outcome.evidence_support_request_id is not None:
                request = request_map.get(outcome.evidence_support_request_id)
                if (
                    request is None
                    or request.phase is not ReviewEvaluationPhase.EVIDENCE_SUPPORT
                    or request.finding_id != outcome.finding_id
                ):
                    raise _error("Finding outcome Evidence support request binding is invalid")
        novel_request_ids = {
            item.request_id
            for item in self.judge_requests
            if item.phase is ReviewEvaluationPhase.NOVEL_FACTUALITY
        }
        outcome_novel_request_ids = {
            item.novel_request_id
            for item in outcomes.values()
            if item.novel_request_id is not None
        }
        if novel_request_ids != outcome_novel_request_ids:
            raise _error("novel factuality requests and Finding outcomes differ")
        support_request_ids = {
            item.request_id
            for item in self.judge_requests
            if item.phase is ReviewEvaluationPhase.EVIDENCE_SUPPORT
        }
        outcome_support_request_ids = {
            item.evidence_support_request_id
            for item in outcomes.values()
            if item.evidence_support_request_id is not None
        }
        if support_request_ids != outcome_support_request_ids:
            raise _error("Evidence support requests and Finding outcomes differ")
        resolution_ids: list[str] = []
        result_digests: list[str] = []
        for receipts in (self.judge_decisions, self.judge_failures, self.judge_ungraded):
            for receipt in receipts:
                request = request_map.get(receipt.request_id)
                if request is None:
                    raise _error("Judge receipt references an unknown request")
                if (
                    receipt.task is not request.task
                    or receipt.request_digest != request.request_digest
                    or receipt.blind_request_id != request.blind_request_id
                    or receipt.evaluator_execution_digest != self.evaluator_execution_digest
                ):
                    raise _error("Judge receipt binding differs from its request/evaluator")
                resolution_ids.append(receipt.request_id)
                result_digests.append(receipt.judge_result_digest)
        if len(resolution_ids) != len(set(resolution_ids)):
            raise _error("a Judge request has more than one receipt")
        if len(result_digests) != len(set(result_digests)):
            raise _error("a Judge result digest resolves more than one request")
        resolved_ids = set(resolution_ids)
        pending_ids = set(request_map) - resolved_ids
        pending_count = len(pending_ids)
        if pending_ids and (
            self.phase is ReviewEvaluationPhase.COMPLETE
            or any(request_map[item].phase is not self.phase for item in pending_ids)
        ):
            raise _error("pending Judge requests do not belong to the active phase")
        semantic_unknown = sum(
            1
            for receipt in self.judge_decisions
            if (
                getattr(receipt.decision, "relation", None)
                is FindingMatchRelation.UNKNOWN
                or getattr(receipt.decision, "factuality", None)
                is NovelFactuality.UNKNOWN
                or getattr(receipt.decision, "support", None)
                is EvidenceSupport.UNKNOWN
            )
        )
        canonical_status = (
            ReviewEvaluationStatus.PENDING_JUDGE
            if pending_count
            else (
                ReviewEvaluationStatus.GRADED
                if self.phase is ReviewEvaluationPhase.COMPLETE
                and not self.judge_failures
                and not self.judge_ungraded
                and not semantic_unknown
                and not any(
                    item.disposition is FindingDisposition.NOVEL_DISALLOWED
                    for item in outcomes.values()
                )
                else ReviewEvaluationStatus.UNGRADED
            )
        )
        if self.status is not canonical_status:
            raise _error("Review evaluation status is not canonical")
        expected_coverage = ReviewCoverage(
            judge_request_count=len(request_map),
            judge_graded_count=len(self.judge_decisions),
            judge_failed_count=len(self.judge_failures),
            judge_ungraded_count=len(self.judge_ungraded),
            judge_pending_count=pending_count,
            semantic_unknown_count=semantic_unknown,
            finding_count=len(generated),
            finding_resolved_count=sum(
                item.issue_resolution is FindingResolution.RESOLVED
                for item in outcomes.values()
            ),
            evidence_result_count=len(evidence),
        )
        if self.coverage != expected_coverage:
            raise _error("Review coverage is not the canonical projection")

        matched_truth_ids = {
            item.matched_expected_truth_id
            for item in outcomes.values()
            if item.matched_expected_truth_id is not None
        }
        integrity_counts = {
            integrity: sum(
                item.integrity is integrity for item in evidence.values()
            )
            for integrity in EvidenceIntegrity
        }
        support_counts = {
            support: sum(
                item.evidence_support is support for item in outcomes.values()
            )
            for support in EvidenceSupport
        }
        expected_metrics = ReviewMetricInputs(
            scorable=self.status is ReviewEvaluationStatus.GRADED,
            generated_finding_count=len(generated),
            expected_truth_count=len(expected),
            required_expected_truth_count=sum(
                item.required for item in expected.values()
            ),
            matched_finding_count=sum(
                item.disposition is FindingDisposition.MATCHED
                for item in outcomes.values()
            ),
            matched_expected_truth_count=len(matched_truth_ids),
            matched_required_truth_count=sum(
                expected[item].required for item in matched_truth_ids
            ),
            duplicate_finding_count=sum(
                item.disposition is FindingDisposition.DUPLICATE
                for item in outcomes.values()
            ),
            known_invalid_finding_count=sum(
                item.disposition is FindingDisposition.KNOWN_INVALID
                for item in outcomes.values()
            ),
            plausible_novel_count=sum(
                item.disposition is FindingDisposition.NOVEL_ALLOWED
                and item.issue_judgement.value == "plausible"
                for item in outcomes.values()
            ),
            fabricated_finding_count=sum(
                item.issue_judgement.value == "fabricated"
                for item in outcomes.values()
            ),
            unknown_finding_count=sum(
                item.issue_judgement.value == "unknown"
                for item in outcomes.values()
            ),
            unmatched_expected_truth_count=len(self.unmatched_expected_truth_ids),
            unmatched_required_truth_count=sum(
                expected[item].required for item in self.unmatched_expected_truth_ids
            ),
            evidence_valid_count=integrity_counts[EvidenceIntegrity.VALID],
            evidence_invalid_count=integrity_counts[EvidenceIntegrity.INVALID],
            evidence_missing_count=integrity_counts[EvidenceIntegrity.MISSING],
            evidence_supported_count=support_counts[EvidenceSupport.SUPPORTED],
            evidence_weak_count=support_counts[EvidenceSupport.WEAK],
            evidence_unsupported_count=support_counts[EvidenceSupport.UNSUPPORTED],
            evidence_support_unknown_count=support_counts[EvidenceSupport.UNKNOWN],
            strict_publishable_count=sum(
                item.strict_publishable for item in outcomes.values()
            ),
        )
        if self.metrics != expected_metrics:
            raise _error("Review metric inputs are not the canonical projection")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluator_revision": self.evaluator_revision,
            "evaluator_execution_digest": self.evaluator_execution_digest,
            "submission_digest": self.submission_digest,
            "submission_review_digest": self.submission_review_digest,
            "submission_evidence_digest": self.submission_evidence_digest,
            "eval_input_digest": self.eval_input_digest,
            "review_truth_digest": self.review_truth_digest,
            "deterministic_context_digest": self.deterministic_context_digest,
            "review_policy_version": self.review_policy_version,
            "assignment_policy_version": self.assignment_policy_version,
            "location_policy_version": self.location_policy_version,
            "evidence_integrity_policy_version": self.evidence_integrity_policy_version,
            "truth_completeness": self.truth_completeness.value,
            "novel_finding_policy": self.novel_finding_policy.value,
            "status": self.status.value,
            "phase": self.phase.value,
            "generated_findings": [item.to_dict() for item in self.generated_findings],
            "expected_truth_findings": [item.to_dict() for item in self.expected_truth_findings],
            "known_invalid_truth_findings": [item.to_dict() for item in self.known_invalid_truth_findings],
            "location_candidates": [item.to_dict() for item in self.location_candidates],
            "known_invalid_candidates": [item.to_dict() for item in self.known_invalid_candidates],
            "expected_candidates": [item.to_dict() for item in self.expected_candidates],
            "assignments": [item.to_dict() for item in self.assignments],
            "finding_outcomes": [item.to_dict() for item in self.finding_outcomes],
            "unmatched_expected_truth_ids": list(self.unmatched_expected_truth_ids),
            "judge_requests": [item.to_dict() for item in self.judge_requests],
            "judge_decisions": [item.to_dict() for item in self.judge_decisions],
            "judge_failures": [item.to_dict() for item in self.judge_failures],
            "judge_ungraded": [item.to_dict() for item in self.judge_ungraded],
            "evidence_integrity_results": [_evidence_result_to_dict(item) for item in self.evidence_integrity_results],
            "coverage": self.coverage.to_dict(),
            "metrics": self.metrics.to_dict(),
            "reason_codes": [item.value for item in self.reason_codes],
            "limit_failure": None if self.limit_failure is None else self.limit_failure.to_dict(),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def digest(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def _parse_unbound(cls, value: Any) -> Dict[str, Any]:
        fields = tuple(field.name for field in dataclass_fields(cls))
        payload = _strict_object(value, fields, "Review evaluation")
        for name, maximum in (
            ("generated_findings", MAX_REVIEW_FINDINGS),
            ("expected_truth_findings", MAX_REVIEW_TRUTH_FINDINGS),
            ("known_invalid_truth_findings", MAX_REVIEW_TRUTH_FINDINGS),
            ("location_candidates", MAX_REVIEW_LOCATION_AUDITS),
            ("known_invalid_candidates", MAX_REVIEW_CANDIDATES),
            ("expected_candidates", MAX_REVIEW_CANDIDATES),
            ("assignments", MAX_REVIEW_ASSIGNMENTS),
            ("finding_outcomes", MAX_REVIEW_FINDINGS),
            ("unmatched_expected_truth_ids", MAX_REVIEW_TRUTH_FINDINGS),
            ("judge_requests", MAX_REVIEW_JUDGE_REQUESTS),
            ("judge_decisions", MAX_REVIEW_JUDGE_REQUESTS),
            ("judge_failures", MAX_REVIEW_JUDGE_REQUESTS),
            ("judge_ungraded", MAX_REVIEW_JUDGE_REQUESTS),
            ("evidence_integrity_results", MAX_REVIEW_EVIDENCE_RESULTS),
            ("reason_codes", 128),
        ):
            _array(payload[name], name, maximum)
        try:
            _json_tree(payload, "Review evaluation")
            if len(canonical_json_bytes(payload)) > MAX_REVIEW_EVALUATION_BYTES:
                raise _error("Review evaluation exceeds its canonical byte budget")
        except (SchemaError, ValueError) as exc:
            raise _error(str(exc)) from exc
        return payload

    @classmethod
    def from_dict(
        cls,
        value: Any,
        *,
        submission: EvalSubmission,
        review_truth: ReviewTruth,
        evaluator: "ReviewEvaluator",
        judge_results: Sequence[JudgeExecutionResult],
    ) -> "ReviewEvaluationResult":
        """Hydrate only after a complete source-bound deterministic replay."""

        if type(evaluator) is not ReviewEvaluator:
            raise _error("Review hydration requires the real ReviewEvaluator")
        parsed = cls._parse_unbound(value)
        replayed = evaluator.evaluate(
            submission,
            review_truth,
            judge_results=judge_results,
        )
        if canonical_json_bytes(parsed) != canonical_json_bytes(replayed.to_dict()):
            raise _error("persisted Review evaluation differs from deterministic replay")
        return replayed

    @classmethod
    def from_json(
        cls,
        data: Any,
        *,
        submission: EvalSubmission,
        review_truth: ReviewTruth,
        evaluator: "ReviewEvaluator",
        judge_results: Sequence[JudgeExecutionResult],
    ) -> "ReviewEvaluationResult":
        try:
            payload = _strict_json_loads(
                data,
                MAX_REVIEW_EVALUATION_BYTES,
                "Review evaluation JSON",
            )
        except (SchemaError, ValueError) as exc:
            raise _error(str(exc)) from exc
        return cls.from_dict(
            payload,
            submission=submission,
            review_truth=review_truth,
            evaluator=evaluator,
            judge_results=judge_results,
        )

    serialize = to_dict
    hydrate = from_dict


class _LimitExceeded(Exception):
    def __init__(
        self,
        scope: ReviewLimitScope,
        observed: int,
        limit: int,
        reason_code: ReviewReasonCode,
        phase: ReviewEvaluationPhase,
    ) -> None:
        super().__init__(scope.value)
        self.failure = ReviewLimitFailure(scope, observed, limit, reason_code)
        self.phase = phase


@dataclass(frozen=True)
class _EvaluationBase:
    submission: EvalSubmission
    review: SubmissionReview
    truth: ReviewTruth
    findings: Tuple[SubmissionFinding, ...]
    evidence: Tuple[SubmissionEvidence, ...]
    evidence_results: Tuple[EvidenceIntegrityResult, ...]
    evidence_by_finding: Mapping[str, EvidenceIntegrityResult]
    evidence_by_id: Mapping[str, SubmissionEvidence]
    submission_digest: str
    submission_review_digest: str
    submission_evidence_digest: str
    review_truth_digest: str
    deterministic_context_digest: str


@dataclass(frozen=True)
class _PairState:
    finding: SubmissionFinding
    truth_kind: ReviewTruthKind
    truth: Union[ExpectedFinding, KnownInvalidFinding]
    match_kind: FindingMatchKind
    request: Optional[ReviewJudgeRequestRecord]
    relation: Optional[FindingMatchRelation]
    score_ppm: Optional[int]
    edge_weight: Optional[int]
    resolution: FindingResolution
    severity_assessment: Optional[SeverityAssessment]
    actionability: Optional[ActionabilityAssessment]
    reason_codes: Tuple[ReviewReasonCode, ...]

    def candidate(
        self,
        *,
        selected: bool,
        location_records: Sequence[LocationAuditRecord],
    ) -> ReviewCandidateRecord:
        best = max((item.match.score for item in location_records), default=None)
        return ReviewCandidateRecord(
            finding_id=self.finding.finding_id,
            truth_kind=self.truth_kind,
            truth_id=self.truth.truth_id,
            match_kind=self.match_kind,
            request_id=None if self.request is None else self.request.request_id,
            blind_request_id=(
                None if self.request is None else self.request.blind_request_id
            ),
            relation=self.relation,
            score_ppm=self.score_ppm,
            edge_weight=self.edge_weight,
            selected=selected,
            resolution=self.resolution,
            severity_assessment=self.severity_assessment,
            actionability=self.actionability,
            location_match_count=sum(
                1 for item in location_records if item.match.matched
            ),
            best_location_score=best,
            best_location_matched=(
                None
                if best is None
                else any(
                    item.match.matched and item.match.score == best
                    for item in location_records
                )
            ),
            reason_codes=self.reason_codes,
        )


@dataclass
class _OutcomeDraft:
    finding_id: str
    issue_resolution: FindingResolution
    issue_judgement: Any
    disposition: FindingDisposition
    matched_expected_truth_id: Optional[str] = None
    matched_known_invalid_truth_id: Optional[str] = None
    duplicate_truth_id: Optional[str] = None
    duplicate_of_finding_id: Optional[str] = None
    novel_request_id: Optional[str] = None
    severity_assessment: Optional[SeverityAssessment] = None
    actionability: Optional[ActionabilityAssessment] = None
    evidence_support_resolution: EvidenceSupportResolution = EvidenceSupportResolution.NOT_REQUESTED
    evidence_support: EvidenceSupport = EvidenceSupport.UNKNOWN
    evidence_support_request_id: Optional[str] = None
    reasons: Tuple[ReviewReasonCode, ...] = ()


class _JudgeResultRegistry:
    def __init__(
        self,
        values: Sequence[JudgeExecutionResult],
        evaluator_execution: EvaluatorExecutionConfig,
    ) -> None:
        if type(values) not in (list, tuple):
            raise _error("judge_results must be a bounded list or tuple")
        if len(values) > MAX_REVIEW_JUDGE_REQUESTS:
            raise _error("judge_results exceeds the Review request limit")
        execution_digest = evaluator_execution.digest()
        by_source: Dict[str, JudgeExecutionResult] = {}
        blind_ids: set[str] = set()
        result_digests: set[str] = set()
        response_bytes = 0
        response_tokens = 0
        for index, result in enumerate(values):
            if type(result) is not JudgeExecutionResult:
                raise _error(f"judge_results[{index}] is not JudgeExecutionResult")
            if result.request.task is JudgeTask.INTENT_EQUIVALENCE:
                raise _error("Review evaluator cannot consume an Intent Judge result")
            if result.evaluator_execution_digest != execution_digest:
                raise _error("Judge result belongs to another evaluator execution")
            source_id = result.request.source_request_id
            if source_id in by_source:
                raise _error("judge_results contains duplicate source request IDs")
            if result.request.request_id in blind_ids:
                raise _error("judge_results contains duplicate blind request IDs")
            digest = result.digest()
            if digest in result_digests:
                raise _error("one Judge result digest resolves multiple requests")
            by_source[source_id] = result
            blind_ids.add(result.request.request_id)
            result_digests.add(digest)
            for attempt in result.attempts:
                response_bytes += attempt.output_size_bytes
                response_tokens += max(0, math.ceil(attempt.output_size_bytes / 4))
        budgets = evaluator_execution.judge_budgets
        if response_bytes > budgets.max_total_judge_response_bytes:
            raise _error("Judge results exceed the aggregate response byte budget")
        if response_tokens > budgets.max_total_judge_response_tokens:
            raise _error("Judge results exceed the aggregate response token budget")
        self._by_source = by_source
        self._used: set[str] = set()
        self.decisions: list[ReviewJudgeDecisionReceipt] = []
        self.failures: list[ReviewJudgeFailureReceipt] = []
        self.ungraded: list[ReviewJudgeUngradedReceipt] = []

    def resolve(
        self, record: ReviewJudgeRequestRecord
    ) -> Optional[JudgeExecutionResult]:
        result = self._by_source.get(record.request_id)
        if result is None:
            return None
        if result.request != record.request:
            raise _error("Judge result request differs from canonical Review request")
        self._used.add(record.request_id)
        common = {
            "request_id": record.request_id,
            "task": record.task,
            "request_digest": record.request_digest,
            "evaluator_execution_digest": result.evaluator_execution_digest,
            "judge_result_digest": result.digest(),
            "blind_request_id": record.blind_request_id,
        }
        if result.status is JudgeRunStatus.GRADED:
            if result.decision is None:
                raise _error("graded Judge result lacks a decision")
            self.decisions.append(
                ReviewJudgeDecisionReceipt(decision=result.decision, **common)
            )
        elif result.status is JudgeRunStatus.JUDGE_FAILED:
            if result.failure is None:
                raise _error("failed Judge result lacks a failure")
            self.failures.append(
                ReviewJudgeFailureReceipt(failure=result.failure, **common)
            )
        elif result.status is JudgeRunStatus.UNGRADED:
            if result.ungraded_reason is None:
                raise _error("ungraded Judge result lacks a reason")
            self.ungraded.append(
                ReviewJudgeUngradedReceipt(
                    ungraded_reason=result.ungraded_reason,
                    **common,
                )
            )
        else:
            raise _error("Judge result has an unsupported terminal status")
        return result

    def ensure_no_extra_results(self) -> None:
        extra = set(self._by_source) - self._used
        if extra:
            raise _error("judge_results contains results for non-canonical/future requests")


@dataclass(frozen=True)
class ReviewEvaluator:
    """Reconcile one completed immutable Submission against Review truth.

    ``evaluate`` never calls a model.  It accepts zero or more typed
    ``JudgeExecutionResult`` values and returns the next canonical stage.  A
    composition root may execute the emitted blind requests and call it again;
    ``evaluate_with_judge`` provides that bounded loop as a convenience.
    """

    eval_input: EvalInput
    replay: PreparedRepositoryReplay | FrozenContextReplay
    trial_id: str
    target_materialization_id: str
    evaluator_execution: EvaluatorExecutionConfig
    rubrics: JudgeRubricCatalog = DEFAULT_JUDGE_RUBRICS
    location_matcher: Optional[LocationMatcher] = None
    evidence_checker: Optional[EvidenceIntegrityChecker] = None
    command_attestations: Tuple[CommandOutputAttestation, ...] = ()
    context_bundle: ReviewContextBundle = ReviewContextBundle()
    evaluator_revision: str = REVIEW_EVALUATOR_REVISION
    review_evaluator_context: ReviewEvaluatorContext = ReviewEvaluatorContext(
        truth_contexts=()
    )

    def __post_init__(self) -> None:
        if type(self.eval_input) is not EvalInput:
            raise _error("ReviewEvaluator requires canonical EvalInput")
        _id(self.trial_id, "ReviewEvaluator.trial_id")
        _id(
            self.target_materialization_id,
            "ReviewEvaluator.target_materialization_id",
        )
        if not isinstance(self.evaluator_execution, EvaluatorExecutionConfig):
            raise _error("ReviewEvaluator requires EvaluatorExecutionConfig")
        self.evaluator_execution.validate_runtime_policy_support()
        if type(self.review_evaluator_context) is not ReviewEvaluatorContext:
            raise _error(
                "review_evaluator_context must be the typed ReviewEvaluatorContext"
            )
        if type(self.rubrics) is not JudgeRubricCatalog:
            raise _error("ReviewEvaluator requires a JudgeRubricCatalog")
        attestations = tuple(self.command_attestations)
        if any(type(item) is not CommandOutputAttestation for item in attestations):
            raise _error("command_attestations contains an invalid item")
        object.__setattr__(self, "command_attestations", attestations)
        if type(self.context_bundle) is not ReviewContextBundle:
            raise _error("context_bundle must be the scoped ReviewContextBundle")
        _id(self.evaluator_revision, "ReviewEvaluator.evaluator_revision")

        matcher = self.location_matcher
        if self.eval_input.review_target.kind is ReviewTargetKind.REPOSITORY:
            if type(self.replay) is not PreparedRepositoryReplay:
                raise _error(
                    "Repository ReviewEvaluator requires PreparedRepositoryReplay"
                )
            repository = repository_from_eval_input(self.eval_input)
            if (
                self.replay.repository_descriptor_digest != repository.digest()
                or self.replay.base_revision != repository.base_revision
                or self.replay.head_revision != repository.head_revision
            ):
                raise _error("ReviewEvaluator replay is not bound to EvalInput")
            expected_catalog = SidePathCatalog.from_replay(self.replay)
            if matcher is None:
                matcher = LocationMatcher(expected_catalog)
                object.__setattr__(self, "location_matcher", matcher)
            elif type(matcher) is not LocationMatcher:
                raise _error("location_matcher must be LocationMatcher")
            elif matcher.side_paths.digest() != expected_catalog.digest():
                raise _error("location_matcher is not bound to the supplied replay")
        else:
            if type(self.replay) is not FrozenContextReplay:
                raise _error(
                    "Frozen ReviewEvaluator requires FrozenContextReplay"
                )
            if matcher is not None:
                raise _error(
                    "Frozen ReviewEvaluator cannot use a Repository location matcher"
                )

        checker = self.evidence_checker
        if checker is None:
            checker = EvidenceIntegrityChecker(
                self.eval_input,
                self.replay,
                self.trial_id,
                self.target_materialization_id,
                attestations,
            )
            object.__setattr__(self, "evidence_checker", checker)
        elif type(checker) is not EvidenceIntegrityChecker:
            raise _error("evidence_checker must be EvidenceIntegrityChecker")
        elif (
            checker.eval_input != self.eval_input
            or checker.replay is not self.replay
            or checker.trial_id != self.trial_id
            or checker.target_materialization_id
            != self.target_materialization_id
            or checker.command_attestations != attestations
        ):
            raise _error("evidence_checker is not exactly bound to evaluator inputs")

        # Fail early if the configured Judge identities cannot grade the three
        # Review tasks emitted by this evaluator.
        for kind in (
            JudgeKind.FINDING_EQUIVALENCE,
            JudgeKind.NOVEL_FACTUALITY,
            JudgeKind.EVIDENCE_SUPPORT,
        ):
            profile = self.evaluator_execution.evaluator.profile(kind)
            rubric = self.rubrics.for_task(JudgeTask(kind.value))
            system_prompt = (
                GLOBAL_JUDGE_SYSTEM_PROMPT
                + "\nTask rubric:\n"
                + rubric.instruction
            )
            if (
                profile.rubric_id != rubric.rubric_id
                or profile.rubric_digest != rubric.rubric_digest
                or profile.rubric_version != rubric.rubric_version
                or profile.response_schema_version != rubric.response_schema
                or profile.response_schema_digest
                != canonical_sha256(rubric.response_schema)
                or profile.system_prompt_version != JUDGE_SYSTEM_PROMPT_VERSION
                or profile.system_prompt_digest != canonical_sha256(system_prompt)
                or profile.context_builder_version
                != JUDGE_CONTEXT_BUILDER_VERSION
                or profile.parser_version != JUDGE_PARSER_VERSION
            ):
                raise _error("EvaluatorExecutionConfig Judge profile differs from Review rubric")

    @property
    def deterministic_context_digest(self) -> str:
        assert self.evidence_checker is not None
        if type(self.replay) is PreparedRepositoryReplay:
            assert self.location_matcher is not None
            replay_binding = {
                "kind": ReviewTargetKind.REPOSITORY.value,
                "prepared_repository_id": self.replay.prepared_repository_id,
                "repository_descriptor_digest": (
                    self.replay.repository_descriptor_digest
                ),
                "base_revision": self.replay.base_revision,
                "head_revision": self.replay.head_revision,
                "side_path_catalog_digest": (
                    self.location_matcher.side_paths.digest()
                ),
                "location_policy": self.location_matcher.policy.to_dict(),
            }
        else:
            assert type(self.replay) is FrozenContextReplay
            replay_binding = {
                "kind": ReviewTargetKind.FROZEN_CONTEXT.value,
                "bundle_id": self.replay.bundle_id,
                "record_id": self.replay.record_id,
                "context_ref": self.replay.context_ref,
                "context_format": self.replay.context_format,
                "rendered_sha256": self.replay.rendered_sha256,
                "rendered_utf8_bytes": self.replay.rendered_utf8_bytes,
                "source_binding_digest": self.replay.source_binding_digest,
                "replay_binding_digest": self.replay.replay_binding_digest,
            }
        return canonical_sha256(
            {
                "schema_version": "eval_review_deterministic_context_v2",
                "eval_input_digest": self.eval_input.digest(),
                "target_materialization_id": self.target_materialization_id,
                "replay_binding": replay_binding,
                "evidence_policy_version": EVIDENCE_INTEGRITY_POLICY_VERSION,
                "trial_id": self.trial_id,
                "command_attestation_digests": sorted(
                    item.digest() for item in self.command_attestations
                ),
                "context_bundle_digest": self.context_bundle.digest(),
                "review_evaluator_context_digest": (
                    self.review_evaluator_context.digest()
                ),
                "review_evaluator_context_policy_version": (
                    self.evaluator_execution.review_evaluator_context_policy_version
                ),
                "judge_rubric_catalog_digest": self.rubrics.catalog_digest,
            }
        )

    def _base(
        self,
        submission: EvalSubmission,
        truth: ReviewTruth,
    ) -> _EvaluationBase:
        if type(submission) is not EvalSubmission:
            raise _error("submission must be EvalSubmission")
        if type(truth) is not ReviewTruth:
            raise _error("review_truth must be ReviewTruth")
        if submission.review is None:
            raise _error("Review evaluation requires a Submission Review")
        if submission.task_id != self.eval_input.task_id:
            raise _error("Submission task_id differs from EvalInput")
        if submission.eval_input_digest != self.eval_input.digest():
            raise _error("Submission eval_input_digest differs from EvalInput")
        if (
            submission.target_materialization_id
            != self.target_materialization_id
        ):
            raise _error(
                "Submission target_materialization_id differs from ReviewEvaluator"
            )
        if submission.trial_id != self.trial_id:
            raise _error("Submission trial_id differs from ReviewEvaluator")
        self._validate_truth_claim_conflicts(truth)
        review = submission.review
        findings = tuple(review.findings)
        self._validate_context_selectors(findings=findings, truth=truth)
        evidence = tuple(submission.evidence)
        assert self.evidence_checker is not None
        evidence_results = self.evidence_checker.check_all(findings, evidence)
        return _EvaluationBase(
            submission=submission,
            review=review,
            truth=truth,
            findings=findings,
            evidence=evidence,
            evidence_results=evidence_results,
            evidence_by_finding={item.finding_id: item for item in evidence_results},
            evidence_by_id={item.evidence_id: item for item in evidence},
            submission_digest=submission.digest(),
            submission_review_digest=canonical_sha256(review.to_dict()),
            submission_evidence_digest=canonical_sha256(
                [item.to_dict() for item in evidence]
            ),
            review_truth_digest=canonical_sha256(truth.to_dict()),
            deterministic_context_digest=self.deterministic_context_digest,
        )

    def _validate_context_selectors(
        self,
        *,
        findings: Sequence[SubmissionFinding],
        truth: ReviewTruth,
    ) -> None:
        finding_ids = {item.finding_id for item in findings}
        expected_ids = {item.truth_id for item in truth.expected_findings}
        invalid_ids = {item.truth_id for item in truth.known_invalid_findings}
        review_truth_ids = expected_ids | invalid_ids
        for entry in self.review_evaluator_context.truth_contexts:
            if entry.truth_id not in review_truth_ids:
                raise _error(
                    "review evaluator context truth selector references an unknown truth Finding"
                )
        for entry in self.context_bundle.finding_entries:
            if entry.finding_id not in finding_ids:
                raise _error("context finding selector references an unknown Finding")
        for entry in self.context_bundle.pair_entries:
            if entry.finding_id not in finding_ids:
                raise _error("context pair selector references an unknown Finding")
            valid_ids = expected_ids if entry.truth_kind is ReviewTruthKind.EXPECTED else invalid_ids
            if entry.truth_id not in valid_ids:
                raise _error("context pair selector references an unknown truth Finding")

    def _evaluator_context_sources(
        self,
        *,
        truth_id: str,
        task: EvaluatorContextTask,
    ) -> Tuple[JudgeContextSource, ...]:
        for entry in self.review_evaluator_context.truth_contexts:
            if entry.truth_id != truth_id:
                continue
            if task not in entry.allowed_tasks:
                return ()
            sources = tuple(
                sorted(
                    (
                        _swe_truth_diff_hunk_source(truth_id, source)
                        for source in entry.sources
                    ),
                    key=lambda item: item.source_id,
                )
            )
            return _merge_context_sources(
                sources,
                "truth-scoped evaluator context sources",
            )
        return ()

    @staticmethod
    def _validate_truth_claim_conflicts(truth: ReviewTruth) -> None:
        # Inputs have already passed strict Unicode validation.  NFC is used
        # here solely to prevent visually identical expected/invalid claims
        # from acquiring contradictory labels.
        expected_claims = {
            unicodedata.normalize("NFC", item.claim): item.truth_id
            for item in truth.expected_findings
        }
        for item in truth.known_invalid_findings:
            canonical = unicodedata.normalize("NFC", item.claim)
            if canonical in expected_claims:
                raise _error(
                    "ReviewTruth contains the same canonical claim as expected and known-invalid"
                )

    def _location_audits(
        self, base: _EvaluationBase
    ) -> Tuple[
        Tuple[LocationAuditRecord, ...],
        Mapping[Tuple[str, ReviewTruthKind, str], Tuple[LocationAuditRecord, ...]],
    ]:
        location_scorable_expected = tuple(
            item
            for item in base.truth.expected_findings
            if item.metric_authority.location_scorable
        )
        truth_locations = sum(
            len(item.locations)
            for item in (
                *location_scorable_expected,
                *base.truth.known_invalid_findings,
            )
        )
        observed = len(base.findings) * truth_locations
        if observed > MAX_REVIEW_LOCATION_AUDITS:
            raise _LimitExceeded(
                ReviewLimitScope.LOCATION_AUDITS,
                observed,
                MAX_REVIEW_LOCATION_AUDITS,
                ReviewReasonCode.LOCATION_LIMIT_EXCEEDED,
                ReviewEvaluationPhase.KNOWN_INVALID,
            )
        if self.location_matcher is None:
            if truth_locations:
                raise _error(
                    "Frozen Review evaluation cannot perform Repository location matching"
                )
            return (), {}
        records: list[LocationAuditRecord] = []
        grouped: Dict[
            Tuple[str, ReviewTruthKind, str], list[LocationAuditRecord]
        ] = {}
        truth_groups = (
            (ReviewTruthKind.EXPECTED, location_scorable_expected),
            (ReviewTruthKind.KNOWN_INVALID, base.truth.known_invalid_findings),
        )
        for finding in base.findings:
            for truth_kind, truths in truth_groups:
                targets = tuple(
                    TruthLocationTarget(item.truth_id, index, location)
                    for item in truths
                    for index, location in enumerate(item.locations)
                )
                candidates = self.location_matcher.generate_candidates(
                    finding, targets
                )
                for candidate in candidates:
                    record = LocationAuditRecord(
                        finding_id=finding.finding_id,
                        truth_kind=truth_kind,
                        truth_id=candidate.truth_id,
                        truth_location_index=candidate.truth_index,
                        truth_location=candidate.truth_location,
                        match=candidate.match,
                    )
                    records.append(record)
                    grouped.setdefault(
                        (finding.finding_id, truth_kind, candidate.truth_id), []
                    ).append(record)
        ordered = tuple(
            sorted(
                records,
                key=lambda item: (
                    item.finding_id,
                    item.truth_kind.value,
                    item.truth_id,
                    item.truth_location_index,
                ),
            )
        )
        return ordered, {
            key: tuple(
                sorted(value, key=lambda item: item.truth_location_index)
            )
            for key, value in grouped.items()
        }

    def _equivalence_request(
        self,
        *,
        base: _EvaluationBase,
        finding: SubmissionFinding,
        truth_kind: ReviewTruthKind,
        truth: Union[ExpectedFinding, KnownInvalidFinding],
        phase: ReviewEvaluationPhase,
    ) -> ReviewJudgeRequestRecord:
        request_id = stable_id(
            "review-finding-equivalence-v1",
            phase.value,
            finding.finding_id,
            truth_kind.value,
            truth.truth_id,
            base.submission_review_digest,
            base.review_truth_digest,
            base.deterministic_context_digest,
        )
        context_sources = _merge_context_sources(
            (
                *self.context_bundle.resolve_pair(
                    finding.finding_id,
                    truth_kind,
                    truth.truth_id,
                ),
                *self._evaluator_context_sources(
                    truth_id=truth.truth_id,
                    task=EvaluatorContextTask.FINDING_EQUIVALENCE,
                ),
            ),
            "Finding-equivalence context",
        )
        request = build_finding_equivalence_judge_input(
            request_id,
            finding,
            truth,
            evidence=self._valid_evidence(base, finding),
            context_sources=context_sources,
            rubrics=self.rubrics,
        )
        return ReviewJudgeRequestRecord(
            request_id=request_id,
            phase=phase,
            finding_id=finding.finding_id,
            truth_kind=truth_kind,
            truth_id=truth.truth_id,
            request=request,
        )

    def _valid_evidence(
        self, base: _EvaluationBase, finding: SubmissionFinding
    ) -> Tuple[SubmissionEvidence, ...]:
        result = base.evidence_by_finding[finding.finding_id]
        valid_ids = {
            item.evidence_id
            for item in result.item_results
            if item.integrity is EvidenceIntegrity.VALID
        }
        seen: set[str] = set()
        values = []
        for evidence_id in finding.evidence_refs:
            if evidence_id in seen or evidence_id not in valid_ids:
                continue
            seen.add(evidence_id)
            evidence = base.evidence_by_id.get(evidence_id)
            if evidence is not None:
                values.append(evidence)
        return tuple(values)

    def _novel_request(
        self, base: _EvaluationBase, finding: SubmissionFinding
    ) -> ReviewJudgeRequestRecord:
        request_id = stable_id(
            "review-novel-factuality-v1",
            finding.finding_id,
            base.submission_review_digest,
            base.review_truth_digest,
            base.deterministic_context_digest,
        )
        request = build_novel_factuality_judge_input(
            request_id,
            finding,
            evidence=self._valid_evidence(base, finding),
            context_sources=self.context_bundle.resolve_finding(finding.finding_id),
            rubrics=self.rubrics,
        )
        return ReviewJudgeRequestRecord(
            request_id=request_id,
            phase=ReviewEvaluationPhase.NOVEL_FACTUALITY,
            finding_id=finding.finding_id,
            truth_kind=None,
            truth_id=None,
            request=request,
        )

    def _support_request(
        self,
        base: _EvaluationBase,
        finding: SubmissionFinding,
        anchors: Sequence[EvidenceAnchor],
        truth_id: Optional[str],
        truth_kind: Optional[ReviewTruthKind] = None,
    ) -> ReviewJudgeRequestRecord:
        valid_evidence = self._valid_evidence(base, finding)
        if not valid_evidence:
            raise _error("Evidence support request requires valid Evidence")
        request_id = stable_id(
            "review-evidence-support-v1",
            finding.finding_id,
            truth_id,
            [item.evidence_id for item in valid_evidence],
            [item.to_dict() for item in anchors],
            base.submission_review_digest,
            base.review_truth_digest,
            base.deterministic_context_digest,
        )
        request = build_evidence_support_judge_input(
            request_id,
            finding,
            valid_evidence,
            anchors=tuple(anchors),
            context_sources=(
                self.context_bundle.resolve_pair(
                    finding.finding_id, truth_kind, truth_id
                )
                if truth_kind is not None and truth_id is not None
                else self.context_bundle.resolve_finding(finding.finding_id)
            ),
            rubrics=self.rubrics,
        )
        return ReviewJudgeRequestRecord(
            request_id=request_id,
            phase=ReviewEvaluationPhase.EVIDENCE_SUPPORT,
            finding_id=finding.finding_id,
            truth_kind=None,
            truth_id=truth_id,
            request=request,
        )

    def _validate_request_budget(
        self,
        requests: Sequence[ReviewJudgeRequestRecord],
        phase: ReviewEvaluationPhase,
    ) -> None:
        if len(requests) > MAX_REVIEW_JUDGE_REQUESTS:
            raise _LimitExceeded(
                ReviewLimitScope.JUDGE_REQUESTS,
                len(requests),
                MAX_REVIEW_JUDGE_REQUESTS,
                ReviewReasonCode.JUDGE_REQUEST_LIMIT_EXCEEDED,
                phase,
            )
        try:
            JudgeInputArtifact.create(
                self.evaluator_execution,
                tuple(item.request for item in requests),
            )
        except Exception as exc:
            text = str(exc).lower()
            budgets = self.evaluator_execution.judge_budgets
            if "token" in text:
                scope = ReviewLimitScope.JUDGE_REQUEST_TOKENS
                limit = budgets.max_total_judge_request_tokens
            else:
                scope = ReviewLimitScope.JUDGE_REQUEST_BYTES
                limit = budgets.max_total_judge_request_bytes
            raise _LimitExceeded(
                scope,
                limit + 1,
                limit,
                ReviewReasonCode.JUDGE_REQUEST_LIMIT_EXCEEDED,
                phase,
            ) from exc

    def _pair_state(
        self,
        *,
        base: _EvaluationBase,
        finding: SubmissionFinding,
        truth_kind: ReviewTruthKind,
        truth: Union[ExpectedFinding, KnownInvalidFinding],
        phase: ReviewEvaluationPhase,
        registry: _JudgeResultRegistry,
    ) -> Tuple[_PairState, Optional[ReviewJudgeRequestRecord]]:
        if finding.claim == truth.claim:
            severity = None
            if (
                type(truth) is ExpectedFinding
                and truth.metric_authority.severity_scorable
            ):
                assert truth.severity is not None
                severity = _severity_assessment(finding.severity, truth.severity)
            return (
                _PairState(
                    finding=finding,
                    truth_kind=truth_kind,
                    truth=truth,
                    match_kind=FindingMatchKind.EXACT,
                    request=None,
                    relation=FindingMatchRelation.EQUIVALENT,
                    score_ppm=None,
                    edge_weight=EXACT_REVIEW_EDGE_WEIGHT,
                    resolution=FindingResolution.RESOLVED,
                    severity_assessment=severity,
                    actionability=_deterministic_actionability(finding),
                    reason_codes=(ReviewReasonCode.DETERMINISTIC_EXACT,),
                ),
                None,
            )

        request = self._equivalence_request(
            base=base,
            finding=finding,
            truth_kind=truth_kind,
            truth=truth,
            phase=phase,
        )
        result = registry.resolve(request)
        if result is None:
            return (
                _PairState(
                    finding=finding,
                    truth_kind=truth_kind,
                    truth=truth,
                    match_kind=FindingMatchKind.SEMANTIC,
                    request=request,
                    relation=None,
                    score_ppm=None,
                    edge_weight=None,
                    resolution=FindingResolution.PENDING_JUDGE,
                    severity_assessment=None,
                    actionability=None,
                    reason_codes=(ReviewReasonCode.JUDGE_PENDING,),
                ),
                request,
            )
        if result.status is JudgeRunStatus.JUDGE_FAILED:
            return (
                _PairState(
                    finding=finding,
                    truth_kind=truth_kind,
                    truth=truth,
                    match_kind=FindingMatchKind.SEMANTIC,
                    request=request,
                    relation=None,
                    score_ppm=None,
                    edge_weight=None,
                    resolution=FindingResolution.JUDGE_FAILED,
                    severity_assessment=None,
                    actionability=None,
                    reason_codes=(ReviewReasonCode.JUDGE_FAILED,),
                ),
                request,
            )
        if result.status is JudgeRunStatus.UNGRADED:
            return (
                _PairState(
                    finding=finding,
                    truth_kind=truth_kind,
                    truth=truth,
                    match_kind=FindingMatchKind.SEMANTIC,
                    request=request,
                    relation=None,
                    score_ppm=None,
                    edge_weight=None,
                    resolution=FindingResolution.UNGRADED,
                    severity_assessment=None,
                    actionability=None,
                    reason_codes=(ReviewReasonCode.JUDGE_UNGRADED,),
                ),
                request,
            )
        decision = result.decision
        if type(decision) is not FindingEquivalenceJudgeDecision:
            raise _error("finding equivalence result has the wrong typed decision")
        severity_assessment = decision.severity_assessment
        if (
            type(truth) is ExpectedFinding
            and not truth.metric_authority.severity_scorable
        ):
            severity_assessment = None
        if decision.relation is FindingMatchRelation.UNKNOWN:
            resolution = FindingResolution.UNGRADED
            edge_weight = None
            reasons = (ReviewReasonCode.JUDGE_UNKNOWN,)
        elif decision.relation is FindingMatchRelation.EQUIVALENT:
            resolution = FindingResolution.RESOLVED
            edge_weight = SEMANTIC_REVIEW_EDGE_WEIGHT_BASE + decision.score_ppm
            reasons = (ReviewReasonCode.SEMANTIC_EQUIVALENT,)
        elif decision.relation is FindingMatchRelation.PARTIALLY_EQUIVALENT:
            resolution = FindingResolution.RESOLVED
            edge_weight = None
            reasons = (ReviewReasonCode.SEMANTIC_PARTIAL,)
        else:
            resolution = FindingResolution.RESOLVED
            edge_weight = None
            reasons = (ReviewReasonCode.SEMANTIC_DIFFERENT,)
        return (
            _PairState(
                finding=finding,
                truth_kind=truth_kind,
                truth=truth,
                match_kind=FindingMatchKind.SEMANTIC,
                request=request,
                relation=decision.relation,
                score_ppm=decision.score_ppm,
                edge_weight=edge_weight,
                resolution=resolution,
                severity_assessment=severity_assessment,
                actionability=decision.actionability,
                reason_codes=reasons,
            ),
            request,
        )

    @staticmethod
    def _ungraded_drafts(
        findings: Sequence[SubmissionFinding],
        reason: ReviewReasonCode,
    ) -> Dict[str, _OutcomeDraft]:
        from .models import IssueJudgement

        return {
            item.finding_id: _OutcomeDraft(
                finding_id=item.finding_id,
                issue_resolution=FindingResolution.UNGRADED,
                issue_judgement=IssueJudgement.UNKNOWN,
                disposition=FindingDisposition.UNGRADED,
                reasons=(reason,),
            )
            for item in findings
        }

    @staticmethod
    def _location_records_for(
        location_map: Mapping[Tuple[str, ReviewTruthKind, str], Tuple[LocationAuditRecord, ...]],
        finding_id: str,
        truth_kind: ReviewTruthKind,
        truth_id: str,
    ) -> Tuple[LocationAuditRecord, ...]:
        return location_map.get((finding_id, truth_kind, truth_id), ())

    @staticmethod
    def _selected_pair_map(
        states: Sequence[_PairState],
    ) -> Dict[Tuple[str, str], _PairState]:
        eligible = [item for item in states if item.edge_weight is not None]
        return {(item.finding.finding_id, item.truth.truth_id): item for item in eligible}

    def _make_outcome(
        self,
        draft: _OutcomeDraft,
        evidence_result: EvidenceIntegrityResult,
    ) -> FindingOutcome:
        reasons = list(draft.reasons)
        if evidence_result.integrity is EvidenceIntegrity.VALID:
            reasons.append(ReviewReasonCode.EVIDENCE_VALID)
        elif evidence_result.integrity is EvidenceIntegrity.INVALID:
            reasons.append(ReviewReasonCode.EVIDENCE_INVALID)
        else:
            reasons.append(ReviewReasonCode.EVIDENCE_MISSING)
        if draft.evidence_support_resolution is EvidenceSupportResolution.NOT_REQUESTED:
            reasons.append(ReviewReasonCode.EVIDENCE_SUPPORT_NOT_REQUESTED)
        elif draft.evidence_support_resolution is EvidenceSupportResolution.PENDING_JUDGE:
            reasons.append(ReviewReasonCode.JUDGE_PENDING)
        elif draft.evidence_support_resolution is EvidenceSupportResolution.JUDGE_FAILED:
            reasons.append(ReviewReasonCode.JUDGE_FAILED)
        elif draft.evidence_support_resolution is EvidenceSupportResolution.UNGRADED:
            reasons.append(ReviewReasonCode.JUDGE_UNGRADED)
        elif draft.evidence_support is EvidenceSupport.SUPPORTED:
            reasons.append(ReviewReasonCode.EVIDENCE_SUPPORT_SUPPORTED)
        elif draft.evidence_support is EvidenceSupport.WEAK:
            reasons.append(ReviewReasonCode.EVIDENCE_SUPPORT_WEAK)
        elif draft.evidence_support is EvidenceSupport.UNSUPPORTED:
            reasons.append(ReviewReasonCode.EVIDENCE_SUPPORT_UNSUPPORTED)
        else:
            reasons.append(ReviewReasonCode.EVIDENCE_SUPPORT_UNKNOWN)
        publishable = _strict_publishable(
            draft.issue_resolution,
            draft.issue_judgement,
            draft.disposition,
            evidence_result.integrity,
            draft.evidence_support_resolution,
            draft.evidence_support,
        )
        if not publishable:
            reasons.append(ReviewReasonCode.NOT_STRICT_PUBLISHABLE)
        unique_reasons = tuple(sorted(set(reasons), key=lambda item: item.value))
        return FindingOutcome(
            finding_id=draft.finding_id,
            issue_resolution=draft.issue_resolution,
            issue_judgement=draft.issue_judgement,
            disposition=draft.disposition,
            matched_expected_truth_id=draft.matched_expected_truth_id,
            matched_known_invalid_truth_id=draft.matched_known_invalid_truth_id,
            duplicate_truth_id=draft.duplicate_truth_id,
            duplicate_of_finding_id=draft.duplicate_of_finding_id,
            novel_request_id=draft.novel_request_id,
            severity_assessment=draft.severity_assessment,
            actionability=draft.actionability,
            evidence_integrity=evidence_result.integrity,
            evidence_support_resolution=draft.evidence_support_resolution,
            evidence_support=draft.evidence_support,
            evidence_support_request_id=draft.evidence_support_request_id,
            strict_publishable=publishable,
            reason_codes=unique_reasons,
        )

    def _assemble(
        self,
        *,
        base: _EvaluationBase,
        location_records: Sequence[LocationAuditRecord],
        known_candidates: Sequence[ReviewCandidateRecord],
        expected_candidates: Sequence[ReviewCandidateRecord],
        assignments: Sequence[ReviewAssignmentRecord],
        drafts: Mapping[str, _OutcomeDraft],
        unmatched_expected: Sequence[str],
        requests: Sequence[ReviewJudgeRequestRecord],
        registry: _JudgeResultRegistry,
        status: ReviewEvaluationStatus,
        phase: ReviewEvaluationPhase,
        extra_reasons: Sequence[ReviewReasonCode] = (),
        limit_failure: Optional[ReviewLimitFailure] = None,
    ) -> ReviewEvaluationResult:
        outcomes = tuple(
            self._make_outcome(drafts[item.finding_id], base.evidence_by_finding[item.finding_id])
            for item in base.findings
        )
        requests_ordered = tuple(sorted(requests, key=lambda item: item.request_id))
        decisions = tuple(sorted(registry.decisions, key=lambda item: item.request_id))
        failures = tuple(sorted(registry.failures, key=lambda item: item.request_id))
        ungraded = tuple(sorted(registry.ungraded, key=lambda item: item.request_id))
        pending = len(requests_ordered) - len(decisions) - len(failures) - len(ungraded)
        semantic_unknown = sum(
            1
            for receipt in decisions
            if (
                getattr(receipt.decision, "relation", None) is FindingMatchRelation.UNKNOWN
                or getattr(receipt.decision, "factuality", None) is NovelFactuality.UNKNOWN
                or getattr(receipt.decision, "support", None) is EvidenceSupport.UNKNOWN
            )
        )
        coverage = ReviewCoverage(
            judge_request_count=len(requests_ordered),
            judge_graded_count=len(decisions),
            judge_failed_count=len(failures),
            judge_ungraded_count=len(ungraded),
            judge_pending_count=pending,
            semantic_unknown_count=semantic_unknown,
            finding_count=len(base.findings),
            finding_resolved_count=sum(item.issue_resolution is FindingResolution.RESOLVED for item in outcomes),
            evidence_result_count=len(base.evidence_results),
        )
        expected_ids = {item.truth_id: item for item in base.truth.expected_findings}
        matched_truth_ids = {
            item.matched_expected_truth_id
            for item in outcomes
            if item.matched_expected_truth_id is not None
        }
        integrity_counts = {
            EvidenceIntegrity.VALID: 0,
            EvidenceIntegrity.INVALID: 0,
            EvidenceIntegrity.MISSING: 0,
        }
        for item in base.evidence_results:
            integrity_counts[item.integrity] += 1
        support_counts = {
            EvidenceSupport.SUPPORTED: 0,
            EvidenceSupport.WEAK: 0,
            EvidenceSupport.UNSUPPORTED: 0,
            EvidenceSupport.UNKNOWN: 0,
        }
        for item in outcomes:
            support_counts[item.evidence_support] += 1
        metrics = ReviewMetricInputs(
            scorable=status is ReviewEvaluationStatus.GRADED,
            generated_finding_count=len(base.findings),
            expected_truth_count=len(expected_ids),
            required_expected_truth_count=sum(item.required for item in expected_ids.values()),
            matched_finding_count=sum(item.disposition is FindingDisposition.MATCHED for item in outcomes),
            matched_expected_truth_count=len(matched_truth_ids),
            matched_required_truth_count=sum(
                1 for item in matched_truth_ids if expected_ids[item].required
            ),
            duplicate_finding_count=sum(item.disposition is FindingDisposition.DUPLICATE for item in outcomes),
            known_invalid_finding_count=sum(item.disposition is FindingDisposition.KNOWN_INVALID for item in outcomes),
            plausible_novel_count=sum(
                item.disposition is FindingDisposition.NOVEL_ALLOWED and item.issue_judgement.value == "plausible"
                for item in outcomes
            ),
            fabricated_finding_count=sum(item.issue_judgement.value == "fabricated" for item in outcomes),
            unknown_finding_count=sum(item.issue_judgement.value == "unknown" for item in outcomes),
            unmatched_expected_truth_count=len(tuple(unmatched_expected)),
            unmatched_required_truth_count=sum(
                expected_ids[item].required for item in unmatched_expected if item in expected_ids
            ),
            evidence_valid_count=integrity_counts[EvidenceIntegrity.VALID],
            evidence_invalid_count=integrity_counts[EvidenceIntegrity.INVALID],
            evidence_missing_count=integrity_counts[EvidenceIntegrity.MISSING],
            evidence_supported_count=support_counts[EvidenceSupport.SUPPORTED],
            evidence_weak_count=support_counts[EvidenceSupport.WEAK],
            evidence_unsupported_count=support_counts[EvidenceSupport.UNSUPPORTED],
            evidence_support_unknown_count=support_counts[EvidenceSupport.UNKNOWN],
            strict_publishable_count=sum(item.strict_publishable for item in outcomes),
        )
        reasons = set(extra_reasons)
        for item in outcomes:
            reasons.update(item.reason_codes)
        if tuple(unmatched_expected):
            reasons.add(ReviewReasonCode.OPTIONAL_TRUTH_MISSED)
            if any(expected_ids[item].required for item in unmatched_expected if item in expected_ids):
                reasons.add(ReviewReasonCode.REQUIRED_TRUTH_MISSED)
        if pending:
            reasons.add(ReviewReasonCode.JUDGE_PENDING)
        if failures:
            reasons.add(ReviewReasonCode.JUDGE_FAILED)
        if ungraded or semantic_unknown:
            reasons.add(ReviewReasonCode.JUDGE_UNGRADED)
        values: Dict[str, Any] = dict(
            schema_version=REVIEW_EVALUATION_SCHEMA_VERSION,
            evaluator_revision=self.evaluator_revision,
            evaluator_execution_digest=self.evaluator_execution.digest(),
            submission_digest=base.submission_digest,
            submission_review_digest=base.submission_review_digest,
            submission_evidence_digest=base.submission_evidence_digest,
            eval_input_digest=self.eval_input.digest(),
            review_truth_digest=base.review_truth_digest,
            deterministic_context_digest=base.deterministic_context_digest,
            review_policy_version=REVIEW_MATCH_POLICY_VERSION,
            assignment_policy_version=ASSIGNMENT_POLICY_VERSION,
            location_policy_version=self.location_matcher.policy.version if self.location_matcher is not None else REVIEW_LOCATION_POLICY_VERSION,
            evidence_integrity_policy_version=EVIDENCE_INTEGRITY_POLICY_VERSION,
            truth_completeness=base.truth.completeness,
            novel_finding_policy=base.truth.novel_finding_policy,
            status=status,
            phase=phase,
            generated_findings=tuple(sorted(base.findings, key=lambda item: item.finding_id)),
            expected_truth_findings=base.truth.expected_findings,
            known_invalid_truth_findings=base.truth.known_invalid_findings,
            location_candidates=tuple(sorted(location_records, key=lambda item: (item.finding_id, item.truth_kind.value, item.truth_id, item.truth_location_index))),
            known_invalid_candidates=tuple(sorted(known_candidates, key=lambda item: (item.finding_id, item.truth_id))),
            expected_candidates=tuple(sorted(expected_candidates, key=lambda item: (item.finding_id, item.truth_id))),
            assignments=tuple(sorted(assignments, key=lambda item: (item.finding_id, item.truth_id))),
            finding_outcomes=outcomes,
            unmatched_expected_truth_ids=tuple(sorted(set(unmatched_expected))),
            judge_requests=requests_ordered,
            judge_decisions=decisions,
            judge_failures=failures,
            judge_ungraded=ungraded,
            evidence_integrity_results=tuple(sorted(base.evidence_results, key=lambda item: item.finding_id)),
            coverage=coverage,
            metrics=metrics,
            reason_codes=tuple(sorted(reasons, key=lambda item: item.value)),
            limit_failure=limit_failure,
        )
        model_fields = dataclass_fields(ReviewEvaluationResult)
        if set(values) != {field.name for field in model_fields}:
            raise AssertionError("internal Review result construction is incomplete")
        result = object.__new__(ReviewEvaluationResult)
        for field in model_fields:
            object.__setattr__(result, field.name, values[field.name])
        result.__post_init__()
        return result

    def _limit_result(
        self,
        base: _EvaluationBase,
        failure: _LimitExceeded,
    ) -> ReviewEvaluationResult:
        """Return a small harness-owned ungraded artifact on any hard limit."""

        empty_registry = _JudgeResultRegistry((), self.evaluator_execution)
        drafts = self._ungraded_drafts(
            base.findings,
            failure.failure.reason_code,
        )
        return self._assemble(
            base=base,
            location_records=(),
            known_candidates=(),
            expected_candidates=(),
            assignments=(),
            drafts=drafts,
            unmatched_expected=tuple(item.truth_id for item in base.truth.expected_findings),
            requests=(),
            registry=empty_registry,
            status=ReviewEvaluationStatus.UNGRADED,
            phase=failure.phase,
            extra_reasons=(failure.failure.reason_code,),
            limit_failure=failure.failure,
        )

    @staticmethod
    def _known_hit_selection(
        states: Sequence[_PairState],
    ) -> Dict[str, _PairState]:
        selected: Dict[str, _PairState] = {}
        for state in states:
            if state.relation is not FindingMatchRelation.EQUIVALENT:
                continue
            finding_id = state.finding.finding_id
            previous = selected.get(finding_id)
            if previous is None or (
                (-int(state.edge_weight or 0), state.truth.truth_id)
                < (-int(previous.edge_weight or 0), previous.truth.truth_id)
            ):
                selected[finding_id] = state
        return selected

    @staticmethod
    def _draft_from_known_invalid(state: _PairState) -> _OutcomeDraft:
        from .models import IssueJudgement

        return _OutcomeDraft(
            finding_id=state.finding.finding_id,
            issue_resolution=FindingResolution.RESOLVED,
            issue_judgement=IssueJudgement.FABRICATED,
            disposition=FindingDisposition.KNOWN_INVALID,
            matched_known_invalid_truth_id=state.truth.truth_id,
            severity_assessment=state.severity_assessment,
            actionability=state.actionability,
            reasons=(ReviewReasonCode.KNOWN_INVALID_MATCH,),
        )

    @staticmethod
    def _draft_from_assignment(state: _PairState) -> _OutcomeDraft:
        from .models import IssueJudgement

        return _OutcomeDraft(
            finding_id=state.finding.finding_id,
            issue_resolution=FindingResolution.RESOLVED,
            issue_judgement=IssueJudgement.CONFIRMED,
            disposition=FindingDisposition.MATCHED,
            matched_expected_truth_id=state.truth.truth_id,
            severity_assessment=state.severity_assessment,
            actionability=state.actionability,
            reasons=(ReviewReasonCode.EXPECTED_MATCH,),
        )

    @staticmethod
    def _draft_from_duplicate(
        state: _PairState,
        duplicate_of_finding_id: str,
    ) -> _OutcomeDraft:
        from .models import IssueJudgement

        return _OutcomeDraft(
            finding_id=state.finding.finding_id,
            issue_resolution=FindingResolution.RESOLVED,
            issue_judgement=IssueJudgement.PLAUSIBLE,
            disposition=FindingDisposition.DUPLICATE,
            duplicate_truth_id=state.truth.truth_id,
            duplicate_of_finding_id=duplicate_of_finding_id,
            severity_assessment=state.severity_assessment,
            actionability=state.actionability,
            reasons=(ReviewReasonCode.DUPLICATE_FINDING,),
        )

    def _evaluate_staged(
        self,
        base: _EvaluationBase,
        registry: _JudgeResultRegistry,
    ) -> ReviewEvaluationResult:
        from .models import IssueJudgement

        location_records, location_map = self._location_audits(base)
        requests: list[ReviewJudgeRequestRecord] = []
        known_states: list[_PairState] = []
        known_truths = tuple(base.truth.known_invalid_findings)
        expected_truths = tuple(base.truth.expected_findings)

        # Stage 1: form the complete known-invalid graph before assigning any
        # expected truth.  Exact pairs remain in the graph even though they do
        # not need a model request.
        for finding in base.findings:
            for truth in known_truths:
                state, request = self._pair_state(
                    base=base,
                    finding=finding,
                    truth_kind=ReviewTruthKind.KNOWN_INVALID,
                    truth=truth,
                    phase=ReviewEvaluationPhase.KNOWN_INVALID,
                    registry=registry,
                )
                known_states.append(state)
                if request is not None:
                    requests.append(request)
        if len(known_states) > MAX_REVIEW_CANDIDATES:
            raise _LimitExceeded(
                ReviewLimitScope.CANDIDATES,
                len(known_states),
                MAX_REVIEW_CANDIDATES,
                ReviewReasonCode.CANDIDATE_LIMIT_EXCEEDED,
                ReviewEvaluationPhase.KNOWN_INVALID,
            )
        self._validate_request_budget(requests, ReviewEvaluationPhase.KNOWN_INVALID)
        known_unresolved = [
            item
            for item in known_states
            if item.resolution is not FindingResolution.RESOLVED
        ]
        if known_unresolved:
            known_candidates = tuple(
                item.candidate(
                    selected=False,
                    location_records=self._location_records_for(
                        location_map,
                        item.finding.finding_id,
                        item.truth_kind,
                        item.truth.truth_id,
                    ),
                )
                for item in known_states
            )
            drafts = self._ungraded_drafts(
                base.findings,
                (
                    ReviewReasonCode.JUDGE_PENDING
                    if any(item.resolution is FindingResolution.PENDING_JUDGE for item in known_unresolved)
                    else ReviewReasonCode.JUDGE_UNGRADED
                ),
            )
            status = (
                ReviewEvaluationStatus.PENDING_JUDGE
                if any(item.resolution is FindingResolution.PENDING_JUDGE for item in known_unresolved)
                else ReviewEvaluationStatus.UNGRADED
            )
            registry.ensure_no_extra_results()
            return self._assemble(
                base=base,
                location_records=location_records,
                known_candidates=known_candidates,
                expected_candidates=(),
                assignments=(),
                drafts=drafts,
                unmatched_expected=tuple(item.truth_id for item in expected_truths),
                requests=requests,
                registry=registry,
                status=status,
                phase=ReviewEvaluationPhase.KNOWN_INVALID,
                extra_reasons=(
                    ReviewReasonCode.JUDGE_PENDING
                    if status is ReviewEvaluationStatus.PENDING_JUDGE
                    else ReviewReasonCode.JUDGE_UNGRADED,
                ),
            )

        known_hits = self._known_hit_selection(known_states)
        known_candidates = tuple(
            item.candidate(
                selected=(known_hits.get(item.finding.finding_id) is item),
                location_records=self._location_records_for(
                    location_map,
                    item.finding.finding_id,
                    item.truth_kind,
                    item.truth.truth_id,
                ),
            )
            for item in known_states
        )
        drafts: Dict[str, _OutcomeDraft] = {
            finding_id: self._draft_from_known_invalid(state)
            for finding_id, state in known_hits.items()
        }
        eligible_findings = tuple(
            item for item in base.findings if item.finding_id not in known_hits
        )
        expected_pair_count = len(known_states) + len(eligible_findings) * len(expected_truths)
        if expected_pair_count > MAX_REVIEW_CANDIDATES:
            raise _LimitExceeded(
                ReviewLimitScope.CANDIDATES,
                expected_pair_count,
                MAX_REVIEW_CANDIDATES,
                ReviewReasonCode.CANDIDATE_LIMIT_EXCEEDED,
                ReviewEvaluationPhase.EXPECTED_ASSIGNMENT,
            )

        # Stage 2: expected truth candidates and one-to-one assignment.
        expected_states: list[_PairState] = []
        for finding in eligible_findings:
            for truth in expected_truths:
                state, request = self._pair_state(
                    base=base,
                    finding=finding,
                    truth_kind=ReviewTruthKind.EXPECTED,
                    truth=truth,
                    phase=ReviewEvaluationPhase.EXPECTED_ASSIGNMENT,
                    registry=registry,
                )
                expected_states.append(state)
                if request is not None:
                    requests.append(request)
        self._validate_request_budget(requests, ReviewEvaluationPhase.EXPECTED_ASSIGNMENT)
        expected_unresolved = [
            item for item in expected_states if item.resolution is not FindingResolution.RESOLVED
        ]
        if expected_unresolved:
            for finding in eligible_findings:
                drafts[finding.finding_id] = _OutcomeDraft(
                    finding_id=finding.finding_id,
                    issue_resolution=(
                        FindingResolution.PENDING_JUDGE
                        if any(item.finding.finding_id == finding.finding_id and item.resolution is FindingResolution.PENDING_JUDGE for item in expected_unresolved)
                        else FindingResolution.UNGRADED
                    ),
                    issue_judgement=IssueJudgement.UNKNOWN,
                    disposition=FindingDisposition.UNGRADED,
                    reasons=(
                        ReviewReasonCode.JUDGE_PENDING
                        if any(item.finding.finding_id == finding.finding_id and item.resolution is FindingResolution.PENDING_JUDGE for item in expected_unresolved)
                        else ReviewReasonCode.JUDGE_UNGRADED,
                    ),
                )
            expected_candidates = tuple(
                item.candidate(
                    selected=False,
                    location_records=self._location_records_for(
                        location_map,
                        item.finding.finding_id,
                        item.truth_kind,
                        item.truth.truth_id,
                    ),
                )
                for item in expected_states
            )
            status = (
                ReviewEvaluationStatus.PENDING_JUDGE
                if any(item.resolution is FindingResolution.PENDING_JUDGE for item in expected_unresolved)
                else ReviewEvaluationStatus.UNGRADED
            )
            registry.ensure_no_extra_results()
            return self._assemble(
                base=base,
                location_records=location_records,
                known_candidates=known_candidates,
                expected_candidates=expected_candidates,
                assignments=(),
                drafts=drafts,
                unmatched_expected=tuple(item.truth_id for item in expected_truths),
                requests=requests,
                registry=registry,
                status=status,
                phase=ReviewEvaluationPhase.EXPECTED_ASSIGNMENT,
                extra_reasons=(
                    ReviewReasonCode.JUDGE_PENDING
                    if status is ReviewEvaluationStatus.PENDING_JUDGE
                    else ReviewReasonCode.JUDGE_UNGRADED,
                ),
            )

        edges = tuple(
            WeightedAssignmentEdge(
                item.finding.finding_id,
                item.truth.truth_id,
                item.edge_weight,
            )
            for item in expected_states
            if item.edge_weight is not None
        )
        assignment = maximum_weight_bipartite_assignment(
            tuple(item.finding_id for item in eligible_findings),
            tuple(item.truth_id for item in expected_truths),
            edges,
            edge_limit=MAX_REVIEW_CANDIDATES,
        )
        state_by_pair = self._selected_pair_map(expected_states)
        selected_pairs = {(item.left_id, item.right_id): item for item in assignment.matches}
        expected_candidates = tuple(
            item.candidate(
                selected=(item.finding.finding_id, item.truth.truth_id) in selected_pairs,
                location_records=self._location_records_for(
                    location_map,
                    item.finding.finding_id,
                    item.truth_kind,
                    item.truth.truth_id,
                ),
            )
            for item in expected_states
        )
        assignments = tuple(
            ReviewAssignmentRecord(
                finding_id=item.left_id,
                truth_id=item.right_id,
                match_kind=state_by_pair[(item.left_id, item.right_id)].match_kind,
                weight=item.weight,
                request_id=(
                    state_by_pair[(item.left_id, item.right_id)].request.request_id
                    if state_by_pair[(item.left_id, item.right_id)].request is not None
                    else None
                ),
            )
            for item in assignment.matches
        )
        assigned_by_finding = {
            item.left_id: state_by_pair[(item.left_id, item.right_id)]
            for item in assignment.matches
        }
        assigned_by_truth = {
            item.right_id: item.left_id for item in assignment.matches
        }
        novel_findings: list[SubmissionFinding] = []
        for finding in eligible_findings:
            finding_id = finding.finding_id
            assigned = assigned_by_finding.get(finding_id)
            if assigned is not None:
                drafts[finding_id] = self._draft_from_assignment(assigned)
                continue
            duplicate_states = [
                item
                for item in expected_states
                if item.finding.finding_id == finding_id
                and item.edge_weight is not None
                and item.truth.truth_id in assigned_by_truth
            ]
            if duplicate_states:
                duplicate_states.sort(key=lambda item: (-int(item.edge_weight or 0), item.truth.truth_id))
                duplicate = duplicate_states[0]
                drafts[finding_id] = self._draft_from_duplicate(
                    duplicate,
                    assigned_by_truth[duplicate.truth.truth_id],
                )
            else:
                novel_findings.append(finding)

        unmatched_expected = tuple(assignment.unmatched_right)

        # Stage 3: the truth-completeness policy controls whether unmatched
        # Findings receive a factuality request.
        novel_requests: Dict[str, ReviewJudgeRequestRecord] = {}
        if novel_findings and base.truth.novel_finding_policy is NovelFindingPolicy.FORBID:
            for finding in novel_findings:
                drafts[finding.finding_id] = _OutcomeDraft(
                    finding_id=finding.finding_id,
                    issue_resolution=FindingResolution.UNGRADED,
                    issue_judgement=IssueJudgement.UNKNOWN,
                    disposition=FindingDisposition.NOVEL_DISALLOWED,
                    reasons=(ReviewReasonCode.NOVEL_FORBID,),
                )
        else:
            for finding in novel_findings:
                request = self._novel_request(base, finding)
                novel_requests[finding.finding_id] = request
                requests.append(request)
            self._validate_request_budget(requests, ReviewEvaluationPhase.NOVEL_FACTUALITY)
            novel_pending = False
            for finding in novel_findings:
                request = novel_requests[finding.finding_id]
                result = registry.resolve(request)
                if result is None:
                    novel_pending = True
                    drafts[finding.finding_id] = _OutcomeDraft(
                        finding_id=finding.finding_id,
                        issue_resolution=FindingResolution.PENDING_JUDGE,
                        issue_judgement=IssueJudgement.UNKNOWN,
                        disposition=FindingDisposition.UNGRADED,
                        novel_request_id=request.request_id,
                        reasons=(ReviewReasonCode.JUDGE_PENDING,),
                    )
                    continue
                if result.status is JudgeRunStatus.JUDGE_FAILED:
                    drafts[finding.finding_id] = _OutcomeDraft(
                        finding_id=finding.finding_id,
                        issue_resolution=FindingResolution.JUDGE_FAILED,
                        issue_judgement=IssueJudgement.UNKNOWN,
                        disposition=FindingDisposition.UNGRADED,
                        novel_request_id=request.request_id,
                        reasons=(ReviewReasonCode.JUDGE_FAILED,),
                    )
                    continue
                if result.status is JudgeRunStatus.UNGRADED:
                    drafts[finding.finding_id] = _OutcomeDraft(
                        finding_id=finding.finding_id,
                        issue_resolution=FindingResolution.UNGRADED,
                        issue_judgement=IssueJudgement.UNKNOWN,
                        disposition=FindingDisposition.UNGRADED,
                        novel_request_id=request.request_id,
                        reasons=(ReviewReasonCode.JUDGE_UNGRADED,),
                    )
                    continue
                decision = result.decision
                if type(decision) is not NovelFactualityJudgeDecision:
                    raise _error("novel factuality result has the wrong typed decision")
                if decision.factuality is NovelFactuality.UNKNOWN:
                    drafts[finding.finding_id] = _OutcomeDraft(
                        finding_id=finding.finding_id,
                        issue_resolution=FindingResolution.UNGRADED,
                        issue_judgement=IssueJudgement.UNKNOWN,
                        disposition=FindingDisposition.UNGRADED,
                        novel_request_id=request.request_id,
                        severity_assessment=decision.severity_assessment,
                        actionability=decision.actionability,
                        reasons=(ReviewReasonCode.NOVEL_UNKNOWN,),
                    )
                elif decision.factuality is NovelFactuality.PLAUSIBLE:
                    drafts[finding.finding_id] = _OutcomeDraft(
                        finding_id=finding.finding_id,
                        issue_resolution=FindingResolution.RESOLVED,
                        issue_judgement=IssueJudgement.PLAUSIBLE,
                        disposition=FindingDisposition.NOVEL_ALLOWED,
                        novel_request_id=request.request_id,
                        severity_assessment=decision.severity_assessment,
                        actionability=decision.actionability,
                        reasons=(ReviewReasonCode.NOVEL_PLAUSIBLE,),
                    )
                else:
                    drafts[finding.finding_id] = _OutcomeDraft(
                        finding_id=finding.finding_id,
                        issue_resolution=FindingResolution.RESOLVED,
                        issue_judgement=IssueJudgement.FABRICATED,
                        disposition=FindingDisposition.NOVEL_ALLOWED,
                        novel_request_id=request.request_id,
                        severity_assessment=decision.severity_assessment,
                        actionability=decision.actionability,
                        reasons=(ReviewReasonCode.NOVEL_FABRICATED,),
                    )
            if novel_pending:
                registry.ensure_no_extra_results()
                return self._assemble(
                    base=base,
                    location_records=location_records,
                    known_candidates=known_candidates,
                    expected_candidates=expected_candidates,
                    assignments=assignments,
                    drafts=drafts,
                    unmatched_expected=unmatched_expected,
                    requests=requests,
                    registry=registry,
                    status=ReviewEvaluationStatus.PENDING_JUDGE,
                    phase=ReviewEvaluationPhase.NOVEL_FACTUALITY,
                    extra_reasons=(ReviewReasonCode.JUDGE_PENDING,),
                )

        # Stage 4: Evidence support is independent from issue matching.  Only
        # confirmed expected issues and verified-plausible novel issues enter
        # this stage, and only valid Evidence is ever sent to the Judge.
        expected_by_id = {item.truth_id: item for item in expected_truths}
        support_targets: list[Tuple[SubmissionFinding, _OutcomeDraft, Tuple[EvidenceAnchor, ...], Optional[ReviewTruthKind]]] = []
        for finding in base.findings:
            draft = drafts[finding.finding_id]
            if draft.disposition is FindingDisposition.MATCHED:
                truth = expected_by_id[draft.matched_expected_truth_id or ""]
                support_targets.append((finding, draft, truth.evidence_anchors, ReviewTruthKind.EXPECTED))
            elif draft.disposition is FindingDisposition.NOVEL_ALLOWED and draft.issue_judgement is IssueJudgement.PLAUSIBLE:
                support_targets.append((finding, draft, (), None))

        support_requests: list[Tuple[SubmissionFinding, _OutcomeDraft, ReviewJudgeRequestRecord]] = []
        for finding, draft, anchors, truth_kind in support_targets:
            if not self._valid_evidence(base, finding):
                continue
            truth_id = draft.matched_expected_truth_id
            request = self._support_request(
                base,
                finding,
                anchors,
                truth_id,
                truth_kind=truth_kind,
            )
            support_requests.append((finding, draft, request))
            requests.append(request)
            draft.evidence_support_request_id = request.request_id
            draft.evidence_support_resolution = EvidenceSupportResolution.PENDING_JUDGE
        self._validate_request_budget(requests, ReviewEvaluationPhase.EVIDENCE_SUPPORT)
        support_pending = False
        for finding, draft, request in support_requests:
            result = registry.resolve(request)
            if result is None:
                support_pending = True
                continue
            if result.status is JudgeRunStatus.JUDGE_FAILED:
                draft.evidence_support_resolution = EvidenceSupportResolution.JUDGE_FAILED
                draft.evidence_support = EvidenceSupport.UNKNOWN
                continue
            if result.status is JudgeRunStatus.UNGRADED:
                draft.evidence_support_resolution = EvidenceSupportResolution.UNGRADED
                draft.evidence_support = EvidenceSupport.UNKNOWN
                continue
            decision = result.decision
            if type(decision) is not EvidenceSupportJudgeDecision:
                raise _error("Evidence support result has the wrong typed decision")
            draft.evidence_support = decision.support
            draft.evidence_support_resolution = (
                EvidenceSupportResolution.UNGRADED
                if decision.support is EvidenceSupport.UNKNOWN
                else EvidenceSupportResolution.RESOLVED
            )

        registry.ensure_no_extra_results()
        if support_pending:
            status = ReviewEvaluationStatus.PENDING_JUDGE
            phase = ReviewEvaluationPhase.EVIDENCE_SUPPORT
        else:
            has_ungraded = bool(registry.failures or registry.ungraded)
            has_semantic_unknown = any(
                getattr(receipt.decision, "relation", None) is FindingMatchRelation.UNKNOWN
                or getattr(receipt.decision, "factuality", None) is NovelFactuality.UNKNOWN
                or getattr(receipt.decision, "support", None) is EvidenceSupport.UNKNOWN
                for receipt in registry.decisions
            )
            has_forbid_novel = bool(
                novel_findings
                and base.truth.novel_finding_policy is NovelFindingPolicy.FORBID
            )
            status = (
                ReviewEvaluationStatus.UNGRADED
                if has_ungraded or has_semantic_unknown or has_forbid_novel
                else ReviewEvaluationStatus.GRADED
            )
            phase = ReviewEvaluationPhase.COMPLETE
        return self._assemble(
            base=base,
            location_records=location_records,
            known_candidates=known_candidates,
            expected_candidates=expected_candidates,
            assignments=assignments,
            drafts=drafts,
            unmatched_expected=unmatched_expected,
            requests=requests,
            registry=registry,
            status=status,
            phase=phase,
            extra_reasons=(
                (ReviewReasonCode.NOVEL_FORBID,)
                if novel_findings and base.truth.novel_finding_policy is NovelFindingPolicy.FORBID
                else ()
            ),
        )

    def evaluate(
        self,
        submission: EvalSubmission,
        review_truth: ReviewTruth,
        *,
        judge_results: Sequence[JudgeExecutionResult] = (),
    ) -> ReviewEvaluationResult:
        """Return the canonical staged result without invoking a model."""

        base = self._base(submission, review_truth)
        registry = _JudgeResultRegistry(judge_results, self.evaluator_execution)
        try:
            return self._evaluate_staged(base, registry)
        except _LimitExceeded as exc:
            return self._limit_result(base, exc)

    def evaluate_with_judge(
        self,
        submission: EvalSubmission,
        review_truth: ReviewTruth,
        judge: Any,
        *,
        max_rounds: int = 8,
    ) -> ReviewEvaluationResult:
        """Execute emitted requests through a supplied typed SemanticJudge.

        The Judge object is intentionally supplied by the composition root;
        this domain layer never constructs a Provider adapter or exposes
        product Runtime state.
        """

        if type(max_rounds) is not int or not 1 <= max_rounds <= 64:
            raise _error("max_rounds must be between 1 and 64")
        if not hasattr(judge, "execute") or not callable(judge.execute):
            raise _error("judge must expose a callable execute method")
        results: list[JudgeExecutionResult] = []
        for _ in range(max_rounds):
            current = self.evaluate(
                submission,
                review_truth,
                judge_results=tuple(results),
            )
            resolved = {
                item.request_id
                for item in (*current.judge_decisions, *current.judge_failures, *current.judge_ungraded)
            }
            pending = [
                item.request
                for item in current.judge_requests
                if item.request_id not in resolved
            ]
            if not pending:
                return current
            for request in pending:
                result = judge.execute(request)
                if type(result) is not JudgeExecutionResult:
                    raise _error("Judge.execute must return JudgeExecutionResult")
                results.append(result)
        raise _error("Judge rounds exhausted while Review requests remain pending")


__all__ = [
    "REVIEW_EVALUATION_SCHEMA_VERSION",
    "REVIEW_EVALUATOR_REVISION",
    "REVIEW_MATCH_POLICY_VERSION",
    "REVIEW_LOCATION_POLICY_VERSION",
    "MAX_REVIEW_EVALUATION_BYTES",
    "MAX_REVIEW_FINDINGS",
    "MAX_REVIEW_TRUTH_FINDINGS",
    "MAX_REVIEW_CANDIDATES",
    "MAX_REVIEW_LOCATION_AUDITS",
    "MAX_REVIEW_JUDGE_REQUESTS",
    "ReviewEvaluationError",
    "ReviewEvaluationStatus",
    "ReviewEvaluationPhase",
    "ReviewTruthKind",
    "FindingMatchKind",
    "FindingDisposition",
    "FindingResolution",
    "EvidenceSupportResolution",
    "ReviewReasonCode",
    "ReviewLimitScope",
    "ReviewContextBundle",
    "ReviewFindingContextEntry",
    "ReviewPairContextEntry",
    "ReviewLimitFailure",
    "LocationAuditRecord",
    "ReviewCandidateRecord",
    "ReviewAssignmentRecord",
    "FindingOutcome",
    "ReviewJudgeRequestRecord",
    "ReviewJudgeDecisionReceipt",
    "ReviewJudgeFailureReceipt",
    "ReviewJudgeUngradedReceipt",
    "ReviewCoverage",
    "ReviewMetricInputs",
    "ReviewEvaluationResult",
    "ReviewEvaluator",
]
