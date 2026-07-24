"""Deterministic Intent grading and the boundary to semantic Judges.

Task 8 deliberately owns the *protocol* around semantic matching, not the
model call.  It projects the canonical Submission Intent into assignment
units, performs all exact/normalized work locally, emits bounded Judge
requests for the remaining same-dimension pairs, and merges only typed Judge
decisions.  No trace, Runtime, Session, Memory, or repository workspace is
consulted here.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, fields as dataclass_fields
from enum import Enum
import unicodedata
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple, cast

from .assignment import (
    ASSIGNMENT_POLICY_VERSION,
    AssignedPair,
    WeightedAssignmentEdge,
    maximum_weight_bipartite_assignment,
)
from .clarification import (
    MaterialClaimCandidateDecision,
    MaterialClaimMatchOutcome,
    MaterialClaimMatchReceipt,
)
from .models import (
    ClarificationAction,
    ClarificationPolicy,
    ClarificationScript,
    IntentClaimJudgement,
    IntentClaimSource,
    IntentDimension,
    IntentResult,
    IntentTruth,
    MAX_CLAIM_CHARS,
    MAX_CLARIFICATION_QUESTIONS,
    MAX_IDENTIFIER_CHARS,
    MAX_INTENT_CLAIMS,
    MAX_TEXT_LIST_ITEMS,
    SubmissionClarificationExchange,
    SubmissionIntent,
    _strict_json_loads,
    canonical_json,
    canonical_json_bytes,
    canonical_sha256,
    stable_id,
    _json_tree,
)


INTENT_EVALUATION_SCHEMA_VERSION = "eval_intent_evaluation_v1"
INTENT_EVALUATOR_REVISION = "intent-evaluator-v1"
INTENT_NORMALIZATION_POLICY_VERSION = "unicode-nfc-whitespace-casefold-v1"
MAX_INTENT_CANDIDATE_EDGES = 65_536
MAX_INTENT_SCORE_PPM = 999_999
MAX_INTENT_EVALUATION_BYTES = 256 * 1024 * 1024
# Deterministic duplicate matrices can be larger than the Judge boundary.  A
# separate total record budget keeps that case bounded while the 65,536 limit
# remains specifically the unresolved semantic/Judge candidate budget.
MAX_INTENT_TOTAL_CANDIDATES = 131_072
MAX_INTENT_REQUEST_TEXT_BYTES = 64 * 1024 * 1024
MAX_INTENT_JUDGE_REASON_REF_BYTES = 8 * 1024 * 1024
MAX_INTENT_CANDIDATE_RECORD_BYTES = 64 * 1024 * 1024
MAX_INTENT_JUDGE_REQUEST_BYTES = 64 * 1024 * 1024
MAX_INTENT_JUDGE_DECISION_BYTES = 16 * 1024 * 1024
MAX_INTENT_JUDGE_FAILURE_BYTES = 16 * 1024 * 1024
MAX_INTENT_JUDGE_UNGRADED_BYTES = 16 * 1024 * 1024
MAX_INTENT_PROJECTED_CLAIMS = (
    MAX_INTENT_CLAIMS + 1 + 3 * MAX_TEXT_LIST_ITEMS
)
UNSCORABLE_INTENT_TRUTH_DIGEST = canonical_sha256(
    {
        "scorable": False,
        "authority": None,
        "expected_claims": [],
        "forbidden_claims": [],
        "clarification_policy": None,
    }
)
EXACT_INTENT_WEIGHT = 4_000_000
NORMALIZED_INTENT_WEIGHT = 3_900_000
SEMANTIC_FULL_INTENT_WEIGHT_BASE = 2_000_000
SEMANTIC_PARTIAL_INTENT_WEIGHT_BASE = 1_000_000

# Kept local so Intent hydration can fail closed without importing Judge
# execution code and creating a package cycle.  These values intentionally
# mirror JudgeFailureCode and JudgeUngradedReason.
INTENT_JUDGE_FAILURE_CODES = frozenset(
    {
        "adapter_capability_missing",
        "context_budget_exceeded",
        "deadline_exceeded",
        "provider_error",
        "timeout",
        "invalid_response",
        "invalid_output",
        "output_limit_exceeded",
        "output_truncated",
        "adapter_identity_mismatch",
        "unsafe_output",
        "attempts_exhausted",
    }
)
INTENT_JUDGE_UNGRADED_REASONS = frozenset(
    {"upstream_missing", "not_scorable", "policy_skipped"}
)


class IntentEvaluationError(ValueError):
    """The evaluator input or a Judge merge violates the v1 contract."""


class IntentJudgeRelation(str, Enum):
    EQUIVALENT = "equivalent"
    PARTIALLY_EQUIVALENT = "partially_equivalent"
    CONTRADICTED = "contradicted"
    DIFFERENT = "different"
    UNKNOWN = "unknown"


class IntentEvaluationStatus(str, Enum):
    GRADED = "graded"
    PENDING_JUDGE = "pending_judge"
    UNGRADED = "ungraded"
    NOT_SCORABLE = "not_scorable"


class IntentTruthKind(str, Enum):
    EXPECTED = "expected"
    FORBIDDEN = "forbidden"


class IntentClaimOrigin(str, Enum):
    STRUCTURED = "structured"
    PROVENANCE = "provenance"
    STRUCTURED_OVERLAY = "structured_overlay"


class IntentMatchKind(str, Enum):
    EXACT = "exact"
    NORMALIZED = "normalized"
    SEMANTIC = "semantic"


class IntentReasonCode(str, Enum):
    SUBMISSION_INTENT_MISSING = "submission_intent_missing"
    INTENT_TRUTH_UNSCORABLE = "intent_truth_unscorable"
    CANDIDATE_EDGE_LIMIT_EXCEEDED = "candidate_edge_limit_exceeded"
    JUDGE_PENDING = "judge_pending"
    JUDGE_FAILED = "judge_failed"
    JUDGE_UNGRADED = "judge_ungraded"
    JUDGE_UNKNOWN = "judge_unknown"
    JUDGE_DIFFERENT = "judge_different"
    DETERMINISTIC_EXACT = "deterministic_exact"
    DETERMINISTIC_NORMALIZED = "deterministic_normalized"
    SEMANTIC_EQUIVALENT = "semantic_equivalent"
    SEMANTIC_PARTIAL = "semantic_partial"
    SEMANTIC_CONTRADICTED = "semantic_contradicted"
    MATCHED_EXPECTED = "matched_expected"
    MATCHED_FORBIDDEN = "matched_forbidden"
    UNMATCHED_DUPLICATE = "unmatched_duplicate"
    NO_TRUTH_CANDIDATE = "no_truth_candidate"
    REQUIRED_TRUTH_MISSED = "required_truth_missed"
    OPTIONAL_TRUTH_MISSED = "optional_truth_missed"
    FORBIDDEN_TRUTH_HIT = "forbidden_truth_hit"
    REQUIRED_CLARIFICATION_NOT_ASKED = "required_clarification_not_asked"
    CLARIFICATION_WRONG_DIMENSION = "clarification_wrong_dimension"
    CLARIFICATION_WRONG_MATERIAL_CLAIM = "clarification_wrong_material_claim"
    CLARIFICATION_UNMATCHED = "clarification_unmatched"
    CLARIFICATION_AMBIGUOUS = "clarification_ambiguous"
    CLARIFICATION_ROUND_LIMIT = "clarification_round_limit"
    CLARIFICATION_ANSWER_NOT_CONSUMED = "clarification_answer_not_consumed"
    CLARIFICATION_UNNECESSARY_QUESTION = "clarification_unnecessary_question"
    CLARIFICATION_UNNECESSARY_BLOCKING = "clarification_unnecessary_blocking"
    CLARIFICATION_ANSWER_NOT_APPLIED = "clarification_answer_not_applied"
    CLARIFICATION_UNRESOLVED = "clarification_unresolved"
    CLARIFICATION_RECEIPT_MISSING = "clarification_receipt_missing"
    CLARIFICATION_RECEIPT_MISMATCH = "clarification_receipt_mismatch"
    CLARIFICATION_DECISION_CORRECT = "clarification_decision_correct"


def _error(message: str) -> IntentEvaluationError:
    return IntentEvaluationError(message)


def _id(value: Any, context: str) -> str:
    if type(value) is not str or not value or len(value) > MAX_IDENTIFIER_CHARS:
        raise _error(f"{context} must be a non-empty bounded identifier")
    if value != value.strip() or any(
        character.isspace() or ord(character) < 32 or ord(character) == 127
        for character in value
    ):
        raise _error(f"{context} must not contain whitespace or controls")
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError as exc:
        raise _error(f"{context} must contain valid Unicode") from exc
    return value


def _text(value: Any, context: str, *, allow_empty: bool = False) -> str:
    if type(value) is not str or len(value) > MAX_CLAIM_CHARS:
        raise _error(f"{context} must be a bounded string")
    if not allow_empty and not value:
        raise _error(f"{context} must be non-empty")
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError as exc:
        raise _error(f"{context} must contain valid Unicode") from exc
    return value


def _enum(enum_type: type[Enum], value: Any, context: str) -> Any:
    if not isinstance(value, enum_type):
        raise _error(f"{context} must be {enum_type.__name__}")
    return value


def _enum_value(enum_type: type[Enum], value: Any, context: str) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise _error(f"{context} has an invalid enum value") from exc


def _optional_enum(enum_type: type[Enum], value: Any, context: str) -> Any:
    if value is None:
        return None
    return _enum_value(enum_type, value, context)


def _score(value: Any, context: str) -> int:
    if type(value) is not int or value < 0 or value > MAX_INTENT_SCORE_PPM:
        raise _error(
            f"{context} must be an integer from 0 through {MAX_INTENT_SCORE_PPM}"
        )
    return value


def _optional_score(value: Any, context: str) -> Optional[int]:
    if value is None:
        return None
    return _score(value, context)


def _optional_non_negative_int(value: Any, context: str) -> Optional[int]:
    if value is None:
        return None
    if type(value) is not int or value < 0:
        raise _error(f"{context} must be a non-negative integer or null")
    return value


def _optional_bool(value: Any, context: str) -> Optional[bool]:
    if value is not None and type(value) is not bool:
        raise _error(f"{context} must be bool or null")
    return value


def _digest(value: Any, context: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _error(f"{context} must be a lowercase SHA-256 digest")
    return value


def _optional_digest(value: Any, context: str) -> Optional[str]:
    if value is None:
        return None
    return _digest(value, context)


def _reason_tuple(
    values: Iterable[IntentReasonCode], context: str
) -> Tuple[IntentReasonCode, ...]:
    result = []
    for index, value in enumerate(values):
        if index >= 4_096:
            raise _error(f"{context} exceeds its item limit")
        result.append(_enum(IntentReasonCode, value, f"{context}[{index}]"))
    return tuple(sorted(set(result), key=lambda item: item.value))


def _reason_values(value: Any, context: str) -> Tuple[IntentReasonCode, ...]:
    items = _bounded_array(value, context, 64)
    return tuple(
        _enum_value(IntentReasonCode, item, f"{context}[{index}]")
        for index, item in enumerate(items)
    )


def normalize_intent_text(value: str) -> str:
    """Apply the versioned, deliberately conservative Intent normalizer."""

    _text(value, "intent text")
    normalized = unicodedata.normalize("NFC", value)
    normalized = " ".join(normalized.split())
    return normalized.casefold()


@dataclass(frozen=True)
class GeneratedIntentClaim:
    generated_id: str
    dimension: IntentDimension
    text: str
    normalized_text: str
    source: Optional[IntentClaimSource]
    provenance_claim_id: Optional[str]
    origin: IntentClaimOrigin

    def __post_init__(self) -> None:
        _id(self.generated_id, "generated claim.generated_id")
        _enum(IntentDimension, self.dimension, "generated claim.dimension")
        _text(self.text, "generated claim.text")
        if self.normalized_text != normalize_intent_text(self.text):
            raise _error("generated claim.normalized_text does not match text")
        if self.source is not None:
            _enum(IntentClaimSource, self.source, "generated claim.source")
        if self.provenance_claim_id is not None:
            _id(self.provenance_claim_id, "generated claim.provenance_claim_id")
        _enum(IntentClaimOrigin, self.origin, "generated claim.origin")
        if self.origin is IntentClaimOrigin.STRUCTURED:
            if self.provenance_claim_id is not None or self.source is not None:
                raise _error("plain structured claim cannot have provenance metadata")
        elif self.provenance_claim_id is None or self.source is None:
            raise _error("provenance-backed claim requires claim ID and source")

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_id": self.generated_id,
            "dimension": self.dimension.value,
            "text": self.text,
            "normalized_text": self.normalized_text,
            "source": None if self.source is None else self.source.value,
            "provenance_claim_id": self.provenance_claim_id,
            "origin": self.origin.value,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "GeneratedIntentClaim":
        payload = _strict_object(
            value,
            (
                "generated_id",
                "dimension",
                "text",
                "normalized_text",
                "source",
                "provenance_claim_id",
                "origin",
            ),
            "generated claim",
        )
        return cls(
            generated_id=_id(payload["generated_id"], "generated claim.generated_id"),
            dimension=_enum_value(
                IntentDimension, payload["dimension"], "generated claim.dimension"
            ),
            text=_text(payload["text"], "generated claim.text"),
            normalized_text=_text(
                payload["normalized_text"], "generated claim.normalized_text"
            ),
            source=_optional_enum(
                IntentClaimSource, payload["source"], "generated claim.source"
            ),
            provenance_claim_id=(
                None
                if payload["provenance_claim_id"] is None
                else _id(
                    payload["provenance_claim_id"],
                    "generated claim.provenance_claim_id",
                )
            ),
            origin=_enum_value(
                IntentClaimOrigin, payload["origin"], "generated claim.origin"
            ),
        )


@dataclass(frozen=True)
class IntentTruthClaim:
    truth_id: str
    dimension: IntentDimension
    text: str
    kind: IntentTruthKind
    required: Optional[bool]

    def __post_init__(self) -> None:
        _id(self.truth_id, "truth claim.truth_id")
        _enum(IntentDimension, self.dimension, "truth claim.dimension")
        _text(self.text, "truth claim.text")
        _enum(IntentTruthKind, self.kind, "truth claim.kind")
        if self.kind is IntentTruthKind.EXPECTED:
            if type(self.required) is not bool:
                raise _error("expected truth claim.required must be bool")
        elif self.required is not None:
            raise _error("forbidden truth claim.required must be null")

    @property
    def normalized_text(self) -> str:
        return normalize_intent_text(self.text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "truth_id": self.truth_id,
            "dimension": self.dimension.value,
            "text": self.text,
            "kind": self.kind.value,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "IntentTruthClaim":
        payload = _strict_object(
            value,
            ("truth_id", "dimension", "text", "kind", "required"),
            "truth claim",
        )
        return cls(
            truth_id=_id(payload["truth_id"], "truth claim.truth_id"),
            dimension=_enum_value(
                IntentDimension, payload["dimension"], "truth claim.dimension"
            ),
            text=_text(payload["text"], "truth claim.text"),
            kind=_enum_value(IntentTruthKind, payload["kind"], "truth claim.kind"),
            required=payload["required"],
        )


@dataclass(frozen=True)
class IntentSemanticJudgeRequest:
    request_id: str
    generated_id: str
    truth_id: str
    dimension: IntentDimension
    generated_text: str
    truth_text: str
    truth_kind: IntentTruthKind

    def __post_init__(self) -> None:
        _id(self.request_id, "Judge request.request_id")
        _id(self.generated_id, "Judge request.generated_id")
        _id(self.truth_id, "Judge request.truth_id")
        _enum(IntentDimension, self.dimension, "Judge request.dimension")
        _text(self.generated_text, "Judge request.generated_text")
        _text(self.truth_text, "Judge request.truth_text")
        _enum(IntentTruthKind, self.truth_kind, "Judge request.truth_kind")

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "generated_id": self.generated_id,
            "truth_id": self.truth_id,
            "dimension": self.dimension.value,
            "generated_text": self.generated_text,
            "truth_text": self.truth_text,
            "truth_kind": self.truth_kind.value,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "IntentSemanticJudgeRequest":
        payload = _strict_object(
            value,
            (
                "request_id",
                "generated_id",
                "truth_id",
                "dimension",
                "generated_text",
                "truth_text",
                "truth_kind",
            ),
            "Judge request",
        )
        return cls(
            request_id=_id(payload["request_id"], "Judge request.request_id"),
            generated_id=_id(payload["generated_id"], "Judge request.generated_id"),
            truth_id=_id(payload["truth_id"], "Judge request.truth_id"),
            dimension=_enum_value(
                IntentDimension, payload["dimension"], "Judge request.dimension"
            ),
            generated_text=_text(payload["generated_text"], "Judge request.generated_text"),
            truth_text=_text(payload["truth_text"], "Judge request.truth_text"),
            truth_kind=_enum_value(
                IntentTruthKind, payload["truth_kind"], "Judge request.truth_kind"
            ),
        )


@dataclass(frozen=True)
class IntentSemanticJudgeDecision:
    request_id: str
    relation: IntentJudgeRelation
    score_ppm: int
    reason_refs: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _id(self.request_id, "Judge decision.request_id")
        _enum(IntentJudgeRelation, self.relation, "Judge decision.relation")
        _score(self.score_ppm, "Judge decision.score_ppm")
        refs = []
        for index, value in enumerate(self.reason_refs):
            if index >= 16:
                raise _error("Judge decision.reason_refs exceeds its item limit")
            refs.append(_id(value, "Judge decision.reason_refs item"))
        if not refs:
            raise _error("Judge decision.reason_refs must not be empty")
        if len(refs) != len(set(refs)):
            raise _error("Judge decision.reason_refs must not contain duplicates")
        object.__setattr__(self, "reason_refs", tuple(sorted(refs)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "relation": self.relation.value,
            "score_ppm": self.score_ppm,
            "reason_refs": list(self.reason_refs),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "IntentSemanticJudgeDecision":
        payload = _strict_object(
            value,
            ("request_id", "relation", "score_ppm", "reason_refs"),
            "Judge decision",
        )
        refs = _bounded_array(payload["reason_refs"], "Judge decision.reason_refs", 16)
        return cls(
            request_id=_id(payload["request_id"], "Judge decision.request_id"),
            relation=_enum_value(
                IntentJudgeRelation, payload["relation"], "Judge decision.relation"
            ),
            score_ppm=_score(payload["score_ppm"], "Judge decision.score_ppm"),
            reason_refs=tuple(
                _id(value, "Judge decision.reason_refs item") for value in refs
            ),
        )


@dataclass(frozen=True)
class IntentSemanticJudgeFailure:
    request_id: str
    failure_code: str
    evaluator_execution_digest: str
    judge_result_digest: str

    def __post_init__(self) -> None:
        _id(self.request_id, "Judge failure.request_id")
        failure_code = _id(self.failure_code, "Judge failure.failure_code")
        if failure_code not in INTENT_JUDGE_FAILURE_CODES:
            raise _error("Judge failure has unsupported failure_code")
        _digest(
            self.evaluator_execution_digest,
            "Judge failure.evaluator_execution_digest",
        )
        _digest(self.judge_result_digest, "Judge failure.judge_result_digest")

    @classmethod
    def from_dict(cls, value: Any) -> "IntentSemanticJudgeFailure":
        payload = _strict_object(
            value,
            (
                "request_id",
                "failure_code",
                "evaluator_execution_digest",
                "judge_result_digest",
            ),
            "Intent Judge failure",
        )
        return cls(
            request_id=_id(payload["request_id"], "Judge failure.request_id"),
            failure_code=_id(payload["failure_code"], "Judge failure.failure_code"),
            evaluator_execution_digest=_digest(
                payload["evaluator_execution_digest"],
                "Judge failure.evaluator_execution_digest",
            ),
            judge_result_digest=_digest(
                payload["judge_result_digest"],
                "Judge failure.judge_result_digest",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "failure_code": self.failure_code,
            "evaluator_execution_digest": self.evaluator_execution_digest,
            "judge_result_digest": self.judge_result_digest,
        }


@dataclass(frozen=True)
class IntentSemanticJudgeUngraded:
    request_id: str
    ungraded_reason: str
    evaluator_execution_digest: str
    judge_result_digest: str

    def __post_init__(self) -> None:
        _id(self.request_id, "Judge ungraded.request_id")
        ungraded_reason = _id(
            self.ungraded_reason, "Judge ungraded.ungraded_reason"
        )
        if ungraded_reason not in INTENT_JUDGE_UNGRADED_REASONS:
            raise _error("Judge ungraded has unsupported ungraded_reason")
        _digest(
            self.evaluator_execution_digest,
            "Judge ungraded.evaluator_execution_digest",
        )
        _digest(self.judge_result_digest, "Judge ungraded.judge_result_digest")

    @classmethod
    def from_dict(cls, value: Any) -> "IntentSemanticJudgeUngraded":
        payload = _strict_object(
            value,
            (
                "request_id",
                "ungraded_reason",
                "evaluator_execution_digest",
                "judge_result_digest",
            ),
            "Intent Judge ungraded",
        )
        return cls(
            request_id=_id(payload["request_id"], "Judge ungraded.request_id"),
            ungraded_reason=_id(
                payload["ungraded_reason"], "Judge ungraded.ungraded_reason"
            ),
            evaluator_execution_digest=_digest(
                payload["evaluator_execution_digest"],
                "Judge ungraded.evaluator_execution_digest",
            ),
            judge_result_digest=_digest(
                payload["judge_result_digest"],
                "Judge ungraded.judge_result_digest",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "ungraded_reason": self.ungraded_reason,
            "evaluator_execution_digest": self.evaluator_execution_digest,
            "judge_result_digest": self.judge_result_digest,
        }


def _validate_decision_budgets(
    decisions: Iterable[IntentSemanticJudgeDecision],
) -> None:
    reason_ref_bytes = 0
    decision_bytes = 2  # JSON array brackets.
    for index, decision in enumerate(decisions):
        if index:
            decision_bytes += 1  # JSON array comma.
        decision_bytes += len(canonical_json(decision.to_dict()).encode("utf-8"))
        if decision_bytes > MAX_INTENT_JUDGE_DECISION_BYTES:
            raise _error("Judge decisions exceed the case byte budget")
        reason_ref_bytes += len(
            canonical_json(list(decision.reason_refs)).encode("utf-8")
        )
        if reason_ref_bytes > MAX_INTENT_JUDGE_REASON_REF_BYTES:
            raise _error("Judge decision reason refs exceed the case byte budget")


def _validate_failure_budgets(
    failures: Iterable[IntentSemanticJudgeFailure],
) -> None:
    if _records_exceed_byte_budget(
        failures,
        maximum=MAX_INTENT_JUDGE_FAILURE_BYTES,
    ):
        raise _error("Judge failures exceed the case byte budget")


def _validate_ungraded_budgets(
    receipts: Iterable[IntentSemanticJudgeUngraded],
) -> None:
    if _records_exceed_byte_budget(
        receipts,
        maximum=MAX_INTENT_JUDGE_UNGRADED_BYTES,
    ):
        raise _error("Judge ungraded receipts exceed the case byte budget")


def _records_exceed_byte_budget(
    records: Iterable[Any],
    *,
    maximum: int,
) -> bool:
    """Return whether canonical JSON array encoding exceeds ``maximum`` bytes."""

    total = 2  # JSON array brackets.
    for index, record in enumerate(records):
        if index:
            total += 1  # JSON array comma.
        payload = record.to_dict() if hasattr(record, "to_dict") else record
        total += len(canonical_json(payload).encode("utf-8"))
        if total > maximum:
            return True
    return False


def _validate_request_budgets(
    requests: Iterable[IntentSemanticJudgeRequest],
) -> None:
    request_items = tuple(requests)
    if _records_exceed_byte_budget(
        request_items,
        maximum=MAX_INTENT_JUDGE_REQUEST_BYTES,
    ):
        raise _error("Judge requests exceed the case byte budget")
    request_text_bytes = sum(
        len(request.generated_text.encode("utf-8"))
        + len(request.truth_text.encode("utf-8"))
        for request in request_items
    )
    if request_text_bytes > MAX_INTENT_REQUEST_TEXT_BYTES:
        raise _error("Judge request texts exceed the case byte budget")


@dataclass(frozen=True)
class IntentCandidateRecord:
    generated_id: str
    truth_id: str
    truth_kind: IntentTruthKind
    match_kind: Optional[IntentMatchKind]
    request_id: Optional[str]
    relation: Optional[IntentJudgeRelation]
    score_ppm: Optional[int]
    edge_weight: Optional[int]
    selected: bool
    judgement: Optional[IntentClaimJudgement]
    reason_codes: Tuple[IntentReasonCode, ...]

    def __post_init__(self) -> None:
        _id(self.generated_id, "candidate.generated_id")
        _id(self.truth_id, "candidate.truth_id")
        _enum(IntentTruthKind, self.truth_kind, "candidate.truth_kind")
        _optional_enum(IntentMatchKind, self.match_kind, "candidate.match_kind")
        if self.request_id is not None:
            _id(self.request_id, "candidate.request_id")
        _optional_enum(IntentJudgeRelation, self.relation, "candidate.relation")
        _optional_score(self.score_ppm, "candidate.score_ppm")
        if self.edge_weight is not None and (
            type(self.edge_weight) is not int or self.edge_weight < 1
        ):
            raise _error("candidate.edge_weight must be a positive integer")
        if type(self.selected) is not bool:
            raise _error("candidate.selected must be bool")
        _optional_enum(IntentClaimJudgement, self.judgement, "candidate.judgement")
        object.__setattr__(
            self, "reason_codes", _reason_tuple(self.reason_codes, "candidate.reason_codes")
        )
        if self.match_kind is None:
            raise _error("candidate.match_kind must be present")
        if self.match_kind is IntentMatchKind.SEMANTIC and self.request_id is None:
            raise _error("semantic candidate requires request_id")
        if self.match_kind is not IntentMatchKind.SEMANTIC and self.request_id is not None:
            raise _error("deterministic candidate cannot have request_id")
        if self.match_kind is not IntentMatchKind.SEMANTIC:
            if (
                self.relation is not IntentJudgeRelation.EQUIVALENT
                or self.score_ppm is not None
                or self.judgement is None
                or self.edge_weight is None
            ):
                raise _error("deterministic candidate has invalid match metadata")
        elif self.relation is None:
            if any(
                value is not None
                for value in (self.score_ppm, self.judgement, self.edge_weight)
            ):
                raise _error("pending semantic candidate must not contain a decision")
        elif self.score_ppm is None:
            raise _error("decided semantic candidate requires score_ppm")
        if self.selected and (self.edge_weight is None or self.judgement is None):
            raise _error("selected candidate requires an eligible judged edge")

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_id": self.generated_id,
            "truth_id": self.truth_id,
            "truth_kind": self.truth_kind.value,
            "match_kind": None if self.match_kind is None else self.match_kind.value,
            "request_id": self.request_id,
            "relation": None if self.relation is None else self.relation.value,
            "score_ppm": self.score_ppm,
            "edge_weight": self.edge_weight,
            "selected": self.selected,
            "judgement": None if self.judgement is None else self.judgement.value,
            "reason_codes": [item.value for item in self.reason_codes],
        }

    @classmethod
    def from_dict(cls, value: Any) -> "IntentCandidateRecord":
        payload = _strict_object(
            value,
            (
                "generated_id",
                "truth_id",
                "truth_kind",
                "match_kind",
                "request_id",
                "relation",
                "score_ppm",
                "edge_weight",
                "selected",
                "judgement",
                "reason_codes",
            ),
            "Intent candidate",
        )
        selected = payload["selected"]
        if type(selected) is not bool:
            raise _error("Intent candidate.selected must be bool")
        return cls(
            generated_id=_id(payload["generated_id"], "Intent candidate.generated_id"),
            truth_id=_id(payload["truth_id"], "Intent candidate.truth_id"),
            truth_kind=_enum_value(
                IntentTruthKind, payload["truth_kind"], "Intent candidate.truth_kind"
            ),
            match_kind=_optional_enum(
                IntentMatchKind, payload["match_kind"], "Intent candidate.match_kind"
            ),
            request_id=(
                None
                if payload["request_id"] is None
                else _id(payload["request_id"], "Intent candidate.request_id")
            ),
            relation=_optional_enum(
                IntentJudgeRelation, payload["relation"], "Intent candidate.relation"
            ),
            score_ppm=_optional_score(
                payload["score_ppm"], "Intent candidate.score_ppm"
            ),
            edge_weight=_optional_non_negative_int(
                payload["edge_weight"], "Intent candidate.edge_weight"
            ),
            selected=selected,
            judgement=_optional_enum(
                IntentClaimJudgement,
                payload["judgement"],
                "Intent candidate.judgement",
            ),
            reason_codes=_reason_values(
                payload["reason_codes"], "Intent candidate.reason_codes"
            ),
        )


@dataclass(frozen=True)
class IntentClaimOutcome:
    generated_id: str
    judgement: IntentClaimJudgement
    matched_truth_id: Optional[str]
    matched_truth_kind: Optional[IntentTruthKind]
    match_kind: Optional[IntentMatchKind]
    reason_codes: Tuple[IntentReasonCode, ...]

    def __post_init__(self) -> None:
        _id(self.generated_id, "claim outcome.generated_id")
        _enum(IntentClaimJudgement, self.judgement, "claim outcome.judgement")
        if self.matched_truth_id is not None:
            _id(self.matched_truth_id, "claim outcome.matched_truth_id")
            if self.matched_truth_kind is None or self.match_kind is None:
                raise _error("matched claim outcome requires truth kind and match kind")
        elif self.matched_truth_kind is not None or self.match_kind is not None:
            raise _error("unmatched claim outcome cannot have match metadata")
        _reason_tuple(self.reason_codes, "claim outcome.reason_codes")
        object.__setattr__(
            self, "reason_codes", _reason_tuple(self.reason_codes, "claim outcome.reason_codes")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "generated_id": self.generated_id,
            "judgement": self.judgement.value,
            "matched_truth_id": self.matched_truth_id,
            "matched_truth_kind": (
                None if self.matched_truth_kind is None else self.matched_truth_kind.value
            ),
            "match_kind": None if self.match_kind is None else self.match_kind.value,
            "reason_codes": [item.value for item in self.reason_codes],
        }

    @classmethod
    def from_dict(cls, value: Any) -> "IntentClaimOutcome":
        payload = _strict_object(
            value,
            (
                "generated_id",
                "judgement",
                "matched_truth_id",
                "matched_truth_kind",
                "match_kind",
                "reason_codes",
            ),
            "Intent claim outcome",
        )
        return cls(
            generated_id=_id(
                payload["generated_id"], "Intent claim outcome.generated_id"
            ),
            judgement=_enum_value(
                IntentClaimJudgement,
                payload["judgement"],
                "Intent claim outcome.judgement",
            ),
            matched_truth_id=(
                None
                if payload["matched_truth_id"] is None
                else _id(
                    payload["matched_truth_id"],
                    "Intent claim outcome.matched_truth_id",
                )
            ),
            matched_truth_kind=_optional_enum(
                IntentTruthKind,
                payload["matched_truth_kind"],
                "Intent claim outcome.matched_truth_kind",
            ),
            match_kind=_optional_enum(
                IntentMatchKind,
                payload["match_kind"],
                "Intent claim outcome.match_kind",
            ),
            reason_codes=_reason_values(
                payload["reason_codes"], "Intent claim outcome.reason_codes"
            ),
        )


@dataclass(frozen=True)
class ClarificationExchangeEvaluation:
    turn_index: int
    question_id: str
    matched_answer_id: Optional[str]
    receipt_digest: Optional[str]
    matcher_digest: Optional[str]
    material: Optional[bool]
    answer_consumed: Optional[bool]
    update_applied: Optional[bool]
    reason_codes: Tuple[IntentReasonCode, ...]

    def __post_init__(self) -> None:
        if (
            type(self.turn_index) is not int
            or not 1 <= self.turn_index <= MAX_CLARIFICATION_QUESTIONS
        ):
            raise _error("clarification evaluation.turn_index is out of bounds")
        _id(self.question_id, "clarification evaluation.question_id")
        if self.matched_answer_id is not None:
            _id(self.matched_answer_id, "clarification evaluation.matched_answer_id")
        if self.receipt_digest is not None:
            _digest(self.receipt_digest, "clarification evaluation.receipt_digest")
        if self.matcher_digest is not None:
            _digest(self.matcher_digest, "clarification evaluation.matcher_digest")
        if (self.receipt_digest is None) != (self.matcher_digest is None):
            raise _error("clarification evaluation receipt binding is incomplete")
        for value, context in (
            (self.material, "clarification evaluation.material"),
            (self.answer_consumed, "clarification evaluation.answer_consumed"),
            (self.update_applied, "clarification evaluation.update_applied"),
        ):
            if value is not None and type(value) is not bool:
                raise _error(f"{context} must be bool or null")
        if self.receipt_digest is None:
            if any(
                value is not None
                for value in (
                    self.material,
                    self.answer_consumed,
                    self.update_applied,
                )
            ):
                raise _error(
                    "clarification without a receipt cannot contain inferred grading"
                )
        else:
            if self.material is None or self.answer_consumed is None:
                raise _error(
                    "receipt-bound clarification requires materiality and consumption"
                )
            if self.answer_consumed and not self.material:
                raise _error("a non-material clarification cannot consume an answer")
            if self.update_applied is not None and not self.answer_consumed:
                raise _error("an unconsumed answer cannot update Intent")
        object.__setattr__(
            self,
            "reason_codes",
            _reason_tuple(
                self.reason_codes, "clarification evaluation.reason_codes"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "question_id": self.question_id,
            "matched_answer_id": self.matched_answer_id,
            "receipt_digest": self.receipt_digest,
            "matcher_digest": self.matcher_digest,
            "material": self.material,
            "answer_consumed": self.answer_consumed,
            "update_applied": self.update_applied,
            "reason_codes": [item.value for item in self.reason_codes],
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ClarificationExchangeEvaluation":
        payload = _strict_object(
            value,
            (
                "turn_index",
                "question_id",
                "matched_answer_id",
                "receipt_digest",
                "matcher_digest",
                "material",
                "answer_consumed",
                "update_applied",
                "reason_codes",
            ),
            "clarification exchange evaluation",
        )
        turn_index = payload["turn_index"]
        if type(turn_index) is not int:
            raise _error("clarification exchange evaluation.turn_index must be int")
        return cls(
            turn_index=turn_index,
            question_id=_id(
                payload["question_id"],
                "clarification exchange evaluation.question_id",
            ),
            matched_answer_id=(
                None
                if payload["matched_answer_id"] is None
                else _id(
                    payload["matched_answer_id"],
                    "clarification exchange evaluation.matched_answer_id",
                )
            ),
            receipt_digest=(
                None
                if payload["receipt_digest"] is None
                else _digest(
                    payload["receipt_digest"],
                    "clarification exchange evaluation.receipt_digest",
                )
            ),
            matcher_digest=(
                None
                if payload["matcher_digest"] is None
                else _digest(
                    payload["matcher_digest"],
                    "clarification exchange evaluation.matcher_digest",
                )
            ),
            material=_optional_bool(
                payload["material"], "clarification exchange evaluation.material"
            ),
            answer_consumed=_optional_bool(
                payload["answer_consumed"],
                "clarification exchange evaluation.answer_consumed",
            ),
            update_applied=_optional_bool(
                payload["update_applied"],
                "clarification exchange evaluation.update_applied",
            ),
            reason_codes=_reason_values(
                payload["reason_codes"],
                "clarification exchange evaluation.reason_codes",
            ),
        )


@dataclass(frozen=True)
class ClarificationEvaluation:
    policy: Optional[ClarificationPolicy]
    decision_correct: Optional[bool]
    complete: Optional[bool]
    exchanges: Tuple[ClarificationExchangeEvaluation, ...]
    reason_codes: Tuple[IntentReasonCode, ...]

    def __post_init__(self) -> None:
        _optional_enum(
            ClarificationPolicy, self.policy, "clarification evaluation.policy"
        )
        for value, context in (
            (self.decision_correct, "clarification evaluation.decision_correct"),
            (self.complete, "clarification evaluation.complete"),
        ):
            if value is not None and type(value) is not bool:
                raise _error(f"{context} must be bool or null")
        exchanges = tuple(self.exchanges)
        for index, exchange in enumerate(exchanges, start=1):
            if type(exchange) is not ClarificationExchangeEvaluation:
                raise _error("clarification evaluation.exchanges contains invalid item")
            if exchange.turn_index != index:
                raise _error("clarification evaluation turns must be contiguous")
        object.__setattr__(self, "exchanges", exchanges)
        object.__setattr__(
            self,
            "reason_codes",
            _reason_tuple(
                self.reason_codes, "clarification evaluation.reason_codes"
            ),
        )
        missing_submission = (
            IntentReasonCode.SUBMISSION_INTENT_MISSING in self.reason_codes
        )
        if self.policy is None:
            if self.decision_correct is not None or self.complete is not None:
                raise _error("unscorable clarification cannot contain a decision")
        elif missing_submission:
            if self.decision_correct is not None or self.complete is not None:
                raise _error("missing Intent clarification must remain ungraded")
        else:
            if self.complete is None:
                raise _error("scorable clarification requires completeness")
            proven_material = any(
                item.material is True and item.answer_consumed is True
                for item in exchanges
            )
            unknown_material = any(item.material is None for item in exchanges)
            expected_decision = (
                (
                    True
                    if proven_material
                    else (None if unknown_material else False)
                )
                if self.policy is ClarificationPolicy.REQUIRED
                else (
                    True
                    if self.policy is ClarificationPolicy.OPTIONAL
                    else not exchanges
                )
            )
            if self.decision_correct is not expected_decision:
                raise _error("clarification decision does not match its policy")
            if self.complete and (
                self.decision_correct is None
                or unknown_material
                or any(item.answer_consumed is False for item in exchanges)
                or any(item.update_applied is False for item in exchanges)
            ):
                raise _error("complete clarification contains unresolved sub-results")
        matcher_digests = {
            item.matcher_digest
            for item in exchanges
            if item.matcher_digest is not None
        }
        if len(matcher_digests) > 1:
            raise _error("clarification exchanges use different matcher bindings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": None if self.policy is None else self.policy.value,
            "decision_correct": self.decision_correct,
            "complete": self.complete,
            "exchanges": [item.to_dict() for item in self.exchanges],
            "reason_codes": [item.value for item in self.reason_codes],
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ClarificationEvaluation":
        payload = _strict_object(
            value,
            ("policy", "decision_correct", "complete", "exchanges", "reason_codes"),
            "clarification evaluation",
        )
        exchanges = _bounded_array(
            payload["exchanges"],
            "clarification evaluation.exchanges",
            MAX_CLARIFICATION_QUESTIONS,
        )
        return cls(
            policy=_optional_enum(
                ClarificationPolicy,
                payload["policy"],
                "clarification evaluation.policy",
            ),
            decision_correct=_optional_bool(
                payload["decision_correct"],
                "clarification evaluation.decision_correct",
            ),
            complete=_optional_bool(
                payload["complete"], "clarification evaluation.complete"
            ),
            exchanges=tuple(
                ClarificationExchangeEvaluation.from_dict(item) for item in exchanges
            ),
            reason_codes=_reason_values(
                payload["reason_codes"], "clarification evaluation.reason_codes"
            ),
        )


@dataclass(frozen=True)
class IntentMetricInputs:
    scorable: bool
    generated_claim_count: Optional[int]
    supported_claim_count: Optional[int]
    partially_supported_claim_count: Optional[int]
    unsupported_claim_count: Optional[int]
    contradicted_claim_count: Optional[int]
    unknown_claim_count: Optional[int]
    required_truth_count: Optional[int]
    required_supported_count: Optional[int]
    required_partially_supported_count: Optional[int]
    required_missed_count: Optional[int]
    optional_truth_count: Optional[int]
    optional_supported_count: Optional[int]
    forbidden_truth_count: Optional[int]
    forbidden_hit_count: Optional[int]
    clarification_numerator: Optional[int]
    clarification_denominator: Optional[int]
    intent_case_pass: Optional[bool]

    def __post_init__(self) -> None:
        if type(self.scorable) is not bool:
            raise _error("metric inputs.scorable must be bool")
        fields = (
            "generated_claim_count",
            "supported_claim_count",
            "partially_supported_claim_count",
            "unsupported_claim_count",
            "contradicted_claim_count",
            "unknown_claim_count",
            "required_truth_count",
            "required_supported_count",
            "required_partially_supported_count",
            "required_missed_count",
            "optional_truth_count",
            "optional_supported_count",
            "forbidden_truth_count",
            "forbidden_hit_count",
            "clarification_numerator",
            "clarification_denominator",
        )
        for name in fields:
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise _error(f"metric inputs.{name} must be a non-negative integer or null")
        if self.intent_case_pass is not None and type(self.intent_case_pass) is not bool:
            raise _error("metric inputs.intent_case_pass must be bool or null")
        if not self.scorable and any(getattr(self, name) is not None for name in fields):
            raise _error("unscorable metric inputs must use null denominators and counts")
        if not self.scorable and self.intent_case_pass is not None:
            raise _error("unscorable metric inputs must have null case pass")
        required_fields = fields[:-2]
        if self.scorable and any(
            getattr(self, name) is None for name in required_fields
        ):
            raise _error("scorable metric inputs require all claim/truth counts")
        if (self.clarification_numerator is None) != (
            self.clarification_denominator is None
        ):
            raise _error("clarification metric numerator/denominator coverage differs")
        if self.clarification_denominator is not None and (
            self.clarification_denominator != 1
            or self.clarification_numerator not in (0, 1)
        ):
            raise _error("clarification metric must be one bounded decision")
        if self.scorable:
            required_truth_count = cast(int, self.required_truth_count)
            required_supported_count = cast(int, self.required_supported_count)
            required_partial_count = cast(
                int, self.required_partially_supported_count
            )
            required_missed_count = cast(int, self.required_missed_count)
            optional_truth_count = cast(int, self.optional_truth_count)
            optional_supported_count = cast(int, self.optional_supported_count)
            forbidden_truth_count = cast(int, self.forbidden_truth_count)
            forbidden_hit_count = cast(int, self.forbidden_hit_count)
            if required_supported_count + required_partial_count > required_truth_count:
                raise _error("required truth metric counts exceed denominator")
            if required_missed_count != required_truth_count - required_supported_count:
                raise _error("required missed count is inconsistent")
            if optional_supported_count > optional_truth_count:
                raise _error("optional supported count exceeds denominator")
            if forbidden_hit_count > forbidden_truth_count:
                raise _error("forbidden hit count exceeds denominator")

    @property
    def intent_claim_precision_numerator(self) -> Optional[int]:
        return self.supported_claim_count

    @property
    def intent_claim_precision_denominator(self) -> Optional[int]:
        return self.generated_claim_count

    @property
    def intent_claim_recall_numerator(self) -> Optional[int]:
        return self.required_supported_count

    @property
    def intent_claim_recall_denominator(self) -> Optional[int]:
        return self.required_truth_count

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in (
                "scorable",
                "generated_claim_count",
                "supported_claim_count",
                "partially_supported_claim_count",
                "unsupported_claim_count",
                "contradicted_claim_count",
                "unknown_claim_count",
                "required_truth_count",
                "required_supported_count",
                "required_partially_supported_count",
                "required_missed_count",
                "optional_truth_count",
                "optional_supported_count",
                "forbidden_truth_count",
                "forbidden_hit_count",
                "clarification_numerator",
                "clarification_denominator",
                "intent_case_pass",
            )
        }

    @classmethod
    def from_dict(cls, value: Any) -> "IntentMetricInputs":
        names = (
            "scorable",
            "generated_claim_count",
            "supported_claim_count",
            "partially_supported_claim_count",
            "unsupported_claim_count",
            "contradicted_claim_count",
            "unknown_claim_count",
            "required_truth_count",
            "required_supported_count",
            "required_partially_supported_count",
            "required_missed_count",
            "optional_truth_count",
            "optional_supported_count",
            "forbidden_truth_count",
            "forbidden_hit_count",
            "clarification_numerator",
            "clarification_denominator",
            "intent_case_pass",
        )
        payload = _strict_object(value, names, "Intent metric inputs")
        if type(payload["scorable"]) is not bool:
            raise _error("Intent metric inputs.scorable must be bool")
        values = {
            name: _optional_non_negative_int(
                payload[name], f"Intent metric inputs.{name}"
            )
            for name in names[1:-1]
        }
        return cls(
            scorable=payload["scorable"],
            **values,
            intent_case_pass=_optional_bool(
                payload["intent_case_pass"],
                "Intent metric inputs.intent_case_pass",
            ),
        )


@dataclass(frozen=True)
class IntentEvaluationResult:
    schema_version: str
    evaluator_revision: str
    submission_intent_digest: Optional[str]
    intent_truth_digest: str
    clarification_script_digest: str
    policy_version: str
    normalization_version: str
    status: IntentEvaluationStatus
    generated_claims: Tuple[GeneratedIntentClaim, ...]
    truth_claims: Tuple[IntentTruthClaim, ...]
    candidates: Tuple[IntentCandidateRecord, ...]
    assignments: Tuple[AssignedPair, ...]
    claim_outcomes: Tuple[IntentClaimOutcome, ...]
    unmatched_generated_ids: Tuple[str, ...]
    unmatched_expected_truth_ids: Tuple[str, ...]
    unmatched_forbidden_truth_ids: Tuple[str, ...]
    judge_requests: Tuple[IntentSemanticJudgeRequest, ...]
    judge_decisions: Tuple[IntentSemanticJudgeDecision, ...]
    judge_failures: Tuple[IntentSemanticJudgeFailure, ...]
    judge_ungraded: Tuple[IntentSemanticJudgeUngraded, ...]
    clarification: ClarificationEvaluation
    metrics: IntentMetricInputs
    reason_codes: Tuple[IntentReasonCode, ...]

    def __post_init__(self) -> None:
        if self.schema_version != INTENT_EVALUATION_SCHEMA_VERSION:
            raise _error("unsupported Intent evaluation schema version")
        _id(self.evaluator_revision, "Intent evaluation.evaluator_revision")
        _optional_digest(
            self.submission_intent_digest,
            "Intent evaluation.submission_intent_digest",
        )
        _digest(self.intent_truth_digest, "Intent evaluation.intent_truth_digest")
        _digest(
            self.clarification_script_digest,
            "Intent evaluation.clarification_script_digest",
        )
        if self.policy_version != ASSIGNMENT_POLICY_VERSION:
            raise _error("unsupported Intent assignment policy version")
        if self.normalization_version != INTENT_NORMALIZATION_POLICY_VERSION:
            raise _error("unsupported Intent normalization version")
        _enum(IntentEvaluationStatus, self.status, "Intent evaluation.status")
        for name, expected_type, maximum, sort_key in (
            (
                "generated_claims",
                GeneratedIntentClaim,
                MAX_INTENT_PROJECTED_CLAIMS,
                lambda item: item.generated_id,
            ),
            (
                "truth_claims",
                IntentTruthClaim,
                MAX_INTENT_CLAIMS,
                lambda item: (item.dimension.value, item.truth_id),
            ),
            (
                "candidates",
                IntentCandidateRecord,
                MAX_INTENT_TOTAL_CANDIDATES,
                lambda item: (item.generated_id, item.truth_id),
            ),
            (
                "assignments",
                AssignedPair,
                MAX_INTENT_CLAIMS,
                lambda item: (item.left_id, item.right_id),
            ),
            (
                "claim_outcomes",
                IntentClaimOutcome,
                MAX_INTENT_PROJECTED_CLAIMS,
                lambda item: item.generated_id,
            ),
            (
                "judge_requests",
                IntentSemanticJudgeRequest,
                MAX_INTENT_CANDIDATE_EDGES,
                lambda item: item.request_id,
            ),
            (
                "judge_decisions",
                IntentSemanticJudgeDecision,
                MAX_INTENT_CANDIDATE_EDGES,
                lambda item: item.request_id,
            ),
            (
                "judge_failures",
                IntentSemanticJudgeFailure,
                MAX_INTENT_CANDIDATE_EDGES,
                lambda item: item.request_id,
            ),
            (
                "judge_ungraded",
                IntentSemanticJudgeUngraded,
                MAX_INTENT_CANDIDATE_EDGES,
                lambda item: item.request_id,
            ),
        ):
            values = tuple(getattr(self, name))
            if len(values) > maximum:
                raise _error(f"Intent evaluation.{name} exceeds its item limit")
            if any(type(item) is not expected_type for item in values):
                raise _error(f"Intent evaluation.{name} contains an invalid item")
            object.__setattr__(self, name, tuple(sorted(values, key=sort_key)))
        if _records_exceed_byte_budget(
            self.candidates,
            maximum=MAX_INTENT_CANDIDATE_RECORD_BYTES,
        ):
            raise _error("Intent candidate records exceed the case byte budget")
        _validate_request_budgets(self.judge_requests)
        _validate_decision_budgets(self.judge_decisions)
        _validate_failure_budgets(self.judge_failures)
        _validate_ungraded_budgets(self.judge_ungraded)
        for name in (
            "unmatched_generated_ids",
            "unmatched_expected_truth_ids",
            "unmatched_forbidden_truth_ids",
        ):
            values = tuple(
                _id(value, f"Intent evaluation.{name} item")
                for value in getattr(self, name)
            )
            if values != tuple(sorted(values)) or len(values) != len(set(values)):
                raise _error(f"Intent evaluation.{name} must be unique and sorted")
            object.__setattr__(self, name, values)
        if type(self.clarification) is not ClarificationEvaluation:
            raise _error("Intent evaluation.clarification has an invalid type")
        if type(self.metrics) is not IntentMetricInputs:
            raise _error("Intent evaluation.metrics has an invalid type")
        object.__setattr__(
            self, "reason_codes", _reason_tuple(self.reason_codes, "Intent evaluation.reason_codes")
        )
        self._validate_cross_references()

    def _validate_cross_references(self) -> None:
        generated = {item.generated_id: item for item in self.generated_claims}
        truth = {item.truth_id: item for item in self.truth_claims}
        clarification_marks_missing = (
            IntentReasonCode.SUBMISSION_INTENT_MISSING
            in self.clarification.reason_codes
        )
        result_marks_missing = (
            IntentReasonCode.SUBMISSION_INTENT_MISSING in self.reason_codes
        )
        if clarification_marks_missing != result_marks_missing:
            raise _error("Submission Intent coverage differs across result sections")
        if self.submission_intent_digest is None:
            if (
                self.generated_claims
                or self.status is not IntentEvaluationStatus.UNGRADED
                or IntentReasonCode.SUBMISSION_INTENT_MISSING
                not in self.reason_codes
            ):
                raise _error("missing Submission Intent binding is inconsistent")
        elif IntentReasonCode.SUBMISSION_INTENT_MISSING in self.reason_codes:
            raise _error("present Submission Intent is marked missing")
        if len(generated) != len(self.generated_claims):
            raise _error("Intent evaluation contains duplicate generated IDs")
        if len(truth) != len(self.truth_claims):
            raise _error("Intent evaluation contains duplicate truth IDs")

        structural_groups: dict[tuple[str, str, str], list[GeneratedIntentClaim]] = {}
        for item in self.generated_claims:
            if item.origin in {
                IntentClaimOrigin.STRUCTURED,
                IntentClaimOrigin.STRUCTURED_OVERLAY,
            }:
                structural_groups.setdefault(
                    (item.dimension.value, item.normalized_text, item.text), []
                ).append(item)
        for (dimension, _normalized, text), items in structural_groups.items():
            expected_ids = {
                stable_id("intent-generated-structured-v1", dimension, text, index)
                for index in range(len(items))
            }
            if {item.generated_id for item in items} != expected_ids:
                raise _error("structured generated claim ID is not canonical")
        for item in self.generated_claims:
            if item.origin is IntentClaimOrigin.PROVENANCE:
                if item.provenance_claim_id is None:
                    raise _error("provenance generated claim has no claim ID")
                expected_id = stable_id(
                    "intent-generated-provenance-v1",
                    item.provenance_claim_id,
                    item.dimension.value,
                    item.text,
                )
                if item.generated_id != expected_id:
                    raise _error("provenance generated claim ID is not canonical")
        provenance_ids = [
            item.provenance_claim_id
            for item in self.generated_claims
            if item.provenance_claim_id is not None
        ]
        if len(provenance_ids) != len(set(provenance_ids)):
            raise _error("a provenance claim is attached to multiple generated claims")

        candidate_map = {
            (item.generated_id, item.truth_id): item for item in self.candidates
        }
        if len(candidate_map) != len(self.candidates):
            raise _error("Intent evaluation contains duplicate candidate pairs")
        generated_by_dimension = Counter(
            item.dimension for item in self.generated_claims
        )
        truth_by_dimension = Counter(item.dimension for item in self.truth_claims)
        expected_candidate_count = sum(
            generated_by_dimension[dimension] * truth_by_dimension[dimension]
            for dimension in IntentDimension
        )
        if self.metrics.scorable and len(candidate_map) != expected_candidate_count:
            raise _error(
                "Intent candidates do not cover every canonical same-dimension pair"
            )
        if (
            not self.metrics.scorable
            and IntentReasonCode.CANDIDATE_EDGE_LIMIT_EXCEEDED in self.reason_codes
        ):
            _seeds, _requests, limit_exceeded = IntentEvaluator._candidate_seeds(
                self.generated_claims,
                self.truth_claims,
            )
            if not limit_exceeded:
                raise _error(
                    "candidate-limit outcome is not the canonical result for its claim graph"
                )
        request_map = {item.request_id: item for item in self.judge_requests}
        decision_map = {item.request_id: item for item in self.judge_decisions}
        failure_map = {item.request_id: item for item in self.judge_failures}
        ungraded_map = {item.request_id: item for item in self.judge_ungraded}
        if len(request_map) != len(self.judge_requests):
            raise _error("Intent evaluation contains duplicate Judge request IDs")
        if len(decision_map) != len(self.judge_decisions):
            raise _error("Intent evaluation contains duplicate Judge decision IDs")
        if len(failure_map) != len(self.judge_failures):
            raise _error("Intent evaluation contains duplicate Judge failure IDs")
        if len(ungraded_map) != len(self.judge_ungraded):
            raise _error("Intent evaluation contains duplicate Judge ungraded IDs")
        if not set(decision_map).issubset(request_map):
            raise _error("Intent evaluation contains a decision for an unknown request")
        if not set(failure_map).issubset(request_map):
            raise _error("Intent evaluation contains a failure for an unknown request")
        if not set(ungraded_map).issubset(request_map):
            raise _error("Intent evaluation contains ungraded receipt for an unknown request")
        resolution_ids = (
            list(decision_map) + list(failure_map) + list(ungraded_map)
        )
        if len(resolution_ids) != len(set(resolution_ids)):
            raise _error("a Judge request cannot have more than one resolution")
        provenance_receipts = (*self.judge_failures, *self.judge_ungraded)
        execution_digests = {
            item.evaluator_execution_digest for item in provenance_receipts
        }
        if len(execution_digests) > 1:
            raise _error("Judge receipts bind multiple evaluator executions")
        result_digests = [item.judge_result_digest for item in provenance_receipts]
        if len(result_digests) != len(set(result_digests)):
            raise _error("Judge result digest cannot resolve multiple requests")

        for request in self.judge_requests:
            generated_item = generated.get(request.generated_id)
            truth_item = truth.get(request.truth_id)
            if generated_item is None or truth_item is None:
                raise _error("Judge request references an unknown claim")
            if (
                request.dimension is not generated_item.dimension
                or request.dimension is not truth_item.dimension
                or request.generated_text != generated_item.text
                or request.truth_text != truth_item.text
                or request.truth_kind is not truth_item.kind
                or request.request_id != _request_id(generated_item, truth_item)
            ):
                raise _error("Judge request does not match its canonical claim pair")
            decision = decision_map.get(request.request_id)
            if decision is not None and not set(decision.reason_refs).issubset(
                {request.generated_id, request.truth_id}
            ):
                raise _error(
                    "Judge decision reason refs cross the canonical request boundary"
                )

        for candidate in self.candidates:
            generated_item = generated.get(candidate.generated_id)
            truth_item = truth.get(candidate.truth_id)
            if generated_item is None or truth_item is None:
                raise _error("candidate references an unknown claim")
            if generated_item.dimension is not truth_item.dimension:
                raise _error("candidate crosses Intent dimensions")
            if candidate.truth_kind is not truth_item.kind:
                raise _error("candidate truth kind does not match truth claim")
            if candidate.match_kind is IntentMatchKind.EXACT:
                if generated_item.text != truth_item.text:
                    raise _error("exact candidate texts are not identical")
                expected_relation = IntentJudgeRelation.EQUIVALENT
                expected_judgement = _relation_judgement(
                    truth_item.kind, expected_relation
                )
                if (
                    candidate.relation is not expected_relation
                    or candidate.score_ppm is not None
                    or candidate.judgement is not expected_judgement
                    or candidate.edge_weight != EXACT_INTENT_WEIGHT
                ):
                    raise _error("exact candidate metadata is inconsistent")
                expected_reasons = (IntentReasonCode.DETERMINISTIC_EXACT,)
            elif candidate.match_kind is IntentMatchKind.NORMALIZED:
                if (
                    generated_item.text == truth_item.text
                    or generated_item.normalized_text != truth_item.normalized_text
                ):
                    raise _error("normalized candidate texts are inconsistent")
                expected_relation = IntentJudgeRelation.EQUIVALENT
                expected_judgement = _relation_judgement(
                    truth_item.kind, expected_relation
                )
                if (
                    candidate.relation is not expected_relation
                    or candidate.score_ppm is not None
                    or candidate.judgement is not expected_judgement
                    or candidate.edge_weight != NORMALIZED_INTENT_WEIGHT
                ):
                    raise _error("normalized candidate metadata is inconsistent")
                expected_reasons = (IntentReasonCode.DETERMINISTIC_NORMALIZED,)
            else:
                if generated_item.normalized_text == truth_item.normalized_text:
                    raise _error("semantic candidate bypasses deterministic matching")
                request = request_map.get(candidate.request_id or "")
                if request is None or (
                    request.generated_id != candidate.generated_id
                    or request.truth_id != candidate.truth_id
                ):
                    raise _error("semantic candidate does not bind its Judge request")
                decision = decision_map.get(request.request_id)
                failure = failure_map.get(request.request_id)
                ungraded = ungraded_map.get(request.request_id)
                if decision is None:
                    if any(
                        value is not None
                        for value in (
                            candidate.relation,
                            candidate.score_ppm,
                            candidate.judgement,
                            candidate.edge_weight,
                        )
                    ):
                        raise _error("unresolved candidate contains fabricated Judge data")
                    expected_reasons = (
                        (IntentReasonCode.JUDGE_FAILED,)
                        if failure is not None
                        else (
                            (IntentReasonCode.JUDGE_UNGRADED,)
                            if ungraded is not None
                            else (IntentReasonCode.JUDGE_PENDING,)
                        )
                    )
                else:
                    expected_judgement = _relation_judgement(
                        truth_item.kind, decision.relation
                    )
                    expected_weight = (
                        None
                        if expected_judgement is None
                        else _edge_weight(
                            IntentMatchKind.SEMANTIC,
                            decision.relation,
                            decision.score_ppm,
                        )
                    )
                    if (
                        candidate.relation is not decision.relation
                        or candidate.score_ppm != decision.score_ppm
                        or candidate.judgement is not expected_judgement
                        or candidate.edge_weight != expected_weight
                    ):
                        raise _error("semantic candidate does not match Judge decision")
                    expected_reasons = (_semantic_reason(decision.relation),)
            if candidate.reason_codes != expected_reasons:
                raise _error("candidate reason codes are inconsistent")

        semantic_request_ids = {
            item.request_id
            for item in self.candidates
            if item.match_kind is IntentMatchKind.SEMANTIC
        }
        if semantic_request_ids != set(request_map):
            raise _error("Judge requests must cover semantic candidates exactly")

        expected_assignment = maximum_weight_bipartite_assignment(
            tuple(generated),
            tuple(truth),
            (
                WeightedAssignmentEdge(
                    candidate.generated_id,
                    candidate.truth_id,
                    candidate.edge_weight,
                )
                for candidate in self.candidates
                if candidate.edge_weight is not None
            ),
            edge_limit=MAX_INTENT_TOTAL_CANDIDATES,
        )
        if self.assignments != expected_assignment.matches:
            raise _error(
                "assignments are not the canonical global maximum-weight result"
            )

        assignments = {(item.left_id, item.right_id): item for item in self.assignments}
        if len(assignments) != len(self.assignments):
            raise _error("Intent evaluation contains duplicate assignments")
        if len({item.left_id for item in self.assignments}) != len(self.assignments):
            raise _error("generated claim appears in more than one assignment")
        if len({item.right_id for item in self.assignments}) != len(self.assignments):
            raise _error("truth claim appears in more than one assignment")
        for pair, assignment in assignments.items():
            candidate = candidate_map.get(pair)
            if (
                candidate is None
                or not candidate.selected
                or candidate.edge_weight != assignment.weight
                or candidate.judgement is None
            ):
                raise _error("assignment does not match one selected candidate edge")
        if {pair for pair, item in candidate_map.items() if item.selected} != set(assignments):
            raise _error("selected candidate set does not equal assignments")

        outcomes = {item.generated_id: item for item in self.claim_outcomes}
        if len(outcomes) != len(self.claim_outcomes) or set(outcomes) != set(generated):
            raise _error("claim outcomes must cover generated claims exactly once")
        assigned_by_generated = {item.left_id: item for item in self.assignments}
        candidates_by_generated: dict[str, list[IntentCandidateRecord]] = {}
        for candidate in self.candidates:
            candidates_by_generated.setdefault(candidate.generated_id, []).append(
                candidate
            )
        for generated_id, outcome in outcomes.items():
            assignment = assigned_by_generated.get(generated_id)
            if assignment is None:
                if outcome.matched_truth_id is not None:
                    raise _error("unassigned claim outcome contains match metadata")
                if not self.metrics.scorable:
                    if (
                        outcome.judgement is not IntentClaimJudgement.UNKNOWN
                        or outcome.reason_codes != self.reason_codes
                    ):
                        raise _error("ungraded claim outcome is inconsistent")
                    continue
                relevant = candidates_by_generated.get(generated_id, [])
                if any(
                    item.relation is IntentJudgeRelation.UNKNOWN
                    for item in relevant
                ):
                    expected_judgement = IntentClaimJudgement.UNKNOWN
                    expected_reasons = (IntentReasonCode.JUDGE_UNKNOWN,)
                elif any(
                    IntentReasonCode.JUDGE_FAILED in item.reason_codes
                    for item in relevant
                ):
                    expected_judgement = IntentClaimJudgement.UNKNOWN
                    expected_reasons = (IntentReasonCode.JUDGE_FAILED,)
                elif any(
                    IntentReasonCode.JUDGE_UNGRADED in item.reason_codes
                    for item in relevant
                ):
                    expected_judgement = IntentClaimJudgement.UNKNOWN
                    expected_reasons = (IntentReasonCode.JUDGE_UNGRADED,)
                elif any(
                    item.match_kind is IntentMatchKind.SEMANTIC
                    and item.relation is None
                    for item in relevant
                ):
                    expected_judgement = IntentClaimJudgement.UNKNOWN
                    expected_reasons = (IntentReasonCode.JUDGE_PENDING,)
                elif not relevant:
                    expected_judgement = IntentClaimJudgement.UNSUPPORTED
                    expected_reasons = (IntentReasonCode.NO_TRUTH_CANDIDATE,)
                elif any(item.edge_weight is not None for item in relevant):
                    expected_judgement = IntentClaimJudgement.UNSUPPORTED
                    expected_reasons = (IntentReasonCode.UNMATCHED_DUPLICATE,)
                elif all(
                    item.relation is IntentJudgeRelation.DIFFERENT
                    for item in relevant
                ):
                    expected_judgement = IntentClaimJudgement.UNSUPPORTED
                    expected_reasons = (IntentReasonCode.JUDGE_DIFFERENT,)
                else:
                    expected_judgement = IntentClaimJudgement.UNSUPPORTED
                    expected_reasons = tuple(
                        sorted(
                            {
                                reason
                                for item in relevant
                                for reason in item.reason_codes
                            },
                            key=lambda reason: reason.value,
                        )
                    ) or (IntentReasonCode.NO_TRUTH_CANDIDATE,)
                if (
                    outcome.judgement is not expected_judgement
                    or outcome.reason_codes != expected_reasons
                ):
                    raise _error("unmatched claim outcome is inconsistent")
                continue
            candidate = candidate_map[(assignment.left_id, assignment.right_id)]
            expected_outcome_reasons = _reason_tuple(
                candidate.reason_codes
                + (
                    IntentReasonCode.MATCHED_FORBIDDEN
                    if candidate.truth_kind is IntentTruthKind.FORBIDDEN
                    else IntentReasonCode.MATCHED_EXPECTED,
                ),
                "derived matched claim outcome reasons",
            )
            if (
                outcome.matched_truth_id != assignment.right_id
                or outcome.matched_truth_kind is not candidate.truth_kind
                or outcome.match_kind is not candidate.match_kind
                or outcome.judgement is not candidate.judgement
                or outcome.reason_codes != expected_outcome_reasons
            ):
                raise _error("claim outcome does not match its selected candidate")

        expected_unmatched_generated = tuple(sorted(set(generated) - set(assigned_by_generated)))
        if self.unmatched_generated_ids != expected_unmatched_generated:
            raise _error("unmatched generated IDs are inconsistent with assignments")
        assigned_truth = {item.right_id for item in self.assignments}
        expected_unmatched_expected = tuple(
            sorted(
                item.truth_id
                for item in self.truth_claims
                if item.kind is IntentTruthKind.EXPECTED
                and item.truth_id not in assigned_truth
            )
        )
        expected_unmatched_forbidden = tuple(
            sorted(
                item.truth_id
                for item in self.truth_claims
                if item.kind is IntentTruthKind.FORBIDDEN
                and item.truth_id not in assigned_truth
            )
        )
        if self.unmatched_expected_truth_ids != expected_unmatched_expected:
            raise _error("unmatched expected truth IDs are inconsistent")
        if self.unmatched_forbidden_truth_ids != expected_unmatched_forbidden:
            raise _error("unmatched forbidden truth IDs are inconsistent")

        if self.metrics.scorable:
            counts = (
                self.metrics.supported_claim_count,
                self.metrics.partially_supported_claim_count,
                self.metrics.unsupported_claim_count,
                self.metrics.contradicted_claim_count,
                self.metrics.unknown_claim_count,
            )
            if any(value is None for value in counts) or sum(counts) != len(outcomes):
                raise _error("Intent metric claim counts are inconsistent")
            actual_counts = Counter(item.judgement for item in self.claim_outcomes)
            expected_counts = (
                actual_counts[IntentClaimJudgement.SUPPORTED],
                actual_counts[IntentClaimJudgement.PARTIALLY_SUPPORTED],
                actual_counts[IntentClaimJudgement.UNSUPPORTED],
                actual_counts[IntentClaimJudgement.CONTRADICTED],
                actual_counts[IntentClaimJudgement.UNKNOWN],
            )
            if counts != expected_counts:
                raise _error("Intent metric claim classifications were tampered")
            required_truth = [
                item
                for item in self.truth_claims
                if item.kind is IntentTruthKind.EXPECTED and item.required is True
            ]
            optional_truth = [
                item
                for item in self.truth_claims
                if item.kind is IntentTruthKind.EXPECTED and item.required is False
            ]
            matched_outcomes = {
                item.matched_truth_id: item
                for item in self.claim_outcomes
                if item.matched_truth_id is not None
            }
            required_supported = sum(
                item.truth_id in matched_outcomes
                and matched_outcomes[item.truth_id].judgement
                is IntentClaimJudgement.SUPPORTED
                for item in required_truth
            )
            required_partial = sum(
                item.truth_id in matched_outcomes
                and matched_outcomes[item.truth_id].judgement
                is IntentClaimJudgement.PARTIALLY_SUPPORTED
                for item in required_truth
            )
            optional_supported = sum(
                item.truth_id in matched_outcomes
                and matched_outcomes[item.truth_id].judgement
                is IntentClaimJudgement.SUPPORTED
                for item in optional_truth
            )
            forbidden_hits = sum(
                item.matched_truth_kind is IntentTruthKind.FORBIDDEN
                for item in self.claim_outcomes
            )
            if (
                self.metrics.generated_claim_count != len(self.claim_outcomes)
                or self.metrics.required_truth_count != len(required_truth)
                or self.metrics.required_supported_count != required_supported
                or self.metrics.required_partially_supported_count
                != required_partial
                or self.metrics.required_missed_count
                != len(required_truth) - required_supported
                or self.metrics.optional_truth_count != len(optional_truth)
                or self.metrics.optional_supported_count != optional_supported
                or self.metrics.forbidden_truth_count
                != sum(
                    item.kind is IntentTruthKind.FORBIDDEN
                    for item in self.truth_claims
                )
                or self.metrics.forbidden_hit_count != forbidden_hits
            ):
                raise _error("Intent truth metric inputs are inconsistent")
            expected_clarification_numerator = (
                None
                if self.clarification.decision_correct is None
                else int(self.clarification.decision_correct)
            )
            if (
                self.metrics.clarification_numerator
                != expected_clarification_numerator
                or self.metrics.clarification_denominator
                != (
                    None
                    if expected_clarification_numerator is None
                    else 1
                )
            ):
                raise _error("clarification metric inputs are inconsistent")
            expected_case_pass = (
                None
                if self.status is not IntentEvaluationStatus.GRADED
                or actual_counts[IntentClaimJudgement.UNKNOWN]
                else (
                    required_supported == len(required_truth)
                    and actual_counts[
                        IntentClaimJudgement.PARTIALLY_SUPPORTED
                    ]
                    == 0
                    and actual_counts[IntentClaimJudgement.UNSUPPORTED] == 0
                    and actual_counts[IntentClaimJudgement.CONTRADICTED] == 0
                    and forbidden_hits == 0
                    and self.clarification.decision_correct is True
                    and self.clarification.complete is True
                )
            )
            if self.metrics.intent_case_pass is not expected_case_pass:
                raise _error("Intent case pass is inconsistent")
        elif self.status is IntentEvaluationStatus.GRADED:
            raise _error("graded Intent evaluation must have scorable metric inputs")

        null_metric_reasons = {
            IntentReasonCode.SUBMISSION_INTENT_MISSING,
            IntentReasonCode.INTENT_TRUTH_UNSCORABLE,
            IntentReasonCode.CANDIDATE_EDGE_LIMIT_EXCEEDED,
        }
        expected_metric_coverage = not bool(
            null_metric_reasons.intersection(self.reason_codes)
        )
        if self.metrics.scorable is not expected_metric_coverage:
            raise _error("Intent metric coverage does not match evaluation reason")
        if not self.metrics.scorable and (
            self.candidates
            or self.assignments
            or self.judge_requests
            or self.judge_decisions
            or self.judge_failures
            or self.judge_ungraded
            or any(
                item.judgement is not IntentClaimJudgement.UNKNOWN
                for item in self.claim_outcomes
            )
        ):
            raise _error("null-coverage Intent evaluation contains graded matching data")

        pending_request_ids = (
            set(request_map)
            - set(decision_map)
            - set(failure_map)
            - set(ungraded_map)
        )
        if bool(pending_request_ids) != (
            self.status is IntentEvaluationStatus.PENDING_JUDGE
        ):
            raise _error("Intent evaluation status does not match pending Judge work")
        if self.metrics.scorable:
            expected_status = (
                IntentEvaluationStatus.PENDING_JUDGE
                if pending_request_ids
                else (
                    IntentEvaluationStatus.UNGRADED
                    if failure_map
                    or ungraded_map
                    or any(
                        decision.relation is IntentJudgeRelation.UNKNOWN
                        for decision in self.judge_decisions
                    )
                    or IntentReasonCode.CLARIFICATION_RECEIPT_MISSING
                    in self.clarification.reason_codes
                    else IntentEvaluationStatus.GRADED
                )
            )
            if self.status is not expected_status:
                raise _error("Intent evaluation status is not canonical")
        if (
            IntentReasonCode.INTENT_TRUTH_UNSCORABLE in self.reason_codes
            and self.status is not IntentEvaluationStatus.NOT_SCORABLE
        ):
            raise _error("unscorable Intent truth has the wrong evaluation status")
        if IntentReasonCode.INTENT_TRUTH_UNSCORABLE in self.reason_codes and (
            self.truth_claims or self.clarification.policy is not None
        ):
            raise _error(
                "unscorable Intent result must have empty truth claims and null clarification policy"
            )
        if (
            IntentReasonCode.INTENT_TRUTH_UNSCORABLE in self.reason_codes
            and self.intent_truth_digest != UNSCORABLE_INTENT_TRUTH_DIGEST
        ):
            raise _error(
                "unscorable Intent result is not bound to canonical unscorable truth"
            )
        if (
            self.status is IntentEvaluationStatus.NOT_SCORABLE
            and IntentReasonCode.INTENT_TRUTH_UNSCORABLE not in self.reason_codes
        ):
            raise _error("not-scorable status lacks unscorable Intent truth")
        if (
            IntentReasonCode.SUBMISSION_INTENT_MISSING in self.reason_codes
            or IntentReasonCode.CANDIDATE_EDGE_LIMIT_EXCEEDED in self.reason_codes
        ) and self.status is not IntentEvaluationStatus.UNGRADED:
            raise _error("ungraded Harness outcome has the wrong evaluation status")
        if self.status is IntentEvaluationStatus.GRADED and {
            IntentReasonCode.JUDGE_PENDING,
            IntentReasonCode.JUDGE_FAILED,
            IntentReasonCode.JUDGE_UNGRADED,
            IntentReasonCode.JUDGE_UNKNOWN,
            IntentReasonCode.CLARIFICATION_RECEIPT_MISSING,
        }.intersection(self.reason_codes):
            raise _error("graded Intent evaluation contains unresolved work")
        if self.status is IntentEvaluationStatus.GRADED and any(
            decision.relation is IntentJudgeRelation.UNKNOWN
            for decision in self.judge_decisions
        ):
            raise _error("graded Intent evaluation contains unknown Judge decisions")

        if self.metrics.scorable:
            expected_result_reasons = list(self.clarification.reason_codes)
            if pending_request_ids:
                expected_result_reasons.append(IntentReasonCode.JUDGE_PENDING)
            if failure_map:
                expected_result_reasons.append(IntentReasonCode.JUDGE_FAILED)
            if ungraded_map:
                expected_result_reasons.append(IntentReasonCode.JUDGE_UNGRADED)
            if any(
                decision.relation is IntentJudgeRelation.UNKNOWN
                for decision in self.judge_decisions
            ):
                expected_result_reasons.append(IntentReasonCode.JUDGE_UNKNOWN)
            if self.metrics.required_missed_count:
                expected_result_reasons.append(
                    IntentReasonCode.REQUIRED_TRUTH_MISSED
                )
            optional_truth_ids = {
                item.truth_id
                for item in self.truth_claims
                if item.kind is IntentTruthKind.EXPECTED
                and item.required is False
            }
            if optional_truth_ids.intersection(
                self.unmatched_expected_truth_ids
            ):
                expected_result_reasons.append(
                    IntentReasonCode.OPTIONAL_TRUTH_MISSED
                )
            if self.metrics.forbidden_hit_count:
                expected_result_reasons.append(IntentReasonCode.FORBIDDEN_TRUTH_HIT)
        elif IntentReasonCode.SUBMISSION_INTENT_MISSING in self.reason_codes:
            expected_result_reasons = [
                IntentReasonCode.SUBMISSION_INTENT_MISSING
            ]
        elif IntentReasonCode.INTENT_TRUTH_UNSCORABLE in self.reason_codes:
            expected_result_reasons = [IntentReasonCode.INTENT_TRUTH_UNSCORABLE]
        else:
            expected_result_reasons = [
                IntentReasonCode.CANDIDATE_EDGE_LIMIT_EXCEEDED
            ]
        if self.reason_codes != _reason_tuple(
            expected_result_reasons, "derived Intent evaluation reasons"
        ):
            raise _error("Intent evaluation reason codes are inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluator_revision": self.evaluator_revision,
            "submission_intent_digest": self.submission_intent_digest,
            "intent_truth_digest": self.intent_truth_digest,
            "clarification_script_digest": self.clarification_script_digest,
            "policy_version": self.policy_version,
            "normalization_version": self.normalization_version,
            "status": self.status.value,
            "generated_claims": [item.to_dict() for item in self.generated_claims],
            "truth_claims": [item.to_dict() for item in self.truth_claims],
            "candidates": [item.to_dict() for item in self.candidates],
            "assignments": [
                {
                    "left_id": item.left_id,
                    "right_id": item.right_id,
                    "weight": item.weight,
                }
                for item in self.assignments
            ],
            "claim_outcomes": [item.to_dict() for item in self.claim_outcomes],
            "unmatched_generated_ids": list(self.unmatched_generated_ids),
            "unmatched_expected_truth_ids": list(self.unmatched_expected_truth_ids),
            "unmatched_forbidden_truth_ids": list(self.unmatched_forbidden_truth_ids),
            "judge_requests": [item.to_dict() for item in self.judge_requests],
            "judge_decisions": [item.to_dict() for item in self.judge_decisions],
            "judge_failures": [item.to_dict() for item in self.judge_failures],
            "judge_ungraded": [item.to_dict() for item in self.judge_ungraded],
            "clarification": self.clarification.to_dict(),
            "metrics": self.metrics.to_dict(),
            "reason_codes": [item.value for item in self.reason_codes],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def digest(self) -> str:
        return canonical_sha256(self.to_dict())

    @classmethod
    def _parse_unbound(cls, value: Any) -> dict[str, Any]:
        """Validate an untrusted payload without hydrating it as authoritative."""

        fields = tuple(field.name for field in dataclass_fields(cls))
        payload = _strict_object(value, fields, "Intent evaluation")
        generated = _bounded_array(
            payload["generated_claims"],
            "Intent evaluation.generated_claims",
            MAX_INTENT_PROJECTED_CLAIMS,
        )
        truth = _bounded_array(
            payload["truth_claims"], "Intent evaluation.truth_claims", MAX_INTENT_CLAIMS
        )
        candidates = _bounded_array(
            payload["candidates"], "Intent evaluation.candidates", MAX_INTENT_TOTAL_CANDIDATES
        )
        assignments = _bounded_array(
            payload["assignments"], "Intent evaluation.assignments", MAX_INTENT_CLAIMS
        )
        outcomes = _bounded_array(
            payload["claim_outcomes"],
            "Intent evaluation.claim_outcomes",
            MAX_INTENT_PROJECTED_CLAIMS,
        )
        requests = _bounded_array(
            payload["judge_requests"],
            "Intent evaluation.judge_requests",
            MAX_INTENT_CANDIDATE_EDGES,
        )
        decisions = _bounded_array(
            payload["judge_decisions"],
            "Intent evaluation.judge_decisions",
            MAX_INTENT_CANDIDATE_EDGES,
        )
        failures = _bounded_array(
            payload["judge_failures"],
            "Intent evaluation.judge_failures",
            MAX_INTENT_CANDIDATE_EDGES,
        )
        ungraded = _bounded_array(
            payload["judge_ungraded"],
            "Intent evaluation.judge_ungraded",
            MAX_INTENT_CANDIDATE_EDGES,
        )
        for name, maximum in (
            ("unmatched_generated_ids", MAX_INTENT_PROJECTED_CLAIMS),
            ("unmatched_expected_truth_ids", MAX_INTENT_CLAIMS),
            ("unmatched_forbidden_truth_ids", MAX_INTENT_CLAIMS),
            ("reason_codes", 128),
        ):
            _bounded_array(payload[name], f"Intent evaluation.{name}", maximum)
        try:
            _json_tree(payload, "Intent evaluation")
            if len(canonical_json_bytes(payload)) > MAX_INTENT_EVALUATION_BYTES:
                raise _error("Intent evaluation exceeds its canonical byte budget")
        except ValueError as exc:
            raise _error(str(exc)) from exc

        assignment_items = []
        for index, item in enumerate(assignments):
            assignment = _strict_object(
                item, ("left_id", "right_id", "weight"), f"Intent evaluation.assignments[{index}]"
            )
            assignment_weight = _optional_non_negative_int(
                assignment["weight"], "assignment.weight"
            )
            if assignment_weight is None or assignment_weight < 1:
                raise _error("assignment.weight must be a positive integer")
            assignment_items.append(
                AssignedPair(
                    left_id=_id(assignment["left_id"], "assignment.left_id"),
                    right_id=_id(assignment["right_id"], "assignment.right_id"),
                    weight=assignment_weight,
                )
            )
        cls(
            schema_version=_text(payload["schema_version"], "Intent evaluation.schema_version"),
            evaluator_revision=_id(
                payload["evaluator_revision"], "Intent evaluation.evaluator_revision"
            ),
            submission_intent_digest=_optional_digest(
                payload["submission_intent_digest"],
                "Intent evaluation.submission_intent_digest",
            ),
            intent_truth_digest=_digest(
                payload["intent_truth_digest"],
                "Intent evaluation.intent_truth_digest",
            ),
            clarification_script_digest=_digest(
                payload["clarification_script_digest"],
                "Intent evaluation.clarification_script_digest",
            ),
            policy_version=_text(payload["policy_version"], "Intent evaluation.policy_version"),
            normalization_version=_text(
                payload["normalization_version"], "Intent evaluation.normalization_version"
            ),
            status=_enum_value(
                IntentEvaluationStatus, payload["status"], "Intent evaluation.status"
            ),
            generated_claims=tuple(GeneratedIntentClaim.from_dict(item) for item in generated),
            truth_claims=tuple(IntentTruthClaim.from_dict(item) for item in truth),
            candidates=tuple(IntentCandidateRecord.from_dict(item) for item in candidates),
            assignments=tuple(assignment_items),
            claim_outcomes=tuple(IntentClaimOutcome.from_dict(item) for item in outcomes),
            unmatched_generated_ids=tuple(
                _id(item, "Intent evaluation.unmatched_generated_ids item")
                for item in _bounded_array(
                    payload["unmatched_generated_ids"],
                    "Intent evaluation.unmatched_generated_ids",
                    MAX_INTENT_PROJECTED_CLAIMS,
                )
            ),
            unmatched_expected_truth_ids=tuple(
                _id(item, "Intent evaluation.unmatched_expected_truth_ids item")
                for item in _bounded_array(
                    payload["unmatched_expected_truth_ids"],
                    "Intent evaluation.unmatched_expected_truth_ids",
                    MAX_INTENT_CLAIMS,
                )
            ),
            unmatched_forbidden_truth_ids=tuple(
                _id(item, "Intent evaluation.unmatched_forbidden_truth_ids item")
                for item in _bounded_array(
                    payload["unmatched_forbidden_truth_ids"],
                    "Intent evaluation.unmatched_forbidden_truth_ids",
                    MAX_INTENT_CLAIMS,
                )
            ),
            judge_requests=tuple(
                IntentSemanticJudgeRequest.from_dict(item) for item in requests
            ),
            judge_decisions=tuple(
                IntentSemanticJudgeDecision.from_dict(item) for item in decisions
            ),
            judge_failures=tuple(
                IntentSemanticJudgeFailure.from_dict(item) for item in failures
            ),
            judge_ungraded=tuple(
                IntentSemanticJudgeUngraded.from_dict(item) for item in ungraded
            ),
            clarification=ClarificationEvaluation.from_dict(payload["clarification"]),
            metrics=IntentMetricInputs.from_dict(payload["metrics"]),
            reason_codes=_reason_values(
                payload["reason_codes"], "Intent evaluation.reason_codes"
            ),
        )
        return payload

    @classmethod
    def from_dict(
        cls,
        value: Any,
        *,
        evaluator: "IntentEvaluator",
        submission_intent: Optional[SubmissionIntent],
        intent_truth: IntentTruth,
        clarification_script: ClarificationScript,
        clarification_match_receipts: Sequence[MaterialClaimMatchReceipt],
        semantic_decisions: Sequence[IntentSemanticJudgeDecision],
        semantic_failures: Sequence[IntentSemanticJudgeFailure],
        semantic_ungraded: Sequence[IntentSemanticJudgeUngraded],
    ) -> "IntentEvaluationResult":
        """Hydrate only after a complete source-bound deterministic replay."""

        if type(evaluator) is not IntentEvaluator:
            raise _error("Intent hydration requires the real IntentEvaluator")
        parsed = cls._parse_unbound(value)
        replayed = evaluator.evaluate(
            submission_intent,
            intent_truth,
            clarification_script,
            receipts=clarification_match_receipts,
            semantic_decisions=semantic_decisions,
            semantic_failures=semantic_failures,
            semantic_ungraded=semantic_ungraded,
        )
        if canonical_json_bytes(parsed) != canonical_json_bytes(replayed.to_dict()):
            raise _error("persisted Intent evaluation differs from deterministic replay")
        return replayed

    @classmethod
    def from_json(
        cls,
        data: Any,
        *,
        evaluator: "IntentEvaluator",
        submission_intent: Optional[SubmissionIntent],
        intent_truth: IntentTruth,
        clarification_script: ClarificationScript,
        clarification_match_receipts: Sequence[MaterialClaimMatchReceipt],
        semantic_decisions: Sequence[IntentSemanticJudgeDecision],
        semantic_failures: Sequence[IntentSemanticJudgeFailure],
        semantic_ungraded: Sequence[IntentSemanticJudgeUngraded],
    ) -> "IntentEvaluationResult":
        try:
            parsed = _strict_json_loads(
                data, MAX_INTENT_EVALUATION_BYTES, "Intent evaluation JSON"
            )
        except ValueError as exc:
            message = str(exc)
            if "duplicate object key" in message:
                message = message.replace(
                    "JSON contains duplicate object key", "duplicate JSON key"
                )
            raise IntentEvaluationError(message) from exc
        return cls.from_dict(
            parsed,
            evaluator=evaluator,
            submission_intent=submission_intent,
            intent_truth=intent_truth,
            clarification_script=clarification_script,
            clarification_match_receipts=clarification_match_receipts,
            semantic_decisions=semantic_decisions,
            semantic_failures=semantic_failures,
            semantic_ungraded=semantic_ungraded,
        )

    serialize = to_dict
    hydrate = from_dict


def _strict_object(value: Any, fields: Sequence[str], context: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != set(fields) or len(value) != len(fields):
        raise _error(f"{context} has unknown or missing fields")
    return value


def _bounded_array(value: Any, context: str, maximum: int) -> list[Any]:
    if type(value) is not list or len(value) > maximum:
        raise _error(f"{context} must be a bounded list")
    return value


def _project_generated_claims(intent: SubmissionIntent) -> Tuple[GeneratedIntentClaim, ...]:
    structural: list[tuple[IntentDimension, str, str]] = []
    if intent.goal is not None:
        structural.append(
            (
                IntentDimension.GOAL,
                intent.goal,
                normalize_intent_text(intent.goal),
            )
        )
    structural.extend(
        (
            IntentDimension.ACCEPTANCE_CRITERION,
            value,
            normalize_intent_text(value),
        )
        for value in intent.acceptance_criteria
    )
    structural.extend(
        (IntentDimension.SCOPE, value, normalize_intent_text(value))
        for value in intent.scope
    )
    structural.extend(
        (IntentDimension.CONSTRAINT, value, normalize_intent_text(value))
        for value in intent.constraints
    )
    structural.sort(key=lambda item: (item[0].value, item[2], item[1]))

    provenance = sorted(intent.claims, key=lambda item: item.claim_id)
    provenance_normalized: dict[str, str] = {}
    overlay_queues = defaultdict(deque)
    for claim in provenance:
        normalized = normalize_intent_text(claim.text)
        provenance_normalized[claim.claim_id] = normalized
        overlay_queues[(claim.dimension, normalized)].append(claim)
    consumed: set[str] = set()
    occurrence: Counter[tuple[str, str, str]] = Counter()
    result: list[GeneratedIntentClaim] = []
    for dimension, text, normalized in structural:
        key = (dimension.value, normalized, text)
        index = occurrence[key]
        occurrence[key] += 1
        # A structural unit keeps the same semantic identity whether or not a
        # provenance claim overlays it.  The opaque Agent claim ID is audit
        # metadata, never the identity of evaluator-generated semantics.
        generated_id = stable_id(
            "intent-generated-structured-v1", dimension.value, text, index
        )
        queue = overlay_queues[(dimension, normalized)]
        overlay = queue.popleft() if queue else None
        if overlay is not None:
            consumed.add(overlay.claim_id)
            origin = IntentClaimOrigin.STRUCTURED_OVERLAY
            source = overlay.source
            provenance_id = overlay.claim_id
        else:
            origin = IntentClaimOrigin.STRUCTURED
            source = None
            provenance_id = None
        result.append(
            GeneratedIntentClaim(
                generated_id=generated_id,
                dimension=dimension,
                text=text,
                normalized_text=normalized,
                source=source,
                provenance_claim_id=provenance_id,
                origin=origin,
            )
        )

    for claim in provenance:
        if claim.claim_id in consumed:
            continue
        result.append(
            GeneratedIntentClaim(
                generated_id=stable_id(
                    "intent-generated-provenance-v1",
                    claim.claim_id,
                    claim.dimension.value,
                    claim.text,
                ),
                dimension=claim.dimension,
                text=claim.text,
                normalized_text=provenance_normalized[claim.claim_id],
                source=claim.source,
                provenance_claim_id=claim.claim_id,
                origin=IntentClaimOrigin.PROVENANCE,
            )
        )
    return tuple(sorted(result, key=lambda item: item.generated_id))


def _truth_claims(truth: IntentTruth) -> Tuple[IntentTruthClaim, ...]:
    result = [
        IntentTruthClaim(
            truth_id=item.truth_id,
            dimension=item.dimension,
            text=item.text,
            kind=IntentTruthKind.EXPECTED,
            required=item.required,
        )
        for item in truth.expected_claims
    ]
    result.extend(
        IntentTruthClaim(
            truth_id=item.truth_id,
            dimension=item.dimension,
            text=item.text,
            kind=IntentTruthKind.FORBIDDEN,
            required=None,
        )
        for item in truth.forbidden_claims
    )
    return tuple(sorted(result, key=lambda item: (item.dimension.value, item.truth_id)))


def _request_id(generated: GeneratedIntentClaim, truth: IntentTruthClaim) -> str:
    return stable_id(
        "intent-semantic-request-v1",
        generated.generated_id,
        truth.truth_id,
        generated.dimension.value,
        generated.text,
        truth.text,
        truth.kind.value,
    )


def _material_receipt_digest(receipt: MaterialClaimMatchReceipt) -> str:
    return canonical_sha256(
        {
            "turn_index": receipt.turn_index,
            "question_id": receipt.question_id,
            "dimension": receipt.dimension.value,
            "actual_claim_digest": receipt.actual_claim_digest,
            "matcher_digest": receipt.matcher_digest,
            "candidates": [
                {
                    "answer_id": item.answer_id,
                    "request_digest": item.request_digest,
                    "equivalent": item.equivalent,
                    "action_eligible": item.action_eligible,
                }
                for item in receipt.candidates
            ],
            "outcome": receipt.outcome.value,
            "matched_answer_id": receipt.matched_answer_id,
        }
    )


def _relation_judgement(
    truth_kind: IntentTruthKind, relation: IntentJudgeRelation
) -> Optional[IntentClaimJudgement]:
    if truth_kind is IntentTruthKind.EXPECTED:
        return {
            IntentJudgeRelation.EQUIVALENT: IntentClaimJudgement.SUPPORTED,
            IntentJudgeRelation.PARTIALLY_EQUIVALENT: IntentClaimJudgement.PARTIALLY_SUPPORTED,
            IntentJudgeRelation.CONTRADICTED: IntentClaimJudgement.CONTRADICTED,
        }.get(relation)
    return {
        IntentJudgeRelation.EQUIVALENT: IntentClaimJudgement.CONTRADICTED,
        IntentJudgeRelation.PARTIALLY_EQUIVALENT: IntentClaimJudgement.CONTRADICTED,
    }.get(relation)


def _edge_weight(
    match_kind: IntentMatchKind,
    relation: Optional[IntentJudgeRelation],
    score: int,
) -> int:
    if match_kind is IntentMatchKind.EXACT:
        return EXACT_INTENT_WEIGHT
    if match_kind is IntentMatchKind.NORMALIZED:
        return NORMALIZED_INTENT_WEIGHT
    if relation is IntentJudgeRelation.PARTIALLY_EQUIVALENT:
        return SEMANTIC_PARTIAL_INTENT_WEIGHT_BASE + score
    return SEMANTIC_FULL_INTENT_WEIGHT_BASE + score


def _semantic_reason(relation: IntentJudgeRelation) -> IntentReasonCode:
    return {
        IntentJudgeRelation.UNKNOWN: IntentReasonCode.JUDGE_UNKNOWN,
        IntentJudgeRelation.DIFFERENT: IntentReasonCode.JUDGE_DIFFERENT,
        IntentJudgeRelation.PARTIALLY_EQUIVALENT: IntentReasonCode.SEMANTIC_PARTIAL,
        IntentJudgeRelation.CONTRADICTED: IntentReasonCode.SEMANTIC_CONTRADICTED,
        IntentJudgeRelation.EQUIVALENT: IntentReasonCode.SEMANTIC_EQUIVALENT,
    }[relation]


@dataclass(frozen=True)
class _CandidateSpec:
    generated: GeneratedIntentClaim
    truth: IntentTruthClaim
    match_kind: IntentMatchKind
    request: Optional[IntentSemanticJudgeRequest]
    relation: Optional[IntentJudgeRelation]
    score_ppm: Optional[int]
    judgement: Optional[IntentClaimJudgement]
    edge_weight: Optional[int]
    reason_codes: Tuple[IntentReasonCode, ...]


def _candidate_record(
    spec: _CandidateSpec,
    *,
    selected: bool,
) -> IntentCandidateRecord:
    return IntentCandidateRecord(
        generated_id=spec.generated.generated_id,
        truth_id=spec.truth.truth_id,
        truth_kind=spec.truth.kind,
        match_kind=spec.match_kind,
        request_id=None if spec.request is None else spec.request.request_id,
        relation=spec.relation,
        score_ppm=spec.score_ppm,
        edge_weight=spec.edge_weight,
        selected=selected,
        judgement=spec.judgement,
        reason_codes=spec.reason_codes,
    )


class IntentEvaluator:
    """Evaluate one immutable Intent against evaluator-side truth."""

    def __init__(self, *, evaluator_revision: str = INTENT_EVALUATOR_REVISION) -> None:
        self.evaluator_revision = _id(evaluator_revision, "evaluator_revision")

    def evaluate(
        self,
        intent: Optional[SubmissionIntent],
        truth: IntentTruth,
        clarification_script: ClarificationScript,
        *,
        transcript: Optional[Sequence[SubmissionClarificationExchange]] = None,
        receipts: Sequence[MaterialClaimMatchReceipt] = (),
        semantic_decisions: Sequence[IntentSemanticJudgeDecision] = (),
        semantic_failures: Sequence[IntentSemanticJudgeFailure] = (),
        semantic_ungraded: Sequence[IntentSemanticJudgeUngraded] = (),
    ) -> IntentEvaluationResult:
        if intent is not None and type(intent) is not SubmissionIntent:
            raise _error("intent must be SubmissionIntent or null")
        if type(truth) is not IntentTruth:
            raise _error("truth must be IntentTruth")
        if type(clarification_script) is not ClarificationScript:
            raise _error("clarification_script must be ClarificationScript")

        submission_intent_digest = (
            None if intent is None else canonical_sha256(intent.to_dict())
        )
        intent_truth_digest = canonical_sha256(truth.to_dict())
        clarification_script_digest = canonical_sha256(
            clarification_script.to_dict()
        )

        transcript_items = tuple(
            ()
            if transcript is None and intent is None
            else (
                intent.clarification_questions
                if transcript is None and intent is not None
                else transcript or ()
            )
        )
        if any(
            type(item) is not SubmissionClarificationExchange
            for item in transcript_items
        ):
            raise _error("transcript contains an invalid clarification exchange")
        if intent is None and transcript_items:
            raise _error("a missing SubmissionIntent cannot have a transcript")
        if (
            intent is not None
            and transcript is not None
            and transcript_items != intent.clarification_questions
        ):
            raise _error(
                "transcript must equal SubmissionIntent clarification transcript"
            )

        projected = () if intent is None else _project_generated_claims(intent)
        truth_items = _truth_claims(truth)

        if intent is None:
            if receipts:
                raise _error("clarification receipts cannot exist without a transcript")
            if semantic_decisions:
                raise _error("Judge decisions cannot exist without an Intent")
            if semantic_failures:
                raise _error("Judge failures cannot exist without an Intent")
            if semantic_ungraded:
                raise _error("Judge ungraded receipts cannot exist without an Intent")
            clarification = ClarificationEvaluation(
                policy=truth.clarification_policy,
                decision_correct=None,
                complete=None,
                exchanges=(),
                reason_codes=(IntentReasonCode.SUBMISSION_INTENT_MISSING,),
            )
            return self._unmatched_result(
                status=IntentEvaluationStatus.UNGRADED,
                generated=(),
                truth_items=truth_items,
                clarification=clarification,
                reason_codes=(IntentReasonCode.SUBMISSION_INTENT_MISSING,),
                submission_intent_digest=submission_intent_digest,
                intent_truth_digest=intent_truth_digest,
                clarification_script_digest=clarification_script_digest,
            )

        clarification = self._clarification(
            projected=projected,
            intent_status=intent.status,
            policy=truth.clarification_policy,
            transcript=transcript_items,
            script=clarification_script,
            receipts=receipts,
        )

        if not truth.scorable:
            if semantic_decisions:
                raise _error("unscorable Intent truth cannot accept Judge decisions")
            if semantic_failures:
                raise _error("unscorable Intent truth cannot accept Judge failures")
            if semantic_ungraded:
                raise _error(
                    "unscorable Intent truth cannot accept Judge ungraded receipts"
                )
            return self._unmatched_result(
                status=IntentEvaluationStatus.NOT_SCORABLE,
                generated=projected,
                truth_items=truth_items,
                clarification=clarification,
                reason_codes=(IntentReasonCode.INTENT_TRUTH_UNSCORABLE,),
                submission_intent_digest=submission_intent_digest,
                intent_truth_digest=intent_truth_digest,
                clarification_script_digest=clarification_script_digest,
            )

        seeds, requests, limit_exceeded = self._candidate_seeds(
            projected, truth_items
        )
        if limit_exceeded:
            if semantic_decisions or semantic_failures or semantic_ungraded:
                raise _error(
                    "candidate limit exceeded before Judge results could be bound"
                )
            return self._unmatched_result(
                status=IntentEvaluationStatus.UNGRADED,
                generated=projected,
                truth_items=truth_items,
                clarification=clarification,
                reason_codes=(IntentReasonCode.CANDIDATE_EDGE_LIMIT_EXCEEDED,),
                submission_intent_digest=submission_intent_digest,
                intent_truth_digest=intent_truth_digest,
                clarification_script_digest=clarification_script_digest,
            )

        decision_map = self._validate_decisions(semantic_decisions, requests)
        failure_map = self._validate_failures(
            semantic_failures,
            requests,
            decision_map,
        )
        ungraded_map = self._validate_ungraded(
            semantic_ungraded,
            requests,
            decision_map,
            failure_map,
        )
        specs = tuple(
            self._resolve_candidate(
                seed,
                decision_map,
                failure_map,
                ungraded_map,
            )
            for seed in seeds
        )
        edges = tuple(
            WeightedAssignmentEdge(
                spec.generated.generated_id,
                spec.truth.truth_id,
                spec.edge_weight,
            )
            for spec in specs
            if spec.edge_weight is not None
        )
        assignment = maximum_weight_bipartite_assignment(
            tuple(item.generated_id for item in projected),
            tuple(item.truth_id for item in truth_items),
            edges,
            edge_limit=MAX_INTENT_TOTAL_CANDIDATES,
        )
        selected = {
            (item.left_id, item.right_id): item for item in assignment.matches
        }
        candidates = tuple(
            _candidate_record(
                spec,
                selected=(
                    spec.generated.generated_id,
                    spec.truth.truth_id,
                )
                in selected,
            )
            for spec in specs
        )
        if _records_exceed_byte_budget(
            candidates,
            maximum=MAX_INTENT_CANDIDATE_RECORD_BYTES,
        ):
            raise _error(
                "resolved Intent candidate records exceed the case byte budget"
            )
        outcomes = self._outcomes(projected, specs, selected)

        pending = any(
            request.request_id not in decision_map
            and request.request_id not in failure_map
            and request.request_id not in ungraded_map
            for request in requests
        )
        judge_failed = bool(failure_map)
        judge_ungraded = bool(ungraded_map)
        judge_unknown = any(
            decision.relation is IntentJudgeRelation.UNKNOWN
            for decision in decision_map.values()
        )
        receipt_missing = (
            IntentReasonCode.CLARIFICATION_RECEIPT_MISSING
            in clarification.reason_codes
        )
        if pending:
            status = IntentEvaluationStatus.PENDING_JUDGE
        elif judge_failed or judge_ungraded or judge_unknown or receipt_missing:
            status = IntentEvaluationStatus.UNGRADED
        else:
            status = IntentEvaluationStatus.GRADED

        metrics = self._metrics(truth_items, outcomes, clarification, status)
        reasons = list(clarification.reason_codes)
        if pending:
            reasons.append(IntentReasonCode.JUDGE_PENDING)
        if judge_failed:
            reasons.append(IntentReasonCode.JUDGE_FAILED)
        if judge_ungraded:
            reasons.append(IntentReasonCode.JUDGE_UNGRADED)
        if judge_unknown:
            reasons.append(IntentReasonCode.JUDGE_UNKNOWN)
        if metrics.required_missed_count:
            reasons.append(IntentReasonCode.REQUIRED_TRUTH_MISSED)
        optional_truth_ids = {
            item.truth_id
            for item in truth_items
            if item.kind is IntentTruthKind.EXPECTED and item.required is False
        }
        if optional_truth_ids.intersection(assignment.unmatched_right):
            reasons.append(IntentReasonCode.OPTIONAL_TRUTH_MISSED)
        if metrics.forbidden_hit_count:
            reasons.append(IntentReasonCode.FORBIDDEN_TRUTH_HIT)

        return IntentEvaluationResult(
            schema_version=INTENT_EVALUATION_SCHEMA_VERSION,
            evaluator_revision=self.evaluator_revision,
            submission_intent_digest=submission_intent_digest,
            intent_truth_digest=intent_truth_digest,
            clarification_script_digest=clarification_script_digest,
            policy_version=ASSIGNMENT_POLICY_VERSION,
            normalization_version=INTENT_NORMALIZATION_POLICY_VERSION,
            status=status,
            generated_claims=projected,
            truth_claims=truth_items,
            candidates=candidates,
            assignments=assignment.matches,
            claim_outcomes=outcomes,
            unmatched_generated_ids=assignment.unmatched_left,
            unmatched_expected_truth_ids=tuple(
                sorted(
                    item.truth_id
                    for item in truth_items
                    if item.kind is IntentTruthKind.EXPECTED
                    and item.truth_id in assignment.unmatched_right
                )
            ),
            unmatched_forbidden_truth_ids=tuple(
                sorted(
                    item.truth_id
                    for item in truth_items
                    if item.kind is IntentTruthKind.FORBIDDEN
                    and item.truth_id in assignment.unmatched_right
                )
            ),
            judge_requests=requests,
            judge_decisions=tuple(decision_map.values()),
            judge_failures=tuple(failure_map.values()),
            judge_ungraded=tuple(ungraded_map.values()),
            clarification=clarification,
            metrics=metrics,
            reason_codes=tuple(reasons),
        )

    @staticmethod
    def _candidate_seeds(
        generated: Sequence[GeneratedIntentClaim],
        truth_items: Sequence[IntentTruthClaim],
    ) -> tuple[
        Tuple[_CandidateSpec, ...],
        Tuple[IntentSemanticJudgeRequest, ...],
        bool,
    ]:
        seeds: list[_CandidateSpec] = []
        requests: list[IntentSemanticJudgeRequest] = []
        pair_count = 0
        unresolved_count = 0
        request_text_bytes = 0
        request_record_bytes = 2  # JSON array brackets.
        candidate_record_bytes = 2  # JSON array brackets, with selected=false.
        for generated_item in generated:
            for truth_item in truth_items:
                if generated_item.dimension is not truth_item.dimension:
                    continue
                pair_count += 1
                if pair_count > MAX_INTENT_TOTAL_CANDIDATES:
                    return (), (), True
                if generated_item.text == truth_item.text:
                    relation = IntentJudgeRelation.EQUIVALENT
                    kind = IntentMatchKind.EXACT
                    judgement = _relation_judgement(truth_item.kind, relation)
                    seed = _CandidateSpec(
                        generated_item,
                        truth_item,
                        kind,
                        None,
                        relation,
                        None,
                        judgement,
                        _edge_weight(kind, relation, 0),
                        (IntentReasonCode.DETERMINISTIC_EXACT,),
                    )
                elif generated_item.normalized_text == truth_item.normalized_text:
                    relation = IntentJudgeRelation.EQUIVALENT
                    kind = IntentMatchKind.NORMALIZED
                    judgement = _relation_judgement(truth_item.kind, relation)
                    seed = _CandidateSpec(
                        generated_item,
                        truth_item,
                        kind,
                        None,
                        relation,
                        None,
                        judgement,
                        _edge_weight(kind, relation, 0),
                        (IntentReasonCode.DETERMINISTIC_NORMALIZED,),
                    )
                else:
                    unresolved_count += 1
                    if unresolved_count > MAX_INTENT_CANDIDATE_EDGES:
                        return (), (), True
                    request_text_bytes += len(
                        generated_item.text.encode("utf-8")
                    ) + len(truth_item.text.encode("utf-8"))
                    if request_text_bytes > MAX_INTENT_REQUEST_TEXT_BYTES:
                        return (), (), True
                    request = IntentSemanticJudgeRequest(
                        request_id=_request_id(generated_item, truth_item),
                        generated_id=generated_item.generated_id,
                        truth_id=truth_item.truth_id,
                        dimension=generated_item.dimension,
                        generated_text=generated_item.text,
                        truth_text=truth_item.text,
                        truth_kind=truth_item.kind,
                    )
                    if requests:
                        request_record_bytes += 1  # JSON array comma.
                    request_record_bytes += len(
                        canonical_json(request.to_dict()).encode("utf-8")
                    )
                    if request_record_bytes > MAX_INTENT_JUDGE_REQUEST_BYTES:
                        return (), (), True
                    requests.append(request)
                    seed = _CandidateSpec(
                        generated_item,
                        truth_item,
                        IntentMatchKind.SEMANTIC,
                        request,
                        None,
                        None,
                        None,
                        None,
                        (IntentReasonCode.JUDGE_PENDING,),
                    )
                seeds.append(seed)
                if pair_count > 1:
                    candidate_record_bytes += 1  # JSON array comma.
                candidate_record_bytes += len(
                    canonical_json(
                        _candidate_record(seed, selected=False).to_dict()
                    ).encode("utf-8")
                )
        generated_groups = Counter(
            (item.dimension, item.normalized_text) for item in generated
        )
        truth_groups = Counter(
            (item.dimension, item.normalized_text) for item in truth_items
        )
        deterministic_selected_count = sum(
            min(count, truth_groups[group])
            for group, count in generated_groups.items()
        )
        # In canonical JSON, replacing selected=false with selected=true saves
        # exactly one byte.  Pending semantic pairs have no eligible edge, so
        # deterministic groups fully determine the selected count.
        candidate_record_bytes -= deterministic_selected_count
        if candidate_record_bytes > MAX_INTENT_CANDIDATE_RECORD_BYTES:
            return (), (), True
        return (
            tuple(
                sorted(
                    seeds,
                    key=lambda item: (
                        item.generated.generated_id,
                        item.truth.truth_id,
                    ),
                )
            ),
            tuple(sorted(requests, key=lambda item: item.request_id)),
            False,
        )

    @staticmethod
    def _validate_decisions(
        decisions: Sequence[IntentSemanticJudgeDecision],
        requests: Sequence[IntentSemanticJudgeRequest],
    ) -> dict[str, IntentSemanticJudgeDecision]:
        request_ids = {item.request_id for item in requests}
        result: dict[str, IntentSemanticJudgeDecision] = {}
        for index, decision in enumerate(decisions):
            if type(decision) is not IntentSemanticJudgeDecision:
                raise _error(f"semantic_decisions[{index}] has invalid type")
            if decision.request_id not in request_ids:
                raise _error("Judge decision references an unknown request")
            if decision.request_id in result:
                raise _error("duplicate Judge decision request_id")
            result[decision.request_id] = decision
        _validate_decision_budgets(result.values())
        return dict(sorted(result.items()))

    @staticmethod
    def _validate_failures(
        failures: Sequence[IntentSemanticJudgeFailure],
        requests: Sequence[IntentSemanticJudgeRequest],
        decisions: Mapping[str, IntentSemanticJudgeDecision],
    ) -> dict[str, IntentSemanticJudgeFailure]:
        request_ids = {item.request_id for item in requests}
        result: dict[str, IntentSemanticJudgeFailure] = {}
        for index, failure in enumerate(failures):
            if type(failure) is not IntentSemanticJudgeFailure:
                raise _error(f"semantic_failures[{index}] has invalid type")
            if failure.request_id not in request_ids:
                raise _error("Judge failure references an unknown request")
            if failure.request_id in decisions:
                raise _error("a Judge request cannot have both a decision and a failure")
            if failure.request_id in result:
                raise _error("duplicate Judge failure request_id")
            result[failure.request_id] = failure
        _validate_failure_budgets(result.values())
        return dict(sorted(result.items()))

    @staticmethod
    def _validate_ungraded(
        receipts: Sequence[IntentSemanticJudgeUngraded],
        requests: Sequence[IntentSemanticJudgeRequest],
        decisions: Mapping[str, IntentSemanticJudgeDecision],
        failures: Mapping[str, IntentSemanticJudgeFailure],
    ) -> dict[str, IntentSemanticJudgeUngraded]:
        request_ids = {item.request_id for item in requests}
        result: dict[str, IntentSemanticJudgeUngraded] = {}
        for index, receipt in enumerate(receipts):
            if type(receipt) is not IntentSemanticJudgeUngraded:
                raise _error(f"semantic_ungraded[{index}] has invalid type")
            if receipt.request_id not in request_ids:
                raise _error("Judge ungraded receipt references an unknown request")
            if receipt.request_id in decisions or receipt.request_id in failures:
                raise _error("a Judge request cannot have more than one resolution")
            if receipt.request_id in result:
                raise _error("duplicate Judge ungraded request_id")
            result[receipt.request_id] = receipt
        _validate_ungraded_budgets(result.values())
        provenance_receipts = (*failures.values(), *result.values())
        execution_digests = {
            item.evaluator_execution_digest for item in provenance_receipts
        }
        if len(execution_digests) > 1:
            raise _error("Judge receipts bind multiple evaluator executions")
        result_digests = [item.judge_result_digest for item in provenance_receipts]
        if len(result_digests) != len(set(result_digests)):
            raise _error("Judge result digest cannot resolve multiple requests")
        return dict(sorted(result.items()))

    @staticmethod
    def _resolve_candidate(
        seed: _CandidateSpec,
        decisions: Mapping[str, IntentSemanticJudgeDecision],
        failures: Mapping[str, IntentSemanticJudgeFailure],
        ungraded: Mapping[str, IntentSemanticJudgeUngraded],
    ) -> _CandidateSpec:
        if seed.request is None:
            return seed
        decision = decisions.get(seed.request.request_id)
        if decision is None:
            if (
                seed.request.request_id not in failures
                and seed.request.request_id not in ungraded
            ):
                return seed
            return _CandidateSpec(
                seed.generated,
                seed.truth,
                seed.match_kind,
                seed.request,
                None,
                None,
                None,
                None,
                (
                    IntentReasonCode.JUDGE_FAILED
                    if seed.request.request_id in failures
                    else IntentReasonCode.JUDGE_UNGRADED,
                ),
            )
        relation = decision.relation
        judgement = _relation_judgement(seed.truth.kind, relation)
        reasons = (_semantic_reason(relation),)
        weight = (
            None
            if judgement is None
            else _edge_weight(
                IntentMatchKind.SEMANTIC, relation, decision.score_ppm
            )
        )
        return _CandidateSpec(
            seed.generated,
            seed.truth,
            seed.match_kind,
            seed.request,
            relation,
            decision.score_ppm,
            judgement,
            weight,
            reasons,
        )

    @staticmethod
    def _outcomes(
        generated: Sequence[GeneratedIntentClaim],
        specs: Sequence[_CandidateSpec],
        selected: Mapping[tuple[str, str], AssignedPair],
    ) -> Tuple[IntentClaimOutcome, ...]:
        by_generated: dict[str, list[_CandidateSpec]] = {}
        for spec in specs:
            by_generated.setdefault(spec.generated.generated_id, []).append(spec)
        outcomes = []
        for generated_item in generated:
            relevant = by_generated.get(generated_item.generated_id, [])
            chosen = next(
                (
                    spec
                    for spec in relevant
                    if (generated_item.generated_id, spec.truth.truth_id) in selected
                ),
                None,
            )
            if chosen is not None:
                if chosen.judgement is None:
                    raise _error("selected candidate has no claim judgement")
                outcomes.append(
                    IntentClaimOutcome(
                        generated_id=generated_item.generated_id,
                        judgement=chosen.judgement,
                        matched_truth_id=chosen.truth.truth_id,
                        matched_truth_kind=chosen.truth.kind,
                        match_kind=chosen.match_kind,
                        reason_codes=chosen.reason_codes
                        + (
                            IntentReasonCode.MATCHED_FORBIDDEN
                            if chosen.truth.kind is IntentTruthKind.FORBIDDEN
                            else IntentReasonCode.MATCHED_EXPECTED,
                        ),
                    )
                )
                continue

            has_unknown = any(
                spec.relation is IntentJudgeRelation.UNKNOWN for spec in relevant
            )
            has_failed = any(
                IntentReasonCode.JUDGE_FAILED in spec.reason_codes
                for spec in relevant
            )
            has_ungraded = any(
                IntentReasonCode.JUDGE_UNGRADED in spec.reason_codes
                for spec in relevant
            )
            has_pending = any(
                spec.request is not None
                and spec.relation is None
                and IntentReasonCode.JUDGE_FAILED not in spec.reason_codes
                and IntentReasonCode.JUDGE_UNGRADED not in spec.reason_codes
                for spec in relevant
            )
            if has_unknown:
                judgement = IntentClaimJudgement.UNKNOWN
                reasons = (IntentReasonCode.JUDGE_UNKNOWN,)
            elif has_failed:
                judgement = IntentClaimJudgement.UNKNOWN
                reasons = (IntentReasonCode.JUDGE_FAILED,)
            elif has_ungraded:
                judgement = IntentClaimJudgement.UNKNOWN
                reasons = (IntentReasonCode.JUDGE_UNGRADED,)
            elif has_pending:
                judgement = IntentClaimJudgement.UNKNOWN
                reasons = (IntentReasonCode.JUDGE_PENDING,)
            elif not relevant:
                judgement = IntentClaimJudgement.UNSUPPORTED
                reasons = (IntentReasonCode.NO_TRUTH_CANDIDATE,)
            elif any(spec.edge_weight is not None for spec in relevant):
                judgement = IntentClaimJudgement.UNSUPPORTED
                reasons = (IntentReasonCode.UNMATCHED_DUPLICATE,)
            elif all(
                spec.relation is IntentJudgeRelation.DIFFERENT
                for spec in relevant
            ):
                judgement = IntentClaimJudgement.UNSUPPORTED
                reasons = (IntentReasonCode.JUDGE_DIFFERENT,)
            else:
                judgement = IntentClaimJudgement.UNSUPPORTED
                reasons = tuple(
                    sorted(
                        {
                            reason
                            for spec in relevant
                            for reason in spec.reason_codes
                        },
                        key=lambda reason: reason.value,
                    )
                ) or (IntentReasonCode.NO_TRUTH_CANDIDATE,)
            outcomes.append(
                IntentClaimOutcome(
                    generated_id=generated_item.generated_id,
                    judgement=judgement,
                    matched_truth_id=None,
                    matched_truth_kind=None,
                    match_kind=None,
                    reason_codes=reasons,
                )
            )
        return tuple(sorted(outcomes, key=lambda item: item.generated_id))

    def _clarification(
        self,
        *,
        projected: Sequence[GeneratedIntentClaim],
        intent_status: IntentResult,
        policy: Optional[ClarificationPolicy],
        transcript: Sequence[SubmissionClarificationExchange],
        script: ClarificationScript,
        receipts: Sequence[MaterialClaimMatchReceipt],
    ) -> ClarificationEvaluation:
        receipt_map = self._receipt_map(transcript, receipts)
        answers = {answer.answer_id: answer for answer in script.answers}
        claimed_consumed: set[str] = set()
        matcher_digest: Optional[str] = None
        evaluations: list[ClarificationExchangeEvaluation] = []
        all_reasons: list[IntentReasonCode] = []

        for exchange in transcript:
            reasons: list[IntentReasonCode] = []
            answer = (
                None
                if exchange.matched_answer_id is None
                else answers.get(exchange.matched_answer_id)
            )
            if exchange.matched_answer_id is not None and answer is None:
                raise _error("clarification transcript references an unknown answer")
            if answer is not None:
                self._validate_exchange_answer(exchange, answer)

            receipt = receipt_map.get((exchange.turn_index, exchange.question_id))
            if receipt is None:
                material = None
                consumed = None
                update = None
                receipt_digest = None
                receipt_matcher_digest = None
                reasons.append(IntentReasonCode.CLARIFICATION_RECEIPT_MISSING)
                if answer is not None and answer.dimension is not exchange.dimension:
                    reasons.append(IntentReasonCode.CLARIFICATION_WRONG_DIMENSION)
            else:
                matcher_digest = self._validate_receipt(
                    receipt=receipt,
                    exchange=exchange,
                    script=script,
                    answers=answers,
                    consumed_answer_ids=claimed_consumed,
                    expected_matcher_digest=matcher_digest,
                )
                receipt_digest = _material_receipt_digest(receipt)
                receipt_matcher_digest = receipt.matcher_digest
                material = receipt.outcome is MaterialClaimMatchOutcome.MATCHED
                consumed = material
                if receipt.outcome is MaterialClaimMatchOutcome.UNMATCHED:
                    reasons.append(IntentReasonCode.CLARIFICATION_UNMATCHED)
                    if self._looks_like_wrong_dimension(exchange, script):
                        reasons.append(IntentReasonCode.CLARIFICATION_WRONG_DIMENSION)
                    else:
                        reasons.append(
                            IntentReasonCode.CLARIFICATION_WRONG_MATERIAL_CLAIM
                        )
                elif receipt.outcome is MaterialClaimMatchOutcome.AMBIGUOUS:
                    reasons.append(IntentReasonCode.CLARIFICATION_AMBIGUOUS)
                elif receipt.outcome is MaterialClaimMatchOutcome.ROUND_LIMIT:
                    reasons.append(IntentReasonCode.CLARIFICATION_ROUND_LIMIT)
                if not consumed:
                    reasons.append(
                        IntentReasonCode.CLARIFICATION_ANSWER_NOT_CONSUMED
                    )
                update = self._clarification_update(
                    projected, exchange, answer if material else None
                )
                if update is False:
                    reasons.append(
                        IntentReasonCode.CLARIFICATION_ANSWER_NOT_APPLIED
                    )

            if exchange.action is None or exchange.action is ClarificationAction.DEFER:
                reasons.append(IntentReasonCode.CLARIFICATION_UNRESOLVED)
            if exchange.matched_answer_id is not None:
                claimed_consumed.add(exchange.matched_answer_id)
            all_reasons.extend(reasons)
            evaluations.append(
                ClarificationExchangeEvaluation(
                    turn_index=exchange.turn_index,
                    question_id=exchange.question_id,
                    matched_answer_id=exchange.matched_answer_id,
                    receipt_digest=receipt_digest,
                    matcher_digest=receipt_matcher_digest,
                    material=material,
                    answer_consumed=consumed,
                    update_applied=update,
                    reason_codes=tuple(reasons),
                )
            )

        if policy is None:
            all_reasons.append(IntentReasonCode.INTENT_TRUTH_UNSCORABLE)
            return ClarificationEvaluation(
                policy=None,
                decision_correct=None,
                complete=None,
                exchanges=tuple(evaluations),
                reason_codes=tuple(all_reasons),
            )

        proven_material = any(
            item.material is True and item.answer_consumed is True
            for item in evaluations
        )
        has_unknown_receipt = any(item.material is None for item in evaluations)
        if policy is ClarificationPolicy.REQUIRED:
            decision_correct = (
                True
                if proven_material
                else (None if has_unknown_receipt else False)
            )
            if decision_correct is False:
                all_reasons.append(
                    IntentReasonCode.REQUIRED_CLARIFICATION_NOT_ASKED
                )
        elif policy is ClarificationPolicy.OPTIONAL:
            decision_correct = True
        else:
            decision_correct = not transcript
            if transcript:
                all_reasons.append(
                    IntentReasonCode.CLARIFICATION_UNNECESSARY_QUESTION
                )

        resolved = (
            intent_status is IntentResult.SUFFICIENT
            and decision_correct is not None
            and not has_unknown_receipt
            and all(item.answer_consumed is not False for item in evaluations)
            and all(item.update_applied is not False for item in evaluations)
            and all(
                exchange.action is not None
                and exchange.action is not ClarificationAction.DEFER
                for exchange in transcript
            )
        )
        if policy is ClarificationPolicy.REQUIRED:
            resolved = resolved and decision_correct is True
        if intent_status is not IntentResult.SUFFICIENT:
            all_reasons.append(IntentReasonCode.CLARIFICATION_UNRESOLVED)
        if (
            policy is ClarificationPolicy.NOT_REQUIRED
            and (intent_status is not IntentResult.SUFFICIENT or not resolved)
        ):
            all_reasons.append(
                IntentReasonCode.CLARIFICATION_UNNECESSARY_BLOCKING
            )
        if decision_correct is True:
            all_reasons.append(IntentReasonCode.CLARIFICATION_DECISION_CORRECT)

        return ClarificationEvaluation(
            policy=policy,
            decision_correct=decision_correct,
            complete=resolved,
            exchanges=tuple(evaluations),
            reason_codes=tuple(all_reasons),
        )

    @staticmethod
    def _receipt_map(
        transcript: Sequence[SubmissionClarificationExchange],
        receipts: Sequence[MaterialClaimMatchReceipt],
    ) -> dict[tuple[int, str], MaterialClaimMatchReceipt]:
        valid_keys = {(item.turn_index, item.question_id) for item in transcript}
        result: dict[tuple[int, str], MaterialClaimMatchReceipt] = {}
        for index, receipt in enumerate(receipts):
            if type(receipt) is not MaterialClaimMatchReceipt:
                raise _error(f"receipts[{index}] has an invalid type")
            key = (receipt.turn_index, receipt.question_id)
            if key not in valid_keys:
                raise _error("clarification receipt does not belong to transcript")
            if key in result:
                raise _error("duplicate clarification receipt")
            result[key] = receipt
        return result

    def _validate_receipt(
        self,
        *,
        receipt: MaterialClaimMatchReceipt,
        exchange: SubmissionClarificationExchange,
        script: ClarificationScript,
        answers: Mapping[str, Any],
        consumed_answer_ids: set[str],
        expected_matcher_digest: Optional[str],
    ) -> str:
        if type(receipt.turn_index) is not int or receipt.turn_index != exchange.turn_index:
            raise _error("clarification receipt turn does not match transcript")
        _id(receipt.question_id, "clarification receipt.question_id")
        if receipt.question_id != exchange.question_id:
            raise _error("clarification receipt question does not match transcript")
        if (
            not isinstance(receipt.dimension, IntentDimension)
            or receipt.dimension is not exchange.dimension
        ):
            raise _error("clarification receipt dimension does not match transcript")
        _digest(receipt.actual_claim_digest, "clarification receipt.actual_claim_digest")
        matcher_digest = _digest(
            receipt.matcher_digest, "clarification receipt.matcher_digest"
        )
        if expected_matcher_digest is not None and matcher_digest != expected_matcher_digest:
            raise _error("clarification receipts use different matcher bindings")
        expected_actual_digest = canonical_sha256(
            {
                "dimension": exchange.dimension.value,
                "material_claim": exchange.material_claim,
            }
        )
        if receipt.actual_claim_digest != expected_actual_digest:
            raise _error("clarification receipt material-claim hash mismatch")
        if type(receipt.candidates) is not tuple:
            raise _error("clarification receipt.candidates must be an immutable tuple")
        if not isinstance(receipt.outcome, MaterialClaimMatchOutcome):
            raise _error("clarification receipt.outcome has an invalid type")
        if receipt.matched_answer_id is not None:
            _id(
                receipt.matched_answer_id,
                "clarification receipt.matched_answer_id",
            )

        decisions = []
        for index, candidate in enumerate(receipt.candidates):
            if type(candidate) is not MaterialClaimCandidateDecision:
                raise _error(
                    f"clarification receipt.candidates[{index}] has invalid type"
                )
            answer_id = _id(
                candidate.answer_id,
                "clarification receipt candidate.answer_id",
            )
            answer = answers.get(answer_id)
            if answer is None:
                raise _error("clarification receipt candidate references unknown answer")
            if answer.dimension is not exchange.dimension:
                raise _error("clarification receipt candidate crosses dimensions")
            if answer_id in consumed_answer_ids:
                raise _error("clarification receipt reuses a consumed answer")
            if (
                type(candidate.equivalent) is not bool
                or type(candidate.action_eligible) is not bool
            ):
                raise _error("clarification receipt candidate booleans are invalid")
            if (
                answer.action is not ClarificationAction.CONFIRM
                and not candidate.action_eligible
            ):
                raise _error("non-confirm clarification answer must be action eligible")
            expected_request_digest = canonical_sha256(
                {
                    "matcher_digest": matcher_digest,
                    "dimension": exchange.dimension.value,
                    "actual_claim": exchange.material_claim,
                    "scripted_claim": answer.material_claim,
                    "answer_id": answer.answer_id,
                }
            )
            if _digest(
                candidate.request_digest,
                "clarification receipt candidate.request_digest",
            ) != expected_request_digest:
                raise _error("clarification receipt candidate hash mismatch")
            decisions.append(candidate)
        candidate_ids = tuple(item.answer_id for item in decisions)
        if candidate_ids != tuple(sorted(candidate_ids)) or len(
            candidate_ids
        ) != len(set(candidate_ids)):
            raise _error("clarification receipt candidates must be unique and sorted")

        round_limited = exchange.turn_index > script.max_rounds
        expected_candidate_ids = tuple(
            answer.answer_id
            for answer in script.answers
            if answer.dimension is exchange.dimension
            and answer.answer_id not in consumed_answer_ids
        )
        if round_limited:
            if (
                receipt.outcome is not MaterialClaimMatchOutcome.ROUND_LIMIT
                or decisions
                or receipt.matched_answer_id is not None
            ):
                raise _error("round-limit clarification receipt is inconsistent")
        else:
            if receipt.outcome is MaterialClaimMatchOutcome.ROUND_LIMIT:
                raise _error("clarification receipt claims a premature round limit")
            if candidate_ids != expected_candidate_ids:
                raise _error("clarification receipt candidate set is incomplete")
            eligible = [
                item
                for item in decisions
                if item.equivalent and item.action_eligible
            ]
            expected_outcome = (
                MaterialClaimMatchOutcome.MATCHED
                if len(eligible) == 1
                else (
                    MaterialClaimMatchOutcome.AMBIGUOUS
                    if len(eligible) > 1
                    else MaterialClaimMatchOutcome.UNMATCHED
                )
            )
            expected_answer_id = (
                eligible[0].answer_id if len(eligible) == 1 else None
            )
            if (
                receipt.outcome is not expected_outcome
                or receipt.matched_answer_id != expected_answer_id
            ):
                raise _error("clarification receipt outcome is inconsistent")
        if receipt.matched_answer_id != exchange.matched_answer_id:
            raise _error("clarification receipt match does not equal transcript")
        return matcher_digest

    @staticmethod
    def _validate_exchange_answer(exchange: Any, answer: Any) -> None:
        if exchange.action is not answer.action or exchange.response != answer.response:
            raise _error("clarification transcript does not preserve scripted answer")
        if answer.action is ClarificationAction.CORRECT:
            if exchange.resolved_values != answer.corrected_values:
                raise _error("clarification correction does not match scripted values")
        elif answer.action is not ClarificationAction.CONFIRM and exchange.resolved_values:
            raise _error("clarification transcript has unexpected resolved values")

    @staticmethod
    def _looks_like_wrong_dimension(
        exchange: SubmissionClarificationExchange,
        script: ClarificationScript,
    ) -> bool:
        normalized = normalize_intent_text(exchange.material_claim)
        return any(
            answer.dimension is not exchange.dimension
            and normalize_intent_text(answer.material_claim) == normalized
            for answer in script.answers
        )

    @staticmethod
    def _clarification_update(
        projected: Sequence[GeneratedIntentClaim],
        exchange: SubmissionClarificationExchange,
        answer: Any,
    ) -> Optional[bool]:
        if answer is None or exchange.action in {
            None,
            ClarificationAction.SKIP,
            ClarificationAction.DEFER,
        }:
            return None
        final_claims = {
            item.normalized_text
            for item in projected
            if item.dimension is exchange.dimension
        }
        original_claims = {
            normalize_intent_text(exchange.material_claim),
            normalize_intent_text(answer.material_claim),
        }
        if exchange.action is ClarificationAction.REJECT:
            return original_claims.isdisjoint(final_claims)
        resolved = {
            normalize_intent_text(item) for item in exchange.resolved_values
        }
        if exchange.action is ClarificationAction.CONFIRM:
            return bool(resolved) and resolved.issubset(final_claims)
        return (
            bool(resolved)
            and resolved.issubset(final_claims)
            and original_claims.isdisjoint(final_claims)
        )

    @staticmethod
    def _metrics(
        truth_items: Sequence[IntentTruthClaim],
        outcomes: Sequence[IntentClaimOutcome],
        clarification: ClarificationEvaluation,
        status: IntentEvaluationStatus,
    ) -> IntentMetricInputs:
        counts = Counter(item.judgement for item in outcomes)
        required = [
            item
            for item in truth_items
            if item.kind is IntentTruthKind.EXPECTED and item.required is True
        ]
        optional = [
            item
            for item in truth_items
            if item.kind is IntentTruthKind.EXPECTED and item.required is False
        ]
        matched = {
            item.matched_truth_id: item
            for item in outcomes
            if item.matched_truth_id is not None
        }
        required_supported = sum(
            item.truth_id in matched
            and matched[item.truth_id].judgement is IntentClaimJudgement.SUPPORTED
            for item in required
        )
        required_partial = sum(
            item.truth_id in matched
            and matched[item.truth_id].judgement
            is IntentClaimJudgement.PARTIALLY_SUPPORTED
            for item in required
        )
        optional_supported = sum(
            item.truth_id in matched
            and matched[item.truth_id].judgement is IntentClaimJudgement.SUPPORTED
            for item in optional
        )
        forbidden_hits = sum(
            item.matched_truth_kind is IntentTruthKind.FORBIDDEN
            for item in outcomes
        )
        clarification_numerator = (
            None
            if clarification.decision_correct is None
            else int(clarification.decision_correct)
        )
        clarification_denominator = (
            None if clarification.decision_correct is None else 1
        )
        unknown = counts[IntentClaimJudgement.UNKNOWN]
        case_pass = (
            None
            if status is not IntentEvaluationStatus.GRADED or unknown
            else (
                required_supported == len(required)
                and counts[IntentClaimJudgement.PARTIALLY_SUPPORTED] == 0
                and counts[IntentClaimJudgement.UNSUPPORTED] == 0
                and counts[IntentClaimJudgement.CONTRADICTED] == 0
                and forbidden_hits == 0
                and clarification.decision_correct is True
                and clarification.complete is True
            )
        )
        return IntentMetricInputs(
            scorable=True,
            generated_claim_count=len(outcomes),
            supported_claim_count=counts[IntentClaimJudgement.SUPPORTED],
            partially_supported_claim_count=counts[
                IntentClaimJudgement.PARTIALLY_SUPPORTED
            ],
            unsupported_claim_count=counts[IntentClaimJudgement.UNSUPPORTED],
            contradicted_claim_count=counts[IntentClaimJudgement.CONTRADICTED],
            unknown_claim_count=unknown,
            required_truth_count=len(required),
            required_supported_count=required_supported,
            required_partially_supported_count=required_partial,
            required_missed_count=len(required) - required_supported,
            optional_truth_count=len(optional),
            optional_supported_count=optional_supported,
            forbidden_truth_count=sum(
                item.kind is IntentTruthKind.FORBIDDEN for item in truth_items
            ),
            forbidden_hit_count=forbidden_hits,
            clarification_numerator=clarification_numerator,
            clarification_denominator=clarification_denominator,
            intent_case_pass=case_pass,
        )

    @staticmethod
    def _metrics_unscorable() -> IntentMetricInputs:
        return IntentMetricInputs(
            scorable=False,
            generated_claim_count=None,
            supported_claim_count=None,
            partially_supported_claim_count=None,
            unsupported_claim_count=None,
            contradicted_claim_count=None,
            unknown_claim_count=None,
            required_truth_count=None,
            required_supported_count=None,
            required_partially_supported_count=None,
            required_missed_count=None,
            optional_truth_count=None,
            optional_supported_count=None,
            forbidden_truth_count=None,
            forbidden_hit_count=None,
            clarification_numerator=None,
            clarification_denominator=None,
            intent_case_pass=None,
        )

    def _unmatched_result(
        self,
        *,
        status: IntentEvaluationStatus,
        generated: Sequence[GeneratedIntentClaim],
        truth_items: Sequence[IntentTruthClaim],
        clarification: ClarificationEvaluation,
        reason_codes: Tuple[IntentReasonCode, ...],
        submission_intent_digest: Optional[str],
        intent_truth_digest: str,
        clarification_script_digest: str,
    ) -> IntentEvaluationResult:
        outcomes = tuple(
            IntentClaimOutcome(
                generated_id=item.generated_id,
                judgement=IntentClaimJudgement.UNKNOWN,
                matched_truth_id=None,
                matched_truth_kind=None,
                match_kind=None,
                reason_codes=reason_codes,
            )
            for item in generated
        )
        return IntentEvaluationResult(
            schema_version=INTENT_EVALUATION_SCHEMA_VERSION,
            evaluator_revision=self.evaluator_revision,
            submission_intent_digest=submission_intent_digest,
            intent_truth_digest=intent_truth_digest,
            clarification_script_digest=clarification_script_digest,
            policy_version=ASSIGNMENT_POLICY_VERSION,
            normalization_version=INTENT_NORMALIZATION_POLICY_VERSION,
            status=status,
            generated_claims=tuple(generated),
            truth_claims=tuple(truth_items),
            candidates=(),
            assignments=(),
            claim_outcomes=outcomes,
            unmatched_generated_ids=tuple(
                sorted(item.generated_id for item in generated)
            ),
            unmatched_expected_truth_ids=tuple(
                sorted(
                    item.truth_id
                    for item in truth_items
                    if item.kind is IntentTruthKind.EXPECTED
                )
            ),
            unmatched_forbidden_truth_ids=tuple(
                sorted(
                    item.truth_id
                    for item in truth_items
                    if item.kind is IntentTruthKind.FORBIDDEN
                )
            ),
            judge_requests=(),
            judge_decisions=(),
            judge_failures=(),
            judge_ungraded=(),
            clarification=clarification,
            metrics=self._metrics_unscorable(),
            reason_codes=reason_codes,
        )


def evaluate_intent(
    intent: Optional[SubmissionIntent],
    truth: IntentTruth,
    clarification_script: ClarificationScript,
    **kwargs: Any,
) -> IntentEvaluationResult:
    """Convenience wrapper around :class:`IntentEvaluator`."""

    return IntentEvaluator().evaluate(intent, truth, clarification_script, **kwargs)


__all__ = [
    "INTENT_EVALUATION_SCHEMA_VERSION",
    "INTENT_EVALUATOR_REVISION",
    "INTENT_NORMALIZATION_POLICY_VERSION",
    "MAX_INTENT_CANDIDATE_EDGES",
    "MAX_INTENT_TOTAL_CANDIDATES",
    "MAX_INTENT_REQUEST_TEXT_BYTES",
    "MAX_INTENT_CANDIDATE_RECORD_BYTES",
    "MAX_INTENT_JUDGE_REQUEST_BYTES",
    "MAX_INTENT_JUDGE_DECISION_BYTES",
    "MAX_INTENT_JUDGE_FAILURE_BYTES",
    "MAX_INTENT_JUDGE_UNGRADED_BYTES",
    "MAX_INTENT_JUDGE_REASON_REF_BYTES",
    "MAX_INTENT_PROJECTED_CLAIMS",
    "UNSCORABLE_INTENT_TRUTH_DIGEST",
    "MAX_INTENT_SCORE_PPM",
    "MAX_INTENT_EVALUATION_BYTES",
    "EXACT_INTENT_WEIGHT",
    "NORMALIZED_INTENT_WEIGHT",
    "SEMANTIC_FULL_INTENT_WEIGHT_BASE",
    "SEMANTIC_PARTIAL_INTENT_WEIGHT_BASE",
    "INTENT_JUDGE_FAILURE_CODES",
    "INTENT_JUDGE_UNGRADED_REASONS",
    "IntentEvaluationError",
    "IntentJudgeRelation",
    "IntentEvaluationStatus",
    "IntentTruthKind",
    "IntentClaimOrigin",
    "IntentMatchKind",
    "IntentReasonCode",
    "GeneratedIntentClaim",
    "IntentTruthClaim",
    "IntentSemanticJudgeRequest",
    "IntentSemanticJudgeDecision",
    "IntentSemanticJudgeFailure",
    "IntentSemanticJudgeUngraded",
    "IntentCandidateRecord",
    "IntentClaimOutcome",
    "ClarificationExchangeEvaluation",
    "ClarificationEvaluation",
    "IntentMetricInputs",
    "IntentEvaluationResult",
    "normalize_intent_text",
    "IntentEvaluator",
    "evaluate_intent",
]
