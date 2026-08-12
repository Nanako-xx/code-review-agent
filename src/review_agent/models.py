from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum


BENCHMARK_AUTO_ACCEPT_BASIS = "benchmark_auto_accept"


class IntentSource(str, Enum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"


class IntentStatus(str, Enum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


class IntentField(str, Enum):
    GOAL = "goal"
    ACCEPTANCE_CRITERIA = "acceptance_criteria"
    SCOPE = "scope"
    CONSTRAINTS = "constraints"


class IntentOrigin(str, Enum):
    USER_INPUT = "user_input"
    REQUEST_METADATA = "request_metadata"
    PROJECT_RULE = "project_rule"
    REPOSITORY_DOCUMENT = "repository_document"
    REPOSITORY_TEST = "repository_test"
    COMMIT_MESSAGE = "commit_message"
    LLM_INFERENCE = "llm_inference"
    USER_CONFIRMATION = "user_confirmation"
    BENCHMARK_AUTO_ACCEPT = "benchmark_auto_accept"
    USER_CORRECTION = "user_correction"
    CHANGED_FILES = "changed_files"
    PROJECT_MEMORY = "project_memory"


class IntentConfidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class IntentClaimState(str, Enum):
    ACTIVE = "active"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    INVALID = "invalid"


class ConclusionImpact(str, Enum):
    BLOCKING = "blocking"
    MATERIAL = "material"
    SUPPLEMENTAL = "supplemental"


class ClarificationStatus(str, Enum):
    PENDING = "pending"
    OPEN = "open"
    CONFIRMED = "confirmed"
    CORRECTED = "corrected"
    REJECTED = "rejected"
    SKIPPED = "skipped"
    SKIPPED_NON_INTERACTIVE = "skipped_non_interactive"


class IntentDecisionAction(str, Enum):
    CONFIRMED = "confirmed"
    CORRECTED = "corrected"
    REJECTED = "rejected"
    SKIPPED = "skipped"
    SKIPPED_NON_INTERACTIVE = "skipped_non_interactive"


def _canonical_intent_value(value: str) -> str:
    return " ".join(value.split()).casefold()


def _stable_intent_id(prefix: str, *parts: object) -> str:
    encoded = json.dumps(parts, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def _validate_non_empty_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{name} must not have leading or trailing whitespace")


def _validate_text_list(values: object, name: str) -> None:
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a list")
    seen: set[str] = set()
    for value in values:
        _validate_non_empty_text(value, f"{name} item")
        if value in seen:
            raise ValueError(f"{name} must not contain duplicate values")
        seen.add(value)


_STABLE_MEMORY_ID = re.compile(r"^MEM-[0-9a-f]{64}$")
_STABLE_FEEDBACK_ID = re.compile(r"^FB-[0-9a-f]{64}$")
_POLICY_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:+#/@-]{0,511}$")
MAX_STAGE_HARD_POLICY_ITEMS = 64
MAX_STAGE_HARD_POLICY_BYTES = 32_768
_MEMORY_KINDS = frozenset(
    {
        "architecture_boundary",
        "business_invariant",
        "review_rule",
        "compatibility_requirement",
        "verification_command",
        "incident_lesson",
        "high_risk_module",
    }
)


def _validate_stable_identifier(
    value: object,
    pattern: re.Pattern[str],
    name: str,
) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical stable ID")
    return value


def _validate_policy_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or _POLICY_IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{name} must be a bounded policy identifier")
    return value


def _canonical_text_tuple(
    values: object,
    name: str,
    *,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple")
    normalized: set[str] = set()
    for value in values:
        _validate_non_empty_text(value, f"{name} item")
        normalized.add(value)
    if not allow_empty and not normalized:
        raise ValueError(f"{name} must not be empty")
    return tuple(sorted(normalized))


def _canonical_memory_ids(values: object, name: str) -> tuple[str, ...]:
    normalized = _canonical_text_tuple(values, name, allow_empty=False)
    return tuple(
        _validate_stable_identifier(value, _STABLE_MEMORY_ID, f"{name} item")
        for value in normalized
    )


def _canonical_feedback_ids(values: object, name: str) -> tuple[str, ...]:
    normalized = _canonical_text_tuple(values, name, allow_empty=False)
    return tuple(
        _validate_stable_identifier(value, _STABLE_FEEDBACK_ID, f"{name} item")
        for value in normalized
    )


@dataclass(frozen=True)
class IntentClaim:
    field: IntentField
    value: str
    source: IntentSource
    origin: IntentOrigin
    confidence: IntentConfidence
    source_refs: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    claim_state: IntentClaimState = IntentClaimState.ACTIVE
    conclusion_impact: ConclusionImpact = ConclusionImpact.MATERIAL
    claim_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.field, IntentField):
            raise ValueError("intent claim field must be an IntentField")
        _validate_non_empty_text(self.value, "intent claim value")
        if not isinstance(self.source, IntentSource):
            raise ValueError("intent claim source must be an IntentSource")
        if not isinstance(self.origin, IntentOrigin):
            raise ValueError("intent claim origin must be an IntentOrigin")
        if not isinstance(self.confidence, IntentConfidence):
            raise ValueError("intent claim confidence must be an IntentConfidence")
        _validate_text_list(self.source_refs, "intent claim source_refs")
        _validate_text_list(self.evidence_refs, "intent claim evidence_refs")
        if not isinstance(self.claim_state, IntentClaimState):
            raise ValueError("intent claim claim_state must be an IntentClaimState")
        if not isinstance(self.conclusion_impact, ConclusionImpact):
            raise ValueError("intent claim conclusion_impact must be a ConclusionImpact")

        expected_id = _stable_intent_id(
            "claim", self.field.value, _canonical_intent_value(self.value)
        )
        if self.claim_id and self.claim_id != expected_id:
            raise ValueError("intent claim claim_id does not match its stable identity")
        object.__setattr__(self, "claim_id", expected_id)
        object.__setattr__(self, "source_refs", list(self.source_refs))
        object.__setattr__(self, "evidence_refs", list(self.evidence_refs))


@dataclass(frozen=True)
class ClarificationQuestion:
    field: IntentField
    question: str
    rationale: str
    proposed_values: list[str] = field(default_factory=list)
    claim_ids: list[str] = field(default_factory=list)
    status: ClarificationStatus = ClarificationStatus.PENDING
    user_response: str | None = None
    continuation_basis: str | None = None
    resolved_values: list[str] = field(default_factory=list)
    decision_id: str | None = None
    question_id: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.field, IntentField):
            raise ValueError("clarification field must be an IntentField")
        _validate_non_empty_text(self.question, "clarification question")
        _validate_non_empty_text(self.rationale, "clarification rationale")
        _validate_text_list(self.proposed_values, "clarification proposed_values")
        _validate_text_list(self.claim_ids, "clarification claim_ids")
        _validate_text_list(self.resolved_values, "clarification resolved_values")
        if not isinstance(self.status, ClarificationStatus):
            raise ValueError("clarification status must be a ClarificationStatus")
        if self.user_response is not None:
            _validate_non_empty_text(self.user_response, "clarification user_response")
        if self.continuation_basis is not None:
            _validate_non_empty_text(
                self.continuation_basis, "clarification continuation_basis"
            )

        expected_id = _stable_intent_id(
            "question",
            self.field.value,
            sorted(_canonical_intent_value(value) for value in self.proposed_values),
            sorted(self.claim_ids),
        )
        if self.question_id and self.question_id != expected_id:
            raise ValueError("clarification question_id does not match its stable identity")
        object.__setattr__(self, "question_id", expected_id)

        terminal = self.status not in {
            ClarificationStatus.PENDING,
            ClarificationStatus.OPEN,
        }
        if terminal:
            _validate_non_empty_text(self.decision_id, "clarification decision_id")
        elif any(
            value
            for value in (
                self.user_response,
                self.continuation_basis,
                self.decision_id,
                self.resolved_values,
            )
        ):
            raise ValueError("open clarification cannot contain decision result fields")
        if self.status is ClarificationStatus.CONFIRMED and not self.resolved_values:
            raise ValueError("confirmed clarification must contain resolved_values")
        if self.status is ClarificationStatus.CORRECTED:
            if not self.resolved_values:
                raise ValueError("corrected clarification must contain resolved_values")
            if self.user_response is None:
                raise ValueError("corrected clarification must contain user_response")
        if self.status in {
            ClarificationStatus.SKIPPED,
            ClarificationStatus.SKIPPED_NON_INTERACTIVE,
        } and self.continuation_basis is None:
            raise ValueError("skipped clarification must contain continuation_basis")

        object.__setattr__(self, "proposed_values", list(self.proposed_values))
        object.__setattr__(self, "claim_ids", list(self.claim_ids))
        object.__setattr__(self, "resolved_values", list(self.resolved_values))


@dataclass(frozen=True)
class IntentDecision:
    question_id: str
    action: IntentDecisionAction
    corrected_values: list[str] = field(default_factory=list)
    user_response: str | None = None
    continuation_basis: str | None = None
    decision_id: str = ""

    def __post_init__(self) -> None:
        _validate_non_empty_text(self.question_id, "intent decision question_id")
        if not isinstance(self.action, IntentDecisionAction):
            raise ValueError("intent decision action must be an IntentDecisionAction")
        _validate_text_list(self.corrected_values, "intent decision corrected_values")
        if self.user_response is not None:
            _validate_non_empty_text(self.user_response, "intent decision user_response")
        if self.continuation_basis is not None:
            _validate_non_empty_text(
                self.continuation_basis, "intent decision continuation_basis"
            )
        if self.action is IntentDecisionAction.CORRECTED:
            if not self.corrected_values:
                raise ValueError("corrected decision must contain corrected_values")
            if self.user_response is None:
                raise ValueError("corrected decision must contain user_response")
        elif self.corrected_values:
            raise ValueError("only corrected decisions may contain corrected_values")
        if self.action in {
            IntentDecisionAction.SKIPPED,
            IntentDecisionAction.SKIPPED_NON_INTERACTIVE,
        } and self.continuation_basis is None:
            raise ValueError("skipped decision must contain continuation_basis")
        if (
            self.action is IntentDecisionAction.CONFIRMED
            and self.continuation_basis
            not in {None, BENCHMARK_AUTO_ACCEPT_BASIS}
        ):
            raise ValueError(
                "confirmed decision continuation_basis must be "
                "benchmark_auto_accept"
            )

        expected_id = _stable_intent_id(
            "decision",
            self.question_id,
            self.action.value,
            [_canonical_intent_value(value) for value in self.corrected_values],
            self.user_response,
            self.continuation_basis,
        )
        if self.decision_id and self.decision_id != expected_id:
            raise ValueError("intent decision_id does not match its stable identity")
        object.__setattr__(self, "decision_id", expected_id)
        object.__setattr__(self, "corrected_values", list(self.corrected_values))


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class MemoryDiagnosticCode(str, Enum):
    UNAVAILABLE = "memory_unavailable"
    STALE = "stale"
    HARD_POLICY_OVERFLOW = "hard_policy_overflow"
    POLICY_REJECTED = "policy_rejected"


@dataclass(frozen=True)
class MemoryDiagnostic:
    """Visible, content-free degradation emitted at a stage boundary."""

    code: MemoryDiagnosticCode
    message: str
    memory_ids: tuple[str, ...] = ()
    blocking: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.code, MemoryDiagnosticCode):
            raise ValueError("memory diagnostic code must be a MemoryDiagnosticCode")
        _validate_non_empty_text(self.message, "memory diagnostic message")
        if type(self.blocking) is not bool:
            raise ValueError("memory diagnostic blocking must be a boolean")
        ids = (
            _canonical_memory_ids(self.memory_ids, "memory diagnostic memory_ids")
            if self.memory_ids
            else ()
        )
        mandatory_blocking = self.code in {
            MemoryDiagnosticCode.HARD_POLICY_OVERFLOW,
            MemoryDiagnosticCode.POLICY_REJECTED,
        }
        object.__setattr__(self, "memory_ids", ids)
        if mandatory_blocking:
            object.__setattr__(self, "blocking", True)

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code.value,
            "message": self.message,
            "memory_ids": list(self.memory_ids),
            "blocking": self.blocking,
        }

    def to_model_dict(
        self,
        *,
        excluded_memory_ids: tuple[str, ...] = (),
    ) -> dict[str, object] | None:
        """Return a diagnostic that cannot disclose an excluded Memory source."""

        excluded = frozenset(excluded_memory_ids)
        if any(memory_id in self.message for memory_id in excluded):
            return None
        visible_ids = tuple(
            memory_id for memory_id in self.memory_ids if memory_id not in excluded
        )
        if self.memory_ids and not visible_ids:
            return None
        return {
            "code": self.code.value,
            "message": self.message,
            "memory_ids": list(visible_ids),
            "blocking": self.blocking,
        }


def hard_policy_overflow_diagnostic(
    stage: str,
    policy_items: object,
    memory_ids: object,
    *,
    max_items: int = MAX_STAGE_HARD_POLICY_ITEMS,
    max_bytes: int = MAX_STAGE_HARD_POLICY_BYTES,
) -> MemoryDiagnostic | None:
    """Independently bound one stage's authoritative policy projection.

    The caller keeps the complete typed policy.  An overflow therefore becomes a
    blocking diagnostic instead of an implicit truncation or authority loss.
    """

    _validate_policy_identifier(stage, "hard policy stage")
    if type(max_items) is not int or max_items < 0:
        raise ValueError("max_items must be a non-negative integer")
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    if isinstance(policy_items, (str, bytes)) or not isinstance(
        policy_items, (list, tuple)
    ):
        raise ValueError("policy_items must be a list or tuple")
    items = tuple(policy_items)
    ids = (
        _canonical_memory_ids(memory_ids, "hard policy memory_ids")
        if memory_ids
        else ()
    )
    normalized_items: list[object] = []
    for item in items:
        try:
            encoded = json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as error:
            raise ValueError("policy_items must be canonically JSON serializable") from error
        normalized_items.append(json.loads(encoded))
    payload_bytes = len(
        json.dumps(
            {
                "stage": stage,
                "policy_items": normalized_items,
                "memory_ids": list(ids),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    required_items = max(len(items), len(ids))
    over_count = required_items > max_items
    over_bytes = payload_bytes > max_bytes
    if not over_count and not over_bytes:
        return None
    exceeded = (
        "records and bytes"
        if over_count and over_bytes
        else "records"
        if over_count
        else "bytes"
    )
    return MemoryDiagnostic(
        code=MemoryDiagnosticCode.HARD_POLICY_OVERFLOW,
        message=f"{stage} hard policy exceeds the stage projection {exceeded} budget",
        memory_ids=ids,
    )


@dataclass(frozen=True)
class MemoryReference:
    """The maximum common Memory identity exposed to authoritative stages."""

    memory_id: str
    kind: str
    source_refs: tuple[str, ...]
    local_only: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "memory_id",
            _validate_stable_identifier(
                self.memory_id,
                _STABLE_MEMORY_ID,
                "memory reference memory_id",
            ),
        )
        if self.kind not in _MEMORY_KINDS:
            raise ValueError("memory reference kind is unsupported")
        object.__setattr__(
            self,
            "source_refs",
            _canonical_text_tuple(
                self.source_refs,
                "memory reference source_refs",
                allow_empty=False,
            ),
        )
        if type(self.local_only) is not bool:
            raise ValueError("memory reference local_only must be a boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "kind": self.kind,
            "authority": "human_approved_project_memory",
            "source_refs": list(self.source_refs),
            "local_only": self.local_only,
        }

    def to_model_dict(self) -> dict[str, object]:
        if self.local_only:
            raise ValueError("local-only Memory cannot enter a model payload")
        return {
            "memory_id": self.memory_id,
            "kind": self.kind,
            "authority": "human_approved_project_memory",
            "source_refs": list(self.source_refs),
        }


@dataclass(frozen=True)
class IntentMemoryClaim:
    field: IntentField
    value: str
    memory: MemoryReference
    confidence: IntentConfidence = IntentConfidence.HIGH
    conclusion_impact: ConclusionImpact = ConclusionImpact.MATERIAL

    def __post_init__(self) -> None:
        if not isinstance(self.field, IntentField):
            raise ValueError("intent memory claim field must be an IntentField")
        _validate_non_empty_text(self.value, "intent memory claim value")
        if not isinstance(self.memory, MemoryReference):
            raise ValueError("intent memory claim memory must be a MemoryReference")
        if self.memory.kind not in {
            "architecture_boundary",
            "business_invariant",
            "compatibility_requirement",
        }:
            raise ValueError("intent memory claim uses a stage-disallowed memory kind")
        if not isinstance(self.confidence, IntentConfidence):
            raise ValueError("intent memory claim confidence must be IntentConfidence")
        if not isinstance(self.conclusion_impact, ConclusionImpact):
            raise ValueError(
                "intent memory claim conclusion_impact must be ConclusionImpact"
            )

    def to_dict(self, *, for_model: bool = False) -> dict[str, object]:
        return {
            "field": self.field.value,
            "value": self.value,
            "source": IntentSource.INFERRED.value,
            "origin": IntentOrigin.PROJECT_MEMORY.value,
            "confidence": self.confidence.value,
            "conclusion_impact": self.conclusion_impact.value,
            "memory": (
                self.memory.to_model_dict()
                if for_model
                else self.memory.to_dict()
            ),
        }


@dataclass(frozen=True)
class IntentMemoryProjection:
    claims: tuple[IntentMemoryClaim, ...] = ()
    diagnostics: tuple[MemoryDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        claims = tuple(self.claims)
        diagnostics = tuple(self.diagnostics)
        if any(not isinstance(item, IntentMemoryClaim) for item in claims):
            raise ValueError("intent memory claims must contain IntentMemoryClaim values")
        if any(not isinstance(item, MemoryDiagnostic) for item in diagnostics):
            raise ValueError("intent memory diagnostics must contain MemoryDiagnostic values")
        ids = [item.memory.memory_id for item in claims]
        if len(ids) != len(set(ids)):
            raise ValueError("intent memory claims must not repeat a memory_id")
        object.__setattr__(
            self,
            "claims",
            tuple(sorted(claims, key=lambda item: item.memory.memory_id)),
        )
        object.__setattr__(
            self,
            "diagnostics",
            tuple(sorted(diagnostics, key=lambda item: (item.code.value, item.message))),
        )

    def to_dict(self, *, for_model: bool = False) -> dict[str, object]:
        claims = (
            tuple(item for item in self.claims if not item.memory.local_only)
            if for_model
            else self.claims
        )
        return {
            "schema_version": "intent_memory_projection_v1",
            "claims": [item.to_dict(for_model=for_model) for item in claims],
            "diagnostics": [
                rendered
                for item in self.diagnostics
                if (
                    rendered := (
                        item.to_model_dict(
                            excluded_memory_ids=tuple(
                                claim.memory.memory_id
                                for claim in self.claims
                                if claim.memory.local_only
                            )
                        )
                        if for_model
                        else item.to_dict()
                    )
                )
                is not None
            ],
        }


@dataclass(frozen=True)
class MemoryRiskSignal:
    signal_ref: str
    summary: str
    memory: MemoryReference

    def __post_init__(self) -> None:
        if not isinstance(self.memory, MemoryReference):
            raise ValueError("memory risk signal memory must be a MemoryReference")
        if self.memory.kind not in {"incident_lesson", "high_risk_module"}:
            raise ValueError("memory risk signal uses a stage-disallowed memory kind")
        expected_ref = f"memory:{self.memory.memory_id}"
        if self.signal_ref != expected_ref:
            raise ValueError("memory risk signal_ref must bind its memory_id")
        _validate_non_empty_text(self.summary, "memory risk signal summary")

    def to_dict(self) -> dict[str, object]:
        return {
            "signal_ref": self.signal_ref,
            "summary": self.summary,
            "memory": self.memory.to_dict(),
        }


@dataclass(frozen=True)
class CompiledRiskFloor:
    minimum_level: RiskLevel
    memory_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.minimum_level, RiskLevel):
            raise ValueError("compiled risk floor minimum_level must be a RiskLevel")
        object.__setattr__(
            self,
            "memory_ids",
            _canonical_memory_ids(self.memory_ids, "compiled risk floor memory_ids"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "minimum_level": self.minimum_level.value,
            "memory_ids": list(self.memory_ids),
        }


@dataclass(frozen=True)
class RiskMemoryProjection:
    signals: tuple[MemoryRiskSignal, ...] = ()
    risk_floor: CompiledRiskFloor | None = None
    policy_sources: tuple[MemoryReference, ...] = ()
    diagnostics: tuple[MemoryDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        signals = tuple(self.signals)
        policy_sources = tuple(self.policy_sources)
        diagnostics = tuple(self.diagnostics)
        if any(not isinstance(item, MemoryRiskSignal) for item in signals):
            raise ValueError("risk memory signals must contain MemoryRiskSignal values")
        if self.risk_floor is not None and not isinstance(
            self.risk_floor, CompiledRiskFloor
        ):
            raise ValueError("risk_floor must be a CompiledRiskFloor or None")
        if any(not isinstance(item, MemoryDiagnostic) for item in diagnostics):
            raise ValueError("risk memory diagnostics must contain MemoryDiagnostic values")
        if any(not isinstance(item, MemoryReference) for item in policy_sources):
            raise ValueError("risk policy_sources must contain MemoryReference values")
        refs = [item.signal_ref for item in signals]
        if len(refs) != len(set(refs)):
            raise ValueError("risk memory signals must not repeat signal_ref")
        source_ids = [item.memory_id for item in policy_sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("risk policy_sources must not repeat a memory_id")
        policy_sources = tuple(
            sorted(policy_sources, key=lambda item: item.memory_id)
        )
        if self.risk_floor is not None:
            signal_sources = {
                item.memory.memory_id: item.memory for item in signals
            }
            policy_by_id = {item.memory_id: item for item in policy_sources}
            for memory_id in self.risk_floor.memory_ids:
                if memory_id not in policy_by_id and memory_id in signal_sources:
                    policy_by_id[memory_id] = signal_sources[memory_id]
            policy_sources = tuple(
                policy_by_id[memory_id] for memory_id in sorted(policy_by_id)
            )
        object.__setattr__(
            self,
            "signals",
            tuple(sorted(signals, key=lambda item: item.signal_ref)),
        )
        object.__setattr__(self, "policy_sources", policy_sources)
        object.__setattr__(
            self,
            "diagnostics",
            tuple(sorted(diagnostics, key=lambda item: (item.code.value, item.message))),
        )

    @property
    def local_only_memory_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    item.memory.memory_id
                    for item in self.signals
                    if item.memory.local_only
                }
                | {
                    item.memory_id
                    for item in self.policy_sources
                    if item.local_only
                }
            )
        )

    @property
    def memory_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {item.memory.memory_id for item in self.signals}
                | {item.memory_id for item in self.policy_sources}
                | (
                    set(self.risk_floor.memory_ids)
                    if self.risk_floor is not None
                    else set()
                )
            )
        )

    def to_dict(self, *, for_model: bool = False) -> dict[str, object]:
        if for_model:
            # Informational statements and the authoritative local floor are kept
            # out of the same model channel. Runtime applies the floor locally.
            return {
                "schema_version": "risk_memory_projection_v1",
                "diagnostics": [
                    rendered
                    for item in self.diagnostics
                    if (
                        rendered := item.to_model_dict(
                            excluded_memory_ids=self.local_only_memory_ids
                        )
                    )
                    is not None
                ],
            }
        return {
            "schema_version": "risk_memory_projection_v1",
            "signals": [item.to_dict() for item in self.signals],
            "risk_floor": None if self.risk_floor is None else self.risk_floor.to_dict(),
            "policy_sources": [item.to_dict() for item in self.policy_sources],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


@dataclass(frozen=True)
class CompiledMemoryRequirement:
    requirement_id: str
    memory_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "requirement_id",
            _validate_policy_identifier(self.requirement_id, "requirement_id"),
        )
        object.__setattr__(
            self,
            "memory_ids",
            _canonical_memory_ids(self.memory_ids, "requirement memory_ids"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "requirement_id": self.requirement_id,
            "memory_ids": list(self.memory_ids),
        }


@dataclass(frozen=True)
class VerificationTemplateHint:
    command_template_id: str
    memory_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "command_template_id",
            _validate_policy_identifier(
                self.command_template_id,
                "command_template_id",
            ),
        )
        object.__setattr__(
            self,
            "memory_ids",
            _canonical_memory_ids(self.memory_ids, "verification hint memory_ids"),
        )

    def to_dict(self) -> dict[str, object]:
        # Deliberately no command, executable, environment, or callable field.
        return {
            "command_template_id": self.command_template_id,
            "memory_ids": list(self.memory_ids),
        }


@dataclass(frozen=True)
class PlannerPerspectiveHint:
    perspective_id: str
    source_feedback_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "perspective_id",
            _validate_policy_identifier(self.perspective_id, "perspective_id"),
        )
        object.__setattr__(
            self,
            "source_feedback_ids",
            _canonical_feedback_ids(
                self.source_feedback_ids,
                "perspective hint source_feedback_ids",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "perspective_id": self.perspective_id,
            "source_feedback_ids": list(self.source_feedback_ids),
        }


def _canonical_typed_tuple(
    values: object,
    expected_type: type,
    name: str,
    key,
) -> tuple:
    if isinstance(values, (str, bytes)) or not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a list or tuple")
    items = tuple(values)
    if any(not isinstance(item, expected_type) for item in items):
        raise ValueError(f"{name} contains an unsupported value")
    keys = [key(item) for item in items]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{name} must not repeat its typed identity")
    return tuple(sorted(items, key=key))


@dataclass(frozen=True)
class PlannerMemoryProjection:
    required_contracts: tuple[CompiledMemoryRequirement, ...] = ()
    required_checks: tuple[CompiledMemoryRequirement, ...] = ()
    verification_hints: tuple[VerificationTemplateHint, ...] = ()
    perspective_hints: tuple[PlannerPerspectiveHint, ...] = ()
    selected_memory: tuple[MemoryReference, ...] = ()
    diagnostics: tuple[MemoryDiagnostic, ...] = ()
    feedback_summary_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "required_contracts",
            _canonical_typed_tuple(
                self.required_contracts,
                CompiledMemoryRequirement,
                "required_contracts",
                lambda item: item.requirement_id,
            ),
        )
        object.__setattr__(
            self,
            "required_checks",
            _canonical_typed_tuple(
                self.required_checks,
                CompiledMemoryRequirement,
                "required_checks",
                lambda item: item.requirement_id,
            ),
        )
        object.__setattr__(
            self,
            "verification_hints",
            _canonical_typed_tuple(
                self.verification_hints,
                VerificationTemplateHint,
                "verification_hints",
                lambda item: item.command_template_id,
            ),
        )
        object.__setattr__(
            self,
            "perspective_hints",
            _canonical_typed_tuple(
                self.perspective_hints,
                PlannerPerspectiveHint,
                "perspective_hints",
                lambda item: item.perspective_id,
            ),
        )
        object.__setattr__(
            self,
            "selected_memory",
            _canonical_typed_tuple(
                self.selected_memory,
                MemoryReference,
                "selected_memory",
                lambda item: item.memory_id,
            ),
        )
        object.__setattr__(
            self,
            "diagnostics",
            _canonical_typed_tuple(
                self.diagnostics,
                MemoryDiagnostic,
                "memory diagnostics",
                lambda item: (item.code.value, item.message, item.memory_ids),
            ),
        )
        if self.feedback_summary_hash is not None:
            if (
                not isinstance(self.feedback_summary_hash, str)
                or re.fullmatch(r"[0-9a-f]{64}", self.feedback_summary_hash) is None
            ):
                raise ValueError("feedback_summary_hash must be a SHA-256 digest or None")
        selected_ids = {item.memory_id for item in self.selected_memory}
        policy_ids = {
            memory_id
            for item in (
                *self.required_contracts,
                *self.required_checks,
                *self.verification_hints,
            )
            for memory_id in item.memory_ids
        }
        if not policy_ids.issubset(selected_ids):
            raise ValueError(
                "planner policy provenance must be present in selected_memory"
            )

    @property
    def local_only_memory_ids(self) -> tuple[str, ...]:
        return tuple(
            item.memory_id for item in self.selected_memory if item.local_only
        )

    def to_dict(self, *, for_model: bool = False) -> dict[str, object]:
        if for_model:
            return {
                "schema_version": "planner_memory_projection_v1",
                "required_contracts": [
                    {"requirement_id": item.requirement_id}
                    for item in self.required_contracts
                ],
                "required_checks": [
                    {"requirement_id": item.requirement_id}
                    for item in self.required_checks
                ],
                "verification_hints": [
                    {"command_template_id": item.command_template_id}
                    for item in self.verification_hints
                ],
                "perspective_hints": [
                    {"perspective_id": item.perspective_id}
                    for item in self.perspective_hints
                ],
                "diagnostics": [
                    rendered
                    for item in self.diagnostics
                    if (
                        rendered := item.to_model_dict(
                            excluded_memory_ids=self.local_only_memory_ids
                        )
                    )
                    is not None
                ],
                "feedback_summary_hash": self.feedback_summary_hash,
            }
        return {
            "schema_version": "planner_memory_projection_v1",
            "required_contracts": [item.to_dict() for item in self.required_contracts],
            "required_checks": [item.to_dict() for item in self.required_checks],
            "verification_hints": [item.to_dict() for item in self.verification_hints],
            "perspective_hints": [item.to_dict() for item in self.perspective_hints],
            "selected_memory": [item.to_dict() for item in self.selected_memory],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "feedback_summary_hash": self.feedback_summary_hash,
        }


@dataclass(frozen=True)
class CompletionMemoryProjection:
    required_contracts: tuple[CompiledMemoryRequirement, ...] = ()
    required_checks: tuple[CompiledMemoryRequirement, ...] = ()
    diagnostics: tuple[MemoryDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "required_contracts",
            _canonical_typed_tuple(
                self.required_contracts,
                CompiledMemoryRequirement,
                "completion required_contracts",
                lambda item: item.requirement_id,
            ),
        )
        object.__setattr__(
            self,
            "required_checks",
            _canonical_typed_tuple(
                self.required_checks,
                CompiledMemoryRequirement,
                "completion required_checks",
                lambda item: item.requirement_id,
            ),
        )
        object.__setattr__(
            self,
            "diagnostics",
            _canonical_typed_tuple(
                self.diagnostics,
                MemoryDiagnostic,
                "completion diagnostics",
                lambda item: (item.code.value, item.message, item.memory_ids),
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "completion_memory_projection_v1",
            "required_contracts": [item.to_dict() for item in self.required_contracts],
            "required_checks": [item.to_dict() for item in self.required_checks],
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }


@dataclass(frozen=True)
class FinalRiskMemoryProjection:
    applied_memory: tuple[MemoryReference, ...] = ()
    risk_signals: tuple[MemoryRiskSignal, ...] = ()
    risk_floor: CompiledRiskFloor | None = None
    diagnostics: tuple[MemoryDiagnostic, ...] = ()
    residual_risk: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "applied_memory",
            _canonical_typed_tuple(
                self.applied_memory,
                MemoryReference,
                "applied_memory",
                lambda item: item.memory_id,
            ),
        )
        object.__setattr__(
            self,
            "risk_signals",
            _canonical_typed_tuple(
                self.risk_signals,
                MemoryRiskSignal,
                "final risk signals",
                lambda item: item.signal_ref,
            ),
        )
        if self.risk_floor is not None and not isinstance(
            self.risk_floor, CompiledRiskFloor
        ):
            raise ValueError("final risk_floor must be a CompiledRiskFloor or None")
        object.__setattr__(
            self,
            "diagnostics",
            _canonical_typed_tuple(
                self.diagnostics,
                MemoryDiagnostic,
                "final risk diagnostics",
                lambda item: (item.code.value, item.message, item.memory_ids),
            ),
        )
        object.__setattr__(
            self,
            "residual_risk",
            _canonical_text_tuple(self.residual_risk, "residual_risk"),
        )
        applied_ids = {item.memory_id for item in self.applied_memory}
        if any(item.memory.memory_id not in applied_ids for item in self.risk_signals):
            raise ValueError("final risk signals must reference applied memory")
        # ``risk_floor.memory_ids`` is itself the authoritative provenance. A
        # separately projected reference is desirable but cannot be required:
        # doing so would reintroduce kind/sensitivity filtering at this boundary.

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "final_risk_memory_projection_v1",
            "applied_memory": [item.to_dict() for item in self.applied_memory],
            "risk_signals": [item.to_dict() for item in self.risk_signals],
            "risk_floor": None if self.risk_floor is None else self.risk_floor.to_dict(),
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "residual_risk": list(self.residual_risk),
        }


# Stage-first and Memory-first spellings are both intentionally stable for callers.
MemoryIntentProjection = IntentMemoryProjection
MemoryRiskProjection = RiskMemoryProjection
MemoryPlannerProjection = PlannerMemoryProjection
MemoryCompletionProjection = CompletionMemoryProjection
MemoryFinalRiskProjection = FinalRiskMemoryProjection


class ContractItemStatus(str, Enum):
    COVERED = "covered"
    PARTIAL = "partial"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class ReviewerResultStatus(str, Enum):
    COMPLETED = "completed"
    PARTIAL = "partial"
    BLOCKED = "blocked"
    FAILED = "failed"


class ReviewerTerminationReason(str, Enum):
    LEGACY_UNKNOWN = "legacy_unknown"
    COMPLETED = "completed"
    REVIEWER_PARTIAL = "reviewer_partial"
    REVIEWER_BLOCKED = "reviewer_blocked"
    PROVIDER_RETRY_EXHAUSTED = "provider_retry_exhausted"
    TURN_BUDGET_EXHAUSTED = "turn_budget_exhausted"
    TOOL_BUDGET_EXHAUSTED = "tool_budget_exhausted"
    TOKEN_BUDGET_EXHAUSTED = "token_budget_exhausted"
    TIME_BUDGET_EXHAUSTED = "time_budget_exhausted"
    RUNTIME_FAILURE = "runtime_failure"


DEFAULT_REVIEWER_MAX_OUTPUT_TOKENS = 8192
DEFAULT_REVIEWER_MAX_TOTAL_TOKENS = 65536
DEFAULT_REVIEWER_MAX_ELAPSED_SECONDS = 300.0
DEFAULT_REVIEWER_MAX_PROVIDER_ATTEMPTS = 2


@dataclass(frozen=True)
class ReviewRequest:
    repository_path: str
    base_revision: str
    head_revision: str
    title: str | None = None
    description: str | None = None
    linked_requirements: tuple[str, ...] = ()
    user_intent: str | None = None
    review_focus: str | None = None
    project_rules: tuple[str, ...] = ()
    existing_ci_evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class IntentPacket:
    goal: str | None
    acceptance_criteria: list[str] = field(default_factory=list)
    scope: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    sources: dict[str, IntentSource] = field(default_factory=dict)
    status: IntentStatus = IntentStatus.INSUFFICIENT
    uncertainties: list[str] = field(default_factory=list)
    provenance: list[IntentClaim] = field(default_factory=list)
    clarifications: list[ClarificationQuestion] = field(default_factory=list)


@dataclass(frozen=True)
class QualityGateResult:
    name: str
    status: str
    command: list[str]
    summary: str
    observation_ref: str | None = None
    category: str = "unknown"
    cost: str = "cheap"
    source: str = "legacy"
    blocking: bool = False
    reason: str | None = None
    exit_code: int | None = None
    duration_seconds: float = 0.0
    output_truncated: bool = False
    sandbox: str = "legacy"

    def __post_init__(self) -> None:
        _validate_non_empty_text(self.name, "quality gate name")
        if self.status not in {
            "passed",
            "failed",
            "skipped",
            "unavailable",
            "timed_out",
            "error",
        }:
            raise ValueError("quality gate status is unsupported")
        if not isinstance(self.command, list) or not self.command:
            raise ValueError("quality gate command must be a non-empty list")
        for argument in self.command:
            _validate_non_empty_text(argument, "quality gate command argument")
        _validate_non_empty_text(self.summary, "quality gate summary")
        if self.observation_ref is not None:
            _validate_non_empty_text(
                self.observation_ref,
                "quality gate observation_ref",
            )
        if self.category not in {
            "compile",
            "format",
            "type",
            "lint",
            "build",
            "test",
            "security",
            "unknown",
        }:
            raise ValueError("quality gate category is unsupported")
        if self.cost not in {"cheap", "expensive"}:
            raise ValueError("quality gate cost is unsupported")
        if self.source not in {"builtin", "repository_config", "legacy"}:
            raise ValueError("quality gate source is unsupported")
        if type(self.blocking) is not bool:
            raise ValueError("quality gate blocking must be a boolean")
        if self.reason is not None:
            _validate_non_empty_text(self.reason, "quality gate reason")
        if self.status in {"skipped", "unavailable", "timed_out", "error"} and (
            self.reason is None
        ):
            raise ValueError(
                f"quality gate status {self.status} must include a reason"
            )
        if self.exit_code is not None and type(self.exit_code) is not int:
            raise ValueError("quality gate exit_code must be an integer or null")
        if (
            isinstance(self.duration_seconds, bool)
            or not isinstance(self.duration_seconds, (int, float))
            or not math.isfinite(self.duration_seconds)
            or self.duration_seconds < 0
        ):
            raise ValueError(
                "quality gate duration_seconds must be a finite non-negative number"
            )
        if type(self.output_truncated) is not bool:
            raise ValueError("quality gate output_truncated must be a boolean")
        _validate_non_empty_text(self.sandbox, "quality gate sandbox")
        object.__setattr__(self, "command", list(self.command))
        object.__setattr__(self, "duration_seconds", float(self.duration_seconds))


@dataclass(frozen=True)
class RiskAssessmentPacket:
    change_summary: dict[str, object]
    deterministic_signals: dict[str, object]
    intent_status: IntentStatus
    intent_uncertainties: list[str]
    diff_excerpt: list[str]
    changed_symbols: list[dict[str, object]] = field(default_factory=list)
    signal_catalog: dict[str, str] = field(default_factory=dict)
    memory_projection: RiskMemoryProjection | None = None

    def __post_init__(self) -> None:
        if self.memory_projection is not None and not isinstance(
            self.memory_projection,
            RiskMemoryProjection,
        ):
            raise ValueError(
                "memory_projection must be a RiskMemoryProjection or None"
            )


@dataclass(frozen=True)
class RiskAssessment:
    level: RiskLevel
    dimensions: dict[str, str]
    reasons: list[str]
    signal_refs: list[str]
    uncertainties: list[str]
    suggested_focus: list[str]


@dataclass(frozen=True)
class InitialContext:
    changed_files: list[str] = field(default_factory=list)
    diff_ranges: list[str] = field(default_factory=list)
    code_ranges: list[str] = field(default_factory=list)
    quality_gate_summary: dict[str, str] = field(default_factory=dict)
    observation_refs: list[str] = field(default_factory=list)
    signal_refs: list[str] = field(default_factory=list)
    selected_memory_refs: list[str] = field(default_factory=list)
    verification_template_hints: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        _validate_text_list(self.selected_memory_refs, "selected_memory_refs")
        _validate_text_list(
            self.verification_template_hints,
            "verification_template_hints",
        )
        for memory_id in self.selected_memory_refs:
            _validate_stable_identifier(
                memory_id,
                _STABLE_MEMORY_ID,
                "selected_memory_refs item",
            )
        for template_id in self.verification_template_hints:
            _validate_policy_identifier(
                template_id,
                "verification_template_hints item",
            )
        object.__setattr__(self, "selected_memory_refs", list(self.selected_memory_refs))
        object.__setattr__(
            self,
            "verification_template_hints",
            list(self.verification_template_hints),
        )


@dataclass(frozen=True)
class ReviewProfile:
    reviewer_count: int
    max_turns_per_reviewer: int
    max_tool_calls_per_reviewer: int
    reviewer_roles: list[str]
    max_output_tokens: int = DEFAULT_REVIEWER_MAX_OUTPUT_TOKENS
    max_total_tokens: int = DEFAULT_REVIEWER_MAX_TOTAL_TOKENS
    max_elapsed_seconds: float = DEFAULT_REVIEWER_MAX_ELAPSED_SECONDS
    max_provider_attempts: int = DEFAULT_REVIEWER_MAX_PROVIDER_ATTEMPTS

    @classmethod
    def for_risk(cls, risk: RiskLevel) -> "ReviewProfile":
        profiles = {
            RiskLevel.LOW: cls(
                1,
                6,
                12,
                ["core"],
                max_output_tokens=8192,
                max_total_tokens=32768,
                max_elapsed_seconds=120.0,
                max_provider_attempts=2,
            ),
            RiskLevel.MEDIUM: cls(
                2,
                10,
                24,
                ["core", "adversarial"],
                max_output_tokens=8192,
                max_total_tokens=65536,
                max_elapsed_seconds=300.0,
                max_provider_attempts=2,
            ),
            RiskLevel.HIGH: cls(
                3,
                16,
                40,
                ["core", "adversarial", "dynamic_specialist"],
                max_output_tokens=8192,
                max_total_tokens=131072,
                max_elapsed_seconds=600.0,
                max_provider_attempts=3,
            ),
            RiskLevel.CRITICAL: cls(
                4,
                24,
                64,
                ["core", "adversarial", "security_specialist", "domain_specialist"],
                max_output_tokens=8192,
                max_total_tokens=262144,
                max_elapsed_seconds=900.0,
                max_provider_attempts=3,
            ),
        }
        return profiles[risk]


@dataclass(frozen=True)
class Assignment:
    role: str
    mission: str
    assignment_reason: list[str]
    assigned_contract: list[str]
    required_checks: list[str]
    initial_context: InitialContext
    max_turns: int
    max_tool_calls: int
    max_output_tokens: int = DEFAULT_REVIEWER_MAX_OUTPUT_TOKENS
    max_total_tokens: int = DEFAULT_REVIEWER_MAX_TOTAL_TOKENS
    max_elapsed_seconds: float = DEFAULT_REVIEWER_MAX_ELAPSED_SECONDS
    max_provider_attempts: int = DEFAULT_REVIEWER_MAX_PROVIDER_ATTEMPTS
    repository_permission: str = "read_only"
    command_permission: str = "safe_checks_only"
    assignment_id: str = ""
    role_kind: str = "legacy"
    perspective_key: str = "legacy"
    planner_source: str = "legacy"

    def __post_init__(self) -> None:
        _validate_non_empty_text(self.role, "assignment role")
        _validate_non_empty_text(self.mission, "assignment mission")
        _validate_text_list(self.assignment_reason, "assignment_reason")
        _validate_text_list(self.assigned_contract, "assigned_contract")
        _validate_text_list(self.required_checks, "required_checks")
        if not isinstance(self.initial_context, InitialContext):
            raise ValueError("initial_context must be an InitialContext")
        if type(self.max_turns) is not int or self.max_turns <= 0:
            raise ValueError("max_turns must be a positive integer")
        if type(self.max_tool_calls) is not int or self.max_tool_calls < 0:
            raise ValueError("max_tool_calls must be a non-negative integer")
        integer_budgets = {
            "max_output_tokens": self.max_output_tokens,
            "max_total_tokens": self.max_total_tokens,
            "max_provider_attempts": self.max_provider_attempts,
        }
        for name, value in integer_budgets.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(self.max_elapsed_seconds, bool)
            or not isinstance(self.max_elapsed_seconds, (int, float))
            or not math.isfinite(self.max_elapsed_seconds)
            or self.max_elapsed_seconds <= 0
        ):
            raise ValueError("max_elapsed_seconds must be a positive number")
        if self.repository_permission != "read_only":
            raise ValueError("repository_permission must be read_only")
        if self.command_permission != "safe_checks_only":
            raise ValueError("command_permission must be safe_checks_only")
        if self.assignment_id:
            _validate_non_empty_text(self.assignment_id, "assignment_id")
        if self.role_kind not in {"legacy", "core", "adversarial", "specialist"}:
            raise ValueError("role_kind is unsupported")
        _validate_non_empty_text(self.perspective_key, "perspective_key")
        if self.planner_source not in {
            "legacy",
            "local",
            "model",
            "runtime_injected",
            "semantic_reconciler",
        }:
            raise ValueError("planner_source is unsupported")
        object.__setattr__(self, "assignment_reason", list(self.assignment_reason))
        object.__setattr__(self, "assigned_contract", list(self.assigned_contract))
        object.__setattr__(self, "required_checks", list(self.required_checks))


@dataclass(frozen=True)
class ReviewerRuntimeMetadata:
    provider_attempts: int = 0
    model_turns: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    usage_available: bool = False
    elapsed_seconds: float = 0.0
    termination_reason: ReviewerTerminationReason = ReviewerTerminationReason.LEGACY_UNKNOWN

    def __post_init__(self) -> None:
        counters = {
            "provider_attempts": self.provider_attempts,
            "model_turns": self.model_turns,
            "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }
        for name, value in counters.items():
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if type(self.usage_available) is not bool:
            raise ValueError("usage_available must be a boolean")
        if (
            isinstance(self.elapsed_seconds, bool)
            or not isinstance(self.elapsed_seconds, (int, float))
            or not math.isfinite(self.elapsed_seconds)
            or self.elapsed_seconds < 0
        ):
            raise ValueError("elapsed_seconds must be a non-negative number")
        if not isinstance(self.termination_reason, ReviewerTerminationReason):
            raise ValueError("termination_reason must be a ReviewerTerminationReason")


@dataclass(frozen=True)
class ModelInvocationEnvelope:
    system: str
    tools: list[dict[str, object]]
    messages: list[dict[str, object]]
    parameters: dict[str, object]


@dataclass(frozen=True)
class ContractAssessment:
    contract: str
    status: ContractItemStatus
    summary: str
    evidence_refs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReviewerFinding:
    claim: str
    severity: str
    confidence: str
    evidence_refs: list[str] = field(default_factory=list)
    suggested_action: str | None = None
    path: str | None = None
    line: int | None = None
    impact: str = ""
    verification_performed: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReviewerResult:
    contract_assessments: list[ContractAssessment] = field(default_factory=list)
    confirmed_findings: list[ReviewerFinding] = field(default_factory=list)
    rejected_hypotheses: list[str] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    observation_refs: list[str] = field(default_factory=list)
    investigation_summary: str = ""
    status: ReviewerResultStatus = ReviewerResultStatus.PARTIAL
