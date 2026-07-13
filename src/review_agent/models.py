from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum


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
    USER_CORRECTION = "user_correction"
    CHANGED_FILES = "changed_files"


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


DEFAULT_REVIEWER_MAX_OUTPUT_TOKENS = 4096
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


@dataclass(frozen=True)
class RiskAssessmentPacket:
    change_summary: dict[str, object]
    deterministic_signals: dict[str, object]
    intent_status: IntentStatus
    intent_uncertainties: list[str]
    diff_excerpt: list[str]


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
                max_output_tokens=4096,
                max_total_tokens=32768,
                max_elapsed_seconds=120.0,
                max_provider_attempts=2,
            ),
            RiskLevel.MEDIUM: cls(
                2,
                10,
                24,
                ["core", "adversarial"],
                max_output_tokens=4096,
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

    def __post_init__(self) -> None:
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
