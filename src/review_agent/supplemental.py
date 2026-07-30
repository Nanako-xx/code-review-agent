from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import re
from typing import Any

from review_agent.context import (
    REVIEWER_TOOL_NAMES,
    normalize_reviewer_allowed_tools,
)
from review_agent.models import (
    Assignment,
    DEFAULT_REVIEWER_MAX_OUTPUT_TOKENS,
    InitialContext,
    RiskLevel,
)
from review_agent.session import SupplementalPolicy


SUPPLEMENTAL_POLICY_VERSION = "supplemental_policy_v1"
SUPPLEMENTAL_PLAN_SCHEMA_VERSION = "supplemental_plan_v1"
SUPPLEMENTAL_CONTRACT_PREFIX = "supplemental_investigation:"
DEFAULT_SUPPLEMENTAL_ALLOWED_TOOLS = REVIEWER_TOOL_NAMES

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_STOP_REASONS = {
    "resolved",
    "no_requests",
    "model_fallback",
    "task_failure",
    "budget_exhausted",
    "max_waves",
    "unavailable",
}


def _clean_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return " ".join(value.split())


def _require_non_negative_number(value: object, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative finite number")


@dataclass(frozen=True)
class SupplementalInvestigationRequest:
    source_disagreement_id: str
    question: str
    required_evidence: tuple[str, ...]
    preferred_perspective: str
    source_candidate_ids: tuple[str, ...]
    reason_refs: tuple[str, ...] = ()
    request_id: str = ""

    def __post_init__(self) -> None:
        _require_safe_id(self.source_disagreement_id, "source_disagreement_id")
        question = _clean_text(self.question, "question")
        perspective = _clean_text(
            self.preferred_perspective,
            "preferred_perspective",
        )
        required_evidence = _canonical_text_items(
            self.required_evidence,
            "required_evidence",
        )
        candidate_ids = _canonical_ids(
            self.source_candidate_ids,
            "source_candidate_ids",
            allow_empty=False,
        )
        reason_refs = _canonical_ids(
            self.reason_refs,
            "reason_refs",
            allow_empty=True,
        )
        computed_id = _request_id(
            question,
            required_evidence,
            perspective,
            candidate_ids,
        )
        if self.request_id and self.request_id != computed_id:
            raise ValueError("request_id does not match request content")
        object.__setattr__(self, "question", question)
        object.__setattr__(self, "preferred_perspective", perspective)
        object.__setattr__(self, "required_evidence", required_evidence)
        object.__setattr__(self, "source_candidate_ids", candidate_ids)
        object.__setattr__(self, "reason_refs", reason_refs)
        object.__setattr__(self, "request_id", computed_id)


@dataclass(frozen=True)
class SupplementalRuntimeLimits:
    max_waves: int
    max_tasks: int
    max_tasks_per_wave: int
    max_concurrency: int
    max_turns_per_task: int
    max_tool_calls_per_task: int
    max_total_tokens_per_task: int
    max_total_tokens: int
    max_elapsed_seconds: float
    policy_version: str = SUPPLEMENTAL_POLICY_VERSION

    def __post_init__(self) -> None:
        integer_fields = {
            "max_waves": self.max_waves,
            "max_tasks": self.max_tasks,
            "max_tasks_per_wave": self.max_tasks_per_wave,
            "max_concurrency": self.max_concurrency,
            "max_turns_per_task": self.max_turns_per_task,
            "max_tool_calls_per_task": self.max_tool_calls_per_task,
            "max_total_tokens_per_task": self.max_total_tokens_per_task,
            "max_total_tokens": self.max_total_tokens,
        }
        for name, value in integer_fields.items():
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        _require_non_negative_number(
            self.max_elapsed_seconds,
            "max_elapsed_seconds",
        )
        _clean_text(self.policy_version, "policy_version")
        object.__setattr__(self, "max_elapsed_seconds", float(self.max_elapsed_seconds))

    @property
    def budget_limits(self) -> BudgetAmount:
        return BudgetAmount(
            tasks=self.max_tasks,
            tool_calls=self.max_tasks * self.max_tool_calls_per_task,
            tokens=self.max_total_tokens,
            elapsed_seconds=self.max_elapsed_seconds,
        )


@dataclass(frozen=True)
class ReviewerBudgetCaps:
    max_output_tokens: int = DEFAULT_REVIEWER_MAX_OUTPUT_TOKENS
    max_total_tokens: int = 2**63 - 1
    max_elapsed_seconds: float = 1_000_000_000.0
    max_provider_attempts: int = 2

    def __post_init__(self) -> None:
        for name, value in {
            "max_output_tokens": self.max_output_tokens,
            "max_total_tokens": self.max_total_tokens,
            "max_provider_attempts": self.max_provider_attempts,
        }.items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        _require_positive_number(
            self.max_elapsed_seconds,
            "max_elapsed_seconds",
        )
        object.__setattr__(self, "max_elapsed_seconds", float(self.max_elapsed_seconds))


@dataclass(frozen=True)
class BudgetAmount:
    tasks: int = 0
    tool_calls: int = 0
    tokens: int = 0
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name, value in {
            "tasks": self.tasks,
            "tool_calls": self.tool_calls,
            "tokens": self.tokens,
        }.items():
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        _require_non_negative_number(self.elapsed_seconds, "elapsed_seconds")
        object.__setattr__(self, "elapsed_seconds", float(self.elapsed_seconds))

    def __add__(self, other: BudgetAmount) -> BudgetAmount:
        if not isinstance(other, BudgetAmount):
            return NotImplemented
        return BudgetAmount(
            tasks=self.tasks + other.tasks,
            tool_calls=self.tool_calls + other.tool_calls,
            tokens=self.tokens + other.tokens,
            elapsed_seconds=self.elapsed_seconds + other.elapsed_seconds,
        )

    def subtract_floor(self, other: BudgetAmount) -> BudgetAmount:
        return BudgetAmount(
            tasks=max(0, self.tasks - other.tasks),
            tool_calls=max(0, self.tool_calls - other.tool_calls),
            tokens=max(0, self.tokens - other.tokens),
            elapsed_seconds=max(0.0, self.elapsed_seconds - other.elapsed_seconds),
        )

    def max_with(self, other: BudgetAmount) -> BudgetAmount:
        return BudgetAmount(
            tasks=max(self.tasks, other.tasks),
            tool_calls=max(self.tool_calls, other.tool_calls),
            tokens=max(self.tokens, other.tokens),
            elapsed_seconds=max(self.elapsed_seconds, other.elapsed_seconds),
        )


@dataclass(frozen=True)
class UnknownConsumption:
    tokens: int = 0
    elapsed_seconds: float = 0.0
    invocation_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if type(self.tokens) is not int or self.tokens < 0:
            raise ValueError("tokens must be a non-negative integer")
        _require_non_negative_number(self.elapsed_seconds, "elapsed_seconds")
        invocation_ids = _canonical_ids(
            self.invocation_ids,
            "invocation_ids",
            allow_empty=True,
        )
        object.__setattr__(self, "elapsed_seconds", float(self.elapsed_seconds))
        object.__setattr__(self, "invocation_ids", invocation_ids)

    @property
    def budget_amount(self) -> BudgetAmount:
        return BudgetAmount(
            tokens=self.tokens,
            elapsed_seconds=self.elapsed_seconds,
        )


class BudgetExceededError(RuntimeError):
    pass


class BudgetLedger:
    """Main-thread ledger for durable reservation, charging, and crash windows."""

    def __init__(
        self,
        *,
        limits: BudgetAmount,
        charged: BudgetAmount | None = None,
        unknown_consumed: UnknownConsumption | None = None,
        stop_reason: str | None = None,
        reservations: Mapping[str, BudgetAmount] | None = None,
    ) -> None:
        if not isinstance(limits, BudgetAmount):
            raise ValueError("limits must be a BudgetAmount")
        self.limits = limits
        self.charged = charged or BudgetAmount()
        self.unknown_consumed = unknown_consumed or UnknownConsumption()
        if not isinstance(self.charged, BudgetAmount):
            raise ValueError("charged must be a BudgetAmount")
        if not isinstance(self.unknown_consumed, UnknownConsumption):
            raise ValueError("unknown_consumed must be UnknownConsumption")
        if stop_reason is not None and stop_reason not in _STOP_REASONS:
            raise ValueError("stop_reason is unsupported")
        self.stop_reason = stop_reason
        self._reservations: dict[str, BudgetAmount] = {}
        for reservation_id, amount in (reservations or {}).items():
            _require_reservation_id(reservation_id)
            if not isinstance(amount, BudgetAmount):
                raise ValueError("reservation amount must be a BudgetAmount")
            self._reservations[reservation_id] = amount
        self.reserved = _sum_budget_amounts(self._reservations.values())
        self._set_exhausted_if_consumed()

    @classmethod
    def for_policy(cls, limits: SupplementalRuntimeLimits) -> BudgetLedger:
        if not isinstance(limits, SupplementalRuntimeLimits):
            raise ValueError("limits must be SupplementalRuntimeLimits")
        return cls(limits=limits.budget_limits)

    @property
    def remaining(self) -> BudgetAmount:
        consumed = self.charged + self.unknown_consumed.budget_amount + self.reserved
        return self.limits.subtract_floor(consumed)

    @property
    def active_reservation_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._reservations))

    def reserve(self, reservation_id: str, amount: BudgetAmount) -> BudgetAmount:
        _require_reservation_id(reservation_id)
        if not isinstance(amount, BudgetAmount):
            raise ValueError("reservation amount must be a BudgetAmount")
        existing = self._reservations.get(reservation_id)
        if existing is not None:
            if existing != amount:
                raise ValueError(
                    f"reservation {reservation_id} already exists with different limits"
                )
            return existing
        if self.stop_reason == "budget_exhausted":
            raise BudgetExceededError("global supplemental budget is exhausted")
        exceeded = _exceeded_dimensions(amount, self.remaining)
        if exceeded:
            self.stop_reason = "budget_exhausted"
            raise BudgetExceededError(
                "supplemental reservation exceeds remaining " + ", ".join(exceeded)
            )
        self._reservations[reservation_id] = amount
        self.reserved = self.reserved + amount
        return amount

    def charge(
        self,
        reservation_id: str,
        consumption: BudgetAmount,
        *,
        usage_available: bool,
    ) -> BudgetAmount:
        if type(usage_available) is not bool:
            raise ValueError("usage_available must be a boolean")
        if not isinstance(consumption, BudgetAmount):
            raise ValueError("consumption must be a BudgetAmount")
        reservation = self._pop_reservation(reservation_id)
        if usage_available:
            # A submitted task itself is never free, even if a malformed usage
            # payload reports zero task consumption.
            charged = BudgetAmount(
                tasks=max(reservation.tasks, consumption.tasks),
                tool_calls=consumption.tool_calls,
                tokens=consumption.tokens,
                elapsed_seconds=consumption.elapsed_seconds,
            )
        else:
            charged = reservation.max_with(consumption)
        self.charged = self.charged + charged
        self._set_exhausted_if_consumed()
        return charged

    def mark_unknown(self, reservation_id: str, *, invocation_id: str) -> None:
        _require_reservation_id(invocation_id)
        reservation = self._pop_reservation(reservation_id)
        # The task and all reserved tool calls remain consumed. Tokens/time are
        # kept separately so reports can identify the at-least-once window.
        self.charged = self.charged + BudgetAmount(
            tasks=reservation.tasks,
            tool_calls=reservation.tool_calls,
        )
        self.unknown_consumed = UnknownConsumption(
            tokens=self.unknown_consumed.tokens + reservation.tokens,
            elapsed_seconds=(
                self.unknown_consumed.elapsed_seconds
                + reservation.elapsed_seconds
            ),
            invocation_ids=(
                *self.unknown_consumed.invocation_ids,
                invocation_id,
            ),
        )
        self._set_exhausted_if_consumed()

    def mark_stopped(self, reason: str) -> None:
        if reason not in _STOP_REASONS:
            raise ValueError("stop reason is unsupported")
        self.stop_reason = reason

    def to_dict(self) -> dict[str, Any]:
        return {
            "limits": asdict(self.limits),
            "reserved": asdict(self.reserved),
            "charged": asdict(self.charged),
            "unknown_consumed": asdict(self.unknown_consumed),
            "remaining": asdict(self.remaining),
            "active_reservations": {
                key: asdict(self._reservations[key])
                for key in sorted(self._reservations)
            },
            "stop_reason": self.stop_reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> BudgetLedger:
        if not isinstance(payload, Mapping):
            raise ValueError("budget ledger payload must be an object")
        reservations_payload = payload.get("active_reservations", {})
        if not isinstance(reservations_payload, Mapping):
            raise ValueError("active_reservations must be an object")
        ledger = cls(
            limits=_budget_amount_from_dict(payload.get("limits"), "limits"),
            charged=_budget_amount_from_dict(payload.get("charged"), "charged"),
            unknown_consumed=_unknown_from_dict(
                payload.get("unknown_consumed"),
            ),
            stop_reason=payload.get("stop_reason"),
            reservations={
                str(key): _budget_amount_from_dict(
                    value,
                    f"active_reservations.{key}",
                )
                for key, value in reservations_payload.items()
            },
        )
        persisted_reserved = _budget_amount_from_dict(
            payload.get("reserved"),
            "reserved",
        )
        if ledger.reserved != persisted_reserved:
            raise ValueError("reserved total does not match active reservations")
        return ledger

    def _pop_reservation(self, reservation_id: str) -> BudgetAmount:
        _require_reservation_id(reservation_id)
        try:
            reservation = self._reservations.pop(reservation_id)
        except KeyError as error:
            raise KeyError(f"unknown reservation: {reservation_id}") from error
        self.reserved = self.reserved.subtract_floor(reservation)
        return reservation

    def _set_exhausted_if_consumed(self) -> None:
        consumed = self.charged + self.unknown_consumed.budget_amount + self.reserved
        if _exceeded_dimensions(consumed, self.limits) or consumed == self.limits:
            self.stop_reason = "budget_exhausted"


@dataclass(frozen=True)
class SupplementalTaskSpec:
    request_id: str
    wave_id: str
    task_id: str
    source_candidate_ids: tuple[str, ...]
    source_disagreement_id: str
    assignment: Assignment
    allowed_tools: tuple[str, ...]
    budget_reservation: BudgetAmount
    bootstrap_policy: str = "targeted_only"

    def __post_init__(self) -> None:
        _require_reservation_id(self.request_id)
        _require_reservation_id(self.wave_id)
        _require_reservation_id(self.task_id)
        _require_safe_id(self.source_disagreement_id, "source_disagreement_id")
        candidate_ids = _canonical_ids(
            self.source_candidate_ids,
            "source_candidate_ids",
            allow_empty=False,
        )
        if not isinstance(self.assignment, Assignment):
            raise ValueError("assignment must be an Assignment")
        if self.assignment.planner_source != "semantic_reconciler":
            raise ValueError(
                "supplemental assignment planner_source must be semantic_reconciler"
            )
        if self.assignment.role_kind != "specialist":
            raise ValueError("supplemental assignment must use specialist role_kind")
        expected_contract = (
            f"{SUPPLEMENTAL_CONTRACT_PREFIX}{self.source_disagreement_id}"
        )
        if self.assignment.assigned_contract != [expected_contract]:
            raise ValueError(
                "supplemental assignment Contract must be isolated from Portfolio coverage"
            )
        if self.assignment.repository_permission != "read_only":
            raise ValueError("supplemental repository permission must be read_only")
        if self.assignment.command_permission != "safe_checks_only":
            raise ValueError(
                "supplemental command permission must be safe_checks_only"
            )
        if self.bootstrap_policy != "targeted_only":
            raise ValueError("supplemental bootstrap_policy must be targeted_only")
        allowed_tools = normalize_reviewer_allowed_tools(self.allowed_tools)
        if not isinstance(self.budget_reservation, BudgetAmount):
            raise ValueError("budget_reservation must be a BudgetAmount")
        expected_task_id = stable_task_id(
            wave_id=self.wave_id,
            request_id=self.request_id,
            compiled_assignment=self.assignment,
        )
        if self.task_id != expected_task_id:
            raise ValueError("task_id does not match compiled task content")
        object.__setattr__(self, "source_candidate_ids", candidate_ids)
        object.__setattr__(self, "allowed_tools", allowed_tools)

    @property
    def counts_toward_initial_coverage(self) -> bool:
        return False

    @property
    def origin(self) -> str:
        return "supplemental"


@dataclass(frozen=True)
class SupplementalPlan:
    review_id: str
    base_sha: str
    head_sha: str
    wave_index: int
    wave_id: str
    trigger_digest: str
    limits: SupplementalRuntimeLimits
    status: str
    tasks: tuple[SupplementalTaskSpec, ...] = ()
    request_ids: tuple[str, ...] = ()
    dropped_request_ids: tuple[str, ...] = ()
    policy_actions: tuple[str, ...] = ()
    schema_version: str = SUPPLEMENTAL_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.wave_index) is not int or self.wave_index < 1:
            raise ValueError("wave_index must be a positive integer")
        if self.status not in {
            "planned",
            "not_needed",
            "policy_limited",
            "max_waves",
        }:
            raise ValueError("supplemental plan status is unsupported")
        if not isinstance(self.limits, SupplementalRuntimeLimits):
            raise ValueError("limits must be SupplementalRuntimeLimits")
        if any(task.wave_id != self.wave_id for task in self.tasks):
            raise ValueError("all supplemental tasks must belong to plan wave_id")
        if self.status == "planned" and not self.tasks:
            raise ValueError("planned supplemental plan must contain tasks")
        if self.status != "planned" and self.tasks:
            raise ValueError("non-planned supplemental plan cannot contain tasks")
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(
            self,
            "request_ids",
            _canonical_ids(self.request_ids, "request_ids", allow_empty=True),
        )
        object.__setattr__(
            self,
            "dropped_request_ids",
            _canonical_ids(
                self.dropped_request_ids,
                "dropped_request_ids",
                allow_empty=True,
            ),
        )
        object.__setattr__(
            self,
            "policy_actions",
            tuple(dict.fromkeys(self.policy_actions)),
        )

    @property
    def max_concurrency(self) -> int:
        return min(self.limits.max_concurrency, len(self.tasks))


def _runtime_limits_from_policy(policy: SupplementalPolicy) -> SupplementalRuntimeLimits:
    return SupplementalRuntimeLimits(
        max_waves=policy.max_waves,
        max_tasks=policy.max_tasks,
        max_tasks_per_wave=policy.max_tasks_per_wave,
        max_concurrency=policy.max_concurrency,
        max_turns_per_task=policy.max_turns_per_task,
        max_tool_calls_per_task=policy.max_tool_calls_per_task,
        max_total_tokens_per_task=policy.max_tokens_per_task,
        max_total_tokens=policy.max_total_tokens,
        max_elapsed_seconds=policy.max_elapsed_seconds,
        policy_version=policy.version,
    )


_RISK_LIMITS = {
    risk_level: _runtime_limits_from_policy(
        SupplementalPolicy.for_risk(risk_level)
    )
    for risk_level in RiskLevel
}


def limits_for_risk(
    risk_level: RiskLevel,
    configured_policy: object | None = None,
) -> SupplementalRuntimeLimits:
    """Expand risk locally, then clamp it by immutable Session policy values."""

    if not isinstance(risk_level, RiskLevel):
        raise ValueError("risk_level must be a RiskLevel")
    default = _RISK_LIMITS[risk_level]
    if configured_policy is None:
        return default
    values: dict[str, int | float] = {}
    aliases = {
        "max_waves": ("max_waves",),
        "max_tasks": ("max_tasks", "max_total_tasks", "max_tasks_total"),
        "max_tasks_per_wave": ("max_tasks_per_wave",),
        "max_concurrency": ("max_concurrency",),
        "max_turns_per_task": ("max_turns_per_task", "max_turns"),
        "max_tool_calls_per_task": (
            "max_tool_calls_per_task",
            "max_tool_calls",
        ),
        "max_total_tokens_per_task": (
            "max_total_tokens_per_task",
            "max_tokens_per_task",
            "max_task_tokens",
        ),
        "max_total_tokens": ("max_total_tokens", "max_global_tokens"),
        "max_elapsed_seconds": (
            "max_elapsed_seconds",
            "max_global_elapsed_seconds",
        ),
    }
    for field_name, field_aliases in aliases.items():
        configured = _policy_field(
            configured_policy,
            field_aliases,
            getattr(default, field_name),
        )
        _validate_policy_limit(field_name, configured)
        values[field_name] = min(getattr(default, field_name), configured)
    policy_version = _policy_field(
        configured_policy,
        ("policy_version", "version"),
        default.policy_version,
    )
    # Independent clamping can otherwise produce an impossible combination
    # (for example one wave, two tasks per wave, but three total tasks).
    values["max_tasks"] = min(
        int(values["max_tasks"]),
        int(values["max_waves"]) * int(values["max_tasks_per_wave"]),
    )
    return SupplementalRuntimeLimits(
        **values,
        policy_version=_clean_text(policy_version, "policy_version"),
    )


def effective_policy_for_risk(
    risk_level: RiskLevel,
    configured_policy: object | None = None,
) -> SupplementalPolicy:
    """Return the immutable Runtime policy after local risk expansion."""

    limits = limits_for_risk(risk_level, configured_policy)
    return SupplementalPolicy(
        version=limits.policy_version,
        risk_level=risk_level.value,
        max_waves=limits.max_waves,
        max_tasks=limits.max_tasks,
        max_tasks_per_wave=limits.max_tasks_per_wave,
        max_concurrency=limits.max_concurrency,
        max_turns_per_task=limits.max_turns_per_task,
        max_tool_calls_per_task=limits.max_tool_calls_per_task,
        max_tokens_per_task=limits.max_total_tokens_per_task,
        max_total_tokens=limits.max_total_tokens,
        max_elapsed_seconds=limits.max_elapsed_seconds,
    )


def stable_request_id(
    request: SupplementalInvestigationRequest,
) -> str:
    if not isinstance(request, SupplementalInvestigationRequest):
        raise ValueError("request must be SupplementalInvestigationRequest")
    return _request_id(
        request.question,
        request.required_evidence,
        request.preferred_perspective,
        request.source_candidate_ids,
    )


def stable_wave_id(
    *,
    review_id: str,
    base_sha: str,
    head_sha: str,
    wave_index: int,
    trigger_digest: str,
    policy_version: str,
) -> str:
    if type(wave_index) is not int or wave_index < 1:
        raise ValueError("wave_index must be a positive integer")
    return _stable_id(
        "W",
        "supplemental_wave_v1",
        _clean_text(review_id, "review_id"),
        _clean_text(base_sha, "base_sha").casefold(),
        _clean_text(head_sha, "head_sha").casefold(),
        wave_index,
        _clean_text(trigger_digest, "trigger_digest"),
        _clean_text(policy_version, "policy_version"),
    )


def stable_assignment_digest(assignment: Assignment) -> str:
    if not isinstance(assignment, Assignment):
        raise ValueError("assignment must be an Assignment")
    encoded = _canonical_json(
        ["supplemental_assignment_digest_v1", asdict(assignment)]
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def stable_task_id(
    *,
    wave_id: str,
    request_id: str,
    compiled_assignment: Assignment | str,
) -> str:
    assignment_digest = (
        stable_assignment_digest(compiled_assignment)
        if isinstance(compiled_assignment, Assignment)
        else _clean_text(compiled_assignment, "compiled_assignment")
    )
    return _stable_id(
        "STASK",
        "supplemental_task_v1",
        _clean_text(wave_id, "wave_id"),
        _clean_text(request_id, "request_id"),
        assignment_digest,
    )


def stable_invocation_id(
    *,
    task_or_batch_id: str,
    logical_turn: int,
    request_digest: str,
    provider_attempt: int | None = None,
) -> str:
    if type(logical_turn) is not int or logical_turn < 0:
        raise ValueError("logical_turn must be a non-negative integer")
    if provider_attempt is not None and (
        type(provider_attempt) is not int or provider_attempt < 1
    ):
        raise ValueError("provider_attempt must be a positive integer")
    # provider_attempt is deliberately validated but excluded: retries of the
    # same logical turn share one invocation identity.
    return _stable_id(
        "INV",
        "reviewer_invocation_v1",
        _clean_text(task_or_batch_id, "task_or_batch_id"),
        logical_turn,
        _clean_text(request_digest, "request_digest"),
    )


def deduplicate_supplemental_requests(
    requests: Iterable[SupplementalInvestigationRequest],
) -> tuple[tuple[SupplementalInvestigationRequest, ...], tuple[str, ...]]:
    rows = tuple(requests)
    if any(not isinstance(item, SupplementalInvestigationRequest) for item in rows):
        raise ValueError(
            "requests must contain SupplementalInvestigationRequest values"
        )
    ordered = sorted(rows, key=_request_sort_key)
    unique: list[SupplementalInvestigationRequest] = []
    dropped: list[str] = []
    seen: set[str] = set()
    for request in ordered:
        if request.request_id in seen:
            dropped.append(request.request_id)
            continue
        seen.add(request.request_id)
        unique.append(request)
    return tuple(unique), tuple(dropped)


def compile_supplemental_plan(
    *,
    review_id: str,
    base_sha: str,
    head_sha: str,
    risk_level: RiskLevel,
    wave_index: int,
    trigger_digest: str,
    requests: Iterable[SupplementalInvestigationRequest],
    configured_policy: object | None = None,
    prior_task_count: int = 0,
    prior_request_ids: Iterable[str] = (),
    initial_context_by_request: Mapping[str, InitialContext] | None = None,
    reviewer_budget_caps: ReviewerBudgetCaps | None = None,
    allowed_tools: Iterable[str] = DEFAULT_SUPPLEMENTAL_ALLOWED_TOOLS,
) -> SupplementalPlan:
    if type(wave_index) is not int or wave_index < 1:
        raise ValueError("wave_index must be a positive integer")
    if type(prior_task_count) is not int or prior_task_count < 0:
        raise ValueError("prior_task_count must be a non-negative integer")
    limits = limits_for_risk(risk_level, configured_policy)
    wave_id = stable_wave_id(
        review_id=review_id,
        base_sha=base_sha,
        head_sha=head_sha,
        wave_index=wave_index,
        trigger_digest=trigger_digest,
        policy_version=limits.policy_version,
    )
    unique, duplicate_ids = deduplicate_supplemental_requests(requests)
    prior_ids = set(
        _canonical_ids(prior_request_ids, "prior_request_ids", allow_empty=True)
    )
    policy_actions = [
        f"enforced_supplemental_risk_limits:{risk_level.value}",
        "enforced_supplemental_permissions:read_only:safe_checks_only",
        "isolated_supplemental_contract_coverage",
        "enforced_bootstrap:targeted_only",
    ]
    dropped_ids = list(duplicate_ids)
    policy_actions.extend(
        f"deduplicated_request:{request_id}" for request_id in duplicate_ids
    )
    remaining_requests: list[SupplementalInvestigationRequest] = []
    for request in unique:
        if request.request_id in prior_ids:
            dropped_ids.append(request.request_id)
            policy_actions.append(
                f"deduplicated_request:{request.request_id}:prior_wave"
            )
            continue
        remaining_requests.append(request)

    if wave_index > limits.max_waves:
        dropped_ids.extend(item.request_id for item in remaining_requests)
        policy_actions.append(
            f"rejected_wave_limit:{wave_index}:max={limits.max_waves}"
        )
        return _empty_plan(
            review_id=review_id,
            base_sha=base_sha,
            head_sha=head_sha,
            wave_index=wave_index,
            wave_id=wave_id,
            trigger_digest=trigger_digest,
            limits=limits,
            status="max_waves",
            dropped_ids=dropped_ids,
            policy_actions=policy_actions,
        )

    capacity = min(
        limits.max_tasks_per_wave,
        max(0, limits.max_tasks - prior_task_count),
    )
    task_budget_available = (
        limits.max_turns_per_task > 0
        and limits.max_total_tokens_per_task > 0
        and limits.max_total_tokens > 0
        and limits.max_elapsed_seconds > 0
    )
    if capacity <= 0 or not task_budget_available:
        dropped_ids.extend(item.request_id for item in remaining_requests)
        policy_actions.append("rejected_task_limit:global_or_wave_capacity")
        return _empty_plan(
            review_id=review_id,
            base_sha=base_sha,
            head_sha=head_sha,
            wave_index=wave_index,
            wave_id=wave_id,
            trigger_digest=trigger_digest,
            limits=limits,
            status="policy_limited",
            dropped_ids=dropped_ids,
            policy_actions=policy_actions,
        )
    if not remaining_requests:
        return _empty_plan(
            review_id=review_id,
            base_sha=base_sha,
            head_sha=head_sha,
            wave_index=wave_index,
            wave_id=wave_id,
            trigger_digest=trigger_digest,
            limits=limits,
            status="not_needed",
            dropped_ids=dropped_ids,
            policy_actions=policy_actions,
        )

    selected = remaining_requests[:capacity]
    truncated = remaining_requests[capacity:]
    for request in truncated:
        dropped_ids.append(request.request_id)
        policy_actions.append(
            f"truncated_request:{request.request_id}:policy_capacity"
        )
    contexts = initial_context_by_request or {}
    if any(
        not isinstance(key, str) or not isinstance(value, InitialContext)
        for key, value in contexts.items()
    ):
        raise ValueError(
            "initial_context_by_request must map request IDs to InitialContext"
        )
    caps = reviewer_budget_caps or ReviewerBudgetCaps()
    if not isinstance(caps, ReviewerBudgetCaps):
        raise ValueError("reviewer_budget_caps must be ReviewerBudgetCaps")
    effective_tools = normalize_reviewer_allowed_tools(allowed_tools)
    tasks = tuple(
        _compile_task(
            wave_id=wave_id,
            request=request,
            limits=limits,
            context=contexts.get(request.request_id),
            caps=caps,
            allowed_tools=effective_tools,
        )
        for request in selected
    )
    return SupplementalPlan(
        review_id=_clean_text(review_id, "review_id"),
        base_sha=_clean_text(base_sha, "base_sha"),
        head_sha=_clean_text(head_sha, "head_sha"),
        wave_index=wave_index,
        wave_id=wave_id,
        trigger_digest=_clean_text(trigger_digest, "trigger_digest"),
        limits=limits,
        status="planned",
        tasks=tasks,
        request_ids=tuple(task.request_id for task in tasks),
        dropped_request_ids=tuple(dropped_ids),
        policy_actions=tuple(policy_actions),
    )


def is_supplemental_assignment(assignment: Assignment) -> bool:
    return (
        isinstance(assignment, Assignment)
        and assignment.planner_source == "semantic_reconciler"
        and bool(assignment.assigned_contract)
        and all(
            contract.startswith(SUPPLEMENTAL_CONTRACT_PREFIX)
            for contract in assignment.assigned_contract
        )
    )


def _compile_task(
    *,
    wave_id: str,
    request: SupplementalInvestigationRequest,
    limits: SupplementalRuntimeLimits,
    context: InitialContext | None,
    caps: ReviewerBudgetCaps,
    allowed_tools: tuple[str, ...],
) -> SupplementalTaskSpec:
    role, perspective_key = _role_for_perspective(
        request.preferred_perspective
    )
    normalized_context = _targeted_context(request, context)
    max_total_tokens = min(
        limits.max_total_tokens_per_task,
        caps.max_total_tokens,
    )
    max_elapsed_seconds = min(
        limits.max_elapsed_seconds / max(1, limits.max_tasks),
        caps.max_elapsed_seconds,
    )
    max_tool_calls = limits.max_tool_calls_per_task if allowed_tools else 0
    assignment_id = _stable_id(
        "SASSIGN",
        "supplemental_assignment_v1",
        wave_id,
        request.request_id,
        perspective_key,
    )
    assignment = Assignment(
        role=role,
        mission=request.question,
        assignment_reason=[
            f"resolve disagreement {request.source_disagreement_id}",
            "investigate only the material question compiled by Runtime",
        ],
        assigned_contract=[
            f"{SUPPLEMENTAL_CONTRACT_PREFIX}{request.source_disagreement_id}"
        ],
        required_checks=[
            *request.required_evidence,
            "cite only Runtime-authorized Observation IDs",
            "record unresolved evidence as uncertainty",
        ],
        initial_context=normalized_context,
        max_turns=limits.max_turns_per_task,
        max_tool_calls=max_tool_calls,
        max_output_tokens=min(caps.max_output_tokens, max_total_tokens),
        max_total_tokens=max_total_tokens,
        max_elapsed_seconds=max_elapsed_seconds,
        max_provider_attempts=caps.max_provider_attempts,
        repository_permission="read_only",
        command_permission="safe_checks_only",
        assignment_id=assignment_id,
        role_kind="specialist",
        perspective_key=perspective_key,
        planner_source="semantic_reconciler",
    )
    task_id = stable_task_id(
        wave_id=wave_id,
        request_id=request.request_id,
        compiled_assignment=assignment,
    )
    return SupplementalTaskSpec(
        request_id=request.request_id,
        wave_id=wave_id,
        task_id=task_id,
        source_candidate_ids=request.source_candidate_ids,
        source_disagreement_id=request.source_disagreement_id,
        assignment=assignment,
        allowed_tools=allowed_tools,
        bootstrap_policy="targeted_only",
        budget_reservation=BudgetAmount(
            tasks=1,
            tool_calls=max_tool_calls,
            tokens=max_total_tokens,
            elapsed_seconds=max_elapsed_seconds,
        ),
    )


def _targeted_context(
    request: SupplementalInvestigationRequest,
    context: InitialContext | None,
) -> InitialContext:
    if context is None:
        return InitialContext(
            observation_refs=[
                ref for ref in request.reason_refs if ref.startswith("O-")
            ],
            signal_refs=[
                request.source_disagreement_id,
                *request.source_candidate_ids,
                *(ref for ref in request.reason_refs if not ref.startswith("O-")),
            ],
        )
    return InitialContext(
        changed_files=sorted(set(context.changed_files)),
        diff_ranges=sorted(set(context.diff_ranges)),
        code_ranges=sorted(set(context.code_ranges)),
        quality_gate_summary={
            key: context.quality_gate_summary[key]
            for key in sorted(context.quality_gate_summary)
        },
        observation_refs=sorted(set(context.observation_refs)),
        signal_refs=sorted(set(context.signal_refs)),
    )


def _empty_plan(
    *,
    review_id: str,
    base_sha: str,
    head_sha: str,
    wave_index: int,
    wave_id: str,
    trigger_digest: str,
    limits: SupplementalRuntimeLimits,
    status: str,
    dropped_ids: Iterable[str],
    policy_actions: Iterable[str],
) -> SupplementalPlan:
    return SupplementalPlan(
        review_id=_clean_text(review_id, "review_id"),
        base_sha=_clean_text(base_sha, "base_sha"),
        head_sha=_clean_text(head_sha, "head_sha"),
        wave_index=wave_index,
        wave_id=wave_id,
        trigger_digest=_clean_text(trigger_digest, "trigger_digest"),
        limits=limits,
        status=status,
        dropped_request_ids=tuple(dropped_ids),
        policy_actions=tuple(policy_actions),
    )


def _request_id(
    question: str,
    required_evidence: Iterable[str],
    preferred_perspective: str,
    source_candidate_ids: Iterable[str],
) -> str:
    return _stable_id(
        "SREQ",
        "supplemental_request_v1",
        _canonical_text(question),
        sorted(_canonical_text(item) for item in required_evidence),
        _canonical_text(preferred_perspective),
        sorted(source_candidate_ids),
    )


def _role_for_perspective(perspective: str) -> tuple[str, str]:
    canonical = _canonical_text(perspective)
    role_prefixes = (
        ("security", "Security Supplemental Reviewer"),
        ("concurr", "Concurrency Supplemental Reviewer"),
        ("async", "Concurrency Supplemental Reviewer"),
        ("performance", "Performance Supplemental Reviewer"),
        ("data", "Data Integrity Supplemental Reviewer"),
        ("api", "API Supplemental Reviewer"),
        ("test", "Test Evidence Supplemental Reviewer"),
    )
    role = "Supplemental Specialist Reviewer"
    for keyword, candidate_role in role_prefixes:
        if keyword in canonical:
            role = candidate_role
            break
    slug = re.sub(r"[^a-z0-9]+", "_", canonical).strip("_") or "general"
    return role, f"supplemental:{slug}"


def _request_sort_key(
    request: SupplementalInvestigationRequest,
) -> tuple[str, str, str, str]:
    return (
        request.request_id,
        _canonical_text(request.source_disagreement_id),
        _canonical_text(request.question),
        request.question,
    )


def _stable_id(prefix: str, namespace: str, *parts: object) -> str:
    encoded = _canonical_json([namespace, *parts])
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"{prefix}-{digest}"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _canonical_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _canonical_text_items(values: object, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValueError(f"{name} must be an iterable of strings")
    cleaned = [_clean_text(value, f"{name} item") for value in values]
    if not cleaned:
        raise ValueError(f"{name} must not be empty")
    by_canonical: dict[str, str] = {}
    for value in sorted(cleaned, key=lambda item: (_canonical_text(item), item)):
        by_canonical.setdefault(_canonical_text(value), value)
    return tuple(by_canonical[key] for key in sorted(by_canonical))


def _canonical_ids(
    values: object,
    name: str,
    *,
    allow_empty: bool,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Iterable):
        raise ValueError(f"{name} must be an iterable of strings")
    cleaned = tuple(_clean_text(value, f"{name} item") for value in values)
    if not allow_empty and not cleaned:
        raise ValueError(f"{name} must not be empty")
    return tuple(sorted(set(cleaned)))


def _require_safe_id(value: object, name: str) -> None:
    text = _clean_text(value, name)
    if not _SAFE_ID.fullmatch(text):
        raise ValueError(f"{name} must be a stable safe identifier")


def _require_reservation_id(value: object) -> None:
    _clean_text(value, "reservation_id")


def _require_positive_number(value: object, name: str) -> None:
    _require_non_negative_number(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be a positive finite number")


def _sum_budget_amounts(amounts: Iterable[BudgetAmount]) -> BudgetAmount:
    total = BudgetAmount()
    for amount in amounts:
        total = total + amount
    return total


def _exceeded_dimensions(
    requested: BudgetAmount,
    available: BudgetAmount,
) -> list[str]:
    exceeded = []
    if requested.tasks > available.tasks:
        exceeded.append("tasks")
    if requested.tool_calls > available.tool_calls:
        exceeded.append("tool_calls")
    if requested.tokens > available.tokens:
        exceeded.append("tokens")
    if requested.elapsed_seconds > available.elapsed_seconds:
        exceeded.append("elapsed_seconds")
    return exceeded


def _policy_field(
    policy: object,
    names: tuple[str, ...],
    default: Any,
) -> Any:
    for name in names:
        if isinstance(policy, Mapping) and name in policy:
            return policy[name]
        if hasattr(policy, name):
            return getattr(policy, name)
    return default


def _validate_policy_limit(name: str, value: object) -> None:
    if name == "max_elapsed_seconds":
        _require_non_negative_number(value, name)
        return
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _budget_amount_from_dict(value: Any, name: str) -> BudgetAmount:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    expected = {"tasks", "tool_calls", "tokens", "elapsed_seconds"}
    if set(value) != expected:
        raise ValueError(f"{name} must contain exactly {', '.join(sorted(expected))}")
    return BudgetAmount(
        tasks=value["tasks"],
        tool_calls=value["tool_calls"],
        tokens=value["tokens"],
        elapsed_seconds=value["elapsed_seconds"],
    )


def _unknown_from_dict(value: Any) -> UnknownConsumption:
    if not isinstance(value, Mapping):
        raise ValueError("unknown_consumed must be an object")
    expected = {"tokens", "elapsed_seconds", "invocation_ids"}
    if set(value) != expected:
        raise ValueError(
            "unknown_consumed must contain exactly "
            + ", ".join(sorted(expected))
        )
    return UnknownConsumption(
        tokens=value["tokens"],
        elapsed_seconds=value["elapsed_seconds"],
        invocation_ids=tuple(value["invocation_ids"]),
    )
