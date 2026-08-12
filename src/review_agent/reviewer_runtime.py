from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Any, Mapping
import math

from review_agent.models import (
    Assignment,
    ReviewerResult,
    ReviewerResultStatus,
    ReviewerRuntimeMetadata,
    ReviewerTerminationReason,
)


@dataclass(frozen=True)
class ProviderUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    available: bool = False


@dataclass(frozen=True)
class ReviewerRuntimeLimitsV2:
    max_elapsed_seconds: float = 1_800.0
    max_provider_attempts: int = 3
    tool_timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        for name, value in (
            ("max_elapsed_seconds", self.max_elapsed_seconds),
            ("tool_timeout_seconds", self.tool_timeout_seconds),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a positive finite number")
            object.__setattr__(self, name, float(value))
        if (
            type(self.max_provider_attempts) is not int
            or self.max_provider_attempts <= 0
        ):
            raise ValueError("max_provider_attempts must be positive")


@dataclass
class ReviewerRuntimeStateV2:
    active_elapsed_seconds: float = 0.0
    provider_attempts: int = 0
    model_turns: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    all_usage_available: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.active_elapsed_seconds, bool)
            or not isinstance(self.active_elapsed_seconds, (int, float))
            or not math.isfinite(self.active_elapsed_seconds)
            or self.active_elapsed_seconds < 0
        ):
            raise ValueError("active_elapsed_seconds must be non-negative")
        self.active_elapsed_seconds = float(self.active_elapsed_seconds)

    def consume_active(self, elapsed_seconds: float) -> None:
        if (
            isinstance(elapsed_seconds, bool)
            or not isinstance(elapsed_seconds, (int, float))
            or not math.isfinite(elapsed_seconds)
            or elapsed_seconds < 0
        ):
            raise ValueError("elapsed_seconds must be non-negative")
        self.active_elapsed_seconds += float(elapsed_seconds)

    def record_provider_attempt(self, raw: Mapping[str, Any] | None) -> ProviderUsage:
        usage = provider_usage_from_raw(raw)
        self.provider_attempts += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.total_tokens += usage.total_tokens
        self.all_usage_available = self.all_usage_available and usage.available
        return usage

    def remaining_seconds(self, limits: ReviewerRuntimeLimitsV2) -> float:
        if not isinstance(limits, ReviewerRuntimeLimitsV2):
            raise ValueError("limits must be ReviewerRuntimeLimitsV2")
        return max(0.0, limits.max_elapsed_seconds - self.active_elapsed_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_elapsed_seconds": round(self.active_elapsed_seconds, 6),
            "provider_attempts": self.provider_attempts,
            "model_turns": self.model_turns,
            "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "usage_available": (
                self.provider_attempts > 0 and self.all_usage_available
            ),
        }


def request_parameters_v2(
    base_parameters: Mapping[str, Any],
    runtime: ReviewerRuntimeStateV2,
    limits: ReviewerRuntimeLimitsV2,
) -> dict[str, Any]:
    if not isinstance(runtime, ReviewerRuntimeStateV2):
        raise ValueError("runtime must be ReviewerRuntimeStateV2")
    remaining = runtime.remaining_seconds(limits)
    if remaining <= 0:
        raise TimeoutError("Reviewer active-time limit is exhausted")
    parameters = {
        key: value
        for key, value in dict(base_parameters).items()
        if key
        not in {
            "max_turns",
            "max_tool_calls",
            "max_total_tokens",
            "max_output_tokens",
        }
    }
    parameters["timeout_seconds"] = remaining
    return parameters


@dataclass
class RuntimeTracker:
    started_at: float
    provider_attempts: int = 0
    model_turns: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    all_usage_available: bool = True

    @classmethod
    def start(cls) -> "RuntimeTracker":
        return cls(started_at=time.monotonic())

    def elapsed_seconds(self) -> float:
        return max(0.0, time.monotonic() - self.started_at)

    def record_attempt(self, raw: Mapping[str, Any] | None) -> ProviderUsage:
        usage = provider_usage_from_raw(raw)
        self.provider_attempts += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.total_tokens += usage.total_tokens
        self.all_usage_available = self.all_usage_available and usage.available
        return usage

    def snapshot(
        self,
        termination_reason: ReviewerTerminationReason,
    ) -> ReviewerRuntimeMetadata:
        return ReviewerRuntimeMetadata(
            provider_attempts=self.provider_attempts,
            model_turns=self.model_turns,
            tool_calls=self.tool_calls,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            total_tokens=self.total_tokens,
            usage_available=(
                self.provider_attempts > 0 and self.all_usage_available
            ),
            elapsed_seconds=self.elapsed_seconds(),
            termination_reason=termination_reason,
        )


def provider_usage_from_raw(raw: Mapping[str, Any] | None) -> ProviderUsage:
    if not isinstance(raw, Mapping) or not isinstance(raw.get("usage"), Mapping):
        return ProviderUsage()

    usage = raw["usage"]
    input_tokens = _usage_counter(usage, "input_tokens", "prompt_tokens")
    output_tokens = _usage_counter(
        usage,
        "output_tokens",
        "completion_tokens",
    )
    total_tokens = _usage_counter(usage, "total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    return ProviderUsage(
        input_tokens=input_tokens or 0,
        output_tokens=output_tokens or 0,
        total_tokens=total_tokens or 0,
        available=(
            input_tokens is not None
            and output_tokens is not None
            and total_tokens is not None
        ),
    )


def budget_reason_before_call(
    assignment: Assignment,
    runtime: RuntimeTracker,
) -> ReviewerTerminationReason | None:
    if runtime.elapsed_seconds() >= assignment.max_elapsed_seconds:
        return ReviewerTerminationReason.TIME_BUDGET_EXHAUSTED
    if runtime.total_tokens >= assignment.max_total_tokens:
        return ReviewerTerminationReason.TOKEN_BUDGET_EXHAUSTED
    return None


def budget_reason_after_call(
    assignment: Assignment,
    runtime: RuntimeTracker,
) -> ReviewerTerminationReason | None:
    if runtime.elapsed_seconds() >= assignment.max_elapsed_seconds:
        return ReviewerTerminationReason.TIME_BUDGET_EXHAUSTED
    # A final response that lands exactly on the limit may still be consumed. A
    # subsequent turn is prevented by budget_reason_before_call().
    if runtime.total_tokens > assignment.max_total_tokens:
        return ReviewerTerminationReason.TOKEN_BUDGET_EXHAUSTED
    return None


def remaining_seconds(assignment: Assignment, runtime: RuntimeTracker) -> float:
    return max(0.001, assignment.max_elapsed_seconds - runtime.elapsed_seconds())


def request_parameters(
    base_parameters: Mapping[str, Any],
    assignment: Assignment,
    runtime: RuntimeTracker,
) -> dict[str, Any]:
    remaining_total = max(1, assignment.max_total_tokens - runtime.total_tokens)
    return {
        **dict(base_parameters),
        "max_output_tokens": min(
            assignment.max_output_tokens,
            remaining_total,
        ),
        "timeout_seconds": remaining_seconds(assignment, runtime),
    }


def termination_reason_for_result(
    result: ReviewerResult,
) -> ReviewerTerminationReason:
    reasons = {
        ReviewerResultStatus.COMPLETED: ReviewerTerminationReason.COMPLETED,
        ReviewerResultStatus.PARTIAL: ReviewerTerminationReason.REVIEWER_PARTIAL,
        ReviewerResultStatus.BLOCKED: ReviewerTerminationReason.REVIEWER_BLOCKED,
        ReviewerResultStatus.FAILED: ReviewerTerminationReason.RUNTIME_FAILURE,
    }
    return reasons[result.status]


def termination_summary(reason: ReviewerTerminationReason) -> str:
    return reason.value.replace("_", " ")


def reviewer_runtime_to_dict(runtime: ReviewerRuntimeMetadata) -> dict[str, Any]:
    payload = asdict(runtime)
    payload["termination_reason"] = runtime.termination_reason.value
    return payload


def _usage_counter(usage: Mapping[str, Any], *names: str) -> int | None:
    for name in names:
        value = usage.get(name)
        if type(value) is int and value >= 0:
            return value
    return None
