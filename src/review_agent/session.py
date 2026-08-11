from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import PurePosixPath, PureWindowsPath
import re
from types import MappingProxyType
from typing import Any, Mapping, TypeVar
from urllib.parse import urlsplit

from review_agent.memory_models import MemoryExecutionConfig
from review_agent.revision import RepositoryIdentity, ResolvedRevisions
from review_agent.run_state import RunPhase, RunStatus


LEGACY_SESSION_SCHEMA_VERSION = 1
PREVIOUS_SESSION_SCHEMA_VERSION = 2
MODEL_STAGE_SESSION_SCHEMA_VERSION = 3
SEMANTIC_RECONCILIATION_SESSION_SCHEMA_VERSION = 4
SESSION_SCHEMA_VERSION = 5
SESSION_V6_SCHEMA_VERSION = 6
SUPPORTED_SESSION_SCHEMA_VERSIONS = (
    LEGACY_SESSION_SCHEMA_VERSION,
    PREVIOUS_SESSION_SCHEMA_VERSION,
    MODEL_STAGE_SESSION_SCHEMA_VERSION,
    SEMANTIC_RECONCILIATION_SESSION_SCHEMA_VERSION,
    SESSION_SCHEMA_VERSION,
)
RESUMABLE_SESSION_SCHEMA_VERSIONS = (
    PREVIOUS_SESSION_SCHEMA_VERSION,
    MODEL_STAGE_SESSION_SCHEMA_VERSION,
    SEMANTIC_RECONCILIATION_SESSION_SCHEMA_VERSION,
    SESSION_SCHEMA_VERSION,
)
DEFAULT_MODEL_STAGE_API_KEY_ENV = "REVIEW_AGENT_API_KEY"
DEFAULT_MODEL_STAGE_MAX_OUTPUT_TOKENS = 4096
DEFAULT_MODEL_STAGE_MAX_PROVIDER_ATTEMPTS = 2
DEFAULT_MODEL_STAGE_MAX_ELAPSED_SECONDS = 60.0


def reviewer_runtime_limits_v2_to_dict() -> dict[str, object]:
    """Project the three v6 Reviewer safety limits into Session configuration."""

    from review_agent.reviewer_runtime import ReviewerRuntimeLimitsV2

    limits = ReviewerRuntimeLimitsV2()
    return {
        "max_elapsed_seconds": limits.max_elapsed_seconds,
        "max_provider_attempts": limits.max_provider_attempts,
        "tool_timeout_seconds": limits.tool_timeout_seconds,
    }
LEGACY_SESSION_PHASES = (
    RunPhase.PREFLIGHT,
    RunPhase.REPOSITORY_INTELLIGENCE,
    RunPhase.REVIEWERS,
    RunPhase.RECONCILIATION,
    RunPhase.COMPLETION,
    RunPhase.FINAL_RISK,
    RunPhase.REPORTING,
)
PREVIOUS_SESSION_PHASES = (
    RunPhase.PREFLIGHT,
    RunPhase.QUALITY_GATES,
    RunPhase.REPOSITORY_INTELLIGENCE,
    RunPhase.INTENT_DISCOVERY,
    RunPhase.INTENT_RESOLUTION,
    RunPhase.PLANNING,
    RunPhase.REVIEWERS,
    RunPhase.RECONCILIATION,
    RunPhase.COMPLETION,
    RunPhase.FINAL_RISK,
    RunPhase.REPORTING,
)
SEMANTIC_RECONCILIATION_SESSION_PHASES = (
    RunPhase.PREFLIGHT,
    RunPhase.QUALITY_GATES,
    RunPhase.REPOSITORY_INTELLIGENCE,
    RunPhase.INTENT_DISCOVERY,
    RunPhase.INTENT_RESOLUTION,
    RunPhase.PLANNING,
    RunPhase.REVIEWERS,
    RunPhase.RECONCILIATION_ANALYSIS,
    RunPhase.SUPPLEMENTAL_INVESTIGATION,
    RunPhase.RECONCILIATION,
    RunPhase.COMPLETION,
    RunPhase.FINAL_RISK,
    RunPhase.REPORTING,
)
SESSION_PHASES = (
    RunPhase.PREFLIGHT,
    RunPhase.QUALITY_GATES,
    RunPhase.REPOSITORY_INTELLIGENCE,
    RunPhase.MEMORY_SELECTION,
    RunPhase.INTENT_DISCOVERY,
    RunPhase.INTENT_RESOLUTION,
    RunPhase.PLANNING,
    RunPhase.REVIEWERS,
    RunPhase.RECONCILIATION_ANALYSIS,
    RunPhase.SUPPLEMENTAL_INVESTIGATION,
    RunPhase.RECONCILIATION,
    RunPhase.COMPLETION,
    RunPhase.FINAL_RISK,
    RunPhase.MEMORY_PROPOSAL,
    RunPhase.REPORTING,
)
SESSION_V6_PHASES = (
    RunPhase.PREFLIGHT,
    RunPhase.INTENT,
    RunPhase.PLANNING,
    RunPhase.REVIEWERS,
    RunPhase.AGGREGATION,
)

_ENVIRONMENT_VARIABLE_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_GIT_OBJECT_ID_PATTERN = re.compile(r"^(?:[0-9A-Fa-f]{40}|[0-9A-Fa-f]{64})$")
_SHA256_PATTERN = re.compile(r"^[0-9A-Fa-f]{64}$")
_STABLE_RUNTIME_ID_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,31}-[0-9A-Fa-f]{64}$")
_SUPPLEMENTAL_POLICY_VERSION = "supplemental_policy_v1"
_SUPPLEMENTAL_STOP_REASONS = frozenset(
    {
        "resolved",
        "no_requests",
        "model_fallback",
        "task_failure",
        "budget_exhausted",
        "max_waves",
        "unavailable",
    }
)


class PhaseStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_USER = "awaiting_user"
    COMPLETED = "completed"
    FAILED = "failed"
    INVALIDATED = "invalidated"


class SupplementalTaskStatus(str, Enum):
    PENDING = "pending"
    RESERVED = "reserved"
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    INVALIDATED = "invalidated"


class RevisionChangeKind(str, Enum):
    INITIAL = "initial"
    HEAD_MOVED = "head_moved"
    BASE_MOVED = "base_moved"
    BASE_AND_HEAD_MOVED = "base_and_head_moved"


@dataclass(frozen=True)
class ModelStageConfig:
    mode: str = "local"
    provider: str = "none"
    model: str | None = None
    base_url: str | None = None
    api_key_env: str = DEFAULT_MODEL_STAGE_API_KEY_ENV
    max_output_tokens: int = DEFAULT_MODEL_STAGE_MAX_OUTPUT_TOKENS
    max_provider_attempts: int = DEFAULT_MODEL_STAGE_MAX_PROVIDER_ATTEMPTS
    max_elapsed_seconds: float = DEFAULT_MODEL_STAGE_MAX_ELAPSED_SECONDS

    def __post_init__(self) -> None:
        if not isinstance(self.mode, str) or self.mode not in {"local", "model"}:
            raise ValueError("mode must be local or model")
        if not isinstance(self.provider, str) or self.provider not in {
            "none",
            "fake",
            "openai-compatible",
        }:
            raise ValueError(
                "provider must be none, fake, or openai-compatible"
            )
        _validate_optional_non_empty_string(self.model, "model")
        _validate_base_url(self.base_url, "base_url")
        if not isinstance(self.api_key_env, str) or not (
            _ENVIRONMENT_VARIABLE_PATTERN.fullmatch(self.api_key_env)
        ):
            raise ValueError(
                "api_key_env must be an environment variable name, not an API key value"
            )
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be a positive integer")
        if (
            type(self.max_provider_attempts) is not int
            or self.max_provider_attempts <= 0
        ):
            raise ValueError("max_provider_attempts must be a positive integer")
        if (
            isinstance(self.max_elapsed_seconds, bool)
            or not isinstance(self.max_elapsed_seconds, (int, float))
            or not math.isfinite(self.max_elapsed_seconds)
            or self.max_elapsed_seconds <= 0
        ):
            raise ValueError("max_elapsed_seconds must be a positive finite number")

        if self.mode == "local":
            if self.provider != "none":
                raise ValueError("mode=local requires provider=none")
            if self.model is not None or self.base_url is not None:
                raise ValueError("mode=local requires model and base_url to be null")
        elif self.provider == "none":
            raise ValueError("mode=model requires a model-capable provider")

        if self.provider == "openai-compatible":
            if self.model is None:
                raise ValueError(
                    "model is required for openai-compatible provider"
                )
            if self.base_url is None:
                raise ValueError(
                    "base_url is required for openai-compatible provider"
                )

        object.__setattr__(self, "max_elapsed_seconds", float(self.max_elapsed_seconds))


@dataclass(frozen=True)
class SupplementalPolicy:
    """Immutable Runtime ceilings for bounded supplemental investigation."""

    version: str = _SUPPLEMENTAL_POLICY_VERSION
    risk_level: str = "critical"
    max_waves: int = 2
    max_tasks: int = 4
    max_tasks_per_wave: int = 2
    max_concurrency: int = 2
    max_turns_per_task: int = 10
    max_tool_calls_per_task: int = 24
    max_tokens_per_task: int = 65_536
    max_total_tokens: int = 262_144
    max_elapsed_seconds: float = 600.0

    def __post_init__(self) -> None:
        if self.version != _SUPPLEMENTAL_POLICY_VERSION:
            raise ValueError(
                f"version must be {_SUPPLEMENTAL_POLICY_VERSION}"
            )
        if self.risk_level not in {"low", "medium", "high", "critical"}:
            raise ValueError("risk_level must be low, medium, high, or critical")
        for field_name in (
            "max_waves",
            "max_tasks",
            "max_tasks_per_wave",
            "max_concurrency",
            "max_turns_per_task",
            "max_tool_calls_per_task",
            "max_tokens_per_task",
            "max_total_tokens",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.max_tasks_per_wave > self.max_tasks:
            raise ValueError("max_tasks_per_wave must not exceed max_tasks")
        if self.max_concurrency > self.max_tasks_per_wave:
            raise ValueError("max_concurrency must not exceed max_tasks_per_wave")
        if self.max_tasks > self.max_waves * self.max_tasks_per_wave:
            raise ValueError("max_tasks exceeds the capacity of the configured waves")
        if self.max_total_tokens < self.max_tokens_per_task:
            raise ValueError("max_total_tokens must cover at least one task budget")
        if (
            isinstance(self.max_elapsed_seconds, bool)
            or not isinstance(self.max_elapsed_seconds, (int, float))
            or not math.isfinite(self.max_elapsed_seconds)
            or self.max_elapsed_seconds <= 0
        ):
            raise ValueError("max_elapsed_seconds must be a positive finite number")
        object.__setattr__(self, "max_elapsed_seconds", float(self.max_elapsed_seconds))

    @classmethod
    def for_risk(cls, risk_level: str | Enum) -> "SupplementalPolicy":
        value = risk_level.value if isinstance(risk_level, Enum) else risk_level
        if not isinstance(value, str):
            raise ValueError("risk_level must be low, medium, high, or critical")
        values = {
            "low": (1, 1, 1, 1, 4, 8, 16_384, 16_384, 120.0),
            "medium": (1, 2, 2, 2, 6, 12, 32_768, 65_536, 240.0),
            "high": (2, 3, 2, 2, 8, 16, 49_152, 147_456, 480.0),
            "critical": (2, 4, 2, 2, 10, 24, 65_536, 262_144, 600.0),
        }
        try:
            limits = values[value]
        except KeyError as error:
            raise ValueError(
                "risk_level must be low, medium, high, or critical"
            ) from error
        return cls(
            risk_level=value,
            max_waves=limits[0],
            max_tasks=limits[1],
            max_tasks_per_wave=limits[2],
            max_concurrency=limits[3],
            max_turns_per_task=limits[4],
            max_tool_calls_per_task=limits[5],
            max_tokens_per_task=limits[6],
            max_total_tokens=limits[7],
            max_elapsed_seconds=limits[8],
        )

    @property
    def max_total_tool_calls(self) -> int:
        return self.max_tasks * self.max_tool_calls_per_task


def _require_effective_policy_within_configured_ceiling(
    effective: SupplementalPolicy,
    configured: SupplementalPolicy,
) -> None:
    if effective.version != configured.version:
        raise ValueError("effective supplemental policy version differs from config")
    for field_name in (
        "max_waves",
        "max_tasks",
        "max_tasks_per_wave",
        "max_concurrency",
        "max_turns_per_task",
        "max_tool_calls_per_task",
        "max_tokens_per_task",
        "max_total_tokens",
        "max_elapsed_seconds",
    ):
        if getattr(effective, field_name) > getattr(configured, field_name):
            raise ValueError(
                f"effective supplemental policy exceeds configured {field_name}"
            )


@dataclass(frozen=True)
class SupplementalBudget:
    tasks: int = 0
    tool_calls: int = 0
    tokens: int = 0
    elapsed_seconds: float = 0.0

    def __post_init__(self) -> None:
        for field_name in ("tasks", "tool_calls", "tokens"):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if (
            isinstance(self.elapsed_seconds, bool)
            or not isinstance(self.elapsed_seconds, (int, float))
            or not math.isfinite(self.elapsed_seconds)
            or self.elapsed_seconds < 0
        ):
            raise ValueError("elapsed_seconds must be a finite non-negative number")
        object.__setattr__(self, "elapsed_seconds", float(self.elapsed_seconds))

    def __add__(self, other: object) -> "SupplementalBudget":
        if not isinstance(other, SupplementalBudget):
            return NotImplemented
        return SupplementalBudget(
            tasks=self.tasks + other.tasks,
            tool_calls=self.tool_calls + other.tool_calls,
            tokens=self.tokens + other.tokens,
            elapsed_seconds=self.elapsed_seconds + other.elapsed_seconds,
        )

    def is_zero(self) -> bool:
        return self == SupplementalBudget()

    def fits_within(self, ceiling: "SupplementalBudget") -> bool:
        return (
            self.tasks <= ceiling.tasks
            and self.tool_calls <= ceiling.tool_calls
            and self.tokens <= ceiling.tokens
            and self.elapsed_seconds <= ceiling.elapsed_seconds
        )


@dataclass(frozen=True)
class ReviewExecutionConfig:
    reviewer_provider: str
    reviewer_model: str | None
    reviewer_base_url: str | None
    reviewer_api_key_env: str
    reviewer_mode: str
    reviewer_loop: str
    non_interactive: bool
    risk_assessor: ModelStageConfig = field(default_factory=ModelStageConfig)
    portfolio_planner: ModelStageConfig = field(default_factory=ModelStageConfig)
    semantic_reconciler: ModelStageConfig = field(default_factory=ModelStageConfig)
    supplemental_policy: SupplementalPolicy = field(default_factory=SupplementalPolicy)
    memory: MemoryExecutionConfig | None = None
    memory_curator: ModelStageConfig = field(default_factory=ModelStageConfig)

    def __post_init__(self) -> None:
        if not _ENVIRONMENT_VARIABLE_PATTERN.fullmatch(self.reviewer_api_key_env):
            raise ValueError(
                "reviewer_api_key_env must be an environment variable name, "
                "not an API key value"
            )
        _validate_base_url(self.reviewer_base_url, "reviewer_base_url")
        if type(self.non_interactive) is not bool:
            raise ValueError("non_interactive must be a boolean")
        if not isinstance(self.risk_assessor, ModelStageConfig):
            raise ValueError("risk_assessor must be a ModelStageConfig")
        if not isinstance(self.portfolio_planner, ModelStageConfig):
            raise ValueError("portfolio_planner must be a ModelStageConfig")
        if not isinstance(self.semantic_reconciler, ModelStageConfig):
            raise ValueError("semantic_reconciler must be a ModelStageConfig")
        if not isinstance(self.supplemental_policy, SupplementalPolicy):
            raise ValueError("supplemental_policy must be a SupplementalPolicy")
        if self.memory is not None and not isinstance(
            self.memory,
            MemoryExecutionConfig,
        ):
            raise ValueError("memory must be a MemoryExecutionConfig or null")
        if not isinstance(self.memory_curator, ModelStageConfig):
            raise ValueError("memory_curator must be a ModelStageConfig")


def model_stage_config_to_dict(config: ModelStageConfig) -> dict[str, Any]:
    if not isinstance(config, ModelStageConfig):
        raise TypeError("config must be ModelStageConfig")
    return {
        "mode": config.mode,
        "provider": config.provider,
        "model": config.model,
        "base_url": config.base_url,
        "api_key_env": config.api_key_env,
        "max_output_tokens": config.max_output_tokens,
        "max_provider_attempts": config.max_provider_attempts,
        "max_elapsed_seconds": config.max_elapsed_seconds,
    }


def supplemental_policy_to_dict(config: SupplementalPolicy) -> dict[str, Any]:
    if not isinstance(config, SupplementalPolicy):
        raise TypeError("config must be SupplementalPolicy")
    return {
        "version": config.version,
        "risk_level": config.risk_level,
        "max_waves": config.max_waves,
        "max_tasks": config.max_tasks,
        "max_tasks_per_wave": config.max_tasks_per_wave,
        "max_concurrency": config.max_concurrency,
        "max_turns_per_task": config.max_turns_per_task,
        "max_tool_calls_per_task": config.max_tool_calls_per_task,
        "max_tokens_per_task": config.max_tokens_per_task,
        "max_total_tokens": config.max_total_tokens,
        "max_elapsed_seconds": config.max_elapsed_seconds,
    }


def review_execution_config_to_dict(
    execution: ReviewExecutionConfig,
) -> dict[str, Any]:
    if not isinstance(execution, ReviewExecutionConfig):
        raise TypeError("execution must be ReviewExecutionConfig")
    return {
        "reviewer_provider": execution.reviewer_provider,
        "reviewer_model": execution.reviewer_model,
        "reviewer_base_url": execution.reviewer_base_url,
        "reviewer_api_key_env": execution.reviewer_api_key_env,
        "reviewer_mode": execution.reviewer_mode,
        "reviewer_loop": execution.reviewer_loop,
        "non_interactive": execution.non_interactive,
        "risk_assessor": model_stage_config_to_dict(execution.risk_assessor),
        "portfolio_planner": model_stage_config_to_dict(
            execution.portfolio_planner
        ),
        "semantic_reconciler": model_stage_config_to_dict(
            execution.semantic_reconciler
        ),
        "supplemental_policy": supplemental_policy_to_dict(
            execution.supplemental_policy
        ),
        "memory": (
            None if execution.memory is None else execution.memory.to_dict()
        ),
        "memory_curator": model_stage_config_to_dict(
            execution.memory_curator
        ),
    }


@dataclass(frozen=True)
class ArtifactDescriptor:
    name: str
    path: str
    sha256: str
    schema: str
    phase: RunPhase
    revision_binding: str | None

    def __post_init__(self) -> None:
        _require_non_empty_string(self.name, "name")
        _require_non_empty_string(self.schema, "schema")
        _validate_artifact_path(self.path)
        if not isinstance(self.sha256, str) or not _SHA256_PATTERN.fullmatch(
            self.sha256
        ):
            raise ValueError("sha256 must be a full 64-character hexadecimal digest")
        if not isinstance(self.phase, RunPhase) or self.phase not in SESSION_PHASES:
            raise ValueError("phase must be one of the persisted SESSION_PHASES")


@dataclass(frozen=True)
class ReviewerTaskCheckpoint:
    status: PhaseStatus = PhaseStatus.PENDING
    attempts: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    artifacts: tuple[str, ...] = field(default_factory=tuple)
    error: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, PhaseStatus):
            raise ValueError("status must be a PhaseStatus")
        if self.status is PhaseStatus.AWAITING_USER:
            raise ValueError("reviewer task cannot be awaiting_user")
        if type(self.attempts) is not int or self.attempts < 0:
            raise ValueError("attempts must be a non-negative integer")
        artifacts = _immutable_string_tuple(self.artifacts, "artifacts")
        for field_name in ("started_at", "completed_at", "error"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{field_name} must be a non-empty string or null")
        if self.status is PhaseStatus.PENDING:
            if (
                self.attempts != 0
                or self.started_at is not None
                or self.completed_at is not None
                or artifacts
                or self.error is not None
            ):
                raise ValueError("pending reviewer task must not contain attempt state")
        elif self.status is PhaseStatus.RUNNING:
            if (
                self.attempts < 1
                or self.started_at is None
                or self.completed_at is not None
                or artifacts
                or self.error is not None
            ):
                raise ValueError("running reviewer task has inconsistent attempt state")
        elif self.status is PhaseStatus.COMPLETED:
            if (
                self.attempts < 1
                or self.started_at is None
                or self.completed_at is None
                or not artifacts
                or self.error is not None
            ):
                raise ValueError("completed reviewer task has inconsistent attempt state")
        elif self.status in {PhaseStatus.FAILED, PhaseStatus.INVALIDATED}:
            if (
                self.attempts < 1
                or self.started_at is None
                or self.completed_at is not None
                or artifacts
                or self.error is None
            ):
                raise ValueError(
                    f"{self.status.value} reviewer task has inconsistent attempt state"
                )
        object.__setattr__(self, "artifacts", artifacts)


@dataclass(frozen=True)
class SupplementalTaskCheckpoint:
    task_id: str
    assignment_digest: str
    status: SupplementalTaskStatus = SupplementalTaskStatus.PENDING
    attempts: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    artifacts: tuple[str, ...] = field(default_factory=tuple)
    reservation: SupplementalBudget = field(default_factory=SupplementalBudget)
    charged: SupplementalBudget = field(default_factory=SupplementalBudget)
    unknown_consumed: SupplementalBudget = field(default_factory=SupplementalBudget)
    unknown_invocation_ids: tuple[str, ...] = field(default_factory=tuple)
    error: str | None = None

    def __post_init__(self) -> None:
        _validate_stable_runtime_id(self.task_id, "task_id", "STASK")
        if not isinstance(self.assignment_digest, str) or not _SHA256_PATTERN.fullmatch(
            self.assignment_digest
        ):
            raise ValueError(
                "assignment_digest must be a full 64-character hexadecimal digest"
            )
        if not isinstance(self.status, SupplementalTaskStatus):
            raise ValueError("status must be a SupplementalTaskStatus")
        if type(self.attempts) is not int or self.attempts < 0:
            raise ValueError("attempts must be a non-negative integer")
        artifacts = _immutable_string_tuple(self.artifacts, "artifacts")
        if len(set(artifacts)) != len(artifacts) or any(not item for item in artifacts):
            raise ValueError("artifacts must contain unique non-empty strings")
        for field_name in ("started_at", "completed_at", "error"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, str) or not value or value != value.strip()
            ):
                raise ValueError(f"{field_name} must be a non-empty string or null")
        for field_name in ("reservation", "charged", "unknown_consumed"):
            if not isinstance(getattr(self, field_name), SupplementalBudget):
                raise ValueError(f"{field_name} must be a SupplementalBudget")
        invocation_ids = _immutable_string_tuple(
            self.unknown_invocation_ids,
            "unknown_invocation_ids",
        )
        if len(set(invocation_ids)) != len(invocation_ids):
            raise ValueError("unknown_invocation_ids must not contain duplicates")
        for invocation_id in invocation_ids:
            _validate_stable_runtime_id(
                invocation_id,
                "unknown invocation id",
                "INV",
            )
        if self.unknown_consumed.is_zero() != (not invocation_ids):
            raise ValueError(
                "unknown_consumed and unknown_invocation_ids must be recorded together"
            )

        if self.status is SupplementalTaskStatus.PENDING:
            if (
                self.attempts != 0
                or self.started_at is not None
                or self.completed_at is not None
                or artifacts
                or not self.reservation.is_zero()
                or not self.charged.is_zero()
                or not self.unknown_consumed.is_zero()
                or self.error is not None
            ):
                raise ValueError("pending supplemental task must not contain attempt state")
        elif self.status is SupplementalTaskStatus.RESERVED:
            if (
                self.started_at is not None
                or self.completed_at is not None
                or artifacts
                or self.reservation.tasks != 1
                or self.error is not None
            ):
                raise ValueError("reserved supplemental task has inconsistent state")
        elif self.status is SupplementalTaskStatus.RUNNING:
            if (
                self.attempts < 1
                or self.started_at is None
                or self.completed_at is not None
                or artifacts
                or self.reservation.tasks != 1
                or self.error is not None
            ):
                raise ValueError("running supplemental task has inconsistent state")
        elif self.status in {
            SupplementalTaskStatus.COMPLETED,
            SupplementalTaskStatus.PARTIAL,
        }:
            if (
                self.attempts < 1
                or self.started_at is None
                or self.completed_at is None
                or not artifacts
                or not self.reservation.is_zero()
                or self.charged.tasks < 1
                or (
                    self.status is SupplementalTaskStatus.COMPLETED
                    and self.error is not None
                )
                or (
                    self.status is SupplementalTaskStatus.PARTIAL
                    and self.error is None
                )
            ):
                raise ValueError(
                    f"{self.status.value} supplemental task has inconsistent state"
                )
        elif self.status is SupplementalTaskStatus.FAILED:
            if (
                self.attempts < 1
                or self.started_at is None
                or self.completed_at is not None
                or not self.reservation.is_zero()
                or self.error is None
            ):
                raise ValueError("failed supplemental task has inconsistent state")
        elif self.status is SupplementalTaskStatus.INVALIDATED:
            if (
                self.completed_at is not None
                or artifacts
                or not self.reservation.is_zero()
                or self.error is None
            ):
                raise ValueError("invalidated supplemental task has inconsistent state")
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "unknown_invocation_ids", invocation_ids)


@dataclass(frozen=True)
class ReviewWaveCheckpoint:
    wave_id: str
    wave_index: int
    trigger_digest: str
    effective_policy: SupplementalPolicy
    status: PhaseStatus = PhaseStatus.PENDING
    attempts: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    artifacts: tuple[str, ...] = field(default_factory=tuple)
    tasks: Mapping[str, SupplementalTaskCheckpoint] = field(default_factory=dict)
    stop_reason: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        _validate_stable_runtime_id(self.wave_id, "wave_id", "W")
        if type(self.wave_index) is not int or self.wave_index < 1:
            raise ValueError("wave_index must be a positive integer")
        if not isinstance(self.trigger_digest, str) or not _SHA256_PATTERN.fullmatch(
            self.trigger_digest
        ):
            raise ValueError(
                "trigger_digest must be a full 64-character hexadecimal digest"
            )
        if not isinstance(self.status, PhaseStatus) or self.status is PhaseStatus.AWAITING_USER:
            raise ValueError("status must be a non-awaiting PhaseStatus")
        if not isinstance(self.effective_policy, SupplementalPolicy):
            raise ValueError("effective_policy must be a SupplementalPolicy")
        if type(self.attempts) is not int or self.attempts < 0:
            raise ValueError("attempts must be a non-negative integer")
        artifacts = _immutable_string_tuple(self.artifacts, "artifacts")
        if len(set(artifacts)) != len(artifacts) or any(not item for item in artifacts):
            raise ValueError("artifacts must contain unique non-empty strings")
        if not isinstance(self.tasks, Mapping):
            raise ValueError("tasks must be a mapping")
        tasks = dict(self.tasks)
        if not tasks:
            raise ValueError("review wave must contain at least one supplemental task")
        for task_id, checkpoint in tasks.items():
            if not isinstance(checkpoint, SupplementalTaskCheckpoint):
                raise ValueError(
                    "wave task values must be SupplementalTaskCheckpoint instances"
                )
            if task_id != checkpoint.task_id:
                raise ValueError("task registry key must match checkpoint.task_id")
        for field_name in ("started_at", "completed_at", "error"):
            value = getattr(self, field_name)
            if value is not None and (
                not isinstance(value, str) or not value or value != value.strip()
            ):
                raise ValueError(f"{field_name} must be a non-empty string or null")
        if self.stop_reason is not None and self.stop_reason not in _SUPPLEMENTAL_STOP_REASONS:
            raise ValueError("stop_reason has an unsupported value")

        if self.status is PhaseStatus.PENDING:
            if (
                self.attempts != 0
                or self.started_at is not None
                or self.completed_at is not None
                or artifacts
                or self.stop_reason is not None
                or self.error is not None
            ):
                raise ValueError("pending review wave must not contain attempt state")
        elif self.status is PhaseStatus.RUNNING:
            if (
                self.attempts < 1
                or self.started_at is None
                or self.completed_at is not None
                or artifacts
                or self.stop_reason is not None
                or self.error is not None
            ):
                raise ValueError("running review wave has inconsistent state")
        elif self.status is PhaseStatus.COMPLETED:
            nonterminal = [
                task_id
                for task_id, task in tasks.items()
                if task.status
                not in {
                    SupplementalTaskStatus.COMPLETED,
                    SupplementalTaskStatus.PARTIAL,
                    SupplementalTaskStatus.FAILED,
                }
            ]
            if nonterminal:
                raise ValueError(
                    "completed review wave contains nonterminal tasks: "
                    + ", ".join(nonterminal)
                )
            task_artifacts = {
                artifact_name
                for task in tasks.values()
                for artifact_name in task.artifacts
            }
            if (
                self.attempts < 1
                or self.started_at is None
                or self.completed_at is None
                or not artifacts
                or not task_artifacts.issubset(artifacts)
                or self.stop_reason is None
                or self.error is not None
            ):
                raise ValueError("completed review wave has inconsistent state")
        elif self.status in {PhaseStatus.FAILED, PhaseStatus.INVALIDATED}:
            if self.completed_at is not None or self.stop_reason is not None or self.error is None:
                raise ValueError(
                    f"{self.status.value} review wave has inconsistent state"
                )
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "tasks", MappingProxyType(tasks))


@dataclass(frozen=True)
class PhaseCheckpoint:
    status: PhaseStatus = PhaseStatus.PENDING
    attempts: int = 0
    started_at: str | None = None
    completed_at: str | None = None
    artifacts: tuple[str, ...] = field(default_factory=tuple)
    error: str | None = None
    tasks: Mapping[str, ReviewerTaskCheckpoint] = field(default_factory=dict)
    user_decisions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, PhaseStatus):
            raise ValueError("status must be a PhaseStatus")
        if type(self.attempts) is not int or self.attempts < 0:
            raise ValueError("attempts must be a non-negative integer")
        artifacts = _immutable_string_tuple(self.artifacts, "artifacts")
        for field_name in ("started_at", "completed_at", "error"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{field_name} must be a non-empty string or null")
        if not isinstance(self.tasks, Mapping):
            raise ValueError("tasks must be a mapping")
        tasks = dict(self.tasks)
        for task_name, checkpoint in tasks.items():
            if (
                not isinstance(task_name, str)
                or not re.fullmatch(r"reviewer-[0-9]+", task_name)
            ):
                raise ValueError("reviewer task names must use reviewer-<index>")
            if not isinstance(checkpoint, ReviewerTaskCheckpoint):
                raise ValueError(
                    "reviewer task values must be ReviewerTaskCheckpoint instances"
                )
        if not isinstance(self.user_decisions, Mapping):
            raise ValueError("user_decisions must be a mapping")
        user_decisions = dict(self.user_decisions)
        for event_id, artifact_name in user_decisions.items():
            _require_non_empty_string(event_id, "user decision event id")
            _require_non_empty_string(artifact_name, "user decision artifact name")
        if not set(user_decisions.values()).issubset(artifacts):
            raise ValueError(
                "user decision artifacts must be retained by the phase checkpoint"
            )
        if self.status is PhaseStatus.AWAITING_USER:
            if (
                self.attempts < 1
                or self.started_at is None
                or self.completed_at is not None
                or not artifacts
                or self.error is not None
                or tasks
            ):
                raise ValueError(
                    "awaiting_user phase must have a started attempt and committed "
                    "artifacts, no completion, error, or reviewer tasks"
                )
        if self.status is PhaseStatus.PENDING and user_decisions:
            raise ValueError("pending phase cannot contain user decisions")
        object.__setattr__(self, "artifacts", artifacts)
        object.__setattr__(self, "tasks", MappingProxyType(tasks))
        object.__setattr__(
            self,
            "user_decisions",
            MappingProxyType(user_decisions),
        )


@dataclass(frozen=True)
class SessionManifest:
    schema_version: int
    review_id: str
    parent_review_id: str | None
    root_review_id: str
    repository: RepositoryIdentity
    revisions: ResolvedRevisions
    original_base_sha: str
    incremental_from_sha: str | None
    revision_change_kind: RevisionChangeKind
    execution: ReviewExecutionConfig
    status: RunStatus
    current_phase: RunPhase
    last_successful_phase: RunPhase | None
    phases: Mapping[str, PhaseCheckpoint]
    artifacts: Mapping[str, ArtifactDescriptor]
    errors: tuple[str, ...]
    created_at: str
    updated_at: str
    supplemental_waves: Mapping[str, ReviewWaveCheckpoint] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version not in SUPPORTED_SESSION_SCHEMA_VERSIONS:
            raise ValueError(
                "schema_version must be a supported Session schema version: "
                + ", ".join(str(version) for version in SUPPORTED_SESSION_SCHEMA_VERSIONS)
            )
        _require_non_empty_string(self.review_id, "review_id")
        _require_non_empty_string(self.root_review_id, "root_review_id")
        if self.parent_review_id is not None:
            _require_non_empty_string(self.parent_review_id, "parent_review_id")
        if not isinstance(self.revision_change_kind, RevisionChangeKind):
            raise ValueError("revision_change_kind must be a RevisionChangeKind")
        if not isinstance(self.status, RunStatus):
            raise ValueError("status must be a RunStatus")
        if not isinstance(self.current_phase, RunPhase):
            raise ValueError("current_phase must be a RunPhase")
        if self.last_successful_phase is not None and not isinstance(
            self.last_successful_phase,
            RunPhase,
        ):
            raise ValueError("last_successful_phase must be a RunPhase or null")
        if not isinstance(self.execution, ReviewExecutionConfig):
            raise ValueError("execution must be a ReviewExecutionConfig")
        if self.schema_version < MODEL_STAGE_SESSION_SCHEMA_VERSION and (
            self.execution.risk_assessor.mode != "local"
            or self.execution.portfolio_planner.mode != "local"
        ):
            raise ValueError(
                "schema v1/v2 Sessions must use local risk_assessor and "
                "portfolio_planner configurations"
            )
        if self.schema_version < SEMANTIC_RECONCILIATION_SESSION_SCHEMA_VERSION and (
            self.execution.semantic_reconciler != ModelStageConfig()
            or self.execution.supplemental_policy != SupplementalPolicy()
        ):
            raise ValueError(
                "schema v1/v2/v3 Sessions must use local semantic_reconciler "
                "and the legacy supplemental policy default"
            )
        if self.schema_version < SESSION_SCHEMA_VERSION:
            if (
                self.execution.memory is not None
                or self.execution.memory_curator != ModelStageConfig()
            ):
                raise ValueError(
                    "schema v1/v2/v3/v4 Sessions must use legacy memory-off "
                    "and no-curator execution semantics"
                )
        elif self.execution.memory is None:
            raise ValueError(
                "schema v5 Sessions require a fixed MemoryExecutionConfig"
            )

        _validate_manifest_object_ids(self)
        _validate_manifest_lineage(self)

        if not isinstance(self.phases, Mapping):
            raise ValueError("phases must be a mapping")
        phases = dict(self.phases)
        layout_phases = session_phases_for_schema(self.schema_version)
        expected_phase_names = {phase.value for phase in layout_phases}
        allowed_current_phases = {
            RunPhase.CREATED,
            RunPhase.COMPLETED,
            RunPhase.FAILED,
            *layout_phases,
        }
        if self.current_phase not in allowed_current_phases:
            raise ValueError(
                "current_phase must belong to the Session schema layout or be "
                "a lifecycle phase"
            )
        if (
            self.last_successful_phase is not None
            and self.last_successful_phase not in layout_phases
        ):
            raise ValueError(
                "last_successful_phase must belong to the Session schema layout"
            )
        if set(phases) != expected_phase_names:
            raise ValueError(
                "phases must contain exactly the persisted phases for its "
                "Session schema version"
            )
        if any(
            not isinstance(checkpoint, PhaseCheckpoint)
            for checkpoint in phases.values()
        ):
            raise ValueError("phases values must be PhaseCheckpoint instances")
        for phase_name, checkpoint in phases.items():
            if phase_name != RunPhase.REVIEWERS.value and checkpoint.tasks:
                raise ValueError(
                    "reviewer task checkpoints are allowed only on the reviewers phase"
                )
            if (
                phase_name != RunPhase.INTENT_RESOLUTION.value
                and checkpoint.user_decisions
            ):
                raise ValueError(
                    "user decisions are allowed only on the intent_resolution phase"
                )
            if (
                checkpoint.status is PhaseStatus.AWAITING_USER
                and phase_name != RunPhase.INTENT_RESOLUTION.value
            ):
                raise ValueError(
                    "awaiting_user checkpoint is allowed only on intent_resolution"
                )
        reviewer_checkpoint = phases[RunPhase.REVIEWERS.value]
        if reviewer_checkpoint.status is PhaseStatus.COMPLETED:
            incomplete_tasks = [
                task_name
                for task_name, task in reviewer_checkpoint.tasks.items()
                if task.status is not PhaseStatus.COMPLETED
            ]
            if incomplete_tasks:
                raise ValueError(
                    "completed reviewers phase contains incomplete tasks: "
                    + ", ".join(incomplete_tasks)
                )
            task_artifacts = {
                artifact_name
                for task in reviewer_checkpoint.tasks.values()
                for artifact_name in task.artifacts
            }
            if not task_artifacts.issubset(reviewer_checkpoint.artifacts):
                raise ValueError(
                    "completed reviewers phase omits reviewer task artifacts"
                )

        if not isinstance(self.artifacts, Mapping):
            raise ValueError("artifacts must be a mapping")
        artifacts = dict(self.artifacts)
        for registry_name, descriptor in artifacts.items():
            if not isinstance(registry_name, str):
                raise ValueError("artifact registry keys must be strings")
            if not isinstance(descriptor, ArtifactDescriptor):
                raise ValueError(
                    "artifact registry values must be ArtifactDescriptor instances"
                )
            if registry_name != descriptor.name:
                raise ValueError(
                    f"artifact registry key {registry_name!r} must match "
                    f"descriptor.name {descriptor.name!r}"
                )
            if descriptor.phase not in layout_phases:
                raise ValueError(
                    "artifact phase must belong to the Session schema layout"
                )

        awaiting_phases = [
            phase_name
            for phase_name, checkpoint in phases.items()
            if checkpoint.status is PhaseStatus.AWAITING_USER
        ]
        if self.status is RunStatus.AWAITING_USER:
            if (
                self.schema_version < PREVIOUS_SESSION_SCHEMA_VERSION
                or self.current_phase is not RunPhase.INTENT_RESOLUTION
                or awaiting_phases != [RunPhase.INTENT_RESOLUTION.value]
            ):
                raise ValueError(
                    "awaiting_user Session must be schema v2 or later and have only the "
                    "intent_resolution checkpoint awaiting_user"
                )
        elif awaiting_phases:
            raise ValueError(
                "an awaiting_user phase requires the Session status awaiting_user"
            )

        if RunPhase.INTENT_RESOLUTION.value in phases:
            intent_resolution = phases[RunPhase.INTENT_RESOLUTION.value]
            if intent_resolution.status is PhaseStatus.AWAITING_USER:
                for artifact_name in intent_resolution.artifacts:
                    descriptor = artifacts.get(artifact_name)
                    if descriptor is None:
                        raise ValueError(
                            "awaiting_user intent_resolution artifacts must be "
                            f"registered: {artifact_name}"
                        )
                    if descriptor.phase is not RunPhase.INTENT_RESOLUTION:
                        raise ValueError(
                            "awaiting_user intent_resolution artifacts must belong "
                            "to intent_resolution"
                        )

        if not isinstance(self.supplemental_waves, Mapping):
            raise ValueError("supplemental_waves must be a mapping")
        supplemental_waves = dict(self.supplemental_waves)
        if (
            self.schema_version < SEMANTIC_RECONCILIATION_SESSION_SCHEMA_VERSION
            and supplemental_waves
        ):
            raise ValueError("schema v1/v2/v3 Sessions cannot contain supplemental_waves")
        expected_wave_index = 1
        task_ids: set[str] = set()
        total_tasks = 0
        effective_policy: SupplementalPolicy | None = None
        for wave_id, wave in supplemental_waves.items():
            if not isinstance(wave, ReviewWaveCheckpoint):
                raise ValueError(
                    "supplemental_waves values must be ReviewWaveCheckpoint instances"
                )
            if wave_id != wave.wave_id:
                raise ValueError("supplemental wave registry key must match wave.wave_id")
            if wave.wave_index != expected_wave_index:
                raise ValueError("supplemental wave indexes must be contiguous from 1")
            expected_wave_index += 1
            if effective_policy is None:
                effective_policy = wave.effective_policy
            elif wave.effective_policy != effective_policy:
                raise ValueError(
                    "supplemental waves must share one effective Runtime policy"
                )
            _require_effective_policy_within_configured_ceiling(
                wave.effective_policy,
                self.execution.supplemental_policy,
            )
            if wave.wave_index > wave.effective_policy.max_waves:
                raise ValueError("supplemental wave exceeds policy max_waves")
            if len(wave.tasks) > wave.effective_policy.max_tasks_per_wave:
                raise ValueError("supplemental wave exceeds policy max_tasks_per_wave")
            duplicate_task_ids = task_ids.intersection(wave.tasks)
            if duplicate_task_ids:
                raise ValueError(
                    "supplemental task IDs must be unique across waves: "
                    + ", ".join(sorted(duplicate_task_ids))
                )
            task_ids.update(wave.tasks)
            total_tasks += len(wave.tasks)
            for artifact_name in wave.artifacts:
                descriptor = artifacts.get(artifact_name)
                if descriptor is None:
                    raise ValueError(
                        f"supplemental wave artifact must be registered: {artifact_name}"
                    )
                if descriptor.phase is not RunPhase.SUPPLEMENTAL_INVESTIGATION:
                    raise ValueError(
                        "supplemental wave artifacts must belong to "
                        "supplemental_investigation"
                    )
        if effective_policy is not None and total_tasks > effective_policy.max_tasks:
            raise ValueError("supplemental tasks exceed policy max_tasks")

        if supplemental_waves:
            supplemental_phase = phases[RunPhase.SUPPLEMENTAL_INVESTIGATION.value]
            if supplemental_phase.status is PhaseStatus.COMPLETED:
                incomplete_waves = [
                    wave_id
                    for wave_id, wave in supplemental_waves.items()
                    if wave.status is not PhaseStatus.COMPLETED
                ]
                if incomplete_waves:
                    raise ValueError(
                        "completed supplemental phase contains incomplete waves: "
                        + ", ".join(incomplete_waves)
                    )
                wave_artifacts = {
                    artifact_name
                    for wave in supplemental_waves.values()
                    for artifact_name in wave.artifacts
                }
                if not wave_artifacts.issubset(supplemental_phase.artifacts):
                    raise ValueError(
                        "completed supplemental phase omits review wave artifacts"
                    )

        errors = _immutable_string_tuple(self.errors, "errors")
        object.__setattr__(self, "phases", MappingProxyType(phases))
        object.__setattr__(self, "artifacts", MappingProxyType(artifacts))
        object.__setattr__(self, "errors", errors)
        object.__setattr__(
            self,
            "supplemental_waves",
            MappingProxyType(supplemental_waves),
        )


def _validate_manifest_object_ids(manifest: SessionManifest) -> None:
    object_ids = {
        "resolved_base_sha": manifest.revisions.resolved_base_sha,
        "resolved_head_sha": manifest.revisions.resolved_head_sha,
        "original_base_sha": manifest.original_base_sha,
    }
    if manifest.incremental_from_sha is not None:
        object_ids["incremental_from_sha"] = manifest.incremental_from_sha

    for field_name, object_id in object_ids.items():
        if not isinstance(object_id, str) or not _GIT_OBJECT_ID_PATTERN.fullmatch(
            object_id
        ):
            raise ValueError(
                f"{field_name} must be a full 40- or 64-character hexadecimal "
                "Git object ID"
            )
    if len({len(object_id) for object_id in object_ids.values()}) != 1:
        raise ValueError(
            "resolved_base_sha, resolved_head_sha, original_base_sha, and "
            "incremental_from_sha must use the same object ID format"
        )


def _validate_manifest_lineage(manifest: SessionManifest) -> None:
    if manifest.revision_change_kind is RevisionChangeKind.INITIAL:
        if manifest.parent_review_id is not None:
            raise ValueError("initial Session parent_review_id must be null")
        if manifest.root_review_id != manifest.review_id:
            raise ValueError("initial Session root_review_id must equal review_id")
        if (
            manifest.original_base_sha.casefold()
            != manifest.revisions.resolved_base_sha.casefold()
        ):
            raise ValueError(
                "initial Session original_base_sha must equal resolved_base_sha"
            )
        if manifest.incremental_from_sha is not None:
            raise ValueError("initial Session incremental_from_sha must be null")
        return

    if manifest.parent_review_id is None:
        raise ValueError("child Session parent_review_id must be present")
    if manifest.parent_review_id == manifest.review_id:
        raise ValueError("child Session parent_review_id must not self-reference")
    if manifest.root_review_id == manifest.review_id:
        raise ValueError("child Session root_review_id must not self-reference")

    if manifest.revision_change_kind is RevisionChangeKind.HEAD_MOVED:
        if manifest.incremental_from_sha is None:
            raise ValueError("HEAD_MOVED Session incremental_from_sha must be present")
        return

    if manifest.incremental_from_sha is not None:
        raise ValueError(
            "Base drift Session incremental_from_sha must be null because it "
            "requires a full re-review"
        )


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _validate_optional_non_empty_string(value: Any, field_name: str) -> None:
    if value is not None and (
        not isinstance(value, str) or not value or value != value.strip()
    ):
        raise ValueError(f"{field_name} must be a non-empty string or null")


def _validate_base_url(value: Any, field_name: str) -> None:
    if value is None:
        return
    _validate_optional_non_empty_string(value, field_name)
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            f"{field_name} must be an HTTP(S) base URL without credentials, "
            "query parameters, or fragments"
        )


def _validate_artifact_path(path: Any) -> None:
    if not isinstance(path, str) or not path or path != path.strip():
        raise ValueError("path must be a non-empty canonical relative path")
    if "\\" in path:
        raise ValueError("path must use canonical forward-slash separators")

    posix_path = PurePosixPath(path)
    windows_path = PureWindowsPath(path)
    path_parts = path.split("/")
    if (
        posix_path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or any(part in {"", ".", ".."} for part in path_parts)
        or posix_path.as_posix() != path
    ):
        raise ValueError(
            "path must be a canonical relative path without parent traversal"
        )


def _immutable_string_tuple(values: Any, field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{field_name} must be a collection of strings")
    try:
        frozen_values = tuple(values)
    except TypeError as error:
        raise ValueError(f"{field_name} must be a collection of strings") from error
    if any(not isinstance(value, str) for value in frozen_values):
        raise ValueError(f"{field_name} must contain only strings")
    return frozen_values


def _validate_stable_runtime_id(
    value: Any,
    field_name: str,
    prefix: str,
) -> str:
    if (
        not isinstance(value, str)
        or not _STABLE_RUNTIME_ID_PATTERN.fullmatch(value)
        or not value.startswith(f"{prefix}-")
    ):
        raise ValueError(
            f"{field_name} must use {prefix}- followed by a 64-character "
            "hexadecimal digest"
        )
    return value


def session_phases_for_schema(schema_version: int) -> tuple[RunPhase, ...]:
    if schema_version == LEGACY_SESSION_SCHEMA_VERSION:
        return LEGACY_SESSION_PHASES
    if schema_version in {
        PREVIOUS_SESSION_SCHEMA_VERSION,
        MODEL_STAGE_SESSION_SCHEMA_VERSION,
    }:
        return PREVIOUS_SESSION_PHASES
    if schema_version == SEMANTIC_RECONCILIATION_SESSION_SCHEMA_VERSION:
        return SEMANTIC_RECONCILIATION_SESSION_PHASES
    if schema_version == SESSION_SCHEMA_VERSION:
        return SESSION_PHASES
    raise ValueError(f"unsupported session schema_version: {schema_version}")


def initial_session_manifest(
    *,
    review_id: str,
    repository: RepositoryIdentity,
    revisions: ResolvedRevisions,
    execution: ReviewExecutionConfig,
    now: str,
) -> SessionManifest:
    return SessionManifest(
        schema_version=SESSION_SCHEMA_VERSION,
        review_id=review_id,
        parent_review_id=None,
        root_review_id=review_id,
        repository=repository,
        revisions=revisions,
        original_base_sha=revisions.resolved_base_sha,
        incremental_from_sha=None,
        revision_change_kind=RevisionChangeKind.INITIAL,
        execution=execution,
        status=RunStatus.CREATED,
        current_phase=RunPhase.CREATED,
        last_successful_phase=None,
        phases={phase.value: PhaseCheckpoint() for phase in SESSION_PHASES},
        artifacts={},
        errors=(),
        created_at=now,
        updated_at=now,
    )


def child_session_manifest(
    *,
    review_id: str,
    parent: SessionManifest,
    repository: RepositoryIdentity,
    revisions: ResolvedRevisions,
    change_kind: RevisionChangeKind,
    now: str,
    execution: ReviewExecutionConfig | None = None,
) -> SessionManifest:
    if change_kind is RevisionChangeKind.INITIAL:
        raise ValueError("child Session change_kind must describe revision drift")
    incremental_from_sha = (
        parent.revisions.resolved_head_sha
        if change_kind is RevisionChangeKind.HEAD_MOVED
        else None
    )
    if parent.schema_version == SESSION_SCHEMA_VERSION:
        if execution is not None and execution != parent.execution:
            raise ValueError(
                "v5 revision-drift child must preserve the parent's fixed "
                "execution config"
            )
        child_execution = parent.execution
    else:
        if execution is None or execution.memory is None:
            raise ValueError(
                "legacy revision-drift child requires an explicit v5 execution "
                "config with a fixed MemoryExecutionConfig"
            )
        legacy_projection = ReviewExecutionConfig(
            reviewer_provider=execution.reviewer_provider,
            reviewer_model=execution.reviewer_model,
            reviewer_base_url=execution.reviewer_base_url,
            reviewer_api_key_env=execution.reviewer_api_key_env,
            reviewer_mode=execution.reviewer_mode,
            reviewer_loop=execution.reviewer_loop,
            non_interactive=execution.non_interactive,
            risk_assessor=execution.risk_assessor,
            portfolio_planner=execution.portfolio_planner,
            semantic_reconciler=execution.semantic_reconciler,
            supplemental_policy=execution.supplemental_policy,
        )
        if legacy_projection != parent.execution:
            raise ValueError(
                "legacy revision-drift child must preserve the parent's fixed "
                "non-memory execution config"
            )
        child_execution = execution
    return SessionManifest(
        schema_version=SESSION_SCHEMA_VERSION,
        review_id=review_id,
        parent_review_id=parent.review_id,
        root_review_id=parent.root_review_id,
        repository=repository,
        revisions=revisions,
        original_base_sha=parent.original_base_sha,
        incremental_from_sha=incremental_from_sha,
        revision_change_kind=change_kind,
        execution=child_execution,
        status=RunStatus.CREATED,
        current_phase=RunPhase.CREATED,
        last_successful_phase=None,
        phases={phase.value: PhaseCheckpoint() for phase in SESSION_PHASES},
        artifacts={},
        errors=(),
        created_at=now,
        updated_at=now,
    )


def session_manifest_to_dict(manifest: SessionManifest) -> dict[str, Any]:
    def budget_payload(budget: SupplementalBudget) -> dict[str, Any]:
        return {
            "tasks": budget.tasks,
            "tool_calls": budget.tool_calls,
            "tokens": budget.tokens,
            "elapsed_seconds": budget.elapsed_seconds,
        }

    def supplemental_task_payload(
        checkpoint: SupplementalTaskCheckpoint,
    ) -> dict[str, Any]:
        return {
            "task_id": checkpoint.task_id,
            "assignment_digest": checkpoint.assignment_digest,
            "status": checkpoint.status.value,
            "attempts": checkpoint.attempts,
            "started_at": checkpoint.started_at,
            "completed_at": checkpoint.completed_at,
            "artifacts": list(checkpoint.artifacts),
            "reservation": budget_payload(checkpoint.reservation),
            "charged": budget_payload(checkpoint.charged),
            "unknown_consumed": budget_payload(checkpoint.unknown_consumed),
            "unknown_invocation_ids": list(checkpoint.unknown_invocation_ids),
            "error": checkpoint.error,
        }

    def wave_payload(checkpoint: ReviewWaveCheckpoint) -> dict[str, Any]:
        return {
            "wave_id": checkpoint.wave_id,
            "wave_index": checkpoint.wave_index,
            "trigger_digest": checkpoint.trigger_digest,
            "effective_policy": supplemental_policy_to_dict(
                checkpoint.effective_policy
            ),
            "status": checkpoint.status.value,
            "attempts": checkpoint.attempts,
            "started_at": checkpoint.started_at,
            "completed_at": checkpoint.completed_at,
            "artifacts": list(checkpoint.artifacts),
            "tasks": {
                task_id: supplemental_task_payload(task)
                for task_id, task in checkpoint.tasks.items()
            },
            "stop_reason": checkpoint.stop_reason,
            "error": checkpoint.error,
        }

    def phase_payload(checkpoint: PhaseCheckpoint) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": checkpoint.status.value,
            "attempts": checkpoint.attempts,
            "started_at": checkpoint.started_at,
            "completed_at": checkpoint.completed_at,
            "artifacts": list(checkpoint.artifacts),
            "error": checkpoint.error,
            "tasks": {
                task_name: {
                    "status": task.status.value,
                    "attempts": task.attempts,
                    "started_at": task.started_at,
                    "completed_at": task.completed_at,
                    "artifacts": list(task.artifacts),
                    "error": task.error,
                }
                for task_name, task in checkpoint.tasks.items()
            },
        }
        if manifest.schema_version >= PREVIOUS_SESSION_SCHEMA_VERSION:
            payload["user_decisions"] = dict(checkpoint.user_decisions)
        return payload

    complete_execution_payload = review_execution_config_to_dict(
        manifest.execution
    )
    execution_payload = {
        key: complete_execution_payload[key]
        for key in (
            "reviewer_provider",
            "reviewer_model",
            "reviewer_base_url",
            "reviewer_api_key_env",
            "reviewer_mode",
            "reviewer_loop",
            "non_interactive",
        )
    }
    if manifest.schema_version >= MODEL_STAGE_SESSION_SCHEMA_VERSION:
        execution_payload.update(
            {
                "risk_assessor": complete_execution_payload["risk_assessor"],
                "portfolio_planner": complete_execution_payload[
                    "portfolio_planner"
                ],
            }
        )
    if manifest.schema_version >= SEMANTIC_RECONCILIATION_SESSION_SCHEMA_VERSION:
        execution_payload.update(
            {
                "semantic_reconciler": complete_execution_payload[
                    "semantic_reconciler"
                ],
                "supplemental_policy": complete_execution_payload[
                    "supplemental_policy"
                ],
            }
        )
    if manifest.schema_version >= SESSION_SCHEMA_VERSION:
        if manifest.execution.memory is None:
            raise ValueError(
                "schema v5 Sessions require a fixed MemoryExecutionConfig"
            )
        execution_payload.update(
            {
                "memory": complete_execution_payload["memory"],
                "memory_curator": complete_execution_payload["memory_curator"],
            }
        )

    payload = {
        "schema_version": manifest.schema_version,
        "review_id": manifest.review_id,
        "parent_review_id": manifest.parent_review_id,
        "root_review_id": manifest.root_review_id,
        "repository": {
            "canonical_path": manifest.repository.canonical_path,
            "git_common_dir": manifest.repository.git_common_dir,
            "origin_url": manifest.repository.origin_url,
        },
        "revisions": {
            "requested_base": manifest.revisions.requested_base,
            "requested_head": manifest.revisions.requested_head,
            "resolved_base_sha": manifest.revisions.resolved_base_sha,
            "resolved_head_sha": manifest.revisions.resolved_head_sha,
            "original_base_sha": manifest.original_base_sha,
            "incremental_from_sha": manifest.incremental_from_sha,
            "change_kind": manifest.revision_change_kind.value,
        },
        "execution": execution_payload,
        "status": manifest.status.value,
        "current_phase": manifest.current_phase.value,
        "last_successful_phase": (
            manifest.last_successful_phase.value
            if manifest.last_successful_phase is not None
            else None
        ),
        "phases": {
            name: phase_payload(checkpoint)
            for name, checkpoint in manifest.phases.items()
        },
        "artifacts": {
            name: {
                "name": descriptor.name,
                "path": descriptor.path,
                "sha256": descriptor.sha256,
                "schema": descriptor.schema,
                "phase": descriptor.phase.value,
                "revision_binding": descriptor.revision_binding,
            }
            for name, descriptor in manifest.artifacts.items()
        },
        "errors": list(manifest.errors),
        "created_at": manifest.created_at,
        "updated_at": manifest.updated_at,
    }
    if (
        manifest.schema_version
        >= SEMANTIC_RECONCILIATION_SESSION_SCHEMA_VERSION
    ):
        payload["supplemental_waves"] = {
            wave_id: wave_payload(wave)
            for wave_id, wave in manifest.supplemental_waves.items()
        }
    return payload


def session_manifest_from_dict(payload: Mapping[str, Any]) -> SessionManifest:
    root = _object(payload, "session")
    schema_version = _integer(root, "schema_version", "session")
    if schema_version not in SUPPORTED_SESSION_SCHEMA_VERSIONS:
        raise ValueError(
            "unsupported session schema_version: "
            f"{schema_version}; expected one of "
            + ", ".join(str(version) for version in SUPPORTED_SESSION_SCHEMA_VERSIONS)
        )
    root_fields = {
        "schema_version",
        "review_id",
        "parent_review_id",
        "root_review_id",
        "repository",
        "revisions",
        "execution",
        "status",
        "current_phase",
        "last_successful_phase",
        "phases",
        "artifacts",
        "errors",
        "created_at",
        "updated_at",
    }
    if schema_version >= SEMANTIC_RECONCILIATION_SESSION_SCHEMA_VERSION:
        root_fields.add("supplemental_waves")
    _exact_fields(
        root,
        root_fields,
        "session",
    )

    repository_payload = _object_field(root, "repository", "session")
    _exact_fields(
        repository_payload,
        {"canonical_path", "git_common_dir", "origin_url"},
        "session.repository",
    )
    repository = RepositoryIdentity(
        canonical_path=_string(repository_payload, "canonical_path", "session.repository"),
        git_common_dir=_string(repository_payload, "git_common_dir", "session.repository"),
        origin_url=_optional_string(repository_payload, "origin_url", "session.repository"),
    )

    revisions_payload = _object_field(root, "revisions", "session")
    _exact_fields(
        revisions_payload,
        {
            "requested_base",
            "requested_head",
            "resolved_base_sha",
            "resolved_head_sha",
            "original_base_sha",
            "incremental_from_sha",
            "change_kind",
        },
        "session.revisions",
    )
    revisions = ResolvedRevisions(
        requested_base=_string(revisions_payload, "requested_base", "session.revisions"),
        requested_head=_string(revisions_payload, "requested_head", "session.revisions"),
        resolved_base_sha=_string(
            revisions_payload,
            "resolved_base_sha",
            "session.revisions",
        ),
        resolved_head_sha=_string(
            revisions_payload,
            "resolved_head_sha",
            "session.revisions",
        ),
    )

    execution_payload = _object_field(root, "execution", "session")
    execution_fields = {
        "reviewer_provider",
        "reviewer_model",
        "reviewer_base_url",
        "reviewer_api_key_env",
        "reviewer_mode",
        "reviewer_loop",
        "non_interactive",
    }
    if schema_version >= MODEL_STAGE_SESSION_SCHEMA_VERSION:
        execution_fields |= {"risk_assessor", "portfolio_planner"}
    if schema_version >= SEMANTIC_RECONCILIATION_SESSION_SCHEMA_VERSION:
        execution_fields |= {"semantic_reconciler", "supplemental_policy"}
    if schema_version >= SESSION_SCHEMA_VERSION:
        execution_fields |= {"memory", "memory_curator"}
    _exact_fields(
        execution_payload,
        execution_fields,
        "session.execution",
    )
    risk_assessor = ModelStageConfig()
    portfolio_planner = ModelStageConfig()
    semantic_reconciler = ModelStageConfig()
    supplemental_policy = SupplementalPolicy()
    memory: MemoryExecutionConfig | None = None
    memory_curator = ModelStageConfig()
    if schema_version >= MODEL_STAGE_SESSION_SCHEMA_VERSION:
        risk_assessor = _model_stage_config_from_dict(
            _object_field(execution_payload, "risk_assessor", "session.execution"),
            "session.execution.risk_assessor",
        )
        portfolio_planner = _model_stage_config_from_dict(
            _object_field(execution_payload, "portfolio_planner", "session.execution"),
            "session.execution.portfolio_planner",
        )
    if schema_version >= SEMANTIC_RECONCILIATION_SESSION_SCHEMA_VERSION:
        semantic_reconciler = _model_stage_config_from_dict(
            _object_field(
                execution_payload,
                "semantic_reconciler",
                "session.execution",
            ),
            "session.execution.semantic_reconciler",
        )
        supplemental_policy = _supplemental_policy_from_dict(
            _object_field(
                execution_payload,
                "supplemental_policy",
                "session.execution",
            ),
            "session.execution.supplemental_policy",
        )
    if schema_version >= SESSION_SCHEMA_VERSION:
        memory = MemoryExecutionConfig.from_dict(
            _object_field(execution_payload, "memory", "session.execution")
        )
        memory_curator = _model_stage_config_from_dict(
            _object_field(
                execution_payload,
                "memory_curator",
                "session.execution",
            ),
            "session.execution.memory_curator",
        )
    execution = ReviewExecutionConfig(
        reviewer_provider=_string(
            execution_payload,
            "reviewer_provider",
            "session.execution",
        ),
        reviewer_model=_optional_string(
            execution_payload,
            "reviewer_model",
            "session.execution",
        ),
        reviewer_base_url=_optional_string(
            execution_payload,
            "reviewer_base_url",
            "session.execution",
        ),
        reviewer_api_key_env=_string(
            execution_payload,
            "reviewer_api_key_env",
            "session.execution",
        ),
        reviewer_mode=_string(
            execution_payload,
            "reviewer_mode",
            "session.execution",
        ),
        reviewer_loop=_string(
            execution_payload,
            "reviewer_loop",
            "session.execution",
        ),
        non_interactive=_boolean(
            execution_payload,
            "non_interactive",
            "session.execution",
        ),
        risk_assessor=risk_assessor,
        portfolio_planner=portfolio_planner,
        semantic_reconciler=semantic_reconciler,
        supplemental_policy=supplemental_policy,
        memory=memory,
        memory_curator=memory_curator,
    )

    phases_payload = _object_field(root, "phases", "session")
    layout_phases = session_phases_for_schema(schema_version)
    expected_phase_names = {phase.value for phase in layout_phases}
    _exact_fields(phases_payload, expected_phase_names, "session.phases")
    phases = {
        phase.value: _phase_checkpoint_from_dict(
            _object_field(phases_payload, phase.value, "session.phases"),
            f"session.phases.{phase.value}",
            schema_version=schema_version,
        )
        for phase in layout_phases
    }

    artifacts_payload = _object_field(root, "artifacts", "session")
    artifacts: dict[str, ArtifactDescriptor] = {}
    for artifact_name, artifact_payload in artifacts_payload.items():
        if not isinstance(artifact_name, str):
            raise ValueError("session.artifacts keys must be strings")
        descriptor = _artifact_descriptor_from_dict(
            _object(artifact_payload, f"session.artifacts.{artifact_name}"),
            f"session.artifacts.{artifact_name}",
        )
        if descriptor.name != artifact_name:
            raise ValueError(
                f"session.artifacts.{artifact_name}.name must match its registry key"
            )
        artifacts[artifact_name] = descriptor

    supplemental_waves: dict[str, ReviewWaveCheckpoint] = {}
    if schema_version >= SEMANTIC_RECONCILIATION_SESSION_SCHEMA_VERSION:
        waves_payload = _object_field(root, "supplemental_waves", "session")
        for wave_id, wave_payload in waves_payload.items():
            if not isinstance(wave_id, str):
                raise ValueError("session.supplemental_waves keys must be strings")
            wave = _review_wave_checkpoint_from_dict(
                _object(
                    wave_payload,
                    f"session.supplemental_waves.{wave_id}",
                ),
                f"session.supplemental_waves.{wave_id}",
            )
            if wave.wave_id != wave_id:
                raise ValueError(
                    f"session.supplemental_waves.{wave_id}.wave_id must match "
                    "its registry key"
                )
            supplemental_waves[wave_id] = wave

    last_successful_phase_value = root["last_successful_phase"]
    last_successful_phase = (
        None
        if last_successful_phase_value is None
        else _enum_value(
            RunPhase,
            last_successful_phase_value,
            "session.last_successful_phase",
        )
    )

    return SessionManifest(
        schema_version=schema_version,
        review_id=_string(root, "review_id", "session"),
        parent_review_id=_optional_string(root, "parent_review_id", "session"),
        root_review_id=_string(root, "root_review_id", "session"),
        repository=repository,
        revisions=revisions,
        original_base_sha=_string(
            revisions_payload,
            "original_base_sha",
            "session.revisions",
        ),
        incremental_from_sha=_optional_string(
            revisions_payload,
            "incremental_from_sha",
            "session.revisions",
        ),
        revision_change_kind=_enum_field(
            RevisionChangeKind,
            revisions_payload,
            "change_kind",
            "session.revisions",
        ),
        execution=execution,
        status=_enum_field(RunStatus, root, "status", "session"),
        current_phase=_enum_field(RunPhase, root, "current_phase", "session"),
        last_successful_phase=last_successful_phase,
        phases=phases,
        artifacts=artifacts,
        errors=_string_list(root, "errors", "session"),
        created_at=_string(root, "created_at", "session"),
        updated_at=_string(root, "updated_at", "session"),
        supplemental_waves=supplemental_waves,
    )


def _phase_checkpoint_from_dict(
    payload: Mapping[str, Any],
    context: str,
    *,
    schema_version: int,
) -> PhaseCheckpoint:
    required = {"status", "attempts", "started_at", "completed_at", "artifacts", "error"}
    optional = {"tasks"}
    if schema_version >= PREVIOUS_SESSION_SCHEMA_VERSION:
        required |= {"tasks", "user_decisions"}
        optional = set()
    missing = required - set(payload)
    if missing:
        raise ValueError(
            f"{context} is missing required field(s): {', '.join(sorted(missing))}"
        )
    unexpected = set(payload) - required - optional
    if unexpected:
        names = ", ".join(sorted(str(name).casefold() for name in unexpected))
        raise ValueError(f"{context} contains unsupported field(s): {names}")
    attempts = _integer(payload, "attempts", context)
    if attempts < 0:
        raise ValueError(f"{context}.attempts must be non-negative")
    tasks_payload = payload.get("tasks", {})
    tasks_object = _object(tasks_payload, f"{context}.tasks")
    tasks = {
        task_name: _reviewer_task_checkpoint_from_dict(
            _object(task_payload, f"{context}.tasks.{task_name}"),
            f"{context}.tasks.{task_name}",
        )
        for task_name, task_payload in tasks_object.items()
    }
    decisions_payload = payload.get("user_decisions", {})
    decisions_object = _object(decisions_payload, f"{context}.user_decisions")
    user_decisions = {
        _require_non_empty_string(event_id, f"{context}.user_decisions event id"):
        _require_non_empty_string(
            artifact_name,
            f"{context}.user_decisions.{event_id}",
        )
        for event_id, artifact_name in decisions_object.items()
    }
    return PhaseCheckpoint(
        status=_enum_field(PhaseStatus, payload, "status", context),
        attempts=attempts,
        started_at=_optional_string(payload, "started_at", context),
        completed_at=_optional_string(payload, "completed_at", context),
        artifacts=_string_list(payload, "artifacts", context),
        error=_optional_string(payload, "error", context),
        tasks=tasks,
        user_decisions=user_decisions,
    )


def _model_stage_config_from_dict(
    payload: Mapping[str, Any],
    context: str,
) -> ModelStageConfig:
    _exact_fields(
        payload,
        {
            "mode",
            "provider",
            "model",
            "base_url",
            "api_key_env",
            "max_output_tokens",
            "max_provider_attempts",
            "max_elapsed_seconds",
        },
        context,
    )
    return ModelStageConfig(
        mode=_string(payload, "mode", context),
        provider=_string(payload, "provider", context),
        model=_optional_string(payload, "model", context),
        base_url=_optional_string(payload, "base_url", context),
        api_key_env=_string(payload, "api_key_env", context),
        max_output_tokens=_integer(payload, "max_output_tokens", context),
        max_provider_attempts=_integer(
            payload,
            "max_provider_attempts",
            context,
        ),
        max_elapsed_seconds=_number(
            payload,
            "max_elapsed_seconds",
            context,
        ),
    )


def _supplemental_policy_from_dict(
    payload: Mapping[str, Any],
    context: str,
) -> SupplementalPolicy:
    _exact_fields(
        payload,
        {
            "version",
            "risk_level",
            "max_waves",
            "max_tasks",
            "max_tasks_per_wave",
            "max_concurrency",
            "max_turns_per_task",
            "max_tool_calls_per_task",
            "max_tokens_per_task",
            "max_total_tokens",
            "max_elapsed_seconds",
        },
        context,
    )
    return SupplementalPolicy(
        version=_string(payload, "version", context),
        risk_level=_string(payload, "risk_level", context),
        max_waves=_integer(payload, "max_waves", context),
        max_tasks=_integer(payload, "max_tasks", context),
        max_tasks_per_wave=_integer(payload, "max_tasks_per_wave", context),
        max_concurrency=_integer(payload, "max_concurrency", context),
        max_turns_per_task=_integer(payload, "max_turns_per_task", context),
        max_tool_calls_per_task=_integer(
            payload,
            "max_tool_calls_per_task",
            context,
        ),
        max_tokens_per_task=_integer(payload, "max_tokens_per_task", context),
        max_total_tokens=_integer(payload, "max_total_tokens", context),
        max_elapsed_seconds=_number(payload, "max_elapsed_seconds", context),
    )


def _supplemental_budget_from_dict(
    payload: Mapping[str, Any],
    context: str,
) -> SupplementalBudget:
    _exact_fields(
        payload,
        {"tasks", "tool_calls", "tokens", "elapsed_seconds"},
        context,
    )
    return SupplementalBudget(
        tasks=_integer(payload, "tasks", context),
        tool_calls=_integer(payload, "tool_calls", context),
        tokens=_integer(payload, "tokens", context),
        elapsed_seconds=_number(payload, "elapsed_seconds", context),
    )


def _supplemental_task_checkpoint_from_dict(
    payload: Mapping[str, Any],
    context: str,
) -> SupplementalTaskCheckpoint:
    _exact_fields(
        payload,
        {
            "task_id",
            "assignment_digest",
            "status",
            "attempts",
            "started_at",
            "completed_at",
            "artifacts",
            "reservation",
            "charged",
            "unknown_consumed",
            "unknown_invocation_ids",
            "error",
        },
        context,
    )
    return SupplementalTaskCheckpoint(
        task_id=_string(payload, "task_id", context),
        assignment_digest=_string(payload, "assignment_digest", context),
        status=_enum_field(SupplementalTaskStatus, payload, "status", context),
        attempts=_integer(payload, "attempts", context),
        started_at=_optional_string(payload, "started_at", context),
        completed_at=_optional_string(payload, "completed_at", context),
        artifacts=_string_list(payload, "artifacts", context),
        reservation=_supplemental_budget_from_dict(
            _object_field(payload, "reservation", context),
            f"{context}.reservation",
        ),
        charged=_supplemental_budget_from_dict(
            _object_field(payload, "charged", context),
            f"{context}.charged",
        ),
        unknown_consumed=_supplemental_budget_from_dict(
            _object_field(payload, "unknown_consumed", context),
            f"{context}.unknown_consumed",
        ),
        unknown_invocation_ids=_string_list(
            payload,
            "unknown_invocation_ids",
            context,
        ),
        error=_optional_string(payload, "error", context),
    )


def _review_wave_checkpoint_from_dict(
    payload: Mapping[str, Any],
    context: str,
) -> ReviewWaveCheckpoint:
    _exact_fields(
        payload,
        {
            "wave_id",
            "wave_index",
            "trigger_digest",
            "effective_policy",
            "status",
            "attempts",
            "started_at",
            "completed_at",
            "artifacts",
            "tasks",
            "stop_reason",
            "error",
        },
        context,
    )
    tasks_payload = _object_field(payload, "tasks", context)
    tasks: dict[str, SupplementalTaskCheckpoint] = {}
    for task_id, task_payload in tasks_payload.items():
        if not isinstance(task_id, str):
            raise ValueError(f"{context}.tasks keys must be strings")
        task_context = f"{context}.tasks.{task_id}"
        task = _supplemental_task_checkpoint_from_dict(
            _object(task_payload, task_context),
            task_context,
        )
        if task.task_id != task_id:
            raise ValueError(f"{task_context}.task_id must match its registry key")
        tasks[task_id] = task
    return ReviewWaveCheckpoint(
        wave_id=_string(payload, "wave_id", context),
        wave_index=_integer(payload, "wave_index", context),
        trigger_digest=_string(payload, "trigger_digest", context),
        effective_policy=_supplemental_policy_from_dict(
            _object_field(payload, "effective_policy", context),
            f"{context}.effective_policy",
        ),
        status=_enum_field(PhaseStatus, payload, "status", context),
        attempts=_integer(payload, "attempts", context),
        started_at=_optional_string(payload, "started_at", context),
        completed_at=_optional_string(payload, "completed_at", context),
        artifacts=_string_list(payload, "artifacts", context),
        tasks=tasks,
        stop_reason=_optional_string(payload, "stop_reason", context),
        error=_optional_string(payload, "error", context),
    )


def _reviewer_task_checkpoint_from_dict(
    payload: Mapping[str, Any],
    context: str,
) -> ReviewerTaskCheckpoint:
    _exact_fields(
        payload,
        {"status", "attempts", "started_at", "completed_at", "artifacts", "error"},
        context,
    )
    attempts = _integer(payload, "attempts", context)
    if attempts < 0:
        raise ValueError(f"{context}.attempts must be non-negative")
    return ReviewerTaskCheckpoint(
        status=_enum_field(PhaseStatus, payload, "status", context),
        attempts=attempts,
        started_at=_optional_string(payload, "started_at", context),
        completed_at=_optional_string(payload, "completed_at", context),
        artifacts=_string_list(payload, "artifacts", context),
        error=_optional_string(payload, "error", context),
    )


def _artifact_descriptor_from_dict(
    payload: Mapping[str, Any],
    context: str,
) -> ArtifactDescriptor:
    _exact_fields(
        payload,
        {"name", "path", "sha256", "schema", "phase", "revision_binding"},
        context,
    )
    return ArtifactDescriptor(
        name=_string(payload, "name", context),
        path=_string(payload, "path", context),
        sha256=_string(payload, "sha256", context),
        schema=_string(payload, "schema", context),
        phase=_enum_field(RunPhase, payload, "phase", context),
        revision_binding=_optional_string(payload, "revision_binding", context),
    )


def _object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _object_field(
    payload: Mapping[str, Any],
    field_name: str,
    context: str,
) -> Mapping[str, Any]:
    return _object(payload[field_name], f"{context}.{field_name}")


def _exact_fields(
    payload: Mapping[str, Any],
    expected: set[str],
    context: str,
) -> None:
    keys = set(payload)
    missing = expected - keys
    if missing:
        raise ValueError(
            f"{context} is missing required field(s): {', '.join(sorted(missing))}"
        )
    unexpected = keys - expected
    if unexpected:
        names = ", ".join(sorted(str(name).casefold() for name in unexpected))
        raise ValueError(f"{context} contains unsupported field(s): {names}")


def _string(payload: Mapping[str, Any], field_name: str, context: str) -> str:
    value = payload[field_name]
    if not isinstance(value, str):
        raise ValueError(f"{context}.{field_name} must be a string")
    return value


def _optional_string(
    payload: Mapping[str, Any],
    field_name: str,
    context: str,
) -> str | None:
    value = payload[field_name]
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{context}.{field_name} must be a string or null")
    return value


def _integer(payload: Mapping[str, Any], field_name: str, context: str) -> int:
    value = payload[field_name]
    if type(value) is not int:
        raise ValueError(f"{context}.{field_name} must be an integer")
    return value


def _number(payload: Mapping[str, Any], field_name: str, context: str) -> int | float:
    value = payload[field_name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context}.{field_name} must be a number")
    return value


def _boolean(payload: Mapping[str, Any], field_name: str, context: str) -> bool:
    value = payload[field_name]
    if type(value) is not bool:
        raise ValueError(f"{context}.{field_name} must be a boolean")
    return value


def _string_list(
    payload: Mapping[str, Any],
    field_name: str,
    context: str,
) -> list[str]:
    value = payload[field_name]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{context}.{field_name} must be a list of strings")
    return list(value)


EnumType = TypeVar("EnumType", bound=Enum)


def _enum_field(
    enum_type: type[EnumType],
    payload: Mapping[str, Any],
    field_name: str,
    context: str,
) -> EnumType:
    return _enum_value(enum_type, payload[field_name], f"{context}.{field_name}")


def _enum_value(
    enum_type: type[EnumType],
    value: Any,
    context: str,
) -> EnumType:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be a string")
    try:
        return enum_type(value)
    except ValueError as error:
        raise ValueError(f"{context} has unsupported value: {value}") from error


_V6_SESSION_ID = re.compile(r"^SESSION-[0-9a-f]{64}$")
_V6_PR_ID = re.compile(r"^PR-[0-9a-f]{64}$")
_V6_SNAPSHOT_ID = re.compile(r"^S-[0-9a-f]{64}$")
_V6_ARTIFACT_ID = re.compile(r"^A-[0-9a-f]{64}$")
_V6_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class SessionV6ArtifactRef:
    logical_name: str
    artifact_id: str
    relative_path: str
    sha256: str

    def __post_init__(self) -> None:
        if type(self.logical_name) is not str or not self.logical_name.strip():
            raise ValueError("Session v6 logical artifact name is invalid")
        if _V6_ARTIFACT_ID.fullmatch(self.artifact_id) is None:
            raise ValueError("Session v6 artifact_id is invalid")
        if type(self.relative_path) is not str or not self.relative_path:
            raise ValueError("Session v6 artifact path is invalid")
        posix = PurePosixPath(self.relative_path)
        windows = PureWindowsPath(self.relative_path)
        if (
            posix.is_absolute()
            or windows.is_absolute()
            or "\\" in self.relative_path
            or any(part in {"", ".", ".."} for part in posix.parts)
        ):
            raise ValueError("Session v6 artifact path must be repository-relative")
        if _V6_SHA256.fullmatch(self.sha256) is None:
            raise ValueError("Session v6 artifact hash is invalid")

    def to_dict(self) -> dict[str, str]:
        return {
            "logical_name": self.logical_name,
            "artifact_id": self.artifact_id,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SessionV6ArtifactRef":
        if type(payload) is not dict or set(payload) != {
            "logical_name",
            "artifact_id",
            "relative_path",
            "sha256",
        }:
            raise ValueError("Session v6 artifact reference schema is invalid")
        return cls(**dict(payload))


@dataclass(frozen=True)
class SessionV6PhaseCheckpoint:
    status: PhaseStatus = PhaseStatus.PENDING
    attempt: int = 0
    artifacts: tuple[SessionV6ArtifactRef, ...] = ()
    error_code: str | None = None
    invalidation_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, PhaseStatus):
            raise ValueError("Session v6 Phase status is invalid")
        if self.status is PhaseStatus.AWAITING_USER:
            raise ValueError("Session v6 has no awaiting-user Phase state")
        if type(self.attempt) is not int or self.attempt < 0:
            raise ValueError("Session v6 Phase attempt is invalid")
        if type(self.artifacts) is not tuple or any(
            type(item) is not SessionV6ArtifactRef for item in self.artifacts
        ):
            raise ValueError("Session v6 Phase artifacts are invalid")
        names = [item.logical_name for item in self.artifacts]
        if len(names) != len(set(names)):
            raise ValueError("Session v6 Phase artifact names must be unique")
        if self.status is PhaseStatus.PENDING:
            if self.attempt != 0 or self.artifacts or self.error_code is not None:
                raise ValueError("Pending Session v6 Phase contains progress")
            if self.invalidation_reason is not None:
                raise ValueError("Pending Session v6 Phase is invalidated")
        elif self.status is PhaseStatus.RUNNING:
            if self.attempt < 1 or self.error_code is not None:
                raise ValueError("Running Session v6 Phase state is invalid")
            if self.invalidation_reason is not None:
                raise ValueError("Running Session v6 Phase is invalidated")
        elif self.status is PhaseStatus.COMPLETED:
            if self.attempt < 1 or self.error_code is not None:
                raise ValueError("Completed Session v6 Phase state is invalid")
            if self.invalidation_reason is not None:
                raise ValueError("Completed Session v6 Phase is invalidated")
        elif self.status is PhaseStatus.FAILED:
            if self.attempt < 1 or not _v6_text_or_none(self.error_code):
                raise ValueError("Failed Session v6 Phase requires an error code")
            if self.invalidation_reason is not None:
                raise ValueError("Failed Session v6 Phase is invalidated")
        elif self.status is PhaseStatus.INVALIDATED:
            if not _v6_text_or_none(self.invalidation_reason):
                raise ValueError(
                    "Invalidated Session v6 Phase requires a reason"
                )
            if self.error_code is not None:
                raise ValueError("Invalidated Session v6 Phase has an error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "attempt": self.attempt,
            "artifacts": [item.to_dict() for item in self.artifacts],
            "error_code": self.error_code,
            "invalidation_reason": self.invalidation_reason,
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "SessionV6PhaseCheckpoint":
        if type(payload) is not dict or set(payload) != {
            "status",
            "attempt",
            "artifacts",
            "error_code",
            "invalidation_reason",
        }:
            raise ValueError("Session v6 Phase checkpoint schema is invalid")
        artifacts = payload["artifacts"]
        if type(artifacts) is not list:
            raise ValueError("Session v6 Phase artifacts must be an array")
        return cls(
            status=PhaseStatus(payload["status"]),
            attempt=payload["attempt"],
            artifacts=tuple(SessionV6ArtifactRef.from_dict(item) for item in artifacts),
            error_code=payload["error_code"],
            invalidation_reason=payload["invalidation_reason"],
        )


@dataclass(frozen=True)
class SessionV6Manifest:
    session_id: str
    pr_id: str
    snapshot_id: str
    status: RunStatus
    current_phase: RunPhase
    phases: Mapping[str, SessionV6PhaseCheckpoint]
    revision: int = 0
    schema_version: int = SESSION_V6_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SESSION_V6_SCHEMA_VERSION:
            raise ValueError("Session v6 schema_version is invalid")
        if _V6_SESSION_ID.fullmatch(self.session_id) is None:
            raise ValueError("Session v6 session_id is invalid")
        if _V6_PR_ID.fullmatch(self.pr_id) is None:
            raise ValueError("Session v6 pr_id is invalid")
        if _V6_SNAPSHOT_ID.fullmatch(self.snapshot_id) is None:
            raise ValueError("Session v6 snapshot_id is invalid")
        if not isinstance(self.status, RunStatus) or self.status is RunStatus.AWAITING_USER:
            raise ValueError("Session v6 Run status is invalid")
        if not isinstance(self.current_phase, RunPhase):
            raise ValueError("Session v6 current_phase is invalid")
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("Session v6 revision is invalid")
        if not isinstance(self.phases, Mapping) or set(self.phases) != {
            phase.value for phase in SESSION_V6_PHASES
        }:
            raise ValueError("Session v6 Phase layout is invalid")
        normalized = {
            phase.value: self.phases[phase.value]
            for phase in SESSION_V6_PHASES
        }
        if any(
            type(checkpoint) is not SessionV6PhaseCheckpoint
            for checkpoint in normalized.values()
        ):
            raise ValueError("Session v6 Phase checkpoint is invalid")
        object.__setattr__(self, "phases", MappingProxyType(normalized))
        self._validate_state()

    def _validate_state(self) -> None:
        checkpoints = [self.phases[phase.value] for phase in SESSION_V6_PHASES]
        seen_incomplete = False
        for checkpoint in checkpoints:
            if checkpoint.status is PhaseStatus.COMPLETED:
                if seen_incomplete:
                    raise ValueError("Session v6 completed Phases are not a prefix")
            else:
                seen_incomplete = True
        all_completed = all(
            checkpoint.status is PhaseStatus.COMPLETED
            for checkpoint in checkpoints
        )
        if self.status is RunStatus.COMPLETED:
            if self.current_phase is not RunPhase.COMPLETED or not all_completed:
                raise ValueError("Completed Session v6 state is inconsistent")
            return
        if self.current_phase not in SESSION_V6_PHASES:
            raise ValueError("Active Session v6 current Phase is invalid")
        current = self.phases[self.current_phase.value]
        if self.status is RunStatus.CREATED:
            if (
                self.current_phase is not RunPhase.PREFLIGHT
                or any(item.status is not PhaseStatus.PENDING for item in checkpoints)
            ):
                raise ValueError("Created Session v6 state is inconsistent")
        elif self.status is RunStatus.FAILED:
            if current.status is not PhaseStatus.FAILED:
                raise ValueError("Failed Session v6 current Phase is inconsistent")
        elif self.status is RunStatus.RUNNING:
            if current.status not in {
                PhaseStatus.PENDING,
                PhaseStatus.RUNNING,
                PhaseStatus.FAILED,
                PhaseStatus.INVALIDATED,
            }:
                raise ValueError("Running Session v6 current Phase is inconsistent")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "revision": self.revision,
            "session_id": self.session_id,
            "pr_id": self.pr_id,
            "snapshot_id": self.snapshot_id,
            "status": self.status.value,
            "current_phase": self.current_phase.value,
            "phases": {
                phase.value: self.phases[phase.value].to_dict()
                for phase in SESSION_V6_PHASES
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SessionV6Manifest":
        if type(payload) is not dict or set(payload) != {
            "schema_version",
            "revision",
            "session_id",
            "pr_id",
            "snapshot_id",
            "status",
            "current_phase",
            "phases",
        }:
            raise ValueError("Session v6 manifest schema is invalid")
        phases = payload["phases"]
        if type(phases) is not dict:
            raise ValueError("Session v6 phases must be an object")
        return cls(
            schema_version=payload["schema_version"],
            revision=payload["revision"],
            session_id=payload["session_id"],
            pr_id=payload["pr_id"],
            snapshot_id=payload["snapshot_id"],
            status=RunStatus(payload["status"]),
            current_phase=RunPhase(payload["current_phase"]),
            phases={
                name: SessionV6PhaseCheckpoint.from_dict(value)
                for name, value in phases.items()
            },
        )


def new_session_v6_manifest(
    *,
    session_id: str,
    pr_id: str,
    snapshot_id: str,
) -> SessionV6Manifest:
    return SessionV6Manifest(
        session_id=session_id,
        pr_id=pr_id,
        snapshot_id=snapshot_id,
        status=RunStatus.CREATED,
        current_phase=RunPhase.PREFLIGHT,
        phases={
            phase.value: SessionV6PhaseCheckpoint()
            for phase in SESSION_V6_PHASES
        },
    )


def session_v6_manifest_to_dict(manifest: SessionV6Manifest) -> dict[str, Any]:
    if type(manifest) is not SessionV6Manifest:
        raise ValueError("manifest must be SessionV6Manifest")
    return manifest.to_dict()


def session_v6_manifest_from_dict(
    payload: Mapping[str, Any],
) -> SessionV6Manifest:
    return SessionV6Manifest.from_dict(payload)


def _v6_text_or_none(value: object) -> bool:
    return type(value) is str and bool(value.strip()) and "\x00" not in value
