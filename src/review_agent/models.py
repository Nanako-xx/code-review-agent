from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class IntentSource(str, Enum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"


class IntentStatus(str, Enum):
    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


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

    @classmethod
    def for_risk(cls, risk: RiskLevel) -> "ReviewProfile":
        profiles = {
            RiskLevel.LOW: cls(1, 6, 12, ["core"]),
            RiskLevel.MEDIUM: cls(2, 10, 24, ["core", "adversarial"]),
            RiskLevel.HIGH: cls(3, 16, 40, ["core", "adversarial", "dynamic_specialist"]),
            RiskLevel.CRITICAL: cls(4, 24, 64, ["core", "adversarial", "security_specialist", "domain_specialist"]),
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
    repository_permission: str = "read_only"
    command_permission: str = "safe_checks_only"


@dataclass(frozen=True)
class ModelInvocationEnvelope:
    system: str
    tools: list[dict[str, object]]
    messages: list[dict[str, object]]
    parameters: dict[str, object]
