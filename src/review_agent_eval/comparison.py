"""Strict, source-bound paired comparison for completed Eval v2 Evaluations.

This module consumes hydrated Evaluation bundles and canonical TrialScore
statistics only.  It never runs an Agent, Judge, acquisition client, network
operation, or product Runtime.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .analysis_artifacts import AnalysisSourceBinding, bind_analysis_source
from .artifacts import ArtifactIntegrityError
from .cases import RunCaseSnapshot
from .config import (
    EvalRunConfig,
    MAX_TRIAL_COUNT,
    derive_evaluation_id,
    derive_trial_id,
)
from .metrics import CoreMetric, MetricKind, PPM_SCALE
from .models import (
    SubmissionStatus,
    _JsonModel,
    _strict_json_loads,
    canonical_json_bytes,
    canonical_sha256,
    stable_id,
)
from .statistics import (
    MAX_BOOTSTRAP_CASES,
    MAX_BOOTSTRAP_ITERATIONS,
    MAX_BOOTSTRAP_SEED,
    MAX_RUN_BOOTSTRAP_DRAWS,
    CaseContributionV1,
    MetricDirection,
    MetricUnit,
    RunStatisticsV1,
    StatisticsCoverageV1,
    StatisticsMetricStatus,
    StatisticsPolicyV1,
    _sample_metric_value,
    compute_run_statistics,
)


COMPARISON_POLICY_SCHEMA_VERSION = "comparison_policy_v1"
COMPARISON_COMPATIBILITY_SCHEMA_VERSION = "comparison_compatibility_v1"
RUN_COMPARISON_SCHEMA_VERSION = "run_comparison_v1"
COMPARISON_ALGORITHM_VERSION = "strict-paired-comparison-v1"
MIN_COMPARISON_TRIAL_COUNT = 3
MAX_COMPARISON_BYTES = 768 * 1024 * 1024

REQUIRED_CASE_FIELDS = (
    "suite.id",
    "suite.version",
    "suite.manifest_digest",
    "case_snapshot.digest",
    "cases.task_ids",
    "cases.versions",
    "cases.canonical_case_digests",
    "cases.input_digests",
    "cases.truth_digests",
    "cases.snapshot_entry_digests",
    "cases.truth_completeness",
    "cases.novel_finding_policy",
    "cases.metric_authority_profile_digests",
    "cases.protocol_ids",
)
REQUIRED_EVALUATOR_FIELDS = (
    "trial.count",
    "target.kind",
    "wire_contract.digest",
    "materialization.protocol",
    "isolation.profile",
    "clarification_matcher.digest",
    "evaluator.evaluation_revision",
    "evaluator.execution.digest",
    "evaluator.configuration.digest",
    "evaluator.judge_profiles.digest",
    "evaluator.judge_rubrics.digest",
    "evaluator.judge_execution.digest",
    "metrics_policy.digest",
    "metric_authority.profile.digest",
    "metric_authority.policy.digest",
    "scoring_policy.digest",
    "statistics_policy.digest",
)
_PROJECTION_FIELDS = REQUIRED_CASE_FIELDS + REQUIRED_EVALUATOR_FIELDS
_AGENT_FIELDS = (
    "run_id",
    "evaluation_id",
    "run_instance_key",
    "agent_config_digest",
    "agent.id",
    "agent.name",
    "agent.version",
    "agent.commit",
    "agent.model",
    "agent.provider",
    "agent.parameters_digest",
    "agent.prompt_config_digest",
    "adapter_capabilities_digest",
    "adapter.id",
    "adapter.version",
    "adapter.internal_policy_digest",
    "agent.resource_policy_digest",
)


class ComparisonError(ValueError):
    """A comparison policy, source, or result is invalid."""


class ComparisonStatus(str, Enum):
    COMPARABLE = "comparable"
    NOT_COMPARABLE = "not_comparable"


class DeltaClassification(str, Enum):
    IMPROVED = "improved"
    REGRESSED = "regressed"
    UNCHANGED = "unchanged"
    NOT_SCORABLE = "not_scorable"


class DeltaNullReason(str, Enum):
    NOT_SCORABLE_OR_INCOMPLETE_COVERAGE = "not_scorable_or_incomplete_coverage"


class DeltaIntervalStatus(str, Enum):
    AVAILABLE = "available"
    INSUFFICIENT_CASE_POPULATION = "insufficient_case_population"
    NOT_SCORABLE = "not_scorable"
    INCOMPLETE_REPLICATES = "incomplete_replicates"


def _error(message: str) -> ComparisonError:
    return ComparisonError(message)


def _exact(value: Any, fields: Iterable[str], context: str) -> Dict[str, Any]:
    expected = set(fields)
    if type(value) is not dict or set(value) != expected or len(value) != len(expected):
        raise _error(f"{context} has unknown or missing fields")
    return value


def _array(value: Any, context: str, maximum: int) -> list[Any]:
    if type(value) is not list or len(value) > maximum:
        raise _error(f"{context} must be a bounded list")
    return value


def _identifier(value: Any, context: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 512
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise _error(f"{context} must be a bounded non-empty identifier")
    return value


def _digest(value: Any, context: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _error(f"{context} must be a lowercase SHA-256 digest")
    return value


def _integer(
    value: Any,
    context: str,
    *,
    minimum: int = 0,
    maximum: Optional[int] = None,
) -> int:
    if type(value) is not int or value < minimum or (
        maximum is not None and value > maximum
    ):
        raise _error(f"{context} is outside its integer bounds")
    return value


def _signed_integer(value: Any, context: str) -> int:
    if type(value) is not int:
        raise _error(f"{context} must be an integer")
    return value


def _optional_integer(value: Any, context: str) -> Optional[int]:
    return None if value is None else _integer(value, context)


def _optional_signed(value: Any, context: str) -> Optional[int]:
    return None if value is None else _signed_integer(value, context)


def _enum(enum_type: type[Enum], value: Any, context: str) -> Any:
    if type(value) is not str:
        raise _error(f"{context} must be an enum string")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise _error(f"{context} has an unknown value") from exc


def _json_value(value: Any, context: str) -> Any:
    try:
        data = canonical_json_bytes(value)
        if len(data) > 16 * 1024 * 1024:
            raise _error(f"{context} exceeds its byte bound")
        return json.loads(data.decode("utf-8"))
    except ComparisonError:
        raise
    except (TypeError, ValueError, UnicodeError) as exc:
        raise _error(f"{context} is not canonical safe JSON") from exc


def _from_json(model_type: Any, data: Any, context: str) -> Any:
    return model_type.from_dict(_strict_json_loads(data, MAX_COMPARISON_BYTES, context))


@dataclass(frozen=True)
class VerifiedRunEvaluation:
    """A hydrated Evaluation and the exact immutable roots that verify it."""

    bundle: Any = field(repr=False)
    run_config: EvalRunConfig
    case_snapshot: RunCaseSnapshot
    source_binding: AnalysisSourceBinding

    def __post_init__(self) -> None:
        if type(self.run_config) is not EvalRunConfig:
            raise TypeError("run_config must be an EvalRunConfig")
        if type(self.case_snapshot) is not RunCaseSnapshot:
            raise TypeError("case_snapshot must be a RunCaseSnapshot")
        if type(self.source_binding) is not AnalysisSourceBinding:
            raise TypeError("source_binding must be an AnalysisSourceBinding")
        rebound = bind_analysis_source(
            self.bundle,
            run_config=self.run_config,
            case_snapshot=self.case_snapshot,
        )
        if rebound != self.source_binding:
            raise ArtifactIntegrityError(
                "supplied Analysis source binding differs from exact source replay"
            )

    @classmethod
    def create(
        cls,
        bundle: Any,
        *,
        run_config: EvalRunConfig,
        case_snapshot: RunCaseSnapshot,
    ) -> "VerifiedRunEvaluation":
        binding = bind_analysis_source(
            bundle,
            run_config=run_config,
            case_snapshot=case_snapshot,
        )
        return cls(bundle, run_config, case_snapshot, binding)

    def verify(self) -> AnalysisSourceBinding:
        rebound = bind_analysis_source(
            self.bundle,
            run_config=self.run_config,
            case_snapshot=self.case_snapshot,
        )
        if rebound != self.source_binding:
            raise ArtifactIntegrityError(
                "verified Evaluation changed after its source binding was supplied"
            )
        return rebound

    @property
    def trials(self) -> Tuple[Any, ...]:
        values = getattr(self.bundle, "trials", None)
        if type(values) is not tuple:
            raise ArtifactIntegrityError("verified Evaluation Trials are not canonical")
        return values

    @property
    def run_id(self) -> str:
        return self.source_binding.run_id

    @property
    def evaluation_id(self) -> str:
        return self.source_binding.evaluation_id


@dataclass(frozen=True)
class ComparisonPolicyV1(_JsonModel):
    schema_version: str
    statistics_policy: StatisticsPolicyV1
    required_case_fields: Tuple[str, ...]
    required_evaluator_fields: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != COMPARISON_POLICY_SCHEMA_VERSION:
            raise _error("ComparisonPolicyV1 schema_version is unsupported")
        if type(self.statistics_policy) is not StatisticsPolicyV1:
            raise _error("ComparisonPolicyV1 statistics_policy is invalid")
        canonical_statistics = StatisticsPolicyV1.from_dict(
            self.statistics_policy.to_dict()
        )
        if canonical_statistics != self.statistics_policy:
            raise _error("ComparisonPolicyV1 statistics_policy is not canonical")
        if type(self.required_case_fields) not in (tuple, list) or tuple(
            self.required_case_fields
        ) != REQUIRED_CASE_FIELDS:
            raise _error(
                "ComparisonPolicyV1 required_case_fields must equal the closed hard set"
            )
        if type(self.required_evaluator_fields) not in (tuple, list) or tuple(
            self.required_evaluator_fields
        ) != REQUIRED_EVALUATOR_FIELDS:
            raise _error(
                "ComparisonPolicyV1 required_evaluator_fields must equal the closed hard set"
            )
        object.__setattr__(self, "statistics_policy", canonical_statistics)
        object.__setattr__(self, "required_case_fields", REQUIRED_CASE_FIELDS)
        object.__setattr__(
            self, "required_evaluator_fields", REQUIRED_EVALUATOR_FIELDS
        )

    @classmethod
    def default(cls, statistics_policy: StatisticsPolicyV1) -> "ComparisonPolicyV1":
        return cls(
            COMPARISON_POLICY_SCHEMA_VERSION,
            statistics_policy,
            REQUIRED_CASE_FIELDS,
            REQUIRED_EVALUATOR_FIELDS,
        )

    @property
    def algorithm_version(self) -> str:
        return COMPARISON_ALGORITHM_VERSION

    @property
    def policy_digest(self) -> str:
        return canonical_sha256(self.to_dict())

    @property
    def algorithm_digest(self) -> str:
        return canonical_sha256(
            {
                "algorithm_version": COMPARISON_ALGORITHM_VERSION,
                "policy_digest": self.policy_digest,
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "statistics_policy": self.statistics_policy.to_dict(),
            "required_case_fields": list(self.required_case_fields),
            "required_evaluator_fields": list(self.required_evaluator_fields),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "ComparisonPolicyV1":
        payload = _exact(
            value,
            (
                "schema_version",
                "statistics_policy",
                "required_case_fields",
                "required_evaluator_fields",
            ),
            "ComparisonPolicyV1",
        )
        return cls(
            payload["schema_version"],
            StatisticsPolicyV1.from_dict(payload["statistics_policy"]),
            tuple(
                _array(
                    payload["required_case_fields"],
                    "ComparisonPolicyV1.required_case_fields",
                    len(REQUIRED_CASE_FIELDS),
                )
            ),
            tuple(
                _array(
                    payload["required_evaluator_fields"],
                    "ComparisonPolicyV1.required_evaluator_fields",
                    len(REQUIRED_EVALUATOR_FIELDS),
                )
            ),
        )

    @classmethod
    def from_json(cls, data: Any) -> "ComparisonPolicyV1":
        return _from_json(cls, data, "ComparisonPolicyV1 JSON")


@dataclass(frozen=True, init=False)
class CompatibilityFieldV1(_JsonModel):
    path: str
    _value_json: str = field(repr=False)

    def __init__(self, path: str, value: Any) -> None:
        _identifier(path, "compatibility field path")
        canonical = _json_value(value, f"compatibility field {path}")
        object.__setattr__(self, "path", path)
        object.__setattr__(
            self,
            "_value_json",
            canonical_json_bytes(canonical).decode("utf-8"),
        )

    @property
    def value(self) -> Any:
        return json.loads(self._value_json)

    def to_dict(self) -> Dict[str, Any]:
        return {"path": self.path, "value": self.value}

    @classmethod
    def from_dict(cls, value: Any) -> "CompatibilityFieldV1":
        payload = _exact(value, ("path", "value"), "CompatibilityFieldV1")
        return cls(payload["path"], payload["value"])


@dataclass(frozen=True)
class CompatibilityProjectionV1(_JsonModel):
    fields: Tuple[CompatibilityFieldV1, ...]

    def __post_init__(self) -> None:
        values = tuple(self.fields)
        if (
            any(type(item) is not CompatibilityFieldV1 for item in values)
            or tuple(item.path for item in values) != _PROJECTION_FIELDS
        ):
            raise _error("compatibility projection does not cover the closed hard fields")
        object.__setattr__(self, "fields", values)

    def value(self, path: str) -> Any:
        return next(item.value for item in self.fields if item.path == path)

    @property
    def projection_digest(self) -> str:
        return canonical_sha256(
            {"fields": [item.to_dict() for item in self.fields]}
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "projection_digest": self.projection_digest,
            "fields": [item.to_dict() for item in self.fields],
        }

    @classmethod
    def from_dict(cls, value: Any) -> "CompatibilityProjectionV1":
        payload = _exact(
            value, ("projection_digest", "fields"), "CompatibilityProjectionV1"
        )
        fields = _array(
            payload["fields"], "CompatibilityProjectionV1.fields", len(_PROJECTION_FIELDS)
        )
        result = cls(tuple(CompatibilityFieldV1.from_dict(item) for item in fields))
        if payload["projection_digest"] != result.projection_digest:
            raise _error("compatibility projection digest is not canonical")
        return result


@dataclass(frozen=True)
class AgentProvenanceV1(_JsonModel):
    fields: Tuple[CompatibilityFieldV1, ...]

    def __post_init__(self) -> None:
        values = tuple(self.fields)
        if (
            any(type(item) is not CompatibilityFieldV1 for item in values)
            or tuple(item.path for item in values) != _AGENT_FIELDS
        ):
            raise _error("Agent provenance does not cover its canonical fields")
        object.__setattr__(self, "fields", values)

    def value(self, path: str) -> Any:
        return next(item.value for item in self.fields if item.path == path)

    @property
    def provenance_digest(self) -> str:
        return canonical_sha256({"fields": [item.to_dict() for item in self.fields]})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provenance_digest": self.provenance_digest,
            "fields": [item.to_dict() for item in self.fields],
        }

    @classmethod
    def from_dict(cls, value: Any) -> "AgentProvenanceV1":
        payload = _exact(value, ("provenance_digest", "fields"), "AgentProvenanceV1")
        fields = _array(payload["fields"], "AgentProvenanceV1.fields", len(_AGENT_FIELDS))
        result = cls(tuple(CompatibilityFieldV1.from_dict(item) for item in fields))
        if payload["provenance_digest"] != result.provenance_digest:
            raise _error("Agent provenance digest is not canonical")
        return result


@dataclass(frozen=True, init=False)
class AgentFieldDeltaV1(_JsonModel):
    field: str
    _baseline_json: str = field(repr=False)
    _candidate_json: str = field(repr=False)

    def __init__(self, field: str, baseline: Any, candidate: Any) -> None:
        if field not in _AGENT_FIELDS:
            raise _error("Agent delta field is outside the closed provenance set")
        baseline_value = _json_value(baseline, "Agent delta baseline")
        candidate_value = _json_value(candidate, "Agent delta candidate")
        if baseline_value == candidate_value:
            raise _error("Agent delta may only contain changed fields")
        object.__setattr__(self, "field", field)
        object.__setattr__(self, "_baseline_json", canonical_json_bytes(baseline_value).decode())
        object.__setattr__(self, "_candidate_json", canonical_json_bytes(candidate_value).decode())

    @property
    def baseline(self) -> Any:
        return json.loads(self._baseline_json)

    @property
    def candidate(self) -> Any:
        return json.loads(self._candidate_json)

    def to_dict(self) -> Dict[str, Any]:
        return {"field": self.field, "baseline": self.baseline, "candidate": self.candidate}

    @classmethod
    def from_dict(cls, value: Any) -> "AgentFieldDeltaV1":
        payload = _exact(value, ("field", "baseline", "candidate"), "AgentFieldDeltaV1")
        return cls(payload["field"], payload["baseline"], payload["candidate"])


def _agent_changes(
    baseline: AgentProvenanceV1, candidate: AgentProvenanceV1
) -> Tuple[AgentFieldDeltaV1, ...]:
    return tuple(
        AgentFieldDeltaV1(path, baseline.value(path), candidate.value(path))
        for path in _AGENT_FIELDS
        if baseline.value(path) != candidate.value(path)
    )


@dataclass(frozen=True)
class AgentDeltaV1(_JsonModel):
    baseline: AgentProvenanceV1
    candidate: AgentProvenanceV1
    changes: Tuple[AgentFieldDeltaV1, ...]

    def __post_init__(self) -> None:
        if type(self.baseline) is not AgentProvenanceV1 or type(self.candidate) is not AgentProvenanceV1:
            raise _error("AgentDeltaV1 provenance is invalid")
        expected = _agent_changes(self.baseline, self.candidate)
        if tuple(self.changes) != expected:
            raise _error("AgentDeltaV1 changes differ from recomputed provenance delta")
        object.__setattr__(self, "changes", expected)

    @property
    def delta_id(self) -> str:
        return stable_id("agent-delta-v1", self._identity_dict())

    def _identity_dict(self) -> Dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "changes": [item.to_dict() for item in self.changes],
        }

    def to_dict(self) -> Dict[str, Any]:
        return {**self._identity_dict(), "delta_id": self.delta_id}

    @classmethod
    def from_dict(cls, value: Any) -> "AgentDeltaV1":
        payload = _exact(value, ("delta_id", "baseline", "candidate", "changes"), "AgentDeltaV1")
        changes = _array(payload["changes"], "AgentDeltaV1.changes", len(_AGENT_FIELDS))
        result = cls(
            AgentProvenanceV1.from_dict(payload["baseline"]),
            AgentProvenanceV1.from_dict(payload["candidate"]),
            tuple(AgentFieldDeltaV1.from_dict(item) for item in changes),
        )
        if payload["delta_id"] != result.delta_id:
            raise _error("AgentDeltaV1 delta_id is not canonical")
        return result


@dataclass(frozen=True)
class ComparisonCompatibilityV1(_JsonModel):
    schema_version: str
    policy_digest: str
    baseline_projection: CompatibilityProjectionV1
    candidate_projection: CompatibilityProjectionV1
    shared_projection: Optional[CompatibilityProjectionV1]
    agent_delta: AgentDeltaV1

    def __post_init__(self) -> None:
        if self.schema_version != COMPARISON_COMPATIBILITY_SCHEMA_VERSION:
            raise _error("ComparisonCompatibilityV1 schema_version is unsupported")
        _digest(self.policy_digest, "comparison compatibility policy_digest")
        if type(self.baseline_projection) is not CompatibilityProjectionV1 or type(self.candidate_projection) is not CompatibilityProjectionV1:
            raise _error("comparison compatibility projections are invalid")
        expected_shared = (
            self.baseline_projection
            if self.baseline_projection == self.candidate_projection
            else None
        )
        if self.shared_projection != expected_shared:
            raise _error("shared compatibility projection differs from both source projections")
        if type(self.agent_delta) is not AgentDeltaV1:
            raise _error("comparison Agent delta is invalid")

    @property
    def compatibility_id(self) -> str:
        return stable_id("comparison-compatibility-v1", self._identity_dict())

    def _identity_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_digest": self.policy_digest,
            "baseline_projection": self.baseline_projection.to_dict(),
            "candidate_projection": self.candidate_projection.to_dict(),
            "shared_projection": None if self.shared_projection is None else self.shared_projection.to_dict(),
            "agent_delta": self.agent_delta.to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {**self._identity_dict(), "compatibility_id": self.compatibility_id}

    @classmethod
    def from_dict(cls, value: Any) -> "ComparisonCompatibilityV1":
        payload = _exact(
            value,
            (
                "compatibility_id",
                "schema_version",
                "policy_digest",
                "baseline_projection",
                "candidate_projection",
                "shared_projection",
                "agent_delta",
            ),
            "ComparisonCompatibilityV1",
        )
        result = cls(
            payload["schema_version"],
            payload["policy_digest"],
            CompatibilityProjectionV1.from_dict(payload["baseline_projection"]),
            CompatibilityProjectionV1.from_dict(payload["candidate_projection"]),
            None if payload["shared_projection"] is None else CompatibilityProjectionV1.from_dict(payload["shared_projection"]),
            AgentDeltaV1.from_dict(payload["agent_delta"]),
        )
        if payload["compatibility_id"] != result.compatibility_id:
            raise _error("ComparisonCompatibilityV1 compatibility_id is not canonical")
        return result


@dataclass(frozen=True)
class MetricValueV1(_JsonModel):
    status: StatisticsMetricStatus
    numerator: Optional[int]
    denominator: Optional[int]
    value: Optional[int]
    coverage: StatisticsCoverageV1

    def __post_init__(self) -> None:
        if type(self.status) is not StatisticsMetricStatus:
            raise _error("MetricValueV1 status is invalid")
        if self.status is StatisticsMetricStatus.AVAILABLE:
            _integer(self.numerator, "MetricValueV1 numerator")
            _integer(self.denominator, "MetricValueV1 denominator")
            _integer(self.value, "MetricValueV1 value")
        elif self.status is StatisticsMetricStatus.ZERO_DENOMINATOR:
            if self.numerator is not None:
                _integer(self.numerator, "MetricValueV1 zero numerator")
            if self.denominator is not None:
                _integer(self.denominator, "MetricValueV1 zero denominator")
            if self.value is not None:
                raise _error("zero-denominator MetricValueV1 must have null value")
        elif self.numerator is not None or self.denominator is not None or self.value is not None:
            raise _error("unavailable MetricValueV1 must have null numeric fields")
        if type(self.coverage) is not StatisticsCoverageV1:
            raise _error("MetricValueV1 coverage is invalid")

    @classmethod
    def from_source(cls, value: Any) -> "MetricValueV1":
        return cls(value.status, value.numerator, value.denominator, value.value, value.coverage)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value": self.value,
            "coverage": self.coverage.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "MetricValueV1":
        payload = _exact(value, ("status", "numerator", "denominator", "value", "coverage"), "MetricValueV1")
        return cls(
            _enum(StatisticsMetricStatus, payload["status"], "MetricValueV1.status"),
            _optional_integer(payload["numerator"], "MetricValueV1.numerator"),
            _optional_integer(payload["denominator"], "MetricValueV1.denominator"),
            _optional_integer(payload["value"], "MetricValueV1.value"),
            StatisticsCoverageV1.from_dict(payload["coverage"]),
        )


def _delta_fields(
    baseline: MetricValueV1,
    candidate: MetricValueV1,
    direction: MetricDirection,
) -> tuple[Optional[int], DeltaClassification, Optional[DeltaNullReason]]:
    if (
        baseline.status is not StatisticsMetricStatus.AVAILABLE
        or candidate.status is not StatisticsMetricStatus.AVAILABLE
        or type(baseline.value) is not int
        or type(candidate.value) is not int
    ):
        return (
            None,
            DeltaClassification.NOT_SCORABLE,
            DeltaNullReason.NOT_SCORABLE_OR_INCOMPLETE_COVERAGE,
        )
    delta = candidate.value - baseline.value
    if delta == 0:
        classification = DeltaClassification.UNCHANGED
    elif (
        direction is MetricDirection.HIGHER_IS_BETTER and delta > 0
    ) or (
        direction is MetricDirection.LOWER_IS_BETTER and delta < 0
    ):
        classification = DeltaClassification.IMPROVED
    else:
        classification = DeltaClassification.REGRESSED
    return delta, classification, None


@dataclass(frozen=True)
class ContributionDeltaV1(_JsonModel):
    metric: CoreMetric
    unit: MetricUnit
    direction: MetricDirection
    baseline: MetricValueV1
    candidate: MetricValueV1
    absolute_delta: Optional[int]
    classification: DeltaClassification
    null_reason: Optional[DeltaNullReason]

    def __post_init__(self) -> None:
        if type(self.metric) is not CoreMetric or type(self.unit) is not MetricUnit or type(self.direction) is not MetricDirection:
            raise _error("ContributionDeltaV1 metric metadata is invalid")
        if type(self.baseline) is not MetricValueV1 or type(self.candidate) is not MetricValueV1:
            raise _error("ContributionDeltaV1 values are invalid")
        expected = _delta_fields(self.baseline, self.candidate, self.direction)
        if (self.absolute_delta, self.classification, self.null_reason) != expected:
            raise _error("ContributionDeltaV1 differs from recomputed delta")

    @classmethod
    def create(cls, baseline: Any, candidate: Any) -> "ContributionDeltaV1":
        if (
            baseline.metric is not candidate.metric
            or baseline.unit is not candidate.unit
            or baseline.direction is not candidate.direction
        ):
            raise _error("paired contributions use incompatible metric metadata")
        b = MetricValueV1.from_source(baseline)
        c = MetricValueV1.from_source(candidate)
        delta, classification, reason = _delta_fields(b, c, baseline.direction)
        return cls(baseline.metric, baseline.unit, baseline.direction, b, c, delta, classification, reason)

    @property
    def delta_id(self) -> str:
        return stable_id("contribution-delta-v1", self._identity_dict())

    def _identity_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric.value,
            "unit": self.unit.value,
            "direction": self.direction.value,
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "absolute_delta": self.absolute_delta,
            "classification": self.classification.value,
            "null_reason": None if self.null_reason is None else self.null_reason.value,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {**self._identity_dict(), "delta_id": self.delta_id}

    @classmethod
    def from_dict(cls, value: Any) -> "ContributionDeltaV1":
        payload = _exact(value, ("delta_id", "metric", "unit", "direction", "baseline", "candidate", "absolute_delta", "classification", "null_reason"), "ContributionDeltaV1")
        result = cls(
            _enum(CoreMetric, payload["metric"], "contribution metric"),
            _enum(MetricUnit, payload["unit"], "contribution unit"),
            _enum(MetricDirection, payload["direction"], "contribution direction"),
            MetricValueV1.from_dict(payload["baseline"]),
            MetricValueV1.from_dict(payload["candidate"]),
            _optional_signed(payload["absolute_delta"], "contribution delta"),
            _enum(DeltaClassification, payload["classification"], "contribution classification"),
            None if payload["null_reason"] is None else _enum(DeltaNullReason, payload["null_reason"], "contribution null_reason"),
        )
        if payload["delta_id"] != result.delta_id:
            raise _error("ContributionDeltaV1 delta_id is not canonical")
        return result


@dataclass(frozen=True)
class PairedBootstrapCoverageV1(_JsonModel):
    total_case_count: int
    scorable_pair_count: int
    not_scorable_pair_count: int
    incomplete_replicate_count: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _integer(getattr(self, name), f"PairedBootstrapCoverageV1.{name}")
        if self.scorable_pair_count + self.not_scorable_pair_count != self.total_case_count:
            raise _error("paired bootstrap coverage does not cover all Cases")

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Any) -> "PairedBootstrapCoverageV1":
        return cls(**_exact(value, tuple(cls.__dataclass_fields__), "PairedBootstrapCoverageV1"))


@dataclass(frozen=True)
class PairedDeltaConfidenceIntervalV1(_JsonModel):
    metric: CoreMetric
    unit: MetricUnit
    status: DeltaIntervalStatus
    lower_bound: Optional[int]
    upper_bound: Optional[int]
    seed: int
    iterations: int
    confidence_level_ppm: int
    coverage: PairedBootstrapCoverageV1

    def __post_init__(self) -> None:
        if type(self.metric) is not CoreMetric or type(self.unit) is not MetricUnit or type(self.status) is not DeltaIntervalStatus:
            raise _error("paired delta interval metadata is invalid")
        _integer(self.seed, "paired interval seed", maximum=MAX_BOOTSTRAP_SEED)
        _integer(self.iterations, "paired interval iterations", minimum=1, maximum=MAX_BOOTSTRAP_ITERATIONS)
        _integer(self.confidence_level_ppm, "paired interval confidence", minimum=1, maximum=PPM_SCALE - 1)
        if type(self.coverage) is not PairedBootstrapCoverageV1:
            raise _error("paired delta interval coverage is invalid")
        if self.coverage.incomplete_replicate_count > self.iterations:
            raise _error("paired interval incomplete count exceeds iterations")
        if self.status is DeltaIntervalStatus.AVAILABLE:
            lower = _signed_integer(self.lower_bound, "paired interval lower")
            upper = _signed_integer(self.upper_bound, "paired interval upper")
            if lower > upper or self.coverage.incomplete_replicate_count:
                raise _error("available paired interval is inconsistent")
        elif self.lower_bound is not None or self.upper_bound is not None:
            raise _error("unavailable paired interval must use null bounds")

    @property
    def interval_id(self) -> str:
        return stable_id("paired-delta-interval-v1", self._identity_dict())

    def _identity_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric.value,
            "unit": self.unit.value,
            "status": self.status.value,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "seed": self.seed,
            "iterations": self.iterations,
            "confidence_level_ppm": self.confidence_level_ppm,
            "coverage": self.coverage.to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {**self._identity_dict(), "interval_id": self.interval_id}

    @classmethod
    def from_dict(cls, value: Any) -> "PairedDeltaConfidenceIntervalV1":
        payload = _exact(value, ("interval_id", "metric", "unit", "status", "lower_bound", "upper_bound", "seed", "iterations", "confidence_level_ppm", "coverage"), "PairedDeltaConfidenceIntervalV1")
        result = cls(
            _enum(CoreMetric, payload["metric"], "paired interval metric"),
            _enum(MetricUnit, payload["unit"], "paired interval unit"),
            _enum(DeltaIntervalStatus, payload["status"], "paired interval status"),
            _optional_signed(payload["lower_bound"], "paired interval lower"),
            _optional_signed(payload["upper_bound"], "paired interval upper"),
            payload["seed"], payload["iterations"], payload["confidence_level_ppm"],
            PairedBootstrapCoverageV1.from_dict(payload["coverage"]),
        )
        if payload["interval_id"] != result.interval_id:
            raise _error("paired delta interval ID is not canonical")
        return result


def _paired_delta_interval(
    baseline: Sequence[CaseContributionV1],
    candidate: Sequence[CaseContributionV1],
    policy: StatisticsPolicyV1,
) -> PairedDeltaConfidenceIntervalV1:
    baseline_values = tuple(baseline)
    candidate_values = tuple(candidate)
    if not baseline_values or len(baseline_values) > MAX_BOOTSTRAP_CASES or len(baseline_values) != len(candidate_values):
        raise _error("paired bootstrap populations are empty, oversized, or unequal")
    baseline_by_key = {
        (item.task_id, item.case_version, item.canonical_case_digest): item
        for item in baseline_values
    }
    candidate_by_key = {
        (item.task_id, item.case_version, item.canonical_case_digest): item
        for item in candidate_values
    }
    if (
        len(baseline_by_key) != len(baseline_values)
        or len(candidate_by_key) != len(candidate_values)
        or set(baseline_by_key) != set(candidate_by_key)
    ):
        raise _error("paired bootstrap populations do not share exact Case keys")
    ordered_keys = tuple(sorted(baseline_by_key))
    baseline_values = tuple(baseline_by_key[key] for key in ordered_keys)
    candidate_values = tuple(candidate_by_key[key] for key in ordered_keys)
    paired_values = tuple(
        (baseline_by_key[key], candidate_by_key[key])
        for key in ordered_keys
    )
    first = baseline_values[0]
    if any(
        b.metric is not first.metric
        or c.metric is not first.metric
        or b.kind is not first.kind
        or c.kind is not first.kind
        or b.unit is not first.unit
        or c.unit is not first.unit
        for b, c in paired_values
    ):
        raise _error("paired bootstrap populations use incompatible metrics")
    scorable = tuple(
        b.status is StatisticsMetricStatus.AVAILABLE
        and c.status is StatisticsMetricStatus.AVAILABLE
        and type(b.value) is int
        and type(c.value) is int
        for b, c in paired_values
    )

    def make(
        status: DeltaIntervalStatus,
        lower: Optional[int] = None,
        upper: Optional[int] = None,
        incomplete: int = 0,
    ) -> PairedDeltaConfidenceIntervalV1:
        return PairedDeltaConfidenceIntervalV1(
            first.metric,
            first.unit,
            status,
            lower,
            upper,
            policy.bootstrap_seed,
            policy.bootstrap_iterations,
            policy.confidence_level_ppm,
            PairedBootstrapCoverageV1(
                len(scorable), sum(scorable), len(scorable) - sum(scorable), incomplete
            ),
        )

    if len(baseline_values) < 2:
        return make(DeltaIntervalStatus.INSUFFICIENT_CASE_POPULATION)
    if not all(scorable):
        return make(DeltaIntervalStatus.NOT_SCORABLE)
    if len(baseline_values) * policy.bootstrap_iterations > 50_000_000:
        raise _error("paired bootstrap draw budget is exceeded")
    generator = random.Random(policy.bootstrap_seed)
    replicates: list[int] = []
    incomplete = 0
    for _iteration in range(policy.bootstrap_iterations):
        indices = tuple(generator.randrange(len(baseline_values)) for _ in baseline_values)
        baseline_sample = tuple(baseline_values[index] for index in indices)
        candidate_sample = tuple(candidate_values[index] for index in indices)
        baseline_value = _sample_metric_value(first.metric, first.kind, baseline_sample)
        candidate_value = _sample_metric_value(first.metric, first.kind, candidate_sample)
        if baseline_value is None or candidate_value is None:
            incomplete += 1
        else:
            replicates.append(candidate_value - baseline_value)
    if incomplete:
        return make(DeltaIntervalStatus.INCOMPLETE_REPLICATES, incomplete=incomplete)
    replicates.sort()
    tail_ppm = (PPM_SCALE - policy.confidence_level_ppm) // 2
    lower_index = tail_ppm * (policy.bootstrap_iterations - 1) // PPM_SCALE
    upper_numerator = (PPM_SCALE - tail_ppm) * (policy.bootstrap_iterations - 1)
    upper_index = (upper_numerator + PPM_SCALE - 1) // PPM_SCALE
    return make(DeltaIntervalStatus.AVAILABLE, replicates[lower_index], replicates[upper_index])


@dataclass(frozen=True)
class MetricDeltaV1(_JsonModel):
    metric: CoreMetric
    unit: MetricUnit
    direction: MetricDirection
    baseline: MetricValueV1
    candidate: MetricValueV1
    absolute_delta: Optional[int]
    classification: DeltaClassification
    null_reason: Optional[DeltaNullReason]
    confidence_interval: PairedDeltaConfidenceIntervalV1

    def __post_init__(self) -> None:
        if type(self.metric) is not CoreMetric or type(self.unit) is not MetricUnit or type(self.direction) is not MetricDirection:
            raise _error("MetricDeltaV1 metric metadata is invalid")
        if type(self.baseline) is not MetricValueV1 or type(self.candidate) is not MetricValueV1:
            raise _error("MetricDeltaV1 values are invalid")
        expected = _delta_fields(self.baseline, self.candidate, self.direction)
        if (self.absolute_delta, self.classification, self.null_reason) != expected:
            raise _error("MetricDeltaV1 differs from recomputed delta")
        if type(self.confidence_interval) is not PairedDeltaConfidenceIntervalV1 or self.confidence_interval.metric is not self.metric or self.confidence_interval.unit is not self.unit:
            raise _error("MetricDeltaV1 paired confidence interval is incompatible")

    @property
    def delta_id(self) -> str:
        return stable_id("metric-delta-v1", self._identity_dict())

    def _identity_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric.value,
            "unit": self.unit.value,
            "direction": self.direction.value,
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "absolute_delta": self.absolute_delta,
            "classification": self.classification.value,
            "null_reason": None if self.null_reason is None else self.null_reason.value,
            "confidence_interval": self.confidence_interval.to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {**self._identity_dict(), "delta_id": self.delta_id}

    @classmethod
    def from_dict(cls, value: Any) -> "MetricDeltaV1":
        payload = _exact(value, ("delta_id", "metric", "unit", "direction", "baseline", "candidate", "absolute_delta", "classification", "null_reason", "confidence_interval"), "MetricDeltaV1")
        result = cls(
            _enum(CoreMetric, payload["metric"], "metric delta metric"),
            _enum(MetricUnit, payload["unit"], "metric delta unit"),
            _enum(MetricDirection, payload["direction"], "metric delta direction"),
            MetricValueV1.from_dict(payload["baseline"]),
            MetricValueV1.from_dict(payload["candidate"]),
            _optional_signed(payload["absolute_delta"], "metric absolute_delta"),
            _enum(DeltaClassification, payload["classification"], "metric classification"),
            None if payload["null_reason"] is None else _enum(DeltaNullReason, payload["null_reason"], "metric null_reason"),
            PairedDeltaConfidenceIntervalV1.from_dict(payload["confidence_interval"]),
        )
        if payload["delta_id"] != result.delta_id:
            raise _error("MetricDeltaV1 delta_id is not canonical")
        return result


@dataclass(frozen=True)
class JudgeCoverageV1(_JsonModel):
    request_count: int
    graded_count: int
    failure_count: int
    ungraded_count: int
    semantic_unknown_count: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _integer(getattr(self, name), f"JudgeCoverageV1.{name}")
        if self.graded_count + self.failure_count + self.ungraded_count != self.request_count or self.semantic_unknown_count > self.graded_count:
            raise _error("JudgeCoverageV1 counts are inconsistent")

    @classmethod
    def from_statistics(cls, coverage: StatisticsCoverageV1) -> "JudgeCoverageV1":
        return cls(
            coverage.judge_request_count,
            coverage.judge_graded_count,
            coverage.judge_failure_count,
            coverage.judge_ungraded_count,
            coverage.judge_semantic_unknown_count,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Any) -> "JudgeCoverageV1":
        return cls(**_exact(value, tuple(cls.__dataclass_fields__), "JudgeCoverageV1"))


@dataclass(frozen=True)
class JudgeCoverageDeltaV1(_JsonModel):
    baseline: JudgeCoverageV1
    candidate: JudgeCoverageV1
    request_delta: int
    graded_delta: int
    failure_delta: int
    ungraded_delta: int
    semantic_unknown_delta: int

    def __post_init__(self) -> None:
        if type(self.baseline) is not JudgeCoverageV1 or type(self.candidate) is not JudgeCoverageV1:
            raise _error("Judge coverage delta values are invalid")
        expected = tuple(
            getattr(self.candidate, name) - getattr(self.baseline, name)
            for name in ("request_count", "graded_count", "failure_count", "ungraded_count", "semantic_unknown_count")
        )
        if (self.request_delta, self.graded_delta, self.failure_delta, self.ungraded_delta, self.semantic_unknown_delta) != expected:
            raise _error("Judge coverage delta differs from recomputed counts")

    @classmethod
    def create(cls, baseline: StatisticsCoverageV1, candidate: StatisticsCoverageV1) -> "JudgeCoverageDeltaV1":
        b = JudgeCoverageV1.from_statistics(baseline)
        c = JudgeCoverageV1.from_statistics(candidate)
        return cls(b, c, c.request_count-b.request_count, c.graded_count-b.graded_count, c.failure_count-b.failure_count, c.ungraded_count-b.ungraded_count, c.semantic_unknown_count-b.semantic_unknown_count)

    @property
    def baseline_ungraded_count(self) -> int:
        return self.baseline.ungraded_count

    @property
    def candidate_ungraded_count(self) -> int:
        return self.candidate.ungraded_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict(), "candidate": self.candidate.to_dict(),
            "request_delta": self.request_delta, "graded_delta": self.graded_delta,
            "failure_delta": self.failure_delta, "ungraded_delta": self.ungraded_delta,
            "semantic_unknown_delta": self.semantic_unknown_delta,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "JudgeCoverageDeltaV1":
        payload = _exact(value, ("baseline", "candidate", "request_delta", "graded_delta", "failure_delta", "ungraded_delta", "semantic_unknown_delta"), "JudgeCoverageDeltaV1")
        return cls(
            JudgeCoverageV1.from_dict(payload["baseline"]), JudgeCoverageV1.from_dict(payload["candidate"]),
            _signed_integer(payload["request_delta"], "Judge request delta"),
            _signed_integer(payload["graded_delta"], "Judge graded delta"),
            _signed_integer(payload["failure_delta"], "Judge failure delta"),
            _signed_integer(payload["ungraded_delta"], "Judge ungraded delta"),
            _signed_integer(payload["semantic_unknown_delta"], "Judge unknown delta"),
        )


@dataclass(frozen=True)
class TrialReferenceV1(_JsonModel):
    run_id: str
    evaluation_id: str
    task_id: str
    case_version: int
    canonical_case_digest: str
    trial_index: int
    trial_id: str
    submission_status: SubmissionStatus
    judge_coverage: JudgeCoverageV1

    def __post_init__(self) -> None:
        _identifier(self.run_id, "TrialReferenceV1.run_id")
        _identifier(self.evaluation_id, "TrialReferenceV1.evaluation_id")
        _identifier(self.task_id, "TrialReferenceV1.task_id")
        _integer(self.case_version, "TrialReferenceV1.case_version", minimum=1)
        _digest(self.canonical_case_digest, "TrialReferenceV1.case digest")
        _integer(self.trial_index, "TrialReferenceV1.trial_index", minimum=1, maximum=MAX_TRIAL_COUNT)
        if self.trial_id != derive_trial_id(self.run_id, self.task_id, self.trial_index):
            raise _error("TrialReferenceV1 trial_id is not canonical")
        if type(self.submission_status) is not SubmissionStatus or type(self.judge_coverage) is not JudgeCoverageV1:
            raise _error("TrialReferenceV1 outcome metadata is invalid")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id, "evaluation_id": self.evaluation_id,
            "task_id": self.task_id, "case_version": self.case_version,
            "canonical_case_digest": self.canonical_case_digest,
            "trial_index": self.trial_index, "trial_id": self.trial_id,
            "submission_status": self.submission_status.value,
            "judge_coverage": self.judge_coverage.to_dict(),
        }

    @property
    def judge_ungraded_count(self) -> int:
        return self.judge_coverage.ungraded_count

    @classmethod
    def from_dict(cls, value: Any) -> "TrialReferenceV1":
        payload = _exact(value, ("run_id", "evaluation_id", "task_id", "case_version", "canonical_case_digest", "trial_index", "trial_id", "submission_status", "judge_coverage"), "TrialReferenceV1")
        return cls(
            payload["run_id"], payload["evaluation_id"], payload["task_id"], payload["case_version"],
            payload["canonical_case_digest"], payload["trial_index"], payload["trial_id"],
            _enum(SubmissionStatus, payload["submission_status"], "TrialReferenceV1.submission_status"),
            JudgeCoverageV1.from_dict(payload["judge_coverage"]),
        )


@dataclass(frozen=True)
class PairedTrialDeltaV1(_JsonModel):
    task_id: str
    case_version: int
    canonical_case_digest: str
    trial_index: int
    baseline: TrialReferenceV1
    candidate: TrialReferenceV1
    metric_deltas: Tuple[ContributionDeltaV1, ...]
    judge_coverage_delta: JudgeCoverageDeltaV1

    def __post_init__(self) -> None:
        key = (self.task_id, self.case_version, self.canonical_case_digest, self.trial_index)
        for ref in (self.baseline, self.candidate):
            if type(ref) is not TrialReferenceV1 or (ref.task_id, ref.case_version, ref.canonical_case_digest, ref.trial_index) != key:
                raise _error("paired Trial reference differs from the exact pairing key")
        deltas = tuple(self.metric_deltas)
        if tuple(item.metric for item in deltas) != tuple(sorted(CoreMetric, key=lambda item: item.value)):
            raise _error("paired Trial metric deltas are incomplete or noncanonical")
        expected_judge = JudgeCoverageDeltaV1(
            self.baseline.judge_coverage, self.candidate.judge_coverage,
            self.candidate.judge_coverage.request_count-self.baseline.judge_coverage.request_count,
            self.candidate.judge_coverage.graded_count-self.baseline.judge_coverage.graded_count,
            self.candidate.judge_coverage.failure_count-self.baseline.judge_coverage.failure_count,
            self.candidate.judge_coverage.ungraded_count-self.baseline.judge_coverage.ungraded_count,
            self.candidate.judge_coverage.semantic_unknown_count-self.baseline.judge_coverage.semantic_unknown_count,
        )
        if self.judge_coverage_delta != expected_judge:
            raise _error("paired Trial Judge coverage differs from its references")
        object.__setattr__(self, "metric_deltas", deltas)

    def metric_delta(self, metric: CoreMetric) -> ContributionDeltaV1:
        return next(item for item in self.metric_deltas if item.metric is metric)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id, "case_version": self.case_version,
            "canonical_case_digest": self.canonical_case_digest, "trial_index": self.trial_index,
            "baseline": self.baseline.to_dict(), "candidate": self.candidate.to_dict(),
            "metric_deltas": [item.to_dict() for item in self.metric_deltas],
            "judge_coverage_delta": self.judge_coverage_delta.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "PairedTrialDeltaV1":
        payload = _exact(value, ("task_id", "case_version", "canonical_case_digest", "trial_index", "baseline", "candidate", "metric_deltas", "judge_coverage_delta"), "PairedTrialDeltaV1")
        values = _array(payload["metric_deltas"], "PairedTrialDeltaV1.metric_deltas", len(CoreMetric))
        return cls(
            payload["task_id"], payload["case_version"], payload["canonical_case_digest"], payload["trial_index"],
            TrialReferenceV1.from_dict(payload["baseline"]), TrialReferenceV1.from_dict(payload["candidate"]),
            tuple(ContributionDeltaV1.from_dict(item) for item in values),
            JudgeCoverageDeltaV1.from_dict(payload["judge_coverage_delta"]),
        )


@dataclass(frozen=True)
class CaseDeltaV1(_JsonModel):
    task_id: str
    case_version: int
    canonical_case_digest: str
    metric_deltas: Tuple[ContributionDeltaV1, ...]
    paired_trials: Tuple[PairedTrialDeltaV1, ...]
    judge_coverage_delta: JudgeCoverageDeltaV1

    def __post_init__(self) -> None:
        _identifier(self.task_id, "CaseDeltaV1.task_id")
        _integer(self.case_version, "CaseDeltaV1.case_version", minimum=1)
        _digest(self.canonical_case_digest, "CaseDeltaV1.case digest")
        deltas = tuple(self.metric_deltas)
        if tuple(item.metric for item in deltas) != tuple(sorted(CoreMetric, key=lambda item: item.value)):
            raise _error("Case metric deltas are incomplete or noncanonical")
        trials = tuple(self.paired_trials)
        if not trials or tuple(item.trial_index for item in trials) != tuple(range(1, len(trials)+1)) or any((item.task_id, item.case_version, item.canonical_case_digest) != (self.task_id, self.case_version, self.canonical_case_digest) for item in trials):
            raise _error("Case paired Trials are incomplete or noncanonical")
        if type(self.judge_coverage_delta) is not JudgeCoverageDeltaV1:
            raise _error("Case Judge coverage delta is invalid")
        object.__setattr__(self, "metric_deltas", deltas)
        object.__setattr__(self, "paired_trials", trials)

    def metric_delta(self, metric: CoreMetric) -> ContributionDeltaV1:
        return next(item for item in self.metric_deltas if item.metric is metric)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id, "case_version": self.case_version,
            "canonical_case_digest": self.canonical_case_digest,
            "metric_deltas": [item.to_dict() for item in self.metric_deltas],
            "paired_trials": [item.to_dict() for item in self.paired_trials],
            "judge_coverage_delta": self.judge_coverage_delta.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "CaseDeltaV1":
        payload = _exact(value, ("task_id", "case_version", "canonical_case_digest", "metric_deltas", "paired_trials", "judge_coverage_delta"), "CaseDeltaV1")
        metrics = _array(payload["metric_deltas"], "CaseDeltaV1.metric_deltas", len(CoreMetric))
        trials = _array(payload["paired_trials"], "CaseDeltaV1.paired_trials", MAX_TRIAL_COUNT)
        return cls(
            payload["task_id"], payload["case_version"], payload["canonical_case_digest"],
            tuple(ContributionDeltaV1.from_dict(item) for item in metrics),
            tuple(PairedTrialDeltaV1.from_dict(item) for item in trials),
            JudgeCoverageDeltaV1.from_dict(payload["judge_coverage_delta"]),
        )


def _projection_incompatibilities(
    baseline: CompatibilityProjectionV1,
    candidate: CompatibilityProjectionV1,
) -> Tuple[str, ...]:
    return tuple(path for path in _PROJECTION_FIELDS if baseline.value(path) != candidate.value(path))


def _case_key(value: Any) -> tuple[str, int, str]:
    return value.task_id, value.case_version, value.canonical_case_digest


def _trial_key(value: Any) -> tuple[str, int, str, int]:
    return value.task_id, value.case_version, value.canonical_case_digest, value.trial_index


def _trial_map(source: VerifiedRunEvaluation) -> Dict[tuple[str, int, str, int], Any]:
    result: Dict[tuple[str, int, str, int], Any] = {}
    for trial in source.trials:
        score = trial.trial_score
        key = (score.task_id, score.case_version, score.canonical_case_digest, score.trial_index)
        if key in result:
            raise ArtifactIntegrityError("verified Evaluation contains duplicate exact Trial pairing keys")
        result[key] = trial
    return result


def _compatibility_projection(
    source: VerifiedRunEvaluation,
    policy: ComparisonPolicyV1,
) -> CompatibilityProjectionV1:
    trials = source.trials
    first_score = trials[0].trial_score
    compatibility = first_score.compatibility
    by_task: Dict[str, Any] = {}
    score_by_task: Dict[str, Any] = {}
    for trial in trials:
        by_task.setdefault(trial.task_id, trial.eval_case)
        score_by_task.setdefault(trial.task_id, trial.trial_score)
    cases = tuple(by_task[key] for key in sorted(by_task))
    snapshot_entries = tuple(source.case_snapshot.case(case.task_id) for case in cases)
    judge_profiles = source.bundle.evaluator_execution.evaluator.judge_profiles
    scoring_policy = {
        name: getattr(compatibility, name)
        for name in (
            "protocol_id",
            "intent_evaluator_revision",
            "review_evaluator_revision",
            "intent_policy_version",
            "intent_normalization_version",
            "review_policy_version",
            "assignment_policy_version",
            "location_policy_version",
            "evidence_policy_version",
        )
    }
    values: Mapping[str, Any] = {
        "suite.id": source.run_config.suite.suite_id,
        "suite.version": source.run_config.suite.suite_version,
        "suite.manifest_digest": source.run_config.suite.manifest_digest,
        "case_snapshot.digest": source.case_snapshot.digest(),
        "cases.task_ids": [case.task_id for case in cases],
        "cases.versions": [{"task_id": case.task_id, "value": case.case_version} for case in cases],
        "cases.canonical_case_digests": [{"task_id": case.task_id, "value": case.digest()} for case in cases],
        "cases.input_digests": [{"task_id": case.task_id, "value": case.eval_input().digest()} for case in cases],
        "cases.truth_digests": [
            {
                "task_id": case.task_id,
                "value": canonical_sha256(
                    {
                        "intent_truth": case.intent_truth.to_dict(),
                        "review_truth": case.review_truth.to_dict(),
                        "clarification_script": case.clarification_script.to_dict(),
                        "review_evaluator_context": case.review_evaluator_context.to_dict(),
                    }
                ),
            }
            for case in cases
        ],
        "cases.snapshot_entry_digests": [{"task_id": item.task_id, "value": canonical_sha256(item.to_dict())} for item in snapshot_entries],
        "cases.truth_completeness": [{"task_id": case.task_id, "value": case.review_truth.completeness.value} for case in cases],
        "cases.novel_finding_policy": [{"task_id": case.task_id, "value": case.review_truth.novel_finding_policy.value} for case in cases],
        "cases.metric_authority_profile_digests": [{"task_id": case.task_id, "value": score_by_task[case.task_id].compatibility.metric_authority_profile_digest} for case in cases],
        "cases.protocol_ids": [{"task_id": item.task_id, "value": item.manifest_case.protocol_id} for item in snapshot_entries],
        "trial.count": source.run_config.trial_count,
        "target.kind": compatibility.target_kind.value,
        "wire_contract.digest": compatibility.wire_contract_digest,
        "materialization.protocol": source.run_config.materializer_protocol,
        "isolation.profile": compatibility.isolation_profile,
        "clarification_matcher.digest": compatibility.clarification_matcher_config_digest,
        "evaluator.evaluation_revision": source.bundle.evaluation_revision,
        "evaluator.execution.digest": source.bundle.evaluator_execution.digest(),
        "evaluator.configuration.digest": source.bundle.evaluator_execution.evaluator_config_digest,
        "evaluator.judge_profiles.digest": canonical_sha256([item.to_dict() for item in judge_profiles]),
        "evaluator.judge_rubrics.digest": canonical_sha256([{"kind": item.kind.value, "rubric_id": item.rubric_id, "rubric_version": item.rubric_version, "rubric_digest": item.rubric_digest} for item in judge_profiles]),
        "evaluator.judge_execution.digest": canonical_sha256({"judge_budgets": source.bundle.evaluator_execution.judge_budgets.to_dict(), "cache_policy_version": source.bundle.evaluator_execution.cache_policy_version}),
        "metrics_policy.digest": compatibility.metrics_policy_digest,
        "metric_authority.profile.digest": compatibility.metric_authority_profile_digest,
        "metric_authority.policy.digest": compatibility.metric_authority_policy_digest,
        "scoring_policy.digest": canonical_sha256(scoring_policy),
        "statistics_policy.digest": canonical_sha256(policy.statistics_policy.to_dict()),
    }
    return CompatibilityProjectionV1(tuple(CompatibilityFieldV1(path, values[path]) for path in _PROJECTION_FIELDS))


def _agent_provenance(source: VerifiedRunEvaluation) -> AgentProvenanceV1:
    config = source.run_config
    agent = config.agent
    capabilities = config.adapter_capabilities
    values: Mapping[str, Any] = {
        "run_id": source.run_id,
        "evaluation_id": source.evaluation_id,
        "run_instance_key": config.run_instance_key,
        "agent_config_digest": config.agent_config_digest,
        "agent.id": agent.agent_id,
        "agent.name": agent.agent_name,
        "agent.version": agent.agent_version,
        "agent.commit": agent.commit,
        "agent.model": agent.model,
        "agent.provider": agent.provider,
        "agent.parameters_digest": canonical_sha256(agent.to_dict()["parameters"]),
        "agent.prompt_config_digest": agent.prompt_config_digest,
        "adapter_capabilities_digest": config.adapter_capabilities_digest,
        "adapter.id": capabilities.adapter_id,
        "adapter.version": capabilities.adapter_version,
        "adapter.internal_policy_digest": canonical_sha256(
            {
                "evidence_kinds": [item.value for item in capabilities.evidence_kinds],
                "clarification_protocol": capabilities.clarification_protocol,
                "trace_protocol": capabilities.trace_protocol,
                "subprocess_wire_version": capabilities.subprocess_wire_version,
            }
        ),
        "agent.resource_policy_digest": canonical_sha256(
            {
                "agent_timeout_seconds": config.resource_budgets.agent_timeout_seconds,
                "max_agent_output_bytes": config.resource_budgets.max_agent_output_bytes,
                "max_trace_bytes": config.resource_budgets.max_trace_bytes,
                "max_parallel_trials": config.resource_budgets.max_parallel_trials,
            }
        ),
    }
    return AgentProvenanceV1(tuple(CompatibilityFieldV1(path, values[path]) for path in _AGENT_FIELDS))


def _metric_deltas(
    baseline: RunStatisticsV1,
    candidate: RunStatisticsV1,
    policy: StatisticsPolicyV1,
) -> Tuple[MetricDeltaV1, ...]:
    case_count = len(baseline.metrics[0].case_contributions)
    if (
        case_count
        * policy.bootstrap_iterations
        * len(CoreMetric)
        > MAX_RUN_BOOTSTRAP_DRAWS
    ):
        raise _error(
            "total paired bootstrap work exceeds the comparison resource budget"
        )
    result = []
    for metric in sorted(CoreMetric, key=lambda item: item.value):
        b_source = baseline.metric(metric)
        c_source = candidate.metric(metric)
        if b_source.unit is not c_source.unit or b_source.direction is not c_source.direction:
            raise _error("paired Run metrics use incompatible metadata")
        b = MetricValueV1.from_source(b_source)
        c = MetricValueV1.from_source(c_source)
        delta, classification, reason = _delta_fields(b, c, b_source.direction)
        result.append(
            MetricDeltaV1(
                metric, b_source.unit, b_source.direction, b, c,
                delta, classification, reason,
                _paired_delta_interval(b_source.case_contributions, c_source.case_contributions, policy),
            )
        )
    return tuple(result)


def _case_deltas(
    baseline: RunStatisticsV1,
    candidate: RunStatisticsV1,
    baseline_trials: Mapping[tuple[str, int, str, int], Any],
    candidate_trials: Mapping[tuple[str, int, str, int], Any],
) -> Tuple[CaseDeltaV1, ...]:
    metrics = tuple(sorted(CoreMetric, key=lambda item: item.value))
    first_baseline = baseline.metric(metrics[0]).case_contributions
    first_candidate = candidate.metric(metrics[0]).case_contributions
    case_keys = tuple(_case_key(item) for item in first_baseline)
    if case_keys != tuple(_case_key(item) for item in first_candidate):
        raise _error("Run statistics do not share exact Case contribution keys")
    result = []
    for case_key in case_keys:
        aggregate_deltas = []
        for metric in metrics:
            b = next(item for item in baseline.metric(metric).case_contributions if _case_key(item) == case_key)
            c = next(item for item in candidate.metric(metric).case_contributions if _case_key(item) == case_key)
            aggregate_deltas.append(ContributionDeltaV1.create(b, c))
        paired_trials = []
        for trial_index in range(1, baseline.trial_count + 1):
            key = (*case_key, trial_index)
            b_trial = baseline_trials[key]
            c_trial = candidate_trials[key]
            per_trial_deltas = []
            first_b_contribution = None
            first_c_contribution = None
            for metric in metrics:
                b_projection = baseline.trial_metric(trial_index, metric)
                c_projection = candidate.trial_metric(trial_index, metric)
                b_contribution = next(item for item in b_projection.case_contributions if _case_key(item) == case_key)
                c_contribution = next(item for item in c_projection.case_contributions if _case_key(item) == case_key)
                first_b_contribution = first_b_contribution or b_contribution
                first_c_contribution = first_c_contribution or c_contribution
                per_trial_deltas.append(ContributionDeltaV1.create(b_contribution, c_contribution))
            assert first_b_contribution is not None and first_c_contribution is not None
            b_judge = JudgeCoverageV1.from_statistics(first_b_contribution.coverage)
            c_judge = JudgeCoverageV1.from_statistics(first_c_contribution.coverage)
            b_ref = TrialReferenceV1(
                baseline.source_binding.run_id, baseline.source_binding.evaluation_id,
                *case_key, trial_index, b_trial.trial_id, b_trial.submission.status, b_judge,
            )
            c_ref = TrialReferenceV1(
                candidate.source_binding.run_id, candidate.source_binding.evaluation_id,
                *case_key, trial_index, c_trial.trial_id, c_trial.submission.status, c_judge,
            )
            paired_trials.append(
                PairedTrialDeltaV1(
                    *case_key, trial_index, b_ref, c_ref, tuple(per_trial_deltas),
                    JudgeCoverageDeltaV1(
                        b_judge, c_judge,
                        c_judge.request_count-b_judge.request_count,
                        c_judge.graded_count-b_judge.graded_count,
                        c_judge.failure_count-b_judge.failure_count,
                        c_judge.ungraded_count-b_judge.ungraded_count,
                        c_judge.semantic_unknown_count-b_judge.semantic_unknown_count,
                    ),
                )
            )
        b_case_coverage = next(item for item in baseline.metric(metrics[0]).case_contributions if _case_key(item) == case_key).coverage
        c_case_coverage = next(item for item in candidate.metric(metrics[0]).case_contributions if _case_key(item) == case_key).coverage
        result.append(CaseDeltaV1(*case_key, tuple(aggregate_deltas), tuple(paired_trials), JudgeCoverageDeltaV1.create(b_case_coverage, c_case_coverage)))
    return tuple(result)


def _validate_case_delta_sources(
    cases: Sequence[CaseDeltaV1],
    baseline: RunStatisticsV1,
    candidate: RunStatisticsV1,
) -> None:
    metrics = tuple(sorted(CoreMetric, key=lambda item: item.value))
    expected_case_keys = tuple(_case_key(item) for item in baseline.metric(metrics[0]).case_contributions)
    if tuple((item.task_id, item.case_version, item.canonical_case_digest) for item in cases) != expected_case_keys:
        raise _error("CaseDeltaV1 keys differ from nested Run statistics")
    for case in cases:
        key = (case.task_id, case.case_version, case.canonical_case_digest)
        for metric in metrics:
            delta = case.metric_delta(metric)
            b = next(item for item in baseline.metric(metric).case_contributions if _case_key(item) == key)
            c = next(item for item in candidate.metric(metric).case_contributions if _case_key(item) == key)
            if delta != ContributionDeltaV1.create(b, c):
                raise _error("Case metric delta differs from recomputed nested statistics")
        b_case = next(item for item in baseline.metric(metrics[0]).case_contributions if _case_key(item) == key)
        c_case = next(item for item in candidate.metric(metrics[0]).case_contributions if _case_key(item) == key)
        if case.judge_coverage_delta != JudgeCoverageDeltaV1.create(
            b_case.coverage, c_case.coverage
        ):
            raise _error("Case Judge coverage differs from nested statistics")
        if tuple(item.trial_index for item in case.paired_trials) != tuple(
            range(1, baseline.trial_count + 1)
        ):
            raise _error("Case paired Trial refs differ from nested Trial count")
        for paired in case.paired_trials:
            if (
                paired.baseline.run_id != baseline.source_binding.run_id
                or paired.baseline.evaluation_id
                != baseline.source_binding.evaluation_id
                or paired.candidate.run_id != candidate.source_binding.run_id
                or paired.candidate.evaluation_id
                != candidate.source_binding.evaluation_id
            ):
                raise _error(
                    "paired Trial refs differ from nested source bindings"
                )
            for metric in metrics:
                delta = paired.metric_delta(metric)
                b = next(item for item in baseline.trial_metric(paired.trial_index, metric).case_contributions if _case_key(item) == key)
                c = next(item for item in candidate.trial_metric(paired.trial_index, metric).case_contributions if _case_key(item) == key)
                if delta != ContributionDeltaV1.create(b, c):
                    raise _error("Trial metric delta differs from recomputed nested statistics")
            b_first = next(item for item in baseline.trial_metric(paired.trial_index, metrics[0]).case_contributions if _case_key(item) == key)
            c_first = next(item for item in candidate.trial_metric(paired.trial_index, metrics[0]).case_contributions if _case_key(item) == key)
            if paired.baseline.judge_coverage != JudgeCoverageV1.from_statistics(b_first.coverage) or paired.candidate.judge_coverage != JudgeCoverageV1.from_statistics(c_first.coverage):
                raise _error("Trial reference Judge coverage differs from nested statistics")
            expected_failure = b_first.coverage.agent_failure_count == 1
            if (paired.baseline.submission_status is SubmissionStatus.COMPLETED) == expected_failure:
                raise _error("baseline Trial outcome differs from nested failure coverage")
            expected_failure = c_first.coverage.agent_failure_count == 1
            if (paired.candidate.submission_status is SubmissionStatus.COMPLETED) == expected_failure:
                raise _error("candidate Trial outcome differs from nested failure coverage")


@dataclass(frozen=True)
class RunComparisonV1(_JsonModel):
    schema_version: str
    comparison_id: str
    status: ComparisonStatus
    baseline_binding: AnalysisSourceBinding
    candidate_binding: AnalysisSourceBinding
    compatibility: ComparisonCompatibilityV1
    baseline_statistics: RunStatisticsV1
    candidate_statistics: RunStatisticsV1
    metric_deltas: Tuple[MetricDeltaV1, ...]
    case_deltas: Tuple[CaseDeltaV1, ...]
    incompatibilities: Tuple[str, ...]
    algorithm_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != RUN_COMPARISON_SCHEMA_VERSION:
            raise _error("RunComparisonV1 schema_version is unsupported")
        if type(self.status) is not ComparisonStatus:
            raise _error("RunComparisonV1 status is invalid")
        if type(self.baseline_binding) is not AnalysisSourceBinding or type(self.candidate_binding) is not AnalysisSourceBinding or self.baseline_binding == self.candidate_binding:
            raise _error("RunComparisonV1 requires two distinct typed source bindings")
        if type(self.baseline_statistics) is not RunStatisticsV1 or type(self.candidate_statistics) is not RunStatisticsV1:
            raise _error("RunComparisonV1 nested statistics are invalid")
        if self.baseline_statistics.source_binding != self.baseline_binding or self.candidate_statistics.source_binding != self.candidate_binding:
            raise _error("RunComparisonV1 bindings differ from nested statistics")
        if self.baseline_statistics.bootstrap_policy != self.candidate_statistics.bootstrap_policy:
            raise _error("RunComparisonV1 nested StatisticsPolicy values differ")
        policy = ComparisonPolicyV1.default(self.baseline_statistics.bootstrap_policy)
        if self.algorithm_digest != policy.algorithm_digest or self.compatibility.policy_digest != policy.policy_digest:
            raise _error("RunComparisonV1 algorithm/policy digest differs from nested policy")
        for label, projection, statistics, provenance, binding in (
            (
                "baseline",
                self.compatibility.baseline_projection,
                self.baseline_statistics,
                self.compatibility.agent_delta.baseline,
                self.baseline_binding,
            ),
            (
                "candidate",
                self.compatibility.candidate_projection,
                self.candidate_statistics,
                self.compatibility.agent_delta.candidate,
                self.candidate_binding,
            ),
        ):
            case_sources = statistics.metrics[0].case_contributions
            try:
                projected_evaluation_id = derive_evaluation_id(
                    binding.run_id,
                    projection.value("evaluator.execution.digest"),
                    projection.value("evaluator.evaluation_revision"),
                )
            except (TypeError, ValueError) as exc:
                raise _error(
                    f"RunComparisonV1 {label} evaluator projection "
                    "cannot derive a canonical Evaluation identity"
                ) from exc
            if projected_evaluation_id != binding.evaluation_id:
                raise _error(
                    f"RunComparisonV1 {label} evaluator projection differs "
                    "from its source binding Evaluation identity"
                )
            if (
                projection.value("case_snapshot.digest")
                != binding.case_snapshot_digest
            ):
                raise _error(
                    f"RunComparisonV1 {label} projection case_snapshot.digest "
                    "differs from its source binding"
                )
            if (
                projection.value("trial.count") != statistics.trial_count
                or projection.value("cases.task_ids")
                != [item.task_id for item in case_sources]
                or projection.value("cases.versions")
                != [
                    {"task_id": item.task_id, "value": item.case_version}
                    for item in case_sources
                ]
                or projection.value("cases.canonical_case_digests")
                != [
                    {
                        "task_id": item.task_id,
                        "value": item.canonical_case_digest,
                    }
                    for item in case_sources
                ]
                or projection.value("statistics_policy.digest")
                != canonical_sha256(statistics.bootstrap_policy.to_dict())
            ):
                raise _error(
                    f"RunComparisonV1 {label} projection differs from nested statistics"
                )
            if (
                provenance.value("run_id") != binding.run_id
                or provenance.value("evaluation_id") != binding.evaluation_id
            ):
                raise _error(
                    f"RunComparisonV1 {label} provenance differs from nested binding"
                )
        expected_incompatibilities = list(_projection_incompatibilities(self.compatibility.baseline_projection, self.compatibility.candidate_projection))
        if self.baseline_statistics.trial_count < MIN_COMPARISON_TRIAL_COUNT or self.candidate_statistics.trial_count < MIN_COMPARISON_TRIAL_COUNT:
            if "trial.count" not in expected_incompatibilities:
                expected_incompatibilities.append("trial.count")
        expected_incompatibilities = tuple(path for path in _PROJECTION_FIELDS if path in expected_incompatibilities)
        if tuple(self.incompatibilities) != expected_incompatibilities:
            raise _error("RunComparisonV1 incompatibilities differ from recomputed projections")
        metrics = tuple(self.metric_deltas)
        cases = tuple(self.case_deltas)
        if expected_incompatibilities:
            if self.status is not ComparisonStatus.NOT_COMPARABLE or metrics or cases:
                raise _error("not_comparable RunComparisonV1 must not contain partial deltas")
        else:
            if self.status is not ComparisonStatus.COMPARABLE:
                raise _error("compatible projections require comparable status")
            expected_metrics = _metric_deltas(self.baseline_statistics, self.candidate_statistics, policy.statistics_policy)
            if metrics != expected_metrics:
                raise _error("RunComparisonV1 metric deltas differ from recomputed nested statistics")
            _validate_case_delta_sources(cases, self.baseline_statistics, self.candidate_statistics)
        object.__setattr__(self, "metric_deltas", metrics)
        object.__setattr__(self, "case_deltas", cases)
        object.__setattr__(self, "incompatibilities", expected_incompatibilities)
        if self.comparison_id != stable_id("run-comparison-v1", self._identity_dict()):
            raise _error("RunComparisonV1 comparison_id is not canonical")
        canonical_json_bytes(self.to_dict())

    def _identity_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "baseline_binding": self.baseline_binding.to_dict(),
            "candidate_binding": self.candidate_binding.to_dict(),
            "compatibility": self.compatibility.to_dict(),
            "baseline_statistics": self.baseline_statistics.to_dict(),
            "candidate_statistics": self.candidate_statistics.to_dict(),
            "metric_deltas": [item.to_dict() for item in self.metric_deltas],
            "case_deltas": [item.to_dict() for item in self.case_deltas],
            "incompatibilities": list(self.incompatibilities),
            "algorithm_digest": self.algorithm_digest,
        }

    @classmethod
    def create(cls, **values: Any) -> "RunComparisonV1":
        identity = {
            "schema_version": RUN_COMPARISON_SCHEMA_VERSION,
            "status": values["status"].value,
            "baseline_binding": values["baseline_binding"].to_dict(),
            "candidate_binding": values["candidate_binding"].to_dict(),
            "compatibility": values["compatibility"].to_dict(),
            "baseline_statistics": values["baseline_statistics"].to_dict(),
            "candidate_statistics": values["candidate_statistics"].to_dict(),
            "metric_deltas": [item.to_dict() for item in values["metric_deltas"]],
            "case_deltas": [item.to_dict() for item in values["case_deltas"]],
            "incompatibilities": list(values["incompatibilities"]),
            "algorithm_digest": values["algorithm_digest"],
        }
        return cls(RUN_COMPARISON_SCHEMA_VERSION, stable_id("run-comparison-v1", identity), **values)

    @property
    def judge_coverage_delta(self) -> Optional[JudgeCoverageDeltaV1]:
        if self.status is not ComparisonStatus.COMPARABLE:
            return None
        first = tuple(sorted(CoreMetric, key=lambda item: item.value))[0]
        return JudgeCoverageDeltaV1.create(
            self.baseline_statistics.metric(first).coverage,
            self.candidate_statistics.metric(first).coverage,
        )

    def metric_delta(self, metric: CoreMetric) -> MetricDeltaV1:
        if type(metric) is not CoreMetric:
            raise TypeError("metric must be CoreMetric")
        return next(item for item in self.metric_deltas if item.metric is metric)

    def to_dict(self) -> Dict[str, Any]:
        return {**self._identity_dict(), "comparison_id": self.comparison_id}

    @classmethod
    def from_dict(cls, value: Any) -> "RunComparisonV1":
        payload = _exact(value, ("comparison_id", "schema_version", "status", "baseline_binding", "candidate_binding", "compatibility", "baseline_statistics", "candidate_statistics", "metric_deltas", "case_deltas", "incompatibilities", "algorithm_digest"), "RunComparisonV1")
        metrics = _array(payload["metric_deltas"], "RunComparisonV1.metric_deltas", len(CoreMetric))
        cases = _array(payload["case_deltas"], "RunComparisonV1.case_deltas", MAX_BOOTSTRAP_CASES)
        incompatibilities = _array(payload["incompatibilities"], "RunComparisonV1.incompatibilities", len(_PROJECTION_FIELDS))
        return cls(
            payload["schema_version"], payload["comparison_id"],
            _enum(ComparisonStatus, payload["status"], "RunComparisonV1.status"),
            AnalysisSourceBinding.from_dict(payload["baseline_binding"]),
            AnalysisSourceBinding.from_dict(payload["candidate_binding"]),
            ComparisonCompatibilityV1.from_dict(payload["compatibility"]),
            RunStatisticsV1.from_dict(payload["baseline_statistics"]),
            RunStatisticsV1.from_dict(payload["candidate_statistics"]),
            tuple(MetricDeltaV1.from_dict(item) for item in metrics),
            tuple(CaseDeltaV1.from_dict(item) for item in cases),
            tuple(_identifier(item, "RunComparisonV1 incompatibility") for item in incompatibilities),
            payload["algorithm_digest"],
        )

    @classmethod
    def from_json(cls, data: Any) -> "RunComparisonV1":
        return _from_json(cls, data, "RunComparisonV1 JSON")


def compare_runs(
    baseline: VerifiedRunEvaluation,
    candidate: VerifiedRunEvaluation,
    policy: ComparisonPolicyV1,
) -> RunComparisonV1:
    """Compare two completed, source-bound Evaluations without executing them."""

    if type(baseline) is not VerifiedRunEvaluation or type(candidate) is not VerifiedRunEvaluation:
        raise TypeError("baseline and candidate must be VerifiedRunEvaluation")
    if type(policy) is not ComparisonPolicyV1:
        raise TypeError("policy must be ComparisonPolicyV1")
    canonical_policy = ComparisonPolicyV1.from_dict(policy.to_dict())
    baseline_binding = baseline.verify()
    candidate_binding = candidate.verify()
    if baseline_binding == candidate_binding:
        raise _error("strict paired comparison requires two distinct Evaluations")
    baseline_statistics = compute_run_statistics(
        baseline.bundle,
        run_config=baseline.run_config,
        case_snapshot=baseline.case_snapshot,
        policy=canonical_policy.statistics_policy,
    )
    candidate_statistics = compute_run_statistics(
        candidate.bundle,
        run_config=candidate.run_config,
        case_snapshot=candidate.case_snapshot,
        policy=canonical_policy.statistics_policy,
    )
    baseline_projection = _compatibility_projection(baseline, canonical_policy)
    candidate_projection = _compatibility_projection(candidate, canonical_policy)
    baseline_agent = _agent_provenance(baseline)
    candidate_agent = _agent_provenance(candidate)
    compatibility = ComparisonCompatibilityV1(
        COMPARISON_COMPATIBILITY_SCHEMA_VERSION,
        canonical_policy.policy_digest,
        baseline_projection,
        candidate_projection,
        baseline_projection if baseline_projection == candidate_projection else None,
        AgentDeltaV1(baseline_agent, candidate_agent, _agent_changes(baseline_agent, candidate_agent)),
    )
    baseline_trials = _trial_map(baseline)
    candidate_trials = _trial_map(candidate)
    incompatibilities = list(_projection_incompatibilities(baseline_projection, candidate_projection))
    pairing_keys_differ = set(baseline_trials) != set(candidate_trials)
    if baseline_statistics.trial_count < MIN_COMPARISON_TRIAL_COUNT or candidate_statistics.trial_count < MIN_COMPARISON_TRIAL_COUNT:
        if "trial.count" not in incompatibilities:
            incompatibilities.append("trial.count")
    # A valid hard Case/Trial mismatch is reported as not_comparable.  A
    # residual key mismatch with otherwise identical projections is source
    # corruption and must never be zipped or silently dropped.
    if pairing_keys_differ and not incompatibilities:
        raise _error("exact Trial pairing keys differ despite source-bound projections")
    ordered_incompatibilities = tuple(path for path in _PROJECTION_FIELDS if path in incompatibilities)
    if ordered_incompatibilities:
        status = ComparisonStatus.NOT_COMPARABLE
        metric_deltas: Tuple[MetricDeltaV1, ...] = ()
        case_deltas: Tuple[CaseDeltaV1, ...] = ()
    else:
        status = ComparisonStatus.COMPARABLE
        metric_deltas = _metric_deltas(baseline_statistics, candidate_statistics, canonical_policy.statistics_policy)
        case_deltas = _case_deltas(baseline_statistics, candidate_statistics, baseline_trials, candidate_trials)
    final_baseline_binding = baseline.verify()
    final_candidate_binding = candidate.verify()
    if (
        final_baseline_binding != baseline_binding
        or final_candidate_binding != candidate_binding
        or baseline_statistics.source_binding != baseline_binding
        or candidate_statistics.source_binding != candidate_binding
    ):
        raise ArtifactIntegrityError(
            "comparison source bindings changed during analysis"
        )
    if (
        _compatibility_projection(baseline, canonical_policy)
        != baseline_projection
        or _compatibility_projection(candidate, canonical_policy)
        != candidate_projection
        or _agent_provenance(baseline) != baseline_agent
        or _agent_provenance(candidate) != candidate_agent
    ):
        raise ArtifactIntegrityError(
            "comparison compatibility sources changed during analysis"
        )
    return RunComparisonV1.create(
        status=status,
        baseline_binding=baseline_binding,
        candidate_binding=candidate_binding,
        compatibility=compatibility,
        baseline_statistics=baseline_statistics,
        candidate_statistics=candidate_statistics,
        metric_deltas=metric_deltas,
        case_deltas=case_deltas,
        incompatibilities=ordered_incompatibilities,
        algorithm_digest=canonical_policy.algorithm_digest,
    )


__all__ = [
    "COMPARISON_POLICY_SCHEMA_VERSION",
    "COMPARISON_COMPATIBILITY_SCHEMA_VERSION",
    "RUN_COMPARISON_SCHEMA_VERSION",
    "COMPARISON_ALGORITHM_VERSION",
    "MIN_COMPARISON_TRIAL_COUNT",
    "MAX_COMPARISON_BYTES",
    "REQUIRED_CASE_FIELDS",
    "REQUIRED_EVALUATOR_FIELDS",
    "ComparisonError",
    "ComparisonStatus",
    "DeltaClassification",
    "DeltaNullReason",
    "DeltaIntervalStatus",
    "VerifiedRunEvaluation",
    "ComparisonPolicyV1",
    "CompatibilityFieldV1",
    "CompatibilityProjectionV1",
    "AgentProvenanceV1",
    "AgentFieldDeltaV1",
    "AgentDeltaV1",
    "ComparisonCompatibilityV1",
    "MetricValueV1",
    "ContributionDeltaV1",
    "PairedBootstrapCoverageV1",
    "PairedDeltaConfidenceIntervalV1",
    "MetricDeltaV1",
    "JudgeCoverageV1",
    "JudgeCoverageDeltaV1",
    "TrialReferenceV1",
    "PairedTrialDeltaV1",
    "CaseDeltaV1",
    "RunComparisonV1",
    "compare_runs",
]
