"""Versioned, source-bound metrics for the core code-review Eval system.

This module is deterministic.  It never invokes an Agent or an LLM, and it
never treats Prompt/model/Runtime/Risk configuration as a product score.  It
turns trusted Intent/Review evaluator outputs into per-Trial contributions,
then aggregates numerators and denominators (never percentages) at Case and
group level.

The public top-level score models are controlled artifacts.  Callers create
them through :class:`TrialScorer` and :class:`MetricsAggregator`; hydration
requires the original source objects and replays the score before accepting a
persisted payload.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields as dataclass_fields
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .assignment import ASSIGNMENT_POLICY_VERSION
from .cases import CaseDimension, SuiteCase, WireContractV2
from .config import (
    DEFAULT_METRIC_AUTHORITY_POLICY_VERSION,
    EvalRunConfig,
    EvaluatorExecutionConfig,
    derive_evaluation_id,
    validate_evaluation_id,
)
from .evidence_checker import EVIDENCE_INTEGRITY_POLICY_VERSION
from .intent_evaluator import (
    INTENT_EVALUATION_SCHEMA_VERSION,
    INTENT_EVALUATOR_REVISION,
    INTENT_NORMALIZATION_POLICY_VERSION,
    IntentEvaluationResult,
    IntentEvaluationStatus,
    IntentJudgeRelation,
)
from .match_location import LOCATION_MATCH_POLICY_VERSION
from .models import (
    EvalCase,
    EvalSubmission,
    EvidenceSupport,
    FailureCode,
    FindingSeverity,
    IssueJudgement,
    MetricAuthority,
    NovelFindingPolicy,
    ReviewTargetKind,
    SchemaError,
    SubmissionStatus,
    SubmissionUsage,
    TraceRef,
    TruthCompleteness,
    UnsupportedProtocolVersionError,
    canonical_json,
    canonical_json_bytes,
    canonical_sha256,
    stable_id,
    _json_tree,
    _strict_json_loads,
)
from .review_evaluator import (
    REVIEW_EVALUATION_SCHEMA_VERSION,
    REVIEW_MATCH_POLICY_VERSION,
    EvidenceSupportResolution,
    FindingDisposition,
    ReviewEvaluationPhase,
    ReviewEvaluationResult,
    ReviewEvaluationStatus,
    ReviewTruthKind,
    REVIEW_EVALUATOR_REVISION,
    REVIEW_LOCATION_POLICY_VERSION,
)


TRIAL_SCORE_SCHEMA_VERSION = "eval_trial_score_v1"
CASE_SCORE_SCHEMA_VERSION = "eval_case_score_v1"
AGGREGATE_SCORE_SCHEMA_VERSION = "eval_aggregate_score_v1"
METRICS_POLICY_VERSION = "core-code-review-metrics-v1"
SEVERITY_WEIGHT_POLICY_VERSION = "severity-weight-policy-v1"
LINE_METRIC_POLICY_VERSION = "assigned-truth-location-v1"
METRICS_AGGREGATOR_REVISION = "metrics-aggregator-v1"
PPM_SCALE = 1_000_000
MISSING_COST_UNIT = "currency-unspecified"

MAX_SCORE_BYTES = 256 * 1024 * 1024
MAX_TRIAL_SCORES = 65_536
MAX_CASE_SCORES = 16_384
MAX_BREAKDOWN_ITEMS = 256
MAX_AUTHORITY_COMBINATIONS = 16


class MetricsError(ValueError):
    """A metrics source, score artifact, or aggregate is invalid."""


class FailureOutcomePolicy(str, Enum):
    COUNT_AS_MISSED = "count_as_missed-v1"
    EXCLUDE_WITH_COVERAGE = "exclude_with_coverage-v1"


class MetricKind(str, Enum):
    RATE = "rate"
    MEAN = "mean"
    COUNT = "count"


class MetricSourceStatus(str, Enum):
    GRADED = "graded"
    FAILURE_AS_MISS = "failure_as_miss"
    NOT_SCORABLE = "not_scorable"
    UNGRADED = "ungraded"
    FAILURE_EXCLUDED = "failure_excluded"
    MISSING = "missing"


class MetricNullReason(str, Enum):
    ZERO_DENOMINATOR = "zero_denominator"
    NOT_SCORABLE = "not_scorable"
    UNGRADED = "ungraded"
    FAILURE_EXCLUDED = "failure_excluded"
    MISSING = "missing"


class CoreMetric(str, Enum):
    INTENT_CLAIM_PRECISION = "intent_claim_precision"
    INTENT_CLAIM_RECALL = "intent_claim_recall"
    INTENT_PARTIALLY_SUPPORTED_RATE = "intent_partially_supported_rate"
    INTENT_UNSUPPORTED_RATE = "intent_unsupported_rate"
    INTENT_CONTRADICTED_RATE = "intent_contradicted_rate"
    INTENT_UNKNOWN_RATE = "intent_unknown_rate"
    CLARIFICATION_ACCURACY = "clarification_accuracy"
    INTENT_CASE_PASS_RATE = "intent_case_pass_rate"
    ISSUE_PRECISION = "issue_precision"
    ISSUE_RECALL = "issue_recall"
    ISSUE_F1 = "issue_f1"
    SEVERITY_WEIGHTED_RECALL = "severity_weighted_recall"
    CRITICAL_HIGH_MISS_COUNT = "critical_high_miss_count"
    FABRICATED_FINDINGS_PER_PR = "fabricated_findings_per_pr"
    FABRICATED_RATE = "fabricated_rate"
    PLAUSIBLE_RATE = "plausible_rate"
    REVIEW_UNKNOWN_RATE = "review_unknown_rate"
    LINE_PRECISION = "line_precision"
    LINE_RECALL = "line_recall"
    EVIDENCE_VALIDITY = "evidence_validity"
    EVIDENCE_SUPPORT_RATE = "evidence_support_rate"
    PUBLISHABLE_FINDING_PRECISION = "publishable_finding_precision"
    AGENT_FAILURE_RATE = "agent_failure_rate"
    JUDGE_FAILURE_RATE = "judge_failure_rate"
    JUDGE_UNGRADED_RATE = "judge_ungraded_rate"
    JUDGE_SEMANTIC_UNKNOWN_RATE = "judge_semantic_unknown_rate"


_METRIC_KINDS: Mapping[CoreMetric, MetricKind] = {
    **{metric: MetricKind.RATE for metric in CoreMetric},
    CoreMetric.CRITICAL_HIGH_MISS_COUNT: MetricKind.COUNT,
    CoreMetric.FABRICATED_FINDINGS_PER_PR: MetricKind.MEAN,
}
_CONTRIBUTION_METRICS = tuple(
    metric for metric in CoreMetric if metric is not CoreMetric.ISSUE_F1
)
_INTENT_METRICS = (
    CoreMetric.INTENT_CLAIM_PRECISION,
    CoreMetric.INTENT_CLAIM_RECALL,
    CoreMetric.INTENT_PARTIALLY_SUPPORTED_RATE,
    CoreMetric.INTENT_UNSUPPORTED_RATE,
    CoreMetric.INTENT_CONTRADICTED_RATE,
    CoreMetric.INTENT_UNKNOWN_RATE,
    CoreMetric.CLARIFICATION_ACCURACY,
    CoreMetric.INTENT_CASE_PASS_RATE,
)
_REVIEW_METRICS = (
    CoreMetric.ISSUE_PRECISION,
    CoreMetric.ISSUE_RECALL,
    CoreMetric.SEVERITY_WEIGHTED_RECALL,
    CoreMetric.CRITICAL_HIGH_MISS_COUNT,
    CoreMetric.FABRICATED_FINDINGS_PER_PR,
    CoreMetric.FABRICATED_RATE,
    CoreMetric.PLAUSIBLE_RATE,
    CoreMetric.REVIEW_UNKNOWN_RATE,
    CoreMetric.LINE_PRECISION,
    CoreMetric.LINE_RECALL,
    CoreMetric.EVIDENCE_VALIDITY,
    CoreMetric.EVIDENCE_SUPPORT_RATE,
    CoreMetric.PUBLISHABLE_FINDING_PRECISION,
)


def _error(message: str) -> MetricsError:
    return MetricsError(message)


def _strict_object(value: Any, fields: Sequence[str], context: str) -> Dict[str, Any]:
    if type(value) is not dict or set(value) != set(fields) or len(value) != len(fields):
        raise _error(f"{context} has unknown or missing fields")
    return value


def _array(value: Any, context: str, maximum: int) -> list[Any]:
    if type(value) is not list or len(value) > maximum:
        raise _error(f"{context} must be a bounded array")
    return value


def _id(value: Any, context: str, maximum: int = 512) -> str:
    if type(value) is not str or not value or len(value) > maximum:
        raise _error(f"{context} must be a bounded non-empty identifier")
    if value != value.strip() or any(
        character.isspace() or ord(character) < 32 or ord(character) == 127
        for character in value
    ):
        raise _error(f"{context} contains whitespace or controls")
    try:
        value.encode("utf-8", "strict")
    except UnicodeEncodeError as exc:
        raise _error(f"{context} contains invalid Unicode") from exc
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


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise _error(f"{context} must be an integer >= {minimum}")
    return value


def _number(value: Any, context: str) -> Any:
    if type(value) not in (int, float) or type(value) is bool:
        raise _error(f"{context} must be a finite number")
    if not math.isfinite(float(value)) or value < 0:
        raise _error(f"{context} must be a finite non-negative number")
    return value


def _enum(enum_type: type[Enum], value: Any, context: str) -> Any:
    if type(value) is not str:
        raise _error(f"{context} must be an enum string")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise _error(f"{context} has an unknown value") from exc


def _ratio_ppm(numerator: int, denominator: int) -> Optional[int]:
    if denominator == 0:
        return None
    return (numerator * PPM_SCALE + denominator // 2) // denominator


def _canonical_payload(value: Any, context: str, maximum: int = MAX_SCORE_BYTES) -> bytes:
    try:
        _json_tree(value, context)
        result = canonical_json_bytes(value)
    except (SchemaError, ValueError) as exc:
        raise _error(str(exc)) from exc
    if len(result) > maximum:
        raise _error(f"{context} exceeds its canonical byte budget")
    return result


def _sealed_instance(cls: type, values: Mapping[str, Any]) -> Any:
    """Internal construction primitive; public score constructors always reject."""

    model_fields = dataclass_fields(cls)
    if set(values) != {field.name for field in model_fields}:
        raise AssertionError(f"internal {cls.__name__} construction is incomplete")
    result = object.__new__(cls)
    for field in model_fields:
        object.__setattr__(result, field.name, values[field.name])
    result.__post_init__()
    return result


@dataclass(frozen=True)
class SeverityWeight:
    severity: FindingSeverity
    weight: int

    def __post_init__(self) -> None:
        if type(self.severity) is not FindingSeverity:
            raise _error("severity weight has an invalid severity")
        _integer(self.weight, "severity weight.weight", minimum=1)

    def to_dict(self) -> Dict[str, Any]:
        return {"severity": self.severity.value, "weight": self.weight}


@dataclass(frozen=True)
class SeverityWeightPolicy:
    version: str
    weights: Tuple[SeverityWeight, ...]
    digest: str

    def __post_init__(self) -> None:
        if self.version != SEVERITY_WEIGHT_POLICY_VERSION:
            raise _error("unsupported severity weight policy version")
        weights = tuple(self.weights)
        if any(type(item) is not SeverityWeight for item in weights):
            raise _error("severity weight policy contains an invalid item")
        ordered = tuple(sorted(weights, key=lambda item: item.severity.value))
        if weights != ordered or {item.severity for item in weights} != set(FindingSeverity):
            raise _error("severity weight policy must cover every severity canonically")
        canonical_weights = {
            FindingSeverity.LOW: 1,
            FindingSeverity.MEDIUM: 2,
            FindingSeverity.HIGH: 4,
            FindingSeverity.CRITICAL: 8,
        }
        if {item.severity: item.weight for item in weights} != canonical_weights:
            raise _error("severity-weight-policy-v1 has fixed 1/2/4/8 weights")
        if self.digest != canonical_sha256(self._identity_dict()):
            raise _error("severity weight policy digest is not canonical")

    def _identity_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "weights": [item.to_dict() for item in self.weights],
        }

    @classmethod
    def create(cls, weights: Mapping[FindingSeverity, int]) -> "SeverityWeightPolicy":
        values = tuple(
            sorted(
                (SeverityWeight(severity, weight) for severity, weight in weights.items()),
                key=lambda item: item.severity.value,
            )
        )
        identity = {
            "version": SEVERITY_WEIGHT_POLICY_VERSION,
            "weights": [item.to_dict() for item in values],
        }
        return cls(SEVERITY_WEIGHT_POLICY_VERSION, values, canonical_sha256(identity))

    def weight_for(self, severity: FindingSeverity) -> int:
        if type(severity) is not FindingSeverity:
            raise _error("severity weight lookup requires FindingSeverity")
        return next(item.weight for item in self.weights if item.severity is severity)

    def to_dict(self) -> Dict[str, Any]:
        return {**self._identity_dict(), "digest": self.digest}


DEFAULT_SEVERITY_WEIGHT_POLICY = SeverityWeightPolicy.create(
    {
        FindingSeverity.LOW: 1,
        FindingSeverity.MEDIUM: 2,
        FindingSeverity.HIGH: 4,
        FindingSeverity.CRITICAL: 8,
    }
)


@dataclass(frozen=True)
class LineMetricPolicy:
    version: str
    precision_rule: str
    recall_rule: str
    digest: str

    def __post_init__(self) -> None:
        if self.version != LINE_METRIC_POLICY_VERSION:
            raise _error("unsupported line metric policy version")
        _id(self.precision_rule, "line policy.precision_rule")
        _id(self.recall_rule, "line policy.recall_rule")
        if (
            self.precision_rule
            != "located-final-assignments/located-target-final-assignments"
            or self.recall_rule
            != "located-required-final-assignments/located-required-truth"
        ):
            raise _error("assigned-truth-location-v1 has fixed line rules")
        if self.digest != canonical_sha256(self._identity_dict()):
            raise _error("line metric policy digest is not canonical")

    def _identity_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "precision_rule": self.precision_rule,
            "recall_rule": self.recall_rule,
        }

    @classmethod
    def create(cls) -> "LineMetricPolicy":
        identity = {
            "version": LINE_METRIC_POLICY_VERSION,
            "precision_rule": "located-final-assignments/located-target-final-assignments",
            "recall_rule": "located-required-final-assignments/located-required-truth",
        }
        return cls(
            version=identity["version"],
            precision_rule=identity["precision_rule"],
            recall_rule=identity["recall_rule"],
            digest=canonical_sha256(identity),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {**self._identity_dict(), "digest": self.digest}


DEFAULT_LINE_METRIC_POLICY = LineMetricPolicy.create()


@dataclass(frozen=True)
class MetricsPolicy:
    version: str
    failure_outcome_policy: FailureOutcomePolicy
    severity_weights: SeverityWeightPolicy
    line_metrics: LineMetricPolicy
    digest: str

    def __post_init__(self) -> None:
        if self.version != METRICS_POLICY_VERSION:
            raise _error("unsupported metrics policy version")
        if type(self.failure_outcome_policy) is not FailureOutcomePolicy:
            raise _error("metrics policy has an invalid failure outcome policy")
        if type(self.severity_weights) is not SeverityWeightPolicy:
            raise _error("metrics policy requires SeverityWeightPolicy")
        if type(self.line_metrics) is not LineMetricPolicy:
            raise _error("metrics policy requires LineMetricPolicy")
        if self.digest != canonical_sha256(self._identity_dict()):
            raise _error("metrics policy digest is not canonical")

    def _identity_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "failure_outcome_policy": self.failure_outcome_policy.value,
            "severity_weights": self.severity_weights.to_dict(),
            "line_metrics": self.line_metrics.to_dict(),
        }

    @classmethod
    def create(
        cls,
        failure_outcome_policy: FailureOutcomePolicy = FailureOutcomePolicy.COUNT_AS_MISSED,
        severity_weights: SeverityWeightPolicy = DEFAULT_SEVERITY_WEIGHT_POLICY,
        line_metrics: LineMetricPolicy = DEFAULT_LINE_METRIC_POLICY,
    ) -> "MetricsPolicy":
        identity = {
            "version": METRICS_POLICY_VERSION,
            "failure_outcome_policy": failure_outcome_policy.value,
            "severity_weights": severity_weights.to_dict(),
            "line_metrics": line_metrics.to_dict(),
        }
        return cls(
            METRICS_POLICY_VERSION,
            failure_outcome_policy,
            severity_weights,
            line_metrics,
            canonical_sha256(identity),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {**self._identity_dict(), "digest": self.digest}


DEFAULT_METRICS_POLICY = MetricsPolicy.create()


def _canonical_metrics_policy(value: Any) -> MetricsPolicy:
    """Rebuild v1 policy data so mutated frozen instances are not trusted."""

    if type(value) is not MetricsPolicy:
        raise _error("metrics policy must be MetricsPolicy")
    try:
        canonical = MetricsPolicy.create(value.failure_outcome_policy)
        supplied = canonical_json_bytes(value.to_dict())
    except (AttributeError, SchemaError, TypeError, ValueError) as exc:
        raise _error(f"metrics policy is invalid: {exc}") from exc
    if supplied != canonical_json_bytes(canonical.to_dict()):
        raise _error("metrics policy is not a canonical v1 snapshot")
    return canonical


@dataclass(frozen=True)
class MetricContribution:
    metric: CoreMetric
    kind: MetricKind
    source_status: MetricSourceStatus
    numerator: Optional[int]
    denominator: Optional[int]

    def __post_init__(self) -> None:
        if type(self.metric) is not CoreMetric or self.metric is CoreMetric.ISSUE_F1:
            raise _error("metric contribution has an invalid metric")
        if type(self.kind) is not MetricKind or self.kind is not _METRIC_KINDS[self.metric]:
            raise _error("metric contribution kind is not canonical")
        if type(self.source_status) is not MetricSourceStatus:
            raise _error("metric contribution source status is invalid")
        included = self.source_status in {
            MetricSourceStatus.GRADED,
            MetricSourceStatus.FAILURE_AS_MISS,
        }
        if included:
            numerator = _integer(self.numerator, "metric contribution.numerator")
            denominator = _integer(self.denominator, "metric contribution.denominator")
            if self.kind is MetricKind.RATE and numerator > denominator:
                raise _error("rate contribution numerator exceeds denominator")
            if self.kind in {MetricKind.MEAN, MetricKind.COUNT} and denominator != 1:
                raise _error("per-Trial mean/count contribution denominator must equal one")
        elif self.numerator is not None or self.denominator is not None:
            raise _error("excluded metric contribution must use null numerator/denominator")

    @property
    def included(self) -> bool:
        return self.source_status in {
            MetricSourceStatus.GRADED,
            MetricSourceStatus.FAILURE_AS_MISS,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric.value,
            "kind": self.kind.value,
            "source_status": self.source_status.value,
            "numerator": self.numerator,
            "denominator": self.denominator,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "MetricContribution":
        payload = _strict_object(
            value,
            ("metric", "kind", "source_status", "numerator", "denominator"),
            "Metric contribution",
        )
        return cls(
            metric=_enum(CoreMetric, payload["metric"], "metric contribution.metric"),
            kind=_enum(MetricKind, payload["kind"], "metric contribution.kind"),
            source_status=_enum(
                MetricSourceStatus,
                payload["source_status"],
                "metric contribution.source_status",
            ),
            numerator=(
                None
                if payload["numerator"] is None
                else _integer(payload["numerator"], "metric contribution.numerator")
            ),
            denominator=(
                None
                if payload["denominator"] is None
                else _integer(payload["denominator"], "metric contribution.denominator")
            ),
        )


@dataclass(frozen=True)
class MetricCoverage:
    total_trial_count: int
    included_trial_count: int
    failure_as_miss_count: int
    zero_denominator_count: int
    not_scorable_count: int
    ungraded_count: int
    failure_excluded_count: int
    missing_count: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _integer(getattr(self, name), f"metric coverage.{name}")
        if (
            self.included_trial_count
            + self.not_scorable_count
            + self.ungraded_count
            + self.failure_excluded_count
            + self.missing_count
            != self.total_trial_count
        ):
            raise _error("metric coverage statuses do not cover Trials")
        if self.failure_as_miss_count > self.included_trial_count:
            raise _error("metric coverage failure-as-miss count exceeds included Trials")
        if self.zero_denominator_count > self.included_trial_count:
            raise _error("metric coverage zero denominators exceed included Trials")

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Any) -> "MetricCoverage":
        fields = tuple(cls.__dataclass_fields__)
        payload = _strict_object(value, fields, "Metric coverage")
        return cls(**{name: payload[name] for name in fields})


@dataclass(frozen=True)
class DerivedMetricCoverage:
    metric: CoreMetric
    coverage: MetricCoverage

    def __post_init__(self) -> None:
        if self.metric not in {
            CoreMetric.ISSUE_PRECISION,
            CoreMetric.ISSUE_RECALL,
        }:
            raise _error("derived metric coverage has an invalid source metric")
        if type(self.coverage) is not MetricCoverage:
            raise _error("derived metric coverage requires MetricCoverage")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric.value,
            "coverage": self.coverage.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "DerivedMetricCoverage":
        payload = _strict_object(
            value,
            ("metric", "coverage"),
            "Derived metric coverage",
        )
        return cls(
            metric=_enum(
                CoreMetric,
                payload["metric"],
                "derived metric coverage.metric",
            ),
            coverage=MetricCoverage.from_dict(payload["coverage"]),
        )


@dataclass(frozen=True)
class MetricAggregate:
    metric: CoreMetric
    kind: MetricKind
    numerator: Optional[int]
    denominator: Optional[int]
    value_ppm: Optional[int]
    null_reason: Optional[MetricNullReason]
    coverage: Optional[MetricCoverage]
    derived_coverages: Tuple[DerivedMetricCoverage, ...] = ()

    def __post_init__(self) -> None:
        if type(self.metric) is not CoreMetric:
            raise _error("metric aggregate has an invalid metric")
        if type(self.kind) is not MetricKind or self.kind is not _METRIC_KINDS[self.metric]:
            raise _error("metric aggregate kind is not canonical")
        derived = tuple(self.derived_coverages)
        if self.metric is CoreMetric.ISSUE_F1:
            if self.coverage is not None:
                raise _error("derived F1 must not publish ambiguous direct coverage")
            if tuple(item.metric for item in derived) != (
                CoreMetric.ISSUE_PRECISION,
                CoreMetric.ISSUE_RECALL,
            ):
                raise _error("derived F1 must bind precision and recall coverage")
            if derived[0].coverage.total_trial_count != derived[1].coverage.total_trial_count:
                raise _error("derived F1 source coverage populations differ")
        else:
            if type(self.coverage) is not MetricCoverage:
                raise _error("base metric aggregate requires MetricCoverage")
            if derived:
                raise _error("base metric aggregate cannot contain derived coverage")
        object.__setattr__(self, "derived_coverages", derived)
        if (self.numerator is None) != (self.denominator is None):
            raise _error("metric aggregate numerator/denominator coverage differs")
        if self.numerator is None:
            if self.value_ppm is not None or self.null_reason is None:
                raise _error("unavailable metric aggregate has invalid value/null reason")
            return
        numerator = _integer(self.numerator, "metric aggregate.numerator")
        denominator = _integer(self.denominator, "metric aggregate.denominator")
        if self.kind is MetricKind.RATE and numerator > denominator:
            raise _error("rate aggregate numerator exceeds denominator")
        if self.kind is MetricKind.COUNT:
            if self.value_ppm is not None or self.null_reason is not None:
                raise _error("count aggregate must expose its count, not a scaled value")
            return
        expected = _ratio_ppm(numerator, denominator)
        if denominator == 0:
            if self.value_ppm is not None or self.null_reason is not MetricNullReason.ZERO_DENOMINATOR:
                raise _error("zero-denominator metric aggregate is not canonical")
        elif self.value_ppm != expected or self.null_reason is not None:
            raise _error("metric aggregate scaled value is not canonical")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric.value,
            "kind": self.kind.value,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value_ppm": self.value_ppm,
            "null_reason": None if self.null_reason is None else self.null_reason.value,
            "coverage": None if self.coverage is None else self.coverage.to_dict(),
            "derived_coverages": [item.to_dict() for item in self.derived_coverages],
        }

    @classmethod
    def from_dict(cls, value: Any) -> "MetricAggregate":
        payload = _strict_object(
            value,
            (
                "metric",
                "kind",
                "numerator",
                "denominator",
                "value_ppm",
                "null_reason",
                "coverage",
                "derived_coverages",
            ),
            "Metric aggregate",
        )
        return cls(
            metric=_enum(CoreMetric, payload["metric"], "metric aggregate.metric"),
            kind=_enum(MetricKind, payload["kind"], "metric aggregate.kind"),
            numerator=(None if payload["numerator"] is None else _integer(payload["numerator"], "metric aggregate.numerator")),
            denominator=(None if payload["denominator"] is None else _integer(payload["denominator"], "metric aggregate.denominator")),
            value_ppm=(None if payload["value_ppm"] is None else _integer(payload["value_ppm"], "metric aggregate.value_ppm")),
            null_reason=(None if payload["null_reason"] is None else _enum(MetricNullReason, payload["null_reason"], "metric aggregate.null_reason")),
            coverage=(
                None
                if payload["coverage"] is None
                else MetricCoverage.from_dict(payload["coverage"])
            ),
            derived_coverages=tuple(
                DerivedMetricCoverage.from_dict(item)
                for item in _array(
                    payload["derived_coverages"],
                    "Metric aggregate derived coverage",
                    2,
                )
            ),
        )


@dataclass(frozen=True)
class NumericUsageAggregate:
    field: str
    unit: str
    sum_value: Optional[Any]
    observed_count: int
    population_count: int
    mean_value: Optional[Any]

    def __post_init__(self) -> None:
        _id(self.field, "usage aggregate.field")
        _id(self.unit, "usage aggregate.unit")
        _integer(self.observed_count, "usage aggregate.observed_count")
        _integer(self.population_count, "usage aggregate.population_count")
        if self.observed_count > self.population_count:
            raise _error("usage observed count exceeds population")
        if self.observed_count == 0:
            if self.sum_value is not None or self.mean_value is not None:
                raise _error("missing usage aggregate must use null values")
        else:
            total = _number(self.sum_value, "usage aggregate.sum_value")
            mean = _number(self.mean_value, "usage aggregate.mean_value")
            expected = total / self.observed_count
            if not math.isclose(float(mean), float(expected), rel_tol=0.0, abs_tol=1e-12):
                raise _error("usage aggregate mean is inconsistent")

    @property
    def missing_count(self) -> int:
        return self.population_count - self.observed_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "unit": self.unit,
            "sum_value": self.sum_value,
            "observed_count": self.observed_count,
            "population_count": self.population_count,
            "missing_count": self.missing_count,
            "mean_value": self.mean_value,
        }


@dataclass(frozen=True)
class UsageAggregate:
    elapsed_seconds: NumericUsageAggregate
    input_tokens: NumericUsageAggregate
    output_tokens: NumericUsageAggregate
    total_tokens: NumericUsageAggregate
    tool_calls: NumericUsageAggregate
    cost: NumericUsageAggregate
    cost_currency: Optional[str]

    def __post_init__(self) -> None:
        expected = {
            "elapsed_seconds": "seconds",
            "input_tokens": "tokens",
            "output_tokens": "tokens",
            "total_tokens": "tokens",
            "tool_calls": "calls",
        }
        for name, unit in expected.items():
            value = getattr(self, name)
            if type(value) is not NumericUsageAggregate or value.field != name or value.unit != unit:
                raise _error(f"usage aggregate.{name} has an invalid identity")
        if type(self.cost) is not NumericUsageAggregate or self.cost.field != "cost_amount":
            raise _error("usage cost aggregate identity is invalid")
        if self.cost_currency is None:
            if self.cost.unit != MISSING_COST_UNIT or self.cost.observed_count != 0:
                raise _error("missing usage cost must retain explicit zero-observation coverage")
        elif (
            type(self.cost_currency) is not str
            or len(self.cost_currency) != 3
            or self.cost_currency.upper() != self.cost_currency
            or self.cost.unit != self.cost_currency
        ):
            raise _error("usage cost currency is invalid")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "elapsed_seconds": self.elapsed_seconds.to_dict(),
            "input_tokens": self.input_tokens.to_dict(),
            "output_tokens": self.output_tokens.to_dict(),
            "total_tokens": self.total_tokens.to_dict(),
            "tool_calls": self.tool_calls.to_dict(),
            "cost": self.cost.to_dict(),
            "cost_currency": self.cost_currency,
        }


@dataclass(frozen=True)
class IntentScoreBinding:
    result_digest: str
    evaluator_revision: str
    policy_version: str
    normalization_version: str
    status: IntentEvaluationStatus
    judge_request_count: int
    judge_graded_count: int
    judge_failed_count: int
    judge_ungraded_count: int
    semantic_unknown_count: int

    def __post_init__(self) -> None:
        _digest(self.result_digest, "Intent score binding.result_digest")
        _id(self.evaluator_revision, "Intent score binding.evaluator_revision")
        _id(self.policy_version, "Intent score binding.policy_version")
        _id(self.normalization_version, "Intent score binding.normalization_version")
        if type(self.status) is not IntentEvaluationStatus or self.status is IntentEvaluationStatus.PENDING_JUDGE:
            raise _error("Intent score binding requires a terminal evaluator status")
        for name in (
            "judge_request_count",
            "judge_graded_count",
            "judge_failed_count",
            "judge_ungraded_count",
            "semantic_unknown_count",
        ):
            _integer(getattr(self, name), f"Intent score binding.{name}")
        if self.judge_graded_count + self.judge_failed_count + self.judge_ungraded_count != self.judge_request_count:
            raise _error("Intent Judge coverage is inconsistent")
        if self.semantic_unknown_count > self.judge_graded_count:
            raise _error("Intent semantic unknown count exceeds graded decisions")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result_digest": self.result_digest,
            "evaluator_revision": self.evaluator_revision,
            "policy_version": self.policy_version,
            "normalization_version": self.normalization_version,
            "status": self.status.value,
            "judge_request_count": self.judge_request_count,
            "judge_graded_count": self.judge_graded_count,
            "judge_failed_count": self.judge_failed_count,
            "judge_ungraded_count": self.judge_ungraded_count,
            "semantic_unknown_count": self.semantic_unknown_count,
        }


@dataclass(frozen=True)
class ReviewScoreBinding:
    result_digest: str
    evaluator_revision: str
    review_policy_version: str
    assignment_policy_version: str
    location_policy_version: str
    evidence_policy_version: str
    status: ReviewEvaluationStatus
    phase: ReviewEvaluationPhase
    judge_request_count: int
    judge_graded_count: int
    judge_failed_count: int
    judge_ungraded_count: int
    judge_pending_count: int
    semantic_unknown_count: int
    finding_count: int
    finding_resolved_count: int

    def __post_init__(self) -> None:
        _digest(self.result_digest, "Review score binding.result_digest")
        for name in (
            "evaluator_revision",
            "review_policy_version",
            "assignment_policy_version",
            "location_policy_version",
            "evidence_policy_version",
        ):
            _id(getattr(self, name), f"Review score binding.{name}")
        if type(self.status) is not ReviewEvaluationStatus or self.status is ReviewEvaluationStatus.PENDING_JUDGE:
            raise _error("Review score binding requires a terminal evaluator status")
        if type(self.phase) is not ReviewEvaluationPhase:
            raise _error("Review score binding phase is invalid")
        for name in (
            "judge_request_count",
            "judge_graded_count",
            "judge_failed_count",
            "judge_ungraded_count",
            "judge_pending_count",
            "semantic_unknown_count",
            "finding_count",
            "finding_resolved_count",
        ):
            _integer(getattr(self, name), f"Review score binding.{name}")
        if self.judge_pending_count != 0:
            raise _error("terminal Review score cannot retain pending Judge work")
        if self.judge_graded_count + self.judge_failed_count + self.judge_ungraded_count != self.judge_request_count:
            raise _error("Review Judge coverage is inconsistent")
        if self.semantic_unknown_count > self.judge_graded_count:
            raise _error("Review semantic unknown count exceeds graded decisions")
        if self.finding_resolved_count > self.finding_count:
            raise _error("Review resolved Finding count exceeds Finding count")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result_digest": self.result_digest,
            "evaluator_revision": self.evaluator_revision,
            "review_policy_version": self.review_policy_version,
            "assignment_policy_version": self.assignment_policy_version,
            "location_policy_version": self.location_policy_version,
            "evidence_policy_version": self.evidence_policy_version,
            "status": self.status.value,
            "phase": self.phase.value,
            "judge_request_count": self.judge_request_count,
            "judge_graded_count": self.judge_graded_count,
            "judge_failed_count": self.judge_failed_count,
            "judge_ungraded_count": self.judge_ungraded_count,
            "judge_pending_count": self.judge_pending_count,
            "semantic_unknown_count": self.semantic_unknown_count,
            "finding_count": self.finding_count,
            "finding_resolved_count": self.finding_resolved_count,
        }


def _metric_authority_policy_snapshot(version: str) -> Dict[str, str]:
    """Return the canonical semantics bound by an execution policy version."""

    supported_version = _id(version, "metric authority policy.version")
    if supported_version != DEFAULT_METRIC_AUTHORITY_POLICY_VERSION:
        raise UnsupportedProtocolVersionError(
            expected=DEFAULT_METRIC_AUTHORITY_POLICY_VERSION,
            actual=supported_version,
        )
    return {
        "version": supported_version,
        "severity_eligibility": "required-and-severity-scorable",
        "location_precision_eligibility": "all-expected-and-location-scorable",
        "location_recall_eligibility": "required-and-location-scorable",
    }


def _metric_authority_policy_digest(version: str) -> str:
    return canonical_sha256(_metric_authority_policy_snapshot(version))


@dataclass(frozen=True)
class MetricAuthorityProfile:
    """Canonical set of authority combinations present in one EvalCase."""

    authorities: Tuple[MetricAuthority, ...]

    def __post_init__(self) -> None:
        values = tuple(self.authorities)
        if any(type(item) is not MetricAuthority for item in values):
            raise _error("metric authority profile contains an invalid authority")
        canonical = {
            canonical_json(item.to_dict()): MetricAuthority.from_dict(item.to_dict())
            for item in values
        }
        if len(canonical) > MAX_AUTHORITY_COMBINATIONS:
            raise _error("metric authority profile exceeds its combination limit")
        ordered = tuple(canonical[key] for key in sorted(canonical))
        object.__setattr__(self, "authorities", ordered)
        _canonical_payload(self.to_dict(), "metric authority profile")

    @classmethod
    def from_eval_case(cls, eval_case: EvalCase) -> "MetricAuthorityProfile":
        if type(eval_case) is not EvalCase:
            raise _error("metric authority profile requires an EvalCase")
        return cls(
            tuple(
                item.metric_authority
                for item in eval_case.review_truth.expected_findings
            )
        )

    @classmethod
    def from_dict(cls, value: Any) -> "MetricAuthorityProfile":
        payload = _strict_object(
            value,
            ("authorities",),
            "metric authority profile",
        )
        return cls(
            tuple(
                MetricAuthority.from_dict(item)
                for item in _array(
                    payload["authorities"],
                    "metric authority profile.authorities",
                    MAX_AUTHORITY_COMBINATIONS,
                )
            )
        )

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {"authorities": [item.to_dict() for item in self.authorities]}


@dataclass(frozen=True)
class MetricAuthorityCoverage:
    """Case-level eligible/excluded truth population for authority metrics."""

    expected_truth_count: int
    required_expected_truth_count: int
    severity_eligible_required_truth_count: int
    severity_excluded_required_truth_count: int
    location_precision_eligible_truth_count: int
    location_precision_excluded_truth_count: int
    location_recall_eligible_required_truth_count: int
    location_recall_excluded_required_truth_count: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _integer(getattr(self, name), f"metric authority coverage.{name}")
        if self.required_expected_truth_count > self.expected_truth_count:
            raise _error("required authority truth count exceeds all expected truths")
        if (
            self.severity_eligible_required_truth_count
            + self.severity_excluded_required_truth_count
            != self.required_expected_truth_count
        ):
            raise _error("severity authority coverage does not cover required truths")
        if (
            self.location_precision_eligible_truth_count
            + self.location_precision_excluded_truth_count
            != self.expected_truth_count
        ):
            raise _error("location precision authority coverage does not cover truths")
        if (
            self.location_recall_eligible_required_truth_count
            + self.location_recall_excluded_required_truth_count
            != self.required_expected_truth_count
        ):
            raise _error("location recall authority coverage does not cover required truths")

    @classmethod
    def from_eval_case(cls, eval_case: EvalCase) -> "MetricAuthorityCoverage":
        if type(eval_case) is not EvalCase:
            raise _error("metric authority coverage requires an EvalCase")
        expected = tuple(eval_case.review_truth.expected_findings)
        required = tuple(item for item in expected if item.required)
        severity_eligible = sum(
            item.metric_authority.severity_scorable for item in required
        )
        precision_eligible = sum(
            item.metric_authority.location_scorable for item in expected
        )
        recall_eligible = sum(
            item.metric_authority.location_scorable for item in required
        )
        return cls(
            expected_truth_count=len(expected),
            required_expected_truth_count=len(required),
            severity_eligible_required_truth_count=severity_eligible,
            severity_excluded_required_truth_count=len(required) - severity_eligible,
            location_precision_eligible_truth_count=precision_eligible,
            location_precision_excluded_truth_count=len(expected) - precision_eligible,
            location_recall_eligible_required_truth_count=recall_eligible,
            location_recall_excluded_required_truth_count=(
                len(required) - recall_eligible
            ),
        )

    @classmethod
    def aggregate(
        cls,
        values: Sequence["MetricAuthorityCoverage"],
    ) -> "MetricAuthorityCoverage":
        items = tuple(values)
        if not items or any(type(item) is not cls for item in items):
            raise _error("authority coverage aggregation requires typed Case coverage")
        return cls(
            **{
                name: sum(getattr(item, name) for item in items)
                for name in cls.__dataclass_fields__
            }
        )

    @classmethod
    def from_dict(cls, value: Any) -> "MetricAuthorityCoverage":
        field_names = tuple(cls.__dataclass_fields__)
        payload = _strict_object(value, field_names, "metric authority coverage")
        return cls(**{name: payload[name] for name in field_names})

    def to_dict(self) -> Dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class ScoreCompatibilityKey:
    run_id: str
    evaluation_id: str
    suite_id: str
    suite_version: str
    manifest_digest: str
    case_snapshot_id: str
    case_snapshot_digest: str
    trial_count: int
    protocol_id: str
    target_kind: ReviewTargetKind
    wire_contract: WireContractV2
    wire_contract_digest: str
    adapter_capabilities_digest: str
    isolation_profile: str
    truth_completeness: TruthCompleteness
    novel_finding_policy: NovelFindingPolicy
    metric_authority_profile: MetricAuthorityProfile
    metric_authority_profile_digest: str
    metric_authority_policy_version: str
    metric_authority_policy_digest: str
    agent_config_digest: str
    clarification_matcher_config_digest: str
    evaluator_execution_digest: str
    metrics_policy_digest: str
    metrics_policy: MetricsPolicy
    intent_evaluator_revision: str
    review_evaluator_revision: str
    intent_policy_version: str
    intent_normalization_version: str
    review_policy_version: str
    assignment_policy_version: str
    location_policy_version: str
    evidence_policy_version: str

    def __post_init__(self) -> None:
        for name in (
            "run_id",
            "evaluation_id",
            "suite_id",
            "suite_version",
            "case_snapshot_id",
            "protocol_id",
            "isolation_profile",
            "metric_authority_policy_version",
            "intent_evaluator_revision",
            "review_evaluator_revision",
            "intent_policy_version",
            "intent_normalization_version",
            "review_policy_version",
            "assignment_policy_version",
            "location_policy_version",
            "evidence_policy_version",
        ):
            _id(getattr(self, name), f"compatibility.{name}")
        _integer(self.trial_count, "compatibility.trial_count", minimum=1)
        for name in (
            "manifest_digest",
            "case_snapshot_digest",
            "wire_contract_digest",
            "adapter_capabilities_digest",
            "metric_authority_profile_digest",
            "metric_authority_policy_digest",
            "agent_config_digest",
            "clarification_matcher_config_digest",
            "evaluator_execution_digest",
            "metrics_policy_digest",
        ):
            _digest(getattr(self, name), f"compatibility.{name}")
        if type(self.target_kind) is not ReviewTargetKind:
            raise _error("compatibility target kind is invalid")
        if type(self.wire_contract) is not WireContractV2:
            raise _error("compatibility wire contract snapshot is invalid")
        if self.target_kind is not self.wire_contract.review_target_kind:
            raise _error("compatibility target kind differs from wire contract")
        if canonical_sha256(self.wire_contract.to_dict()) != self.wire_contract_digest:
            raise _error("compatibility wire contract snapshot/digest differ")
        if (
            type(self.metric_authority_profile) is not MetricAuthorityProfile
            or self.metric_authority_profile.digest
            != self.metric_authority_profile_digest
        ):
            raise _error("compatibility authority profile snapshot/digest differ")
        if (
            _metric_authority_policy_digest(
                self.metric_authority_policy_version
            )
            != self.metric_authority_policy_digest
        ):
            raise _error("compatibility authority policy version/digest differ")
        if (
            type(self.metrics_policy) is not MetricsPolicy
            or self.metrics_policy.digest != self.metrics_policy_digest
        ):
            raise _error("compatibility metrics policy snapshot/digest differ")
        if type(self.truth_completeness) is not TruthCompleteness:
            raise _error("compatibility truth completeness is invalid")
        if type(self.novel_finding_policy) is not NovelFindingPolicy:
            raise _error("compatibility novel Finding policy is invalid")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "evaluation_id": self.evaluation_id,
            "suite_id": self.suite_id,
            "suite_version": self.suite_version,
            "manifest_digest": self.manifest_digest,
            "case_snapshot_id": self.case_snapshot_id,
            "case_snapshot_digest": self.case_snapshot_digest,
            "trial_count": self.trial_count,
            "protocol_id": self.protocol_id,
            "target_kind": self.target_kind.value,
            "wire_contract": self.wire_contract.to_dict(),
            "wire_contract_digest": self.wire_contract_digest,
            "adapter_capabilities_digest": self.adapter_capabilities_digest,
            "isolation_profile": self.isolation_profile,
            "truth_completeness": self.truth_completeness.value,
            "novel_finding_policy": self.novel_finding_policy.value,
            "metric_authority_profile": self.metric_authority_profile.to_dict(),
            "metric_authority_profile_digest": self.metric_authority_profile_digest,
            "metric_authority_policy_version": self.metric_authority_policy_version,
            "metric_authority_policy_digest": self.metric_authority_policy_digest,
            "agent_config_digest": self.agent_config_digest,
            "clarification_matcher_config_digest": self.clarification_matcher_config_digest,
            "evaluator_execution_digest": self.evaluator_execution_digest,
            "metrics_policy_digest": self.metrics_policy_digest,
            "metrics_policy": self.metrics_policy.to_dict(),
            "intent_evaluator_revision": self.intent_evaluator_revision,
            "review_evaluator_revision": self.review_evaluator_revision,
            "intent_policy_version": self.intent_policy_version,
            "intent_normalization_version": self.intent_normalization_version,
            "review_policy_version": self.review_policy_version,
            "assignment_policy_version": self.assignment_policy_version,
            "location_policy_version": self.location_policy_version,
            "evidence_policy_version": self.evidence_policy_version,
        }


@dataclass(frozen=True, init=False)
class TrialScore:
    schema_version: str
    score_id: str
    aggregator_revision: str
    evaluation_revision: str
    compatibility: ScoreCompatibilityKey
    authority_coverage: MetricAuthorityCoverage
    task_id: str
    case_version: int
    trial_index: int
    trial_id: str
    canonical_case_digest: str
    eval_input_digest: str
    dimensions: Tuple[CaseDimension, ...]
    submission_digest: str
    submission_status: SubmissionStatus
    failure_code: Optional[FailureCode]
    failure_retryable: Optional[bool]
    intent_binding: Optional[IntentScoreBinding]
    review_binding: Optional[ReviewScoreBinding]
    contributions: Tuple[MetricContribution, ...]
    usage: SubmissionUsage
    trace_ref: Optional[TraceRef]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("TrialScore must be created by TrialScorer.score or source-bound hydration")

    def __post_init__(self) -> None:
        if self.schema_version != TRIAL_SCORE_SCHEMA_VERSION:
            raise _error("unsupported Trial score schema version")
        if self.aggregator_revision != METRICS_AGGREGATOR_REVISION:
            raise _error("unsupported metrics aggregator revision")
        _id(self.evaluation_revision, "Trial score.evaluation_revision")
        if type(self.compatibility) is not ScoreCompatibilityKey:
            raise _error("Trial score requires a compatibility key")
        if type(self.authority_coverage) is not MetricAuthorityCoverage:
            raise _error("Trial score authority coverage is invalid")
        _id(self.task_id, "Trial score.task_id")
        _integer(self.case_version, "Trial score.case_version", minimum=1)
        _integer(self.trial_index, "Trial score.trial_index", minimum=1)
        _id(self.trial_id, "Trial score.trial_id")
        _digest(self.canonical_case_digest, "Trial score.canonical_case_digest")
        _digest(self.eval_input_digest, "Trial score.eval_input_digest")
        dimensions = tuple(self.dimensions)
        if any(type(item) is not CaseDimension for item in dimensions):
            raise _error("Trial score dimensions contain an invalid item")
        if dimensions != tuple(sorted(dimensions, key=lambda item: item.name)):
            raise _error("Trial score dimensions are not canonically ordered")
        if len({item.name for item in dimensions}) != len(dimensions):
            raise _error("Trial score dimensions contain duplicate names")
        object.__setattr__(self, "dimensions", dimensions)
        _digest(self.submission_digest, "Trial score.submission_digest")
        if type(self.submission_status) is not SubmissionStatus:
            raise _error("Trial score submission status is invalid")
        if self.submission_status is SubmissionStatus.COMPLETED:
            if self.failure_code is not None or self.failure_retryable is not None:
                raise _error("completed Trial score cannot contain failure metadata")
        else:
            if type(self.failure_code) is not FailureCode or type(self.failure_retryable) is not bool:
                raise _error("failed Trial score requires typed failure metadata")
        if self.intent_binding is not None and type(self.intent_binding) is not IntentScoreBinding:
            raise _error("Trial score Intent binding is invalid")
        if self.review_binding is not None and type(self.review_binding) is not ReviewScoreBinding:
            raise _error("Trial score Review binding is invalid")
        contributions = tuple(self.contributions)
        if any(type(item) is not MetricContribution for item in contributions):
            raise _error("Trial score contributions contain an invalid item")
        if contributions != tuple(sorted(contributions, key=lambda item: item.metric.value)):
            raise _error("Trial score contributions are not canonically ordered")
        if {item.metric for item in contributions} != set(_CONTRIBUTION_METRICS):
            raise _error("Trial score contributions do not cover the v1 metric set")
        object.__setattr__(self, "contributions", contributions)
        if type(self.usage) is not SubmissionUsage:
            raise _error("Trial score usage is invalid")
        if self.trace_ref is not None and type(self.trace_ref) is not TraceRef:
            raise _error("Trial score trace_ref is invalid")
        expected_id = stable_id("trial-score-v1", self._identity_dict())
        if self.score_id != expected_id:
            raise _error("Trial score ID is not canonical")
        _canonical_payload(self.to_dict(), "Trial score")

    def _identity_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "aggregator_revision": self.aggregator_revision,
            "evaluation_revision": self.evaluation_revision,
            "compatibility": self.compatibility.to_dict(),
            "authority_coverage": self.authority_coverage.to_dict(),
            "task_id": self.task_id,
            "case_version": self.case_version,
            "trial_index": self.trial_index,
            "trial_id": self.trial_id,
            "canonical_case_digest": self.canonical_case_digest,
            "eval_input_digest": self.eval_input_digest,
            "dimensions": [item.to_dict() for item in self.dimensions],
            "submission_digest": self.submission_digest,
            "submission_status": self.submission_status.value,
            "failure_code": None if self.failure_code is None else self.failure_code.value,
            "failure_retryable": self.failure_retryable,
            "intent_binding": None if self.intent_binding is None else self.intent_binding.to_dict(),
            "review_binding": None if self.review_binding is None else self.review_binding.to_dict(),
            "contributions": [item.to_dict() for item in self.contributions],
            "usage": self.usage.to_dict(),
            "trace_ref": None if self.trace_ref is None else self.trace_ref.to_dict(),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {**self._identity_dict(), "score_id": self.score_id}

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def digest(self) -> str:
        return canonical_sha256(self.to_dict())

    def contribution(self, metric: CoreMetric) -> MetricContribution:
        return next(item for item in self.contributions if item.metric is metric)

    @classmethod
    def from_dict(
        cls,
        value: Any,
        *,
        scorer: "TrialScorer",
        run_config: EvalRunConfig,
        evaluator_execution: EvaluatorExecutionConfig,
        evaluation_revision: str,
        eval_case: EvalCase,
        submission: EvalSubmission,
        trial_index: int,
        intent_result: Optional[IntentEvaluationResult],
        review_result: Optional[ReviewEvaluationResult],
    ) -> "TrialScore":
        if type(value) is not dict or type(scorer) is not TrialScorer:
            raise _error(
                "Trial score hydration requires an object and TrialScorer"
            )
        trusted_scorer = TrialScorer._canonical_clone(scorer)
        replayed = TrialScorer.score(
            trusted_scorer,
            run_config=run_config,
            evaluator_execution=evaluator_execution,
            evaluation_revision=evaluation_revision,
            eval_case=eval_case,
            submission=submission,
            trial_index=trial_index,
            intent_result=intent_result,
            review_result=review_result,
        )
        if _canonical_payload(value, "Trial score payload") != canonical_json_bytes(replayed.to_dict()):
            raise _error("persisted Trial score differs from source-bound replay")
        return replayed

    @classmethod
    def from_json(cls, data: Any, **sources: Any) -> "TrialScore":
        try:
            payload = _strict_json_loads(data, MAX_SCORE_BYTES, "Trial score JSON")
        except (SchemaError, ValueError) as exc:
            raise _error(str(exc)) from exc
        return cls.from_dict(payload, **sources)


@dataclass(frozen=True)
class ScoreRef:
    score_id: str
    score_digest: str
    task_id: str
    trial_id: Optional[str]

    def __post_init__(self) -> None:
        _id(self.score_id, "score ref.score_id")
        _digest(self.score_digest, "score ref.score_digest")
        _id(self.task_id, "score ref.task_id")
        if self.trial_id is not None:
            _id(self.trial_id, "score ref.trial_id")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score_id": self.score_id,
            "score_digest": self.score_digest,
            "task_id": self.task_id,
            "trial_id": self.trial_id,
        }


@dataclass(frozen=True)
class CountBreakdown:
    key: str
    count: int

    def __post_init__(self) -> None:
        _id(self.key, "count breakdown.key")
        _integer(self.count, "count breakdown.count")

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "count": self.count}


@dataclass(frozen=True, init=False)
class CaseScore:
    schema_version: str
    score_id: str
    aggregator_revision: str
    compatibility: ScoreCompatibilityKey
    authority_coverage: MetricAuthorityCoverage
    task_id: str
    case_version: int
    canonical_case_digest: str
    dimensions: Tuple[CaseDimension, ...]
    planned_trial_count: int
    terminal_trial_count: int
    intent_scored_trial_count: int
    review_scored_trial_count: int
    fully_scored_trial_count: int
    trial_scores: Tuple[ScoreRef, ...]
    metrics: Tuple[MetricAggregate, ...]
    usage: UsageAggregate
    submission_status_breakdown: Tuple[CountBreakdown, ...]
    failure_code_breakdown: Tuple[CountBreakdown, ...]
    failed_trial_ids: Tuple[str, ...]
    ungraded_trial_ids: Tuple[str, ...]
    critical_high_miss_trial_ids: Tuple[str, ...]
    fabricated_trial_ids: Tuple[str, ...]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("CaseScore must be created by MetricsAggregator")

    def __post_init__(self) -> None:
        if self.schema_version != CASE_SCORE_SCHEMA_VERSION or self.aggregator_revision != METRICS_AGGREGATOR_REVISION:
            raise _error("unsupported Case score schema/revision")
        if type(self.compatibility) is not ScoreCompatibilityKey:
            raise _error("Case score compatibility key is invalid")
        if type(self.authority_coverage) is not MetricAuthorityCoverage:
            raise _error("Case score authority coverage is invalid")
        _id(self.task_id, "Case score.task_id")
        _integer(self.case_version, "Case score.case_version", minimum=1)
        _digest(self.canonical_case_digest, "Case score.canonical_case_digest")
        for name in (
            "planned_trial_count",
            "terminal_trial_count",
            "intent_scored_trial_count",
            "review_scored_trial_count",
            "fully_scored_trial_count",
        ):
            _integer(getattr(self, name), f"Case score.{name}")
        if (
            self.terminal_trial_count > self.planned_trial_count
            or self.intent_scored_trial_count > self.terminal_trial_count
            or self.review_scored_trial_count > self.terminal_trial_count
            or self.fully_scored_trial_count
            > min(self.intent_scored_trial_count, self.review_scored_trial_count)
        ):
            raise _error("Case score Trial coverage is inconsistent")
        self._validate_collections()
        if type(self.usage) is not UsageAggregate:
            raise _error("Case score usage is invalid")
        expected_id = stable_id("case-score-v1", self._identity_dict())
        if self.score_id != expected_id:
            raise _error("Case score ID is not canonical")
        _canonical_payload(self.to_dict(), "Case score")

    def _validate_collections(self) -> None:
        dimensions = tuple(self.dimensions)
        if any(type(item) is not CaseDimension for item in dimensions) or dimensions != tuple(sorted(dimensions, key=lambda item: item.name)):
            raise _error("Case score dimensions are invalid")
        object.__setattr__(self, "dimensions", dimensions)
        refs = tuple(self.trial_scores)
        if any(type(item) is not ScoreRef or item.trial_id is None for item in refs):
            raise _error("Case score Trial refs are invalid")
        if refs != tuple(sorted(refs, key=lambda item: item.trial_id or "")) or len({item.trial_id for item in refs}) != len(refs):
            raise _error("Case score Trial refs are not unique/canonical")
        object.__setattr__(self, "trial_scores", refs)
        metrics = tuple(self.metrics)
        if any(type(item) is not MetricAggregate for item in metrics) or metrics != tuple(sorted(metrics, key=lambda item: item.metric.value)) or {item.metric for item in metrics} != set(CoreMetric):
            raise _error("Case score metrics do not cover the v1 metric set")
        object.__setattr__(self, "metrics", metrics)
        for name in ("submission_status_breakdown", "failure_code_breakdown"):
            values = tuple(getattr(self, name))
            if any(type(item) is not CountBreakdown for item in values) or values != tuple(sorted(values, key=lambda item: item.key)):
                raise _error(f"Case score {name} is invalid")
            object.__setattr__(self, name, values)
        for name in ("failed_trial_ids", "ungraded_trial_ids", "critical_high_miss_trial_ids", "fabricated_trial_ids"):
            values = tuple(_id(item, f"Case score.{name} item") for item in getattr(self, name))
            if values != tuple(sorted(set(values))):
                raise _error(f"Case score {name} is not unique/canonical")
            object.__setattr__(self, name, values)

    def _identity_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "aggregator_revision": self.aggregator_revision,
            "compatibility": self.compatibility.to_dict(),
            "authority_coverage": self.authority_coverage.to_dict(),
            "task_id": self.task_id,
            "case_version": self.case_version,
            "canonical_case_digest": self.canonical_case_digest,
            "dimensions": [item.to_dict() for item in self.dimensions],
            "planned_trial_count": self.planned_trial_count,
            "terminal_trial_count": self.terminal_trial_count,
            "intent_scored_trial_count": self.intent_scored_trial_count,
            "review_scored_trial_count": self.review_scored_trial_count,
            "fully_scored_trial_count": self.fully_scored_trial_count,
            "trial_scores": [item.to_dict() for item in self.trial_scores],
            "metrics": [item.to_dict() for item in self.metrics],
            "usage": self.usage.to_dict(),
            "submission_status_breakdown": [item.to_dict() for item in self.submission_status_breakdown],
            "failure_code_breakdown": [item.to_dict() for item in self.failure_code_breakdown],
            "failed_trial_ids": list(self.failed_trial_ids),
            "ungraded_trial_ids": list(self.ungraded_trial_ids),
            "critical_high_miss_trial_ids": list(self.critical_high_miss_trial_ids),
            "fabricated_trial_ids": list(self.fabricated_trial_ids),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {**self._identity_dict(), "score_id": self.score_id}

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def digest(self) -> str:
        return canonical_sha256(self.to_dict())

    def metric(self, metric: CoreMetric) -> MetricAggregate:
        return next(item for item in self.metrics if item.metric is metric)

    @classmethod
    def from_dict(
        cls,
        value: Any,
        *,
        aggregator: "MetricsAggregator",
        trials: Sequence[TrialScore],
        planned_trial_count: Optional[int] = None,
    ) -> "CaseScore":
        if type(value) is not dict or type(aggregator) is not MetricsAggregator:
            raise _error("Case score hydration requires an object and MetricsAggregator")
        replayed = MetricsAggregator.aggregate_case(
            aggregator,
            trials,
            planned_trial_count=planned_trial_count,
        )
        if _canonical_payload(value, "Case score payload") != canonical_json_bytes(
            replayed.to_dict()
        ):
            raise _error("persisted Case score differs from source-bound replay")
        return replayed

    @classmethod
    def from_json(cls, data: Any, **sources: Any) -> "CaseScore":
        try:
            payload = _strict_json_loads(data, MAX_SCORE_BYTES, "Case score JSON")
        except (SchemaError, ValueError) as exc:
            raise _error(str(exc)) from exc
        return cls.from_dict(payload, **sources)


@dataclass(frozen=True, init=False)
class AggregateScore:
    schema_version: str
    score_id: str
    aggregator_revision: str
    compatibility: ScoreCompatibilityKey
    authority_coverage: MetricAuthorityCoverage
    group_dimensions: Tuple[CaseDimension, ...]
    case_count: int
    planned_trial_count: int
    terminal_trial_count: int
    intent_scored_trial_count: int
    review_scored_trial_count: int
    fully_scored_trial_count: int
    case_scores: Tuple[ScoreRef, ...]
    metrics: Tuple[MetricAggregate, ...]
    usage: UsageAggregate
    submission_status_breakdown: Tuple[CountBreakdown, ...]
    failure_code_breakdown: Tuple[CountBreakdown, ...]
    failed_trial_ids: Tuple[str, ...]
    ungraded_trial_ids: Tuple[str, ...]
    critical_high_miss_trial_ids: Tuple[str, ...]
    fabricated_trial_ids: Tuple[str, ...]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("AggregateScore must be created by MetricsAggregator")

    def __post_init__(self) -> None:
        if self.schema_version != AGGREGATE_SCORE_SCHEMA_VERSION or self.aggregator_revision != METRICS_AGGREGATOR_REVISION:
            raise _error("unsupported Aggregate score schema/revision")
        if type(self.compatibility) is not ScoreCompatibilityKey:
            raise _error("Aggregate score compatibility key is invalid")
        if type(self.authority_coverage) is not MetricAuthorityCoverage:
            raise _error("Aggregate score authority coverage is invalid")
        for name in (
            "case_count",
            "planned_trial_count",
            "terminal_trial_count",
            "intent_scored_trial_count",
            "review_scored_trial_count",
            "fully_scored_trial_count",
        ):
            _integer(getattr(self, name), f"Aggregate score.{name}")
        if (
            self.terminal_trial_count > self.planned_trial_count
            or self.intent_scored_trial_count > self.terminal_trial_count
            or self.review_scored_trial_count > self.terminal_trial_count
            or self.fully_scored_trial_count
            > min(self.intent_scored_trial_count, self.review_scored_trial_count)
        ):
            raise _error("Aggregate Trial coverage is inconsistent")
        dimensions = tuple(self.group_dimensions)
        if any(type(item) is not CaseDimension for item in dimensions) or dimensions != tuple(sorted(dimensions, key=lambda item: item.name)):
            raise _error("Aggregate group dimensions are invalid")
        object.__setattr__(self, "group_dimensions", dimensions)
        refs = tuple(self.case_scores)
        if any(type(item) is not ScoreRef or item.trial_id is not None for item in refs):
            raise _error("Aggregate Case refs are invalid")
        if refs != tuple(sorted(refs, key=lambda item: item.task_id)) or len({item.task_id for item in refs}) != len(refs):
            raise _error("Aggregate Case refs are not unique/canonical")
        object.__setattr__(self, "case_scores", refs)
        if len(refs) != self.case_count:
            raise _error("Aggregate case count differs from source refs")
        metrics = tuple(self.metrics)
        if any(type(item) is not MetricAggregate for item in metrics) or metrics != tuple(sorted(metrics, key=lambda item: item.metric.value)) or {item.metric for item in metrics} != set(CoreMetric):
            raise _error("Aggregate metrics do not cover the v1 metric set")
        object.__setattr__(self, "metrics", metrics)
        if type(self.usage) is not UsageAggregate:
            raise _error("Aggregate usage is invalid")
        for name in ("submission_status_breakdown", "failure_code_breakdown"):
            values = tuple(getattr(self, name))
            if any(type(item) is not CountBreakdown for item in values) or values != tuple(sorted(values, key=lambda item: item.key)):
                raise _error(f"Aggregate {name} is invalid")
            object.__setattr__(self, name, values)
        for name in ("failed_trial_ids", "ungraded_trial_ids", "critical_high_miss_trial_ids", "fabricated_trial_ids"):
            values = tuple(_id(item, f"Aggregate score.{name} item") for item in getattr(self, name))
            if values != tuple(sorted(set(values))):
                raise _error(f"Aggregate {name} is not unique/canonical")
            object.__setattr__(self, name, values)
        expected_id = stable_id("aggregate-score-v1", self._identity_dict())
        if self.score_id != expected_id:
            raise _error("Aggregate score ID is not canonical")
        _canonical_payload(self.to_dict(), "Aggregate score")

    def _identity_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "aggregator_revision": self.aggregator_revision,
            "compatibility": self.compatibility.to_dict(),
            "authority_coverage": self.authority_coverage.to_dict(),
            "group_dimensions": [item.to_dict() for item in self.group_dimensions],
            "case_count": self.case_count,
            "planned_trial_count": self.planned_trial_count,
            "terminal_trial_count": self.terminal_trial_count,
            "intent_scored_trial_count": self.intent_scored_trial_count,
            "review_scored_trial_count": self.review_scored_trial_count,
            "fully_scored_trial_count": self.fully_scored_trial_count,
            "case_scores": [item.to_dict() for item in self.case_scores],
            "metrics": [item.to_dict() for item in self.metrics],
            "usage": self.usage.to_dict(),
            "submission_status_breakdown": [item.to_dict() for item in self.submission_status_breakdown],
            "failure_code_breakdown": [item.to_dict() for item in self.failure_code_breakdown],
            "failed_trial_ids": list(self.failed_trial_ids),
            "ungraded_trial_ids": list(self.ungraded_trial_ids),
            "critical_high_miss_trial_ids": list(self.critical_high_miss_trial_ids),
            "fabricated_trial_ids": list(self.fabricated_trial_ids),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {**self._identity_dict(), "score_id": self.score_id}

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def digest(self) -> str:
        return canonical_sha256(self.to_dict())

    def metric(self, metric: CoreMetric) -> MetricAggregate:
        return next(item for item in self.metrics if item.metric is metric)

    @classmethod
    def from_dict(
        cls,
        value: Any,
        *,
        aggregator: "MetricsAggregator",
        case_scores: Sequence[CaseScore],
        source_trials: Sequence[TrialScore],
        group_dimensions: Sequence[CaseDimension] = (),
    ) -> "AggregateScore":
        if type(value) is not dict or type(aggregator) is not MetricsAggregator:
            raise _error(
                "Aggregate score hydration requires an object and MetricsAggregator"
            )
        replayed = MetricsAggregator.aggregate_cases(
            aggregator,
            case_scores,
            group_dimensions=group_dimensions,
            source_trials=source_trials,
        )
        if _canonical_payload(
            value, "Aggregate score payload"
        ) != canonical_json_bytes(replayed.to_dict()):
            raise _error(
                "persisted Aggregate score differs from source-bound replay"
            )
        return replayed

    @classmethod
    def from_json(cls, data: Any, **sources: Any) -> "AggregateScore":
        try:
            payload = _strict_json_loads(
                data, MAX_SCORE_BYTES, "Aggregate score JSON"
            )
        except (SchemaError, ValueError) as exc:
            raise _error(str(exc)) from exc
        return cls.from_dict(payload, **sources)


class TrialScorer:
    """Create one source-bound TrialScore from immutable evaluator outputs."""

    __slots__ = (
        "metrics_policy",
        "intent_evaluator_revision",
        "review_evaluator_revision",
        "intent_policy_version",
        "intent_normalization_version",
        "review_policy_version",
        "assignment_policy_version",
        "location_policy_version",
        "evidence_policy_version",
        "_sealed",
    )

    def __init__(
        self,
        metrics_policy: MetricsPolicy = DEFAULT_METRICS_POLICY,
        *,
        intent_evaluator_revision: str = INTENT_EVALUATOR_REVISION,
        review_evaluator_revision: str = REVIEW_EVALUATOR_REVISION,
        intent_policy_version: str = ASSIGNMENT_POLICY_VERSION,
        intent_normalization_version: str = INTENT_NORMALIZATION_POLICY_VERSION,
        review_policy_version: str = REVIEW_MATCH_POLICY_VERSION,
        assignment_policy_version: str = ASSIGNMENT_POLICY_VERSION,
        location_policy_version: str = LOCATION_MATCH_POLICY_VERSION,
        evidence_policy_version: str = EVIDENCE_INTEGRITY_POLICY_VERSION,
    ) -> None:
        metrics_policy = _canonical_metrics_policy(metrics_policy)
        object.__setattr__(self, "metrics_policy", metrics_policy)
        object.__setattr__(self, "intent_evaluator_revision", _id(
            intent_evaluator_revision,
            "TrialScorer.intent_evaluator_revision",
        ))
        object.__setattr__(self, "review_evaluator_revision", _id(
            review_evaluator_revision,
            "TrialScorer.review_evaluator_revision",
        ))
        object.__setattr__(self, "intent_policy_version", _id(intent_policy_version, "TrialScorer.intent_policy_version"))
        object.__setattr__(self, "intent_normalization_version", _id(intent_normalization_version, "TrialScorer.intent_normalization_version"))
        object.__setattr__(self, "review_policy_version", _id(review_policy_version, "TrialScorer.review_policy_version"))
        object.__setattr__(self, "assignment_policy_version", _id(assignment_policy_version, "TrialScorer.assignment_policy_version"))
        object.__setattr__(self, "location_policy_version", _id(location_policy_version, "TrialScorer.location_policy_version"))
        object.__setattr__(self, "evidence_policy_version", _id(evidence_policy_version, "TrialScorer.evidence_policy_version"))
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("TrialScorer configuration is immutable")
        object.__setattr__(self, name, value)

    @classmethod
    def _canonical_clone(cls, value: Any) -> "TrialScorer":
        if type(value) is not cls:
            raise _error("TrialScorer snapshot has an invalid type")
        try:
            return cls(
                _canonical_metrics_policy(value.metrics_policy),
                intent_evaluator_revision=value.intent_evaluator_revision,
                review_evaluator_revision=value.review_evaluator_revision,
                intent_policy_version=value.intent_policy_version,
                intent_normalization_version=value.intent_normalization_version,
                review_policy_version=value.review_policy_version,
                assignment_policy_version=value.assignment_policy_version,
                location_policy_version=value.location_policy_version,
                evidence_policy_version=value.evidence_policy_version,
            )
        except (AttributeError, SchemaError, TypeError, ValueError) as exc:
            raise _error(f"TrialScorer snapshot is invalid: {exc}") from exc

    @staticmethod
    def _included(metric: CoreMetric, numerator: int, denominator: int, status: MetricSourceStatus = MetricSourceStatus.GRADED) -> MetricContribution:
        return MetricContribution(metric, _METRIC_KINDS[metric], status, numerator, denominator)

    @staticmethod
    def _excluded(metric: CoreMetric, status: MetricSourceStatus) -> MetricContribution:
        return MetricContribution(metric, _METRIC_KINDS[metric], status, None, None)

    def _intent_contributions(
        self,
        eval_case: EvalCase,
        submission: EvalSubmission,
        result: Optional[IntentEvaluationResult],
    ) -> Tuple[MetricContribution, ...]:
        truth = eval_case.intent_truth
        if result is not None and result.status is IntentEvaluationStatus.GRADED:
            metrics = result.metrics
            if not metrics.scorable:
                raise _error("graded Intent result has unscorable metric inputs")
            values = {
                CoreMetric.INTENT_CLAIM_PRECISION: (metrics.supported_claim_count, metrics.generated_claim_count),
                CoreMetric.INTENT_CLAIM_RECALL: (metrics.required_supported_count, metrics.required_truth_count),
                CoreMetric.INTENT_PARTIALLY_SUPPORTED_RATE: (metrics.partially_supported_claim_count, metrics.generated_claim_count),
                CoreMetric.INTENT_UNSUPPORTED_RATE: (metrics.unsupported_claim_count, metrics.generated_claim_count),
                CoreMetric.INTENT_CONTRADICTED_RATE: (metrics.contradicted_claim_count, metrics.generated_claim_count),
                CoreMetric.INTENT_UNKNOWN_RATE: (metrics.unknown_claim_count, metrics.generated_claim_count),
            }
            contributions = [
                self._included(metric, int(pair[0]), int(pair[1]))
                for metric, pair in values.items()
            ]
            contributions.append(
                self._included(
                    CoreMetric.CLARIFICATION_ACCURACY,
                    int(metrics.clarification_numerator),
                    int(metrics.clarification_denominator),
                )
                if metrics.clarification_numerator is not None
                else self._excluded(CoreMetric.CLARIFICATION_ACCURACY, MetricSourceStatus.MISSING)
            )
            contributions.append(
                self._included(CoreMetric.INTENT_CASE_PASS_RATE, int(bool(metrics.intent_case_pass)), 1)
                if metrics.intent_case_pass is not None
                else self._excluded(CoreMetric.INTENT_CASE_PASS_RATE, MetricSourceStatus.UNGRADED)
            )
            return tuple(contributions)
        if result is not None and result.status is IntentEvaluationStatus.NOT_SCORABLE:
            return tuple(self._excluded(metric, MetricSourceStatus.NOT_SCORABLE) for metric in _INTENT_METRICS)
        if not truth.scorable:
            return tuple(self._excluded(metric, MetricSourceStatus.NOT_SCORABLE) for metric in _INTENT_METRICS)
        if result is not None:
            return tuple(
                self._excluded(metric, MetricSourceStatus.UNGRADED)
                for metric in _INTENT_METRICS
            )
        if submission.intent is not None:
            return tuple(
                self._excluded(metric, MetricSourceStatus.MISSING)
                for metric in _INTENT_METRICS
            )
        failed = submission.status is not SubmissionStatus.COMPLETED
        if failed and self.metrics_policy.failure_outcome_policy is FailureOutcomePolicy.COUNT_AS_MISSED:
            required_count = sum(item.required for item in truth.expected_claims)
            values = {
                CoreMetric.INTENT_CLAIM_RECALL: (0, required_count),
                CoreMetric.INTENT_CASE_PASS_RATE: (0, 1),
            }
            return tuple(
                self._included(
                    metric,
                    values[metric][0],
                    values[metric][1],
                    MetricSourceStatus.FAILURE_AS_MISS,
                )
                if metric in values
                else self._excluded(
                    metric,
                    (
                        MetricSourceStatus.MISSING
                        if metric is CoreMetric.CLARIFICATION_ACCURACY
                        else MetricSourceStatus.FAILURE_EXCLUDED
                    ),
                )
                for metric in _INTENT_METRICS
            )
        status = MetricSourceStatus.FAILURE_EXCLUDED if failed else MetricSourceStatus.UNGRADED
        return tuple(self._excluded(metric, status) for metric in _INTENT_METRICS)

    def _review_contributions(
        self,
        eval_case: EvalCase,
        submission: EvalSubmission,
        result: Optional[ReviewEvaluationResult],
    ) -> Tuple[MetricContribution, ...]:
        truth = eval_case.review_truth
        required_truths = tuple(item for item in truth.expected_findings if item.required)
        severity_eligible_truths = tuple(
            item
            for item in required_truths
            if item.metric_authority.severity_scorable
        )
        precision_location_eligible_truths = tuple(
            item
            for item in truth.expected_findings
            if item.metric_authority.location_scorable
        )
        recall_location_eligible_truths = tuple(
            item
            for item in required_truths
            if item.metric_authority.location_scorable
        )
        severity_eligible_ids = {
            item.truth_id for item in severity_eligible_truths
        }
        precision_location_eligible_ids = {
            item.truth_id for item in precision_location_eligible_truths
        }
        recall_location_eligible_ids = {
            item.truth_id for item in recall_location_eligible_truths
        }
        weighted_denominator = sum(
            self.metrics_policy.severity_weights.weight_for(item.severity)
            for item in severity_eligible_truths
        )
        located_required_denominator = len(recall_location_eligible_truths)
        severe_required = sum(
            item.severity in {FindingSeverity.HIGH, FindingSeverity.CRITICAL}
            for item in severity_eligible_truths
        )
        severity_metrics = {
            CoreMetric.SEVERITY_WEIGHTED_RECALL,
            CoreMetric.CRITICAL_HIGH_MISS_COUNT,
        }

        def excluded_for_authority(
            metric: CoreMetric,
            fallback: MetricSourceStatus,
        ) -> MetricContribution:
            if metric in severity_metrics and not severity_eligible_truths:
                return self._excluded(metric, MetricSourceStatus.NOT_SCORABLE)
            if (
                metric is CoreMetric.LINE_PRECISION
                and not precision_location_eligible_truths
            ):
                return self._excluded(metric, MetricSourceStatus.NOT_SCORABLE)
            if (
                metric is CoreMetric.LINE_RECALL
                and not recall_location_eligible_truths
            ):
                return self._excluded(metric, MetricSourceStatus.NOT_SCORABLE)
            return self._excluded(metric, fallback)

        if result is not None and result.status is ReviewEvaluationStatus.GRADED:
            metrics = result.metrics
            if not metrics.scorable:
                raise _error("graded Review result has unscorable metric inputs")
            expected_by_id = {item.truth_id: item for item in result.expected_truth_findings}
            matched_severity_ids = {
                item.matched_expected_truth_id
                for item in result.finding_outcomes
                if item.matched_expected_truth_id is not None
                and item.matched_expected_truth_id in severity_eligible_ids
            }
            weighted_numerator = sum(
                self.metrics_policy.severity_weights.weight_for(expected_by_id[item].severity)
                for item in matched_severity_ids
            )
            severe_misses = sum(
                item.truth_id in severity_eligible_ids
                and item.severity in {FindingSeverity.HIGH, FindingSeverity.CRITICAL}
                and item.truth_id in result.unmatched_expected_truth_ids
                for item in result.expected_truth_findings
            )
            located_assignment_count = 0
            located_assignment_denominator = 0
            located_required_numerator = 0
            for assignment in result.assignments:
                if assignment.truth_id not in precision_location_eligible_ids:
                    continue
                located_assignment_denominator += 1
                matched = any(
                    record.finding_id == assignment.finding_id
                    and record.truth_kind is ReviewTruthKind.EXPECTED
                    and record.truth_id == assignment.truth_id
                    and record.match.matched
                    for record in result.location_candidates
                )
                if matched:
                    located_assignment_count += 1
                    if assignment.truth_id in recall_location_eligible_ids:
                        located_required_numerator += 1
            plausible = sum(item.issue_judgement is IssueJudgement.PLAUSIBLE for item in result.finding_outcomes)
            finding_count = metrics.generated_finding_count
            values = {
                CoreMetric.ISSUE_PRECISION: (metrics.matched_finding_count, finding_count),
                CoreMetric.ISSUE_RECALL: (metrics.matched_required_truth_count, metrics.required_expected_truth_count),
                CoreMetric.SEVERITY_WEIGHTED_RECALL: (weighted_numerator, weighted_denominator),
                CoreMetric.CRITICAL_HIGH_MISS_COUNT: (severe_misses, 1),
                CoreMetric.FABRICATED_FINDINGS_PER_PR: (metrics.fabricated_finding_count, 1),
                CoreMetric.FABRICATED_RATE: (metrics.fabricated_finding_count, finding_count),
                CoreMetric.PLAUSIBLE_RATE: (plausible, finding_count),
                CoreMetric.REVIEW_UNKNOWN_RATE: (metrics.unknown_finding_count, finding_count),
                CoreMetric.LINE_PRECISION: (located_assignment_count, located_assignment_denominator),
                CoreMetric.LINE_RECALL: (located_required_numerator, located_required_denominator),
                CoreMetric.EVIDENCE_VALIDITY: (metrics.evidence_valid_count, finding_count),
                CoreMetric.EVIDENCE_SUPPORT_RATE: (metrics.evidence_supported_count, finding_count),
                CoreMetric.PUBLISHABLE_FINDING_PRECISION: (metrics.strict_publishable_count, finding_count),
            }
            return tuple(
                excluded_for_authority(metric, MetricSourceStatus.NOT_SCORABLE)
                if (
                    (metric in severity_metrics and not severity_eligible_truths)
                    or (
                        metric is CoreMetric.LINE_PRECISION
                        and not precision_location_eligible_truths
                    )
                    or (
                        metric is CoreMetric.LINE_RECALL
                        and not recall_location_eligible_truths
                    )
                )
                else self._included(metric, *values[metric])
                for metric in _REVIEW_METRICS
            )
        if result is not None:
            return tuple(
                excluded_for_authority(metric, MetricSourceStatus.UNGRADED)
                for metric in _REVIEW_METRICS
            )
        if submission.review is not None:
            return tuple(
                excluded_for_authority(metric, MetricSourceStatus.MISSING)
                for metric in _REVIEW_METRICS
            )
        failed = submission.status is not SubmissionStatus.COMPLETED
        if failed and self.metrics_policy.failure_outcome_policy is FailureOutcomePolicy.COUNT_AS_MISSED:
            missed_values = {
                CoreMetric.ISSUE_RECALL: (0, len(required_truths)),
                CoreMetric.SEVERITY_WEIGHTED_RECALL: (0, weighted_denominator),
                CoreMetric.LINE_RECALL: (0, located_required_denominator),
                CoreMetric.CRITICAL_HIGH_MISS_COUNT: (severe_required, 1),
            }
            contributions = []
            for metric in _REVIEW_METRICS:
                if metric in severity_metrics and not severity_eligible_truths:
                    contributions.append(
                        self._excluded(metric, MetricSourceStatus.NOT_SCORABLE)
                    )
                elif (
                    metric is CoreMetric.LINE_PRECISION
                    and not precision_location_eligible_truths
                ):
                    contributions.append(
                        self._excluded(metric, MetricSourceStatus.NOT_SCORABLE)
                    )
                elif (
                    metric is CoreMetric.LINE_RECALL
                    and not recall_location_eligible_truths
                ):
                    contributions.append(
                        self._excluded(metric, MetricSourceStatus.NOT_SCORABLE)
                    )
                elif metric in missed_values:
                    numerator, denominator = missed_values[metric]
                    contributions.append(self._included(metric, numerator, denominator, MetricSourceStatus.FAILURE_AS_MISS))
                else:
                    contributions.append(self._excluded(metric, MetricSourceStatus.FAILURE_EXCLUDED))
            return tuple(contributions)
        status = MetricSourceStatus.FAILURE_EXCLUDED if failed else MetricSourceStatus.UNGRADED
        return tuple(
            excluded_for_authority(metric, status) for metric in _REVIEW_METRICS
        )

    def _judge_contributions(
        self,
        intent_result: Optional[IntentEvaluationResult],
        review_result: Optional[ReviewEvaluationResult],
    ) -> Tuple[MetricContribution, ...]:
        intent_requests = 0 if intent_result is None else len(intent_result.judge_requests)
        review_requests = 0 if review_result is None else review_result.coverage.judge_request_count
        requests = intent_requests + review_requests
        failed = (0 if intent_result is None else len(intent_result.judge_failures)) + (0 if review_result is None else review_result.coverage.judge_failed_count)
        ungraded = (0 if intent_result is None else len(intent_result.judge_ungraded)) + (0 if review_result is None else review_result.coverage.judge_ungraded_count)
        semantic_unknown = (
            0
            if intent_result is None
            else sum(item.relation is IntentJudgeRelation.UNKNOWN for item in intent_result.judge_decisions)
        ) + (0 if review_result is None else review_result.coverage.semantic_unknown_count)
        return (
            self._included(CoreMetric.JUDGE_FAILURE_RATE, failed, requests),
            self._included(CoreMetric.JUDGE_UNGRADED_RATE, ungraded, requests),
            self._included(CoreMetric.JUDGE_SEMANTIC_UNKNOWN_RATE, semantic_unknown, requests),
        )

    def score(
        self,
        *,
        run_config: EvalRunConfig,
        evaluator_execution: EvaluatorExecutionConfig,
        evaluation_revision: str,
        eval_case: EvalCase,
        submission: EvalSubmission,
        trial_index: int,
        intent_result: Optional[IntentEvaluationResult],
        review_result: Optional[ReviewEvaluationResult],
    ) -> TrialScore:
        if type(run_config) is not EvalRunConfig or type(evaluator_execution) is not EvaluatorExecutionConfig:
            raise _error("Trial scoring requires canonical Run/Evaluator execution config")
        evaluator_execution.validate_runtime_policy_support()
        if type(eval_case) is not EvalCase or type(submission) is not EvalSubmission:
            raise _error("Trial scoring requires canonical Case and Submission")
        suite_case = run_config.suite.case(eval_case.task_id)
        if submission.task_id != eval_case.task_id or submission.agent_id != run_config.agent.agent_id:
            raise _error("Submission identity differs from Run/Case")
        expected_trial_id = run_config.trial_id(eval_case.task_id, trial_index)
        if submission.trial_id != expected_trial_id:
            raise _error("Submission trial_id differs from Run binding")
        if (
            suite_case.case_version != eval_case.case_version
            or suite_case.canonical_case_digest != eval_case.digest()
            or suite_case.eval_input_digest != eval_case.eval_input().digest()
            or suite_case.truth_completeness is not eval_case.review_truth.completeness
            or eval_case.source.suite != run_config.suite.suite_id
        ):
            raise _error("EvalCase differs from immutable Suite binding")
        execution_digest = evaluator_execution.digest()
        location_policy_version = (
            self.location_policy_version
            if eval_case.eval_input().review_target.kind
            is ReviewTargetKind.REPOSITORY
            else REVIEW_LOCATION_POLICY_VERSION
        )
        evaluation_id = derive_evaluation_id(run_config.run_id, execution_digest, evaluation_revision)
        validate_evaluation_id(evaluation_id, run_config.run_id, execution_digest, evaluation_revision)
        if intent_result is not None:
            if type(intent_result) is not IntentEvaluationResult or intent_result.status is IntentEvaluationStatus.PENDING_JUDGE:
                raise _error("Intent score source is not terminal")
            expected_intent_digest = None if submission.intent is None else canonical_sha256(submission.intent.to_dict())
            if (
                intent_result.submission_intent_digest != expected_intent_digest
                or intent_result.intent_truth_digest != eval_case.intent_truth.digest()
                or intent_result.clarification_script_digest != eval_case.clarification_script.digest()
                or intent_result.policy_version != self.intent_policy_version
                or intent_result.normalization_version != self.intent_normalization_version
                or intent_result.evaluator_revision != self.intent_evaluator_revision
            ):
                raise _error("Intent result is not bound to scoring sources/policy")
            for item in (*intent_result.judge_failures, *intent_result.judge_ungraded):
                if item.evaluator_execution_digest != execution_digest:
                    raise _error("Intent Judge receipt belongs to another evaluator execution")
        if review_result is not None:
            if type(review_result) is not ReviewEvaluationResult or review_result.status is ReviewEvaluationStatus.PENDING_JUDGE:
                raise _error("Review score source is not terminal")
            if (
                review_result.submission_digest != submission.digest()
                or review_result.eval_input_digest != eval_case.eval_input().digest()
                or review_result.review_truth_digest != eval_case.review_truth.digest()
                or review_result.evaluator_execution_digest != execution_digest
                or review_result.review_policy_version != self.review_policy_version
                or review_result.assignment_policy_version != self.assignment_policy_version
                or review_result.location_policy_version != location_policy_version
                or review_result.evidence_integrity_policy_version != self.evidence_policy_version
                or review_result.evaluator_revision != self.review_evaluator_revision
            ):
                raise _error("Review result is not bound to scoring sources/policy")
        if intent_result is not None and submission.intent is None:
            raise _error("Intent evaluator result has no Submission Intent source")
        if review_result is not None and submission.review is None:
            raise _error("Review evaluator result has no Submission Review source")

        intent_binding = None
        if intent_result is not None:
            intent_binding = IntentScoreBinding(
                result_digest=intent_result.digest(),
                evaluator_revision=intent_result.evaluator_revision,
                policy_version=intent_result.policy_version,
                normalization_version=intent_result.normalization_version,
                status=intent_result.status,
                judge_request_count=len(intent_result.judge_requests),
                judge_graded_count=len(intent_result.judge_decisions),
                judge_failed_count=len(intent_result.judge_failures),
                judge_ungraded_count=len(intent_result.judge_ungraded),
                semantic_unknown_count=sum(item.relation is IntentJudgeRelation.UNKNOWN for item in intent_result.judge_decisions),
            )
        review_binding = None
        if review_result is not None:
            coverage = review_result.coverage
            review_binding = ReviewScoreBinding(
                result_digest=review_result.digest(),
                evaluator_revision=review_result.evaluator_revision,
                review_policy_version=review_result.review_policy_version,
                assignment_policy_version=review_result.assignment_policy_version,
                location_policy_version=review_result.location_policy_version,
                evidence_policy_version=review_result.evidence_integrity_policy_version,
                status=review_result.status,
                phase=review_result.phase,
                judge_request_count=coverage.judge_request_count,
                judge_graded_count=coverage.judge_graded_count,
                judge_failed_count=coverage.judge_failed_count,
                judge_ungraded_count=coverage.judge_ungraded_count,
                judge_pending_count=coverage.judge_pending_count,
                semantic_unknown_count=coverage.semantic_unknown_count,
                finding_count=coverage.finding_count,
                finding_resolved_count=coverage.finding_resolved_count,
            )
        contributions = [
            *self._intent_contributions(eval_case, submission, intent_result),
            *self._review_contributions(eval_case, submission, review_result),
            self._included(
                CoreMetric.AGENT_FAILURE_RATE,
                int(submission.status is not SubmissionStatus.COMPLETED),
                1,
            ),
            *self._judge_contributions(intent_result, review_result),
        ]
        contributions = sorted(contributions, key=lambda item: item.metric.value)
        authority_profile = MetricAuthorityProfile.from_eval_case(eval_case)
        authority_coverage = MetricAuthorityCoverage.from_eval_case(eval_case)
        wire_contract_digest = canonical_sha256(run_config.wire_contract.to_dict())
        authority_policy_version = (
            evaluator_execution.metric_authority_policy_version
        )
        compatibility = ScoreCompatibilityKey(
            run_id=run_config.run_id,
            evaluation_id=evaluation_id,
            suite_id=run_config.suite.suite_id,
            suite_version=run_config.suite.suite_version,
            manifest_digest=run_config.suite.manifest_digest,
            case_snapshot_id=run_config.suite.case_snapshot_id,
            case_snapshot_digest=run_config.suite.case_snapshot_digest,
            trial_count=run_config.trial_count,
            protocol_id=suite_case.protocol_id,
            target_kind=run_config.wire_contract.review_target_kind,
            wire_contract=run_config.wire_contract,
            wire_contract_digest=wire_contract_digest,
            adapter_capabilities_digest=run_config.adapter_capabilities_digest,
            isolation_profile=run_config.adapter_capabilities.isolation_profile,
            truth_completeness=eval_case.review_truth.completeness,
            novel_finding_policy=eval_case.review_truth.novel_finding_policy,
            metric_authority_profile=authority_profile,
            metric_authority_profile_digest=authority_profile.digest,
            metric_authority_policy_version=authority_policy_version,
            metric_authority_policy_digest=_metric_authority_policy_digest(
                authority_policy_version
            ),
            agent_config_digest=run_config.agent_config_digest,
            clarification_matcher_config_digest=run_config.clarification_matcher_config_digest,
            evaluator_execution_digest=execution_digest,
            metrics_policy_digest=self.metrics_policy.digest,
            metrics_policy=self.metrics_policy,
            intent_evaluator_revision=self.intent_evaluator_revision,
            review_evaluator_revision=self.review_evaluator_revision,
            intent_policy_version=self.intent_policy_version,
            intent_normalization_version=self.intent_normalization_version,
            review_policy_version=self.review_policy_version,
            assignment_policy_version=self.assignment_policy_version,
            location_policy_version=location_policy_version,
            evidence_policy_version=self.evidence_policy_version,
        )
        trace_ref = submission.trace_ref
        identity = {
            "schema_version": TRIAL_SCORE_SCHEMA_VERSION,
            "aggregator_revision": METRICS_AGGREGATOR_REVISION,
            "evaluation_revision": evaluation_revision,
            "compatibility": compatibility.to_dict(),
            "authority_coverage": authority_coverage.to_dict(),
            "task_id": eval_case.task_id,
            "case_version": eval_case.case_version,
            "trial_index": trial_index,
            "trial_id": submission.trial_id,
            "canonical_case_digest": eval_case.digest(),
            "eval_input_digest": eval_case.eval_input().digest(),
            "dimensions": [item.to_dict() for item in suite_case.dimensions],
            "submission_digest": submission.digest(),
            "submission_status": submission.status.value,
            "failure_code": None if submission.failure is None else submission.failure.code.value,
            "failure_retryable": None if submission.failure is None else submission.failure.retryable,
            "intent_binding": None if intent_binding is None else intent_binding.to_dict(),
            "review_binding": None if review_binding is None else review_binding.to_dict(),
            "contributions": [item.to_dict() for item in contributions],
            "usage": submission.usage.to_dict(),
            "trace_ref": None if trace_ref is None else trace_ref.to_dict(),
        }
        score_id = stable_id("trial-score-v1", identity)
        return _sealed_instance(
            TrialScore,
            {
                **identity,
                "score_id": score_id,
                "compatibility": compatibility,
                "authority_coverage": authority_coverage,
                "submission_status": submission.status,
                "failure_code": None if submission.failure is None else submission.failure.code,
                "intent_binding": intent_binding,
                "review_binding": review_binding,
                "dimensions": tuple(suite_case.dimensions),
                "contributions": tuple(contributions),
                "usage": submission.usage,
                "trace_ref": trace_ref,
            },
        )


class MetricsAggregator:
    """Aggregate trusted TrialScore values by ratio-of-sums."""

    # The aggregator is intentionally stateless.  Removing an instance
    # ``__dict__`` prevents hydration callers from shadowing aggregation
    # methods on an otherwise exact-type object.
    __slots__ = ()

    @staticmethod
    def _coverage(contributions: Sequence[MetricContribution]) -> MetricCoverage:
        return MetricCoverage(
            total_trial_count=len(contributions),
            included_trial_count=sum(item.included for item in contributions),
            failure_as_miss_count=sum(item.source_status is MetricSourceStatus.FAILURE_AS_MISS for item in contributions),
            zero_denominator_count=sum(item.included and item.denominator == 0 for item in contributions),
            not_scorable_count=sum(item.source_status is MetricSourceStatus.NOT_SCORABLE for item in contributions),
            ungraded_count=sum(item.source_status is MetricSourceStatus.UNGRADED for item in contributions),
            failure_excluded_count=sum(item.source_status is MetricSourceStatus.FAILURE_EXCLUDED for item in contributions),
            missing_count=sum(item.source_status is MetricSourceStatus.MISSING for item in contributions),
        )

    @staticmethod
    def _null_reason(coverage: MetricCoverage) -> MetricNullReason:
        if coverage.ungraded_count:
            return MetricNullReason.UNGRADED
        if coverage.failure_excluded_count:
            return MetricNullReason.FAILURE_EXCLUDED
        if coverage.not_scorable_count:
            return MetricNullReason.NOT_SCORABLE
        return MetricNullReason.MISSING

    def _aggregate_metric(self, metric: CoreMetric, trials: Sequence[TrialScore]) -> MetricAggregate:
        values = [item.contribution(metric) for item in trials]
        coverage = self._coverage(values)
        kind = _METRIC_KINDS[metric]
        included = [item for item in values if item.included]
        if not included:
            return MetricAggregate(metric, kind, None, None, None, self._null_reason(coverage), coverage)
        numerator = sum(int(item.numerator) for item in included)
        denominator = sum(int(item.denominator) for item in included)
        if kind is MetricKind.COUNT:
            return MetricAggregate(metric, kind, numerator, denominator, None, None, coverage)
        value = _ratio_ppm(numerator, denominator)
        return MetricAggregate(
            metric,
            kind,
            numerator,
            denominator,
            value,
            MetricNullReason.ZERO_DENOMINATOR if denominator == 0 else None,
            coverage,
        )

    @staticmethod
    def _f1(precision: MetricAggregate, recall: MetricAggregate) -> MetricAggregate:
        if precision.coverage is None or recall.coverage is None:
            raise _error("F1 sources require direct coverage")
        derived = (
            DerivedMetricCoverage(CoreMetric.ISSUE_PRECISION, precision.coverage),
            DerivedMetricCoverage(CoreMetric.ISSUE_RECALL, recall.coverage),
        )
        if precision.numerator is None or recall.numerator is None:
            reason = precision.null_reason or recall.null_reason or MetricNullReason.MISSING
            return MetricAggregate(
                CoreMetric.ISSUE_F1,
                MetricKind.RATE,
                None,
                None,
                None,
                reason,
                None,
                derived,
            )
        if precision.denominator == 0 or recall.denominator == 0:
            return MetricAggregate(
                CoreMetric.ISSUE_F1,
                MetricKind.RATE,
                0,
                0,
                None,
                MetricNullReason.ZERO_DENOMINATOR,
                None,
                derived,
            )
        numerator = 2 * precision.numerator * recall.numerator
        denominator = precision.numerator * recall.denominator + recall.numerator * precision.denominator
        if denominator == 0:
            numerator, denominator = 0, 1
        return MetricAggregate(
            CoreMetric.ISSUE_F1,
            MetricKind.RATE,
            numerator,
            denominator,
            _ratio_ppm(numerator, denominator),
            None,
            None,
            derived,
        )

    @staticmethod
    def _numeric_usage(field: str, unit: str, values: Sequence[Any], population: int) -> NumericUsageAggregate:
        observed = [item for item in values if item is not None]
        if not observed:
            return NumericUsageAggregate(field, unit, None, 0, population, None)
        if all(type(item) is int for item in observed):
            total: Any = sum(observed)
        else:
            total = math.fsum(float(item) for item in observed)
        return NumericUsageAggregate(field, unit, total, len(observed), population, total / len(observed))

    def _usage(self, trials: Sequence[TrialScore]) -> UsageAggregate:
        population = len(trials)
        currencies = {item.usage.cost_currency for item in trials if item.usage.cost_currency is not None}
        if len(currencies) > 1:
            raise _error("cannot aggregate cost across currencies")
        currency = next(iter(currencies), None)
        return UsageAggregate(
            elapsed_seconds=self._numeric_usage("elapsed_seconds", "seconds", [item.usage.elapsed_seconds for item in trials], population),
            input_tokens=self._numeric_usage("input_tokens", "tokens", [item.usage.input_tokens for item in trials], population),
            output_tokens=self._numeric_usage("output_tokens", "tokens", [item.usage.output_tokens for item in trials], population),
            total_tokens=self._numeric_usage("total_tokens", "tokens", [item.usage.total_tokens for item in trials], population),
            tool_calls=self._numeric_usage("tool_calls", "calls", [item.usage.tool_calls for item in trials], population),
            cost=self._numeric_usage(
                "cost_amount",
                MISSING_COST_UNIT if currency is None else currency,
                [item.usage.cost_amount for item in trials],
                population,
            ),
            cost_currency=currency,
        )

    @staticmethod
    def _breakdown(values: Iterable[str]) -> Tuple[CountBreakdown, ...]:
        counts: Dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        if len(counts) > MAX_BREAKDOWN_ITEMS:
            raise _error("count breakdown exceeds its key limit")
        return tuple(CountBreakdown(key, count) for key, count in sorted(counts.items()))

    def _aggregate_metrics(self, trials: Sequence[TrialScore]) -> Tuple[MetricAggregate, ...]:
        metrics = [self._aggregate_metric(metric, trials) for metric in _CONTRIBUTION_METRICS]
        by_id = {item.metric: item for item in metrics}
        metrics.append(self._f1(by_id[CoreMetric.ISSUE_PRECISION], by_id[CoreMetric.ISSUE_RECALL]))
        return tuple(sorted(metrics, key=lambda item: item.metric.value))

    @staticmethod
    def _validate_trials(trials: Sequence[TrialScore]) -> Tuple[TrialScore, ...]:
        values = tuple(trials)
        if not values or len(values) > MAX_TRIAL_SCORES or any(type(item) is not TrialScore for item in values):
            raise _error("Trial score collection is empty, oversized, or invalid")
        if len({item.trial_id for item in values}) != len(values):
            raise _error("Trial score collection contains duplicate trial IDs")
        compatibility = values[0].compatibility
        if any(item.compatibility != compatibility for item in values):
            raise _error("incompatible Trial scores cannot be aggregated")
        return tuple(sorted(values, key=lambda item: (item.task_id, item.trial_index)))

    def aggregate_case(
        self,
        trials: Sequence[TrialScore],
        *,
        planned_trial_count: Optional[int] = None,
    ) -> CaseScore:
        values = self._validate_trials(trials)
        first = values[0]
        if any(
            item.task_id != first.task_id
            or item.case_version != first.case_version
            or item.canonical_case_digest != first.canonical_case_digest
            or item.dimensions != first.dimensions
            for item in values
        ):
            raise _error("Case aggregation sources refer to different Cases")
        authority_coverage = first.authority_coverage
        if any(item.authority_coverage != authority_coverage for item in values):
            raise _error("one Case contains inconsistent authority coverage")
        planned = (
            first.compatibility.trial_count
            if planned_trial_count is None
            else _integer(planned_trial_count, "planned_trial_count", minimum=1)
        )
        if planned != first.compatibility.trial_count:
            raise _error("planned Trial count differs from immutable Run binding")
        if len(values) > planned:
            raise _error("terminal Trial scores exceed planned Trial count")
        metrics = self._aggregate_metrics(values)
        refs = tuple(
            sorted(
                (
                    ScoreRef(
                        item.score_id,
                        item.digest(),
                        item.task_id,
                        item.trial_id,
                    )
                    for item in values
                ),
                key=lambda ref: ref.trial_id or "",
            )
        )
        failed_ids = tuple(sorted(item.trial_id for item in values if item.submission_status is not SubmissionStatus.COMPLETED))
        ungraded_ids = tuple(
            sorted(
                item.trial_id
                for item in values
                if (item.intent_binding is not None and item.intent_binding.status is IntentEvaluationStatus.UNGRADED)
                or (item.review_binding is not None and item.review_binding.status is ReviewEvaluationStatus.UNGRADED)
            )
        )
        severe_ids = tuple(sorted(item.trial_id for item in values if int(item.contribution(CoreMetric.CRITICAL_HIGH_MISS_COUNT).numerator or 0) > 0))
        fabricated_ids = tuple(sorted(item.trial_id for item in values if int(item.contribution(CoreMetric.FABRICATED_FINDINGS_PER_PR).numerator or 0) > 0))
        identity = {
            "schema_version": CASE_SCORE_SCHEMA_VERSION,
            "aggregator_revision": METRICS_AGGREGATOR_REVISION,
            "compatibility": first.compatibility.to_dict(),
            "authority_coverage": authority_coverage.to_dict(),
            "task_id": first.task_id,
            "case_version": first.case_version,
            "canonical_case_digest": first.canonical_case_digest,
            "dimensions": [item.to_dict() for item in first.dimensions],
            "planned_trial_count": planned,
            "terminal_trial_count": len(values),
            "intent_scored_trial_count": sum(
                item.intent_binding is not None
                and item.intent_binding.status is IntentEvaluationStatus.GRADED
                for item in values
            ),
            "review_scored_trial_count": sum(
                item.review_binding is not None
                and item.review_binding.status is ReviewEvaluationStatus.GRADED
                for item in values
            ),
            "fully_scored_trial_count": sum(
                item.intent_binding is not None
                and item.intent_binding.status is IntentEvaluationStatus.GRADED
                and item.review_binding is not None
                and item.review_binding.status is ReviewEvaluationStatus.GRADED
                for item in values
            ),
            "trial_scores": [item.to_dict() for item in refs],
            "metrics": [item.to_dict() for item in metrics],
            "usage": self._usage(values).to_dict(),
            "submission_status_breakdown": [item.to_dict() for item in self._breakdown(item.submission_status.value for item in values)],
            "failure_code_breakdown": [item.to_dict() for item in self._breakdown(item.failure_code.value for item in values if item.failure_code is not None)],
            "failed_trial_ids": list(failed_ids),
            "ungraded_trial_ids": list(ungraded_ids),
            "critical_high_miss_trial_ids": list(severe_ids),
            "fabricated_trial_ids": list(fabricated_ids),
        }
        score_id = stable_id("case-score-v1", identity)
        return _sealed_instance(
            CaseScore,
            {
                **identity,
                "score_id": score_id,
                "compatibility": first.compatibility,
                "authority_coverage": authority_coverage,
                "dimensions": first.dimensions,
                "trial_scores": refs,
                "metrics": metrics,
                "usage": self._usage(values),
                "submission_status_breakdown": self._breakdown(item.submission_status.value for item in values),
                "failure_code_breakdown": self._breakdown(item.failure_code.value for item in values if item.failure_code is not None),
                "failed_trial_ids": failed_ids,
                "ungraded_trial_ids": ungraded_ids,
                "critical_high_miss_trial_ids": severe_ids,
                "fabricated_trial_ids": fabricated_ids,
            },
        )

    def aggregate_cases(
        self,
        case_scores: Sequence[CaseScore],
        *,
        group_dimensions: Sequence[CaseDimension] = (),
        source_trials: Sequence[TrialScore],
    ) -> AggregateScore:
        cases = tuple(case_scores)
        if not cases or len(cases) > MAX_CASE_SCORES or any(type(item) is not CaseScore for item in cases):
            raise _error("Case score collection is empty, oversized, or invalid")
        if len({item.task_id for item in cases}) != len(cases):
            raise _error("Case score collection contains duplicate task IDs")
        compatibility = cases[0].compatibility
        if any(item.compatibility != compatibility for item in cases):
            raise _error("incompatible Case scores cannot be aggregated")
        trials = self._validate_trials(source_trials)
        if trials[0].compatibility != compatibility:
            raise _error("Aggregate source Trials are incompatible with Case scores")
        case_trial_ids = {ref.trial_id for case in cases for ref in case.trial_scores}
        if {item.trial_id for item in trials} != case_trial_ids:
            raise _error("Aggregate source Trials differ from Case score refs")
        refs_by_trial_id = {
            ref.trial_id: ref
            for case in cases
            for ref in case.trial_scores
        }
        for trial in trials:
            expected_ref = ScoreRef(
                trial.score_id,
                trial.digest(),
                trial.task_id,
                trial.trial_id,
            )
            if refs_by_trial_id[trial.trial_id] != expected_ref:
                raise _error("Aggregate source Trial digest differs from Case score ref")
        trials_by_task: Dict[str, list[TrialScore]] = {}
        for trial in trials:
            trials_by_task.setdefault(trial.task_id, []).append(trial)
        for case in cases:
            case_trials = tuple(trials_by_task.get(case.task_id, ()))
            if (
                not case_trials
                or any(
                    item.authority_coverage != case.authority_coverage
                    for item in case_trials
                )
            ):
                raise _error(
                    "Aggregate Case authority coverage differs from source Trials"
                )
        authority_coverage = MetricAuthorityCoverage.aggregate(
            tuple(item.authority_coverage for item in cases)
        )
        metrics = self._aggregate_metrics(trials)
        refs = tuple(
            sorted(
                (ScoreRef(item.score_id, item.digest(), item.task_id, None) for item in cases),
                key=lambda item: item.task_id,
            )
        )
        dimensions = tuple(sorted(tuple(group_dimensions), key=lambda item: item.name))
        if len({item.name for item in dimensions}) != len(dimensions):
            raise _error("Aggregate group dimensions contain duplicates")
        for dimension in dimensions:
            if any(dimension not in case.dimensions for case in cases):
                raise _error(
                    "Aggregate group dimensions are not a common Case projection"
                )
        identity = {
            "schema_version": AGGREGATE_SCORE_SCHEMA_VERSION,
            "aggregator_revision": METRICS_AGGREGATOR_REVISION,
            "compatibility": compatibility.to_dict(),
            "authority_coverage": authority_coverage.to_dict(),
            "group_dimensions": [item.to_dict() for item in dimensions],
            "case_count": len(cases),
            "planned_trial_count": sum(item.planned_trial_count for item in cases),
            "terminal_trial_count": len(trials),
            "intent_scored_trial_count": sum(
                item.intent_binding is not None
                and item.intent_binding.status is IntentEvaluationStatus.GRADED
                for item in trials
            ),
            "review_scored_trial_count": sum(
                item.review_binding is not None
                and item.review_binding.status is ReviewEvaluationStatus.GRADED
                for item in trials
            ),
            "fully_scored_trial_count": sum(
                item.intent_binding is not None
                and item.intent_binding.status is IntentEvaluationStatus.GRADED
                and item.review_binding is not None
                and item.review_binding.status is ReviewEvaluationStatus.GRADED
                for item in trials
            ),
            "case_scores": [item.to_dict() for item in refs],
            "metrics": [item.to_dict() for item in metrics],
            "usage": self._usage(trials).to_dict(),
            "submission_status_breakdown": [item.to_dict() for item in self._breakdown(item.submission_status.value for item in trials)],
            "failure_code_breakdown": [item.to_dict() for item in self._breakdown(item.failure_code.value for item in trials if item.failure_code is not None)],
            "failed_trial_ids": sorted(item.trial_id for item in trials if item.submission_status is not SubmissionStatus.COMPLETED),
            "ungraded_trial_ids": sorted(
                item.trial_id
                for item in trials
                if (item.intent_binding is not None and item.intent_binding.status is IntentEvaluationStatus.UNGRADED)
                or (item.review_binding is not None and item.review_binding.status is ReviewEvaluationStatus.UNGRADED)
            ),
            "critical_high_miss_trial_ids": sorted(item.trial_id for item in trials if int(item.contribution(CoreMetric.CRITICAL_HIGH_MISS_COUNT).numerator or 0) > 0),
            "fabricated_trial_ids": sorted(item.trial_id for item in trials if int(item.contribution(CoreMetric.FABRICATED_FINDINGS_PER_PR).numerator or 0) > 0),
        }
        score_id = stable_id("aggregate-score-v1", identity)
        return _sealed_instance(
            AggregateScore,
            {
                **identity,
                "score_id": score_id,
                "compatibility": compatibility,
                "authority_coverage": authority_coverage,
                "group_dimensions": dimensions,
                "case_scores": refs,
                "metrics": metrics,
                "usage": self._usage(trials),
                "submission_status_breakdown": self._breakdown(item.submission_status.value for item in trials),
                "failure_code_breakdown": self._breakdown(item.failure_code.value for item in trials if item.failure_code is not None),
                "failed_trial_ids": tuple(identity["failed_trial_ids"]),
                "ungraded_trial_ids": tuple(identity["ungraded_trial_ids"]),
                "critical_high_miss_trial_ids": tuple(identity["critical_high_miss_trial_ids"]),
                "fabricated_trial_ids": tuple(identity["fabricated_trial_ids"]),
            },
        )

    def group_case_scores(
        self,
        case_scores: Sequence[CaseScore],
        trial_scores: Sequence[TrialScore],
        *,
        dimension_names: Sequence[str] = (),
    ) -> Tuple[AggregateScore, ...]:
        names = tuple(_id(item, "group dimension name") for item in dimension_names)
        if len(names) != len(set(names)):
            raise _error("group dimension names contain duplicates")
        cases = tuple(case_scores)
        if (
            not cases
            or len(cases) > MAX_CASE_SCORES
            or any(type(item) is not CaseScore for item in cases)
            or len({item.task_id for item in cases}) != len(cases)
        ):
            raise _error("Case score grouping collection is empty, oversized, or invalid")
        trials = tuple(trial_scores)
        if (
            not trials
            or len(trials) > MAX_TRIAL_SCORES
            or any(type(item) is not TrialScore for item in trials)
            or len({item.trial_id for item in trials}) != len(trials)
        ):
            raise _error("Trial score grouping collection is empty, oversized, or invalid")
        trials_by_id = {item.trial_id: item for item in trials}
        referenced_trial_ids = {
            ref.trial_id
            for case in cases
            for ref in case.trial_scores
        }
        if set(trials_by_id) != referenced_trial_ids:
            raise _error("group source Trials differ from Case score refs")
        groups: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], list[CaseScore]] = {}
        for case in cases:
            dimensions = {item.name: item.value for item in case.dimensions}
            missing = set(names) - set(dimensions)
            if missing:
                raise _error("requested group dimension is missing from a Case")
            selected = tuple((name, dimensions[name]) for name in names)
            key = (canonical_sha256(case.compatibility.to_dict()), selected)
            groups.setdefault(key, []).append(case)
        results = []
        for (_compatibility_digest, selected), cases in sorted(groups.items(), key=lambda item: item[0]):
            trial_ids = {ref.trial_id for case in cases for ref in case.trial_scores}
            selected_trials = tuple(trials_by_id[item] for item in sorted(trial_ids) if item in trials_by_id)
            if len(selected_trials) != len(trial_ids):
                raise _error("group source Trial score is missing")
            dimensions = tuple(CaseDimension(name, value) for name, value in selected)
            results.append(
                self.aggregate_cases(
                    tuple(cases),
                    group_dimensions=dimensions,
                    source_trials=selected_trials,
                )
            )
        return tuple(sorted(results, key=lambda item: (canonical_sha256(item.compatibility.to_dict()), tuple((d.name, d.value) for d in item.group_dimensions))))


__all__ = [
    "TRIAL_SCORE_SCHEMA_VERSION",
    "CASE_SCORE_SCHEMA_VERSION",
    "AGGREGATE_SCORE_SCHEMA_VERSION",
    "METRICS_POLICY_VERSION",
    "SEVERITY_WEIGHT_POLICY_VERSION",
    "LINE_METRIC_POLICY_VERSION",
    "METRICS_AGGREGATOR_REVISION",
    "PPM_SCALE",
    "MISSING_COST_UNIT",
    "MetricsError",
    "FailureOutcomePolicy",
    "MetricKind",
    "MetricSourceStatus",
    "MetricNullReason",
    "CoreMetric",
    "SeverityWeight",
    "SeverityWeightPolicy",
    "DEFAULT_SEVERITY_WEIGHT_POLICY",
    "LineMetricPolicy",
    "DEFAULT_LINE_METRIC_POLICY",
    "MetricsPolicy",
    "DEFAULT_METRICS_POLICY",
    "MetricContribution",
    "MetricCoverage",
    "DerivedMetricCoverage",
    "MetricAggregate",
    "NumericUsageAggregate",
    "UsageAggregate",
    "IntentScoreBinding",
    "ReviewScoreBinding",
    "ScoreCompatibilityKey",
    "TrialScore",
    "ScoreRef",
    "CountBreakdown",
    "CaseScore",
    "AggregateScore",
    "TrialScorer",
    "MetricsAggregator",
]
