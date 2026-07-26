"""Source-bound repeated-Trial statistics for Eval v2.

The module only reorganizes canonical :class:`TrialScore` contributions.  It
does not inspect raw Intent/Finding/Evidence semantics and never invokes an
Agent or Judge.
"""

from __future__ import annotations

import math
import random
import statistics as stdlib_statistics
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .analysis_artifacts import AnalysisSourceBinding, bind_analysis_source
from .cases import RunCaseSnapshot
from .config import EvalRunConfig, MAX_TRIAL_COUNT
from .metrics import (
    CoreMetric,
    MetricAggregate,
    MetricCoverage,
    MetricKind,
    MetricNullReason,
    MetricsAggregator,
    PPM_SCALE,
    TrialScore,
)
from .models import (
    _JsonModel,
    _strict_json_loads,
    canonical_json_bytes,
    stable_id,
)


RUN_STATISTICS_SCHEMA_VERSION = "run_statistics_v1"
STATISTICS_ALGORITHM_VERSION = "repeated-trial-statistics-v1"
MAX_STATISTICS_BYTES = 256 * 1024 * 1024
MAX_BOOTSTRAP_SEED = (1 << 63) - 1
MAX_BOOTSTRAP_ITERATIONS = 100_000
MAX_BOOTSTRAP_CASES = 16_384
MAX_BOOTSTRAP_DRAWS = 50_000_000


class StatisticsError(ValueError):
    """A repeated-Trial policy, source, or result is invalid."""


class MetricUnit(str, Enum):
    PPM = "ppm"
    COUNT = "count"


class MetricDirection(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class StatisticsMetricStatus(str, Enum):
    AVAILABLE = "available"
    ZERO_DENOMINATOR = "zero_denominator"
    NOT_SCORABLE = "not_scorable"
    UNGRADED = "ungraded"
    FAILURE_EXCLUDED = "failure_excluded"
    MISSING = "missing"


class DispersionNullReason(str, Enum):
    NO_AVAILABLE_REPLICATES = "no_available_replicates"
    INSUFFICIENT_REPLICATES = "insufficient_replicates"


class ConfidenceIntervalStatus(str, Enum):
    AVAILABLE = "available"
    INSUFFICIENT_CASE_POPULATION = "insufficient_case_population"
    ZERO_DENOMINATOR = "zero_denominator"
    INCOMPLETE_REPLICATES = "incomplete_replicates"
    NOT_SCORABLE = "not_scorable"
    UNGRADED = "ungraded"
    FAILURE_EXCLUDED = "failure_excluded"
    MISSING = "missing"


_LOWER_IS_BETTER = frozenset(
    {
        CoreMetric.INTENT_PARTIALLY_SUPPORTED_RATE,
        CoreMetric.INTENT_UNSUPPORTED_RATE,
        CoreMetric.INTENT_CONTRADICTED_RATE,
        CoreMetric.INTENT_UNKNOWN_RATE,
        CoreMetric.CRITICAL_HIGH_MISS_COUNT,
        CoreMetric.FABRICATED_FINDINGS_PER_PR,
        CoreMetric.FABRICATED_RATE,
        CoreMetric.REVIEW_UNKNOWN_RATE,
        CoreMetric.AGENT_FAILURE_RATE,
        CoreMetric.JUDGE_FAILURE_RATE,
        CoreMetric.JUDGE_UNGRADED_RATE,
        CoreMetric.JUDGE_SEMANTIC_UNKNOWN_RATE,
    }
)
_METRIC_KINDS: Mapping[CoreMetric, MetricKind] = {
    **{metric: MetricKind.RATE for metric in CoreMetric},
    CoreMetric.CRITICAL_HIGH_MISS_COUNT: MetricKind.COUNT,
    CoreMetric.FABRICATED_FINDINGS_PER_PR: MetricKind.MEAN,
}


def _error(message: str) -> StatisticsError:
    return StatisticsError(message)


def _exact(value: Any, fields: Iterable[str], context: str) -> Dict[str, Any]:
    expected = set(fields)
    if type(value) is not dict or set(value) != expected or len(value) != len(expected):
        raise _error(f"{context} has unknown or missing fields")
    return value


def _array(value: Any, context: str, maximum: int) -> list[Any]:
    if type(value) is not list or len(value) > maximum:
        raise _error(f"{context} must be a bounded list")
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
        suffix = "" if maximum is None else f" and <= {maximum}"
        raise _error(f"{context} must be an integer >= {minimum}{suffix}")
    return value


def _optional_integer(value: Any, context: str) -> Optional[int]:
    return None if value is None else _integer(value, context)


def _enum(enum_type: type[Enum], value: Any, context: str) -> Any:
    if type(value) is not str:
        raise _error(f"{context} must be an enum string")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise _error(f"{context} has an unknown value") from exc


def _identifier(value: Any, context: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 512
        or value != value.strip()
        or any(character.isspace() or ord(character) < 32 for character in value)
    ):
        raise _error(f"{context} must be a bounded non-empty identifier")
    return value


def _digest(value: Any, context: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise _error(f"{context} must be a lowercase SHA-256 digest")
    return value


def _ratio_ppm(numerator: int, denominator: int) -> Optional[int]:
    if denominator == 0:
        return None
    return (numerator * PPM_SCALE + denominator // 2) // denominator


def _metric_unit(metric: CoreMetric) -> MetricUnit:
    return MetricUnit.COUNT if _METRIC_KINDS[metric] is MetricKind.COUNT else MetricUnit.PPM


def _metric_direction(metric: CoreMetric) -> MetricDirection:
    return (
        MetricDirection.LOWER_IS_BETTER
        if metric in _LOWER_IS_BETTER
        else MetricDirection.HIGHER_IS_BETTER
    )


def _metric_status(value: MetricAggregate) -> StatisticsMetricStatus:
    if value.numerator is not None:
        if value.kind is not MetricKind.COUNT and value.denominator == 0:
            return StatisticsMetricStatus.ZERO_DENOMINATOR
        return StatisticsMetricStatus.AVAILABLE
    mapping = {
        MetricNullReason.ZERO_DENOMINATOR: StatisticsMetricStatus.ZERO_DENOMINATOR,
        MetricNullReason.NOT_SCORABLE: StatisticsMetricStatus.NOT_SCORABLE,
        MetricNullReason.UNGRADED: StatisticsMetricStatus.UNGRADED,
        MetricNullReason.FAILURE_EXCLUDED: StatisticsMetricStatus.FAILURE_EXCLUDED,
        MetricNullReason.MISSING: StatisticsMetricStatus.MISSING,
    }
    try:
        return mapping[value.null_reason]
    except KeyError as exc:  # pragma: no cover - guarded by MetricAggregate
        raise _error("MetricAggregate has no canonical Statistics status") from exc


def _metric_value(value: MetricAggregate) -> Optional[int]:
    if value.kind is MetricKind.COUNT:
        return value.numerator
    return value.value_ppm


def _validate_metric_value(
    *,
    metric: CoreMetric,
    kind: MetricKind,
    unit: MetricUnit,
    direction: MetricDirection,
    status: StatisticsMetricStatus,
    numerator: Optional[int],
    denominator: Optional[int],
    value: Optional[int],
    context: str,
) -> None:
    if type(metric) is not CoreMetric:
        raise _error(f"{context}.metric is invalid")
    if type(kind) is not MetricKind or kind is not _METRIC_KINDS[metric]:
        raise _error(f"{context}.kind is not canonical")
    if type(unit) is not MetricUnit or unit is not _metric_unit(metric):
        raise _error(f"{context}.unit is not canonical")
    if type(direction) is not MetricDirection or direction is not _metric_direction(metric):
        raise _error(f"{context}.direction is not canonical")
    if type(status) is not StatisticsMetricStatus:
        raise _error(f"{context}.status is invalid")
    if status in {
        StatisticsMetricStatus.AVAILABLE,
        StatisticsMetricStatus.ZERO_DENOMINATOR,
    }:
        checked_numerator = _integer(numerator, f"{context}.numerator")
        checked_denominator = _integer(denominator, f"{context}.denominator")
        if kind is MetricKind.RATE and checked_numerator > checked_denominator:
            raise _error(f"{context} rate numerator exceeds denominator")
        if status is StatisticsMetricStatus.ZERO_DENOMINATOR:
            if kind is MetricKind.COUNT or checked_denominator != 0 or value is not None:
                raise _error(f"{context} zero-denominator value is not canonical")
            return
        if kind is MetricKind.COUNT:
            if value != checked_numerator:
                raise _error(f"{context} count value is not canonical")
        elif checked_denominator == 0 or value != _ratio_ppm(
            checked_numerator, checked_denominator
        ):
            raise _error(f"{context} scaled value is not canonical")
    elif numerator is not None or denominator is not None or value is not None:
        raise _error(f"{context} unavailable value must use null fields")


def _from_json(model_type: Any, data: Any, context: str) -> Any:
    return model_type.from_dict(
        _strict_json_loads(data, MAX_STATISTICS_BYTES, context)
    )


@dataclass(frozen=True)
class StatisticsPolicyV1(_JsonModel):
    algorithm_version: str
    bootstrap_seed: int
    bootstrap_iterations: int
    confidence_level_ppm: int

    def __post_init__(self) -> None:
        if self.algorithm_version != STATISTICS_ALGORITHM_VERSION:
            raise _error("StatisticsPolicyV1 algorithm_version is unsupported")
        _integer(
            self.bootstrap_seed,
            "StatisticsPolicyV1.bootstrap_seed",
            maximum=MAX_BOOTSTRAP_SEED,
        )
        _integer(
            self.bootstrap_iterations,
            "StatisticsPolicyV1.bootstrap_iterations",
            minimum=1,
            maximum=MAX_BOOTSTRAP_ITERATIONS,
        )
        _integer(
            self.confidence_level_ppm,
            "StatisticsPolicyV1.confidence_level_ppm",
            minimum=1,
            maximum=PPM_SCALE - 1,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm_version": self.algorithm_version,
            "bootstrap_seed": self.bootstrap_seed,
            "bootstrap_iterations": self.bootstrap_iterations,
            "confidence_level_ppm": self.confidence_level_ppm,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "StatisticsPolicyV1":
        payload = _exact(
            value,
            (
                "algorithm_version",
                "bootstrap_seed",
                "bootstrap_iterations",
                "confidence_level_ppm",
            ),
            "StatisticsPolicyV1",
        )
        return cls(**payload)

    @classmethod
    def from_json(cls, data: Any) -> "StatisticsPolicyV1":
        return _from_json(cls, data, "StatisticsPolicyV1 JSON")


@dataclass(frozen=True)
class MetricSourceCoverageV1(_JsonModel):
    metric: CoreMetric
    total_trial_count: int
    included_trial_count: int
    failure_as_miss_count: int
    zero_denominator_count: int
    not_scorable_count: int
    ungraded_count: int
    failure_excluded_count: int
    missing_count: int

    def __post_init__(self) -> None:
        if type(self.metric) is not CoreMetric:
            raise _error("MetricSourceCoverageV1.metric is invalid")
        for name in self.__dataclass_fields__:
            if name != "metric":
                _integer(getattr(self, name), f"MetricSourceCoverageV1.{name}")
        if (
            self.included_trial_count
            + self.not_scorable_count
            + self.ungraded_count
            + self.failure_excluded_count
            + self.missing_count
            != self.total_trial_count
        ):
            raise _error("MetricSourceCoverageV1 statuses do not cover Trials")
        if (
            self.failure_as_miss_count > self.included_trial_count
            or self.zero_denominator_count > self.included_trial_count
        ):
            raise _error("MetricSourceCoverageV1 subset counts are inconsistent")

    @classmethod
    def from_metric_coverage(
        cls, metric: CoreMetric, coverage: MetricCoverage
    ) -> "MetricSourceCoverageV1":
        if type(coverage) is not MetricCoverage:
            raise _error("source coverage must be MetricCoverage")
        return cls(metric=metric, **coverage.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric.value,
            **{
                name: getattr(self, name)
                for name in self.__dataclass_fields__
                if name != "metric"
            },
        }

    @classmethod
    def from_dict(cls, value: Any) -> "MetricSourceCoverageV1":
        fields = tuple(cls.__dataclass_fields__)
        payload = _exact(value, fields, "MetricSourceCoverageV1")
        return cls(
            metric=_enum(CoreMetric, payload["metric"], "coverage.metric"),
            **{name: payload[name] for name in fields if name != "metric"},
        )


@dataclass(frozen=True)
class StatisticsCoverageV1(_JsonModel):
    total_trial_count: int
    completed_trial_count: int
    agent_failure_count: int
    judge_request_count: int
    judge_graded_count: int
    judge_failure_count: int
    judge_ungraded_count: int
    judge_semantic_unknown_count: int
    metric_sources: Tuple[MetricSourceCoverageV1, ...]

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if name != "metric_sources":
                _integer(getattr(self, name), f"StatisticsCoverageV1.{name}")
        if self.completed_trial_count + self.agent_failure_count != self.total_trial_count:
            raise _error("Statistics coverage does not partition completed/failing Trials")
        if (
            self.judge_graded_count
            + self.judge_failure_count
            + self.judge_ungraded_count
            != self.judge_request_count
            or self.judge_semantic_unknown_count > self.judge_graded_count
        ):
            raise _error("Statistics Judge coverage is inconsistent")
        sources = tuple(self.metric_sources)
        if (
            not sources
            or len(sources) > 2
            or any(type(item) is not MetricSourceCoverageV1 for item in sources)
            or sources != tuple(sorted(sources, key=lambda item: item.metric.value))
            or len({item.metric for item in sources}) != len(sources)
            or any(item.total_trial_count != self.total_trial_count for item in sources)
        ):
            raise _error("Statistics metric source coverage is not canonical")
        object.__setattr__(self, "metric_sources", sources)

    def to_dict(self) -> Dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "metric_sources"
        } | {"metric_sources": [item.to_dict() for item in self.metric_sources]}

    @classmethod
    def from_dict(cls, value: Any) -> "StatisticsCoverageV1":
        fields = tuple(cls.__dataclass_fields__)
        payload = _exact(value, fields, "StatisticsCoverageV1")
        sources = _array(payload["metric_sources"], "coverage.metric_sources", 2)
        return cls(
            **{name: payload[name] for name in fields if name != "metric_sources"},
            metric_sources=tuple(
                MetricSourceCoverageV1.from_dict(item) for item in sources
            ),
        )


@dataclass(frozen=True)
class DerivedCaseContributionV1(_JsonModel):
    metric: CoreMetric
    status: StatisticsMetricStatus
    numerator: Optional[int]
    denominator: Optional[int]

    def __post_init__(self) -> None:
        if self.metric not in {CoreMetric.ISSUE_PRECISION, CoreMetric.ISSUE_RECALL}:
            raise _error("derived Case contribution has an invalid metric")
        _validate_metric_value(
            metric=self.metric,
            kind=MetricKind.RATE,
            unit=MetricUnit.PPM,
            direction=MetricDirection.HIGHER_IS_BETTER,
            status=self.status,
            numerator=self.numerator,
            denominator=self.denominator,
            value=(
                _ratio_ppm(self.numerator, self.denominator)
                if self.numerator is not None and self.denominator
                else None
            ),
            context="DerivedCaseContributionV1",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric.value,
            "status": self.status.value,
            "numerator": self.numerator,
            "denominator": self.denominator,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "DerivedCaseContributionV1":
        payload = _exact(
            value,
            ("metric", "status", "numerator", "denominator"),
            "DerivedCaseContributionV1",
        )
        return cls(
            metric=_enum(CoreMetric, payload["metric"], "derived.metric"),
            status=_enum(
                StatisticsMetricStatus, payload["status"], "derived.status"
            ),
            numerator=_optional_integer(payload["numerator"], "derived.numerator"),
            denominator=_optional_integer(
                payload["denominator"], "derived.denominator"
            ),
        )


@dataclass(frozen=True)
class CaseContributionV1(_JsonModel):
    task_id: str
    case_version: int
    canonical_case_digest: str
    trial_index: Optional[int]
    metric: CoreMetric
    kind: MetricKind
    unit: MetricUnit
    direction: MetricDirection
    status: StatisticsMetricStatus
    numerator: Optional[int]
    denominator: Optional[int]
    value: Optional[int]
    coverage: StatisticsCoverageV1
    derived_contributions: Tuple[DerivedCaseContributionV1, ...] = ()

    def __post_init__(self) -> None:
        _identifier(self.task_id, "CaseContributionV1.task_id")
        _integer(self.case_version, "CaseContributionV1.case_version", minimum=1)
        _digest(self.canonical_case_digest, "CaseContributionV1.case digest")
        if self.trial_index is not None:
            _integer(
                self.trial_index,
                "CaseContributionV1.trial_index",
                minimum=1,
                maximum=MAX_TRIAL_COUNT,
            )
        _validate_metric_value(
            metric=self.metric,
            kind=self.kind,
            unit=self.unit,
            direction=self.direction,
            status=self.status,
            numerator=self.numerator,
            denominator=self.denominator,
            value=self.value,
            context="CaseContributionV1",
        )
        if type(self.coverage) is not StatisticsCoverageV1:
            raise _error("CaseContributionV1.coverage is invalid")
        derived = tuple(self.derived_contributions)
        if self.metric is CoreMetric.ISSUE_F1:
            expected_sources = (
                CoreMetric.ISSUE_PRECISION,
                CoreMetric.ISSUE_RECALL,
            )
            if tuple(item.metric for item in derived) != (
                CoreMetric.ISSUE_PRECISION,
                CoreMetric.ISSUE_RECALL,
            ):
                raise _error("F1 Case contribution lacks canonical source contributions")
        else:
            expected_sources = (self.metric,)
            if derived:
                raise _error("base Case contribution cannot contain derived contributions")
        if tuple(item.metric for item in self.coverage.metric_sources) != expected_sources:
            raise _error("Case contribution coverage sources differ from its metric")
        object.__setattr__(self, "derived_contributions", derived)

    def _identity_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "case_version": self.case_version,
            "canonical_case_digest": self.canonical_case_digest,
            "trial_index": self.trial_index,
            "metric": self.metric.value,
            "kind": self.kind.value,
            "unit": self.unit.value,
            "direction": self.direction.value,
            "status": self.status.value,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value": self.value,
            "coverage": self.coverage.to_dict(),
            "derived_contributions": [
                item.to_dict() for item in self.derived_contributions
            ],
        }

    @property
    def contribution_id(self) -> str:
        return stable_id("case-contribution-v1", self._identity_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {**self._identity_dict(), "contribution_id": self.contribution_id}

    @classmethod
    def from_dict(cls, value: Any) -> "CaseContributionV1":
        fields = (
            "contribution_id",
            "task_id",
            "case_version",
            "canonical_case_digest",
            "trial_index",
            "metric",
            "kind",
            "unit",
            "direction",
            "status",
            "numerator",
            "denominator",
            "value",
            "coverage",
            "derived_contributions",
        )
        payload = _exact(value, fields, "CaseContributionV1")
        derived = _array(
            payload["derived_contributions"], "case derived contributions", 2
        )
        result = cls(
            task_id=payload["task_id"],
            case_version=payload["case_version"],
            canonical_case_digest=payload["canonical_case_digest"],
            trial_index=payload["trial_index"],
            metric=_enum(CoreMetric, payload["metric"], "case metric"),
            kind=_enum(MetricKind, payload["kind"], "case kind"),
            unit=_enum(MetricUnit, payload["unit"], "case unit"),
            direction=_enum(MetricDirection, payload["direction"], "case direction"),
            status=_enum(
                StatisticsMetricStatus, payload["status"], "case status"
            ),
            numerator=_optional_integer(payload["numerator"], "case numerator"),
            denominator=_optional_integer(payload["denominator"], "case denominator"),
            value=_optional_integer(payload["value"], "case value"),
            coverage=StatisticsCoverageV1.from_dict(payload["coverage"]),
            derived_contributions=tuple(
                DerivedCaseContributionV1.from_dict(item) for item in derived
            ),
        )
        if payload["contribution_id"] != result.contribution_id:
            raise _error("CaseContributionV1 contribution_id is not canonical")
        return result


@dataclass(frozen=True)
class BootstrapCoverageV1(_JsonModel):
    total_case_count: int
    available_case_count: int
    zero_denominator_case_count: int
    not_scorable_case_count: int
    ungraded_case_count: int
    failure_excluded_case_count: int
    missing_case_count: int
    incomplete_replicate_count: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _integer(getattr(self, name), f"BootstrapCoverageV1.{name}")
        if (
            self.available_case_count
            + self.zero_denominator_case_count
            + self.not_scorable_case_count
            + self.ungraded_case_count
            + self.failure_excluded_case_count
            + self.missing_case_count
            != self.total_case_count
        ):
            raise _error("Bootstrap Case statuses do not cover the population")

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Any) -> "BootstrapCoverageV1":
        fields = tuple(cls.__dataclass_fields__)
        return cls(**_exact(value, fields, "BootstrapCoverageV1"))


@dataclass(frozen=True)
class ConfidenceIntervalV1(_JsonModel):
    metric: CoreMetric
    kind: MetricKind
    unit: MetricUnit
    status: ConfidenceIntervalStatus
    lower_bound: Optional[int]
    upper_bound: Optional[int]
    seed: int
    iterations: int
    confidence_level_ppm: int
    coverage: BootstrapCoverageV1

    def __post_init__(self) -> None:
        if type(self.metric) is not CoreMetric:
            raise _error("ConfidenceIntervalV1.metric is invalid")
        if type(self.kind) is not MetricKind or self.kind is not _METRIC_KINDS[self.metric]:
            raise _error("ConfidenceIntervalV1.kind is not canonical")
        if type(self.unit) is not MetricUnit or self.unit is not _metric_unit(self.metric):
            raise _error("ConfidenceIntervalV1.unit is not canonical")
        if type(self.status) is not ConfidenceIntervalStatus:
            raise _error("ConfidenceIntervalV1.status is invalid")
        _integer(self.seed, "ConfidenceIntervalV1.seed", maximum=MAX_BOOTSTRAP_SEED)
        _integer(
            self.iterations,
            "ConfidenceIntervalV1.iterations",
            minimum=1,
            maximum=MAX_BOOTSTRAP_ITERATIONS,
        )
        _integer(
            self.confidence_level_ppm,
            "ConfidenceIntervalV1.confidence_level_ppm",
            minimum=1,
            maximum=PPM_SCALE - 1,
        )
        if type(self.coverage) is not BootstrapCoverageV1:
            raise _error("ConfidenceIntervalV1.coverage is invalid")
        if self.coverage.incomplete_replicate_count > self.iterations:
            raise _error("Bootstrap incomplete replicate count exceeds iterations")
        if self.status is ConfidenceIntervalStatus.AVAILABLE:
            lower = _integer(self.lower_bound, "ConfidenceIntervalV1.lower_bound")
            upper = _integer(self.upper_bound, "ConfidenceIntervalV1.upper_bound")
            if lower > upper or self.coverage.incomplete_replicate_count:
                raise _error("available Confidence interval is inconsistent")
        elif self.lower_bound is not None or self.upper_bound is not None:
            raise _error("unavailable Confidence interval must use null bounds")

    def _identity_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric.value,
            "kind": self.kind.value,
            "unit": self.unit.value,
            "status": self.status.value,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "seed": self.seed,
            "iterations": self.iterations,
            "confidence_level_ppm": self.confidence_level_ppm,
            "coverage": self.coverage.to_dict(),
        }

    @property
    def interval_id(self) -> str:
        return stable_id("confidence-interval-v1", self._identity_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {**self._identity_dict(), "interval_id": self.interval_id}

    @classmethod
    def from_dict(cls, value: Any) -> "ConfidenceIntervalV1":
        fields = (
            "interval_id",
            "metric",
            "kind",
            "unit",
            "status",
            "lower_bound",
            "upper_bound",
            "seed",
            "iterations",
            "confidence_level_ppm",
            "coverage",
        )
        payload = _exact(value, fields, "ConfidenceIntervalV1")
        result = cls(
            metric=_enum(CoreMetric, payload["metric"], "interval metric"),
            kind=_enum(MetricKind, payload["kind"], "interval kind"),
            unit=_enum(MetricUnit, payload["unit"], "interval unit"),
            status=_enum(
                ConfidenceIntervalStatus,
                payload["status"],
                "interval status",
            ),
            lower_bound=_optional_integer(payload["lower_bound"], "interval lower"),
            upper_bound=_optional_integer(payload["upper_bound"], "interval upper"),
            seed=payload["seed"],
            iterations=payload["iterations"],
            confidence_level_ppm=payload["confidence_level_ppm"],
            coverage=BootstrapCoverageV1.from_dict(payload["coverage"]),
        )
        if payload["interval_id"] != result.interval_id:
            raise _error("ConfidenceIntervalV1 interval_id is not canonical")
        return result


def _bootstrap_coverage(
    values: Sequence[CaseContributionV1], incomplete: int = 0
) -> BootstrapCoverageV1:
    statuses = [item.status for item in values]
    return BootstrapCoverageV1(
        total_case_count=len(values),
        available_case_count=sum(
            status is StatisticsMetricStatus.AVAILABLE for status in statuses
        ),
        zero_denominator_case_count=sum(
            status is StatisticsMetricStatus.ZERO_DENOMINATOR for status in statuses
        ),
        not_scorable_case_count=sum(
            status is StatisticsMetricStatus.NOT_SCORABLE for status in statuses
        ),
        ungraded_case_count=sum(
            status is StatisticsMetricStatus.UNGRADED for status in statuses
        ),
        failure_excluded_case_count=sum(
            status is StatisticsMetricStatus.FAILURE_EXCLUDED for status in statuses
        ),
        missing_case_count=sum(
            status is StatisticsMetricStatus.MISSING for status in statuses
        ),
        incomplete_replicate_count=incomplete,
    )


def _null_interval_status(coverage: BootstrapCoverageV1) -> ConfidenceIntervalStatus:
    if coverage.ungraded_case_count:
        return ConfidenceIntervalStatus.UNGRADED
    if coverage.failure_excluded_case_count:
        return ConfidenceIntervalStatus.FAILURE_EXCLUDED
    if coverage.not_scorable_case_count:
        return ConfidenceIntervalStatus.NOT_SCORABLE
    if coverage.zero_denominator_case_count:
        return ConfidenceIntervalStatus.ZERO_DENOMINATOR
    return ConfidenceIntervalStatus.MISSING


def _sample_metric_value(
    metric: CoreMetric,
    kind: MetricKind,
    sample: Sequence[CaseContributionV1],
) -> Optional[int]:
    if metric is CoreMetric.ISSUE_F1:
        sums: Dict[CoreMetric, tuple[int, int, int]] = {}
        for source_metric in (CoreMetric.ISSUE_PRECISION, CoreMetric.ISSUE_RECALL):
            components = [
                component
                for item in sample
                for component in item.derived_contributions
                if component.metric is source_metric
                and component.status is StatisticsMetricStatus.AVAILABLE
            ]
            if not components:
                return None
            sums[source_metric] = (
                sum(int(item.numerator) for item in components),
                sum(int(item.denominator) for item in components),
                len(components),
            )
        p_num, p_den, _ = sums[CoreMetric.ISSUE_PRECISION]
        r_num, r_den, _ = sums[CoreMetric.ISSUE_RECALL]
        if p_den == 0 or r_den == 0:
            return None
        numerator = 2 * p_num * r_num
        denominator = p_num * r_den + r_num * p_den
        if denominator == 0:
            return 0
        return _ratio_ppm(numerator, denominator)
    available = [
        item for item in sample if item.status is StatisticsMetricStatus.AVAILABLE
    ]
    if not available:
        return None
    numerator = sum(int(item.numerator) for item in available)
    if kind is MetricKind.COUNT:
        return numerator
    denominator = sum(int(item.denominator) for item in available)
    return _ratio_ppm(numerator, denominator)


def paired_bootstrap_interval(
    case_contributions: Sequence[CaseContributionV1],
    *,
    seed: int,
    iterations: int,
    confidence_level_ppm: int,
) -> ConfidenceIntervalV1:
    """Return a deterministic Case-clustered percentile interval.

    Every draw samples complete Case records with replacement and recomputes
    the ratio from sampled numerators and denominators.  Invalid replicates are
    counted and make the interval explicitly unavailable; they are never
    silently discarded.
    """

    _integer(seed, "bootstrap seed", maximum=MAX_BOOTSTRAP_SEED)
    _integer(
        iterations,
        "bootstrap iterations",
        minimum=1,
        maximum=MAX_BOOTSTRAP_ITERATIONS,
    )
    _integer(
        confidence_level_ppm,
        "bootstrap confidence_level_ppm",
        minimum=1,
        maximum=PPM_SCALE - 1,
    )
    try:
        raw = tuple(case_contributions)
    except TypeError as exc:
        raise _error("Case contributions must be a sequence") from exc
    if (
        not raw
        or len(raw) > MAX_BOOTSTRAP_CASES
        or any(type(item) is not CaseContributionV1 for item in raw)
    ):
        raise _error("Case contribution population is empty, oversized, or invalid")
    if len(raw) * iterations > MAX_BOOTSTRAP_DRAWS:
        raise _error("bootstrap draw budget is exceeded")
    values = tuple(
        sorted(
            raw,
            key=lambda item: (
                item.task_id,
                item.case_version,
                item.canonical_case_digest,
                -1 if item.trial_index is None else item.trial_index,
            ),
        )
    )
    identities = tuple(
        (item.task_id, item.case_version, item.canonical_case_digest) for item in values
    )
    first = values[0]
    if len(identities) != len(set(identities)):
        raise _error("bootstrap population contains duplicate Case records")
    if any(
        item.metric is not first.metric
        or item.kind is not first.kind
        or item.unit is not first.unit
        or item.direction is not first.direction
        or item.trial_index != first.trial_index
        for item in values
    ):
        raise _error("bootstrap Case contributions are not compatible")

    coverage = _bootstrap_coverage(values)

    def result(
        status: ConfidenceIntervalStatus,
        lower: Optional[int] = None,
        upper: Optional[int] = None,
        *,
        final_coverage: BootstrapCoverageV1 = coverage,
    ) -> ConfidenceIntervalV1:
        return ConfidenceIntervalV1(
            metric=first.metric,
            kind=first.kind,
            unit=first.unit,
            status=status,
            lower_bound=lower,
            upper_bound=upper,
            seed=seed,
            iterations=iterations,
            confidence_level_ppm=confidence_level_ppm,
            coverage=final_coverage,
        )

    if len(values) < 2:
        return result(ConfidenceIntervalStatus.INSUFFICIENT_CASE_POPULATION)
    if _sample_metric_value(first.metric, first.kind, values) is None:
        return result(_null_interval_status(coverage))

    generator = random.Random(seed)
    replicates: list[int] = []
    incomplete = 0
    for _iteration in range(iterations):
        sample = tuple(values[generator.randrange(len(values))] for _ in values)
        sampled_value = _sample_metric_value(first.metric, first.kind, sample)
        if sampled_value is None:
            incomplete += 1
        else:
            replicates.append(sampled_value)
    final_coverage = _bootstrap_coverage(values, incomplete)
    if incomplete:
        return result(
            ConfidenceIntervalStatus.INCOMPLETE_REPLICATES,
            final_coverage=final_coverage,
        )
    replicates.sort()
    tail_ppm = (PPM_SCALE - confidence_level_ppm) // 2
    lower_index = tail_ppm * (iterations - 1) // PPM_SCALE
    upper_numerator = (PPM_SCALE - tail_ppm) * (iterations - 1)
    upper_index = (upper_numerator + PPM_SCALE - 1) // PPM_SCALE
    return result(
        ConfidenceIntervalStatus.AVAILABLE,
        replicates[lower_index],
        replicates[upper_index],
        final_coverage=final_coverage,
    )


@dataclass(frozen=True)
class DispersionCoverageV1(_JsonModel):
    total_replicate_count: int
    available_replicate_count: int
    zero_denominator_count: int
    not_scorable_count: int
    ungraded_count: int
    failure_excluded_count: int
    missing_count: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _integer(getattr(self, name), f"DispersionCoverageV1.{name}")
        if (
            self.available_replicate_count
            + self.zero_denominator_count
            + self.not_scorable_count
            + self.ungraded_count
            + self.failure_excluded_count
            + self.missing_count
            != self.total_replicate_count
        ):
            raise _error("Dispersion statuses do not cover replicates")

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, value: Any) -> "DispersionCoverageV1":
        fields = tuple(cls.__dataclass_fields__)
        return cls(**_exact(value, fields, "DispersionCoverageV1"))


@dataclass(frozen=True)
class DispersionV1(_JsonModel):
    unit: MetricUnit
    minimum: Optional[int]
    maximum: Optional[int]
    standard_deviation: Optional[float]
    null_reason: Optional[DispersionNullReason]
    coverage: DispersionCoverageV1

    def __post_init__(self) -> None:
        if type(self.unit) is not MetricUnit:
            raise _error("DispersionV1.unit is invalid")
        if type(self.coverage) is not DispersionCoverageV1:
            raise _error("DispersionV1.coverage is invalid")
        available = self.coverage.available_replicate_count
        if available == 0:
            if (
                self.minimum is not None
                or self.maximum is not None
                or self.standard_deviation is not None
                or self.null_reason is not DispersionNullReason.NO_AVAILABLE_REPLICATES
            ):
                raise _error("empty Dispersion is not canonical")
            return
        minimum = _integer(self.minimum, "DispersionV1.minimum")
        maximum = _integer(self.maximum, "DispersionV1.maximum")
        if minimum > maximum:
            raise _error("Dispersion minimum exceeds maximum")
        if available == 1:
            if (
                self.standard_deviation is not None
                or self.null_reason is not DispersionNullReason.INSUFFICIENT_REPLICATES
            ):
                raise _error("single-replicate Dispersion is not canonical")
        elif (
            type(self.standard_deviation) is not float
            or not math.isfinite(self.standard_deviation)
            or self.standard_deviation < 0
            or self.null_reason is not None
        ):
            raise _error("available Dispersion standard deviation is invalid")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unit": self.unit.value,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "standard_deviation": self.standard_deviation,
            "null_reason": (
                None if self.null_reason is None else self.null_reason.value
            ),
            "coverage": self.coverage.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> "DispersionV1":
        payload = _exact(
            value,
            (
                "unit",
                "minimum",
                "maximum",
                "standard_deviation",
                "null_reason",
                "coverage",
            ),
            "DispersionV1",
        )
        deviation = payload["standard_deviation"]
        if deviation is not None and type(deviation) not in (int, float):
            raise _error("Dispersion standard_deviation must be numeric or null")
        return cls(
            unit=_enum(MetricUnit, payload["unit"], "dispersion unit"),
            minimum=_optional_integer(payload["minimum"], "dispersion minimum"),
            maximum=_optional_integer(payload["maximum"], "dispersion maximum"),
            standard_deviation=(None if deviation is None else float(deviation)),
            null_reason=(
                None
                if payload["null_reason"] is None
                else _enum(
                    DispersionNullReason,
                    payload["null_reason"],
                    "dispersion null_reason",
                )
            ),
            coverage=DispersionCoverageV1.from_dict(payload["coverage"]),
        )


def _dispersion(
    unit: MetricUnit, projections: Sequence["TrialMetricProjectionV1"]
) -> DispersionV1:
    statuses = [item.status for item in projections]
    values = [
        int(item.value)
        for item in projections
        if item.status is StatisticsMetricStatus.AVAILABLE and item.value is not None
    ]
    coverage = DispersionCoverageV1(
        total_replicate_count=len(projections),
        available_replicate_count=len(values),
        zero_denominator_count=sum(
            status is StatisticsMetricStatus.ZERO_DENOMINATOR for status in statuses
        ),
        not_scorable_count=sum(
            status is StatisticsMetricStatus.NOT_SCORABLE for status in statuses
        ),
        ungraded_count=sum(
            status is StatisticsMetricStatus.UNGRADED for status in statuses
        ),
        failure_excluded_count=sum(
            status is StatisticsMetricStatus.FAILURE_EXCLUDED for status in statuses
        ),
        missing_count=sum(
            status is StatisticsMetricStatus.MISSING for status in statuses
        ),
    )
    if not values:
        return DispersionV1(
            unit,
            None,
            None,
            None,
            DispersionNullReason.NO_AVAILABLE_REPLICATES,
            coverage,
        )
    if len(values) == 1:
        return DispersionV1(
            unit,
            values[0],
            values[0],
            None,
            DispersionNullReason.INSUFFICIENT_REPLICATES,
            coverage,
        )
    return DispersionV1(
        unit,
        min(values),
        max(values),
        float(stdlib_statistics.stdev(values)),
        None,
        coverage,
    )


@dataclass(frozen=True)
class TrialMetricProjectionV1(_JsonModel):
    trial_index: int
    metric: CoreMetric
    kind: MetricKind
    unit: MetricUnit
    direction: MetricDirection
    status: StatisticsMetricStatus
    numerator: Optional[int]
    denominator: Optional[int]
    value: Optional[int]
    coverage: StatisticsCoverageV1
    case_contributions: Tuple[CaseContributionV1, ...]

    def __post_init__(self) -> None:
        _integer(
            self.trial_index,
            "TrialMetricProjectionV1.trial_index",
            minimum=1,
            maximum=MAX_TRIAL_COUNT,
        )
        _validate_metric_value(
            metric=self.metric,
            kind=self.kind,
            unit=self.unit,
            direction=self.direction,
            status=self.status,
            numerator=self.numerator,
            denominator=self.denominator,
            value=self.value,
            context="TrialMetricProjectionV1",
        )
        if type(self.coverage) is not StatisticsCoverageV1:
            raise _error("TrialMetricProjectionV1.coverage is invalid")
        contributions = tuple(self.case_contributions)
        identities = tuple(
            (item.task_id, item.case_version, item.canonical_case_digest)
            for item in contributions
        )
        if (
            not contributions
            or any(
                type(item) is not CaseContributionV1
                or item.metric is not self.metric
                or item.trial_index != self.trial_index
                for item in contributions
            )
            or contributions
            != tuple(
                sorted(
                    contributions,
                    key=lambda item: (
                        item.task_id,
                        item.case_version,
                        item.canonical_case_digest,
                    ),
                )
            )
            or len(identities) != len(set(identities))
        ):
            raise _error("Trial metric Case contributions are not canonical")
        object.__setattr__(self, "case_contributions", contributions)

    def _identity_dict(self) -> Dict[str, Any]:
        return {
            "trial_index": self.trial_index,
            "metric": self.metric.value,
            "kind": self.kind.value,
            "unit": self.unit.value,
            "direction": self.direction.value,
            "status": self.status.value,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value": self.value,
            "coverage": self.coverage.to_dict(),
            "case_contributions": [
                item.to_dict() for item in self.case_contributions
            ],
        }

    @property
    def projection_id(self) -> str:
        return stable_id("trial-metric-projection-v1", self._identity_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {**self._identity_dict(), "projection_id": self.projection_id}

    @classmethod
    def from_dict(cls, value: Any) -> "TrialMetricProjectionV1":
        fields = (
            "projection_id",
            "trial_index",
            "metric",
            "kind",
            "unit",
            "direction",
            "status",
            "numerator",
            "denominator",
            "value",
            "coverage",
            "case_contributions",
        )
        payload = _exact(value, fields, "TrialMetricProjectionV1")
        cases = _array(
            payload["case_contributions"],
            "Trial projection Case contributions",
            MAX_BOOTSTRAP_CASES,
        )
        result = cls(
            trial_index=payload["trial_index"],
            metric=_enum(CoreMetric, payload["metric"], "projection metric"),
            kind=_enum(MetricKind, payload["kind"], "projection kind"),
            unit=_enum(MetricUnit, payload["unit"], "projection unit"),
            direction=_enum(
                MetricDirection, payload["direction"], "projection direction"
            ),
            status=_enum(
                StatisticsMetricStatus, payload["status"], "projection status"
            ),
            numerator=_optional_integer(payload["numerator"], "projection numerator"),
            denominator=_optional_integer(
                payload["denominator"], "projection denominator"
            ),
            value=_optional_integer(payload["value"], "projection value"),
            coverage=StatisticsCoverageV1.from_dict(payload["coverage"]),
            case_contributions=tuple(CaseContributionV1.from_dict(item) for item in cases),
        )
        if payload["projection_id"] != result.projection_id:
            raise _error("TrialMetricProjectionV1 projection_id is not canonical")
        return result


@dataclass(frozen=True)
class StatisticsMetricV1(_JsonModel):
    metric: CoreMetric
    kind: MetricKind
    unit: MetricUnit
    direction: MetricDirection
    status: StatisticsMetricStatus
    numerator: Optional[int]
    denominator: Optional[int]
    value: Optional[int]
    coverage: StatisticsCoverageV1
    dispersion: DispersionV1
    confidence_interval: ConfidenceIntervalV1
    case_contributions: Tuple[CaseContributionV1, ...]

    def __post_init__(self) -> None:
        _validate_metric_value(
            metric=self.metric,
            kind=self.kind,
            unit=self.unit,
            direction=self.direction,
            status=self.status,
            numerator=self.numerator,
            denominator=self.denominator,
            value=self.value,
            context="StatisticsMetricV1",
        )
        if type(self.coverage) is not StatisticsCoverageV1:
            raise _error("StatisticsMetricV1.coverage is invalid")
        if type(self.dispersion) is not DispersionV1 or self.dispersion.unit is not self.unit:
            raise _error("StatisticsMetricV1.dispersion is invalid")
        interval = self.confidence_interval
        if (
            type(interval) is not ConfidenceIntervalV1
            or interval.metric is not self.metric
            or interval.kind is not self.kind
            or interval.unit is not self.unit
        ):
            raise _error("StatisticsMetricV1 confidence interval is incompatible")
        contributions = tuple(self.case_contributions)
        identities = tuple(
            (item.task_id, item.case_version, item.canonical_case_digest)
            for item in contributions
        )
        if (
            not contributions
            or any(
                type(item) is not CaseContributionV1
                or item.metric is not self.metric
                or item.trial_index is not None
                for item in contributions
            )
            or contributions
            != tuple(
                sorted(
                    contributions,
                    key=lambda item: (
                        item.task_id,
                        item.case_version,
                        item.canonical_case_digest,
                    ),
                )
            )
            or len(identities) != len(set(identities))
        ):
            raise _error("Statistics metric Case contributions are not canonical")
        if interval.coverage.total_case_count != len(contributions):
            raise _error("Statistics metric interval Case coverage differs")
        object.__setattr__(self, "case_contributions", contributions)

    def _identity_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric.value,
            "kind": self.kind.value,
            "unit": self.unit.value,
            "direction": self.direction.value,
            "status": self.status.value,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "value": self.value,
            "coverage": self.coverage.to_dict(),
            "dispersion": self.dispersion.to_dict(),
            "confidence_interval": self.confidence_interval.to_dict(),
            "case_contributions": [
                item.to_dict() for item in self.case_contributions
            ],
        }

    @property
    def metric_id(self) -> str:
        return stable_id("statistics-metric-v1", self._identity_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {**self._identity_dict(), "metric_id": self.metric_id}

    @classmethod
    def from_dict(cls, value: Any) -> "StatisticsMetricV1":
        fields = (
            "metric_id",
            "metric",
            "kind",
            "unit",
            "direction",
            "status",
            "numerator",
            "denominator",
            "value",
            "coverage",
            "dispersion",
            "confidence_interval",
            "case_contributions",
        )
        payload = _exact(value, fields, "StatisticsMetricV1")
        cases = _array(
            payload["case_contributions"],
            "Statistics metric Case contributions",
            MAX_BOOTSTRAP_CASES,
        )
        result = cls(
            metric=_enum(CoreMetric, payload["metric"], "statistics metric"),
            kind=_enum(MetricKind, payload["kind"], "statistics kind"),
            unit=_enum(MetricUnit, payload["unit"], "statistics unit"),
            direction=_enum(
                MetricDirection, payload["direction"], "statistics direction"
            ),
            status=_enum(
                StatisticsMetricStatus, payload["status"], "statistics status"
            ),
            numerator=_optional_integer(payload["numerator"], "statistics numerator"),
            denominator=_optional_integer(
                payload["denominator"], "statistics denominator"
            ),
            value=_optional_integer(payload["value"], "statistics value"),
            coverage=StatisticsCoverageV1.from_dict(payload["coverage"]),
            dispersion=DispersionV1.from_dict(payload["dispersion"]),
            confidence_interval=ConfidenceIntervalV1.from_dict(
                payload["confidence_interval"]
            ),
            case_contributions=tuple(CaseContributionV1.from_dict(item) for item in cases),
        )
        if payload["metric_id"] != result.metric_id:
            raise _error("StatisticsMetricV1 metric_id is not canonical")
        return result


@dataclass(frozen=True)
class RunStatisticsV1(_JsonModel):
    schema_version: str
    source_binding: AnalysisSourceBinding
    trial_count: int
    metrics: Tuple[StatisticsMetricV1, ...]
    trial_metrics: Tuple[TrialMetricProjectionV1, ...]
    bootstrap_policy: StatisticsPolicyV1

    def __post_init__(self) -> None:
        if self.schema_version != RUN_STATISTICS_SCHEMA_VERSION:
            raise _error("RunStatisticsV1 schema_version is unsupported")
        if type(self.source_binding) is not AnalysisSourceBinding:
            raise _error("RunStatisticsV1 source_binding is invalid")
        _integer(
            self.trial_count,
            "RunStatisticsV1.trial_count",
            minimum=1,
            maximum=MAX_TRIAL_COUNT,
        )
        if type(self.bootstrap_policy) is not StatisticsPolicyV1:
            raise _error("RunStatisticsV1 bootstrap_policy is invalid")
        metrics = tuple(self.metrics)
        expected_metrics = tuple(sorted(CoreMetric, key=lambda item: item.value))
        if (
            any(type(item) is not StatisticsMetricV1 for item in metrics)
            or tuple(item.metric for item in metrics) != expected_metrics
        ):
            raise _error("RunStatisticsV1 metrics do not cover CoreMetric canonically")
        projections = tuple(self.trial_metrics)
        expected_projection_keys = tuple(
            (index, metric)
            for index in range(1, self.trial_count + 1)
            for metric in expected_metrics
        )
        if (
            any(type(item) is not TrialMetricProjectionV1 for item in projections)
            or tuple((item.trial_index, item.metric) for item in projections)
            != expected_projection_keys
        ):
            raise _error("RunStatisticsV1 Trial projections are incomplete/noncanonical")
        source_trial_count = len(self.source_binding.trial_score_digests)
        case_count = len(metrics[0].case_contributions)
        if case_count * self.trial_count != source_trial_count:
            raise _error("RunStatisticsV1 source Trial coverage is inconsistent")
        metric_cases = tuple(
            (item.task_id, item.case_version, item.canonical_case_digest)
            for item in metrics[0].case_contributions
        )
        for metric in metrics:
            if (
                metric.coverage.total_trial_count != source_trial_count
                or tuple(
                    (item.task_id, item.case_version, item.canonical_case_digest)
                    for item in metric.case_contributions
                )
                != metric_cases
                or any(
                    item.coverage.total_trial_count != self.trial_count
                    for item in metric.case_contributions
                )
                or metric.confidence_interval.seed
                != self.bootstrap_policy.bootstrap_seed
                or metric.confidence_interval.iterations
                != self.bootstrap_policy.bootstrap_iterations
                or metric.confidence_interval.confidence_level_ppm
                != self.bootstrap_policy.confidence_level_ppm
            ):
                raise _error("RunStatisticsV1 metric binding/coverage is inconsistent")
            metric_projections = tuple(
                item for item in projections if item.metric is metric.metric
            )
            if any(
                item.coverage.total_trial_count != case_count
                or tuple(
                    (
                        contribution.task_id,
                        contribution.case_version,
                        contribution.canonical_case_digest,
                    )
                    for contribution in item.case_contributions
                )
                != metric_cases
                or any(
                    contribution.coverage.total_trial_count != 1
                    for contribution in item.case_contributions
                )
                for item in metric_projections
            ) or _dispersion(metric.unit, metric_projections) != metric.dispersion:
                raise _error("RunStatisticsV1 dispersion/projection coverage differs")
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "trial_metrics", projections)
        canonical_json_bytes(self._identity_dict())

    def _identity_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_binding": self.source_binding.to_dict(),
            "trial_count": self.trial_count,
            "metrics": [item.to_dict() for item in self.metrics],
            "trial_metrics": [item.to_dict() for item in self.trial_metrics],
            "bootstrap_policy": self.bootstrap_policy.to_dict(),
        }

    @property
    def statistics_id(self) -> str:
        return stable_id("run-statistics-v1", self._identity_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {**self._identity_dict(), "statistics_id": self.statistics_id}

    @classmethod
    def from_dict(cls, value: Any) -> "RunStatisticsV1":
        fields = (
            "statistics_id",
            "schema_version",
            "source_binding",
            "trial_count",
            "metrics",
            "trial_metrics",
            "bootstrap_policy",
        )
        payload = _exact(value, fields, "RunStatisticsV1")
        metrics = _array(payload["metrics"], "RunStatisticsV1.metrics", len(CoreMetric))
        projections = _array(
            payload["trial_metrics"],
            "RunStatisticsV1.trial_metrics",
            MAX_TRIAL_COUNT * len(CoreMetric),
        )
        result = cls(
            schema_version=payload["schema_version"],
            source_binding=AnalysisSourceBinding.from_dict(payload["source_binding"]),
            trial_count=payload["trial_count"],
            metrics=tuple(StatisticsMetricV1.from_dict(item) for item in metrics),
            trial_metrics=tuple(
                TrialMetricProjectionV1.from_dict(item) for item in projections
            ),
            bootstrap_policy=StatisticsPolicyV1.from_dict(payload["bootstrap_policy"]),
        )
        if payload["statistics_id"] != result.statistics_id:
            raise _error("RunStatisticsV1 statistics_id is not canonical")
        return result

    @classmethod
    def from_json(cls, data: Any) -> "RunStatisticsV1":
        return _from_json(cls, data, "RunStatisticsV1 JSON")

    def metric(self, metric: CoreMetric) -> StatisticsMetricV1:
        if type(metric) is not CoreMetric:
            raise TypeError("metric must be CoreMetric")
        return next(item for item in self.metrics if item.metric is metric)

    def trial_metric(
        self, trial_index: int, metric: CoreMetric
    ) -> TrialMetricProjectionV1:
        return next(
            item
            for item in self.trial_metrics
            if item.trial_index == trial_index and item.metric is metric
        )


def _source_coverages(value: MetricAggregate) -> Tuple[MetricSourceCoverageV1, ...]:
    if value.coverage is not None:
        return (MetricSourceCoverageV1.from_metric_coverage(value.metric, value.coverage),)
    return tuple(
        MetricSourceCoverageV1.from_metric_coverage(item.metric, item.coverage)
        for item in value.derived_coverages
    )


def _sum_contribution(scores: Sequence[TrialScore], metric: CoreMetric) -> tuple[int, int]:
    contributions = [item.contribution(metric) for item in scores]
    if any(item.numerator is None or item.denominator is None for item in contributions):
        raise _error(f"reliability metric {metric.value} is unexpectedly unavailable")
    return (
        sum(int(item.numerator) for item in contributions),
        sum(int(item.denominator) for item in contributions),
    )


def _statistics_coverage(
    value: MetricAggregate, scores: Sequence[TrialScore]
) -> StatisticsCoverageV1:
    failures, trial_population = _sum_contribution(scores, CoreMetric.AGENT_FAILURE_RATE)
    judge_failures, judge_requests = _sum_contribution(
        scores, CoreMetric.JUDGE_FAILURE_RATE
    )
    judge_ungraded, ungraded_requests = _sum_contribution(
        scores, CoreMetric.JUDGE_UNGRADED_RATE
    )
    semantic_unknown, unknown_requests = _sum_contribution(
        scores, CoreMetric.JUDGE_SEMANTIC_UNKNOWN_RATE
    )
    if (
        trial_population != len(scores)
        or ungraded_requests != judge_requests
        or unknown_requests != judge_requests
        or judge_failures + judge_ungraded > judge_requests
    ):
        raise _error("TrialScore reliability coverage is inconsistent")
    return StatisticsCoverageV1(
        total_trial_count=len(scores),
        completed_trial_count=len(scores) - failures,
        agent_failure_count=failures,
        judge_request_count=judge_requests,
        judge_graded_count=judge_requests - judge_failures - judge_ungraded,
        judge_failure_count=judge_failures,
        judge_ungraded_count=judge_ungraded,
        judge_semantic_unknown_count=semantic_unknown,
        metric_sources=_source_coverages(value),
    )


def _aggregate_scores(
    scores: Sequence[TrialScore], *, planned_trial_count: int
) -> tuple[Any, Tuple[Any, ...], Dict[str, Tuple[TrialScore, ...]]]:
    ordered = tuple(sorted(scores, key=lambda item: (item.task_id, item.trial_index)))
    by_task: Dict[str, list[TrialScore]] = {}
    for score in ordered:
        by_task.setdefault(score.task_id, []).append(score)
    frozen_by_task = {
        task_id: tuple(items) for task_id, items in sorted(by_task.items())
    }
    aggregator = MetricsAggregator()
    cases = tuple(
        aggregator.aggregate_case(
            items,
            planned_trial_count=planned_trial_count,
        )
        for items in frozen_by_task.values()
    )
    aggregate = aggregator.aggregate_cases(cases, source_trials=ordered)
    return aggregate, cases, frozen_by_task


def _derived_case_contributions(case_score: Any) -> Tuple[DerivedCaseContributionV1, ...]:
    result = []
    for metric in (CoreMetric.ISSUE_PRECISION, CoreMetric.ISSUE_RECALL):
        value = case_score.metric(metric)
        result.append(
            DerivedCaseContributionV1(
                metric=metric,
                status=_metric_status(value),
                numerator=value.numerator,
                denominator=value.denominator,
            )
        )
    return tuple(result)


def _case_contribution(
    case_score: Any,
    scores: Sequence[TrialScore],
    metric: CoreMetric,
    *,
    trial_index: Optional[int],
) -> CaseContributionV1:
    value = case_score.metric(metric)
    return CaseContributionV1(
        task_id=case_score.task_id,
        case_version=case_score.case_version,
        canonical_case_digest=case_score.canonical_case_digest,
        trial_index=trial_index,
        metric=metric,
        kind=value.kind,
        unit=_metric_unit(metric),
        direction=_metric_direction(metric),
        status=_metric_status(value),
        numerator=value.numerator,
        denominator=value.denominator,
        value=_metric_value(value),
        coverage=_statistics_coverage(value, scores),
        derived_contributions=(
            _derived_case_contributions(case_score)
            if metric is CoreMetric.ISSUE_F1
            else ()
        ),
    )


def _projection(
    aggregate: Any,
    cases: Sequence[Any],
    scores_by_task: Mapping[str, Tuple[TrialScore, ...]],
    metric: CoreMetric,
    trial_index: int,
) -> TrialMetricProjectionV1:
    value = aggregate.metric(metric)
    all_scores = tuple(
        score for task_id in sorted(scores_by_task) for score in scores_by_task[task_id]
    )
    contributions = tuple(
        _case_contribution(
            case,
            scores_by_task[case.task_id],
            metric,
            trial_index=trial_index,
        )
        for case in cases
    )
    return TrialMetricProjectionV1(
        trial_index=trial_index,
        metric=metric,
        kind=value.kind,
        unit=_metric_unit(metric),
        direction=_metric_direction(metric),
        status=_metric_status(value),
        numerator=value.numerator,
        denominator=value.denominator,
        value=_metric_value(value),
        coverage=_statistics_coverage(value, all_scores),
        case_contributions=contributions,
    )


def compute_run_statistics(
    bundle: Any,
    *,
    run_config: EvalRunConfig,
    case_snapshot: RunCaseSnapshot,
    policy: StatisticsPolicyV1,
) -> RunStatisticsV1:
    """Bind a hydrated Evaluation and organize its existing Core metrics."""

    source_binding = bind_analysis_source(
        bundle,
        run_config=run_config,
        case_snapshot=case_snapshot,
    )
    if type(policy) is not StatisticsPolicyV1:
        raise TypeError("policy must be StatisticsPolicyV1")
    canonical_policy = StatisticsPolicyV1.from_dict(policy.to_dict())
    scores = tuple(
        sorted(
            (item.trial_score for item in bundle.trials),
            key=lambda item: (item.task_id, item.trial_index),
        )
    )
    if len(scores) != len(source_binding.trial_score_digests):
        raise _error("bound TrialScore population changed after source validation")

    projection_values: list[TrialMetricProjectionV1] = []
    for trial_index in range(1, run_config.trial_count + 1):
        index_scores = tuple(item for item in scores if item.trial_index == trial_index)
        aggregate, cases, by_task = _aggregate_scores(
            index_scores,
            planned_trial_count=run_config.trial_count,
        )
        for metric in sorted(CoreMetric, key=lambda item: item.value):
            projection_values.append(
                _projection(
                    aggregate,
                    cases,
                    by_task,
                    metric,
                    trial_index,
                )
            )

    aggregate, cases, by_task = _aggregate_scores(
        scores,
        planned_trial_count=run_config.trial_count,
    )
    metrics: list[StatisticsMetricV1] = []
    for metric in sorted(CoreMetric, key=lambda item: item.value):
        value = aggregate.metric(metric)
        contributions = tuple(
            _case_contribution(
                case,
                by_task[case.task_id],
                metric,
                trial_index=None,
            )
            for case in cases
        )
        interval = paired_bootstrap_interval(
            contributions,
            seed=canonical_policy.bootstrap_seed,
            iterations=canonical_policy.bootstrap_iterations,
            confidence_level_ppm=canonical_policy.confidence_level_ppm,
        )
        metric_projections = tuple(
            item for item in projection_values if item.metric is metric
        )
        metrics.append(
            StatisticsMetricV1(
                metric=metric,
                kind=value.kind,
                unit=_metric_unit(metric),
                direction=_metric_direction(metric),
                status=_metric_status(value),
                numerator=value.numerator,
                denominator=value.denominator,
                value=_metric_value(value),
                coverage=_statistics_coverage(value, scores),
                dispersion=_dispersion(_metric_unit(metric), metric_projections),
                confidence_interval=interval,
                case_contributions=contributions,
            )
        )
    return RunStatisticsV1(
        schema_version=RUN_STATISTICS_SCHEMA_VERSION,
        source_binding=source_binding,
        trial_count=run_config.trial_count,
        metrics=tuple(metrics),
        trial_metrics=tuple(projection_values),
        bootstrap_policy=canonical_policy,
    )


__all__ = [
    "RUN_STATISTICS_SCHEMA_VERSION",
    "STATISTICS_ALGORITHM_VERSION",
    "MAX_BOOTSTRAP_SEED",
    "MAX_BOOTSTRAP_ITERATIONS",
    "MAX_BOOTSTRAP_CASES",
    "StatisticsError",
    "MetricUnit",
    "MetricDirection",
    "StatisticsMetricStatus",
    "DispersionNullReason",
    "ConfidenceIntervalStatus",
    "StatisticsPolicyV1",
    "MetricSourceCoverageV1",
    "StatisticsCoverageV1",
    "DerivedCaseContributionV1",
    "CaseContributionV1",
    "BootstrapCoverageV1",
    "ConfidenceIntervalV1",
    "DispersionCoverageV1",
    "DispersionV1",
    "TrialMetricProjectionV1",
    "StatisticsMetricV1",
    "RunStatisticsV1",
    "paired_bootstrap_interval",
    "compute_run_statistics",
]
