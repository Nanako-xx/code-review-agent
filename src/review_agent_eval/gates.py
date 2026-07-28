"""Pre-registered, source-bound regression gates.

The gate layer is deliberately a small protocol boundary.  A policy is
prepared from a verified baseline and an immutable candidate :class:`EvalRunConfig`
before candidate results exist.  Evaluation later consumes only the typed
comparison and the explicitly named calibration results.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
import hashlib
import re
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .analysis_artifacts import (
    AnalysisArtifactStore,
    AnalysisReceipt,
    AnalysisSourceBinding,
)
from .artifacts import ArtifactIntegrityError
from .calibration import CalibrationResultV1, CalibrationStatus
from .cases import CaseSplit, SuiteKind
from .comparison import (
    ComparisonStatus,
    MetricDeltaV1,
    RunComparisonV1,
    VerifiedRunEvaluation,
)
from .config import EvalRunConfig, MAX_TRIAL_COUNT, validate_path_segment, validate_run_id
from .metrics import CoreMetric, MetricKind, MetricSourceStatus
from .models import (
    CaseOrigin,
    IntentAuthority,
    SchemaError,
    _JsonModel,
    _strict_json_loads,
    canonical_json_bytes,
    canonical_sha256,
    stable_id,
)
from .statistics import (
    MetricUnit,
    PPM_SCALE,
    StatisticsMetricStatus,
)
from .judge import JudgeTask


GATE_POLICY_SCHEMA_VERSION = "gate_policy_v1"
GATE_RESULT_SCHEMA_VERSION = "gate_result_v1"
GATE_ALGORITHM_VERSION = "preregistered-regression-gate-v1"
MAX_GATE_CONSTRAINTS = len(CoreMetric) * 6
MAX_GATE_CALIBRATIONS = 16
MAX_GATE_REFS = 65_536
MAX_GATE_INTEGER = (1 << 63) - 1
MAX_GATE_BYTES = 256 * 1024 * 1024

# This is deliberately an object identity rather than a serializable marker.
# A FrozenGatePolicy can only be issued by the private factory below, after an
# AnalysisArtifactStore has replayed the baseline/RunConfig preparation and
# verified the receipt.  The public evaluator checks this seal before it ever
# unwraps the policy.
_FROZEN_GATE_POLICY_SEAL = object()
_LEGACY_GATE_HYDRATION: ContextVar[bool] = ContextVar(
    "legacy_gate_hydration", default=False
)


class GateError(ValueError):
    """A gate policy, result, or source replay is invalid."""


class GateEligibility(str, Enum):
    RELEASE_BLOCKING = "release_blocking"
    DIAGNOSTIC_ONLY = "diagnostic_only"


class GateDecision(str, Enum):
    PROMOTE = "promote"
    BLOCK = "block"
    INELIGIBLE = "ineligible"


class GateCheckStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_COMPARABLE = "not_comparable"
    NOT_SCORABLE = "not_scorable"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    NOT_CONFIGURED = "not_configured"
    PENDING = "pending"
    # Artifact compatibility only; evaluators never emit this legacy spelling.
    INELIGIBLE = "ineligible"


class GateConstraintScope(str, Enum):
    CANDIDATE_ABSOLUTE = "candidate_absolute"
    BASELINE_DELTA = "baseline_delta"
    CASE_ABSOLUTE = "case_absolute"
    TRIAL_ABSOLUTE = "trial_absolute"
    CASE_DELTA = "case_delta"
    TRIAL_DELTA = "trial_delta"


class GateOperator(str, Enum):
    AT_LEAST = "at_least"
    AT_MOST = "at_most"


class GateCheckReason(str, Enum):
    POLICY_MISMATCH = "policy_mismatch"
    NOT_COMPARABLE = "not_comparable"
    NOT_SCORABLE = "not_scorable"
    NOT_CONFIGURED = "not_configured"
    ZERO_DENOMINATOR = "zero_denominator"
    UNGRADED = "ungraded"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    FAILED_COVERAGE = "insufficient_coverage"
    MISSING_VALUE = "missing_value"
    AUTHORITY_INSUFFICIENT = "authority_insufficient"
    CALIBRATION_MISSING = "calibration_missing"
    CALIBRATION_NOT_ELIGIBLE = "calibration_not_eligible"
    CALIBRATION_PENDING_HUMAN_LABELS = "calibration_pending_human_labels"
    CALIBRATION_INSUFFICIENT_COVERAGE = "calibration_insufficient_coverage"
    CALIBRATION_FAILED_THRESHOLDS = "calibration_failed_thresholds"
    UNIT_MISMATCH = "unit_mismatch"
    THRESHOLD_FAILED = "threshold_failed"


_STATUS_REASONS = {
    GateCheckStatus.NOT_COMPARABLE: frozenset(
        {GateCheckReason.NOT_COMPARABLE, GateCheckReason.POLICY_MISMATCH}
    ),
    GateCheckStatus.NOT_SCORABLE: frozenset(
        {
            GateCheckReason.POLICY_MISMATCH,
            GateCheckReason.NOT_SCORABLE,
            GateCheckReason.ZERO_DENOMINATOR,
            GateCheckReason.UNGRADED,
            GateCheckReason.MISSING_VALUE,
            GateCheckReason.AUTHORITY_INSUFFICIENT,
            GateCheckReason.CALIBRATION_MISSING,
            GateCheckReason.CALIBRATION_NOT_ELIGIBLE,
            GateCheckReason.CALIBRATION_FAILED_THRESHOLDS,
            GateCheckReason.UNIT_MISMATCH,
        }
    ),
    GateCheckStatus.INSUFFICIENT_COVERAGE: frozenset(
        {
            GateCheckReason.INSUFFICIENT_COVERAGE,
            GateCheckReason.FAILED_COVERAGE,
            GateCheckReason.CALIBRATION_INSUFFICIENT_COVERAGE,
        }
    ),
    GateCheckStatus.NOT_CONFIGURED: frozenset(
        {GateCheckReason.NOT_CONFIGURED}
    ),
    GateCheckStatus.PENDING: frozenset(
        {GateCheckReason.CALIBRATION_PENDING_HUMAN_LABELS}
    ),
}


class GateReferenceKind(str, Enum):
    CASE = "case"
    TRIAL = "trial"


_SCOPE_ALIASES = {
    "candidate": GateConstraintScope.CANDIDATE_ABSOLUTE,
    "absolute": GateConstraintScope.CANDIDATE_ABSOLUTE,
    "delta": GateConstraintScope.BASELINE_DELTA,
    "baseline": GateConstraintScope.BASELINE_DELTA,
    "case": GateConstraintScope.CASE_ABSOLUTE,
    "trial": GateConstraintScope.TRIAL_ABSOLUTE,
}
_OPERATOR_ALIASES = {
    "gte": GateOperator.AT_LEAST,
    ">=": GateOperator.AT_LEAST,
    "lte": GateOperator.AT_MOST,
    "<=": GateOperator.AT_MOST,
}
_RATE_METRICS = frozenset(metric for metric in CoreMetric if metric is not CoreMetric.CRITICAL_HIGH_MISS_COUNT and metric is not CoreMetric.FABRICATED_FINDINGS_PER_PR)
_COUNT_METRICS = frozenset({CoreMetric.CRITICAL_HIGH_MISS_COUNT})
_MEAN_METRICS = frozenset({CoreMetric.FABRICATED_FINDINGS_PER_PR})
_INTENT_METRICS = frozenset(
    {
        CoreMetric.INTENT_CLAIM_PRECISION,
        CoreMetric.INTENT_CLAIM_RECALL,
        CoreMetric.INTENT_PARTIALLY_SUPPORTED_RATE,
        CoreMetric.INTENT_UNSUPPORTED_RATE,
        CoreMetric.INTENT_CONTRADICTED_RATE,
        CoreMetric.INTENT_UNKNOWN_RATE,
        CoreMetric.INTENT_CASE_PASS_RATE,
    }
)
_FINDING_METRICS = frozenset(
    {
        CoreMetric.ISSUE_PRECISION,
        CoreMetric.ISSUE_RECALL,
        CoreMetric.ISSUE_F1,
        CoreMetric.SEVERITY_WEIGHTED_RECALL,
        CoreMetric.CRITICAL_HIGH_MISS_COUNT,
        CoreMetric.LINE_PRECISION,
        CoreMetric.LINE_RECALL,
    }
)
_NOVEL_METRICS = frozenset(
    {
        CoreMetric.FABRICATED_FINDINGS_PER_PR,
        CoreMetric.FABRICATED_RATE,
        CoreMetric.PLAUSIBLE_RATE,
        CoreMetric.REVIEW_UNKNOWN_RATE,
    }
)
_EVIDENCE_METRICS = frozenset({CoreMetric.EVIDENCE_SUPPORT_RATE})
_SEMANTIC_PROFILES: Dict[CoreMetric, Tuple[JudgeTask, ...]] = {
    **{metric: (JudgeTask.INTENT_EQUIVALENCE,) for metric in _INTENT_METRICS},
    **{metric: (JudgeTask.FINDING_EQUIVALENCE,) for metric in _FINDING_METRICS},
    **{metric: (JudgeTask.NOVEL_FACTUALITY,) for metric in _NOVEL_METRICS},
    **{metric: (JudgeTask.EVIDENCE_SUPPORT,) for metric in _EVIDENCE_METRICS},
    CoreMetric.PUBLISHABLE_FINDING_PRECISION: (
        JudgeTask.FINDING_EQUIVALENCE,
        JudgeTask.NOVEL_FACTUALITY,
        JudgeTask.EVIDENCE_SUPPORT,
    ),
}
_AUTHORITY_METRICS = frozenset(
    {
        CoreMetric.SEVERITY_WEIGHTED_RECALL,
        CoreMetric.CRITICAL_HIGH_MISS_COUNT,
        CoreMetric.LINE_PRECISION,
        CoreMetric.LINE_RECALL,
    }
)
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_DECIMAL_RE = re.compile(r"^-?(0|[1-9][0-9]*)(\.[0-9]+)?$")


def _error(message: str) -> GateError:
    return GateError(message)


def _exact(value: Any, fields: Iterable[str], context: str) -> Dict[str, Any]:
    expected = set(fields)
    if type(value) is not dict or set(value) != expected:
        raise _error(f"{context} has unknown or missing fields")
    return value


def _array(value: Any, context: str, maximum: int) -> list[Any]:
    if type(value) is not list or len(value) > maximum:
        raise _error(f"{context} must be a bounded array")
    return value


def _from_json(model_type: Any, data: Any, context: str) -> Any:
    return model_type.from_dict(_strict_json_loads(data, MAX_GATE_BYTES, context))


def _digest(value: Any, context: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise _error(f"{context} must be a lowercase SHA-256 digest")
    return value


def _identifier(value: Any, context: str) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 512
        or value != value.strip()
        or any(ord(c) < 32 or c.isspace() for c in value)
    ):
        raise _error(f"{context} must be a bounded identifier")
    return value


def _opaque_stable_ref(value: Any, prefix: str, context: str) -> str:
    checked = _identifier(value, context)
    marker = prefix + "-"
    suffix = checked[len(marker) :] if checked.startswith(marker) else ""
    if len(suffix) != 64 or _DIGEST_RE.fullmatch(suffix) is None:
        raise _error(f"{context} must be an opaque {prefix} identity")
    return checked


def _enum(enum_type: type[Enum], value: Any, context: str) -> Any:
    if isinstance(value, enum_type) and type(value) is enum_type:
        return value
    if type(value) is not str:
        raise _error(f"{context} must be a closed enum value")
    try:
        return enum_type(value)
    except ValueError as exc:
        raise _error(f"{context} has an unsupported value") from exc


def _scope(value: Any) -> GateConstraintScope:
    if isinstance(value, GateConstraintScope) and type(value) is GateConstraintScope:
        return value
    if type(value) is str and value in _SCOPE_ALIASES:
        return _SCOPE_ALIASES[value]
    return _enum(GateConstraintScope, value, "constraint.scope")


def _operator(value: Any) -> GateOperator:
    if isinstance(value, GateOperator) and type(value) is GateOperator:
        return value
    if type(value) is str and value in _OPERATOR_ALIASES:
        return _OPERATOR_ALIASES[value]
    return _enum(GateOperator, value, "constraint.operator")


def _strict_int(value: Any, context: str, *, minimum: int = 0, maximum: int = MAX_GATE_INTEGER) -> int:
    if type(value) is not int or value < minimum or value > maximum:
        raise _error(f"{context} must be an integer in the permitted range")
    return value


def _strict_optional_int(value: Any, context: str, *, maximum: int = MAX_GATE_INTEGER) -> Optional[int]:
    if value is None:
        return None
    return _strict_int(value, context, maximum=maximum)


def _metric_kind(metric: CoreMetric) -> MetricKind:
    if metric in _COUNT_METRICS:
        return MetricKind.COUNT
    if metric in _MEAN_METRICS:
        return MetricKind.MEAN
    return MetricKind.RATE


def _expected_unit(metric: CoreMetric) -> MetricUnit:
    return MetricUnit.COUNT if _metric_kind(metric) is MetricKind.COUNT else MetricUnit.PPM


def _canonical_threshold(metric: CoreMetric, scope: GateConstraintScope, value: Any) -> int:
    if type(value) is bool or type(value) not in (int, str):
        raise _error("constraint.threshold must be an integer or canonical decimal string")
    if type(value) is str:
        if value == "-0" or _DECIMAL_RE.fullmatch(value) is None:
            raise _error("constraint.threshold is not a canonical decimal")
        try:
            decimal = Decimal(value)
        except InvalidOperation as exc:
            raise _error("constraint.threshold is not numeric") from exc
        if not decimal.is_finite() or decimal != decimal.to_integral_value():
            raise _error("current CoreMetric units require an integral threshold")
        value = int(decimal)
    else:
        value = int(value)
    delta_scope = scope in {
        GateConstraintScope.BASELINE_DELTA,
        GateConstraintScope.CASE_DELTA,
        GateConstraintScope.TRIAL_DELTA,
    }
    if not delta_scope and value < 0:
        raise _error("absolute thresholds may not be negative")
    kind = _metric_kind(metric)
    if kind is MetricKind.RATE and not -PPM_SCALE <= value <= PPM_SCALE:
        raise _error("rate thresholds must be within signed ppm bounds")
    if abs(value) > MAX_GATE_INTEGER:
        raise _error("threshold exceeds the safe numeric bound")
    return value


def _constraint_sort_key(value: "MetricConstraintV1") -> tuple[Any, ...]:
    return (value.metric.value, value.scope.value, value.operator.value, value.threshold, value.unit.value, value.required, value.min_coverage_ppm if value.min_coverage_ppm is not None else -1)


@dataclass(frozen=True)
class MetricConstraintV1(_JsonModel):
    metric: CoreMetric
    scope: GateConstraintScope
    operator: GateOperator
    threshold: int | str
    unit: MetricUnit
    required: bool
    min_coverage_ppm: int | None

    def __post_init__(self) -> None:
        if type(self.metric) is not CoreMetric:
            raise _error("MetricConstraintV1.metric is invalid")
        scope = _scope(self.scope)
        operator = _operator(self.operator)
        unit = _enum(MetricUnit, self.unit, "constraint.unit")
        if unit is not _expected_unit(self.metric):
            raise _error("constraint.unit differs from Task 2 metric metadata")
        threshold = _canonical_threshold(self.metric, scope, self.threshold)
        if type(self.required) is not bool:
            raise _error("constraint.required must be a boolean")
        coverage = _strict_optional_int(self.min_coverage_ppm, "constraint.min_coverage_ppm", maximum=PPM_SCALE)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "min_coverage_ppm", coverage)

    @property
    def constraint_id(self) -> str:
        return stable_id("gate-constraint-v1", self._identity_dict())

    def _identity_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric.value,
            "scope": self.scope.value,
            "operator": self.operator.value,
            "threshold": self.threshold,
            "unit": self.unit.value,
            "required": self.required,
            "min_coverage_ppm": self.min_coverage_ppm,
        }

    def to_dict(self) -> Dict[str, Any]:
        return self._identity_dict()

    @classmethod
    def from_dict(cls, value: Any) -> "MetricConstraintV1":
        payload = _exact(value, ("metric", "scope", "operator", "threshold", "unit", "required", "min_coverage_ppm"), "MetricConstraintV1")
        return cls(
            _enum(CoreMetric, payload["metric"], "constraint.metric"),
            _scope(payload["scope"]),
            _operator(payload["operator"]),
            payload["threshold"],
            _enum(MetricUnit, payload["unit"], "constraint.unit"),
            payload["required"],
            payload["min_coverage_ppm"],
        )

    @classmethod
    def from_json(cls, data: Any) -> "MetricConstraintV1":
        return _from_json(cls, data, "MetricConstraintV1 JSON")


@dataclass(frozen=True)
class GatePolicyV1(_JsonModel):
    schema_version: str
    policy_id: str
    baseline_binding: AnalysisSourceBinding
    candidate_run_id: str
    candidate_run_config_digest: str
    case_snapshot_digest: str
    trial_count: int
    comparison_policy_digest: str
    calibration_result_digests: Tuple[str, ...]
    eligibility: GateEligibility
    constraints: Tuple[MetricConstraintV1, ...]

    def __post_init__(self) -> None:
        if self.schema_version != GATE_POLICY_SCHEMA_VERSION:
            raise _error("GatePolicyV1 schema_version is unsupported")
        if type(self.baseline_binding) is not AnalysisSourceBinding:
            raise _error("GatePolicyV1.baseline_binding is invalid")
        validate_run_id(self.candidate_run_id)
        for name in ("candidate_run_config_digest", "case_snapshot_digest", "comparison_policy_digest"):
            _digest(getattr(self, name), f"GatePolicyV1.{name}")
        _strict_int(self.trial_count, "GatePolicyV1.trial_count", minimum=1, maximum=MAX_TRIAL_COUNT)
        eligibility = _enum(GateEligibility, self.eligibility, "GatePolicyV1.eligibility")
        digests = tuple(self.calibration_result_digests)
        if type(self.calibration_result_digests) not in (tuple, list) or any(type(item) is not str for item in digests):
            raise _error("GatePolicyV1 calibration_result_digests must be typed digests")
        digests = tuple(_digest(item, "GatePolicyV1 calibration result digest") for item in digests)
        if len(digests) > MAX_GATE_CALIBRATIONS or len(digests) != len(set(digests)) or digests != tuple(sorted(digests)):
            raise _error("GatePolicyV1 calibration digests are not canonical")
        constraints = tuple(self.constraints)
        if type(self.constraints) not in (tuple, list) or any(type(item) is not MetricConstraintV1 for item in constraints):
            raise _error("GatePolicyV1 constraints must contain typed constraints")
        if len(constraints) > MAX_GATE_CONSTRAINTS or len({item.constraint_id for item in constraints}) != len(constraints):
            raise _error("GatePolicyV1 constraints are duplicate or excessive")
        if constraints != tuple(sorted(constraints, key=_constraint_sort_key)):
            raise _error("GatePolicyV1 constraints are not canonical")
        if eligibility is GateEligibility.RELEASE_BLOCKING and (
            not constraints or not any(item.required for item in constraints)
        ):
            raise _error(
                "release_blocking policy must configure at least one required constraint"
            )
        if not constraints and digests:
            raise _error("an empty diagnostic policy may not pre-register unused calibrations")
        object.__setattr__(self, "eligibility", eligibility)
        object.__setattr__(self, "calibration_result_digests", digests)
        object.__setattr__(self, "constraints", constraints)
        identity = self._identity_dict()
        if self.policy_id != stable_id("gate-policy-v1", identity):
            raise _error("GatePolicyV1 policy_id is not canonical")

    @classmethod
    def create(cls, **values: Any) -> "GatePolicyV1":
        constraints = tuple(sorted(tuple(values["constraints"]), key=_constraint_sort_key))
        digests = tuple(sorted(tuple(values.get("calibration_result_digests", ()))) )
        identity = {
            "schema_version": GATE_POLICY_SCHEMA_VERSION,
            "baseline_binding": values["baseline_binding"].to_dict(),
            "candidate_run_id": values["candidate_run_id"],
            "candidate_run_config_digest": values["candidate_run_config_digest"],
            "case_snapshot_digest": values["case_snapshot_digest"],
            "trial_count": values["trial_count"],
            "comparison_policy_digest": values["comparison_policy_digest"],
            "calibration_result_digests": list(digests),
            "eligibility": _enum(GateEligibility, values["eligibility"], "GatePolicyV1.eligibility").value,
            "constraints": [item.to_dict() for item in constraints],
        }
        return cls(
            GATE_POLICY_SCHEMA_VERSION,
            stable_id("gate-policy-v1", identity),
            values["baseline_binding"],
            values["candidate_run_id"],
            values["candidate_run_config_digest"],
            values["case_snapshot_digest"],
            values["trial_count"],
            values["comparison_policy_digest"],
            digests,
            _enum(GateEligibility, values["eligibility"], "GatePolicyV1.eligibility"),
            constraints,
        )

    def _identity_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "baseline_binding": self.baseline_binding.to_dict(),
            "candidate_run_id": self.candidate_run_id,
            "candidate_run_config_digest": self.candidate_run_config_digest,
            "case_snapshot_digest": self.case_snapshot_digest,
            "trial_count": self.trial_count,
            "comparison_policy_digest": self.comparison_policy_digest,
            "calibration_result_digests": list(self.calibration_result_digests),
            "eligibility": self.eligibility.value,
            "constraints": [item.to_dict() for item in self.constraints],
        }

    @property
    def policy_digest(self) -> str:
        return self.digest()

    @property
    def algorithm_digest(self) -> str:
        return canonical_sha256({
            "algorithm_version": GATE_ALGORITHM_VERSION,
            "policy_digest": self.policy_digest,
            "candidate_run_id": self.candidate_run_id,
            "candidate_run_config_digest": self.candidate_run_config_digest,
            "comparison_policy_digest": self.comparison_policy_digest,
        })

    def to_dict(self) -> Dict[str, Any]:
        return {**self._identity_dict(), "policy_id": self.policy_id}

    @classmethod
    def from_dict(cls, value: Any) -> "GatePolicyV1":
        payload = _exact(value, ("schema_version", "policy_id", "baseline_binding", "candidate_run_id", "candidate_run_config_digest", "case_snapshot_digest", "trial_count", "comparison_policy_digest", "calibration_result_digests", "eligibility", "constraints"), "GatePolicyV1")
        digests = _array(payload["calibration_result_digests"], "GatePolicyV1.calibration_result_digests", MAX_GATE_CALIBRATIONS)
        constraints = _array(payload["constraints"], "GatePolicyV1.constraints", MAX_GATE_CONSTRAINTS)
        return cls(
            payload["schema_version"], payload["policy_id"],
            AnalysisSourceBinding.from_dict(payload["baseline_binding"]),
            payload["candidate_run_id"], payload["candidate_run_config_digest"],
            payload["case_snapshot_digest"], payload["trial_count"],
            payload["comparison_policy_digest"], tuple(digests),
            _enum(GateEligibility, payload["eligibility"], "GatePolicyV1.eligibility"),
            tuple(MetricConstraintV1.from_dict(item) for item in constraints),
        )

    @classmethod
    def from_json(cls, data: Any) -> "GatePolicyV1":
        return _from_json(cls, data, "GatePolicyV1 JSON")


@dataclass(frozen=True, init=False)
class FrozenGatePolicy:
    """A Store-issued, source-replayed gate policy.

    ``GatePolicyV1`` is intentionally still a useful proposal/serialization
    type.  It is not, however, an evaluation capability.  This wrapper is
    the capability boundary: it carries the exact policy receipt and the
    replayed baseline/candidate identities, and its constructor is sealed so
    callers cannot manufacture one by simply deserializing a policy.
    """

    policy: GatePolicyV1
    receipt: AnalysisReceipt
    receipt_digest: str
    artifact_id: str
    baseline_binding: AnalysisSourceBinding
    candidate_run_config_digest: str
    _store_identity_digest: str = field(repr=False)
    _seal: object = field(repr=False, compare=False)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError(
            "FrozenGatePolicy is Store-issued; use AnalysisArtifactStore"
        )

    @classmethod
    def _issue(
        cls,
        *,
        policy: GatePolicyV1,
        receipt: AnalysisReceipt,
        store_identity_digest: str,
        seal: object,
    ) -> "FrozenGatePolicy":
        if seal is not _FROZEN_GATE_POLICY_SEAL:
            raise TypeError("FrozenGatePolicy can only be issued by its Store")
        _validate_frozen_gate_policy_parts(
            policy=policy,
            receipt=receipt,
            receipt_digest=receipt.digest(),
            artifact_id=receipt.artifact_id,
            baseline_binding=policy.baseline_binding,
            candidate_run_config_digest=policy.candidate_run_config_digest,
            store_identity_digest=store_identity_digest,
            seal=seal,
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "policy", policy)
        object.__setattr__(instance, "receipt", receipt)
        object.__setattr__(instance, "receipt_digest", receipt.digest())
        object.__setattr__(instance, "artifact_id", receipt.artifact_id)
        object.__setattr__(instance, "baseline_binding", policy.baseline_binding)
        object.__setattr__(
            instance,
            "candidate_run_config_digest",
            policy.candidate_run_config_digest,
        )
        object.__setattr__(instance, "_store_identity_digest", store_identity_digest)
        object.__setattr__(instance, "_seal", seal)
        return instance

    @property
    def policy_digest(self) -> str:
        return self.policy.policy_digest

    @property
    def candidate_run_id(self) -> str:
        return self.policy.candidate_run_id

    @property
    def calibration_result_digests(self) -> Tuple[str, ...]:
        return self.policy.calibration_result_digests

    @property
    def eligibility(self) -> GateEligibility:
        return self.policy.eligibility

    @property
    def constraints(self) -> Tuple[MetricConstraintV1, ...]:
        return self.policy.constraints

    @property
    def case_snapshot_digest(self) -> str:
        return self.policy.case_snapshot_digest

    @property
    def trial_count(self) -> int:
        return self.policy.trial_count

    @property
    def comparison_policy_digest(self) -> str:
        return self.policy.comparison_policy_digest


def _validate_frozen_gate_policy_parts(
    *,
    policy: Any,
    receipt: Any,
    receipt_digest: Any,
    artifact_id: Any,
    baseline_binding: Any,
    candidate_run_config_digest: Any,
    store_identity_digest: Any,
    seal: Any,
) -> None:
    """Validate the non-serializable provenance carried by a frozen policy."""

    if seal is not _FROZEN_GATE_POLICY_SEAL:
        raise ArtifactIntegrityError("FrozenGatePolicy seal is invalid")
    if type(policy) is not GatePolicyV1:
        raise ArtifactIntegrityError("FrozenGatePolicy policy is not canonical")
    if type(receipt) is not AnalysisReceipt:
        raise ArtifactIntegrityError("FrozenGatePolicy receipt is invalid")
    if receipt.kind != "gate-policy":
        raise ArtifactIntegrityError("FrozenGatePolicy receipt has the wrong kind")
    if type(baseline_binding) is not AnalysisSourceBinding:
        raise ArtifactIntegrityError("FrozenGatePolicy baseline binding is invalid")
    if baseline_binding != policy.baseline_binding:
        raise ArtifactIntegrityError("FrozenGatePolicy baseline binding is not replayed")
    if candidate_run_config_digest != policy.candidate_run_config_digest:
        raise ArtifactIntegrityError(
            "FrozenGatePolicy candidate RunConfig digest is not replayed"
        )
    _digest(receipt_digest, "FrozenGatePolicy.receipt_digest")
    _digest(store_identity_digest, "FrozenGatePolicy.store_identity_digest")
    if artifact_id != receipt.artifact_id:
        raise ArtifactIntegrityError("FrozenGatePolicy artifact identity is invalid")
    if receipt_digest != receipt.digest():
        raise ArtifactIntegrityError("FrozenGatePolicy receipt digest is invalid")
    if receipt.source_bindings != (policy.baseline_binding,):
        raise ArtifactIntegrityError(
            "FrozenGatePolicy receipt does not bind the baseline"
        )
    if receipt.algorithm_digest != policy.algorithm_digest:
        raise ArtifactIntegrityError(
            "FrozenGatePolicy receipt does not bind the canonical policy"
        )
    try:
        canonical_policy = GatePolicyV1.from_dict(policy.to_dict())
        canonical_receipt = AnalysisReceipt.from_dict(receipt.to_dict())
    except (SchemaError, TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(
            "FrozenGatePolicy nested policy/receipt is not canonical"
        ) from exc
    if canonical_policy != policy or canonical_receipt != receipt:
        raise ArtifactIntegrityError(
            "FrozenGatePolicy nested policy/receipt differs from canonical bytes"
        )
    refs = [
        item
        for item in receipt.artifacts
        if item.relative_path.rsplit("/", 1)[-1] == "gate_policy.json"
    ]
    expected_data = canonical_json_bytes(policy.to_dict())
    if (
        len(refs) != 1
        or refs[0].sha256 != hashlib.sha256(expected_data).hexdigest()
        or refs[0].size_bytes != len(expected_data)
    ):
        raise ArtifactIntegrityError(
            "FrozenGatePolicy receipt does not bind gate_policy.json"
        )


def _validate_frozen_gate_policy(
    frozen: Any,
    *,
    store_identity_digest: str | None = None,
) -> GatePolicyV1:
    if type(frozen) is not FrozenGatePolicy:
        raise TypeError(
            "evaluate_gate requires a Store-issued FrozenGatePolicy"
        )
    try:
        _validate_frozen_gate_policy_parts(
            policy=frozen.policy,
            receipt=frozen.receipt,
            receipt_digest=frozen.receipt_digest,
            artifact_id=frozen.artifact_id,
            baseline_binding=frozen.baseline_binding,
            candidate_run_config_digest=frozen.candidate_run_config_digest,
            store_identity_digest=frozen._store_identity_digest,
            seal=frozen._seal,
        )
    except ArtifactIntegrityError:
        raise
    except (AttributeError, GateError, SchemaError, TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(
            "FrozenGatePolicy provenance is malformed"
        ) from exc
    if (
        store_identity_digest is not None
        and frozen._store_identity_digest != store_identity_digest
    ):
        raise ArtifactIntegrityError(
            "FrozenGatePolicy belongs to another AnalysisArtifactStore"
        )
    return frozen.policy


def _freeze_gate_policy(
    policy: GatePolicyV1,
    receipt: AnalysisReceipt,
    *,
    store_identity_digest: str,
) -> FrozenGatePolicy:
    """Private Store seam for issuing an evaluation-capable policy."""

    return FrozenGatePolicy._issue(
        policy=policy,
        receipt=receipt,
        store_identity_digest=store_identity_digest,
        seal=_FROZEN_GATE_POLICY_SEAL,
    )


def _ref_sort_key(value: "GateFailureRefV1") -> tuple[str, str]:
    return value.kind.value, value.reference_id


@dataclass(frozen=True)
class GateFailureRefV1(_JsonModel):
    kind: GateReferenceKind
    reference_id: str
    actual: int | str | None
    threshold: int | str
    unit: MetricUnit
    reason: GateCheckReason

    def __post_init__(self) -> None:
        kind = _enum(GateReferenceKind, self.kind, "GateFailureRefV1.kind")
        _opaque_stable_ref(
            self.reference_id,
            (
                "gate-case-ref-v1"
                if kind is GateReferenceKind.CASE
                else "gate-trial-ref-v1"
            ),
            "GateFailureRefV1.reference_id",
        )
        unit = _enum(MetricUnit, self.unit, "GateFailureRefV1.unit")
        reason = _enum(GateCheckReason, self.reason, "GateFailureRefV1.reason")
        if self.actual is not None and (
            type(self.actual) is not int or abs(self.actual) > MAX_GATE_INTEGER
        ):
            raise _error("GateFailureRefV1.actual is not a bounded integer")
        if type(self.threshold) not in (int, str) or type(self.threshold) is bool:
            raise _error("GateFailureRefV1.threshold is invalid")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "reason", reason)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "reference_id": self.reference_id,
            "actual": self.actual,
            "threshold": self.threshold,
            "unit": self.unit.value,
            "reason": self.reason.value,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "GateFailureRefV1":
        payload = _exact(value, ("kind", "reference_id", "actual", "threshold", "unit", "reason"), "GateFailureRefV1")
        return cls(
            _enum(GateReferenceKind, payload["kind"], "failure ref.kind"),
            payload["reference_id"], payload["actual"], payload["threshold"],
            _enum(MetricUnit, payload["unit"], "failure ref.unit"),
            _enum(GateCheckReason, payload["reason"], "failure ref.reason"),
        )


@dataclass(frozen=True)
class GateCheckV1(_JsonModel):
    constraint_id: str
    metric: CoreMetric
    scope: GateConstraintScope
    operator: GateOperator
    required: bool
    status: GateCheckStatus
    actual: int | str | None
    threshold: int | str
    unit: MetricUnit
    coverage_ppm: int | None
    min_coverage_ppm: int | None
    metric_ref: str | None
    failure_refs: Tuple[GateFailureRefV1, ...]
    calibration_result_digests: Tuple[str, ...]
    reasons: Tuple[GateCheckReason, ...]

    def __post_init__(self) -> None:
        _identifier(self.constraint_id, "GateCheckV1.constraint_id")
        metric = _enum(CoreMetric, self.metric, "GateCheckV1.metric")
        scope = _scope(self.scope)
        operator = _operator(self.operator)
        status = _enum(GateCheckStatus, self.status, "GateCheckV1.status")
        unit = _enum(MetricUnit, self.unit, "GateCheckV1.unit")
        if unit is not _expected_unit(metric):
            raise _error("GateCheckV1.unit differs from metric metadata")
        if type(self.required) is not bool:
            raise _error("GateCheckV1.required must be boolean")
        threshold = _canonical_threshold(metric, scope, self.threshold)
        if self.actual is not None and (
            type(self.actual) is not int or abs(self.actual) > MAX_GATE_INTEGER
        ):
            raise _error("GateCheckV1.actual is not a bounded integer")
        coverage = _strict_optional_int(self.coverage_ppm, "GateCheckV1.coverage_ppm", maximum=PPM_SCALE)
        minimum = _strict_optional_int(self.min_coverage_ppm, "GateCheckV1.min_coverage_ppm", maximum=PPM_SCALE)
        if self.metric_ref is not None:
            _opaque_stable_ref(
                self.metric_ref,
                "metric-delta-v1",
                "GateCheckV1.metric_ref",
            )
        refs = tuple(self.failure_refs)
        if type(self.failure_refs) not in (tuple, list) or any(type(item) is not GateFailureRefV1 for item in refs):
            raise _error("GateCheckV1 failure_refs are invalid")
        if len(refs) > MAX_GATE_REFS or refs != tuple(sorted(refs, key=_ref_sort_key)):
            raise _error("GateCheckV1 failure_refs are not canonical")
        if len({(item.kind, item.reference_id) for item in refs}) != len(refs):
            raise _error("GateCheckV1 failure_refs contain duplicates")
        if any(item.threshold != threshold or item.unit is not unit for item in refs):
            raise _error("GateCheckV1 failure refs differ from the configured threshold")
        digests = tuple(self.calibration_result_digests)
        if type(self.calibration_result_digests) not in (tuple, list):
            raise _error("GateCheckV1 calibration refs are invalid")
        digests = tuple(_digest(item, "GateCheckV1 calibration ref") for item in digests)
        if len(digests) != len(set(digests)) or digests != tuple(sorted(digests)):
            raise _error("GateCheckV1 calibration refs are not canonical")
        reasons = tuple(_enum(GateCheckReason, item, "GateCheckV1.reason") for item in self.reasons)
        if len(reasons) != len(set(reasons)) or reasons != tuple(sorted(reasons, key=lambda item: item.value)):
            raise _error("GateCheckV1 reasons are not canonical")
        legacy_status = status is GateCheckStatus.INELIGIBLE
        if legacy_status and not _LEGACY_GATE_HYDRATION.get():
            raise _error("legacy ineligible GateCheckV1 is hydration-only")
        if not legacy_status:
            if status is GateCheckStatus.PASS and reasons:
                raise _error("passing GateCheckV1 may not carry reasons")
            if status is GateCheckStatus.FAIL and reasons != (
                GateCheckReason.THRESHOLD_FAILED,
            ):
                raise _error("failed GateCheckV1 requires only threshold_failed")
            if status not in {GateCheckStatus.PASS, GateCheckStatus.FAIL}:
                allowed_reasons = _STATUS_REASONS[status]
                if not reasons or not set(reasons).issubset(allowed_reasons):
                    raise _error("GateCheckV1 status and reasons are incompatible")
                if status is GateCheckStatus.NOT_COMPARABLE and (
                    GateCheckReason.NOT_COMPARABLE not in reasons
                ):
                    raise _error("not_comparable GateCheckV1 requires not_comparable")
                if status is GateCheckStatus.NOT_CONFIGURED and reasons != (
                    GateCheckReason.NOT_CONFIGURED,
                ):
                    raise _error("not_configured GateCheckV1 requires not_configured")
                if status is GateCheckStatus.PENDING and reasons != (
                    GateCheckReason.CALIBRATION_PENDING_HUMAN_LABELS,
                ):
                    raise _error("pending GateCheckV1 requires pending calibration")
        if status not in {GateCheckStatus.PASS, GateCheckStatus.FAIL} and not reasons:
            raise _error("unscored GateCheckV1 requires a typed reason")
        if status is GateCheckStatus.FAIL and GateCheckReason.THRESHOLD_FAILED not in reasons:
            raise _error("failed GateCheckV1 requires threshold_failed")
        if status is GateCheckStatus.PASS and self.actual is None:
            raise _error("passing GateCheckV1 requires an actual value")
        if status is GateCheckStatus.FAIL and not refs:
            raise _error("failed GateCheckV1 requires source-bound failure refs")
        if status is GateCheckStatus.PASS and not _satisfies(operator, self.actual, threshold):
            raise _error("passing GateCheckV1 does not satisfy its threshold")
        if status is GateCheckStatus.FAIL and _satisfies(operator, self.actual, threshold):
            raise _error("failed GateCheckV1 satisfies its threshold")
        if (
            status in {GateCheckStatus.PASS, GateCheckStatus.FAIL}
            and minimum is not None
            and (coverage is None or coverage < minimum)
        ):
            raise _error("scored GateCheckV1 does not meet configured coverage")
        if status is GateCheckStatus.NOT_CONFIGURED:
            if self.required or self.actual is not None or coverage is not None or minimum is not None:
                raise _error("not_configured GateCheckV1 carries configured data")
            expected_constraint_id = stable_id(
                "gate-not-configured-v1", {"metric": metric.value}
            )
        else:
            expected_constraint_id = stable_id("gate-constraint-v1", {
                "metric": metric.value,
                "scope": scope.value,
                "operator": operator.value,
                "threshold": threshold,
                "unit": unit.value,
                "required": self.required,
                "min_coverage_ppm": minimum,
            })
        if self.constraint_id != expected_constraint_id:
            raise _error("GateCheckV1 constraint_id is not canonical")
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "scope", scope)
        object.__setattr__(self, "operator", operator)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "coverage_ppm", coverage)
        object.__setattr__(self, "min_coverage_ppm", minimum)
        object.__setattr__(self, "failure_refs", refs)
        object.__setattr__(self, "calibration_result_digests", digests)
        object.__setattr__(self, "reasons", reasons)

    @property
    def check_id(self) -> str:
        return stable_id("gate-check-v1", self._identity_dict())

    @property
    def case_refs(self) -> Tuple[str, ...]:
        return tuple(item.reference_id for item in self.failure_refs if item.kind is GateReferenceKind.CASE)

    @property
    def trial_refs(self) -> Tuple[str, ...]:
        return tuple(item.reference_id for item in self.failure_refs if item.kind is GateReferenceKind.TRIAL)

    @property
    def reason(self) -> GateCheckReason | None:
        return self.reasons[0] if self.reasons else None

    def _identity_dict(self) -> Dict[str, Any]:
        return {
            "constraint_id": self.constraint_id,
            "metric": self.metric.value,
            "scope": self.scope.value,
            "operator": self.operator.value,
            "required": self.required,
            "status": self.status.value,
            "actual": self.actual,
            "threshold": self.threshold,
            "unit": self.unit.value,
            "coverage_ppm": self.coverage_ppm,
            "min_coverage_ppm": self.min_coverage_ppm,
            "metric_ref": self.metric_ref,
            "failure_refs": [item.to_dict() for item in self.failure_refs],
            "calibration_result_digests": list(self.calibration_result_digests),
            "reasons": [item.value for item in self.reasons],
        }

    def to_dict(self) -> Dict[str, Any]:
        return {**self._identity_dict(), "check_id": self.check_id}

    @classmethod
    def from_dict(cls, value: Any) -> "GateCheckV1":
        payload = _exact(value, ("check_id", "constraint_id", "metric", "scope", "operator", "required", "status", "actual", "threshold", "unit", "coverage_ppm", "min_coverage_ppm", "metric_ref", "failure_refs", "calibration_result_digests", "reasons"), "GateCheckV1")
        refs = _array(payload["failure_refs"], "GateCheckV1.failure_refs", MAX_GATE_REFS)
        is_legacy = payload["status"] == GateCheckStatus.INELIGIBLE.value
        token = _LEGACY_GATE_HYDRATION.set(True) if is_legacy else None
        try:
            result = cls(
                payload["constraint_id"], _enum(CoreMetric, payload["metric"], "check.metric"),
                _scope(payload["scope"]), _operator(payload["operator"]), payload["required"],
                _enum(GateCheckStatus, payload["status"], "check.status"), payload["actual"],
                payload["threshold"], _enum(MetricUnit, payload["unit"], "check.unit"),
                payload["coverage_ppm"], payload["min_coverage_ppm"], payload["metric_ref"],
                tuple(GateFailureRefV1.from_dict(item) for item in refs),
                tuple(_array(payload["calibration_result_digests"], "check.calibration_result_digests", MAX_GATE_CALIBRATIONS)),
                tuple(_enum(GateCheckReason, item, "check.reason") for item in _array(payload["reasons"], "check.reasons", len(GateCheckReason))),
            )
        finally:
            if token is not None:
                _LEGACY_GATE_HYDRATION.reset(token)
        if payload["check_id"] != result.check_id:
            raise _error("GateCheckV1 check_id is not canonical")
        return result

    @classmethod
    def from_json(cls, data: Any) -> "GateCheckV1":
        return _from_json(cls, data, "GateCheckV1 JSON")


@dataclass(frozen=True)
class GateResultV1(_JsonModel):
    schema_version: str
    gate_result_id: str
    policy_digest: str
    policy_artifact_id: str
    policy_receipt_digest: str
    comparison_id: str
    decision: GateDecision
    checks: Tuple[GateCheckV1, ...]

    def __post_init__(self) -> None:
        if self.schema_version != GATE_RESULT_SCHEMA_VERSION:
            raise _error("GateResultV1 schema_version is unsupported")
        _digest(self.policy_digest, "GateResultV1.policy_digest")
        _opaque_stable_ref(
            self.policy_artifact_id,
            "analysis-artifact-v1",
            "GateResultV1.policy_artifact_id",
        )
        _digest(
            self.policy_receipt_digest,
            "GateResultV1.policy_receipt_digest",
        )
        _identifier(self.comparison_id, "GateResultV1.comparison_id")
        decision = _enum(GateDecision, self.decision, "GateResultV1.decision")
        checks = tuple(self.checks)
        if type(self.checks) not in (tuple, list) or any(type(item) is not GateCheckV1 for item in checks):
            raise _error("GateResultV1 checks are invalid")
        if checks != tuple(sorted(checks, key=lambda item: item.constraint_id)) or len({item.constraint_id for item in checks}) != len(checks):
            raise _error("GateResultV1 checks are not canonical")
        required = tuple(item for item in checks if item.required)
        if (
            any(item.status is GateCheckStatus.INELIGIBLE for item in checks)
            and not _LEGACY_GATE_HYDRATION.get()
        ):
            raise _error("legacy ineligible GateResultV1 is hydration-only")
        if checks and any(
            item.calibration_result_digests
            != checks[0].calibration_result_digests
            for item in checks[1:]
        ):
            raise _error("GateResultV1 checks disagree on calibration bindings")
        unscored_required = tuple(
            item
            for item in required
            if item.status not in {GateCheckStatus.PASS, GateCheckStatus.FAIL}
        )
        if decision is GateDecision.PROMOTE and any(
            item.status is not GateCheckStatus.PASS for item in required
        ):
            raise _error("promote GateResultV1 has a non-passing required check")
        if decision is GateDecision.PROMOTE and not required:
            raise _error(
                "promote GateResultV1 requires at least one required check"
            )
        if decision is GateDecision.BLOCK and not any(
            item.status is GateCheckStatus.FAIL for item in required
        ):
            raise _error("block GateResultV1 has no failed required check")
        if decision is GateDecision.BLOCK and unscored_required:
            raise _error("ineligible required checks take precedence over block")
        object.__setattr__(self, "decision", decision)
        object.__setattr__(self, "checks", checks)
        identity = self._identity_dict()
        if self.gate_result_id != stable_id("gate-result-v1", identity):
            raise _error("GateResultV1 gate_result_id is not canonical")

    @classmethod
    def create(cls, **values: Any) -> "GateResultV1":
        checks = tuple(sorted(tuple(values["checks"]), key=lambda item: item.constraint_id))
        if any(item.status is GateCheckStatus.INELIGIBLE for item in checks):
            raise _error("legacy ineligible checks cannot create a new GateResultV1")
        identity = {
            "schema_version": GATE_RESULT_SCHEMA_VERSION,
            "policy_digest": values["policy_digest"],
            "policy_artifact_id": values["policy_artifact_id"],
            "policy_receipt_digest": values["policy_receipt_digest"],
            "comparison_id": values["comparison_id"],
            "decision": _enum(GateDecision, values["decision"], "GateResultV1.decision").value,
            "checks": [item.to_dict() for item in checks],
        }
        return cls(
            GATE_RESULT_SCHEMA_VERSION,
            stable_id("gate-result-v1", identity),
            values["policy_digest"], values["policy_artifact_id"],
            values["policy_receipt_digest"], values["comparison_id"],
            _enum(GateDecision, values["decision"], "GateResultV1.decision"),
            checks,
        )

    @property
    def calibration_result_digests(self) -> Tuple[str, ...]:
        values = set()
        for check in self.checks:
            values.update(check.calibration_result_digests)
        return tuple(sorted(values))

    @property
    def algorithm_digest(self) -> str:
        return canonical_sha256({
            "algorithm_version": GATE_ALGORITHM_VERSION,
            "policy_digest": self.policy_digest,
            "policy_artifact_id": self.policy_artifact_id,
            "policy_receipt_digest": self.policy_receipt_digest,
            "comparison_id": self.comparison_id,
            "calibration_result_digests": list(self.calibration_result_digests),
        })

    def _identity_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_digest": self.policy_digest,
            "policy_artifact_id": self.policy_artifact_id,
            "policy_receipt_digest": self.policy_receipt_digest,
            "comparison_id": self.comparison_id,
            "decision": self.decision.value,
            "checks": [item.to_dict() for item in self.checks],
        }

    def to_dict(self) -> Dict[str, Any]:
        return {**self._identity_dict(), "gate_result_id": self.gate_result_id}

    @classmethod
    def from_dict(cls, value: Any) -> "GateResultV1":
        payload = _exact(value, ("schema_version", "gate_result_id", "policy_digest", "policy_artifact_id", "policy_receipt_digest", "comparison_id", "decision", "checks"), "GateResultV1")
        checks = _array(payload["checks"], "GateResultV1.checks", MAX_GATE_CONSTRAINTS)
        is_legacy = any(
            type(item) is dict
            and item.get("status") == GateCheckStatus.INELIGIBLE.value
            for item in checks
        )
        token = _LEGACY_GATE_HYDRATION.set(True) if is_legacy else None
        try:
            return cls(
                payload["schema_version"], payload["gate_result_id"], payload["policy_digest"],
                payload["policy_artifact_id"], payload["policy_receipt_digest"],
                payload["comparison_id"], _enum(GateDecision, payload["decision"], "result.decision"),
                tuple(GateCheckV1.from_dict(item) for item in checks),
            )
        finally:
            if token is not None:
                _LEGACY_GATE_HYDRATION.reset(token)

    @classmethod
    def from_json(cls, data: Any) -> "GateResultV1":
        return _from_json(cls, data, "GateResultV1 JSON")


def _canonical_policy(policy: GatePolicyV1) -> GatePolicyV1:
    if type(policy) is not GatePolicyV1:
        raise TypeError("policy must be a GatePolicyV1")
    try:
        result = GatePolicyV1.from_dict(policy.to_dict())
    except (GateError, SchemaError, TypeError, ValueError) as exc:
        raise ArtifactIntegrityError("gate policy fails canonical hydration") from exc
    if result != policy or canonical_json_bytes(result.to_dict()) != canonical_json_bytes(policy.to_dict()):
        raise ArtifactIntegrityError("gate policy differs from canonical hydration")
    return result


def _canonical_comparison(comparison: RunComparisonV1) -> RunComparisonV1:
    if type(comparison) is not RunComparisonV1:
        raise TypeError("comparison must be a RunComparisonV1")
    try:
        result = RunComparisonV1.from_dict(comparison.to_dict())
    except (GateError, SchemaError, TypeError, ValueError) as exc:
        raise ArtifactIntegrityError("comparison fails canonical hydration") from exc
    if result != comparison:
        raise ArtifactIntegrityError("comparison differs from canonical hydration")
    return result


def _candidate_plan_is_canonical(candidate: EvalRunConfig) -> EvalRunConfig:
    if type(candidate) is not EvalRunConfig:
        raise TypeError("candidate_run_config must be an EvalRunConfig")
    try:
        result = EvalRunConfig.from_dict(candidate.to_dict())
    except (SchemaError, TypeError, ValueError) as exc:
        raise ArtifactIntegrityError("candidate RunConfig fails canonical hydration") from exc
    if result != candidate or result.digest() != candidate.digest():
        raise ArtifactIntegrityError("candidate RunConfig differs from canonical hydration")
    return result


def _case_evaluations(baseline: VerifiedRunEvaluation) -> Tuple[Any, ...]:
    by_id: Dict[str, Any] = {}
    for trial in baseline.trials:
        case = getattr(trial, "eval_case", None)
        task_id = getattr(case, "task_id", None)
        if task_id is not None:
            by_id.setdefault(task_id, case)
    return tuple(by_id[key] for key in sorted(by_id))


def _metric_authority_available(baseline: VerifiedRunEvaluation, metric: CoreMetric) -> bool:
    cases = _case_evaluations(baseline)
    if not cases:
        return False
    if metric in _INTENT_METRICS:
        return all(
            getattr(case.intent_truth, "scorable", False)
            and getattr(case.intent_truth, "authority", None) is not None
            and getattr(case.intent_truth, "authority", None) is not IntentAuthority.SYNTHETIC
            for case in cases
        )
    if metric in _AUTHORITY_METRICS:
        for case in cases:
            expected = tuple(case.review_truth.expected_findings)
            required = tuple(item for item in expected if item.required)
            if metric is CoreMetric.SEVERITY_WEIGHTED_RECALL or metric is CoreMetric.CRITICAL_HIGH_MISS_COUNT:
                if any(not item.metric_authority.severity_scorable for item in required):
                    return False
            elif metric is CoreMetric.LINE_RECALL:
                if any(not item.metric_authority.location_scorable for item in required):
                    return False
            elif metric is CoreMetric.LINE_PRECISION:
                if any(not item.metric_authority.location_scorable for item in expected):
                    return False
    # A baseline with no source-scored contribution cannot authorize a release
    # gate for that metric.  This is a data-quality check, not a business
    # threshold.
    for trial in baseline.trials:
        score = getattr(trial, "trial_score", None)
        if score is None:
            return False
        contribution = score.contribution(metric)
        if getattr(contribution, "source_status", None) not in {
            MetricSourceStatus.NOT_SCORABLE,
            MetricSourceStatus.MISSING,
        }:
            return True
    return False


def _trusted_release_authority(
    baseline: VerifiedRunEvaluation,
    constraints: Sequence[MetricConstraintV1],
) -> bool:
    manifest = baseline.case_snapshot.manifest
    source_kind = manifest.source.kind
    entries = baseline.case_snapshot.cases
    if source_kind is SuiteKind.CORE:
        if not all(item.split is CaseSplit.REGRESSION and item.source.origin is CaseOrigin.HAND_AUTHORED for item in entries):
            return False
    elif source_kind is SuiteKind.PRIVATE:
        if not all(item.split is CaseSplit.HELD_OUT and item.source.origin is CaseOrigin.PRIVATE for item in entries):
            return False
    else:
        return False
    if any(
        getattr(case.intent_truth, "authority", None)
        is IntentAuthority.SYNTHETIC
        for case in _case_evaluations(baseline)
    ):
        return False
    return all(_metric_authority_available(baseline, item.metric) for item in constraints)


def _hard_plan_compatibility(baseline: VerifiedRunEvaluation, candidate: EvalRunConfig) -> None:
    base = baseline.run_config
    fields = (
        "suite", "wire_contract", "suite_preparation_binding_digest",
        "clarification_matcher", "clarification_matcher_config_digest",
        "evaluator", "evaluator_config_digest", "target_kinds",
        "materializer_protocol", "trial_count",
    )
    for name in fields:
        if getattr(base, name) != getattr(candidate, name):
            raise _error(f"candidate RunConfig is incompatible at {name}")


def prepare_gate_policy(
    baseline: VerifiedRunEvaluation,
    candidate_run_config: EvalRunConfig,
    *,
    policy: GatePolicyV1,
) -> GatePolicyV1:
    """Validate and freeze a policy using only the baseline and candidate plan.

    This function never reads a candidate Evaluation, Submission, Score, or
    Comparison.  The returned value is a proposal until it is committed with
    :meth:`AnalysisArtifactStore.publish_gate_policy`; publication is the
    create-only freeze point.
    """

    if type(baseline) is not VerifiedRunEvaluation:
        raise TypeError("baseline must be a VerifiedRunEvaluation")
    baseline_binding = baseline.verify()
    candidate = _candidate_plan_is_canonical(candidate_run_config)
    canonical = _canonical_policy(policy)
    # Keep this check even though GatePolicyV1.__post_init__ enforces it.  A
    # hostile caller can bypass dataclass construction with object.__new__,
    # and preparation is the last proposal-facing boundary before Store
    # publication.
    if canonical.eligibility is GateEligibility.RELEASE_BLOCKING and (
        not canonical.constraints
        or not any(item.required for item in canonical.constraints)
    ):
        raise _error(
            "release_blocking policy must configure at least one required constraint"
        )
    if canonical.baseline_binding != baseline_binding:
        raise _error("GatePolicy baseline binding differs from verified baseline")
    if canonical.candidate_run_id != candidate.run_id or canonical.candidate_run_config_digest != candidate.digest():
        raise _error("GatePolicy candidate identity differs from the frozen Run plan")
    if canonical.case_snapshot_digest != baseline.case_snapshot.digest() or candidate.suite.case_snapshot_digest != canonical.case_snapshot_digest:
        raise _error("GatePolicy and candidate do not bind the baseline Case Snapshot")
    if canonical.trial_count != candidate.trial_count or candidate.trial_count != baseline.run_config.trial_count:
        raise _error("GatePolicy trial_count differs from the paired Run plans")
    _hard_plan_compatibility(baseline, candidate)
    if baseline.verify() != baseline_binding:
        raise ArtifactIntegrityError("baseline changed while preparing gate policy")
    trusted = _trusted_release_authority(baseline, canonical.constraints)
    effective = canonical.eligibility
    if effective is GateEligibility.RELEASE_BLOCKING and (not trusted or candidate.trial_count < 3):
        effective = GateEligibility.DIAGNOSTIC_ONLY
    if effective is GateEligibility.DIAGNOSTIC_ONLY and not canonical.constraints:
        return GatePolicyV1.create(
            baseline_binding=baseline_binding,
            candidate_run_id=candidate.run_id,
            candidate_run_config_digest=candidate.digest(),
            case_snapshot_digest=canonical.case_snapshot_digest,
            trial_count=candidate.trial_count,
            comparison_policy_digest=canonical.comparison_policy_digest,
            calibration_result_digests=canonical.calibration_result_digests,
            eligibility=effective,
            constraints=(),
        )
    return GatePolicyV1.create(
        baseline_binding=baseline_binding,
        candidate_run_id=candidate.run_id,
        candidate_run_config_digest=candidate.digest(),
        case_snapshot_digest=canonical.case_snapshot_digest,
        trial_count=candidate.trial_count,
        comparison_policy_digest=canonical.comparison_policy_digest,
        calibration_result_digests=canonical.calibration_result_digests,
        eligibility=effective,
        constraints=canonical.constraints,
    )


def _normalize_calibrations(calibrations: Mapping[Any, CalibrationResultV1]) -> Dict[JudgeTask, CalibrationResultV1]:
    if not isinstance(calibrations, Mapping):
        raise TypeError("calibrations must be a mapping")
    result: Dict[JudgeTask, CalibrationResultV1] = {}
    for raw_profile, raw_result in calibrations.items():
        profile = raw_profile if type(raw_profile) is JudgeTask else _enum(JudgeTask, raw_profile, "calibration profile")
        if type(raw_result) is not CalibrationResultV1:
            raise TypeError("calibration mapping values must be CalibrationResultV1")
        try:
            canonical = CalibrationResultV1.from_dict(raw_result.to_dict())
        except (SchemaError, TypeError, ValueError) as exc:
            raise ArtifactIntegrityError("calibration result fails canonical hydration") from exc
        if canonical != raw_result:
            raise ArtifactIntegrityError("calibration result differs from canonical hydration")
        if len(canonical.profiles) != 1 or canonical.profiles[0].profile is not profile:
            raise _error("calibration mapping key does not match its typed profile")
        if profile in result:
            raise _error("calibration profiles are duplicated")
        result[profile] = canonical
    return result


def _global_mismatch_reasons(policy: GatePolicyV1, comparison: RunComparisonV1, calibrations: Mapping[JudgeTask, CalibrationResultV1]) -> Tuple[GateCheckReason, ...]:
    reasons = []
    if comparison.baseline_binding != policy.baseline_binding:
        reasons.append(GateCheckReason.POLICY_MISMATCH)
    if comparison.candidate_binding.run_id != policy.candidate_run_id or comparison.candidate_binding.run_config_digest != policy.candidate_run_config_digest:
        reasons.append(GateCheckReason.POLICY_MISMATCH)
    if comparison.baseline_binding.case_snapshot_digest != policy.case_snapshot_digest or comparison.candidate_binding.case_snapshot_digest != policy.case_snapshot_digest:
        reasons.append(GateCheckReason.POLICY_MISMATCH)
    if comparison.baseline_statistics.trial_count != policy.trial_count or comparison.candidate_statistics.trial_count != policy.trial_count:
        reasons.append(GateCheckReason.POLICY_MISMATCH)
    if comparison.compatibility.policy_digest != policy.comparison_policy_digest:
        reasons.append(GateCheckReason.POLICY_MISMATCH)
    actual_digests = tuple(sorted(result.digest() for result in calibrations.values()))
    if actual_digests != policy.calibration_result_digests:
        reasons.append(GateCheckReason.POLICY_MISMATCH)
    if comparison.status is ComparisonStatus.NOT_COMPARABLE:
        reasons.append(GateCheckReason.NOT_COMPARABLE)
    return tuple(sorted(set(reasons), key=lambda item: item.value))


def _status_reason(status: StatisticsMetricStatus) -> GateCheckReason:
    return {
        StatisticsMetricStatus.ZERO_DENOMINATOR: GateCheckReason.ZERO_DENOMINATOR,
        StatisticsMetricStatus.UNGRADED: GateCheckReason.UNGRADED,
        StatisticsMetricStatus.NOT_SCORABLE: GateCheckReason.NOT_SCORABLE,
        StatisticsMetricStatus.FAILURE_EXCLUDED: GateCheckReason.FAILED_COVERAGE,
        StatisticsMetricStatus.MISSING: GateCheckReason.MISSING_VALUE,
    }.get(status, GateCheckReason.NOT_SCORABLE)


def _coverage_ppm(value: Any) -> int:
    coverage = value.coverage
    total = coverage.total_trial_count
    if total <= 0:
        return 0
    values = []
    for source in coverage.metric_sources:
        usable = max(0, source.included_trial_count - source.zero_denominator_count)
        values.append((usable * PPM_SCALE) // total)
    if coverage.judge_request_count:
        judge_usable = max(0, coverage.judge_graded_count - coverage.judge_semantic_unknown_count)
        values.append((judge_usable * PPM_SCALE) // coverage.judge_request_count)
    return min(values) if values else 0


def _coverage_disposition(
    coverage_ppm: int,
    minimum_ppm: int | None,
) -> GateCheckStatus | None:
    """Return the non-passing coverage state, if the policy does not allow it."""

    if minimum_ppm is None:
        return (
            None
            if coverage_ppm == PPM_SCALE
            else GateCheckStatus.INSUFFICIENT_COVERAGE
        )
    return (
        None
        if coverage_ppm >= minimum_ppm
        else GateCheckStatus.INSUFFICIENT_COVERAGE
    )


def _value_for_scope(delta: MetricDeltaV1, scope: GateConstraintScope) -> tuple[Any, Any, str]:
    if scope in {GateConstraintScope.BASELINE_DELTA, GateConstraintScope.CASE_DELTA, GateConstraintScope.TRIAL_DELTA}:
        return delta.absolute_delta, delta.candidate, "delta"
    return delta.candidate.value, delta.candidate, "candidate"


def _local_values(comparison: RunComparisonV1, constraint: MetricConstraintV1) -> list[tuple[GateReferenceKind, str, Any, Any]]:
    values: list[tuple[GateReferenceKind, str, Any, Any]] = []
    for case in comparison.case_deltas:
        case_delta = case.metric_delta(constraint.metric)
        actual, source, mode = _value_for_scope(case_delta, constraint.scope)
        case_ref = stable_id("gate-case-ref-v1", {
            "comparison_id": comparison.comparison_id,
            "case_delta_digest": canonical_sha256(case.to_dict()),
            "metric": constraint.metric.value,
            "scope": constraint.scope.value,
        })
        values.append((GateReferenceKind.CASE, case_ref, actual, source))
        for trial in case.paired_trials:
            trial_delta = trial.metric_delta(constraint.metric)
            trial_actual, trial_source, _ = _value_for_scope(trial_delta, constraint.scope)
            trial_ref = stable_id("gate-trial-ref-v1", {
                "comparison_id": comparison.comparison_id,
                "paired_trial_digest": canonical_sha256(trial.to_dict()),
                "metric": constraint.metric.value,
                "scope": constraint.scope.value,
            })
            values.append((GateReferenceKind.TRIAL, trial_ref, trial_actual, trial_source))
    return values


def _satisfies(operator: GateOperator, actual: Any, threshold: Any) -> bool:
    if type(actual) is not int or type(threshold) is not int:
        return False
    return actual >= threshold if operator is GateOperator.AT_LEAST else actual <= threshold


def _make_check(
    constraint: MetricConstraintV1,
    *,
    status: GateCheckStatus,
    actual: Any,
    coverage_ppm: int | None,
    metric_ref: str | None,
    failure_refs: Sequence[GateFailureRefV1] = (),
    calibration_result_digests: Sequence[str] = (),
    reasons: Sequence[GateCheckReason] = (),
) -> GateCheckV1:
    return GateCheckV1(
        constraint.constraint_id, constraint.metric, constraint.scope,
        constraint.operator, constraint.required, status, actual,
        constraint.threshold, constraint.unit, coverage_ppm,
        constraint.min_coverage_ppm, metric_ref, tuple(sorted(failure_refs, key=_ref_sort_key)),
        tuple(sorted(calibration_result_digests)),
        tuple(sorted(set(reasons), key=lambda item: item.value)),
    )


def _ineligible_check(
    constraint: MetricConstraintV1,
    reasons: Sequence[GateCheckReason],
    *,
    status: GateCheckStatus = GateCheckStatus.NOT_SCORABLE,
    metric_ref: str | None = None,
    actual: Any = None,
    coverage_ppm: int | None = None,
    failure_refs: Sequence[GateFailureRefV1] = (),
    calibrations: Sequence[str] = (),
) -> GateCheckV1:
    return _make_check(
        constraint,
        status=status,
        actual=actual,
        coverage_ppm=coverage_ppm,
        metric_ref=metric_ref,
        failure_refs=failure_refs,
        calibration_result_digests=calibrations,
        reasons=reasons,
    )


def _not_configured_check(
    metric: CoreMetric,
    calibration_result_digests: Sequence[str],
) -> GateCheckV1:
    """Surface an omitted metric without synthesizing a business threshold."""

    # The placeholder operator/threshold are schema transport only.  The
    # status-specific constraint ID above makes this explicitly distinct from
    # a configured MetricConstraintV1 and it is never evaluated.
    constraint = MetricConstraintV1(
        metric,
        GateConstraintScope.CANDIDATE_ABSOLUTE,
        GateOperator.AT_LEAST,
        0,
        _expected_unit(metric),
        False,
        None,
    )
    return GateCheckV1(
        stable_id("gate-not-configured-v1", {"metric": metric.value}),
        constraint.metric,
        constraint.scope,
        constraint.operator,
        False,
        GateCheckStatus.NOT_CONFIGURED,
        None,
        constraint.threshold,
        constraint.unit,
        None,
        None,
        None,
        (),
        tuple(sorted(calibration_result_digests)),
        (GateCheckReason.NOT_CONFIGURED,),
    )


def _unavailable_status(reasons: Sequence[GateCheckReason]) -> GateCheckStatus:
    values = set(reasons)
    if GateCheckReason.NOT_COMPARABLE in values:
        return GateCheckStatus.NOT_COMPARABLE
    if GateCheckReason.CALIBRATION_PENDING_HUMAN_LABELS in values:
        return GateCheckStatus.PENDING
    if (
        GateCheckReason.INSUFFICIENT_COVERAGE in values
        or GateCheckReason.FAILED_COVERAGE in values
        or GateCheckReason.CALIBRATION_INSUFFICIENT_COVERAGE in values
    ):
        return GateCheckStatus.INSUFFICIENT_COVERAGE
    return GateCheckStatus.NOT_SCORABLE


def _evaluate_frozen_policy(
    policy: FrozenGatePolicy,
    comparison: RunComparisonV1,
    calibrations: Mapping[Any, CalibrationResultV1],
) -> GateResultV1:
    """Pure gate logic for a policy already live-verified by its Store."""

    canonical_policy = _validate_frozen_gate_policy(policy)
    canonical_comparison = _canonical_comparison(comparison)
    canonical_calibrations = _normalize_calibrations(calibrations)
    global_reasons = _global_mismatch_reasons(canonical_policy, canonical_comparison, canonical_calibrations)
    checks: list[GateCheckV1] = []
    for constraint in canonical_policy.constraints:
        if global_reasons:
            checks.append(
                _ineligible_check(
                    constraint,
                    global_reasons,
                    status=_unavailable_status(global_reasons),
                    calibrations=canonical_policy.calibration_result_digests,
                )
            )
            continue
        delta = canonical_comparison.metric_delta(constraint.metric)
        metric_ref = delta.delta_id
        required_profiles = _SEMANTIC_PROFILES.get(constraint.metric, ())
        calibration_reasons: list[GateCheckReason] = []
        for profile in required_profiles:
            result = canonical_calibrations.get(profile)
            if result is None:
                calibration_reasons.append(GateCheckReason.CALIBRATION_MISSING)
            elif result.status is not CalibrationStatus.GATE_ELIGIBLE or result.profiles[0].status is not CalibrationStatus.GATE_ELIGIBLE:
                calibration_reasons.append(
                    {
                        CalibrationStatus.PENDING_HUMAN_LABELS: GateCheckReason.CALIBRATION_PENDING_HUMAN_LABELS,
                        CalibrationStatus.INSUFFICIENT_COVERAGE: GateCheckReason.CALIBRATION_INSUFFICIENT_COVERAGE,
                        CalibrationStatus.FAILED_THRESHOLDS: GateCheckReason.CALIBRATION_FAILED_THRESHOLDS,
                    }.get(
                        result.status,
                        GateCheckReason.CALIBRATION_NOT_ELIGIBLE,
                    )
                )
        if calibration_reasons:
            checks.append(
                _ineligible_check(
                    constraint,
                    calibration_reasons,
                    status=_unavailable_status(calibration_reasons),
                    metric_ref=metric_ref,
                    calibrations=canonical_policy.calibration_result_digests,
                )
            )
            continue
        baseline_value = delta.baseline
        candidate_value = delta.candidate
        if baseline_value.status is not StatisticsMetricStatus.AVAILABLE or candidate_value.status is not StatisticsMetricStatus.AVAILABLE:
            statuses = tuple(
                _status_reason(status)
                for status in (baseline_value.status, candidate_value.status)
                if status is not StatisticsMetricStatus.AVAILABLE
            )
            invalid_refs = []
            for kind, ref, local_actual, local_source in _local_values(canonical_comparison, constraint):
                if local_source.status is not StatisticsMetricStatus.AVAILABLE or local_actual is None:
                    reason = _status_reason(local_source.status)
                    invalid_refs.append(
                        GateFailureRefV1(
                            kind,
                            ref,
                            local_actual,
                            constraint.threshold,
                            constraint.unit,
                            reason,
                        )
                    )
            checks.append(
                _ineligible_check(
                    constraint,
                    statuses,
                    status=_unavailable_status(statuses),
                    metric_ref=metric_ref,
                    failure_refs=invalid_refs,
                    calibrations=canonical_policy.calibration_result_digests,
                )
            )
            continue
        if delta.unit is not constraint.unit:
            checks.append(_ineligible_check(constraint, (GateCheckReason.UNIT_MISMATCH,), metric_ref=metric_ref, calibrations=canonical_policy.calibration_result_digests))
            continue
        actual, source, _ = _value_for_scope(delta, constraint.scope)
        coverage = min(_coverage_ppm(baseline_value), _coverage_ppm(candidate_value))
        local = _local_values(canonical_comparison, constraint)
        unavailable_refs = tuple(
            GateFailureRefV1(
                kind,
                ref,
                local_actual,
                constraint.threshold,
                constraint.unit,
                _status_reason(local_source.status),
            )
            for kind, ref, local_actual, local_source in local
            if local_source.status is not StatisticsMetricStatus.AVAILABLE
            or local_actual is None
        )
        coverage_status = _coverage_disposition(
            coverage,
            constraint.min_coverage_ppm,
        )
        partial_coverage_allowed = (
            coverage_status is None
            and constraint.min_coverage_ppm is not None
        )
        if coverage_status is not None or (
            unavailable_refs and not partial_coverage_allowed
        ):
            checks.append(
                _ineligible_check(
                    constraint,
                    (GateCheckReason.INSUFFICIENT_COVERAGE,),
                    status=GateCheckStatus.INSUFFICIENT_COVERAGE,
                    metric_ref=metric_ref,
                    actual=actual,
                    coverage_ppm=coverage,
                    failure_refs=unavailable_refs,
                    calibrations=canonical_policy.calibration_result_digests,
                )
            )
            continue
        relevant_kind = (
            GateReferenceKind.CASE
            if constraint.scope in {
                GateConstraintScope.CASE_ABSOLUTE,
                GateConstraintScope.CASE_DELTA,
            }
            else GateReferenceKind.TRIAL
            if constraint.scope in {
                GateConstraintScope.TRIAL_ABSOLUTE,
                GateConstraintScope.TRIAL_DELTA,
            }
            else None
        )
        relevant = tuple(
            item for item in local if relevant_kind is None or item[0] is relevant_kind
        )
        if relevant_kind is not None:
            scorable_relevant = tuple(
                item
                for item in relevant
                if item[3].status is StatisticsMetricStatus.AVAILABLE
                and item[2] is not None
            )
            if not scorable_relevant:
                checks.append(
                    _ineligible_check(
                        constraint,
                        (GateCheckReason.INSUFFICIENT_COVERAGE,),
                        status=GateCheckStatus.INSUFFICIENT_COVERAGE,
                        metric_ref=metric_ref,
                        coverage_ppm=coverage,
                        failure_refs=unavailable_refs,
                        calibrations=canonical_policy.calibration_result_digests,
                    )
                )
                continue
            local_actuals = tuple(item[2] for item in scorable_relevant)
            if local_actuals:
                actual = (
                    min(local_actuals)
                    if constraint.operator is GateOperator.AT_LEAST
                    else max(local_actuals)
                )
        local_failures: list[GateFailureRefV1] = list(unavailable_refs)
        for kind, ref, local_actual, local_source in local:
            if local_source.status is not StatisticsMetricStatus.AVAILABLE or local_actual is None:
                continue
            elif not _satisfies(constraint.operator, local_actual, constraint.threshold):
                local_failures.append(GateFailureRefV1(kind, ref, local_actual, constraint.threshold, constraint.unit, GateCheckReason.THRESHOLD_FAILED))
        threshold_values = (
            scorable_relevant if relevant_kind is not None else relevant
        )
        passed = (
            all(
                _satisfies(constraint.operator, item[2], constraint.threshold)
                for item in threshold_values
            )
            if relevant_kind is not None
            else _satisfies(constraint.operator, actual, constraint.threshold)
        )
        if passed:
            checks.append(_make_check(constraint, status=GateCheckStatus.PASS, actual=actual, coverage_ppm=coverage, metric_ref=metric_ref, failure_refs=local_failures, calibration_result_digests=canonical_policy.calibration_result_digests))
        else:
            if not local_failures and local:
                local_failures = [GateFailureRefV1(kind, ref, local_actual, constraint.threshold, constraint.unit, GateCheckReason.THRESHOLD_FAILED) for kind, ref, local_actual, _source in local]
            checks.append(_make_check(constraint, status=GateCheckStatus.FAIL, actual=actual, coverage_ppm=coverage, metric_ref=metric_ref, failure_refs=local_failures, calibration_result_digests=canonical_policy.calibration_result_digests, reasons=(GateCheckReason.THRESHOLD_FAILED,)))
    configured_metrics = {item.metric for item in canonical_policy.constraints}
    checks.extend(
        _not_configured_check(metric, canonical_policy.calibration_result_digests)
        for metric in CoreMetric
        if metric not in configured_metrics
    )
    if canonical_policy.eligibility is GateEligibility.DIAGNOSTIC_ONLY:
        decision = GateDecision.INELIGIBLE
    elif global_reasons or any(
        item.required
        and item.status not in {GateCheckStatus.PASS, GateCheckStatus.FAIL}
        for item in checks
    ):
        decision = GateDecision.INELIGIBLE
    elif not any(item.required for item in checks):
        # Defensive non-vacuity rule for a forged/legacy prepared policy.
        decision = GateDecision.INELIGIBLE
    elif any(item.status is GateCheckStatus.FAIL and item.required for item in checks):
        decision = GateDecision.BLOCK
    else:
        decision = GateDecision.PROMOTE
    return GateResultV1.create(
        policy_digest=canonical_policy.policy_digest,
        policy_artifact_id=policy.artifact_id,
        policy_receipt_digest=policy.receipt_digest,
        comparison_id=canonical_comparison.comparison_id,
        decision=decision,
        checks=tuple(checks),
    )


def evaluate_gate(
    store: AnalysisArtifactStore,
    policy: FrozenGatePolicy,
    comparison: RunComparisonV1,
    calibrations: Mapping[Any, CalibrationResultV1],
) -> GateResultV1:
    """Live-verify and evaluate only a Store-published frozen policy.

    The raw ``GatePolicyV1`` remains proposal/serialization data and is
    intentionally rejected here.  Every call re-opens the policy artifact and
    receipt through the concrete Store before any comparison is evaluated.
    """

    if type(store) is not AnalysisArtifactStore:
        raise TypeError("store must be a concrete AnalysisArtifactStore")
    verified_policy = store._require_frozen_gate_policy(policy)
    return _evaluate_frozen_policy(verified_policy, comparison, calibrations)


__all__ = [
    "GATE_POLICY_SCHEMA_VERSION",
    "GATE_RESULT_SCHEMA_VERSION",
    "GATE_ALGORITHM_VERSION",
    "GateError",
    "GateEligibility",
    "GateDecision",
    "GateCheckStatus",
    "GateConstraintScope",
    "GateOperator",
    "ConstraintScope",
    "MetricConstraintScope",
    "ConstraintOperator",
    "MetricConstraintOperator",
    "GateCheckReason",
    "GateReferenceKind",
    "MetricUnit",
    "MetricConstraintV1",
    "GatePolicyV1",
    "FrozenGatePolicy",
    "GateFailureRefV1",
    "GateCheckV1",
    "GateResultV1",
    "SEMANTIC_METRIC_PROFILES",
    "prepare_gate_policy",
    "evaluate_gate",
]


SEMANTIC_METRIC_PROFILES = dict(_SEMANTIC_PROFILES)

# Compatibility aliases keep callers on closed enums without accepting
# arbitrary scope/operator strings.
ConstraintScope = GateConstraintScope
MetricConstraintScope = GateConstraintScope
ConstraintOperator = GateOperator
MetricConstraintOperator = GateOperator
