"""Blind, source-bound calibration of the four semantic Judge profiles.

Calibration consumes already-hydrated Evaluation artifacts.  It never creates
an Agent, Judge, provider adapter, acquisition client, or network request.
"""

from __future__ import annotations

import hashlib
import os
import stat
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .artifacts import (
    ArtifactConflictError,
    ArtifactIntegrityError,
    ArtifactSecurityError,
    ArtifactStore,
    _ReadBudget,
    _hardlinked_file,
    _unsafe_node,
)
from .comparison import VerifiedRunEvaluation
from .config import JudgeKind, validate_path_segment
from .intent_evaluator import IntentJudgeRelation, IntentMatchKind
from .judge import (
    ActionabilityAssessment,
    BlindJudgeInput,
    FindingMatchRelation,
    JudgeExecutionResult,
    JudgeFailureCode,
    JudgeRunStatus,
    JudgeTask,
    NovelFactuality,
    SeverityAssessment,
)
from .models import (
    EvidenceSupport,
    FindingSeverity,
    SchemaError,
    _JsonModel,
    _strict_json_loads,
    canonical_json,
    canonical_json_bytes,
    canonical_sha256,
    stable_id,
)
from .review_evaluator import FindingMatchKind


CALIBRATION_SELECTION_POLICY_SCHEMA_VERSION = "calibration_selection_policy_v1"
CALIBRATION_ITEM_SCHEMA_VERSION = "calibration_item_v1"
CALIBRATION_SELECTION_RECORD_SCHEMA_VERSION = "calibration_selection_record_v1"
CALIBRATION_PACKAGE_SCHEMA_VERSION = "calibration_package_v1"
CALIBRATION_PACKAGE_MANIFEST_SCHEMA_VERSION = "calibration_package_manifest_v1"
HUMAN_LABEL_SCHEMA_VERSION = "human_label_v1"
HUMAN_LABEL_SET_SCHEMA_VERSION = "human_label_set_v1"
HUMAN_REVIEWER_PROVENANCE_SCHEMA_VERSION = "human_reviewer_provenance_v1"
ADJUDICATION_SCHEMA_VERSION = "adjudication_v1"
HUMAN_ADJUDICATION_SCHEMA_VERSION = ADJUDICATION_SCHEMA_VERSION
PROFILE_CALIBRATION_SCHEMA_VERSION = "profile_calibration_v1"
AUXILIARY_CALIBRATION_SCHEMA_VERSION = "auxiliary_calibration_v1"
CALIBRATION_RESULT_SCHEMA_VERSION = "calibration_result_v1"
CALIBRATION_ALGORITHM_VERSION = "blind-judge-calibration-v1"
CALIBRATION_BLINDED_REQUEST_SCHEMA_VERSION = "calibration_blinded_request_v1"
CALIBRATION_EXPORT_RECEIPT_SCHEMA_VERSION = "calibration_export_receipt_v1"

PPM_SCALE = 1_000_000
MAX_CALIBRATION_SEED = (1 << 63) - 1
MAX_CALIBRATION_ITEMS = 100_000
MAX_CALIBRATION_BYTES = 256 * 1024 * 1024
MAX_TRUSTED_PROVENANCE_DIGESTS = 4096
JUDGE_FAILED_OUTCOME = "__judge_failed__"
JUDGE_UNGRADED_OUTCOME = "__ungraded__"
_RESERVED_OUTCOME_SENTINELS = (
    JUDGE_FAILED_OUTCOME,
    JUDGE_UNGRADED_OUTCOME,
)


class CalibrationError(ValueError):
    """A calibration policy, package, label, or replay is invalid."""


class CalibrationStatus(str, Enum):
    PENDING_HUMAN_LABELS = "pending_human_labels"
    INSUFFICIENT_COVERAGE = "insufficient_coverage"
    FAILED_THRESHOLDS = "failed_thresholds"
    GATE_ELIGIBLE = "gate_eligible"


class CalibrationSelectionCategory(str, Enum):
    MANDATORY = "mandatory"
    SEEDED = "seeded"


class CalibrationAuxiliaryDimension(str, Enum):
    SEVERITY_ASSESSMENT = "severity_assessment"
    ACTIONABILITY = "actionability"


class AuxiliaryCalibrationApplicability(str, Enum):
    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"


class ReviewerProvenanceKind(str, Enum):
    HUMAN = "human"
    FIXTURE = "fixture"
    SYNTHETIC = "synthetic"


class KappaNullReason(str, Enum):
    NO_ELIGIBLE_LABELS = "no_eligible_labels"
    ZERO_EXPECTED_DISAGREEMENT = "zero_expected_disagreement"


class ClassMetricNullReason(str, Enum):
    NO_RECORDED_PREDICTIONS = "no_recorded_predictions"
    NO_HUMAN_LABELS = "no_human_labels"


_ALLOWED_LABELS: Mapping[JudgeTask, Tuple[str, ...]] = {
    JudgeTask.INTENT_EQUIVALENCE: tuple(item.value for item in IntentJudgeRelation),
    JudgeTask.FINDING_EQUIVALENCE: tuple(item.value for item in FindingMatchRelation),
    JudgeTask.NOVEL_FACTUALITY: tuple(item.value for item in NovelFactuality),
    JudgeTask.EVIDENCE_SUPPORT: tuple(item.value for item in EvidenceSupport),
}
_SELECTION_REASONS = frozenset(
    {
        "mandatory_semantic_unknown",
        "mandatory_high_critical_fabricated",
        "mandatory_deterministic_conflict",
        "seeded_normal_stratum",
    }
)
_MANDATORY_REASONS = _SELECTION_REASONS - {"seeded_normal_stratum"}
_HEX = frozenset("0123456789abcdef")
_FORBIDDEN_BLIND_KEYS = frozenset(
    {
        "agent",
        "agent_id",
        "agent_name",
        "baseline",
        "baseline_id",
        "candidate",
        "candidate_id",
        "decision",
        "expected_winner",
        "failure",
        "judge_decision",
        "judge_failure",
        "judge_result",
        "judge_result_digest",
        "model",
        "provider",
        "request_id",
        "source_request_id",
    }
)
_FORBIDDEN_BLIND_VALUE_MARKERS = (
    "expected winner",
    "judge decision",
    "judge failure",
    "judge result",
)
_STRUCTURED_BLIND_VALUE_MARKERS = (
    "baseline",
    "candidate",
    "decision",
    "failure",
)
_STRUCTURED_BLIND_VALUE_STATUSES = (
    "baseline",
    "candidate",
    "error",
    "failed",
    "loser",
    "rejected",
    "selected",
    "success",
    "winner",
)
_IDENTITY_FIELDS = frozenset(
    {
        "adapter_config_digest",
        "adapter_id",
        "adapter_version",
        "agent_config_digest",
        "agent_id",
        "agent_name",
        "agent_version",
        "commit",
        "evaluation_id",
        "evaluation_revision",
        "evaluator_config_digest",
        "evaluator_id",
        "judge_id",
        "judge_version",
        "model",
        "model_artifact_digest",
        "prompt_config_digest",
        "provider",
        "run_id",
        "run_instance_key",
        "system_prompt_digest",
        "system_prompt_version",
    }
)
_BASELINE_CANDIDATE_IDENTITY_FIELDS = frozenset(
    {
        "baseline",
        "baseline_agent_id",
        "baseline_id",
        "baseline_identity",
        "candidate",
        "candidate_agent_id",
        "candidate_id",
        "candidate_identity",
    }
)


def _error(message: str) -> CalibrationError:
    return CalibrationError(message)


def _exact(value: Any, fields: Iterable[str], context: str) -> Dict[str, Any]:
    expected = set(fields)
    if type(value) is not dict or set(value) != expected or len(value) != len(expected):
        raise _error(f"{context} has unknown or missing fields")
    return value


def _array(value: Any, context: str, maximum: int = MAX_CALIBRATION_ITEMS) -> list[Any]:
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
        raise _error(f"{context} is outside its integer bounds")
    return value


def _digest(value: Any, context: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise _error(f"{context} must be a lowercase SHA-256 digest")
    return value


def _text(value: Any, context: str, maximum: int = 4096) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > maximum
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        raise _error(f"{context} must be bounded non-empty text")
    return value


def _optional_text(value: Any, context: str, maximum: int = 4096) -> Optional[str]:
    return None if value is None else _text(value, context, maximum)


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


def _signed_ratio_ppm(numerator: int, denominator: int) -> int:
    sign = -1 if numerator < 0 else 1
    return sign * ((abs(numerator) * PPM_SCALE + denominator // 2) // denominator)


def _canonical_payload(value: Any, context: str) -> str:
    try:
        encoded = canonical_json(value)
        replayed = _strict_json_loads(encoded, MAX_CALIBRATION_BYTES, context)
    except (SchemaError, TypeError, ValueError) as exc:
        raise _error(f"{context} is not canonical JSON") from exc
    if type(replayed) is not dict:
        raise _error(f"{context} must be a JSON object")
    return encoded


def _allowed_labels(profile: JudgeTask) -> Tuple[str, ...]:
    if type(profile) is not JudgeTask:
        raise _error("profile must be a JudgeTask")
    return _ALLOWED_LABELS[profile]


def _identity_strings(value: Any) -> Tuple[str, ...]:
    if type(value) is str:
        return (value,) if value else ()
    if type(value) is dict:
        result: list[str] = []
        for child in value.values():
            result.extend(_identity_strings(child))
        return tuple(result)
    if type(value) in (list, tuple):
        result = []
        for child in value:
            result.extend(_identity_strings(child))
        return tuple(result)
    return ()


def _collapsed_blind_token(value: str) -> str:
    """NFKC/case-fold text and retain only Unicode letters and numbers."""

    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(
        character
        for character in normalized
        if unicodedata.category(character)[:1] in {"L", "N"}
    )


def _contains_structured_blind_value(
    normalized: str,
    *,
    markers: Sequence[str],
    allowed_values: Sequence[str],
) -> bool:
    for marker in markers:
        offset = 0
        while True:
            start = normalized.find(marker, offset)
            if start < 0:
                break
            end = start + len(marker)
            before_is_boundary = start == 0 or not normalized[start - 1].isalnum()
            if (
                before_is_boundary
                and end < len(normalized)
                and not normalized[end].isalnum()
            ):
                value_start = end
                while value_start < len(normalized) and not normalized[value_start].isalnum():
                    value_start += 1
                tail = _collapsed_blind_token(normalized[value_start:])
                if any(tail.startswith(value) for value in allowed_values):
                    return True
            offset = start + 1
    return False


def _forbidden_identity_values(
    evaluation: VerifiedRunEvaluation,
) -> Tuple[str, ...]:
    if type(evaluation) is not VerifiedRunEvaluation:
        raise TypeError("evaluation must be a VerifiedRunEvaluation")
    roots = (
        evaluation.source_binding.to_dict(),
        evaluation.run_config.to_dict(),
        evaluation.bundle.evaluator_execution.to_dict(),
        {
            "evaluation_id": evaluation.evaluation_id,
            "evaluation_revision": evaluation.bundle.evaluation_revision,
            "run_id": evaluation.run_id,
        },
    )
    values: set[str] = set()

    def visit(value: Any) -> None:
        if type(value) is not dict:
            return
        for raw_key, child in value.items():
            key = str(raw_key).casefold().replace("-", "_").replace(" ", "_")
            if (
                key in _IDENTITY_FIELDS
                or key in _BASELINE_CANDIDATE_IDENTITY_FIELDS
                or key.endswith("_config_digest")
                or (
                    type(child) is str
                    and any(
                        marker in key
                        for marker in ("baseline", "candidate", "prompt")
                    )
                )
            ):
                values.update(_identity_strings(child))
            if type(child) is dict:
                visit(child)
            elif type(child) in (list, tuple):
                for nested in child:
                    visit(nested)

    for root in roots:
        visit(root)
    return tuple(sorted(values, key=lambda item: (item.casefold(), item)))


def _assert_blind_payload(
    value: Any,
    *,
    forbidden_identity_values: Sequence[str],
    context: str,
) -> None:
    identities = tuple(
        (
            item,
            unicodedata.normalize("NFKC", item).casefold(),
            _collapsed_blind_token(item),
        )
        for item in forbidden_identity_values
        if type(item) is str and item
    )
    forbidden_key_tokens = tuple(
        _collapsed_blind_token(item) for item in _FORBIDDEN_BLIND_KEYS
    )
    forbidden_value_tokens = tuple(
        _collapsed_blind_token(item) for item in _FORBIDDEN_BLIND_VALUE_MARKERS
    )
    structured_value_markers = tuple(
        unicodedata.normalize("NFKC", item).casefold()
        for item in _STRUCTURED_BLIND_VALUE_MARKERS
    )
    structured_allowed_values = tuple(
        sorted(
            {
                *(
                    _collapsed_blind_token(label)
                    for labels in _ALLOWED_LABELS.values()
                    for label in labels
                ),
                *(
                    _collapsed_blind_token(item.value)
                    for item in SeverityAssessment
                ),
                *(
                    _collapsed_blind_token(item.value)
                    for item in ActionabilityAssessment
                ),
                *(
                    _collapsed_blind_token(item.value)
                    for item in JudgeFailureCode
                ),
                *(
                    _collapsed_blind_token(item.value)
                    for item in JudgeRunStatus
                ),
                *(
                    _collapsed_blind_token(item)
                    for item in _RESERVED_OUTCOME_SENTINELS
                ),
                *(
                    _collapsed_blind_token(item)
                    for item in _STRUCTURED_BLIND_VALUE_STATUSES
                ),
            },
            key=lambda item: (-len(item), item),
        )
    )
    reserved_outcome_sentinels = tuple(
        unicodedata.normalize("NFKC", item).casefold()
        for item in _RESERVED_OUTCOME_SENTINELS
    )

    def visit(current: Any, path: str) -> None:
        if type(current) is dict:
            for raw_key, child in current.items():
                if type(raw_key) is not str:
                    raise ArtifactSecurityError(
                        f"{context} contains a non-string key at {path}"
                    )
                normalized_key = unicodedata.normalize("NFKC", raw_key).casefold()
                collapsed_key = _collapsed_blind_token(raw_key)
                if any(marker in collapsed_key for marker in forbidden_key_tokens):
                    raise ArtifactSecurityError(
                        f"{context} contains forbidden key {raw_key!r} at {path}"
                    )
                for identity, normalized_identity, collapsed_identity in identities:
                    if normalized_key == normalized_identity or (
                        len(normalized_identity) >= 4
                        and normalized_identity in normalized_key
                    ) or (
                        collapsed_identity
                        and (
                            collapsed_key == collapsed_identity
                            or (
                                len(collapsed_identity) >= 4
                                and collapsed_identity in collapsed_key
                            )
                        )
                    ):
                        raise ArtifactSecurityError(
                            f"{context} contains forbidden source identity "
                            f"{identity!r} in a key at {path}"
                        )
                visit(child, f"{path}.{raw_key}")
            return
        if type(current) in (list, tuple):
            for index, child in enumerate(current):
                visit(child, f"{path}[{index}]")
            return
        if type(current) is not str:
            return
        normalized = unicodedata.normalize("NFKC", current).casefold()
        collapsed = _collapsed_blind_token(current)
        for identity, normalized_identity, collapsed_identity in identities:
            if normalized == normalized_identity or (
                len(normalized_identity) >= 4 and normalized_identity in normalized
            ) or (
                collapsed_identity
                and (
                    collapsed == collapsed_identity
                    or (
                        len(collapsed_identity) >= 4
                        and collapsed_identity in collapsed
                    )
                )
            ):
                raise ArtifactSecurityError(
                    f"{context} contains forbidden source identity {identity!r} at {path}"
                )
        if (
            any(sentinel in normalized for sentinel in reserved_outcome_sentinels)
            or any(marker in collapsed for marker in forbidden_value_tokens)
            or _contains_structured_blind_value(
                normalized,
                markers=structured_value_markers,
                allowed_values=structured_allowed_values,
            )
        ):
            raise ArtifactSecurityError(
                f"{context} contains forbidden Judge-result text at {path}"
            )

    visit(value, "$")


@dataclass(frozen=True)
class CalibrationSelectionPolicyV1(_JsonModel):
    schema_version: str
    algorithm_version: str
    selection_seed: int
    max_items_per_profile: int
    max_normal_items_per_stratum: int
    minimum_human_labels: int
    minimum_human_coverage_ppm: int
    minimum_labels_per_class: int
    minimum_exact_agreement_ppm: int
    minimum_cohen_kappa_ppm: int
    minimum_auxiliary_human_coverage_ppm: int
    minimum_auxiliary_labels_per_class: int
    minimum_auxiliary_exact_agreement_ppm: int
    minimum_auxiliary_cohen_kappa_ppm: Optional[int]
    trusted_reviewer_provenance_digests: Tuple[str, ...]
    trusted_adjudicator_provenance_digests: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != CALIBRATION_SELECTION_POLICY_SCHEMA_VERSION:
            raise _error("CalibrationSelectionPolicyV1 schema_version is unsupported")
        if self.algorithm_version != CALIBRATION_ALGORITHM_VERSION:
            raise _error("CalibrationSelectionPolicyV1 algorithm_version is unsupported")
        _integer(
            self.selection_seed,
            "CalibrationSelectionPolicyV1.selection_seed",
            maximum=MAX_CALIBRATION_SEED,
        )
        maximum = _integer(
            self.max_items_per_profile,
            "CalibrationSelectionPolicyV1.max_items_per_profile",
            minimum=1,
            maximum=MAX_CALIBRATION_ITEMS,
        )
        _integer(
            self.max_normal_items_per_stratum,
            "CalibrationSelectionPolicyV1.max_normal_items_per_stratum",
            minimum=1,
            maximum=maximum,
        )
        _integer(
            self.minimum_human_labels,
            "CalibrationSelectionPolicyV1.minimum_human_labels",
            maximum=maximum,
        )
        _integer(
            self.minimum_human_coverage_ppm,
            "CalibrationSelectionPolicyV1.minimum_human_coverage_ppm",
            maximum=PPM_SCALE,
        )
        _integer(
            self.minimum_labels_per_class,
            "CalibrationSelectionPolicyV1.minimum_labels_per_class",
            maximum=maximum,
        )
        _integer(
            self.minimum_exact_agreement_ppm,
            "CalibrationSelectionPolicyV1.minimum_exact_agreement_ppm",
            maximum=PPM_SCALE,
        )
        _integer(
            self.minimum_cohen_kappa_ppm,
            "CalibrationSelectionPolicyV1.minimum_cohen_kappa_ppm",
            minimum=-PPM_SCALE,
            maximum=PPM_SCALE,
        )
        _integer(
            self.minimum_auxiliary_human_coverage_ppm,
            "CalibrationSelectionPolicyV1.minimum_auxiliary_human_coverage_ppm",
            maximum=PPM_SCALE,
        )
        _integer(
            self.minimum_auxiliary_labels_per_class,
            "CalibrationSelectionPolicyV1.minimum_auxiliary_labels_per_class",
            maximum=maximum,
        )
        _integer(
            self.minimum_auxiliary_exact_agreement_ppm,
            "CalibrationSelectionPolicyV1.minimum_auxiliary_exact_agreement_ppm",
            maximum=PPM_SCALE,
        )
        if self.minimum_auxiliary_cohen_kappa_ppm is not None:
            _integer(
                self.minimum_auxiliary_cohen_kappa_ppm,
                "CalibrationSelectionPolicyV1.minimum_auxiliary_cohen_kappa_ppm",
                minimum=-PPM_SCALE,
                maximum=PPM_SCALE,
            )
        for name in (
            "trusted_reviewer_provenance_digests",
            "trusted_adjudicator_provenance_digests",
        ):
            values = tuple(getattr(self, name))
            if (
                len(values) > MAX_TRUSTED_PROVENANCE_DIGESTS
                or values != tuple(sorted(set(values)))
            ):
                raise _error(
                    f"CalibrationSelectionPolicyV1.{name} is not canonical"
                )
            for value in values:
                _digest(value, f"CalibrationSelectionPolicyV1.{name}")
            object.__setattr__(self, name, values)

    @classmethod
    def from_dict(cls, value: Any) -> "CalibrationSelectionPolicyV1":
        payload = _exact(
            value,
            (
                "schema_version",
                "algorithm_version",
                "selection_seed",
                "max_items_per_profile",
                "max_normal_items_per_stratum",
                "minimum_human_labels",
                "minimum_human_coverage_ppm",
                "minimum_labels_per_class",
                "minimum_exact_agreement_ppm",
                "minimum_cohen_kappa_ppm",
                "minimum_auxiliary_human_coverage_ppm",
                "minimum_auxiliary_labels_per_class",
                "minimum_auxiliary_exact_agreement_ppm",
                "minimum_auxiliary_cohen_kappa_ppm",
                "trusted_reviewer_provenance_digests",
                "trusted_adjudicator_provenance_digests",
            ),
            "CalibrationSelectionPolicyV1",
        )
        return cls(
            **{
                **payload,
                "trusted_reviewer_provenance_digests": tuple(
                    _array(
                        payload["trusted_reviewer_provenance_digests"],
                        "trusted_reviewer_provenance_digests",
                        MAX_TRUSTED_PROVENANCE_DIGESTS,
                    )
                ),
                "trusted_adjudicator_provenance_digests": tuple(
                    _array(
                        payload["trusted_adjudicator_provenance_digests"],
                        "trusted_adjudicator_provenance_digests",
                        MAX_TRUSTED_PROVENANCE_DIGESTS,
                    )
                ),
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm_version": self.algorithm_version,
            "selection_seed": self.selection_seed,
            "max_items_per_profile": self.max_items_per_profile,
            "max_normal_items_per_stratum": self.max_normal_items_per_stratum,
            "minimum_human_labels": self.minimum_human_labels,
            "minimum_human_coverage_ppm": self.minimum_human_coverage_ppm,
            "minimum_labels_per_class": self.minimum_labels_per_class,
            "minimum_exact_agreement_ppm": self.minimum_exact_agreement_ppm,
            "minimum_cohen_kappa_ppm": self.minimum_cohen_kappa_ppm,
            "minimum_auxiliary_human_coverage_ppm": (
                self.minimum_auxiliary_human_coverage_ppm
            ),
            "minimum_auxiliary_labels_per_class": (
                self.minimum_auxiliary_labels_per_class
            ),
            "minimum_auxiliary_exact_agreement_ppm": (
                self.minimum_auxiliary_exact_agreement_ppm
            ),
            "minimum_auxiliary_cohen_kappa_ppm": (
                self.minimum_auxiliary_cohen_kappa_ppm
            ),
            "trusted_reviewer_provenance_digests": list(
                self.trusted_reviewer_provenance_digests
            ),
            "trusted_adjudicator_provenance_digests": list(
                self.trusted_adjudicator_provenance_digests
            ),
        }


def _item_identity(
    profile: JudgeTask,
    rubric_version: str,
    context_builder_version: str,
    payload: Mapping[str, Any],
) -> str:
    return stable_id(
        "calibration-item-v1",
        profile.value,
        rubric_version,
        context_builder_version,
        dict(payload),
    )


@dataclass(frozen=True)
class CalibrationItemV1(_JsonModel):
    schema_version: str
    calibration_item_id: str
    profile: JudgeTask
    rubric_id: str
    rubric_version: str
    rubric_digest: str
    context_builder_version: str
    dimension: Optional[str]
    blinded_request_payload_json: str
    payload_digest: str
    allowed_labels: Tuple[str, ...]
    source_digest: str

    def __post_init__(self) -> None:
        if self.schema_version != CALIBRATION_ITEM_SCHEMA_VERSION:
            raise _error("CalibrationItemV1 schema_version is unsupported")
        if type(self.profile) is not JudgeTask:
            raise _error("CalibrationItemV1.profile is invalid")
        validate_path_segment(self.calibration_item_id, "calibration_item_id")
        _text(self.rubric_id, "CalibrationItemV1.rubric_id")
        _text(self.rubric_version, "CalibrationItemV1.rubric_version")
        _digest(self.rubric_digest, "CalibrationItemV1.rubric_digest")
        _text(
            self.context_builder_version,
            "CalibrationItemV1.context_builder_version",
        )
        if self.dimension is not None:
            _text(self.dimension, "CalibrationItemV1.dimension")
        if (self.profile is JudgeTask.INTENT_EQUIVALENCE) != (
            self.dimension is not None
        ):
            raise _error("CalibrationItemV1 Intent dimension binding is incomplete")
        try:
            payload = _strict_json_loads(
                self.blinded_request_payload_json,
                MAX_CALIBRATION_BYTES,
                "CalibrationItemV1.blinded_request_payload",
            )
        except (SchemaError, ValueError) as exc:
            raise _error("CalibrationItemV1 blinded payload is invalid") from exc
        if type(payload) is not dict or canonical_json(payload) != self.blinded_request_payload_json:
            raise _error("CalibrationItemV1 blinded payload is not canonical")
        _assert_blind_payload(
            payload,
            forbidden_identity_values=(),
            context="CalibrationItemV1 blinded payload",
        )
        if self.payload_digest != canonical_sha256(payload):
            raise _error("CalibrationItemV1 payload_digest is not canonical")
        labels = tuple(self.allowed_labels)
        if labels != _allowed_labels(self.profile):
            raise _error("CalibrationItemV1 allowed_labels are not canonical")
        object.__setattr__(self, "allowed_labels", labels)
        _digest(self.source_digest, "CalibrationItemV1.source_digest")
        expected_id = _item_identity(
            self.profile,
            self.rubric_version,
            self.context_builder_version,
            payload,
        )
        if self.calibration_item_id != expected_id:
            raise _error("CalibrationItemV1 calibration_item_id is not canonical")

    @property
    def blinded_request_payload(self) -> Dict[str, Any]:
        return _strict_json_loads(
            self.blinded_request_payload_json,
            MAX_CALIBRATION_BYTES,
            "CalibrationItemV1.blinded_request_payload",
        )

    @classmethod
    def from_dict(cls, value: Any) -> "CalibrationItemV1":
        payload = _exact(
            value,
            (
                "schema_version",
                "calibration_item_id",
                "profile",
                "rubric_id",
                "rubric_version",
                "rubric_digest",
                "context_builder_version",
                "dimension",
                "blinded_request_payload",
                "payload_digest",
                "allowed_labels",
                "source_digest",
            ),
            "CalibrationItemV1",
        )
        labels = _array(payload["allowed_labels"], "CalibrationItemV1.allowed_labels", 16)
        return cls(
            schema_version=payload["schema_version"],
            calibration_item_id=payload["calibration_item_id"],
            profile=_enum(JudgeTask, payload["profile"], "CalibrationItemV1.profile"),
            rubric_id=payload["rubric_id"],
            rubric_version=payload["rubric_version"],
            rubric_digest=payload["rubric_digest"],
            context_builder_version=payload["context_builder_version"],
            dimension=payload["dimension"],
            blinded_request_payload_json=_canonical_payload(
                payload["blinded_request_payload"],
                "CalibrationItemV1.blinded_request_payload",
            ),
            payload_digest=payload["payload_digest"],
            allowed_labels=tuple(labels),
            source_digest=payload["source_digest"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "calibration_item_id": self.calibration_item_id,
            "profile": self.profile.value,
            "rubric_id": self.rubric_id,
            "rubric_version": self.rubric_version,
            "rubric_digest": self.rubric_digest,
            "context_builder_version": self.context_builder_version,
            "dimension": self.dimension,
            "blinded_request_payload": self.blinded_request_payload,
            "payload_digest": self.payload_digest,
            "allowed_labels": list(self.allowed_labels),
            "source_digest": self.source_digest,
        }


@dataclass(frozen=True)
class CalibrationSelectionRecordV1(_JsonModel):
    schema_version: str
    selection_record_id: str
    calibration_item_id: str
    item_digest: str
    source_digest: str
    selection_order: int
    selection_seed: int

    def __post_init__(self) -> None:
        if self.schema_version != CALIBRATION_SELECTION_RECORD_SCHEMA_VERSION:
            raise _error("CalibrationSelectionRecordV1 schema_version is unsupported")
        validate_path_segment(self.selection_record_id, "selection_record_id")
        validate_path_segment(self.calibration_item_id, "calibration_item_id")
        _digest(self.item_digest, "CalibrationSelectionRecordV1.item_digest")
        _digest(self.source_digest, "CalibrationSelectionRecordV1.source_digest")
        _integer(self.selection_order, "selection_order", minimum=1)
        _integer(self.selection_seed, "selection_seed", maximum=MAX_CALIBRATION_SEED)
        expected = stable_id(
            "calibration-selection-v1",
            {
                "schema_version": self.schema_version,
                "calibration_item_id": self.calibration_item_id,
                "item_digest": self.item_digest,
                "source_digest": self.source_digest,
                "selection_order": self.selection_order,
                "selection_seed": self.selection_seed,
            },
        )
        if self.selection_record_id != expected:
            raise _error("Calibration selection_record_id is not canonical")

    @classmethod
    def create(
        cls,
        *,
        calibration_item_id: str,
        item_digest: str,
        source_digest: str,
        selection_order: int,
        selection_seed: int,
    ) -> "CalibrationSelectionRecordV1":
        identity = {
            "schema_version": CALIBRATION_SELECTION_RECORD_SCHEMA_VERSION,
            "calibration_item_id": calibration_item_id,
            "item_digest": item_digest,
            "source_digest": source_digest,
            "selection_order": selection_order,
            "selection_seed": selection_seed,
        }
        return cls(
            selection_record_id=stable_id("calibration-selection-v1", identity),
            schema_version=identity["schema_version"],
            calibration_item_id=calibration_item_id,
            item_digest=item_digest,
            source_digest=source_digest,
            selection_order=selection_order,
            selection_seed=selection_seed,
        )

    @classmethod
    def from_dict(cls, value: Any) -> "CalibrationSelectionRecordV1":
        payload = _exact(
            value,
            (
                "schema_version",
                "selection_record_id",
                "calibration_item_id",
                "item_digest",
                "source_digest",
                "selection_order",
                "selection_seed",
            ),
            "CalibrationSelectionRecordV1",
        )
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "selection_record_id": self.selection_record_id,
            "calibration_item_id": self.calibration_item_id,
            "item_digest": self.item_digest,
            "source_digest": self.source_digest,
            "selection_order": self.selection_order,
            "selection_seed": self.selection_seed,
        }


@dataclass(frozen=True)
class CalibrationPackageV1(_JsonModel):
    schema_version: str
    package_id: str
    profile: JudgeTask
    policy: CalibrationSelectionPolicyV1
    source_digest: str
    payload_digest: str
    selection_digest: str
    items: Tuple[CalibrationItemV1, ...]
    selection_records: Tuple[CalibrationSelectionRecordV1, ...]
    status: CalibrationStatus

    def __post_init__(self) -> None:
        if self.schema_version != CALIBRATION_PACKAGE_SCHEMA_VERSION:
            raise _error("CalibrationPackageV1 schema_version is unsupported")
        validate_path_segment(self.package_id, "calibration package_id")
        if type(self.profile) is not JudgeTask:
            raise _error("CalibrationPackageV1.profile is invalid")
        if type(self.policy) is not CalibrationSelectionPolicyV1:
            raise _error("CalibrationPackageV1.policy is invalid")
        replayed_policy = CalibrationSelectionPolicyV1.from_dict(self.policy.to_dict())
        if replayed_policy != self.policy:
            raise _error("CalibrationPackageV1.policy is not canonical")
        object.__setattr__(self, "policy", replayed_policy)
        _digest(self.source_digest, "CalibrationPackageV1.source_digest")
        _digest(self.payload_digest, "CalibrationPackageV1.payload_digest")
        _digest(self.selection_digest, "CalibrationPackageV1.selection_digest")
        items = tuple(self.items)
        records = tuple(self.selection_records)
        if (
            len(items) > MAX_CALIBRATION_ITEMS
            or any(type(item) is not CalibrationItemV1 for item in items)
            or tuple(item.calibration_item_id for item in items)
            != tuple(sorted(item.calibration_item_id for item in items))
            or len({item.calibration_item_id for item in items}) != len(items)
        ):
            raise _error("CalibrationPackageV1 items are not canonical")
        if (
            len(records) > self.policy.max_items_per_profile
            or any(type(item) is not CalibrationSelectionRecordV1 for item in records)
            or tuple(item.selection_order for item in records)
            != tuple(range(1, len(records) + 1))
            or len({item.selection_record_id for item in records}) != len(records)
        ):
            raise _error("CalibrationPackageV1 selection records are not canonical")
        item_map = {item.calibration_item_id: item for item in items}
        if set(item_map) != {record.calibration_item_id for record in records}:
            raise _error("CalibrationPackageV1 items and selections differ")
        if any(
            record.item_digest != item_map[record.calibration_item_id].digest()
            or record.selection_seed != self.policy.selection_seed
            for record in records
        ):
            raise _error("CalibrationPackageV1 selection binding is invalid")
        if any(item.profile is not self.profile for item in items):
            raise _error("CalibrationPackageV1 mixes profiles")
        if self.payload_digest != canonical_sha256([item.to_dict() for item in items]):
            raise _error("CalibrationPackageV1 payload_digest is not canonical")
        if self.selection_digest != canonical_sha256(
            [item.to_dict() for item in records]
        ):
            raise _error("CalibrationPackageV1 selection_digest is not canonical")
        if self.status is not CalibrationStatus.PENDING_HUMAN_LABELS:
            raise _error("exported CalibrationPackageV1 must be pending_human_labels")
        expected_id = stable_id(
            "calibration-package-v1",
            self.profile.value,
            self.policy.digest(),
            self.source_digest,
            self.payload_digest,
            self.selection_digest,
        )
        if self.package_id != expected_id:
            raise _error("CalibrationPackageV1 package_id is not canonical")
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "selection_records", records)
        if len(canonical_json_bytes(self.to_dict())) > MAX_CALIBRATION_BYTES:
            raise _error("CalibrationPackageV1 exceeds its byte limit")

    @classmethod
    def from_dict(cls, value: Any) -> "CalibrationPackageV1":
        payload = _exact(
            value,
            (
                "schema_version",
                "package_id",
                "profile",
                "policy",
                "source_digest",
                "payload_digest",
                "selection_digest",
                "items",
                "selection_records",
                "status",
            ),
            "CalibrationPackageV1",
        )
        items = _array(payload["items"], "CalibrationPackageV1.items")
        records = _array(
            payload["selection_records"],
            "CalibrationPackageV1.selection_records",
        )
        return cls(
            schema_version=payload["schema_version"],
            package_id=payload["package_id"],
            profile=_enum(JudgeTask, payload["profile"], "CalibrationPackageV1.profile"),
            policy=CalibrationSelectionPolicyV1.from_dict(payload["policy"]),
            source_digest=payload["source_digest"],
            payload_digest=payload["payload_digest"],
            selection_digest=payload["selection_digest"],
            items=tuple(CalibrationItemV1.from_dict(item) for item in items),
            selection_records=tuple(
                CalibrationSelectionRecordV1.from_dict(item) for item in records
            ),
            status=_enum(CalibrationStatus, payload["status"], "CalibrationPackageV1.status"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "package_id": self.package_id,
            "profile": self.profile.value,
            "policy": self.policy.to_dict(),
            "source_digest": self.source_digest,
            "payload_digest": self.payload_digest,
            "selection_digest": self.selection_digest,
            "items": [item.to_dict() for item in self.items],
            "selection_records": [item.to_dict() for item in self.selection_records],
            "status": self.status.value,
        }

    def to_blind_dict(self) -> Dict[str, Any]:
        """Return the human-facing payload without source result metadata."""

        return {
            "schema_version": self.schema_version,
            "package_id": self.package_id,
            "package_digest": self.digest(),
            "profile": self.profile.value,
            "policy_digest": self.policy.digest(),
            "selection_seed": self.policy.selection_seed,
            "source_digest": self.source_digest,
            "payload_digest": self.payload_digest,
            "selection_digest": self.selection_digest,
            "status": self.status.value,
            "selections": [record.to_dict() for record in self.selection_records],
            "items": [item.to_dict() for item in self.items],
        }


@dataclass(frozen=True)
class CalibrationPackageManifestV1(_JsonModel):
    schema_version: str
    package_id: str
    package_digest: str
    profile: JudgeTask
    policy_digest: str
    source_digest: str
    payload_digest: str
    selection_digest: str
    selection_seed: int
    selected_count: int
    selection_records: Tuple[CalibrationSelectionRecordV1, ...]
    status: CalibrationStatus

    def __post_init__(self) -> None:
        if self.schema_version != CALIBRATION_PACKAGE_MANIFEST_SCHEMA_VERSION:
            raise _error("CalibrationPackageManifestV1 schema_version is unsupported")
        validate_path_segment(self.package_id, "calibration package_id")
        _digest(self.package_digest, "package_digest")
        if type(self.profile) is not JudgeTask:
            raise _error("Calibration package manifest profile is invalid")
        for name in (
            "policy_digest",
            "source_digest",
            "payload_digest",
            "selection_digest",
        ):
            _digest(getattr(self, name), name)
        _integer(self.selection_seed, "selection_seed", maximum=MAX_CALIBRATION_SEED)
        count = _integer(self.selected_count, "selected_count", maximum=MAX_CALIBRATION_ITEMS)
        records = tuple(self.selection_records)
        if (
            len(records) != count
            or any(type(item) is not CalibrationSelectionRecordV1 for item in records)
            or tuple(item.selection_order for item in records)
            != tuple(range(1, count + 1))
            or canonical_sha256([item.to_dict() for item in records])
            != self.selection_digest
        ):
            raise _error("Calibration package manifest selections are invalid")
        if any(item.selection_seed != self.selection_seed for item in records):
            raise _error("Calibration package manifest seed binding is invalid")
        if self.status is not CalibrationStatus.PENDING_HUMAN_LABELS:
            raise _error("Calibration package manifest status is invalid")
        object.__setattr__(self, "selection_records", records)

    @classmethod
    def from_package(cls, package: CalibrationPackageV1) -> "CalibrationPackageManifestV1":
        if type(package) is not CalibrationPackageV1:
            raise TypeError("package must be a CalibrationPackageV1")
        return cls(
            schema_version=CALIBRATION_PACKAGE_MANIFEST_SCHEMA_VERSION,
            package_id=package.package_id,
            package_digest=package.digest(),
            profile=package.profile,
            policy_digest=package.policy.digest(),
            source_digest=package.source_digest,
            payload_digest=package.payload_digest,
            selection_digest=package.selection_digest,
            selection_seed=package.policy.selection_seed,
            selected_count=len(package.selection_records),
            selection_records=package.selection_records,
            status=package.status,
        )

    @classmethod
    def from_dict(cls, value: Any) -> "CalibrationPackageManifestV1":
        payload = _exact(
            value,
            (
                "schema_version",
                "package_id",
                "package_digest",
                "profile",
                "policy_digest",
                "source_digest",
                "payload_digest",
                "selection_digest",
                "selection_seed",
                "selected_count",
                "selection_records",
                "status",
            ),
            "CalibrationPackageManifestV1",
        )
        records = _array(payload["selection_records"], "selection_records")
        return cls(
            **{
                **payload,
                "profile": _enum(JudgeTask, payload["profile"], "profile"),
                "selection_records": tuple(
                    CalibrationSelectionRecordV1.from_dict(item) for item in records
                ),
                "status": _enum(CalibrationStatus, payload["status"], "status"),
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "package_id": self.package_id,
            "package_digest": self.package_digest,
            "profile": self.profile.value,
            "policy_digest": self.policy_digest,
            "source_digest": self.source_digest,
            "payload_digest": self.payload_digest,
            "selection_digest": self.selection_digest,
            "selection_seed": self.selection_seed,
            "selected_count": self.selected_count,
            "selection_records": [item.to_dict() for item in self.selection_records],
            "status": self.status.value,
        }


@dataclass(frozen=True)
class HumanReviewerProvenanceV1(_JsonModel):
    schema_version: str
    provenance_id: str
    kind: ReviewerProvenanceKind
    reviewer_id: str
    provenance_ref: str
    attestation_ref: Optional[str]

    def __post_init__(self) -> None:
        if self.schema_version != HUMAN_REVIEWER_PROVENANCE_SCHEMA_VERSION:
            raise _error("HumanReviewerProvenanceV1 schema_version is unsupported")
        validate_path_segment(self.provenance_id, "reviewer provenance_id")
        if type(self.kind) is not ReviewerProvenanceKind:
            raise _error("reviewer provenance kind is invalid")
        _text(self.reviewer_id, "reviewer provenance reviewer_id")
        _text(self.provenance_ref, "reviewer provenance_ref")
        _optional_text(self.attestation_ref, "reviewer attestation_ref")
        if self.kind is ReviewerProvenanceKind.HUMAN:
            if self.attestation_ref is None:
                raise _error("human reviewer provenance requires an attestation ref")
        elif self.attestation_ref is not None:
            raise _error("fixture/synthetic provenance cannot carry human attestation")
        expected = stable_id(
            "human-reviewer-provenance-v1",
            {
                "schema_version": self.schema_version,
                "kind": self.kind.value,
                "reviewer_id": self.reviewer_id,
                "provenance_ref": self.provenance_ref,
                "attestation_ref": self.attestation_ref,
            },
        )
        if self.provenance_id != expected:
            raise _error("reviewer provenance_id is not canonical")

    @classmethod
    def create(
        cls,
        *,
        kind: ReviewerProvenanceKind,
        reviewer_id: str,
        provenance_ref: str,
        attestation_ref: Optional[str],
    ) -> "HumanReviewerProvenanceV1":
        if type(kind) is not ReviewerProvenanceKind:
            raise TypeError("kind must be a ReviewerProvenanceKind")
        identity = {
            "schema_version": HUMAN_REVIEWER_PROVENANCE_SCHEMA_VERSION,
            "kind": kind.value,
            "reviewer_id": reviewer_id,
            "provenance_ref": provenance_ref,
            "attestation_ref": attestation_ref,
        }
        return cls(
            provenance_id=stable_id("human-reviewer-provenance-v1", identity),
            schema_version=identity["schema_version"],
            kind=kind,
            reviewer_id=reviewer_id,
            provenance_ref=provenance_ref,
            attestation_ref=attestation_ref,
        )

    @classmethod
    def from_dict(cls, value: Any) -> "HumanReviewerProvenanceV1":
        payload = _exact(
            value,
            (
                "schema_version",
                "provenance_id",
                "kind",
                "reviewer_id",
                "provenance_ref",
                "attestation_ref",
            ),
            "HumanReviewerProvenanceV1",
        )
        return cls(
            **{
                **payload,
                "kind": _enum(
                    ReviewerProvenanceKind,
                    payload["kind"],
                    "reviewer provenance kind",
                ),
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provenance_id": self.provenance_id,
            "kind": self.kind.value,
            "reviewer_id": self.reviewer_id,
            "provenance_ref": self.provenance_ref,
            "attestation_ref": self.attestation_ref,
        }


@dataclass(frozen=True)
class AdjudicationV1(_JsonModel):
    schema_version: str
    adjudication_id: str
    adjudication_ref: str
    package_id: str
    package_digest: str
    calibration_item_id: str
    item_digest: str
    profile: JudgeTask
    original_label_refs: Tuple[str, ...]
    final_primary_label: str
    final_severity_assessment: Optional[SeverityAssessment]
    final_actionability: Optional[ActionabilityAssessment]
    adjudicator_provenance: HumanReviewerProvenanceV1
    blind_attestation: bool

    def __post_init__(self) -> None:
        if self.schema_version != ADJUDICATION_SCHEMA_VERSION:
            raise _error("AdjudicationV1 schema_version is unsupported")
        validate_path_segment(self.adjudication_id, "adjudication_id")
        validate_path_segment(self.package_id, "adjudication package_id")
        validate_path_segment(
            self.calibration_item_id,
            "adjudication calibration_item_id",
        )
        _text(self.adjudication_ref, "adjudication_ref")
        _digest(self.package_digest, "adjudication package_digest")
        _digest(self.item_digest, "adjudication item_digest")
        if type(self.profile) is not JudgeTask:
            raise _error("adjudication profile is invalid")
        refs = tuple(self.original_label_refs)
        if not refs or len(refs) > 16 or refs != tuple(sorted(set(refs))):
            raise _error("adjudication original_label_refs are not canonical")
        for ref in refs:
            _digest(ref, "adjudication original label ref")
        if self.final_primary_label not in _allowed_labels(self.profile):
            raise _error("adjudication final primary label is out of vocabulary")
        if self.final_severity_assessment is not None:
            if type(self.final_severity_assessment) is not SeverityAssessment:
                raise _error("adjudication final severity assessment is invalid")
        if self.final_actionability is not None:
            if type(self.final_actionability) is not ActionabilityAssessment:
                raise _error("adjudication final actionability is invalid")
        if self.profile not in {
            JudgeTask.FINDING_EQUIVALENCE,
            JudgeTask.NOVEL_FACTUALITY,
        } and (
            self.final_severity_assessment is not None
            or self.final_actionability is not None
        ):
            raise _error("adjudication auxiliary labels do not apply to this profile")
        if type(self.adjudicator_provenance) is not HumanReviewerProvenanceV1:
            raise _error("adjudicator provenance is invalid")
        if self.adjudicator_provenance.kind is not ReviewerProvenanceKind.HUMAN:
            raise _error("adjudication requires HUMAN adjudicator provenance")
        if type(self.blind_attestation) is not bool or not self.blind_attestation:
            raise _error("adjudication requires blind attestation")
        identity = self.to_dict()
        identity.pop("adjudication_id")
        if self.adjudication_id != stable_id("adjudication-v1", identity):
            raise _error("adjudication_id is not canonical")
        object.__setattr__(self, "original_label_refs", refs)

    @classmethod
    def create(
        cls,
        *,
        package: CalibrationPackageV1,
        item: CalibrationItemV1,
        adjudication_ref: str,
        original_label_refs: Iterable[str],
        final_primary_label: str,
        final_severity_assessment: Optional[SeverityAssessment],
        final_actionability: Optional[ActionabilityAssessment],
        adjudicator_provenance: HumanReviewerProvenanceV1,
        blind_attestation: bool,
    ) -> "AdjudicationV1":
        if type(package) is not CalibrationPackageV1:
            raise TypeError("package must be a CalibrationPackageV1")
        if type(item) is not CalibrationItemV1:
            raise TypeError("item must be a CalibrationItemV1")
        if item.calibration_item_id not in {
            value.calibration_item_id for value in package.items
        }:
            raise _error("adjudication item is unknown to its package")
        fields = {
            "schema_version": ADJUDICATION_SCHEMA_VERSION,
            "adjudication_ref": adjudication_ref,
            "package_id": package.package_id,
            "package_digest": package.digest(),
            "calibration_item_id": item.calibration_item_id,
            "item_digest": item.digest(),
            "profile": package.profile,
            "original_label_refs": tuple(sorted(tuple(original_label_refs))),
            "final_primary_label": final_primary_label,
            "final_severity_assessment": final_severity_assessment,
            "final_actionability": final_actionability,
            "adjudicator_provenance": adjudicator_provenance,
            "blind_attestation": blind_attestation,
        }
        serialized = {
            **fields,
            "profile": package.profile.value,
            "original_label_refs": list(fields["original_label_refs"]),
            "final_severity_assessment": (
                None
                if final_severity_assessment is None
                else final_severity_assessment.value
            ),
            "final_actionability": (
                None if final_actionability is None else final_actionability.value
            ),
            "adjudicator_provenance": adjudicator_provenance.to_dict(),
        }
        return cls(
            adjudication_id=stable_id("adjudication-v1", serialized),
            **fields,
        )

    @classmethod
    def from_dict(cls, value: Any) -> "AdjudicationV1":
        payload = _exact(
            value,
            (
                "schema_version",
                "adjudication_id",
                "adjudication_ref",
                "package_id",
                "package_digest",
                "calibration_item_id",
                "item_digest",
                "profile",
                "original_label_refs",
                "final_primary_label",
                "final_severity_assessment",
                "final_actionability",
                "adjudicator_provenance",
                "blind_attestation",
            ),
            "AdjudicationV1",
        )
        return cls(
            **{
                **payload,
                "profile": _enum(JudgeTask, payload["profile"], "adjudication profile"),
                "original_label_refs": tuple(
                    _array(payload["original_label_refs"], "original_label_refs", 16)
                ),
                "adjudicator_provenance": HumanReviewerProvenanceV1.from_dict(
                    payload["adjudicator_provenance"]
                ),
                "final_severity_assessment": (
                    None
                    if payload["final_severity_assessment"] is None
                    else _enum(
                        SeverityAssessment,
                        payload["final_severity_assessment"],
                        "final_severity_assessment",
                    )
                ),
                "final_actionability": (
                    None
                    if payload["final_actionability"] is None
                    else _enum(
                        ActionabilityAssessment,
                        payload["final_actionability"],
                        "final_actionability",
                    )
                ),
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "adjudication_id": self.adjudication_id,
            "adjudication_ref": self.adjudication_ref,
            "package_id": self.package_id,
            "package_digest": self.package_digest,
            "calibration_item_id": self.calibration_item_id,
            "item_digest": self.item_digest,
            "profile": self.profile.value,
            "original_label_refs": list(self.original_label_refs),
            "final_primary_label": self.final_primary_label,
            "final_severity_assessment": (
                None
                if self.final_severity_assessment is None
                else self.final_severity_assessment.value
            ),
            "final_actionability": (
                None if self.final_actionability is None else self.final_actionability.value
            ),
            "adjudicator_provenance": self.adjudicator_provenance.to_dict(),
            "blind_attestation": self.blind_attestation,
        }


HumanAdjudicationV1 = AdjudicationV1


def _timestamp(value: Any) -> str:
    text = _text(value, "HumanLabelV1.labeled_at", 64)
    if not text.endswith("Z"):
        raise _error("HumanLabelV1.labeled_at must be canonical UTC RFC3339")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise _error("HumanLabelV1.labeled_at is invalid") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise _error("HumanLabelV1.labeled_at must be UTC")
    return text


@dataclass(frozen=True)
class HumanLabelV1(_JsonModel):
    schema_version: str
    package_id: str
    package_digest: str
    package_source_digest: str
    calibration_item_id: str
    item_digest: str
    source_digest: str
    profile: JudgeTask
    label: str
    severity_assessment: Optional[SeverityAssessment]
    actionability: Optional[ActionabilityAssessment]
    reviewer_provenance: HumanReviewerProvenanceV1
    blind_attestation: bool
    labeled_at: str
    disputed: bool
    adjudication: Optional[AdjudicationV1]

    def __post_init__(self) -> None:
        if self.schema_version != HUMAN_LABEL_SCHEMA_VERSION:
            raise _error("HumanLabelV1 schema_version is unsupported")
        validate_path_segment(self.package_id, "HumanLabelV1.package_id")
        validate_path_segment(
            self.calibration_item_id,
            "HumanLabelV1.calibration_item_id",
        )
        for name in (
            "package_digest",
            "package_source_digest",
            "item_digest",
            "source_digest",
        ):
            _digest(getattr(self, name), f"HumanLabelV1.{name}")
        if type(self.profile) is not JudgeTask:
            raise _error("HumanLabelV1.profile is invalid")
        if self.label not in _allowed_labels(self.profile):
            raise _error("HumanLabelV1 label is out of vocabulary")
        if self.severity_assessment is not None:
            if type(self.severity_assessment) is not SeverityAssessment:
                raise _error("HumanLabelV1.severity_assessment is invalid")
        if self.actionability is not None:
            if type(self.actionability) is not ActionabilityAssessment:
                raise _error("HumanLabelV1.actionability is invalid")
        if self.profile not in {
            JudgeTask.FINDING_EQUIVALENCE,
            JudgeTask.NOVEL_FACTUALITY,
        } and (self.severity_assessment is not None or self.actionability is not None):
            raise _error("HumanLabelV1 auxiliary labels do not apply to this profile")
        if type(self.reviewer_provenance) is not HumanReviewerProvenanceV1:
            raise _error("HumanLabelV1 reviewer provenance is invalid")
        if type(self.blind_attestation) is not bool:
            raise _error("HumanLabelV1.blind_attestation must be bool")
        _timestamp(self.labeled_at)
        if type(self.disputed) is not bool:
            raise _error("HumanLabelV1.disputed must be bool")
        if self.adjudication is not None and type(self.adjudication) is not AdjudicationV1:
            raise _error("HumanLabelV1 adjudication is invalid")
        if self.adjudication is not None and not self.disputed:
            raise _error("HumanLabelV1 adjudication requires a disputed label")
        if self.adjudication is not None:
            adjudication = self.adjudication
            if (
                adjudication.package_id != self.package_id
                or adjudication.package_digest != self.package_digest
                or adjudication.calibration_item_id != self.calibration_item_id
                or adjudication.item_digest != self.item_digest
                or adjudication.profile is not self.profile
                or self.original_label_digest not in adjudication.original_label_refs
            ):
                raise _error("HumanLabelV1 adjudication binding is invalid")
            if (
                adjudication.adjudicator_provenance.digest()
                == self.reviewer_provenance.digest()
                or adjudication.adjudicator_provenance.reviewer_id
                == self.reviewer_provenance.reviewer_id
            ):
                raise _error("adjudicator must be independent from the original reviewer")

    @property
    def eligible(self) -> bool:
        """Eligibility cannot be established without a package-bound trust policy."""

        return False

    @property
    def original_label_digest(self) -> str:
        return canonical_sha256(
            {
                "schema_version": self.schema_version,
                "package_id": self.package_id,
                "package_digest": self.package_digest,
                "package_source_digest": self.package_source_digest,
                "calibration_item_id": self.calibration_item_id,
                "item_digest": self.item_digest,
                "source_digest": self.source_digest,
                "profile": self.profile.value,
                "label": self.label,
                "severity_assessment": (
                    None
                    if self.severity_assessment is None
                    else self.severity_assessment.value
                ),
                "actionability": (
                    None if self.actionability is None else self.actionability.value
                ),
                "reviewer_provenance": self.reviewer_provenance.to_dict(),
                "blind_attestation": self.blind_attestation,
                "labeled_at": self.labeled_at,
                "disputed": self.disputed,
            }
        )

    @property
    def effective_label(self) -> str:
        if self.disputed and self.adjudication is not None:
            return self.adjudication.final_primary_label
        return self.label

    @property
    def effective_severity_assessment(self) -> Optional[str]:
        if self.disputed and self.adjudication is not None:
            value = self.adjudication.final_severity_assessment
        else:
            value = self.severity_assessment
        return None if value is None else value.value

    @property
    def effective_actionability(self) -> Optional[str]:
        if self.disputed and self.adjudication is not None:
            value = self.adjudication.final_actionability
        else:
            value = self.actionability
        return None if value is None else value.value

    def eligible_under(self, policy: CalibrationSelectionPolicyV1) -> bool:
        if type(policy) is not CalibrationSelectionPolicyV1:
            raise TypeError("policy must be CalibrationSelectionPolicyV1")
        if (
            not self.blind_attestation
            or self.reviewer_provenance.kind is not ReviewerProvenanceKind.HUMAN
            or self.reviewer_provenance.digest()
            not in policy.trusted_reviewer_provenance_digests
        ):
            return False
        if not self.disputed:
            return True
        return bool(
            self.adjudication is not None
            and self.adjudication.blind_attestation
            and self.adjudication.adjudicator_provenance.kind
            is ReviewerProvenanceKind.HUMAN
            and self.adjudication.adjudicator_provenance.digest()
            in policy.trusted_adjudicator_provenance_digests
            and self.adjudication.adjudicator_provenance.digest()
            != self.reviewer_provenance.digest()
            and self.adjudication.adjudicator_provenance.reviewer_id
            != self.reviewer_provenance.reviewer_id
        )

    @classmethod
    def create(
        cls,
        *,
        package: CalibrationPackageV1,
        item: CalibrationItemV1,
        label: str,
        severity_assessment: Optional[SeverityAssessment],
        actionability: Optional[ActionabilityAssessment],
        reviewer_provenance: HumanReviewerProvenanceV1,
        blind_attestation: bool,
        labeled_at: str,
        disputed: bool,
        adjudication: Optional[AdjudicationV1],
    ) -> "HumanLabelV1":
        if type(package) is not CalibrationPackageV1:
            raise TypeError("package must be a CalibrationPackageV1")
        if type(item) is not CalibrationItemV1:
            raise TypeError("item must be a CalibrationItemV1")
        if item.calibration_item_id not in {
            value.calibration_item_id for value in package.items
        }:
            raise _error("Human label item is unknown to its package")
        return cls(
            schema_version=HUMAN_LABEL_SCHEMA_VERSION,
            package_id=package.package_id,
            package_digest=package.digest(),
            package_source_digest=package.source_digest,
            calibration_item_id=item.calibration_item_id,
            item_digest=item.digest(),
            source_digest=item.source_digest,
            profile=package.profile,
            label=label,
            severity_assessment=severity_assessment,
            actionability=actionability,
            reviewer_provenance=reviewer_provenance,
            blind_attestation=blind_attestation,
            labeled_at=labeled_at,
            disputed=disputed,
            adjudication=adjudication,
        )

    @classmethod
    def from_dict(cls, value: Any) -> "HumanLabelV1":
        payload = _exact(
            value,
            (
                "schema_version",
                "package_id",
                "package_digest",
                "package_source_digest",
                "calibration_item_id",
                "item_digest",
                "source_digest",
                "profile",
                "label",
                "severity_assessment",
                "actionability",
                "reviewer_provenance",
                "blind_attestation",
                "labeled_at",
                "disputed",
                "adjudication",
            ),
            "HumanLabelV1",
        )
        return cls(
            **{
                **payload,
                "profile": _enum(JudgeTask, payload["profile"], "HumanLabelV1.profile"),
                "reviewer_provenance": HumanReviewerProvenanceV1.from_dict(
                    payload["reviewer_provenance"]
                ),
                "adjudication": (
                    None
                    if payload["adjudication"] is None
                    else AdjudicationV1.from_dict(payload["adjudication"])
                ),
                "severity_assessment": (
                    None
                    if payload["severity_assessment"] is None
                    else _enum(
                        SeverityAssessment,
                        payload["severity_assessment"],
                        "HumanLabelV1.severity_assessment",
                    )
                ),
                "actionability": (
                    None
                    if payload["actionability"] is None
                    else _enum(
                        ActionabilityAssessment,
                        payload["actionability"],
                        "HumanLabelV1.actionability",
                    )
                ),
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "package_id": self.package_id,
            "package_digest": self.package_digest,
            "package_source_digest": self.package_source_digest,
            "calibration_item_id": self.calibration_item_id,
            "item_digest": self.item_digest,
            "source_digest": self.source_digest,
            "profile": self.profile.value,
            "label": self.label,
            "severity_assessment": (
                None if self.severity_assessment is None else self.severity_assessment.value
            ),
            "actionability": (
                None if self.actionability is None else self.actionability.value
            ),
            "reviewer_provenance": self.reviewer_provenance.to_dict(),
            "blind_attestation": self.blind_attestation,
            "labeled_at": self.labeled_at,
            "disputed": self.disputed,
            "adjudication": (
                None if self.adjudication is None else self.adjudication.to_dict()
            ),
        }


@dataclass(frozen=True)
class HumanLabelSetV1(_JsonModel):
    schema_version: str
    label_set_id: str
    package_id: str
    package_digest: str
    package_source_digest: str
    profile: JudgeTask
    labels: Tuple[HumanLabelV1, ...]

    def __post_init__(self) -> None:
        if self.schema_version != HUMAN_LABEL_SET_SCHEMA_VERSION:
            raise _error("HumanLabelSetV1 schema_version is unsupported")
        validate_path_segment(self.label_set_id, "HumanLabelSetV1.label_set_id")
        validate_path_segment(self.package_id, "HumanLabelSetV1.package_id")
        _digest(self.package_digest, "HumanLabelSetV1.package_digest")
        _digest(
            self.package_source_digest,
            "HumanLabelSetV1.package_source_digest",
        )
        if type(self.profile) is not JudgeTask:
            raise _error("HumanLabelSetV1.profile is invalid")
        labels = tuple(self.labels)
        if (
            len(labels) > MAX_CALIBRATION_ITEMS
            or any(type(item) is not HumanLabelV1 for item in labels)
            or tuple(item.calibration_item_id for item in labels)
            != tuple(sorted(item.calibration_item_id for item in labels))
            or len({item.calibration_item_id for item in labels}) != len(labels)
        ):
            raise _error("HumanLabelSetV1 labels are duplicate or non-canonical")
        if any(
            item.package_id != self.package_id
            or item.package_digest != self.package_digest
            or item.package_source_digest != self.package_source_digest
            or item.profile is not self.profile
            for item in labels
        ):
            raise _error("HumanLabelSetV1 nested package/profile binding is invalid")
        expected_id = stable_id(
            "human-label-set-v1",
            self.package_id,
            self.package_digest,
            self.package_source_digest,
            [item.to_dict() for item in labels],
        )
        if self.label_set_id != expected_id:
            raise _error("HumanLabelSetV1 label_set_id is not canonical")
        object.__setattr__(self, "labels", labels)

    @classmethod
    def create(
        cls,
        *,
        package: CalibrationPackageV1,
        labels: Iterable[HumanLabelV1],
    ) -> "HumanLabelSetV1":
        if type(package) is not CalibrationPackageV1:
            raise TypeError("package must be a CalibrationPackageV1")
        values = tuple(sorted(tuple(labels), key=lambda item: item.calibration_item_id))
        item_map = {item.calibration_item_id: item for item in package.items}
        if len(item_map) != len(package.items):
            raise _error("Calibration package contains duplicate items")
        if len({item.calibration_item_id for item in values}) != len(values):
            raise _error("HumanLabelSetV1 contains duplicate labels")
        for label in values:
            if type(label) is not HumanLabelV1:
                raise TypeError("labels must contain HumanLabelV1 values")
            item = item_map.get(label.calibration_item_id)
            if item is None:
                raise _error("HumanLabelSetV1 contains an unknown item")
            if (
                label.package_id != package.package_id
                or label.package_digest != package.digest()
                or label.package_source_digest != package.source_digest
                or label.item_digest != item.digest()
                or label.source_digest != item.source_digest
                or label.profile is not package.profile
                or label.label not in item.allowed_labels
            ):
                raise _error("HumanLabelSetV1 label package/item/source digest binding is invalid")
        identity = [item.to_dict() for item in values]
        return cls(
            schema_version=HUMAN_LABEL_SET_SCHEMA_VERSION,
            label_set_id=stable_id(
                "human-label-set-v1",
                package.package_id,
                package.digest(),
                package.source_digest,
                identity,
            ),
            package_id=package.package_id,
            package_digest=package.digest(),
            package_source_digest=package.source_digest,
            profile=package.profile,
            labels=values,
        )

    @classmethod
    def from_dict(
        cls,
        value: Any,
        *,
        package: CalibrationPackageV1,
    ) -> "HumanLabelSetV1":
        payload = _exact(
            value,
            (
                "schema_version",
                "label_set_id",
                "package_id",
                "package_digest",
                "package_source_digest",
                "profile",
                "labels",
            ),
            "HumanLabelSetV1",
        )
        labels = _array(payload["labels"], "HumanLabelSetV1.labels")
        result = cls(
            schema_version=payload["schema_version"],
            label_set_id=payload["label_set_id"],
            package_id=payload["package_id"],
            package_digest=payload["package_digest"],
            package_source_digest=payload["package_source_digest"],
            profile=_enum(JudgeTask, payload["profile"], "HumanLabelSetV1.profile"),
            labels=tuple(HumanLabelV1.from_dict(item) for item in labels),
        )
        rebound = cls.create(package=package, labels=result.labels)
        if rebound != result:
            raise _error("HumanLabelSetV1 differs from exact package replay")
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label_set_id": self.label_set_id,
            "package_id": self.package_id,
            "package_digest": self.package_digest,
            "package_source_digest": self.package_source_digest,
            "profile": self.profile.value,
            "labels": [item.to_dict() for item in self.labels],
        }


@dataclass(frozen=True)
class ConfusionMatrixCellV1(_JsonModel):
    human_label: str
    recorded_label: str
    count: int

    def __post_init__(self) -> None:
        _text(self.human_label, "ConfusionMatrixCellV1.human_label")
        _text(self.recorded_label, "ConfusionMatrixCellV1.recorded_label")
        _integer(self.count, "ConfusionMatrixCellV1.count")

    @classmethod
    def from_dict(cls, value: Any) -> "ConfusionMatrixCellV1":
        return cls(**_exact(value, ("human_label", "recorded_label", "count"), "ConfusionMatrixCellV1"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "human_label": self.human_label,
            "recorded_label": self.recorded_label,
            "count": self.count,
        }


@dataclass(frozen=True)
class ClassCalibrationV1(_JsonModel):
    label: str
    human_count: int
    recorded_count: int
    true_positive_count: int
    precision_ppm: Optional[int]
    precision_null_reason: Optional[ClassMetricNullReason]
    recall_ppm: Optional[int]
    recall_null_reason: Optional[ClassMetricNullReason]

    def __post_init__(self) -> None:
        _text(self.label, "ClassCalibrationV1.label")
        human = _integer(self.human_count, "ClassCalibrationV1.human_count")
        recorded = _integer(self.recorded_count, "ClassCalibrationV1.recorded_count")
        true_positive = _integer(
            self.true_positive_count,
            "ClassCalibrationV1.true_positive_count",
        )
        if true_positive > min(human, recorded):
            raise _error("ClassCalibrationV1 true positives exceed support")
        expected_precision = _ratio_ppm(true_positive, recorded)
        expected_recall = _ratio_ppm(true_positive, human)
        if self.precision_ppm != expected_precision or self.recall_ppm != expected_recall:
            raise _error("ClassCalibrationV1 precision/recall are not canonical")
        expected_precision_null = (
            ClassMetricNullReason.NO_RECORDED_PREDICTIONS if recorded == 0 else None
        )
        expected_recall_null = (
            ClassMetricNullReason.NO_HUMAN_LABELS if human == 0 else None
        )
        if (
            self.precision_null_reason is not expected_precision_null
            or self.recall_null_reason is not expected_recall_null
        ):
            raise _error("ClassCalibrationV1 null reasons are not canonical")

    @classmethod
    def from_dict(cls, value: Any) -> "ClassCalibrationV1":
        payload = _exact(
            value,
            (
                "label",
                "human_count",
                "recorded_count",
                "true_positive_count",
                "precision_ppm",
                "precision_null_reason",
                "recall_ppm",
                "recall_null_reason",
            ),
            "ClassCalibrationV1",
        )
        return cls(
            **{
                **payload,
                "precision_null_reason": (
                    None
                    if payload["precision_null_reason"] is None
                    else _enum(
                        ClassMetricNullReason,
                        payload["precision_null_reason"],
                        "precision_null_reason",
                    )
                ),
                "recall_null_reason": (
                    None
                    if payload["recall_null_reason"] is None
                    else _enum(
                        ClassMetricNullReason,
                        payload["recall_null_reason"],
                        "recall_null_reason",
                    )
                ),
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "human_count": self.human_count,
            "recorded_count": self.recorded_count,
            "true_positive_count": self.true_positive_count,
            "precision_ppm": self.precision_ppm,
            "precision_null_reason": (
                None if self.precision_null_reason is None else self.precision_null_reason.value
            ),
            "recall_ppm": self.recall_ppm,
            "recall_null_reason": (
                None if self.recall_null_reason is None else self.recall_null_reason.value
            ),
        }


def _kappa_from_matrix(
    labels: Sequence[str],
    outcomes: Sequence[str],
    counts: Mapping[tuple[str, str], int],
) -> tuple[Optional[int], Optional[KappaNullReason]]:
    total = sum(counts.values())
    if total == 0:
        return None, KappaNullReason.NO_ELIGIBLE_LABELS
    agreement = sum(counts.get((label, label), 0) for label in labels)
    expected_numerator = 0
    for label in labels:
        human_count = sum(counts.get((label, outcome), 0) for outcome in outcomes)
        recorded_count = sum(counts.get((human, label), 0) for human in labels)
        expected_numerator += human_count * recorded_count
    denominator = total * total - expected_numerator
    if denominator == 0:
        return None, KappaNullReason.ZERO_EXPECTED_DISAGREEMENT
    return (
        _signed_ratio_ppm(agreement * total - expected_numerator, denominator),
        None,
    )


@dataclass(frozen=True)
class LabelCountV1(_JsonModel):
    label: str
    count: int

    def __post_init__(self) -> None:
        _text(self.label, "LabelCountV1.label")
        _integer(self.count, "LabelCountV1.count")

    @classmethod
    def from_dict(cls, value: Any) -> "LabelCountV1":
        return cls(**_exact(value, ("label", "count"), "LabelCountV1"))

    def to_dict(self) -> Dict[str, Any]:
        return {"label": self.label, "count": self.count}


@dataclass(frozen=True)
class AuxiliaryCalibrationV1(_JsonModel):
    schema_version: str
    auxiliary_calibration_id: str
    dimension: CalibrationAuxiliaryDimension
    applicability: AuxiliaryCalibrationApplicability
    allowed_labels: Tuple[str, ...]
    recorded_outcomes: Tuple[str, ...]
    applicable_item_count: int
    labeled_item_count: int
    eligible_labeled_item_count: int
    pending_item_count: int
    labeled_item_coverage_ppm: Optional[int]
    eligible_item_coverage_ppm: Optional[int]
    applicable_occurrence_count: int
    labeled_occurrence_count: int
    eligible_labeled_occurrence_count: int
    unique_human_label_counts: Tuple[LabelCountV1, ...]
    confusion_matrix: Tuple[ConfusionMatrixCellV1, ...]
    exact_agreement_numerator: int
    exact_agreement_denominator: int
    exact_agreement_ppm: Optional[int]
    class_metrics: Tuple[ClassCalibrationV1, ...]
    cohen_kappa_ppm: Optional[int]
    cohen_kappa_null_reason: Optional[KappaNullReason]
    disagreement_count: int
    disagreement_occurrence_refs: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.schema_version != AUXILIARY_CALIBRATION_SCHEMA_VERSION:
            raise _error("AuxiliaryCalibrationV1 schema_version is unsupported")
        validate_path_segment(
            self.auxiliary_calibration_id,
            "auxiliary_calibration_id",
        )
        if type(self.dimension) is not CalibrationAuxiliaryDimension:
            raise _error("auxiliary calibration dimension is invalid")
        if type(self.applicability) is not AuxiliaryCalibrationApplicability:
            raise _error("auxiliary calibration applicability is invalid")
        canonical_labels = (
            tuple(item.value for item in SeverityAssessment)
            if self.dimension is CalibrationAuxiliaryDimension.SEVERITY_ASSESSMENT
            else tuple(item.value for item in ActionabilityAssessment)
        )
        labels = tuple(self.allowed_labels)
        outcomes = tuple(self.recorded_outcomes)
        if labels != canonical_labels or outcomes != canonical_labels:
            raise _error("auxiliary calibration labels are not canonical")
        applicable_items = _integer(
            self.applicable_item_count,
            "applicable_item_count",
        )
        labeled_items = _integer(self.labeled_item_count, "labeled_item_count")
        eligible_items = _integer(
            self.eligible_labeled_item_count,
            "eligible_labeled_item_count",
        )
        pending_items = _integer(self.pending_item_count, "pending_item_count")
        if (
            labeled_items + pending_items != applicable_items
            or eligible_items > labeled_items
        ):
            raise _error("auxiliary item coverage counts are inconsistent")
        if (
            self.labeled_item_coverage_ppm
            != _ratio_ppm(labeled_items, applicable_items)
            or self.eligible_item_coverage_ppm
            != _ratio_ppm(eligible_items, applicable_items)
        ):
            raise _error("auxiliary item coverage ratios are not canonical")
        applicable_occurrences = _integer(
            self.applicable_occurrence_count,
            "applicable_occurrence_count",
        )
        labeled_occurrences = _integer(
            self.labeled_occurrence_count,
            "labeled_occurrence_count",
        )
        eligible_occurrences = _integer(
            self.eligible_labeled_occurrence_count,
            "eligible_labeled_occurrence_count",
        )
        if (
            applicable_occurrences < applicable_items
            or labeled_occurrences > applicable_occurrences
            or eligible_occurrences > labeled_occurrences
            or eligible_occurrences < eligible_items
        ):
            raise _error("auxiliary occurrence coverage counts are inconsistent")
        if (
            self.applicability is AuxiliaryCalibrationApplicability.NOT_APPLICABLE
            and (applicable_items != 0 or applicable_occurrences != 0)
        ) or (
            self.applicability is AuxiliaryCalibrationApplicability.APPLICABLE
            and (applicable_items == 0 or applicable_occurrences == 0)
        ):
            raise _error("auxiliary applicability differs from authoritative coverage")
        unique_counts = tuple(self.unique_human_label_counts)
        if (
            len(unique_counts) != len(labels)
            or any(type(item) is not LabelCountV1 for item in unique_counts)
            or tuple(item.label for item in unique_counts) != labels
            or sum(item.count for item in unique_counts) != eligible_items
        ):
            raise _error("auxiliary unique class support is not canonical")
        matrix = tuple(self.confusion_matrix)
        expected_pairs = tuple((human, recorded) for human in labels for recorded in outcomes)
        if (
            len(matrix) != len(expected_pairs)
            or any(type(item) is not ConfusionMatrixCellV1 for item in matrix)
            or tuple((item.human_label, item.recorded_label) for item in matrix)
            != expected_pairs
        ):
            raise _error("auxiliary confusion matrix is not canonical")
        counts = {(item.human_label, item.recorded_label): item.count for item in matrix}
        if sum(counts.values()) != eligible_occurrences:
            raise _error("auxiliary confusion matrix coverage is inconsistent")
        agreement = sum(counts[(label, label)] for label in labels)
        if (
            self.exact_agreement_numerator != agreement
            or self.exact_agreement_denominator != eligible_occurrences
            or self.exact_agreement_ppm != _ratio_ppm(agreement, eligible_occurrences)
        ):
            raise _error("auxiliary exact agreement is not canonical")
        metrics = tuple(self.class_metrics)
        if (
            len(metrics) != len(labels)
            or any(type(item) is not ClassCalibrationV1 for item in metrics)
            or tuple(item.label for item in metrics) != labels
        ):
            raise _error("auxiliary class metrics are not canonical")
        for metric in metrics:
            human_count = sum(counts[(metric.label, outcome)] for outcome in outcomes)
            recorded_count = sum(counts[(human, metric.label)] for human in labels)
            if (
                metric.human_count != human_count
                or metric.recorded_count != recorded_count
                or metric.true_positive_count != counts[(metric.label, metric.label)]
            ):
                raise _error("auxiliary class metrics differ from matrix")
        kappa, kappa_null = _kappa_from_matrix(labels, outcomes, counts)
        if self.cohen_kappa_ppm != kappa or self.cohen_kappa_null_reason is not kappa_null:
            raise _error("auxiliary Cohen kappa is not canonical")
        refs = tuple(self.disagreement_occurrence_refs)
        disagreement_count = _integer(self.disagreement_count, "disagreement_count")
        if (
            refs != tuple(sorted(set(refs)))
            or len(refs) != disagreement_count
            or disagreement_count != eligible_occurrences - agreement
        ):
            raise _error("auxiliary disagreement refs are invalid")
        for ref in refs:
            validate_path_segment(ref, "auxiliary disagreement occurrence ref")
        identity = self.to_dict()
        identity.pop("auxiliary_calibration_id")
        if self.auxiliary_calibration_id != stable_id(
            "auxiliary-calibration-v1",
            identity,
        ):
            raise _error("auxiliary_calibration_id is not canonical")
        object.__setattr__(self, "allowed_labels", labels)
        object.__setattr__(self, "recorded_outcomes", outcomes)
        object.__setattr__(self, "unique_human_label_counts", unique_counts)
        object.__setattr__(self, "confusion_matrix", matrix)
        object.__setattr__(self, "class_metrics", metrics)
        object.__setattr__(self, "disagreement_occurrence_refs", refs)

    @classmethod
    def create(cls, **fields: Any) -> "AuxiliaryCalibrationV1":
        identity = {"schema_version": AUXILIARY_CALIBRATION_SCHEMA_VERSION, **fields}
        serialized = {
            key: (
                value.value
                if isinstance(value, Enum)
                else [item.to_dict() if isinstance(item, _JsonModel) else item for item in value]
                if type(value) in (tuple, list)
                else value
            )
            for key, value in identity.items()
        }
        return cls(
            auxiliary_calibration_id=stable_id(
                "auxiliary-calibration-v1",
                serialized,
            ),
            **identity,
        )

    @classmethod
    def from_dict(cls, value: Any) -> "AuxiliaryCalibrationV1":
        fields = (
            "schema_version", "auxiliary_calibration_id", "dimension", "applicability",
            "allowed_labels", "recorded_outcomes", "applicable_item_count",
            "labeled_item_count", "eligible_labeled_item_count", "pending_item_count",
            "labeled_item_coverage_ppm", "eligible_item_coverage_ppm",
            "applicable_occurrence_count", "labeled_occurrence_count",
            "eligible_labeled_occurrence_count", "unique_human_label_counts",
            "confusion_matrix", "exact_agreement_numerator",
            "exact_agreement_denominator", "exact_agreement_ppm", "class_metrics",
            "cohen_kappa_ppm", "cohen_kappa_null_reason", "disagreement_count",
            "disagreement_occurrence_refs",
        )
        payload = _exact(value, fields, "AuxiliaryCalibrationV1")
        return cls(
            **{
                **payload,
                "dimension": _enum(CalibrationAuxiliaryDimension, payload["dimension"], "dimension"),
                "applicability": _enum(
                    AuxiliaryCalibrationApplicability,
                    payload["applicability"],
                    "applicability",
                ),
                "allowed_labels": tuple(_array(payload["allowed_labels"], "allowed_labels", 16)),
                "recorded_outcomes": tuple(_array(payload["recorded_outcomes"], "recorded_outcomes", 16)),
                "unique_human_label_counts": tuple(
                    LabelCountV1.from_dict(item)
                    for item in _array(payload["unique_human_label_counts"], "unique_human_label_counts", 16)
                ),
                "confusion_matrix": tuple(
                    ConfusionMatrixCellV1.from_dict(item)
                    for item in _array(payload["confusion_matrix"], "confusion_matrix", 256)
                ),
                "class_metrics": tuple(
                    ClassCalibrationV1.from_dict(item)
                    for item in _array(payload["class_metrics"], "class_metrics", 16)
                ),
                "cohen_kappa_null_reason": (
                    None
                    if payload["cohen_kappa_null_reason"] is None
                    else _enum(KappaNullReason, payload["cohen_kappa_null_reason"], "cohen_kappa_null_reason")
                ),
                "disagreement_occurrence_refs": tuple(
                    _array(payload["disagreement_occurrence_refs"], "disagreement_occurrence_refs")
                ),
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "auxiliary_calibration_id": self.auxiliary_calibration_id,
            "dimension": self.dimension.value,
            "applicability": self.applicability.value,
            "allowed_labels": list(self.allowed_labels),
            "recorded_outcomes": list(self.recorded_outcomes),
            "applicable_item_count": self.applicable_item_count,
            "labeled_item_count": self.labeled_item_count,
            "eligible_labeled_item_count": self.eligible_labeled_item_count,
            "pending_item_count": self.pending_item_count,
            "labeled_item_coverage_ppm": self.labeled_item_coverage_ppm,
            "eligible_item_coverage_ppm": self.eligible_item_coverage_ppm,
            "applicable_occurrence_count": self.applicable_occurrence_count,
            "labeled_occurrence_count": self.labeled_occurrence_count,
            "eligible_labeled_occurrence_count": self.eligible_labeled_occurrence_count,
            "unique_human_label_counts": [item.to_dict() for item in self.unique_human_label_counts],
            "confusion_matrix": [item.to_dict() for item in self.confusion_matrix],
            "exact_agreement_numerator": self.exact_agreement_numerator,
            "exact_agreement_denominator": self.exact_agreement_denominator,
            "exact_agreement_ppm": self.exact_agreement_ppm,
            "class_metrics": [item.to_dict() for item in self.class_metrics],
            "cohen_kappa_ppm": self.cohen_kappa_ppm,
            "cohen_kappa_null_reason": (
                None if self.cohen_kappa_null_reason is None else self.cohen_kappa_null_reason.value
            ),
            "disagreement_count": self.disagreement_count,
            "disagreement_occurrence_refs": list(self.disagreement_occurrence_refs),
        }


@dataclass(frozen=True)
class ProfileCalibrationV1(_JsonModel):
    schema_version: str
    profile_calibration_id: str
    profile: JudgeTask
    package_id: str
    package_digest: str
    policy_digest: str
    label_set_id: str
    label_set_digest: str
    allowed_labels: Tuple[str, ...]
    recorded_outcomes: Tuple[str, ...]
    selected_count: int
    labeled_count: int
    eligible_labeled_count: int
    pending_label_count: int
    unattested_label_count: int
    unadjudicated_dispute_count: int
    labeled_coverage_ppm: Optional[int]
    eligible_coverage_ppm: Optional[int]
    selected_occurrence_count: int
    labeled_occurrence_count: int
    eligible_labeled_occurrence_count: int
    judge_graded_count: int
    semantic_unknown_count: int
    judge_failure_count: int
    judge_ungraded_count: int
    unique_primary_label_counts: Tuple[LabelCountV1, ...]
    confusion_matrix: Tuple[ConfusionMatrixCellV1, ...]
    exact_agreement_numerator: int
    exact_agreement_denominator: int
    exact_agreement_ppm: Optional[int]
    class_metrics: Tuple[ClassCalibrationV1, ...]
    cohen_kappa_ppm: Optional[int]
    cohen_kappa_null_reason: Optional[KappaNullReason]
    disagreement_count: int
    disagreement_item_refs: Tuple[str, ...]
    auxiliary_calibrations: Tuple[AuxiliaryCalibrationV1, ...]
    status: CalibrationStatus

    def __post_init__(self) -> None:
        if self.schema_version != PROFILE_CALIBRATION_SCHEMA_VERSION:
            raise _error("ProfileCalibrationV1 schema_version is unsupported")
        validate_path_segment(self.profile_calibration_id, "profile_calibration_id")
        if type(self.profile) is not JudgeTask:
            raise _error("ProfileCalibrationV1.profile is invalid")
        validate_path_segment(self.package_id, "ProfileCalibrationV1.package_id")
        validate_path_segment(self.label_set_id, "ProfileCalibrationV1.label_set_id")
        for name in ("package_digest", "policy_digest", "label_set_digest"):
            _digest(getattr(self, name), f"ProfileCalibrationV1.{name}")
        labels = tuple(self.allowed_labels)
        outcomes = tuple(self.recorded_outcomes)
        if labels != _allowed_labels(self.profile):
            raise _error("ProfileCalibrationV1 allowed_labels are not canonical")
        if outcomes != labels + (JUDGE_FAILED_OUTCOME, JUDGE_UNGRADED_OUTCOME):
            raise _error("ProfileCalibrationV1 recorded_outcomes are not canonical")
        selected = _integer(self.selected_count, "selected_count")
        labeled = _integer(self.labeled_count, "labeled_count")
        eligible = _integer(self.eligible_labeled_count, "eligible_labeled_count")
        pending = _integer(self.pending_label_count, "pending_label_count")
        unattested = _integer(self.unattested_label_count, "unattested_label_count")
        disputes = _integer(
            self.unadjudicated_dispute_count,
            "unadjudicated_dispute_count",
        )
        if labeled + pending != selected or eligible > labeled:
            raise _error("ProfileCalibrationV1 label coverage counts are inconsistent")
        if unattested > labeled or disputes > labeled:
            raise _error("ProfileCalibrationV1 provenance counts are inconsistent")
        expected_labeled_coverage = _ratio_ppm(labeled, selected)
        expected_eligible_coverage = _ratio_ppm(eligible, selected)
        if (
            self.labeled_coverage_ppm != expected_labeled_coverage
            or self.eligible_coverage_ppm != expected_eligible_coverage
        ):
            raise _error("ProfileCalibrationV1 coverage ratios are not canonical")
        selected_occurrences = _integer(
            self.selected_occurrence_count,
            "selected_occurrence_count",
        )
        labeled_occurrences = _integer(
            self.labeled_occurrence_count,
            "labeled_occurrence_count",
        )
        eligible_occurrences = _integer(
            self.eligible_labeled_occurrence_count,
            "eligible_labeled_occurrence_count",
        )
        if (
            selected_occurrences < selected
            or labeled_occurrences > selected_occurrences
            or eligible_occurrences > labeled_occurrences
            or eligible_occurrences < eligible
        ):
            raise _error("ProfileCalibrationV1 occurrence coverage is inconsistent")
        graded = _integer(self.judge_graded_count, "judge_graded_count")
        unknown = _integer(self.semantic_unknown_count, "semantic_unknown_count")
        failed = _integer(self.judge_failure_count, "judge_failure_count")
        ungraded = _integer(self.judge_ungraded_count, "judge_ungraded_count")
        if graded + failed + ungraded != selected_occurrences or unknown > graded:
            raise _error("ProfileCalibrationV1 Judge coverage is inconsistent")
        unique_counts = tuple(self.unique_primary_label_counts)
        if (
            len(unique_counts) != len(labels)
            or any(type(item) is not LabelCountV1 for item in unique_counts)
            or tuple(item.label for item in unique_counts) != labels
            or sum(item.count for item in unique_counts) != eligible
        ):
            raise _error("ProfileCalibrationV1 unique class support is not canonical")
        matrix = tuple(self.confusion_matrix)
        expected_pairs = tuple((human, recorded) for human in labels for recorded in outcomes)
        if (
            len(matrix) != len(expected_pairs)
            or any(type(item) is not ConfusionMatrixCellV1 for item in matrix)
            or tuple((item.human_label, item.recorded_label) for item in matrix)
            != expected_pairs
        ):
            raise _error("ProfileCalibrationV1 confusion matrix is not canonical")
        counts = {(item.human_label, item.recorded_label): item.count for item in matrix}
        if sum(counts.values()) != eligible_occurrences:
            raise _error("ProfileCalibrationV1 confusion matrix coverage is inconsistent")
        agreement = sum(counts[(label, label)] for label in labels)
        if (
            self.exact_agreement_numerator != agreement
            or self.exact_agreement_denominator != eligible_occurrences
            or self.exact_agreement_ppm != _ratio_ppm(agreement, eligible_occurrences)
        ):
            raise _error("ProfileCalibrationV1 exact agreement is not canonical")
        metrics = tuple(self.class_metrics)
        if (
            len(metrics) != len(labels)
            or any(type(item) is not ClassCalibrationV1 for item in metrics)
            or tuple(item.label for item in metrics) != labels
        ):
            raise _error("ProfileCalibrationV1 class metrics are not canonical")
        for metric in metrics:
            human_count = sum(counts[(metric.label, outcome)] for outcome in outcomes)
            recorded_count = sum(counts[(human, metric.label)] for human in labels)
            if (
                metric.human_count != human_count
                or metric.recorded_count != recorded_count
                or metric.true_positive_count != counts[(metric.label, metric.label)]
            ):
                raise _error("ProfileCalibrationV1 class metrics differ from matrix")
        kappa, kappa_null = _kappa_from_matrix(labels, outcomes, counts)
        if self.cohen_kappa_ppm != kappa or self.cohen_kappa_null_reason is not kappa_null:
            raise _error("ProfileCalibrationV1 Cohen kappa is not canonical")
        disagreement_count = _integer(self.disagreement_count, "disagreement_count")
        refs = tuple(self.disagreement_item_refs)
        if (
            refs != tuple(sorted(set(refs)))
            or len(refs) != disagreement_count
            or disagreement_count != eligible_occurrences - agreement
        ):
            raise _error("ProfileCalibrationV1 disagreement metadata is invalid")
        for ref in refs:
            validate_path_segment(ref, "disagreement item ref")
        auxiliary = tuple(self.auxiliary_calibrations)
        if (
            len(auxiliary) != len(CalibrationAuxiliaryDimension)
            or any(type(item) is not AuxiliaryCalibrationV1 for item in auxiliary)
            or tuple(item.dimension for item in auxiliary)
            != tuple(CalibrationAuxiliaryDimension)
        ):
            raise _error("ProfileCalibrationV1 auxiliary calibrations are not canonical")
        if type(self.status) is not CalibrationStatus:
            raise _error("ProfileCalibrationV1.status is invalid")
        identity = self.to_dict()
        identity.pop("profile_calibration_id")
        expected_id = stable_id("profile-calibration-v1", identity)
        if self.profile_calibration_id != expected_id:
            raise _error("ProfileCalibrationV1 profile_calibration_id is not canonical")
        object.__setattr__(self, "allowed_labels", labels)
        object.__setattr__(self, "recorded_outcomes", outcomes)
        object.__setattr__(self, "unique_primary_label_counts", unique_counts)
        object.__setattr__(self, "confusion_matrix", matrix)
        object.__setattr__(self, "class_metrics", metrics)
        object.__setattr__(self, "disagreement_item_refs", refs)
        object.__setattr__(self, "auxiliary_calibrations", auxiliary)

    @classmethod
    def create(cls, **fields: Any) -> "ProfileCalibrationV1":
        identity = {
            "schema_version": PROFILE_CALIBRATION_SCHEMA_VERSION,
            **fields,
        }
        serialized = {
            key: (
                value.value
                if isinstance(value, Enum)
                else [item.to_dict() if isinstance(item, _JsonModel) else item for item in value]
                if type(value) in (tuple, list)
                else value
            )
            for key, value in identity.items()
        }
        return cls(
            profile_calibration_id=stable_id("profile-calibration-v1", serialized),
            **identity,
        )

    @classmethod
    def from_dict(cls, value: Any) -> "ProfileCalibrationV1":
        fields = (
            "schema_version", "profile_calibration_id", "profile", "package_id",
            "package_digest", "policy_digest", "label_set_id", "label_set_digest",
            "allowed_labels", "recorded_outcomes", "selected_count", "labeled_count",
            "eligible_labeled_count", "pending_label_count", "unattested_label_count",
            "unadjudicated_dispute_count", "labeled_coverage_ppm",
            "eligible_coverage_ppm", "selected_occurrence_count",
            "labeled_occurrence_count", "eligible_labeled_occurrence_count",
            "judge_graded_count", "semantic_unknown_count",
            "judge_failure_count", "judge_ungraded_count", "confusion_matrix",
            "unique_primary_label_counts",
            "exact_agreement_numerator", "exact_agreement_denominator",
            "exact_agreement_ppm", "class_metrics", "cohen_kappa_ppm",
            "cohen_kappa_null_reason", "disagreement_count",
            "disagreement_item_refs", "auxiliary_calibrations", "status",
        )
        payload = _exact(value, fields, "ProfileCalibrationV1")
        matrix = _array(payload["confusion_matrix"], "confusion_matrix", 512)
        metrics = _array(payload["class_metrics"], "class_metrics", 32)
        return cls(
            **{
                **payload,
                "profile": _enum(JudgeTask, payload["profile"], "profile"),
                "allowed_labels": tuple(_array(payload["allowed_labels"], "allowed_labels", 16)),
                "recorded_outcomes": tuple(_array(payload["recorded_outcomes"], "recorded_outcomes", 18)),
                "unique_primary_label_counts": tuple(
                    LabelCountV1.from_dict(item)
                    for item in _array(payload["unique_primary_label_counts"], "unique_primary_label_counts", 16)
                ),
                "confusion_matrix": tuple(ConfusionMatrixCellV1.from_dict(item) for item in matrix),
                "class_metrics": tuple(ClassCalibrationV1.from_dict(item) for item in metrics),
                "cohen_kappa_null_reason": (
                    None if payload["cohen_kappa_null_reason"] is None else _enum(
                        KappaNullReason, payload["cohen_kappa_null_reason"], "cohen_kappa_null_reason"
                    )
                ),
                "disagreement_item_refs": tuple(_array(payload["disagreement_item_refs"], "disagreement_item_refs")),
                "auxiliary_calibrations": tuple(
                    AuxiliaryCalibrationV1.from_dict(item)
                    for item in _array(payload["auxiliary_calibrations"], "auxiliary_calibrations", 2)
                ),
                "status": _enum(CalibrationStatus, payload["status"], "status"),
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_calibration_id": self.profile_calibration_id,
            "profile": self.profile.value,
            "package_id": self.package_id,
            "package_digest": self.package_digest,
            "policy_digest": self.policy_digest,
            "label_set_id": self.label_set_id,
            "label_set_digest": self.label_set_digest,
            "allowed_labels": list(self.allowed_labels),
            "recorded_outcomes": list(self.recorded_outcomes),
            "selected_count": self.selected_count,
            "labeled_count": self.labeled_count,
            "eligible_labeled_count": self.eligible_labeled_count,
            "pending_label_count": self.pending_label_count,
            "unattested_label_count": self.unattested_label_count,
            "unadjudicated_dispute_count": self.unadjudicated_dispute_count,
            "labeled_coverage_ppm": self.labeled_coverage_ppm,
            "eligible_coverage_ppm": self.eligible_coverage_ppm,
            "selected_occurrence_count": self.selected_occurrence_count,
            "labeled_occurrence_count": self.labeled_occurrence_count,
            "eligible_labeled_occurrence_count": self.eligible_labeled_occurrence_count,
            "judge_graded_count": self.judge_graded_count,
            "semantic_unknown_count": self.semantic_unknown_count,
            "judge_failure_count": self.judge_failure_count,
            "judge_ungraded_count": self.judge_ungraded_count,
            "unique_primary_label_counts": [
                item.to_dict() for item in self.unique_primary_label_counts
            ],
            "confusion_matrix": [item.to_dict() for item in self.confusion_matrix],
            "exact_agreement_numerator": self.exact_agreement_numerator,
            "exact_agreement_denominator": self.exact_agreement_denominator,
            "exact_agreement_ppm": self.exact_agreement_ppm,
            "class_metrics": [item.to_dict() for item in self.class_metrics],
            "cohen_kappa_ppm": self.cohen_kappa_ppm,
            "cohen_kappa_null_reason": (
                None if self.cohen_kappa_null_reason is None else self.cohen_kappa_null_reason.value
            ),
            "disagreement_count": self.disagreement_count,
            "disagreement_item_refs": list(self.disagreement_item_refs),
            "auxiliary_calibrations": [
                item.to_dict() for item in self.auxiliary_calibrations
            ],
            "status": self.status.value,
        }


@dataclass(frozen=True)
class CalibrationResultV1(_JsonModel):
    schema_version: str
    calibration_result_id: str
    algorithm_version: str
    source_digest: str
    package_id: str
    package_digest: str
    payload_digest: str
    policy: CalibrationSelectionPolicyV1
    label_set_id: str
    label_set_digest: str
    profiles: Tuple[ProfileCalibrationV1, ...]
    status: CalibrationStatus

    def __post_init__(self) -> None:
        if self.schema_version != CALIBRATION_RESULT_SCHEMA_VERSION:
            raise _error("CalibrationResultV1 schema_version is unsupported")
        if self.algorithm_version != CALIBRATION_ALGORITHM_VERSION:
            raise _error("CalibrationResultV1 algorithm_version is unsupported")
        validate_path_segment(self.calibration_result_id, "calibration_result_id")
        validate_path_segment(self.package_id, "CalibrationResultV1.package_id")
        validate_path_segment(self.label_set_id, "CalibrationResultV1.label_set_id")
        for name in ("source_digest", "package_digest", "payload_digest", "label_set_digest"):
            _digest(getattr(self, name), f"CalibrationResultV1.{name}")
        if type(self.policy) is not CalibrationSelectionPolicyV1:
            raise _error("CalibrationResultV1.policy is invalid")
        profiles = tuple(self.profiles)
        if len(profiles) != 1 or type(profiles[0]) is not ProfileCalibrationV1:
            raise _error("CalibrationResultV1 must contain exactly one independent profile")
        profile = profiles[0]
        if (
            profile.package_id != self.package_id
            or profile.package_digest != self.package_digest
            or profile.policy_digest != self.policy.digest()
            or profile.label_set_id != self.label_set_id
            or profile.label_set_digest != self.label_set_digest
            or profile.status is not self.status
        ):
            raise _error("CalibrationResultV1 nested bindings are inconsistent")
        expected_status = _status_for(
            policy=self.policy,
            selected=profile.selected_count,
            labeled=profile.labeled_count,
            eligible=profile.eligible_labeled_count,
            class_counts={
                item.label: item.count
                for item in profile.unique_primary_label_counts
            },
            exact_agreement_ppm=profile.exact_agreement_ppm,
            kappa_ppm=profile.cohen_kappa_ppm,
            auxiliary_calibrations=profile.auxiliary_calibrations,
        )
        if self.status is not expected_status:
            raise _error(
                "CalibrationResultV1 status differs from coverage and thresholds"
            )
        identity = self.to_dict()
        identity.pop("calibration_result_id")
        if self.calibration_result_id != stable_id("calibration-result-v1", identity):
            raise _error("CalibrationResultV1 calibration_result_id is not canonical")
        object.__setattr__(self, "profiles", profiles)

    @classmethod
    def create(cls, **fields: Any) -> "CalibrationResultV1":
        identity = {
            "schema_version": CALIBRATION_RESULT_SCHEMA_VERSION,
            "algorithm_version": CALIBRATION_ALGORITHM_VERSION,
            **fields,
        }
        serialized = {
            **identity,
            "policy": identity["policy"].to_dict(),
            "profiles": [item.to_dict() for item in identity["profiles"]],
            "status": identity["status"].value,
        }
        return cls(
            calibration_result_id=stable_id("calibration-result-v1", serialized),
            **identity,
        )

    @classmethod
    def from_dict(cls, value: Any) -> "CalibrationResultV1":
        payload = _exact(
            value,
            (
                "schema_version", "calibration_result_id", "algorithm_version",
                "source_digest", "package_id", "package_digest", "payload_digest",
                "policy", "label_set_id", "label_set_digest", "profiles", "status",
            ),
            "CalibrationResultV1",
        )
        profiles = _array(payload["profiles"], "CalibrationResultV1.profiles", 4)
        return cls(
            **{
                **payload,
                "policy": CalibrationSelectionPolicyV1.from_dict(payload["policy"]),
                "profiles": tuple(ProfileCalibrationV1.from_dict(item) for item in profiles),
                "status": _enum(CalibrationStatus, payload["status"], "status"),
            }
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "calibration_result_id": self.calibration_result_id,
            "algorithm_version": self.algorithm_version,
            "source_digest": self.source_digest,
            "package_id": self.package_id,
            "package_digest": self.package_digest,
            "payload_digest": self.payload_digest,
            "policy": self.policy.to_dict(),
            "label_set_id": self.label_set_id,
            "label_set_digest": self.label_set_digest,
            "profiles": [item.to_dict() for item in self.profiles],
            "status": self.status.value,
        }


@dataclass(frozen=True)
class _SelectionCandidate:
    item: CalibrationItemV1
    result: JudgeExecutionResult
    source_digest: str
    recorded_outcome: str
    primary_label: Optional[str]
    severity_assessment: Optional[str]
    actionability: Optional[str]
    stratum: str
    reasons: Tuple[str, ...]
    seed_rank: str


def _profile_snapshot(result: JudgeExecutionResult) -> Any:
    return result.evaluator_execution.evaluator.profile(
        JudgeKind(result.request.task.value)
    )


def _decision_projection(
    result: JudgeExecutionResult,
) -> tuple[str, Optional[str], Optional[str], Optional[str]]:
    if result.status is JudgeRunStatus.JUDGE_FAILED:
        return JUDGE_FAILED_OUTCOME, None, None, None
    if result.status is JudgeRunStatus.UNGRADED:
        return JUDGE_UNGRADED_OUTCOME, None, None, None
    decision = result.decision
    if decision is None:
        raise ArtifactIntegrityError("graded Judge result has no persisted decision")
    severity = getattr(decision, "severity_assessment", None)
    actionability = getattr(decision, "actionability", None)
    if result.request.task is JudgeTask.INTENT_EQUIVALENCE:
        label = decision.relation.value
    elif result.request.task is JudgeTask.FINDING_EQUIVALENCE:
        label = decision.relation.value
    elif result.request.task is JudgeTask.NOVEL_FACTUALITY:
        label = decision.factuality.value
    else:
        label = decision.support.value
    return (
        label,
        label,
        None if severity is None else severity.value,
        None if actionability is None else actionability.value,
    )


def _blind_payload(request: BlindJudgeInput) -> tuple[Dict[str, Any], Optional[str]]:
    labels = _allowed_labels(request.task)
    dimension: Optional[str] = None
    if request.task is JudgeTask.INTENT_EQUIVALENCE:
        dimensions = {item.metadata["dimension"] for item in request.items}
        if len(dimensions) != 1:
            raise ArtifactIntegrityError("Intent calibration item crosses dimensions")
        dimension = next(iter(dimensions))
    auxiliary: Dict[str, list[str]] = {}
    if request.task in {
        JudgeTask.FINDING_EQUIVALENCE,
        JudgeTask.NOVEL_FACTUALITY,
    }:
        auxiliary = {
            "severity_assessment": [item.value for item in SeverityAssessment],
            "actionability": [item.value for item in ActionabilityAssessment],
        }
    payload = {
        "schema_version": CALIBRATION_BLINDED_REQUEST_SCHEMA_VERSION,
        "profile": request.task.value,
        "rubric": {
            "rubric_id": request.rubric.rubric_id,
            "rubric_version": request.rubric.rubric_version,
            "rubric_digest": request.rubric.rubric_digest,
            "instruction": request.rubric.instruction,
        },
        "dimension": dimension,
        "items": [item.to_model_dict() for item in request.items],
        "context_blocks": [item.to_model_dict() for item in request.contexts],
        "allowed_reason_refs": list(request.allowed_reason_refs),
        "allowed_labels": list(labels),
        "auxiliary_allowed_labels": auxiliary,
    }
    return payload, dimension


def _calibration_item(
    result: JudgeExecutionResult,
    *,
    forbidden_identity_values: Sequence[str],
) -> CalibrationItemV1:
    request = result.request
    profile = _profile_snapshot(result)
    if (
        profile.rubric_id != request.rubric.rubric_id
        or profile.rubric_version != request.rubric.rubric_version
        or profile.rubric_digest != request.rubric.rubric_digest
    ):
        raise ArtifactIntegrityError("calibration request rubric differs from profile")
    payload, dimension = _blind_payload(request)
    _assert_blind_payload(
        payload,
        forbidden_identity_values=forbidden_identity_values,
        context="calibration blinded request",
    )
    payload_json = _canonical_payload(payload, "calibration blinded request")
    payload_digest = canonical_sha256(payload)
    item_id = _item_identity(
        request.task,
        request.rubric.rubric_version,
        profile.context_builder_version,
        payload,
    )
    source_digest = canonical_sha256(
        {
            "profile": request.task.value,
            "source_request_digest": request.source_request_digest,
            "request_payload_digest": payload_digest,
            "rubric_digest": request.rubric.rubric_digest,
            "context_builder_version": profile.context_builder_version,
        }
    )
    return CalibrationItemV1(
        schema_version=CALIBRATION_ITEM_SCHEMA_VERSION,
        calibration_item_id=item_id,
        profile=request.task,
        rubric_id=request.rubric.rubric_id,
        rubric_version=request.rubric.rubric_version,
        rubric_digest=request.rubric.rubric_digest,
        context_builder_version=profile.context_builder_version,
        dimension=dimension,
        blinded_request_payload_json=payload_json,
        payload_digest=payload_digest,
        allowed_labels=_allowed_labels(request.task),
        source_digest=source_digest,
    )


def _deterministic_judge_conflict(
    trial: Any,
    result: JudgeExecutionResult,
) -> bool:
    if result.status is not JudgeRunStatus.GRADED or result.decision is None:
        return False
    request_id = result.request.source_request_id
    if result.request.task is JudgeTask.INTENT_EQUIVALENCE:
        evaluation = trial.intent_result
        if evaluation is None:
            return False
        candidate = next(
            (
                item
                for item in evaluation.candidates
                if item.request_id == request_id
            ),
            None,
        )
        if (
            candidate is None
            or result.decision.relation is not IntentJudgeRelation.EQUIVALENT
            or candidate.selected
        ):
            return False
        return any(
            item.selected
            and item.match_kind is not IntentMatchKind.SEMANTIC
            and (
                item.generated_id == candidate.generated_id
                or item.truth_id == candidate.truth_id
            )
            for item in evaluation.candidates
        )
    # Novel factuality and Evidence support deliberately have no deterministic
    # semantic label for the same request.  Evidence integrity, for example,
    # is not a deterministic substitute for Evidence support.
    if result.request.task is not JudgeTask.FINDING_EQUIVALENCE:
        return False
    evaluation = trial.review_result
    if evaluation is None:
        return False
    candidates = (
        *evaluation.known_invalid_candidates,
        *evaluation.expected_candidates,
    )
    candidate = next(
        (item for item in candidates if item.request_id == request_id),
        None,
    )
    if (
        candidate is None
        or result.decision.relation is not FindingMatchRelation.EQUIVALENT
        or candidate.selected
    ):
        return False
    return any(
        item.selected
        and item.match_kind is FindingMatchKind.EXACT
        and (
            item.finding_id == candidate.finding_id
            or item.truth_id == candidate.truth_id
        )
        for item in candidates
    )


def _high_critical_fabricated(trial: Any, result: JudgeExecutionResult) -> bool:
    if result.request.task not in {
        JudgeTask.FINDING_EQUIVALENCE,
        JudgeTask.NOVEL_FACTUALITY,
    }:
        return False
    review = trial.review_result
    if review is None:
        return False
    request_record = {
        item.request_id: item for item in review.judge_requests
    }.get(result.request.source_request_id)
    if request_record is None:
        return False
    finding = {item.finding_id: item for item in review.generated_findings}.get(
        request_record.finding_id
    )
    outcome = {item.finding_id: item for item in review.finding_outcomes}.get(
        request_record.finding_id
    )
    return bool(
        finding is not None
        and finding.severity in {FindingSeverity.HIGH, FindingSeverity.CRITICAL}
        and outcome is not None
        and outcome.issue_judgement.value == "fabricated"
    )


def _candidate_for(
    trial: Any,
    result: JudgeExecutionResult,
    *,
    policy: CalibrationSelectionPolicyV1,
    forbidden_identity_values: Sequence[str],
) -> _SelectionCandidate:
    item = _calibration_item(
        result,
        forbidden_identity_values=forbidden_identity_values,
    )
    outcome, primary, severity, actionability = _decision_projection(result)
    reasons = []
    if primary == "unknown":
        reasons.append("mandatory_semantic_unknown")
    if _high_critical_fabricated(trial, result):
        reasons.append("mandatory_high_critical_fabricated")
    if _deterministic_judge_conflict(trial, result):
        reasons.append("mandatory_deterministic_conflict")
    if reasons:
        stratum = "mandatory"
    else:
        reasons.append("seeded_normal_stratum")
        stratum = f"normal:{outcome}"
    source_digest = canonical_sha256(
        {
            "trial_score_digest": trial.trial_score.digest(),
            "judge_input_digest": trial.judge_input.digest(),
            "judge_output_digest": trial.judge_output.digest(),
            "judge_result_digest": result.digest(),
            "request_digest": result.request.digest(),
        }
    )
    seed_rank = canonical_sha256(
        {
            "algorithm": policy.algorithm_version,
            "seed": policy.selection_seed,
            "profile": result.request.task.value,
            "stratum": stratum,
            "calibration_item_id": item.calibration_item_id,
            "source_digest": source_digest,
        }
    )
    return _SelectionCandidate(
        item=item,
        result=result,
        source_digest=source_digest,
        recorded_outcome=outcome,
        primary_label=primary,
        severity_assessment=severity,
        actionability=actionability,
        stratum=stratum,
        reasons=tuple(sorted(reasons)),
        seed_rank=seed_rank,
    )


def _select_candidates(
    candidates: Sequence[_SelectionCandidate],
    policy: CalibrationSelectionPolicyV1,
) -> Tuple[_SelectionCandidate, ...]:
    mandatory = [
        item for item in candidates if set(item.reasons).intersection(_MANDATORY_REASONS)
    ]
    mandatory.sort(
        key=lambda item: (
            item.reasons,
            item.item.calibration_item_id,
            item.source_digest,
        )
    )
    if len(mandatory) > policy.max_items_per_profile:
        raise _error("mandatory calibration selections exceed the profile bound")
    normals: Dict[str, list[_SelectionCandidate]] = {}
    for item in candidates:
        if item not in mandatory:
            normals.setdefault(item.stratum, []).append(item)
    sampled: list[_SelectionCandidate] = []
    for stratum in sorted(normals):
        ordered = sorted(
            normals[stratum],
            key=lambda item: (
                item.seed_rank,
                item.item.calibration_item_id,
                item.source_digest,
            ),
        )
        sampled.extend(ordered[: policy.max_normal_items_per_stratum])
    remaining = policy.max_items_per_profile - len(mandatory)
    sampled = sorted(
        sampled,
        key=lambda item: (
            item.seed_rank,
            item.stratum,
            item.item.calibration_item_id,
            item.source_digest,
        ),
    )[:remaining]
    selected = mandatory + sampled
    return tuple(
        sorted(
            selected,
            key=lambda item: (
                canonical_sha256(
                    {
                        "algorithm": policy.algorithm_version,
                        "selection_seed": policy.selection_seed,
                        "profile": item.item.profile.value,
                        "calibration_item_id": item.item.calibration_item_id,
                        "source_digest": item.source_digest,
                    }
                ),
                item.item.calibration_item_id,
                item.source_digest,
            ),
        )
    )


def _build_package_with_selection(
    evaluation: VerifiedRunEvaluation,
    *,
    profile: JudgeTask,
    policy: CalibrationSelectionPolicyV1,
) -> tuple[CalibrationPackageV1, Tuple[_SelectionCandidate, ...]]:
    if type(evaluation) is not VerifiedRunEvaluation:
        raise TypeError("evaluation must be a VerifiedRunEvaluation")
    if type(profile) is not JudgeTask:
        raise TypeError("profile must be a JudgeTask")
    if type(policy) is not CalibrationSelectionPolicyV1:
        raise TypeError("policy must be a CalibrationSelectionPolicyV1")
    source_binding = evaluation.verify()
    canonical_policy = CalibrationSelectionPolicyV1.from_dict(policy.to_dict())
    if canonical_policy != policy:
        raise _error("calibration selection policy is not canonical")
    forbidden_identity_values = _forbidden_identity_values(evaluation)
    candidates: list[_SelectionCandidate] = []
    source_artifacts = []
    for trial in evaluation.trials:
        trial.judge_output.validate_against(trial.judge_input)
        source_artifacts.append(
            {
                "trial_score_digest": trial.trial_score.digest(),
                "judge_input_digest": trial.judge_input.digest(),
                "judge_output_digest": trial.judge_output.digest(),
                "intent_result_digest": (
                    None if trial.intent_result is None else trial.intent_result.digest()
                ),
                "review_result_digest": (
                    None if trial.review_result is None else trial.review_result.digest()
                ),
            }
        )
        for result in trial.judge_output.results:
            if result.request.task is profile:
                candidates.append(
                    _candidate_for(
                        trial,
                        result,
                        policy=canonical_policy,
                        forbidden_identity_values=forbidden_identity_values,
                    )
                )
    selected = _select_candidates(candidates, canonical_policy)
    item_map = {item.item.calibration_item_id: item.item for item in selected}
    items = tuple(sorted(item_map.values(), key=lambda item: item.calibration_item_id))
    records = []
    for order, selected_item in enumerate(selected, start=1):
        item = selected_item.item
        records.append(
            CalibrationSelectionRecordV1.create(
                calibration_item_id=item.calibration_item_id,
                item_digest=item.digest(),
                source_digest=selected_item.source_digest,
                selection_order=order,
                selection_seed=canonical_policy.selection_seed,
            )
        )
    record_tuple = tuple(records)
    payload_digest = canonical_sha256([item.to_dict() for item in items])
    selection_digest = canonical_sha256([item.to_dict() for item in record_tuple])
    source_digest = canonical_sha256(
        {
            "analysis_source_binding": source_binding.to_dict(),
            "profile": profile.value,
            "source_artifacts": source_artifacts,
        }
    )
    package_id = stable_id(
        "calibration-package-v1",
        profile.value,
        canonical_policy.digest(),
        source_digest,
        payload_digest,
        selection_digest,
    )
    package = CalibrationPackageV1(
        schema_version=CALIBRATION_PACKAGE_SCHEMA_VERSION,
        package_id=package_id,
        profile=profile,
        policy=canonical_policy,
        source_digest=source_digest,
        payload_digest=payload_digest,
        selection_digest=selection_digest,
        items=items,
        selection_records=record_tuple,
        status=CalibrationStatus.PENDING_HUMAN_LABELS,
    )
    return package, selected


def _build_package(
    evaluation: VerifiedRunEvaluation,
    *,
    profile: JudgeTask,
    policy: CalibrationSelectionPolicyV1,
) -> CalibrationPackageV1:
    package, _selected = _build_package_with_selection(
        evaluation,
        profile=profile,
        policy=policy,
    )
    return package


def _portable_key(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().rstrip(" .")


def _validate_external_root(value: Path) -> Path:
    if not isinstance(value, Path):
        raise TypeError("output_root must be a pathlib.Path")
    raw = os.fspath(value)
    if ".." in value.parts:
        raise ArtifactSecurityError("calibration output_root contains traversal")
    if os.name == "nt":
        tail = raw[2:] if len(raw) >= 2 and raw[1] == ":" else raw
        if ":" in tail:
            raise ArtifactSecurityError("calibration output_root contains an ADS path")
    absolute = Path(os.path.abspath(raw))
    forbidden_roots = {_portable_key(".eval-runs"), _portable_key(".eval-analyses")}
    if any(_portable_key(part) in forbidden_roots for part in absolute.parts):
        raise ValueError("calibration payload may not be written to the Analysis or Run Store")
    existing = absolute
    while not os.path.lexists(existing):
        if existing.parent == existing:
            raise ArtifactSecurityError("calibration output_root has no existing ancestor")
        existing = existing.parent
    current = existing
    while True:
        try:
            metadata = os.lstat(current)
        except OSError as exc:
            raise ArtifactSecurityError("could not inspect calibration output ancestry") from exc
        if _unsafe_node(metadata) or not stat.S_ISDIR(metadata.st_mode):
            raise ArtifactSecurityError(
                "calibration output ancestry contains a link, reparse point, or non-directory"
            )
        git_marker = current / ".git"
        if os.path.lexists(git_marker):
            raise ValueError("calibration payload may not be written inside a repository")
        if current.parent == current:
            break
        current = current.parent
    return absolute


class _CalibrationExportStorage(ArtifactStore):
    def __init__(self, root: Path) -> None:
        self._initialize_storage_root(
            root,
            max_file_bytes=MAX_CALIBRATION_BYTES,
            max_total_read_bytes=MAX_CALIBRATION_BYTES * 2,
            create_root=True,
            required_root_name=None,
            reject_hardlinks=True,
        )


def _export_entries(storage: _CalibrationExportStorage, directory: Path) -> set[str]:
    if not os.path.lexists(directory):
        return set()
    storage._assert_directory(directory)
    names: set[str] = set()
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                metadata = os.lstat(entry.path)
                if _unsafe_node(metadata) or not stat.S_ISREG(metadata.st_mode):
                    raise ArtifactSecurityError(
                        "calibration export contains a symlink, reparse point, or unsafe entry"
                    )
                if _hardlinked_file(metadata):
                    raise ArtifactSecurityError(
                        "calibration export artifact has an unsafe hardlink count"
                    )
                names.add(entry.name)
                if len(names) > 2:
                    raise ArtifactIntegrityError(
                        "calibration export has unknown artifacts"
                    )
    except (ArtifactSecurityError, ArtifactIntegrityError):
        raise
    except OSError as exc:
        raise ArtifactSecurityError("could not inspect calibration export") from exc
    if len({_portable_key(name) for name in names}) != len(names):
        raise ArtifactSecurityError("calibration export has a portable name collision")
    return names


def _write_external_package(package: CalibrationPackageV1, output_root: Path) -> None:
    root = _validate_external_root(output_root)
    storage = _CalibrationExportStorage(root)
    directory = storage._within_root(storage.root / package.package_id)
    storage._ensure_directory(directory)
    payload_name = "calibration_package.json"
    receipt_name = "receipt.json"
    payload_data = canonical_json_bytes(package.to_blind_dict())
    receipt = {
        "schema_version": CALIBRATION_EXPORT_RECEIPT_SCHEMA_VERSION,
        "package_id": package.package_id,
        "package_digest": package.digest(),
        "payload_digest": hashlib.sha256(payload_data).hexdigest(),
        "payload_size_bytes": len(payload_data),
    }
    receipt_data = canonical_json_bytes(receipt)
    names = _export_entries(storage, directory)
    if names - {payload_name, receipt_name}:
        raise ArtifactIntegrityError("calibration export has an unknown artifact")
    budget = _ReadBudget(storage.max_total_read_bytes)
    payload_path = directory / payload_name
    receipt_path = directory / receipt_name
    if payload_name in names:
        stored_payload = storage._read_bytes(
            payload_path,
            expected_sha256=hashlib.sha256(payload_data).hexdigest(),
            expected_size=len(payload_data),
            budget=budget,
        )
        if stored_payload != payload_data:
            raise ArtifactConflictError("existing calibration payload differs")
    else:
        if receipt_name in names:
            raise ArtifactIntegrityError("calibration receipt exists without payload")
        storage._write_bytes_exclusive(payload_path, payload_data)
    if receipt_name in names:
        stored_receipt = storage._read_bytes(
            receipt_path,
            expected_sha256=hashlib.sha256(receipt_data).hexdigest(),
            expected_size=len(receipt_data),
            budget=budget,
        )
        if stored_receipt != receipt_data:
            raise ArtifactConflictError("existing calibration receipt differs")
    else:
        storage._write_bytes_exclusive(receipt_path, receipt_data)
    if _export_entries(storage, directory) != {payload_name, receipt_name}:
        raise ArtifactIntegrityError("calibration export publication is incomplete")


def export_calibration_package(
    evaluation: VerifiedRunEvaluation,
    *,
    profile: JudgeTask,
    policy: CalibrationSelectionPolicyV1,
    output_root: Path,
) -> CalibrationPackageV1:
    """Export a create-only blind package from persisted Evaluation artifacts."""

    package = _build_package(evaluation, profile=profile, policy=policy)
    _assert_blind_payload(
        package.to_blind_dict(),
        forbidden_identity_values=_forbidden_identity_values(evaluation),
        context="calibration package export",
    )
    _write_external_package(package, output_root)
    return package


def _status_for(
    *,
    policy: CalibrationSelectionPolicyV1,
    selected: int,
    labeled: int,
    eligible: int,
    class_counts: Mapping[str, int],
    exact_agreement_ppm: Optional[int],
    kappa_ppm: Optional[int],
    auxiliary_calibrations: Sequence[AuxiliaryCalibrationV1],
) -> CalibrationStatus:
    if labeled == 0 or eligible == 0:
        return CalibrationStatus.PENDING_HUMAN_LABELS
    eligible_coverage = _ratio_ppm(eligible, selected)
    if (
        eligible < policy.minimum_human_labels
        or eligible != selected
        or eligible_coverage is None
        or eligible_coverage < policy.minimum_human_coverage_ppm
    ):
        return CalibrationStatus.INSUFFICIENT_COVERAGE
    for auxiliary in auxiliary_calibrations:
        if auxiliary.applicability is AuxiliaryCalibrationApplicability.NOT_APPLICABLE:
            continue
        if (
            auxiliary.eligible_labeled_item_count
            != auxiliary.applicable_item_count
            or auxiliary.eligible_item_coverage_ppm is None
            or auxiliary.eligible_item_coverage_ppm
            < policy.minimum_auxiliary_human_coverage_ppm
        ):
            return CalibrationStatus.INSUFFICIENT_COVERAGE
    if (
        any(
            count < policy.minimum_labels_per_class
            for count in class_counts.values()
        )
        or exact_agreement_ppm is None
        or exact_agreement_ppm < policy.minimum_exact_agreement_ppm
        or kappa_ppm is None
        or kappa_ppm < policy.minimum_cohen_kappa_ppm
    ):
        return CalibrationStatus.FAILED_THRESHOLDS
    for auxiliary in auxiliary_calibrations:
        if auxiliary.applicability is AuxiliaryCalibrationApplicability.NOT_APPLICABLE:
            continue
        unique_counts = {
            item.label: item.count
            for item in auxiliary.unique_human_label_counts
        }
        if (
            any(
                count < policy.minimum_auxiliary_labels_per_class
                for count in unique_counts.values()
            )
            or auxiliary.exact_agreement_ppm is None
            or auxiliary.exact_agreement_ppm
            < policy.minimum_auxiliary_exact_agreement_ppm
            or (
                policy.minimum_auxiliary_cohen_kappa_ppm is not None
                and (
                    auxiliary.cohen_kappa_ppm is None
                    or auxiliary.cohen_kappa_ppm
                    < policy.minimum_auxiliary_cohen_kappa_ppm
                )
            )
        ):
            return CalibrationStatus.FAILED_THRESHOLDS
    return CalibrationStatus.GATE_ELIGIBLE


def _class_metrics_for_counts(
    labels: Sequence[str],
    outcomes: Sequence[str],
    counts: Mapping[tuple[str, str], int],
) -> Tuple[ClassCalibrationV1, ...]:
    metrics = []
    for label in labels:
        human_count = sum(counts[(label, outcome)] for outcome in outcomes)
        recorded_count = sum(counts[(human, label)] for human in labels)
        true_positive = counts[(label, label)]
        metrics.append(
            ClassCalibrationV1(
                label=label,
                human_count=human_count,
                recorded_count=recorded_count,
                true_positive_count=true_positive,
                precision_ppm=_ratio_ppm(true_positive, recorded_count),
                precision_null_reason=(
                    ClassMetricNullReason.NO_RECORDED_PREDICTIONS
                    if recorded_count == 0
                    else None
                ),
                recall_ppm=_ratio_ppm(true_positive, human_count),
                recall_null_reason=(
                    ClassMetricNullReason.NO_HUMAN_LABELS
                    if human_count == 0
                    else None
                ),
            )
        )
    return tuple(metrics)


def _score_auxiliary_dimension(
    dimension: CalibrationAuxiliaryDimension,
    occurrences: Sequence[tuple[CalibrationSelectionRecordV1, _SelectionCandidate]],
    label_map: Mapping[str, HumanLabelV1],
    policy: CalibrationSelectionPolicyV1,
) -> AuxiliaryCalibrationV1:
    labels = (
        tuple(item.value for item in SeverityAssessment)
        if dimension is CalibrationAuxiliaryDimension.SEVERITY_ASSESSMENT
        else tuple(item.value for item in ActionabilityAssessment)
    )
    counts = {(human, recorded): 0 for human in labels for recorded in labels}
    applicable_items: set[str] = set()
    labeled_items: set[str] = set()
    eligible_items: Dict[str, str] = {}
    applicable_occurrences = 0
    labeled_occurrences = 0
    eligible_occurrences = 0
    disagreements: list[str] = []
    for record, selected in occurrences:
        recorded = (
            selected.severity_assessment
            if dimension is CalibrationAuxiliaryDimension.SEVERITY_ASSESSMENT
            else selected.actionability
        )
        if recorded is None:
            continue
        applicable_occurrences += 1
        applicable_items.add(record.calibration_item_id)
        human = label_map.get(record.calibration_item_id)
        if human is None:
            continue
        human_value = (
            human.effective_severity_assessment
            if dimension is CalibrationAuxiliaryDimension.SEVERITY_ASSESSMENT
            else human.effective_actionability
        )
        if human_value is None:
            continue
        labeled_occurrences += 1
        labeled_items.add(record.calibration_item_id)
        if not human.eligible_under(policy):
            continue
        eligible_occurrences += 1
        eligible_items[record.calibration_item_id] = human_value
        counts[(human_value, recorded)] += 1
        if human_value != recorded:
            disagreements.append(record.selection_record_id)
    matrix = tuple(
        ConfusionMatrixCellV1(human, recorded, counts[(human, recorded)])
        for human in labels
        for recorded in labels
    )
    agreement = sum(counts[(label, label)] for label in labels)
    kappa, kappa_null = _kappa_from_matrix(labels, labels, counts)
    unique_counts = tuple(
        LabelCountV1(
            label,
            sum(1 for value in eligible_items.values() if value == label),
        )
        for label in labels
    )
    applicable_item_count = len(applicable_items)
    labeled_item_count = len(labeled_items)
    eligible_item_count = len(eligible_items)
    return AuxiliaryCalibrationV1.create(
        dimension=dimension,
        applicability=(
            AuxiliaryCalibrationApplicability.APPLICABLE
            if applicable_occurrences
            else AuxiliaryCalibrationApplicability.NOT_APPLICABLE
        ),
        allowed_labels=labels,
        recorded_outcomes=labels,
        applicable_item_count=applicable_item_count,
        labeled_item_count=labeled_item_count,
        eligible_labeled_item_count=eligible_item_count,
        pending_item_count=applicable_item_count - labeled_item_count,
        labeled_item_coverage_ppm=_ratio_ppm(
            labeled_item_count,
            applicable_item_count,
        ),
        eligible_item_coverage_ppm=_ratio_ppm(
            eligible_item_count,
            applicable_item_count,
        ),
        applicable_occurrence_count=applicable_occurrences,
        labeled_occurrence_count=labeled_occurrences,
        eligible_labeled_occurrence_count=eligible_occurrences,
        unique_human_label_counts=unique_counts,
        confusion_matrix=matrix,
        exact_agreement_numerator=agreement,
        exact_agreement_denominator=eligible_occurrences,
        exact_agreement_ppm=_ratio_ppm(agreement, eligible_occurrences),
        class_metrics=_class_metrics_for_counts(labels, labels, counts),
        cohen_kappa_ppm=kappa,
        cohen_kappa_null_reason=kappa_null,
        disagreement_count=len(disagreements),
        disagreement_occurrence_refs=tuple(sorted(disagreements)),
    )


def score_calibration(
    evaluation: VerifiedRunEvaluation,
    *,
    package: CalibrationPackageV1,
    labels: HumanLabelSetV1,
) -> CalibrationResultV1:
    """Compare persisted Judge outcomes with independently supplied human labels."""

    if type(evaluation) is not VerifiedRunEvaluation:
        raise TypeError("evaluation must be a VerifiedRunEvaluation")
    if type(package) is not CalibrationPackageV1:
        raise TypeError("package must be a CalibrationPackageV1")
    if type(labels) is not HumanLabelSetV1:
        raise TypeError("labels must be a HumanLabelSetV1")
    evaluation.verify()
    canonical_package = CalibrationPackageV1.from_dict(package.to_dict())
    replayed_package, replayed_selection = _build_package_with_selection(
        evaluation,
        profile=package.profile,
        policy=package.policy,
    )
    if (
        canonical_json_bytes(canonical_package.to_dict())
        != canonical_json_bytes(package.to_dict())
        or canonical_json_bytes(replayed_package.to_dict())
        != canonical_json_bytes(package.to_dict())
    ):
        raise ArtifactIntegrityError(
            "calibration package differs from exact Evaluation source replay"
        )
    canonical_labels = HumanLabelSetV1.from_dict(
        labels.to_dict(),
        package=package,
    )
    if canonical_json_bytes(canonical_labels.to_dict()) != canonical_json_bytes(
        labels.to_dict()
    ):
        raise ArtifactIntegrityError("human labels differ from canonical package replay")
    label_map = {item.calibration_item_id: item for item in canonical_labels.labels}
    allowed = package.items[0].allowed_labels if package.items else _allowed_labels(package.profile)
    outcomes = allowed + (JUDGE_FAILED_OUTCOME, JUDGE_UNGRADED_OUTCOME)
    counts = {(human, recorded): 0 for human in allowed for recorded in outcomes}
    disagreements: list[str] = []
    graded = 0
    semantic_unknown = 0
    failed = 0
    ungraded = 0
    if len(replayed_selection) != len(package.selection_records):
        raise ArtifactIntegrityError("calibration selection replay length differs")
    occurrences = tuple(zip(package.selection_records, replayed_selection))
    for record, selected in occurrences:
        if (
            record.calibration_item_id != selected.item.calibration_item_id
            or record.source_digest != selected.source_digest
        ):
            raise ArtifactIntegrityError(
                "calibration selection differs from source request replay"
            )
    selected_item_ids = {record.calibration_item_id for record, _ in occurrences}
    if selected_item_ids != {item.calibration_item_id for item in package.items}:
        raise ArtifactIntegrityError("calibration unique item replay differs")
    labeled_item_ids = selected_item_ids.intersection(label_map)
    eligible_label_map = {
        item_id: label_map[item_id]
        for item_id in labeled_item_ids
        if label_map[item_id].eligible_under(package.policy)
    }
    selected_count = len(selected_item_ids)
    labeled_count = len(labeled_item_ids)
    eligible_count = len(eligible_label_map)
    unattested = sum(
        1 for item_id in labeled_item_ids if not label_map[item_id].blind_attestation
    )
    disputes = sum(
        1
        for item_id in labeled_item_ids
        if label_map[item_id].disputed and label_map[item_id].adjudication is None
    )
    labeled_occurrence_count = 0
    eligible_occurrence_count = 0
    for record, selected in occurrences:
        recorded_outcome = selected.recorded_outcome
        if recorded_outcome == JUDGE_FAILED_OUTCOME:
            failed += 1
        elif recorded_outcome == JUDGE_UNGRADED_OUTCOME:
            ungraded += 1
        else:
            graded += 1
            if selected.primary_label == "unknown":
                semantic_unknown += 1
        human = label_map.get(record.calibration_item_id)
        if human is None:
            continue
        labeled_occurrence_count += 1
        if record.calibration_item_id not in eligible_label_map:
            continue
        eligible_occurrence_count += 1
        effective_label = human.effective_label
        counts[(effective_label, recorded_outcome)] += 1
        if effective_label != recorded_outcome:
            disagreements.append(record.selection_record_id)
    matrix = tuple(
        ConfusionMatrixCellV1(human, recorded, counts[(human, recorded)])
        for human in allowed
        for recorded in outcomes
    )
    agreement = sum(counts[(label, label)] for label in allowed)
    exact_ppm = _ratio_ppm(agreement, eligible_occurrence_count)
    class_metrics = _class_metrics_for_counts(allowed, outcomes, counts)
    unique_class_counts = {
        label: sum(
            1
            for human in eligible_label_map.values()
            if human.effective_label == label
        )
        for label in allowed
    }
    unique_primary_label_counts = tuple(
        LabelCountV1(label, unique_class_counts[label]) for label in allowed
    )
    kappa, kappa_null = _kappa_from_matrix(allowed, outcomes, counts)
    auxiliary_calibrations = tuple(
        _score_auxiliary_dimension(
            dimension,
            occurrences,
            label_map,
            package.policy,
        )
        for dimension in CalibrationAuxiliaryDimension
    )
    status = _status_for(
        policy=package.policy,
        selected=selected_count,
        labeled=labeled_count,
        eligible=eligible_count,
        class_counts=unique_class_counts,
        exact_agreement_ppm=exact_ppm,
        kappa_ppm=kappa,
        auxiliary_calibrations=auxiliary_calibrations,
    )
    profile_fields = {
        "profile": package.profile,
        "package_id": package.package_id,
        "package_digest": package.digest(),
        "policy_digest": package.policy.digest(),
        "label_set_id": labels.label_set_id,
        "label_set_digest": labels.digest(),
        "allowed_labels": allowed,
        "recorded_outcomes": outcomes,
        "selected_count": selected_count,
        "labeled_count": labeled_count,
        "eligible_labeled_count": eligible_count,
        "pending_label_count": selected_count - labeled_count,
        "unattested_label_count": unattested,
        "unadjudicated_dispute_count": disputes,
        "labeled_coverage_ppm": _ratio_ppm(labeled_count, selected_count),
        "eligible_coverage_ppm": _ratio_ppm(eligible_count, selected_count),
        "selected_occurrence_count": len(occurrences),
        "labeled_occurrence_count": labeled_occurrence_count,
        "eligible_labeled_occurrence_count": eligible_occurrence_count,
        "judge_graded_count": graded,
        "semantic_unknown_count": semantic_unknown,
        "judge_failure_count": failed,
        "judge_ungraded_count": ungraded,
        "unique_primary_label_counts": unique_primary_label_counts,
        "confusion_matrix": matrix,
        "exact_agreement_numerator": agreement,
        "exact_agreement_denominator": eligible_occurrence_count,
        "exact_agreement_ppm": exact_ppm,
        "class_metrics": class_metrics,
        "cohen_kappa_ppm": kappa,
        "cohen_kappa_null_reason": kappa_null,
        "disagreement_count": len(disagreements),
        "disagreement_item_refs": tuple(sorted(disagreements)),
        "auxiliary_calibrations": auxiliary_calibrations,
        "status": status,
    }
    profile_result = ProfileCalibrationV1.create(**profile_fields)
    return CalibrationResultV1.create(
        source_digest=package.source_digest,
        package_id=package.package_id,
        package_digest=package.digest(),
        payload_digest=package.payload_digest,
        policy=package.policy,
        label_set_id=labels.label_set_id,
        label_set_digest=labels.digest(),
        profiles=(profile_result,),
        status=status,
    )


__all__ = [
    "CALIBRATION_SELECTION_POLICY_SCHEMA_VERSION",
    "CALIBRATION_ITEM_SCHEMA_VERSION",
    "CALIBRATION_SELECTION_RECORD_SCHEMA_VERSION",
    "CALIBRATION_PACKAGE_SCHEMA_VERSION",
    "CALIBRATION_PACKAGE_MANIFEST_SCHEMA_VERSION",
    "HUMAN_LABEL_SCHEMA_VERSION",
    "HUMAN_LABEL_SET_SCHEMA_VERSION",
    "HUMAN_REVIEWER_PROVENANCE_SCHEMA_VERSION",
    "ADJUDICATION_SCHEMA_VERSION",
    "HUMAN_ADJUDICATION_SCHEMA_VERSION",
    "PROFILE_CALIBRATION_SCHEMA_VERSION",
    "AUXILIARY_CALIBRATION_SCHEMA_VERSION",
    "CALIBRATION_RESULT_SCHEMA_VERSION",
    "CALIBRATION_ALGORITHM_VERSION",
    "CalibrationError",
    "CalibrationStatus",
    "CalibrationSelectionCategory",
    "CalibrationAuxiliaryDimension",
    "AuxiliaryCalibrationApplicability",
    "ReviewerProvenanceKind",
    "KappaNullReason",
    "ClassMetricNullReason",
    "CalibrationSelectionPolicyV1",
    "CalibrationItemV1",
    "CalibrationSelectionRecordV1",
    "CalibrationPackageV1",
    "CalibrationPackageManifestV1",
    "HumanReviewerProvenanceV1",
    "AdjudicationV1",
    "HumanAdjudicationV1",
    "HumanLabelV1",
    "HumanLabelSetV1",
    "ConfusionMatrixCellV1",
    "ClassCalibrationV1",
    "LabelCountV1",
    "AuxiliaryCalibrationV1",
    "ProfileCalibrationV1",
    "CalibrationResultV1",
    "export_calibration_package",
    "score_calibration",
]
